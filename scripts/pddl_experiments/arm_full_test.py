import argparse
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))

import load_multi_tangent
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_planning as pp
from collision import Element, create_couplers, init_pb
from mobile_base_planner import extract_obstacles_from_bars
from multi_tangent.collision import create_collision_bodies
from multi_tangent.convert import flatten_list
from parse import parse_mt_geometric
from pybullet_planning import Attachment, Euler, Point, Pose, get_distance, interpolate_poses, invert, multiply
from robot_setup import RobotSetup
from scipy.spatial import ConvexHull
from stream import get_pick_gen_fn, get_place_gen_fn, get_transfer_gen_fn, get_transit_gen_fn
from utils import CounterModule, CounterValue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Problem info (Input)
    parser.add_argument(
        "--mt_file_name",
        default="one_tet_MT_layer_0_contact.json",
        help='The name of the multi tangent file to solve (json file\'s name, e.g. "box_MT_layer_1.json")',
    )

    args = parser.parse_args()

    # Load process file
    mt_file_name = args.mt_file_name
    line_pt_pairs, contact_id_pairs, bar_radius = parse_mt_geometric(mt_file_name)
    line_pts_flattened = flatten_list(np.array(line_pt_pairs))
    vertices = flatten_list(line_pt_pairs)

    # reach_distance = 1.0
    # sample_max_distance = 0.75
    # safety_distance = 0.5
    # sampling_number = 200

    min_z = np.min(line_pts_flattened, axis=0)[2]
    line_pts_flattened = [np.array([0, 0, -min_z]) + point for point in line_pts_flattened]

    radius_per_edge = [bar_radius] * int(len(line_pts_flattened) / 2)

    init_pb()
    goal_poses = {}
    bar_attachments = []
    counter_modules = []
    with pp.LockRenderer():
        robot_setup0 = RobotSetup("r0")
        element_bodies = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)
        half_coupler_from_contact_pair = create_couplers(line_pts_flattened, contact_id_pairs)
        # temp = element_bodies[1]
        # element_bodies[1] = element_bodies[2]
        # element_bodies[2] = temp

    for i, e in enumerate(element_bodies):
        goal_poses[i] = pp.get_pose(e)
        ipad_link_pose = pp.get_link_pose(robot_setup0.robot, pp.link_from_name(robot_setup0.robot, "ipad_rack_link"))
        delta_pose = Pose(point=[0, 0, 0.5], euler=Euler(roll=-np.pi / 2, pitch=0, yaw=0))
        bar_pose = multiply(ipad_link_pose, delta_pose)
        pp.set_pose(e, bar_pose)
        # pp.draw_pose(bar_pose, length=0.5)
        bar_attachments.append(
            pp.create_attachment(robot_setup0.robot, pp.link_from_name(robot_setup0.robot, "ipad_rack_link"), e)
        )
        counter_modules.append(CounterModule())

    robot_setup0.update_attachments(bar_attachments)
    robot_setup0.set_joint_positions(robot_setup0.base_joints, [-5, -5, 0])
    last_pose_2d = np.array([-5, -5, 0])

    for attachment in bar_attachments:
        attachment.assign()

    element_from_index = {
        i: Element(i, e, pp.get_pose(e), goal_poses[i], [line_pts_flattened[2 * i], line_pts_flattened[2 * i + 1]])
        for i, e in enumerate(element_bodies)
    }
    place_gen = get_place_gen_fn(robot_setup0, element_from_index, [], verbose=False, collisions=True, teleops=False)
    pick_gen = get_pick_gen_fn(robot_setup0, element_from_index, [], verbose=False, collisions=True, teleops=False)
    transfer_gen = get_transfer_gen_fn(
        robot_setup0, element_from_index, [], verbose=False, collisions=True, teleops=False
    )
    transit_gen = get_transit_gen_fn(robot_setup0, element_from_index, [], verbose=True, collisions=True, teleops=False)

    for i in element_from_index:
        cur_attachment = bar_attachments.pop(0)
        counter_handle: CounterModule = counter_modules[i]
        place_handle = counter_handle.create_handle("place")
        pick_handle = counter_handle.create_handle("pick")
        transfer_handle = counter_handle.create_handle("transfer")

        assembled = list(range(i))
        unassembled = list(range(i + 1, len(element_from_index)))
        with pp.LockRenderer():
            place_cmd, place_grasp_mask, grasp_attach, grasp, pregrasp_pose = next(
                place_gen(
                    i, assembled=assembled, unassembled=unassembled, attachments=bar_attachments, counter=place_handle
                )
            )
        cur_attachment.assign()

        unassembled = list(range(i, len(element_from_index)))
        with pp.LockRenderer():
            pick_cmd, pick_grasp_mask = next(pick_gen(i, grasp, assembled, unassembled, bar_attachments, pick_handle))
        # pick_cmd, pick_grasp_mask = [], []

        unassembled = list(range(i + 1, len(element_from_index)))
        with pp.LockRenderer():
            transfer_cmd, transfer_grasp_mask = next(
                transfer_gen(
                    i,
                    grasp_attach,
                    pick_cmd[-1],
                    place_cmd[0],
                    assembled=assembled,
                    unassembled=unassembled,
                    attachments=bar_attachments,
                    counter=transfer_handle,
                )
            )
        # transfer_cmd, transfer_grasp_mask = [], []

        pp.set_pose(element_from_index[i].body, element_from_index[i].goal_pose)
        target_pose_2d = np.array(place_cmd[0][:3])
        with pp.LockRenderer():
            transit_cmd, transit_grasp_mask = next(
                transit_gen(
                    i,
                    last_pose_2d,
                    target_pose_2d,
                    assembled,
                    [cur_attachment] + bar_attachments,
                )
            )

        manipulator_traj = pick_cmd + transfer_cmd + place_cmd
        base_traj = transit_cmd
        manipulator_traj_grasp_mask = pick_grasp_mask + transfer_grasp_mask + place_grasp_mask
        base_traj_grasp_mask = transit_grasp_mask

        last_pose_2d = base_traj[-1][:3]

        cur_attachment.assign()
        p.removeAllUserParameters()

        base_param_slider = p.addUserDebugParameter("base playback", 0.0, 1.0, 0.0)
        manipulator_param_slider = p.addUserDebugParameter("manipulator playback", 0.0, 1.0, 0.0)
        continue_button = p.addUserDebugParameter("continue", 1, 0, 0)
        prev_continue_button_value = p.readUserDebugParameter(continue_button)

        logger_path = os.path.join(cur_dir, "logs", args.mt_file_name.split(".")[0])
        counter_handle.save(logger_path, "bar_" + str(i) + ".json")

        while True:
            current_continue_button_value = p.readUserDebugParameter(continue_button)
            if current_continue_button_value > prev_continue_button_value:
                pp.set_pose(element_from_index[i].body, element_from_index[i].goal_pose)
                prev_continue_button_value = current_continue_button_value
                break

            manipulator_traj_param_value = p.readUserDebugParameter(manipulator_param_slider)
            manipulator_traj_idx = int(manipulator_traj_param_value * (len(manipulator_traj) - 1))
            if manipulator_traj_idx != 0:
                traj_pose = manipulator_traj[manipulator_traj_idx]
                grasp_attach_flag = manipulator_traj_grasp_mask[manipulator_traj_idx]
                robot_setup0.update_attachments(bar_attachments)
            else:
                base_traj_param_value = p.readUserDebugParameter(base_param_slider)
                base_traj_idx = int(base_traj_param_value * (len(base_traj) - 1))
                traj_pose = base_traj[base_traj_idx]
                grasp_attach_flag = base_traj_grasp_mask[base_traj_idx]
                robot_setup0.update_attachments([cur_attachment] + bar_attachments)
            robot_setup0.set_joint_positions(robot_setup0.control_joints, traj_pose)

            if grasp_attach_flag:
                grasp_attach.assign()
