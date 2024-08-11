import argparse

import load_multi_tangent
import numpy as np
import pybullet as p
import pybullet_planning as pp
from collision import Element, create_couplers, init_pb
from multi_tangent.collision import create_collision_bodies
from multi_tangent.convert import flatten_list
from parse import parse_mt_geometric
from robot_setup import RobotSetup
from stream import get_place_gen_fn

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

    min_z = np.min(line_pts_flattened, axis=0)[2]
    line_pts_flattened = [np.array([0, 0, -min_z]) + point for point in line_pts_flattened]

    radius_per_edge = [bar_radius] * int(len(line_pts_flattened) / 2)

    init_pb()
    goal_poses = {}
    with pp.LockRenderer():
        element_bodies = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)
        half_coupler_from_contact_pair = create_couplers(line_pts_flattened, contact_id_pairs)
        for i, e in enumerate(element_bodies):
            goal_poses[i] = pp.get_pose(e)
            # pp.draw_pose(goal_poses[i], length=0.5)
            pp.set_pose(e, pp.Pose(pp.Point(2, 2, 0)))

        robot_setup0 = RobotSetup("r0")

    element_from_index = {
        i: Element(i, e, pp.get_pose(e), goal_poses[i], [line_pts_flattened[2 * i], line_pts_flattened[2 * i + 1]])
        for i, e in enumerate(element_bodies)
    }
    place_gen = get_place_gen_fn(robot_setup0, element_from_index, [], verbose=False, collisions=True, teleops=False)

    traj_param_slider = p.addUserDebugParameter("trajectory playback", 0.0, 1.0, 0.0)

    continue_button = p.addUserDebugParameter("continue", 1, 0, 0)
    prev_continue_button_value = p.readUserDebugParameter(continue_button)

    for i in element_from_index:
        command, pregrasp_length, bar_attach = next(place_gen(i, assembled=list(range(i)), diagnosis=False))

        while True:
            current_continue_button_value = p.readUserDebugParameter(continue_button)
            if current_continue_button_value > prev_continue_button_value:
                pp.set_pose(element_from_index[i].body, element_from_index[i].goal_pose)
                prev_continue_button_value = current_continue_button_value
                break

            traj_param_value = p.readUserDebugParameter(traj_param_slider)
            traj_idx = int(traj_param_value * (len(command) - 1))
            traj_pose = command[traj_idx]
            pp.set_joint_positions(robot_setup0.robot, robot_setup0.control_joints, traj_pose)
            robot_setup0.ee_attachment.assign()
            # print("pregrasp_length = ", pregrasp_length, "traj_idx = ", traj_idx)
            if traj_idx < pregrasp_length:
                bar_attach.assign()
        # pp.wait_if_gui()
