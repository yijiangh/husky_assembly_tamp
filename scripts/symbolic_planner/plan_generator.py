import argparse
import os

import numpy as np
import pybullet as p

cur_dir = os.path.dirname(os.path.abspath(__file__))

# -------------------- self-defined modules --------------------#
import load_multi_tangent
import pybullet_planning as pp
from collision import Element, create_couplers, init_pb
from element_object import ElementObject, ElementStatus
from multi_tangent.collision import create_collision_bodies
from multi_tangent.convert import flatten_list
from parse import parse_mt_geometric
from planner import Planner
from robot import Robot, PathItem, PathWithIndex
from robot_setup import RobotSetup

if __name__ == "__main__":
    with pp.HideOutput():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--mt_file_name",
            default="one_tet_MT_contact.json",
            # default="tower_integral_one_len_MT_layer_0.json",
            # default="triangle_reciprocal_MT_contact.json",
            help='The name of the multi tangent file to solve (json file\'s name, e.g. "tower_integral_one_len_MT_layer_0.json")',
        )
        args = parser.parse_args()

        # -------------------- Load process file --------------------#
        mt_file_name = args.mt_file_name
        line_pt_pairs, contact_id_pairs, bar_radius = parse_mt_geometric(mt_file_name)
        line_pt_pairs: list[list[list]]  # bar list
        contact_id_pairs: list[list]  # contact pairs
        bar_radius: float
        line_pts_flattened: list[np.ndarray] = flatten_list(np.array(line_pt_pairs))  # numpy points list
        vertices: list[list] = flatten_list(line_pt_pairs)  # points list

        # -------------------- Eliminate Z-axis deviation --------------------#
        min_z = np.min(line_pts_flattened, axis=0)[2]
        line_pts_flattened = [np.array([0, 0, -min_z]) + point for point in line_pts_flattened]

        radius_per_edge = [bar_radius] * int(len(line_pts_flattened) / 2)

        # -------------------- Elements Init --------------------#
        init_pb()
        goal_poses = []
        with pp.LockRenderer():
            element_bodies = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)
            half_coupler_from_contact_pair = create_couplers(line_pts_flattened, contact_id_pairs)
            for i, e in enumerate(element_bodies):
                pp.add_text(str(i), pp.get_point(e))
                # pp.set_color(e, (1, 0, 0, 0.1))
                goal_poses.append(pp.get_pose(e))
                pp.set_pose(e, pp.Pose(point=(5, 0, 0), euler=pp.Euler(0, 1.5708, 0)))
        element_from_index = {
            i: Element(i, e, pp.get_pose(e), goal_poses[i], [line_pts_flattened[2 * i], line_pts_flattened[2 * i + 1]])
            for i, e in enumerate(element_bodies)
        }

        #-------------------- Path storage --------------------#
        path_storage = PathWithIndex()

        # -------------------- Robots Init --------------------#
        with pp.HideOutput():
            robots = []
            robot_num = 1
            for i in range(robot_num):
                rb = RobotSetup(f"r{i}")
                robots.append(Robot(i, rb, element_from_index, [], path_storage))

        grounded_elements_index = [0, 1, 2]  # one_tet_MT_contact
        # grounded_elements_index = [0, 1, 4, 19]  # tower_integral_one_len_MT_layer_0
        # grounded_elements_index = [0, 1, 2]  # triangle_reciprocal_MT_contact

        # -------------------- Plan --------------------#
        planner = Planner(robot_num=robot_num, robots=robots)
        path_index = planner.Plan(element_from_index, contact_id_pairs, grounded_elements_index)
        element_object_list = Planner.GetElementObjects(element_from_index, contact_id_pairs, grounded_elements_index)

        # -------------------- Visualization --------------------#
        assembled = []
        # for rb in robots:
        #     pp.set_pose(rb.robot_setup.robot, pp.Pose(point=(5, 0, 0), euler=pp.Euler(0, 0, 0)))
        for step_num, index_list in enumerate(path_index):
            for index in index_list:
                element_object_list[index].Assemble(assembled)
                assembled.append(index)
                Planner.UpdateElements(assembled, element_object_list)
            with pp.LockRenderer():
                for element_obj in element_object_list:
                    if element_obj.status == ElementStatus.float:
                        pp.set_color(element_obj.body, pp.YELLOW)
                    elif element_obj.status == ElementStatus.rotate:
                        pp.set_color(element_obj.body, pp.GREEN)
                    elif element_obj.status == ElementStatus.fixed:
                        pp.set_color(element_obj.body, pp.RED)
                        pp.set_pose(element_obj.body, element_obj.goal_pose)
                    else:
                        pp.set_color(element_obj.body, (1, 0, 0, 0.1))
                for index in index_list:
                    pp.set_color(element_object_list[index].body, pp.GREY)

            pp.wait_for_user(
                f"step: {step_num+1}/{len(path_index)} ,cur update index: {index_list}/0~{len(element_bodies)-1}"
            )

        # -------------------- Path Visualize --------------------#
        assembled_list = []
        for element_obj in element_object_list:
            pp.set_color(element_obj.body, (1, 0, 0, 0.1))
            pp.set_pose(element_obj.body, pp.Pose(point=(5, 0, 0), euler=pp.Euler(0, 1.5708, 0)))
        for step_num, index_list in enumerate(path_index):
            index = index_list[0] # TODO: 考虑多机器人协作的问题
            path = path_storage.get(index)
            base_path_item = path.base_path
            path_item = path.manipulator_path
            base_path_item: PathItem
            path_item: PathItem

            p.removeAllUserParameters()

            base_param_slider = p.addUserDebugParameter("base playback", 0.0, 1.0, 0.0)
            manipulator_param_slider = p.addUserDebugParameter("manipulator playback", 0.0, 1.0, 0.0)
            continue_button = p.addUserDebugParameter("continue", 1, 0, 0)
            prev_continue_button_value = p.readUserDebugParameter(continue_button)

            manipulator_traj = path_item.conf
            manipulator_traj_grasp_mask = path_item.mask
            grasp_attach = path_item.attach

            if base_path_item is not None:
                base_traj = base_path_item.conf
                base_traj_grasp_mask = base_path_item.mask
        
            pp.set_color(path_item.element_index, pp.RED)

            for i in assembled_list:
                pp.set_pose(element_from_index[i].body, element_from_index[i].goal_pose)

            is_first = True

            while True:
                current_continue_button_value = p.readUserDebugParameter(continue_button)
                if current_continue_button_value > prev_continue_button_value:
                    assembled_list.append(i)
                    prev_continue_button_value = current_continue_button_value
                    break

                manipulator_traj_param_value = p.readUserDebugParameter(manipulator_param_slider)
                manipulator_traj_idx = int(manipulator_traj_param_value * (len(manipulator_traj) - 1))
                
                if base_path_item is not None:
                    base_traj_param_value = p.readUserDebugParameter(base_param_slider)
                    base_traj_idx = int(base_traj_param_value * (len(base_traj) - 1))

                traj_pose = manipulator_traj[manipulator_traj_idx]
                grasp_attach_flag = manipulator_traj_grasp_mask[manipulator_traj_idx]

                if base_path_item != None and manipulator_traj_idx == 0:
                    traj_pose = base_traj[base_traj_idx]
                    grasp_attach_flag = base_traj_grasp_mask[base_traj_idx]
                robots[0].robot_setup.set_joint_positions(robots[0].robot_setup.control_joints, traj_pose)

                if is_first:
                    robots[0].UpdateElementsRobot(path_item.element_index)
                    is_first = False

                if grasp_attach_flag:
                    grasp_attach.assign()
