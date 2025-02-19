import argparse
import os
import random
from copy import deepcopy
from datetime import datetime

import numpy as np
import pybullet as p
import pybullet_planning as pp

# -------------------- self-defined modules --------------------#
import utils.load_multi_tangent as load_multi_tangent
from multi_tangent.collision import create_collision_bodies
from multi_tangent.convert import flatten_list
from robot.robot import PathItem, PathWithIndex, Robot, ConcretePath
from robot.robot_setup import RobotSetup
from symbolic_planner.element_object import ElementObject, ElementStatus
from symbolic_planner.planner import Planner
from utils.collision import Element, create_couplers, init_pb
from utils.params import *
from utils.parse import parse_mt_geometric
from utils.utils import CounterModule

log_dir = os.path.join(HERE, f"logs/{MT_FILE_NAME}")

# mt_file_name = "tower_integral_one_len_MT_layer_0"
# grounded_elements_index = [0, 1, 4, 19]  # tower_integral_one_len_MT_layer_0

if __name__ == "__main__":

    # plan_manipulator_path 稳定失败种子
    # random.seed(128363)
    # np.random.seed(98765)

    with pp.HideOutput():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--mt_file_name",
            default=MT_FILE_NAME + ".json",
            help='The name of the multi tangent file to solve (json file\'s name, e.g. "tower_integral_one_len_MT_layer_0.json")',
        )
        parser.add_argument(
            "--visualization",
            type=int,
            default=1,
            help="Visualization flag in the form of int.",
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
                goal_poses.append(pp.get_pose(e))
                pp.set_pose(e, pp.Pose(point=(5, 0, 0), euler=pp.Euler(0, 1.5708, 0)))
        element_from_index = {
            i: Element(i, e, pp.get_pose(e), goal_poses[i], [line_pts_flattened[2 * i], line_pts_flattened[2 * i + 1]])
            for i, e in enumerate(element_bodies)
        }

        # -------------------- Path storage --------------------#
        path_storage = PathWithIndex()

        # -------------------- Counter module --------------------#
        counter = CounterModule()

        # -------------------- Robots Init --------------------#
        with pp.HideOutput():
            robots = []
            robot_num = ROBOT_NUM
            for i in range(robot_num):
                rb = RobotSetup(f"r{i}")
                robots.append(Robot(i, rb, element_from_index, counter, [], path_storage))

        grounded_elements_index = GROUNDED_ELEMENTS_INDEX

        # -------------------- Plan --------------------#
        planner = Planner(robot_num=robot_num, robots=robots)
        path_index = planner.Plan(element_from_index, contact_id_pairs, grounded_elements_index)
        element_object_list = Planner.GetElementObjects(element_from_index, contact_id_pairs, grounded_elements_index)

        # -------------------- save log file --------------------#
        current_time_str = datetime.now().strftime("%y%m%d_%H%M%S")
        counter.save(log_dir, f"{current_time_str}.json")

        # -------------------- Sequence Visualization --------------------#
        if args.visualization:
            assembled = []
            for step_num, index_list in enumerate(path_index):
                for element_index in index_list:
                    element_object_list[element_index].Assemble(assembled)
                    assembled.append(element_index)
                    Planner.UpdateElements(deepcopy(assembled), element_object_list)
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
                    for element_index in index_list:
                        pp.set_color(element_object_list[element_index].body, pp.GREY)
                        pp.draw_pose(pp.get_pose(element_object_list[element_index].body), length=0.5)

                pp.wait_for_user(
                    f"step: {step_num+1}/{len(path_index)} ,cur update index: {index_list}/0~{len(element_bodies)-1}"
                )

            if MANIPULATOR_PLANNER == "default":
                pass
            else:
                # -------------------- Path Visualize --------------------#
                assembled_list = []

                # init element far away
                for element_obj in element_object_list:
                    pp.set_color(element_obj.body, (1, 0, 0, 0.1))
                    pp.set_pose(element_obj.body, pp.Pose(point=(5, 0, 0), euler=pp.Euler(0, 1.5708, 0)))

                for step_num, index_list in enumerate(path_index):

                    # **************************************************************************
                    # 初始化GUI控件
                    # **************************************************************************

                    p.removeAllUserParameters()

                    base_sliders = []
                    manipulator_sliders = []

                    for element_index in index_list:
                        # 获取Path对应的robot index
                        path = path_storage.get(element_index)
                        robot_index = path.robot_index
                        # path_item: PathItem = path.manipulator_path
                        # base_path_item: PathItem = path.base_path

                        base_slider = p.addUserDebugParameter(f"base playback r{robot_index}", 0.0, 1.0, 0.0)
                        manipulator_slider = p.addUserDebugParameter(
                            f"manipulator playback r{robot_index}", 0.0, 1.0, 0.0
                        )
                        base_sliders.append(base_slider)
                        manipulator_sliders.append(manipulator_slider)

                    continue_button = p.addUserDebugParameter("continue", 1, 0, 0)
                    prev_continue_button_value = p.readUserDebugParameter(continue_button)

                    # **************************************************************************
                    # 获取Path
                    # **************************************************************************

                    manipulator_traj_list = []
                    manipulator_traj_mask_list = []
                    grasp_attach_list = []
                    robot_index_list = []
                    base_traj_list = []
                    base_traj_mask_list = []

                    for index, element_index in enumerate(index_list):

                        path: ConcretePath = path_storage.get(element_index)

                        robot_index_list.append(path.robot_index)

                        manipulator_path_item: PathItem = path.manipulator_path
                        base_path_item: PathItem = path.base_path

                        manipulator_traj_list.append(manipulator_path_item.conf)
                        manipulator_traj_mask_list.append(manipulator_path_item.mask)
                        grasp_attach_list.append(path.attachment)
                        pp.set_color(element_index, pp.RED)

                        if base_path_item is not None:
                            base_traj_list.append(base_path_item.conf)
                            base_traj_mask_list.append(base_path_item.mask)
                        else:
                            base_traj_list.append([])
                            base_traj_mask_list.append([])

                    # **************************************************************************
                    # 其他element初始化
                    # **************************************************************************

                    for i in assembled_list:
                        pp.set_pose(element_from_index[i].body, element_from_index[i].goal_pose)

                    # **************************************************************************
                    # 读取slider和button并设置关节角
                    # **************************************************************************

                    is_first = True

                    while True:

                        # 更新continue
                        current_continue_button_value = p.readUserDebugParameter(continue_button)
                        if current_continue_button_value > prev_continue_button_value:
                            for temp_index in index_list:
                                if temp_index not in assembled_list:
                                    assembled_list.append(temp_index)
                            prev_continue_button_value = current_continue_button_value
                            break

                        # 更新每个silder对应的轨迹
                        for (
                            manipulator_traj,
                            manipulator_traj_mask,
                            base_traj,
                            base_traj_mask,
                            grasp_attach,
                            robot_index,
                            manipulator_slider,
                            base_slider,
                            element_index,
                        ) in zip(
                            manipulator_traj_list,
                            manipulator_traj_mask_list,
                            base_traj_list,
                            base_traj_mask_list,
                            grasp_attach_list,
                            robot_index_list,
                            manipulator_sliders,
                            base_sliders,
                            index_list,
                        ):
                            manipulator_traj_param_value = p.readUserDebugParameter(manipulator_slider)
                            manipulator_traj_idx = int(manipulator_traj_param_value * (len(manipulator_traj) - 1))
                            traj_pose = manipulator_traj[manipulator_traj_idx]
                            grasp_attach_flag = manipulator_traj_mask[manipulator_traj_idx]

                            if len(base_traj) != 0 and manipulator_traj_idx == 0:
                                base_traj_param_value = p.readUserDebugParameter(base_slider)
                                base_traj_idx = int(base_traj_param_value * (len(base_traj) - 1))
                                traj_pose = base_traj[base_traj_idx]
                                grasp_attach_flag = base_traj_mask[base_traj_idx]

                            robots[robot_index].robot_setup.set_joint_positions(
                                robots[robot_index].robot_setup.control_joints, traj_pose
                            )

                            if is_first:
                                robots[robot_index].UpdateElementsRobot(element_index)

                            if grasp_attach_flag:
                                grasp_attach.assign()

                        is_first = False
