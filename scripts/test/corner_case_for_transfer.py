import argparse
import os
import random
import signal
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import pybullet as p
import pybullet_planning as pp

# Third Party
import torch

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import Capsule, Cuboid, Cylinder, Mesh, Sphere, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from multi_tangent.collision import create_collision_bodies
from multi_tangent.convert import flatten_list
from robot.robot_setup import RobotSetup
from utils.collision import Element, create_couplers, init_pb
from utils.utils import CounterModule, SetSeeds


class PlanningThread(threading.Thread):
    def __init__(self, func, *args, **kwargs):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.done = False
        self.daemon = True

    def run(self):
        try:
            # 执行传入的函数并保存结果
            self.result = self.func(*self.args, **self.kwargs)
            self.done = True
        except Exception as e:
            print(f"\nPlanning error: {e}")
            self.done = True


if __name__ == "__main__":
    # seed = 128363
    # SetSeeds(seed)

    init_pb()
    line_pts_flattened = [
        # 4根竖着的棍子
        np.array([0, 0, 0]),
        np.array([0, 0, 1]),
        np.array([0, 1, 0]),
        np.array([0, 1, 1]),
        np.array([1, 1, 0]),
        np.array([1, 1, 1]),
        np.array([1, 0, 0]),
        np.array([1, 0, 1]),
        # 4根横着的棍子
        np.array([0, 0, 1]),
        np.array([1, 0, 1]),
        np.array([1, 0, 1]),
        np.array([1, 1, 1]),
        np.array([1, 1, 1]),
        np.array([0, 1, 1]),
        np.array([0, 1, 1]),
        np.array([0, 0, 1]),
        # 4根内部横着的棍子
        np.array([0, 0.2, 1]),
        np.array([1, 0.2, 1]),
        np.array([0.6, 0, 1]),
        np.array([0.6, 1, 1]),
        np.array([1, 0.8, 1]),
        np.array([0, 0.8, 1]),
        np.array([0.2, 1, 1]),
        np.array([0.2, 0, 1]),
    ]

    radius_per_edge = [0.01] * int(len(line_pts_flattened) / 2)
    element_bodies = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)

    rb = RobotSetup("r0")
    pp.set_pose(rb.robot, pp.Pose(point=(-0.5, 0.5, 0), euler=pp.Euler(0, 0, 0)))

    line_pts_grasped = [np.array([0, 0, 0]), np.array([0, 0, 1])]
    grasped_element = create_collision_bodies(line_pts_grasped, [0.01], viewer=True)[0]
    pp.set_pose(
        grasped_element,
        pp.multiply(
            pp.get_link_pose(rb.robot, rb.tool_link), pp.Pose(point=(0, 0.1, 0.15), euler=pp.Euler(1.5708, 0, 0))
        ),
    )
    attachment = pp.create_attachment(rb.robot, rb.tool_link, grasped_element)

    target_q = np.array([1.323, 0.331, -1.753, -0.397, 1.653, 1.819])
    init_q = np.array([np.pi / 2, 0.0, 0.0, -np.pi, 0.0, -np.pi / 2])

    # **************************************************************************
    # BIRRT plan
    # **************************************************************************

    input = pp.wait_for_user("start birrt plan?")

    path = None
    if input == "y" or input == "Y":

        # 创建并启动规划线程，将计划函数作为参数传入
        planning_thread = PlanningThread(
            rb.plan_manipulator_path,  # 传入规划函数
            init_q,
            target_q,
            [attachment],
            element_bodies,
            max_time=600,
            max_iterations=10000,
        )
        planning_thread.start()

        start_time = time.time()
        try:
            while not planning_thread.done:
                elapsed_time = time.time() - start_time
                print(f"\rPlanning... current time: {elapsed_time:.2f} s", end="", flush=True)
                time.sleep(0.1)

            elapsed_time = time.time() - start_time
            path = planning_thread.result

            if path is not None:
                print(f"\rPlan success! Total time: {elapsed_time:.2f} s!", flush=True)
                input = pp.wait_for_user("\nvisualize planned path?")
                if input == "y" or input == "Y":
                    replay_slider = p.addUserDebugParameter("replay", 0, 1, 0)
                    continue_button = p.addUserDebugParameter("continue", 1, 0, 0)
                    prev_continue_button_value = p.readUserDebugParameter(continue_button)
                    while True:
                        replay = p.readUserDebugParameter(replay_slider)
                        current_continue_button_value = p.readUserDebugParameter(continue_button)
                        idx = int(replay * (len(path) - 1))
                        conf = path[idx]
                        rb.set_joint_positions(rb.arm_joints, conf)
                        attachment.assign()
                        time.sleep(1.0 / 240)
                        if current_continue_button_value > prev_continue_button_value:
                            break
                        prev_continue_button_value = current_continue_button_value
            else:
                print(f"\rBIRRT plan failed, total time: {elapsed_time:.2f} s!", flush=True)

        except KeyboardInterrupt:
            print("\nexit!")
            exit()

    # **************************************************************************
    # curobo plan
    # **************************************************************************

    input = pp.wait_for_user("\nstart curobo plan?")

    rb.set_joint_positions(rb.arm_joints, target_q)
    attachment.assign()

    if input == "y" or input == "Y":
        p.removeAllUserParameters()
        tensor_args = TensorDeviceType()

        # load robot
        config_file = load_yaml(join_path(get_robot_path(), "husky_ur5_e.yml"))["robot_cfg"]
        robot_cfg = RobotConfig.from_dict(config_file, tensor_args)
        kin_model = CudaRobotModel(robot_cfg.kinematics)

        # load obstacles
        obstacles = []
        for element_body in element_bodies:
            pose = pp.multiply(pp.invert(pp.get_pose(rb.robot)), pp.get_pose(element_body))
            point = list(pose[0])
            quat = [pose[1][3], pose[1][0], pose[1][1], pose[1][2]]
            obstacle = Cylinder(
                name=f"element_{element_body}",
                radius=0.01,
                height=1.0,
                pose=point + quat,
                color=[1.0, 0, 0, 1.0],
            )
            obstacles.append(obstacle)

        q_init = torch.tensor(init_q, dtype=torch.float32).reshape(1, 6).cuda()
        q_target = torch.tensor(target_q, dtype=torch.float32).reshape(1, 6).cuda()

        # load grasp object
        rb.set_joint_positions(rb.arm_joints, init_q)
        attachment.assign()
        grasp_element_pose = pp.multiply(pp.invert(pp.get_pose(rb.robot)), pp.get_pose(grasped_element))
        grasp_pose_point = list(grasp_element_pose[0])
        grasp_pose_quat = [
            grasp_element_pose[1][3],
            grasp_element_pose[1][0],
            grasp_element_pose[1][1],
            grasp_element_pose[1][2],
        ]
        grasp_object = Cylinder(
            name="grasp_element",
            radius=0.01,
            height=1.0,
            pose=grasp_pose_point + grasp_pose_quat,
            color=[1.0, 0, 0, 1.0],
        )

        # create world config
        world_config = WorldConfig(cylinder=obstacles + [grasp_object])
        world_config_obb = WorldConfig.create_obb_world(world_config)

        # world_config_obb.save_world_as_mesh("debug_mesh.obj")

        # create motion gen
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            "husky_ur5_e.yml", world_config_obb, interpolation_dt=0.01
        )
        motion_gen = MotionGen(motion_gen_config)
        motion_gen.warmup()

        # attach element
        init_js = JointState(
            position=q_init, velocity=q_init, acceleration=q_init, jerk=q_init, joint_names=robot_cfg.cspace.joint_names
        )
        motion_gen.attach_objects_to_robot(
            init_js, ["grasp_element"], sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE
        )

        # set target and start
        out = kin_model.get_state(q_target)
        goal_pose = Pose.from_list(
            out.ee_pose.position.cpu().numpy().flatten().tolist()
            + out.ee_pose.quaternion.cpu().numpy().flatten().tolist()
        )
        start_state = JointState.from_position(q_init, joint_names=robot_cfg.cspace.joint_names)

        # 创建并启动 Curobo 规划线程
        planning_thread = PlanningThread(
            motion_gen.plan_single,
            start_state,
            goal_pose,
            MotionGenPlanConfig(max_attempts=6000, timeout=600),
        )
        planning_thread.start()

        start_time = time.time()
        try:
            while not planning_thread.done:
                elapsed_time = time.time() - start_time
                print(f"\rPlanning... current time: {elapsed_time:.2f} s ", end="", flush=True)
                time.sleep(0.1)

            elapsed_time = time.time() - start_time
            result = planning_thread.result

            if result.success:
                print(f"\rPlan success! Total time: {elapsed_time:.2f} s! ", flush=True)
                path = result.get_interpolated_plan().position.cpu().numpy()
                # print("Plan result: \n", path)
                input = pp.wait_for_user("\nvisualize planned path? ")
                if input == "y" or input == "Y":
                    replay_slider = p.addUserDebugParameter("replay", 0, 1, 0)
                    continue_button = p.addUserDebugParameter("continue", 1, 0, 0)
                    prev_continue_button_value = p.readUserDebugParameter(continue_button)
                    while True:
                        replay = p.readUserDebugParameter(replay_slider)
                        current_continue_button_value = p.readUserDebugParameter(continue_button)
                        idx = int(replay * (len(path) - 1))
                        conf = path[idx]
                        rb.set_joint_positions(rb.arm_joints, conf)
                        attachment.assign()
                        time.sleep(1.0 / 240)
                        if current_continue_button_value > prev_continue_button_value:
                            break
                        prev_continue_button_value = current_continue_button_value
            else:
                print(f"\rCurobo plan failed, total time: {elapsed_time:.2f} s! ", flush=True)

        except KeyboardInterrupt:
            print("\nexit!")
            exit()

    # **************************************************************************
    # manual control
    # **************************************************************************

    input = pp.wait_for_user("\nstart manual control?")

    if input == "y" or input == "Y":

        p.removeAllUserParameters()

        j0_slider = p.addUserDebugParameter("joint contorl j0", -2 * np.pi, 2 * np.pi, target_q[0])
        j1_slider = p.addUserDebugParameter("joint contorl j1", -2 * np.pi, 2 * np.pi, target_q[1])
        j2_slider = p.addUserDebugParameter("joint contorl j2", -2 * np.pi, 2 * np.pi, target_q[2])
        j3_slider = p.addUserDebugParameter("joint contorl j3", -2 * np.pi, 2 * np.pi, target_q[3])
        j4_slider = p.addUserDebugParameter("joint contorl j4", -2 * np.pi, 2 * np.pi, target_q[4])
        j5_slider = p.addUserDebugParameter("joint contorl j5", -2 * np.pi, 2 * np.pi, target_q[5])

        record_button = p.addUserDebugParameter("record", 1, 0, 0)
        prev_record_button_value = p.readUserDebugParameter(record_button)

        save_button = p.addUserDebugParameter("save", 1, 0, 0)
        prev_save_button_value = p.readUserDebugParameter(save_button)

        replay_button = p.addUserDebugParameter("replay", 1, 0, 0)
        prev_replay_button_value = p.readUserDebugParameter(replay_button)

        record = [target_q.tolist()]
        last_conf = target_q

        while True:
            j0 = p.readUserDebugParameter(j0_slider)
            j1 = p.readUserDebugParameter(j1_slider)
            j2 = p.readUserDebugParameter(j2_slider)
            j3 = p.readUserDebugParameter(j3_slider)
            j4 = p.readUserDebugParameter(j4_slider)
            j5 = p.readUserDebugParameter(j5_slider)

            rb.set_joint_positions(rb.arm_joints, np.array([j0, j1, j2, j3, j4, j5]))
            attachment.assign()

            current_record_button_value = p.readUserDebugParameter(record_button)
            if current_record_button_value > prev_record_button_value:
                cur_conf = np.array([j0, j1, j2, j3, j4, j5])
                confs = np.linspace(last_conf, cur_conf, 120)
                for conf in confs:
                    record.append(conf.tolist())
                print("record success!")
                last_conf = cur_conf
            prev_record_button_value = current_record_button_value

            current_save_button_value = p.readUserDebugParameter(save_button)
            if current_save_button_value > prev_save_button_value:
                np.save("record.npy", np.array(record))
                print("save record success!")
            prev_save_button_value = current_save_button_value

            current_replay_button_value = p.readUserDebugParameter(replay_button)
            if current_replay_button_value > prev_replay_button_value:  # replay
                current_record = np.load(
                    "/home/jeong/summer_research/eth_ws/src/husky_assembly/scripts/record_corner.npy"
                )
                for i in range(len(current_record)):
                    rb.set_joint_positions(rb.arm_joints, np.array(current_record[i]))
                    attachment.assign()
                    time.sleep(1.0 / 60)
            prev_replay_button_value = current_replay_button_value

            time.sleep(1.0 / 240)
