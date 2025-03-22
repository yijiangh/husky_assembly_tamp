import argparse
import json
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
from motion_planner.trajectory_curobo_solver import TrajectoryCuroboSolver
from motion_planner.trajectory_ompl_solver import TrajectoryOMPLSolver
from multi_tangent.collision import create_collision_bodies
from multi_tangent.convert import flatten_list
from robot.robot_setup import RobotSetup
from utils.collision import Element, create_couplers, init_pb
from utils.params import *
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

    parser = argparse.ArgumentParser(description="Corner case for transfer planning")
    parser.add_argument("--birrt", action="store_true", help="Enable BIRRT planning")
    parser.add_argument("--curobo", action="store_true", help="Enable cuRobo planning")
    parser.add_argument("--eitstar", action="store_true", help="Enable OMPL ETIStar planning")
    parser.add_argument("--manual", action="store_true", help="Enable manual control")
    parser.add_argument("--repeat", type=int, default=1, help="Number of repetitions for the planning")
    parser.add_argument("--save", action="store_true", help="Whether to save the results")
    parser.add_argument("--visualize", action="store_true", help="Whether to visualize the results")
    parser.add_argument("--random", action="store_true", help="Enable random planning")
    args = parser.parse_args()

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

    for body in element_bodies:
        print(f"body: {body}, pose: {pp.get_pose(body)}")

    pp.wait_for_user("Press Enter to exit...")
    exit()

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
    grasp_attachment = pp.create_attachment(rb.robot, rb.tool_link, grasped_element)

    target_q = np.array([1.323, 0.331, -1.753, -0.397, 1.653, 1.819])
    init_q = np.array([np.pi / 2, 0.0, 0.0, -np.pi, 0.0, -np.pi / 2])

    results = {"BIRRT": [], "cuRobo": [], "EITStar": []}

    for repeat_id in range(args.repeat):

        if args.random:
            seed = int.from_bytes(os.urandom(4), byteorder="big")
            SetSeeds(seed)
            print(f"\n-------------------- current seed: {seed} --------------------\n")

        # **************************************************************************
        # BIRRT plan
        # **************************************************************************

        if args.birrt:
            print("\n========================================")
            print(f"{repeat_id+1}th BIRRT planning")
            print("========================================\n")

            # 创建并启动规划线程，将计划函数作为参数传入
            planning_thread = PlanningThread(
                rb.plan_manipulator_path,  # 传入规划函数
                init_q,
                target_q,
                [grasp_attachment],
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

                cur_result = (seed, path is not None, elapsed_time)
                results["BIRRT"].append(cur_result)

                if path is not None:
                    print(f"\rPlan success! Total time: {elapsed_time:.2f} s!", flush=True)
                    if args.visualize:
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
                                grasp_attachment.assign()
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

        if args.curobo:
            print("\n========================================")
            print(f"{repeat_id+1}th cuRobo planning")
            print("========================================\n")

            p.removeAllUserParameters()

            curobo_planner = TrajectoryCuroboSolver(rb, TensorDeviceType())
            planning_thread = PlanningThread(
                curobo_planner.plan,
                init_q,
                target_q,
                600,
                10000,
                element_bodies,
                grasped_element=grasped_element,
                grasped_attachment=grasp_attachment,
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

                cur_result = (seed, result["success"], elapsed_time)
                results["cuRobo"].append(cur_result)

                if result["success"]:
                    print(f"\rPlan success! Total time: {elapsed_time:.2f} s! ", flush=True)
                    path = result["path"]
                    # print("Plan result: \n", path)
                    if args.visualize:
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
                                grasp_attachment.assign()
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
        # ompl plan
        # **************************************************************************

        planner = "EITstar"

        if args.eitstar:
            print("\n========================================")
            print(f"{repeat_id+1}th EITStar planning")
            print("========================================\n")

            p.removeAllUserParameters()

            extra_disabled_collisions = [
                (
                    (rb.robot, pp.link_from_name(rb.robot, "ur_arm_wrist_3_link")),
                    (rb.ee_attachment.child, pp.BASE_LINK),
                ),
                (
                    (rb.ee_attachment.child, pp.BASE_LINK),
                    (grasp_attachment.child, pp.BASE_LINK),
                ),
            ]

            collision_fn = pp.get_collision_fn(
                rb.robot,
                rb.arm_joints,
                obstacles=element_bodies,
                attachments=[grasp_attachment, rb.ee_attachment] + rb.attachments,
                self_collisions=True,
                disabled_collisions=rb.disabled_collisions,
                extra_disabled_collisions=extra_disabled_collisions,
                max_distance=0.0,
            )

            ompl_planner = TrajectoryOMPLSolver(collision_fn, planner=planner)

            planning_thread = PlanningThread(ompl_planner.plan, init_q, target_q, time=600)

            planning_thread.start()

            start_time = time.time()
            try:
                while not planning_thread.done:
                    elapsed_time = time.time() - start_time
                    print(f"\rPlanning... current time: {elapsed_time:.2f} s ", end="", flush=True)
                    time.sleep(0.1)

                elapsed_time = time.time() - start_time
                path = planning_thread.result

                cur_result = (seed, path is not None, elapsed_time)
                results["EITStar"].append(cur_result)

                if path is not None:
                    print(f"\rPlan success! Total time: {elapsed_time:.2f} s!", flush=True)
                    if args.visualize:
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
                                grasp_attachment.assign()
                                time.sleep(1.0 / 240)
                                if current_continue_button_value > prev_continue_button_value:
                                    break
                                prev_continue_button_value = current_continue_button_value
                else:
                    print(f"\r{planner} plan failed, total time: {elapsed_time:.2f} s!", flush=True)

            except KeyboardInterrupt:
                print("\nexit!")
                exit()

    if args.save:
        file_path = os.path.join(LOG_DIR, "corner_case_for_transfer.json")
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)

    # **************************************************************************
    # manual control
    # **************************************************************************

    if args.manual:

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
            grasp_attachment.assign()

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
                    grasp_attachment.assign()
                    time.sleep(1.0 / 60)
            prev_replay_button_value = current_replay_button_value

            time.sleep(1.0 / 240)
