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

from curobo.types.base import TensorDeviceType

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from motion_planner.trajectory_curobo_solver import TrajectoryCuroboSolver
from motion_planner.trajectory_ompl_solver import TrajectoryOMPLSolver
from multi_tangent.collision import create_collision_bodies
from multi_tangent.convert import flatten_list
from robot.robot_setup import RobotSetup
from utils.collision import init_pb
from utils.params import *
from utils.utils import SetSeeds, HideOutput
from model.scene_parse import SceneParser


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
            self.result = self.func(*self.args, **self.kwargs)
            self.done = True
        except Exception as e:
            print(f"\nPlanning error: {e}")
            self.done = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corner case for transfer planning")

    parser.add_argument("--birrt", action="store_true", help="Enable BIRRT planning")
    parser.add_argument("--curobo", action="store_true", help="Enable cuRobo planning")
    # parser.add_argument("--eitstar", action="store_true", help="Enable OMPL ETIStar planning")
    parser.add_argument("--save", action="store_true", help="Whether to save the results")
    parser.add_argument("--random", action="store_true", help="Enable random planning")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--manual", action="store_true", help="Enable manual control")
    parser.add_argument("--confirm", action="store_true", help="Enable manual confirm")
    parser.add_argument("--validation", action="store_true", help="Enable validation")

    parser.add_argument("--repeat", type=int, default=1, help="Number of repetitions for the planning")
    parser.add_argument("--scene", type=str, default="cuboid_1", help="Scene name")
    parser.add_argument("--task", type=str, default="task_1", help="Task number")
    parser.add_argument("--max_attempts", type=int, default=-1, help="Maximum number of attempts")
    args = parser.parse_args()

    init_pb()

    scene_parser = SceneParser(os.path.join(HERE, "model", "scenes", f"{args.scene}", f"{args.task}.yml"))
    scene_parser.load_scene()

    line_pts_flattened, radius_per_edge = scene_parser.get_element_info()
    element_bodies = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)

    rb = RobotSetup("r0")
    robot_pose_2d = scene_parser.get_robot_pose_2d()
    pp.set_pose(
        rb.robot, pp.Pose(point=[robot_pose_2d[0], robot_pose_2d[1], 0], euler=pp.Euler(0, 0, robot_pose_2d[2]))
    )

    line_pts_grasped = [np.array([0, 0, 0]), np.array([0, 0, 1])]
    grasped_element = create_collision_bodies(line_pts_grasped, [0.01], viewer=True)[0]
    pp.set_pose(
        grasped_element,
        pp.multiply(
            pp.get_link_pose(rb.robot, rb.tool_link),
            pp.Pose(point=scene_parser.get_robot_grasp_offset(), euler=pp.Euler(1.5708, 0, 0)),
        ),
    )
    grasp_attachment = pp.create_attachment(rb.robot, rb.tool_link, grasped_element)

    target_q = np.array(scene_parser.get_robot_target_pose())
    init_q = np.array(scene_parser.get_robot_start_pose())

    success_counts = {"BIRRT": 0, "cuRobo": 0}

    # Create data directories if they don't exist
    if args.save:
        data_dir = os.path.join(HERE, "model", "data", args.scene, args.task)
        for planner in ["BIRRT", "cuRobo"]:
            planner_dir = os.path.join(data_dir, planner)
            os.makedirs(planner_dir, exist_ok=True)

    if args.confirm:
        pp.wait_for_user("Press Enter to start generation...")

    loop_count = 0
    max_attempts = args.max_attempts if args.max_attempts != -1 else float("inf")

    while (args.birrt and success_counts["BIRRT"] < args.repeat) or (
        args.curobo and success_counts["cuRobo"] < args.repeat
    ):
        if loop_count >= max_attempts:
            print(f"Max attempts reached: {max_attempts}")
            break

        if args.validation:
            if success_counts["BIRRT"] + success_counts["cuRobo"] > 0:
                break

        if args.random:
            seed = int.from_bytes(os.urandom(4), byteorder="big")
            SetSeeds(seed)
            print(f"\n-------------------- current seed: {seed} --------------------\n")

        # **************************************************************************
        # BIRRT plan
        # **************************************************************************

        if args.birrt and success_counts["BIRRT"] < args.repeat:
            print("\n========================================")
            print(f"{loop_count}th BIRRT planning (success: {success_counts['BIRRT']}/{args.repeat})")
            print("========================================\n")

            # Create and start planning thread
            planning_thread = PlanningThread(
                rb.plan_manipulator_path,
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

                if path is not None:
                    print(f"\rPlan success! Total time: {elapsed_time:.2f} s!", flush=True)
                    success_counts["BIRRT"] += 1

                    # Save trajectory if requested
                    if args.save:
                        plan_id = len(os.listdir(os.path.join(data_dir, "BIRRT")))
                        save_path = os.path.join(data_dir, "BIRRT", f"plan_{plan_id}.npy")
                        np.save(save_path, np.array(path))
                        print(f"Saved trajectory to {save_path}")

                    if args.visualize:
                        p.removeAllUserParameters()
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

        if args.curobo and success_counts["cuRobo"] < args.repeat:
            print("\n========================================")
            print(f"{loop_count}th cuRobo planning (success: {success_counts['cuRobo']}/{args.repeat})")
            print("========================================\n")

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

                if result["success"]:
                    print(f"\rPlan success! Total time: {elapsed_time:.2f} s! ", flush=True)
                    success_counts["cuRobo"] += 1
                    path = result["path"]

                    if args.visualize:
                        p.removeAllUserParameters()
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

                    # Save trajectory if requested
                    if args.save:
                        plan_id = len(os.listdir(os.path.join(data_dir, "cuRobo")))
                        save_path = os.path.join(data_dir, "cuRobo", f"plan_{plan_id}.npy")
                        np.save(save_path, np.array(path))
                        print(f"Saved trajectory to {save_path}")

                else:
                    print(f"\rCurobo plan failed, total time: {elapsed_time:.2f} s! ", flush=True)

            except KeyboardInterrupt:
                print("\nexit!")
                exit()

        loop_count += 1

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
