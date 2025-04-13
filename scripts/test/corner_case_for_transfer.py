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
from model.scene_parse import SceneParser
from motion_planner.trajectory_curobo_solver import TrajectoryCuroboSolver
from motion_planner.trajectory_ompl_solver import TrajectoryOMPLSolver
from motion_planner.trajectory_tampor_solver import TrajectoryTAMPORSolver
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
            with pp.LockRenderer():
                self.result = self.func(*self.args, **self.kwargs)
            self.done = True
        except Exception as e:
            print(f"\nPlanning error: {e}")
            self.done = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corner case for transfer planning")
    parser.add_argument("--birrt", action="store_true", help="Enable BIRRT planning")
    parser.add_argument("--curobo", action="store_true", help="Enable cuRobo planning")
    parser.add_argument("--tampor", action="store_true", help="Enable TAMPOR planning")
    parser.add_argument("--ompl", nargs="+", default=[], choices=["RRTConnect", "BITstar", "EITstar", "RRTstar", "PRM", "EST", "FMT"], help="OMPL algorithms to use")
    parser.add_argument("--manual", action="store_true", help="Enable manual control")
    parser.add_argument("--repeat", type=int, default=1, help="Number of repetitions for the planning")
    parser.add_argument("--save", action="store_true", help="Whether to save the results")
    parser.add_argument("--visualize", action="store_true", help="Whether to visualize the results")
    parser.add_argument("--random", action="store_true", help="Enable random planning")
    parser.add_argument("--scene", type=str, default="cuboid_1", help="Scene name")
    parser.add_argument("--task", type=str, default="task_1", help="Task number")
    parser.add_argument("--max_time", type=float, default=600.0, help="Maximum time for the planning")
    args = parser.parse_args()

    init_pb()

    # 设置保存路径的时间戳
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 使用SceneParser加载场景
    scene_parser = SceneParser(os.path.join(HERE, "model", "scenes", f"{args.scene}", f"{args.task}.yml"))
    scene_parser.load_scene()

    # 获取场景信息
    line_pts_flattened, radius_per_edge = scene_parser.get_element_info()
    element_bodies = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)
    channel_info = scene_parser.get_channel_info()
    grasp_offset = scene_parser.get_robot_grasp_offset()
    pose_2d = scene_parser.get_robot_pose_2d(output_type="array")
    start_q = np.array(scene_parser.get_robot_start_pose())
    target_q = np.array(scene_parser.get_robot_target_pose())

    # 设置机器人
    rb = RobotSetup("r0")
    pp.set_pose(rb.robot, pp.Pose(point=[pose_2d[0], pose_2d[1], 0], euler=pp.Euler(0, 0, pose_2d[2])))

    # 设置抓取物体
    line_pts_grasped = [np.array([0, 0, 0]), np.array([0, 0, 1])]
    grasped_element = create_collision_bodies(line_pts_grasped, [0.01], viewer=True)[0]
    pp.set_pose(grasped_element, pp.multiply(pp.get_link_pose(rb.robot, rb.tool_link), pp.Pose(point=grasp_offset, euler=pp.Euler(1.5708, 0, 0))))
    grasp_attachment = pp.create_attachment(rb.robot, rb.tool_link, grasped_element)
    rb.update_attachments([grasp_attachment])

    results = {"BIRRT": [], "cuRobo": [], "TAMPOR": []}

    # 为每个OMPL算法初始化结果存储
    for algo in args.ompl:
        results[f"OMPL_{algo}"] = []

    for repeat_id in range(args.repeat):
        if args.random:
            seed = int.from_bytes(os.urandom(4), byteorder="big")
            print(f"\n-------------------- current seed: {seed} --------------------\n")

        # **************************************************************************
        # BIRRT plan
        # **************************************************************************

        if args.birrt:
            print("\n========================================")
            print(f"{repeat_id+1}th BIRRT planning")
            print("========================================\n")

            SetSeeds(seed)

            planning_thread = PlanningThread(rb.plan_manipulator_path, start_q, target_q, rb.attachments, element_bodies, max_time=args.max_time, max_iterations=10000)
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
                    cur_result = (seed, path is not None, elapsed_time)
                    
                    # 保存路径
                    if args.save:
                        save_dir = os.path.join(LOG_DIR, "corner_case", time_stamp, args.scene, args.task, "BIRRT")
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{repeat_id}.npy")
                        np.save(save_path, path)
                else:
                    cur_result = (seed, False, args.max_time)
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

            SetSeeds(seed)

            p.removeAllUserParameters()

            curobo_planner = TrajectoryCuroboSolver(rb, TensorDeviceType())
            planning_thread = PlanningThread(curobo_planner.plan, start_q, target_q, args.max_time, 10000, element_bodies, grasped_element=grasped_element, grasped_attachment=grasp_attachment)

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
                    cur_result = (seed, result["success"], elapsed_time)
                    path = result["path"]
                    
                    # 保存路径
                    if args.save:
                        save_dir = os.path.join(LOG_DIR, "corner_case", time_stamp, args.scene, args.task, "cuRobo")
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{repeat_id}.npy")
                        np.save(save_path, path)
                else:
                    cur_result = (seed, False, args.max_time)
                results["cuRobo"].append(cur_result)

                if result["success"]:
                    print(f"\rPlan success! Total time: {elapsed_time:.2f} s! ", flush=True)
                    path = result["path"]
                    if args.visualize:
                        input = pp.wait_for_user("\nvisualize planned path?")
                        if input == "y" or input == "Y":
                            # for body in element_bodies[:10]:
                            #     pp.set_color(body, [0, 0, 1, 1])
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
                            # for body in element_bodies[:10]:
                            #     pp.set_color(body, [1, 0, 0, 1])
                else:
                    print(f"\rCurobo plan failed, total time: {elapsed_time:.2f} s! ", flush=True)

            except KeyboardInterrupt:
                print("\nexit!")
                exit()

        # **************************************************************************
        # ompl plan
        # **************************************************************************

        for ompl_algo in args.ompl:
            print("\n========================================")
            print(f"{repeat_id+1}th OMPL {ompl_algo} planning")
            print("========================================\n")

            SetSeeds(seed)

            p.removeAllUserParameters()
            
            collision_fn = rb.create_collision_fn(element_bodies)

            ompl_planner = TrajectoryOMPLSolver(collision_fn, planner=ompl_algo, robot_id=rb.robot, arm_joints=rb.arm_joints)

            planning_thread = PlanningThread(ompl_planner.plan, start_q, target_q, max_time=args.max_time)

            planning_thread.start()

            start_time = time.time()
            try:
                while not planning_thread.done:
                    elapsed_time = time.time() - start_time
                    print(f"\rPlanning... current time: {elapsed_time:.2f} s ", end="", flush=True)
                    time.sleep(0.1)

                elapsed_time = time.time() - start_time
                path = planning_thread.result

                if path["success"]:
                    cur_result = (seed, path["success"], elapsed_time)
                    
                    # 保存路径
                    if args.save:
                        save_dir = os.path.join(LOG_DIR, "corner_case", time_stamp, args.scene, args.task, f"OMPL_{ompl_algo}")
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{repeat_id}.npy")
                        np.save(save_path, path["path"])
                else:
                    cur_result = (seed, False, args.max_time)
                results[f"OMPL_{ompl_algo}"].append(cur_result)

                if path["success"]:
                    print(f"\rPlan success! Total time: {elapsed_time:.2f} s!", flush=True)
                    if args.visualize:
                        input = pp.wait_for_user(f"\nvisualize {ompl_algo} planned path?")
                        if input == "y" or input == "Y":
                            replay_slider = p.addUserDebugParameter("replay", 0, 1, 0)
                            continue_button = p.addUserDebugParameter("continue", 1, 0, 0)
                            prev_continue_button_value = p.readUserDebugParameter(continue_button)
                            while True:
                                replay = p.readUserDebugParameter(replay_slider)
                                current_continue_button_value = p.readUserDebugParameter(continue_button)
                                idx = int(replay * (len(path["path"]) - 1))
                                conf = path["path"][idx]
                                rb.set_joint_positions(rb.arm_joints, conf)
                                grasp_attachment.assign()
                                print(f"collision_fn: {collision_fn(conf)}, joint_positions: {conf}")
                                time.sleep(1.0 / 240)
                                if current_continue_button_value > prev_continue_button_value:
                                    break
                                prev_continue_button_value = current_continue_button_value
                else:
                    print(f"\rOMPL {ompl_algo} plan failed, total time: {elapsed_time:.2f} s!", flush=True)

            except KeyboardInterrupt:
                print("\nexit!")
                exit()

        # **************************************************************************
        # TAMPOR plan
        # **************************************************************************

        if args.tampor:
            print("\n========================================")
            print(f"{repeat_id+1}th TAMPOR planning")
            print("========================================\n")

            SetSeeds(seed)

            tampor_planner = TrajectoryTAMPORSolver(rb, channel_info, grasp_offset, eval_max_attempts=1000)
            planning_thread = PlanningThread(tampor_planner.plan, start_q, target_q, element_bodies, grasp_attachment, key_frame_num=25, grow_tree_max_nodes=1000, max_time=args.max_time, step_max_time=30.0, verbose=False)

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
                    cur_result = (seed, result["success"], elapsed_time)
                    
                    # 保存路径
                    if args.save and "path" in result:
                        path = result["path"]
                        save_dir = os.path.join(LOG_DIR, "corner_case", time_stamp, args.scene, args.task, "TAMPOR")
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{repeat_id}.npy")
                        np.save(save_path, path)
                else:
                    cur_result = (seed, False, args.max_time)
                results["TAMPOR"].append(cur_result)

                if result["success"]:
                    print(f"\rPlan success! Total time: {elapsed_time:.2f} s! ", flush=True)
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
                    print(f"\rTAMPOR plan failed, total time: {elapsed_time:.2f} s! ", flush=True)

            except KeyboardInterrupt:
                print("\nexit!")
                exit()

        # 保存结果到日志文件
        if args.save:
            log_dir = os.path.join(LOG_DIR, "corner_case", time_stamp)
            os.makedirs(log_dir, exist_ok=True)
            file_path = os.path.join(log_dir, "log.json")
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
                current_record = np.load("/home/jeong/summer_research/eth_ws/src/husky_assembly/scripts/record_corner.npy")
                for i in range(len(current_record)):
                    rb.set_joint_positions(rb.arm_joints, np.array(current_record[i]))
                    grasp_attachment.assign()
                    time.sleep(1.0 / 60)
            prev_replay_button_value = current_replay_button_value

            time.sleep(1.0 / 240)
