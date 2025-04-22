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
from model.scene_parse import SceneParser
from motion_planner.trajectory_curobo_solver import TrajectoryCuroboSolver
from motion_planner.trajectory_ompl_solver import TrajectoryOMPLSolver
from motion_planner.trajectory_tampor_solver import TrajectoryTAMPORSolver
from multi_tangent.collision import create_collision_bodies
from multi_tangent.convert import flatten_list
from robot.robot_setup import RobotSetup
from utils.collision import Element, create_couplers, init_pb
from utils.params import *
from utils.util import CounterModule, SetSeeds, PrintManager

# 初始化PrintManager实例
printer = PrintManager()


class PlanningThread(threading.Thread):
    def __init__(self, func, *args, **kwargs):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = {"success": False, "path": None}
        self.done = False
        self.daemon = True

    def run(self):
        try:
            with pp.LockRenderer():
                self.result = self.func(*self.args, **self.kwargs)
                if isinstance(self.result, dict):
                    pass
                elif isinstance(self.result, np.ndarray):
                    self.result = {"success": True, "path": self.result}
                elif self.result is None:
                    self.result = {"success": False, "path": None}
            self.done = True
        except Exception as e:
            printer.error(f"\n规划错误: {e}")
            self.done = True
            self.result = {"success": False, "path": None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corner case for transfer planning")

    parser.add_argument("--birrt", action="store_true", help="Enable BIRRT planning")
    parser.add_argument("--curobo", action="store_true", help="Enable cuRobo planning")
    parser.add_argument("--tampor", action="store_true", help="Enable TAMPOR planning")
    parser.add_argument("--ompl", nargs="+", default=[], choices=["RRTConnect", "BITstar", "EITstar", "RRTstar", "PRM", "EST", "FMT", "BFMT", "LazyRRT", "STRIDE"], help="OMPL algorithms to use")

    parser.add_argument("--save", action="store_true", help="Whether to save the results")

    parser.add_argument("--manual", action="store_true", help="Enable manual control")

    parser.add_argument("--visualize", action="store_true", help="Whether to visualize the results")

    parser.add_argument("--scene", type=str, default="cuboid_1", help="Scene name")
    parser.add_argument("--task", type=str, default="task_1", help="Task number")

    parser.add_argument("--seed", type=int, default=None, help="Fixed seed value for reproducible results")
    parser.add_argument("--repeat", type=int, default=1, help="Number of repetitions for the planning")
    parser.add_argument("--random", action="store_true", help="Enable random planning")

    parser.add_argument("--max_time", type=float, default=600.0, help="Maximum time for the planning")
    parser.add_argument("--joint_angles", type=str, help="Joint angles for manual mode (comma-separated, e.g. '1.0,0.5,-1.0,0.8,1.2,0.3')")
    args = parser.parse_args()

    init_pb()

    # 使用SceneParser加载场景
    scene_parser = SceneParser(os.path.join(HERE, "model", "scenes", f"{args.scene}", f"{args.task}.yml"))

    with pp.LockRenderer():
        # 设置机器人
        rb = scene_parser.create_robot("r0")
        # 设置抓取物体
        attachment_body, grasp_attachment, approximate_attachment_body, approximate_attachment = scene_parser.create_attachment(rb, approximate=True)
        rb.update_attachments([grasp_attachment])
        # 加载场景元素
        element_bodies, element_infos = scene_parser.create_elements(color=[1, 0, 0, 1])

    # 获取场景信息
    start_q = np.array(scene_parser.get_robot_start_pose())
    target_q = np.array(scene_parser.get_robot_target_pose())
    pose_2d = scene_parser.get_robot_pose_2d(output_type="array")
    channel_info = scene_parser.get_channel_info()
    grasp_pose = scene_parser.get_robot_grasp_pose()

    # 定义要执行的seeds
    seeds_to_run = []

    # 如果提供了seed参数
    if args.seed is not None:
        printer.info(f"将使用指定的seed: {args.seed}")
        seeds_to_run = [args.seed]
    # 如果提供了random和repeat参数
    elif args.random and args.repeat > 0:
        printer.info(f"将生成 {args.repeat} 个随机seed")
        for i in range(args.repeat):
            seed = int.from_bytes(os.urandom(4), byteorder="big")
            seeds_to_run.append(seed)
    # 如果是手动模式
    elif args.manual:
        seeds_to_run = []
    # 如果都不满足则退出
    else:
        parser.error("必须提供--seed 或 同时提供--random和--repeat参数")
        sys.exit(1)

    # 加载已有日志文件中的结果（如果存在）
    log_dir = os.path.join(HERE, "logs", args.scene, args.task)
    log_file_path = os.path.join(log_dir, "log.json")
    if os.path.exists(log_file_path):
        printer.info(f"从 {log_file_path} 加载已有结果")
        try:
            with open(log_file_path, "r") as f:
                results = json.load(f)

            # 确保所有必要的算法键存在
            for algo_key in ["BIRRT", "cuRobo", "TAMPOR"]:
                if algo_key not in results:
                    results[algo_key] = []

            # 确保所有OMPL算法键存在
            for algo in args.ompl:
                algo_key = f"OMPL_{algo}"
                if algo_key not in results:
                    results[algo_key] = []
        except Exception as e:
            printer.error(f"读取日志文件失败: {e}，将创建新的日志文件")
            results = {"BIRRT": [], "cuRobo": [], "TAMPOR": []}
            for algo in args.ompl:
                results[f"OMPL_{algo}"] = []
    else:
        printer.info(f"日志文件 {log_file_path} 不存在，将创建新的日志文件")
        results = {"BIRRT": [], "cuRobo": [], "TAMPOR": []}
        for algo in args.ompl:
            results[f"OMPL_{algo}"] = []

    # 执行规划
    for repeat_id, seed in enumerate(seeds_to_run):
        if seed is not None:
            printer.info(f"\n-------------------- 当前seed: {seed} --------------------\n")

        # **************************************************************************
        # BIRRT plan
        # **************************************************************************
        if args.birrt:
            printer.info("\n========================================")
            printer.info(f"{repeat_id+1}th BIRRT planning")
            printer.info("========================================\n")

            # 检查是否已存在此seed的结果
            skip_planning = False
            if "BIRRT" in results:
                for existing_result in results["BIRRT"]:
                    if existing_result and len(existing_result) > 0 and existing_result[0] == seed:
                        printer.info(f"Seed {seed} for BIRRT already exists in log, skipping.")
                        skip_planning = True
                        break

            # 只有在需要规划时才执行
            if not skip_planning:
                planning_thread = PlanningThread(rb.plan_manipulator_path, start_q, target_q, rb.attachments, element_bodies, max_time=args.max_time, max_iterations=10000)
                planning_thread.start()

                start_time = time.time()
                try:
                    while not planning_thread.done:
                        elapsed_time = time.time() - start_time
                        print(f"\rPlanning... current time: {elapsed_time:.2f} s", end="", flush=True)
                        time.sleep(0.1)

                    elapsed_time = time.time() - start_time
                    result = planning_thread.result

                    if result["success"]:
                        cur_result = (seed, result["success"], elapsed_time)

                        # 保存路径
                        if args.save:
                            save_dir = os.path.join(log_dir, "BIRRT")
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, f"{seed}.npy")
                            np.save(save_path, result["path"])
                    else:
                        cur_result = (seed, False, args.max_time)

                    # 添加结果（避免重复添加同一个seed的结果）
                    append_result = True
                    if "BIRRT" not in results:
                        results["BIRRT"] = []
                    for existing_result in results["BIRRT"]:
                        if existing_result[0] == seed:
                            append_result = False
                            break
                    if append_result:
                        results["BIRRT"].append(cur_result)

                    if result["success"]:
                        printer.success(f"\rPlanning success! Total time: {elapsed_time:.2f} s!")
                        if args.visualize:
                            input = pp.wait_for_user("\nVisualize planned path?")
                            if input == "y" or input == "Y":
                                replay_slider = p.addUserDebugParameter("replay", 0, 1, 0)
                                continue_button = p.addUserDebugParameter("continue", 1, 0, 0)
                                prev_continue_button_value = p.readUserDebugParameter(continue_button)
                                while True:
                                    replay = p.readUserDebugParameter(replay_slider)
                                    current_continue_button_value = p.readUserDebugParameter(continue_button)
                                    idx = int(replay * (len(result["path"]) - 1))
                                    conf = result["path"][idx]
                                    rb.set_joint_positions(rb.arm_joints, conf)
                                    time.sleep(1.0 / 240)
                                    if current_continue_button_value > prev_continue_button_value:
                                        break
                                    prev_continue_button_value = current_continue_button_value
                    else:
                        printer.warning(f"\rBIRRT planning failed, total time: {elapsed_time:.2f} s!")

                except KeyboardInterrupt:
                    printer.warning("\nExit!")
                    exit()

        # **************************************************************************
        # curobo plan TODO: 需要修改
        # **************************************************************************

        if args.curobo:
            printer.info("\n========================================")
            printer.info(f"{repeat_id+1}th cuRobo planning")
            printer.info("========================================\n")

            # 检查是否已存在此seed的结果
            skip_planning = False
            if "cuRobo" in results:
                for existing_result in results["cuRobo"]:
                    if existing_result and len(existing_result) > 0 and existing_result[0] == seed:
                        printer.info(f"Seed {seed} for cuRobo already exists in log, skipping.")
                        skip_planning = True
                        break

            # 只有在需要规划时才执行
            if not skip_planning:

                p.removeAllUserParameters()

                curobo_planner = TrajectoryCuroboSolver(rb, TensorDeviceType())
                planning_thread = PlanningThread(
                    curobo_planner.plan,
                    start_q,
                    target_q,
                    args.max_time,
                    10000,
                    element_bodies,
                    element_infos,
                    grasped_approximate_info=scene_parser.get_robot_grasp_approximate(),
                    grasped_approximate_body=approximate_attachment_body,
                    grasped_approximate_attachment=approximate_attachment,
                    collision_fn=rb.create_collision_fn(element_bodies),
                )

                planning_thread.start()

                start_time = time.time()
                try:
                    while not planning_thread.done:
                        elapsed_time = time.time() - start_time
                        print(f"\rPlanning... current time: {elapsed_time:.2f} s", end="", flush=True)
                        time.sleep(0.1)

                    elapsed_time = time.time() - start_time
                    result = planning_thread.result

                    if result["success"]:
                        cur_result = (seed, result["success"], elapsed_time)
                        path = result["path"]

                        # 保存路径
                        if args.save:
                            save_dir = os.path.join(log_dir, "cuRobo")
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, f"{seed}.npy")
                            np.save(save_path, path)
                    else:
                        cur_result = (seed, False, args.max_time)

                    # 添加结果（避免重复添加同一个seed的结果）
                    append_result = True
                    if "cuRobo" not in results:
                        results["cuRobo"] = []
                    for existing_result in results["cuRobo"]:
                        if existing_result[0] == seed:
                            append_result = False
                            break
                    if append_result:
                        results["cuRobo"].append(cur_result)

                    if result["success"]:
                        printer.success(f"\rPlanning success! Total time: {elapsed_time:.2f} s! ")
                        path = result["path"]
                        if args.visualize:
                            input = pp.wait_for_user("\nVisualize planned path?")
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
                                    time.sleep(1.0 / 240)
                                    if current_continue_button_value > prev_continue_button_value:
                                        break
                                    prev_continue_button_value = current_continue_button_value
                                # for body in element_bodies[:10]:
                                #     pp.set_color(body, [1, 0, 0, 1])
                    else:
                        printer.warning(f"\rCurobo planning failed, total time: {elapsed_time:.2f} s! ")

                except KeyboardInterrupt:
                    printer.warning("\nExit!")
                    exit()

        # **************************************************************************
        # ompl plan
        # **************************************************************************

        for ompl_algo in args.ompl:
            printer.info("\n========================================")
            printer.info(f"{repeat_id+1}th OMPL {ompl_algo} planning")
            printer.info("========================================\n")

            # 检查是否已存在此seed的结果
            algo_key = f"OMPL_{ompl_algo}"
            skip_planning = False
            if algo_key in results:
                for existing_result in results[algo_key]:
                    if existing_result and len(existing_result) > 0 and existing_result[0] == seed:
                        printer.info(f"Seed {seed} for {algo_key} already exists in log, skipping.")
                        skip_planning = True
                        break

            # 只有在需要规划时才执行
            if not skip_planning:
                if seed is not None:
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
                        print(f"\rPlanning... current time: {elapsed_time:.2f} s", end="", flush=True)
                        time.sleep(0.1)

                    elapsed_time = time.time() - start_time
                    path = planning_thread.result

                    if path["success"]:
                        cur_result = (seed, path["success"], elapsed_time)

                        # 保存路径
                        if args.save:
                            save_dir = os.path.join(log_dir, f"OMPL_{ompl_algo}")
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, f"{seed}.npy")
                            np.save(save_path, path["path"])
                    else:
                        cur_result = (seed, False, args.max_time)

                    # 添加结果（避免重复添加同一个seed的结果）
                    append_result = True
                    if algo_key in results:
                        for existing_result in results[algo_key]:
                            if existing_result[0] == seed:
                                append_result = False
                                break
                    else:
                        results[algo_key] = []

                    if append_result:
                        results[algo_key].append(cur_result)

                    if path["success"]:
                        printer.success(f"\rPlanning success! Total time: {elapsed_time:.2f} s!")
                        if args.visualize:
                            input = pp.wait_for_user(f"\nVisualize {ompl_algo} planning path?")
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
                                    print(f"collision_fn: {collision_fn(conf)}, joint_positions: {conf}")
                                    time.sleep(1.0 / 240)
                                    if current_continue_button_value > prev_continue_button_value:
                                        break
                                    prev_continue_button_value = current_continue_button_value
                    else:
                        printer.warning(f"\rOMPL {ompl_algo} planning failed, total time: {elapsed_time:.2f} s!")

                except KeyboardInterrupt:
                    printer.warning("\nExit!")
                    exit()

        # **************************************************************************
        # TAMPOR plan
        # **************************************************************************

        if args.tampor:
            printer.info("\n========================================")
            printer.info(f"{repeat_id+1}th TAMPOR planning")
            printer.info("========================================\n")

            # 检查是否已存在此seed的结果
            skip_planning = False
            if "TAMPOR" in results:
                for existing_result in results["TAMPOR"]:
                    if existing_result and len(existing_result) > 0 and existing_result[0] == seed:
                        printer.info(f"Seed {seed} for TAMPOR already exists in log, skipping.")
                        skip_planning = True
                        break

            # 只有在需要规划时才执行
            if not skip_planning:
                if seed is not None:
                    SetSeeds(seed)

                tampor_planner = TrajectoryTAMPORSolver(rb, channel_info, grasp_pose, eval_max_attempts=1000)
                planning_thread = PlanningThread(tampor_planner.plan, start_q, target_q, element_bodies, grasp_attachment, max_time=args.max_time, init_step_max_time=args.max_time / 3.0, step_max_time=20.0, key_frame_num=20, verbose=True)

                planning_thread.start()

                start_time = time.time()
                try:
                    while not planning_thread.done:
                        elapsed_time = time.time() - start_time
                        # print(f"\rPlanning... current time: {elapsed_time:.2f} s ", end="", flush=True)
                        time.sleep(0.1)

                    elapsed_time = time.time() - start_time
                    result = planning_thread.result

                    if result["success"]:
                        cur_result = (seed, result["success"], elapsed_time)

                        # 保存路径
                        if args.save and "path" in result:
                            path = result["path"]
                            save_dir = os.path.join(log_dir, "TAMPOR")
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, f"{seed}.npy")
                            np.save(save_path, path)
                    else:
                        cur_result = (seed, False, args.max_time)

                    # 添加结果（避免重复添加同一个seed的结果）
                    append_result = True
                    if "TAMPOR" not in results:
                        results["TAMPOR"] = []
                    for existing_result in results["TAMPOR"]:
                        if existing_result[0] == seed:
                            append_result = False
                            break
                    if append_result:
                        results["TAMPOR"].append(cur_result)

                    if result["success"]:
                        printer.success(f"\rPlanning success! Total time: {elapsed_time:.2f} s! ")
                        if args.visualize:
                            input = pp.wait_for_user("\nVisualize planned path?")
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
                        printer.warning(f"\rTAMPOR planning failed, total time: {elapsed_time:.2f} s! ")

                except KeyboardInterrupt:
                    printer.warning("\nExit!")
                    exit()

        # 保存结果到日志文件
        if args.save:
            os.makedirs(log_dir, exist_ok=True)
            file_path = os.path.join(log_dir, "log.json")
            with open(file_path, "w") as f:
                json.dump(results, f, indent=4)

    # **************************************************************************
    # manual control
    # **************************************************************************

    if args.manual:
        p.removeAllUserParameters()

        # 如果提供了关节角度参数，使用它们；否则使用target_q
        if args.joint_angles:
            try:
                # 解析命令行参数中的关节角度
                initial_angles = np.array([float(angle) for angle in args.joint_angles.split(",")])
                if len(initial_angles) != 6:
                    printer.warning(f"警告: 提供的关节角度数量不正确，需要6个，但获得了{len(initial_angles)}个。使用target_q代替。")
                    initial_angles = target_q
            except Exception as e:
                printer.warning(f"解析关节角度失败: {e}。使用target_q代替。")
                initial_angles = target_q
        else:
            initial_angles = target_q

        printer.info("\n使用初始关节角度:")
        for i, angle in enumerate(initial_angles):
            printer.info(f"joint {i}: {angle:.4f} rad ({np.degrees(angle):.2f}°)")

        j0_slider = p.addUserDebugParameter("joint contorl j0", -2 * np.pi, 2 * np.pi, initial_angles[0])
        j1_slider = p.addUserDebugParameter("joint contorl j1", -2 * np.pi, 2 * np.pi, initial_angles[1])
        j2_slider = p.addUserDebugParameter("joint contorl j2", -2 * np.pi, 2 * np.pi, initial_angles[2])
        j3_slider = p.addUserDebugParameter("joint contorl j3", -2 * np.pi, 2 * np.pi, initial_angles[3])
        j4_slider = p.addUserDebugParameter("joint contorl j4", -2 * np.pi, 2 * np.pi, initial_angles[4])
        j5_slider = p.addUserDebugParameter("joint contorl j5", -2 * np.pi, 2 * np.pi, initial_angles[5])

        record_button = p.addUserDebugParameter("record", 1, 0, 0)
        prev_record_button_value = p.readUserDebugParameter(record_button)

        save_button = p.addUserDebugParameter("save", 1, 0, 0)
        prev_save_button_value = p.readUserDebugParameter(save_button)

        replay_button = p.addUserDebugParameter("replay", 1, 0, 0)
        prev_replay_button_value = p.readUserDebugParameter(replay_button)

        record = [initial_angles.tolist()]
        last_conf = initial_angles

        while True:
            j0 = p.readUserDebugParameter(j0_slider)
            j1 = p.readUserDebugParameter(j1_slider)
            j2 = p.readUserDebugParameter(j2_slider)
            j3 = p.readUserDebugParameter(j3_slider)
            j4 = p.readUserDebugParameter(j4_slider)
            j5 = p.readUserDebugParameter(j5_slider)

            rb.set_joint_positions(rb.arm_joints, np.array([j0, j1, j2, j3, j4, j5]))

            current_record_button_value = p.readUserDebugParameter(record_button)
            if current_record_button_value > prev_record_button_value:
                cur_conf = np.array([j0, j1, j2, j3, j4, j5])
                confs = np.linspace(last_conf, cur_conf, 120)
                for conf in confs:
                    record.append(conf.tolist())
                printer.success("Record success!")
                last_conf = cur_conf
            prev_record_button_value = current_record_button_value

            current_save_button_value = p.readUserDebugParameter(save_button)
            if current_save_button_value > prev_save_button_value:
                np.save("record.npy", np.array(record))
                printer.success("Save record success!")
            prev_save_button_value = current_save_button_value

            current_replay_button_value = p.readUserDebugParameter(replay_button)
            if current_replay_button_value > prev_replay_button_value:  # replay
                current_record = np.load("/home/jeong/summer_research/eth_ws/src/husky_assembly/scripts/record_corner.npy")
                for i in range(len(current_record)):
                    rb.set_joint_positions(rb.arm_joints, np.array(current_record[i]))
                    time.sleep(1.0 / 60)
            prev_replay_button_value = current_replay_button_value

            time.sleep(1.0 / 240)
