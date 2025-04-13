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

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from model.scene_parse import SceneParser
from multi_tangent.collision import create_collision_bodies
from multi_tangent.convert import flatten_list
from robot.robot_setup import RobotSetup
from utils.collision import Element, create_couplers, init_pb
from utils.params import *
from utils.util import CounterModule, SetSeeds


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
    parser = argparse.ArgumentParser(description="Manual control")
    parser.add_argument("--scene", type=str, default="cuboid_1", help="Scene name")
    parser.add_argument("--task", type=str, default="task_1", help="Task number")
    parser.add_argument("--joint_angles", type=str, help="Joint angles for manual mode (comma-separated, e.g. '1.0,0.5,-1.0,0.8,1.2,0.3')")
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

    # **************************************************************************
    # manual control
    # **************************************************************************
    p.removeAllUserParameters()

    # 如果提供了关节角度参数，使用它们；否则使用target_q
    if args.joint_angles:
        try:
            # 解析命令行参数中的关节角度
            initial_angles = np.array([float(angle) for angle in args.joint_angles.split(",")])
            if len(initial_angles) != 6:
                print(f"警告: 提供的关节角度数量不正确，需要6个，但获得了{len(initial_angles)}个。使用target_q代替。")
                initial_angles = target_q
        except Exception as e:
            print(f"解析关节角度失败: {e}。使用target_q代替。")
            initial_angles = target_q
    else:
        initial_angles = target_q

    print("\n使用初始关节角度:")
    for i, angle in enumerate(initial_angles):
        print(f"关节 {i}: {angle:.4f} rad ({np.degrees(angle):.2f}°)")

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
