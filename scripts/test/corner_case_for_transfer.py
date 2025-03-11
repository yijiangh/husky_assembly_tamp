import argparse
import os
import random
import sys
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import pybullet as p
import pybullet_planning as pp
import time

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from multi_tangent.collision import create_collision_bodies
from multi_tangent.convert import flatten_list
from robot.robot_setup import RobotSetup
from utils.collision import Element, create_couplers, init_pb
from utils.utils import CounterModule, SetSeeds

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
        np.array([0.8, 0, 1]),
        np.array([0.8, 1, 1]),
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

    input = pp.wait_for_user("start birrt plan?")

    path = None
    if input == "y" or input == "Y":
        start_time = time.time()
        path = rb.plan_manipulator_path(
            init_q, target_q, [attachment], element_bodies, max_time=600, max_iterations=1000
        )
        end_time = time.time()
        print(f"planning time: {end_time - start_time}")

    if path is not None:

        input = pp.wait_for_user("visualize planned path?")

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

    input = pp.wait_for_user("start manual control?")

    if input == "y" or input == "Y":

        p.removeAllUserParameters()

        j0_slider = p.addUserDebugParameter("joint contorl j0", -2 * np.pi, 2 * np.pi, target_q[0])
        j1_slider = p.addUserDebugParameter("joint contorl j1", -2 * np.pi, 2 * np.pi, target_q[1])
        j2_slider = p.addUserDebugParameter("joint contorl j2", -2 * np.pi, 2 * np.pi, target_q[2])
        j3_slider = p.addUserDebugParameter("joint contorl j3", -2 * np.pi, 2 * np.pi, target_q[3])
        j4_slider = p.addUserDebugParameter("joint contorl j4", -2 * np.pi, 2 * np.pi, target_q[4])
        j5_slider = p.addUserDebugParameter("joint contorl j5", -2 * np.pi, 2 * np.pi, target_q[5])

        while True:
            j0 = p.readUserDebugParameter(j0_slider)
            j1 = p.readUserDebugParameter(j1_slider)
            j2 = p.readUserDebugParameter(j2_slider)
            j3 = p.readUserDebugParameter(j3_slider)
            j4 = p.readUserDebugParameter(j4_slider)
            j5 = p.readUserDebugParameter(j5_slider)

            rb.set_joint_positions(rb.arm_joints, np.array([j0, j1, j2, j3, j4, j5]))
            attachment.assign()

            time.sleep(1.0 / 240)
