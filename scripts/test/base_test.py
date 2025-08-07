import argparse
import math
import os
import sys
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import pybullet
import pybullet_planning as pp
from pybullet_planning.interfaces.planner_interface.joint_motion_planning import get_difference_fn, get_refine_fn

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from robot.dual_arm_projection import DualArmProjection
from model.target_parse import TargetParser
from robot.robot_setup import RobotSetup
from utils.params import DATA_DIR

if __name__ == "__main__":
    design_study_path = os.path.join(DATA_DIR, "husky_assembly_design_study")
    design_case = "250707_RobotX_box_demo"
    start_cell_state_path = os.path.join(design_study_path, design_case, "RobotCellStates", "robotx_box_A6-S4_start_RobotCellState.json")
    target_cell_state_path = os.path.join(design_study_path, design_case, "RobotCellStates", "robotx_box_A6-S4_end_RobotCellState.json")

    # **************************************************************************
    # Start Configuration
    # **************************************************************************
    print("Initializing start configuration...")
    robot_setup = RobotSetup("r0", robot_type="husky_dual", robot_cell_state_path=start_cell_state_path, use_scene_parser_gui=True, scene_parser_verbose=True)
    print("✓ Start configuration initialized.")

    start_conf = np.array(robot_setup.arm_target_angles)
    start_conf = (start_conf + np.pi) % (2 * np.pi) - np.pi
    start_pos = pp.get_pose(robot_setup.robot)
    start_pos_2d = np.array([start_pos[0][0], start_pos[0][1], (pp.euler_from_quat(start_pos[1])[2] + np.pi) % (2 * np.pi) - np.pi])
    print(f"Start pose 2D: {start_pos_2d}")

    robot_setup.set_joint_positions(robot_setup.arm_joints, start_conf)
    pp.draw_pose(start_pos, length=0.5)
    
    # -------------------- setup target parser --------------------#
    target_parser = TargetParser(os.path.join(design_study_path, design_case), "robotx_box_A6-S4_start_GraspTargets.json")
    world_from_bar = target_parser.world_from_bar
    world_from_bar_pos = world_from_bar[0]
    min_dist = np.inf
    for id in robot_setup.obstacles:
        pose = pp.get_pose(id)
        position = pose[0]
        dist = np.linalg.norm(np.array(position) - np.array(world_from_bar_pos))
        if dist < min_dist:
            min_dist = dist
            min_dist_id = id
    print(f"Min distance: {min_dist}, id: {min_dist_id}")
    pp.set_color(min_dist_id, pp.YELLOW)
    robot_setup.remove_obstacle(min_dist_id)
    robot_setup.set_joint_positions(robot_setup.arm_joints, start_conf)
    attachment = pp.create_attachment(robot_setup.robot, robot_setup.tool_link_right, min_dist_id)
    robot_setup.update_attachments([attachment])

    # **************************************************************************
    # Last Configuration
    # **************************************************************************
    last_pos_2d = np.array([-start_pos_2d[0], -start_pos_2d[1], start_pos_2d[2]])
    robot_setup.set_base_pose_2d(*last_pos_2d.tolist())
    last_pos = pp.get_pose(robot_setup.robot)
    print(f"Last pose 2D: {last_pos_2d}")
    
    pp.draw_pose(last_pos, length=0.5)
    
    pp.wait_for_user()
