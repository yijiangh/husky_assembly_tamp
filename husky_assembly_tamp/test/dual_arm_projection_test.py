import argparse
import os
import sys
import time

import numpy as np
import pybullet
import pybullet_planning as pp

from husky_assembly_tamp.robot.dual_arm_projection import DualArmProjection
from husky_assembly_tamp.robot.robot_setup import RobotSetup
from husky_assembly_tamp.utils.params import DATA_DIR

if __name__ == "__main__":
    design_study_path = os.path.join(DATA_DIR, "husky_assembly_design_study")
    design_case = "250707_RobotX_box_demo"
    robot_cell_state_path = os.path.join(design_study_path, design_case, "RobotCellStates", "robotx_box_A6-S4_end_RobotCellState.json")

    robot_setup = RobotSetup(robot_name="husky_with_scene", robot_type="husky_dual", robot_cell_state_path=robot_cell_state_path, use_scene_parser_gui=True, scene_parser_verbose=True)

    world_from_left = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
    world_from_right = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
    desired_right_from_left = pp.multiply(pp.invert(world_from_right), world_from_left)
    projector = DualArmProjection(robot_setup, desired_right_from_left)

    left_tool0_pose = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
    right_tool0_pose = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
    left_arm_angles = list(pp.get_joint_positions(robot_setup.robot, robot_setup.arm_joints_left))
    right_arm_angles = list(pp.get_joint_positions(robot_setup.robot, robot_setup.arm_joints_right))
    pp.draw_pose(left_tool0_pose)
    pp.draw_pose(right_tool0_pose)

    sliders = []
    for i in range(6):
        sliders.append(pybullet.addUserDebugParameter(f"joint_{i}", -np.pi * 2, np.pi * 2, right_arm_angles[i]))

    while True:
        for i in range(6):
            right_arm_angles[i] = pybullet.readUserDebugParameter(sliders[i])
        
        projected_conf = projector.project(right_arm_angles, left_arm_angles)
        if projected_conf is None:
            continue
        
        print(f"Projected conf: {projected_conf}")
        right_arm_angles = projected_conf[6:]
        left_arm_angles = projected_conf[:6]

        robot_setup.set_joint_positions(robot_setup.arm_joints_right, right_arm_angles)
        robot_setup.set_joint_positions(robot_setup.arm_joints_left, left_arm_angles)
        time.sleep(0.01)
