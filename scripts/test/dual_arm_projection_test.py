import os
import sys
import time
import argparse

import numpy as np
import pybullet
import pybullet_planning as pp

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from robot.robot_setup import RobotSetup
from utils.params import DATA_DIR
from dual_constrain_test import RelativeEndEffectorConstraint
from ConstrainedPlanningCommon import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", action="store_true", help="Dump found solution path and planning graph.")
    parser.add_argument("--bench", action="store_true", help="Run benchmark instead of single planning run.")
    parser.add_argument("--interpolate-points", type=int, default=300, help="Number of points to interpolate the trajectory to (default 300)")
    parser.add_argument("--plot-violations", action="store_true", help="Compute and plot constraint violations along the trajectory")

    addSpaceOption(parser)
    addPlannerOption(parser)
    addConstrainedOptions(parser)
    addAtlasOptions(parser)

    args = parser.parse_args()
    
    design_study_path = os.path.join(DATA_DIR, "husky_assembly_design_study")
    design_case = "test"
    robot_cell_state_path = os.path.join(design_study_path, design_case, "RobotCellStates", "robotx_box_A0-G_RobotCellState.json")
    
    robot_setup = RobotSetup(robot_name="husky_with_scene", robot_type="husky_dual", robot_cell_state_path=robot_cell_state_path, use_scene_parser_gui=True, scene_parser_verbose=True)
    
    constraint = RelativeEndEffectorConstraint(12, robot_setup)
    cp = ConstrainedProblem(args.space, constraint.createSpace(), constraint, args)
    projection = constraint.getProjection(cp.space)
    
    left_tool0_pose = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
    right_tool0_pose = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
    left_arm_angles = list(pp.get_joint_positions(robot_setup.robot, robot_setup.arm_joints_left))
    right_arm_angles = list(pp.get_joint_positions(robot_setup.robot, robot_setup.arm_joints_right))
    pp.draw_pose(left_tool0_pose)
    pp.draw_pose(right_tool0_pose)
    
    sliders = []
    for i in range(6):
        sliders.append(pybullet.addUserDebugParameter(f"joint_{i}", -np.pi * 2, np.pi * 2, right_arm_angles[i]))
        
    projection_button = pybullet.addUserDebugParameter("projection", 1, 0, 0)
    
    projection_button_state = 0
    while True:
        for i in range(6):
            right_arm_angles[i] = pybullet.readUserDebugParameter(sliders[i])
        
        if pybullet.readUserDebugParameter(projection_button) != projection_button_state:
            projection_button_state = pybullet.readUserDebugParameter(projection_button)
            temp = [0] * 12
            projection.project(np.concatenate([left_arm_angles, right_arm_angles]), temp)
            print(temp)
            
            left_arm_angles = temp[:6]
            right_arm_angles = temp[6:]
            
        robot_setup.set_joint_positions(robot_setup.arm_joints_right, right_arm_angles)
        robot_setup.set_joint_positions(robot_setup.arm_joints_left, left_arm_angles)
        time.sleep(0.01)
    
    # pp.draw_pose(pp.multiply(pp.get_pose(robot_setup.robot), robot_setup.base_from_connect_left), length=0.5)
    # # pp.draw_pose(pp.get_link_pose(robot_setup.robot, pp.link_from_name(robot_setup.robot, robot_setup.robot_params["onboard_link_left"])), length=0.3)
    # pp.draw_pose(pp.multiply(pp.get_pose(robot_setup.robot), robot_setup.base_from_connect_right), length=0.5)
    # # pp.draw_pose(pp.get_link_pose(robot_setup.robot, pp.link_from_name(robot_setup.robot, robot_setup.robot_params["onboard_link_right"])), length=0.3)

    # pp.wait_for_user()
    
    # left_tool0_pose = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
    # right_tool0_pose = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
    # left_arm_angles = pp.get_joint_positions(robot_setup.robot, robot_setup.arm_joints_left)
    # right_arm_angles = pp.get_joint_positions(robot_setup.robot, robot_setup.arm_joints_right)
    # pp.draw_pose(left_tool0_pose)
    # pp.draw_pose(right_tool0_pose)
    
    # pp.wait_for_user()
    
    # pose_delta = pp.Pose(point = [0, 0, -0.1], euler = [0, 0, 0])
    # new_left_tool0_pose = pp.multiply(left_tool0_pose, pose_delta)
    # new_left_arm_angles = robot_setup.get_left_arm_ik_solution(new_left_tool0_pose, left_arm_angles)
    # robot_setup.set_joint_positions(robot_setup.arm_joints_left, new_left_arm_angles)
    # pp.draw_pose(new_left_tool0_pose)
    
    # pp.wait_for_user()
    
    # new_right_tool0_pose = pp.multiply(right_tool0_pose, pose_delta)
    # new_right_arm_angles = robot_setup.get_right_arm_ik_solution(new_right_tool0_pose, right_arm_angles)
    # robot_setup.set_joint_positions(robot_setup.arm_joints_right, new_right_arm_angles)
    # pp.draw_pose(new_right_tool0_pose)
    
    # pp.wait_for_user()