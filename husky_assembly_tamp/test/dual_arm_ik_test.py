import os
import sys

import pybullet_planning as pp

from husky_assembly_tamp.robot.robot_setup import RobotSetup
from husky_assembly_tamp.utils.params import DATA_DIR

if __name__ == "__main__":
    design_study_path = os.path.join(DATA_DIR, "husky_assembly_design_study")
    design_case = "test"
    robot_cell_state_path = os.path.join(design_study_path, design_case, "RobotCellStates", "robotx_box_A0-G_RobotCellState.json")
    
    robot_setup = RobotSetup(robot_name="husky_with_scene", robot_type="husky_dual", robot_cell_state_path=robot_cell_state_path, use_scene_parser_gui=True, scene_parser_verbose=True)
    
    # pp.wait_for_user()
    
    pp.draw_pose(pp.multiply(pp.get_pose(robot_setup.robot), robot_setup.base_from_connect_left), length=0.5)
    # pp.draw_pose(pp.get_link_pose(robot_setup.robot, pp.link_from_name(robot_setup.robot, robot_setup.robot_params["onboard_link_left"])), length=0.3)
    pp.draw_pose(pp.multiply(pp.get_pose(robot_setup.robot), robot_setup.base_from_connect_right), length=0.5)
    # pp.draw_pose(pp.get_link_pose(robot_setup.robot, pp.link_from_name(robot_setup.robot, robot_setup.robot_params["onboard_link_right"])), length=0.3)

    pp.wait_for_user()
    
    left_tool0_pose = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
    right_tool0_pose = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
    left_arm_angles = pp.get_joint_positions(robot_setup.robot, robot_setup.arm_joints_left)
    right_arm_angles = pp.get_joint_positions(robot_setup.robot, robot_setup.arm_joints_right)
    pp.draw_pose(left_tool0_pose)
    pp.draw_pose(right_tool0_pose)
    
    pp.wait_for_user()
    
    pose_delta = pp.Pose(point = [0, 0, -0.1], euler = [0, 0, 0])
    new_left_tool0_pose = pp.multiply(left_tool0_pose, pose_delta)
    new_left_arm_angles = robot_setup.get_left_arm_ik_solution(new_left_tool0_pose, left_arm_angles)
    robot_setup.set_joint_positions(robot_setup.arm_joints_left, new_left_arm_angles)
    pp.draw_pose(new_left_tool0_pose)
    
    pp.wait_for_user()
    
    new_right_tool0_pose = pp.multiply(right_tool0_pose, pose_delta)
    new_right_arm_angles = robot_setup.get_right_arm_ik_solution(new_right_tool0_pose, right_arm_angles)
    robot_setup.set_joint_positions(robot_setup.arm_joints_right, new_right_arm_angles)
    pp.draw_pose(new_right_tool0_pose)
    
    pp.wait_for_user()