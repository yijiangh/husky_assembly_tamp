#!/usr/bin/env python3
from typing import Callable, Union

import numpy as np
import pybullet_planning as pp


class DualArmProjection:
    """
    Independent projection class for dual-arm relative constraint.

    This class projects a full dual-arm configuration to satisfy the relative
    constraint by computing the left arm configuration that maintains the desired
    relative pose between the end effectors.

    Args:
        robot_setup: RobotSetup instance for IK calculations
        desired_right_from_left: Desired relative pose of left EE in right EE's frame
    """

    def __init__(self, robot_setup, desired_right_from_left):
        self.robot_setup = robot_setup
        self.desired_right_from_left = desired_right_from_left

    def project(self, right: np.ndarray, left_init_guess: np.ndarray) -> Union[np.ndarray, None]:
        """
        Project a state to satisfy the relative constraint.

        Args:
            right: Right arm joint angles
            left_init_guess: Initial guess for left arm joint angles
        """
        q_right = np.array(right)
        q_left = np.array(left_init_guess)
        self.robot_setup.set_joint_positions(self.robot_setup.arm_joints_right, q_right)

        world_from_right = pp.get_link_pose(self.robot_setup.robot, self.robot_setup.tool_link_right)
        world_from_left = pp.multiply(world_from_right, self.desired_right_from_left)
        q_left_new = self.robot_setup.get_left_arm_ik_solution(world_from_left, q_left)

        if q_left_new is None:
            return None

        return np.concatenate([q_left_new, q_right])

    def project_inv(self, left: np.ndarray, right_init_guess: np.ndarray) -> Union[np.ndarray, None]:
        """
        Inverse projection: project a state to satisfy the relative constraint by computing
        the right arm configuration that maintains the desired relative pose with the left arm.

        Args:
            left: Left arm joint angles
            right_init_guess: Initial guess for right arm joint angles
        """
        q_left = np.array(left)
        q_right = np.array(right_init_guess)
        self.robot_setup.set_joint_positions(self.robot_setup.arm_joints_left, q_left)

        world_from_left = pp.get_link_pose(self.robot_setup.robot, self.robot_setup.tool_link_left)
        world_from_right = pp.multiply(world_from_left, pp.invert(self.desired_right_from_left))
        q_right_new = self.robot_setup.get_right_arm_ik_solution(world_from_right, q_right)

        if q_right_new is None:
            return None

        return np.concatenate([q_left, q_right_new])

    def project_multiple(self, right: np.ndarray, max_attempts: int = 100, collision_fn: Callable[[np.ndarray], bool] = None) -> Union[np.ndarray, None]:
        """
        Project multiple states to satisfy the relative constraint.

        Args:
            right: Right arm joint angles
            left_init_guesses: Initial guess for left arm joint angles
        """
        projected_confs = []
        for _ in range(max_attempts):
            left_init_guess = np.random.uniform(-np.pi, np.pi, 6)
            projected_conf = self.project(right, left_init_guess)
            if projected_conf is not None:
                projected_confs.append(projected_conf)
        if not projected_confs:
            return None
        unique_confs = []
        atol = 1e-2
        for conf in projected_confs:
            is_duplicate = False
            for uconf in unique_confs:
                if np.allclose(conf, uconf, atol=atol):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_confs.append(conf)
        projected_confs = unique_confs
        if collision_fn is not None:
            projected_confs = [conf for conf in projected_confs if not collision_fn(conf)]
        if len(projected_confs) == 0:
            return None
        return np.stack(projected_confs)


if __name__ == "__main__":
    import argparse
    import os
    import sys
    import time

    import numpy as np
    import pybullet
    import pybullet_planning as pp

    HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(HERE)

    from ConstrainedPlanningCommon import *
    from dual_arm_projection import DualArmProjection
    from dual_constrain_test import RelativeEndEffectorConstraint
    from robot.robot_setup import RobotSetup
    from utils.params import DATA_DIR
    
    design_study_path = os.path.join(DATA_DIR, "husky_assembly_design_study")
    design_case = "250707_RobotX_box_demo"
    robot_cell_state_path = os.path.join(design_study_path, design_case, "RobotCellStates", "robotx_box_A6-S4_end_RobotCellState.json")
    
    robot_setup = RobotSetup(robot_name="husky_with_scene", robot_type="husky_dual", robot_cell_state_path=robot_cell_state_path, use_scene_parser_gui=True, scene_parser_verbose=True)

    world_from_left = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
    world_from_right = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
    desired_right_from_left = pp.multiply(pp.invert(world_from_right), world_from_left)
    projector = DualArmProjection(robot_setup, desired_right_from_left)
    
    target_conf = np.array([2.22637564e-03, -3.51527382e-01, 1.42532484e+00, -2.30987362e+00, 1.78120551e+00, -1.55214616e+00, 5.17115363e-01, -1.11289822e+00, -7.18635362e-01, 2.17292434e+00, -1.29150914e+00, 1.42157927e+00])
    projected_confs = projector.project_multiple(target_conf[6:], collision_fn=robot_setup.create_invalid_fn(desired_right_from_left, obstacle_bodies=robot_setup.obstacles, resolution=1e-2))
    
    slider = pybullet.addUserDebugParameter("traj_idx", 0, projected_confs.shape[0] - 1, 0)
    current_index = -1

    while True:
        idx = int(pybullet.readUserDebugParameter(slider))
        if idx != current_index:
            current_index = idx
            conf = projected_confs[current_index]
            robot_setup.set_joint_positions(robot_setup.arm_joints, conf)
        time.sleep(0.01)