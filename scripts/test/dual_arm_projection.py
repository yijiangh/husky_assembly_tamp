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
