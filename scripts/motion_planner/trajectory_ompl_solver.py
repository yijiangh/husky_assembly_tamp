import math
import os
import sys
import time
from copy import deepcopy
from typing import Callable, Dict, List, Union

import numpy as np
import pybullet as p

from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from utils.params import *
from motion_planner.pb_ompl import pb_ompl
from utils.util import interpolate


class PbOMPLRobotWrapper(pb_ompl.PbOMPLRobot):
    """
    Wrapper for PbOMPLRobot class to adapt it to our collision checking function
    """

    def __init__(self, robot_id, joint_indices):
        self.id = robot_id
        self.joint_idx = joint_indices
        self.num_dim = len(joint_indices)
        self.state = [0] * self.num_dim
        self.joint_bounds = []
        self._set_manual_joint_bounds()

    def _set_manual_joint_bounds(self):
        # Manually set joint boundaries to avoid reading from PyBullet
        for _ in range(self.num_dim):
            self.joint_bounds.append([-2 * math.pi, 2 * math.pi])

    def get_joint_bounds(self):
        if not self.joint_bounds:
            self._set_manual_joint_bounds()
        return self.joint_bounds

    def _is_not_fixed(self, joint_idx):
        return True  # We have already filtered fixed joints


class TrajectoryOMPLSolver:
    def __init__(
        self,
        collision_fn: Callable[[np.ndarray], bool],
        robot_id: int,
        arm_joints: List[int],
        obstacles: List[int] = None,
        planner: str = "RRTConnect",
    ):
        self.collision_fn = collision_fn
        self.robot_id = robot_id
        self.arm_joints = arm_joints
        self.obstacles = obstacles if obstacles else []
        self.planner = planner

        self.setup_pb_ompl(planner)

    def setup_pb_ompl(self, planner_name):
        """Setup pb_ompl planner"""
        # Create robot wrapper
        self.robot = PbOMPLRobotWrapper(self.robot_id, self.arm_joints)

        # Initialize pb_ompl interface
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles)

        # Set planner
        self.pb_ompl_interface.set_planner(planner_name)

        self.pb_ompl_interface.si.setStateValidityCheckingResolution(0.0005)

        if hasattr(self.pb_ompl_interface.space, "setLongestValidSegmentFraction"):
            self.pb_ompl_interface.space.setLongestValidSegmentFraction(0.0005)

        if hasattr(self.pb_ompl_interface.planner, "setRange"):
            self.pb_ompl_interface.planner.setRange(0.01)

        # Configure collision detection
        if self.collision_fn:
            # Replace default collision detection with custom function
            def custom_is_state_valid(state):
                state_arr = np.array([state[i] for i in range(self.robot.num_dim)])
                return not self.collision_fn(state_arr)

            self.pb_ompl_interface.ss.setStateValidityChecker(ob.StateValidityCheckerFn(custom_is_state_valid))

    def isStateValid(self, state):
        """
        Regular OMPL state validity check
        """
        state_arr = np.zeros(6)
        for i in range(6):
            state_arr[i] = state[i]
        if self.collision_fn is not None:
            return not self.collision_fn(state_arr)
        else:
            return True

    def plan(self, q_init: np.ndarray, q_target: np.ndarray, max_time: float = 10.0, max_attempts: int = 100, collision_fn: Callable = None) -> Dict:
        """
        Plan a path

        Parameters:
            q_init: Initial joint configuration
            q_target: Target joint configuration
            max_time: Maximum planning time (seconds)
            max_attempts: Maximum number of planning attempts
            collision_fn: Collision checking function

        Returns:
            Dictionary containing success status and path
        """
        start_time = time.time()

        # Use the provided collision checking function if available
        if collision_fn is not None:
            collision_check = collision_fn
        else:
            collision_check = self.collision_fn

        # Validate start and goal configurations
        if collision_check(q_init):
            print("Start configuration is in collision")
            return {"success": False, "path": None}

        if collision_check(q_target):
            print("Goal configuration is in collision")
            return {"success": False, "path": None}

        # Loop until successful planning or timeout
        attempts = 0
        while time.time() - start_time < max_time and attempts < max_attempts:
            attempts += 1

            # Calculate remaining time
            remaining_time = max_time - (time.time() - start_time)
            if remaining_time <= 0:
                break

            # Execute one planning attempt
            self.setup_pb_ompl(self.planner)
            res, path = self.pb_ompl_interface.plan_start_goal(q_init.tolist(), q_target.tolist(), allowed_time=remaining_time)

            if res:
                # Convert path to numpy array
                path_array = np.array(path)

                # Check if the path is empty
                if len(path_array) == 0:
                    print("Planner returned an empty path")
                    continue

                # Check if start and end points match the input
                start_diff = np.linalg.norm(path_array[0] - q_init)
                goal_diff = np.linalg.norm(path_array[-1] - q_target)

                if start_diff > 1e-6:
                    print(f"Path start point doesn't match input start point, difference: {start_diff}")
                    continue

                if goal_diff > 1e-6:
                    print(f"Path end point doesn't match input target point, difference: {goal_diff}")
                    continue

                path_array = interpolate(path_array, max(50000, len(path_array)))

                # Verify each configuration in the path
                collision_free = True
                for conf in path_array:
                    if collision_check(conf):
                        collision_free = False
                        break

                if collision_free:
                    return {"success": True, "path": path_array}
                else:
                    print(f"Found path on attempt {attempts}, but it contains collisions. Retrying...")
            else:
                print(f"Planning attempt {attempts} failed. Retrying...")

        # If all attempts failed
        return {"success": False, "path": None}
