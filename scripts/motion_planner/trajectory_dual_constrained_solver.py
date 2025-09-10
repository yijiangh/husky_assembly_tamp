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

from model.target_parse import TargetParser
from robot.dual_arm_projection import DualArmProjection
from robot.robot_setup import RobotSetup
from utils.params import DATA_DIR, PROJECT_DIR

DEFAULT_RESOLUTION = math.radians(1.0)


class TrajectoryDualConstrainedSolver:
    """
    Dual-arm constrained trajectory solver for robot motion planning.

    This class encapsulates the functionality for planning trajectories between start and target
    configurations while maintaining dual-arm constraints (e.g., keeping objects grasped by both arms).

    Key Features:
    - Dual-arm constraint projection
    - Collision-aware planning
    - RRT-based path planning with custom extend functions
    - Interactive trajectory visualization and playback

    Example usage:
        ```python
        # Initialize solver
        solver = TrajectoryDualConstrainedSolver(robot_setup, target_parser)

        # Plan trajectory
        path = solver.plan(start_conf, target_conf, max_time=600, visualization=True)

        # Interactive playback
        if path:
            solver.interactive_trajectory_playback(path)
        ```
    """

    def __init__(self, robot_setup: RobotSetup, target_parser: TargetParser, resolution: float = DEFAULT_RESOLUTION):
        """
        Initialize the trajectory solver.

        Args:
            robot_setup: Configured RobotSetup instance with the target scene
            target_parser: TargetParser instance for handling grasp targets
            resolution: Angular resolution for motion planning (default: 1 radian)
        """
        self.robot_setup = robot_setup
        self.target_parser = target_parser
        self.resolution = resolution
        self.projector = None
        self.desired_right_from_left = None
        self.collision_fn = None
        self.invalid_fn = None
        self.start_projected_confs = None
        self.target_projected_confs = None

    def plan(self, start_conf: np.ndarray, target_conf: np.ndarray, max_time: int = 600, max_projection_attempts: int = 100, visualization: bool = True) -> Optional[List[np.ndarray]]:
        """
        Main planning method that finds a trajectory from start to target configuration.

        Args:
            start_conf: Starting joint configuration (12 DOF)
            target_conf: Target joint configuration (12 DOF)
            max_time: Maximum planning time in seconds
            max_projection_attempts: Maximum attempts for constraint projection
            visualization: Whether to enable visualization

        Returns:
            List of joint configurations representing the path, or None if no path found
        """
        # Normalize configurations to [-pi, pi]
        start_conf = self._normalize_angles(start_conf)
        target_conf = self._normalize_angles(target_conf)

        print(f"Start configuration: {list(start_conf)}")
        print(f"Target configuration: {list(target_conf)}")

        # Setup constraint projection
        self._setup_constraint_projection(target_conf)

        # Setup collision checking
        self._setup_collision_checking()

        # Generate projected configurations
        self._generate_projected_configurations(start_conf, target_conf, max_projection_attempts)

        # Setup planning functions
        sample_fn = self._get_sample_fn()
        extend_fn_continuous = self._get_extend_fn(check_continuous=True)
        extend_fn_direct = self._get_extend_fn(check_continuous=False)
        distance_fn = self._get_distance_fn()
        self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, start_conf)
        start = pp.get_link_pose(self.robot_setup.robot, self.robot_setup.tool_link_right)
        self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, target_conf)
        target = pp.get_link_pose(self.robot_setup.robot, self.robot_setup.tool_link_right)
        draw_fn = self._get_draw_fn(start, target) if visualization else None

        # Try direct path first
        path = self._try_direct_path(extend_fn_direct)

        if path is None:
            print("No direct path found. Using RRT-based planning...")
            path = self.robot_setup.plan_manipulator_path(
                start_conf, target_conf, attachments=[], obstacles=self.robot_setup.obstacles, sample_fn=sample_fn, collision_fn=self.invalid_fn, extend_fn=extend_fn_continuous, max_time=max_time, draw_fn=draw_fn, distance_fn=distance_fn
            )

        if path is not None:
            path = self._post_process_path(path)
            if visualization:
                self._visualize_path(path)
            print(f"Path found with {len(path)} waypoints")
            return path
        else:
            print("No path found.")
            return None

    def _normalize_angles(self, conf: np.ndarray) -> np.ndarray:
        """Normalize joint angles to [-pi, pi] range."""
        conf = np.array(conf)
        return (conf + np.pi) % (2 * np.pi) - np.pi

    def _setup_constraint_projection(self, target_conf: np.ndarray):
        """Setup dual arm constraint projection."""
        self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, target_conf)

        world_from_left = pp.get_link_pose(self.robot_setup.robot, self.robot_setup.tool_link_left)
        world_from_right = pp.get_link_pose(self.robot_setup.robot, self.robot_setup.tool_link_right)
        self.desired_right_from_left = pp.multiply(pp.invert(world_from_right), world_from_left)
        self.projector = DualArmProjection(self.robot_setup, self.desired_right_from_left)

    def _setup_collision_checking(self):
        """Setup collision checking functions."""
        self.collision_fn = self.robot_setup.create_collision_fn(obstacle_bodies=self.robot_setup.obstacles)
        self.invalid_fn = self.robot_setup.create_invalid_fn(self.desired_right_from_left, obstacle_bodies=self.robot_setup.obstacles, resolution=1e-2)

    def _generate_projected_configurations(self, start_conf: np.ndarray, target_conf: np.ndarray, max_attempts: int):
        """Generate projected configurations for start and target."""

        # Helper to normalize returns from projector to a consistent ndarray with shape (N, 12)
        def to_array_or_empty(confs):
            if confs is None:
                return np.empty((0, 12))
            arr = np.array(confs)
            if arr.ndim == 1:
                # Single configuration of length 12
                if arr.size == 12:
                    return arr.reshape(1, 12)
                # Unexpected shape; treat as empty to be safe
                return np.empty((0, 12))
            return arr

        # Project configurations using left arm as primary
        start_projected_confs_left = to_array_or_empty(self.projector.project_multiple(start_conf[6:], max_attempts=max_attempts, collision_fn=self.collision_fn))
        target_projected_confs_left = to_array_or_empty(self.projector.project_multiple(target_conf[6:], max_attempts=max_attempts, collision_fn=self.collision_fn))

        # Project configurations using right arm as primary
        start_projected_confs_right = to_array_or_empty(self.projector.project_multiple_inv(start_conf[:6], max_attempts=max_attempts, collision_fn=self.collision_fn))
        target_projected_confs_right = to_array_or_empty(self.projector.project_multiple_inv(target_conf[:6], max_attempts=max_attempts, collision_fn=self.collision_fn))

        print(f"Start projected confs (left primary): {start_projected_confs_left.shape}")
        print(f"Target projected confs (left primary): {target_projected_confs_left.shape}")
        print(f"Start projected confs (right primary): {start_projected_confs_right.shape}")
        print(f"Target projected confs (right primary): {target_projected_confs_right.shape}")

        # Generate all combinations where available; otherwise fall back to the available side
        start_projected_confs: List[np.ndarray] = []
        target_projected_confs: List[np.ndarray] = []

        if start_projected_confs_left.shape[0] > 0 and start_projected_confs_right.shape[0] > 0:
            for left_conf in start_projected_confs_left:
                for right_conf in start_projected_confs_right:
                    start_projected_confs.append(np.concatenate([left_conf[:6], right_conf[6:]]))
        elif start_projected_confs_left.shape[0] > 0:
            start_projected_confs.extend(list(start_projected_confs_left))
        elif start_projected_confs_right.shape[0] > 0:
            start_projected_confs.extend(list(start_projected_confs_right))

        if target_projected_confs_left.shape[0] > 0 and target_projected_confs_right.shape[0] > 0:
            for left_conf in target_projected_confs_left:
                for right_conf in target_projected_confs_right:
                    target_projected_confs.append(np.concatenate([left_conf[:6], right_conf[6:]]))
        elif target_projected_confs_left.shape[0] > 0:
            target_projected_confs.extend(list(target_projected_confs_left))
        elif target_projected_confs_right.shape[0] > 0:
            target_projected_confs.extend(list(target_projected_confs_right))

        self.start_projected_confs = np.array(start_projected_confs) if len(start_projected_confs) > 0 else np.empty((0, 12))
        self.target_projected_confs = np.array(target_projected_confs) if len(target_projected_confs) > 0 else np.empty((0, 12))

        print(f"Total start projected configurations: {self.start_projected_confs.shape}")
        print(f"Total target projected configurations: {self.target_projected_confs.shape}")

    def _get_sample_fn(self):
        """Create sampling function for configuration space."""
        lower, upper = [-np.pi] * 12, [np.pi] * 12
        cache = []

        def fn():
            if len(cache) == 0:
                print("Generating cache...")
                while len(cache) < 100:
                    right_conf = np.random.uniform(lower[6:], upper[6:])
                    projected_confs = self.projector.project_multiple(right_conf, max_attempts=10, collision_fn=self.collision_fn)

                    if projected_confs is not None:
                        cache.append(projected_confs[0])
                        print(f"Cache: {len(cache)}")

                print("Cache generated!")

            sample = cache.pop()
            print(f"Cache: {len(cache)}")
            return sample

        return fn

    def _get_draw_fn(self, start, target):
        """Create drawing function for visualization."""
        pose_cache = set()
        segment_cache = set()

        def pose_to_tuple(pose, decimals=3):
            pos, orn = pose
            pos_tuple = tuple(np.round(pos, decimals=decimals))
            orn_tuple = tuple(np.round(orn, decimals=decimals))
            return pos_tuple + orn_tuple

        def segment_to_tuple(pose1, pose2, decimals=3):
            t1 = pose_to_tuple(pose1, decimals)
            t2 = pose_to_tuple(pose2, decimals)
            return tuple(sorted([t1, t2]))

        start_tree_set = set()
        target_tree_set = set()

        start_tree_set.add(pose_to_tuple(start))
        target_tree_set.add(pose_to_tuple(target))

        def fn(conf, segment, valid=None, valid_right=None):
            self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, conf)
            pose_1 = pp.get_link_pose(self.robot_setup.robot, self.robot_setup.tool_link_right)
            pose_1_tuple = pose_to_tuple(pose_1)

            if pose_1_tuple not in pose_cache:
                pose_cache.add(pose_1_tuple)

            if len(segment) > 0:
                self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, segment[1])
                pose_2 = pp.get_link_pose(self.robot_setup.robot, self.robot_setup.tool_link_right)
                pose_2_tuple = pose_to_tuple(pose_2)

                if pose_2_tuple not in pose_cache:
                    pose_cache.add(pose_2_tuple)

                color = pp.BROWN
                if pose_1_tuple in start_tree_set:
                    color = pp.BLUE
                    start_tree_set.add(pose_2_tuple)
                elif pose_2_tuple in start_tree_set:
                    color = pp.BLUE
                    start_tree_set.add(pose_1_tuple)
                elif pose_1_tuple in target_tree_set:
                    color = pp.RED
                    target_tree_set.add(pose_2_tuple)
                elif pose_2_tuple in target_tree_set:
                    color = pp.RED
                    target_tree_set.add(pose_1_tuple)

                seg_tuple = segment_to_tuple(pose_1, pose_2)
                if seg_tuple not in segment_cache:
                    pp.add_line(pose_1[0], pose_2[0], width=1.0, color=color)
                    segment_cache.add(seg_tuple)

        return fn

    def _get_circular_diff(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Compute shortest angular difference between two angle arrays."""
        q1 = self._normalize_angles(q1)
        q2 = self._normalize_angles(q2)
        diff = q2 - q1
        diff = self._normalize_angles(diff)
        return diff

    def _create_invalid_configuration(self) -> np.ndarray:
        """Create a configuration that is guaranteed to fail collision checking."""
        data = np.array([2.15557306, -1.05715414, 1.63506225, -0.25357488, 1.23252519, -1.42178216, 1.31437349, 1.25663757, 0.39683294, -3.90218878, -1.71960878, 0.05148935])
        return self._normalize_angles(data)

    def _get_extend_fn(self, check_continuous: bool = True):
        """Create extension function for path planning."""
        resolutions = np.array([1.0 if j in [] else 5.0 / 180.0 * np.pi for j in self.robot_setup.arm_joints])
        norm = 2

        def fn(q1, q2):
            q1_right = np.array(q1[6:])
            q2_right = np.array(q2[6:])

            right_diff = self._get_circular_diff(q1_right, q2_right)
            right_steps = int(np.ceil(np.linalg.norm(right_diff / resolutions[6:], ord=norm)))

            q_left_init = np.array(q1[:6])
            q_left_target = np.array(q2[:6])

            for i in range(right_steps + 1):
                if right_steps == 0:
                    t = 0.0
                else:
                    t = i / right_steps

                q_right_interp = q1_right + t * right_diff
                q_right_interp = self._normalize_angles(q_right_interp)

                with pp.LockRenderer():
                    projected_conf = self.projector.project(q_right_interp, q_left_init)

                if check_continuous:
                    # Continuous version with stricter constraints
                    if t < 1 - 0.01 and projected_conf is not None and np.linalg.norm(self._get_circular_diff(projected_conf[:6], q_left_init)) < 0.5:
                        q_left_init = np.array(projected_conf[:6])
                        yield tuple(projected_conf)
                    elif t >= 1 - 0.01 and projected_conf is not None and np.linalg.norm(self._get_circular_diff(projected_conf[:6], q_left_target)) < 0.1:
                        q_left_init = np.array(projected_conf[:6])
                        yield tuple(projected_conf)
                    else:
                        collision_conf = self._create_invalid_configuration()
                        yield tuple(collision_conf)
                else:
                    # Direct version
                    if projected_conf is not None and np.linalg.norm(self._get_circular_diff(projected_conf[:6], q_left_init)) < 0.5:
                        q_left_init = np.array(projected_conf[:6])
                        yield tuple(projected_conf)
                    else:
                        collision_conf = self._create_invalid_configuration()
                        yield tuple(collision_conf)

        return fn

    def _get_distance_fn(self):
        """Create distance function for path planning."""
        weights = 1 * np.ones(len(self.robot_setup.arm_joints))

        def fn(q1, q2):
            diff = np.array(self._get_circular_diff(q2, q1))
            return np.sqrt(np.dot(weights, diff * diff))

        return fn

    def _try_direct_path(self, extend_fn_direct):
        """Try to find a direct path between start and target configurations."""
        print("Checking for direct paths...")
        i = 0
        for s in self.start_projected_confs:
            for t in self.target_projected_confs:
                path = pp.direct_path(s, t, extend_fn_direct, self.invalid_fn)
                if path is not None:
                    print("Direct path found!")
                    return path
                else:
                    print(f"No direct path between configurations {i}th")
                i += 1

        print("No direct path found for any start-target pair.")
        return None

    def _post_process_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """Post-process the path to remove unnecessary trailing frames."""
        if len(path) <= 1:
            return path

        # Remove trailing frames where the right arm joint angles remain unchanged
        right_arm_indices = list(range(6, 12))
        last_right = tuple(np.round(np.array(path[-1])[right_arm_indices], decimals=6))

        # Find the first index from the end where right arm changes
        cutoff_idx = len(path) - 1
        for i in reversed(range(len(path) - 1)):
            right_i = tuple(np.round(np.array(path[i])[right_arm_indices], decimals=6))
            if right_i != last_right:
                cutoff_idx = i + 1
                break

        # Keep up to the first occurrence of the repeated right arm, discard the rest
        return path[: cutoff_idx + 1]

    def _visualize_path(self, path: List[np.ndarray]):
        """Visualize the planned path."""
        if len(path) <= 1:
            return

        prev_conf = path[0]
        self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, prev_conf)
        prev_pose = pp.get_link_pose(self.robot_setup.robot, self.robot_setup.tool_link_right)

        for conf in path[1:]:
            self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, conf)
            pose = pp.get_link_pose(self.robot_setup.robot, self.robot_setup.tool_link_right)
            pp.add_line(prev_pose[0], pose[0], color=[0, 1, 0, 0.5], width=2.0)
            prev_pose = pose

    def interactive_trajectory_playback(self, path: List[np.ndarray]):
        """Interactive trajectory playback with slider control."""
        if path is None or len(path) == 0:
            print("No trajectory to playback")
            return

        result_traj = np.array(path)
        slider = pybullet.addUserDebugParameter("traj_idx", 0, result_traj.shape[0] - 1, 0)
        current_index = -1

        try:
            while True:
                idx = int(pybullet.readUserDebugParameter(slider))
                if idx != current_index:
                    current_index = idx
                    conf = result_traj[current_index]
                    self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, conf)
                    print(f"Conf: {conf}, is valid: {not self.invalid_fn(conf)}")
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            print("Trajectory playback ended.")

    @staticmethod
    def initialize_robot_setup_for_planning(robot_name: str, robot_type: str, target_cell_state_path: str, use_scene_parser_gui: bool = True, scene_parser_verbose: bool = True) -> Tuple[RobotSetup, np.ndarray, DualArmProjection]:
        """
        Initialize robot setup for dual-arm constrained motion planning.

        This method encapsulates the complete initialization process for robot setup, including:
        1. Creating and configuring the RobotSetup instance
        2. Computing and normalizing the target joint configuration
        3. Calculating the relative transformation between left and right tool poses
        4. Creating the dual-arm constraint projector

        Args:
            robot_name (str): Unique identifier for the robot instance (e.g., "r0")
            robot_type (str): Type of robot to initialize (e.g., "husky_dual")
            target_cell_state_path (str): File path to the robot cell state JSON file containing
                                        target configuration and scene information
            use_scene_parser_gui (bool, optional): Whether to enable GUI for scene parsing.
                                                 Defaults to True.
            scene_parser_verbose (bool, optional): Whether to enable verbose output during
                                                 scene parsing. Defaults to True.

        Returns:
            Tuple[RobotSetup, np.ndarray, DualArmProjection]: A tuple containing:
                - robot_setup (RobotSetup): Fully configured robot setup instance with loaded
                                          scene and target configuration
                - target_conf (np.ndarray): Normalized target joint configuration (12 DOF)
                                           with angles in [-π, π] range
                - projector (DualArmProjection): Dual-arm constraint projector configured
                                                with the relative transformation between
                                                left and right tool poses

        Raises:
            FileNotFoundError: If the target_cell_state_path does not exist
            ValueError: If the robot setup fails to initialize properly
            RuntimeError: If unable to compute tool poses or create projector

        Example:
            ```python
            # Initialize robot setup for planning
            robot_setup, target_conf, projector = TrajectoryDualConstrainedSolver.initialize_robot_setup_for_planning(
                robot_name="r0",
                robot_type="husky_dual",
                target_cell_state_path="/path/to/target_state.json"
            )

            # Create solver with initialized components
            target_parser = TargetParser(design_path, targets_file)
            solver = TrajectoryDualConstrainedSolver(robot_setup, target_parser)
            ```

        Note:
            The target configuration is automatically normalized to the [-π, π] range to ensure
            consistent angle representation for motion planning algorithms. The dual-arm projector
            maintains the relative pose constraint between the left and right tool links as
            computed from the target configuration.
        """
        print("Initializing robot setup for planning...")

        # Create RobotSetup instance with specified parameters
        robot_setup = RobotSetup(robot_name, robot_type=robot_type, robot_cell_state_path=target_cell_state_path, use_scene_parser_gui=use_scene_parser_gui, scene_parser_verbose=scene_parser_verbose)

        # Extract and normalize target configuration
        target_conf = np.array(robot_setup.arm_target_angles)
        target_conf = (target_conf + np.pi) % (2 * np.pi) - np.pi
        print(f"Target configuration: {list(target_conf)}")

        # Compute relative transformation between left and right tool poses
        world_from_left = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
        world_from_right = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
        desired_right_from_left = pp.multiply(pp.invert(world_from_right), world_from_left)

        # Create dual-arm constraint projector
        projector = DualArmProjection(robot_setup, desired_right_from_left)

        print("✓ Robot setup initialization completed")
        return robot_setup, target_conf, projector

    def generate_start_configuration(self, projector: DualArmProjection, delta_pose_point: List[float] = [0.4, 0.0, 0.75], delta_pose_euler: List[float] = [-1.5708, 1.5708, 0], tool_index: int = 1, max_attempts: int = 100) -> np.ndarray:
        """
        Generate a valid start configuration for dual-arm constrained motion planning.

        This method computes a feasible starting joint configuration by:
        1. Computing the target bar pose from robot base pose and relative delta pose
        2. Using dual-arm constraint projection to find valid configurations
        3. Normalizing joint angles to [-π, π] range
        4. Setting the robot to the computed start configuration

        Args:
            projector (DualArmProjection): Dual-arm constraint projector for generating
                                         valid configurations that maintain relative constraints
            delta_pose_point (List[float], optional): Relative position offset from robot base
                                                    in meters [x, y, z]. Defaults to [0.4, 0.0, 0.75].
            delta_pose_euler (List[float], optional): Relative orientation in Euler angles
                                                    [roll, pitch, yaw] in radians.
                                                    Defaults to [-1.5708, 1.5708, 0].
            tool_index (int, optional): Index of the tool transformation in target_parser.tools_from_bar.
                                       Used to define the grasp relationship. Defaults to 1.
            max_attempts (int, optional): Maximum number of attempts for configuration generation.
                                        Higher values increase success probability but take longer.
                                        Defaults to 100.

        Returns:
            np.ndarray: Normalized start joint configuration (12 DOF) with angles in [-π, π] range.
                       The configuration satisfies dual-arm constraints and collision-free requirements.

        Raises:
            SystemExit: If no valid start configuration can be found after max_attempts.
                       This indicates the problem may be infeasible or requires different parameters.
            ValueError: If target_parser is not properly initialized or tool_index is invalid.
            RuntimeError: If IK solution handles are not available or projector fails.

        Example:
            ```python
            # Initialize solver components
            robot_setup, target_conf, projector = TrajectoryDualConstrainedSolver.initialize_robot_setup_for_planning(...)
            target_parser = TargetParser(design_path, targets_file)
            solver = TrajectoryDualConstrainedSolver(robot_setup, target_parser)

            # Generate start configuration with default parameters
            start_conf = solver.generate_start_configuration(projector)

            # Generate start configuration with custom pose
            start_conf = solver.generate_start_configuration(
                projector,
                delta_pose_point=[0.5, 0.1, 0.8],
                delta_pose_euler=[-1.57, 1.57, 0.1],
                tool_index=0,
                max_attempts=200
            )
            ```

        Note:
            - The delta pose defines the target position for the bar/object relative to the robot base
            - The tool_index selects which tool transformation to use for grasp constraint
            - The method automatically sets the robot to the computed configuration
            - Joint angles are normalized to ensure consistent representation for planning algorithms
            - Collision checking is automatically performed during configuration generation
        """
        print("Initializing start configuration...")

        # Compute target bar pose from robot base and relative delta
        delta_pose = pp.Pose(point=delta_pose_point, euler=delta_pose_euler)
        base_pose = pp.get_pose(self.robot_setup.robot)
        bar_pose = pp.multiply(base_pose, delta_pose)

        # Get IK solution handles for both arms
        left_start_ik_handle = self.robot_setup.get_left_arm_ik_solution
        right_start_ik_handle = self.robot_setup.get_right_arm_ik_solution

        # Generate valid configurations using dual-arm constraint projection
        start_confs = projector.create_valid_confs(
            right_start_ik_handle, bar_pose, pp.invert(self.target_parser.tools_from_bar[tool_index]), delta=0, max_attempts=max_attempts, collision_fn=self.robot_setup.create_collision_fn(obstacle_bodies=self.robot_setup.obstacles)
        )

        # Select first valid configuration or exit if none found
        if start_confs is not None:
            start_conf = start_confs[0]
        else:
            print(f"✗ Failed to generate start configuration after {max_attempts} attempts")
            print("Consider adjusting delta_pose_point, delta_pose_euler, or increasing max_attempts")
            exit()

        # Normalize configuration to [-π, π] range
        start_conf = np.array(start_conf)
        start_conf = (start_conf + np.pi) % (2 * np.pi) - np.pi
        print(f"Start configuration: {list(start_conf)}")

        # Set robot to computed configuration
        self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, start_conf)

        print("✓ Start configuration generated successfully")
        return start_conf


def main():
    """
    Example usage of TrajectoryDualConstrainedSolver.
    """
    # Configuration paths
    design_study_path = os.path.join(DATA_DIR, "husky_assembly_design_study")
    design_case = "250806_RobotX_box_redo"
    target_name = "robotx_box_A6-S4_end"
    target_cell_state_path = os.path.join(design_study_path, design_case, "RobotCellStates", f"{target_name}_RobotCellState.json")

    # ------------------------------------------------------------------
    # Initialize Robot Setup for Planning
    # ------------------------------------------------------------------
    robot_setup, target_conf, projector = TrajectoryDualConstrainedSolver.initialize_robot_setup_for_planning(
        robot_name="r0", robot_type="husky_dual", target_cell_state_path=target_cell_state_path, use_scene_parser_gui=True, scene_parser_verbose=True
    )

    # ------------------------------------------------------------------
    # Initialize Target Parser
    # ------------------------------------------------------------------
    target_parser = TargetParser(os.path.join(design_study_path, design_case), "robotx_box_A6-S4_end_GraspTargets.json")

    # ------------------------------------------------------------------
    # Initialize Trajectory Solver
    # ------------------------------------------------------------------
    print("Initializing TrajectoryDualConstrainedSolver...")
    solver = TrajectoryDualConstrainedSolver(robot_setup, target_parser)

    # ------------------------------------------------------------------
    # Get Start Configuration
    # ------------------------------------------------------------------
    start_conf = solver.generate_start_configuration(projector)

    # ------------------------------------------------------------------
    # Plan Trajectory
    # ------------------------------------------------------------------
    print("Planning trajectory...")
    path = solver.plan(
        start_conf=start_conf,
        target_conf=target_conf,
        max_time=36000,  # Maximum planning time in seconds, conservatively set to 10 hours
        max_projection_attempts=100,  # Maximum attempts for constraint projection
        visualization=True,  # Enable visualization
    )

    # if path is not None:
    #     print(f"✓ Trajectory found with {len(path)} waypoints")

    #     # Wait for user to examine the path
    #     pp.wait_for_user()

    #     # Optional: Interactive trajectory playback
    #     print("Starting interactive trajectory playback...")
    #     print("Use the slider to control trajectory position. Press Ctrl+C to exit.")
    #     solver.interactive_trajectory_playback(path)

    # else:
    #     print("✗ No trajectory found")
    #     pp.wait_for_user()

    # ------------------------------------------------------------------
    # Save Trajectory
    # ------------------------------------------------------------------
    from compas.data import json_dump, json_load
    from compas_fab.robots import JointTrajectory, JointTrajectoryPoint
    from compas_fab.robots import Duration

    if path is not None:
        points = []
        for i, conf in enumerate(path):
            conf: np.ndarray
            point = JointTrajectoryPoint(joint_values=conf.tolist(), joint_types=robot_setup.joint_types, time_from_start=Duration(secs=i * 0.5, nsecs=0))
            points.append(point)

        trajectory = JointTrajectory(joint_names=robot_setup.joint_names, trajectory_points=points)
        print(f"Created trajectory with {len(points)} points")
        
        json_file = os.path.join(PROJECT_DIR, "data", f"{target_name}_robot_trajectory.json")
        json_dump(trajectory, json_file)
        print(f"Trajectory saved to {json_file}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    print("Cleaning up...")
    robot_setup.cleanup()
    print("✓ Example completed")


if __name__ == "__main__":
    main()
