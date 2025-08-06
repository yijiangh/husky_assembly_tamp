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
from utils.params import DATA_DIR

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
        draw_fn = self._get_draw_fn() if visualization else None

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
        # Remove the target object from obstacles and attach it to the robot
        world_from_bar = self.target_parser.world_from_bar
        world_from_bar_pos = world_from_bar[0]
        min_dist = np.inf
        min_dist_id = None

        for id in self.robot_setup.obstacles:
            pose = pp.get_pose(id)
            position = pose[0]
            dist = np.linalg.norm(np.array(position) - np.array(world_from_bar_pos))
            if dist < min_dist:
                min_dist = dist
                min_dist_id = id

        if min_dist_id is not None:
            print(f"Removing object with min distance: {min_dist}, id: {min_dist_id}")
            pp.set_color(min_dist_id, pp.YELLOW)
            self.robot_setup.remove_obstacle(min_dist_id)
            attachment = pp.create_attachment(self.robot_setup.robot, self.robot_setup.tool_link_right, min_dist_id)
            self.robot_setup.update_attachments([attachment])

        self.collision_fn = self.robot_setup.create_collision_fn(obstacle_bodies=self.robot_setup.obstacles)
        self.invalid_fn = self.robot_setup.create_invalid_fn(self.desired_right_from_left, obstacle_bodies=self.robot_setup.obstacles, resolution=1e-2)

    def _generate_projected_configurations(self, start_conf: np.ndarray, target_conf: np.ndarray, max_attempts: int):
        """Generate projected configurations for start and target."""
        # Project configurations using left arm as primary
        start_projected_confs_left = self.projector.project_multiple(start_conf[6:], max_attempts=max_attempts, collision_fn=self.collision_fn)
        target_projected_confs_left = self.projector.project_multiple(target_conf[6:], max_attempts=max_attempts, collision_fn=self.collision_fn)

        # Project configurations using right arm as primary
        start_projected_confs_right = self.projector.project_multiple_inv(start_conf[:6], max_attempts=max_attempts, collision_fn=self.collision_fn)
        target_projected_confs_right = self.projector.project_multiple_inv(target_conf[:6], max_attempts=max_attempts, collision_fn=self.collision_fn)

        print(f"Start projected confs (left primary): {start_projected_confs_left.shape}")
        print(f"Target projected confs (left primary): {target_projected_confs_left.shape}")
        print(f"Start projected confs (right primary): {start_projected_confs_right.shape}")
        print(f"Target projected confs (right primary): {target_projected_confs_right.shape}")

        # Generate all combinations
        start_projected_confs = []
        target_projected_confs = []

        for left_conf in start_projected_confs_left:
            for right_conf in start_projected_confs_right:
                start_projected_confs.append(np.concatenate([left_conf[:6], right_conf[6:]]))

        for left_conf in target_projected_confs_left:
            for right_conf in target_projected_confs_right:
                target_projected_confs.append(np.concatenate([left_conf[:6], right_conf[6:]]))

        self.start_projected_confs = np.array(start_projected_confs)
        self.target_projected_confs = np.array(target_projected_confs)

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

    def _get_draw_fn(self):
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
        for s in self.start_projected_confs:
            for t in self.target_projected_confs:
                path = pp.direct_path(s, t, extend_fn_direct, self.invalid_fn)
                if path is not None:
                    print("Direct path found!")
                    return path
                else:
                    print(f"No direct path between configurations")

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


def main():
    """
    Example usage of TrajectoryDualConstrainedSolver.
    """
    # Configuration paths
    design_study_path = os.path.join(DATA_DIR, "husky_assembly_design_study")
    design_case = "250707_RobotX_box_demo"
    start_cell_state_path = os.path.join(design_study_path, design_case, "RobotCellStates", "robotx_box_A6-S4_start_RobotCellState.json")
    target_cell_state_path = os.path.join(design_study_path, design_case, "RobotCellStates", "robotx_box_A6-S4_end_RobotCellState.json")

    # ------------------------------------------------------------------
    # Get Start Configuration
    # ------------------------------------------------------------------
    print("Initializing start configuration...")
    robot_setup_start = RobotSetup("r0", robot_type="husky_dual", robot_cell_state_path=start_cell_state_path, use_scene_parser_gui=True, scene_parser_verbose=True)

    start_conf = np.array(robot_setup_start.arm_target_angles)
    start_conf = (start_conf + np.pi) % (2 * np.pi) - np.pi
    print(f"Start configuration: {list(start_conf)}")

    # Clean up start setup
    pp.disconnect()
    del robot_setup_start

    # ------------------------------------------------------------------
    # Initialize Robot Setup for Planning
    # ------------------------------------------------------------------
    print("Initializing robot setup for planning...")
    robot_setup = RobotSetup("r0", robot_type="husky_dual", robot_cell_state_path=target_cell_state_path, use_scene_parser_gui=True, scene_parser_verbose=True)

    target_conf = np.array(robot_setup.arm_target_angles)
    target_conf = (target_conf + np.pi) % (2 * np.pi) - np.pi
    print(f"Target configuration: {list(target_conf)}")

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
    # Plan Trajectory
    # ------------------------------------------------------------------
    print("Planning trajectory...")
    path = solver.plan(start_conf=start_conf, target_conf=target_conf, max_time=600, max_projection_attempts=100, visualization=True)  # Maximum planning time in seconds  # Maximum attempts for constraint projection  # Enable visualization

    if path is not None:
        print(f"✓ Trajectory found with {len(path)} waypoints")

        # Wait for user to examine the path
        pp.wait_for_user()

        # Optional: Interactive trajectory playback
        print("Starting interactive trajectory playback...")
        print("Use the slider to control trajectory position. Press Ctrl+C to exit.")
        solver.interactive_trajectory_playback(path)

    else:
        print("✗ No trajectory found")
        pp.wait_for_user()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    print("Cleaning up...")
    robot_setup.cleanup()
    print("✓ Example completed")


if __name__ == "__main__":
    main()
