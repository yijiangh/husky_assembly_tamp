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

# Define DEFAULT_RESOLUTION if not imported
DEFAULT_RESOLUTION = math.radians(1.0)  # 0.05

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-o", "--output", action="store_true", help="Dump found solution path and planning graph.")
    # parser.add_argument("--bench", action="store_true", help="Run benchmark instead of single planning run.")
    # parser.add_argument("--interpolate-points", type=int, default=300, help="Number of points to interpolate the trajectory to (default 300)")
    # parser.add_argument("--plot-violations", action="store_true", help="Compute and plot constraint violations along the trajectory")

    # addSpaceOption(parser)
    # addPlannerOption(parser)
    # addConstrainedOptions(parser)
    # addAtlasOptions(parser)

    # args = parser.parse_args()
    design_study_path = os.path.join(DATA_DIR, "husky_assembly_design_study")
    design_case = "250707_RobotX_box_demo"
    start_cell_state_path = os.path.join(design_study_path, design_case, "RobotCellStates", "robotx_box_A6-S4_start_RobotCellState.json")
    target_cell_state_path = os.path.join(design_study_path, design_case, "RobotCellStates", "robotx_box_A6-S4_end_RobotCellState.json")

    # ------------------------------------------------------------------
    # Start Configuration
    # ------------------------------------------------------------------
    print("Initializing start configuration...")
    robot_setup = RobotSetup("r0", robot_type="husky_dual", robot_cell_state_path=start_cell_state_path, use_scene_parser_gui=True, scene_parser_verbose=True)
    print("✓ Start configuration initialized.")

    start_conf = np.array(robot_setup.arm_target_angles)
    start_conf = (start_conf + np.pi) % (2 * np.pi) - np.pi
    print(f"Start configuration: {list(start_conf)}")
    robot_setup.set_joint_positions(robot_setup.arm_joints, start_conf)

    pp.disconnect()
    del robot_setup

    # Normalize start_conf to be within [-pi, pi]
    start_conf = (start_conf + np.pi) % (2 * np.pi) - np.pi

    # ------------------------------------------------------------------
    # Environment & Robot Setup
    # ------------------------------------------------------------------
    print("Initializing PyBullet environment and robot setup...")
    robot_setup = RobotSetup("r0", robot_type="husky_dual", robot_cell_state_path=target_cell_state_path, use_scene_parser_gui=True, scene_parser_verbose=True)
    print("✓ Robot setup complete.")

    target_conf = np.array(robot_setup.arm_target_angles)
    target_conf = (target_conf + np.pi) % (2 * np.pi) - np.pi
    print(f"Target configuration: {list(target_conf)}")
    robot_setup.set_joint_positions(robot_setup.arm_joints, target_conf)

    world_from_left = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
    world_from_right = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
    desired_right_from_left = pp.multiply(pp.invert(world_from_right), world_from_left)
    projector = DualArmProjection(robot_setup, desired_right_from_left)

    # -------------------- setup target parser --------------------#
    target_parser = TargetParser(os.path.join(design_study_path, design_case), "robotx_box_A6-S4_end_GraspTargets.json")
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
    robot_setup.set_joint_positions(robot_setup.arm_joints, target_conf)
    attachment = pp.create_attachment(robot_setup.robot, robot_setup.tool_link_right, min_dist_id)
    robot_setup.update_attachments([attachment])

    # -------------------- Set up collision checking --------------------#
    collision_fn = robot_setup.create_collision_fn(obstacle_bodies=robot_setup.obstacles)

    # -------------------- Projected configurations --------------------#
    start_projected_confs_left = projector.project_multiple(start_conf[6:], max_attempts=100, collision_fn=collision_fn)
    print(f"Projected configurations for start configuration: {start_projected_confs_left.shape}")
    target_projected_confs_left = projector.project_multiple(target_conf[6:], max_attempts=100, collision_fn=collision_fn)
    print(f"Projected configurations for target configuration: {target_projected_confs_left.shape}")

    start_projected_confs_right = projector.project_multiple_inv(start_conf[:6], max_attempts=100, collision_fn=collision_fn)
    print(f"Projected configurations for start configuration: {start_projected_confs_right.shape}")
    target_projected_confs_right = projector.project_multiple_inv(target_conf[:6], max_attempts=100, collision_fn=collision_fn)
    print(f"Projected configurations for target configuration: {target_projected_confs_right.shape}")

    start_projected_confs = []
    target_projected_confs = []
    for left_conf in start_projected_confs_left:
        for right_conf in start_projected_confs_right:
            start_projected_confs.append(np.concatenate([left_conf[:6], right_conf[6:]]))
    for left_conf in target_projected_confs_left:
        for right_conf in target_projected_confs_right:
            target_projected_confs.append(np.concatenate([left_conf[:6], right_conf[6:]]))
    start_projected_confs = np.array(start_projected_confs)
    target_projected_confs = np.array(target_projected_confs)
    print(f"Start projected configurations: {start_projected_confs.shape}")
    print(f"Target projected configurations: {target_projected_confs.shape}")

    robot_setup.set_joint_positions(robot_setup.arm_joints, start_projected_confs[0])
    pose = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
    pp.draw_pose(pose, length=0.2)

    robot_setup.set_joint_positions(robot_setup.arm_joints, target_projected_confs[0])
    pose = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
    pp.draw_pose(pose, length=0.2)

    def get_sample_fn():
        lower, upper = [-np.pi] * 12, [np.pi] * 12
        # cache = list(start_projected_confs) + list(target_projected_confs)
        cache = []

        def fn():
            if len(cache) == 0:
                print("Generating cache...")
                while len(cache) < 100:
                    right_conf = np.random.uniform(lower[6:], upper[6:])
                    projected_confs = projector.project_multiple(right_conf, max_attempts=10, collision_fn=collision_fn)

                    # -------------------- Multiple samples --------------------#
                    # if projected_confs is not None:
                    #     cache.extend(list(projected_confs))
                    #     for conf in projected_confs:
                    #         robot_setup.set_joint_positions(robot_setup.arm_joints, conf)
                    #         pose = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
                    #         # pp.draw_pose(pose, length=0.05)
                    #     print(f"Cache: {len(cache)}")

                    # -------------------- Single sample --------------------#
                    if projected_confs is not None:
                        cache.append(projected_confs[0])
                        robot_setup.set_joint_positions(robot_setup.arm_joints, projected_confs[0])
                        pose = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
                        # pp.draw_pose(pose, length=0.05)
                        print(f"Cache: {len(cache)}")

                print("Cache generated!")
            sample = cache.pop()
            print(f"Cache: {len(cache)}")
            return sample

        return fn

    def get_draw_fn():
        pose_cache = set()
        segment_cache = set()

        def pose_to_tuple(pose, decimals=3):
            # Convert pose (position, orientation) to a tuple of rounded floats for hashing
            pos, orn = pose
            pos_tuple = tuple(np.round(pos, decimals=decimals))
            orn_tuple = tuple(np.round(orn, decimals=decimals))
            return pos_tuple + orn_tuple

        def segment_to_tuple(pose1, pose2, decimals=3):
            # Segment is unordered, so sort the two pose tuples
            t1 = pose_to_tuple(pose1, decimals)
            t2 = pose_to_tuple(pose2, decimals)
            return tuple(sorted([t1, t2]))

        start_tree_set = set()
        robot_setup.set_joint_positions(robot_setup.arm_joints, start_conf)
        start_pose_tuple = pose_to_tuple(pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right))
        start_tree_set.add(start_pose_tuple)

        target_tree_set = set()
        robot_setup.set_joint_positions(robot_setup.arm_joints, target_conf)
        target_pose_tuple = pose_to_tuple(pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right))
        target_tree_set.add(target_pose_tuple)

        def fn(conf, segment, valid=None, valid_right=None):
            robot_setup.set_joint_positions(robot_setup.arm_joints, conf)
            pose_1 = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
            pose_1_tuple = pose_to_tuple(pose_1)

            # Draw pose if not already drawn
            if pose_1_tuple not in pose_cache:
                # pp.draw_pose(pose_1, length=0.025)
                pose_cache.add(pose_1_tuple)

            if len(segment) > 0:
                robot_setup.set_joint_positions(robot_setup.arm_joints, segment[1])
                pose_2 = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
                pose_2_tuple = pose_to_tuple(pose_2)

                # Draw pose_2 if not already drawn
                if pose_2_tuple not in pose_cache:
                    # pp.draw_pose(pose_2, length=0.025)
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

            else:
                pass

        return fn

    def get_circular_diff(q1, q2):
        """Compute the shortest angular difference between two angle arrays."""
        q1 = np.array(q1)
        q2 = np.array(q2)
        # Normalize angles to [-pi, pi]
        q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
        q2 = (q2 + np.pi) % (2 * np.pi) - np.pi
        # Compute shortest angular distance for each joint
        diff = q2 - q1
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return diff

    def create_invalid_configuration():
        """Create a configuration that is guaranteed to fail collision checking."""
        data = np.array([2.15557306, -1.05715414, 1.63506225, -0.25357488, 1.23252519, -1.42178216, 1.31437349, 1.25663757, 0.39683294, -3.90218878, -1.71960878, 0.05148935])
        # Convert the data to the range [-pi, pi]
        data = (data + np.pi) % (2 * np.pi) - np.pi
        return data

    def get_extend_fn(body, joints, projector: DualArmProjection, resolutions=None, norm=2, check_continuous=True):
        if resolutions is None:
            resolutions = DEFAULT_RESOLUTION * np.ones(len(joints))

        def fn(q1, q2):
            q1_right = np.array(q1[6:])
            q2_right = np.array(q2[6:])

            right_diff = get_circular_diff(q1_right, q2_right)
            right_steps = int(np.ceil(np.linalg.norm(right_diff / resolutions[6:], ord=norm)))

            q_left_init = np.array(q1[:6])
            q_left_target = np.array(q2[:6])

            for i in range(right_steps + 1):
                if right_steps == 0:
                    t = 0.0
                else:
                    t = i / right_steps

                q_right_interp = q1_right + t * right_diff
                q_right_interp = (q_right_interp + np.pi) % (2 * np.pi) - np.pi

                with pp.LockRenderer():
                    projected_conf = projector.project(q_right_interp, q_left_init)

                if projected_conf is not None and np.linalg.norm(get_circular_diff(projected_conf[:6], q_left_init)) < 0.5:
                    q_left_init = np.array(projected_conf[:6])
                    yield tuple(projected_conf)
                else:
                    collision_conf = create_invalid_configuration()
                    yield tuple(collision_conf)

        def fn_continuous(q1, q2):
            q1_right = np.array(q1[6:])
            q2_right = np.array(q2[6:])

            right_diff = get_circular_diff(q1_right, q2_right)
            right_steps = int(np.ceil(np.linalg.norm(right_diff / resolutions[6:], ord=norm)))

            q_left_init = np.array(q1[:6])
            q_left_target = np.array(q2[:6])

            for i in range(right_steps + 1):
                if right_steps == 0:
                    t = 0.0
                else:
                    t = i / right_steps

                q_right_interp = q1_right + t * right_diff
                q_right_interp = (q_right_interp + np.pi) % (2 * np.pi) - np.pi

                with pp.LockRenderer():
                    projected_conf = projector.project(q_right_interp, q_left_init)

                if t < 1 - 0.01 and projected_conf is not None and np.linalg.norm(get_circular_diff(projected_conf[:6], q_left_init)) < 0.5:
                    q_left_init = np.array(projected_conf[:6])
                    yield tuple(projected_conf)
                elif t >= 1 - 0.01 and projected_conf is not None and np.linalg.norm(get_circular_diff(projected_conf[:6], q_left_target)) < 0.1:
                    q_left_init = np.array(projected_conf[:6])
                    yield tuple(projected_conf)
                else:
                    collision_conf = create_invalid_configuration()
                    yield tuple(collision_conf)

        if check_continuous:
            return fn_continuous
        else:
            return fn

    def get_distance_fn(body, joints, weights=None):
        if weights is None:
            weights = 1 * np.ones(len(joints))
        difference_fn = get_circular_diff

        def fn(q1, q2):
            diff = np.array(difference_fn(q2, q1))
            return np.sqrt(np.dot(weights, diff * diff))

        return fn

    resolutions = np.array([1.0 if j in [] else 5.0 / 180.0 * np.pi for j in robot_setup.arm_joints])

    sample_fn = get_sample_fn()
    extend_fn_continuous = get_extend_fn(robot_setup.robot, robot_setup.arm_joints, projector, resolutions=resolutions, check_continuous=True)
    extend_fn_direct = get_extend_fn(robot_setup.robot, robot_setup.arm_joints, projector, resolutions=resolutions, check_continuous=False)
    invalid_fn = robot_setup.create_invalid_fn(desired_right_from_left, obstacle_bodies=robot_setup.obstacles, resolution=1e-2)
    distance_fn = get_distance_fn(robot_setup.robot, robot_setup.arm_joints)
    draw_fn = get_draw_fn()

    # Cross-iterate all pairs of start and target, check if there is a direct path
    has_direct_path = False
    for s in start_projected_confs:
        for t in target_projected_confs:
            path = pp.direct_path(s, t, extend_fn_direct, invalid_fn)
            if path is not None:
                print(f"Direct path found!")
                has_direct_path = True
                break
            else:
                print(f"No direct path between {s} and {t}")
        if has_direct_path:
            break
    if not has_direct_path:
        print("No direct path found for any start-target pair.")
        path = robot_setup.plan_manipulator_path(
            start_conf, target_conf, attachments=[], obstacles=robot_setup.obstacles, sample_fn=sample_fn, collision_fn=invalid_fn, extend_fn=extend_fn_continuous, max_time=600, draw_fn=draw_fn, distance_fn=distance_fn
        )

    if path is not None:
        # Remove trailing frames where the right arm joint angles remain unchanged, keeping only the first occurrence
        if len(path) > 1:
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
            path = path[: cutoff_idx + 1]
        print(f"Path: {path}")
        prev_conf = path[0]
        robot_setup.set_joint_positions(robot_setup.arm_joints, prev_conf)
        prev_pose = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
        for idx, conf in enumerate(path[1:]):
            robot_setup.set_joint_positions(robot_setup.arm_joints, conf)
            pose = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
            pp.add_line(prev_pose[0], pose[0], color=[0, 1, 0, 0.5], width=2.0)
            prev_pose = pose
    else:
        print("No path found.")

    pp.wait_for_user()

    if path is not None:
        # print("Interpolating path...")
        # interp_path = []
        # start_conf = path[0]
        # for temp_conf in path[1:]:
        #     interp_path.extend(pp.direct_path(start_conf, temp_conf, extend_fn_refine, invalid_fn))
        #     start_conf = temp_conf

        # result_traj = np.array(interp_path)
        result_traj = np.array(path)

        slider = pybullet.addUserDebugParameter("traj_idx", 0, result_traj.shape[0] - 1, 0)
        current_index = -1

        try:
            while True:
                idx = int(pybullet.readUserDebugParameter(slider))
                if idx != current_index:
                    current_index = idx
                    conf = result_traj[current_index]
                    robot_setup.set_joint_positions(robot_setup.arm_joints, conf)
                    print(f"Conf: {conf}, is valid: {not invalid_fn(conf)}")
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            # Cleanup visualization elements
            print("Cleaning up visualization...")
            robot_setup.cleanup()
            print("✓ Cleanup complete.")
