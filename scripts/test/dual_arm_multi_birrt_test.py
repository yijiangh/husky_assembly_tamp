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
from robot.robot_setup import RobotSetup
from utils.params import DATA_DIR

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
    design_case = "test"
    target_cell_state_path = os.path.join(design_study_path, design_case, "RobotCellStates", "robotx_box_A0-G_RobotCellState.json")
    start_cell_state_path = os.path.join(design_study_path, design_case, "RobotCellStates", "robotx_box_A8-S7_start_state_RobotCellState.json")

    # ------------------------------------------------------------------
    # Start Configuration
    # ------------------------------------------------------------------
    # print("Initializing start configuration...")
    # robot_setup = RobotSetup("r0", robot_type="husky_dual", robot_cell_state_path=start_cell_state_path, use_scene_parser_gui=True, scene_parser_verbose=True)
    # print("✓ Start configuration initialized.")

    # start_conf = np.array(robot_setup.arm_target_angles)
    # start_conf = (start_conf + np.pi) % (2 * np.pi) - np.pi
    # print(f"Start configuration: {list(start_conf)}")
    # robot_setup.set_joint_positions(robot_setup.arm_joints, start_conf)

    # world_from_right = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
    # world_from_left = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
    # collision_fn = robot_setup.create_collision_fn(obstacle_bodies=robot_setup.obstacles)
    # projector = DualArmProjection(robot_setup, pp.multiply(pp.invert(world_from_right), world_from_left))

    # for idx, conf in enumerate(projected_confs):
    #     robot.set_joint_positions(robot.arm_joints, conf)
    #     pp.wait_for_user(f"Projected configuration {idx}: {list(conf)}")
    # pp.wait_for_user()

    # pp.disconnect()
    # del robot_setup

    start_conf = np.array([1.44588847, -1.07000539, 2.06615573, 2.78551221, -0.66673551, 0.12182059, -4.49743795, -0.19841623, 1.19049788, -3.16904378, -2.07907343, -0.72975159])

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

    collision_fn = robot_setup.create_collision_fn(obstacle_bodies=robot_setup.obstacles)
    start_projected_confs = projector.project_multiple(start_conf[6:], max_attempts=100, collision_fn=collision_fn)
    print(f"Projected configurations for start configuration: {start_projected_confs.shape}")
    target_projected_confs = projector.project_multiple(target_conf[6:], max_attempts=100, collision_fn=collision_fn)
    print(f"Projected configurations for target configuration: {target_projected_confs.shape}")

    def get_sample_fn():
        lower, upper = [-np.pi] * 12, [np.pi] * 12
        cache = list(start_projected_confs) + list(target_projected_confs)

        def fn():
            if len(cache) == 0:
                print("Generating cache...")
                while len(cache) < 100:
                    right_conf = np.random.uniform(lower[6:], upper[6:])
                    projected_confs = projector.project_multiple(right_conf, max_attempts=10, collision_fn=collision_fn)
                    if projected_confs is not None:
                        cache.extend(list(projected_confs))
                        print(f"Cache: {len(cache)}")
                print("Cache generated!")
            sample = cache.pop()
            print(f"Cache: {len(cache)}")
            return sample

        return fn

    resolutions = np.array([1.0 if j in [] else 1.0 / 180.0 * np.pi for j in robot_setup.arm_joints])

    sample_fn = get_sample_fn()
    extend_fn = pp.get_extend_fn(robot_setup.robot, robot_setup.arm_joints, resolutions=resolutions)
    invalid_fn = robot_setup.create_invalid_fn(desired_right_from_left, obstacle_bodies=robot_setup.obstacles, resolution=1e-2)

    # while True:
    #     sample = sample_fn()
    #     if sample is None:
    #         continue
    #     robot_setup.set_joint_positions(robot_setup.arm_joints, sample)
    #     pp.wait_for_user(f"Sample: {sample}")

    # def circular_distance_fn(q1, q2):
    #     q1 = np.array(q1)
    #     q2 = np.array(q2)
    #     # Normalize angles to [-pi, pi]
    #     q1 = np.mod(q1 + np.pi, 2 * np.pi) - np.pi
    #     q2 = np.mod(q2 + np.pi, 2 * np.pi) - np.pi
    #     # Compute shortest angular distance for each joint
    #     diff = np.array(q1) - np.array(q2)
    #     diff = (diff + np.pi) % (2 * np.pi) - np.pi
    #     return np.linalg.norm(diff)

    # distance_fn = circular_distance_fn

    # path = robot_setup.plan_manipulator_path(start_conf, target_conf, attachments=[], obstacles=robot_setup.obstacles, sample_fn=sample_fn, collision_fn=collision_fn, extend_fn=extend_fn, distance_fn=distance_fn, max_time=600)
    path = robot_setup.plan_manipulator_path(start_conf, target_conf, attachments=[], obstacles=robot_setup.obstacles, sample_fn=sample_fn, collision_fn=invalid_fn, extend_fn=extend_fn, max_time=600)

    print(f"Path: {path}")

    if path is not None:
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
