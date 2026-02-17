"""
Trajectory Dual-Arm Constrained Solver - Testbench

Standalone testbench for the dual-arm constrained motion planner.
Loads the robot directly from URDF (no compas_fab), provides a GUI
for configuring start/end bar poses and grasps, and includes timing
instrumentation for bottleneck identification.

Usage:
    cd external/husky_assembly_tamp/scripts
    python -m motion_planner.trajectory_testbenc
"""

import argparse
import cProfile
import json
import os
import subprocess
import pstats
import sys
import time

import numpy as np
import pybullet
import pybullet_planning as pp

from compas.data import json_dump, json_load
from compas_fab.robots import JointTrajectory, JointTrajectoryPoint

from husky_assembly_tamp.robot.dual_arm_projection import DualArmProjection
from husky_assembly_tamp.robot.robot_setup import (
    HUSKY_DUAL_ARM_JOINT_NAMES,
    HUSKY_DUAL_INIT_ARM_JOINT_ANGLES,
    HUSKY_DUAL_URDF_PATH,
    RobotSetup,
)
import husky_assembly_tamp.motion_planner.trajectory_dual_cart_constrained_solver as solver_mod
from husky_assembly_tamp.motion_planner.trajectory_dual_cart_constrained_solver import (
    TrajectoryDualCartConstrainedSolver,
)
from husky_assembly_tamp.utils.util import normalize_angles, setup_logger, reinit_logger_stream


# Global logger instance
logger = setup_logger("trajectory_testbench")


# ---------------------------------------------------------------------------
# GUI helpers
# ---------------------------------------------------------------------------

class Button:
    def __init__(self, name, cid=0):
        self.cid = cid
        self.dbg_param = pybullet.addUserDebugParameter(name, 1.0, 0.0, 0.0, physicsClientId=cid)
        self.prev_value = pybullet.readUserDebugParameter(self.dbg_param, physicsClientId=cid)

    def pressed(self):
        new_value = pybullet.readUserDebugParameter(self.dbg_param, physicsClientId=self.cid)
        if new_value != self.prev_value:
            self.prev_value = new_value
            return True
        return False


class PoseSliders:
    """Six sliders for position (x, y, z) and orientation (roll, pitch, yaw)."""

    def __init__(self, prefix, cid=0, defaults=None, pos_range=2.0, angle_range=np.pi):
        self.cid = cid
        if defaults is None:
            defaults = [0.4, 0.0, 0.75, np.pi, np.pi / 2, np.pi / 2]

        labels = ["x", "y", "z", "roll", "pitch", "yaw"]
        mins = [-pos_range, -pos_range, 0.0, -angle_range, -angle_range, -angle_range]
        maxs = [pos_range, pos_range, 2.0, angle_range, angle_range, angle_range]

        self.sliders = []
        for i, label in enumerate(labels):
            s = pybullet.addUserDebugParameter(
                f"{prefix} {label}", mins[i], maxs[i], defaults[i], physicsClientId=cid
            )
            self.sliders.append(s)
        self._prev = list(defaults)

    def read(self):
        vals = [pybullet.readUserDebugParameter(s, physicsClientId=self.cid) for s in self.sliders]
        return vals

    def changed(self):
        vals = self.read()
        if not np.allclose(vals, self._prev, atol=1e-6):
            self._prev = list(vals)
            return True
        return False

    def as_pose(self):
        vals = self.read()
        return pp.Pose(point=vals[:3], euler=pp.Euler(*vals[3:]))


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self.records = {}

    def start(self, name):
        self.records[name] = time.time()

    def stop(self, name):
        elapsed = time.time() - self.records[name]
        self.records[name] = elapsed
        return elapsed

    def summary(self):
        logger.info("--- Timing Summary ---")
        for name, val in self.records.items():
            logger.info(f"  {name}: {val:.3f}s")
        logger.info("----------------------")


# ---------------------------------------------------------------------------
# JSON loading helpers
# ---------------------------------------------------------------------------

def load_grasp_targets(json_path):
    """Parse GraspTargets JSON -> list of (world_from_bar, world_from_tool0) PyBullet poses."""
    with open(json_path) as f:
        raw = json.load(f)
    targets = []
    for item in raw:
        d = item["data"]
        world_from_bar = pp.pose_from_tform(np.array(d["world_from_bar"]["data"]["matrix"]))
        world_from_tool0 = pp.pose_from_tform(np.array(d["world_from_tool0"]["data"]["matrix"]))
        targets.append((world_from_bar, world_from_tool0))
    return targets


def load_robot_cell_state(json_path):
    """Parse RobotCellState JSON -> joint_values (12 floats).

    Lightweight loading using standard json, no compas_fab scene reconstruction.
    """
    with open(json_path) as f:
        data = json.load(f)
    state = data["data"]
    return state["robot_configuration"]["data"]["joint_values"]


def pose_to_slider_values(pose):
    """Convert PyBullet pose (point, quat) to slider values [x, y, z, roll, pitch, yaw].

    Euler angles are wrapped to [-pi, pi] so they stay within slider bounds.
    pybullet's getEulerFromQuaternion can return values outside this range
    (e.g. yaw = -3pi/2 instead of +pi/2).
    """
    point, quat = pose
    euler = pp.euler_from_quat(quat)
    # Wrap euler angles to [-pi, pi]
    euler = [((e + np.pi) % (2 * np.pi)) - np.pi for e in euler]
    return list(point) + euler


# ---------------------------------------------------------------------------
# Trajectory save / load helpers
# ---------------------------------------------------------------------------

def save_path_as_joint_trajectory(path, joint_names, out_path):
    """Save a list of joint configurations as a compas_fab JointTrajectory JSON.

    Parameters
    ----------
    path : list of array-like
        Each element is a list/array of joint values (12 floats for dual arm).
    joint_names : list of str
        Joint names matching the values in each configuration.
    out_path : str
        File path to write the JSON to.
    """
    points = []
    for conf in path:
        pt = JointTrajectoryPoint(
            joint_values=[float(v) for v in conf],
            joint_names=joint_names,
        )
        points.append(pt)
    traj = JointTrajectory(
        trajectory_points=points,
        joint_names=joint_names,
    )
    json_dump(traj, out_path)
    logger.info(f"Trajectory saved to {out_path}  ({len(path)} points)")


def load_joint_trajectory_as_path(json_path, joint_names=None):
    """Load a compas_fab JointTrajectory JSON and return a list of numpy arrays.

    Parameters
    ----------
    json_path : str
        Path to the JointTrajectory JSON file.
    joint_names : list of str, optional
        If given, reorder loaded values to match this joint name order.

    Returns
    -------
    list of np.ndarray
    """
    traj = json_load(json_path)
    if joint_names and traj.joint_names and traj.joint_names != joint_names:
        # Reorder to match expected joint_names
        idx_map = [traj.joint_names.index(n) for n in joint_names]
        return [np.array([pt.joint_values[i] for i in idx_map]) for pt in traj.points]
    return [np.array(pt.joint_values) for pt in traj.points]


def scan_trajectory_files(directory):
    """Return sorted list of *_JointTrajectory.json filenames in directory."""
    if not os.path.isdir(directory):
        return []
    files = [f for f in os.listdir(directory) if f.endswith("_JointTrajectory.json")]
    files.sort()
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trajectory Dual-Arm Constrained Solver - Testbench")
    default_data_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..",
        "data", "husky_assembly_design_study", "250904_transfer_path_test", "RobotCellStates",
    )
    parser.add_argument("--grasp-json", type=str,
                        default=os.path.join(default_data_dir, "IK_test__GraspTargets.json"),
                        help="Path to GraspTargets JSON file")
    parser.add_argument("--start-state", type=str,
                        default=os.path.join(default_data_dir, "IK_test__20250905_101010_RobotCellState.json"),
                        help="Path to RobotCellState JSON for start configuration")
    parser.add_argument("--end-state", type=str, 
                        default=os.path.join(default_data_dir, "IK_test__20250909_235058_RobotCellState.json"),
                        help="Path to RobotCellState JSON for goal configuration")
    parser.add_argument("--traj-dir", type=str,
                        default=default_data_dir,
                        help="Directory to save/load JointTrajectory JSON files")
    args = parser.parse_args()
    timer = Timer()

    # ------------------------------------------------------------------
    # 1. Direct URDF loading (no compas_fab)
    # ------------------------------------------------------------------
    timer.start("robot_loading")

    cid = pp.connect(use_gui=True)
    # On Windows, pybullet.GUI can invalidate stdout/stderr handles (WinError 6).
    # Reopen them to avoid OSError on subsequent print() calls.
    if sys.platform == "win32":
        sys.stdout = open("CONOUT$", "w")
        sys.stderr = open("CONOUT$", "w")
        # Re-initialize logger's console handler after stdout is restored
        reinit_logger_stream(logger)

    # pp.connect disables the debug GUI panel — re-enable it for sliders/buttons
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 1, physicsClientId=cid)
    pybullet.resetDebugVisualizerCamera(
        cameraDistance=2.5, cameraYaw=45, cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5], physicsClientId=cid,
    )

    with pp.LockRenderer():
        robot = pp.load_pybullet(HUSKY_DUAL_URDF_PATH, fixed_base=True)

    arm_joints = pp.joints_from_names(robot, HUSKY_DUAL_ARM_JOINT_NAMES)
    pp.set_joint_positions(robot, arm_joints, HUSKY_DUAL_INIT_ARM_JOINT_ANGLES)

    # Create a simple bar (box primitive)
    bar_height = 1
    bar = pp.create_cylinder(radius=0.015, height=bar_height, color=(0.8, 0.4, 0.1, 0.6))

    # Place bar at a reasonable initial position relative to robot
    # init_bar_pose = pp.Pose(point=[0.4, 0.0, 0.75], euler=pp.Euler(np.pi, np.pi / 2, np.pi / 2))
    # pp.set_pose(bar, init_bar_pose)

    # Create RobotSetup via robot_data dict (bypasses SceneParser/compas_fab)
    robot_setup = RobotSetup(
        robot_name="r0",
        robot_type="husky_dual",
        robot_data={
            "robot_id": robot,
            "obstacles": [],
            "target_bar": bar,
            "joint_values": list(HUSKY_DUAL_INIT_ARM_JOINT_ANGLES),
        },
    )

    load_time = timer.stop("robot_loading")
    logger.info(f"Robot loaded in {load_time:.3f}s (direct URDF, no compas_fab)")

    # ------------------------------------------------------------------
    # 2. Load scene from JSON (if provided)
    # ------------------------------------------------------------------
    start_defaults = [0.4, 0.0, 0.75, np.pi, np.pi / 2, np.pi / 2]
    end_defaults = [0.3, 0.2, 0.6, np.pi, np.pi / 2, 0.0]

    grasp_bar_from_right = None
    grasp_bar_from_left = None
    projector = None
    solver = None

    start_confs = None
    end_confs = None
    start_conf_idx_slider = None  # created in GUI section
    end_conf_idx_slider = None  # created in GUI section

    path = None
    path_current_idx = -1

    # Load grasp first (needed to derive bar pose from FK)
    if args.grasp_json:
        grasp_targets = load_grasp_targets(args.grasp_json)
        # targets[0] = right arm, targets[1] = left arm
        world_from_bar_l, world_from_tool0_left = grasp_targets[0]
        world_from_bar_r, world_from_tool0_right = grasp_targets[1]

        grasp_bar_from_right = pp.multiply(pp.invert(world_from_bar_r), world_from_tool0_right)
        grasp_bar_from_left = pp.multiply(pp.invert(world_from_bar_l), world_from_tool0_left)

        desired_right_from_left = pp.multiply(pp.invert(world_from_tool0_right), world_from_tool0_left)
        projector = DualArmProjection(robot_setup, desired_right_from_left)

        solver_mod.bar_from_right = grasp_bar_from_right
        solver_mod.bar_from_left = grasp_bar_from_left

        logger.info(f"Grasp loaded from {args.grasp_json}")
        logger.debug(f"  bar_from_right: pos={np.round(grasp_bar_from_right[0], 4)}, "
              f"quat={np.round(grasp_bar_from_right[1], 4)}")
        logger.debug(f"  bar_from_left:  pos={np.round(grasp_bar_from_left[0], 4)}, "
              f"quat={np.round(grasp_bar_from_left[1], 4)}")

    # Load start/end states and derive bar pose via FK + grasp
    if args.start_state:
        joint_values = load_robot_cell_state(args.start_state)
        pp.set_joint_positions(robot, arm_joints, joint_values)
        robot_setup.robot_data["joint_values"] = list(joint_values)
        if grasp_bar_from_left is not None:
            world_from_tool0_left = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
            bar_pose_left = pp.multiply(world_from_tool0_left, pp.invert(grasp_bar_from_left))

            # Draw tool0 poses for left and right arms (start state)
            pp.draw_pose(world_from_tool0_left)
            pp.add_text("start_tool0_left", world_from_tool0_left[0], color=(0.0, 0.5, 1.0, 1.0))

            world_from_tool0_right = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
            bar_pose_right = pp.multiply(world_from_tool0_right, pp.invert(grasp_bar_from_right))

            pp.draw_pose(world_from_tool0_right)
            pp.add_text("start_tool0_right", world_from_tool0_right[0], color=(1.0, 0.5, 0.0, 1.0))

            pos_equal = np.allclose(bar_pose_left[0], bar_pose_right[0], atol=1e-4)
            quat_equal = np.allclose(bar_pose_left[1], bar_pose_right[1], atol=1e-4)
            if not pos_equal or not quat_equal:
                logger.warning("Bar poses computed from left and right arms do not match!")
                logger.warning(f"  From left:  pos={np.round(bar_pose_left[0], 4)}, quat={np.round(bar_pose_left[1], 4)}")
                logger.warning(f"  From right: pos={np.round(bar_pose_right[0], 4)}, quat={np.round(bar_pose_right[1], 4)}")

            # Use left arm result (default), but this block can easily be swapped to right if desired
            bar_pose = bar_pose_left
            start_defaults = pose_to_slider_values(bar_pose)
            pp.set_pose(bar, bar_pose)
            pp.draw_pose(bar_pose)
            pp.add_text("Loaded Start", bar_pose[0], color=(0.0, 0.8, 0.0, 1.0))

        logger.info(f"Start state loaded from {args.start_state}")
        logger.debug(f"  joint_values: {[round(v, 4) for v in joint_values]}")
        if grasp_bar_from_left is not None:
            logger.debug(f"  bar_pose (from FK): pos={np.round(bar_pose[0], 4)}, quat={np.round(bar_pose[1], 4)}")

    if args.end_state:
        end_joint_values = load_robot_cell_state(args.end_state)
        if grasp_bar_from_left is not None:
            # Temporarily set end joints to compute FK, then restore start
            pp.set_joint_positions(robot, arm_joints, end_joint_values)
            world_from_tool0_left = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
            end_bar_pose = pp.multiply(world_from_tool0_left, pp.invert(grasp_bar_from_left))
            end_defaults = pose_to_slider_values(end_bar_pose)
            pp.draw_pose(end_bar_pose)
            pp.add_text("Loaded End", end_bar_pose[0], color=(0.8, 0.0, 0.0, 1.0))

            # Draw tool0 poses for left and right arms (end state)
            pp.draw_pose(world_from_tool0_left)
            pp.add_text("end_tool0_left", world_from_tool0_left[0], color=(0.0, 0.5, 1.0, 1.0))

            # Do a left-right agreement check for the end state bar pose, similar to the start state
            world_from_tool0_right = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
            pp.draw_pose(world_from_tool0_right)
            pp.add_text("end_tool0_right", world_from_tool0_right[0], color=(1.0, 0.5, 0.0, 1.0))

            end_bar_pose_right = pp.multiply(world_from_tool0_right, pp.invert(grasp_bar_from_right))

            pos_equal = np.allclose(end_bar_pose[0], end_bar_pose_right[0], atol=1e-4)
            quat_equal = np.allclose(end_bar_pose[1], end_bar_pose_right[1], atol=1e-4)
            if not pos_equal or not quat_equal:
                logger.warning("Bar poses for END computed from left and right arms do not match!")
                logger.warning(f"  From left:  pos={np.round(end_bar_pose[0], 4)}, quat={np.round(end_bar_pose[1], 4)}")
                logger.warning(f"  From right: pos={np.round(end_bar_pose_right[0], 4)}, quat={np.round(end_bar_pose_right[1], 4)}")

            # Restore start configuration
            if args.start_state:
                pp.set_joint_positions(robot, arm_joints, joint_values)
        logger.info(f"End state loaded from {args.end_state}")
        logger.debug(f"  joint_values: {[round(v, 4) for v in end_joint_values]}")
        if grasp_bar_from_left is not None:
            logger.debug(f"  bar_pose (from FK): pos={np.round(end_bar_pose[0], 4)}, quat={np.round(end_bar_pose[1], 4)}")

    # ------------------------------------------------------------------
    # 3. Create GUI
    # ------------------------------------------------------------------
    logger.info("=== GUI Controls ===")
    logger.info("  Buttons: Plan Path (auto-solves IK for start & end)")
    logger.info("  Sliders: Start bar pose, End bar pose")
    logger.info("====================")

    # btn_set_grasp = Button("Set Grasp From Current", cid=cid)
    btn_plan = Button("Plan Path", cid=cid)
    path_slider = pybullet.addUserDebugParameter("Path t", 0.0, 1.0, 0.0, physicsClientId=cid)
    start_conf_idx_slider = pybullet.addUserDebugParameter("Start conf t", 0.0, 1.0, 0.0, physicsClientId=cid)
    end_conf_idx_slider = pybullet.addUserDebugParameter("End conf t", 0.0, 1.0, 0.0, physicsClientId=cid)

    # Mode selector: 0=start, 1=end, 2=path playback
    mode_slider = pybullet.addUserDebugParameter("Mode (0=start, 1=end, 2=path)", 0, 2, 0, physicsClientId=cid)

    # ---- Pose sliders (for adjusting start/end bar poses) ----
    start_sliders = PoseSliders("Start", cid=cid, defaults=start_defaults)
    end_sliders = PoseSliders("End", cid=cid, defaults=end_defaults)
    # grasp_sliders = PoseSliders(
    #     "Grasp",
    #     cid=cid,
    #     defaults=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #     pos_range=0.3,
    #     angle_range=np.pi,
    # )

    # Ghost bars for showing start (green) and end (red) simultaneously
    ghost_start = pp.create_cylinder(radius=0.015, height=bar_height, color=(0.0, 0.8, 0.0, 0.4))
    ghost_end = pp.create_cylinder(radius=0.015, height=bar_height, color=(0.8, 0.0, 0.0, 0.4))

    # ---- Trajectory file loading UI ----
    available_trajs = scan_trajectory_files(args.traj_dir)
    if available_trajs:
        logger.info(f"Found {len(available_trajs)} trajectory files in {args.traj_dir}")
        for i, f in enumerate(available_trajs):
            logger.info(f"  [{i}] {f}")
    else:
        logger.info(f"No trajectory files found in {args.traj_dir}")
    traj_file_slider = pybullet.addUserDebugParameter("Traj file idx", 0.0, 1.0, 0.0, physicsClientId=cid)
    btn_load_traj = Button("Load Trajectory", cid=cid)

    # ------------------------------------------------------------------
    # 4. Main loop
    # ------------------------------------------------------------------
    logger.info("Ready. Adjust sliders and press buttons in the PyBullet GUI.")

    while True:
        try:
            # Read current slider poses
            start_pose = start_sliders.as_pose()
            end_pose = end_sliders.as_pose()

            # Show ghost bars for start and end
            pp.set_pose(ghost_start, start_pose)
            pp.set_pose(ghost_end, end_pose)

            # Mode: 0=start, 1=end, 2=path playback
            mode = int(round(pybullet.readUserDebugParameter(mode_slider, physicsClientId=cid)))
            if mode == 0:
                pp.set_pose(bar, start_pose)
            elif mode == 1:
                pp.set_pose(bar, end_pose)
            # mode == 2: bar pose updated by path playback below

            # ---- Set Grasp ----
            # if btn_set_grasp.pressed():
            #     # Compute grasp from current right/left tool pose relative to bar
            #     world_from_right = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
            #     world_from_left = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
            #     world_from_bar = pp.get_pose(bar)

            #     grasp_bar_from_right = pp.multiply(pp.invert(world_from_bar), world_from_right)
            #     grasp_bar_from_left = pp.multiply(pp.invert(world_from_bar), world_from_left)

            #     # Compute relative transform for dual-arm projection
            #     desired_right_from_left = pp.multiply(pp.invert(world_from_right), world_from_left)
            #     projector = DualArmProjection(robot_setup, desired_right_from_left)

            #     # Set globals for the solver
            #     solver_mod.bar_from_right = grasp_bar_from_right
            #     solver_mod.bar_from_left = grasp_bar_from_left

            #     print(f"Grasp set!")
            #     print(f"  bar_from_right: pos={np.round(grasp_bar_from_right[0], 4)}, "
            #           f"quat={np.round(grasp_bar_from_right[1], 4)}")
            #     print(f"  bar_from_left:  pos={np.round(grasp_bar_from_left[0], 4)}, "
            #           f"quat={np.round(grasp_bar_from_left[1], 4)}")

            #     # Reset path since grasp changed
            #     path = None
            #     start_confs = None
            #     end_confs = None

            # ---- Plan Path (auto-solves IK) ----
            if btn_plan.pressed():
                if projector is None or grasp_bar_from_right is None:
                    logger.error("Set grasp first!")
                else:
                    collision_fn = robot_setup.create_collision_fn(obstacle_bodies=robot_setup.obstacles)

                    # Solve Start IK
                    timer.start("solve_start_ik")
                    with pp.LockRenderer():
                        start_confs = projector.create_valid_confs(
                            robot_setup.ik_solver_right,
                            start_pose,
                            grasp_bar_from_right,
                            delta=np.pi,
                            max_attempts=20,
                            collision_fn=collision_fn,
                        )
                    elapsed = timer.stop("solve_start_ik")
                    if start_confs is not None:
                        start_confs = normalize_angles(start_confs)
                        logger.info(f"Start IK: {len(start_confs)} solutions found in {elapsed:.3f}s")
                    else:
                        logger.warning(f"Start IK: no solution found ({elapsed:.3f}s)")
                        continue

                    # Solve End IK
                    timer.start("solve_end_ik")
                    with pp.LockRenderer():
                        end_confs = projector.create_valid_confs(
                            robot_setup.ik_solver_right,
                            end_pose,
                            grasp_bar_from_right,
                            delta=np.pi,
                            max_attempts=20,
                            collision_fn=collision_fn,
                        )
                    elapsed = timer.stop("solve_end_ik")
                    if end_confs is not None:
                        end_confs = normalize_angles(end_confs)
                        logger.info(f"End IK: {len(end_confs)} solutions found in {elapsed:.3f}s")
                    else:
                        logger.warning(f"End IK: no solution found ({elapsed:.3f}s)")
                        continue

                    start_conf = start_confs[0]
                    end_conf = end_confs[0]

                    # Create solver (uses cached collision fn)
                    solver = TrajectoryDualCartConstrainedSolver(robot_setup, None, projector)

                    logger.info("Planning path...")
                    timer.start("plan_path")
                    profile_path = os.path.join(os.path.dirname(__file__), "plan_profile.prof")
                    profiler = cProfile.Profile()
                    profiler.enable()
                    path = solver.plan(
                        start_conf=start_conf,
                        target_conf=end_conf,
                        max_time=60,
                        max_iterations=5000,
                        max_attempts=10,
                        use_draw=True,
                        verbose=True,
                    )
                    profiler.disable()
                    elapsed = timer.stop("plan_path")

                    # Save and display profile
                    profiler.dump_stats(profile_path)
                    logger.info(f"Profile saved to {profile_path}")

                    # Print top 30 cumulative-time entries
                    stats = pstats.Stats(profiler)
                    stats.sort_stats("cumulative")
                    stats.print_stats(30)

                    # # Launch snakeviz in background
                    subprocess.Popen([sys.executable, "-m", "snakeviz", profile_path])
                    logger.info("Launched snakeviz (check your browser)")

                    if path is not None:
                        logger.info(f"Path found! {len(path)} waypoints in {elapsed:.3f}s")
                        path_current_idx = -1
                        # Save trajectory as JointTrajectory JSON
                        os.makedirs(args.traj_dir, exist_ok=True)
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        traj_filename = f"testbench_{timestamp}_JointTrajectory.json"
                        traj_path = os.path.join(args.traj_dir, traj_filename)
                        save_path_as_joint_trajectory(path, HUSKY_DUAL_ARM_JOINT_NAMES, traj_path)
                        # Refresh file list for the load slider
                        available_trajs = scan_trajectory_files(args.traj_dir)
                        logger.info(f"Trajectory dir now has {len(available_trajs)} files")
                    else:
                        logger.warning(f"No path found ({elapsed:.3f}s)")

                    timer.summary()

            # ---- Load Trajectory from file ----
            if btn_load_traj.pressed():
                available_trajs = scan_trajectory_files(args.traj_dir)
                if not available_trajs:
                    logger.warning(f"No trajectory files in {args.traj_dir}")
                else:
                    t = pybullet.readUserDebugParameter(traj_file_slider, physicsClientId=cid)
                    idx = int(round(t * (len(available_trajs) - 1)))
                    idx = max(0, min(idx, len(available_trajs) - 1))
                    selected_file = available_trajs[idx]
                    traj_path = os.path.join(args.traj_dir, selected_file)
                    logger.info(f"Loading trajectory [{idx}/{len(available_trajs)}]: {selected_file}")
                    try:
                        path = load_joint_trajectory_as_path(traj_path, HUSKY_DUAL_ARM_JOINT_NAMES)
                        path_current_idx = -1
                        logger.info(f"Loaded {len(path)} waypoints from {selected_file}")
                        # Switch to path playback mode
                        mode = 2
                    except Exception as e:
                        logger.error(f"Failed to load trajectory: {e}")

            # ---- Browse start configs (mode 0) ----
            if mode == 0 and start_confs is not None:
                t = pybullet.readUserDebugParameter(start_conf_idx_slider, physicsClientId=cid)
                idx = int(round(t * (len(start_confs) - 1)))
                idx = max(0, min(idx, len(start_confs) - 1))
                robot_setup.set_joint_positions(robot_setup.arm_joints, start_confs[idx])

            # ---- Browse end configs (mode 1) ----
            if mode == 1 and end_confs is not None:
                t = pybullet.readUserDebugParameter(end_conf_idx_slider, physicsClientId=cid)
                idx = int(round(t * (len(end_confs) - 1)))
                idx = max(0, min(idx, len(end_confs) - 1))
                robot_setup.set_joint_positions(robot_setup.arm_joints, end_confs[idx])

            # ---- Path playback (mode 2) ----
            if mode == 2 and path is not None:
                t = pybullet.readUserDebugParameter(path_slider, physicsClientId=cid)
                idx = int(round(t * (len(path) - 1)))
                idx = max(0, min(idx, len(path) - 1))
                if idx != path_current_idx:
                    path_current_idx = idx
                    robot_setup.set_joint_positions(robot_setup.arm_joints, path[idx])
                    # Update bar pose from FK to match robot configuration
                    if grasp_bar_from_left is not None:
                        world_from_tool0 = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
                        bar_pose = pp.multiply(world_from_tool0, pp.invert(grasp_bar_from_left))
                        pp.set_pose(bar, bar_pose)

            time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Exiting...")
            break

    pp.disconnect()


if __name__ == "__main__":
    main()
