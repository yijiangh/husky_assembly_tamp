"""
Trajectory Dual-Arm Constrained Solver - Testbench

Standalone testbench for the dual-arm constrained motion planner.
Loads the robot directly from URDF (no compas_fab), provides a GUI
for configuring start/end bar poses and grasps, and includes timing
instrumentation for bottleneck identification.

Usage:
    cd external/husky_assembly_tamp
    python -m husky_assembly_tamp.motion_planner.trajectory_testbench [--planner birrt|constrained_bimanual]
"""

import argparse
import cProfile
import csv
import json
import os
import subprocess
import pstats
import sys
import time
from typing import Optional, Sequence, Tuple

import numpy as np
import pybullet
import pybullet_planning as pp

from compas.data import json_dump, json_load
from compas_fab.robots import JointTrajectory, JointTrajectoryPoint
from compas_robots.model import Joint

from husky_assembly_tamp.robot.dual_arm_projection import DualArmProjection
from husky_assembly_tamp.robot.robot_setup import (
    HUSKY_DUAL_ARM_JOINT_NAMES,
    HUSKY_DUAL_INIT_ARM_JOINT_ANGLES,
    HUSKY_DUAL_URDF_PATH,
    RobotSetup,
)
import husky_assembly_tamp.motion_planner.trajectory_dual_cart_constrained_solver as solver_mod
from husky_assembly_tamp.motion_planner.planner_backends import get_backend, list_backends
from husky_assembly_tamp.motion_planner.trajectory_dual_cart_constrained_solver import (
    TrajectoryDualCartConstrainedSolver,
)
from husky_assembly_tamp.utils.util import normalize_angles, setup_logger, reinit_logger_stream


# Global logger instance
logger = setup_logger("trajectory_testbench")


PoseLike = Tuple[Sequence[float], Sequence[float]]


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
            joint_types=[Joint.REVOLUTE] * len(conf),
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


def _is_pose_waypoint(wp):
    """True if waypoint looks like a pose tuple: (pos[3], quat[4])."""
    if not isinstance(wp, (list, tuple)) or len(wp) != 2:
        return False
    try:
        pos = np.asarray(wp[0], dtype=float).reshape(-1)
        quat = np.asarray(wp[1], dtype=float).reshape(-1)
    except Exception:
        return False
    return pos.shape[0] == 3 and quat.shape[0] == 4


def _is_joint_waypoint(wp):
    """True if waypoint looks like a full dual-arm joint vector."""
    try:
        arr = np.asarray(wp, dtype=float).reshape(-1)
    except Exception:
        return False
    return arr.shape[0] == len(HUSKY_DUAL_ARM_JOINT_NAMES)


def _apply_path_waypoint(
    waypoint,
    robot_setup: RobotSetup,
    grasp_bar_from_left: Optional[PoseLike],
    bar: int,
):
    """Apply path waypoint for visualization.

    - Joint waypoint: set robot joints, then update bar from FK
    - Pose waypoint: set bar pose directly (Stage 1 task-space path)
    """
    if _is_joint_waypoint(waypoint):
        robot_setup.set_joint_positions(robot_setup.arm_joints, waypoint)
        if grasp_bar_from_left is not None:
            world_from_tool0 = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
            bar_pose = pp.multiply(world_from_tool0, pp.invert(grasp_bar_from_left))
            pp.set_pose(bar, bar_pose)
        return

    if _is_pose_waypoint(waypoint):
        pp.set_pose(bar, waypoint)
        return

    logger.warning(f"Unsupported waypoint format in path playback: {type(waypoint)}")


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
    parser.add_argument(
        "--planner",
        choices=list_backends(),
        default="birrt",
        help=f"Planner backend to use (default: birrt). Available: {', '.join(list_backends())}",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Planning stage: 1=task-space + floating collision (Stage 1b default), 2=IK on (no collision), 3=full",
    )
    parser.add_argument(
        "--stage1-no-collision",
        action="store_true",
        help="When --stage 1, disable floating-bar collision and run legacy Stage 1 (task-space only)",
    )
    parser.add_argument(
        "--dist-metric",
        choices=["feature", "pose6d"],
        default="feature",
        help="Task-space distance metric (default: feature)",
    )
    parser.add_argument(
        "--ladder-search",
        choices=["shortest", "enumerate"],
        default="shortest",
        help="Ladder graph search mode (default: shortest)",
    )
    parser.add_argument(
        "--expand-delta",
        type=float,
        default=np.pi / 4.0,
        help="Delta (rad) for IK sweep during ladder expansion",
    )
    parser.add_argument(
        "--start-goal-delta",
        type=float,
        default=np.pi / 4.0,
        help="Delta (rad) for IK sweep at start/goal",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headless (no PyBullet GUI)",
    )
    parser.add_argument(
        "--goal-bias",
        type=float,
        default=0.1,
        help="Goal bias probability for task-space sampling (0..1)",
    )
    parser.add_argument(
        "--guide-bias",
        type=float,
        default=0.2,
        help="Guide-pose sampling probability for Stage 3 (0..1)",
    )
    parser.add_argument(
        "--warm-start-first",
        action="store_true",
        help="Try warm-start smoothing before running Stage 3 RRT",
    )
    parser.add_argument(
        "--return-task-path",
        action="store_true",
        help="In Stage 1, return the raw task-space path for diagnosis",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=30.0,
        help="Max planning time per attempt (seconds)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2000,
        help="Max RRT iterations per attempt",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Number of random restarts",
    )
    parser.add_argument(
        "--smooth-iterations",
        type=int,
        default=10,
        help="Joint-space shortcut smoothing iterations (set 0 to disable)",
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable joint-space shortcut smoothing",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for planner sampling (default: random)",
    )
    parser.add_argument(
        "--failure-analysis",
        action="store_true",
        help="Run multi-seed failure-distribution and per-stage comparison analysis",
    )
    parser.add_argument(
        "--analysis-trials",
        type=int,
        default=30,
        help="Number of seeds/attempts to evaluate in failure analysis mode",
    )
    parser.add_argument(
        "--analysis-seed-start",
        type=int,
        default=0,
        help="Start seed for failure analysis (seeds = start..start+trials-1)",
    )
    parser.add_argument(
        "--analysis-outdir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "reports"),
        help="Output directory for failure-analysis CSV/JSON/plots",
    )
    parser.add_argument(
        "--analysis-no-plot",
        action="store_true",
        help="Skip generating PNG plots for failure-analysis",
    )
    args = parser.parse_args()
    backend: TrajectoryDualCartConstrainedSolver = get_backend(args.planner)
    logger.info(f"Planner backend: {backend.name} — {backend.description}")
    timer = Timer()

    is_mac = sys.platform == "darwin"
    if is_mac:
        logger.info("macOS detected — skipping GUI buttons/sliders, will auto-run planning")

    # ------------------------------------------------------------------
    # 1. Direct URDF loading (no compas_fab)
    # ------------------------------------------------------------------
    timer.start("robot_loading")

    use_gui = (not args.no_gui)
    cid = pp.connect(use_gui=use_gui)
    # On Windows, pybullet.GUI can invalidate stdout/stderr handles (WinError 6).
    # Reopen them to avoid OSError on subsequent print() calls.
    if sys.platform == "win32":
        sys.stdout = open("CONOUT$", "w")
        sys.stderr = open("CONOUT$", "w")
        # Re-initialize logger's console handler after stdout is restored
        reinit_logger_stream(logger)

    # pp.connect disables the debug GUI panel — re-enable it for sliders/buttons
    if use_gui:
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 1, physicsClientId=cid)

    if use_gui:
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=2.5, cameraYaw=45, cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5], physicsClientId=cid,
        )

    with pp.LockRenderer():
        robot = pp.load_pybullet(HUSKY_DUAL_URDF_PATH, fixed_base=True)

    arm_joints = pp.joints_from_names(robot, HUSKY_DUAL_ARM_JOINT_NAMES)
    pp.set_joint_positions(robot, arm_joints, HUSKY_DUAL_INIT_ARM_JOINT_ANGLES)

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

    grasp_bar_from_right: Optional[PoseLike] = None
    grasp_bar_from_left: Optional[PoseLike] = None
    projector: Optional[DualArmProjection] = None
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
    # 3. Plan path & create GUI
    # ------------------------------------------------------------------
    # On macOS, pybullet GUI buttons/sliders are broken on Apple Silicon.
    # Skip interactive controls and auto-run planning, then visualize.
    if is_mac or args.no_gui:
        path = _run_planning_headless(
            projector, grasp_bar_from_right, robot_setup, backend, timer,
            start_defaults, end_defaults, args, bar_height,
        )
        if use_gui:
            _run_visualization_loop(path, robot_setup, grasp_bar_from_left, bar, cid)
    else:
        _run_interactive_gui(
            projector, grasp_bar_from_right, grasp_bar_from_left, robot_setup, backend, timer,
            start_defaults, end_defaults, start_confs, end_confs, path, path_current_idx,
            args, bar, bar_height, cid,
        )

    pp.disconnect()


def _execute_plan(
    start_pose: PoseLike,
    end_pose: PoseLike,
    projector: DualArmProjection,
    grasp_bar_from_right: PoseLike,
    robot_setup: RobotSetup,
    backend: TrajectoryDualCartConstrainedSolver,
    timer: Timer,
    traj_dir: str,
    args: argparse.Namespace,
):
    """Shared IK + planning core used by both headless and interactive GUI paths.

    Returns
    -------
    tuple: (path, start_confs, end_confs)
        path is None if planning failed; start_confs/end_confs are None if IK failed.
    """
    enable_ik = args.stage >= 2
    enable_collision = args.stage >= 3
    if args.stage == 1:
        enable_collision = (not args.stage1_no_collision)
    if enable_ik:
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
            return None, None, None

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
            return None, start_confs, None

        start_conf = start_confs[0]
        end_conf = end_confs[0]
    else:
        # Stage 1: no IK needed for endpoints; keep current robot configuration as a seed.
        q_now = np.asarray(pp.get_joint_positions(robot_setup.robot, robot_setup.arm_joints), dtype=float)
        start_confs = [q_now]
        end_confs = [q_now]
        start_conf = q_now
        end_conf = q_now
        logger.info("Stage 1: skipping start/end IK solve; planning directly in task space.")

    logger.info(
        f"Planning path with '{backend.name}' "
        f"(stage={args.stage}{'b' if (args.stage == 1 and enable_collision) else ''}, "
        f"metric={args.dist_metric}, ladder={args.ladder_search})..."
    )
    timer.start("plan_path")
    profile_path = os.path.join(os.path.dirname(__file__), "plan_profile.prof")
    profiler = cProfile.Profile()
    profiler.enable()
    # Stage 3 warm-start: run Stage 2 first to get a guide path (if requested)
    warm_start_path = None
    guide_poses = None
    if args.stage == 3:
        logger.info("Stage 3 warm-start: running Stage 2 to seed guide poses...")
        warm_start_path = backend.plan(
            start_conf=start_conf,
            goal_conf=end_conf,
            robot_setup=robot_setup,
            projector=projector,
            max_time=args.max_time,
            max_iterations=args.max_iterations,
            max_attempts=max(1, args.max_attempts // 2),
            use_draw=False,
            verbose=False,
            # key flags
            enable_ik=True,
            enable_collision=False,
            ####
            dist_metric=args.dist_metric,
            ladder_search=args.ladder_search,
            ladder_expand_delta=args.expand_delta,
            start_goal_delta=args.start_goal_delta,
            goal_sample_prob=args.goal_bias,
            random_seed=args.random_seed,
        )
        if warm_start_path is not None:
            guide_poses = []
            for q in warm_start_path:
                robot_setup.set_joint_positions(robot_setup.arm_joints, q)
                world_from_tool0 = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
                bar_pose = pp.multiply(world_from_tool0, pp.invert(grasp_bar_from_right))
                guide_poses.append(bar_pose)
            logger.info(f"Stage 3 warm-start: {len(guide_poses)} guide poses prepared.")
        else:
            logger.warning("Stage 3 warm-start: failed to get Stage 2 path.")

    path = backend.plan(
        start_conf=start_conf,
        goal_conf=end_conf,
        robot_setup=robot_setup,
        projector=projector,
        max_time=args.max_time,
        max_iterations=args.max_iterations,
        max_attempts=args.max_attempts,
        use_draw=not args.no_gui,
        verbose=True,
        enable_ik=enable_ik,
        enable_collision=enable_collision,
        dist_metric=args.dist_metric,
        ladder_search=args.ladder_search,
        ladder_expand_delta=args.expand_delta,
        start_goal_delta=args.start_goal_delta,
        goal_sample_prob=args.goal_bias,
        return_task_path=args.return_task_path,
        guide_poses=guide_poses,
        warm_start_path=warm_start_path,
        warm_start_first=args.warm_start_first,
        smooth_iterations=(0 if args.no_smoothing else args.smooth_iterations),
        start_bar_pose=(start_pose if not enable_ik else None),
        target_bar_pose=(end_pose if not enable_ik else None),
        random_seed=args.random_seed,
    )
    profiler.disable()
    elapsed = timer.stop("plan_path")

    # Save and display profile
    profiler.dump_stats(profile_path)
    logger.info(f"Profile saved to {profile_path}")
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(30)
    subprocess.Popen([sys.executable, "-m", "snakeviz", profile_path])
    logger.info("Launched snakeviz (check your browser)")

    if path is not None:
        logger.info(f"Path found! {len(path)} waypoints in {elapsed:.3f}s")
        if len(path) > 0 and isinstance(path[0], (list, tuple, np.ndarray)) and len(path[0]) == len(HUSKY_DUAL_ARM_JOINT_NAMES):
            os.makedirs(traj_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            traj_filename = f"testbench_{timestamp}_JointTrajectory.json"
            traj_path = os.path.join(traj_dir, traj_filename)
            save_path_as_joint_trajectory(path, HUSKY_DUAL_ARM_JOINT_NAMES, traj_path)
        else:
            logger.info("Stage 1 returned task-space poses; skipping JointTrajectory save.")
    else:
        logger.warning(f"No path found ({elapsed:.3f}s)")

    timer.summary()
    return path, start_confs, end_confs


def _run_failure_distribution_analysis(
    start_pose: PoseLike,
    end_pose: PoseLike,
    projector: DualArmProjection,
    grasp_bar_from_right: PoseLike,
    robot_setup: RobotSetup,
    backend: TrajectoryDualCartConstrainedSolver,
    args: argparse.Namespace,
):
    """Run multi-seed stage comparison and classify dominant failure source."""
    if args.analysis_trials <= 0:
        logger.warning("failure-analysis requested with non-positive --analysis-trials; skipping.")
        return
    os.makedirs(args.analysis_outdir, exist_ok=True)

    # Endpoint IK is shared across Stage 2/3 to isolate planner-stage effects from endpoint setup jitter.
    collision_fn = robot_setup.create_collision_fn(obstacle_bodies=robot_setup.obstacles)
    with pp.LockRenderer():
        start_confs = projector.create_valid_confs(
            robot_setup.ik_solver_right,
            start_pose,
            grasp_bar_from_right,
            delta=np.pi,
            max_attempts=40,
            collision_fn=collision_fn,
        )
        end_confs = projector.create_valid_confs(
            robot_setup.ik_solver_right,
            end_pose,
            grasp_bar_from_right,
            delta=np.pi,
            max_attempts=40,
            collision_fn=collision_fn,
        )

    if start_confs is None or end_confs is None:
        logger.error("Failure analysis aborted: endpoint IK failed (cannot evaluate Stage 2/3).")
        return

    start_conf = normalize_angles(start_confs)[0]
    end_conf = normalize_angles(end_confs)[0]

    def _plot_tree_3d(tree_data, stage_label, out_path):
        t1 = tree_data.get("tree1", {"points": [], "edges": []})
        t2 = tree_data.get("tree2", {"points": [], "edges": []})
        p1 = np.asarray(t1.get("points", []), dtype=float).reshape((-1, 3)) if t1.get("points") else np.zeros((0, 3))
        p2 = np.asarray(t2.get("points", []), dtype=float).reshape((-1, 3)) if t2.get("points") else np.zeros((0, 3))
        e1 = t1.get("edges", [])
        e2 = t2.get("edges", [])

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        if len(p1) > 0:
            ax.scatter(p1[:, 0], p1[:, 1], p1[:, 2], s=6, c="#d62728", alpha=0.7, label="tree A")
            for i, j in e1:
                ax.plot([p1[i, 0], p1[j, 0]], [p1[i, 1], p1[j, 1]], [p1[i, 2], p1[j, 2]], c="#d62728", alpha=0.35, linewidth=0.8)
        if len(p2) > 0:
            ax.scatter(p2[:, 0], p2[:, 1], p2[:, 2], s=6, c="#1f77b4", alpha=0.7, label="tree B")
            for i, j in e2:
                ax.plot([p2[i, 0], p2[j, 0]], [p2[i, 1], p2[j, 1]], [p2[i, 2], p2[j, 2]], c="#1f77b4", alpha=0.35, linewidth=0.8)

        sp = np.asarray(tree_data.get("start_pose", start_pose[0]), dtype=float)
        gp = np.asarray(tree_data.get("goal_pose", end_pose[0]), dtype=float)
        ax.scatter([sp[0]], [sp[1]], [sp[2]], c="green", s=80, marker="o", label="start")
        ax.scatter([gp[0]], [gp[1]], [gp[2]], c="black", s=80, marker="x", label="goal")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"{stage_label} tree structure (seed {args.analysis_seed_start})")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()

    def _plan_stage(stage_id, seed, collect_tree=False):
        enable_ik = stage_id >= 2
        enable_collision = stage_id >= 3
        if stage_id == 1:
            enable_collision = (not args.stage1_no_collision)
        t0 = time.perf_counter()
        diagnostics = {}
        tree_data = {} if collect_tree else None
        path = backend.plan(
            start_conf=start_conf,
            goal_conf=end_conf,
            robot_setup=robot_setup,
            projector=projector,
            max_time=args.max_time,
            max_iterations=args.max_iterations,
            max_attempts=1,
            use_draw=False,
            verbose=False,
            enable_ik=enable_ik,
            enable_collision=enable_collision,
            dist_metric=args.dist_metric,
            ladder_search=args.ladder_search,
            ladder_expand_delta=args.expand_delta,
            start_goal_delta=args.start_goal_delta,
            goal_sample_prob=args.goal_bias,
            return_task_path=(stage_id == 1),
            guide_poses=None,
            warm_start_path=None,
            warm_start_first=False,
            smooth_iterations=0,
            start_bar_pose=(start_pose if stage_id == 1 else None),
            target_bar_pose=(end_pose if stage_id == 1 else None),
            random_seed=seed,
            diagnostics_out=diagnostics,
            debug_tree_out=tree_data,
        )
        dt = time.perf_counter() - t0
        return path, dt, diagnostics, tree_data

    records = []
    counts = {
        "success": 0,
        "task_space_failure": 0,
        "ik_failure": 0,
        "collision_failure": 0,
    }
    stage_success = {1: 0, 2: 0, 3: 0}
    stage_runtime_sums = {1: 0.0, 2: 0.0, 3: 0.0}

    for i in range(args.analysis_trials):
        seed = args.analysis_seed_start + i
        s1_path, t1, s1_diag, _ = _plan_stage(1, seed)
        s2_path, t2, s2_diag, _ = _plan_stage(2, seed)
        s3_path, t3, s3_diag, _ = _plan_stage(3, seed)

        s1_ok = s1_path is not None
        s2_ok = s2_path is not None
        s3_ok = s3_path is not None

        stage_success[1] += int(s1_ok)
        stage_success[2] += int(s2_ok)
        stage_success[3] += int(s3_ok)
        stage_runtime_sums[1] += t1
        stage_runtime_sums[2] += t2
        stage_runtime_sums[3] += t3

        if not s1_ok:
            category = "task_space_failure"
        elif not s2_ok:
            category = "ik_failure"
        elif not s3_ok:
            category = "collision_failure"
        else:
            category = "success"
        counts[category] += 1

        records.append(
            {
                "seed": seed,
                "stage1_success": int(s1_ok),
                "stage2_success": int(s2_ok),
                "stage3_success": int(s3_ok),
                "stage1_runtime_s": round(t1, 4),
                "stage2_runtime_s": round(t2, 4),
                "stage3_runtime_s": round(t3, 4),
                "stage1_rrt_failed": int(s1_diag.get("rrt_failed", 0)),
                "stage2_rrt_failed": int(s2_diag.get("rrt_failed", 0)),
                "stage3_rrt_failed": int(s3_diag.get("rrt_failed", 0)),
                "stage2_ladder_failed": int(s2_diag.get("ladder_failed", 0)),
                "stage3_ladder_failed": int(s3_diag.get("ladder_failed", 0)),
                "category": category,
            }
        )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.analysis_outdir, f"failure_analysis_{timestamp}.csv")
    json_path = os.path.join(args.analysis_outdir, f"failure_analysis_{timestamp}.json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "trials": args.analysis_trials,
        "seed_start": args.analysis_seed_start,
        "max_time_per_attempt_s": args.max_time,
        "counts": counts,
        "stage_success_rate": {
            "stage1": stage_success[1] / max(1, args.analysis_trials),
            "stage2": stage_success[2] / max(1, args.analysis_trials),
            "stage3": stage_success[3] / max(1, args.analysis_trials),
        },
        "stage_avg_runtime_s": {
            "stage1": stage_runtime_sums[1] / max(1, args.analysis_trials),
            "stage2": stage_runtime_sums[2] / max(1, args.analysis_trials),
            "stage3": stage_runtime_sums[3] / max(1, args.analysis_trials),
        },
    }
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    logger.info("Failure analysis summary:")
    logger.info(f"  task_space_failure: {counts['task_space_failure']}")
    logger.info(f"  ik_failure:         {counts['ik_failure']}")
    logger.info(f"  collision_failure:  {counts['collision_failure']}")
    logger.info(f"  success:            {counts['success']}")
    logger.info(
        "  stage success rates: "
        f"S1={summary['stage_success_rate']['stage1']:.2%}, "
        f"S2={summary['stage_success_rate']['stage2']:.2%}, "
        f"S3={summary['stage_success_rate']['stage3']:.2%}"
    )
    logger.info(f"Saved analysis CSV: {csv_path}")
    logger.info(f"Saved analysis JSON: {json_path}")

    if args.analysis_no_plot:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"Skipping plots (matplotlib unavailable): {e}")
        return

    fig1_path = os.path.join(args.analysis_outdir, f"failure_distribution_{timestamp}.png")
    fig2_path = os.path.join(args.analysis_outdir, f"stage_success_{timestamp}.png")

    labels = ["task_space_failure", "ik_failure", "collision_failure", "success"]
    vals = [counts[k] for k in labels]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, vals, color=["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"])
    plt.ylabel("Count")
    plt.title("Failure Distribution Across Seeds")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=180)
    plt.close()

    stage_labels = ["Stage 1", "Stage 2", "Stage 3"]
    stage_vals = [
        summary["stage_success_rate"]["stage1"],
        summary["stage_success_rate"]["stage2"],
        summary["stage_success_rate"]["stage3"],
    ]
    plt.figure(figsize=(6, 4))
    plt.bar(stage_labels, stage_vals, color=["#777777", "#337ab7", "#5cb85c"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Success Rate")
    plt.title("Per-Stage Success Rate")
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=180)
    plt.close()

    logger.info(f"Saved plot: {fig1_path}")
    logger.info(f"Saved plot: {fig2_path}")

    # Tree-structure comparison using the first seed in this analysis batch
    for stage_id, label in [(1, "Stage 1"), (2, "Stage 2"), (3, "Stage 3")]:
        _, _, _, tree_data = _plan_stage(stage_id, args.analysis_seed_start, collect_tree=True)
        if tree_data is None:
            continue
        tree_path = os.path.join(
            args.analysis_outdir,
            f"tree_structure_stage{stage_id}_seed{args.analysis_seed_start}_{timestamp}.png",
        )
        _plot_tree_3d(tree_data, label, tree_path)
        logger.info(f"Saved tree plot: {tree_path}")


def _run_planning_headless(
    projector: Optional[DualArmProjection],
    grasp_bar_from_right: Optional[PoseLike],
    robot_setup: RobotSetup,
    backend: TrajectoryDualCartConstrainedSolver,
    timer: Timer,
    start_defaults: Sequence[float],
    end_defaults: Sequence[float],
    args: argparse.Namespace,
    bar_height: float,
):
    """Derive poses from loaded JSON defaults and auto-run planning (macOS path)."""
    if projector is None or grasp_bar_from_right is None:
        logger.error("Cannot auto-plan: grasp data not loaded. Provide --grasp-json.")
        return None
    assert projector is not None
    assert grasp_bar_from_right is not None

    start_pose = pp.Pose(point=start_defaults[:3], euler=pp.Euler(*start_defaults[3:]))
    end_pose = pp.Pose(point=end_defaults[:3], euler=pp.Euler(*end_defaults[3:]))

    # Ghost bars for start/end visualization
    ghost_start = pp.create_cylinder(radius=0.015, height=bar_height, color=(0.0, 0.8, 0.0, 0.4))
    ghost_end = pp.create_cylinder(radius=0.015, height=bar_height, color=(0.8, 0.0, 0.0, 0.4))
    pp.set_pose(ghost_start, start_pose)
    pp.set_pose(ghost_end, end_pose)

    if args.failure_analysis:
        _run_failure_distribution_analysis(
            start_pose, end_pose, projector, grasp_bar_from_right, robot_setup, backend, args
        )
        return None

    path, _, _ = _execute_plan(
        start_pose, end_pose, projector, grasp_bar_from_right,
        robot_setup, backend, timer, args.traj_dir, args,
    )
    return path


def _run_visualization_loop(
    path,
    robot_setup: RobotSetup,
    grasp_bar_from_left: Optional[PoseLike],
    bar: int,
    cid: int,
):
    """Minimal GUI loop for path playback only (macOS path).

    Uses a single 'Path t' slider to scrub through the trajectory.
    """
    if path is None:
        logger.info("No path to visualize. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Exiting...")
        return

    logger.info(f"Visualizing path ({len(path)} waypoints). Use 'Path t' slider to scrub.")
    path_slider = pybullet.addUserDebugParameter("Path t", 0.0, 1.0, 0.0, physicsClientId=cid)
    path_current_idx = -1

    while True:
        try:
            t = pybullet.readUserDebugParameter(path_slider, physicsClientId=cid)
            idx = int(round(t * (len(path) - 1)))
            idx = max(0, min(idx, len(path) - 1))
            if idx != path_current_idx:
                path_current_idx = idx
                _apply_path_waypoint(path[idx], robot_setup, grasp_bar_from_left, bar)

            time.sleep(0.01)
        except KeyboardInterrupt:
            logger.info("Exiting...")
            break


def _run_interactive_gui(
    projector: Optional[DualArmProjection],
    grasp_bar_from_right: Optional[PoseLike],
    grasp_bar_from_left: Optional[PoseLike],
    robot_setup: RobotSetup,
    backend: TrajectoryDualCartConstrainedSolver,
    timer: Timer,
    start_defaults, end_defaults, start_confs, end_confs, path, path_current_idx,
    args: argparse.Namespace,
    bar: int,
    bar_height: float,
    cid: int,
):
    """Full interactive GUI with buttons and sliders (Windows/Linux path)."""
    logger.info("=== GUI Controls ===")
    logger.info("  Buttons: Plan Path (auto-solves IK for start & end)")
    logger.info("  Sliders: Start bar pose, End bar pose")
    logger.info("====================")

    btn_plan = Button("Plan Path", cid=cid)
    path_slider = pybullet.addUserDebugParameter("Path t", 0.0, 1.0, 0.0, physicsClientId=cid)
    start_conf_idx_slider = pybullet.addUserDebugParameter("Start conf t", 0.0, 1.0, 0.0, physicsClientId=cid)
    end_conf_idx_slider = pybullet.addUserDebugParameter("End conf t", 0.0, 1.0, 0.0, physicsClientId=cid)

    mode_slider = pybullet.addUserDebugParameter("Mode (0=start, 1=end, 2=path)", 0, 2, 0, physicsClientId=cid)

    start_sliders = PoseSliders("Start", cid=cid, defaults=start_defaults)
    end_sliders = PoseSliders("End", cid=cid, defaults=end_defaults)

    ghost_start = pp.create_cylinder(radius=0.015, height=bar_height, color=(0.0, 0.8, 0.0, 0.4))
    ghost_end = pp.create_cylinder(radius=0.015, height=bar_height, color=(0.8, 0.0, 0.0, 0.4))

    available_trajs = scan_trajectory_files(args.traj_dir)
    if available_trajs:
        logger.info(f"Found {len(available_trajs)} trajectory files in {args.traj_dir}")
        for i, f in enumerate(available_trajs):
            logger.info(f"  [{i}] {f}")
    else:
        logger.info(f"No trajectory files found in {args.traj_dir}")
    traj_file_slider = pybullet.addUserDebugParameter("Traj file idx", 0.0, 1.0, 0.0, physicsClientId=cid)
    btn_load_traj = Button("Load Trajectory", cid=cid)

    logger.info("Ready. Adjust sliders and press buttons in the PyBullet GUI.")

    while True:
        try:
            start_pose = start_sliders.as_pose()
            end_pose = end_sliders.as_pose()

            pp.set_pose(ghost_start, start_pose)
            pp.set_pose(ghost_end, end_pose)

            mode = int(round(pybullet.readUserDebugParameter(mode_slider, physicsClientId=cid)))
            if mode == 0:
                pp.set_pose(bar, start_pose)
            elif mode == 1:
                pp.set_pose(bar, end_pose)

            # ---- Plan Path (auto-solves IK) ----
            if btn_plan.pressed():
                if projector is None or grasp_bar_from_right is None:
                    logger.error("Set grasp first!")
                else:
                    path, start_confs, end_confs = _execute_plan(
                        start_pose, end_pose, projector, grasp_bar_from_right,
                        robot_setup, backend, timer, args.traj_dir, args,
                    )
                    if path is not None:
                        path_current_idx = -1
                        available_trajs = scan_trajectory_files(args.traj_dir)
                        logger.info(f"Trajectory dir now has {len(available_trajs)} files")

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
                    _apply_path_waypoint(path[idx], robot_setup, grasp_bar_from_left, bar)

            time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Exiting...")
            break


if __name__ == "__main__":
    main()
