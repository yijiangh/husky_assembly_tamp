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
        help="Planning stage: 1=task-space only (no collision), 2=IK on (no collision), 3=full",
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
    args = parser.parse_args()
    backend = get_backend(args.planner)
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


def _execute_plan(start_pose, end_pose, projector, grasp_bar_from_right, robot_setup, backend, timer, traj_dir, args):
    """Shared IK + planning core used by both headless and interactive GUI paths.

    Returns
    -------
    tuple: (path, start_confs, end_confs)
        path is None if planning failed; start_confs/end_confs are None if IK failed.
    """
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

    enable_ik = args.stage >= 2
    enable_collision = args.stage >= 3
    logger.info(
        f"Planning path with '{backend.name}' "
        f"(stage={args.stage}, metric={args.dist_metric}, ladder={args.ladder_search})..."
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
            enable_ik=True,
            enable_collision=False,
            dist_metric=args.dist_metric,
            ladder_search=args.ladder_search,
            ladder_expand_delta=args.expand_delta,
            start_goal_delta=args.start_goal_delta,
            goal_sample_prob=args.goal_bias,
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


def _run_planning_headless(
    projector, grasp_bar_from_right, robot_setup, backend, timer,
    start_defaults, end_defaults, args, bar_height,
):
    """Derive poses from loaded JSON defaults and auto-run planning (macOS path)."""
    if projector is None or grasp_bar_from_right is None:
        logger.error("Cannot auto-plan: grasp data not loaded. Provide --grasp-json.")
        return None

    start_pose = pp.Pose(point=start_defaults[:3], euler=pp.Euler(*start_defaults[3:]))
    end_pose = pp.Pose(point=end_defaults[:3], euler=pp.Euler(*end_defaults[3:]))

    # Ghost bars for start/end visualization
    ghost_start = pp.create_cylinder(radius=0.015, height=bar_height, color=(0.0, 0.8, 0.0, 0.4))
    ghost_end = pp.create_cylinder(radius=0.015, height=bar_height, color=(0.8, 0.0, 0.0, 0.4))
    pp.set_pose(ghost_start, start_pose)
    pp.set_pose(ghost_end, end_pose)

    path, _, _ = _execute_plan(
        start_pose, end_pose, projector, grasp_bar_from_right,
        robot_setup, backend, timer, args.traj_dir, args,
    )
    return path


def _run_visualization_loop(path, robot_setup, grasp_bar_from_left, bar, cid):
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
                robot_setup.set_joint_positions(robot_setup.arm_joints, path[idx])
                if grasp_bar_from_left is not None:
                    world_from_tool0 = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
                    bar_pose = pp.multiply(world_from_tool0, pp.invert(grasp_bar_from_left))
                    pp.set_pose(bar, bar_pose)

            time.sleep(0.01)
        except KeyboardInterrupt:
            logger.info("Exiting...")
            break


def _run_interactive_gui(
    projector, grasp_bar_from_right, grasp_bar_from_left, robot_setup, backend, timer,
    start_defaults, end_defaults, start_confs, end_confs, path, path_current_idx,
    args, bar, bar_height, cid,
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
                    robot_setup.set_joint_positions(robot_setup.arm_joints, path[idx])
                    if grasp_bar_from_left is not None:
                        world_from_tool0 = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
                        bar_pose = pp.multiply(world_from_tool0, pp.invert(grasp_bar_from_left))
                        pp.set_pose(bar, bar_pose)

            time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("Exiting...")
            break


if __name__ == "__main__":
    main()
