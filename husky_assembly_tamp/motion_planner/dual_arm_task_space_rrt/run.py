"""
Dual-arm task-space RRT runner (Stages 1/2/3).

This is a clean restart from the original design intent:
- task-space sampling
- single-tree RRT
- Stage 1: no IK in the planner loop
- Stage 2: seed-chained dual-arm IK in extend, collision still off
- Stage 3: seed-chained dual-arm IK with robot collision on
- no ladder graph
- optional floating-body collision against a fixed robot

"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet
import pybullet_planning as pp
from pybullet_planning.interfaces.geometry.mesh import Mesh, create_mesh

from husky_assembly_tamp.utils.params import DATA_DIR
from husky_assembly_tamp.utils.util import setup_logger

from .core import (
    BAR_BOX_DIMS,
    DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
    DEFAULT_USE_ANGLE_NORMALIZATION,
    STAGE3_GRASP_MASK_LINKS,
    TOOL_LINK_LEFT,
    TOOL_LINK_RIGHT,
    FullConf,
    GraspTarget,
    PoseLike,
    bar_orientation_from_grasps,
    derive_constrained_start,
    get_bar_feature_points,
    get_joint_collision_fn,
    get_pose_collision_fn,
    plan_pose_rrt,
    solve_endpoint_dual_arm_ik,
    summarize_joint_continuity,
    validate_dual_arm_bar_pose,
)
from .path_validation import validate_stage_trajectory
from .smooth import smooth_dual_arm_pose_path
from .trajectory_io import save_path_as_joint_trajectory


logger = setup_logger("dual_arm_task_space_rrt_run", file_mode="w")

HUSKY_DUAL_URDF_PATH = os.path.join(
    DATA_DIR,
    "husky_urdf/mt_husky_dual_ur5_e_moveit_config/urdf/husky_dual_ur5_e_no_base_joint_All_Calibrated.urdf",
)
HUSKY_DUAL_SRDF_PATH = os.path.join(
    DATA_DIR,
    "husky_urdf/mt_husky_dual_ur5_e_moveit_config/config/dual_arm_husky.srdf",
)
HUSKY_DUAL_ARM_JOINT_NAMES = [
    "left_ur_arm_shoulder_pan_joint",
    "left_ur_arm_shoulder_lift_joint",
    "left_ur_arm_elbow_joint",
    "left_ur_arm_wrist_1_joint",
    "left_ur_arm_wrist_2_joint",
    "left_ur_arm_wrist_3_joint",
    "right_ur_arm_shoulder_pan_joint",
    "right_ur_arm_shoulder_lift_joint",
    "right_ur_arm_elbow_joint",
    "right_ur_arm_wrist_1_joint",
    "right_ur_arm_wrist_2_joint",
    "right_ur_arm_wrist_3_joint",
]
INIT_ARM_JOINT_ANGLES = np.array([0.0, -np.pi / 2.0, 0.0, 0.0, 0.0, 0.0] * 2, dtype=float)
STAGE1_DEBUG_START_OFFSET = np.array([-0.5, 0.0, 0.5], dtype=float)
DEFAULT_HOME_LEFT_TOOL_Z_OFFSET = 0.2
MOBILE_BASE_FROM_TOOL0_LEFT_HOME: PoseLike = (
    np.array([0.3974141597747803, 0.16023626923561096, 0.8621799349784851], dtype=float),
    np.array([-0.5000003576278687, 0.4999987483024597, -0.499999463558197, 0.5000012516975403], dtype=float),
    # np.array([0.4999987483024597, 0.5000003576278687, 0.5000012516975403, 0.499999463558197], dtype=float)
)
BarMeshSpec = Dict[str, Any]


def _normalize_vector(vector: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(arr)
    if norm <= 0.0:
        raise ValueError(f"Cannot normalize zero-length vector: {vector}")
    return arr / norm


def frame_data_to_pose(frame_data: Dict[str, Any]) -> PoseLike:
    data = frame_data["data"] if "data" in frame_data else frame_data
    origin = np.asarray(data["point"], dtype=float)
    xaxis = _normalize_vector(data["xaxis"])
    yaxis = _normalize_vector(data["yaxis"])
    zaxis = _normalize_vector(np.cross(xaxis, yaxis))
    rotation = np.column_stack([xaxis, yaxis, zaxis])
    tform = np.eye(4, dtype=float)
    tform[:3, :3] = rotation
    tform[:3, 3] = origin
    return pp.pose_from_tform(tform)


@contextlib.contextmanager
def suppress_native_output(enabled: bool = True):
    if not enabled:
        yield
        return

    redirected_fds: List[Tuple[int, int]] = []
    devnull_fd: Optional[int] = None
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        for stream in (sys.stdout, sys.stderr):
            if stream is None or not hasattr(stream, "fileno"):
                continue
            try:
                stream.flush()
                fd = stream.fileno()
            except (AttributeError, OSError, ValueError):
                continue
            saved_fd = os.dup(fd)
            os.dup2(devnull_fd, fd)
            redirected_fds.append((fd, saved_fd))
        yield
    finally:
        for fd, saved_fd in reversed(redirected_fds):
            os.dup2(saved_fd, fd)
            os.close(saved_fd)
        if devnull_fd is not None:
            os.close(devnull_fd)


def triangulate_faces(face_vertices: Sequence[int]) -> List[Tuple[int, int, int]]:
    if len(face_vertices) < 3:
        return []
    if len(face_vertices) == 3:
        return [tuple(int(v) for v in face_vertices)]
    anchor = int(face_vertices[0])
    triangles: List[Tuple[int, int, int]] = []
    for idx in range(1, len(face_vertices) - 1):
        triangles.append((anchor, int(face_vertices[idx]), int(face_vertices[idx + 1])))
    return triangles


def compas_mesh_data_to_pybullet_mesh(mesh_data: Dict[str, Any]) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    vertex_data = mesh_data["data"]["vertex"]
    vertex_keys = sorted(vertex_data.keys(), key=lambda key: int(key))
    old_to_new = {int(key): idx for idx, key in enumerate(vertex_keys)}
    vertices = [
        (
            float(vertex_data[key]["x"]),
            float(vertex_data[key]["y"]),
            float(vertex_data[key]["z"]),
        )
        for key in vertex_keys
    ]
    faces: List[Tuple[int, int, int]] = []
    for _, face in sorted(mesh_data["data"]["face"].items(), key=lambda item: int(item[0])):
        remapped = [old_to_new[int(vertex_idx)] for vertex_idx in face]
        faces.extend(triangulate_faces(remapped))
    return vertices, faces


def mesh_vertices_aabb_dims(vertices: Sequence[Sequence[float]]) -> Tuple[float, float, float]:
    verts = np.asarray(vertices, dtype=float)
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    dims = maxs - mins
    return float(dims[0]), float(dims[1]), float(dims[2])


def create_bar_mesh_body(mesh_spec: BarMeshSpec, color: Tuple[float, float, float, float], collision: bool = True) -> int:
    mesh = Mesh(mesh_spec["vertices"], mesh_spec["faces"])
    return create_mesh(mesh, mass=pp.STATIC_MASS, collision=collision, color=color)


def get_goal_pose_from_grasp_targets(grasp_targets: Sequence[GraspTarget]) -> PoseLike:
    if not grasp_targets:
        raise ValueError("Expected at least one grasp target.")
    goal_pose = grasp_targets[0][0]
    for idx, (other_pose, _) in enumerate(grasp_targets[1:], start=1):
        if not pp.is_pose_close(goal_pose, other_pose, pos_tolerance=1e-4, ori_tolerance=1e-4):
            logger.warning(
                "Goal bar pose from grasp target %d does not match grasp target 0 exactly; using grasp target 0.",
                idx,
            )
    return goal_pose


def derive_home_start_poses_from_grasps(
    grasp_targets: Sequence[GraspTarget],
    mobile_base_from_tool0_left: PoseLike = MOBILE_BASE_FROM_TOOL0_LEFT_HOME,
) -> Dict[str, PoseLike]:
    if len(grasp_targets) < 2:
        raise ValueError("Expected two grasp targets to derive the shared home start pose.")
    mobile_base_from_bar_left, mobile_base_from_tool0_left_goal = grasp_targets[0]
    mobile_base_from_bar_right, mobile_base_from_tool0_right_goal = grasp_targets[1]
    bar_from_tool0_left = pp.multiply(pp.invert(mobile_base_from_bar_left), mobile_base_from_tool0_left_goal)
    tool0_left_from_bar = pp.invert(bar_from_tool0_left)
    bar_from_tool0_right = pp.multiply(pp.invert(mobile_base_from_bar_right), mobile_base_from_tool0_right_goal)
    mobile_base_from_bar_start = pp.multiply(mobile_base_from_tool0_left, tool0_left_from_bar)
    mobile_base_from_tool0_right_start = pp.multiply(mobile_base_from_bar_start, bar_from_tool0_right)
    return {
        "mobile_base_from_tool0_left_start": mobile_base_from_tool0_left,
        "mobile_base_from_bar_start": mobile_base_from_bar_start,
        "mobile_base_from_tool0_right_start": mobile_base_from_tool0_right_start,
        "tool0_left_from_bar": tool0_left_from_bar,
        "bar_from_tool0_right": bar_from_tool0_right,
    }


# --- gdrive convention (2026-05+) ---------------------------------------------
# New design-study datasets live on gdrive and follow a different convention
# than the legacy design study:
#   - cell state filename: <bar_tag>_<phase>.json  (e.g. B3_approach.json),
#     not *_RobotCellState.json. No paired *_GraspTargets.json.
#   - rigid bodies tagged by role: active_bar_*, active_<other>_* (rigidly
#     bound to the bar at install), env_*. RobotCell.json sits at the
#     dataset root, RobotCellStates/ holds the per-bar/per-phase states.
#   - grasps come from FK at the cell state's joint values vs the active
#     bar's frame (the cell state is the single source of truth for grasps).
#   - the husky's robot_base_frame is non-identity. To keep the planner
#     entirely in mobile-base coords (same convention as the legacy code),
#     the gdrive loader transforms bar / env poses into mobile-base frame.
GDRIVE_DATA_DIRECTORY = (
    "/home/yijiangh/Insync/yijiang94817@gmail.com/Google Drive - Shared with me/2025-03 Husky Assembly/data_design_study"
)
GDRIVE_DEFAULT_PROBLEM = "2026-05-14_foc_demo_reduced"


def compas_frame_to_pose(frame: Any) -> PoseLike:
    """Convert a compas Frame-like object to a pybullet_planning pose."""
    return (
        np.asarray(frame.point, dtype=float),
        np.asarray(frame.quaternion.xyzw, dtype=float),
    )


def load_gdrive_active_bar_mesh(robot_cell_json: str, body_name: str) -> "BarMeshSpec":
    """Load a bar mesh from RobotCell.json keyed by the full body name
    ('active_bar_B3', 'env_bar_B1', ...)."""
    with open(robot_cell_json) as f:
        robot_cell = json.load(f)
    rigid_body_models = robot_cell["data"]["rigid_body_models"]
    if body_name not in rigid_body_models:
        raise KeyError(f"Body {body_name} not found in {robot_cell_json}")
    collision_meshes = rigid_body_models[body_name]["collision_meshes"]
    if not collision_meshes:
        raise ValueError(f"Body {body_name} has no collision meshes in {robot_cell_json}")
    vertices, faces = compas_mesh_data_to_pybullet_mesh(collision_meshes[0])
    return {
        "name": body_name,
        "body_name": body_name,
        "vertices": vertices,
        "faces": faces,
        "aabb_dims": mesh_vertices_aabb_dims(vertices),
    }


def _fk_dual_arm_grasps_in_mb_frame(
    goal_conf: np.ndarray,
    mb_from_bar_goal: PoseLike,
) -> Tuple[PoseLike, PoseLike]:
    """FK both tool0 links at goal_conf to derive grasp_bar_from_left/right.

    Operates in MOBILE-BASE frame (husky URDF is fixed_base at world origin
    in the temp pybullet client, which matches the planner's convention).
    Frame-invariant relative transforms: bar_from_tool0_* = inv(bar) * tool0.
    """
    pp.connect(use_gui=False)
    try:
        robot = pp.load_pybullet(HUSKY_DUAL_URDF_PATH, fixed_base=True)
        arm_joints = pp.joints_from_names(robot, HUSKY_DUAL_ARM_JOINT_NAMES)
        tool_link_left = pp.link_from_name(robot, TOOL_LINK_LEFT)
        tool_link_right = pp.link_from_name(robot, TOOL_LINK_RIGHT)
        pp.set_joint_positions(robot, arm_joints, np.asarray(goal_conf, dtype=float))
        mb_from_tool0_L = pp.get_link_pose(robot, tool_link_left)
        mb_from_tool0_R = pp.get_link_pose(robot, tool_link_right)
    finally:
        pp.disconnect()
    bar_inv = pp.invert(mb_from_bar_goal)
    grasp_bar_from_left = pp.multiply(bar_inv, mb_from_tool0_L)
    grasp_bar_from_right = pp.multiply(bar_inv, mb_from_tool0_R)
    return grasp_bar_from_left, grasp_bar_from_right


def _resolve_gdrive_state_path(state_arg: str, problem: Optional[str] = None) -> Tuple[str, str, str]:
    """Resolve a state arg to (state_path, problem_dir, robot_cell_json).

    `state_arg` may be an absolute path OR a bare filename like
    'B3_approach.json'. In the latter case `problem` is required and the
    file is looked up under GDRIVE_DATA_DIRECTORY/<problem>/RobotCellStates/.
    """
    if os.path.isabs(state_arg) and os.path.isfile(state_arg):
        state_path = state_arg
        problem_dir = os.path.dirname(os.path.dirname(state_path))
    else:
        if not problem:
            raise ValueError(
                "When --gdrive-state is a bare filename you must also pass "
                "--gdrive-problem (the dataset directory under "
                f"{GDRIVE_DATA_DIRECTORY!r}).")
        problem_dir = os.path.join(GDRIVE_DATA_DIRECTORY, problem)
        state_path = os.path.join(problem_dir, "RobotCellStates", state_arg)
        if not os.path.isfile(state_path):
            raise FileNotFoundError(f"Gdrive state file not found: {state_path}")
    robot_cell_json = os.path.join(problem_dir, "RobotCell.json")
    if not os.path.isfile(robot_cell_json):
        raise FileNotFoundError(f"RobotCell.json not found at {robot_cell_json}")
    return state_path, problem_dir, robot_cell_json


def _resolve_gdrive_bar_action_path(action_arg: str, problem: Optional[str] = None) -> Tuple[str, str, str]:
    """Resolve a BarAction arg to (action_path, problem_dir, robot_cell_json)."""
    if os.path.isabs(action_arg) and os.path.isfile(action_arg):
        action_path = action_arg
        problem_dir = os.path.dirname(os.path.dirname(action_path))
    else:
        if not problem:
            raise ValueError(
                "When --gdrive-bar-action is a bare filename you must also pass "
                "--gdrive-problem.")
        problem_dir = os.path.join(GDRIVE_DATA_DIRECTORY, problem)
        action_path = os.path.join(problem_dir, "BarActions", action_arg)
        if not os.path.isfile(action_path):
            raise FileNotFoundError(f"Gdrive BarAction not found: {action_path}")
    robot_cell_json = os.path.join(problem_dir, "RobotCell.json")
    if not os.path.isfile(robot_cell_json):
        raise FileNotFoundError(f"RobotCell.json not found at {robot_cell_json}")
    return action_path, problem_dir, robot_cell_json


def _joint_values_from_robot_cell_state(state: Any) -> np.ndarray:
    """Extract the dual-arm 12-vector from a compas_fab RobotCellState."""
    robot_conf = state.robot_configuration
    name_to_value = {n: v for n, v in zip(robot_conf.joint_names, robot_conf.joint_values)}
    try:
        return np.asarray([name_to_value[n] for n in HUSKY_DUAL_ARM_JOINT_NAMES], dtype=float)
    except KeyError as e:
        raise KeyError(f"BarAction start_state missing joint {e!s}; expected {HUSKY_DUAL_ARM_JOINT_NAMES}")


def _joint_values_from_configuration(configuration: Any) -> np.ndarray:
    """Extract the dual-arm 12-vector from a compas Configuration."""
    name_to_value = {n: v for n, v in zip(configuration.joint_names, configuration.joint_values)}
    try:
        return np.asarray([name_to_value[n] for n in HUSKY_DUAL_ARM_JOINT_NAMES], dtype=float)
    except KeyError as e:
        raise KeyError(f"BarAction target_configuration missing joint {e!s}; expected {HUSKY_DUAL_ARM_JOINT_NAMES}")


def build_gdrive_bar_action_scene_spec(
    action_json: str,
    movement: str | int = "M1",
    problem: Optional[str] = None,
    *,
    include_built_bars: bool = False,
) -> Dict[str, Any]:
    """Build a lightweight scene_spec from a gdrive BarAction movement.

    This intentionally avoids CfabSession / RobotCell materialization. It only
    reads the BarAction, active bar mesh, start RobotCellState, and target EE
    frames needed by the existing pp-only Stage 1/2/3 planner.
    """
    from husky_assembly_teleop.bar_action_io import find_movement, parse_bar_action

    action_path, _, robot_cell_json = _resolve_gdrive_bar_action_path(action_json, problem)
    action = parse_bar_action(action_path)
    movement_index, mv = find_movement(action, movement)
    if mv.start_state is None:
        raise ValueError(f"Movement {mv.movement_id!r} in {action_path} has no start_state.")
    if not mv.target_ee_frames or "left" not in mv.target_ee_frames or "right" not in mv.target_ee_frames:
        raise ValueError(f"Movement {mv.movement_id!r} has no left/right target_ee_frames.")

    start_state = mv.start_state
    active_bar_name = f"bar_{action.active_bar_id}"
    goal_state = None
    goal_conf_source = None
    target_configuration = getattr(mv, "target_configuration", None)
    if target_configuration is not None:
        goal_joint_values = _joint_values_from_configuration(target_configuration)
        goal_state = start_state
        goal_conf_source = "movement.target_configuration"
    elif movement_index + 1 < len(action.movements) and action.movements[movement_index + 1].start_state is not None:
        # For M1-style home->approach moves, the next movement starts at this
        # movement's goal cell state, including the planned dual-arm joints.
        goal_state = action.movements[movement_index + 1].start_state
        goal_joint_values = _joint_values_from_robot_cell_state(goal_state)
        goal_conf_source = f"movement[{movement_index + 1}].start_state"
    else:
        goal_state = start_state
        goal_joint_values = _joint_values_from_robot_cell_state(start_state)
        goal_conf_source = "movement.start_state"

    bar_state = ((goal_state.rigid_body_states or {}).get(active_bar_name)
                 or (start_state.rigid_body_states or {}).get(active_bar_name))
    if bar_state is None or bar_state.attachment_frame is None:
        raise ValueError(f"BarAction start_state has no attached active bar {active_bar_name!r}.")

    start_joint_values = _joint_values_from_robot_cell_state(start_state)

    # BarAction target EE frames are authored in the cell/world frame. Convert
    # to the mobile-base frame because this planner keeps the husky at origin.
    world_from_mobile_base = compas_frame_to_pose(goal_state.robot_base_frame)
    mobile_base_from_world = pp.invert(world_from_mobile_base)
    mobile_base_from_tool0_left_goal = pp.multiply(
        mobile_base_from_world,
        compas_frame_to_pose(mv.target_ee_frames["left"]),
    )
    mobile_base_from_tool0_right_goal = pp.multiply(
        mobile_base_from_world,
        compas_frame_to_pose(mv.target_ee_frames["right"]),
    )

    attached_link = getattr(bar_state, "attached_to_link", None)
    if attached_link == TOOL_LINK_RIGHT:
        tool0_from_bar = compas_frame_to_pose(bar_state.attachment_frame)
        mobile_base_from_bar_goal = pp.multiply(mobile_base_from_tool0_right_goal, tool0_from_bar)
    else:
        # The current gdrive actions attach the active bar to left_ur_arm_tool0.
        # Unknown link names fall back to the same convention and still log the
        # source link in the returned metadata.
        tool0_from_bar = compas_frame_to_pose(bar_state.attachment_frame)
        mobile_base_from_bar_goal = pp.multiply(mobile_base_from_tool0_left_goal, tool0_from_bar)

    grasp_targets = [
        (mobile_base_from_bar_goal, mobile_base_from_tool0_left_goal),
        (mobile_base_from_bar_goal, mobile_base_from_tool0_right_goal),
    ]

    try:
        active_bar_mesh = load_gdrive_active_bar_mesh(robot_cell_json, active_bar_name)
    except (KeyError, ValueError) as e:
        logger.warning(f"active bar mesh missing for {active_bar_name!r}: {e}; using default BAR_BOX_DIMS fallback")
        active_bar_mesh = None

    built_bars: List[Dict[str, Any]] = []
    if include_built_bars:
        logger.warning("BarAction include_built_bars=True is intentionally light; no full RobotCell bodies are imported yet.")

    return {
        "start_joint_values": start_joint_values,
        "end_joint_values": goal_joint_values,
        "grasp_targets": grasp_targets,
        "world_from_bar_goal": mobile_base_from_bar_goal,
        "active_bar_mesh": active_bar_mesh,
        "built_bars": built_bars,
        "_gdrive_bar_action_path": action_path,
        "_gdrive_bar_action_id": action.action_id,
        "_gdrive_bar_action_movement": mv.movement_id,
        "_gdrive_bar_action_movement_index": movement_index,
        "_gdrive_bar_action_goal_conf_source": goal_conf_source,
        "_gdrive_active_bar_name": active_bar_name,
        "_gdrive_active_bar_attached_to_link": attached_link,
        "_gdrive_world_from_mobile_base": world_from_mobile_base,
        "_gdrive_robot_cell_json": robot_cell_json,
    }


def build_gdrive_scene_spec(
    state_json: str,
    problem: Optional[str] = None,
    *,
    include_env_bars: bool = True,
    include_active_extras: bool = True,
) -> Dict[str, Any]:
    """Build a scene_spec dict for run_stage_trial / setup_planning_scene
    from a single gdrive RobotCellState file.

    The husky's robot_base_frame is converted into the planner's mobile-base
    convention: all bar / env poses are expressed in mobile-base frame so
    the planner sees the husky at world origin (matching the legacy code's
    implicit assumption).

    Returns scene_spec with: start_joint_values, end_joint_values (both =
    goal_conf), grasp_targets (mb-frame), world_from_bar_goal (mb-frame),
    active_bar_mesh, built_bars (env_* + active_* extras as static bodies),
    active_bar_name (informational).
    """
    state_path, problem_dir, robot_cell_json = _resolve_gdrive_state_path(state_json, problem)

    with open(state_path) as f:
        state_blob = json.load(f)
    state_data = state_blob["data"]

    rc_data = state_data["robot_configuration"]["data"]
    name_to_value = {n: v for n, v in zip(rc_data["joint_names"], rc_data["joint_values"])}
    try:
        goal_conf = np.array([name_to_value[n] for n in HUSKY_DUAL_ARM_JOINT_NAMES], dtype=float)
    except KeyError as e:
        raise KeyError(f"Cell state {state_path!r} missing joint {e!s}; "
                       f"expected {HUSKY_DUAL_ARM_JOINT_NAMES}")

    world_from_mobile_base = frame_data_to_pose(state_data["robot_base_frame"])
    mobile_base_from_world = pp.invert(world_from_mobile_base)

    active_bar_name: Optional[str] = None
    world_from_bar_goal: Optional[PoseLike] = None
    extra_actives: List[Tuple[str, PoseLike]] = []
    env_bodies: List[Tuple[str, PoseLike]] = []
    for name, rbs_wrap in (state_data.get("rigid_body_states") or {}).items():
        rbs = rbs_wrap.get("data", rbs_wrap)
        frame_wrap = rbs.get("frame")
        if frame_wrap is None:
            continue
        pose_world = frame_data_to_pose(frame_wrap)
        if name.startswith("active_bar_"):
            if active_bar_name is None:
                active_bar_name = name
                world_from_bar_goal = pose_world
        elif name.startswith("active_"):
            extra_actives.append((name, pose_world))
        elif name.startswith("env_"):
            env_bodies.append((name, pose_world))
    if active_bar_name is None or world_from_bar_goal is None:
        raise ValueError(f"No active_bar_* rigid body in {state_path}")

    mb_from_bar_goal = pp.multiply(mobile_base_from_world, world_from_bar_goal)
    grasp_bar_from_left, grasp_bar_from_right = _fk_dual_arm_grasps_in_mb_frame(
        goal_conf, mb_from_bar_goal,
    )
    mb_from_tool0_L_goal = pp.multiply(mb_from_bar_goal, grasp_bar_from_left)
    mb_from_tool0_R_goal = pp.multiply(mb_from_bar_goal, grasp_bar_from_right)
    grasp_targets = [
        (mb_from_bar_goal, mb_from_tool0_L_goal),
        (mb_from_bar_goal, mb_from_tool0_R_goal),
    ]

    try:
        active_bar_mesh = load_gdrive_active_bar_mesh(robot_cell_json, active_bar_name)
    except (KeyError, ValueError) as e:
        # Match husky_monitor._create_rigid_body_obstacle's fallback: when the
        # RobotCell.json was authored for a different active bar (e.g. only B5
        # has mesh data but the cell state is for B6), let setup_planning_scene
        # fall back to BAR_BOX_DIMS via active_bar_mesh=None.
        logger.warning(f"active bar mesh missing for {active_bar_name!r}: {e}; using default BAR_BOX_DIMS fallback")
        active_bar_mesh = None

    built_bars: List[Dict[str, Any]] = []
    if include_env_bars:
        for name, pose_world in env_bodies:
            try:
                mesh = load_gdrive_active_bar_mesh(robot_cell_json, name)
            except (KeyError, ValueError) as e:
                logger.warning(f"skipping env body {name!r}: {e}")
                continue
            mb_pose = pp.multiply(mobile_base_from_world, pose_world)
            built_bars.append({"mesh": mesh, "pose": mb_pose, "collision": True,
                                "color": (0.5, 0.5, 0.55, 0.95)})
    if include_active_extras:
        for name, pose_world in extra_actives:
            try:
                mesh = load_gdrive_active_bar_mesh(robot_cell_json, name)
            except (KeyError, ValueError) as e:
                logger.warning(f"skipping active extra {name!r}: {e}")
                continue
            mb_pose = pp.multiply(mobile_base_from_world, pose_world)
            built_bars.append({"mesh": mesh, "pose": mb_pose, "collision": True,
                                "color": (0.85, 0.45, 0.15, 0.55)})

    return {
        "start_joint_values": goal_conf,
        "end_joint_values": goal_conf,
        "grasp_targets": grasp_targets,
        "world_from_bar_goal": mb_from_bar_goal,
        "active_bar_mesh": active_bar_mesh,
        "built_bars": built_bars,
        # informational only (not consumed by setup_planning_scene)
        "_gdrive_active_bar_name": active_bar_name,
        "_gdrive_world_from_mobile_base": world_from_mobile_base,
        "_gdrive_state_path": state_path,
        "_gdrive_robot_cell_json": robot_cell_json,
    }


def import_static_bar_bodies(static_bar_specs: Sequence[Dict[str, Any]]) -> List[int]:
    bodies: List[int] = []
    for spec in static_bar_specs:
        body = create_bar_mesh_body(
            spec["mesh"],
            color=spec.get("color", (0.45, 0.45, 0.45, 0.95)),
            collision=bool(spec.get("collision", False)),
        )
        pp.set_pose(body, spec["pose"])
        bodies.append(body)
    return bodies


def create_visual_ee_marker(
    pose: PoseLike,
    color: Tuple[float, float, float, float],
    half_extents: Tuple[float, float, float] = (0.03, 0.015, 0.015),
) -> int:
    visual_shape = pybullet.createVisualShape(
        pybullet.GEOM_BOX,
        halfExtents=list(half_extents),
        rgbaColor=list(color),
    )
    return pybullet.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape,
        basePosition=list(np.asarray(pose[0], dtype=float)),
        baseOrientation=list(np.asarray(pose[1], dtype=float)),
    )


def add_grasp_pose_markers(
    start_pose: PoseLike,
    end_pose: PoseLike,
    grasp_bar_from_left: PoseLike,
    grasp_bar_from_right: Optional[PoseLike],
) -> List[int]:
    marker_bodies: List[int] = []

    start_left = pp.multiply(start_pose, grasp_bar_from_left)
    marker_bodies.append(create_visual_ee_marker(start_left, color=(0.1, 0.45, 1.0, 0.7)))
    pp.add_text("Start L", np.asarray(start_left[0], dtype=float) + np.array([0.0, 0.0, 0.06]), color=(0.1, 0.45, 1.0, 1.0))

    goal_left = pp.multiply(end_pose, grasp_bar_from_left)
    marker_bodies.append(create_visual_ee_marker(goal_left, color=(0.1, 0.45, 1.0, 0.35)))
    pp.add_text("Goal L", np.asarray(goal_left[0], dtype=float) + np.array([0.0, 0.0, 0.06]), color=(0.1, 0.45, 1.0, 1.0))

    if grasp_bar_from_right is not None:
        start_right = pp.multiply(start_pose, grasp_bar_from_right)
        marker_bodies.append(create_visual_ee_marker(start_right, color=(1.0, 0.55, 0.1, 0.7)))
        pp.add_text("Start R", np.asarray(start_right[0], dtype=float) + np.array([0.0, 0.0, 0.06]), color=(1.0, 0.55, 0.1, 1.0))

        goal_right = pp.multiply(end_pose, grasp_bar_from_right)
        marker_bodies.append(create_visual_ee_marker(goal_right, color=(1.0, 0.55, 0.1, 0.35)))
        pp.add_text("Goal R", np.asarray(goal_right[0], dtype=float) + np.array([0.0, 0.0, 0.06]), color=(1.0, 0.55, 0.1, 1.0))

    return marker_bodies


def setup_planning_scene(
    scene_spec: Dict[str, Any],
    use_gui: bool = False,
) -> Dict[str, Any]:
    if not os.path.isfile(HUSKY_DUAL_URDF_PATH):
        raise FileNotFoundError(f"URDF not found: {HUSKY_DUAL_URDF_PATH}")
    scene_spec = dict(scene_spec)
    for required_key in ("grasp_targets", "start_joint_values", "end_joint_values"):
        if required_key not in scene_spec:
            raise ValueError(f"scene_spec missing required key {required_key!r}")
    if not scene_spec["grasp_targets"]:
        raise ValueError("scene_spec must contain non-empty 'grasp_targets'")

    cid = pp.connect(use_gui=use_gui)
    if sys.platform == "win32":
        sys.stdout = open("CONOUT$", "w")
        sys.stderr = open("CONOUT$", "w")
    if use_gui:
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 1, physicsClientId=cid)
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=2.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5],
            physicsClientId=cid,
        )

    # Silence PyBullet's URDF loader warnings so planner logs stay readable.
    with pp.LockRenderer(), suppress_native_output():
        robot = pp.load_pybullet(HUSKY_DUAL_URDF_PATH, fixed_base=True)
    arm_joints = pp.joints_from_names(robot, HUSKY_DUAL_ARM_JOINT_NAMES)
    tool_link_left = pp.link_from_name(robot, TOOL_LINK_LEFT)
    tool_link_right = pp.link_from_name(robot, TOOL_LINK_RIGHT)
    pp.set_joint_positions(robot, arm_joints, INIT_ARM_JOINT_ANGLES)

    active_bar_mesh = scene_spec.get("active_bar_mesh")
    bar_box_dims = tuple(active_bar_mesh["aabb_dims"]) if active_bar_mesh is not None else BAR_BOX_DIMS
    bar_label = None if active_bar_mesh is None else active_bar_mesh.get("name", active_bar_mesh.get("body_name"))
    start_text_label = "Start" if bar_label is None else f"Start: {bar_label}"
    if active_bar_mesh is not None:
        bar_body = create_bar_mesh_body(active_bar_mesh, color=(0.8, 0.4, 0.1, 0.65), collision=True)
        ghost_start = create_bar_mesh_body(active_bar_mesh, color=(0.0, 0.8, 0.0, 0.35), collision=False)
        ghost_goal = create_bar_mesh_body(active_bar_mesh, color=(0.8, 0.0, 0.0, 0.35), collision=False)
    else:
        box_width, box_depth, box_length = bar_box_dims
        bar_body = pp.create_box(box_width, box_depth, box_length, color=(0.8, 0.4, 0.1, 0.65))
        ghost_start = pp.create_box(box_width, box_depth, box_length, color=(0.0, 0.8, 0.0, 0.35))
        ghost_goal = pp.create_box(box_width, box_depth, box_length, color=(0.8, 0.0, 0.0, 0.35))

    grasp_targets = scene_spec["grasp_targets"]
    world_from_bar_l, world_from_tool0_left = grasp_targets[0]
    grasp_bar_from_left = pp.multiply(pp.invert(world_from_bar_l), world_from_tool0_left)
    grasp_bar_from_right: Optional[PoseLike] = None
    if len(grasp_targets) >= 2:
        world_from_bar_r, world_from_tool0_right = grasp_targets[1]
        grasp_bar_from_right = pp.multiply(pp.invert(world_from_bar_r), world_from_tool0_right)

    start_joint_values = np.asarray(scene_spec["start_joint_values"], dtype=float)
    end_joint_values = np.asarray(scene_spec["end_joint_values"], dtype=float)

    world_from_bar_grasp = get_goal_pose_from_grasp_targets(grasp_targets)
    start_pose_context = None
    if len(grasp_targets) >= 2:
        mobile_base_from_tool0_left_home = scene_spec.get("mobile_base_from_tool0_left_home", MOBILE_BASE_FROM_TOOL0_LEFT_HOME)

        pp.draw_pose(mobile_base_from_tool0_left_home)
        # pp.draw_pose(mobile_base_from_tool0_right_start)

        start_pose_context = derive_home_start_poses_from_grasps(
            grasp_targets,
            mobile_base_from_tool0_left=mobile_base_from_tool0_left_home,
        )
    world_from_bar_start = scene_spec.get(
        "world_from_bar_start",
        world_from_bar_grasp if start_pose_context is None else start_pose_context["mobile_base_from_bar_start"],
    )
    world_from_bar_goal = scene_spec.get("world_from_bar_goal", world_from_bar_grasp)

    pp.set_joint_positions(robot, arm_joints, start_joint_values)
    pp.set_pose(bar_body, world_from_bar_start)
    pp.set_pose(ghost_start, world_from_bar_start)
    pp.set_pose(ghost_goal, world_from_bar_goal)
    start_text_id = pp.add_text(start_text_label, world_from_bar_start[0], color=(0.0, 0.8, 0.0, 1.0))
    goal_text_id = pp.add_text("Goal", world_from_bar_goal[0], color=(0.8, 0.0, 0.0, 1.0))
    start_pose_axes = pp.draw_pose(world_from_bar_start, length=0.15)
    goal_pose_axes = pp.draw_pose(world_from_bar_goal, length=0.15)
    grasp_marker_bodies = add_grasp_pose_markers(
        start_pose=world_from_bar_start,
        end_pose=world_from_bar_goal,
        grasp_bar_from_left=grasp_bar_from_left,
        grasp_bar_from_right=grasp_bar_from_right,
    )

    static_bar_bodies = import_static_bar_bodies(scene_spec.get("built_bars", []))

    non_obstacle_bodies = {bar_body, ghost_start, ghost_goal, *grasp_marker_bodies}
    collision_obstacles = [body for body in pp.get_bodies() if body not in non_obstacle_bodies]

    return {
        "cid": cid,
        "robot": robot,
        "arm_joints": arm_joints,
        "tool_link_left": tool_link_left,
        "tool_link_right": tool_link_right,
        "bar_body": bar_body,
        "ghost_start": ghost_start,
        "ghost_goal": ghost_goal,
        "world_from_bar_grasp": world_from_bar_grasp,
        "bar_label": bar_label,
        "world_from_bar_start": world_from_bar_start,
        "world_from_bar_goal": world_from_bar_goal,
        "mobile_base_from_tool0_left_home": (
            MOBILE_BASE_FROM_TOOL0_LEFT_HOME if start_pose_context is None else start_pose_context["mobile_base_from_tool0_left_start"]
        ),
        "mobile_base_from_tool0_right_start": (
            None if start_pose_context is None else start_pose_context["mobile_base_from_tool0_right_start"]
        ),
        "tool0_left_from_bar": None if start_pose_context is None else start_pose_context["tool0_left_from_bar"],
        "bar_from_tool0_right": None if start_pose_context is None else start_pose_context["bar_from_tool0_right"],
        "start_pose_axes": start_pose_axes,
        "goal_pose_axes": goal_pose_axes,
        "start_text_id": start_text_id,
        "goal_text_id": goal_text_id,
        "start_text_label": start_text_label,
        "start_pose": world_from_bar_start,
        "end_pose": world_from_bar_goal,
        "start_joint_values": start_joint_values,
        "end_joint_values": end_joint_values,
        "grasp_bar_from_left": grasp_bar_from_left,
        "grasp_bar_from_right": grasp_bar_from_right,
        "bar_box_dims": bar_box_dims,
        "feature_points": get_bar_feature_points(bar_box_dims),
        "grasp_marker_bodies": grasp_marker_bodies,
        "static_bar_bodies": static_bar_bodies,
        "collision_obstacles": collision_obstacles,
    }


def teardown_planning_scene() -> None:
    pp.disconnect()


def run_stage_trial(
    stage: int,
    scene_spec: Dict[str, Any],
    use_gui: bool = False,
    dist_metric: str = "feature",
    goal_bias: float = 0.1,
    position_res: float = 0.01,
    rotation_res: float = 0.025,
    max_time: float = 30.0,
    max_iterations: int = 2000,
    max_attempts: int = 5,
    endpoint_ik_attempts: int = 20,
    random_seed: Optional[int] = None,
    enable_collision: bool = True,
    enable_smoothing: bool = True,
    smooth_max_iterations: int = 100,
    smooth_max_time: float = 10.0,
    smooth_min_cost_improvement: float = 0.0,
    joint_continuity_threshold_rad: Optional[float] = DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
    use_angle_normalization: bool = DEFAULT_USE_ANGLE_NORMALIZATION,
    lock_renderer_during_search: bool = False,
    draw_rrt_tree: bool = True,
    validation_reports_dir: Optional[str] = None,
    debug_tree_out: Optional[Dict] = None,
    start_retries: int = 1,
    **_unused_kwargs: Any,
) -> Dict[str, Any]:
    if stage not in {1, 2, 3}:
        raise ValueError(f"Unsupported stage: {stage}")
    # Always have a local debug dict so we can capture the planner's stop_reason histogram,
    # regardless of whether the caller wants the tree dump.
    if debug_tree_out is None:
        debug_tree_out = {}
    scene = setup_planning_scene(scene_spec=scene_spec, use_gui=use_gui)

    def early_failure(failure_reason: str) -> Dict[str, Any]:
        return {
            "stage": stage,
            "scene": scene,
            "path": None,
            "path_confs": None,
            "path_before_smoothing": None,
            "path_confs_before_smoothing": None,
            "joint_continuity": None,
            "validation_joint_path": None,
            "validation_joint_path_source": None,
            "validation": {
                "joint_continuity_ok": None,
                "collision_free": None,
                "failure_reason": failure_reason,
                "plot_path": None,
            },
            "start_conf": None,
            "goal_conf": None,
            "planning_time_s": 0.0,
            "smoothing_time_s": 0.0,
            "validation_time_s": 0.0,
            "runtime_s": 0.0,
            "path_found": False,
            "success": False,
            "failure_reason": failure_reason,
            "smoothing": None,
        }

    try:
        enable_ik = stage >= 2
        enforce_collision = stage >= 3
        rng = np.random.default_rng(random_seed)
        start_conf = None
        goal_conf = None
        joint_collision_fn = None
        if enable_ik:
            from husky_assembly_tamp.motion_planner.api import derive_grasps_from_state

            # Match the live/headless constrained path: solve the goal first,
            # derive FK-consistent grasps at the goal, then compute a validated
            # home/start pose and start_conf with derive_constrained_start.
            grasp_bar_from_right = scene["grasp_bar_from_right"]
            if grasp_bar_from_right is None:
                raise ValueError(f"Stage {stage} requires both left and right grasp targets.")
            env_obstacles = [body for body in scene["collision_obstacles"] if body != scene["robot"]]
            if enforce_collision:
                joint_collision_fn = get_joint_collision_fn(
                    robot=scene["robot"],
                    arm_joints=scene["arm_joints"],
                    obstacle_bodies=env_obstacles,
                    tool_link_left=scene["tool_link_left"],
                    bar_body=scene["bar_body"],
                    grasp_bar_from_left=scene["grasp_bar_from_left"],
                )
            goal_seed = np.asarray(scene["end_joint_values"], dtype=float)
            if validate_dual_arm_bar_pose(
                robot=scene["robot"],
                arm_joints=scene["arm_joints"],
                tool_link_left=scene["tool_link_left"],
                tool_link_right=scene["tool_link_right"],
                full_conf=goal_seed,
                bar_pose=scene["world_from_bar_goal"],
                grasp_bar_from_left=scene["grasp_bar_from_left"],
                grasp_bar_from_right=grasp_bar_from_right,
                pos_tolerance=1e-3,
                ori_tolerance=1e-2,
            ) and (joint_collision_fn is None or not joint_collision_fn(goal_seed)):
                goal_conf = goal_seed
                logger.info("Using authored goal_conf from scene end_joint_values.")
            else:
                goal_conf = solve_endpoint_dual_arm_ik(
                    robot=scene["robot"],
                    arm_joints=scene["arm_joints"],
                    tool_link_left=scene["tool_link_left"],
                    tool_link_right=scene["tool_link_right"],
                    bar_pose=scene["world_from_bar_goal"],
                    grasp_bar_from_left=scene["grasp_bar_from_left"],
                    grasp_bar_from_right=grasp_bar_from_right,
                    seed_conf=goal_seed,
                    rng=rng,
                    max_attempts=endpoint_ik_attempts,
                    use_angle_normalization=use_angle_normalization,
                    collision_fn=joint_collision_fn,
                )
            if goal_conf is None:
                logger.warning(f"Stage {stage} goal pose has no valid dual-arm IK solution.")
                return early_failure("goal_ik_failure")
            scene["end_joint_values"] = np.asarray(goal_conf, dtype=float)
            scene["grasp_bar_from_left"], scene["grasp_bar_from_right"] = derive_grasps_from_state(
                scene["robot"],
                scene["arm_joints"],
                scene["tool_link_left"],
                scene["tool_link_right"],
                goal_conf,
                scene["world_from_bar_goal"],
            )
            grasp_bar_from_right = scene["grasp_bar_from_right"]
            if enforce_collision:
                # Rebuild after FK-consistent grasp derivation so the bar
                # attachment used for start validation matches the planner.
                joint_collision_fn = get_joint_collision_fn(
                    robot=scene["robot"],
                    arm_joints=scene["arm_joints"],
                    obstacle_bodies=env_obstacles,
                    tool_link_left=scene["tool_link_left"],
                    bar_body=scene["bar_body"],
                    grasp_bar_from_left=scene["grasp_bar_from_left"],
                )
            def _derive_start_for_attempt(attempt_idx: int) -> Tuple[Optional[PoseLike], Optional[np.ndarray]]:
                # attempt_idx==0 keeps the original deterministic behavior so
                # easy targets (e.g. B235) plan identically to the legacy code.
                # Later attempts shuffle the bar-position grid with a different
                # seed AND widen the sweep box (in z particularly) so we can
                # derive starts much closer to extreme goals (e.g. floor-level).
                if start_retries <= 1 or attempt_idx == 0:
                    derive_seed = random_seed
                    shuffle = False
                    sweep_box = ((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))
                else:
                    base = 0 if random_seed is None else int(random_seed)
                    derive_seed = base + 9973 * attempt_idx
                    shuffle = True
                    # Widen sweep box (especially -z) so that for goals far below
                    # the default home (z=0.86) the start can be closer in z.
                    sweep_box = ((-0.4, 0.4), (-0.4, 0.4), (-0.5, 0.3))
                return derive_constrained_start(
                    scene["robot"],
                    scene["arm_joints"],
                    scene["tool_link_left"],
                    scene["tool_link_right"],
                    scene["grasp_bar_from_left"],
                    grasp_bar_from_right,
                    scene["world_from_bar_goal"],
                    seed_conf=goal_conf,
                    bar_body=scene["bar_body"] if enforce_collision else None,
                    obstacles=env_obstacles if enforce_collision else (),
                    random_seed=derive_seed,
                    max_ik_attempts=endpoint_ik_attempts,
                    shuffle_deltas=shuffle,
                    bar_sweep_box=sweep_box,
                )

            world_from_bar_start, start_conf = _derive_start_for_attempt(0)
            if start_conf is None or world_from_bar_start is None:
                logger.warning(f"Stage {stage} start pose has no valid derived dual-arm IK solution.")
                return early_failure("start_ik_failure")
            scene["world_from_bar_start"] = world_from_bar_start
            scene["start_pose"] = world_from_bar_start
            scene["start_joint_values"] = np.asarray(start_conf, dtype=float)
            pp.set_pose(scene["bar_body"], world_from_bar_start)
            pp.set_pose(scene["ghost_start"], world_from_bar_start)
            grasp_marker_bodies = scene.get("grasp_marker_bodies") or []
            if grasp_marker_bodies:
                # Keep GUI markers consistent with the derived start pose.
                pp.set_pose(grasp_marker_bodies[0], pp.multiply(world_from_bar_start, scene["grasp_bar_from_left"]))
                if len(grasp_marker_bodies) >= 3:
                    pp.set_pose(grasp_marker_bodies[2], pp.multiply(world_from_bar_start, grasp_bar_from_right))

        logger.info(f"Running minimal Stage {stage} RRT.")
        logger.info(f"  start pose: {np.round(scene['world_from_bar_start'][0], 4)}")
        logger.info(f"  goal pose:  {np.round(scene['world_from_bar_goal'][0], 4)}")
        logger.info(f"  IK: {'on' if enable_ik else 'off'}")
        logger.info(f"  collision: {'on' if (enable_collision if stage == 1 else enforce_collision) else 'off'}")
        logger.info(f"  position_res: {position_res}")
        logger.info(f"  rotation_res: {rotation_res}")
        if enable_ik:
            logger.info(f"  joint continuity threshold: {joint_continuity_threshold_rad}")
            logger.info(f"  angle normalization: {'on' if use_angle_normalization else 'off'}")
        logger.info(f"  collision obstacles: {len(scene['collision_obstacles'])} bodies")
        logger.info(f"  lock renderer during search: {'on' if (use_gui and lock_renderer_during_search) else 'off'}")

        # The core RRT stays pose-space-first; extra constraints are injected through
        # the IK and collision callbacks rather than changing the tree structure.
        planning_kwargs = dict(
            robot=scene["robot"],
            bar_body=scene["bar_body"],
            obstacle_bodies=scene["collision_obstacles"],
            start_pose=scene["world_from_bar_start"],
            goal_pose=scene["world_from_bar_goal"],
            start_conf=start_conf,
            goal_conf=goal_conf,
            dist_metric=dist_metric,
            goal_sample_prob=goal_bias,
            position_res=position_res,
            rotation_res=rotation_res,
            random_seed=random_seed,
            max_time=max_time,
            max_iterations=max_iterations,
            max_attempts=max_attempts,
            enable_collision=(enable_collision if stage == 1 else enforce_collision),
            enable_ik=enable_ik,
            ik_context={
                "robot": scene["robot"],
                "arm_joints": scene["arm_joints"],
                "tool_link_left": scene["tool_link_left"],
                "tool_link_right": scene["tool_link_right"],
                "grasp_bar_from_left": scene["grasp_bar_from_left"],
                "grasp_bar_from_right": scene["grasp_bar_from_right"],
            }
            if enable_ik
            else None,
            joint_collision_fn=joint_collision_fn,
            feature_points=scene["feature_points"],
            joint_continuity_threshold_rad=(joint_continuity_threshold_rad if enable_ik else None),
            use_angle_normalization=use_angle_normalization,
            use_draw=use_gui and draw_rrt_tree,
            debug_tree_out=debug_tree_out,
        )
        t0 = time.perf_counter()
        planning_time_s = 0.0
        smoothing_time_s = 0.0
        validation_time_s = 0.0
        t_plan = time.perf_counter()

        def _run_plan() -> Tuple[Optional[List[PoseLike]], Optional[List[FullConf]]]:
            if use_gui and lock_renderer_during_search:
                with pp.LockRenderer():
                    return plan_pose_rrt(**planning_kwargs)
            return plan_pose_rrt(**planning_kwargs)

        path, path_confs = _run_plan()
        if path is None and enable_ik and start_retries > 1:
            # Single fixed start can be a dead end for hard problems (e.g. when the
            # goal is far from the home anchor in z). Re-derive the start with a
            # shuffled bar-position grid and retry.
            for retry_idx in range(1, start_retries):
                logger.info(
                    f"plan failed from start #{retry_idx - 1}; re-deriving start (retry {retry_idx}/{start_retries - 1})."
                )
                new_start_pose, new_start_conf = _derive_start_for_attempt(retry_idx)
                if new_start_conf is None or new_start_pose is None:
                    logger.warning(f"start retry {retry_idx}: no valid derived start; skipping.")
                    continue
                scene["world_from_bar_start"] = new_start_pose
                scene["start_pose"] = new_start_pose
                scene["start_joint_values"] = np.asarray(new_start_conf, dtype=float)
                pp.set_pose(scene["bar_body"], new_start_pose)
                pp.set_pose(scene["ghost_start"], new_start_pose)
                grasp_marker_bodies = scene.get("grasp_marker_bodies") or []
                if grasp_marker_bodies:
                    pp.set_pose(grasp_marker_bodies[0], pp.multiply(new_start_pose, scene["grasp_bar_from_left"]))
                    if len(grasp_marker_bodies) >= 3:
                        pp.set_pose(grasp_marker_bodies[2], pp.multiply(new_start_pose, grasp_bar_from_right))
                planning_kwargs["start_pose"] = new_start_pose
                planning_kwargs["start_conf"] = new_start_conf
                start_conf = new_start_conf
                # Reset the debug tree (extend_stop_reasons accumulates) and re-run.
                if debug_tree_out is not None:
                    debug_tree_out.clear()
                logger.info(f"  retry start pose: {np.round(new_start_pose[0], 4)}")
                path, path_confs = _run_plan()
                if path is not None:
                    logger.info(f"plan succeeded on start retry {retry_idx}.")
                    break
        planning_time_s = time.perf_counter() - t_plan

        path_before_smoothing = None if path is None else list(path)
        path_confs_before_smoothing = None if path_confs is None else [np.asarray(conf, dtype=float) for conf in path_confs]
        if path is not None and enable_smoothing:
            smooth_pose_collision_fn = (
                get_pose_collision_fn(scene["bar_body"], scene["collision_obstacles"], True)
                if (stage == 1 and enable_collision)
                else None
            )
            t_smooth = time.perf_counter()
            path, path_confs = smooth_dual_arm_pose_path(
                path_poses=path,
                path_confs=path_confs,
                scene=scene,
                pose_collision_fn=smooth_pose_collision_fn,
                joint_collision_fn=joint_collision_fn,
                dist_metric=dist_metric,
                feature_points=scene["feature_points"],
                position_res=position_res,
                rotation_res=rotation_res,
                joint_continuity_threshold_rad=(joint_continuity_threshold_rad if enable_ik else None),
                use_angle_normalization=use_angle_normalization,
                max_smooth_iterations=smooth_max_iterations,
                max_time=smooth_max_time,
                min_cost_improvement=smooth_min_cost_improvement,
                random_seed=random_seed,
            )
            smoothing_time_s = time.perf_counter() - t_smooth
            logger.info(f"Smoothing reduced path to {len(path)} waypoints.")
        runtime_s = time.perf_counter() - t0

        if path is not None:
            logger.info(f"Found Stage {stage} pose path with {len(path)} waypoints.")
            pp.set_pose(scene["bar_body"], path[-1])
            if enable_ik and path_confs:
                pp.set_joint_positions(scene["robot"], scene["arm_joints"], path_confs[-1])
        else:
            logger.warning(f"No Stage {stage} pose path found.")

        coarse_continuity = (
            summarize_joint_continuity(path_confs, use_angle_normalization=use_angle_normalization)
            if path_confs is not None
            else None
        )

        if path_confs is not None:
            # Forward the smoothed path AS-IS (continuous joint values, can
            # extend beyond ±π). Pre-normalizing here used to introduce
            # spurious ±2π wraps in the validation display + continuity
            # check; the densifier in refine_joint_path_for_validation
            # re-normalizes internally for collision math, so this is a
            # display + continuity-check fidelity win at no correctness
            # cost.
            validation_joint_path = [np.asarray(conf, dtype=float) for conf in path_confs]
            validation_joint_path_source = "planner"
            validation_joint_path_reason = None
        else:
            validation_joint_path = None
            validation_joint_path_source = None
            validation_joint_path_reason = "planner_joint_path_unavailable"
        validation_kwargs = dict(
            stage=stage,
            scene=scene,
            path=path,
            joint_path=validation_joint_path,
            original_joint_path=path_confs_before_smoothing,
            joint_path_source=validation_joint_path_source,
            joint_path_reason=validation_joint_path_reason,
            urdf_path=HUSKY_DUAL_URDF_PATH,
            srdf_path=HUSKY_DUAL_SRDF_PATH,
            grasp_mask_links=STAGE3_GRASP_MASK_LINKS,
            target_label=scene.get("bar_label"),
            position_res=position_res,
            rotation_res=rotation_res,
            use_angle_normalization=use_angle_normalization,
        )
        if validation_reports_dir is not None:
            validation_kwargs["reports_dir"] = validation_reports_dir
        t_validation = time.perf_counter()
        if use_gui:
            with pp.LockRenderer():
                validation = validate_stage_trajectory(**validation_kwargs)
        else:
            validation = validate_stage_trajectory(**validation_kwargs)
        validation_time_s = time.perf_counter() - t_validation
        # Surface the RRT extend_toward stop_reason histogram on the validation dict
        # so it gets logged and saved alongside other diagnostics.
        validation["extend_stop_reasons"] = dict(debug_tree_out.get("extend_stop_reasons") or {})
        log_validation_summary(validation)
        path_found = path is not None
        validated_success = path_found
        if stage >= 2:
            validated_success = validated_success and bool(validation.get("joint_continuity_ok"))
            validated_success = validated_success and bool(validation.get("collision_free"))

        return {
            "stage": stage,
            "scene": scene,
            "path": path,
            "path_confs": path_confs,
            "path_before_smoothing": path_before_smoothing,
            "path_confs_before_smoothing": path_confs_before_smoothing,
            "joint_continuity": coarse_continuity,
            "validation_joint_path": validation_joint_path,
            "validation_joint_path_source": validation_joint_path_source,
            "validation": validation,
            "start_conf": start_conf,
            "goal_conf": goal_conf,
            "planning_time_s": planning_time_s,
            "smoothing_time_s": smoothing_time_s,
            "validation_time_s": validation_time_s,
            "runtime_s": runtime_s,
            "path_found": path_found,
            "success": bool(validated_success),
            "smoothing": None,
        }
    except Exception:
        teardown_planning_scene()
        raise


def log_validation_summary(validation: Dict[str, Any]) -> None:
    plot_path = validation.get("plot_path")
    if plot_path:
        logger.info(f"Saved trajectory validation plot: {plot_path}")

    # RRT `extend_toward` stop_reason histogram — useful for diagnosing failed plans.
    extend_stop_reasons = validation.get("extend_stop_reasons") or {}
    if extend_stop_reasons:
        total = sum(extend_stop_reasons.values())
        # Sort by count descending so the dominant failure mode prints first.
        items = sorted(extend_stop_reasons.items(), key=lambda kv: kv[1], reverse=True)
        parts = [f"{reason}={count} ({count / total * 100:.1f}%)" for reason, count in items]
        logger.info(f"extend_toward stop_reasons (total={total}): " + ", ".join(parts))

    collision_free = validation.get("collision_free")
    if collision_free is None:
        reason = validation.get("joint_path_reason")
        if reason:
            logger.warning(f"Trajectory validation could not run robot-side checks: {reason}")
    else:
        logger.info(f"Trajectory collisions: {'pass' if collision_free else 'fail'}")
        for key, details in validation.get("collision_breakdown", {}).items():
            if details.get("count", 0) <= 0:
                continue
            logger.warning(
                "  %s collisions: %s hits (first waypoint %s)",
                key,
                details["count"],
                details["first_index"],
            )

    joint_continuity_ok = validation.get("joint_continuity_ok")
    if joint_continuity_ok is not None:
        logger.info(
            "Joint continuity: %s (max dq=%.4f rad, threshold=%.4f rad)",
            "pass" if joint_continuity_ok else "fail",
            float(validation.get("joint_continuity_max_delta_rad") or 0.0),
            float(validation.get("joint_continuity_threshold_rad") or 0.0),
        )
        if validation.get("joint_continuity_first_bad_step") is not None:
            logger.warning(f"  first joint continuity violation at waypoint {validation['joint_continuity_first_bad_step']}")

    relative_transform_ok = validation.get("relative_transform_ok")
    if relative_transform_ok is not None:
        max_axis_drift_deg = validation.get("relative_transform_max_axis_angle_deg") or {}
        logger.info(
            "End-effector relative transform drift: %s (max pos=%.6f m, max axis drift xyz=[%.3f, %.3f, %.3f] deg)",
            "pass" if relative_transform_ok else "fail",
            float(validation.get("relative_transform_max_translation_m") or 0.0),
            float(max_axis_drift_deg.get("x") or 0.0),
            float(max_axis_drift_deg.get("y") or 0.0),
            float(max_axis_drift_deg.get("z") or 0.0),
        )


def run_visualization_loop(
    bar_body: int,
    path: Sequence[PoseLike],
    cid: int,
    robot: Optional[int] = None,
    arm_joints: Optional[Sequence[int]] = None,
    path_confs: Optional[Sequence[FullConf]] = None,
    validation: Optional[Dict[str, Any]] = None,
    scene: Optional[Dict[str, Any]] = None,
) -> None:
    # PyBullet GUI sliders can occasionally fail to read; keep the last value so replay stays interactive.
    def read_debug_parameter_safe(param_id: Optional[int], default: float) -> float:
        if param_id is None:
            return default
        try:
            return float(pybullet.readUserDebugParameter(param_id, physicsClientId=cid))
        except pybullet.error:
            return float(default)

    if path is None:
        logger.info("No path to visualize. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            return

    logger.info(f"Visualizing pose path with {len(path)} waypoints.")
    path_slider = pybullet.addUserDebugParameter("Path t", 0.0, 1.0, 0.0, physicsClientId=cid)
    collision_failures = list((validation or {}).get("collision_failures") or [])
    replay_mode_slider = None
    replay_idx_slider = None
    current_replay_idx = -1
    replay_mode_enabled = False
    last_path_slider_value = 0.0
    last_replay_mode_value = 0.0
    last_replay_idx_value = 0.0
    joint_collision_checker = None
    bar_pose_source = (validation or {}).get("bar_pose_source") or (scene or {}).get("bar_pose_source", "left_grasp")
    if collision_failures:
        logger.info(
            "Validation captured %d collision-failing waypoints. Enable 'Collision Replay Mode' to inspect them.",
            len(collision_failures),
        )
        replay_mode_slider = pybullet.addUserDebugParameter("Collision Replay Mode", 0.0, 1.0, 0.0, physicsClientId=cid)
        replay_idx_slider = pybullet.addUserDebugParameter(
            "Collision Failure Index",
            0.0,
            float(len(collision_failures) - 1),
            0.0,
            physicsClientId=cid,
        )
        if scene is not None and bar_pose_source == "left_grasp":
            joint_collision_checker = get_joint_collision_fn(
                robot=scene["robot"],
                arm_joints=scene["arm_joints"],
                obstacle_bodies=[body for body in scene["collision_obstacles"] if body != scene["robot"]],
                tool_link_left=scene["tool_link_left"],
                bar_body=scene["bar_body"],
                grasp_bar_from_left=scene["grasp_bar_from_left"],
            )
    current_idx = -1
    while True:
        try:
            if replay_mode_slider is not None and replay_idx_slider is not None:
                last_replay_mode_value = read_debug_parameter_safe(replay_mode_slider, last_replay_mode_value)
                last_replay_idx_value = read_debug_parameter_safe(replay_idx_slider, last_replay_idx_value)
                replay_mode = last_replay_mode_value >= 0.5
                replay_idx = int(round(last_replay_idx_value))
                replay_idx = max(0, min(replay_idx, len(collision_failures) - 1))
                if replay_mode:
                    failure = collision_failures[replay_idx]
                    waypoint_idx = int(failure["waypoint_index"])
                    dense_waypoint_idx = failure.get("dense_waypoint_index")
                    failure_joint_values = failure.get("joint_values")
                    failure_bar_pose = failure.get("bar_pose")
                    if (not replay_mode_enabled) or replay_idx != current_replay_idx:
                        logger.warning(
                            "Replaying collision failure %d/%d at waypoint %d%s: %s",
                            replay_idx + 1,
                            len(collision_failures),
                            waypoint_idx,
                            "" if dense_waypoint_idx is None else f" (dense {dense_waypoint_idx})",
                            ", ".join(failure.get("collision_keys", [])),
                        )
                        # Make the replay-diagnosis gate explicit so skipped cases are visible in the terminal.
                        failure_collision_keys = list(failure.get("collision_keys", []))
                        has_joint_collision_checker = joint_collision_checker is not None
                        has_path_confs = path_confs is not None
                        waypoint_has_joint_conf = has_path_confs and 0 <= waypoint_idx < len(path_confs)
                        has_recorded_joint_conf = failure_joint_values is not None
                        has_robot_collision_key = any(
                            key in {"robot_self", "robot_static"} for key in failure_collision_keys
                        )
                        if has_joint_collision_checker and has_recorded_joint_conf and has_robot_collision_key:
                            joint_collision_checker(np.asarray(failure_joint_values, dtype=float), diagnosis=True)
                        elif has_joint_collision_checker and has_path_confs and waypoint_has_joint_conf and has_robot_collision_key:
                            joint_collision_checker(path_confs[waypoint_idx], diagnosis=True)
                        else:
                            logger.warning(
                                "  replay diagnosis skipped: checker=%s recorded_conf=%s path_confs=%s conf_at_waypoint=%s robot_collision_key=%s keys=%s",
                                has_joint_collision_checker,
                                has_recorded_joint_conf,
                                has_path_confs,
                                waypoint_has_joint_conf,
                                has_robot_collision_key,
                                failure_collision_keys,
                            )
                        for key in failure_collision_keys:
                            if key in {"bar_robot", "bar_static"}:
                                logger.warning("  %s diagnosis: on-demand floating-body pair diagnosis is disabled in replay", key)
                    current_replay_idx = replay_idx
                    replay_mode_enabled = True
                    if failure_bar_pose is not None:
                        pp.set_pose(
                            bar_body,
                            (
                                np.asarray(failure_bar_pose["position"], dtype=float),
                                np.asarray(failure_bar_pose["quaternion"], dtype=float),
                            ),
                        )
                        if robot is not None and arm_joints is not None and failure_joint_values is not None:
                            pp.set_joint_positions(robot, arm_joints, np.asarray(failure_joint_values, dtype=float))
                        current_idx = waypoint_idx
                    elif 0 <= waypoint_idx < len(path):
                        pp.set_pose(bar_body, path[waypoint_idx])
                        if robot is not None and arm_joints is not None and path_confs is not None and waypoint_idx < len(path_confs):
                            pp.set_joint_positions(robot, arm_joints, path_confs[waypoint_idx])
                        current_idx = waypoint_idx
                    time.sleep(0.01)
                    continue
                replay_mode_enabled = False

            last_path_slider_value = read_debug_parameter_safe(path_slider, last_path_slider_value)
            t = last_path_slider_value
            idx = int(round(t * (len(path) - 1)))
            idx = max(0, min(idx, len(path) - 1))
            if idx != current_idx:
                current_idx = idx
                pp.set_pose(bar_body, path[idx])
                if robot is not None and arm_joints is not None and path_confs is not None and idx < len(path_confs):
                    pp.set_joint_positions(robot, arm_joints, path_confs[idx])
            time.sleep(0.01)
        except KeyboardInterrupt:
            return


# --- post-plan output helpers (folded from real_state_study.py) --------------


def reports_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "reports")


def support_dir() -> str:
    path = os.path.join(reports_dir(), "_support")
    os.makedirs(path, exist_ok=True)
    return path


def pose_to_json(pose) -> Dict[str, List[float]]:
    return {
        "position": [float(v) for v in pose[0]],
        "quaternion": [float(v) for v in pose[1]],
    }


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def save_replay_bundle(
    *,
    scene: Dict[str, Any],
    spec: Dict[str, Any],
    joint_path: Sequence[Any],
    pose_path: Sequence[Any],
    trajectory_json_path: str,
    metadata_json_path: str,
) -> None:
    os.makedirs(os.path.dirname(trajectory_json_path), exist_ok=True)
    save_path_as_joint_trajectory(joint_path, HUSKY_DUAL_ARM_JOINT_NAMES, trajectory_json_path)
    metadata = {
        "scene_spec": {
            "mobile_base_from_tool0_left_home": pose_to_json(scene["mobile_base_from_tool0_left_home"]),
            "world_from_bar_start": pose_to_json(scene["world_from_bar_start"]),
            "start_joint_values": [float(v) for v in scene["start_joint_values"]],
            "end_joint_values": [float(v) for v in scene["end_joint_values"]],
            "world_from_bar_goal": pose_to_json(scene["world_from_bar_goal"]),
            "grasp_targets": [[pose_to_json(bar_pose), pose_to_json(tool_pose)] for bar_pose, tool_pose in spec["grasp_targets"]],
            "active_bar_mesh": to_jsonable(spec["active_bar_mesh"]),
            "built_bars": to_jsonable(spec["built_bars"]),
        },
        "pose_path": [pose_to_json(pose) for pose in pose_path],
        "start_pose": pose_to_json(scene["world_from_bar_start"]),
        "goal_pose": pose_to_json(scene["world_from_bar_goal"]),
    }
    with open(metadata_json_path, "w") as f:
        json.dump(metadata, f, indent=2)


def build_replay_command(
    *,
    trajectory_json_path: str,
    metadata_json_path: str,
) -> str:
    parts = [
        "cd",
        shlex.quote(os.getcwd()),
        "&&",
        shlex.quote(sys.executable),
        "-m",
        "husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.trajectory_replay",
        "--trajectory-json",
        shlex.quote(trajectory_json_path),
        "--metadata-json",
        shlex.quote(metadata_json_path),
    ]
    return " ".join(parts)


def record_trajectory_video(
    scene: Dict[str, Any],
    joint_path: Sequence[Any],
    pose_path: Sequence[Any],
    out_path: str,
    frame_step: int = 2,
    frame_sleep: float = 0.02,
    width: int = 1024,
    height: int = 768,
) -> Optional[str]:
    if len(pose_path) == 0 or len(joint_path) == 0:
        return None

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        logger.warning("Skipping trajectory video capture because imageio is unavailable: %s", exc)
        return None

    pp.set_pose(scene["ghost_start"], scene["world_from_bar_start"])
    pp.set_pose(scene["ghost_goal"], scene["world_from_bar_goal"])
    pp.set_pose(scene["bar_body"], pose_path[0])
    pp.set_joint_positions(scene["robot"], scene["arm_joints"], joint_path[0])

    view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.6, 0.0, 0.7],
        distance=2.3,
        yaw=45.0,
        pitch=-25.0,
        roll=0.0,
        upAxisIndex=2,
    )
    projection_matrix = pybullet.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=float(width) / float(height),
        nearVal=0.02,
        farVal=6.0,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fps = max(1, int(round(1.0 / max(frame_sleep, 1e-6))))
    try:
        with imageio.get_writer(out_path, fps=fps, macro_block_size=1) as writer:
            for index, (conf, pose) in enumerate(zip(joint_path, pose_path)):
                if index % max(1, frame_step) != 0 and index != len(joint_path) - 1:
                    continue
                pp.set_joint_positions(scene["robot"], scene["arm_joints"], conf)
                pp.set_pose(scene["bar_body"], pose)
                pybullet.stepSimulation(physicsClientId=scene["cid"])
                _, _, rgba, _, _ = pybullet.getCameraImage(
                    width=width,
                    height=height,
                    viewMatrix=view_matrix,
                    projectionMatrix=projection_matrix,
                    renderer=pybullet.ER_TINY_RENDERER,
                    physicsClientId=scene["cid"],
                )
                image = np.reshape(np.asarray(rgba, dtype=np.uint8), (height, width, 4))
                writer.append_data(image[:, :, :3])
    except Exception as exc:
        logger.warning("Skipping trajectory video capture because writing %s failed: %s", out_path, exc)
        if os.path.isfile(out_path):
            os.remove(out_path)
        return None

    if not os.path.isfile(out_path):
        return None
    return out_path


def summarize_result(target_name: str, spec: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    validation = result.get("validation", {})
    goal_pose = spec["world_from_bar_goal"]
    active_bar_mesh = spec.get("active_bar_mesh")
    active_bar_body_name = (
        active_bar_mesh.get("body_name") if active_bar_mesh is not None else "(no mesh)"
    )
    bar_box_dims = (
        [float(v) for v in active_bar_mesh["aabb_dims"]] if active_bar_mesh is not None else [0.0, 0.0, 0.0]
    )
    return {
        "target_name": target_name,
        "active_bar_body_name": active_bar_body_name,
        "bar_box_dims": bar_box_dims,
        "goal_position": [float(v) for v in goal_pose[0]],
        "goal_quaternion": [float(v) for v in goal_pose[1]],
        "path_found": bool(result.get("path_found", result.get("path") is not None)),
        "success": bool(result["success"]),
        "runtime_s": float(result.get("runtime_s", 0.0)),
        "planning_time_s": float(result.get("planning_time_s", 0.0)),
        "smoothing_time_s": float(result.get("smoothing_time_s", 0.0)),
        "validation_time_s": float(result.get("validation_time_s", 0.0)),
        "waypoints": int(len(result["path"])) if result["path"] is not None else 0,
        "max_dq_rad": validation.get("joint_continuity_max_delta_rad"),
        "joint_continuity_ok": validation.get("joint_continuity_ok"),
        "collision_free": validation.get("collision_free"),
        "collision_reason": validation.get("failure_reason", result.get("failure_reason")),
        "validation_plot": validation.get("plot_path"),
        "video_mp4": result.get("video_mp4"),
        "trajectory_json": result.get("trajectory_json"),
        "trajectory_metadata_json": result.get("trajectory_metadata_json"),
        "replay_command": result.get("replay_command"),
    }


def write_report(report_path: str, json_relpath: str, args, summaries: List[Dict[str, Any]]) -> None:
    from .path_validation import DEFAULT_DENSE_JOINT_VALIDATION_STEP_RAD

    lines: List[str] = []
    lines.append(f"# Dual-arm task-space RRT Stage {args.stage} run ({datetime.now().strftime('%Y%m%d_%H%M%S')})")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"This report benchmarks Stage {args.stage} against gdrive design-study targets using:")
    lines.append(f"- per-target start bar pose / start_conf derived by `derive_constrained_start`")
    lines.append(f"- per-target goal pose and grasps from the gdrive cell state / BarAction")
    lines.append(f"- per-target active bar mesh from `RobotCell.json`")
    lines.append(f"- planning frame: mobile-base frame")
    lines.append(f"- support JSON: `{json_relpath}`")
    lines.append("")
    lines.append(f"- Design root: `{args.design_root}`")
    lines.append(f"- Targets: `{', '.join(args.targets)}`")
    lines.append(f"- Include built bars: `{args.include_built_bars}`")
    lines.append(f"- Built-bar collision enabled: `{args.enable_built_bar_collision}`")
    lines.append(f"- Position resolution: `{args.position_res}`")
    lines.append(f"- Rotation resolution: `{args.rotation_res}`")
    lines.append(f"- Joint continuity threshold: `{args.joint_continuity_threshold}`")
    lines.append(f"- Dense joint validation step: `{DEFAULT_DENSE_JOINT_VALIDATION_STEP_RAD}`")
    lines.append(f"- Max time: `{args.max_time}`")
    lines.append(f"- Max iterations: `{args.max_iterations}`")
    lines.append(f"- Max attempts: `{args.max_attempts}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Target | Bar body | Bar dims (m) | Goal xyz (m) | Path found | Validated success | Planning time (s) | Smoothing time (s) | Validation time (s) | Waypoints | Max dq (rad) | Collision-free | Failure reason | Validation plot | Video MP4 | Replay command |")
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |")
    for item in summaries:
        validation_plot = item["validation_plot"]
        validation_label = os.path.relpath(validation_plot, reports_dir()) if validation_plot else "-"
        if validation_plot:
            validation_cell = f"[{validation_label}]({validation_label})"
        else:
            validation_cell = "-"
        video_mp4 = item.get("video_mp4")
        video_label = os.path.relpath(video_mp4, reports_dir()) if video_mp4 else "-"
        if video_mp4:
            video_cell = f"[{video_label}]({video_label})"
        else:
            video_cell = "-"
        replay_command = item.get("replay_command")
        replay_cell = "-" if not replay_command else f"`{replay_command}`"
        max_dq_text = "-" if item["max_dq_rad"] is None else f"{item['max_dq_rad']:.4f}"
        lines.append(
            f"| {item['target_name']} | {item['active_bar_body_name']} | "
            f"`{np.round(item['bar_box_dims'], 4).tolist()}` | "
            f"`{np.round(item['goal_position'], 4).tolist()}` | "
            f"{'PASS' if item['path_found'] else 'FAIL'} | "
            f"{'PASS' if item['success'] else 'FAIL'} | "
            f"{item['planning_time_s']:.3f} | "
            f"{item['smoothing_time_s']:.3f} | "
            f"{item['validation_time_s']:.3f} | "
            f"{item['waypoints']} | "
            f"{max_dq_text} | "
            f"{'PASS' if item['collision_free'] else 'FAIL'} | "
            f"{item['collision_reason'] or '-'} | "
            f"{validation_cell} | "
            f"{video_cell} | "
            f"{replay_cell} |"
        )
    lines.append("")
    validation_plot_items = [item for item in summaries if item.get("validation_plot")]
    if validation_plot_items:
        lines.append("## Validation Plots")
        lines.append("")
        for item in validation_plot_items:
            validation_plot = item["validation_plot"]
            validation_label = os.path.relpath(validation_plot, reports_dir())
            lines.append(f"### {item['target_name']}")
            lines.append("")
            lines.append(f"![Validation plot for {item['target_name']}]({validation_label})")
            lines.append("")
    successes = sum(1 for item in summaries if item["success"])
    lines.append(f"Validated Stage {args.stage} success: `{successes} / {len(summaries)}` targets.")
    lines.append("")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


# --- Batch CLI ----------------------------------------------------------------


def _build_target_spec(target_name: str, args) -> Tuple[str, Dict[str, Any]]:
    """Resolve one --targets entry to (target_name_with_movement, scene_spec)."""
    if args.gdrive_bar_action:
        scene_spec = build_gdrive_bar_action_scene_spec(
            target_name,
            movement=args.movement,
            problem=args.gdrive_problem,
            include_built_bars=args.include_built_bars,
        )
        full_name = (
            f"{os.path.splitext(os.path.basename(scene_spec['_gdrive_bar_action_path']))[0]}"
            f"_{scene_spec['_gdrive_bar_action_movement']}"
        )
        logger.info(f"Using gdrive BarAction scene_spec from {scene_spec['_gdrive_bar_action_path']}")
        logger.info(f"  movement={scene_spec['_gdrive_bar_action_movement']!r}, "
                    f"active_bar={scene_spec['_gdrive_active_bar_name']!r}, "
                    f"built_bars={len(scene_spec['built_bars'])}")
    else:
        scene_spec = build_gdrive_scene_spec(
            target_name,
            problem=args.gdrive_problem,
            include_env_bars=not args.gdrive_no_env,
            include_active_extras=not args.gdrive_no_active_extras,
        )
        if not args.enable_built_bar_collision and scene_spec["built_bars"]:
            scene_spec["built_bars"] = [{**b, "collision": False} for b in scene_spec["built_bars"]]
        full_name = os.path.splitext(os.path.basename(scene_spec["_gdrive_state_path"]))[0]
        logger.info(f"Using gdrive scene_spec from {scene_spec['_gdrive_state_path']}")
        logger.info(f"  active_bar={scene_spec['_gdrive_active_bar_name']!r}, "
                    f"built_bars={len(scene_spec['built_bars'])}")
    return full_name, scene_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-arm task-space RRT batch runner (Stages 1/2/3)")
    parser.add_argument("--stage", choices=[1, 2, 3], type=int, default=3, help="Planning stage to run")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI (default off; batch mode)")
    parser.add_argument(
        "--visualize-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When GUI is enabled, open the slider-based path viewer after planning each target",
    )
    parser.add_argument("--goal-bias", type=float, default=0.1, help="Goal sampling probability")
    parser.add_argument("--dist-metric", choices=["feature", "pose6d"], default="feature", help="Task-space distance metric")
    parser.add_argument("--position-res", type=float, default=0.01, help="Translation resolution used during pose extension, in meters")
    parser.add_argument("--rotation-res", type=float, default=0.025, help="Rotation resolution used during pose extension, in radians")
    parser.add_argument("--max-time", type=float, default=30.0, help="Max planning time per attempt")
    parser.add_argument("--max-iterations", type=int, default=2000, help="Max RRT iterations per attempt")
    parser.add_argument("--max-attempts", type=int, default=5, help="Random restarts")
    parser.add_argument("--endpoint-ik-attempts", type=int, default=20, help="Max random seeds used when solving endpoint IK in Stage 2/3")
    parser.add_argument(
        "--start-retries", type=int, default=1,
        help="On planning failure, re-derive the start bar pose this many times with a shuffled, widened sweep box.",
    )
    parser.add_argument(
        "--bidirectional", action="store_true",
        help="Use bidirectional RRT-Connect (two trees rooted at start and goal). Helps hard cases where a single forward tree barely grows.",
    )
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--smoothing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run dual-arm-aware shortcut smoothing on the planned path (use --no-smoothing to disable)",
    )
    parser.add_argument("--smooth-iterations", type=int, default=100, help="Max shortcut iterations for path smoothing")
    parser.add_argument("--smooth-max-time", type=float, default=10.0, help="Max wall time (s) for path smoothing")
    parser.add_argument(
        "--smooth-min-improvement",
        type=float,
        default=0.0,
        help="Minimum pose-path cost improvement required to accept a smoothing shortcut",
    )
    parser.add_argument(
        "--joint-continuity-threshold",
        type=float,
        default=DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
        help="Maximum allowed joint delta between neighboring Stage 2/3 configurations, in radians",
    )
    parser.add_argument(
        "--use-angle-normalization",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_ANGLE_NORMALIZATION,
        help="Wrap joint angles into the principal range before IK propagation and continuity checks",
    )
    parser.add_argument(
        "--floating-collision",
        action="store_true",
        help="Enable floating-bar collision in Stage 1; Stage 3 always enables robot collision checking",
    )
    parser.add_argument(
        "--lock-renderer-during-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lock the PyBullet renderer while the tree is being expanded, then show the result afterward",
    )
    parser.add_argument(
        "--draw-rrt-tree",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw RRT tree edges in the PyBullet GUI while planning (use --no-draw-rrt-tree to disable). Only effective when GUI is on.",
    )
    parser.add_argument("--include-built-bars", action="store_true", help="Import already-built bars into the scene")
    parser.add_argument(
        "--enable-built-bar-collision",
        action="store_true",
        help="Enable collision on imported built bars",
    )
    parser.add_argument("--video-frame-step", type=int, default=1, help="Record every Nth waypoint into batch trajectory videos")
    parser.add_argument("--video-frame-sleep", type=float, default=0.02, help="Replay frame interval used to derive batch video FPS")
    parser.add_argument(
        "--gdrive-state",
        action="store_true",
        help=("Use the gdrive dataset convention. --targets are state filenames "
              "(e.g. 'B3_approach.json' or 'B3_approach') under "
              "GDRIVE_DATA_DIRECTORY/<gdrive-problem>/RobotCellStates/."),
    )
    parser.add_argument(
        "--gdrive-bar-action",
        action="store_true",
        help=("Use gdrive BarAction inputs. --targets are BarAction filenames "
              "(e.g. 'B1.json') under GDRIVE_DATA_DIRECTORY/<gdrive-problem>/BarActions/."),
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=None,
        help="Target filenames (bare or .json). Default = [B3_approach.json] for --gdrive-state or [B1.json] for --gdrive-bar-action.",
    )
    parser.add_argument(
        "--movement",
        type=str,
        default="M1",
        help="Movement selector for --gdrive-bar-action: index string, exact id, or substring (default: M1).",
    )
    parser.add_argument(
        "--gdrive-problem", type=str, default=GDRIVE_DEFAULT_PROBLEM,
        help=(f"Dataset directory under GDRIVE_DATA_DIRECTORY (default {GDRIVE_DEFAULT_PROBLEM!r})."),
    )
    parser.add_argument(
        "--gdrive-no-env", action="store_true",
        help="When using --gdrive-state, skip loading env_* bodies as static obstacles.",
    )
    parser.add_argument(
        "--gdrive-no-active-extras", action="store_true",
        help="When using --gdrive-state, skip loading active_* sibling bodies (joints) as static.",
    )
    args = parser.parse_args()

    if args.bidirectional:
        # Swap plan_pose_rrt for plan_pose_birrt at module level. run_stage_trial
        # imports plan_pose_rrt from .core at module load; mutating the binding
        # here is the smallest change that lets the rest of the pipeline (start
        # retries, smoothing, validation) compose unchanged.
        from .core import plan_pose_birrt as _plan_pose_birrt
        import husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.run as _run_mod
        import husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.core as _core_mod
        _run_mod.plan_pose_rrt = _plan_pose_birrt
        _core_mod.plan_pose_rrt = _plan_pose_birrt
        logger.info("Bidirectional RRT-Connect (plan_pose_birrt) enabled.")

    if args.gdrive_state and args.gdrive_bar_action:
        raise ValueError("--gdrive-state and --gdrive-bar-action are mutually exclusive.")
    if not (args.gdrive_state or args.gdrive_bar_action):
        raise ValueError("Exactly one of --gdrive-state / --gdrive-bar-action is required.")
    if args.targets is None:
        args.targets = ["B1.json"] if args.gdrive_bar_action else ["B3_approach.json"]
    args.targets = [t if t.endswith(".json") else f"{t}.json" for t in args.targets]
    args.design_root = os.path.join(GDRIVE_DATA_DIRECTORY, args.gdrive_problem)
    if isinstance(args.movement, str) and args.movement.isdigit():
        args.movement = int(args.movement)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(support_dir(), f"dual_arm_task_space_rrt_{timestamp}.json")
    report_path = os.path.join(reports_dir(), f"dual_arm_task_space_rrt_report_{timestamp}.md")

    summaries: List[Dict[str, Any]] = []
    for target_arg in args.targets:
        full_name, scene_spec = _build_target_spec(target_arg, args)
        debug_tree_out: Dict = {}
        result = run_stage_trial(
            stage=args.stage,
            scene_spec=scene_spec,
            use_gui=args.gui,
            dist_metric=args.dist_metric,
            goal_bias=args.goal_bias,
            position_res=args.position_res,
            rotation_res=args.rotation_res,
            max_time=args.max_time,
            max_iterations=args.max_iterations,
            max_attempts=args.max_attempts,
            endpoint_ik_attempts=args.endpoint_ik_attempts,
            random_seed=args.random_seed,
            enable_collision=args.floating_collision,
            enable_smoothing=args.smoothing,
            smooth_max_iterations=args.smooth_iterations,
            smooth_max_time=args.smooth_max_time,
            smooth_min_cost_improvement=args.smooth_min_improvement,
            joint_continuity_threshold_rad=args.joint_continuity_threshold,
            use_angle_normalization=args.use_angle_normalization,
            start_retries=args.start_retries,
            lock_renderer_during_search=args.lock_renderer_during_search,
            draw_rrt_tree=args.draw_rrt_tree,
            validation_reports_dir=support_dir(),
            debug_tree_out=debug_tree_out,
        )

        # spec dict used by summarize_result / save_replay_bundle; mirrors the
        # earlier real_state_study.spec shape (target_name, goal_pose, grasp_targets,
        # active_bar_mesh, built_bars).
        spec_for_save = {
            "target_name": full_name,
            "world_from_bar_goal": scene_spec["world_from_bar_goal"],
            "grasp_targets": scene_spec["grasp_targets"],
            "active_bar_mesh": scene_spec.get("active_bar_mesh"),
            "built_bars": scene_spec.get("built_bars", []),
        }
        summary = summarize_result(full_name, spec_for_save, result)
        if result.get("path") is not None and result.get("path_confs") is not None:
            trajectory_json_path = os.path.join(
                support_dir(),
                f"dual_arm_task_space_rrt_stage{args.stage}_{full_name}_{timestamp}_trajectory.json",
            )
            trajectory_metadata_json_path = os.path.join(
                support_dir(),
                f"dual_arm_task_space_rrt_stage{args.stage}_{full_name}_{timestamp}_trajectory_metadata.json",
            )
            save_replay_bundle(
                scene=result["scene"],
                spec=spec_for_save,
                joint_path=result["path_confs"],
                pose_path=result["path"],
                trajectory_json_path=trajectory_json_path,
                metadata_json_path=trajectory_metadata_json_path,
            )
            replay_command = build_replay_command(
                trajectory_json_path=trajectory_json_path,
                metadata_json_path=trajectory_metadata_json_path,
            )
            summary["trajectory_json"] = trajectory_json_path
            summary["trajectory_metadata_json"] = trajectory_metadata_json_path
            summary["replay_command"] = replay_command
            logger.info("Saved target %s replay trajectory: %s", full_name, trajectory_json_path)
        if (
            not args.gui
            and result.get("path") is not None
            and result.get("path_confs") is not None
        ):
            video_path = record_trajectory_video(
                scene=result["scene"],
                joint_path=result["path_confs"],
                pose_path=result["path"],
                out_path=os.path.join(
                    support_dir(),
                    f"dual_arm_task_space_rrt_stage{args.stage}_{full_name}_{timestamp}_trajectory.mp4",
                ),
                frame_step=args.video_frame_step,
                frame_sleep=args.video_frame_sleep,
            )
            summary["video_mp4"] = video_path
            if video_path is not None:
                logger.info("Saved target %s trajectory video: %s", full_name, video_path)
        summaries.append(summary)
        logger.info(
            "Target %s -> success=%s runtime=%.3fs waypoints=%d",
            full_name,
            summaries[-1]["success"],
            summaries[-1]["runtime_s"],
            summaries[-1]["waypoints"],
        )
        if args.gui and args.visualize_path:
            run_visualization_loop(
                result["scene"]["bar_body"],
                result["path"],
                result["scene"]["cid"],
                robot=result["scene"]["robot"],
                arm_joints=result["scene"]["arm_joints"],
                path_confs=result["path_confs"],
                validation=result.get("validation"),
                scene=result.get("scene"),
            )
        teardown_planning_scene()

    payload = {
        "design_root": args.design_root,
        "targets": args.targets,
        "stage": args.stage,
        "mode": f"stage{args.stage}_planning",
        "results": summaries,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    write_report(report_path, os.path.relpath(json_path, reports_dir()), args, summaries)
    logger.info("Saved dual-arm task-space RRT report: %s", report_path)
    logger.info("Saved dual-arm task-space RRT JSON: %s", json_path)


if __name__ == "__main__":
    main()
