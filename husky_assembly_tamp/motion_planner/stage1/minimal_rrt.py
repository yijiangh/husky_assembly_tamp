"""
Minimal Stage 1/2/3 floating-bar RRT.

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
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet
import pybullet_planning as pp
from compas_fab.robots import RobotSemantics
from compas_robots import RobotModel
from pybullet_planning.motion_planners.rrt import TreeNode, configs
from pybullet_planning.interfaces.geometry.mesh import Mesh, create_mesh

from husky_assembly_tamp.motion_planner.stage1.path_validation import validate_stage_trajectory
from husky_assembly_tamp.utils.params import DATA_DIR
from husky_assembly_tamp.utils.util import calculate_pose_error, normalize_angles, setup_logger


logger = setup_logger("stage1_minimal_rrt")

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
TOOL_LINK_LEFT = "left_ur_arm_tool0"
TOOL_LINK_RIGHT = "right_ur_arm_tool0"
STAGE3_GRASP_MASK_LINKS = [
    "left_ur_arm_wrist_3_link",
    "right_ur_arm_wrist_3_link",
    TOOL_LINK_LEFT,
    TOOL_LINK_RIGHT,
]
INIT_ARM_JOINT_ANGLES = np.array([0.0, -np.pi / 2.0, 0.0, 0.0, 0.0, 0.0] * 2, dtype=float)
BAR_RADIUS = 0.015
BAR_LENGTH = 1.0
BAR_BOX_DIMS = (2.0 * BAR_RADIUS, 2.0 * BAR_RADIUS, BAR_LENGTH)
STAGE1_DEBUG_START_OFFSET = np.array([-0.5, 0.0, 0.5], dtype=float)
DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD = 10.0 * np.pi / 180.0
DEFAULT_USE_ANGLE_NORMALIZATION = True
DEFAULT_HOME_LEFT_TOOL_Z_OFFSET = 0.2
MOBILE_BASE_FROM_TOOL0_LEFT_HOME: PoseLike = (
    np.array([0.3974141597747803, 0.16023626923561096, 0.8621799349784851], dtype=float),
    np.array([-0.5000003576278687, 0.4999987483024597, -0.499999463558197, 0.5000012516975403], dtype=float),
    # np.array([0.4999987483024597, 0.5000003576278687, 0.5000012516975403, 0.499999463558197], dtype=float)
)
DESIGN_STUDY_BAR_SEQUENCE = [
    "G1",
    "G2",
    "G3",
    "G4",
    "V1",
    "V1-G1",
    "V1-G2",
    "V2",
    "V2-G1",
    "V2-G2",
    "H1",
    "D1",
    "V3",
]
DESIGN_STUDY_BAR_NAME_TO_INDEX = {name: idx for idx, name in enumerate(DESIGN_STUDY_BAR_SEQUENCE)}


PoseLike = Tuple[np.ndarray, np.ndarray]
GraspTarget = Tuple[PoseLike, PoseLike]
ArmConf = np.ndarray
FullConf = np.ndarray
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


def maybe_normalize_angles(values: Sequence[float] | np.ndarray, use_angle_normalization: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if use_angle_normalization:
        return np.asarray(normalize_angles(arr), dtype=float)
    return arr


def load_grasp_targets(
    json_path: str,
    world_from_mobile_base: Optional[PoseLike] = None,
    swap_grasps: bool = False,
) -> List[GraspTarget]:
    with open(json_path) as f:
        raw = json.load(f)
    targets = []
    mobile_base_from_world = pp.invert(world_from_mobile_base) if world_from_mobile_base is not None else None
    for item in raw:
        d = item["data"]
        world_from_bar = pp.pose_from_tform(np.array(d["world_from_bar"]["data"]["matrix"]))
        world_from_tool0 = pp.pose_from_tform(np.array(d["world_from_tool0"]["data"]["matrix"]))
        if mobile_base_from_world is not None:
            world_from_bar = pp.multiply(mobile_base_from_world, world_from_bar)
            world_from_tool0 = pp.multiply(mobile_base_from_world, world_from_tool0)
        targets.append((world_from_bar, world_from_tool0))
    if swap_grasps and len(targets) >= 2:
        targets[0], targets[1] = targets[1], targets[0]
    return targets


def load_robot_cell_state_data(json_path: str) -> Dict[str, Any]:
    with open(json_path) as f:
        data = json.load(f)
    state = data["data"]
    return {
        "joint_values": np.asarray(state["robot_configuration"]["data"]["joint_values"], dtype=float),
        "world_from_mobile_base": frame_data_to_pose(state["robot_base_frame"]),
    }


def load_robot_cell_state(json_path: str) -> np.ndarray:
    return load_robot_cell_state_data(json_path)["joint_values"]


def get_bar_feature_points(bar_box_dims: Sequence[float] = BAR_BOX_DIMS) -> List[np.ndarray]:
    half_width, half_depth, half_length = 0.5 * np.asarray(bar_box_dims, dtype=float)
    return [
        np.array([sx * half_width, sy * half_depth, sz * half_length], dtype=float)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]


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


def design_study_active_bar_body_name(target_name: str) -> str:
    if target_name not in DESIGN_STUDY_BAR_NAME_TO_INDEX:
        raise KeyError(f"Unknown design-study bar target: {target_name}")
    return f"b{DESIGN_STUDY_BAR_NAME_TO_INDEX[target_name]}_0"


def load_design_study_bar_mesh(robot_cell_json: str, target_name: str) -> BarMeshSpec:
    active_bar_body_name = design_study_active_bar_body_name(target_name)
    with open(robot_cell_json) as f:
        robot_cell = json.load(f)
    rigid_body_models = robot_cell["data"]["rigid_body_models"]
    if active_bar_body_name not in rigid_body_models:
        raise KeyError(f"Bar {active_bar_body_name} not found in {robot_cell_json}")
    collision_meshes = rigid_body_models[active_bar_body_name]["collision_meshes"]
    if not collision_meshes:
        raise ValueError(f"Bar {active_bar_body_name} has no collision meshes in {robot_cell_json}")
    vertices, faces = compas_mesh_data_to_pybullet_mesh(collision_meshes[0])
    return {
        "name": target_name,
        "body_name": active_bar_body_name,
        "vertices": vertices,
        "faces": faces,
        "aabb_dims": mesh_vertices_aabb_dims(vertices),
    }


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


def auto_compute_home_bar_pose(
    grasp_targets: Sequence[GraspTarget],
    mobile_base_from_tool0_left: PoseLike = MOBILE_BASE_FROM_TOOL0_LEFT_HOME,
    forward_direction: np.ndarray = np.array([1.0, 0.0, 0.0]),
    ik_validator: Optional[Callable[[PoseLike], bool]] = None,
    num_geometric_candidates: int = 20,
) -> Dict[str, Any]:
    """Auto-compute the home bar pose by optimizing bar-axis rotation and EE-axis flip."""
    if len(grasp_targets) < 2:
        raise ValueError("Expected two grasp targets to auto-compute the home bar pose.")

    mobile_base_from_bar_left, mobile_base_from_tool0_left_goal = grasp_targets[0]
    mobile_base_from_bar_right, mobile_base_from_tool0_right_goal = grasp_targets[1]
    bar_from_tool0_left = pp.multiply(pp.invert(mobile_base_from_bar_left), mobile_base_from_tool0_left_goal)
    tool0_left_from_bar = pp.invert(bar_from_tool0_left)
    bar_from_tool0_right = pp.multiply(pp.invert(mobile_base_from_bar_right), mobile_base_from_tool0_right_goal)

    forward = np.asarray(forward_direction, dtype=float)
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-9:
        raise ValueError("forward_direction must be non-zero.")
    forward = forward / forward_norm

    all_candidates: List[Tuple[float, float, float, PoseLike]] = []
    for flip_yaw in (0.0, np.pi):
        adjusted_left = pp.multiply(
            mobile_base_from_tool0_left,
            pp.Pose(euler=pp.Euler(yaw=flip_yaw)),
        )
        bar_base = pp.multiply(adjusted_left, tool0_left_from_bar)

        for theta in np.linspace(-np.pi, np.pi, 360, endpoint=False):
            bar_rotated = pp.multiply(bar_base, pp.Pose(euler=pp.Euler(yaw=float(theta))))
            left_ee = pp.multiply(bar_rotated, bar_from_tool0_left)
            right_ee = pp.multiply(bar_rotated, bar_from_tool0_right)

            left_z = np.asarray(pp.tform_from_pose(left_ee), dtype=float)[:3, 2]
            right_z = np.asarray(pp.tform_from_pose(right_ee), dtype=float)[:3, 2]
            avg_z = left_z + right_z
            avg_z_norm = np.linalg.norm(avg_z)
            if avg_z_norm < 1e-9:
                continue
            avg_z = avg_z / avg_z_norm

            score = float(np.dot(avg_z, forward))
            all_candidates.append((score, float(flip_yaw), float(theta), bar_rotated))

    if not all_candidates:
        raise ValueError("Could not generate any home bar pose candidates.")

    all_candidates.sort(key=lambda candidate: -candidate[0])
    chosen = all_candidates[0]

    if ik_validator is not None and num_geometric_candidates > 0:
        top_candidates = all_candidates[:num_geometric_candidates]
        for candidate in top_candidates:
            _, _, _, bar_pose = candidate
            if ik_validator(bar_pose):
                chosen = candidate
                break
        else:
            logger.warning(
                "No IK-feasible candidate found among top %d geometric candidates; falling back to best geometric candidate.",
                num_geometric_candidates,
            )

    best_score, best_flip, best_theta, _ = chosen
    adjusted_left_final = pp.multiply(
        mobile_base_from_tool0_left,
        pp.Pose(euler=pp.Euler(yaw=best_flip)),
    )
    bar_final = pp.multiply(
        pp.multiply(adjusted_left_final, tool0_left_from_bar),
        pp.Pose(euler=pp.Euler(yaw=best_theta)),
    )
    right_tool_final = pp.multiply(bar_final, bar_from_tool0_right)

    return {
        "mobile_base_from_tool0_left_start": adjusted_left_final,
        "mobile_base_from_bar_start": bar_final,
        "mobile_base_from_tool0_right_start": right_tool_final,
        "tool0_left_from_bar": tool0_left_from_bar,
        "bar_from_tool0_right": bar_from_tool0_right,
        "chosen_flip_yaw": best_flip,
        "chosen_bar_axis_theta": best_theta,
        "alignment_score": best_score,
    }


def build_real_design_goal_spec(
    design_root: str,
    target_name: str,
    robot_cell_json: Optional[str] = None,
    include_built_bars: bool = False,
    enable_built_bar_collision: bool = False,
    swap_grasps: bool = False,
) -> Dict[str, Any]:
    design_root_path = Path(design_root)
    robot_cell_states_dir = design_root_path / "RobotCellStates"
    robot_cell_json_path = Path(robot_cell_json) if robot_cell_json is not None else design_root_path / "RobotCell.json"
    state_json = robot_cell_states_dir / f"{target_name}_RobotCellState.json"
    grasp_json = robot_cell_states_dir / f"{target_name}_GraspTargets.json"
    if not state_json.is_file():
        raise FileNotFoundError(f"RobotCellState not found for {target_name}: {state_json}")
    if not grasp_json.is_file():
        raise FileNotFoundError(f"GraspTargets not found for {target_name}: {grasp_json}")
    state_data = load_robot_cell_state_data(str(state_json))
    grasp_targets = load_grasp_targets(
        str(grasp_json),
        world_from_mobile_base=state_data["world_from_mobile_base"],
        swap_grasps=swap_grasps,
    )
    built_bars: List[Dict[str, Any]] = []
    if include_built_bars:
        target_index = DESIGN_STUDY_BAR_NAME_TO_INDEX[target_name]
        for prior_name in DESIGN_STUDY_BAR_SEQUENCE[:target_index]:
            prior_grasp_json = robot_cell_states_dir / f"{prior_name}_GraspTargets.json"
            prior_state_json = robot_cell_states_dir / f"{prior_name}_RobotCellState.json"
            if not prior_grasp_json.is_file() or not prior_state_json.is_file():
                logger.info("Skipping built-bar import for %s because pose files are unavailable.", prior_name)
                continue
            prior_state_data = load_robot_cell_state_data(str(prior_state_json))
            prior_targets = load_grasp_targets(
                str(prior_grasp_json),
                world_from_mobile_base=prior_state_data["world_from_mobile_base"],
                swap_grasps=swap_grasps,
            )
            built_bars.append(
                {
                    "name": prior_name,
                    "mesh": load_design_study_bar_mesh(str(robot_cell_json_path), prior_name),
                    "pose": get_goal_pose_from_grasp_targets(prior_targets),
                    "collision": bool(enable_built_bar_collision),
                    "color": (0.45, 0.45, 0.45, 0.95),
                }
            )
    return {
        "target_name": target_name,
        "state_json": str(state_json),
        "grasp_json": str(grasp_json),
        "robot_state": state_data,
        "grasp_targets": grasp_targets,
        "goal_pose": get_goal_pose_from_grasp_targets(grasp_targets),
        "active_bar_mesh": load_design_study_bar_mesh(str(robot_cell_json_path), target_name),
        "built_bars": built_bars,
    }


def pose_to_feature_vec(pose: PoseLike, feature_points: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    if not feature_points:
        return None
    pts = []
    for p_local in feature_points:
        p_world, _ = pp.multiply(pose, (p_local, [0, 0, 0, 1]))
        pts.append(np.asarray(p_world, dtype=float))
    return np.concatenate(pts, axis=0)


def pose_distance(pose1: PoseLike, pose2: PoseLike, dist_metric: str, feature_points: Sequence[np.ndarray]) -> float:
    pos1, quat1 = pose1
    pos2, quat2 = pose2
    if dist_metric == "feature":
        vec1 = pose_to_feature_vec((pos1, quat1), feature_points)
        vec2 = pose_to_feature_vec((pos2, quat2), feature_points)
        if vec1 is not None and vec2 is not None:
            return float(np.linalg.norm(vec2 - vec1))
    dx = pos2 - pos1
    rot_dist = pp.quat_angle_between(quat1, quat2)
    return float(np.linalg.norm(np.array([dx[0], dx[1], dx[2], rot_dist], dtype=float)))


def _pose_path_cost(path_poses: Sequence[PoseLike], dist_metric: str, feature_points: Sequence[np.ndarray]) -> float:
    if len(path_poses) < 2:
        return 0.0
    return float(
        sum(
            pose_distance(path_poses[idx], path_poses[idx + 1], dist_metric, feature_points)
            for idx in range(len(path_poses) - 1)
        )
    )


def _pose_path_inflection_indices(
    path_poses: Sequence[PoseLike],
    feature_points: Sequence[np.ndarray],
    tolerance: float = 1e-3,
) -> List[int]:
    """Return dense-path indices that act like geometric control points."""
    if not path_poses:
        return []
    if len(path_poses) <= 2:
        return list(range(len(path_poses)))
    feature_vecs = [pose_to_feature_vec(pose, feature_points) for pose in path_poses]
    if any(vec is None for vec in feature_vecs):
        return list(range(len(path_poses)))

    indices = [0]
    anchor_idx = 0
    last_direction: Optional[np.ndarray] = None
    for idx in range(1, len(feature_vecs)):
        anchor_vec = np.asarray(feature_vecs[anchor_idx], dtype=float)
        current_vec = np.asarray(feature_vecs[idx], dtype=float)
        delta = current_vec - anchor_vec
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm <= tolerance:
            continue
        direction = delta / delta_norm
        if last_direction is None:
            last_direction = direction
            continue
        if float(np.linalg.norm(direction - last_direction)) > tolerance:
            waypoint_idx = idx - 1
            if waypoint_idx > indices[-1]:
                indices.append(waypoint_idx)
            anchor_idx = waypoint_idx
            anchor_vec = np.asarray(feature_vecs[anchor_idx], dtype=float)
            delta = current_vec - anchor_vec
            delta_norm = float(np.linalg.norm(delta))
            last_direction = None if delta_norm <= tolerance else (delta / delta_norm)
    if indices[-1] != len(path_poses) - 1:
        indices.append(len(path_poses) - 1)
    return indices


def sample_pose(
    robot: int,
    goal_pose: PoseLike,
    rng: np.random.Generator,
    goal_sample_prob: float,
    workspace_xy: float,
    workspace_z: float,
) -> Tuple[PoseLike, bool]:
    if rng.random() < goal_sample_prob:
        return goal_pose, True
    base_pos, _ = pp.get_pose(robot)
    cx, cy, cz = np.asarray(base_pos, dtype=float)
    x = cx + rng.uniform(-workspace_xy / 2.0, workspace_xy / 2.0)
    y = cy + rng.uniform(-workspace_xy / 2.0, workspace_xy / 2.0)
    z_min = max(0.05, cz)
    z = rng.uniform(z_min, z_min + workspace_z)
    roll = rng.uniform(-np.pi, np.pi)
    pitch = rng.uniform(-np.pi, np.pi)
    yaw = rng.uniform(-np.pi, np.pi)
    return pp.Pose(point=[x, y, z], euler=pp.Euler(roll, pitch, yaw)), False


def nearest_node(
    nodes: List[TreeNode],
    target_pose: PoseLike,
    dist_metric: str,
    feature_points: Sequence[np.ndarray],
    feature_vecs: Dict[int, np.ndarray],
) -> TreeNode:
    if dist_metric == "feature":
        target_vec = pose_to_feature_vec(target_pose, feature_points)
        if target_vec is not None:
            return min(
                nodes,
                key=lambda node: float(np.linalg.norm(feature_vecs[id(node)] - target_vec)),
            )
    return min(nodes, key=lambda node: pose_distance(node.config, target_pose, dist_metric, feature_points))


def export_tree(nodes: List[TreeNode]) -> Dict[str, List[List[float]]]:
    id_to_idx: Dict[int, int] = {}
    points: List[List[float]] = []
    for node in nodes:
        idx = len(points)
        id_to_idx[id(node)] = idx
        pos = np.asarray(node.config[0], dtype=float).reshape(3)
        points.append([float(pos[0]), float(pos[1]), float(pos[2])])
    edges: List[List[int]] = []
    for node in nodes:
        if node.parent is None:
            continue
        pid = id(node.parent)
        cid = id(node)
        if pid in id_to_idx and cid in id_to_idx:
            edges.append([id_to_idx[pid], id_to_idx[cid]])
    return {"points": points, "edges": edges}


def get_pose_collision_fn(bar_body: int, obstacle_bodies: Sequence[int], enable_collision: bool) -> Callable[[PoseLike], bool]:
    if not enable_collision:
        return lambda pose: False
    floating_collision_fn = pp.get_floating_body_collision_fn(
        bar_body,
        obstacles=list(obstacle_bodies),
        disabled_collisions=[],
    )
    return lambda pose: bool(floating_collision_fn(pose))


def get_disabled_collisions_from_link_names(
    robot: int,
    link_name_pairs: Sequence[Tuple[str, str]],
) -> List[Tuple[int, int]]:
    disabled_pairs: List[Tuple[int, int]] = []
    for link1_name, link2_name in link_name_pairs:
        if not (pp.has_link(robot, link1_name) and pp.has_link(robot, link2_name)):
            continue
        disabled_pairs.append((pp.link_from_name(robot, link1_name), pp.link_from_name(robot, link2_name)))
    return disabled_pairs


def get_joint_collision_fn(
    robot: int,
    arm_joints: Sequence[int],
    obstacle_bodies: Sequence[int],
    tool_link_left: int,
    bar_body: int,
    grasp_bar_from_left: PoseLike,
) -> Callable[..., bool]:
    robot_model = RobotModel.from_urdf_file(HUSKY_DUAL_URDF_PATH)
    semantics = RobotSemantics.from_srdf_file(HUSKY_DUAL_SRDF_PATH, robot_model)
    disabled_collisions = get_disabled_collisions_from_link_names(robot, semantics.disabled_collisions)
    attachment = pp.Attachment(robot, tool_link_left, pp.invert(grasp_bar_from_left), bar_body)
    extra_disabled_collisions = []
    for link_name in STAGE3_GRASP_MASK_LINKS:
        if not pp.has_link(robot, link_name):
            continue
        link = pp.link_from_name(robot, link_name)
        extra_disabled_collisions.append(((robot, link), (bar_body, pp.BASE_LINK)))
    collision_fn = pp.get_collision_fn(
        robot,
        arm_joints,
        obstacles=list(obstacle_bodies),
        attachments=[attachment],
        self_collisions=True,
        disabled_collisions=disabled_collisions,
        extra_disabled_collisions=extra_disabled_collisions,
        max_distance=0.0,
    )

    return collision_fn


def add_profile_time(profile_out: Optional[Dict[str, Any]], key: str, dt: float) -> None:
    if profile_out is None:
        return
    profile_out[key] = float(profile_out.get(key, 0.0)) + float(dt)


def bump_profile_count(profile_out: Optional[Dict[str, Any]], key: str, inc: int = 1) -> None:
    if profile_out is None:
        return
    profile_out[key] = int(profile_out.get(key, 0)) + int(inc)


def goal_pose_reached(pose: PoseLike, goal_pose: PoseLike, position_res: float, rotation_res: float) -> bool:
    return bool(
        pp.is_pose_close(
            pose,
            goal_pose,
            pos_tolerance=max(position_res, 1e-6),
            ori_tolerance=max(rotation_res, 1e-6),
        )
    )


def solve_single_arm_ik(
    robot: int,
    arm_joints: Sequence[int],
    tool_link: int,
    full_seed_conf: FullConf,
    target_tool_pose: PoseLike,
    arm_slice: slice,
    use_angle_normalization: bool = DEFAULT_USE_ANGLE_NORMALIZATION,
) -> Optional[FullConf]:
    seed_conf = np.asarray(full_seed_conf, dtype=float)
    pp.set_joint_positions(robot, arm_joints, seed_conf)
    result = np.asarray(
        pybullet.calculateInverseKinematics(
            robot,
            tool_link,
            target_tool_pose[0],
            target_tool_pose[1],
            maxNumIterations=1000,
            residualThreshold=1e-6,
        ),
        dtype=float,
    )
    if result.shape[0] < max(arm_slice.stop, len(seed_conf)):
        return None
    solved_conf = seed_conf.copy()
    solved_conf[arm_slice] = maybe_normalize_angles(result[arm_slice], use_angle_normalization)
    pp.set_joint_positions(robot, arm_joints, solved_conf)
    pose_res = pp.get_link_pose(robot, tool_link)
    pose_err = calculate_pose_error(target_tool_pose, pose_res)
    if np.linalg.norm(pose_err) > 1e-4:
        return None
    return maybe_normalize_angles(solved_conf, use_angle_normalization)


def validate_dual_arm_bar_pose(
    robot: int,
    arm_joints: Sequence[int],
    tool_link_left: int,
    tool_link_right: int,
    full_conf: FullConf,
    bar_pose: PoseLike,
    grasp_bar_from_left: PoseLike,
    grasp_bar_from_right: PoseLike,
) -> bool:
    pp.set_joint_positions(robot, arm_joints, full_conf)
    target_left = pp.multiply(bar_pose, grasp_bar_from_left)
    target_right = pp.multiply(bar_pose, grasp_bar_from_right)
    world_from_left = pp.get_link_pose(robot, tool_link_left)
    world_from_right = pp.get_link_pose(robot, tool_link_right)
    if not pp.is_pose_close(target_left, world_from_left, pos_tolerance=1e-4, ori_tolerance=1e-4):
        return False
    if not pp.is_pose_close(target_right, world_from_right, pos_tolerance=1e-4, ori_tolerance=1e-4):
        return False
    bar_from_left = pp.invert(grasp_bar_from_left)
    bar_from_right = pp.invert(grasp_bar_from_right)
    left_bar_pose = pp.multiply(world_from_left, bar_from_left)
    right_bar_pose = pp.multiply(world_from_right, bar_from_right)
    return bool(
        pp.is_pose_close(left_bar_pose, bar_pose, pos_tolerance=1e-4, ori_tolerance=1e-4)
        and pp.is_pose_close(right_bar_pose, bar_pose, pos_tolerance=1e-4, ori_tolerance=1e-4)
        and pp.is_pose_close(left_bar_pose, right_bar_pose, pos_tolerance=1e-4, ori_tolerance=1e-4)
    )


def solve_dual_arm_pose_ik(
    robot: int,
    arm_joints: Sequence[int],
    tool_link_left: int,
    tool_link_right: int,
    bar_pose: PoseLike,
    grasp_bar_from_left: PoseLike,
    grasp_bar_from_right: PoseLike,
    seed_conf: FullConf,
    use_angle_normalization: bool = DEFAULT_USE_ANGLE_NORMALIZATION,
    profile_out: Optional[Dict[str, Any]] = None,
) -> Optional[FullConf]:
    target_left = pp.multiply(bar_pose, grasp_bar_from_left)
    target_right = pp.multiply(bar_pose, grasp_bar_from_right)
    seed_conf = maybe_normalize_angles(seed_conf, use_angle_normalization)
    t0 = time.perf_counter()
    attempts = (
        ("right", "left"),
        ("left", "right"),
    )
    for order in attempts:
        conf = seed_conf.copy()
        success = True
        for arm_name in order:
            bump_profile_count(profile_out, "ik_calls")
            if arm_name == "right":
                conf_next = solve_single_arm_ik(
                    robot=robot,
                    arm_joints=arm_joints,
                    tool_link=tool_link_right,
                    full_seed_conf=conf,
                    target_tool_pose=target_right,
                    arm_slice=slice(6, 12),
                    use_angle_normalization=use_angle_normalization,
                )
            else:
                conf_next = solve_single_arm_ik(
                    robot=robot,
                    arm_joints=arm_joints,
                    tool_link=tool_link_left,
                    full_seed_conf=conf,
                    target_tool_pose=target_left,
                    arm_slice=slice(0, 6),
                    use_angle_normalization=use_angle_normalization,
                )
            if conf_next is None:
                success = False
                bump_profile_count(profile_out, "ik_failures")
                break
            conf = conf_next
        if success and validate_dual_arm_bar_pose(
            robot=robot,
            arm_joints=arm_joints,
            tool_link_left=tool_link_left,
            tool_link_right=tool_link_right,
            full_conf=conf,
            bar_pose=bar_pose,
            grasp_bar_from_left=grasp_bar_from_left,
            grasp_bar_from_right=grasp_bar_from_right,
        ):
            add_profile_time(profile_out, "ik_time_s", time.perf_counter() - t0)
            bump_profile_count(profile_out, "nodes_with_ik")
            return maybe_normalize_angles(conf, use_angle_normalization)
    add_profile_time(profile_out, "ik_time_s", time.perf_counter() - t0)
    return None


def solve_endpoint_dual_arm_ik(
    robot: int,
    arm_joints: Sequence[int],
    tool_link_left: int,
    tool_link_right: int,
    bar_pose: PoseLike,
    grasp_bar_from_left: PoseLike,
    grasp_bar_from_right: PoseLike,
    seed_conf: FullConf,
    rng: np.random.Generator,
    max_attempts: int,
    use_angle_normalization: bool = DEFAULT_USE_ANGLE_NORMALIZATION,
    profile_out: Optional[Dict[str, Any]] = None,
) -> Optional[FullConf]:
    for attempt in range(max(1, max_attempts)):
        if attempt == 0:
            attempt_seed = np.asarray(seed_conf, dtype=float)
        else:
            attempt_seed = rng.uniform(-np.pi, np.pi, len(seed_conf))
        conf = solve_dual_arm_pose_ik(
            robot=robot,
            arm_joints=arm_joints,
            tool_link_left=tool_link_left,
            tool_link_right=tool_link_right,
            bar_pose=bar_pose,
            grasp_bar_from_left=grasp_bar_from_left,
            grasp_bar_from_right=grasp_bar_from_right,
            seed_conf=attempt_seed,
            use_angle_normalization=use_angle_normalization,
            profile_out=profile_out,
        )
        if conf is not None:
            return conf
    return None


def extend_toward(
    nodes: List[TreeNode],
    source: TreeNode,
    target_pose: PoseLike,
    collision_fn: Callable[[PoseLike], bool],
    joint_collision_fn: Optional[Callable[[FullConf], bool]],
    draw_color: Tuple[float, float, float, float],
    use_draw: bool,
    position_res: float,
    rotation_res: float,
    dist_metric: str,
    feature_points: Sequence[np.ndarray],
    feature_vecs: Dict[int, np.ndarray],
    enable_ik: bool = False,
    node_confs: Optional[Dict[int, FullConf]] = None,
    ik_context: Optional[Dict[str, Any]] = None,
    joint_continuity_threshold_rad: Optional[float] = None,
    use_angle_normalization: bool = DEFAULT_USE_ANGLE_NORMALIZATION,
    profile_out: Optional[Dict[str, Any]] = None,
) -> Tuple[TreeNode, bool, str]:
    current = source
    reached = True
    stop_reason = "reached"
    current_conf = None if node_confs is None else node_confs.get(id(source))
    if enable_ik and (node_confs is None or ik_context is None or current_conf is None):
        return current, False, "ik_failure"
    for pose in list(
        pp.interpolate_poses(
            source.config,
            target_pose,
            pos_step_size=max(position_res, 1e-6),
            ori_step_size=max(rotation_res, 1e-6),
        )
    )[1:]:
        bump_profile_count(profile_out, "poses_checked")
        next_conf = None
        if enable_ik:
            next_conf = solve_dual_arm_pose_ik(
                robot=ik_context["robot"],
                arm_joints=ik_context["arm_joints"],
                tool_link_left=ik_context["tool_link_left"],
                tool_link_right=ik_context["tool_link_right"],
                bar_pose=pose,
                grasp_bar_from_left=ik_context["grasp_bar_from_left"],
                grasp_bar_from_right=ik_context["grasp_bar_from_right"],
                seed_conf=current_conf,
                use_angle_normalization=use_angle_normalization,
                profile_out=profile_out,
            )
            if next_conf is None:
                reached = False
                stop_reason = "ik_failure"
                break
            if joint_continuity_threshold_rad is not None:
                step_delta = np.abs(
                    maybe_normalize_angles(
                        np.asarray(next_conf, dtype=float) - np.asarray(current_conf, dtype=float),
                        use_angle_normalization,
                    )
                )
                if float(np.max(step_delta)) > float(joint_continuity_threshold_rad):
                    reached = False
                    stop_reason = "continuity"
                    bump_profile_count(profile_out, "continuity_rejections")
                    break
            if joint_collision_fn is not None and joint_collision_fn(next_conf):
                reached = False
                stop_reason = "collision"
                bump_profile_count(profile_out, "collision_hits")
                break
        if collision_fn(pose):
            reached = False
            stop_reason = "collision"
            bump_profile_count(profile_out, "collision_hits")
            break
        # this is the version of extension that stops at the first collision of the extend, but still add the valid interp so far into the tree
        pp.VideoSaver
        node = TreeNode(pose, parent=current)
        nodes.append(node)
        if enable_ik and node_confs is not None and next_conf is not None:
            node_confs[id(node)] = next_conf
        bump_profile_count(profile_out, "nodes_created")
        if dist_metric == "feature":
            feature_vec = pose_to_feature_vec(pose, feature_points)
            if feature_vec is not None:
                feature_vecs[id(node)] = feature_vec
        if use_draw:
            pp.add_line(current.config[0], node.config[0], width=1.5, color=draw_color)
        current = node
        if next_conf is not None:
            current_conf = next_conf
    return current, reached, stop_reason


def summarize_joint_continuity(
    joint_path: Optional[Sequence[FullConf]],
    threshold_rad: float = DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
    use_angle_normalization: bool = DEFAULT_USE_ANGLE_NORMALIZATION,
) -> Dict[str, Any]:
    summary = {
        "ok": None,
        "max_delta_rad": None,
        "first_bad_step": None,
        "threshold_rad": float(threshold_rad),
    }
    if joint_path is None:
        return summary
    normalized_joint_path = [maybe_normalize_angles(conf, use_angle_normalization) for conf in joint_path]
    if len(normalized_joint_path) < 2:
        summary["ok"] = True
        summary["max_delta_rad"] = 0.0
        return summary

    step_max_deltas = []
    for prev_conf, next_conf in zip(normalized_joint_path[:-1], normalized_joint_path[1:]):
        step_delta = np.abs(
            maybe_normalize_angles(
                np.asarray(next_conf, dtype=float) - np.asarray(prev_conf, dtype=float),
                use_angle_normalization,
            )
        )
        step_max_deltas.append(float(np.max(step_delta)))
    max_delta = max(step_max_deltas) if step_max_deltas else 0.0
    first_bad_step = next((idx + 1 for idx, delta in enumerate(step_max_deltas) if delta > threshold_rad), None)
    summary["ok"] = first_bad_step is None
    summary["max_delta_rad"] = float(max_delta)
    summary["first_bad_step"] = first_bad_step
    return summary


def reconstruct_joint_path_for_pose_path(
    scene: Dict[str, Any],
    pose_path: Sequence[PoseLike],
    start_conf: FullConf,
    joint_collision_fn: Optional[Callable[[FullConf], bool]] = None,
    joint_continuity_threshold_rad: Optional[float] = None,
    use_angle_normalization: bool = DEFAULT_USE_ANGLE_NORMALIZATION,
    profile_out: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[List[FullConf]], Optional[str]]:
    if not pose_path:
        return [], None
    grasp_bar_from_right = scene["grasp_bar_from_right"]
    if grasp_bar_from_right is None:
        return None, "missing_right_grasp"

    current_conf = maybe_normalize_angles(start_conf, use_angle_normalization)
    joint_path = [current_conf]
    for idx, pose in enumerate(pose_path[1:], start=1):
        next_conf = solve_dual_arm_pose_ik(
            robot=scene["robot"],
            arm_joints=scene["arm_joints"],
            tool_link_left=scene["tool_link_left"],
            tool_link_right=scene["tool_link_right"],
            bar_pose=pose,
            grasp_bar_from_left=scene["grasp_bar_from_left"],
            grasp_bar_from_right=grasp_bar_from_right,
            seed_conf=current_conf,
            use_angle_normalization=use_angle_normalization,
            profile_out=profile_out,
        )
        if next_conf is None:
            return None, f"ik_failure_at_waypoint_{idx}"
        next_conf = maybe_normalize_angles(next_conf, use_angle_normalization)
        if joint_continuity_threshold_rad is not None:
            step_delta = np.abs(
                maybe_normalize_angles(
                    np.asarray(next_conf, dtype=float) - np.asarray(current_conf, dtype=float),
                    use_angle_normalization,
                )
            )
            if float(np.max(step_delta)) > float(joint_continuity_threshold_rad):
                bump_profile_count(profile_out, "continuity_rejections")
                return None, f"continuity_at_waypoint_{idx}"
        if joint_collision_fn is not None:
            t_collision = time.perf_counter()
            in_collision = joint_collision_fn(next_conf)
            add_profile_time(profile_out, "collision_check_time_s", time.perf_counter() - t_collision)
            if in_collision:
                bump_profile_count(profile_out, "collision_hits")
                return None, f"collision_at_waypoint_{idx}"
        current_conf = next_conf
        joint_path.append(current_conf)
    return joint_path, None


def smooth_dual_arm_pose_path(
    path_poses: Sequence[PoseLike],
    path_confs: Optional[Sequence[FullConf]],
    *,
    scene: Dict[str, Any],
    pose_collision_fn: Optional[Callable[[PoseLike], bool]] = None,
    joint_collision_fn: Optional[Callable[[FullConf], bool]] = None,
    dist_metric: str = "feature",
    feature_points: Optional[Sequence[np.ndarray]] = None,
    position_res: float = 0.05,
    rotation_res: float = 0.1,
    joint_continuity_threshold_rad: Optional[float] = None,
    use_angle_normalization: bool = DEFAULT_USE_ANGLE_NORMALIZATION,
    max_smooth_iterations: int = 100,
    max_time: float = 10.0,
    min_cost_improvement: float = 0.0,
    inflection_tolerance: float = 1e-3,
    random_seed: Optional[int] = None,
    profile_out: Optional[Dict[str, Any]] = None,
) -> Tuple[List[PoseLike], Optional[List[FullConf]]]:
    feature_points = list(feature_points) if feature_points is not None else get_bar_feature_points()
    current_poses = list(path_poses)
    current_confs = None if path_confs is None else [np.asarray(conf, dtype=float) for conf in path_confs]
    if current_confs is not None and len(current_confs) != len(current_poses):
        raise ValueError("Pose and joint path lengths must match for smoothing.")

    current_cost = _pose_path_cost(current_poses, dist_metric, feature_points)
    if profile_out is not None:
        profile_out.clear()
        profile_out.update(
            {
                "cost_before": float(current_cost),
                "cost_after": float(current_cost),
                "waypoints_before": len(current_poses),
                "waypoints_after": len(current_poses),
                "shortcut_attempts": 0,
                "cost_rejections": 0,
                "ik_failures": 0,
                "continuity_rejections": 0,
                "collision_rejections": 0,
                "accepts": 0,
                "smooth_time_s": 0.0,
            }
        )
    if len(current_poses) < 3 or max_smooth_iterations <= 0 or max_time <= 0.0:
        return current_poses, current_confs

    rng = np.random.default_rng(random_seed)
    start_time = time.perf_counter()
    inflection_indices = _pose_path_inflection_indices(current_poses, feature_points, inflection_tolerance)
    for _ in range(max_smooth_iterations):
        if (time.perf_counter() - start_time) >= max_time:
            break
        if len(inflection_indices) < 3:
            break

        bump_profile_count(profile_out, "shortcut_attempts")
        ii = int(rng.integers(0, len(inflection_indices) - 2))
        jj = int(rng.integers(ii + 2, len(inflection_indices)))
        i = int(inflection_indices[ii])
        j = int(inflection_indices[jj])
        shortcut = list(
            pp.interpolate_poses(
                current_poses[i],
                current_poses[j],
                pos_step_size=max(position_res, 1e-6),
                ori_step_size=max(rotation_res, 1e-6),
            )
        )
        candidate_poses = list(current_poses[:i]) + shortcut + list(current_poses[j + 1 :])
        new_cost = _pose_path_cost(candidate_poses, dist_metric, feature_points)
        if (current_cost - new_cost) <= min_cost_improvement:
            bump_profile_count(profile_out, "cost_rejections")
            continue

        if pose_collision_fn is not None and any(pose_collision_fn(pose) for pose in shortcut[1:-1]):
            bump_profile_count(profile_out, "collision_rejections")
            continue

        candidate_confs = None
        if current_confs is not None:
            candidate_suffix, failure_reason = reconstruct_joint_path_for_pose_path(
                scene=scene,
                pose_path=candidate_poses[i:],
                start_conf=current_confs[i],
                joint_collision_fn=joint_collision_fn,
                joint_continuity_threshold_rad=joint_continuity_threshold_rad,
                use_angle_normalization=use_angle_normalization,
                profile_out=profile_out,
            )
            if candidate_suffix is None:
                if failure_reason and failure_reason.startswith("ik_failure"):
                    bump_profile_count(profile_out, "ik_failures")
                elif failure_reason and failure_reason.startswith("continuity"):
                    bump_profile_count(profile_out, "continuity_rejections")
                elif failure_reason and failure_reason.startswith("collision"):
                    bump_profile_count(profile_out, "collision_rejections")
                continue
            candidate_confs = list(current_confs[:i]) + list(candidate_suffix)
            if len(candidate_confs) != len(candidate_poses):
                raise RuntimeError("Smoothed pose and joint path lengths diverged.")

        current_poses = candidate_poses
        current_confs = candidate_confs
        current_cost = new_cost
        inflection_indices = _pose_path_inflection_indices(current_poses, feature_points, inflection_tolerance)
        bump_profile_count(profile_out, "accepts")

    if profile_out is not None:
        profile_out["cost_after"] = float(current_cost)
        profile_out["waypoints_after"] = len(current_poses)
        profile_out["smooth_time_s"] = float(time.perf_counter() - start_time)
    return current_poses, current_confs


def update_debug_tree(
    debug_tree_out: Optional[Dict],
    success: bool,
    iterations: int,
    nodes: List[TreeNode],
    start_pose: PoseLike,
    goal_pose: PoseLike,
) -> None:
    if debug_tree_out is None:
        return
    debug_tree_out.clear()
    debug_tree_out["success"] = success
    debug_tree_out["iterations"] = iterations
    debug_tree_out["tree1"] = export_tree(nodes)
    debug_tree_out["tree2"] = {"points": [], "edges": []}
    debug_tree_out["start_pose"] = [float(v) for v in start_pose[0]]
    debug_tree_out["goal_pose"] = [float(v) for v in goal_pose[0]]


def plan_pose_rrt(
    robot: int,
    bar_body: int,
    obstacle_bodies: Sequence[int],
    start_pose: PoseLike,
    goal_pose: PoseLike,
    start_conf: Optional[FullConf] = None,
    goal_conf: Optional[FullConf] = None,
    dist_metric: str = "feature",
    goal_sample_prob: float = 0.1,
    workspace_xy: float = 2.2,
    workspace_z: float = 1.2,
    position_res: float = 0.05,
    rotation_res: float = 0.1,
    random_seed: Optional[int] = None,
    max_time: float = 30.0,
    max_iterations: int = 2000,
    max_attempts: int = 5,
    enable_collision: bool = False,
    enable_ik: bool = False,
    ik_context: Optional[Dict[str, Any]] = None,
    joint_collision_fn: Optional[Callable[[FullConf], bool]] = None,
    feature_points: Optional[Sequence[np.ndarray]] = None,
    joint_continuity_threshold_rad: Optional[float] = None,
    use_angle_normalization: bool = DEFAULT_USE_ANGLE_NORMALIZATION,
    use_draw: bool = True,
    debug_tree_out: Optional[Dict] = None,
    profile_out: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[List[PoseLike]], Optional[List[FullConf]]]:
    """Plan a collision-free path in SE(3) pose space using RRT.

    Args:
        robot: PyBullet body ID of the robot (used for IK and workspace sampling).
        bar_body: PyBullet body ID of the bar being manipulated (used for collision checks).
        obstacle_bodies: PyBullet body IDs of obstacles to check collisions against.
        start_pose: Starting SE(3) pose of the bar, as ((x, y, z), (qx, qy, qz, qw)).
        goal_pose: Goal SE(3) pose of the bar, same format as start_pose.
        start_conf: Full joint configuration of the robot at the start pose. Required
            when enable_ik=True (stage 2/3 planning); unused otherwise.
        goal_conf: Full joint configuration of the robot at the goal pose. Required
            when joint_collision_fn is provided.
        dist_metric: Distance metric for nearest-neighbor lookup. ``"feature"`` uses
            a feature-point projection of the bar pose; ``"pose"`` uses raw
            position/quaternion distance.
        goal_sample_prob: Probability of sampling the goal pose directly instead of
            a random workspace pose on each RRT iteration.
        workspace_xy: Half-width (metres) of the square XY workspace region sampled
            during random pose generation.
        workspace_z: Height (metres) of the workspace Z region sampled during random
            pose generation.
        position_res: Linear step size (metres) used when extending the tree and when
            testing whether the goal has been reached.
        rotation_res: Angular step size (radians) used when extending the tree and
            when testing whether the goal has been reached.
        random_seed: Seed for the NumPy RNG. Pass an integer for reproducible runs;
            ``None`` gives a non-deterministic result.
        max_time: Wall-clock time limit (seconds) per planning attempt before moving
            on to the next attempt.
        max_iterations: Maximum RRT iterations per attempt.
        max_attempts: Number of independent planning attempts before giving up.
        enable_collision: Whether to enable collision checking. In Stage 1 this
            enables floating-bar pose collision; in Stage 3 this enables joint-space
            robot collision through ``joint_collision_fn``.
        enable_ik: Whether to compute and propagate IK solutions along the tree. Must
            be True for stage-2/3 planning; requires start_conf.
        ik_context: Extra context passed to the IK solver when enable_ik=True (e.g.
            preferred joint configuration, solver settings).
        joint_collision_fn: Optional robot collision checker operating on a full
            12-DOF arm configuration. Used by Stage 3 after IK succeeds.
        use_draw: Whether to draw the RRT tree edges in the PyBullet GUI while
            planning. Disable when running headless or for speed.
        debug_tree_out: Optional dict that is populated with the final RRT tree for
            offline inspection (nodes, edges, success flag, iteration count).
        profile_out: Optional dict populated with timing and count statistics for
            each planning run (collision checks, IK calls, node counts, outcome, etc.).

    Returns:
        A tuple ``(path_poses, path_confs)``.  On success, ``path_poses`` is the list
        of SE(3) waypoint poses from start to goal and ``path_confs`` is the
        corresponding list of joint configurations (or ``None`` when enable_ik is
        False).  Returns ``(None, None)`` if no path is found within the given limits.
    """
    rng = np.random.default_rng(random_seed)
    feature_points = list(feature_points) if feature_points is not None else get_bar_feature_points()
    collision_fn = get_pose_collision_fn(bar_body, obstacle_bodies, enable_collision and not enable_ik)
    if profile_out is not None:
        profile_out.clear()
        profile_out.update(
            {
                "attempts": 0,
                "iterations": 0,
                "nodes_created": 0,
                "poses_checked": 0,
                "collision_hits": 0,
                "ik_calls": 0,
                "ik_failures": 0,
                "nodes_with_ik": 0,
                "continuity_rejections": 0,
            }
        )

    if joint_collision_fn is not None:
        if start_conf is None or goal_conf is None:
            raise ValueError("Collision-aware planning requires both start_conf and goal_conf.")
        t_collision = time.perf_counter()
        start_in_collision = joint_collision_fn(start_conf)
        add_profile_time(profile_out, "collision_check_time_s", time.perf_counter() - t_collision)
        if start_in_collision:
            logger.warning("Start configuration is in collision.")
            if profile_out is not None:
                profile_out["outcome"] = "start_in_collision"
                bump_profile_count(profile_out, "collision_hits")
            return None, None
        t_collision = time.perf_counter()
        goal_in_collision = joint_collision_fn(goal_conf)
        add_profile_time(profile_out, "collision_check_time_s", time.perf_counter() - t_collision)
        if goal_in_collision:
            logger.warning("Goal configuration is in collision.")
            if profile_out is not None:
                profile_out["outcome"] = "goal_in_collision"
                bump_profile_count(profile_out, "collision_hits")
            return None, None
    else:
        t_collision = time.perf_counter()
        start_in_collision = collision_fn(start_pose)
        add_profile_time(profile_out, "collision_check_time_s", time.perf_counter() - t_collision)
        if start_in_collision:
            logger.warning("Start pose is in floating-body collision.")
            if profile_out is not None:
                profile_out["outcome"] = "start_in_collision"
                bump_profile_count(profile_out, "collision_hits")
            return None, None
        t_collision = time.perf_counter()
        goal_in_collision = collision_fn(goal_pose)
        add_profile_time(profile_out, "collision_check_time_s", time.perf_counter() - t_collision)
        if goal_in_collision:
            logger.warning("Goal pose is in floating-body collision.")
            if profile_out is not None:
                profile_out["outcome"] = "goal_in_collision"
                bump_profile_count(profile_out, "collision_hits")
            return None, None

    best_tree: List[TreeNode] = []
    total_iterations = 0
    for attempt in range(max_attempts):
        bump_profile_count(profile_out, "attempts")
        start_time = time.time()
        root = TreeNode(start_pose)
        nodes = [root]
        node_confs: Dict[int, FullConf] = {}
        if enable_ik:
            if start_conf is None:
                raise ValueError("Stage 2/3 planning requires start_conf.")
            node_confs[id(root)] = np.asarray(start_conf, dtype=float)
        feature_vecs: Dict[int, np.ndarray] = {}
        bump_profile_count(profile_out, "nodes_created")
        if dist_metric == "feature":
            t_feature = time.perf_counter()
            root_feature = pose_to_feature_vec(start_pose, feature_points)
            add_profile_time(profile_out, "feature_time_s", time.perf_counter() - t_feature)
            if root_feature is not None:
                feature_vecs[id(root)] = root_feature

        for iteration in range(max_iterations):
            total_iterations += 1
            if profile_out is not None:
                profile_out["iterations"] = total_iterations
            if (time.time() - start_time) >= max_time:
                break
            t_sample = time.perf_counter()
            target_pose, _ = sample_pose(robot, goal_pose, rng, goal_sample_prob, workspace_xy, workspace_z)
            add_profile_time(profile_out, "sample_time_s", time.perf_counter() - t_sample)
            t_nearest = time.perf_counter()
            nearest = nearest_node(nodes, target_pose, dist_metric, feature_points, feature_vecs)
            add_profile_time(profile_out, "nearest_time_s", time.perf_counter() - t_nearest)
            t_extend = time.perf_counter()
            new_last, reached, stop_reason = extend_toward(
                nodes=nodes,
                source=nearest,
                target_pose=target_pose,
                collision_fn=collision_fn,
                joint_collision_fn=joint_collision_fn,
                draw_color=(0.85, 0.2, 0.2, 0.45),
                use_draw=use_draw,
                position_res=position_res,
                rotation_res=rotation_res,
                dist_metric=dist_metric,
                feature_points=feature_points,
                feature_vecs=feature_vecs,
                enable_ik=enable_ik,
                node_confs=node_confs,
                ik_context=ik_context,
                joint_continuity_threshold_rad=joint_continuity_threshold_rad,
                use_angle_normalization=use_angle_normalization,
                profile_out=profile_out,
            )
            add_profile_time(profile_out, "extend_tree_time_s", time.perf_counter() - t_extend)
            if not reached:
                if stop_reason == "ik_failure" and enable_ik and profile_out is not None:
                    profile_out["outcome"] = "extend_ik_failure"
                elif stop_reason == "continuity" and enable_ik and profile_out is not None:
                    profile_out["outcome"] = "extend_continuity_failure"
                elif stop_reason == "collision" and profile_out is not None:
                    profile_out["outcome"] = "extend_collision_failure"
                continue
            t_goal_dist = time.perf_counter()
            if goal_pose_reached(new_last.config, goal_pose, position_res, rotation_res):
                add_profile_time(profile_out, "goal_test_time_s", time.perf_counter() - t_goal_dist)
                update_debug_tree(debug_tree_out, True, iteration + 1, nodes, start_pose, goal_pose)
                if profile_out is not None:
                    profile_out["outcome"] = "success"
                path_nodes = new_last.retrace()
                path_poses = configs(path_nodes)
                path_confs = None
                if enable_ik:
                    path_confs = [np.asarray(node_confs[id(node)], dtype=float) for node in path_nodes]
                return path_poses, path_confs
            else:
                add_profile_time(profile_out, "goal_test_time_s", time.perf_counter() - t_goal_dist)
        best_tree = nodes
        logger.info(f"Attempt {attempt + 1}/{max_attempts}: no path found.")

    update_debug_tree(debug_tree_out, False, total_iterations, best_tree, start_pose, goal_pose)
    if profile_out is not None:
        profile_out["iterations"] = total_iterations
        profile_out["outcome"] = "task_space_failure"
    return None, None


def compute_bar_pose_from_state(
    robot: int,
    arm_joints: Sequence[int],
    tool_link: int,
    joint_values: Sequence[float],
    grasp_bar_from_tool: PoseLike,
) -> PoseLike:
    pp.set_joint_positions(robot, arm_joints, joint_values)
    world_from_tool = pp.get_link_pose(robot, tool_link)
    return pp.multiply(world_from_tool, pp.invert(grasp_bar_from_tool))


def build_default_paths() -> Tuple[str, str, str]:
    robot_cell_dir = os.path.join(DATA_DIR, "husky_assembly_design_study", "250904_transfer_path_test", "RobotCellStates")
    grasp_json = os.path.join(robot_cell_dir, "IK_test__GraspTargets.json")
    start_state = os.path.join(robot_cell_dir, "IK_test__20250905_101010_RobotCellState.json")
    end_state = os.path.join(robot_cell_dir, "IK_test__20250909_235058_RobotCellState.json")
    return grasp_json, start_state, end_state


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
    grasp_json: str,
    start_state_json: str,
    end_state_json: str,
    use_gui: bool = False,
    scene_spec: Optional[Dict[str, Any]] = None,
    swap_grasps: bool = False,
) -> Dict[str, Any]:
    if not os.path.isfile(HUSKY_DUAL_URDF_PATH):
        raise FileNotFoundError(f"URDF not found: {HUSKY_DUAL_URDF_PATH}")

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

    scene_spec = dict(scene_spec or {})
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

    grasp_targets = scene_spec.get("grasp_targets")
    if grasp_targets is None:
        grasp_targets = load_grasp_targets(grasp_json, swap_grasps=swap_grasps)
    if len(grasp_targets) < 1:
        raise ValueError(f"Expected at least one grasp target in {grasp_json}")
    world_from_bar_l, world_from_tool0_left = grasp_targets[0]
    grasp_bar_from_left = pp.multiply(pp.invert(world_from_bar_l), world_from_tool0_left)
    grasp_bar_from_right: Optional[PoseLike] = None
    if len(grasp_targets) >= 2:
        world_from_bar_r, world_from_tool0_right = grasp_targets[1]
        grasp_bar_from_right = pp.multiply(pp.invert(world_from_bar_r), world_from_tool0_right)

    start_joint_values = np.asarray(scene_spec.get("start_joint_values", load_robot_cell_state(start_state_json)), dtype=float)
    end_joint_values = np.asarray(scene_spec.get("end_joint_values", load_robot_cell_state(end_state_json)), dtype=float)

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
    grasp_json: str,
    start_state_json: str,
    end_state_json: str,
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
    scene_spec: Optional[Dict[str, Any]] = None,
    validation_reports_dir: Optional[str] = None,
    debug_tree_out: Optional[Dict] = None,
    planner_profile_out: Optional[Dict[str, Any]] = None,
    swap_grasps: bool = False,
) -> Dict[str, Any]:
    if stage not in {1, 2, 3}:
        raise ValueError(f"Unsupported stage: {stage}")
    scene = setup_planning_scene(
        grasp_json,
        start_state_json,
        end_state_json,
        use_gui=use_gui,
        scene_spec=scene_spec,
        swap_grasps=swap_grasps,
    )

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
        smooth_profile: Optional[Dict[str, Any]] = None
        start_conf = None
        goal_conf = None
        joint_collision_fn = None
        if enable_ik:
            # Stages 2/3 solve dual-arm endpoint IK once before search so the tree
            # starts from a robot-feasible grasp state instead of only a bar pose.
            grasp_bar_from_right = scene["grasp_bar_from_right"]
            if grasp_bar_from_right is None:
                raise ValueError(f"Stage {stage} requires both left and right grasp targets.")
            t_endpoint = time.perf_counter()
            start_conf = solve_endpoint_dual_arm_ik(
                robot=scene["robot"],
                arm_joints=scene["arm_joints"],
                tool_link_left=scene["tool_link_left"],
                tool_link_right=scene["tool_link_right"],
                bar_pose=scene["world_from_bar_start"],
                grasp_bar_from_left=scene["grasp_bar_from_left"],
                grasp_bar_from_right=grasp_bar_from_right,
                seed_conf=scene["start_joint_values"],
                rng=rng,
                max_attempts=endpoint_ik_attempts,
                use_angle_normalization=use_angle_normalization,
                profile_out=planner_profile_out,
            )
            if start_conf is None:
                add_profile_time(planner_profile_out, "endpoint_ik_time_s", time.perf_counter() - t_endpoint)
                if planner_profile_out is not None:
                    planner_profile_out["outcome"] = "start_ik_failure"
                logger.warning(f"Stage {stage} start pose has no valid dual-arm IK solution.")
                return early_failure("start_ik_failure")
            goal_conf = solve_endpoint_dual_arm_ik(
                robot=scene["robot"],
                arm_joints=scene["arm_joints"],
                tool_link_left=scene["tool_link_left"],
                tool_link_right=scene["tool_link_right"],
                bar_pose=scene["world_from_bar_goal"],
                grasp_bar_from_left=scene["grasp_bar_from_left"],
                grasp_bar_from_right=grasp_bar_from_right,
                seed_conf=scene["end_joint_values"],
                rng=rng,
                max_attempts=endpoint_ik_attempts,
                use_angle_normalization=use_angle_normalization,
                profile_out=planner_profile_out,
            )
            add_profile_time(planner_profile_out, "endpoint_ik_time_s", time.perf_counter() - t_endpoint)
            if goal_conf is None:
                if planner_profile_out is not None:
                    planner_profile_out["outcome"] = "goal_ik_failure"
                logger.warning(f"Stage {stage} goal pose has no valid dual-arm IK solution.")
                return early_failure("goal_ik_failure")
            if enforce_collision:
                env_obstacles = [body for body in scene["collision_obstacles"] if body != scene["robot"]]
                joint_collision_fn = get_joint_collision_fn(
                    robot=scene["robot"],
                    arm_joints=scene["arm_joints"],
                    obstacle_bodies=env_obstacles,
                    tool_link_left=scene["tool_link_left"],
                    bar_body=scene["bar_body"],
                    grasp_bar_from_left=scene["grasp_bar_from_left"],
                )

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
            use_draw=use_gui,
            debug_tree_out=debug_tree_out,
            profile_out=planner_profile_out,
        )
        t0 = time.perf_counter()
        planning_time_s = 0.0
        smoothing_time_s = 0.0
        validation_time_s = 0.0
        t_plan = time.perf_counter()
        if use_gui and lock_renderer_during_search:
            with pp.LockRenderer():
                path, path_confs = plan_pose_rrt(**planning_kwargs)
        else:
            path, path_confs = plan_pose_rrt(**planning_kwargs)
        planning_time_s = time.perf_counter() - t_plan

        path_before_smoothing = None if path is None else list(path)
        path_confs_before_smoothing = None if path_confs is None else [np.asarray(conf, dtype=float) for conf in path_confs]
        if path is not None and enable_smoothing:
            smooth_profile = {}
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
                profile_out=smooth_profile,
            )
            smoothing_time_s = time.perf_counter() - t_smooth
            if planner_profile_out is not None:
                planner_profile_out["smoothing"] = smooth_profile
            logger.info(
                "Smoothing: cost %.4f -> %.4f, waypoints %d -> %d, accepts %d/%d",
                float(smooth_profile.get("cost_before", 0.0)),
                float(smooth_profile.get("cost_after", 0.0)),
                int(smooth_profile.get("waypoints_before", 0)),
                int(smooth_profile.get("waypoints_after", 0)),
                int(smooth_profile.get("accepts", 0)),
                int(smooth_profile.get("shortcut_attempts", 0)),
            )
        runtime_s = time.perf_counter() - t0

        if path is not None:
            logger.info(f"Found Stage {stage} pose path with {len(path)} waypoints.")
            pp.set_pose(scene["bar_body"], path[-1])
            if enable_ik and path_confs:
                pp.set_joint_positions(scene["robot"], scene["arm_joints"], path_confs[-1])
        else:
            if (
                enable_ik
                and enforce_collision
                and planner_profile_out is not None
                and planner_profile_out.get("outcome") in {"start_in_collision", "goal_in_collision"}
            ):
                diagnosis_conf = start_conf if planner_profile_out["outcome"] == "start_in_collision" else goal_conf
                if diagnosis_conf is not None:
                    joint_collision_fn = get_joint_collision_fn(
                        robot=scene["robot"],
                        arm_joints=scene["arm_joints"],
                        obstacle_bodies=[body for body in scene["collision_obstacles"] if body != scene["robot"]],
                        tool_link_left=scene["tool_link_left"],
                        bar_body=scene["bar_body"],
                        grasp_bar_from_left=scene["grasp_bar_from_left"],
                    )
                    logger.warning(
                        "%s collision diagnosis:",
                        "Start" if planner_profile_out["outcome"] == "start_in_collision" else "Goal",
                    )
                    joint_collision_fn(diagnosis_conf, diagnosis=True)
            logger.warning(f"No Stage {stage} pose path found.")

        coarse_continuity = (
            summarize_joint_continuity(path_confs, use_angle_normalization=use_angle_normalization)
            if path_confs is not None
            else None
        )

        if path_confs is not None:
            validation_joint_path = [maybe_normalize_angles(conf, use_angle_normalization) for conf in path_confs]
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
            "smoothing": smooth_profile,
        }
    except Exception:
        teardown_planning_scene()
        raise


def run_stage1_trial(**kwargs) -> Dict[str, Any]:
    return run_stage_trial(stage=1, **kwargs)


def run_stage2_trial(**kwargs) -> Dict[str, Any]:
    return run_stage_trial(stage=2, **kwargs)


def run_stage3_trial(**kwargs) -> Dict[str, Any]:
    return run_stage_trial(stage=3, **kwargs)


def log_validation_summary(validation: Dict[str, Any]) -> None:
    plot_path = validation.get("plot_path")
    if plot_path:
        logger.info(f"Saved trajectory validation plot: {plot_path}")

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


def main() -> None:
    default_grasp_json, default_start_state, default_end_state = build_default_paths()
    parser = argparse.ArgumentParser(description="Minimal Stage 1/2/3 floating-bar RRT")
    parser.add_argument("--grasp-json", type=str, default=default_grasp_json, help="Path to grasp JSON file")
    parser.add_argument("--start-state", type=str, default=default_start_state, help="Path to start RobotCellState JSON")
    parser.add_argument("--end-state", type=str, default=default_end_state, help="Path to end RobotCellState JSON")
    parser.add_argument("--stage", choices=[1, 2, 3], type=int, default=3, help="Planning stage to run")
    parser.add_argument("--no-gui", action="store_true", help="Run without PyBullet GUI")
    parser.add_argument("--goal-bias", type=float, default=0.1, help="Goal sampling probability")
    parser.add_argument("--dist-metric", choices=["feature", "pose6d"], default="feature", help="Task-space distance metric")
    parser.add_argument("--position-res", type=float, default=0.01, help="Translation resolution used during pose extension, in meters")
    parser.add_argument("--rotation-res", type=float, default=0.025, help="Rotation resolution used during pose extension, in radians")
    parser.add_argument("--swap-grasps", action="store_true", help="Swap the first two grasps loaded from the grasp JSON")
    parser.add_argument("--max-time", type=float, default=30.0, help="Max planning time per attempt")
    parser.add_argument("--max-iterations", type=int, default=2000, help="Max RRT iterations per attempt")
    parser.add_argument("--max-attempts", type=int, default=5, help="Random restarts")
    parser.add_argument("--endpoint-ik-attempts", type=int, default=20, help="Max random seeds used when solving endpoint IK in Stage 2/3")
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
        action="store_true",
        help="Lock the PyBullet renderer while the tree is being expanded, then show the result afterward",
    )
    parser.add_argument(
        "--planner",
        choices=["dual-arm-constrained", "single-arm-free", "dual-arm-free"],
        default="dual-arm-constrained",
        help=(
            "Planning mode: dual-arm-constrained (default, pose-space RRT "
            "maintaining bar grasp), single-arm-free (6-DOF joint-space BiRRT "
            "for one arm), dual-arm-free (12-DOF joint-space BiRRT for both arms)"
        ),
    )
    parser.add_argument(
        "--active-arm",
        choices=["left", "right"],
        default="left",
        help="Which arm to plan for in single-arm-free mode (default: left)",
    )
    parser.set_defaults(floating_collision=False, lock_renderer_during_search=True)
    args = parser.parse_args()

    use_gui = not args.no_gui
    if args.planner == "dual-arm-constrained":
        debug_tree_out: Dict = {}
        result = run_stage_trial(
            stage=args.stage,
            grasp_json=args.grasp_json,
            start_state_json=args.start_state,
            end_state_json=args.end_state,
            use_gui=use_gui,
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
            lock_renderer_during_search=args.lock_renderer_during_search,
            swap_grasps=args.swap_grasps,
            debug_tree_out=debug_tree_out,
        )
    else:
        from husky_assembly_tamp.motion_planner.stage1.free_space_rrt import run_free_space_trial

        result = run_free_space_trial(
            planner_mode=args.planner,
            active_arm=args.active_arm,
            grasp_json=args.grasp_json,
            start_state_json=args.start_state,
            end_state_json=args.end_state,
            use_gui=use_gui,
            max_time=args.max_time,
            max_iterations=args.max_iterations,
            max_attempts=args.max_attempts,
            enable_smoothing=args.smoothing,
            smooth_iterations=args.smooth_iterations,
            random_seed=args.random_seed,
            lock_renderer_during_search=args.lock_renderer_during_search,
            swap_grasps=args.swap_grasps,
            joint_continuity_threshold_rad=args.joint_continuity_threshold,
            use_angle_normalization=args.use_angle_normalization,
            enable_collision=args.floating_collision,
        )

    if use_gui:
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


if __name__ == "__main__":
    main()
