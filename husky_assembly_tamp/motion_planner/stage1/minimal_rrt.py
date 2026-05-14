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
from collections import Counter
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


logger = setup_logger("stage1_minimal_rrt", file_mode="w")

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
DEFAULT_USE_ANGLE_NORMALIZATION = False
DEFAULT_HOME_LEFT_TOOL_Z_OFFSET = 0.2
MOBILE_BASE_FROM_TOOL0_LEFT_HOME: PoseLike = (
    np.array([0.3974141597747803, 0.16023626923561096, 0.8621799349784851], dtype=float),
    np.array([-0.5000003576278687, 0.4999987483024597, -0.499999463558197, 0.5000012516975403], dtype=float),
    # np.array([0.4999987483024597, 0.5000003576278687, 0.5000012516975403, 0.499999463558197], dtype=float)
)
# = MOBILE_BASE_FROM_TOOL0_LEFT_HOME[0] + (0, -0.2, 0); orientation derived from grasps at runtime
MOBILE_BASE_FROM_BAR_HOME_POSITION: np.ndarray = np.array(
    [0.3974, -0.0398, 0.8622], dtype=float
)
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


# DISABLED: 2pi-branch unwrapping of IK outputs against the previous conf. Left in source
# (commented) for reference; call sites are also commented out.
# def unwrap_conf_near_reference(conf: Sequence[float] | np.ndarray, reference: Sequence[float] | np.ndarray) -> np.ndarray:
#     """Choose the equivalent joint branch closest to the previous command."""
#     conf_arr = np.asarray(conf, dtype=float)
#     ref_arr = np.asarray(reference, dtype=float)
#     return ref_arr + np.asarray(normalize_angles(conf_arr - ref_arr), dtype=float)


def joint_step_exceeds_threshold(
    next_conf: Sequence[float] | np.ndarray,
    current_conf: Sequence[float] | np.ndarray,
    threshold_rad: Optional[float],
) -> bool:
    """Check raw command-space delta after branch unwrapping."""
    if threshold_rad is None:
        return False
    step_delta = np.abs(np.asarray(next_conf, dtype=float) - np.asarray(current_conf, dtype=float))
    return bool(float(np.max(step_delta)) > float(threshold_rad))


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


def bar_orientation_from_grasps(
    grasp_targets: Sequence[GraspTarget],
    target_axis_in_mb: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> Tuple[float, float, float, float]:
    """Quaternion (x,y,z,w) aligning the bar-local right->left grasp vector with target_axis_in_mb."""
    if len(grasp_targets) < 2:
        raise ValueError("Expected two grasp targets to derive bar orientation.")
    mobile_base_from_bar_left, mobile_base_from_tool0_left_goal = grasp_targets[0]
    mobile_base_from_bar_right, mobile_base_from_tool0_right_goal = grasp_targets[1]
    bar_from_tool0_left = pp.multiply(pp.invert(mobile_base_from_bar_left), mobile_base_from_tool0_left_goal)
    bar_from_tool0_right = pp.multiply(pp.invert(mobile_base_from_bar_right), mobile_base_from_tool0_right_goal)
    v = np.asarray(bar_from_tool0_left[0], dtype=float) - np.asarray(bar_from_tool0_right[0], dtype=float)
    norm = float(np.linalg.norm(v))
    if norm < 1e-6:
        return (0.0, 0.0, 0.0, 1.0)
    v = v / norm
    target = np.asarray(target_axis_in_mb, dtype=float)
    target = target / float(np.linalg.norm(target))
    cross = np.cross(v, target)
    sin_theta = float(np.linalg.norm(cross))
    cos_theta = float(np.dot(v, target))
    if sin_theta < 1e-9:
        if cos_theta > 0:
            return (0.0, 0.0, 0.0, 1.0)
        # antiparallel: 180 deg around X
        return (1.0, 0.0, 0.0, 0.0)
    axis = cross / sin_theta
    angle = float(np.arctan2(sin_theta, cos_theta))
    quat = pp.quat_from_axis_angle(axis, angle)
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def auto_compute_home_bar_pose(
    grasp_targets: Sequence[GraspTarget],
    mobile_base_from_bar: PoseLike,
    forward_direction: np.ndarray = np.array([1.0, 0.0, 0.0]),
    ik_validator: Optional[Callable[[PoseLike], bool]] = None,
    num_geometric_candidates: int = 20,
    allow_unvalidated_fallback: bool = True,
    bar_axis_step_rad: float = float(np.deg2rad(30.0)),
) -> Dict[str, Any]:
    """Auto-compute the home bar pose by optimizing bar-axis rotation around a fixed bar position+orientation."""
    if len(grasp_targets) < 2:
        raise ValueError("Expected two grasp targets to auto-compute the home bar pose.")

    mobile_base_from_bar_left, mobile_base_from_tool0_left_goal = grasp_targets[0]
    mobile_base_from_bar_right, mobile_base_from_tool0_right_goal = grasp_targets[1]
    bar_from_tool0_left = pp.multiply(pp.invert(mobile_base_from_bar_left), mobile_base_from_tool0_left_goal)
    bar_from_tool0_right = pp.multiply(pp.invert(mobile_base_from_bar_right), mobile_base_from_tool0_right_goal)

    forward = np.asarray(forward_direction, dtype=float)
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-9:
        raise ValueError("forward_direction must be non-zero.")
    forward = forward / forward_norm

    all_candidates: List[Tuple[float, float, PoseLike]] = []
    for theta in np.arange(-np.pi, np.pi, bar_axis_step_rad):
        bar_rotated = pp.multiply(mobile_base_from_bar, pp.Pose(euler=pp.Euler(yaw=float(theta))))
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
        all_candidates.append((score, float(theta), bar_rotated))

    if not all_candidates:
        raise ValueError("Could not generate any home bar pose candidates.")

    all_candidates.sort(key=lambda candidate: -candidate[0])
    chosen = all_candidates[0]
    ik_validated = False

    if ik_validator is not None and num_geometric_candidates > 0:
        top_candidates = all_candidates[:num_geometric_candidates]
        for candidate in top_candidates:
            _, _, bar_pose = candidate
            if ik_validator(bar_pose):
                chosen = candidate
                ik_validated = True
                break
        else:
            if allow_unvalidated_fallback:
                logger.warning(
                    "No IK-feasible candidate found among top %d geometric candidates; falling back to best geometric candidate.",
                    num_geometric_candidates,
                )

    best_score, best_theta, _ = chosen
    bar_final = pp.multiply(mobile_base_from_bar, pp.Pose(euler=pp.Euler(yaw=best_theta)))
    left_tool_final = pp.multiply(bar_final, bar_from_tool0_left)
    right_tool_final = pp.multiply(bar_final, bar_from_tool0_right)

    return {
        "mobile_base_from_bar_start": bar_final,
        "mobile_base_from_tool0_left_start": left_tool_final,
        "mobile_base_from_tool0_right_start": right_tool_final,
        "tool0_left_from_bar": pp.invert(bar_from_tool0_left),
        "bar_from_tool0_right": bar_from_tool0_right,
        "chosen_bar_axis_theta": best_theta,
        "alignment_score": best_score,
        "ik_validated": ik_validated,
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
    pos_tolerance: float = 1e-4,
    ori_tolerance: float = 1e-4,
) -> bool:
    pp.set_joint_positions(robot, arm_joints, full_conf)
    target_left = pp.multiply(bar_pose, grasp_bar_from_left)
    target_right = pp.multiply(bar_pose, grasp_bar_from_right)
    world_from_left = pp.get_link_pose(robot, tool_link_left)
    world_from_right = pp.get_link_pose(robot, tool_link_right)
    if not pp.is_pose_close(target_left, world_from_left, pos_tolerance=pos_tolerance, ori_tolerance=ori_tolerance):
        return False
    if not pp.is_pose_close(target_right, world_from_right, pos_tolerance=pos_tolerance, ori_tolerance=ori_tolerance):
        return False
    bar_from_left = pp.invert(grasp_bar_from_left)
    bar_from_right = pp.invert(grasp_bar_from_right)
    left_bar_pose = pp.multiply(world_from_left, bar_from_left)
    right_bar_pose = pp.multiply(world_from_right, bar_from_right)
    return bool(
        pp.is_pose_close(left_bar_pose, bar_pose, pos_tolerance=pos_tolerance, ori_tolerance=ori_tolerance)
        and pp.is_pose_close(right_bar_pose, bar_pose, pos_tolerance=pos_tolerance, ori_tolerance=ori_tolerance)
        and pp.is_pose_close(left_bar_pose, right_bar_pose, pos_tolerance=pos_tolerance, ori_tolerance=ori_tolerance)
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
) -> Optional[FullConf]:
    target_left = pp.multiply(bar_pose, grasp_bar_from_left)
    target_right = pp.multiply(bar_pose, grasp_bar_from_right)
    seed_conf = maybe_normalize_angles(seed_conf, use_angle_normalization)
    attempts = (
        ("right", "left"),
        ("left", "right"),
    )
    for order in attempts:
        conf = seed_conf.copy()
        success = True
        for arm_name in order:
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
            return maybe_normalize_angles(conf, use_angle_normalization)
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
    collision_fn: Optional[Callable[[np.ndarray], bool]] = None,
    **_unused_kwargs: Any,
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
        )
        if conf is not None:
            if collision_fn is not None and collision_fn(np.asarray(conf, dtype=float)):
                continue
            return conf
    return None


def _grid_in_box(
    box: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    step: float,
) -> List[Tuple[float, float, float]]:
    (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = box
    xs = np.arange(x_lo, x_hi + 0.5 * step, step)
    ys = np.arange(y_lo, y_hi + 0.5 * step, step)
    zs = np.arange(z_lo, z_hi + 0.5 * step, step)
    return [(float(x), float(y), float(z)) for x in xs for y in ys for z in zs]


def derive_constrained_start(
    robot: int,
    arm_joints: Sequence[int],
    tool_link_left: int,
    tool_link_right: int,
    grasp_bar_from_left,
    grasp_bar_from_right,
    world_from_bar_goal,
    seed_conf: Sequence[float],
    *,
    bar_body: Optional[int] = None,
    obstacles: Sequence[int] = (),
    world_from_mobile_base=None,
    bar_sweep_box: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (-0.3, 0.3),
        (-0.3, 0.3),
        (-0.3, 0.3),
    ),
    bar_sweep_step: float = 0.1,
    num_geometric_candidates: int = 20,
    bar_axis_step_rad: float = float(np.deg2rad(30.0)),
    max_ik_attempts: int = 20,
    random_seed: Optional[int] = None,
) -> Tuple[Optional[PoseLike], Optional[np.ndarray]]:
    """Derive a constraint-satisfying start (bar pose, joint conf)."""
    if len(seed_conf) != 12:
        raise ValueError("seed_conf must have length 12")

    identity_pose = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    if world_from_mobile_base is None:
        world_from_mobile_base = identity_pose
    mobile_base_from_world = pp.invert(world_from_mobile_base)

    mb_from_bar_goal = pp.multiply(mobile_base_from_world, world_from_bar_goal)
    mb_from_tool0_left_goal = pp.multiply(mb_from_bar_goal, grasp_bar_from_left)
    mb_from_tool0_right_goal = pp.multiply(mb_from_bar_goal, grasp_bar_from_right)
    grasp_targets_mb = [
        (mb_from_bar_goal, mb_from_tool0_left_goal),
        (mb_from_bar_goal, mb_from_tool0_right_goal),
    ]

    home_bar_quat = bar_orientation_from_grasps(grasp_targets_mb)

    # Anchor the grasp midpoint at the home position. Some bar frames live at
    # one grasp end, so anchoring the frame origin would shift the held bar.
    bar_from_tool0_left_local = pp.multiply(pp.invert(mb_from_bar_goal), mb_from_tool0_left_goal)
    bar_from_tool0_right_local = pp.multiply(pp.invert(mb_from_bar_goal), mb_from_tool0_right_goal)
    grasp_midpoint_in_bar = 0.5 * (
        np.asarray(bar_from_tool0_left_local[0], dtype=float)
        + np.asarray(bar_from_tool0_right_local[0], dtype=float)
    )
    midpoint_in_mb = np.asarray(
        pp.multiply(
            ((0.0, 0.0, 0.0), home_bar_quat),
            (tuple(grasp_midpoint_in_bar.tolist()), (0.0, 0.0, 0.0, 1.0)),
        )[0],
        dtype=float,
    )
    base_pos_mb = np.asarray(MOBILE_BASE_FROM_BAR_HOME_POSITION, dtype=float) - midpoint_in_mb

    rng = np.random.default_rng(random_seed)
    joint_collision_fn = None
    if bar_body is not None:
        joint_collision_fn = get_joint_collision_fn(
            robot=robot,
            arm_joints=arm_joints,
            obstacle_bodies=list(obstacles),
            tool_link_left=tool_link_left,
            bar_body=bar_body,
            grasp_bar_from_left=grasp_bar_from_left,
        )

    deltas = sorted(
        _grid_in_box(bar_sweep_box, bar_sweep_step),
        key=lambda d: float(np.linalg.norm(d)),
    )
    saved_bar_pose = pp.get_pose(bar_body) if bar_body is not None else None
    found: Dict[str, Any] = {"world_from_bar": None, "conf": None}
    chosen_ctx: Optional[Dict[str, Any]] = None

    with pp.WorldSaver():
        for delta in deltas:
            mb_from_bar_candidate = (
                tuple((base_pos_mb + np.asarray(delta, dtype=float)).tolist()),
                home_bar_quat,
            )

            def ik_validator(bar_pose_mb, _found=found):
                world_from_bar = pp.multiply(world_from_mobile_base, bar_pose_mb)
                conf = solve_endpoint_dual_arm_ik(
                    robot=robot,
                    arm_joints=arm_joints,
                    tool_link_left=tool_link_left,
                    tool_link_right=tool_link_right,
                    bar_pose=world_from_bar,
                    grasp_bar_from_left=grasp_bar_from_left,
                    grasp_bar_from_right=grasp_bar_from_right,
                    seed_conf=np.asarray(seed_conf, dtype=float),
                    rng=rng,
                    max_attempts=max_ik_attempts,
                    collision_fn=joint_collision_fn,
                )
                if conf is None:
                    return False
                _found["world_from_bar"] = world_from_bar
                _found["conf"] = conf
                return True

            ctx = auto_compute_home_bar_pose(
                grasp_targets_mb,
                mobile_base_from_bar=mb_from_bar_candidate,
                ik_validator=ik_validator,
                num_geometric_candidates=num_geometric_candidates,
                bar_axis_step_rad=bar_axis_step_rad,
                allow_unvalidated_fallback=False,
            )
            if ctx.get("ik_validated", False):
                chosen_ctx = ctx
                break

    if saved_bar_pose is not None:
        pp.set_pose(bar_body, saved_bar_pose)

    if chosen_ctx is None or found["conf"] is None:
        logger.warning(
            "derive_constrained_start: no collision-free home pose across %d deltas (kinematic_only=%s)",
            len(deltas),
            bar_body is None,
        )
        return None, None

    world_from_bar_start = pp.multiply(world_from_mobile_base, chosen_ctx["mobile_base_from_bar_start"])
    return world_from_bar_start, found["conf"]


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
) -> Tuple[TreeNode, bool, str]:
    """Extend the RRT tree from ``source`` toward ``target_pose`` one interp step at a time.

    Walks the SE(3) line from source.config to target_pose at (position_res, rotation_res)
    granularity. At each intermediate pose: (optionally) solve IK + continuity + joint-collision,
    then pose collision; on failure stop but keep all valid intermediate nodes appended.

    Returns:
        (last_node_added, reached_target, stop_reason)
        ``stop_reason`` ∈ {"reached", "collision", "ik_failure", "continuity"}.
    """
    # `current`/`current_conf` track the frontier as we walk; updated each accepted step.
    current = source
    reached = True
    stop_reason = "reached"
    current_conf = None if node_confs is None else node_confs.get(id(source))
    # IK mode requires both the conf cache and an ik_context; bail before any work if missing.
    if enable_ik and (node_confs is None or ik_context is None or current_conf is None):
        return current, False, "ik_failure"

    # Discretize the SE(3) segment source -> target into waypoints. Skip index 0 (== source).
    for pose in list(
        pp.interpolate_poses(
            source.config,
            target_pose,
            pos_step_size=max(position_res, 1e-6),
            ori_step_size=max(rotation_res, 1e-6),
        )
    )[1:]:
        next_conf = None
        if enable_ik:
            # --- Stage 2/3: solve dual-arm IK at this pose, seeded by previous step's conf
            # for warm-start continuity.
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
            )

            # IK miss -> can't follow the bar pose; stop extension here.
            if next_conf is None:
                reached = False
                stop_reason = "ik_failure"
                break

            # Unwrap revolute joints to be ±pi-closest to the previous conf, so a step that
            # "looks" like a 2pi jump is normalized away before the continuity check.
            # DISABLED: keep raw IK output; let continuity check see the unmodified delta.
            # next_conf = unwrap_conf_near_reference(next_conf, current_conf)

            # Reject IK branch flips: large per-joint deltas between consecutive steps
            # indicate the solver hopped to a different IK solution.
            if joint_step_exceeds_threshold(next_conf, current_conf, joint_continuity_threshold_rad):
                reached = False
                stop_reason = "continuity"
                break

            # Stage 3: full robot-vs-world / self collision in joint space.
            if joint_collision_fn is not None and joint_collision_fn(next_conf):
                reached = False
                stop_reason = "collision"
                break

        # Stage 1 (or always-on belt-and-braces): floating bar collision against obstacles.
        if collision_fn(pose):
            reached = False
            stop_reason = "collision"
            break

        # this is the version of extension that stops at the first collision of the extend, but still add the valid interp so far into the tree
        # --- Accept this waypoint: append as child of `current` and refresh caches.
        node = TreeNode(pose, parent=current)
        nodes.append(node)
        if enable_ik and node_confs is not None and next_conf is not None:
            node_confs[id(node)] = next_conf
        if dist_metric == "feature":
            feature_vec = pose_to_feature_vec(pose, feature_points)
            if feature_vec is not None:
                feature_vecs[id(node)] = feature_vec

        if use_draw:
            # Visualize tree edge in the PyBullet GUI.
            pp.add_line(current.config[0], node.config[0], width=1.5, color=draw_color)

        # Advance the frontier.
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
    command_joint_path = [np.asarray(conf, dtype=float) for conf in joint_path]
    if len(command_joint_path) < 2:
        summary["ok"] = True
        summary["max_delta_rad"] = 0.0
        return summary

    step_max_deltas = []
    for prev_conf, next_conf in zip(command_joint_path[:-1], command_joint_path[1:]):
        step_delta = np.abs(np.asarray(next_conf, dtype=float) - np.asarray(prev_conf, dtype=float))
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
) -> Tuple[Optional[List[FullConf]], Optional[str]]:
    if not pose_path:
        return [], None
    grasp_bar_from_right = scene["grasp_bar_from_right"]
    if grasp_bar_from_right is None:
        return None, "missing_right_grasp"

    current_conf = np.asarray(start_conf, dtype=float)
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
        )
        if next_conf is None:
            return None, f"ik_failure_at_waypoint_{idx}"
        # DISABLED: keep raw IK output; do not normalize to ±pi-closest branch of current_conf.
        # next_conf = unwrap_conf_near_reference(next_conf, current_conf)
        if joint_step_exceeds_threshold(next_conf, current_conf, joint_continuity_threshold_rad):
            return None, f"continuity_at_waypoint_{idx}"
        if joint_collision_fn is not None and joint_collision_fn(next_conf):
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
    **_unused_kwargs: Any,
) -> Tuple[List[PoseLike], Optional[List[FullConf]]]:
    feature_points = list(feature_points) if feature_points is not None else get_bar_feature_points()
    current_poses = list(path_poses)
    current_confs = None if path_confs is None else [np.asarray(conf, dtype=float) for conf in path_confs]
    if current_confs is not None and len(current_confs) != len(current_poses):
        raise ValueError("Pose and joint path lengths must match for smoothing.")

    current_cost = _pose_path_cost(current_poses, dist_metric, feature_points)
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
            continue

        if pose_collision_fn is not None and any(pose_collision_fn(pose) for pose in shortcut[1:-1]):
            continue

        candidate_confs = None
        if current_confs is not None:
            candidate_suffix, _failure_reason = reconstruct_joint_path_for_pose_path(
                scene=scene,
                pose_path=candidate_poses[i:],
                start_conf=current_confs[i],
                joint_collision_fn=joint_collision_fn,
                joint_continuity_threshold_rad=joint_continuity_threshold_rad,
                use_angle_normalization=use_angle_normalization,
            )
            if candidate_suffix is None:
                continue
            candidate_confs = list(current_confs[:i]) + list(candidate_suffix)
            if len(candidate_confs) != len(candidate_poses):
                raise RuntimeError("Smoothed pose and joint path lengths diverged.")

        current_poses = candidate_poses
        current_confs = candidate_confs
        current_cost = new_cost
        inflection_indices = _pose_path_inflection_indices(current_poses, feature_points, inflection_tolerance)

    return current_poses, current_confs


def update_debug_tree(
    debug_tree_out: Optional[Dict],
    success: bool,
    iterations: int,
    nodes: List[TreeNode],
    start_pose: PoseLike,
    goal_pose: PoseLike,
    extend_stop_reasons: Optional[Dict[str, int]] = None,
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
    # Histogram of `extend_toward` stop_reasons across all RRT iterations.
    # Useful for diagnosing failed plans (e.g. mostly "ik_failure" vs "collision").
    debug_tree_out["extend_stop_reasons"] = dict(extend_stop_reasons) if extend_stop_reasons else {}


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
    **_unused_kwargs: Any,
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

    Returns:
        A tuple ``(path_poses, path_confs)``.  On success, ``path_poses`` is the list
        of SE(3) waypoint poses from start to goal and ``path_confs`` is the
        corresponding list of joint configurations (or ``None`` when enable_ik is
        False).  Returns ``(None, None)`` if no path is found within the given limits.
    """
    # --- Setup: RNG, feature points for "feature" distance, and pose-level collision fn.
    # Pose collision fn only used when IK is off (Stage 1). With enable_ik, joint_collision_fn
    # in extend_toward is the authoritative check, so we disable the pose-level one here.
    rng = np.random.default_rng(random_seed)
    feature_points = list(feature_points) if feature_points is not None else get_bar_feature_points()
    collision_fn = get_pose_collision_fn(bar_body, obstacle_bodies, enable_collision and not enable_ik)

    # --- Endpoint feasibility: reject early if start/goal already in collision.
    # Use joint-space check when a robot collision fn is provided (Stage 3),
    # otherwise check the floating bar pose against obstacles (Stage 1).
    if joint_collision_fn is not None:
        if start_conf is None or goal_conf is None:
            raise ValueError("Collision-aware planning requires both start_conf and goal_conf.")
        start_in_collision = joint_collision_fn(start_conf)
        if start_in_collision:
            logger.warning("Start configuration is in collision.")
            return None, None
        goal_in_collision = joint_collision_fn(goal_conf)
        if goal_in_collision:
            logger.warning("Goal configuration is in collision.")
            return None, None
    else:
        start_in_collision = collision_fn(start_pose)
        if start_in_collision:
            logger.warning("Start pose is in floating-body collision.")
            return None, None
        goal_in_collision = collision_fn(goal_pose)
        if goal_in_collision:
            logger.warning("Goal pose is in floating-body collision.")
            return None, None

    # --- Outer loop: independent RRT restarts (each builds a fresh tree).
    # Track histogram of extend_toward stop_reasons across the whole plan call
    # (accumulates across attempts) so callers can see failure-mode breakdown.
    extend_stop_reasons: Counter = Counter()
    best_tree: List[TreeNode] = []
    total_iterations = 0
    for attempt in range(max_attempts):
        start_time = time.time()

        # Root the tree at the start pose. Side caches keyed by id(node):
        #   node_confs   -> full joint config at that node (only when enable_ik)
        #   feature_vecs -> cached feature-point vector for fast nearest lookup
        root = TreeNode(start_pose)
        nodes = [root]
        node_confs: Dict[int, FullConf] = {}
        if enable_ik:
            if start_conf is None:
                raise ValueError("Stage 2/3 planning requires start_conf.")
            node_confs[id(root)] = np.asarray(start_conf, dtype=float)
        feature_vecs: Dict[int, np.ndarray] = {}
        if dist_metric == "feature":
            root_feature = pose_to_feature_vec(start_pose, feature_points)
            if root_feature is not None:
                feature_vecs[id(root)] = root_feature

        # --- Inner loop: standard RRT — sample, nearest, extend, goal-check.
        for iteration in range(max_iterations):
            total_iterations += 1
            # Wall-clock budget per attempt.
            if (time.time() - start_time) >= max_time:
                break
            # 1) Sample: with prob `goal_sample_prob` returns goal_pose, else random workspace pose.
            target_pose, _ = sample_pose(robot, goal_pose, rng, goal_sample_prob, workspace_xy, workspace_z)

            # 2) Nearest-neighbor in current tree under chosen distance metric.
            nearest = nearest_node(nodes, target_pose, dist_metric, feature_points, feature_vecs)

            # 3) Extend: step from `nearest` toward `target_pose` at (position_res, rotation_res).
            #    Stops on collision, IK fail, or joint-discontinuity; appends every valid
            #    intermediate node into `nodes` and updates node_confs/feature_vecs in-place.
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
            )
            extend_stop_reasons[stop_reason] += 1

            # Extension stalled before reaching target — keep iterating.
            if not reached:
                continue

            # 4) Goal test on the newly-added frontier node.
            if goal_pose_reached(new_last.config, goal_pose, position_res, rotation_res):
                update_debug_tree(
                    debug_tree_out, True, iteration + 1, nodes, start_pose, goal_pose,
                    extend_stop_reasons=extend_stop_reasons,
                )
                # Retrace tree from frontier back to root, then reverse to start->goal order.
                path_nodes = new_last.retrace()
                path_poses = configs(path_nodes)
                path_confs = None
                if enable_ik:
                    # Pull cached joint configs for each pose waypoint.
                    path_confs = [np.asarray(node_confs[id(node)], dtype=float) for node in path_nodes]
                return path_poses, path_confs

        # Attempt exhausted (time or iteration cap) — remember tree for debug, then restart.
        best_tree = nodes
        logger.info(f"Attempt {attempt + 1}/{max_attempts}: no path found.")

    # All attempts failed. Dump last tree for inspection and signal failure.
    update_debug_tree(
        debug_tree_out, False, total_iterations, best_tree, start_pose, goal_pose,
        extend_stop_reasons=extend_stop_reasons,
    )
    return None, None


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
            world_from_bar_start, start_conf = derive_constrained_start(
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
                random_seed=random_seed,
                max_ik_attempts=endpoint_ik_attempts,
            )
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
        if use_gui and lock_renderer_during_search:
            with pp.LockRenderer():
                path, path_confs = plan_pose_rrt(**planning_kwargs)
        else:
            path, path_confs = plan_pose_rrt(**planning_kwargs)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Stage 1/2/3 floating-bar RRT")
    parser.add_argument("--stage", choices=[1, 2, 3], type=int, default=3, help="Planning stage to run")
    parser.add_argument("--no-gui", action="store_true", help="Run without PyBullet GUI")
    parser.add_argument("--goal-bias", type=float, default=0.1, help="Goal sampling probability")
    parser.add_argument("--dist-metric", choices=["feature", "pose6d"], default="feature", help="Task-space distance metric")
    parser.add_argument("--position-res", type=float, default=0.01, help="Translation resolution used during pose extension, in meters")
    parser.add_argument("--rotation-res", type=float, default=0.025, help="Rotation resolution used during pose extension, in radians")
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
        "--no-lock-renderer-during-search",
        action="store_true",
        help="Don't Lock the PyBullet renderer while the tree is being expanded, so we can look at tree growth interactively",
    )
    parser.add_argument(
        "--draw-rrt-tree",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw RRT tree edges in the PyBullet GUI while planning (use --no-draw-rrt-tree to disable). Only effective when GUI is on.",
    )
    # gdrive convention: single-state input (no GraspTargets JSON), bodies
    # tagged active_bar_* / active_*_* / env_*.
    parser.add_argument(
        "--gdrive-state", type=str, default=None,
        help=("Path or bare filename of a gdrive-convention RobotCellState "
              "(e.g. 'B3_approach.json'). Grasps come from FK at the cell "
              "state's joint values."),
    )
    parser.add_argument(
        "--gdrive-bar-action", type=str, default=None,
        help=("Path or bare filename of a gdrive BarAction JSON "
              "(e.g. 'B1.json'). When set, target EE frames come from the "
              "selected BarAction movement and no full RobotCell scene is loaded."),
    )
    parser.add_argument(
        "--movement", type=str, default="M1",
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
    # parser.set_defaults(floating_collision=False, lock_renderer_during_search=True)
    args = parser.parse_args()

    use_gui = not args.no_gui
    gdrive_movement: str | int = args.movement
    if isinstance(gdrive_movement, str) and gdrive_movement.isdigit():
        gdrive_movement = int(gdrive_movement)
    if args.gdrive_state is not None and args.gdrive_bar_action is not None:
        raise ValueError("--gdrive-state and --gdrive-bar-action are mutually exclusive.")
    if args.gdrive_state is None and args.gdrive_bar_action is None:
        raise ValueError("Exactly one of --gdrive-state / --gdrive-bar-action is required.")
    gdrive_scene_spec: Dict[str, Any]
    if args.gdrive_state is not None:
        gdrive_scene_spec = build_gdrive_scene_spec(
            args.gdrive_state,
            problem=args.gdrive_problem,
            include_env_bars=not args.gdrive_no_env,
            include_active_extras=not args.gdrive_no_active_extras,
        )
        logger.info(f"Using gdrive scene_spec from {gdrive_scene_spec['_gdrive_state_path']}")
        logger.info(f"  active_bar={gdrive_scene_spec['_gdrive_active_bar_name']!r}, "
                    f"built_bars={len(gdrive_scene_spec['built_bars'])}")
    else:
        gdrive_scene_spec = build_gdrive_bar_action_scene_spec(
            args.gdrive_bar_action,
            movement=gdrive_movement,
            problem=args.gdrive_problem,
        )
        logger.info(f"Using gdrive BarAction scene_spec from {gdrive_scene_spec['_gdrive_bar_action_path']}")
        logger.info(f"  movement={gdrive_scene_spec['_gdrive_bar_action_movement']!r}, "
                    f"active_bar={gdrive_scene_spec['_gdrive_active_bar_name']!r}, "
                    f"built_bars={len(gdrive_scene_spec['built_bars'])}")
    debug_tree_out: Dict = {}
    result = run_stage_trial(
        stage=args.stage,
        scene_spec=gdrive_scene_spec,
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
        lock_renderer_during_search=not args.no_lock_renderer_during_search,
        draw_rrt_tree=args.draw_rrt_tree,
        debug_tree_out=debug_tree_out,
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
