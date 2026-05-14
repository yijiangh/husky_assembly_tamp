from __future__ import annotations

import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet
import pybullet_planning as pp
from compas_fab.robots import RobotSemantics
from compas_robots import RobotModel
from pybullet_planning.motion_planners.rrt import TreeNode, configs

from husky_assembly_tamp.utils.util import calculate_pose_error, normalize_angles, setup_logger


logger = setup_logger("dual_arm_task_space_rrt_core", file_mode="w")


PoseLike = Tuple[np.ndarray, np.ndarray]
GraspTarget = Tuple[PoseLike, PoseLike]
FullConf = np.ndarray
BAR_RADIUS = 0.015
BAR_LENGTH = 1.0
BAR_BOX_DIMS = (2.0 * BAR_RADIUS, 2.0 * BAR_RADIUS, BAR_LENGTH)
DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD = 10.0 * np.pi / 180.0
DEFAULT_USE_ANGLE_NORMALIZATION = False
TOOL_LINK_LEFT = "left_ur_arm_tool0"
TOOL_LINK_RIGHT = "right_ur_arm_tool0"
STAGE3_GRASP_MASK_LINKS = [
    "left_ur_arm_wrist_3_link",
    "right_ur_arm_wrist_3_link",
    TOOL_LINK_LEFT,
    TOOL_LINK_RIGHT,
]
# = MOBILE_BASE_FROM_TOOL0_LEFT_HOME[0] + (0, -0.2, 0); orientation derived from grasps at runtime
MOBILE_BASE_FROM_BAR_HOME_POSITION: np.ndarray = np.array(
    [0.3974, -0.0398, 0.8622], dtype=float
)


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
    from .run import HUSKY_DUAL_URDF_PATH, HUSKY_DUAL_SRDF_PATH
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
