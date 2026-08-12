"""Task-space (SE(3)) RRT motion planner for a dual-arm Husky carrying a bar.

The robot holds one rigid bar with both arms and we want to move that bar from
a start pose to a goal pose without hitting anything. Instead of planning
directly in the 12 arm joints, we plan in the *bar's* SE(3) pose (position +
orientation) and let inverse kinematics figure out the arm joints at each step.
That keeps the two-arm grasp rigid throughout the motion.

Every candidate bar pose is checked in three escalating stages:

    Stage 1 - float the bar (no robot) and check the bar mesh against the
              environment obstacles. Cheap early reject.
    Stage 2 - solve dual-arm IK so both tool flanges follow the grasped bar
              pose. If IK misses, the bar pose is unreachable.
    Stage 3 - with the IK joints set, check the whole robot (arms + tools +
              held bar) for self / world collision in joint space.

Collision checks in Stages 1 and 3 are delegated to the compas_fab ("cfab")
PyBullet planner so the allowed-collision rules are defined in exactly one
place (see ``build_cfab_pose_collision_fn`` and the cfab collision adapter).

Two public planners are provided:

    ``plan_pose_birrt`` - bidirectional RRT-Connect (grows a start tree and a
                          goal tree and stitches them when they meet). This is
                          the DEFAULT planner (benchmarking showed it wins).
    ``plan_pose_rrt``   - single-tree RRT grown from the start pose. Kept for
                          archival comparison only; no longer the default.

Both return ``(path_poses, path_confs)``: the SE(3) waypoints of the bar and,
when IK is enabled, the matching 12-DOF dual-arm joint configuration per
waypoint. Distances are in metres and angles in radians throughout.
"""

from __future__ import annotations

# * Standard library
import os
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# * Third-party: numeric core, PyBullet, and the compas robotics stack
import numpy as np
import pybullet
import pybullet_planning as pp
from compas_fab.robots import RobotSemantics
from compas_robots import RobotModel
from pybullet_planning.motion_planners.rrt import TreeNode, configs

# * This package (husky_assembly_tamp) — data paths + shared helpers
from husky_assembly_tamp.motion_planner import ssik_ik
from husky_assembly_tamp.utils.params import DATA_DIR
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

# * Discrete "home bar" carry anchors for M1 start derivation, in priority
# * order (dict insertion order = sampling order). Each anchor is a
# * bar-independent spec -- bar frames differ per bar (origin can sit at one
# * tip, local Z may flip), so the actual bar quaternion is still derived per
# * bar from the grasps at runtime (see bar_orientation_from_grasps):
# *   midpoint_mb -- where the midpoint of the two tool0 grasps sits (mb frame)
# *   bar_axis_mb -- target direction for the bar's right->left grasp axis
# *   forward_mb  -- direction the tool0 Z axes should face (roll-sweep scoring)
# ? Provenance: authored demo files in
# ? data_design_study/260812_M1_motion_samples/BarActions (identity base frame,
# ? so M1 target_ee_frames ARE the carry pose in the mb frame; each file's
# ? target_configuration proves the pose is IK-feasible):
# ?   horizontal: B3.json  tool0 L [0.635, 0.33, 0.95] / R [0.535, -0.33, 0.85]
# ?               (midpoint kept at the pre-existing constant by choice)
# ?   vertical:   B6.json  tool0 L [0.715, 0.0, 1.17] / R [0.715, 0.0, 0.23]
# ?   back:       B9.json  tool0 L [0.57, 0.0, 1.04]  / R [-0.37, 0.0, 1.04]
HOME_BAR_ANCHORS: Dict[str, Dict[str, np.ndarray]] = {
    # Bar across the robot's front, axis along base-link Y (the original home).
    "horizontal": dict(
        midpoint_mb=MOBILE_BASE_FROM_BAR_HOME_POSITION,
        bar_axis_mb=np.array([0.0, 1.0, 0.0]),
        forward_mb=np.array([1.0, 0.0, 0.0]),
    ),
    # Bar upright in front of the robot, axis along base-link Z.
    "vertical": dict(
        midpoint_mb=np.array([0.715, 0.0, 0.70]),
        bar_axis_mb=np.array([0.0, 0.0, 1.0]),
        forward_mb=np.array([1.0, 0.0, 0.0]),
    ),
    # Bar carried fore-aft over the robot's back, tools facing up.
    "back": dict(
        midpoint_mb=np.array([0.10, 0.0, 1.04]),
        bar_axis_mb=np.array([1.0, 0.0, 0.0]),
        forward_mb=np.array([0.0, 0.0, 1.0]),
    ),
}


def resolve_home_anchors(anchors: Optional[Sequence[str]] = None) -> List[str]:
    """Normalize a home-anchor selection into an ordered list of labels.

    Args:
        anchors (Optional[Sequence[str]]): None or "all" selects every anchor
            in ``HOME_BAR_ANCHORS`` priority order; a single label string or a
            list of labels selects just those (order preserved).

    Returns:
        List[str]: validated anchor labels to sample, in order.

    Raises:
        ValueError: if a label is not a key of ``HOME_BAR_ANCHORS``.
    """
    if anchors is None or anchors == "all":
        return list(HOME_BAR_ANCHORS.keys())
    labels = [anchors] if isinstance(anchors, str) else list(anchors)
    unknown = [label for label in labels if label not in HOME_BAR_ANCHORS]
    if unknown:
        raise ValueError(
            f"Unknown home anchor label(s) {unknown}; valid: {list(HOME_BAR_ANCHORS)}"
        )
    return labels

# Husky dual-arm URDF/SRDF, used by the joint-space collision predicate.
# Defined here (not imported from ``run``) so this module never needs to import
# the standalone runner — importing ``run`` would pull in its heavier deps and
# create a core<->run import cycle.
HUSKY_DUAL_URDF_PATH = os.path.join(
    DATA_DIR,
    "husky_urdf/mt_husky_dual_ur5_e_moveit_config/urdf/husky_dual_ur5_e_no_base_joint_All_Calibrated.urdf",
)
HUSKY_DUAL_SRDF_PATH = os.path.join(
    DATA_DIR,
    "husky_urdf/mt_husky_dual_ur5_e_moveit_config/config/dual_arm_husky.srdf",
)


def maybe_normalize_angles(values: Sequence[float] | np.ndarray, use_angle_normalization: bool) -> np.ndarray:
    """Optionally wrap joint angles into a canonical (-pi, pi] branch.

    A single on/off switch so call sites can choose between raw IK output and
    normalized angles without duplicating the check.

    Args:
        values (Sequence[float] | np.ndarray): joint angles in radians.
        use_angle_normalization (bool): when True, wrap each angle via
            ``normalize_angles``; when False, pass the values through unchanged.

    Returns:
        np.ndarray: the (optionally normalized) angles as a float array.
    """
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
    """Whether any joint moved more than ``threshold_rad`` between two configs.

    Used to reject IK "branch flips": when consecutive pose steps land on
    different IK solutions, one or more joints jump by a large amount even though
    the tool pose barely changed.

    Args:
        next_conf (Sequence[float] | np.ndarray): the candidate next configuration.
        current_conf (Sequence[float] | np.ndarray): the previous configuration.
        threshold_rad (Optional[float]): max allowed per-joint change in radians;
            ``None`` disables the check (always returns False).

    Returns:
        bool: True if the largest per-joint delta exceeds the threshold.
    """
    if threshold_rad is None:
        return False
    step_delta = np.abs(np.asarray(next_conf, dtype=float) - np.asarray(current_conf, dtype=float))
    return bool(float(np.max(step_delta)) > float(threshold_rad))


def get_bar_feature_points(bar_box_dims: Sequence[float] = BAR_BOX_DIMS) -> List[np.ndarray]:
    """Return the 8 corner points of the bar bounding box, in the bar-local frame.

    These corners are the "feature points" used by the ``"feature"`` distance
    metric: projecting all eight through a pose and comparing corner-to-corner
    captures position and orientation differences together in one vector.

    Args:
        bar_box_dims (Sequence[float]): the bar bounding-box side lengths
            (width, depth, length) in metres.

    Returns:
        List[np.ndarray]: eight 3-vectors, one per box corner, in the bar frame.
    """
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
    """Quaternion that aligns the bar's right->left grasp axis with a target axis.

    From the two grasp targets (each a bar pose plus the tool0 pose that grasps
    it), find the bar-local vector pointing from the right grasp to the left
    grasp and return the rotation that turns that vector onto
    ``target_axis_in_mb`` (expressed in the mobile-base frame). Used to pick a
    natural home orientation for the held bar.

    Args:
        grasp_targets (Sequence[GraspTarget]): at least two
            ``(mobile_base_from_bar, mobile_base_from_tool0)`` pairs; index 0 is
            the left grasp, index 1 the right.
        target_axis_in_mb (np.ndarray): the direction, in the mobile-base frame,
            the right->left grasp vector should end up pointing along.

    Returns:
        Tuple[float, float, float, float]: the aligning quaternion as (x, y, z, w).

    Raises:
        ValueError: if fewer than two grasp targets are given.
    """
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
    """Pick a home bar pose by sweeping rotation about the bar's own long axis.

    Holds the bar position and base orientation fixed and spins the bar about
    its long axis in ``bar_axis_step_rad`` increments. Each candidate is scored
    by how well the average of the two tool0 Z-axes points along
    ``forward_direction`` (so the tools face "forward"). Candidates are ranked
    best-first; when an ``ik_validator`` is given, the best IK-feasible one among
    the top ``num_geometric_candidates`` is chosen.

    Args:
        grasp_targets (Sequence[GraspTarget]): two
            ``(mobile_base_from_bar, mobile_base_from_tool0)`` grasp pairs (left
            then right).
        mobile_base_from_bar (PoseLike): the fixed bar position + orientation to
            spin about (only the bar-axis rotation is optimized).
        forward_direction (np.ndarray): the mobile-base direction the tools
            should face; higher alignment scores better.
        ik_validator (Optional[Callable[[PoseLike], bool]]): optional test that
            returns True when a candidate bar pose is IK-feasible.
        num_geometric_candidates (int): how many top-scoring candidates to feed
            through ``ik_validator`` before giving up.
        allow_unvalidated_fallback (bool): when True, fall back to the best
            geometric candidate if none pass IK; when False, leave the result
            un-validated (the caller inspects ``ik_validated``).
        bar_axis_step_rad (float): angular step (radians) of the bar-axis sweep.

    Returns:
        Dict[str, Any]: the chosen home pose plus metadata, including
        ``mobile_base_from_bar_start`` (final bar pose), the two
        ``mobile_base_from_tool0_*_start`` tool poses, the grasp transforms,
        ``chosen_bar_axis_theta``, ``alignment_score``, and ``ik_validated``.

    Raises:
        ValueError: if fewer than two grasp targets are given, if
            ``forward_direction`` is zero, or if no candidate could be generated.
    """
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
    """Project the bar feature points through a pose into a flat world-coord vector.

    Args:
        pose (PoseLike): the bar pose ``((x, y, z), (qx, qy, qz, qw))``.
        feature_points (Sequence[np.ndarray]): bar-local corner points from
            ``get_bar_feature_points``.

    Returns:
        Optional[np.ndarray]: the concatenated world coordinates of every
        feature point (length ``3 * n_points``), or ``None`` if no feature
        points were supplied.
    """
    if not feature_points:
        return None
    pts = []
    for p_local in feature_points:
        p_world, _ = pp.multiply(pose, (p_local, [0, 0, 0, 1]))
        pts.append(np.asarray(p_world, dtype=float))
    return np.concatenate(pts, axis=0)


def pose_distance(pose1: PoseLike, pose2: PoseLike, dist_metric: str, feature_points: Sequence[np.ndarray]) -> float:
    """Distance between two bar poses under the chosen metric.

    Args:
        pose1 (PoseLike): first pose ``((x, y, z), quat_xyzw)``.
        pose2 (PoseLike): second pose, same format.
        dist_metric (str): ``"feature"`` compares projected feature-point vectors
            (blends position and orientation); anything else uses the straight
            position delta combined with the quaternion angle.
        feature_points (Sequence[np.ndarray]): bar-local corners, used only for
            the ``"feature"`` metric.

    Returns:
        float: the scalar distance.
    """
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
    """Total path length as the sum of consecutive pose-to-pose distances.

    Args:
        path_poses (Sequence[PoseLike]): ordered poses along the path.
        dist_metric (str): distance metric passed to ``pose_distance``.
        feature_points (Sequence[np.ndarray]): bar-local corners for the
            ``"feature"`` metric.

    Returns:
        float: the summed segment distances (0.0 for a path shorter than 2 poses).
    """
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
    """Find the "corner" indices of a dense pose path (its geometric control points).

    Walks the feature-vector path and marks an index whenever the travel
    direction changes by more than ``tolerance``. This thins a dense interpolated
    path down to the few waypoints that actually define its shape.

    Args:
        path_poses (Sequence[PoseLike]): the dense ordered poses.
        feature_points (Sequence[np.ndarray]): bar-local corners used to build
            each pose's feature vector.
        tolerance (float): minimum direction change (and minimum travel distance)
            for a point to count as a new segment.

    Returns:
        List[int]: indices into ``path_poses`` for the start, each detected
        inflection, and the end.
    """
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
    """Draw a target bar pose for one RRT iteration (goal-biased sampling).

    With probability ``goal_sample_prob`` returns the goal pose directly;
    otherwise samples a random pose in a box around the robot base (uniform XY
    within ``workspace_xy``, Z above the base up to ``workspace_z``, and a fully
    random orientation).

    Args:
        robot (int): PyBullet body id, used to read the base position the box is
            centred on.
        goal_pose (PoseLike): the goal bar pose returned on a goal-sample.
        rng (np.random.Generator): random source.
        goal_sample_prob (float): probability of returning ``goal_pose`` directly.
        workspace_xy (float): full XY side length (metres) of the sampling box.
        workspace_z (float): Z height (metres) of the sampling box above the base.

    Returns:
        Tuple[PoseLike, bool]: the sampled pose and a flag that is True when the
        pose is the goal pose.
    """
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
    """Return the tree node closest to ``target_pose`` under the chosen metric.

    Args:
        nodes (List[TreeNode]): the current tree nodes.
        target_pose (PoseLike): the pose to find the nearest node to.
        dist_metric (str): ``"feature"`` uses the cached feature-vector distance
            (fast); otherwise falls back to ``pose_distance``.
        feature_points (Sequence[np.ndarray]): bar-local corners for the feature
            metric.
        feature_vecs (Dict[int, np.ndarray]): cache mapping ``id(node)`` to that
            node's feature vector (feature metric only).

    Returns:
        TreeNode: the nearest node.
    """
    if dist_metric == "feature":
        target_vec = pose_to_feature_vec(target_pose, feature_points)
        if target_vec is not None:
            return min(
                nodes,
                key=lambda node: float(np.linalg.norm(feature_vecs[id(node)] - target_vec)),
            )
    return min(nodes, key=lambda node: pose_distance(node.config, target_pose, dist_metric, feature_points))


def export_tree(nodes: List[TreeNode]) -> Dict[str, List[List[float]]]:
    """Flatten a tree into a serializable points + edges dict for debugging.

    Args:
        nodes (List[TreeNode]): the tree nodes; each node's ``config`` is a pose
            whose position becomes a point.

    Returns:
        Dict[str, List[List[float]]]: ``{"points": [[x, y, z], ...], "edges":
        [[parent_idx, child_idx], ...]}`` where the edge indices refer into
        ``points``.
    """
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


def build_cfab_pose_collision_fn(planner, start_state, active_bar_id: str) -> Callable[[PoseLike], bool]:
    """Return a closure ``(world_from_bar_sample: PoseLike) -> bool``.

    True == colliding. The closure clones ``start_state`` per call, overrides
    the active bar's ``attachment_frame`` so cfab's ``set_robot_cell_state``
    lands the bar at the sampled world pose (the bar stays attached, so cfab's
    CC.4 still fires with ``touch_bodies`` respected), and calls
    ``planner.check_collision`` with all CC steps skipped except CC.4.

    Math: ``set_robot_cell_state`` places an attached RB at
    ``link_pose_world * attachment_frame``. To land it at a sampled world
    pose we override ``attachment_frame = inverse(link_pose_world) * sample``.
    The link pose is captured ONCE at start_state (the robot configuration
    does not change between pose samples).

    Note on joints: joints in this cell attach to robot tool0 links, NOT to
    the bar. When the bar is moved to a sampled pose, joints stay anchored
    to the robot — their cfab CC checks reflect the joints' current
    tool0-anchored positions, not their hypothetical positions had they
    followed the bar. That's fine for pose-space pre-reject (bar mesh vs
    environment); joints get their real check at the joint-space layer.
    """
    from compas.geometry import Frame
    from compas_fab.backends import CollisionCheckError

    attached_link_name = start_state.rigid_body_states[active_bar_id].attached_to_link
    if not attached_link_name:
        raise ValueError(
            f"{active_bar_id!r} is not attached to any link in start_state; "
            "pose-space override needs an attached link to compute attachment_frame."
        )
    robot_puid = planner.client.robot_puid
    attached_link_id = planner.client.robot_link_puids[attached_link_name]
    link_pose_world_at_start = pp.get_link_pose(robot_puid, attached_link_id)
    inv_link = pp.invert(link_pose_world_at_start)

    pose_cc_opts = {
        "_skip_cc1": True, "_skip_cc2": True, "_skip_cc3": True,
        "_skip_cc4": False, "_skip_cc5": True, "verbose": False,
    }

    def _pose_collision_fn(world_from_bar_sample: PoseLike) -> bool:
        """Return True if the bar collides when placed at ``world_from_bar_sample``.

        Clones the captured start state, overrides the bar's attachment frame so
        cfab lands it at the sampled world pose, and runs only the CC.4 step of
        ``planner.check_collision``.

        Args:
            world_from_bar_sample (PoseLike): the world bar pose to test.

        Returns:
            bool: True if colliding, False otherwise.
        """
        attach_pose = pp.multiply(inv_link, world_from_bar_sample)
        attach_frame = Frame.from_quaternion(
            [attach_pose[1][3], attach_pose[1][0], attach_pose[1][1], attach_pose[1][2]],
            point=list(attach_pose[0]),
        )
        s = start_state.copy()
        s.rigid_body_states[active_bar_id].attachment_frame = attach_frame
        try:
            planner.check_collision(s, options=pose_cc_opts)
        except CollisionCheckError:
            return True
        return False

    return _pose_collision_fn


def _noop_pose_collision_fn(pose: PoseLike) -> bool:
    """Collision stub that always reports "no collision" (used when disabled).

    Args:
        pose (PoseLike): ignored.

    Returns:
        bool: always False.
    """
    return False


def get_disabled_collisions_from_link_names(
    robot: int,
    link_name_pairs: Sequence[Tuple[str, str]],
) -> List[Tuple[int, int]]:
    """Translate link-name pairs into PyBullet (link-index, link-index) pairs.

    Pairs whose links are missing on the robot are silently skipped.

    Args:
        robot (int): PyBullet body id.
        link_name_pairs (Sequence[Tuple[str, str]]): pairs of link names whose
            mutual collision should be disabled.

    Returns:
        List[Tuple[int, int]]: the resolved link-index pairs.
    """
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
    """Build a joint-space collision predicate for the dual-arm robot + held bar.

    Loads the Husky URDF/SRDF to seed the disabled-collision (self-collision)
    set, attaches the bar to the left tool, and additionally disables collisions
    between the grasp-mask links (wrists/tools) and the bar so the grasp contact
    itself is not flagged. Returns the standard pybullet_planning collision
    function.

    Args:
        robot (int): PyBullet body id.
        arm_joints (Sequence[int]): the movable arm joint indices the predicate
            reads.
        obstacle_bodies (Sequence[int]): body ids to check the robot against.
        tool_link_left (int): link index the bar is attached to.
        bar_body (int): PyBullet body id of the held bar.
        grasp_bar_from_left (PoseLike): the left-tool grasp transform
            (bar-from-tool), used to build the attachment.

    Returns:
        Callable[..., bool]: a predicate ``(conf) -> bool`` that is True when the
        configuration is in collision.
    """
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


def get_joint_collision_fn_cfab(
    cfab_session,
    template_state,
    *,
    joint_names_12: Optional[Sequence[str]] = None,
    cc_options: Optional[dict] = None,
) -> Callable[..., bool]:
    """cfab equivalent of get_joint_collision_fn for Stage 3 RRT.

    Caller must ensure template_state has the held bar attached to the left
    tool (RigidBodyState.attached_to_tool/link set) with STAGE3_GRASP_MASK_LINKS
    listed in its touch_links so the grasp contact is not flagged as a
    collision.
    """
    from husky_assembly_teleop.cfab_collision_adapter import make_cfab_collision_fn

    return make_cfab_collision_fn(
        cfab_session, template_state,
        joint_names_12=joint_names_12,
        cc_options=cc_options,
    )


def goal_pose_reached(pose: PoseLike, goal_pose: PoseLike, position_res: float, rotation_res: float) -> bool:
    """Whether ``pose`` is within position + rotation tolerance of ``goal_pose``.

    Args:
        pose (PoseLike): the pose to test.
        goal_pose (PoseLike): the goal pose.
        position_res (float): position tolerance in metres.
        rotation_res (float): orientation tolerance in radians.

    Returns:
        bool: True if within both tolerances.
    """
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
    """Solve IK for one arm and splice the result into the full 12-DOF config.

    Seeds pybullet's IK from ``full_seed_conf``, solves for ``tool_link`` to
    reach ``target_tool_pose``, writes only the ``arm_slice`` joints back, and
    verifies the achieved tool pose is within tolerance before accepting.

    Args:
        robot (int): PyBullet body id.
        arm_joints (Sequence[int]): all arm joint indices (both arms) the seed is
            applied to.
        tool_link (int): the tool link index to drive to the target.
        full_seed_conf (FullConf): the 12-DOF warm-start configuration.
        target_tool_pose (PoseLike): the world tool0 pose to reach.
        arm_slice (slice): the slice of the 12-vector this arm owns (0:6 left,
            6:12 right).
        use_angle_normalization (bool): whether to normalize the solved angles.

    Returns:
        Optional[FullConf]: the updated 12-DOF configuration, or ``None`` if IK
        returned too few values or missed the target beyond 1e-4.
    """
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
    """Check that a 12-DOF config really holds the bar at ``bar_pose`` with both arms.

    Sets the configuration, then confirms (a) each tool0 reaches its grasp
    target, and (b) the bar pose implied by each tool through the inverse grasp
    matches ``bar_pose`` and the two arms agree with each other -- i.e. the rigid
    dual-arm grasp is self-consistent.

    Args:
        robot (int): PyBullet body id.
        arm_joints (Sequence[int]): arm joint indices set from ``full_conf``.
        tool_link_left (int): left tool0 link index.
        tool_link_right (int): right tool0 link index.
        full_conf (FullConf): the 12-DOF configuration to validate.
        bar_pose (PoseLike): the intended world bar pose.
        grasp_bar_from_left (PoseLike): left grasp transform (bar-from-tool).
        grasp_bar_from_right (PoseLike): right grasp transform (bar-from-tool).
        pos_tolerance (float): position tolerance in metres.
        ori_tolerance (float): orientation tolerance in radians.

    Returns:
        bool: True if both tools reach their targets and the grasp is consistent.
    """
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
    """Solve both arms so they rigidly hold the bar at ``bar_pose``.

    Tries two solve orders (right-then-left, then left-then-right): within an
    order each arm's IK is warm-started from the running configuration so the
    second arm solves around the first. The first order that yields a validated,
    consistent dual-arm grasp (see ``validate_dual_arm_bar_pose``) wins.

    Args:
        robot (int): PyBullet body id.
        arm_joints (Sequence[int]): arm joint indices.
        tool_link_left (int): left tool0 link index.
        tool_link_right (int): right tool0 link index.
        bar_pose (PoseLike): the world bar pose both arms must grasp.
        grasp_bar_from_left (PoseLike): left grasp transform (bar-from-tool).
        grasp_bar_from_right (PoseLike): right grasp transform (bar-from-tool).
        seed_conf (FullConf): the 12-DOF warm-start configuration.
        use_angle_normalization (bool): whether to normalize the solved angles.

    Returns:
        Optional[FullConf]: a validated 12-DOF configuration, or ``None`` if
        neither solve order produced a consistent grasp.
    """
    # * Backend dispatch: analytical ssik (the default) replaces the PyBullet
    # * gradient descent below. Set HUSKY_IK_BACKEND=gradient to get the old path.
    if ssik_ik.ik_backend() == "ssik":
        return _solve_dual_arm_pose_ik_ssik(
            robot=robot,
            arm_joints=arm_joints,
            tool_link_left=tool_link_left,
            tool_link_right=tool_link_right,
            bar_pose=bar_pose,
            grasp_bar_from_left=grasp_bar_from_left,
            grasp_bar_from_right=grasp_bar_from_right,
            seed_conf=seed_conf,
            use_angle_normalization=use_angle_normalization,
        )
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


def _solve_dual_arm_pose_ik_ssik(
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
    """ssik variant of :func:`solve_dual_arm_pose_ik` (warm, per-waypoint).

    Each arm is an independent 6R chain rooted at its own base link, so there is
    no right-then-left / left-then-right solve-order dance: both arms are solved
    directly, each taking its branch nearest (raw distance, after 2*pi
    re-branching) to the seed -- which is exactly what keeps the joint path
    continuous through the RRT / path reconstruction.

    Args:
        robot (int): PyBullet body id.
        arm_joints (Sequence[int]): arm joint indices (both arms).
        tool_link_left (int): left tool0 link index.
        tool_link_right (int): right tool0 link index.
        bar_pose (PoseLike): the world bar pose both arms must grasp.
        grasp_bar_from_left (PoseLike): left grasp transform (bar-from-tool).
        grasp_bar_from_right (PoseLike): right grasp transform (bar-from-tool).
        seed_conf (FullConf): the 12-DOF warm-start configuration.
        use_angle_normalization (bool): whether to normalize the solved angles.

    Returns:
        Optional[FullConf]: a validated 12-DOF configuration, or ``None`` when
        either arm cannot reach its grasp target.
    """
    seed = np.asarray(maybe_normalize_angles(seed_conf, use_angle_normalization), dtype=float)
    conf = seed.copy()
    targets = {
        "left": pp.multiply(bar_pose, grasp_bar_from_left),
        "right": pp.multiply(bar_pose, grasp_bar_from_right),
    }
    for arm in ("left", "right"):
        arm_slice = ssik_ik.ARM_SLICE[arm]
        # All branches come back sorted nearest-to-seed; take the closest one.
        # ! allow_rescue=False: this runs per RRT waypoint, so an unreachable
        # ! pose must fail fast (~1 ms) instead of burning ~40 ms on ssik's
        # ! numeric rescue -- whose loose LM solutions can also branch-flip.
        branches = ssik_ik.ssik_arm_branches(
            robot, arm, targets[arm], seed[arm_slice], allow_rescue=False
        )
        if not branches:
            return None  # this arm cannot reach its tool0 target
        conf[arm_slice] = branches[0]
    # ! Keep the PyBullet FK ground-truth gate the gradient path used: it guards
    # ! against any joint-order / frame-convention mismatch with ssik.
    if not validate_dual_arm_bar_pose(
        robot=robot,
        arm_joints=arm_joints,
        tool_link_left=tool_link_left,
        tool_link_right=tool_link_right,
        full_conf=conf,
        bar_pose=bar_pose,
        grasp_bar_from_left=grasp_bar_from_left,
        grasp_bar_from_right=grasp_bar_from_right,
    ):
        return None
    return maybe_normalize_angles(conf, use_angle_normalization)


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
    """Dual-arm IK for a single pose, with random restarts and optional collision reject.

    Attempt 0 uses ``seed_conf``; later attempts resample a fully random seed.
    Each attempt calls ``solve_dual_arm_pose_ik``; the first solution that also
    passes the optional ``collision_fn`` is returned. Used to find a feasible
    start/goal joint config for the grasped bar.

    Args:
        robot (int): PyBullet body id.
        arm_joints (Sequence[int]): arm joint indices.
        tool_link_left (int): left tool0 link index.
        tool_link_right (int): right tool0 link index.
        bar_pose (PoseLike): the world bar pose to grasp.
        grasp_bar_from_left (PoseLike): left grasp transform (bar-from-tool).
        grasp_bar_from_right (PoseLike): right grasp transform (bar-from-tool).
        seed_conf (FullConf): warm-start for attempt 0.
        rng (np.random.Generator): random source for the restart seeds.
        max_attempts (int): number of seeds to try (>= 1).
        use_angle_normalization (bool): whether to normalize the solved angles.
        collision_fn (Optional[Callable[[np.ndarray], bool]]): optional predicate;
            solutions for which it returns True are rejected.
        **_unused_kwargs: ignored (lets callers pass a shared kwargs bag).

    Returns:
        Optional[FullConf]: the first collision-free solved configuration, or
        ``None`` if every attempt failed.
    """
    # * Backend dispatch: with ssik the whole random-restart loop is pointless --
    # * the analytical solver enumerates EVERY branch once, deterministically, so
    # * ``rng`` / ``max_attempts`` are intentionally ignored on that path.
    if ssik_ik.ik_backend() == "ssik":
        return _solve_endpoint_dual_arm_ik_ssik(
            robot=robot,
            arm_joints=arm_joints,
            tool_link_left=tool_link_left,
            tool_link_right=tool_link_right,
            bar_pose=bar_pose,
            grasp_bar_from_left=grasp_bar_from_left,
            grasp_bar_from_right=grasp_bar_from_right,
            seed_conf=seed_conf,
            collision_fn=collision_fn,
            use_angle_normalization=use_angle_normalization,
        )
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


def _solve_endpoint_dual_arm_ik_ssik(
    robot: int,
    arm_joints: Sequence[int],
    tool_link_left: int,
    tool_link_right: int,
    bar_pose: PoseLike,
    grasp_bar_from_left: PoseLike,
    grasp_bar_from_right: PoseLike,
    seed_conf: FullConf,
    collision_fn: Optional[Callable[[np.ndarray], bool]] = None,
    use_angle_normalization: bool = DEFAULT_USE_ANGLE_NORMALIZATION,
) -> Optional[FullConf]:
    """ssik variant of :func:`solve_endpoint_dual_arm_ik` (cold, single pose).

    Enumerates every analytical branch per arm (up to 8), forms all left/right
    combinations (up to 64), orders them by distance to ``seed_conf``, and
    returns the first combination that passes the FK validation and the
    optional collision reject. Fully deterministic -- no random restarts.

    Args:
        robot (int): PyBullet body id.
        arm_joints (Sequence[int]): arm joint indices.
        tool_link_left (int): left tool0 link index.
        tool_link_right (int): right tool0 link index.
        bar_pose (PoseLike): the world bar pose to grasp.
        grasp_bar_from_left (PoseLike): left grasp transform (bar-from-tool).
        grasp_bar_from_right (PoseLike): right grasp transform (bar-from-tool).
        seed_conf (FullConf): preferred configuration; candidates are tried
            nearest-to-it first.
        collision_fn (Optional[Callable[[np.ndarray], bool]]): optional
            predicate; candidates for which it returns True are rejected.
        use_angle_normalization (bool): whether to normalize the solved angles.

    Returns:
        Optional[FullConf]: the best surviving configuration, or ``None`` when
        no branch combination is reachable + valid + collision-free.
    """
    seed = np.asarray(seed_conf, dtype=float)
    branches_left = ssik_ik.ssik_arm_branches(
        robot, "left", pp.multiply(bar_pose, grasp_bar_from_left), seed[0:6], max_solutions=8
    )
    branches_right = ssik_ik.ssik_arm_branches(
        robot, "right", pp.multiply(bar_pose, grasp_bar_from_right), seed[6:12], max_solutions=8
    )
    if not branches_left or not branches_right:
        return None  # at least one arm cannot reach its grasp target at all
    # * Cross-product of per-arm branches, tried nearest-to-seed first so the
    # * accepted endpoint stays close to where the robot already is.
    candidates = [
        (float(np.linalg.norm(np.concatenate([ql, qr]) - seed)), np.concatenate([ql, qr]))
        for ql in branches_left
        for qr in branches_right
    ]
    candidates.sort(key=lambda pair: pair[0])
    for _distance, conf in candidates:
        conf = maybe_normalize_angles(conf, use_angle_normalization)
        # ! PyBullet FK ground-truth gate (same role as in the gradient path).
        if not validate_dual_arm_bar_pose(
            robot=robot,
            arm_joints=arm_joints,
            tool_link_left=tool_link_left,
            tool_link_right=tool_link_right,
            full_conf=conf,
            bar_pose=bar_pose,
            grasp_bar_from_left=grasp_bar_from_left,
            grasp_bar_from_right=grasp_bar_from_right,
        ):
            continue
        if collision_fn is not None and collision_fn(np.asarray(conf, dtype=float)):
            continue
        return conf
    return None


def _grid_in_box(
    box: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    step: float,
) -> List[Tuple[float, float, float]]:
    """Enumerate a regular 3D grid of points spanning an axis-aligned box.

    Args:
        box (Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]):
            the ``((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi))`` bounds.
        step (float): grid spacing along every axis.

    Returns:
        List[Tuple[float, float, float]]: the grid points (upper bounds included
        within half a step).
    """
    (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = box
    xs = np.arange(x_lo, x_hi + 0.5 * step, step)
    ys = np.arange(y_lo, y_hi + 0.5 * step, step)
    zs = np.arange(z_lo, z_hi + 0.5 * step, step)
    return [(float(x), float(y), float(z)) for x in xs for y in ys for z in zs]


def home_bar_anchor_pose_mb(
    mb_from_bar_goal: PoseLike,
    grasp_bar_from_left: PoseLike,
    grasp_bar_from_right: PoseLike,
    bar_quat_override: Optional[Tuple[float, float, float, float]] = None,
    anchor: str = "horizontal",
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """A "home" bar anchor pose (mobile-base frame) for start derivation.

    Orientation comes from the grasp geometry (``bar_orientation_from_grasps``
    aligned to the anchor's ``bar_axis_mb``) unless ``bar_quat_override`` is
    given; position anchors the GRASP MIDPOINT (not the bar frame origin --
    some bar frames live at one grasp end) at the anchor's ``midpoint_mb``.
    ``derive_constrained_start`` sweeps position deltas around this anchor, and
    the ssik goal/start branch pairing in api.py evaluates branch sets at it.

    Args:
        mb_from_bar_goal (PoseLike): the goal bar pose in the mobile-base frame.
        grasp_bar_from_left (PoseLike): left grasp transform (bar-from-tool).
        grasp_bar_from_right (PoseLike): right grasp transform (bar-from-tool).
        bar_quat_override (Optional[Tuple]): use this orientation (xyzw) for the
            home bar instead of the canonical one. The anchored position is
            recomputed for it (the grasp midpoint moves with the orientation).
        anchor (str): which ``HOME_BAR_ANCHORS`` carry mode to anchor at
            (default "horizontal", the original front carry).

    Returns:
        Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        the anchor as a ``(position, quaternion_xyzw)`` pair in the mobile-base
        frame.
    """
    anchor_spec = HOME_BAR_ANCHORS[anchor]
    mb_from_tool0_left_goal = pp.multiply(mb_from_bar_goal, grasp_bar_from_left)
    mb_from_tool0_right_goal = pp.multiply(mb_from_bar_goal, grasp_bar_from_right)
    grasp_targets_mb = [
        (mb_from_bar_goal, mb_from_tool0_left_goal),
        (mb_from_bar_goal, mb_from_tool0_right_goal),
    ]
    if bar_quat_override is not None:
        home_bar_quat = tuple(bar_quat_override)
    else:
        home_bar_quat = bar_orientation_from_grasps(
            grasp_targets_mb, target_axis_in_mb=anchor_spec["bar_axis_mb"]
        )

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
    base_pos_mb = np.asarray(anchor_spec["midpoint_mb"], dtype=float) - midpoint_in_mb
    return tuple(base_pos_mb.tolist()), home_bar_quat


def home_bar_anchor_variants(
    mb_from_bar_goal: PoseLike,
    grasp_bar_from_left: PoseLike,
    grasp_bar_from_right: PoseLike,
    anchors: Optional[Sequence[str]] = None,
) -> List[Tuple[str, Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]]:
    """Candidate home anchor poses: canonical orientations plus rotated variants.

    Why: with real joint limits, the IK branch sheet that holds the bar at the
    GOAL pose may simply not extend to a canonical home ORIENTATION (measured
    on the hard B226 case: every goal branch is ~176 deg away from every
    canonical-home branch). The home pose is ours to choose -- M1's start is
    derived, and M0 drives the arms to it -- so when a canonical orientation
    is unreachable-in-branch, try the same anchor with the bar rotated:

      * ``roll`` -- about the bar's own long axis (local Z): re-poses the
        wrists while the bar stays put visually; the cheapest branch changer.
      * ``yaw`` -- about the mobile base's vertical axis: swings the bar
        heading; changes the shoulder/elbow posture.

    With several carry anchors selected, the per-anchor lists are interleaved
    round-robin by rotation rank: EVERY anchor's canonical pose comes before
    ANY rotated variant, and rotations stay smallest-first globally -- so an
    early-exiting caller reaches a genuinely different carry mode before
    burning time on large rotations of the first one. Labels are prefixed with
    the anchor, e.g. ``"horizontal/canonical"``, ``"back/yaw+30"``.

    Args:
        mb_from_bar_goal (PoseLike): the goal bar pose in the mobile-base frame.
        grasp_bar_from_left (PoseLike): left grasp transform (bar-from-tool).
        grasp_bar_from_right (PoseLike): right grasp transform (bar-from-tool).
        anchors (Optional[Sequence[str]]): ``HOME_BAR_ANCHORS`` selection
            (None/"all" = every anchor, see ``resolve_home_anchors``).

    Returns:
        List[Tuple[str, Tuple[pos, quat]]]: ``(label, (position, quat_xyzw))``
        anchor candidates in the mobile-base frame, canonicals first.
    """
    anchor_labels = resolve_home_anchors(anchors)

    def _variants_for_anchor(anchor_label: str):
        """One anchor's ordered variant list: canonical, then rotations."""
        canonical_pos, canonical_quat = home_bar_anchor_pose_mb(
            mb_from_bar_goal, grasp_bar_from_left, grasp_bar_from_right,
            anchor=anchor_label,
        )
        variants: List[Tuple[str, Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]] = [
            (f"{anchor_label}/canonical", (canonical_pos, canonical_quat)),
        ]

        def _anchor_for(quat):
            """Re-anchor the grasp midpoint for a rotated orientation."""
            return home_bar_anchor_pose_mb(
                mb_from_bar_goal, grasp_bar_from_left, grasp_bar_from_right,
                bar_quat_override=quat, anchor=anchor_label,
            )

        # Smallest rotations first; roll before yaw at equal magnitude (wrist-limit
        # breaks are the common failure and roll targets exactly those).
        for magnitude_deg in (30, 60, 90, 120, 150, 180):
            angle = float(np.deg2rad(magnitude_deg))
            for sign in (1.0, -1.0):
                if magnitude_deg == 180 and sign < 0:
                    continue  # +180 and -180 are the same rotation
                # Roll: rotate about the bar's LOCAL long axis (post-multiply).
                if magnitude_deg <= 90:
                    roll_quat = pybullet.getQuaternionFromAxisAngle((0.0, 0.0, 1.0), sign * angle)
                    quat = pp.multiply(((0.0, 0.0, 0.0), canonical_quat), ((0.0, 0.0, 0.0), roll_quat))[1]
                    variants.append((f"{anchor_label}/roll{int(sign * magnitude_deg):+d}", _anchor_for(quat)))
                # Yaw: rotate about the mobile base's vertical axis (pre-multiply).
                yaw_quat = pybullet.getQuaternionFromAxisAngle((0.0, 0.0, 1.0), sign * angle)
                quat = pp.multiply(((0.0, 0.0, 0.0), yaw_quat), ((0.0, 0.0, 0.0), canonical_quat))[1]
                variants.append((f"{anchor_label}/yaw{int(sign * magnitude_deg):+d}", _anchor_for(quat)))
        return variants

    per_anchor = [_variants_for_anchor(label) for label in anchor_labels]
    # Round-robin interleave by rotation rank (see docstring). With a single
    # anchor this is exactly that anchor's own list, i.e. the old ordering.
    interleaved: List[Tuple[str, Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]] = []
    for rank in range(max(len(variants) for variants in per_anchor)):
        for variants in per_anchor:
            if rank < len(variants):
                interleaved.append(variants[rank])
    return interleaved


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
    shuffle_deltas: bool = False,
    anchors: Optional[Sequence[str]] = None,
    joint_collision_fn: Optional[Callable[[FullConf], bool]] = None,
) -> Tuple[Optional[PoseLike], Optional[np.ndarray]]:
    """Derive a constraint-satisfying start (bar pose, joint conf).

    When ``shuffle_deltas`` is True, the bar-position grid is randomly permuted
    (driven by ``random_seed``) instead of sorted by distance to the home anchor.
    This lets the caller derive *different* starts on different seeds — useful
    for hard problems where the first-IK-feasible home cannot reach the goal.

    ``anchors`` selects which ``HOME_BAR_ANCHORS`` carry modes to sample
    (None/"all" = every anchor in priority order). Deltas are swept outermost
    and anchors innermost, so every anchor's near-home poses are tried before
    any anchor's far ones; the first IK-validated candidate wins.

    ``joint_collision_fn`` (``conf_12 -> bool``, True == colliding) lets the
    caller inject a collision predicate. When given it is used as-is; otherwise
    one is built from the husky URDF/SRDF via :func:`get_joint_collision_fn`
    (the standalone-runner path). Callers driving a compas_fab planner pass a
    cfab-backed predicate so no URDF file on disk is required.
    """
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

    # "Home" bar anchors (position + orientation in the mobile-base frame) the
    # delta sweep expands around, one per selected carry mode. Shared helper so
    # the ssik branch-pairing in api.py can reason about the same anchor poses.
    anchor_labels = resolve_home_anchors(anchors)
    anchor_poses: Dict[str, Tuple[np.ndarray, Tuple[float, float, float, float]]] = {}
    for anchor_label in anchor_labels:
        anchor_pos, anchor_quat = home_bar_anchor_pose_mb(
            mb_from_bar_goal, grasp_bar_from_left, grasp_bar_from_right,
            anchor=anchor_label,
        )
        anchor_poses[anchor_label] = (np.asarray(anchor_pos, dtype=float), anchor_quat)

    rng = np.random.default_rng(random_seed)
    if joint_collision_fn is None and bar_body is not None:
        joint_collision_fn = get_joint_collision_fn(
            robot=robot,
            arm_joints=arm_joints,
            obstacle_bodies=list(obstacles),
            tool_link_left=tool_link_left,
            bar_body=bar_body,
            grasp_bar_from_left=grasp_bar_from_left,
        )

    deltas = _grid_in_box(bar_sweep_box, bar_sweep_step)
    if shuffle_deltas:
        # Fully random permutation. Different random_seed -> different start.
        rng.shuffle(deltas)
    else:
        deltas = sorted(deltas, key=lambda d: float(np.linalg.norm(d)))
    saved_bar_pose = pp.get_pose(bar_body) if bar_body is not None else None
    found: Dict[str, Any] = {"world_from_bar": None, "conf": None}
    chosen_ctx: Optional[Dict[str, Any]] = None

    def ik_validator(bar_pose_mb, _found=found):
        """Return True if a collision-free dual-arm grasp exists for this bar pose.

        Converts the mobile-base bar pose to world, runs
        ``solve_endpoint_dual_arm_ik``, and on success stashes the world
        bar pose + solved config into the captured ``found`` dict.

        Args:
            bar_pose_mb (PoseLike): candidate bar pose in the mobile-base frame.
            _found (dict): captured accumulator for the winning pose/config.

        Returns:
            bool: True if an IK solution was found (and recorded).
        """
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

    with pp.WorldSaver():
        # Deltas outermost, anchors innermost: every anchor's poses near its
        # home get tried before any anchor's far ones. First IK-validated
        # candidate wins (each candidate also sweeps rotation about the bar
        # axis inside auto_compute_home_bar_pose).
        for delta in deltas:
            for anchor_label in anchor_labels:
                base_pos_mb, home_bar_quat = anchor_poses[anchor_label]
                mb_from_bar_candidate = (
                    tuple((base_pos_mb + np.asarray(delta, dtype=float)).tolist()),
                    home_bar_quat,
                )
                ctx = auto_compute_home_bar_pose(
                    grasp_targets_mb,
                    mobile_base_from_bar=mb_from_bar_candidate,
                    forward_direction=HOME_BAR_ANCHORS[anchor_label]["forward_mb"],
                    ik_validator=ik_validator,
                    num_geometric_candidates=num_geometric_candidates,
                    bar_axis_step_rad=bar_axis_step_rad,
                    allow_unvalidated_fallback=False,
                )
                if ctx.get("ik_validated", False):
                    chosen_ctx = ctx
                    break
            if chosen_ctx is not None:
                break

    if saved_bar_pose is not None:
        pp.set_pose(bar_body, saved_bar_pose)

    if chosen_ctx is None or found["conf"] is None:
        logger.warning(
            "derive_constrained_start: no collision-free home pose across %d deltas x %d anchors (%s, kinematic_only=%s)",
            len(deltas),
            len(anchor_labels),
            "/".join(anchor_labels),
            bar_body is None,
        )
        return None, None

    world_from_bar_start = pp.multiply(world_from_mobile_base, chosen_ctx["mobile_base_from_bar_start"])
    return world_from_bar_start, found["conf"]


def derive_constrained_start_tracked(
    robot: int,
    arm_joints: Sequence[int],
    tool_link_left: int,
    tool_link_right: int,
    grasp_bar_from_left: PoseLike,
    grasp_bar_from_right: PoseLike,
    world_from_bar_goal: PoseLike,
    goal_conf: Sequence[float],
    *,
    world_from_mobile_base: Optional[PoseLike] = None,
    bar_sweep_box: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (-0.3, 0.3),
        (-0.3, 0.3),
        (-0.3, 0.3),
    ),
    bar_sweep_step: float = 0.1,
    position_res: float = 0.01,
    rotation_res: float = 0.025,
    screen_position_res: float = 0.01,
    screen_rotation_res: float = 0.025,
    joint_continuity_threshold_rad: float = DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
    joint_collision_fn: Optional[Callable[[FullConf], bool]] = None,
    use_angle_normalization: bool = DEFAULT_USE_ANGLE_NORMALIZATION,
    anchors: Optional[Sequence[str]] = None,
) -> Tuple[Optional[PoseLike], Optional[np.ndarray], Dict[str, Any]]:
    """Derive M1's start by TRACKING the goal conf backward to a home pose.

    Why this exists (ssik / joint-limits era): the cold endpoint IK in
    :func:`derive_constrained_start` picks the home-pose branch nearest the
    goal seed among those that are collision-free -- but with real joint limits
    the near branches are often colliding or clipped, so the sweep silently
    accepts a FAR branch and no continuous in-limit motion connects start to
    goal (the RRT then dies of continuity stops). This variant removes that
    failure mode by construction: walk the interpolated SE(3) bar segment from
    the GOAL pose (at ``goal_conf``) to each candidate home pose with
    warm-seeded per-waypoint IK. If tracking survives the continuity gate the
    arrival config is on the goal's own branch sheet, so a continuous joint
    path start->goal exists along that segment.

    Collisions ALONG the segment are deliberately ignored -- the RRT's job is
    to route around them. Only the arrival (start) config must be
    collision-free, since it becomes a plan endpoint.

    Three escalations for hard cases (all used automatically):
      1. FREE CORRIDOR: a fully-tracked candidate whose EVERY waypoint is also
         collision-free is a finished M1 plan by itself -- ``info['corridor']``
         then carries the whole ``(poses, confs)`` path (goal->home order) and
         the caller can skip the RRT entirely (direct-connect-first).
      2. ORIENTATION variants: when no home pose at the canonical orientation
         tracks, the same anchor is retried with the bar rolled about its own
         axis / yawed about the base vertical (``home_bar_anchor_variants``) --
         the goal's branch sheet may reach a rotated home even when the
         canonical one is out of reach within joint limits.
      3. PARTIAL start: when nothing fully tracks, the farthest-reaching
         broken walk donates its last collision-free waypoint as the start --
         still on the goal's branch sheet, just closer to the goal; M0 (which
         drives the arms to M1's start) absorbs the difference. ``info`` then
         carries ``partial=True`` and the start-to-goal distance.

    Args:
        robot (int): PyBullet body id.
        arm_joints (Sequence[int]): arm joint indices (both arms).
        tool_link_left (int): left tool0 link index.
        tool_link_right (int): right tool0 link index.
        grasp_bar_from_left (PoseLike): left grasp transform (bar-from-tool).
        grasp_bar_from_right (PoseLike): right grasp transform (bar-from-tool).
        world_from_bar_goal (PoseLike): the goal bar pose.
        goal_conf (Sequence[float]): the solved 12-DOF goal configuration the
            backward track starts from.
        world_from_mobile_base (Optional[PoseLike]): robot base pose (None =
            identity), used to place the canonical home anchor.
        bar_sweep_box: position-delta box swept around the home anchor
            (same convention as :func:`derive_constrained_start`).
        bar_sweep_step (float): grid spacing of the delta sweep, meters.
        position_res (float): linear interpolation step of the DELIVERED
            corridor, meters (match the RRT's resolution).
        rotation_res (float): angular step of the delivered corridor, radians.
        screen_position_res (float): coarse linear step used to SCREEN
            candidates, meters. Tracking every candidate at a very fine
            requested resolution burns the whole time budget on a handful of
            deltas (0.002 rad steps mean 1000+ IK solves per candidate), so
            candidates are screened at this coarser step and only a winning
            corridor is re-tracked at ``position_res``/``rotation_res`` for
            delivery. Screening is never finer than the requested resolution
            (the coarser of the two wins).
        screen_rotation_res (float): angular twin of ``screen_position_res``,
            radians.
        joint_continuity_threshold_rad (float): max per-joint step between
            consecutive tracked waypoints before the track counts as broken.
        joint_collision_fn (Optional[Callable[[FullConf], bool]]): predicate
            for the ARRIVAL config only (True = colliding).
        use_angle_normalization (bool): forwarded to the per-waypoint IK.
        anchors (Optional[Sequence[str]]): ``HOME_BAR_ANCHORS`` carry-mode
            selection (None/"all" = every anchor). The time budget is split
            evenly across the selected anchors so a hopeless first anchor
            cannot starve the others.

    Returns:
        Tuple[Optional[PoseLike], Optional[np.ndarray], Dict[str, Any]]:
        ``(world_from_bar_start, start_conf, info)`` on success, or
        ``(None, None, info)`` when no delta yields a trackable, collision-free
        start. ``info`` carries counters for diagnosis.
    """
    identity_pose = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    if world_from_mobile_base is None:
        world_from_mobile_base = identity_pose

    mb_from_bar_goal = pp.multiply(pp.invert(world_from_mobile_base), world_from_bar_goal)
    # Home anchor candidates, interleaved across the selected carry anchors:
    # every anchor's canonical orientation first, then rotated variants (roll
    # about the bar axis / yaw about the base vertical) smallest-first. The
    # rotations are the escape hatch for goals whose branch sheet cannot reach
    # a canonical home orientation within joint limits (see
    # home_bar_anchor_variants).
    anchor_labels = resolve_home_anchors(anchors)
    anchor_variants = home_bar_anchor_variants(
        mb_from_bar_goal, grasp_bar_from_left, grasp_bar_from_right,
        anchors=anchor_labels,
    )

    # Nearest-to-anchor deltas first, exactly like the cold sweep. Non-canonical
    # orientations only sweep the nearest deltas -- the orientation is the knob
    # being explored there, not the position.
    deltas = sorted(_grid_in_box(bar_sweep_box, bar_sweep_step), key=lambda d: float(np.linalg.norm(d)))
    max_variant_deltas = 60

    # Soft time budget over the WHOLE sweep: with ~13 orientations x dozens of
    # deltas per anchor the worst case is minutes; most breaks happen within a
    # few waypoints so typical cost is far lower. The budget is split EVENLY
    # across the selected anchors (no roll-over, so the total stays bounded and
    # a hopeless first anchor cannot starve the others); a single selected
    # anchor keeps the full budget, matching the old single-anchor behavior.
    # On expiry we fall through to the best partial track found so far.
    max_time = 120.0
    anchor_allowance = max_time / max(1, len(anchor_labels))
    anchor_spent: Dict[str, float] = {label: 0.0 for label in anchor_labels}

    # Screening runs COARSER than the delivered corridor: candidates are
    # screened at the screen_* step (cheap, many candidates fit in the budget)
    # and only a corridor that comes back clean is re-tracked at the requested
    # position_res/rotation_res for delivery. Screening is never finer than
    # the request itself, and when the two coincide the re-track is skipped.
    fine_pos = max(position_res, 1e-6)
    fine_rot = max(rotation_res, 1e-6)
    screen_pos = max(fine_pos, screen_position_res)
    screen_rot = max(fine_rot, screen_rotation_res)
    needs_fine_retrack = screen_pos > fine_pos or screen_rot > fine_rot

    def _track_to(world_from_bar_home: PoseLike, pos_step: float, rot_step: float):
        """Walk goal -> home with warm per-waypoint IK at the given steps.

        Args:
            world_from_bar_home (PoseLike): the candidate home bar pose.
            pos_step (float): linear interpolation step, meters.
            rot_step (float): angular interpolation step, radians.

        Returns:
            Tuple[list, bool]: ``(track, track_ok)`` -- ``track`` is the
            accepted ``(pose, conf)`` prefix starting at the goal; ``track_ok``
            is False when the walk broke on an IK miss or a joint-continuity
            (branch flip) violation, leaving ``track`` as the reachable prefix.
        """
        track = [(world_from_bar_goal, np.asarray(goal_conf, dtype=float))]
        for pose in list(
            pp.interpolate_poses(
                world_from_bar_goal,
                world_from_bar_home,
                pos_step_size=pos_step,
                ori_step_size=rot_step,
            )
        )[1:]:
            next_conf = solve_dual_arm_pose_ik(
                robot=robot,
                arm_joints=arm_joints,
                tool_link_left=tool_link_left,
                tool_link_right=tool_link_right,
                bar_pose=pose,
                grasp_bar_from_left=grasp_bar_from_left,
                grasp_bar_from_right=grasp_bar_from_right,
                seed_conf=track[-1][1],
                use_angle_normalization=use_angle_normalization,
            )
            # Track breaks on IK miss or a branch flip.
            if next_conf is None or joint_step_exceeds_threshold(
                next_conf, track[-1][1], joint_continuity_threshold_rad
            ):
                return track, False
            track.append((pose, np.asarray(next_conf, dtype=float)))
        return track, True

    def _first_blocked_index(track) -> Optional[int]:
        """Index (within ``track``) of the first colliding INTERIOR waypoint.

        Checked coarse-first (every 3rd waypoint) so blocked corridors are
        rejected at a third of the cost; all interior waypoints get covered.

        Args:
            track (list): the fully-tracked ``(pose, conf)`` waypoints.

        Returns:
            Optional[int]: index of the first hit found (in scan order), or
            None when every interior waypoint is collision-free.
        """
        if joint_collision_fn is None:
            return None
        interior = track[1:-1]
        for stride_offset in (0, 1, 2):  # coarse pass 0, then the rest
            for idx in range(stride_offset, len(interior), 3):
                if joint_collision_fn(interior[idx][1]):
                    return idx + 1  # index within `track`
        return None

    info: Dict[str, Any] = {
        "tracked_deltas": 0,
        "track_breaks": 0,
        "arrival_collisions": 0,
        "blocked_corridors": 0,
        "fine_reverify_failures": 0,
    }
    # First fully-tracked candidate with a collision-free ARRIVAL: kept as the
    # start-only answer when no fully collision-free corridor turns up.
    first_start: Optional[Dict[str, Any]] = None
    # Best PARTIAL track seen: the one whose last continuous waypoint sits
    # farthest (in bar position) from the goal. Kept as the last fallback: a
    # start on the goal's own branch sheet partway toward home still gives the
    # RRT a branch-consistent, solvable problem (M0 absorbs the difference).
    best_partial: Optional[Dict[str, Any]] = None

    with pp.WorldSaver():
        for variant_label, (base_pos_mb, home_bar_quat) in anchor_variants:
            # Which carry anchor this variant belongs to ("back/yaw+30" -> "back").
            anchor_label = variant_label.split("/")[0]
            if anchor_spent[anchor_label] >= anchor_allowance:
                continue  # this anchor's time share is spent; others carry on
            base_pos_mb = np.asarray(base_pos_mb, dtype=float)
            variant_deltas = deltas if variant_label.endswith("/canonical") else deltas[:max_variant_deltas]
            mark = time.time()
            for delta in variant_deltas:
                # Charge the previous iteration to this anchor's time share.
                now = time.time()
                anchor_spent[anchor_label] += now - mark
                mark = now
                if anchor_spent[anchor_label] >= anchor_allowance:
                    logger.warning(
                        "derive_constrained_start_tracked: anchor %s spent its %.0fs "
                        "time share at variant %s; moving to the next anchor.",
                        anchor_label, anchor_allowance, variant_label,
                    )
                    break
                home_mb = (
                    tuple((base_pos_mb + np.asarray(delta, dtype=float)).tolist()),
                    home_bar_quat,
                )
                world_from_bar_home = pp.multiply(world_from_mobile_base, home_mb)
                info["tracked_deltas"] += 1

                # --- SCREEN: walk goal -> home with warm per-waypoint IK at the
                # coarse screening step. ``track`` records every accepted
                # (pose, conf) so a broken walk can still donate its farthest
                # reachable prefix as a partial start.
                track, track_ok = _track_to(world_from_bar_home, screen_pos, screen_rot)

                if track_ok:
                    # --- Arrival config must be collision-free (it is a plan endpoint).
                    conf = track[-1][1]
                    if joint_collision_fn is not None and joint_collision_fn(conf):
                        info["arrival_collisions"] += 1
                        continue
                    # --- FREE CORRIDOR check: if every interior waypoint is also
                    # collision-free, this track IS a finished M1 path -- return
                    # it and let the caller skip the RRT (direct-connect-first).
                    first_blocked = _first_blocked_index(track)
                    if first_blocked is None:
                        # --- DELIVER: re-track only this winner at the requested
                        # (finer) resolution -- the corridor becomes the M1 path,
                        # so its spacing must be what the caller asked for, not
                        # the screening step.
                        if needs_fine_retrack:
                            fine_track, fine_ok = _track_to(world_from_bar_home, fine_pos, fine_rot)
                            if fine_ok and (
                                joint_collision_fn is None
                                or not joint_collision_fn(fine_track[-1][1])
                            ) and _first_blocked_index(fine_track) is None:
                                track = fine_track
                                conf = fine_track[-1][1]
                            else:
                                # The coarse screen was a false positive at fine
                                # spacing (continuity, IK or a collision differs
                                # between the two step sizes). Keep the candidate
                                # as a start-only answer -- its coarse arrival
                                # conf is exact IK at the home pose and already
                                # passed the collision gate -- and keep scanning
                                # for another corridor.
                                info["fine_reverify_failures"] += 1
                                logger.warning(
                                    "derive_constrained_start_tracked: corridor at %s "
                                    "failed the fine re-track (%.4f m / %.4f rad); "
                                    "keeping it as a start-only candidate.",
                                    variant_label, fine_pos, fine_rot,
                                )
                                if first_start is None:
                                    first_start = {
                                        "pose": world_from_bar_home,
                                        "conf": conf,
                                        "variant": variant_label,
                                        "delta": [float(v) for v in delta],
                                    }
                                continue
                        info["delta"] = [float(v) for v in delta]
                        info["variant"] = variant_label
                        info["corridor"] = (
                            [wp_pose for wp_pose, _c in track],
                            [wp_conf for _p, wp_conf in track],
                        )
                        return world_from_bar_home, conf, info
                    # Corridor blocked mid-way: keep the FIRST such candidate as
                    # the start-only answer and keep scanning for a free corridor.
                    # Record WHERE it blocked (fraction along goal->home) -- a
                    # diagnostic that separates "blocked right at the goal exit"
                    # (straight corridors hopeless, RRT must detour) from
                    # "blocked near home" (other deltas/orientations may clear).
                    info["blocked_corridors"] += 1
                    blocked_fraction = first_blocked / max(1, len(track) - 1)
                    info.setdefault("blocked_fractions", []).append(round(blocked_fraction, 2))
                    if first_start is None:
                        first_start = {
                            "pose": world_from_bar_home,
                            "conf": conf,
                            "variant": variant_label,
                            "delta": [float(v) for v in delta],
                        }
                    continue

                info["track_breaks"] += 1
                # Remember the farthest-reaching broken walk for the fallback.
                partial_dist = float(np.linalg.norm(
                    np.asarray(track[-1][0][0], dtype=float)
                    - np.asarray(world_from_bar_goal[0], dtype=float)
                ))
                if best_partial is None or partial_dist > best_partial["distance"]:
                    best_partial = {
                        "distance": partial_dist,
                        "track": track,
                        "variant": variant_label,
                        "delta": [float(v) for v in delta],
                    }
            # Charge the last delta iteration too, then move to the next
            # variant (an anchor whose share is spent gets skipped up top).
            anchor_spent[anchor_label] += time.time() - mark

    # --- No fully collision-free corridor: fall back to the first fully-tracked
    # start with a clean arrival (the RRT then searches for the detour).
    if first_start is not None:
        info["delta"] = first_start["delta"]
        info["variant"] = first_start["variant"]
        return first_start["pose"], first_start["conf"], info

    # --- Partial fallback: no home pose fully tracked. Walk the best partial
    # track back from its farthest waypoint until a collision-free config, and
    # use THAT as the start (still on the goal's branch sheet by construction).
    min_partial_distance = 0.02  # m -- below this the "motion" is degenerate
    if best_partial is not None and best_partial["distance"] >= min_partial_distance:
        for pose, conf in reversed(best_partial["track"]):
            dist = float(np.linalg.norm(
                np.asarray(pose[0], dtype=float) - np.asarray(world_from_bar_goal[0], dtype=float)
            ))
            if dist < min_partial_distance:
                break  # walked back too close to the goal -- give up on partial
            if joint_collision_fn is not None and joint_collision_fn(conf):
                continue
            info["variant"] = best_partial["variant"]
            info["delta"] = best_partial["delta"]
            info["partial"] = True
            info["partial_distance_m"] = dist
            logger.warning(
                "derive_constrained_start_tracked: no full track to any home; using a "
                "PARTIAL start %.3f m from the goal (variant %s). M0 will cover the rest.",
                dist, best_partial["variant"],
            )
            return pose, conf, info

    logger.warning(
        "derive_constrained_start_tracked: no trackable collision-free home across "
        "%d delta/orientation candidates over anchors %s (%d track breaks, %d arrival collisions)",
        info["tracked_deltas"], "/".join(anchor_labels),
        info["track_breaks"], info["arrival_collisions"],
    )
    return None, None, info


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
    """Summarize the largest per-step joint jump along a joint-space path.

    Args:
        joint_path (Optional[Sequence[FullConf]]): the ordered configurations, or
            ``None``.
        threshold_rad (float): per-step jump (radians) above which a step counts
            as "bad".
        use_angle_normalization (bool): accepted for signature symmetry; the raw
            command deltas are used as-is.

    Returns:
        Dict[str, Any]: ``{"ok", "max_delta_rad", "first_bad_step",
        "threshold_rad"}``. ``ok`` is ``None`` for a ``None`` path and True/False
        otherwise; ``first_bad_step`` is the 1-based index of the first offending
        step, or ``None``.
    """
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
    """Re-solve a joint path along a pose path, enforcing continuity + collision.

    Walks the pose path from ``start_conf``, warm-starting each step's dual-arm
    IK from the previous config, and bails at the first waypoint that fails IK,
    the continuity check, or collision.

    Args:
        scene (Dict[str, Any]): solver context (robot, arm_joints, tool links,
            and the two grasp transforms).
        pose_path (Sequence[PoseLike]): the ordered bar poses to follow.
        start_conf (FullConf): the joint configuration at the first pose.
        joint_collision_fn (Optional[Callable[[FullConf], bool]]): optional
            joint-space collision predicate.
        joint_continuity_threshold_rad (Optional[float]): max allowed per-step
            joint jump; ``None`` disables the check.
        use_angle_normalization (bool): whether to normalize the solved angles.

    Returns:
        Tuple[Optional[List[FullConf]], Optional[str]]: ``(joint_path, None)`` on
        success, or ``(None, reason)`` where ``reason`` names the failed waypoint
        (e.g. ``"ik_failure_at_waypoint_3"``) or the missing input.
    """
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
    """Populate ``debug_tree_out`` in place with the tree + planning outcome.

    A no-op when ``debug_tree_out`` is ``None``. Records success, iteration
    count, the exported tree, endpoints, and a histogram of ``extend_toward``
    stop reasons for offline diagnosis.

    Args:
        debug_tree_out (Optional[Dict]): the dict to fill (cleared first), or
            ``None`` to skip.
        success (bool): whether a path was found.
        iterations (int): iteration count to record.
        nodes (List[TreeNode]): the tree to export.
        start_pose (PoseLike): the start bar pose.
        goal_pose (PoseLike): the goal bar pose.
        extend_stop_reasons (Optional[Dict[str, int]]): histogram of extend stop
            reasons across the run.

    Returns:
        None. Mutates ``debug_tree_out``.
    """
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
    *,
    planner=None,
    start_state=None,
    active_bar_id: Optional[str] = None,
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
    """Plan a collision-free path in SE(3) pose space using a single-tree RRT.

    ARCHIVAL: this single start-rooted RRT is no longer the default. The
    bidirectional :func:`plan_pose_birrt` is now used everywhere (it won the
    benchmark); this function is retained only for comparison.

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
    # Pose collision fn now uses cfab.check_collision via build_cfab_pose_collision_fn.
    # It runs whenever enable_collision is True, regardless of IK stage — the cfab
    # check is fast enough and is the single source of truth for ACM.
    rng = np.random.default_rng(random_seed)
    feature_points = list(feature_points) if feature_points is not None else get_bar_feature_points()
    if enable_collision:
        if planner is None or start_state is None or active_bar_id is None:
            raise ValueError(
                "plan_pose_rrt with enable_collision=True requires keyword args "
                "`planner`, `start_state`, and `active_bar_id` so the cfab "
                "pose-collision closure can be built."
            )
        collision_fn = build_cfab_pose_collision_fn(planner, start_state, active_bar_id)
    else:
        collision_fn = _noop_pose_collision_fn

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


def plan_pose_birrt(
    robot: int,
    bar_body: int,
    obstacle_bodies: Sequence[int],
    start_pose: PoseLike,
    goal_pose: PoseLike,
    *,
    planner=None,
    start_state=None,
    active_bar_id: Optional[str] = None,
    start_conf: Optional[FullConf] = None,
    goal_conf: Optional[FullConf] = None,
    dist_metric: str = "feature",
    goal_sample_prob: float = 0.05,
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
    """Bidirectional pose-space RRT-Connect with dual-arm IK propagation.

    Grows two trees: T_a rooted at start_pose, T_b rooted at goal_pose. Each
    iteration extends one tree toward a random sample, then tries to connect
    the other tree to that tree's new frontier. Connection succeeds when the
    two-tree meeting poses match AND the joint configs are continuous (no
    branch flip across the connection).
    """
    rng = np.random.default_rng(random_seed)
    feature_points = list(feature_points) if feature_points is not None else get_bar_feature_points()
    if enable_collision:
        if planner is None or start_state is None or active_bar_id is None:
            raise ValueError(
                "plan_pose_birrt with enable_collision=True requires keyword args "
                "`planner`, `start_state`, and `active_bar_id`."
            )
        collision_fn = build_cfab_pose_collision_fn(planner, start_state, active_bar_id)
    else:
        collision_fn = _noop_pose_collision_fn

    if joint_collision_fn is not None:
        if start_conf is None or goal_conf is None:
            raise ValueError("Collision-aware planning requires both start_conf and goal_conf.")
        if joint_collision_fn(start_conf):
            logger.warning("Start configuration is in collision.")
            return None, None
        if joint_collision_fn(goal_conf):
            logger.warning("Goal configuration is in collision.")
            return None, None
    else:
        if collision_fn(start_pose):
            logger.warning("Start pose is in floating-body collision.")
            return None, None
        if collision_fn(goal_pose):
            logger.warning("Goal pose is in floating-body collision.")
            return None, None

    def make_tree(root_pose: PoseLike, root_conf: Optional[FullConf]) -> Tuple[List[TreeNode], Dict, Dict]:
        """Create one BiRRT tree rooted at a pose, with its per-node side caches.

        Args:
            root_pose (PoseLike): the tree's root bar pose.
            root_conf (Optional[FullConf]): the root joint config; required when
                IK is enabled.

        Returns:
            Tuple[List[TreeNode], Dict, Dict]: ``(nodes, node_confs,
            feature_vecs)`` -- the node list plus the per-node config and
            feature-vector caches (both keyed by ``id(node)``).

        Raises:
            ValueError: if IK is enabled but ``root_conf`` is ``None``.
        """
        root = TreeNode(root_pose)
        nodes = [root]
        node_confs: Dict[int, FullConf] = {}
        feature_vecs: Dict[int, np.ndarray] = {}
        if enable_ik:
            if root_conf is None:
                raise ValueError("BiRRT with IK requires start_conf and goal_conf.")
            node_confs[id(root)] = np.asarray(root_conf, dtype=float)
        if dist_metric == "feature":
            fvec = pose_to_feature_vec(root_pose, feature_points)
            if fvec is not None:
                feature_vecs[id(root)] = fvec
        return nodes, node_confs, feature_vecs

    extend_stop_reasons: Counter = Counter()
    best_tree: List[TreeNode] = []
    total_iterations = 0

    for attempt in range(max_attempts):
        start_time = time.time()
        nodes_a, confs_a, fvecs_a = make_tree(start_pose, start_conf)
        nodes_b, confs_b, fvecs_b = make_tree(goal_pose, goal_conf)

        for iteration in range(max_iterations):
            total_iterations += 1
            if (time.time() - start_time) >= max_time:
                break

            # Alternate which tree extends toward the random sample.
            # Convention: TreeA always rooted at start, TreeB at goal. We pick
            # which tree extends first based on size (smaller grows first), but
            # we always retrace the final path as start->goal.
            grow_a_first = len(nodes_a) <= len(nodes_b)
            tree_grow, confs_grow, fvecs_grow = (
                (nodes_a, confs_a, fvecs_a) if grow_a_first else (nodes_b, confs_b, fvecs_b)
            )
            tree_other, confs_other, fvecs_other = (
                (nodes_b, confs_b, fvecs_b) if grow_a_first else (nodes_a, confs_a, fvecs_a)
            )

            # Sample: occasional bias toward the other tree's root (cross-bias).
            if rng.random() < goal_sample_prob:
                target_pose = (nodes_a[0].config if not grow_a_first else nodes_b[0].config)
            else:
                target_pose, _ = sample_pose(robot, goal_pose, rng, 0.0, workspace_xy, workspace_z)

            nearest = nearest_node(tree_grow, target_pose, dist_metric, feature_points, fvecs_grow)
            new_last, _, stop_reason = extend_toward(
                nodes=tree_grow,
                source=nearest,
                target_pose=target_pose,
                collision_fn=collision_fn,
                joint_collision_fn=joint_collision_fn,
                draw_color=(0.85, 0.2, 0.2, 0.45) if grow_a_first else (0.2, 0.45, 0.85, 0.45),
                use_draw=use_draw,
                position_res=position_res,
                rotation_res=rotation_res,
                dist_metric=dist_metric,
                feature_points=feature_points,
                feature_vecs=fvecs_grow,
                enable_ik=enable_ik,
                node_confs=confs_grow,
                ik_context=ik_context,
                joint_continuity_threshold_rad=joint_continuity_threshold_rad,
                use_angle_normalization=use_angle_normalization,
            )
            extend_stop_reasons[stop_reason] += 1
            if new_last is nearest:
                # No progress this iteration; move on.
                continue

            # Connect: try to extend the other tree all the way to `new_last`.
            connect_target = new_last.config
            connect_nearest = nearest_node(tree_other, connect_target, dist_metric, feature_points, fvecs_other)
            connect_last, connect_reached, connect_stop = extend_toward(
                nodes=tree_other,
                source=connect_nearest,
                target_pose=connect_target,
                collision_fn=collision_fn,
                joint_collision_fn=joint_collision_fn,
                draw_color=(0.85, 0.5, 0.0, 0.6),
                use_draw=use_draw,
                position_res=position_res,
                rotation_res=rotation_res,
                dist_metric=dist_metric,
                feature_points=feature_points,
                feature_vecs=fvecs_other,
                enable_ik=enable_ik,
                node_confs=confs_other,
                ik_context=ik_context,
                joint_continuity_threshold_rad=joint_continuity_threshold_rad,
                use_angle_normalization=use_angle_normalization,
            )
            extend_stop_reasons[f"connect_{connect_stop}"] += 1

            if not connect_reached:
                continue

            # Stitch path: identify start-side (tree_a, anchored at start_pose) and
            # goal-side (tree_b, anchored at goal_pose) portions of the BiRRT solution.
            # The "frontier" on the start side and the "junction" on the goal side meet
            # at the same pose; we drop the duplicate seam node.
            if grow_a_first:
                start_side_nodes = list(new_last.retrace())       # start -> new_last
                goal_side_nodes_rev = list(reversed(connect_last.retrace()))  # connect_last -> goal_root
            else:
                start_side_nodes = list(connect_last.retrace())   # start -> connect_last
                goal_side_nodes_rev = list(reversed(new_last.retrace()))      # new_last -> goal_root
            # Drop the goal-side's first node (duplicate of start-side's last).
            goal_side_nodes_rev_tail = goal_side_nodes_rev[1:]

            start_side_poses = [n.config for n in start_side_nodes]
            goal_side_poses = [n.config for n in goal_side_nodes_rev_tail]

            path_poses: List[PoseLike] = start_side_poses + goal_side_poses
            path_confs: Optional[List[FullConf]] = None

            if enable_ik:
                start_side_confs_raw = [np.asarray(confs_a[id(n)], dtype=float) for n in start_side_nodes]
                # goal_side_nodes_rev_tail is [seam+1, ..., goal_root], in path order.
                goal_side_confs_raw = [np.asarray(confs_b[id(n)], dtype=float) for n in goal_side_nodes_rev_tail]
                if not start_side_confs_raw:
                    continue

                def _stitch_forward(seed_conf):
                    """Rebuild the goal-side joint configs by re-IK'ing from the seam.

                    Warm-starts from ``seed_conf`` and solves each goal-side pose
                    in order, rejecting on IK failure, continuity break, or
                    collision (each recorded in ``extend_stop_reasons``).

                    Args:
                        seed_conf (FullConf): the seam configuration to start from.

                    Returns:
                        Optional[List[FullConf]]: the goal-side configs in path
                        order, or ``None`` if any step failed.
                    """
                    out: List[FullConf] = []
                    cur = seed_conf
                    for stitch_pose in goal_side_poses:
                        nxt = solve_dual_arm_pose_ik(
                            robot=ik_context["robot"],
                            arm_joints=ik_context["arm_joints"],
                            tool_link_left=ik_context["tool_link_left"],
                            tool_link_right=ik_context["tool_link_right"],
                            bar_pose=stitch_pose,
                            grasp_bar_from_left=ik_context["grasp_bar_from_left"],
                            grasp_bar_from_right=ik_context["grasp_bar_from_right"],
                            seed_conf=cur,
                            use_angle_normalization=use_angle_normalization,
                        )
                        if nxt is None:
                            extend_stop_reasons["stitch_ik_failure"] += 1
                            return None
                        if joint_step_exceeds_threshold(nxt, cur, joint_continuity_threshold_rad):
                            extend_stop_reasons["stitch_continuity_fail"] += 1
                            return None
                        if joint_collision_fn is not None and joint_collision_fn(nxt):
                            extend_stop_reasons["stitch_collision"] += 1
                            return None
                        out.append(np.asarray(nxt, dtype=float))
                        cur = nxt
                    return out

                def _stitch_backward(seed_conf_at_goal):
                    """Rebuild the start-side joint configs by re-IK'ing from the goal branch.

                    Walks the start-side poses in reverse (goal_root -> start)
                    warm-started from ``seed_conf_at_goal``, then reverses the
                    result into path order and checks the start endpoint agrees
                    with the tree's start config.

                    Args:
                        seed_conf_at_goal (FullConf): the goal-branch config to
                            start the backward solve from.

                    Returns:
                        Optional[List[FullConf]]: the start-side configs in path
                        order (start -> seam), or ``None`` if any step failed or
                        the start endpoint mismatched.
                    """
                    out_rev: List[FullConf] = []
                    cur = seed_conf_at_goal
                    for stitch_pose in reversed([n.config for n in start_side_nodes]):
                        nxt = solve_dual_arm_pose_ik(
                            robot=ik_context["robot"],
                            arm_joints=ik_context["arm_joints"],
                            tool_link_left=ik_context["tool_link_left"],
                            tool_link_right=ik_context["tool_link_right"],
                            bar_pose=stitch_pose,
                            grasp_bar_from_left=ik_context["grasp_bar_from_left"],
                            grasp_bar_from_right=ik_context["grasp_bar_from_right"],
                            seed_conf=cur,
                            use_angle_normalization=use_angle_normalization,
                        )
                        if nxt is None:
                            extend_stop_reasons["stitch_ik_failure"] += 1
                            return None
                        if joint_step_exceeds_threshold(nxt, cur, joint_continuity_threshold_rad):
                            extend_stop_reasons["stitch_continuity_fail"] += 1
                            return None
                        if joint_collision_fn is not None and joint_collision_fn(nxt):
                            extend_stop_reasons["stitch_collision"] += 1
                            return None
                        out_rev.append(np.asarray(nxt, dtype=float))
                        cur = nxt
                    out_rev.reverse()
                    # Final start-side IK must agree with start_conf endpoint (it's the same pose).
                    if joint_step_exceeds_threshold(out_rev[0], start_side_confs_raw[0], joint_continuity_threshold_rad):
                        extend_stop_reasons["stitch_endpoint_mismatch"] += 1
                        return None
                    return out_rev

                # Try forward stitch (rebuild goal side from start-side seam conf)
                stitched_goal = _stitch_forward(start_side_confs_raw[-1])
                if stitched_goal is not None:
                    path_confs = start_side_confs_raw + stitched_goal
                else:
                    # Fallback: rebuild START side from goal-root branch.
                    if goal_side_confs_raw:
                        seed_goal = goal_side_confs_raw[-1]  # goal_root conf
                    else:
                        seed_goal = np.asarray(goal_conf, dtype=float)
                    rebuilt_start = _stitch_backward(seed_goal)
                    if rebuilt_start is None:
                        continue
                    # rebuilt_start[-1] is at the seam pose, on goal branch -> continuous with goal_side_confs_raw.
                    # Seam continuity is guaranteed since both end at goal-side branch.
                    path_confs = rebuilt_start + list(goal_side_confs_raw)

            update_debug_tree(
                debug_tree_out, True, iteration + 1, nodes_a + nodes_b, start_pose, goal_pose,
                extend_stop_reasons=extend_stop_reasons,
            )
            logger.info(
                f"birrt: connected after {iteration + 1} iters; "
                f"tree_a={len(nodes_a)} nodes, tree_b={len(nodes_b)} nodes, path={len(path_poses)} waypoints."
            )
            return path_poses, path_confs

        best_tree = nodes_a + nodes_b
        logger.info(
            f"birrt attempt {attempt + 1}/{max_attempts}: no path found "
            f"(tree_a={len(nodes_a)}, tree_b={len(nodes_b)})."
        )

    update_debug_tree(
        debug_tree_out, False, total_iterations, best_tree, start_pose, goal_pose,
        extend_stop_reasons=extend_stop_reasons,
    )
    return None, None
