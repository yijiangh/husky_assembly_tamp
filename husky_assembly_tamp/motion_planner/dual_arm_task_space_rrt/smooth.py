from __future__ import annotations

import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet_planning as pp

from .core import (
    DEFAULT_USE_ANGLE_NORMALIZATION,
    FullConf,
    PoseLike,
    _pose_path_cost,
    _pose_path_inflection_indices,
    get_bar_feature_points,
    reconstruct_joint_path_for_pose_path,
    repair_joint_path,
)


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


# -----------------------------------------------------------------------------
# * Rounded corners in bar pose space
# -----------------------------------------------------------------------------
#
# A shortcut-smoothed or roadmap path is a chain of straight pieces in bar pose
# space. Its joint path has a corner wherever two pieces meet, and an arm can
# only follow a joint corner by stopping there (or by cutting it, which breaks
# the two-arm grasp). Rounding each corner in pose space, with a curve that
# leaves and joins the straight pieces along their own direction, gives a bar
# path without corners; its joint path, solved again by IK, has none either
# (away from singularities), so a trajectory timer can run through it.

#: Default corner radius (m): how far before and after a corner the bar starts and stops turning off the straight
#: piece, measured as the largest move of any bar feature point. Capped at half of each neighbouring piece, so the
#: pieces' length (the tolerance below) limits it first.
DEFAULT_ROUND_RADIUS = 0.3
#: Default straightness tolerance (m): samples within this of the straight piece between two corners (largest
#: feature-point offset) belong to that piece; anything further makes a corner. A fine tolerance cuts a path into
#: short pieces and its curves stay as tight as the corners were; the joints follow a curve without stopping only
#: when it is wide. On the B6 roadmap carry, 60 mm left 5 corners and moved the bar centre at most 35 mm off its path.
DEFAULT_ROUND_TOLERANCE = 0.06
#: The finest tolerance tried when the straightened pieces themselves fail (m).
DEFAULT_ROUND_MIN_TOLERANCE = 0.001
#: A corner whose radius had to shrink below this (m) is left sharp.
DEFAULT_ROUND_MIN_RADIUS = 0.02

_WAYPOINT = re.compile(r"_at_waypoint_(\d+)$")


def _quat_continuous(quaternions: np.ndarray) -> np.ndarray:
    """Unit quaternions (N x 4, xyzw) with signs chosen so each stays on the side of the one before."""
    out = np.array(quaternions, dtype=float)
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    for index in range(1, len(out)):
        if float(np.dot(out[index - 1], out[index])) < 0.0:
            out[index] = -out[index]
    return out


def _slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    """Spherical interpolation of two unit quaternions (xyzw) on the short arc, ``s`` in [0, 1]."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 1.0 - 1e-12:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = math.acos(min(1.0, dot))
    sin_theta = math.sin(theta)
    return (math.sin((1.0 - s) * theta) * q0 + math.sin(s * theta) * q1) / sin_theta


def _quat_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    """The turn (rad) between two unit quaternions."""
    return 2.0 * math.acos(min(1.0, abs(float(np.dot(q0, q1)))))


def _rotate(quaternion: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Rotate points (K x 3) by a unit quaternion (xyzw)."""
    vector, w = quaternion[:3], quaternion[3]
    cross = np.cross(vector, points)
    return points + 2.0 * w * cross + 2.0 * np.cross(vector, cross)


class _PosePath:
    """Bar poses as arrays, with the feature-point geometry the rounding measures in.

    Distances are the largest move of any bar feature point (the corners of the bar's box): a turn and a move
    count in the same metres.
    """

    def __init__(self, poses: Sequence[PoseLike], feature_points: Sequence[np.ndarray]):
        self.positions = np.array([np.asarray(pose[0], dtype=float) for pose in poses])
        self.quaternions = _quat_continuous(np.array([np.asarray(pose[1], dtype=float) for pose in poses]))
        self.feature_points = np.array(feature_points, dtype=float)

    def __len__(self) -> int:
        return len(self.positions)

    def points(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """The feature points (K x 3) in the world at one pose."""
        return position + _rotate(quaternion, self.feature_points)

    def points_at(self, index: int) -> np.ndarray:
        """The feature points at a pose of the path."""
        return self.points(self.positions[index], self.quaternions[index])

    def distance(self, a: Tuple[np.ndarray, np.ndarray], b: Tuple[np.ndarray, np.ndarray]) -> float:
        """The largest feature-point move between two poses (m)."""
        return float(np.max(np.linalg.norm(self.points(*a) - self.points(*b), axis=1)))

    def pose(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """A pose of the path as (position, quaternion)."""
        return self.positions[index], self.quaternions[index]


def _between(a: Tuple[np.ndarray, np.ndarray], b: Tuple[np.ndarray, np.ndarray],
             s: float) -> Tuple[np.ndarray, np.ndarray]:
    """The pose ``s`` of the way from ``a`` to ``b``: the position straight, the orientation on the short arc."""
    return a[0] + s * (b[0] - a[0]), _slerp(a[1], b[1], s)


def _slerp_many(q0: np.ndarray, q1: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """``_slerp`` at many shares at once: (S,) shares -> (S, 4) quaternions; ``q1`` on ``q0``'s side."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    shares = np.asarray(shares, dtype=float)[:, None]
    if dot > 1.0 - 1e-12:
        q = q0 + shares * (q1 - q0)
        return q / np.linalg.norm(q, axis=1, keepdims=True)
    theta = math.acos(min(1.0, dot))
    return (np.sin((1.0 - shares) * theta) * q0 + np.sin(shares * theta) * q1) / math.sin(theta)


def _piece_points(path: _PosePath, a: Tuple[np.ndarray, np.ndarray], b: Tuple[np.ndarray, np.ndarray],
                  shares: np.ndarray) -> np.ndarray:
    """The feature points (S x K x 3) of the straight piece from ``a`` to ``b`` at each share (S,)."""
    shares = np.asarray(shares, dtype=float)
    positions = a[0] + shares[:, None] * (b[0] - a[0])
    quaternions = _slerp_many(a[1], b[1], shares)
    vector, w = quaternions[:, None, :3], quaternions[:, None, 3:]
    local = path.feature_points[None, :, :]
    cross = np.cross(vector, local)
    return positions[:, None, :] + local + 2.0 * w * cross + 2.0 * np.cross(vector, cross)


def _offsets_from_piece(path: _PosePath, first: int, last: int) -> np.ndarray:
    """The feature-point distance of each pose strictly between ``first`` and ``last`` to the nearest pose of the
    straight piece between them (largest point distance, m): a grid along the piece, refined twice around the best.
    """
    a, b = path.pose(first), path.pose(last)
    samples = np.array([path.points_at(index) for index in range(first + 1, last)])        # M x K x 3

    def distances(shares: np.ndarray) -> np.ndarray:
        """(M, S) distances for per-sample shares (M, S)."""
        m, n = shares.shape
        points = _piece_points(path, a, b, shares.reshape(-1)).reshape(m, n, *samples.shape[1:])
        return np.max(np.linalg.norm(points - samples[:, None], axis=3), axis=2)

    shares = np.tile(np.linspace(0.0, 1.0, 101), (len(samples), 1))
    width = 0.01
    for _ in range(3):
        found = distances(shares)
        best = shares[np.arange(len(samples)), np.argmin(found, axis=1)]
        shares = np.clip(best[:, None] + np.linspace(-width, width, 21)[None, :], 0.0, 1.0)
        width /= 10.0
    return np.min(distances(shares), axis=1)


def pose_path_corners(path_poses: Sequence[PoseLike], feature_points: Optional[Sequence[np.ndarray]] = None,
                      tolerance: float = DEFAULT_ROUND_TOLERANCE) -> List[int]:
    """The corners of a bar pose path: the fewest of its poses whose straight pieces stay within ``tolerance``.

    Ramer-Douglas-Peucker on the pose path. A sample's offset is its feature-point distance to the nearest pose of
    the straight piece (position straight, orientation on the short arc).

    Args:
        path_poses (Sequence[PoseLike]): the bar poses, start first.
        feature_points (Optional[Sequence[np.ndarray]]): bar-local feature points (default: the bar's box corners).
        tolerance (float): the largest feature-point offset a sample may have from its piece (m).

    Returns:
        List[int]: indices into ``path_poses``, both ends included, in order.
    """
    feature_points = list(feature_points) if feature_points is not None else get_bar_feature_points()
    path = _PosePath(path_poses, feature_points)
    n = len(path)
    if n <= 2:
        return list(range(n))
    keep = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:
        first, last = stack.pop()
        if last - first < 2:
            continue
        offsets = _offsets_from_piece(path, first, last)
        worst_index = first + 1 + int(np.argmax(offsets))
        worst = float(offsets[worst_index - first - 1])
        if worst > tolerance:
            keep.add(worst_index)
            stack += [(first, worst_index), (worst_index, last)]
    return sorted(keep)


def _steps_for(a: Tuple[np.ndarray, np.ndarray], b: Tuple[np.ndarray, np.ndarray], position_res: float,
               rotation_res: float) -> int:
    """How many steps keep a move from ``a`` to ``b`` within the resolutions (at least one)."""
    move = float(np.linalg.norm(b[0] - a[0])) / max(position_res, 1e-9)
    turn = _quat_angle(a[1], b[1]) / max(rotation_res, 1e-9)
    return max(1, int(math.ceil(max(move, turn))))


def round_pose_path(path_poses: Sequence[PoseLike], corners: Sequence[int], radii: Sequence[float], *,
                    position_res: float, rotation_res: float,
                    feature_points: Optional[Sequence[np.ndarray]] = None) -> Tuple[List[PoseLike], List[int]]:
    """The straight pieces between ``corners``, each inner corner rounded, sampled at the resolutions.

    At an inner corner V with neighbours P (before) and N (after), the curve starts at A on the piece P-V and
    ends at B on V-N, each ``radius`` (largest feature-point move) from V, at most half the piece. Between them
    it is a quadratic Bezier with control points A, V, B: the position by de Casteljau, the orientation by
    de Casteljau with slerps. A and B lie on their pieces with position and orientation at the same share, so
    the curve leaves and joins the pieces along their own direction in pose space: no corner at A or B.

    Args:
        path_poses (Sequence[PoseLike]): the bar poses, start first.
        corners (Sequence[int]): indices of the corners (``pose_path_corners``), both ends included.
        radii (Sequence[float]): one radius per inner corner (``len(corners) - 2``), m; 0 keeps the corner sharp.
        position_res (float): largest bar move between samples (m).
        rotation_res (float): largest bar turn between samples (rad).
        feature_points (Optional[Sequence[np.ndarray]]): bar-local feature points (default: the box corners).

    Returns:
        Tuple[List[PoseLike], List[int]]: the sampled poses ``((x, y, z), (qx, qy, qz, qw))``, start and end
        exactly the path's, and for each pose the inner corner it rounds (0-based among the inner corners) or -1
        on a straight piece.
    """
    feature_points = list(feature_points) if feature_points is not None else get_bar_feature_points()
    path = _PosePath(path_poses, feature_points)
    if len(radii) != max(0, len(corners) - 2):
        raise ValueError(f"{len(corners)} corners need {max(0, len(corners) - 2)} radii, got {len(radii)}")
    control = [path.pose(index) for index in corners]
    lengths = [path.distance(a, b) for a, b in zip(control[:-1], control[1:])]
    # The share of each piece the curve takes at its start and end (0 at the path's ends).
    head = [0.0] * len(lengths)
    tail = [0.0] * len(lengths)
    for k, radius in enumerate(radii, start=1):
        if radius <= 0.0:
            continue
        if lengths[k - 1] > 0.0:
            tail[k - 1] = min(0.5, radius / lengths[k - 1])
        if lengths[k] > 0.0:
            head[k] = min(0.5, radius / lengths[k])

    poses: List[Tuple[np.ndarray, np.ndarray]] = [control[0]]
    owner: List[int] = [-1]

    def add(pose: Tuple[np.ndarray, np.ndarray], corner: int) -> None:
        poses.append(pose)
        owner.append(corner)

    for k in range(len(lengths)):
        a, b = control[k], control[k + 1]
        start = _between(a, b, head[k])
        end = _between(a, b, 1.0 - tail[k])
        # The straight part of piece k, from where the last curve ended to where the next begins.
        n = _steps_for(start, end, position_res, rotation_res) if tail[k] + head[k] < 1.0 else 0
        for step in range(1, n + 1):
            add(_between(start, end, step / n), -1)
        if k + 1 >= len(lengths):
            break
        # The curve around corner k + 1 (inner corner index k), from end (on piece k) to the start of piece k + 1.
        if tail[k] <= 0.0 and head[k + 1] <= 0.0:
            continue
        nxt = control[k + 2]
        a_pose, v_pose, b_pose = end, b, _between(b, nxt, head[k + 1])
        # The curve moves at most twice as fast as along its longer leg (its derivative at s is
        # 2((1 - s)(V - A) + s(B - V))): enough samples for both resolutions everywhere on it.
        n = 2 * max(_steps_for(a_pose, v_pose, position_res, rotation_res),
                    _steps_for(v_pose, b_pose, position_res, rotation_res))
        for step in range(1, n + 1):
            s = step / n
            first = _between(a_pose, v_pose, s)
            second = _between(v_pose, b_pose, s)
            add(_between(first, second, s), k)
    # Exactly the path's ends, as given (the samples already are, up to rounding and the quaternion's sign).
    poses[0] = tuple(np.asarray(v, dtype=float) for v in path_poses[0])
    poses[-1] = tuple(np.asarray(v, dtype=float) for v in path_poses[-1])
    out = [(tuple(float(v) for v in p), tuple(float(v) for v in q)) for p, q in poses]
    return out, owner


def _failed_index(reason: Optional[str]) -> Optional[int]:
    """The waypoint a re-solve stopped at (``collision_at_waypoint_17`` -> 17), or None."""
    match = _WAYPOINT.search(reason or "")
    return int(match.group(1)) if match else None


def round_dual_arm_pose_path(
    path_poses: Sequence[PoseLike],
    path_confs: Sequence[FullConf],
    *,
    scene: Dict[str, Any],
    joint_collision_fn: Optional[Callable[[FullConf], bool]] = None,
    feature_points: Optional[Sequence[np.ndarray]] = None,
    position_res: float = 0.01,
    rotation_res: float = 0.025,
    radius: float = DEFAULT_ROUND_RADIUS,
    tolerance: float = DEFAULT_ROUND_TOLERANCE,
    min_radius: float = DEFAULT_ROUND_MIN_RADIUS,
    min_tolerance: float = DEFAULT_ROUND_MIN_TOLERANCE,
    joint_continuity_threshold_rad: Optional[float] = None,
    use_angle_normalization: bool = DEFAULT_USE_ANGLE_NORMALIZATION,
    max_rounds: int = 40,
    profile_out: Optional[Dict[str, Any]] = None,
) -> Tuple[List[PoseLike], List[FullConf]]:
    """Round the corners of a held bar's path in pose space and solve its joints again.

    The corners are found on the bar path (``pose_path_corners``) and rounded (``round_pose_path``); the joint
    path is solved again along the rounded poses with ``repair_joint_path`` (IK at every pose, the continuity
    gate, the collision check), from the path's first configuration and landing on its last. A corner whose
    curve fails (no IK, a joint jump, a collision) gets half its radius and the path is solved again; below
    ``min_radius`` that corner stays sharp. A path within the tolerance of one straight piece becomes that piece.
    When the straight pieces fail with every corner sharp, or a sample on one collides or has no IK (the radii
    cannot move it), the corners are looked for again at half the tolerance,
    down to ``min_tolerance``. When nothing works the path comes back unchanged.

    Args:
        path_poses (Sequence[PoseLike]): the bar poses, start first.
        path_confs (Sequence[FullConf]): the joint path along them (its first and last are kept exactly).
        scene (Dict[str, Any]): the solver context of ``reconstruct_joint_path_for_pose_path`` (robot, arm joints,
            tool links, both grasps, optional ``ik_options``).
        joint_collision_fn (Optional[Callable[[FullConf], bool]]): the full collision predicate (True = colliding).
        feature_points (Optional[Sequence[np.ndarray]]): bar-local feature points (default: the box corners).
        position_res (float): largest bar move between samples of the result (m).
        rotation_res (float): largest bar turn between samples of the result (rad).
        radius (float): the corner radius tried first (m, largest feature-point move from the corner).
        tolerance (float): straightness tolerance of the corner search tried first (m).
        min_radius (float): below this radius a corner is left sharp (m).
        min_tolerance (float): the finest tolerance tried (m).
        joint_continuity_threshold_rad (Optional[float]): the per-step joint gate of the re-solve.
        use_angle_normalization (bool): forwarded to the IK.
        max_rounds (int): most re-solves in all before giving up and returning the path unchanged.
        profile_out (Optional[Dict[str, Any]]): filled with ``rounded`` (bool), ``tolerance`` and ``n_corners``
            (of the corner search that worked or was tried last), ``radii`` (final, per inner corner), ``rounds``,
            ``failures``, ``reason`` (when unchanged), ``wall_s``.

    Returns:
        Tuple[List[PoseLike], List[FullConf]]: the rounded poses and joints, or the given ones when rounding failed.
    """
    started = time.perf_counter()
    profile: Dict[str, Any] = profile_out if profile_out is not None else {}
    feature_points = list(feature_points) if feature_points is not None else get_bar_feature_points()
    poses_in = [(tuple(float(v) for v in p), tuple(float(v) for v in q)) for p, q in path_poses]
    confs_in = [np.asarray(conf, dtype=float) for conf in path_confs]
    if len(poses_in) != len(confs_in):
        raise ValueError("Pose and joint path lengths must match for rounding.")
    profile.update({"rounded": False, "rounds": 0, "reason": None, "failures": [], "tolerance": tolerance,
                    "n_corners": 0})

    def unchanged(reason: str) -> Tuple[List[PoseLike], List[FullConf]]:
        profile.update({"reason": reason, "radii": [], "wall_s": time.perf_counter() - started})
        return list(path_poses), confs_in

    if radius <= 0.0:
        return unchanged("radius 0")
    while True:
        corners = pose_path_corners(poses_in, feature_points, tolerance)
        radii = [float(radius)] * max(0, len(corners) - 2)
        profile.update({"tolerance": tolerance, "n_corners": len(radii)})
        # * Each try is the straight pieces between the corners with the rounded ones curved; with no corner (or
        # * every one sharp) it is the straight pieces alone, a shortcut of the path.
        while True:
            if profile["rounds"] >= max_rounds:
                return unchanged(f"no valid rounding in {max_rounds} re-solves")
            profile["rounds"] += 1
            poses, owner = round_pose_path(poses_in, corners, radii, position_res=position_res,
                                           rotation_res=rotation_res, feature_points=feature_points)
            confs, info = repair_joint_path(scene, poses, confs_in[0], confs_in[-1],
                                            joint_collision_fn=joint_collision_fn,
                                            joint_continuity_threshold_rad=joint_continuity_threshold_rad,
                                            use_angle_normalization=use_angle_normalization)
            if confs is not None:
                profile.update({"rounded": True, "radii": list(radii), "n_waypoints": len(poses),
                                "wall_s": time.perf_counter() - started})
                return poses, confs
            if not any(r > 0.0 for r in radii):
                profile["failures"].append({"tolerance": tolerance, "reason": info.get("reason"),
                                            "forward": info.get("forward_reason"),
                                            "backward": info.get("backward_reason"), "shrunk": []})
                break
            targets = _blamed_corners(info, owner, radii, len(poses))
            if targets is None:
                # A collision or no IK on a straight part: that sample is on the chord whatever the radii are.
                profile["failures"].append({"tolerance": tolerance, "reason": info.get("reason"),
                                            "forward": info.get("forward_reason"),
                                            "backward": info.get("backward_reason"), "shrunk": None})
                break
            for k in targets:
                radii[k] = radii[k] / 2.0 if radii[k] / 2.0 >= min_radius else 0.0
            profile["failures"].append({"tolerance": tolerance, "reason": info.get("reason"),
                                        "forward": info.get("forward_reason"),
                                        "backward": info.get("backward_reason"), "shrunk": targets})
        # * The straight pieces fail on their own, with every corner sharp: look for the corners more finely.
        if tolerance / 2.0 < min_tolerance:
            return unchanged(f"the straight pieces fail down to {1000.0 * tolerance:g} mm tolerance")
        tolerance /= 2.0


def _blamed_corners(info: Dict[str, Any], owner: Sequence[int], radii: Sequence[float],
                    n_poses: int) -> Optional[List[int]]:
    """The corners to shrink after a failed re-solve, or None when no radius can help.

    The corner whose curve the forward walk stopped in; from the backward walk's stop when the forward one names
    none; every rounded corner when neither names a waypoint. A collision or no IK on a straight part is None: that
    sample lies on the chord whatever the radii are (only the IK branch the walk brings in differs), so a finer corner
    search is tried instead of shrinking every corner first. A joint jump there
    may come from the curve before it (the walk carries its joints on), so the nearest rounded corner before it is
    blamed, else the one after it.
    """
    reason = info.get("forward_reason")
    index = _failed_index(reason)
    if index is None:
        reason = info.get("backward_reason")
        back = _failed_index(reason)
        index = None if back is None else n_poses - 1 - back
    rounded = [k for k, r in enumerate(radii) if r > 0.0]
    if index is None:
        return rounded
    index = min(max(index, 0), len(owner) - 1)
    if owner[index] >= 0 and radii[owner[index]] > 0.0:
        return [owner[index]]
    if owner[index] < 0 and str(reason).startswith(("collision", "ik_failure")):
        return None
    before = [owner[i] for i in range(index, -1, -1) if owner[i] >= 0 and radii[owner[i]] > 0.0]
    after = [owner[i] for i in range(index, len(owner)) if owner[i] >= 0 and radii[owner[i]] > 0.0]
    return [before[0]] if before else ([after[0]] if after else rounded)
