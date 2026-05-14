from __future__ import annotations

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
