"""Joint-trajectory export/load helpers for Stage 1/2/3 studies."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from compas.data import json_dump, json_load
from compas_fab.robots import JointTrajectory, JointTrajectoryPoint
from compas_robots.model import Joint


def save_path_as_joint_trajectory(path: Sequence[Sequence[float]], joint_names: Sequence[str], out_path: str) -> None:
    points = []
    for conf in path:
        point = JointTrajectoryPoint(
            joint_values=[float(v) for v in conf],
            joint_types=[Joint.REVOLUTE] * len(conf),
            joint_names=list(joint_names),
        )
        points.append(point)
    traj = JointTrajectory(
        trajectory_points=points,
        joint_names=list(joint_names),
    )
    json_dump(traj, out_path)


def load_joint_trajectory_as_path(json_path: str, joint_names: Optional[Sequence[str]] = None) -> List[np.ndarray]:
    traj = json_load(json_path)
    points = getattr(traj, "points", None) or getattr(traj, "trajectory_points", None)
    if points is None:
        raise ValueError(f"JointTrajectory JSON missing points: {json_path}")
    loaded_joint_names = list(getattr(traj, "joint_names", []) or [])
    if joint_names and loaded_joint_names and loaded_joint_names != list(joint_names):
        idx_map = [loaded_joint_names.index(name) for name in joint_names]
        return [np.asarray([point.joint_values[i] for i in idx_map], dtype=float) for point in points]
    return [np.asarray(point.joint_values, dtype=float) for point in points]
