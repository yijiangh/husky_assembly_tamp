"""A last look at a joint path: is it free of collisions between its waypoints too?"""
from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.path_validation import (
    DEFAULT_DENSE_JOINT_VALIDATION_STEP_RAD,
    refine_joint_path_for_validation,
)


def dense_check(confs: Sequence[Sequence[float]], collides_many: Callable[[List[np.ndarray]], List[bool]],
                step_rad: float = DEFAULT_DENSE_JOINT_VALIDATION_STEP_RAD) -> Dict[str, Any]:
    """Collision-check a path along straight joint moves between its waypoints, as a controller tracks it.

    Args:
        confs: The waypoints.
        collides_many: Says for each of some configurations whether it collides.
        step_rad: The largest move of any joint between two checked points.

    Returns:
        Dict[str, Any]: ``ok``, ``n_points``, ``step_deg``, ``seconds`` and ``first_hit`` (None, or ``segment``: the
        waypoint the colliding piece starts at, ``alpha``: how far along it, ``conf``).
    """
    start = perf_counter()
    dense, meta = refine_joint_path_for_validation([np.asarray(conf, dtype=float) for conf in confs], step_rad,
                                                   use_angle_normalization=False)
    first_hit = None
    for point, entry, hit in zip(dense, meta, collides_many(dense) if dense else []):
        if hit:
            first_hit = {"segment": int(entry["segment_index"]), "alpha": round(float(entry["segment_alpha"]), 4),
                         "conf": [float(v) for v in point]}
            break
    return {"ok": first_hit is None, "n_points": len(dense), "step_deg": round(float(np.degrees(step_rad)), 3),
            "seconds": round(perf_counter() - start, 3), "first_hit": first_hit}
