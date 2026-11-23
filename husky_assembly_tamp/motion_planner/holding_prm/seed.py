"""A carry planned earlier as the seed of a plan from a fixed start (``online.plan_on_roadmap(seed_path=...)``).

Every row of a planned carry holds the bar wherever the robot stands, so the rows can join the graph as a chain.
When the robot stands somewhere else than it was planned for, the rows are first solved again so that the bar
follows the planned path in the world, where the obstacles are, and not in the robot's frame (``adapt_seed``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from husky_assembly_tamp.motion_planner.holding_prm.holding import (
    ARMS,
    HoldingModel,
    bar_from_arm,
    interpolate_pose,
    invert_transform,
    linf,
    pose_gap,
    tool_target_from_bar,
)
from husky_assembly_tamp.motion_planner.holding_prm.sampling import arm_branches
from husky_assembly_tamp.motion_planner.ssik_ik import ARM_SLICE

# * The bar is back on its planned world path after this fraction of the carry (by bar travel): the base's offset is
# * taken up while the bar is still out in the open, near the robot.
SETTLE_FRACTION = 0.5
# * A re-solved row may be this far (largest joint) from the row before it; more means the arm changed its branch.
CONTINUITY_RAD = float(np.radians(20.0))
# * Bar turn counted as travel when the rows are spread along the path, metres per radian.
TRAVEL_M_PER_RAD = 0.5


def adapt_seed(model: HoldingModel, seed: Sequence[np.ndarray], live_from_planned: np.ndarray,
               settle: float = SETTLE_FRACTION) -> Tuple[Optional[List[np.ndarray]], str]:
    """Solve a planned carry again for a robot that stands somewhere else, keeping the bar's planned world path.

    The first row stays (the arms are there). From it the bar's offset from its planned world pose shrinks
    linearly with the bar's travel, to nothing at the fraction ``settle`` of the path: from there on the bar and
    the tools are exactly where they were planned, and only the arms differ.

    Args:
        model (HoldingModel): the holding model at the live base.
        seed (Sequence[np.ndarray]): the planned carry, twelve joints a row.
        live_from_planned (np.ndarray): 4x4, the robot's live base pose times the inverse of the base pose the
            carry was planned at (how the whole robot moved in the world).
        settle (float): the fraction of the bar's travel after which it is on its planned world path.

    Returns:
        Tuple[Optional[List[np.ndarray]], str]: the re-solved rows and ""; or None and why (a pose out of reach, or
        an arm that would have to change its branch).
    """
    rows = [np.asarray(row, dtype=float) for row in seed]
    live_poses = [bar_from_arm(model, "left", row[ARM_SLICE["left"]]) for row in rows]
    travel = [0.0]
    for before, after in zip(live_poses[:-1], live_poses[1:]):
        distance, angle = pose_gap(before, after)
        travel.append(travel[-1] + distance + TRAVEL_M_PER_RAD * angle)
    whole = travel[-1] or 1.0
    planned_from_live = invert_transform(live_from_planned)
    identity = np.eye(4)
    adapted = [rows[0]]
    for index in range(1, len(rows)):
        fraction = min(1.0, travel[index] / whole / settle) if settle > 0.0 else 1.0
        # * Where the bar was planned in the world, moved by what is left of the base's offset.
        world_from_bar = interpolate_pose(live_from_planned, identity, fraction) @ planned_from_live @ live_poses[index]
        row = np.empty(12)
        for arm in ARMS:
            part = ARM_SLICE[arm]
            branches = arm_branches(model, arm, tool_target_from_bar(model, arm, world_from_bar), rows[index][part],
                                    allow_rescue=False, counters={})
            if not branches:
                return None, f"row {index} of {len(rows)}: out of the {arm} arm's reach at this base"
            row[part] = branches[0]
        if linf(row, adapted[-1]) > CONTINUITY_RAD:
            return None, (f"row {index} of {len(rows)}: an arm would jump "
                          f"{np.degrees(linf(row, adapted[-1])):.0f} deg at this base")
        adapted.append(row)
    return adapted, ""


def near_seed_end(found: Sequence[Dict[str, Any]], seed_end: np.ndarray,
                  within_rad: float) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    """The goal postures near a seed's last row, and how far the nearest of all is.

    Args:
        found (Sequence[Dict[str, Any]]): goal postures, each with its ``conf``.
        seed_end (np.ndarray): the seed's last row.
        within_rad (float): the largest joint difference that still counts as near.

    Returns:
        Tuple[List[Dict[str, Any]], Optional[float]]: the near ones, and the nearest posture's distance (largest
        joint, radians; None without postures).
    """
    distances = [linf(np.asarray(each["conf"], dtype=float), seed_end) for each in found]
    near = [each for each, distance in zip(found, distances) if distance <= within_rad]
    return near, (min(distances) if distances else None)


def seed_chains(start_node: int, rows: Sequence[np.ndarray],
                goal_confs: Sequence[np.ndarray]) -> List[Tuple[int, List[np.ndarray]]]:
    """A seed's rows as chains for ``walk_in.add_chains``: from the start node, one chain ending on each goal posture.

    Args:
        start_node (int): the start's node.
        rows (Sequence[np.ndarray]): the seed's rows, thinned (``walk_in.chain_of``).
        goal_confs (Sequence[np.ndarray]): the goal postures the seed may end on.

    Returns:
        List[Tuple[int, List[np.ndarray]]]: the chains; one without a goal posture when there is none.
    """
    ends: List[Optional[np.ndarray]] = list(goal_confs) or [None]
    return [(start_node, list(rows) + ([end] if end is not None else [])) for end in ends]
