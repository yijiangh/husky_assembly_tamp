"""Shared ssik helpers for the MOTION planner (PyBullet world, meters).

The keyframe solver already has an ssik backend, but it works in compas /
RobotCellState / millimeter terms. The motion planner (task-space RRT + linear
cartesian loops) lives in raw PyBullet terms: body ids, link indices,
``(pos, quat)`` poses in meters, and a flat 12-vector of arm joints (left 6
then right 6). This module is the thin adapter between that world and the
in-process ssik solver (``keyframe.ssik_inprocess``):

  * convert a world tool0 pose into the arm base-link frame (what ssik expects),
  * re-branch returned solutions onto the seed's 2*pi branch (see below),
  * enumerate branches sorted nearest-to-seed,
  * one-time joint-order guard between the URDF-in-PyBullet and the artifacts.

! The 2*pi re-branch is load-bearing, not cosmetic: ssik ranks and dedups its
! branches with WRAP-to-pi metrics, while our continuity checks use RAW joint
! deltas (~10 deg threshold). UR limits are wider than +-pi (shoulder_pan
! +-2*pi, shoulder_lift +-4.71, wrists +-4.19), so ssik can legally return the
! 2*pi twin of the seed (wrap-distance 0, raw distance 6.28) -- which would
! look like a huge jump and kill the RRT / cartesian tracking. We shift every
! returned joint by k*2*pi toward the seed whenever the shifted value stays
! inside the limits.

Backend selection reuses ``keyframe.config.IK_BACKEND`` (env HUSKY_IK_BACKEND,
"ssik" default / "gradient" fallback) so ONE env var switches both the keyframe
and the motion IK.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pybullet_planning as pp

from husky_assembly_tamp.keyframe import config as keyframe_config
from husky_assembly_tamp.keyframe import ssik_inprocess


# * Which URDF link each arm's ssik solver is rooted at / solves for. Must match
# * the `ssik build` recipe in keyframe.config.SSIK_ARM_BUILD.
ARM_BASE_LINK: Dict[str, str] = {
    "left": "left_ur_arm_base_link",
    "right": "right_ur_arm_base_link",
}

# * Where each arm's 6 joints live inside the flat 12-vector (left 6 then right 6).
ARM_SLICE: Dict[str, slice] = {
    "left": slice(0, 6),
    "right": slice(6, 12),
}

# Robot body ids whose joint order has already been checked against the artifacts.
_JOINT_ORDER_CHECKED: set = set()


def ik_backend() -> str:
    """The active IK backend name, shared with the keyframe solver.

    Returns:
        str: "ssik" (default) or "gradient" -- from ``HUSKY_IK_BACKEND``
        (validated at ``keyframe.config`` import; typos raise there).
    """
    return keyframe_config.IK_BACKEND


def verify_ssik_joint_order(robot: int) -> None:
    """One-time guard: artifact joint order must match the PyBullet URDF chain.

    The 6 values ssik returns are in kinematic base->tool0 order; we splice them
    straight into the 12-vector, which is built from the URDF chain order. This
    asserts (once per robot body id) that every artifact joint name exists in
    the loaded robot and that their PyBullet joint indices are strictly
    increasing, i.e. the chain order agrees. The per-solve FK validation
    downstream would also catch a mismatch, but this fails FAST and with a
    readable message.

    Args:
        robot (int): PyBullet body id of the dual-arm robot.

    Raises:
        AssertionError: when a joint name is missing or the order disagrees.
    """
    if robot in _JOINT_ORDER_CHECKED:
        return
    for arm in ("left", "right"):
        names = ssik_inprocess.joint_names(arm)
        # pp.joint_from_name raises if the name is absent from the body.
        indices = [pp.joint_from_name(robot, n) for n in names]
        assert indices == sorted(indices), (
            f"ssik/{arm}: artifact joint order {names} does not follow the "
            f"URDF chain order in PyBullet (indices {indices})"
        )
    _JOINT_ORDER_CHECKED.add(robot)


def target_in_arm_base(robot: int, arm: str, world_tool_pose) -> np.ndarray:
    """Express a world tool0 pose in the arm's base-link frame (ssik's input).

    Reads the base link's CURRENT world pose from PyBullet each call -- the
    mobile base placement changes between plans/bars, and a getLinkState call
    costs microseconds, so caching would only add a staleness bug.

    Args:
        robot (int): PyBullet body id.
        arm (str): "left" or "right".
        world_tool_pose: PyBullet ``(pos, quat_xyzw)`` world pose of tool0.

    Returns:
        np.ndarray: 4x4 transform of tool0 in the arm base-link frame, meters.
    """
    base_link = pp.link_from_name(robot, ARM_BASE_LINK[arm])
    world_from_base = pp.tform_from_pose(pp.get_link_pose(robot, base_link))
    world_from_tool = pp.tform_from_pose(world_tool_pose)
    return np.linalg.inv(world_from_base) @ world_from_tool


def rebranch_toward_seed(
    q: np.ndarray,
    seed: Sequence[float],
    limits: Sequence,
) -> np.ndarray:
    """Shift each joint by a multiple of 2*pi onto the seed's branch (limit-checked).

    Args:
        q (np.ndarray): a 6-vector IK solution, radians.
        seed (Sequence[float]): the 6-vector we want to stay close to.
        limits (Sequence): per-joint ``(lower, upper)`` bounds; a shifted value
            is only kept when it stays inside them.

    Returns:
        np.ndarray: the re-branched copy of ``q``.
    """
    q = np.array(q, dtype=float)
    seed = np.asarray(seed, dtype=float)
    # How many full turns to add so each joint lands on the seed's branch.
    shift = 2.0 * np.pi * np.round((seed - q) / (2.0 * np.pi))
    for i, (lo, hi) in enumerate(limits):
        candidate = q[i] + shift[i]
        if lo <= candidate <= hi:
            q[i] = candidate
    return q


def ssik_arm_branches(
    robot: int,
    arm: str,
    world_tool_pose,
    seed6: Optional[Sequence[float]],
    max_solutions: Optional[int] = None,
    allow_rescue: bool = True,
) -> List[np.ndarray]:
    """All ssik branches for one arm at a world tool0 pose, nearest-seed first.

    Args:
        robot (int): PyBullet body id.
        arm (str): "left" or "right".
        world_tool_pose: PyBullet ``(pos, quat_xyzw)`` world target for tool0.
        seed6 (Optional[Sequence[float]]): the arm's current 6 joint values;
            when given, results are re-branched toward it and sorted by raw
            L-infinity distance to it.
        max_solutions (Optional[int]): cap on branches (None = all).
        allow_rescue (bool): forward ssik's numeric-rescue switch. Keep True
            for one-off endpoint solves; set False in tracking loops (an
            unreachable waypoint should fail fast, and rescued solutions can
            branch-flip).

    Returns:
        List[np.ndarray]: 6-vectors in kinematic (== 12-vector slice) order;
        empty list = pose unreachable for this arm.
    """
    verify_ssik_joint_order(robot)
    target = target_in_arm_base(robot, arm, world_tool_pose)
    seed_arr = None if seed6 is None else np.asarray(seed6, dtype=float)
    solutions = ssik_inprocess.solve(
        arm, target, max_solutions=max_solutions, q_seed=seed_arr,
        allow_rescue=allow_rescue,
    )
    branches = [q for q, _residual in solutions]
    if seed_arr is not None:
        limits = ssik_inprocess.joint_limits(arm)
        branches = [rebranch_toward_seed(q, seed_arr, limits) for q in branches]
        branches.sort(key=lambda q: float(np.max(np.abs(q - seed_arr))))
    return branches
