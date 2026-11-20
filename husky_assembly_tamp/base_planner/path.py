# * Copy of husky-assembly-teleop `husky_assembly_teleop/plugins/base_planner/path.py` at monitor commit b75b9975.
# * Kept close to the original so it can be re-synced: what is here is verbatim, the rest is cut.
# * Cut: `BasePath` and the timing (`timed_path`, speeds, `speed_scales`, `slowed`), the follower (`driven`,
# *      `move_margins`, `_heading`, the crl_husky imports), the RRT's helpers (`steer_cost`, `steer_points`,
# *      `round_corners`, `_arc`, ARC_STEP).
# * Changed: the module docstring's summary line.
"""
The turn-drive-turn "steer" between two base poses, and a path in small steps.

Pure math, safe on any thread.
"""

from __future__ import annotations

import math

import numpy as np

#: Closer than this counts as already there; the move only turns.
POSITION_TOLERANCE = 0.005  # m
#: Drive after a turn on the spot over which the follower brings the base back onto the path, metres.
# ? A guess of a few lookaheads of the pure pursuit (0.2 m), not measured.
SETTLE_DISTANCE = 0.5


def steer(q1, q2) -> list[tuple[float, float, float]]:
    """The corner poses of the move from `q1` to `q2`: turn, drive forwards or backwards (less turning), turn.

    ! Yaws continue from `q1` and may lie outside (-pi, pi]: the last pose has `q2`'s heading, not its yaw number.

    Args:
        q1: (x, y, yaw) to start from, metres and radians.
        q2: (x, y, yaw) to end at.

    Returns:
        list[tuple[float, float, float]]: One to three poses after `q1`, ending at `q2`.
    """
    x0, y0, yaw0 = (float(value) for value in q1)
    x1, y1, yaw1 = (float(value) for value in q2)
    corners = []
    yaw = yaw0
    if math.hypot(x1 - x0, y1 - y0) > POSITION_TOLERANCE:
        heading = math.atan2(y1 - y0, x1 - x0)
        forward = abs(_angle(heading - yaw0)) + abs(_angle(yaw1 - heading))
        reverse = abs(_angle(heading + math.pi - yaw0)) + abs(_angle(yaw1 - heading - math.pi))
        yaw = yaw0 + _angle((heading if forward <= reverse else heading + math.pi) - yaw0)
        corners += [(x0, y0, yaw), (x1, y1, yaw)]
    corners.append((x1, y1, yaw + _angle(yaw1 - yaw)))
    return corners


def densify(poses, position_step: float, yaw_step: float) -> list[tuple[float, float, float]]:
    """The path through `poses` in small steps, each a straight blend between two poses.

    Args:
        poses: (x, y, yaw) waypoints, yaw unwrapped.
        position_step: Largest step while driving, metres.
        yaw_step: Largest step while turning, radians.

    Returns:
        list[tuple[float, float, float]]: Poses after the first, ending with the last.
    """
    points, last = [], np.array(poses[0], dtype=float)
    for pose in poses[1:]:
        pose = np.array(pose, dtype=float)
        count = max(int(math.ceil(max(np.linalg.norm(pose[:2] - last[:2]) / position_step,
                                      abs(pose[2] - last[2]) / yaw_step))), 1)
        points += [tuple(last + (pose - last) * (i / count)) for i in range(1, count + 1)]
        last = pose
    return points


def _angle(angle: float) -> float:
    """An angle folded into (-pi, pi]: the shortest turn to the same heading."""
    return math.atan2(math.sin(angle), math.cos(angle))
