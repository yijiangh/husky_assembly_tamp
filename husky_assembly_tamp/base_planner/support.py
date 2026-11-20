# * Copies from husky-assembly-teleop at monitor commit b75b9975, kept close to the originals so they can be re-synced:
# * - `husky_assembly_teleop/shared/planner/planner_search.py`: TIME_LIMIT, `PlanResult` (without `hit`: the
# *   monitor's `CollisionHit` is not copied).
# * - `husky_assembly_teleop/plugins/base_planner/algorithms/base.py`: LEAD, POSITION_STEP, YAW_STEP, `yaw_step`, and
# *   `Debug`, here with the same methods but recording nothing.
# * - `husky_assembly_teleop/plugins/base_planner/planner_plugin.py`: ARRIVAL_SLACK, `kept_margin`, `end_margins`.
# * - `husky_assembly_teleop/world/clearance.py`: BASE_PLAN_DISTANCE.
# * Cut: everything else in those files (widgets, ament, ROS, viser, the follower, the plugin).
"""
What the base planner's search needs from the monitor around it: its result, its steps, and the margins at the ends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, TypeVar

from .world import CLEARANCE_STEP, PlanningWorld

#: Give up searching after this long, seconds.
TIME_LIMIT = 20.0

#: Distance a base plan keeps where it can (the base planner's Margin at start), metres: more than STOP_DISTANCE.
BASE_PLAN_DISTANCE = 0.1

#: Straight drive onto the target (and off the start, for the RRT), metres: no turn on the spot there.
LEAD = 0.5
#: Collision-checking resolution along a move: metres driven, and at most radians turned (`yaw_step`).
POSITION_STEP = 0.05
YAW_STEP = 0.05

#: With a narrow margin, what the start or target is closer to than it keeps ARRIVAL_SLACK less than the end has (room
#: for the follower to arrive a little off), but never less than half of it.
ARRIVAL_SLACK = 0.01  # m

P = TypeVar("P")


@dataclass
class PlanResult(Generic[P]):
    """What a search came back with.

    Attributes:
        path: The path, or None if none was found.
        reason: Why not, when `path` is None; empty otherwise.
        seconds: How long it took, syncing the world included.
        direct: Whether the straight move was already free.
        detail: More on a failed search for the log, e.g. what blocked it; too long for a chip.
    """

    path: P | None
    reason: str = ""
    seconds: float = 0.0
    direct: bool = False
    detail: str = ""


class Debug:
    """Where the monitor's search records its drawings and notes; here every call does nothing.

    Kept so the search's `debug.*` calls stay as they are in the monitor.
    """

    #: Never on: the search draws nothing, and measures nothing only its drawings need.
    on = False

    def layer(self, name: str, color: tuple[int, int, int]) -> None:
        """Declare layer `name`; nothing here."""

    def path(self, name: str, poses, color: tuple[int, int, int]) -> None:
        """Draw a path into layer `name`; nothing here."""

    def lines(self, name: str, segments, color: tuple[int, int, int]) -> None:
        """Draw line segments into layer `name`; nothing here."""

    def points(self, name: str, xy, color: tuple[int, int, int]) -> None:
        """Draw points into layer `name`; nothing here."""

    def note(self, text: str) -> None:
        """Add a line of text; nothing here."""


def yaw_step(world: PlanningWorld, own: str, x_icrs) -> float:
    """The turn between checked poses: YAW_STEP, or less where the robot or what it holds reaches far.

    Turning by it about any turning centre in `x_icrs` (crl_husky xICR, metres) moves no part more than POSITION_STEP.
    """
    return min(YAW_STEP, POSITION_STEP / (world.reach(own) + max(abs(x_icr) for x_icr in x_icrs)))


def kept_margin(distance: float) -> float:
    """The margin a path keeps from something its start or target is `distance` from: just below, to CLEARANCE_STEP."""
    return max(0.0, math.floor((distance - 1e-6) / CLEARANCE_STEP) * CLEARANCE_STEP)


def end_margins(distances: dict[str, float], narrow: bool) -> dict[str, float]:
    """The margins kept from what the start or target is already closer to than the plan's margin, by object.

    As much as the end has (`kept_margin`); with a narrow margin (`narrow`), ARRIVAL_SLACK less, but at least half.
    """
    out = {}
    for object_id, distance in distances.items():
        kept = kept_margin(distance)
        out[object_id] = max(kept - ARRIVAL_SLACK, kept / 2.0) if narrow else kept
    return out
