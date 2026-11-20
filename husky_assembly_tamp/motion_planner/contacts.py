"""Contacts a held transfer may not make: what rides on the arms against what stands in the scene.

The core lets the parts of a pending mate touch during the whole movement that ends at the mate (the tool and the
receiver, the two halves, each half and the other's bar). A carry would then cut through the receiver on its way.
"""
from __future__ import annotations

from typing import List, Tuple

from compas_fab.robots import RobotCellState


def strict_carry_state(state: RobotCellState) -> Tuple[RobotCellState, List[Tuple[str, str]]]:
    """A copy of a state where nothing the arms carry or wear may touch a standing body.

    Args:
        state: The scene of a held transfer; not changed.

    Returns:
        Tuple[RobotCellState, List[Tuple[str, str]]]: The copy, and the allowances it lost as (carried body or arm
        tool, standing body). Contacts among the carried bodies and the tools stay, and so do the robot links each
        body may touch.
    """
    out = state.copy()
    bodies = out.rigid_body_states
    carried = {name for name, body in bodies.items() if body.attached_to_link or body.attached_to_tool}
    arm_tools = {name for name, tool in (out.tool_states or {}).items() if tool.attached_to_group}
    removed = set()
    for name, body in bodies.items():
        touches = list(body.touch_bodies or [])
        if name in carried:
            drop = [other for other in touches if other in bodies and other not in carried]
            removed.update((name, other) for other in drop)
        else:
            drop = [other for other in touches if other in carried or other in arm_tools]
            removed.update((other, name) for other in drop)
        if drop:
            body.touch_bodies = [other for other in touches if other not in drop]
    return out, sorted(removed)
