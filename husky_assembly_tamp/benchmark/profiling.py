"""Record where each planning run spends its time.

Two products come out of one profiling session:

    a flame graph   written as speedscope JSON, openable in the dashboard or
                    at speedscope.app, for looking at one run in detail
    a breakdown     a handful of numbers -- how many seconds went into
                    collision checking, into IK, into smoothing -- that can be
                    averaged over thousands of runs and charted

The breakdown is the one that scales. It is computed by walking the sampled
call tree and attributing each frame to the first component whose function
matches, so a second spent inside compas_fab's collision check is counted as
collision checking no matter how deep in the search it happened.

Sampling costs a few percent of runtime. That is measured rather than assumed:
see the ``calibrate`` stage.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

from pyinstrument import Profiler
from pyinstrument.renderers import SpeedscopeRenderer


# * How often the profiler samples the stack. 1 ms is fine-grained enough to
# * resolve a single collision check (~10 ms) while adding only a few percent
# * of overhead.
SAMPLE_INTERVAL_S = 0.001

# * The components a run's time is split into, in priority order: a frame is
# * attributed to the FIRST entry whose function name matches, and its whole
# * subtree goes with it. Order matters where components nest -- collision
# * checking happens inside the IK, so it is listed first and the IK entry
# * then only picks up IK time that was not collision checking.
# *
# * Each entry is (component name, function names to match).
COMPONENT_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("collision", ("check_collision", "_check_collision")),
    ("state_sync", ("set_robot_cell_state", "_set_robot_cell_state")),
    ("ik", ("ssik_arm_branches", "solve", "solve_dual_arm_pose_ik",
            "_solve_dual_arm_pose_ik_ssik")),
    ("fk_gate", ("validate_dual_arm_bar_pose", "_fk_link_pose_pp")),
    ("interpolation", ("interpolate_poses",)),
    ("nearest_node", ("nearest_node", "pose_distance", "pose_to_feature_vec")),
    ("smoothing", ("smooth_dual_arm_pose_path",)),
    ("stitching", ("_stitch_forward", "_stitch_backward")),
)


def _component_for(function_name: str) -> Optional[str]:
    """Return which component a function belongs to, if any.

    Args:
        function_name (str): the function name from a profiler frame.

    Returns:
        Optional[str]: the component name, or None when the frame is not the
        top of any component.
    """
    for component, names in COMPONENT_RULES:
        if function_name in names:
            return component
    return None


def breakdown_from_session(session) -> Dict[str, float]:
    """Split a profiling session's time across the known components.

    Walks the call tree from the root. When a frame is the top of a component,
    its whole subtree is charged to that component and the walk does not
    descend further -- so nested calls are counted once, at the outermost
    component they belong to. Anything never attributed lands in ``other``.

    Args:
        session: a pyinstrument ``Session``.

    Returns:
        Dict[str, float]: seconds per component, plus ``total`` and ``other``.
    """
    totals: Dict[str, float] = {name: 0.0 for name, _ in COMPONENT_RULES}
    root = session.root_frame()
    if root is None:
        return {**totals, "total": 0.0, "other": 0.0}

    def walk(frame) -> None:
        """Charge this frame to a component, or recurse into its children."""
        component = _component_for(frame.function)
        if component is not None:
            totals[component] += float(frame.time)
            return
        for child in frame.children:
            walk(child)

    walk(root)
    total = float(root.time)
    attributed = sum(totals.values())
    return {**totals, "total": total, "other": max(0.0, total - attributed)}


@contextmanager
def profiled(output_path: Optional[str] = None,
             *, enabled: bool = True) -> Iterator[Dict[str, Any]]:
    """Profile a block of work and hand back its breakdown.

    Used as::

        with profiled(path) as profile:
            ...do the planning...
        profile["breakdown"]   # now filled in

    The dict is filled in when the block exits, so read it afterwards.

    Args:
        output_path (Optional[str]): where to write the speedscope flame graph.
            Skipped when None.
        enabled (bool): when False, does nothing and returns an empty result.
            Lets a caller switch profiling off without restructuring the code.

    Yields:
        Dict[str, Any]: filled in on exit with ``breakdown``, ``speedscope``
        (the path written, if any) and ``enabled``.
    """
    result: Dict[str, Any] = {"enabled": bool(enabled)}
    if not enabled:
        yield result
        result.setdefault("breakdown", {})
        return

    profiler = Profiler(interval=SAMPLE_INTERVAL_S)
    profiler.start()
    try:
        yield result
    finally:
        profiler.stop()
        session = profiler.last_session
        result["breakdown"] = breakdown_from_session(session) if session else {}
        if output_path and session is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            rendered = SpeedscopeRenderer().render(session)
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(rendered)
            result["speedscope"] = output_path


def merge_breakdowns(breakdowns: List[Dict[str, float]]) -> Dict[str, float]:
    """Add several breakdowns together.

    Args:
        breakdowns (List[Dict[str, float]]): the breakdowns to add.

    Returns:
        Dict[str, float]: their sum, component by component.
    """
    merged: Dict[str, float] = {}
    for breakdown in breakdowns:
        for key, value in breakdown.items():
            merged[key] = merged.get(key, 0.0) + float(value)
    return merged
