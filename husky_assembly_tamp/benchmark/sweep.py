"""Stage 2: survey every carry pose the robot could start a transfer from.

Before the planner searches, it has to decide where the motion begins. The bar
arrives held in front of the robot in one of three carry poses -- across the
front, upright, or over the back -- and the planner sweeps a grid of small
offsets around each, spinning the bar about its own axis at each offset, until
it finds one the arms can hold without colliding. It takes the first that
works and searches from there.

The benchmark wants the whole picture instead: every offset, every spin, every
anchor, and for each one whether it worked and if not why. That tells us how
much choice the planner really had, whether some anchors are dead weight, and
which links are doing the blocking.

This is production's own sweep with its early exit switched off (see
``derive_constrained_start(exhaustive=True)``). The candidate it would have
picked is still identifiable: it is the first feasible one in visit order.

The sweep is by far the most expensive stage. A candidate the robot can hold
costs one IK solve and one collision check; a candidate that collides costs up
to sixty-four of them, because every way of grasping it has to be ruled out.
The grid step is therefore a knob, and the `measure` stage exists to tell you
what a given step will cost before you commit a night to it.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet_planning as pp

from husky_assembly_tamp.benchmark.direct_path import check_direct_path
from husky_assembly_tamp.benchmark.scene import SceneCtx, pose_to_lists
from husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt import core


# * Production's own sweep box and step: offsets up to 30 cm around each carry
# * anchor, sampled every 10 cm, which is 7x7x7 = 343 offsets per anchor.
DEFAULT_SWEEP_BOX = ((-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3))
DEFAULT_SWEEP_STEP_M = 0.1


def sweep_goal_starts(scene: SceneCtx, goal_conf: Sequence[float], world_from_bar_goal,
                      *, sweep_step_m: float = DEFAULT_SWEEP_STEP_M,
                      sweep_box=DEFAULT_SWEEP_BOX,
                      check_direct_paths: bool = True,
                      verbose: bool = False) -> Dict[str, Any]:
    """Survey every carry-pose candidate for one goal.

    Args:
        scene (SceneCtx): the live scene.
        goal_conf (Sequence[float]): the configuration at the goal.
        world_from_bar_goal: the goal bar pose.
        sweep_step_m (float): spacing of the offset grid, metres. Production
            uses 0.1; a coarser step surveys fewer candidates and costs
            proportionally less.
        sweep_box: the offset box, as three ``(low, high)`` pairs.
        check_direct_paths (bool): also test, for every feasible start, whether
            a straight motion already joins it to the goal.
        verbose (bool): print a line when the survey finishes.

    Returns:
        Dict[str, Any]: the survey --

            ``candidates``      one record per candidate pose tried
            ``starts``          the feasible ones, with configurations
            ``production_pick`` the start production would have used
            ``counts``          how many candidates ended each way
            ``wall_s``          how long the survey took
    """
    collision_fn = scene.reporting_collision_fn()
    report: List[Dict[str, Any]] = []

    scene.set_state(scene.template_state)
    t0 = time.time()
    _pose, conf = core.derive_constrained_start(
        robot=scene.robot_puid,
        arm_joints=scene.arm_joints,
        tool_link_left=scene.tool_link_left,
        tool_link_right=scene.tool_link_right,
        grasp_bar_from_left=scene.grasp_bar_from_left,
        grasp_bar_from_right=scene.grasp_bar_from_right,
        world_from_bar_goal=world_from_bar_goal,
        seed_conf=np.asarray(goal_conf, dtype=float),
        bar_body=scene.bar_body,
        joint_collision_fn=collision_fn,
        bar_sweep_box=sweep_box,
        bar_sweep_step=sweep_step_m,
        exhaustive=True,
        report=report,
    )
    sweep_s = time.time() - t0

    candidates: List[Dict[str, Any]] = []
    starts: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for rank, record in enumerate(report):
        ik_report = record.get("ik") or {}
        reason = ik_report.get("reason", "ik_unreachable_both")
        counts[reason] = counts.get(reason, 0) + 1

        candidate: Dict[str, Any] = {
            "rank": rank,
            "anchor": record["anchor"],
            "delta": record["delta"],
            "world_from_bar": record["world_from_bar"],
            "reason": reason,
            "wall_s": record["wall_s"],
            "n_branches_left": ik_report.get("n_branches_left", 0),
            "n_branches_right": ik_report.get("n_branches_right", 0),
        }
        # Only failures carry their offending link pairs: for a candidate that
        # worked there is nothing to explain.
        if reason == "collision":
            pairs: List[List[str]] = []
            for entry in ik_report.get("pairs", []):
                pairs.extend(entry.get("collision_pairs", []))
            candidate["collision_pairs"] = pairs
        candidates.append(candidate)

        if record.get("conf") is not None:
            starts.append({
                "start_index": len(starts),
                "rank": rank,
                "anchor": record["anchor"],
                "delta": record["delta"],
                "world_from_bar": record["world_from_bar"],
                "conf": record["conf"],
            })

    # Production takes the first feasible candidate; that start is worth
    # marking so the benchmark can compare it against the rest.
    production_pick = starts[0]["start_index"] if starts else None
    if conf is not None and starts:
        if not np.allclose(starts[0]["conf"], conf, atol=1e-9):
            raise RuntimeError(
                "The survey's first feasible start is not the one the sweep "
                "returned. The exhaustive survey no longer reproduces what "
                "production would pick, which invalidates the comparison."
            )

    result: Dict[str, Any] = {
        "candidates": candidates,
        "starts": starts,
        "production_pick": production_pick,
        "counts": counts,
        "n_candidates": len(candidates),
        "n_starts": len(starts),
        "sweep_step_m": sweep_step_m,
        "wall_s": sweep_s,
    }

    if check_direct_paths and starts:
        direct_t0 = time.time()
        for start in starts:
            scene.set_state(scene.template_state)
            start["direct_path"] = check_direct_path(
                scene,
                goal_conf=goal_conf,
                world_from_bar_goal=world_from_bar_goal,
                start_conf=start["conf"],
                world_from_bar_start=_pose_from_record(start["world_from_bar"]),
                collision_fn=collision_fn,
            )
            start["difficulty"] = difficulty_class(start["direct_path"])
        result["direct_path_wall_s"] = time.time() - direct_t0
        result["n_direct_path_free"] = sum(
            1 for start in starts if start.get("direct_path", {}).get("free")
        )
        result["difficulty_counts"] = {}
        for start in starts:
            label = start["difficulty"]
            result["difficulty_counts"][label] = result["difficulty_counts"].get(label, 0) + 1

    if verbose:
        print(f"    sweep: {len(candidates)} candidates, {len(starts)} feasible, "
              f"{sweep_s:.1f} s ({sweep_s / max(1, len(candidates)) * 1000:.0f} ms each), "
              f"reasons {counts}")
    return result


def difficulty_class(direct_path: Dict[str, Any]) -> str:
    """Label how hard a start looks, judged by the straight motion to it.

    Being able to hold the bar at a carry pose says nothing about being able to
    get there from the goal, and most carry poses turn out to be much harder to
    reach than to occupy. The straight-line probe is a cheap, honest proxy for
    that, and grouping starts by it keeps a benchmark batch from being all
    hopeless problems or all trivial ones.

    Args:
        direct_path (Dict[str, Any]): what ``check_direct_path`` found.

    Returns:
        str: one of --

            ``free``     a straight motion already solves it; the search only
                         has to notice
            ``blocked``  a straight motion reaches the start but passes
                         through something; a detour exists to be found
            ``broken``   a straight motion cannot even reach the start, so any
                         solution has to leave the straight line entirely
    """
    if direct_path.get("free"):
        return "free"
    if direct_path.get("track_ok"):
        return "blocked"
    return "broken"


def _pose_from_record(pose_lists) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Turn a recorded ``[[x, y, z], [qx, qy, qz, qw]]`` back into a pose.

    Args:
        pose_lists: the two lists as stored in a survey record.

    Returns:
        tuple: the ``(position, quaternion)`` pose.
    """
    position, quaternion = pose_lists
    return (tuple(float(v) for v in position), tuple(float(v) for v in quaternion))


def run_production_derivation(scene: SceneCtx, goal_conf: Sequence[float],
                              world_from_bar_goal, *,
                              position_res: float = 0.002,
                              rotation_res: float = 0.002) -> Dict[str, Any]:
    """Run the start derivation production actually uses, and record it.

    The sweep surveyed above is production's fallback. Its first choice is a
    different method: walk the bar backwards from the goal towards each carry
    pose and keep the walk that survives. When that walk is also collision-free
    the planner skips searching altogether, so how often it succeeds is one of
    the more interesting things this benchmark can report.

    Args:
        scene (SceneCtx): the live scene.
        goal_conf (Sequence[float]): the configuration at the goal.
        world_from_bar_goal: the goal bar pose.
        position_res (float): the walk's linear step, metres.
        rotation_res (float): the walk's angular step, radians.

    Returns:
        Dict[str, Any]: whether it found a start, whether the direct path was
        free, which anchor variant won, and its own counters.
    """
    collision_fn = scene.collision_fn()
    scene.set_state(scene.template_state)
    t0 = time.time()
    with pp.WorldSaver():
        world_from_bar_start, start_conf, info = core.derive_constrained_start_tracked(
            robot=scene.robot_puid,
            arm_joints=scene.arm_joints,
            tool_link_left=scene.tool_link_left,
            tool_link_right=scene.tool_link_right,
            grasp_bar_from_left=scene.grasp_bar_from_left,
            grasp_bar_from_right=scene.grasp_bar_from_right,
            world_from_bar_goal=world_from_bar_goal,
            goal_conf=np.asarray(goal_conf, dtype=float),
            position_res=position_res,
            rotation_res=rotation_res,
            joint_collision_fn=collision_fn,
        )
    wall_s = time.time() - t0

    # The walk itself can be long; keep its length, not its contents.
    direct_path = info.pop("direct_path", None)
    result: Dict[str, Any] = {
        "found": start_conf is not None,
        "direct_path_free": direct_path is not None,
        "direct_path_waypoints": len(direct_path[0]) if direct_path else 0,
        "wall_s": wall_s,
        "counters": {key: value for key, value in info.items()
                     if isinstance(value, (int, float, str, bool))},
    }
    if start_conf is not None:
        result["start_conf"] = [float(v) for v in start_conf]
        result["world_from_bar_start"] = pose_to_lists(world_from_bar_start)
    return result
