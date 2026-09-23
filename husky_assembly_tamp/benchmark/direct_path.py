"""Ask whether a start and a goal are joined by a plain straight-line motion.

The planner has a shortcut worth measuring separately. Before searching, its
start derivation walks the bar straight through space from the goal back to a
carry pose, solving IK at every step. If that walk survives -- no unreachable
step, no branch flip, nothing in the way -- the walk IS the answer and the
search never runs.

So a "success" from the planner can mean two quite different things, and a
benchmark that lumps them together would mostly be measuring how often the
problem was trivial. Every start/goal pair the benchmark plans is therefore
classified first: does the direct path already solve it?

The walk itself is production's, ``core.track_direct_path`` -- the same
function the start derivation uses.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence

import numpy as np

from husky_assembly_tamp.benchmark.scene import SceneCtx
from husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt import core


# * The step the planner screens candidate walks at (its own defaults). Finer
# * steps are only used once a walk has already proven clean, so screening at
# * these values is what production actually pays.
SCREEN_POSITION_RES_M = 0.01
SCREEN_ROTATION_RES_RAD = 0.025

# How closely the walk must arrive at the intended start configuration. The
# walk solves IK afresh at every step, so it can land on a different (but
# equally valid) solution than the one we set out for; that is worth knowing
# rather than glossing over.
ARRIVAL_TOLERANCE_RAD = 1e-3


def check_direct_path(scene: SceneCtx, goal_conf: Sequence[float], world_from_bar_goal,
                      start_conf: Sequence[float], world_from_bar_start,
                      collision_fn, *,
                      position_res: float = SCREEN_POSITION_RES_M,
                      rotation_res: float = SCREEN_ROTATION_RES_RAD) -> Dict[str, Any]:
    """Walk the bar straight from the goal to a start and report what happened.

    Walking from the goal (not from the start) is deliberate: that is the
    direction production tracks in, and the direction decides which IK solution
    each step is seeded from, so walking the other way could reach a different
    verdict.

    Args:
        scene (SceneCtx): the live scene.
        goal_conf (Sequence[float]): the configuration at the goal pose.
        world_from_bar_goal: the goal bar pose.
        start_conf (Sequence[float]): the configuration the walk should reach.
        world_from_bar_start: the start bar pose.
        collision_fn: the collision predicate.
        position_res (float): linear step of the walk, metres.
        rotation_res (float): angular step of the walk, radians.

    Returns:
        Dict[str, Any]: what the walk found --

            ``track_ok``            the walk reached the start pose at all
            ``reached_fraction``    how far it got, 0 to 1
            ``arrival_matches``     it arrived at the intended configuration
            ``arrival_error_rad``   how far off the arrival was
            ``free``                the whole walk is collision-free, i.e. the
                                    planner would skip its search here
            ``first_blocked_fraction``  where the first collision sits, if any
            ``n_waypoints``         length of the walk
            ``wall_s``              time this check took
    """
    t0 = time.time()
    track, track_ok = core.track_direct_path(
        robot=scene.robot_puid,
        arm_joints=scene.arm_joints,
        tool_link_left=scene.tool_link_left,
        tool_link_right=scene.tool_link_right,
        grasp_bar_from_left=scene.grasp_bar_from_left,
        grasp_bar_from_right=scene.grasp_bar_from_right,
        world_from_bar_from=world_from_bar_goal,
        conf_from=goal_conf,
        world_from_bar_to=world_from_bar_start,
        position_res=position_res,
        rotation_res=rotation_res,
    )

    result: Dict[str, Any] = {
        "track_ok": bool(track_ok),
        "n_waypoints": len(track),
        "free": False,
        "first_blocked_fraction": None,
        "arrival_matches": False,
        "arrival_error_rad": None,
    }

    # How much of the way it got, measured on the bar rather than on joints so
    # it means the same thing for every bar and grasp.
    goal_position = np.asarray(world_from_bar_goal[0], dtype=float)
    start_position = np.asarray(world_from_bar_start[0], dtype=float)
    span = float(np.linalg.norm(start_position - goal_position))
    if span < 1e-9:
        result["reached_fraction"] = 1.0
    else:
        reached = float(np.linalg.norm(np.asarray(track[-1][0][0], dtype=float) - goal_position))
        result["reached_fraction"] = min(1.0, reached / span)

    if track_ok:
        arrival = np.asarray(track[-1][1], dtype=float)
        error = float(np.max(np.abs(arrival - np.asarray(start_conf, dtype=float))))
        result["arrival_error_rad"] = error
        result["arrival_matches"] = error <= ARRIVAL_TOLERANCE_RAD

        blocked = core.first_blocked_index(track, collision_fn)
        if blocked is None:
            result["free"] = True
        else:
            result["first_blocked_fraction"] = float(blocked) / max(1, len(track) - 1)

    result["wall_s"] = time.time() - t0
    return result


def direct_path_trajectory(scene: SceneCtx, goal_conf: Sequence[float], world_from_bar_goal,
                           world_from_bar_start, *,
                           position_res: float, rotation_res: float):
    """Return the direct-path motion from a start to a goal, if there is one.

    The walk runs goal-to-start, the same direction production tracks in, and
    the result is reversed so it reads as a motion the robot would execute.

    Args:
        scene (SceneCtx): the live scene.
        goal_conf (Sequence[float]): the configuration at the goal pose.
        world_from_bar_goal: the goal bar pose.
        world_from_bar_start: the start bar pose.
        position_res (float): linear step, metres.
        rotation_res (float): angular step, radians.

    Returns:
        Optional[list]: the configurations from start to goal, or None when the
        walk did not survive.
    """
    track, track_ok = core.track_direct_path(
        robot=scene.robot_puid,
        arm_joints=scene.arm_joints,
        tool_link_left=scene.tool_link_left,
        tool_link_right=scene.tool_link_right,
        grasp_bar_from_left=scene.grasp_bar_from_left,
        grasp_bar_from_right=scene.grasp_bar_from_right,
        world_from_bar_from=world_from_bar_goal,
        conf_from=goal_conf,
        world_from_bar_to=world_from_bar_start,
        position_res=position_res,
        rotation_res=rotation_res,
    )
    if not track_ok:
        return None
    return [np.asarray(conf, dtype=float) for _pose, conf in reversed(track)]
