"""Collision verdicts of one exported bar scene, printed as one line of JSON.

Used by ``test_holding_prm.py``. It runs as its own program because one Python
process can hold only one simulator scene, and the test compares two scenes:

* ``real``: the exported scene with everything in it (built bars, joints,
  support robots), the robot at its exported base pose. Prints the verdict of
  the full check, of the robot-side check and of the scene-only check.
* ``trimmed``: the scene the roadmap is precomputed in (robot, tools and the
  carried bar only, base at the origin). Prints the verdict of the full check.

Both modes test the same configurations (same random seed). The time per
check of each kind is printed too (``ms``), and what the scene holds. Run as::

    python collision_split_helper.py <real|trimmed> <problem folder> <bar id> <how many>

With a fifth argument, the result is also written to that file (JSON).
"""

from __future__ import annotations

import os
import sys
from json import dumps
from time import perf_counter
from typing import Any, Callable, Dict, List

import numpy as np
from compas.data import json_load

from husky_assembly_tamp.benchmark.scene import load_trimmed_problem, open_scene
from husky_assembly_tamp.motion_planner.holding_prm.sampling import make_collision_fns
from husky_assembly_tamp.motion_planner.holding_prm.scenes import (
    build_scene_cell,
    filled_start_state,
    load_carry_movement,
    problem_source_key,
    scene_body_names,
)


def sample_configurations(home12: np.ndarray, goal12: np.ndarray, count: int) -> np.ndarray:
    """Configurations spread around the line from home to the goal posture.

    Args:
        home12 (np.ndarray): the home configuration (12 joint values).
        goal12 (np.ndarray): the authored goal configuration.
        count (int): how many to draw (besides home and the goal themselves).

    Returns:
        np.ndarray: (count + 2, 12) configurations; a mix of free and colliding ones.
    """
    rng = np.random.default_rng(0)
    confs = [home12, goal12]
    for number in range(count):
        if number % 2 == 0:
            # Somewhere along the line, pushed off it.
            confs.append(home12 + rng.uniform() * (goal12 - home12) + rng.normal(0.0, 0.25, 12))
        else:
            # Near the goal, where the built structure is.
            confs.append(goal12 + rng.normal(0.0, 0.12, 12))
    return np.asarray(confs)


def timed(check: Callable[[np.ndarray], bool], confs: np.ndarray) -> Dict[str, Any]:
    """Run one collision check over the configurations: its verdicts and its time per check (ms)."""
    start = perf_counter()
    hits = [bool(check(conf)) for conf in confs]
    return {"hits": hits, "ms": 1000.0 * (perf_counter() - start) / len(confs)}


def verdicts(mode: str, problem_dir: str, bar_id: str, count: int) -> Dict[str, Any]:
    """Open the scene of the given mode and collision-check the test configurations.

    Args:
        mode (str): "real" or "trimmed" (see the module text).
        problem_dir (str): the exported problem folder.
        bar_id (str): the bar, e.g. "B4".
        count (int): how many random configurations.

    Returns:
        Dict[str, Any]: per check name, True where a configuration collides;
        ``ms`` (time per check of each kind) and ``scene`` (how many bodies of
        each kind the scene shows).
    """
    root = os.environ["HUSKY_BENCH_ROOT"]
    problem, trimmed_cell, trimmed_state = load_trimmed_problem(root, problem_source_key(problem_dir), bar_id)
    _action, movement, _home = load_carry_movement(problem_dir, bar_id)
    names = problem.joint_names_12
    home12 = np.asarray(problem.home_conf12, dtype=float)
    goal12 = np.asarray([float(movement.target_configuration[name]) for name in names])
    confs = sample_configurations(home12, goal12, count)

    if mode == "trimmed":
        scene = open_scene(trimmed_cell, trimmed_state, problem)
        checks = make_collision_fns(scene.planner, trimmed_state, names)
        full = timed(checks["full"], confs)
        return {"full": full["hits"], "ms": {"full": full["ms"]},
                "scene": {key: len(value) for key, value in scene_body_names(trimmed_state).items()}}

    # * The real scene: the exported cell, cut down to what this movement shows.
    whole_cell = json_load(os.path.join(problem_dir, "RobotCell.json"))
    state = filled_start_state(movement.start_state, whole_cell, names, home12)
    cell, small_state = build_scene_cell(whole_cell, state)
    scene = open_scene(cell, small_state, problem)
    checks = make_collision_fns(scene.planner, small_state, names)
    results = {key: timed(checks[key], confs) for key in ("full", "robot_side", "scene_only")}
    return {**{key: result["hits"] for key, result in results.items()},
            "ms": {key: result["ms"] for key, result in results.items()},
            "scene": {key: len(value) for key, value in scene_body_names(small_state).items()}}


if __name__ == "__main__":
    # The scene opening prints its own lines; the result is the line starting with RESULT.
    result = verdicts(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
    print("RESULT " + dumps(result))
    if len(sys.argv) > 5:
        with open(sys.argv[5], "w", encoding="utf-8") as handle:
            handle.write(dumps({"mode": sys.argv[1], "bar": sys.argv[3], **result}))
