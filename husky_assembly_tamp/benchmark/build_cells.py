"""Stage 0: turn an exported design study into per-bar benchmark scenes.

The source export is one big robot cell (a few hundred megabytes, a hundred
rigid bodies) plus one pair of action files per bar. Loading that once takes a
while; loading it in each of sixteen worker processes would be unbearable. So
this stage does it once, trims the cell down per bar (see ``scene.py``), and
writes the small results to the benchmark data root. Every later stage and
every worker reads only those.

It also checks each problem it writes: it poses the robot at the authored home
configuration and confirms the bar really does end up where the recorded grasp
says it should. A mismatch means the grasp was read wrongly, and the benchmark
would go on to measure nonsense, so it stops.
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pybullet_planning as pp
from compas.data import json_load

from husky_assembly_tamp.benchmark.scene import (
    BarProblem,
    build_template_state,
    build_trimmed_cell,
    extract_bar_problem,
    open_scene,
    pose_from_lists,
    save_trimmed_problem,
)


# How closely the bar's simulated pose must match the pose implied by the
# recorded grasp. These are far looser than the numbers actually seen (which
# are at float noise) but tight enough that a wrong grasp cannot slip past.
GRASP_CHECK_POSITION_TOL_M = 1e-6
GRASP_CHECK_ORIENTATION_TOL = 1e-6


def source_key_for(problem_dir: str) -> str:
    """Return a short, stable key identifying one source export.

    The key is the folder name plus a hash of the robot cell's size and
    modification time, so re-exporting the study produces a different key and
    the trimmed scenes cached under the old one are never silently reused.

    Args:
        problem_dir (str): the exported problem folder (holds
            ``RobotCell.json`` and ``BarActions/``).

    Returns:
        str: the cache key, e.g. ``"260921_motion_sample-3f2a1b9c"``.

    Raises:
        FileNotFoundError: if the folder has no ``RobotCell.json``.
    """
    cell_path = os.path.join(problem_dir, "RobotCell.json")
    if not os.path.isfile(cell_path):
        raise FileNotFoundError(
            f"No RobotCell.json in {problem_dir}. Point --problem-dir at an "
            "exported design study folder."
        )
    stat = os.stat(cell_path)
    digest = hashlib.sha256(
        f"{stat.st_size}:{int(stat.st_mtime)}".encode("utf-8")
    ).hexdigest()[:8]
    return f"{os.path.basename(os.path.normpath(problem_dir))}-{digest}"


def find_bar_ids(problem_dir: str) -> List[str]:
    """List the bars an export has a full jointing + release pair for.

    Args:
        problem_dir (str): the exported problem folder.

    Returns:
        List[str]: bar ids in assembly order, e.g. ``["B3", "B6", ...]``.

    Raises:
        FileNotFoundError: if the folder has no ``BarActions`` directory.
    """
    actions_dir = os.path.join(problem_dir, "BarActions")
    if not os.path.isdir(actions_dir):
        raise FileNotFoundError(f"No BarActions folder in {problem_dir}.")
    bar_ids = []
    for name in os.listdir(actions_dir):
        if not name.endswith("__J.json"):
            continue
        bar_id = name[: -len("__J.json")]
        if os.path.isfile(os.path.join(actions_dir, f"{bar_id}__R.json")):
            bar_ids.append(bar_id)
    # Sort by the numeric part so B3 comes before B12.
    return sorted(bar_ids, key=lambda b: int("".join(c for c in b if c.isdigit()) or 0))


def verify_grasp(scene, problem: BarProblem) -> Dict[str, float]:
    """Check that the recorded grasp really describes where the bar hangs.

    Poses the robot at its home configuration, reads where PyBullet actually
    puts the bar, and compares that with the pose computed from the left tool0
    and the recorded grasp. The two must agree: the whole benchmark rests on
    that grasp being right.

    Args:
        scene (SceneCtx): the live scene for this problem.
        problem (BarProblem): the problem whose grasp is being checked.

    Returns:
        Dict[str, float]: the measured ``position_error_m`` and
        ``orientation_error`` (quaternion distance).

    Raises:
        ValueError: if either error exceeds its tolerance.
    """
    conf12 = problem.home_conf12
    state = scene.template_state.copy()
    for name, value in zip(problem.joint_names_12, conf12):
        state.robot_configuration[name] = float(value)
    scene.set_state(state)

    simulated = pp.get_pose(scene.bar_body)
    from_grasp = scene.bar_pose_at(conf12)

    position_error = float(
        np.linalg.norm(np.asarray(simulated[0]) - np.asarray(from_grasp[0]))
    )
    # Quaternions q and -q are the same rotation, so compare the absolute dot.
    dot = abs(float(np.dot(np.asarray(simulated[1]), np.asarray(from_grasp[1]))))
    orientation_error = float(1.0 - min(1.0, dot))

    if (position_error > GRASP_CHECK_POSITION_TOL_M
            or orientation_error > GRASP_CHECK_ORIENTATION_TOL):
        raise ValueError(
            f"{problem.bar_id}: the recorded grasp does not reproduce the bar's "
            f"simulated pose (position off by {position_error:.2e} m, "
            f"orientation off by {orientation_error:.2e}). The grasp was read "
            "wrongly; do not benchmark this bar until it is understood."
        )
    return {
        "position_error_m": position_error,
        "orientation_error": orientation_error,
    }


def build_cells(problem_dir: str, root: str, *, bar_ids: Optional[Sequence[str]] = None,
                verbose: bool = True) -> Dict[str, object]:
    """Trim one export into per-bar benchmark scenes and cache them.

    Args:
        problem_dir (str): the exported problem folder.
        root (str): the benchmark data root to write into.
        bar_ids (Optional[Sequence[str]]): only these bars; all of them by
            default.
        verbose (bool): print progress as each bar is done.

    Returns:
        Dict[str, object]: a summary with the ``source_key``, the list of
        ``bars`` built and the grasp-check errors per bar.
    """
    source_key = source_key_for(problem_dir)
    wanted = list(bar_ids) if bar_ids else find_bar_ids(problem_dir)

    if verbose:
        print(f"[build-cells] source {problem_dir}")
        print(f"[build-cells] cache key {source_key}")
        print(f"[build-cells] {len(wanted)} bar(s): {', '.join(wanted)}")

    load_t0 = time.time()
    rcell = json_load(os.path.join(problem_dir, "RobotCell.json"))
    load_s = time.time() - load_t0
    if verbose:
        print(f"[build-cells] robot cell loaded in {load_s:.1f} s "
              f"({len(rcell.rigid_body_models)} rigid bodies, "
              f"{len(rcell.tool_models)} tools)")

    checks: Dict[str, Dict[str, float]] = {}
    built: List[str] = []
    for bar_id in wanted:
        bar_t0 = time.time()
        actions_dir = os.path.join(problem_dir, "BarActions")
        action = json_load(os.path.join(actions_dir, f"{bar_id}__J.json"))
        release_action = json_load(os.path.join(actions_dir, f"{bar_id}__R.json"))

        problem = extract_bar_problem(action, release_action, rcell)
        problem.source_action = f"{bar_id}__J.json"
        cell = build_trimmed_cell(rcell, problem)
        template_state = build_template_state(action, problem, cell)

        # Open the scene once purely to check the grasp; workers open their own.
        scene = open_scene(cell, template_state, problem)
        try:
            checks[bar_id] = verify_grasp(scene, problem)
        finally:
            scene.close()

        save_trimmed_problem(root, source_key, problem, cell, template_state)
        built.append(bar_id)
        if verbose:
            print(f"[build-cells] {bar_id}: length {problem.bar_length_m:.3f} m, "
                  f"radius {problem.bar_radius_m:.3f} m, "
                  f"joints {problem.joint_rb_names}, "
                  f"grasp check ok, {time.time() - bar_t0:.1f} s")

    return {"source_key": source_key, "bars": built, "grasp_checks": checks,
            "cell_load_s": load_s}
