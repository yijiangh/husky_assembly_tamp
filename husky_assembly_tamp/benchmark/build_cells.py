"""Stage 0: turn an exported process folder into per-bar benchmark scenes.

A process folder (``process.json``, ``actions/``, the robots as URDF files and
the meshes as OBJ files; see ``process_export.py``) describes the whole
assembly: every bar, every connector half, the ground, the robots. A roadmap
is built for one bar at a time, in a scene with only the robot, its tools and
what rides on them, and the worker processes must not read the whole process
each. So this stage reads it once, trims each bar's carry scene down (see
``process_export.trimmed_problem`` and ``scene.py``), and writes the small
results to the benchmark data root. Every later stage and every worker reads
only those.

It also checks each problem it writes: it poses the robot at the authored home
configuration and confirms the bar really does end up where the recorded grasp
says it should. A mismatch means the grasp was read wrongly, and the benchmark
would go on to measure nonsense, so it stops.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pybullet_planning as pp

from husky_assembly_tamp.benchmark.process_export import (
    is_process_folder,
    process_bar_ids,
    process_source_key,
    read_process,
    trimmed_problem,
)
from husky_assembly_tamp.benchmark.scene import BarProblem, open_scene, save_trimmed_problem


# How closely the bar's simulated pose must match the pose implied by the
# recorded grasp. These are far looser than the numbers actually seen (which
# are at float noise) but tight enough that a wrong grasp cannot slip past.
GRASP_CHECK_POSITION_TOL_M = 1e-6
GRASP_CHECK_ORIENTATION_TOL = 1e-6


def check_process_folder(problem_dir: str) -> None:
    """Refuse anything that is not a process folder, with a message that says what one is.

    Args:
        problem_dir (str): the folder given on the command line.

    Raises:
        FileNotFoundError: when the folder has no ``process.json``.
    """
    if not is_process_folder(problem_dir):
        raise FileNotFoundError(
            f"{problem_dir} is not a process folder (no process.json). Point --problem-dir at a folder "
            "written by Rhino's RSExportProcess: process.json, actions/, robots/, meshes/."
        )


def source_key_for(problem_dir: str) -> str:
    """Return a short, stable key identifying one exported process.

    The key is the folder name plus eight hex digits of a hash over the
    content hashes of ``process.json`` and every action file
    (``process_export.process_source_key``). Re-exporting an unchanged design
    keeps the key; a moved bar, a new grasp or a new robot file changes it, so
    the trimmed scenes cached under the old key are never silently reused.
    Copying the data folder to another machine keeps the key the same.

    Args:
        problem_dir (str): the process folder.

    Returns:
        str: the cache key, e.g. ``"261008_2ModulesDemo_purged_process-4be92c6a"``.

    Raises:
        FileNotFoundError: if the folder is not a process folder.
    """
    check_process_folder(problem_dir)
    return process_source_key(problem_dir)


def find_bar_ids(problem_dir: str) -> List[str]:
    """List the bars a process has both a jointing and a release action for.

    Args:
        problem_dir (str): the process folder.

    Returns:
        List[str]: bar ids in schedule order, e.g. ``["B20", "B21", ...]``.

    Raises:
        FileNotFoundError: if the folder is not a process folder.
    """
    check_process_folder(problem_dir)
    return process_bar_ids(read_process(problem_dir))


def verify_grasp(scene, problem: BarProblem) -> Dict[str, float]:
    """Check that the recorded grasp really describes where the bar hangs.

    Poses the robot at its home configuration, reads where PyBullet actually
    puts the bar, and compares that with the pose computed from the left
    flange and the recorded grasp. The two must agree: the whole benchmark
    rests on that grasp being right.

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

    position_error = float(np.linalg.norm(np.asarray(simulated[0]) - np.asarray(from_grasp[0])))
    # Quaternions q and -q are the same rotation, so compare the absolute dot.
    dot = abs(float(np.dot(np.asarray(simulated[1]), np.asarray(from_grasp[1]))))
    orientation_error = float(1.0 - min(1.0, dot))

    if position_error > GRASP_CHECK_POSITION_TOL_M or orientation_error > GRASP_CHECK_ORIENTATION_TOL:
        raise ValueError(
            f"{problem.bar_id}: the recorded grasp does not reproduce the bar's "
            f"simulated pose (position off by {position_error:.2e} m, "
            f"orientation off by {orientation_error:.2e}). The grasp was read "
            "wrongly; do not benchmark this bar until it is understood."
        )
    return {"position_error_m": position_error, "orientation_error": orientation_error}


def build_cells(problem_dir: str, root: str, *, bar_ids: Optional[Sequence[str]] = None,
                verbose: bool = True) -> Dict[str, object]:
    """Trim one process folder into per-bar benchmark scenes and cache them.

    Args:
        problem_dir (str): the process folder (``process.json`` and ``actions/``).
        root (str): the benchmark data root to write into.
        bar_ids (Optional[Sequence[str]]): only these bars; all of them by default.
        verbose (bool): print progress as each bar is done.

    Returns:
        Dict[str, object]: a summary with the ``source_key``, the list of
        ``bars`` built and the grasp-check errors per bar.
    """
    source_key = source_key_for(problem_dir)
    load_t0 = time.time()
    process = read_process(problem_dir)
    load_s = time.time() - load_t0
    wanted = list(bar_ids) if bar_ids else process_bar_ids(process)

    if verbose:
        print(f"[build-cells] source {problem_dir}")
        print(f"[build-cells] cache key {source_key}")
        print(f"[build-cells] process read in {load_s:.1f} s ({len(process.bodies)} bodies, "
              f"{len(process.actions)} actions, schema {process.writer.schema})")
        print(f"[build-cells] {len(wanted)} bar(s): {', '.join(wanted)}")

    checks: Dict[str, Dict[str, float]] = {}
    built: List[str] = []
    for bar_id in wanted:
        bar_t0 = time.time()
        problem, cell, template_state = trimmed_problem(process, bar_id)

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
                  f"{len(problem.joint_rb_names)} halves ride along, "
                  f"grasp check ok, {time.time() - bar_t0:.1f} s")

    return {"source_key": source_key, "bars": built, "grasp_checks": checks, "cell_load_s": load_s}
