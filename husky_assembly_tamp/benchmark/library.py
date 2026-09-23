"""The problem library: the benchmark's fixed set of questions.

A library is a list of transfer problems -- for each bar, a set of goal poses,
and for each goal, the carry poses the robot could start from. Building it is
slow (it is mostly collision checking) but it only has to happen once. After
that, every planner version is measured against the same problems, which is
what makes two benchmark runs comparable at all.

The library is stored as one JSON file per bar plus a small index, so a bar can
be rebuilt without touching the others and a partial library is still usable.
The per-candidate sweep detail is far bigger than the rest and is written
separately, gzipped, since only the dashboard ever reads it.
"""
from __future__ import annotations

import gzip
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence

from husky_assembly_tamp.benchmark.sampling import sample_goals
from husky_assembly_tamp.benchmark.scene import SceneCtx, pose_from_lists
from husky_assembly_tamp.benchmark.sweep import (
    DEFAULT_SWEEP_BOX,
    DEFAULT_SWEEP_STEP_M,
    run_production_derivation,
    sweep_goal_starts,
)


LIBRARY_SCHEMA_VERSION = 1


def library_dir(root: str, library_name: str) -> str:
    """Return the folder one library lives in.

    Args:
        root (str): the benchmark data root.
        library_name (str): the library's name, e.g. ``"cdfm_v1"``.

    Returns:
        str: the folder path (not created).
    """
    return os.path.join(root, "library", library_name)


def bar_path(root: str, library_name: str, bar_id: str) -> str:
    """Return the file holding one bar's goals and starts."""
    return os.path.join(library_dir(root, library_name), "bars", f"{bar_id}.json")


def sweep_path(root: str, library_name: str, bar_id: str) -> str:
    """Return the file holding one bar's per-candidate sweep detail."""
    return os.path.join(library_dir(root, library_name), "sweeps", f"{bar_id}.json.gz")


def index_path(root: str, library_name: str) -> str:
    """Return the library's index file."""
    return os.path.join(library_dir(root, library_name), "index.json")


def build_bar_library(scene: SceneCtx, *, n_goals: int, library_seed: int,
                      sweep_step_m: float = DEFAULT_SWEEP_STEP_M,
                      sweep_box=DEFAULT_SWEEP_BOX,
                      run_production_check: bool = True,
                      verbose: bool = True) -> Dict[str, Any]:
    """Build the goals and starts for one bar.

    Args:
        scene (SceneCtx): the live scene for this bar.
        n_goals (int): how many reachable goals to sample.
        library_seed (int): the library's master seed.
        sweep_step_m (float): spacing of the start-offset grid, metres.
        sweep_box: the start-offset box.
        run_production_check (bool): also run the start derivation production
            actually uses, for comparison.
        verbose (bool): print progress.

    Returns:
        Dict[str, Any]: the bar's library entry, and under ``sweeps`` the
        per-candidate detail (stored separately by :func:`save_bar_library`).
    """
    problem = scene.problem
    t0 = time.time()
    if verbose:
        print(f"[library] {problem.bar_id}: sampling {n_goals} goals "
              f"(bar {problem.bar_length_m:.2f} m)", flush=True)

    sampled = sample_goals(scene, n_goals=n_goals, library_seed=library_seed,
                           verbose=verbose)

    goals: List[Dict[str, Any]] = []
    sweeps: Dict[str, Any] = {}
    for goal in sampled["goals"]:
        world_from_bar_goal = pose_from_lists(goal["world_from_bar"])
        goal_conf = goal["primary_conf"]

        survey = sweep_goal_starts(
            scene, goal_conf, world_from_bar_goal,
            sweep_step_m=sweep_step_m, sweep_box=sweep_box,
            check_direct_paths=True, verbose=verbose,
        )
        # Every feasible start is stored. Which of them a batch actually
        # plans from is decided later, at the planning stage, so that choice
        # can change without re-running the sweep that found them.
        entry = dict(goal)
        entry["starts"] = survey["starts"]
        entry["n_starts_found"] = survey["n_starts"]
        entry["difficulty_counts"] = survey.get("difficulty_counts", {})
        entry["sweep_counts"] = survey["counts"]
        entry["sweep_n_candidates"] = survey["n_candidates"]
        entry["sweep_wall_s"] = survey["wall_s"]
        entry["sweep_step_m"] = survey["sweep_step_m"]
        entry["n_direct_path_free"] = survey.get("n_direct_path_free", 0)

        if run_production_check:
            entry["production_derivation"] = run_production_derivation(
                scene, goal_conf, world_from_bar_goal,
            )

        goals.append(entry)
        sweeps[str(goal["goal_index"])] = survey["candidates"]
        if verbose:
            print(f"  goal {goal['goal_index']:3d}: {survey['n_starts']} starts, "
                  f"difficulty {survey.get('difficulty_counts', {})}", flush=True)

    return {
        "bar_id": problem.bar_id,
        "problem": problem.to_dict(),
        "goals": goals,
        "rejections": sampled["rejections"],
        "n_sampled": sampled["n_sampled"],
        "sweep_step_m": sweep_step_m,
        "wall_s": time.time() - t0,
        "sweeps": sweeps,
    }


def select_starts(starts: Sequence[Dict[str, Any]],
                  max_starts: Optional[int]) -> List[Dict[str, Any]]:
    """Choose which of a goal's starts a batch should plan from.

    A survey turns up hundreds of workable carry poses per goal. Planning from
    all of them, ten seeds each, is far more compute than the answer is worth,
    so a batch takes a sample -- and which sample matters a great deal.

    Taken in survey order the sample would be hundreds of variations on the
    same pose, all equally hard. Taken at random it would be dominated by the
    commonest kind, which is the hardest kind. So the sample is spread across
    both the three carry anchors and the three difficulty classes, and the
    start production itself would have picked always goes in first. The result
    is a batch that says something about easy, medium and hard problems
    instead of just reporting that most random problems are hard.

    Args:
        starts (Sequence[Dict[str, Any]]): every feasible start from the survey.
        max_starts (Optional[int]): how many to plan from; None means all.

    Returns:
        List[Dict[str, Any]]: the chosen starts, in survey order.
    """
    if max_starts is None or len(starts) <= max_starts:
        return list(starts)

    buckets: Dict[Any, List[Dict[str, Any]]] = {}
    for start in starts:
        key = (start.get("difficulty", "broken"), start["anchor"])
        buckets.setdefault(key, []).append(start)
    for bucket in buckets.values():
        bucket.sort(key=lambda start: start["rank"])

    # Production's own pick first: every batch should at least measure the
    # problem the planner would really have been given.
    chosen: List[Dict[str, Any]] = []
    chosen_ids = set()
    production_pick = min(starts, key=lambda start: start["rank"])
    chosen.append(production_pick)
    chosen_ids.add(production_pick["start_index"])

    # Then round-robin over the buckets, easiest class first so a small budget
    # still covers problems that are actually solvable.
    order = sorted(buckets, key=lambda key: (
        {"free": 0, "blocked": 1, "broken": 2}.get(key[0], 3), key[1]))
    position = 0
    while len(chosen) < max_starts:
        added_any = False
        for key in order:
            bucket = buckets[key]
            if position >= len(bucket) or len(chosen) >= max_starts:
                continue
            candidate = bucket[position]
            if candidate["start_index"] not in chosen_ids:
                chosen.append(candidate)
                chosen_ids.add(candidate["start_index"])
            added_any = True
        if not added_any:
            break
        position += 1
    return sorted(chosen, key=lambda start: start["rank"])


def save_bar_library(root: str, library_name: str, entry: Dict[str, Any]) -> None:
    """Write one bar's library entry and its sweep detail.

    Args:
        root (str): the benchmark data root.
        library_name (str): the library's name.
        entry (Dict[str, Any]): what :func:`build_bar_library` returned.
    """
    entry = dict(entry)
    sweeps = entry.pop("sweeps", {})

    path = bar_path(root, library_name, entry["bar_id"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(entry, handle)

    detail = sweep_path(root, library_name, entry["bar_id"])
    os.makedirs(os.path.dirname(detail), exist_ok=True)
    with gzip.open(detail, "wt", encoding="utf-8") as handle:
        json.dump(sweeps, handle)


def load_bar_library(root: str, library_name: str, bar_id: str) -> Dict[str, Any]:
    """Read one bar's library entry.

    Args:
        root (str): the benchmark data root.
        library_name (str): the library's name.
        bar_id (str): which bar.

    Returns:
        Dict[str, Any]: the entry.

    Raises:
        FileNotFoundError: if this bar is not in the library yet.
    """
    path = bar_path(root, library_name, bar_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{bar_id} is not in library {library_name!r} at {path}. Run the "
            "'build-library' stage for it first."
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_sweep_detail(root: str, library_name: str, bar_id: str) -> Dict[str, Any]:
    """Read one bar's per-candidate sweep detail.

    Args:
        root (str): the benchmark data root.
        library_name (str): the library's name.
        bar_id (str): which bar.

    Returns:
        Dict[str, Any]: candidate lists keyed by goal index (as a string).
    """
    path = sweep_path(root, library_name, bar_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No sweep detail for {bar_id} at {path}. Run 'build-library' for it."
        )
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_index(root: str, library_name: str, *, source_key: str,
                library_seed: int, bar_ids: Sequence[str],
                settings: Dict[str, Any]) -> str:
    """Write the library index, summarising what it contains.

    Args:
        root (str): the benchmark data root.
        library_name (str): the library's name.
        source_key (str): which trimmed-scene cache the library was built from.
        library_seed (int): the master seed.
        bar_ids (Sequence[str]): the bars in the library.
        settings (Dict[str, Any]): the sampling and sweep settings used.

    Returns:
        str: the path written.
    """
    summary = []
    for bar_id in bar_ids:
        entry = load_bar_library(root, library_name, bar_id)
        summary.append({
            "bar_id": bar_id,
            "bar_length_m": entry["problem"]["bar_length_m"],
            "n_goals": len(entry["goals"]),
            "n_starts_found": sum(goal["n_starts_found"] for goal in entry["goals"]),
            "n_direct_path_free": sum(goal["n_direct_path_free"] for goal in entry["goals"]),
            "wall_s": entry["wall_s"],
        })

    index = {
        "schema_version": LIBRARY_SCHEMA_VERSION,
        "library_name": library_name,
        "source_key": source_key,
        "library_seed": library_seed,
        "created": time.time(),
        "settings": settings,
        "bars": summary,
    }
    path = index_path(root, library_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)
    return path


def load_index(root: str, library_name: str) -> Dict[str, Any]:
    """Read a library's index.

    Args:
        root (str): the benchmark data root.
        library_name (str): the library's name.

    Returns:
        Dict[str, Any]: the index.

    Raises:
        FileNotFoundError: if the library does not exist.
    """
    path = index_path(root, library_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No library {library_name!r} at {path}. Run 'build-library' first."
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
