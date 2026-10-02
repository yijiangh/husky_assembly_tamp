"""Compare several benchmark batches -- planner variants -- on the same problems.

Each variant is its own batch (``run --name pilot1_v2 --variant v2_nocont``),
so a comparison starts by finding the runs every batch has in common (same bar,
goal, start and seed) and only looks at those. It then answers, per variant:

    how many runs and problems it solved (with confidence intervals),
    whether it solved the SAME problems as the baseline or different ones
        (a paired 2x2 table per problem, and an exact McNemar test on the
        problems where the two disagree),
    how fast it was on the runs both it and the baseline solved (the median
        ratio of fine-pass times with a bootstrap interval, so a slower
        machine day cannot masquerade as a slower planner),
    how often its search found a path that the strict repair then rejected,
        and why.

The result is one JSON file under ``<root>/compare/`` that the dashboard's
comparison page reads, plus an index of the comparisons that exist.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from husky_assembly_tamp.benchmark.library import load_bar_library
from husky_assembly_tamp.benchmark.summarize import (
    load_runs,
    quantiles,
    restart_curve,
    summarize_group,
    wilson_interval,
)


ProblemKey = Tuple[str, int, int]
RunKey = Tuple[str, int, int, int]


def problem_key(run: Dict[str, Any]) -> ProblemKey:
    """Return what identifies a problem: the bar, the goal and the start."""
    return (run["bar_id"], int(run["goal_index"]), int(run["start_index"]))


def run_key(run: Dict[str, Any]) -> RunKey:
    """Return what identifies a run: the problem plus the seed."""
    return (run["bar_id"], int(run["goal_index"]), int(run["start_index"]), int(run["seed"]))


def problem_label(key: ProblemKey) -> str:
    """Return the ``B3/g007/s0042`` form of a problem key, as run ids use."""
    bar_id, goal_index, start_index = key
    return f"{bar_id}/g{goal_index:03d}/s{start_index:04d}"


def load_batch(root: str, run_name: str) -> Dict[str, Any]:
    """Read one batch's config and every run record.

    Args:
        root (str): the benchmark data root.
        run_name (str): the batch's name (its folder under ``runs/``).

    Returns:
        Dict[str, Any]: ``name``, ``config``, ``runs`` (the records) and
        ``by_run`` (the same records keyed by :func:`run_key`).

    Raises:
        FileNotFoundError: when the batch has no results.
    """
    run_dir = os.path.join(root, "runs", run_name)
    runs = load_runs(run_dir)
    config: Dict[str, Any] = {}
    config_path = os.path.join(run_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    return {
        "name": run_name,
        "config": config,
        "runs": runs,
        "by_run": {run_key(run): run for run in runs},
    }


def mcnemar_exact_p(baseline_only: int, variant_only: int) -> float:
    """Return the two-sided exact McNemar p-value for paired yes/no outcomes.

    Only the problems where the two planners disagree carry information. Under
    the hypothesis that neither is better, each disagreement is equally likely
    to go either way, so the count going one way is binomial(n, 1/2).

    Args:
        baseline_only (int): problems only the baseline solved.
        variant_only (int): problems only the variant solved.

    Returns:
        float: the p-value; 1.0 when there is no disagreement at all.
    """
    n = baseline_only + variant_only
    if n == 0:
        return 1.0
    k = min(baseline_only, variant_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))


def paired_problem_table(baseline_solved: Dict[ProblemKey, bool],
                         variant_solved: Dict[ProblemKey, bool]) -> Dict[str, Any]:
    """Cross-tabulate which problems each of two planners solved.

    Args:
        baseline_solved (Dict[ProblemKey, bool]): solved-or-not per problem.
        variant_solved (Dict[ProblemKey, bool]): the same for the variant.

    Returns:
        Dict[str, Any]: the four cell counts, the problems in the two
        disagreeing cells, and the exact McNemar p-value.
    """
    both, baseline_only, variant_only, neither = [], [], [], []
    for key in sorted(baseline_solved):
        base = bool(baseline_solved[key])
        other = bool(variant_solved.get(key, False))
        if base and other:
            both.append(key)
        elif base:
            baseline_only.append(key)
        elif other:
            variant_only.append(key)
        else:
            neither.append(key)
    return {
        "both": len(both),
        "baseline_only": len(baseline_only),
        "variant_only": len(variant_only),
        "neither": len(neither),
        "baseline_only_problems": [problem_label(key) for key in baseline_only],
        "variant_only_problems": [problem_label(key) for key in variant_only],
        "mcnemar_p": mcnemar_exact_p(len(baseline_only), len(variant_only)),
    }


def bootstrap_median_ratio(pairs: Sequence[Tuple[float, float]],
                           n_boot: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """Return the median of variant/baseline time ratios with a 95% interval.

    Args:
        pairs (Sequence[Tuple[float, float]]): ``(baseline_s, variant_s)`` for
            runs both planners solved.
        n_boot (int): bootstrap resamples.
        seed (int): random seed, fixed so the numbers are reproducible.

    Returns:
        Dict[str, Any]: ``n``, ``median_ratio`` and ``ci95``; the ratio is
        ``None`` when there are no pairs.
    """
    if not pairs:
        return {"n": 0, "median_ratio": None, "ci95": [None, None]}
    base = np.asarray([b for b, _ in pairs], dtype=float)
    other = np.asarray([v for _, v in pairs], dtype=float)
    # Guard against a zero baseline time (a degenerate record), not a real case.
    ratios = other / np.maximum(base, 1e-9)
    rng = np.random.default_rng(seed)
    medians = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.integers(0, len(ratios), len(ratios))
        medians[i] = np.median(ratios[sample])
    return {
        "n": int(len(ratios)),
        "median_ratio": float(np.median(ratios)),
        "ci95": [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))],
    }


def fine_time_s(run: Dict[str, Any]) -> Optional[float]:
    """Return a solved run's fine-pass wall time (repair included), if any."""
    fine = (run.get("passes") or {}).get("fine") or {}
    if not run.get("success") or "wall_s" not in fine:
        return None
    return float(fine["wall_s"])


def start_labels(root: str, library_name: Optional[str],
                 problems: Sequence[ProblemKey]) -> Dict[ProblemKey, Dict[str, Any]]:
    """Look up each problem's start difficulty and carry anchor in the library.

    Args:
        root (str): the benchmark data root.
        library_name (Optional[str]): the library the batches were planned
            from; nothing is looked up when unknown.
        problems (Sequence[ProblemKey]): the problems to label.

    Returns:
        Dict[ProblemKey, Dict[str, Any]]: ``difficulty`` and ``anchor`` per
        problem; empty dicts where the library has no entry.
    """
    labels: Dict[ProblemKey, Dict[str, Any]] = {key: {} for key in problems}
    if not library_name:
        return labels
    cache: Dict[str, Dict[Tuple[int, int], Dict[str, Any]]] = {}
    for key in problems:
        bar_id, goal_index, start_index = key
        if bar_id not in cache:
            try:
                entry = load_bar_library(root, library_name, bar_id)
            except (OSError, ValueError, KeyError):
                cache[bar_id] = {}
                continue
            lookup: Dict[Tuple[int, int], Dict[str, Any]] = {}
            for goal in entry.get("goals", []):
                for start in goal.get("starts", []):
                    lookup[(int(goal["goal_index"]), int(start["start_index"]))] = {
                        "difficulty": start.get("difficulty"),
                        "anchor": start.get("anchor"),
                        "rank": start.get("rank"),
                    }
            cache[bar_id] = lookup
        labels[key] = cache[bar_id].get((goal_index, start_index), {})
    return labels


def compare_runs(root: str, run_names: Sequence[str], out_name: str,
                 baseline: Optional[str] = None) -> Dict[str, Any]:
    """Build the comparison of several batches on their common runs.

    Args:
        root (str): the benchmark data root.
        run_names (Sequence[str]): the batches to compare; the first is the
            baseline unless ``baseline`` says otherwise.
        out_name (str): the comparison's name (its file under ``compare/``).
        baseline (Optional[str]): which batch the paired statistics are
            measured against.

    Returns:
        Dict[str, Any]: the comparison payload (also written by
        :func:`write_comparison`).

    Raises:
        ValueError: with fewer than two batches, an unknown baseline, or no
            runs in common.
    """
    if len(run_names) < 2:
        raise ValueError("a comparison needs at least two batches")
    baseline_name = baseline or run_names[0]
    if baseline_name not in run_names:
        raise ValueError(f"baseline {baseline_name!r} is not one of {list(run_names)}")

    batches = {name: load_batch(root, name) for name in run_names}
    common: set = set.intersection(*(set(batch["by_run"]) for batch in batches.values()))
    if not common:
        raise ValueError("the batches have no run (bar, goal, start, seed) in common")
    common_keys = sorted(common)
    problems = sorted({key[:3] for key in common_keys})
    seeds = sorted({key[3] for key in common_keys})
    n_seeds = len(seeds)

    def solved_by_problem(batch: Dict[str, Any]) -> Dict[ProblemKey, bool]:
        """Whether the batch solved each problem with at least one common seed."""
        solved: Dict[ProblemKey, bool] = {key: False for key in problems}
        for key in common_keys:
            if batch["by_run"][key].get("success"):
                solved[key[:3]] = True
        return solved

    def solved_seed_count(batch: Dict[str, Any]) -> Dict[ProblemKey, int]:
        """How many of the common seeds each problem was solved with."""
        counts: Dict[ProblemKey, int] = {key: 0 for key in problems}
        for key in common_keys:
            if batch["by_run"][key].get("success"):
                counts[key[:3]] += 1
        return counts

    base_batch = batches[baseline_name]
    base_solved = solved_by_problem(base_batch)

    variants: List[Dict[str, Any]] = []
    for name in run_names:
        batch = batches[name]
        runs = [batch["by_run"][key] for key in common_keys]
        totals = summarize_group(runs)
        totals["restart_curve"] = restart_curve(runs, n_seeds)
        solved = solved_by_problem(batch)
        n_solved_problems = sum(1 for value in solved.values() if value)

        # Times on the runs that BOTH this batch and the baseline solved.
        pairs: List[Tuple[float, float]] = []
        for key in common_keys:
            base_t = fine_time_s(base_batch["by_run"][key])
            this_t = fine_time_s(batch["by_run"][key])
            if base_t is not None and this_t is not None:
                pairs.append((base_t, this_t))
        both_solved_times = [v for _, v in pairs]

        entry: Dict[str, Any] = {
            "run_name": name,
            "variant": batch["config"].get("variant"),
            "variant_settings": batch["config"].get("variant_settings"),
            "is_baseline": name == baseline_name,
            "totals": totals,
            "n_problems": len(problems),
            "n_solved_problems": n_solved_problems,
            "problem_success_rate": n_solved_problems / len(problems),
            "problem_success_ci95": wilson_interval(n_solved_problems, len(problems)),
            "fine_wall_s_both_solved": quantiles(both_solved_times),
            "time_ratio_vs_baseline": bootstrap_median_ratio(pairs) if name != baseline_name else None,
            "paired_vs_baseline": (paired_problem_table(base_solved, solved)
                                   if name != baseline_name else None),
            "timeouts": totals.get("statuses", {}).get("timeout", 0),
        }
        variants.append(entry)

    labels = start_labels(root, base_batch["config"].get("library"), problems)
    seed_counts = {name: solved_seed_count(batches[name]) for name in run_names}
    problem_rows = []
    for key in problems:
        row = {
            "key": problem_label(key),
            "bar_id": key[0],
            "goal_index": key[1],
            "start_index": key[2],
            **labels.get(key, {}),
            "solved_seeds": {name: seed_counts[name][key] for name in run_names},
        }
        problem_rows.append(row)

    return {
        "schema_version": 1,
        "name": out_name,
        "created": time.time(),
        "baseline": baseline_name,
        "library": base_batch["config"].get("library"),
        "run_names": list(run_names),
        "seeds": seeds,
        "n_common_runs": len(common_keys),
        "n_problems": len(problems),
        "variants": variants,
        "problems": problem_rows,
    }


def write_comparison(root: str, payload: Dict[str, Any]) -> str:
    """Write a comparison under ``<root>/compare/`` and update the index.

    Args:
        root (str): the benchmark data root.
        payload (Dict[str, Any]): what :func:`compare_runs` returned.

    Returns:
        str: the path of the comparison file.
    """
    folder = os.path.join(root, "compare")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{payload['name']}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    index_path = os.path.join(folder, "index.json")
    index: Dict[str, Any] = {"schema_version": 1, "comparisons": []}
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
    entries = [row for row in index.get("comparisons", []) if row.get("name") != payload["name"]]
    entries.insert(0, {
        "name": payload["name"],
        "created": payload["created"],
        "baseline": payload["baseline"],
        "run_names": payload["run_names"],
        "n_common_runs": payload["n_common_runs"],
        "n_problems": payload["n_problems"],
        "path": f"compare/{payload['name']}.json",
    })
    index["comparisons"] = entries
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=1)
    return path
