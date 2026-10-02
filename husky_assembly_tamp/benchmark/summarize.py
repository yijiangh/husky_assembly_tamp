"""Turn a batch of individual runs into the tables the dashboard shows.

Every run is its own JSON file, which is right for a batch that gets
interrupted but useless for answering "is this planner better than that one".
This stage reads them all and produces one summary: success rates with
confidence intervals, runtime quantiles, where the time went, and why failures
failed.

Two statistics deserve a note.

Success rates get a Wilson interval rather than the textbook one. With ten
seeds per problem, proportions near zero or one are common, and the textbook
interval famously gives nonsense there (zero width at a rate of 1.0, and bounds
below zero near 0.0).

Runtimes get quantiles rather than a mean and standard deviation, because a
planner's runtime distribution has a long tail -- most runs are quick and a few
take the whole budget -- and a mean of such a distribution says little.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


def wilson_interval(n_success: int, n_total: int, z: float = 1.96) -> List[float]:
    """Return a confidence interval for a success rate.

    Args:
        n_success (int): how many runs succeeded.
        n_total (int): how many runs there were.
        z (float): the standard-normal quantile; 1.96 gives 95%.

    Returns:
        List[float]: the ``[low, high]`` bounds. ``[0, 1]`` when there are no
        runs at all, since nothing is known then.
    """
    if n_total == 0:
        return [0.0, 1.0]
    proportion = n_success / n_total
    denominator = 1.0 + z * z / n_total
    centre = (proportion + z * z / (2 * n_total)) / denominator
    spread = z * math.sqrt(
        proportion * (1 - proportion) / n_total + z * z / (4 * n_total * n_total)
    ) / denominator
    return [max(0.0, centre - spread), min(1.0, centre + spread)]


def quantiles(values: Sequence[float]) -> Dict[str, float]:
    """Return a few quantiles of a list of numbers.

    Args:
        values (Sequence[float]): the numbers.

    Returns:
        Dict[str, float]: ``p10``, ``p50``, ``p90``, ``max`` and ``mean``; all
        zero when there is nothing to summarise.
    """
    if not values:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    array = np.asarray(values, dtype=float)
    return {
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def load_runs(run_dir: str) -> List[Dict[str, Any]]:
    """Read every run result in a batch.

    Args:
        run_dir (str): the batch's folder.

    Returns:
        List[Dict[str, Any]]: the run records.

    Raises:
        FileNotFoundError: if the batch has no results yet.
    """
    runs_dir = os.path.join(run_dir, "runs")
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(
            f"No results under {runs_dir}. Run the 'run' stage for this batch first."
        )
    records: List[Dict[str, Any]] = []
    for bar_folder in sorted(os.listdir(runs_dir)):
        folder = os.path.join(runs_dir, bar_folder)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(folder, name), "r", encoding="utf-8") as handle:
                records.append(json.load(handle))
    return records


def restart_curve(runs: Iterable[Dict[str, Any]], n_seeds: int) -> List[float]:
    """Return the chance of succeeding within the first k attempts.

    A planner that succeeds one run in three is not a planner that fails: run
    it three times and it usually works. This curve is how that shows up, and
    it is often the more honest headline number.

    Args:
        runs (Iterable[Dict[str, Any]]): the run records.
        n_seeds (int): how many seeds each problem was run with.

    Returns:
        List[float]: entry k is the fraction of problems solved by at least one
        of the first k+1 seeds.
    """
    by_problem: Dict[Any, Dict[int, bool]] = {}
    for run in runs:
        key = (run["bar_id"], run["goal_index"], run["start_index"])
        by_problem.setdefault(key, {})[run["seed"]] = bool(run.get("success"))
    if not by_problem:
        return [0.0] * n_seeds

    curve = []
    for k in range(n_seeds):
        solved = sum(
            1 for seeds in by_problem.values()
            if any(seeds.get(seed, False) for seed in range(k + 1))
        )
        curve.append(solved / len(by_problem))
    return curve


def summarize_group(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise one group of runs.

    Args:
        runs (Sequence[Dict[str, Any]]): the runs in the group.

    Returns:
        Dict[str, Any]: counts, success rate with interval, runtime quantiles
        for each pass, how the passes fared, why failures failed, and the mean
        time breakdown.
    """
    n_runs = len(runs)
    n_success = sum(1 for run in runs if run.get("success"))

    coarse_ok = [run for run in runs
                 if (run.get("passes", {}).get("coarse") or {}).get("success")]
    fine_times = [run["passes"]["fine"]["wall_s"] for run in runs
                  if run.get("success") and "fine" in run.get("passes", {})]
    coarse_times = [run["passes"]["coarse"]["wall_s"] for run in coarse_ok]

    failed_pass = Counter(run.get("failed_pass") for run in runs
                          if run.get("failed_pass"))
    statuses = Counter(run.get("status", "ok") for run in runs)

    # * "Search success" is the search alone, before the repair that re-solves
    # * the path on the calibrated model. Batches made before the repair
    # * existed have no such field; there the search WAS the whole run.
    n_search_success = sum(
        1 for run in runs if run.get("search_success", run.get("success"))
    )

    stop_reasons: Counter = Counter()
    planners: Counter = Counter()
    breakdown_totals: Dict[str, float] = {}
    n_breakdowns = 0
    # Why the repair rejected a searched path, and how long repairs took.
    postprocess_reasons: Counter = Counter()
    repair_times: List[float] = []
    ik_solve_counts: List[int] = []
    first_step_failures = 0
    growth_walks_recorded = 0
    for run in runs:
        for record in (run.get("passes") or {}).values():
            stop_reasons.update(record.get("extend_stop_reasons") or {})
            if record.get("planner"):
                planners[record["planner"]] += 1
            breakdown = (record.get("profile") or {}).get("breakdown") or {}
            if breakdown:
                n_breakdowns += 1
                for key, value in breakdown.items():
                    breakdown_totals[key] = breakdown_totals.get(key, 0.0) + float(value)
            postprocess = record.get("postprocess") or {}
            if postprocess:
                repair_times.append(float(postprocess.get("wall_s", 0.0)))
                if not postprocess.get("success"):
                    postprocess_reasons[postprocess.get("reason") or "unknown"] += 1
            if record.get("n_ik_solves") is not None:
                ik_solve_counts.append(int(record["n_ik_solves"]))
            if record.get("n_first_step_failures") is not None:
                first_step_failures += int(record["n_first_step_failures"])
                growth_walks_recorded += int(record.get("n_growth_walks") or 0)

    mean_breakdown = ({key: value / n_breakdowns
                       for key, value in breakdown_totals.items()}
                      if n_breakdowns else {})

    # * How often a growth walk added no node at all (it failed on its first
    # * waypoint). Exact when the passes counted it; otherwise a lower bound
    # * from the tree sizes: every node in both trees came from a walk that
    # * made progress, and every other growth walk made none.
    growth_walk_keys = ("reached", "continuity", "ik_failure", "collision")
    growth_walks = sum(stop_reasons.get(key, 0) for key in growth_walk_keys)
    tree_nodes = sum(
        int(record.get("tree_a_nodes") or 0) + int(record.get("tree_b_nodes") or 0)
        for run in runs for record in (run.get("passes") or {}).values()
    )
    n_passes = sum(len(run.get("passes") or {}) for run in runs)
    if growth_walks_recorded:
        first_step_failure_rate = first_step_failures / growth_walks_recorded
        first_step_rate_is_bound = False
    elif growth_walks:
        # Two root nodes per pass were not grown; the rest were.
        first_step_failure_rate = max(0.0, 1.0 - (tree_nodes - 2 * n_passes) / growth_walks)
        first_step_rate_is_bound = True
    else:
        first_step_failure_rate = 0.0
        first_step_rate_is_bound = True

    mean_ik_seconds = mean_breakdown.get("ik", 0.0)
    mean_ik_solves = (sum(ik_solve_counts) / len(ik_solve_counts)) if ik_solve_counts else 0.0

    return {
        "n_runs": n_runs,
        "n_success": n_success,
        "success_rate": n_success / n_runs if n_runs else 0.0,
        "success_rate_ci95": wilson_interval(n_success, n_runs),
        "coarse_success_rate": len(coarse_ok) / n_runs if n_runs else 0.0,
        "fine_wall_s": quantiles(fine_times),
        "coarse_wall_s": quantiles(coarse_times),
        "median_wall_s": quantiles([run.get("wall_s", 0.0) for run in runs])["p50"],
        "failed_pass": dict(failed_pass),
        "statuses": dict(statuses),
        "extend_stop_reasons": dict(stop_reasons),
        "planners": dict(planners),
        "mean_breakdown_s": mean_breakdown,
        # --- planner-variant experiments ---
        "n_search_success": n_search_success,
        "search_success_rate": n_search_success / n_runs if n_runs else 0.0,
        "postprocess_reasons": dict(postprocess_reasons),
        "repair_wall_s": quantiles(repair_times),
        "mean_ik_solves": mean_ik_solves,
        # Milliseconds per IK solve, from the sampled IK seconds and the solve
        # count; zero when either was not recorded.
        "ik_ms_per_solve": (1000.0 * mean_ik_seconds / mean_ik_solves) if mean_ik_solves else 0.0,
        "first_step_failure_rate": first_step_failure_rate,
        "first_step_failure_rate_is_lower_bound": first_step_rate_is_bound,
    }


def bar_lengths_from_library(run_dir: str, library_name: Optional[str]) -> Dict[str, float]:
    """Look up each bar's length from the library a batch was planned from.

    Bar length is the main thing results are grouped by, and a run record does
    not carry it -- it belongs to the problem, not the run.

    Args:
        run_dir (str): the batch's folder, two levels below the data root.
        library_name (Optional[str]): which library, from the batch's config.

    Returns:
        Dict[str, float]: length in metres per bar id; empty when the library
        cannot be found, since a missing length should not stop a summary.
    """
    if not library_name:
        return {}
    root = os.path.dirname(os.path.dirname(run_dir))
    index = os.path.join(root, "library", library_name, "index.json")
    if not os.path.isfile(index):
        return {}
    with open(index, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {row["bar_id"]: row.get("bar_length_m", 0.0)
            for row in payload.get("bars", [])}


def summarize_run(run_dir: str) -> Dict[str, Any]:
    """Summarise a whole batch, overall and per bar, goal and start.

    Args:
        run_dir (str): the batch's folder.

    Returns:
        Dict[str, Any]: the summary.
    """
    runs = load_runs(run_dir)
    config = {}
    config_path = os.path.join(run_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)

    library_settings = config.get("library_settings") or {}
    bar_lengths = bar_lengths_from_library(run_dir, config.get("library"))

    by_bar: Dict[str, List[Dict[str, Any]]] = {}
    by_goal: Dict[Any, List[Dict[str, Any]]] = {}
    by_start: Dict[Any, List[Dict[str, Any]]] = {}
    for run in runs:
        by_bar.setdefault(run["bar_id"], []).append(run)
        by_goal.setdefault((run["bar_id"], run["goal_index"]), []).append(run)
        by_start.setdefault(
            (run["bar_id"], run["goal_index"], run["start_index"]), []).append(run)

    per_bar = []
    for bar_id in sorted(by_bar):
        summary = summarize_group(by_bar[bar_id])
        summary["bar_id"] = bar_id
        summary["bar_length_m"] = bar_lengths.get(bar_id, 0.0)
        per_bar.append(summary)

    per_goal = []
    for (bar_id, goal_index) in sorted(by_goal):
        summary = summarize_group(by_goal[(bar_id, goal_index)])
        summary["bar_id"] = bar_id
        summary["goal_index"] = goal_index
        per_goal.append(summary)

    per_start = []
    for key in sorted(by_start):
        bar_id, goal_index, start_index = key
        summary = summarize_group(by_start[key])
        summary["bar_id"] = bar_id
        summary["goal_index"] = goal_index
        summary["start_index"] = start_index
        per_start.append(summary)

    n_seeds = len(config.get("seeds") or []) or 1
    totals = summarize_group(runs)
    totals["restart_curve"] = restart_curve(runs, n_seeds)

    return {
        "run_name": config.get("name", os.path.basename(run_dir)),
        # Which planner variant the batch ran (None for batches made before
        # variants existed, which were all the baseline planner).
        "variant": config.get("variant"),
        "config": config,
        "library_settings": library_settings,
        "totals": totals,
        "per_bar": per_bar,
        "per_goal": per_goal,
        "per_start": per_start,
    }
