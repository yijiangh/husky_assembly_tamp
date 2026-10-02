"""Explain a batch's continuity stops: why did the arm solution jump?

Three quarters of the search's rejected steps are "continuity" stops: the IK
solution at the next 2 mm waypoint sat more than 10 degrees away from the
previous one on some joint. That single number hides three very different
things, and the cure is different for each:

    2pi representative   the solver handed back the same posture one full
                         turn away on a joint (a 2pi twin), which the
                         re-branch step should have folded back -- a bug in
                         the adapter, not a planning problem
    solver noise         two branches sit almost equally close to the seed and
                         the solver picked the other one; the arm barely
                         moved, the joint numbers did -- a solver question
    real flip            the nearest branch genuinely is far away, typically
                         next to a wrist or elbow singularity -- a threshold
                         or sampling question

This module reads the ``*.attempts.json.gz`` files a batch saves next to its
search trees and sorts every continuity stop into one of those bins. Some of
the evidence (the joint deltas) is in every batch; the finer evidence (how many
branches the solver had, how close the runner-up was) is only there when the
batch ran with the ``e0_diag`` variant, which asks the planner to record it.
"""
from __future__ import annotations

import glob
import gzip
import json
import math
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# * A joint delta this close to a whole number of turns is a 2pi twin.
TWO_PI_TOLERANCE_RAD = 0.05
# * Two branches this close to each other (raw L-infinity) are "the same
# * distance" as far as the seed is concerned.
BRANCH_TIE_TOLERANCE_RAD = 0.05
# * Histogram edges for the largest joint jump, in degrees.
DELTA_BINS_DEG = (10, 15, 20, 30, 45, 60, 90, 135, 180, 270, 360, 720)


def attempts_files(run_dir: str) -> List[str]:
    """Return every attempts file a batch saved, sorted.

    Args:
        run_dir (str): the batch's folder (``runs/<name>``).

    Returns:
        List[str]: paths of ``trees/*.attempts.json.gz``.
    """
    return sorted(glob.glob(os.path.join(run_dir, "trees", "*.attempts.json.gz")))


def pass_name_of(path: str) -> str:
    """Return the pass ("coarse"/"fine") an attempts file belongs to.

    File names are ``<bar>_<stem>_<pass>.attempts.json.gz``.

    Args:
        path (str): the file path.

    Returns:
        str: the pass name.
    """
    stem = os.path.basename(path)[: -len(".attempts.json.gz")]
    return stem.rsplit("_", 1)[-1]


def read_attempts(path: str) -> List[Dict[str, Any]]:
    """Read one gzipped attempts file.

    Args:
        path (str): the file path.

    Returns:
        List[Dict[str, Any]]: the attempts.
    """
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def distance_to_whole_turns(delta_rad: float) -> float:
    """How far a joint jump is from the nearest non-zero whole number of turns.

    Args:
        delta_rad (float): the absolute joint jump.

    Returns:
        float: ``min over k >= 1 of |delta - 2 pi k|``.
    """
    turns = max(1, int(round(abs(delta_rad) / (2.0 * math.pi))))
    return abs(abs(delta_rad) - 2.0 * math.pi * turns)


def classify_stop(detail: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Sort one continuity stop into a bin and gather the numbers behind it.

    Args:
        detail (Dict[str, Any]): the attempt's ``detail`` dict. It always has
            ``joint`` and ``delta_rad`` (12 values); an ``e0_diag`` batch adds
            ``n_branches``, ``nearest_linf``, ``second_nearest_linf``,
            ``sin_q3``, ``sin_q5`` and possibly ``nominal_delta_max``.

    Returns:
        Tuple[str, Dict[str, Any]]: the bin name and the measured values.
    """
    deltas = np.abs(np.asarray(detail.get("delta_rad") or [0.0], dtype=float))
    joint = int(detail.get("joint", int(np.argmax(deltas))))
    delta_max = float(deltas.max())
    arm = "left" if joint < 6 else "right"
    mod_turns = distance_to_whole_turns(delta_max)

    values: Dict[str, Any] = {
        "delta_max_rad": delta_max,
        "delta_max_deg": math.degrees(delta_max),
        "joint": joint,
        "arm": arm,
        "delta_mod_2pi": mod_turns,
    }
    for key in ("n_branches", "nearest_linf", "second_nearest_linf",
                "sin_q3", "sin_q5", "nominal_delta_max"):
        if key in detail:
            values[key] = detail[key]

    if mod_turns < TWO_PI_TOLERANCE_RAD:
        return "two_pi_representative", values
    second = detail.get("second_nearest_linf")
    nearest = detail.get("nearest_linf")
    if second is not None and nearest is not None and (second - nearest) < BRANCH_TIE_TOLERANCE_RAD:
        return "solver_noise", values
    nominal = detail.get("nominal_delta_max")
    threshold = float(detail.get("threshold_rad") or math.radians(10.0))
    if nominal is not None and nominal < threshold:
        return "solver_noise", values
    return "real_flip", values


def histogram_deg(values: Iterable[float]) -> Dict[str, int]:
    """Count joint jumps into the degree bins of ``DELTA_BINS_DEG``.

    Args:
        values (Iterable[float]): jumps in degrees.

    Returns:
        Dict[str, int]: counts keyed by a readable bin label.
    """
    edges = list(DELTA_BINS_DEG)
    counts: Counter = Counter()
    for value in values:
        label = f">{edges[-1]}"
        for low, high in zip(edges[:-1], edges[1:]):
            if low <= value < high:
                label = f"{low}-{high}"
                break
        if value < edges[0]:
            label = f"<{edges[0]}"
        counts[label] += 1
    return dict(counts)


def diagnose_run(run_dir: str) -> Dict[str, Any]:
    """Sort every continuity stop of a batch into its bin, per pass.

    Args:
        run_dir (str): the batch's folder (``runs/<name>``).

    Returns:
        Dict[str, Any]: per pass and overall: how many stops, the share in
        each bin, histograms of the jump size and the joint, how many stops
        happened on the very first waypoint of a walk, and the singularity
        indicators when recorded.

    Raises:
        FileNotFoundError: when the batch saved no attempts files.
    """
    files = attempts_files(run_dir)
    if not files:
        raise FileNotFoundError(
            f"No attempts files under {os.path.join(run_dir, 'trees')}; the batch must "
            "be run with trees stored (the default) for this diagnosis."
        )

    per_pass: Dict[str, Dict[str, Any]] = {}
    for path in files:
        pass_name = pass_name_of(path)
        bucket = per_pass.setdefault(pass_name, {
            "n_attempts": 0, "n_continuity": 0, "n_first_step": 0,
            "bins": Counter(), "delta_deg": [], "joints": Counter(),
            "arms": Counter(), "phases": Counter(), "sin_q3": [], "sin_q5": [],
            "has_branch_detail": False, "n_files": 0,
        })
        bucket["n_files"] += 1
        for attempt in read_attempts(path):
            bucket["n_attempts"] += 1
            if attempt.get("reason") != "continuity":
                continue
            bucket["n_continuity"] += 1
            bucket["phases"][attempt.get("phase", "?")] += 1
            if int(attempt.get("n_added") or 0) == 0:
                bucket["n_first_step"] += 1
            bin_name, values = classify_stop(attempt.get("detail") or {})
            bucket["bins"][bin_name] += 1
            bucket["delta_deg"].append(values["delta_max_deg"])
            bucket["joints"][values["joint"]] += 1
            bucket["arms"][values["arm"]] += 1
            if "nearest_linf" in values:
                bucket["has_branch_detail"] = True
            for key in ("sin_q3", "sin_q5"):
                if key in values:
                    bucket[key].append(abs(float(values[key])))

    def finish(bucket: Dict[str, Any]) -> Dict[str, Any]:
        """Turn the raw tallies of one pass into shares and histograms."""
        n = bucket["n_continuity"]
        return {
            "n_files": bucket["n_files"],
            "n_attempts": bucket["n_attempts"],
            "n_continuity": n,
            "share_of_attempts": (n / bucket["n_attempts"]) if bucket["n_attempts"] else 0.0,
            "first_step_share": (bucket["n_first_step"] / n) if n else 0.0,
            "bins": {key: value for key, value in bucket["bins"].items()},
            "bin_shares": {key: value / n for key, value in bucket["bins"].items()} if n else {},
            "delta_deg_histogram": histogram_deg(bucket["delta_deg"]),
            "delta_deg_median": float(np.median(bucket["delta_deg"])) if bucket["delta_deg"] else 0.0,
            "joints": {str(key): value for key, value in sorted(bucket["joints"].items())},
            "arms": dict(bucket["arms"]),
            "phases": dict(bucket["phases"]),
            "has_branch_detail": bucket["has_branch_detail"],
            # Near a wrist singularity |sin q5| is small; near the elbow one |sin q3| is.
            "sin_q3_median": float(np.median(bucket["sin_q3"])) if bucket["sin_q3"] else None,
            "sin_q5_median": float(np.median(bucket["sin_q5"])) if bucket["sin_q5"] else None,
        }

    result: Dict[str, Any] = {"run_dir": run_dir, "passes": {}}
    overall = Counter()
    for pass_name, bucket in per_pass.items():
        result["passes"][pass_name] = finish(bucket)
        overall.update(bucket["bins"])
    total = sum(overall.values())
    result["overall_bins"] = dict(overall)
    result["overall_bin_shares"] = {key: value / total for key, value in overall.items()} if total else {}
    result["n_continuity"] = total
    return result


def format_report(diag: Dict[str, Any]) -> str:
    """Render a diagnosis as a few readable lines.

    Args:
        diag (Dict[str, Any]): what :func:`diagnose_run` returned.

    Returns:
        str: the text.
    """
    lines = [f"continuity stops: {diag['n_continuity']:,}"]
    for key, share in sorted(diag["overall_bin_shares"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {key:24s} {100 * share:5.1f}%  ({diag['overall_bins'][key]:,})")
    for pass_name, info in diag["passes"].items():
        lines.append(f"[{pass_name}] {info['n_continuity']:,} of {info['n_attempts']:,} attempts; "
                     f"{100 * info['first_step_share']:.1f}% on the first waypoint; "
                     f"median jump {info['delta_deg_median']:.1f} deg; "
                     f"branch detail {'yes' if info['has_branch_detail'] else 'no'}")
        lines.append(f"  jump histogram (deg): {info['delta_deg_histogram']}")
        lines.append(f"  joints: {info['joints']}  arms: {info['arms']}  phases: {info['phases']}")
        if info["sin_q5_median"] is not None:
            lines.append(f"  median |sin q3| {info['sin_q3_median']:.3f}, |sin q5| {info['sin_q5_median']:.3f}")
    if not any(info["has_branch_detail"] for info in diag["passes"].values()):
        lines.append("note: no branch detail recorded -- 'solver_noise' cannot be told from "
                     "'real_flip' without an e0_diag batch; only the 2pi test is exact here.")
    return "\n".join(lines)


def write_diagnosis(run_dir: str, diag: Dict[str, Any]) -> str:
    """Write a diagnosis next to the batch's summary.

    Args:
        run_dir (str): the batch's folder.
        diag (Dict[str, Any]): what :func:`diagnose_run` returned.

    Returns:
        str: the path written (``continuity_diag.json``).
    """
    path = os.path.join(run_dir, "continuity_diag.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(diag, handle, indent=1)
    return path
