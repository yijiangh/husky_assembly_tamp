"""Write the files the web dashboard reads.

A batch leaves one JSON file per run plus a library that is far too big to load
into a browser. This stage reshapes both into something a page can fetch a
piece at a time: a small summary for the overview, one file per bar, one per
goal, one per start, and trajectories as compact binary.

Trajectories are the reason for the binary format. A planned motion at two
millimetre resolution is a few hundred waypoints of twelve joint angles, and
written as JSON that is a megabyte of decimal digits per run. As raw 32-bit
floats it is forty kilobytes, and the browser can hand the buffer straight to
the renderer.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import os
import struct
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from husky_assembly_tamp.benchmark.library import load_bar_library, load_sweep_detail
from husky_assembly_tamp.benchmark.summarize import load_runs, summarize_group, summarize_run


DASHBOARD_SCHEMA_VERSION = 1

# Magic bytes at the front of a trajectory file, so a truncated or wrong file
# is rejected rather than rendered as nonsense.
TRAJECTORY_MAGIC = b"HTRJ"


def write_trajectory(path: str, path_confs: Sequence[Sequence[float]]) -> int:
    """Write one planned motion as raw floats with a short header.

    Layout: the four magic bytes, then version, waypoint count and joint count
    as unsigned 32-bit little-endian integers, then the joint angles as 32-bit
    floats, waypoint by waypoint.

    Args:
        path (str): the file to write.
        path_confs (Sequence[Sequence[float]]): the configurations along the
            motion.

    Returns:
        int: how many bytes were written.
    """
    array = np.asarray(path_confs, dtype=np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(TRAJECTORY_MAGIC)
        handle.write(struct.pack("<III", DASHBOARD_SCHEMA_VERSION,
                                 array.shape[0], array.shape[1]))
        handle.write(array.tobytes())
    return os.path.getsize(path)


def write_json(path: str, payload: Any) -> None:
    """Write a JSON file, creating its folder.

    Args:
        path (str): the file to write.
        payload (Any): what to write.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def export_run(root: str, run_name: str, *, library_name: str,
               verbose: bool = True) -> Dict[str, Any]:
    """Reshape one batch into the dashboard's file layout.

    Args:
        root (str): the benchmark data root.
        run_name (str): which batch to export.
        library_name (str): the library it was planned from.
        verbose (bool): print progress.

    Returns:
        Dict[str, Any]: counts of what was written.
    """
    run_dir = os.path.join(root, "runs", run_name)
    out_dir = os.path.join(run_dir, "dashboard")

    summary = summarize_run(run_dir)
    runs = load_runs(run_dir)

    # Group the runs once; every file below is a slice of these.
    by_bar: Dict[str, List[Dict[str, Any]]] = {}
    by_goal: Dict[Any, List[Dict[str, Any]]] = {}
    by_start: Dict[Any, List[Dict[str, Any]]] = {}
    for run in runs:
        by_bar.setdefault(run["bar_id"], []).append(run)
        by_goal.setdefault((run["bar_id"], run["goal_index"]), []).append(run)
        by_start.setdefault(
            (run["bar_id"], run["goal_index"], run["start_index"]), []).append(run)

    bar_ids = sorted(by_bar)
    n_trajectories = 0
    bytes_written = 0

    bars_index = []
    for bar_id in bar_ids:
        entry = load_bar_library(root, library_name, bar_id)
        problem = entry["problem"]
        goals_in_run = sorted({run["goal_index"] for run in by_bar[bar_id]})

        goal_rows = []
        for goal_index in goals_in_run:
            goal = entry["goals"][goal_index]
            goal_runs = by_goal[(bar_id, goal_index)]
            stats = summarize_group(goal_runs)
            goal_rows.append({
                "goal_index": goal_index,
                "world_from_bar": goal["world_from_bar"],
                "primary_conf": goal["primary_conf"],
                "n_free_branch_pairs": goal["n_free_branch_pairs"],
                "n_starts_found": goal["n_starts_found"],
                "difficulty_counts": goal.get("difficulty_counts", {}),
                "n_runs": stats["n_runs"],
                "success_rate": stats["success_rate"],
                "median_wall_s": stats["median_wall_s"],
            })

        write_json(os.path.join(out_dir, "bars", f"{bar_id}.json"), {
            "bar_id": bar_id,
            "problem": problem,
            "goals": goal_rows,
        })
        bars_index.append({
            "bar_id": bar_id,
            "bar_length_m": problem["bar_length_m"],
            "n_goals": len(goal_rows),
        })

        # Per goal: the sweep detail and the starts, which is what the goal
        # page draws.
        sweep_detail = load_sweep_detail(root, library_name, bar_id)
        for goal_index in goals_in_run:
            goal = entry["goals"][goal_index]
            started = {run["start_index"] for run in by_goal[(bar_id, goal_index)]}
            start_rows = []
            for start in goal["starts"]:
                row = {
                    "start_index": start["start_index"],
                    "rank": start["rank"],
                    "anchor": start["anchor"],
                    "delta": start["delta"],
                    "world_from_bar": start["world_from_bar"],
                    "difficulty": start.get("difficulty"),
                    "direct_path": start.get("direct_path", {}),
                    "planned": start["start_index"] in started,
                }
                if row["planned"]:
                    stats = summarize_group(
                        by_start[(bar_id, goal_index, start["start_index"])])
                    row["success_rate"] = stats["success_rate"]
                    row["median_wall_s"] = stats["median_wall_s"]
                    row["conf"] = start["conf"]
                start_rows.append(row)

            write_json(
                os.path.join(out_dir, "bars", bar_id, f"goal{goal_index:03d}.json"),
                {
                    "bar_id": bar_id,
                    "goal_index": goal_index,
                    "world_from_bar": goal["world_from_bar"],
                    "primary_conf": goal["primary_conf"],
                    "free_branch_pairs": goal.get("free_branch_pairs", []),
                    "sweep_counts": goal["sweep_counts"],
                    "difficulty_counts": goal.get("difficulty_counts", {}),
                    "production_derivation": goal.get("production_derivation", {}),
                    "candidates": sweep_detail.get(str(goal_index), []),
                    "starts": start_rows,
                },
            )

            # Per start: the seed-by-seed detail, plus the trajectories.
            for start_index in sorted(started):
                start_runs = by_start[(bar_id, goal_index, start_index)]
                seeds = []
                for run in sorted(start_runs, key=lambda r: r["seed"]):
                    seed_row: Dict[str, Any] = {
                        "seed": run["seed"],
                        "status": run.get("status"),
                        "success": run.get("success"),
                        "failed_pass": run.get("failed_pass"),
                        "wall_s": run.get("wall_s"),
                        "passes": {},
                    }
                    for pass_name, record in (run.get("passes") or {}).items():
                        stem = (f"{bar_id}_g{goal_index:03d}_s{start_index:04d}"
                                f"_k{run['seed']:02d}_{pass_name}")
                        pass_row = {
                            "success": record.get("success"),
                            "wall_s": record.get("wall_s"),
                            "iterations": record.get("iterations"),
                            "planner": record.get("planner"),
                            "tree_a_nodes": record.get("tree_a_nodes"),
                            "tree_b_nodes": record.get("tree_b_nodes"),
                            "n_waypoints": record.get("n_waypoints"),
                            "extend_stop_reasons": record.get("extend_stop_reasons"),
                            "failure_reason": record.get("failure_reason"),
                            "breakdown": (record.get("profile") or {}).get("breakdown", {}),
                            "speedscope": (record.get("profile") or {}).get("speedscope"),
                        }
                        if record.get("path"):
                            rel = os.path.join("traj", f"{stem}.bin")
                            bytes_written += write_trajectory(
                                os.path.join(out_dir, rel), record["path"])
                            n_trajectories += 1
                            pass_row["trajectory"] = rel.replace(os.sep, "/")
                        seed_row["passes"][pass_name] = pass_row
                    seeds.append(seed_row)

                write_json(
                    os.path.join(out_dir, "bars", bar_id, f"goal{goal_index:03d}",
                                 f"start{start_index:04d}.json"),
                    {
                        "bar_id": bar_id,
                        "goal_index": goal_index,
                        "start_index": start_index,
                        "seeds": seeds,
                    },
                )

        if verbose:
            print(f"[export] {bar_id}: {len(goal_rows)} goal(s)", flush=True)

    config = summary.get("config", {})
    source_key = config.get("source_key", "")
    write_json(os.path.join(out_dir, "summary.json"), summary)
    write_json(os.path.join(out_dir, "run.json"), {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "run_name": run_name,
        "library": library_name,
        "config": config,
        "bars": bars_index,
        "joint_order": _joint_order(root, library_name, bar_ids),
        # Where the robot's geometry sits, relative to this folder. It is
        # shared by every run rather than copied into each one: it is three
        # megabytes and depends only on the robot.
        "assets": f"../../../assets/{source_key}",
        "totals": summary["totals"],
    })
    copy_static_site(out_dir)

    _update_catalog(root, run_name, summary)

    if verbose:
        print(f"[export] {len(bar_ids)} bar(s), {n_trajectories} trajectory file(s), "
              f"{bytes_written / 1e6:.1f} MB of motion -> {out_dir}")
    return {"n_bars": len(bar_ids), "n_trajectories": n_trajectories,
            "trajectory_bytes": bytes_written, "out_dir": out_dir}


def copy_static_site(out_dir: str) -> None:
    """Copy the dashboard's own HTML, styles and scripts next to the data.

    The page and the data it reads then sit in one folder, so a run can be
    served, zipped or handed to someone else as a single thing.

    Args:
        out_dir (str): the run's dashboard folder.
    """
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_dir = os.path.dirname(package_dir)
    source = os.path.join(repo_dir, "benchmark", "dashboard")
    if not os.path.isdir(source):
        raise FileNotFoundError(
            f"The dashboard's files are missing from {source}. The benchmark "
            "package and its dashboard folder belong to the same checkout."
        )
    for name in ("index.html", "static"):
        src = os.path.join(source, name)
        dst = os.path.join(out_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    _stamp_asset_versions(out_dir)


def _stamp_asset_versions(out_dir: str) -> None:
    """Make the page ask for its own script and stylesheet by content.

    A browser holds on to a cached script hard, and a fix that is on disk but
    not in the cache looks exactly like a fix that did not work -- which has
    cost real debugging time here. Appending a hash of the file's contents to
    the URL means a changed file is a different URL, so the browser fetches it,
    and an unchanged one still comes from cache.

    Args:
        out_dir (str): the dashboard folder whose index.html to rewrite.
    """
    index_path = os.path.join(out_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as handle:
        html = handle.read()

    for relative in ("static/css/index.css", "static/js/app.js"):
        path = os.path.join(out_dir, relative.replace("/", os.sep))
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as handle:
            digest = hashlib.sha1(handle.read()).hexdigest()[:8]
        html = html.replace(f'"./{relative}"', f'"./{relative}?v={digest}"')

    with open(index_path, "w", encoding="utf-8") as handle:
        handle.write(html)


def _joint_order(root: str, library_name: str, bar_ids: Sequence[str]) -> List[str]:
    """Return the twelve arm joint names, in the order every result uses.

    Args:
        root (str): the benchmark data root.
        library_name (str): the library.
        bar_ids (Sequence[str]): the bars in the run; the first is enough.

    Returns:
        List[str]: the joint names, left arm then right.
    """
    if not bar_ids:
        return []
    entry = load_bar_library(root, library_name, bar_ids[0])
    return entry["problem"]["joint_names_12"]


def _update_catalog(root: str, run_name: str, summary: Dict[str, Any]) -> None:
    """Add or refresh this batch's entry in the catalogue of runs.

    The dashboard reads the catalogue first, to offer a run picker and to
    compare one run against another.

    Args:
        root (str): the benchmark data root.
        run_name (str): the batch being exported.
        summary (Dict[str, Any]): its summary.
    """
    path = os.path.join(root, "runs.json")
    catalog: Dict[str, Any] = {"schema_version": DASHBOARD_SCHEMA_VERSION, "runs": []}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            catalog = json.load(handle)

    config = summary.get("config", {})
    entry = {
        "run_name": run_name,
        "created": config.get("created"),
        "library": config.get("library"),
        "n_runs": summary["totals"]["n_runs"],
        "success_rate": summary["totals"]["success_rate"],
        "path": f"runs/{run_name}/dashboard",
        "provenance": config.get("provenance", {}),
    }
    catalog["runs"] = [row for row in catalog.get("runs", [])
                       if row.get("run_name") != run_name]
    catalog["runs"].append(entry)
    catalog["runs"].sort(key=lambda row: row.get("created") or 0, reverse=True)
    write_json(path, catalog)
