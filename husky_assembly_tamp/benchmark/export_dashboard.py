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
    setup_by_bar: Dict[str, Dict[str, float]] = {}
    for bar_id in bar_ids:
        entry = load_bar_library(root, library_name, bar_id)
        setup = setup_timing(entry)
        setup_by_bar[bar_id] = setup["per_goal_mean_s"]
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
                "planning_median_s": planning_median(stats),
                "setup_s": setup["per_goal"][goal_index],
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
                    # So the goal page can tell "coarse pass only" from solved.
                    row["coarse_success_rate"] = stats.get("coarse_success_rate")
                    row["median_wall_s"] = stats["median_wall_s"]
                    row["planning_median_s"] = planning_median(stats)
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
                    "setup_s": setup["per_goal"][goal_index],
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
                            # Planner-variant experiments: the search's own
                            # verdict and the strict repair's, when recorded.
                            "search_success": record.get("search_success"),
                            "postprocess": record.get("postprocess"),
                            "variant": record.get("variant"),
                            "n_ik_solves": record.get("n_ik_solves"),
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
    # Setup time -- finding goals and surveying starts, before any motion
    # planning -- belongs to the library, not to the runs, so summarize never
    # sees it. It is attached here, beside the planning numbers, so the page
    # can show both and keep them apart.
    for row in summary["per_bar"]:
        row["setup_s_per_goal"] = setup_by_bar.get(row["bar_id"], {})
    n_goals = {bar["bar_id"]: bar["n_goals"] for bar in bars_index}
    total_goals = sum(n_goals.values()) or 1
    summary["totals"]["setup_s_per_goal"] = {
        stage: sum(setup_by_bar[bar_id].get(stage, 0.0) * n_goals[bar_id]
                   for bar_id in setup_by_bar) / total_goals
        for stage in SETUP_STAGES
    }
    write_json(os.path.join(out_dir, "summary.json"), summary)
    write_json(os.path.join(out_dir, "run.json"), {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "run_name": run_name,
        "library": library_name,
        "variant": config.get("variant"),
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
    attach_trees(root, run_name, verbose=verbose)

    if verbose:
        print(f"[export] {len(bar_ids)} bar(s), {n_trajectories} trajectory file(s), "
              f"{bytes_written / 1e6:.1f} MB of motion -> {out_dir}")
    return {"n_bars": len(bar_ids), "n_trajectories": n_trajectories,
            "trajectory_bytes": bytes_written, "out_dir": out_dir}


def planning_median(stats: Dict[str, Any]) -> Optional[float]:
    """Median motion-planning time of the solved runs in a group.

    The median over all runs is the wrong number to show: most runs fail the
    coarse pass at its time limit, so it just reports that limit (about 36 s)
    for every goal. The solved runs' fine pass is the time it takes the planner
    to actually find a motion.

    Args:
        stats (Dict[str, Any]): a group summary from ``summarize_group``.

    Returns:
        Optional[float]: seconds, or None when nothing in the group was solved.
    """
    if not stats.get("n_success"):
        return None
    return float(stats["fine_wall_s"]["p50"])


# The stages of work done before motion planning, in the order they happen.
SETUP_STAGES = ("goal_ik", "start_survey", "straight_line_probes",
                "planner_start_derivation")


def setup_timing(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Split one bar's library-building time into its stages.

    None of this is motion planning. It is the work of posing the benchmark's
    questions: finding reachable goals, surveying the carry poses a transfer
    could start from, probing whether a straight line joins each start to its
    goal, and running the planner's own start derivation for comparison. The
    dashboard reports it separately from planning time so the two are never
    mistaken for each other.

    Args:
        entry (Dict[str, Any]): one bar's library entry.

    Returns:
        Dict[str, Any]: ``per_goal`` -- seconds per stage for each goal index,
        and ``per_goal_mean_s`` -- the mean of each stage over the bar's goals.
    """
    goals = entry["goals"]
    per_goal: Dict[int, Dict[str, float]] = {}
    for goal in goals:
        per_goal[goal["goal_index"]] = {
            "start_survey": float(goal.get("sweep_wall_s", 0.0)),
            "straight_line_probes": float(sum(
                (start.get("direct_path") or {}).get("wall_s", 0.0)
                for start in goal.get("starts", []))),
            "planner_start_derivation": float(
                (goal.get("production_derivation") or {}).get("wall_s", 0.0)),
        }

    # Goal IK is recorded for the whole bar at once. Libraries built before it
    # was stored directly only have the bar's total, so there it is what is
    # left over once every per-goal stage is taken out.
    if "goal_sampling_wall_s" in entry:
        goal_ik_total = float(entry["goal_sampling_wall_s"])
    else:
        accounted = sum(sum(stages.values()) for stages in per_goal.values())
        goal_ik_total = max(0.0, float(entry.get("wall_s", 0.0)) - accounted)
    goal_ik_each = goal_ik_total / max(1, len(goals))
    for stages in per_goal.values():
        stages["goal_ik"] = goal_ik_each
        stages["total"] = sum(stages[stage] for stage in SETUP_STAGES)

    n = max(1, len(per_goal))
    mean = {stage: sum(stages[stage] for stages in per_goal.values()) / n
            for stage in SETUP_STAGES + ("total",)}
    return {"per_goal": per_goal, "per_goal_mean_s": mean}


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
        "variant": config.get("variant"),
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


def attach_trees(root: str, run_name: str, *, verbose: bool = True) -> int:
    """Put any saved search trees where the dashboard's start pages find them.

    Trees come from two places. A batch run with ``--store-trees`` saves them
    for every run. A ``replay`` re-plans chosen starts afterwards, with trees,
    because the original batch did not keep them. A replay uses the same seed
    and settings, but the search stops on a wall-clock limit, so it can stop at
    a different point and even end differently; its own outcome is recorded
    beside its trees, and the page shows both rather than pretending the replay
    is the original.

    Args:
        root (str): the benchmark data root.
        run_name (str): which batch.
        verbose (bool): print what was attached.

    Returns:
        int: how many tree files were attached.
    """
    run_dir = os.path.join(root, "runs", run_name)
    out_dir = os.path.join(run_dir, "dashboard")
    tree_out = os.path.join(out_dir, "trees")

    # (bar, goal, start, seed) -> {"source", "record", "tree_dir"}
    found: Dict[Any, Dict[str, Any]] = {}
    sources = [
        ("original", os.path.join(run_dir, "runs"), os.path.join(run_dir, "trees")),
        ("replay", os.path.join(run_dir, "replays", "runs"),
         os.path.join(run_dir, "replays", "trees")),
    ]
    for source, runs_dir, tree_dir in sources:
        if not os.path.isdir(runs_dir):
            continue
        for bar_folder in os.listdir(runs_dir):
            folder = os.path.join(runs_dir, bar_folder)
            if not os.path.isdir(folder):
                continue
            for name in os.listdir(folder):
                if not name.endswith(".json"):
                    continue
                with open(os.path.join(folder, name), "r", encoding="utf-8") as handle:
                    record = json.load(handle)
                if not any(p.get("tree_file") for p in (record.get("passes") or {}).values()):
                    continue
                key = (record["bar_id"], record["goal_index"],
                       record["start_index"], record["seed"])
                # Trees saved by the original run win over a replay's.
                if key in found and found[key]["source"] == "original":
                    continue
                found[key] = {"source": source, "record": record, "tree_dir": tree_dir}

    by_start: Dict[Any, List[Dict[str, Any]]] = {}
    for (bar_id, goal_index, start_index, _seed), item in found.items():
        by_start.setdefault((bar_id, goal_index, start_index), []).append(item)

    attached = 0
    for (bar_id, goal_index, start_index), items in by_start.items():
        start_path = os.path.join(out_dir, "bars", bar_id, f"goal{goal_index:03d}",
                                  f"start{start_index:04d}.json")
        if not os.path.isfile(start_path):
            continue  # a start this batch never planned; nothing to attach to
        with open(start_path, "r", encoding="utf-8") as handle:
            start_file = json.load(handle)
        seeds = {row["seed"]: row for row in start_file.get("seeds", [])}

        for item in items:
            record = item["record"]
            trees = {}
            for pass_name, pass_record in (record.get("passes") or {}).items():
                tree_file = pass_record.get("tree_file")
                if not tree_file:
                    continue
                src = os.path.join(item["tree_dir"], tree_file)
                if not os.path.isfile(src):
                    continue
                os.makedirs(tree_out, exist_ok=True)
                shutil.copy2(src, os.path.join(tree_out, tree_file))
                trees[pass_name] = {
                    "tree": f"trees/{tree_file}",
                    "success": pass_record.get("success"),
                    "search_success": pass_record.get("search_success"),
                    "postprocess": pass_record.get("postprocess"),
                    "wall_s": pass_record.get("wall_s"),
                    "iterations": pass_record.get("iterations"),
                    "tree_a_nodes": pass_record.get("tree_a_nodes"),
                    "tree_b_nodes": pass_record.get("tree_b_nodes"),
                    "extend_stop_reasons": pass_record.get("extend_stop_reasons"),
                    "tree_error": pass_record.get("tree_error"),
                }
                # The failed attempts travel with the trees, when they were kept.
                attempts_file = pass_record.get("attempts_file")
                if attempts_file and os.path.isfile(os.path.join(item["tree_dir"], attempts_file)):
                    shutil.copy2(os.path.join(item["tree_dir"], attempts_file),
                                 os.path.join(tree_out, attempts_file))
                    trees[pass_name]["attempts"] = f"trees/{attempts_file}"
                    trees[pass_name]["n_attempts"] = pass_record.get("n_attempts")
                attached += 1
            if not trees:
                continue
            row = seeds.setdefault(record["seed"], {"seed": record["seed"], "passes": {}})
            row["trees"] = {
                "source": item["source"],
                "success": record.get("success"),
                "failed_pass": record.get("failed_pass"),
                "passes": trees,
            }

        start_file["seeds"] = sorted(seeds.values(), key=lambda row: row["seed"])
        with open(start_path, "w", encoding="utf-8") as handle:
            json.dump(start_file, handle)

    if verbose:
        print(f"[export] {attached} search tree file(s) attached across "
              f"{len(by_start)} start(s)")
    return attached
