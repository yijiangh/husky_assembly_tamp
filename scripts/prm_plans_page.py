"""The page of a set of bar-carry plans: every bar of a process in build order, each plan in 3D in its own scene.

Reads what ``plan_process_carry.py`` and ``prm_check_solved.py`` saved for one tag in a plans folder and writes
``<plans dir>/dashboard/``: the page (``benchmark/dashboard/plans.html``), ``plans.json`` (one row per bar) and
``bars/<bar>.json`` (the path, the scene's address, every timing and counter). Each bar's roadmap run is exported
too (``prm_pilot.py export``), which writes the 3D file of the scene the plan was made in.

Serve the benchmark data root and open the folder:

    python -m http.server --directory "$HUSKY_BENCH_ROOT" 8000
    http://localhost:8000/prm/plans/<problem>/<set>/dashboard/

Usage:

    python scripts/prm_plans_page.py "$HUSKY_DESIGN_STUDY_DIR/<problem>" --plans-dir <folder> --tag prm
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from husky_assembly_tamp.benchmark.sibling_packages import ensure_sibling_packages  # noqa: E402

ensure_sibling_packages()

from husky_assembly_tamp.benchmark.asset_versions import stamp_page_assets  # noqa: E402
from husky_assembly_tamp.benchmark.paths import long_path  # noqa: E402
from husky_assembly_tamp.motion_planner.holding_prm.pilot import export_dashboard  # noqa: E402
from husky_assembly_tamp.motion_planner.holding_prm.store import (  # noqa: E402
    dashboard_source_dir,
    load_roadmap,
    read_json,
    run_dir,
    write_json_atomic,
)

# * The check's fields the page shows (the rest stay in checks.jsonl).
CHECK_FIELDS = ("ok", "reason", "n_waypoints", "saved_max_step_deg", "bar_travel_m", "bar_turn_deg",
                "joint_travel_deg", "holding_between_pos_mm", "holding_between_rot_deg", "joint_reversals",
                "repair_success", "repair_reason", "repair_s", "saved_waypoints_colliding", "lowest_z_m",
                "lowest_z_above_base_frame_m", "lowest_body", "lowest_at_waypoint", "lowest_z_at_goal_m")


def bar_order(process_dir: str) -> List[Dict[str, Any]]:
    """The bars with a jointing action, in schedule order, with how many bars stand when each is carried.

    Args:
        process_dir (str): the process folder.

    Returns:
        List[Dict[str, Any]]: ``bar``, ``order`` (from 1) and ``built_before`` per bar.
    """
    schedule = read_json(os.path.join(process_dir, "process.json"))["schedule"]
    rows = []
    for name in schedule:
        if not name.endswith("_J"):
            continue
        action = read_json(os.path.join(process_dir, "actions", f"{name}.json"))
        carry = next(m for m in action["movements"] if m.get("coupled") and m.get("path") == "free")
        rows.append({"bar": name[:-2], "order": len(rows) + 1, "built_before": len(carry["start"].get("built") or [])})
    return rows


def checks_by_bar(plans_dir: str, tag: str) -> Dict[str, Dict[str, Any]]:
    """The last check of each bar for a tag, from ``checks.jsonl``."""
    found: Dict[str, Dict[str, Any]] = {}
    path = os.path.join(plans_dir, "checks.jsonl")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("tag") == tag and row.get("bar"):
                    found[row["bar"]] = row
    return found


def copy_page(out_dir: str) -> None:
    """Copy the plans page and the shared scripts into a folder, with content hashes on their addresses."""
    source = dashboard_source_dir()
    os.makedirs(out_dir, exist_ok=True)
    shutil.copytree(long_path(os.path.join(source, "static")), long_path(os.path.join(out_dir, "static")),
                    dirs_exist_ok=True)
    for name in ("plans.html", "index.html"):
        shutil.copyfile(os.path.join(source, "plans.html"), os.path.join(out_dir, name))
    stamp_page_assets(out_dir, ("plans.html", "index.html"), ("static/css/index.css", "static/js/plans-viewer.js"))


def precompute_of(root: str, run_name: str, page: Dict[str, Any]) -> Dict[str, Any]:
    """What a bar's roadmap is and what it cost: size, build phases, tracking files."""
    config = read_json(os.path.join(run_dir(root, run_name), "config.json"))
    roadmap = load_roadmap(root, run_name)
    return {"settings": {key: config["settings"].get(key) for key in ("nodes", "workers", "k", "radius_deg")},
            "n_nodes": roadmap.n_nodes, "n_edges": roadmap.n_edges,
            "n_edges_free": int(sum(bool(free) for free in roadmap.edges["free"])),
            "build_timings": page["precompute"]["build_timings"], "tracking": page["precompute"]["tracking"]}


def export_plans(process_dir: str, plans_dir: str, tag: str, root: str, fallback_tag: str = "") -> str:
    """Write the page of one tag's plans.

    Args:
        process_dir (str): the process folder the plans were made for.
        plans_dir (str): the folder holding ``<bar>_J.solved_motion.<tag>[.plan_info].json`` and ``checks.jsonl``.
        tag (str): which plans.
        root (str): the benchmark data root (it is what gets served).
        fallback_tag (str): for a bar whose ``tag`` plan found no path, show this tag's plan when it found one
            (the row then says which planner made it, and keeps what the first one tried).

    Returns:
        str: the dashboard folder.
    """
    problem = os.path.basename(os.path.normpath(process_dir))
    out_dir = os.path.join(plans_dir, "dashboard")
    copy_page(out_dir)
    os.makedirs(os.path.join(out_dir, "bars"), exist_ok=True)
    checks = checks_by_bar(plans_dir, tag)
    fallback_checks = checks_by_bar(plans_dir, fallback_tag) if fallback_tag else {}

    def address(path: str) -> str:
        """A file under the data root as an address relative to the page."""
        return os.path.relpath(path, out_dir).replace(os.sep, "/")

    rows: List[Dict[str, Any]] = []
    assets: Optional[str] = None
    for row in bar_order(process_dir):
        bar = row["bar"]
        info_path = os.path.join(plans_dir, f"{bar}_J.solved_motion.{tag}.plan_info.json")
        if not os.path.isfile(info_path):
            rows.append({**row, "planned": False})
            continue
        info = read_json(info_path)
        prm = info.get("prm") or {}
        run_name = f"{problem}__{bar}"
        # The scene is the bar's turn whichever planner planned in it: taken from the first plan's files.
        scene_files = (prm.get("scene") or {}).get("scene_files") or {}
        used_tag, first_try, bar_checks = tag, None, checks
        other_path = os.path.join(plans_dir, f"{bar}_J.solved_motion.{fallback_tag}.plan_info.json")
        if fallback_tag and not info.get("ok") and os.path.isfile(other_path) and read_json(other_path).get("ok"):
            first_try = {"backend": info.get("backend"), "failure_reason": info.get("failure_reason"),
                         "plan_s": info.get("plan_s"), "best_gap_left_mm": prm.get("best_gap_left_mm")}
            info, used_tag, bar_checks = read_json(other_path), fallback_tag, fallback_checks
            prm = info.get("prm") or {}
        # The run's own page and the 3D file of the scene this plan was made in.
        page = read_json(os.path.join(export_dashboard(root, run_name), "prm.json"))
        assets = assets or address(os.path.join(root, "assets", page["source_key"]))
        scene = os.path.splitext(scene_files["state"])[0] + ".glb" if scene_files.get("state") else None
        solved_path = os.path.join(plans_dir, f"{bar}_J.solved_motion.{used_tag}.json")
        trajectory: Dict[str, Any] = {}
        if info.get("ok") and os.path.isfile(solved_path):
            solved = read_json(solved_path)
            trajectory = solved["movements"][info["movement"]]["trajectory"]
        check = {key: bar_checks[bar].get(key) for key in CHECK_FIELDS} if bar in bar_checks else None
        precompute = precompute_of(root, run_name, page)
        timings = dict(prm.get("timings") or {})
        rows.append({
            **row, "planned": True, "ok": bool(info.get("ok")), "failure_reason": info.get("failure_reason"),
            "backend": info.get("backend"), "first_try": first_try,
            "plan_s": info.get("plan_s"), "timings": timings, "workers": prm.get("workers"),
            "n_waypoints": info.get("n_waypoints"), "tolerance_mm": prm.get("tolerance_mm"),
            "route_nodes": len(prm.get("route") or []), "check": check,
            # Where the carry starts (class, and whether the pose was precomputed or found by a fallback).
            "start": {key: (prm.get("start") or {}).get(key)
                      for key in ("carry", "label", "source", "midpoint", "roll_deg", "found_after")},
            "fallback_stages": [stage.get("stage") for stage in (prm.get("walk_in") or {}).get("stages", [])],
            "follow_on": {key: (info.get("follow_on") or {}).get(key)
                          for key in ("end_is_authored_posture", "end_off_authored_deg", "end_refused")}
            | {"rejected": len((info.get("follow_on") or {}).get("rejected") or [])} if info.get("follow_on") else None,
            "roadmap": {key: precompute[key] for key in ("n_nodes", "n_edges", "n_edges_free")},
            "build_s": (precompute["build_timings"] or {}).get("total_s"),
            "track_s": sum(float(t.get("wall_s") or 0.0) for t in precompute["tracking"]),
            "file": f"bars/{bar}.json",
        })
        write_json_atomic(os.path.join(out_dir, "bars", f"{bar}.json"), {
            "bar": bar, "run_name": run_name, "movement": info.get("movement"), "problem": page["problem"],
            "scene": address(scene) if scene and os.path.isfile(scene) else None,
            "run_page": address(os.path.join(run_dir(root, run_name), "dashboard")) + f"/#/query/{bar}_{tag}",
            "joint_names": trajectory.get("joint_names"),
            # Seven decimals: the page measures the holding of the bar from these, to a thousandth of a millimetre.
            "positions": [[round(float(v), 7) for v in conf] for conf in trajectory.get("positions", [])],
            "plan": {key: info.get(key) for key in ("ok", "failure_reason", "plan_s", "n_waypoints", "goal",
                                                    "base_shift", "follow_on", "post", "backend", "search")},
            "first_try": first_try,
            "prm": {key: prm.get(key) for key in ("workers", "timings", "counters", "starts", "pretracked_edges",
                                                  "edges_looked_at", "edge_verdicts", "scene_blocked_edges",
                                                  "dropped_in_final_check", "tube", "retracks", "tolerance_mm",
                                                  "route", "route_scores_mm", "start", "n_waypoints_1cm",
                                                  "goals_rejected", "best_gap_left_mm", "scene", "walk_in")},
            "check": bar_checks.get(bar), "precompute": precompute,
        })
        print(f"  {bar}: {'ok' if info.get('ok') else info.get('failure_reason')}")
    write_json_atomic(os.path.join(out_dir, "plans.json"), {
        "problem": problem, "tag": tag, "fallback_tag": fallback_tag,
        "set": os.path.basename(os.path.normpath(plans_dir)), "assets": assets,
        "generated": time.time(), "bars": rows,
    })
    return out_dir


def main(argv: Optional[List[str]] = None) -> int:
    """Run from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("process_dir", help="the process folder the plans were made for")
    parser.add_argument("--plans-dir", required=True, help="the folder holding the saved plans and checks.jsonl")
    parser.add_argument("--tag", default="prm", help="which plans (their --save-tag)")
    parser.add_argument("--fallback-tag", default="",
                        help="for a bar whose --tag plan found no path, show this tag's plan when it found one")
    parser.add_argument("--root", default=os.environ.get("HUSKY_BENCH_ROOT"), help="benchmark data root")
    args = parser.parse_args(argv)
    if not args.root:
        parser.error("give --root or set HUSKY_BENCH_ROOT")
    out_dir = export_plans(args.process_dir, os.path.abspath(args.plans_dir), args.tag, os.path.abspath(args.root),
                           args.fallback_tag)
    print(f"[page] {out_dir}\n       serve the data root and open "
          f"http://localhost:8000/{os.path.relpath(out_dir, os.path.abspath(args.root)).replace(os.sep, '/')}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
