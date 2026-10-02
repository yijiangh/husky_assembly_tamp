"""Search for one good carry (home) pose per class: horizontal, vertical, back.

    stage1   score a grid of candidates (class x grasp-midpoint position x roll)
             on one or more bars: can the pose be held, how far its best
             branch pair is from joint limits and singularities, and whether
             it survives 5 cm / 15 degree nudges (which also keeps it clear of
             the robot body). Today's three anchors are scored the same way
             for comparison.
    starts   write the top candidates of each class as a starts file, which
             ``prm_pilot.py build/query --starts-file`` take, so the roadmap
             can score them against many goals
    stage2   after roadmaps were built with that starts file and queried and
             refined per goal (query tag ``g<goal>_carry``), rank the
             candidates by the number of goals they reach with an exact path
    pick     write one carry pose per class to ``asset/carry_poses.json``
             (a starts file), provisional until a person chooses

Outputs go to ``<root>/prm/carry/<tag>/``. Nothing in the planner or the
benchmark library is changed.

Example (PowerShell, from the repository root, with HUSKY_BENCH_ROOT set):

    .venv\\Scripts\\python.exe scripts\\prm_carry_search.py stage1 --tag carry_v1 --bars B18,B3 --workers 8
    .venv\\Scripts\\python.exe scripts\\prm_carry_search.py starts --tag carry_v1 --top 5
"""
import argparse
import os
import sys
from time import perf_counter, time
from typing import Any, Dict, List, Optional, Tuple

# The package sits one level up from scripts/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from husky_assembly_tamp.motion_planner.holding_prm.carry import CARRY_CLASSES, ROLL_STEP_DEG, candidate_grid  # noqa: E402
from husky_assembly_tamp.motion_planner.holding_prm.parallel import WorkerPool, default_workers, task_carry  # noqa: E402
from husky_assembly_tamp.motion_planner.holding_prm.pilot import chunked  # noqa: E402
from husky_assembly_tamp.motion_planner.holding_prm.sampling import add_counters  # noqa: E402
from husky_assembly_tamp.motion_planner.holding_prm.store import (  # noqa: E402
    prm_root,
    provenance_record,
    read_json,
    write_json_atomic,
)

# * Candidates per scoring task.
CARRY_CHUNK = 16


def carry_dir(root: str, tag: str) -> str:
    """The folder of one carry search: ``<root>/prm/carry/<tag>``."""
    return os.path.join(prm_root(root), "carry", tag)


def candidate_key(candidate: Dict[str, Any]) -> Tuple[str, float, float, float, float]:
    """What identifies a candidate across bars: class, midpoint, roll."""
    return (candidate["carry"], *[round(v, 4) for v in candidate["midpoint"]], round(candidate["roll_deg"], 3))


def score_on_bar(root: str, library: str, bar_id: str, candidates: List[Dict[str, Any]], with_nudges: bool,
                 workers: int, label: str) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Score candidates in one bar's scene.

    Args:
        root (str): the benchmark data root.
        library (str): the library whose trimmed scenes to use.
        bar_id (str): the bar.
        candidates (List[Dict[str, Any]]): ``carry``, ``midpoint``, ``roll_deg`` each.
        with_nudges (bool): also run the nudge test.
        workers (int): worker processes.
        label (str): shown in the progress lines.

    Returns:
        Tuple: the scores (in candidate order) and the work tally.
    """
    source_key = read_json(os.path.join(root, "library", library, "index.json"))["source_key"]
    # Not used by the carry task; the pool wants them (the roadmap's defaults).
    edge_settings = {"collision_step_rad": float(np.radians(10.0)), "viol_step_rad": float(np.radians(2.5)),
                     "rot_deg_per_mm": 0.4}
    tasks = [{"chunk": n, "candidates": list(piece), "with_nudges": with_nudges}
             for n, piece in enumerate(chunked(candidates, CARRY_CHUNK))]
    scores, counters = [], {}
    with WorkerPool(root, source_key, bar_id, workers, None, edge_settings) as pool:
        for result in pool.run(task_carry, tasks, f"{label} {bar_id}"):
            scores.extend(result["scores"])
            add_counters(counters, result["counters"])
    return scores, counters


def combine(per_bar: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Join each candidate's scores over the bars: a candidate is only as good as its worst bar.

    Args:
        per_bar (Dict[str, List[Dict[str, Any]]]): bar -> scores (same candidates, same order).

    Returns:
        List[Dict[str, Any]]: one row per candidate with ``holdable_all``,
        ``min_free``, ``min_comfort`` (and the nudge numbers when present) and
        ``per_bar`` (each bar's own score).
    """
    bars = list(per_bar)
    rows = []
    for index, first in enumerate(per_bar[bars[0]]):
        scores = {bar: per_bar[bar][index] for bar in bars}
        row = {"carry": first["carry"], "midpoint": first["midpoint"], "roll_deg": first["roll_deg"],
               "holdable_all": all(s["n_free"] > 0 for s in scores.values()),
               "min_free": min(s["n_free"] for s in scores.values()),
               "min_comfortable": min(s.get("n_comfortable", 0) for s in scores.values()),
               "min_comfort": min(s.get("best", {}).get("comfort", 0.0) for s in scores.values()),
               "per_bar": scores}
        if all("nudge_holdable" in s for s in scores.values()):
            row["min_nudge_holdable"] = min(s["nudge_holdable"] for s in scores.values())
            # The weakest nudge over all bars: how many free pairs are left there.
            row["min_nudge_free"] = min(min(s["nudge_free"]) for s in scores.values())
        rows.append(row)
    return rows


def spread_pick(rows: List[Dict[str, Any]], count: int, min_distance_m: float = 0.15,
                min_roll_deg: float = 60.0) -> List[Dict[str, Any]]:
    """The best rows in order, skipping one too close to a row already picked (same place and similar roll).

    Args:
        rows (List[Dict[str, Any]]): candidates, best first.
        count (int): how many to pick.
        min_distance_m (float): midpoints closer than this count as the same place...
        min_roll_deg (float): ...when their rolls also differ by less than this.

    Returns:
        List[Dict[str, Any]]: the picked rows.
    """
    picked: List[Dict[str, Any]] = []
    for row in rows:
        close = any(np.linalg.norm(np.subtract(row["midpoint"], p["midpoint"])) < min_distance_m
                    and abs((row["roll_deg"] - p["roll_deg"] + 180.0) % 360.0 - 180.0) < min_roll_deg
                    for p in picked)
        if not close:
            picked.append(row)
        if len(picked) >= count:
            break
    return picked


def today_candidates() -> List[Dict[str, Any]]:
    """Today's three anchor midpoints, at every roll (the planner tries all rolls too)."""
    rolls = np.arange(-180.0, 180.0, ROLL_STEP_DEG)
    return [{"carry": carry, "midpoint": list(spec["today"]), "roll_deg": float(roll), "today": True}
            for carry, spec in CARRY_CLASSES.items() for roll in rolls]


def cmd_stage1(root: str, args: argparse.Namespace) -> None:
    """Grid scoring on every bar, then the nudge test on the best of each class."""
    started = perf_counter()
    bars = [b.strip() for b in args.bars.split(",") if b.strip()]
    grid = candidate_grid(args.step)
    print(f"[carry stage1] {len(grid)} grid candidates on {bars}, {args.workers} workers")
    counters: Dict[str, Any] = {}
    # * 1. Every grid candidate on every bar, without nudges.
    per_bar = {}
    for bar in bars:
        per_bar[bar], counters[f"grid_{bar}"] = score_on_bar(root, args.library, bar, grid, False, args.workers, "grid")
    rows = combine(per_bar)
    # * 2. The best of each class (holdable on every bar, most comfortable
    # * pairs), spread out over places and rolls, plus today's anchors, with nudges.
    finalists = []
    for carry in CARRY_CLASSES:
        members = [r for r in rows if r["carry"] == carry and r["holdable_all"]]
        members.sort(key=lambda r: (r["min_comfortable"], r["min_free"]), reverse=True)
        finalists.extend({"carry": r["carry"], "midpoint": r["midpoint"], "roll_deg": r["roll_deg"]}
                         for r in spread_pick(members, args.finalists))
        print(f"  {carry}: {len(members)} of {sum(1 for r in rows if r['carry'] == carry)} candidates holdable on "
              f"every bar; most comfortable pairs {members[0]['min_comfortable']}" if members
              else f"  {carry}: none holdable")
    finalists.extend(today_candidates())
    nudged = {}
    for bar in bars:
        nudged[bar], counters[f"nudge_{bar}"] = score_on_bar(root, args.library, bar, finalists, True, args.workers,
                                                            "nudges")
    final = combine(nudged)
    for row, candidate in zip(final, finalists):
        row["today"] = bool(candidate.get("today"))
    # * 3. Rank: survives the most nudges on its worst bar, then keeps the most
    # * free pairs at its weakest nudge, then comfortable pairs, then free pairs.
    def rank_key(r: Dict[str, Any]) -> Tuple:
        return (r.get("min_nudge_holdable", 0), r.get("min_nudge_free", 0), r["min_comfortable"], r["min_free"])

    ranking = {}
    for carry in CARRY_CLASSES:
        members = sorted((r for r in final if r["carry"] == carry and not r["today"] and r["holdable_all"]),
                         key=rank_key, reverse=True)
        ranking[carry] = members
        today = [r for r in final if r["carry"] == carry and r["today"]]
        best_today = max(today, key=lambda r: (r["holdable_all"],) + rank_key(r))
        print(f"  {carry}: today's anchor at its best roll ({best_today['roll_deg']:g} deg): holdable on all bars "
              f"{best_today['holdable_all']}, nudges survived {best_today.get('min_nudge_holdable', 0)}/24, "
              f"weakest nudge keeps {best_today.get('min_nudge_free', 0)} pairs, comfortable pairs "
              f"{best_today['min_comfortable']}")
        for r in members[:5]:
            print(f"     candidate midpoint {r['midpoint']}, roll {r['roll_deg']:g} deg: nudges "
                  f"{r.get('min_nudge_holdable', 0)}/24, weakest nudge keeps {r.get('min_nudge_free', 0)} pairs, "
                  f"comfortable pairs {r['min_comfortable']}, free pairs >= {r['min_free']}")
    out = {"tag": args.tag, "library": args.library, "bars": bars, "step_m": args.step, "created": time(),
           "classes": CARRY_CLASSES, "grid": rows, "finalists": final, "ranking": ranking,
           "counters": counters, "wall_s": perf_counter() - started, "provenance": provenance_record()}
    path = os.path.join(carry_dir(root, args.tag), "stage1.json.gz")
    write_json_atomic(path, out)
    print(f"[carry stage1] done in {out['wall_s']:.0f} s; wrote {path}")


def cmd_starts(root: str, args: argparse.Namespace) -> None:
    """Write the top candidates of each class (and today's anchors) as a starts file."""
    stage1 = read_json(os.path.join(carry_dir(root, args.tag), "stage1.json.gz"))
    starts = []
    for carry, members in stage1["ranking"].items():
        for rank, row in enumerate(members[: args.top]):
            starts.append({"label": f"{carry}_{rank}", "carry": carry, "midpoint": row["midpoint"],
                           "roll_deg": row["roll_deg"], "today": False})
        today = [r for r in stage1["finalists"] if r["carry"] == carry and r["today"] and r["holdable_all"]]
        today.sort(key=lambda r: (r.get("min_nudge_holdable", 0), r.get("min_nudge_free", 0),
                                  r["min_comfortable"]), reverse=True)
        for row in today[: args.top_today]:
            starts.append({"label": f"{carry}_today_r{row['roll_deg']:g}", "carry": carry,
                           "midpoint": row["midpoint"], "roll_deg": row["roll_deg"], "today": True})
    path = os.path.join(carry_dir(root, args.tag), "starts.json")
    write_json_atomic(path, {"tag": args.tag, "kind": "carry", "starts": starts})
    print(f"[carry starts] {len(starts)} carry poses -> {path}")


def cmd_stage2(root: str, args: argparse.Namespace) -> None:
    """Rank the carry candidates by how well the roadmap connects them to many goals.

    Reads, for every (roadmap run, goal), the query ``g<goal>_carry`` (whose
    start set is the starts file) and its refinement, and counts per
    candidate: on how many goals its roadmap gap, and its exact gap (a path
    that holds the bar exactly), are within the target tolerance.
    """
    starts = read_json(os.path.join(carry_dir(root, args.tag), "starts.json"))["starts"]
    stats = {k: {"label": s["label"], "carry": s["carry"], "today": s["today"], "midpoint": s["midpoint"],
                 "roll_deg": s["roll_deg"], "goals": []} for k, s in enumerate(starts)}
    for spec in args.runs.replace("+", ";").split(";"):
        run, _, goals = spec.partition(":")
        for goal in [int(g) for g in goals.split(",") if g.strip()]:
            query_tag = f"g{goal:03d}_carry"
            query = read_json(os.path.join(prm_root(root), run, "queries", f"{query_tag}.json.gz"))
            refine_file = os.path.join(prm_root(root), run, "refine", f"{query_tag}_k2.5_p10mm_mid.json.gz")
            refine = read_json(refine_file) if os.path.isfile(refine_file) else {"starts": []}
            exact = {row["start_index"]: row["exact_gap_mm"] for row in refine["starts"]}
            for row in query["starts"]:
                stats[row["start_index"]]["goals"].append({
                    "run": run, "goal": goal, "gap_mm": row["best_gap_mm"],
                    "exact_gap_mm": exact.get(row["start_index"])})
    target = args.target_mm
    for entry in stats.values():
        gaps = [g["gap_mm"] for g in entry["goals"]]
        exact = [g["exact_gap_mm"] for g in entry["goals"]]
        entry["n_goals"] = len(gaps)
        entry["roadmap_within"] = sum(1 for g in gaps if g is not None and g <= target)
        entry["exact_within"] = sum(1 for g in exact if g is not None and g <= target)
        entry["exact_any"] = sum(1 for g in exact if g is not None)
        finite = [g for g in exact if g is not None]
        entry["median_exact_gap_mm"] = float(np.median(finite)) if finite else None
    ranking = {}
    for carry in CARRY_CLASSES:
        members = [e for e in stats.values() if e["carry"] == carry]
        members.sort(key=lambda e: (e["exact_within"], e["exact_any"],
                                    -(e["median_exact_gap_mm"] if e["median_exact_gap_mm"] is not None else 1e9)),
                     reverse=True)
        ranking[carry] = members
        print(f"  {carry}:")
        for e in members:
            median = "-" if e["median_exact_gap_mm"] is None else f"{e['median_exact_gap_mm']:.1f}"
            print(f"     {e['label']:<24s} exact path within {target:g} mm on {e['exact_within']}/{e['n_goals']} goals "
                  f"(any exact path {e['exact_any']}, roadmap within {target:g} mm {e['roadmap_within']}), "
                  f"median exact gap {median} mm, midpoint {e['midpoint']}, roll {e['roll_deg']:g}")
    path = os.path.join(carry_dir(root, args.tag), "stage2.json")
    write_json_atomic(path, {"tag": args.tag, "runs": args.runs, "target_mm": target, "created": time(),
                             "candidates": list(stats.values()), "ranking": ranking})
    print(f"[carry stage2] wrote {path}")


def cmd_pick(root: str, args: argparse.Namespace) -> None:
    """Write one carry pose per class to a file the planner tools can read (a starts file).

    By default the stage-2 winner of each class is picked and the file is
    marked provisional, with the top three of each class listed next to it so
    a person can choose; ``--choose carry=label`` overrides a class's pick.
    """
    stage2 = read_json(os.path.join(carry_dir(root, args.tag), "stage2.json"))
    chosen = dict(part.split("=", 1) for part in args.choose.split(",") if "=" in part)
    starts, alternatives = [], {}
    for carry, members in stage2["ranking"].items():
        candidates = [m for m in members if not m["today"]]
        alternatives[carry] = [{key: m[key] for key in ("label", "midpoint", "roll_deg", "exact_within", "exact_any",
                                                        "roadmap_within", "median_exact_gap_mm", "n_goals")}
                               for m in candidates[:3]]
        pick = next((m for m in candidates if m["label"] == chosen.get(carry)), candidates[0] if candidates else None)
        if pick is None:
            continue
        starts.append({"label": f"{carry}_carry", "carry": carry, "midpoint": pick["midpoint"],
                       "roll_deg": pick["roll_deg"], "from_candidate": pick["label"]})
    out = {"kind": "carry", "provisional": not chosen, "source": f"prm/carry/{args.tag}", "created": time(),
           "note": ("Carry poses by class: grasp midpoint in the mobile-base frame (m) and roll about the bar "
                    "(deg); holding_prm.carry.carry_bar_pose turns them into a bar pose for any bar. "
                    + ("Picked automatically (stage-2 winners); choose from 'alternatives'." if not chosen
                       else "Chosen by hand from 'alternatives'.")),
           "starts": starts, "alternatives": alternatives}
    write_json_atomic(args.out, out)
    for start in starts:
        print(f"  {start['carry']}: {start['from_candidate']} midpoint {start['midpoint']} roll {start['roll_deg']:g}")
    print(f"[carry pick] wrote {args.out} ({'provisional' if not chosen else 'chosen'})")


def build_parser() -> argparse.ArgumentParser:
    """The argument parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    stage1 = commands.add_parser("stage1", help="score the candidate grid and nudge the best")
    stage1.add_argument("--root", default=None)
    stage1.add_argument("--tag", required=True, help="search name, a folder under <root>/prm/carry")
    stage1.add_argument("--library", default="cdfm_v1")
    stage1.add_argument("--bars", default="B18,B3", help="bars every candidate must work for")
    stage1.add_argument("--step", type=float, default=0.1, help="grid spacing of the grasp midpoint (m)")
    stage1.add_argument("--finalists", type=int, default=30, help="candidates per class given the nudge test")
    stage1.add_argument("--workers", type=int, default=default_workers())
    starts = commands.add_parser("starts", help="write the top candidates as a starts file")
    starts.add_argument("--root", default=None)
    starts.add_argument("--tag", required=True)
    starts.add_argument("--top", type=int, default=5, help="candidates per class")
    starts.add_argument("--top-today", type=int, default=1, help="today's anchor at its best rolls, per class")
    stage2 = commands.add_parser("stage2", help="rank candidates by exact paths to many goals")
    stage2.add_argument("--root", default=None)
    stage2.add_argument("--tag", required=True)
    stage2.add_argument("--runs", required=True,
                        help="roadmap runs and their goals, separated by ; or +, e.g. 'B18_carry:0,7,14+B3_carry:4,11,16'")
    stage2.add_argument("--target-mm", type=float, default=10.0, help="roadmap tolerance counted as a success")
    pick = commands.add_parser("pick", help="write one carry pose per class (provisional unless chosen)")
    pick.add_argument("--root", default=None)
    pick.add_argument("--tag", required=True)
    pick.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                    "asset", "carry_poses.json"))
    pick.add_argument("--choose", default="", help="override picks, e.g. 'horizontal=horizontal_2,back=back_0'")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run one subcommand."""
    args = build_parser().parse_args(argv)
    root = args.root or os.environ.get("HUSKY_BENCH_ROOT")
    if not root:
        raise SystemExit("No benchmark data root. Pass --root, or set HUSKY_BENCH_ROOT.")
    root = os.path.abspath(root)
    if args.command == "stage1":
        cmd_stage1(root, args)
    elif args.command == "starts":
        cmd_starts(root, args)
    elif args.command == "stage2":
        cmd_stage2(root, args)
    elif args.command == "pick":
        cmd_pick(root, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
