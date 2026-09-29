"""Command line for the CDFM planner benchmark.

Run the stages in order; each reads what the last one wrote::

    build-cells    trim the exported robot cell into one small scene per bar
    build-library  sample goal poses and survey the starts they can be
                   reached from
    plan           say how much work a batch would be, and how long
    run            plan every start-to-goal transfer, several seeds each
    summarize      aggregate the runs into tables
    export-dashboard  write the files the web dashboard reads
    replay         re-plan one start with its search trees saved
    measure        time the inner loops, to size a batch before committing to it

Every stage takes ``--root``, the folder all benchmark data lives in. Set
``HUSKY_BENCH_ROOT`` once and it can be left off.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from husky_assembly_tamp.benchmark.sweep import DEFAULT_SWEEP_STEP_M


DEFAULT_LIBRARY_NAME = "cdfm_v1"
DEFAULT_LIBRARY_SEED = 20260923


def resolve_root(value: Optional[str]) -> str:
    """Return the benchmark data root, from the flag or the environment.

    Args:
        value (Optional[str]): whatever ``--root`` was given.

    Returns:
        str: the folder path.

    Raises:
        SystemExit: when neither the flag nor ``HUSKY_BENCH_ROOT`` is set.
            Guessing a location would scatter large result folders somewhere
            unexpected, so this asks instead.
    """
    root = value or os.environ.get("HUSKY_BENCH_ROOT")
    if not root:
        raise SystemExit(
            "No benchmark data root. Pass --root, or set HUSKY_BENCH_ROOT to "
            "the folder benchmark data should live in."
        )
    return os.path.abspath(root)


def resolve_source_key(root: str, value: Optional[str]) -> str:
    """Return which trimmed-scene cache to use.

    Args:
        root (str): the benchmark data root.
        value (Optional[str]): an explicit key, or None to pick automatically.

    Returns:
        str: the cache key.

    Raises:
        SystemExit: when no cache exists, or when several do and none was
            named -- picking one arbitrarily would silently benchmark the
            wrong export.
    """
    if value:
        return value
    cells_dir = os.path.join(root, "cells")
    keys = sorted(os.listdir(cells_dir)) if os.path.isdir(cells_dir) else []
    if not keys:
        raise SystemExit(
            f"No trimmed scenes under {cells_dir}. Run 'build-cells' first."
        )
    if len(keys) > 1:
        raise SystemExit(
            f"Several trimmed-scene caches exist ({', '.join(keys)}). Say which "
            "one with --source-key."
        )
    return keys[0]


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the flags every stage understands.

    Args:
        parser (argparse.ArgumentParser): the sub-parser to add them to.
    """
    parser.add_argument("--root", default=None,
                        help="folder all benchmark data lives in "
                             "(or set HUSKY_BENCH_ROOT)")
    parser.add_argument("--source-key", default=None,
                        help="which trimmed-scene cache to use; only needed "
                             "when more than one exists")
    parser.add_argument("--bars", default=None,
                        help="comma-separated bar ids; all of them by default")


def parse_bars(value: Optional[str]) -> Optional[List[str]]:
    """Turn a ``--bars`` value into a list.

    Args:
        value (Optional[str]): the comma-separated value, or None.

    Returns:
        Optional[List[str]]: the bar ids, or None for "all of them".
    """
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def known_bars(root: str, source_key: str) -> List[str]:
    """List the bars that have a trimmed scene.

    Args:
        root (str): the benchmark data root.
        source_key (str): the cache key.

    Returns:
        List[str]: bar ids, in assembly order.
    """
    folder = os.path.join(root, "cells", source_key)
    if not os.path.isdir(folder):
        raise SystemExit(f"No trimmed scenes at {folder}. Run 'build-cells' first.")
    bars = [name for name in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, name, "problem.json"))]
    return sorted(bars, key=lambda b: int("".join(c for c in b if c.isdigit()) or 0))


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def cmd_build_cells(args: argparse.Namespace) -> int:
    """Trim an exported design study into per-bar benchmark scenes."""
    from husky_assembly_tamp.benchmark.build_cells import build_cells

    root = resolve_root(args.root)
    out = build_cells(args.problem_dir, root, bar_ids=parse_bars(args.bars))
    print(f"\n[build-cells] {len(out['bars'])} bar(s) cached under "
          f"cells/{out['source_key']}")
    print(f"[build-cells] use --source-key {out['source_key']} for later stages")
    return 0


def cmd_build_library(args: argparse.Namespace) -> int:
    """Sample goal poses and survey the starts they can be reached from."""
    from husky_assembly_tamp.benchmark.library import build_library, write_index

    root = resolve_root(args.root)
    source_key = resolve_source_key(root, args.source_key)
    bars = parse_bars(args.bars) or known_bars(root, source_key)

    settings = {
        "n_goals": args.goals,
        "sweep_step_m": args.sweep_step,
        "library_seed": args.seed,
    }
    print(f"[build-library] {len(bars)} bar(s), {args.goals} goal(s) each, "
          f"start grid {args.sweep_step} m, {args.workers} worker(s)", flush=True)

    t0 = time.time()
    results = build_library(
        root, args.library, bars,
        source_key=source_key,
        n_goals=args.goals,
        library_seed=args.seed,
        sweep_step_m=args.sweep_step,
        n_workers=args.workers,
        force=args.force,
    )
    failed = [result["bar_id"] for result in results if not result["ok"]]
    built = [bar_id for bar_id in bars if bar_id not in failed]
    if failed:
        print(f"[build-library] {len(failed)} bar(s) failed: {', '.join(failed)}")
    if not built:
        raise SystemExit("No bar was built, so there is no library to index.")

    write_index(root, args.library, source_key=source_key,
                library_seed=args.seed, bar_ids=built, settings=settings)
    print(f"[build-library] library {args.library!r} written in "
          f"{(time.time() - t0) / 60:.1f} min")
    return 0


def collect_tasks(root: str, library_name: str, bars: Sequence[str],
                  *, seeds: Sequence[int],
                  max_starts_per_goal: Optional[int],
                  max_goals: Optional[int]):
    """Build the list of planning tasks a batch should work through.

    Args:
        root (str): the benchmark data root.
        library_name (str): which library to plan from.
        bars (Sequence[str]): which bars.
        seeds (Sequence[int]): the random seeds to run each pair with.
        max_starts_per_goal (Optional[int]): further limit the starts kept in
            the library.
        max_goals (Optional[int]): only the first this many goals per bar.

    Returns:
        Tuple[List[RunTask], Dict]: the tasks and, for each, how its start
        ranked in the survey (used to order the batch).
    """
    from husky_assembly_tamp.benchmark.cdfm import RunTask
    from husky_assembly_tamp.benchmark.library import load_bar_library, select_starts

    tasks: List[RunTask] = []
    start_ranks: Dict[Any, int] = {}
    for bar_id in bars:
        entry = load_bar_library(root, library_name, bar_id)
        goals = entry["goals"]
        if max_goals is not None:
            goals = goals[:max_goals]
        for goal in goals:
            starts = select_starts(goal["starts"], max_starts_per_goal)
            for start in starts:
                key = (bar_id, goal["goal_index"], start["start_index"])
                start_ranks[key] = start["rank"]
                for seed in seeds:
                    tasks.append(RunTask(
                        bar_id=bar_id,
                        goal_index=goal["goal_index"],
                        start_index=start["start_index"],
                        seed=seed,
                        goal_conf=goal["primary_conf"],
                        start_conf=start["conf"],
                    ))
    return tasks, start_ranks


def cmd_plan(args: argparse.Namespace) -> int:
    """Report how big a batch would be, without running it."""
    from husky_assembly_tamp.benchmark.library import load_index

    root = resolve_root(args.root)
    index = load_index(root, args.library)
    bars = parse_bars(args.bars) or [bar["bar_id"] for bar in index["bars"]]
    seeds = list(range(args.seeds))

    tasks, _ranks = collect_tasks(
        root, args.library, bars, seeds=seeds,
        max_starts_per_goal=args.max_starts_per_goal, max_goals=args.max_goals,
    )
    per_run_s = args.assume_run_s
    total_h = len(tasks) * per_run_s / 3600.0
    print(f"[plan] {len(bars)} bar(s), {args.seeds} seed(s)")
    print(f"[plan] {len(tasks)} run(s)")
    print(f"[plan] at {per_run_s:.0f} s per run: {total_h:.1f} h serial, "
          f"{total_h / args.workers:.1f} h on {args.workers} workers")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Plan every start-to-goal transfer in the library."""
    from husky_assembly_tamp.benchmark.library import load_index
    from husky_assembly_tamp.benchmark.runner import (
        BatchRunner, order_tasks, pending_tasks,
    )

    root = resolve_root(args.root)
    source_key = resolve_source_key(root, args.source_key)
    index = load_index(root, args.library)
    bars = parse_bars(args.bars) or [bar["bar_id"] for bar in index["bars"]]
    seeds = list(range(args.seeds))

    run_dir = os.path.join(root, "runs", args.name)
    runs_dir = os.path.join(run_dir, "runs")
    profile_dir = os.path.join(run_dir, "profiles") if args.profile else None
    os.makedirs(runs_dir, exist_ok=True)

    tasks, start_ranks = collect_tasks(
        root, args.library, bars, seeds=seeds,
        max_starts_per_goal=args.max_starts_per_goal, max_goals=args.max_goals,
    )
    tasks = order_tasks(tasks, start_ranks)
    todo = pending_tasks(tasks, runs_dir)

    pass_overrides: Dict[str, Dict[str, float]] = {}
    if args.max_time_coarse is not None:
        pass_overrides["coarse"] = {"max_time": args.max_time_coarse}
    if args.max_time_fine is not None:
        pass_overrides["fine"] = {"max_time": args.max_time_fine}

    config = {
        "root": root,
        "source_key": source_key,
        "profile_dir": profile_dir,
        "enable_profiling": bool(args.profile),
        "store_paths": not args.no_paths,
        "tree_dir": os.path.join(run_dir, "trees") if args.store_trees else None,
        "pass_overrides": pass_overrides,
    }
    write_run_config(run_dir, args, index, config, n_tasks=len(tasks))

    print(f"[run] {len(tasks)} task(s), {len(todo)} still to do, "
          f"{args.workers} worker(s)", flush=True)
    if not todo:
        print("[run] nothing to do; every result is already on disk")
        return 0

    runner = BatchRunner(todo, runs_dir=runs_dir, config=config,
                         n_workers=args.workers)
    summary = runner.run()
    with open(os.path.join(run_dir, "batch_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return 0


def write_run_config(run_dir: str, args: argparse.Namespace, index: Dict[str, Any],
                     config: Dict[str, Any], *, n_tasks: int) -> None:
    """Record exactly what a batch was asked to do.

    Without this a result file is uninterpretable six months later: the
    planner changes, the library changes, and nothing in the result says which
    versions produced it.

    Args:
        run_dir (str): the batch's folder.
        args (argparse.Namespace): the command line it was started with.
        index (Dict[str, Any]): the library index it planned from.
        config (Dict[str, Any]): the settings handed to the workers.
        n_tasks (int): how many runs the batch covers.
    """
    from husky_assembly_tamp.benchmark.cdfm import PASS_SETTINGS
    from husky_assembly_tamp.benchmark.provenance import provenance

    record = {
        "name": args.name,
        "created": time.time(),
        "library": args.library,
        "library_seed": index.get("library_seed"),
        "library_settings": index.get("settings"),
        "source_key": config["source_key"],
        "n_tasks": n_tasks,
        "n_workers": args.workers,
        "seeds": list(range(args.seeds)),
        "pass_settings": PASS_SETTINGS,
        "pass_overrides": config["pass_overrides"],
        "profiling": config["enable_profiling"],
        "command": " ".join(sys.argv),
        "provenance": provenance(),
    }
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)


def cmd_summarize(args: argparse.Namespace) -> int:
    """Aggregate a batch's runs into tables."""
    from husky_assembly_tamp.benchmark.summarize import summarize_run

    root = resolve_root(args.root)
    run_dir = os.path.join(root, "runs", args.name)
    summary = summarize_run(run_dir)
    path = os.path.join(run_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[summarize] {summary['totals']['n_runs']} run(s) -> {path}")
    for row in summary["per_bar"]:
        print(f"  {row['bar_id']:5s} len {row['bar_length_m']:.2f} m  "
              f"runs {row['n_runs']:5d}  success {row['success_rate']:.2f}  "
              f"median {row['median_wall_s']:.1f} s")
    return 0


def cmd_export_dashboard(args: argparse.Namespace) -> int:
    """Write the files the web dashboard reads."""
    from husky_assembly_tamp.benchmark.export_dashboard import export_run

    root = resolve_root(args.root)
    export_run(root, args.name, library_name=args.library)
    print("[export] serve the dashboard with:")
    print("    python -m http.server --directory "
          f"{os.path.join(root, 'runs', args.name, 'dashboard')} 8000")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Re-plan chosen starts with their search trees saved.

    A batch run without ``--store-trees`` keeps only the size of each search
    tree, which is not enough to see why a search failed. This re-plans the
    starts in question with the batch's own seeds and settings, saves both
    trees for every pass, and attaches them to the batch's dashboard.

    A replay is not a byte-for-byte rerun. The search stops on a wall-clock
    limit, so the same seed can stop at a different point, and occasionally end
    the other way. Each replay records its own outcome, and the dashboard shows
    it beside the original's.
    """
    from husky_assembly_tamp.benchmark.cdfm import RunTask
    from husky_assembly_tamp.benchmark.export_dashboard import attach_trees
    from husky_assembly_tamp.benchmark.library import load_bar_library
    from husky_assembly_tamp.benchmark.runner import BatchRunner, pending_tasks

    root = resolve_root(args.root)
    run_dir = os.path.join(root, "runs", args.name)
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_path):
        raise SystemExit(f"No batch called {args.name!r} at {run_dir}.")
    with open(config_path, "r", encoding="utf-8") as handle:
        run_config = json.load(handle)

    entry = load_bar_library(root, run_config["library"], args.bar)
    goal = entry["goals"][args.goal]
    start = next((s for s in goal["starts"] if s["start_index"] == args.start), None)
    if start is None:
        raise SystemExit(
            f"{args.bar} goal {args.goal} has no workable start {args.start}.")
    seeds = (run_config.get("seeds") or list(range(10))) if args.seeds == "all" else [
        int(part) for part in args.seeds.split(",") if part.strip()]

    tasks = [RunTask(bar_id=args.bar, goal_index=args.goal,
                     start_index=args.start, seed=seed,
                     goal_conf=goal["primary_conf"], start_conf=start["conf"])
             for seed in seeds]
    replay_dir = os.path.join(run_dir, "replays")
    runs_dir = os.path.join(replay_dir, "runs")
    todo = tasks if args.force else pending_tasks(tasks, runs_dir)
    print(f"[replay] {args.bar} goal {args.goal} start {args.start}: "
          f"{len(tasks)} seed(s), {len(todo)} to plan", flush=True)

    if todo:
        config = {
            "root": root,
            "source_key": run_config["source_key"],
            "profile_dir": None,
            "enable_profiling": False,
            "store_paths": True,
            "tree_dir": os.path.join(replay_dir, "trees"),
            # The batch's own settings, so the replay asks the same question.
            "pass_overrides": run_config.get("pass_overrides") or {},
        }
        BatchRunner(todo, runs_dir=runs_dir, config=config,
                    n_workers=min(args.workers, len(todo))).run()

    attach_trees(root, args.name)
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    """Time the inner loops so a batch can be sized before it is started."""
    from husky_assembly_tamp.benchmark.measure import measure_costs

    root = resolve_root(args.root)
    source_key = resolve_source_key(root, args.source_key)
    bars = parse_bars(args.bars) or known_bars(root, source_key)[:1]
    measure_costs(root, source_key, bars[0], sweep_steps=args.sweep_steps,
                  n_goals=args.goals)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for every stage."""
    parser = argparse.ArgumentParser(
        prog="benchmark_cdfm",
        description="Benchmark the M1 constrained dual-arm transfer planner.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cells = subparsers.add_parser(
        "build-cells", help="trim an exported design study into per-bar scenes")
    add_common_arguments(cells)
    cells.add_argument("problem_dir",
                       help="the exported problem folder (holds RobotCell.json "
                            "and BarActions/)")
    cells.set_defaults(func=cmd_build_cells)

    library = subparsers.add_parser(
        "build-library", help="sample goals and survey their starts")
    add_common_arguments(library)
    library.add_argument("--library", default=DEFAULT_LIBRARY_NAME)
    library.add_argument("--goals", type=int, default=50,
                         help="goal poses per bar (default 50)")
    library.add_argument("--sweep-step", type=float, default=DEFAULT_SWEEP_STEP_M,
                         help="spacing of the start-offset grid in metres "
                              "(default %(default)s, production's own). Coarser "
                              "surveys fewer starts and costs proportionally less")
    library.add_argument("--seed", type=int, default=DEFAULT_LIBRARY_SEED)
    library.add_argument("--workers", type=int, default=1,
                         help="how many bars to survey at once (default 1). "
                              "Bars are independent, so this scales nearly "
                              "linearly up to the core count")
    library.add_argument("--force", action="store_true",
                         help="rebuild bars that already have a file; by "
                              "default they are skipped, so an interrupted "
                              "build can simply be run again")
    library.set_defaults(func=cmd_build_library)

    plan = subparsers.add_parser("plan", help="size a batch without running it")
    add_common_arguments(plan)
    plan.add_argument("--library", default=DEFAULT_LIBRARY_NAME)
    plan.add_argument("--seeds", type=int, default=10)
    plan.add_argument("--max-starts-per-goal", type=int, default=None,
                      help="plan from at most this many starts per goal, spread "
                           "across the carry anchors and difficulty classes "
                           "(default: every feasible start)")
    plan.add_argument("--max-goals", type=int, default=None)
    plan.add_argument("--workers", type=int, default=12)
    plan.add_argument("--assume-run-s", type=float, default=45.0,
                      help="seconds per run to assume (default %(default)s); "
                           "'measure' gives a better figure")
    plan.set_defaults(func=cmd_plan)

    run = subparsers.add_parser("run", help="plan every transfer in the library")
    add_common_arguments(run)
    run.add_argument("--name", required=True, help="a name for this batch")
    run.add_argument("--library", default=DEFAULT_LIBRARY_NAME)
    run.add_argument("--seeds", type=int, default=10)
    run.add_argument("--max-starts-per-goal", type=int, default=None,
                     help="plan from at most this many starts per goal, spread "
                          "across the carry anchors and difficulty classes "
                          "(default: every feasible start)")
    run.add_argument("--max-goals", type=int, default=None)
    run.add_argument("--workers", type=int, default=12)
    run.add_argument("--max-time-coarse", type=float, default=None,
                     help="override the coarse pass's time budget, seconds")
    run.add_argument("--max-time-fine", type=float, default=None,
                     help="override the fine pass's time budget, seconds")
    run.add_argument("--no-profile", dest="profile", action="store_false",
                     help="skip profiling (a few percent faster, no flame graphs)")
    run.add_argument("--no-paths", action="store_true",
                     help="do not store planned trajectories (much smaller results)")
    run.add_argument("--store-trees", action="store_true",
                     help="save both search trees of every pass, solved or not, "
                          "for the dashboard to draw (tens of KB per pass)")
    run.set_defaults(func=cmd_run, profile=True)

    summarize = subparsers.add_parser("summarize", help="aggregate a batch")
    add_common_arguments(summarize)
    summarize.add_argument("--name", required=True)
    summarize.set_defaults(func=cmd_summarize)

    export = subparsers.add_parser(
        "export-dashboard", help="write the files the web dashboard reads")
    add_common_arguments(export)
    export.add_argument("--name", required=True)
    export.add_argument("--library", default=DEFAULT_LIBRARY_NAME)
    export.set_defaults(func=cmd_export_dashboard)

    replay = subparsers.add_parser(
        "replay", help="re-plan one start with its search trees saved")
    add_common_arguments(replay)
    replay.add_argument("--name", required=True, help="the batch to replay from")
    replay.add_argument("--bar", required=True)
    replay.add_argument("--goal", type=int, required=True)
    replay.add_argument("--start", type=int, required=True)
    replay.add_argument("--seeds", default="all",
                        help="comma-separated seeds, or all of the batch's (default)")
    replay.add_argument("--workers", type=int, default=10)
    replay.add_argument("--force", action="store_true",
                        help="re-plan seeds that already have a replay")
    replay.set_defaults(func=cmd_replay)

    measure = subparsers.add_parser(
        "measure", help="time the inner loops to size a batch")
    add_common_arguments(measure)
    measure.add_argument("--goals", type=int, default=1,
                         help="how many goals to survey (default 1)")
    measure.add_argument("--sweep-steps", default="0.3,0.2",
                         help="comma-separated grid steps to time "
                              "(default %(default)s)")
    measure.set_defaults(func=cmd_measure)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse the command line and run the chosen stage.

    Args:
        argv (Optional[Sequence[str]]): arguments, or None for ``sys.argv``.

    Returns:
        int: the process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
