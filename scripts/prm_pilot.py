"""Command line for the holding-constraint roadmap pilot.

    build   sample a roadmap for one bar and measure its edges
    query   connect a goal and a choice of starts to a built roadmap and search it
    refine  turn a saved query's routes into paths that hold the bar exactly
    track   track every roadmap edge within a score once, for later refinements (overnight work)
    export  copy the roadmap page next to a run, to open it in the browser
    info    print a built run's numbers and its queries

Examples (PowerShell, from the repository root, with HUSKY_BENCH_ROOT set):

    .venv\\Scripts\\python.exe scripts\\prm_pilot.py build --name smoke_prm --library smoke --bar B3 --goals 0 --nodes 200 --workers 2
    .venv\\Scripts\\python.exe scripts\\prm_pilot.py query --name smoke_prm --goal 0 --starts free
    .venv\\Scripts\\python.exe scripts\\prm_pilot.py export --name smoke_prm

Then open http://localhost:8000/prm/smoke_prm/dashboard/ with a web server
running on the data root (``python -m http.server --directory <root> 8000``).

This script and everything it runs are separate from ``benchmark_cdfm.py``:
results go to ``<root>/prm/``, never to ``<root>/runs/``, so a roadmap run and a
planner batch can run side by side.
"""
import argparse
import os
import sys
from typing import Dict, List, Optional

# The package sits one level up from scripts/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from husky_assembly_tamp.motion_planner.holding_prm.parallel import default_workers  # noqa: E402
from husky_assembly_tamp.motion_planner.holding_prm.scenes import problem_source_key  # noqa: E402
from husky_assembly_tamp.motion_planner.holding_prm.store import read_json  # noqa: E402
from husky_assembly_tamp.motion_planner.holding_prm.tracking import TrackSettings  # noqa: E402
from husky_assembly_tamp.motion_planner.holding_prm.pilot import (  # noqa: E402
    BuildSettings,
    QuerySettings,
    build_roadmap,
    export_dashboard,
    print_info,
    query_roadmap,
)
from husky_assembly_tamp.motion_planner.holding_prm.refine import (  # noqa: E402
    RefineSettings,
    pretrack_roadmap,
    refine_query,
)


def resolve_root(value: Optional[str]) -> str:
    """The benchmark data root, from ``--root`` or ``HUSKY_BENCH_ROOT``.

    Args:
        value (Optional[str]): the flag's value.

    Returns:
        str: the folder.

    Raises:
        SystemExit: when neither is set (guessing would scatter results).
    """
    root = value or os.environ.get("HUSKY_BENCH_ROOT")
    if not root:
        raise SystemExit("No benchmark data root. Pass --root, or set HUSKY_BENCH_ROOT.")
    return os.path.abspath(root)


def resolve_source_key(root: str, value: Optional[str]) -> str:
    """Which trimmed-scene cache to use (the same rule as the benchmark's CLI).

    Args:
        root (str): the data root.
        value (Optional[str]): ``--source-key``, if given.

    Returns:
        str: the key.

    Raises:
        SystemExit: when there is no cache, or several and none was named.
    """
    if value:
        return value
    cells = os.path.join(root, "cells")
    keys = sorted(os.listdir(cells)) if os.path.isdir(cells) else []
    if not keys:
        raise SystemExit(f"No trimmed scenes under {cells}; run the benchmark's build-cells stage first.")
    if len(keys) > 1:
        raise SystemExit(f"Several trimmed-scene caches exist ({', '.join(keys)}); pass --source-key.")
    return keys[0]


def parse_goals(value: str) -> Optional[List[int]]:
    """``all`` -> None, ``2,4`` -> [2, 4]."""
    if value == "all":
        return None
    return [int(part) for part in value.split(",") if part.strip()]


def parse_mix(value: str) -> Dict[str, float]:
    """``anchor_box=0.25,anchor_line=0.75`` -> a dict of weights."""
    mix = {}
    for part in value.split(","):
        name, _, weight = part.partition("=")
        mix[name.strip()] = float(weight)
    return mix


def parse_floats(value: str) -> List[float]:
    """``2,5,10`` -> [2.0, 5.0, 10.0]."""
    return [float(part) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    """The argument parser with its four subcommands."""
    defaults = BuildSettings(run_name="", library="", bar_id="", source_key="")
    query_defaults = QuerySettings(run_name="", goal_index=0)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--root", default=None, help="benchmark data root (or HUSKY_BENCH_ROOT)")
        sub.add_argument("--name", required=True, help="run name, a folder under <root>/prm")

    build = commands.add_parser("build", help="sample and measure a roadmap")
    common(build)
    build.add_argument("--library", default="", help="problem library, e.g. cdfm_v1 or smoke")
    build.add_argument("--problem-dir", default="",
                       help="build from an exported problem folder instead of a library: anchors on the "
                            "bar's planned goal and on copies of it for a shifted base (needs build-cells)")
    build.add_argument("--base-error-m", type=float, default=defaults.base_error_m,
                       help="with --problem-dir: how far the base may park from its planned place")
    build.add_argument("--base-error-deg", type=float, default=defaults.base_error_deg,
                       help="with --problem-dir: how far the base may be turned")
    build.add_argument("--base-samples", type=int, default=defaults.base_samples,
                       help="with --problem-dir: how many shifted bases to anchor on")
    build.add_argument("--bar", required=True, help="bar id, e.g. B18")
    build.add_argument("--source-key", default=None, help="trimmed-scene cache (only if several exist)")
    build.add_argument("--goals", default="all", help="goal indices whose pairs become anchors, or all")
    build.add_argument("--anchor-starts", type=int, default=defaults.anchor_starts,
                       help="library starts per goal used as anchors")
    build.add_argument("--anchor-turns", choices=("both", "given", "home"), default=defaults.anchor_turns)
    build.add_argument("--nodes", type=int, default=defaults.nodes, help="sampled nodes to keep")
    build.add_argument("--k", type=int, default=defaults.k, help="neighbours per node")
    build.add_argument("--radius-deg", type=float, default=defaults.radius_deg,
                       help="largest joint move of an edge")
    build.add_argument("--collision-step-deg", type=float, default=defaults.collision_step_deg)
    build.add_argument("--viol-step-deg", type=float, default=defaults.viol_step_deg)
    build.add_argument("--rot-deg-per-mm", type=float, default=defaults.rot_deg_per_mm,
                       help="degrees of rotation error that count as one mm")
    build.add_argument("--box-margin-deg", type=float, default=defaults.box_margin_deg)
    build.add_argument("--gauss-sigma-deg", type=float, default=defaults.gauss_sigma_deg)
    build.add_argument("--line-sigma-deg", type=float, default=defaults.line_sigma_deg)
    build.add_argument("--mode-mix", default=",".join(f"{k}={v}" for k, v in defaults.mode_mix.items()),
                       help="weights of the sampling modes")
    build.add_argument("--max-branches", type=int, default=defaults.max_branches,
                       help="other-arm branches kept per draw, nearest first (0 = all)")
    build.add_argument("--seed", type=int, default=defaults.seed)
    build.add_argument("--workers", type=int, default=default_workers())
    build.add_argument("--lazy", action="store_true", help="skip collision checks of roadmap edges")
    build.add_argument("--starts-file", default="", help="also anchor on these start poses (e.g. carry candidates)")

    query = commands.add_parser("query", help="search a built roadmap for one goal")
    common(query)
    query.add_argument("--goal", type=int, required=True, help="goal index (library numbering)")
    query.add_argument("--starts", default=query_defaults.starts,
                       help="all, a class list (free,blocked,broken), or start indices (0,12)")
    query.add_argument("--max-starts", type=int, default=None, help="at most this many starts (spread)")
    query.add_argument("--k-endpoint", type=int, default=query_defaults.k_endpoint)
    query.add_argument("--radius-endpoint-deg", type=float, default=query_defaults.radius_endpoint_deg)
    query.add_argument("--tolerances-mm", default=",".join(str(t) for t in query_defaults.tolerances_mm))
    query.add_argument("--w-hold", type=float, default=query_defaults.w_hold)
    query.add_argument("--endpoint-check", choices=("lazy", "eager"), default=query_defaults.endpoint_check)
    query.add_argument("--workers", type=int, default=default_workers())
    query.add_argument("--tag", default="", help="query name (default from goal and starts)")
    query.add_argument("--starts-file", default="",
                       help="take the start poses from this file instead of the library (ignores --starts)")

    refine_defaults = RefineSettings(run_name="")
    refine = commands.add_parser("refine", help="track a saved query's routes exactly")
    common(refine)
    refine.add_argument("--query", default=refine_defaults.query_tag, help="the saved query's tag")
    refine.add_argument("--tag", default="", help="this refinement's name (default from the settings)")
    refine.add_argument("--tolerances-mm", default=",".join(str(t) for t in refine_defaults.tolerances_mm),
                        help="roadmap tolerances to report exact routes for")
    refine.add_argument("--max-gap-mm", type=float, default=refine_defaults.max_gap_mm,
                        help="leave out starts whose roadmap gap is above this")
    refine.add_argument("--store-paths-mm", type=float, default=refine_defaults.store_paths_mm,
                        help="store the exact path of every start whose exact gap is at most this")
    refine.add_argument("--knot-step-deg", type=float, default=refine_defaults.knot_step_deg,
                        help="knot spacing along each edge's joint line (0 = end nodes only)")
    refine.add_argument("--position-res-mm", type=float, default=refine_defaults.position_res_m * 1000.0,
                        help="longest bar move between waypoints")
    refine.add_argument("--rotation-res-rad", type=float, default=refine_defaults.rotation_res_rad,
                        help="largest bar turn between waypoints")
    refine.add_argument("--lead", choices=("mid", "left", "right"), default=refine_defaults.lead,
                        help="which bar pose a knot aims for")
    refine.add_argument("--edge-sample", type=int, default=refine_defaults.edge_sample,
                        help="also track this many random edges, for statistics")
    refine.add_argument("--tube-rounds", type=int, default=refine_defaults.tube_rounds,
                        help="rounds of draws around routes still above the target (0 = none)")
    refine.add_argument("--tube-target-mm", type=float, default=refine_defaults.tube_target_mm)
    refine.add_argument("--tube-max-starts", type=int, default=refine_defaults.tube_max_starts)
    refine.add_argument("--tube-draws-per-start", type=int, default=refine_defaults.tube_draws_per_start)
    refine.add_argument("--tube-sigma-deg", type=float, default=refine_defaults.tube_sigma_deg)
    refine.add_argument("--tube-k", type=int, default=refine_defaults.tube_k)
    refine.add_argument("--tube-radius-deg", type=float, default=refine_defaults.tube_radius_deg)
    refine.add_argument("--tube-patience", type=int, default=refine_defaults.tube_patience)
    refine.add_argument("--seed", type=int, default=refine_defaults.seed)
    refine.add_argument("--workers", type=int, default=default_workers())
    refine.add_argument("--pretracked", default="", help="roadmap-edge verdicts from 'track' (overnight work)")

    track = commands.add_parser("track", help="track every roadmap edge within a score once (overnight work)")
    common(track)
    track.add_argument("--max-score-mm", type=float, default=10.0, help="track edges whose score is at most this")
    track.add_argument("--knot-step-deg", type=float, default=refine_defaults.knot_step_deg)
    track.add_argument("--position-res-mm", type=float, default=refine_defaults.position_res_m * 1000.0)
    track.add_argument("--rotation-res-rad", type=float, default=refine_defaults.rotation_res_rad)
    track.add_argument("--lead", choices=("mid", "left", "right"), default=refine_defaults.lead)
    track.add_argument("--workers", type=int, default=default_workers())

    export = commands.add_parser("export", help="copy the roadmap page into the run; also writes the robot's 3D "
                                                 "file and the scene files of saved online plans when missing, "
                                                 "and lists the run in prm/dashboards.json (the page's run dropdown)")
    common(export)
    info = commands.add_parser("info", help="print a run's numbers")
    common(info)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run one subcommand.

    Args:
        argv (Optional[List[str]]): arguments (default: the command line).

    Returns:
        int: the exit code.
    """
    args = build_parser().parse_args(argv)
    root = resolve_root(args.root)
    if args.command == "build":
        if bool(args.library) == bool(args.problem_dir):
            raise SystemExit("Give exactly one of --library (benchmark library) or --problem-dir (exported problem).")
        # An export has its own trimmed-scene cache, named after the folder; a
        # library records which cache it was built from.
        if args.source_key:
            source_key = args.source_key
        elif args.problem_dir:
            source_key = problem_source_key(args.problem_dir)
        else:
            index = os.path.join(root, "library", args.library, "index.json")
            source_key = (read_json(index).get("source_key") if os.path.isfile(index) else None) \
                or resolve_source_key(root, None)
        build_roadmap(root, BuildSettings(
            run_name=args.name, library=args.library, bar_id=args.bar,
            problem_dir=os.path.abspath(args.problem_dir) if args.problem_dir else "",
            base_error_m=args.base_error_m, base_error_deg=args.base_error_deg, base_samples=args.base_samples,
            source_key=source_key, goals=parse_goals(args.goals),
            anchor_starts=args.anchor_starts, anchor_turns=args.anchor_turns, nodes=args.nodes, k=args.k,
            radius_deg=args.radius_deg, collision_step_deg=args.collision_step_deg,
            viol_step_deg=args.viol_step_deg, rot_deg_per_mm=args.rot_deg_per_mm,
            box_margin_deg=args.box_margin_deg, gauss_sigma_deg=args.gauss_sigma_deg,
            line_sigma_deg=args.line_sigma_deg, mode_mix=parse_mix(args.mode_mix),
            max_branches=args.max_branches, seed=args.seed, workers=args.workers, lazy=args.lazy,
            starts_file=args.starts_file,
        ))
    elif args.command == "query":
        query_roadmap(root, QuerySettings(
            run_name=args.name, goal_index=args.goal, starts=args.starts, max_starts=args.max_starts,
            k_endpoint=args.k_endpoint, radius_endpoint_deg=args.radius_endpoint_deg,
            tolerances_mm=parse_floats(args.tolerances_mm), w_hold=args.w_hold,
            endpoint_check=args.endpoint_check, workers=args.workers, tag=args.tag,
            starts_file=args.starts_file,
        ))
    elif args.command == "refine":
        refine_query(root, RefineSettings(
            run_name=args.name, query_tag=args.query, tag=args.tag,
            tolerances_mm=parse_floats(args.tolerances_mm), max_gap_mm=args.max_gap_mm,
            store_paths_mm=args.store_paths_mm, knot_step_deg=args.knot_step_deg,
            position_res_m=args.position_res_mm / 1000.0, rotation_res_rad=args.rotation_res_rad,
            lead=args.lead, edge_sample=args.edge_sample, tube_rounds=args.tube_rounds,
            tube_target_mm=args.tube_target_mm, tube_max_starts=args.tube_max_starts,
            tube_draws_per_start=args.tube_draws_per_start, tube_sigma_deg=args.tube_sigma_deg,
            tube_k=args.tube_k, tube_radius_deg=args.tube_radius_deg, tube_patience=args.tube_patience,
            seed=args.seed, workers=args.workers, pretracked=args.pretracked,
        ))
    elif args.command == "track":
        pretrack_roadmap(root, args.name, args.max_score_mm, TrackSettings(
            knot_step_deg=args.knot_step_deg, position_res_m=args.position_res_mm / 1000.0,
            rotation_res_rad=args.rotation_res_rad, lead=args.lead), args.workers)
    elif args.command == "export":
        out_dir = export_dashboard(root, args.name)
        print(f"[prm export] {out_dir}")
        print(f"  open http://localhost:8000/prm/{args.name}/dashboard/ "
              f"(with a web server on {root})")
    elif args.command == "info":
        print_info(root, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
