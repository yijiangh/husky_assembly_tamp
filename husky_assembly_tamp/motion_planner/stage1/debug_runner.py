"""Stage 1 debug runner with batch analysis, plots, and report generation."""

from __future__ import annotations

import argparse
import cProfile
import csv
import io
import json
import os
import pstats
import time
from typing import Dict, List, Optional, Tuple

from husky_assembly_tamp.motion_planner.stage1.minimal_rrt import (
    build_default_paths,
    run_stage1_trial,
    run_visualization_loop,
    teardown_stage1_scene,
)
from husky_assembly_tamp.utils.util import setup_logger


logger = setup_logger("stage1_debug_runner")

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
PROFILE_TIME_KEYS = [
    "collision_check_time_s",
    "feature_time_s",
    "sample_time_s",
    "nearest_time_s",
    "goal_test_time_s",
    "extend_direct_time_s",
    "extend_tree_time_s",
    "extend_goal_time_s",
]
PROFILE_COUNT_KEYS = [
    "attempts",
    "iterations",
    "nodes_created",
    "poses_checked",
    "collision_hits",
]


def classify_result(success: bool, planner_profile: Dict, enable_collision: bool) -> str:
    if success:
        return "success"
    outcome = planner_profile.get("outcome")
    if outcome in {"start_in_collision", "goal_in_collision"}:
        return "collision_failure"
    if enable_collision and int(planner_profile.get("collision_hits", 0)) > 0:
        return "collision_failure"
    return "task_space_failure"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception as e:
        logger.warning(f"Skipping plots (matplotlib unavailable): {e}")
        return None


def plot_tree_3d(tree_data: Dict, out_path: str, title: str) -> bool:
    plt = maybe_import_matplotlib()
    if plt is None:
        return False
    tree = tree_data.get("tree1", {})
    points = tree.get("points", [])
    edges = tree.get("edges", [])
    if not points:
        return False

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    ax.scatter(xs, ys, zs, s=8, c="#777777", alpha=0.65, label="tree nodes")
    for parent_idx, child_idx in edges:
        p1 = points[parent_idx]
        p2 = points[child_idx]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="#d9534f", alpha=0.35, linewidth=0.8)

    start_pose = tree_data.get("start_pose")
    goal_pose = tree_data.get("goal_pose")
    if start_pose is not None:
        ax.scatter([start_pose[0]], [start_pose[1]], [start_pose[2]], s=70, c="#5cb85c", label="start")
    if goal_pose is not None:
        ax.scatter([goal_pose[0]], [goal_pose[1]], [goal_pose[2]], s=70, c="#337ab7", label="goal")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def plot_failure_distribution(counts: Dict[str, int], out_path: str) -> bool:
    plt = maybe_import_matplotlib()
    if plt is None:
        return False
    labels = ["task_space_failure", "collision_failure", "success"]
    vals = [counts.get(k, 0) for k in labels]
    colors = ["#d9534f", "#5bc0de", "#5cb85c"]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, vals, color=colors)
    plt.ylabel("Count")
    plt.title("Stage 1 Failure Distribution Across Seeds")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def plot_success_rate(success_rate: float, out_path: str) -> bool:
    plt = maybe_import_matplotlib()
    if plt is None:
        return False
    plt.figure(figsize=(4, 4))
    plt.bar(["Stage 1"], [success_rate], color=["#337ab7"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Success Rate")
    plt.title("Stage 1 Success Rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def plot_runtime_by_seed(records: List[Dict], out_path: str) -> bool:
    plt = maybe_import_matplotlib()
    if plt is None or not records:
        return False
    seeds = [r["seed"] for r in records]
    runtimes = [r["runtime_s"] for r in records]
    categories = [r["category"] for r in records]
    color_map = {
        "success": "#5cb85c",
        "task_space_failure": "#d9534f",
        "collision_failure": "#5bc0de",
    }
    colors = [color_map.get(c, "#777777") for c in categories]
    plt.figure(figsize=(8, 4))
    plt.bar(seeds, runtimes, color=colors)
    plt.xlabel("Seed")
    plt.ylabel("Runtime (s)")
    plt.title("Stage 1 Runtime by Seed")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def plot_planner_breakdown(mean_profile: Dict[str, float], out_path: str) -> bool:
    plt = maybe_import_matplotlib()
    if plt is None:
        return False
    keys = [k for k in PROFILE_TIME_KEYS if mean_profile.get(k, 0.0) > 0.0]
    if not keys:
        return False
    labels = [k.replace("_time_s", "") for k in keys]
    vals = [mean_profile[k] for k in keys]
    plt.figure(figsize=(9, 4))
    plt.bar(labels, vals, color="#777777")
    plt.ylabel("Average time per run (s)")
    plt.title("Stage 1 Planner Breakdown")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def write_profile_text(stats_path: str, out_path: str, top_n: int) -> None:
    stream = io.StringIO()
    stats = pstats.Stats(stats_path, stream=stream)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(top_n)
    with open(out_path, "w") as f:
        f.write(stream.getvalue())


def summarize_profile_means(records: List[Dict]) -> Dict[str, float]:
    if not records:
        return {}
    means: Dict[str, float] = {}
    for key in PROFILE_TIME_KEYS + PROFILE_COUNT_KEYS:
        vals = [float(r["planner_profile"].get(key, 0.0)) for r in records]
        means[key] = sum(vals) / max(1, len(vals))
    return means


def write_markdown_report(
    report_path: str,
    timestamp: str,
    args,
    summary: Dict,
    artifacts: Dict[str, Optional[str]],
) -> None:
    lines: List[str] = []
    lines.append(f"# Stage 1 Debugging Report ({timestamp})")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This report summarizes results from:")
    lines.append("")
    for key in [
        "json",
        "csv",
        "failure_distribution",
        "success_rate",
        "runtime_by_seed",
        "tree_plot",
        "planner_breakdown",
        "profile_txt",
    ]:
        path = artifacts.get(key)
        if path:
            lines.append(f"- `{os.path.basename(path)}`")
    lines.append("")
    lines.append("Run setup:")
    lines.append("")
    lines.append(f"- Trials: `{args.analysis_trials}` seeds (`{args.analysis_seed_start}..{args.analysis_seed_start + args.analysis_trials - 1}`)")
    lines.append(f"- Per-attempt max time: `{args.max_time}s`")
    lines.append(f"- Dist metric: `{args.dist_metric}`")
    lines.append(f"- Position resolution: `{args.position_res} m`")
    lines.append(f"- Rotation resolution: `{args.rotation_res} rad`")
    lines.append(f"- Floating collision: `{'on' if args.floating_collision else 'off'}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1) Workspace Tree Visualization")
    lines.append("")
    if artifacts.get("tree_plot"):
        lines.append(f"### Stage 1 (seed {args.analysis_seed_start})")
        lines.append(f"![Stage 1 Tree](./{os.path.basename(artifacts['tree_plot'])})")
        lines.append("")
        lines.append("Observation:")
        lines.append("")
        lines.append("- The tree image shows the task-space exploration footprint used by the single-tree Stage 1 RRT.")
        lines.append("- This is the quickest way to see whether the sampler is exploring broadly or repeatedly getting trapped near the start or obstacle boundary.")
    else:
        lines.append("Tree plot was not generated.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2) Failure Distribution Analysis")
    lines.append("")
    if artifacts.get("failure_distribution"):
        lines.append("### Distribution plot")
        lines.append(f"![Failure Distribution](./{os.path.basename(artifacts['failure_distribution'])})")
        lines.append("")
    lines.append("From `summary.counts`:")
    lines.append("")
    for key in ["task_space_failure", "collision_failure", "success"]:
        count = summary["counts"][key]
        pct = 100.0 * count / max(1, summary["trials"])
        lines.append(f"- `{key}`: **{count} / {summary['trials']}** ({pct:.0f}%)")
    lines.append("")
    dominant = max(summary["counts"], key=summary["counts"].get)
    lines.append("### Bottleneck conclusion")
    lines.append("")
    lines.append(f"Dominant observed outcome in this run is **{dominant}**.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3) Runtime and Bottleneck Breakdown")
    lines.append("")
    if artifacts.get("success_rate"):
        lines.append("### Success-rate plot")
        lines.append(f"![Stage 1 Success Rate](./{os.path.basename(artifacts['success_rate'])})")
        lines.append("")
    if artifacts.get("runtime_by_seed"):
        lines.append("### Runtime-by-seed plot")
        lines.append(f"![Runtime by Seed](./{os.path.basename(artifacts['runtime_by_seed'])})")
        lines.append("")
    if artifacts.get("planner_breakdown"):
        lines.append("### Planner breakdown plot")
        lines.append(f"![Planner Breakdown](./{os.path.basename(artifacts['planner_breakdown'])})")
        lines.append("")
    lines.append("From `summary`:")
    lines.append("")
    lines.append(f"- Stage 1 success rate: **{summary['success_rate']:.0%}**")
    lines.append(f"- Stage 1 avg runtime: **{summary['avg_runtime_s']:.3f} s**")
    lines.append(f"- Stage 1 avg iterations: **{summary['profile_means'].get('iterations', 0.0):.1f}**")
    lines.append(f"- Stage 1 avg nodes created: **{summary['profile_means'].get('nodes_created', 0.0):.1f}**")
    lines.append(f"- Stage 1 avg poses checked: **{summary['profile_means'].get('poses_checked', 0.0):.1f}**")
    lines.append("")
    if artifacts.get("profile_txt"):
        lines.append(f"Detailed `cProfile` summary: `{os.path.basename(artifacts['profile_txt'])}`")
        lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- The runtime plot shows whether failures correlate with long searches or early exits.")
    lines.append("- The planner breakdown plot shows which internal planner phases consume the most time on average.")
    lines.append("- The saved `cProfile` text report is the lower-level function-call view for deeper bottleneck inspection.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Final Answer to Debugging Goals")
    lines.append("")
    lines.append("1. **Workspace tree visualization**: Achieved. A Stage 1 tree image is generated for the first seed in the batch.")
    lines.append("2. **Failure distribution analysis**: Achieved. Successes and failures are categorized across seeds and visualized.")
    lines.append("3. **Runtime / bottleneck analysis**: Achieved. The runner emits aggregate planner timing, per-seed runtime plots, and `cProfile` output.")
    lines.append("")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def run_analysis(args) -> None:
    ensure_dir(args.analysis_outdir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    profile_seed = args.profile_seed if args.profile_seed is not None else args.analysis_seed_start

    records: List[Dict] = []
    counts = {
        "success": 0,
        "task_space_failure": 0,
        "collision_failure": 0,
    }
    tree_data_first: Optional[Dict] = None

    prof_path: Optional[str] = None
    prof_txt_path: Optional[str] = None
    for i in range(args.analysis_trials):
        seed = args.analysis_seed_start + i
        debug_tree_out = {} if seed == args.analysis_seed_start else None
        planner_profile: Dict = {}
        profiler = cProfile.Profile() if seed == profile_seed else None
        try:
            if profiler is not None:
                result = profiler.runcall(
                    run_stage1_trial,
                    grasp_json=args.grasp_json,
                    start_state_json=args.start_state,
                    end_state_json=args.end_state,
                    use_gui=False,
                    dist_metric=args.dist_metric,
                    goal_bias=args.goal_bias,
                    position_res=args.position_res,
                    rotation_res=args.rotation_res,
                    max_time=args.max_time,
                    max_iterations=args.max_iterations,
                    max_attempts=args.max_attempts,
                    random_seed=seed,
                    enable_collision=args.floating_collision,
                    lock_renderer_during_search=args.lock_renderer_during_search,
                    debug_tree_out=debug_tree_out,
                    planner_profile_out=planner_profile,
                )
            else:
                result = run_stage1_trial(
                    grasp_json=args.grasp_json,
                    start_state_json=args.start_state,
                    end_state_json=args.end_state,
                    use_gui=False,
                    dist_metric=args.dist_metric,
                    goal_bias=args.goal_bias,
                    position_res=args.position_res,
                    rotation_res=args.rotation_res,
                    max_time=args.max_time,
                    max_iterations=args.max_iterations,
                    max_attempts=args.max_attempts,
                    random_seed=seed,
                    enable_collision=args.floating_collision,
                    lock_renderer_during_search=args.lock_renderer_during_search,
                    debug_tree_out=debug_tree_out,
                    planner_profile_out=planner_profile,
                )
        finally:
            teardown_stage1_scene()

        if profiler is not None:
            prof_path = os.path.join(args.analysis_outdir, f"plan_profile_seed{seed}_{timestamp}.prof")
            prof_txt_path = os.path.join(args.analysis_outdir, f"plan_profile_seed{seed}_{timestamp}.txt")
            profiler.dump_stats(prof_path)
            write_profile_text(prof_path, prof_txt_path, args.profile_top_n)

        category = classify_result(result["success"], planner_profile, args.floating_collision)
        counts[category] += 1
        record = {
            "seed": seed,
            "success": int(result["success"]),
            "runtime_s": round(result["runtime_s"], 4),
            "category": category,
            "planner_profile": planner_profile,
        }
        for key in PROFILE_TIME_KEYS + PROFILE_COUNT_KEYS:
            record[key] = round(float(planner_profile.get(key, 0.0)), 6)
        records.append(record)
        if debug_tree_out is not None:
            tree_data_first = debug_tree_out

    csv_path = os.path.join(args.analysis_outdir, f"failure_analysis_{timestamp}.csv")
    json_path = os.path.join(args.analysis_outdir, f"failure_analysis_{timestamp}.json")
    report_path = os.path.join(args.analysis_outdir, f"debug_report_{timestamp}.md")

    with open(csv_path, "w", newline="") as f:
        fieldnames = ["seed", "success", "runtime_s", "category"] + PROFILE_TIME_KEYS + PROFILE_COUNT_KEYS
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k) for k in fieldnames})

    profile_means = summarize_profile_means(records)
    summary = {
        "trials": args.analysis_trials,
        "seed_start": args.analysis_seed_start,
        "max_time_per_attempt_s": args.max_time,
        "counts": counts,
        "success_rate": counts["success"] / max(1, args.analysis_trials),
        "avg_runtime_s": sum(r["runtime_s"] for r in records) / max(1, len(records)),
        "profile_means": profile_means,
    }
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    artifacts: Dict[str, Optional[str]] = {
        "csv": csv_path,
        "json": json_path,
        "failure_distribution": None,
        "success_rate": None,
        "runtime_by_seed": None,
        "tree_plot": None,
        "planner_breakdown": None,
        "profile_prof": prof_path,
        "profile_txt": prof_txt_path,
    }

    if not args.analysis_no_plot:
        failure_plot = os.path.join(args.analysis_outdir, f"failure_distribution_{timestamp}.png")
        success_plot = os.path.join(args.analysis_outdir, f"stage1_success_{timestamp}.png")
        runtime_plot = os.path.join(args.analysis_outdir, f"runtime_by_seed_{timestamp}.png")
        breakdown_plot = os.path.join(args.analysis_outdir, f"planner_breakdown_{timestamp}.png")
        if plot_failure_distribution(counts, failure_plot):
            artifacts["failure_distribution"] = failure_plot
        if plot_success_rate(summary["success_rate"], success_plot):
            artifacts["success_rate"] = success_plot
        if plot_runtime_by_seed(records, runtime_plot):
            artifacts["runtime_by_seed"] = runtime_plot
        if plot_planner_breakdown(profile_means, breakdown_plot):
            artifacts["planner_breakdown"] = breakdown_plot
        if tree_data_first is not None:
            tree_plot = os.path.join(
                args.analysis_outdir,
                f"tree_structure_stage1_seed{args.analysis_seed_start}_{timestamp}.png",
            )
            if plot_tree_3d(tree_data_first, tree_plot, f"Stage 1 tree structure (seed {args.analysis_seed_start})"):
                artifacts["tree_plot"] = tree_plot

    write_markdown_report(report_path, timestamp, args, summary, artifacts)
    logger.info(f"Saved analysis CSV: {csv_path}")
    logger.info(f"Saved analysis JSON: {json_path}")
    logger.info(f"Saved debug report: {report_path}")
    if artifacts["profile_prof"]:
        logger.info(f"Saved profile dump: {artifacts['profile_prof']}")
    if artifacts["profile_txt"]:
        logger.info(f"Saved profile text: {artifacts['profile_txt']}")


def parse_args():
    default_grasp_json, default_start_state, default_end_state = build_default_paths()
    parser = argparse.ArgumentParser(description="Stage 1 floating-bar RRT debug runner")
    parser.add_argument("--grasp-json", type=str, default=default_grasp_json, help="Path to grasp JSON file")
    parser.add_argument("--start-state", type=str, default=default_start_state, help="Path to start RobotCellState JSON")
    parser.add_argument("--end-state", type=str, default=default_end_state, help="Path to end RobotCellState JSON")
    parser.add_argument("--gui", action="store_true", help="Run with PyBullet GUI")
    parser.add_argument("--goal-bias", type=float, default=0.1, help="Goal sampling probability")
    parser.add_argument("--dist-metric", choices=["feature", "pose6d"], default="feature", help="Task-space distance metric")
    parser.add_argument("--position-res", type=float, default=0.05, help="Translation resolution used during pose extension, in meters")
    parser.add_argument("--rotation-res", type=float, default=0.1, help="Rotation resolution used during pose extension, in radians")
    parser.add_argument("--max-time", type=float, default=30.0, help="Max planning time per attempt")
    parser.add_argument("--max-iterations", type=int, default=2000, help="Max RRT iterations per attempt")
    parser.add_argument("--max-attempts", type=int, default=5, help="Random restarts")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed for one-shot mode")
    parser.add_argument(
        "--no-floating-collision",
        action="store_false",
        dest="floating_collision",
        help="Disable floating-bar collision against the robot and loaded environment obstacles",
    )
    parser.add_argument(
        "--lock-renderer-during-search",
        action="store_true",
        help="Lock the PyBullet renderer while the tree is being expanded, then show the result afterward",
    )
    parser.add_argument("--analysis-trials", type=int, default=0, help="If > 0, run batch analysis over consecutive seeds")
    parser.add_argument("--analysis-seed-start", type=int, default=0, help="First seed used for batch analysis")
    parser.add_argument("--analysis-outdir", type=str, default=REPORTS_DIR, help="Output directory for analysis artifacts")
    parser.add_argument("--analysis-no-plot", action="store_true", help="Skip matplotlib plot generation during analysis")
    parser.add_argument("--profile-seed", type=int, default=None, help="Seed to capture with cProfile during analysis")
    parser.add_argument("--profile-top-n", type=int, default=30, help="Number of cumulative cProfile rows to include in the text report")
    parser.set_defaults(floating_collision=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analysis_trials > 0:
        run_analysis(args)
        return

    use_gui = args.gui
    debug_tree_out: Dict = {}
    result = run_stage1_trial(
        grasp_json=args.grasp_json,
        start_state_json=args.start_state,
        end_state_json=args.end_state,
        use_gui=use_gui,
        dist_metric=args.dist_metric,
        goal_bias=args.goal_bias,
        position_res=args.position_res,
        rotation_res=args.rotation_res,
        max_time=args.max_time,
        max_iterations=args.max_iterations,
        max_attempts=args.max_attempts,
        random_seed=args.random_seed,
        enable_collision=args.floating_collision,
        lock_renderer_during_search=args.lock_renderer_during_search,
        debug_tree_out=debug_tree_out,
    )

    if use_gui:
        run_visualization_loop(result["scene"]["bar_body"], result["path"], result["scene"]["cid"])

    teardown_stage1_scene()


if __name__ == "__main__":
    main()
