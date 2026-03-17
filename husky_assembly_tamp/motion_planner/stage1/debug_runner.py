"""Stage 1/2/3 debug runner with batch analysis, plots, and report generation."""

from __future__ import annotations

import argparse
import cProfile
import csv
import io
import json
import os
import pstats
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from husky_assembly_tamp.motion_planner.stage1.minimal_rrt import (
    build_default_paths,
    run_stage_trial,
    run_visualization_loop,
    teardown_stage1_scene,
)
from husky_assembly_tamp.utils.util import setup_logger


logger = setup_logger("stage1_debug_runner")

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
SUPPORT_DIRNAME = "_support"
PROFILE_TIME_KEYS = [
    "collision_check_time_s",
    "endpoint_ik_time_s",
    "feature_time_s",
    "sample_time_s",
    "nearest_time_s",
    "ik_time_s",
    "goal_test_time_s",
    "extend_tree_time_s",
]
PROFILE_COUNT_KEYS = [
    "attempts",
    "iterations",
    "nodes_created",
    "poses_checked",
    "collision_hits",
    "ik_calls",
    "ik_failures",
    "nodes_with_ik",
    "continuity_rejections",
]
FAILURE_LABELS = ["task_space_failure", "ik_failure", "continuity_failure", "collision_failure", "success"]
STAGE_ORDER = [1, 2, 3]


def stage_label(stage: int) -> str:
    return f"Stage {stage}"


def resolution_label(position_res: float, rotation_res: float) -> str:
    return f"pos={position_res:.3f}m rot={rotation_res:.3f}rad"


def collision_enabled(stage: int, floating_collision: bool) -> bool:
    return bool(floating_collision or stage >= 3)


def validation_status_label(value: Optional[bool]) -> str:
    if value is None:
        return "N/A"
    return "PASS" if value else "FAIL"


def classify_result(result: Dict[str, Any], planner_profile: Dict, stage: int, enable_collision: bool) -> str:
    if result.get("success"):
        return "success"
    validation = result.get("validation") or {}
    if stage >= 2 and validation.get("joint_continuity_ok") is False:
        return "continuity_failure"
    if stage >= 2 and validation.get("collision_free") is False:
        return "collision_failure"
    outcome = planner_profile.get("outcome")
    if outcome in {"start_in_collision", "goal_in_collision"}:
        return "collision_failure"
    if stage >= 2 and outcome in {"start_ik_failure", "goal_ik_failure", "extend_ik_failure"}:
        return "ik_failure"
    if stage >= 2 and outcome == "extend_continuity_failure":
        return "continuity_failure"
    if enable_collision and int(planner_profile.get("collision_hits", 0)) > 0:
        return "collision_failure"
    if stage >= 2 and int(planner_profile.get("continuity_rejections", 0)) > 0:
        return "continuity_failure"
    if stage >= 2 and int(planner_profile.get("ik_failures", 0)) > 0:
        return "ik_failure"
    return "task_space_failure"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def support_dir(outdir: str) -> str:
    path = os.path.join(outdir, SUPPORT_DIRNAME)
    ensure_dir(path)
    return path


def parse_resolution_sweep_spec(spec: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [part.strip() for part in chunk.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid resolution sweep entry: {chunk!r}")
        pairs.append((float(parts[0]), float(parts[1])))
    if not pairs:
        raise ValueError("Resolution sweep spec is empty.")
    return pairs


def report_relpath(report_path: str, target_path: str) -> str:
    return os.path.relpath(target_path, os.path.dirname(report_path))


def relocate_artifact_to_support(path: Optional[str], support_outdir: str) -> Optional[str]:
    if not path:
        return path
    if not os.path.isfile(path):
        return path
    target_path = os.path.join(support_outdir, os.path.basename(path))
    if os.path.abspath(path) != os.path.abspath(target_path):
        ensure_dir(os.path.dirname(target_path))
        os.replace(path, target_path)
    return target_path


def compute_tree_axis_limits(tree_data_items: Sequence[Dict[str, Any]], padding_ratio: float = 0.08) -> Optional[Dict[str, Tuple[float, float]]]:
    coords: List[Tuple[float, float, float]] = []
    for tree_data in tree_data_items:
        tree = tree_data.get("tree1", {})
        coords.extend(tuple(point[:3]) for point in tree.get("points", []))
        for pose_key in ("start_pose", "goal_pose"):
            pose = tree_data.get(pose_key)
            if pose is not None:
                coords.append(tuple(pose[:3]))
    if not coords:
        return None
    arr = np.asarray(coords, dtype=float)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    span = max(span, 1e-3)
    half_extent = 0.5 * span * (1.0 + padding_ratio)
    return {
        "x": (float(center[0] - half_extent), float(center[0] + half_extent)),
        "y": (float(center[1] - half_extent), float(center[1] + half_extent)),
        "z": (float(center[2] - half_extent), float(center[2] + half_extent)),
    }


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception as e:
        logger.warning(f"Skipping plots (matplotlib unavailable): {e}")
        return None


def plot_tree_3d(
    tree_data: Dict,
    out_path: str,
    title: str,
    axis_limits: Optional[Dict[str, Tuple[float, float]]] = None,
) -> bool:
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
    limits = axis_limits or compute_tree_axis_limits([tree_data])
    if limits is not None:
        ax.set_xlim(*limits["x"])
        ax.set_ylim(*limits["y"])
        ax.set_zlim(*limits["z"])
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def plot_failure_distribution(counts: Dict[str, int], out_path: str, stage: int) -> bool:
    plt = maybe_import_matplotlib()
    if plt is None:
        return False
    labels = ["task_space_failure", "ik_failure", "continuity_failure", "collision_failure", "success"]
    vals = [counts.get(k, 0) for k in labels]
    colors = ["#d9534f", "#f0ad4e", "#9467bd", "#5bc0de", "#5cb85c"]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, vals, color=colors)
    plt.ylabel("Count")
    plt.title(f"{stage_label(stage)} Failure Distribution Across Seeds")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def plot_success_rate(success_rate: float, out_path: str, stage: int) -> bool:
    plt = maybe_import_matplotlib()
    if plt is None:
        return False
    plt.figure(figsize=(4, 4))
    plt.bar([stage_label(stage)], [success_rate], color=["#337ab7"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Success Rate")
    plt.title(f"{stage_label(stage)} Success Rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def plot_runtime_by_seed(records: List[Dict], out_path: str, stage: int) -> bool:
    plt = maybe_import_matplotlib()
    if plt is None or not records:
        return False
    seeds = [r["seed"] for r in records]
    runtimes = [r["runtime_s"] for r in records]
    categories = [r["category"] for r in records]
    color_map = {
        "success": "#5cb85c",
        "task_space_failure": "#d9534f",
        "ik_failure": "#f0ad4e",
        "continuity_failure": "#9467bd",
        "collision_failure": "#5bc0de",
    }
    colors = [color_map.get(c, "#777777") for c in categories]
    plt.figure(figsize=(8, 4))
    plt.bar(seeds, runtimes, color=colors)
    plt.xlabel("Seed")
    plt.ylabel("Runtime (s)")
    plt.title(f"{stage_label(stage)} Runtime by Seed")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def plot_planner_breakdown(mean_profile: Dict[str, float], out_path: str, stage: int) -> bool:
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
    plt.title(f"{stage_label(stage)} Planner Breakdown")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def plot_stage_success_comparison(stage_summaries: Dict[int, Dict[str, Any]], out_path: str) -> bool:
    plt = maybe_import_matplotlib()
    if plt is None or not stage_summaries:
        return False
    stages = sorted(stage_summaries)
    labels = [stage_label(stage) for stage in stages]
    vals = [stage_summaries[stage]["success_rate"] for stage in stages]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals, color=["#5cb85c", "#f0ad4e", "#5bc0de"][: len(labels)])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Success Rate")
    plt.title("Success Rate Comparison Across Stages")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def plot_stage_failure_comparison(stage_summaries: Dict[int, Dict[str, Any]], out_path: str) -> bool:
    plt = maybe_import_matplotlib()
    if plt is None or not stage_summaries:
        return False
    stages = sorted(stage_summaries)
    labels = [stage_label(stage) for stage in stages]
    color_map = {
        "task_space_failure": "#d9534f",
        "ik_failure": "#f0ad4e",
        "continuity_failure": "#9467bd",
        "collision_failure": "#5bc0de",
        "success": "#5cb85c",
    }
    plt.figure(figsize=(8, 4.5))
    bottoms = [0] * len(stages)
    for failure_label in FAILURE_LABELS:
        vals = [stage_summaries[stage]["counts"].get(failure_label, 0) for stage in stages]
        plt.bar(labels, vals, bottom=bottoms, color=color_map[failure_label], label=failure_label)
        bottoms = [bottom + val for bottom, val in zip(bottoms, vals)]
    plt.ylabel("Trial Count")
    plt.title("Failure Distribution Comparison Across Stages")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def plot_stage_runtime_comparison(stage_summaries: Dict[int, Dict[str, Any]], out_path: str) -> bool:
    plt = maybe_import_matplotlib()
    if plt is None or not stage_summaries:
        return False
    stages = sorted(stage_summaries)
    labels = [stage_label(stage) for stage in stages]
    vals = [stage_summaries[stage]["avg_runtime_s"] for stage in stages]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals, color=["#777777", "#999999", "#bbbbbb"][: len(labels)])
    plt.ylabel("Average Runtime (s)")
    plt.title("Average Runtime Comparison Across Stages")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def plot_stage_planner_breakdown_comparison(stage_summaries: Dict[int, Dict[str, Any]], out_path: str) -> bool:
    plt = maybe_import_matplotlib()
    if plt is None or not stage_summaries:
        return False
    stages = sorted(stage_summaries)
    keys = [key for key in PROFILE_TIME_KEYS if any(stage_summaries[stage]["profile_means"].get(key, 0.0) > 0.0 for stage in stages)]
    if not keys:
        return False
    labels = [key.replace("_time_s", "") for key in keys]
    x = list(range(len(keys)))
    width = 0.22
    offsets = [(-width, "#5cb85c"), (0.0, "#f0ad4e"), (width, "#5bc0de")]
    plt.figure(figsize=(10, 4.5))
    for idx, stage in enumerate(stages):
        offset, color = offsets[idx]
        vals = [stage_summaries[stage]["profile_means"].get(key, 0.0) for key in keys]
        xs = [xi + offset for xi in x]
        plt.bar(xs, vals, width=width, color=color, label=stage_label(stage))
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Average time per run (s)")
    plt.title("Planner Breakdown Comparison Across Stages")
    plt.legend(loc="best")
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


def dominant_failure_label(summary: Dict[str, Any]) -> str:
    failure_counts = {label: summary["counts"].get(label, 0) for label in FAILURE_LABELS if label != "success"}
    if not any(failure_counts.values()):
        return "none"
    return max(failure_counts, key=failure_counts.get)


def write_markdown_report(
    report_path: str,
    timestamp: str,
    args,
    stage: int,
    summary: Dict[str, Any],
    artifacts: Dict[str, Optional[str]],
) -> None:
    lines: List[str] = []
    lines.append(f"# {stage_label(stage)} Debugging Report ({timestamp})")
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
        "validation_plot",
        "planner_breakdown",
        "profile_txt",
    ]:
        path = artifacts.get(key)
        if path:
            lines.append(f"- `{report_relpath(report_path, path)}`")
    lines.append("")
    lines.append("Run setup:")
    lines.append("")
    lines.append(f"- Trials: `{args.analysis_trials}` seeds (`{args.analysis_seed_start}..{args.analysis_seed_start + args.analysis_trials - 1}`)")
    lines.append(f"- Per-attempt max time: `{args.max_time}s`")
    lines.append(f"- Dist metric: `{args.dist_metric}`")
    lines.append(f"- Position resolution: `{args.position_res} m`")
    lines.append(f"- Rotation resolution: `{args.rotation_res} rad`")
    if stage >= 2:
        lines.append(f"- Endpoint IK attempts: `{args.endpoint_ik_attempts}`")
        lines.append(f"- Joint continuity threshold: `{args.joint_continuity_threshold} rad`")
    lines.append(f"- Collision: `{'on' if collision_enabled(stage, args.floating_collision) else 'off'}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1) Workspace Tree Visualization")
    lines.append("")
    if artifacts.get("tree_plot"):
        lines.append(f"### {stage_label(stage)} (seed {args.analysis_seed_start})")
        lines.append(f"![{stage_label(stage)} Tree]({report_relpath(report_path, artifacts['tree_plot'])})")
        lines.append("")
        lines.append("Observation:")
        lines.append("")
        lines.append(f"- The tree image shows the task-space exploration footprint used by the single-tree {stage_label(stage)} RRT.")
        lines.append("- This is the quickest way to see whether the sampler is exploring broadly or repeatedly getting trapped near the start or obstacle boundary.")
    else:
        lines.append("Tree plot was not generated.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2) Trajectory Validation")
    lines.append("")
    validation = summary.get("validation_first")
    if artifacts.get("validation_plot"):
        lines.append(f"![Trajectory Validation]({report_relpath(report_path, artifacts['validation_plot'])})")
        lines.append("")
    if validation:
        lines.append("First-seed validation summary:")
        lines.append("")
        lines.append(f"- Collision-free replay: **{validation_status_label(validation.get('collision_free'))}**")
        lines.append(f"- Joint continuity: **{validation_status_label(validation.get('joint_continuity_ok'))}**")
        lines.append(f"- Relative transform consistency: **{validation_status_label(validation.get('relative_transform_ok'))}**")
        lines.append(f"- Joint-path source: `{validation.get('joint_path_source') or 'n/a'}`")
        if stage >= 2:
            lines.append(f"- Max dq: `{float(validation.get('joint_continuity_max_delta_rad') or 0.0):.4f} rad`")
    else:
        lines.append("Validation plot was not generated.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3) Failure Distribution Analysis")
    lines.append("")
    if artifacts.get("failure_distribution"):
        lines.append("### Distribution plot")
        lines.append(f"![Failure Distribution]({report_relpath(report_path, artifacts['failure_distribution'])})")
        lines.append("")
    lines.append("From `summary.counts`:")
    lines.append("")
    for key in FAILURE_LABELS:
        count = summary["counts"][key]
        pct = 100.0 * count / max(1, summary["trials"])
        lines.append(f"- `{key}`: **{count} / {summary['trials']}** ({pct:.0f}%)")
    lines.append("")
    lines.append("### Bottleneck conclusion")
    lines.append("")
    lines.append(f"Dominant failure mode in this run is **{dominant_failure_label(summary)}**.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4) Runtime and Bottleneck Breakdown")
    lines.append("")
    if artifacts.get("success_rate"):
        lines.append("### Validated-success plot")
        lines.append(f"![{stage_label(stage)} Success Rate]({report_relpath(report_path, artifacts['success_rate'])})")
        lines.append("")
    if artifacts.get("runtime_by_seed"):
        lines.append("### Runtime-by-seed plot")
        lines.append(f"![Runtime by Seed]({report_relpath(report_path, artifacts['runtime_by_seed'])})")
        lines.append("")
    if artifacts.get("planner_breakdown"):
        lines.append("### Planner breakdown plot")
        lines.append(f"![Planner Breakdown]({report_relpath(report_path, artifacts['planner_breakdown'])})")
        lines.append("")
    lines.append("From `summary`:")
    lines.append("")
    lines.append(f"- {stage_label(stage)} validated success rate: **{summary['success_rate']:.0%}**")
    lines.append(f"- {stage_label(stage)} task-space path-found rate: **{summary['path_found_rate']:.0%}**")
    lines.append(f"- {stage_label(stage)} avg runtime: **{summary['avg_runtime_s']:.3f} s**")
    lines.append(f"- {stage_label(stage)} avg iterations: **{summary['profile_means'].get('iterations', 0.0):.1f}**")
    lines.append(f"- {stage_label(stage)} avg nodes created: **{summary['profile_means'].get('nodes_created', 0.0):.1f}**")
    lines.append(f"- {stage_label(stage)} avg poses checked: **{summary['profile_means'].get('poses_checked', 0.0):.1f}**")
    if stage >= 2:
        lines.append(f"- {stage_label(stage)} avg IK calls: **{summary['profile_means'].get('ik_calls', 0.0):.1f}**")
        lines.append(f"- {stage_label(stage)} avg IK failures: **{summary['profile_means'].get('ik_failures', 0.0):.1f}**")
        lines.append(
            f"- {stage_label(stage)} avg max dq: **{float(summary.get('avg_joint_continuity_max_delta_rad', 0.0)):.4f} rad**"
        )
    if collision_enabled(stage, args.floating_collision):
        lines.append(f"- {stage_label(stage)} avg collision hits: **{summary['profile_means'].get('collision_hits', 0.0):.1f}**")
    lines.append("")
    if artifacts.get("profile_txt"):
        lines.append(f"Detailed `cProfile` summary: `{report_relpath(report_path, artifacts['profile_txt'])}`")
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
    lines.append(f"1. **Workspace tree visualization**: Achieved. A {stage_label(stage)} tree image is generated for the first seed in the batch.")
    lines.append("2. **Failure distribution analysis**: Achieved. Successes and failures are categorized across seeds and visualized.")
    lines.append("3. **Per-stage trajectory validation support**: Achieved. The report links the first-seed validation replay plot and validation summary.")
    lines.append("")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def run_stage_analysis(args, stage: int, timestamp: str, outdir: str) -> Dict[str, Any]:
    records: List[Dict] = []
    counts = {label: 0 for label in FAILURE_LABELS}
    tree_data_first: Optional[Dict] = None
    validation_first: Optional[Dict[str, Any]] = None
    profile_seed = args.profile_seed if args.profile_seed is not None else args.analysis_seed_start
    support_outdir = support_dir(outdir)

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
                    run_stage_trial,
                    stage=stage,
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
                    endpoint_ik_attempts=args.endpoint_ik_attempts,
                    random_seed=seed,
                    enable_collision=args.floating_collision,
                    joint_continuity_threshold_rad=args.joint_continuity_threshold,
                    lock_renderer_during_search=args.lock_renderer_during_search,
                    debug_tree_out=debug_tree_out,
                    planner_profile_out=planner_profile,
                )
            else:
                result = run_stage_trial(
                    stage=stage,
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
                    endpoint_ik_attempts=args.endpoint_ik_attempts,
                    random_seed=seed,
                    enable_collision=args.floating_collision,
                    joint_continuity_threshold_rad=args.joint_continuity_threshold,
                    lock_renderer_during_search=args.lock_renderer_during_search,
                    debug_tree_out=debug_tree_out,
                    planner_profile_out=planner_profile,
                )
        finally:
            teardown_stage1_scene()

        if profiler is not None:
            prof_path = os.path.join(support_outdir, f"plan_profile_stage{stage}_seed{seed}_{timestamp}.prof")
            prof_txt_path = os.path.join(support_outdir, f"plan_profile_stage{stage}_seed{seed}_{timestamp}.txt")
            profiler.dump_stats(prof_path)
            write_profile_text(prof_path, prof_txt_path, args.profile_top_n)

        if result.get("validation") is not None:
            result["validation"]["plot_path"] = relocate_artifact_to_support(
                result["validation"].get("plot_path"),
                support_outdir,
            )

        category = classify_result(result, planner_profile, stage, collision_enabled(stage, args.floating_collision))
        counts[category] += 1
        record = {
            "stage": stage,
            "seed": seed,
            "path_found": int(result.get("path_found", result["path"] is not None)),
            "success": int(result["success"]),
            "runtime_s": round(result["runtime_s"], 4),
            "category": category,
            "planner_profile": planner_profile,
            "validation": result.get("validation"),
        }
        joint_continuity = result.get("joint_continuity") or {}
        if not joint_continuity:
            joint_continuity = result.get("validation") or {}
        record["joint_continuity_max_delta_rad"] = round(
            float(joint_continuity.get("max_delta_rad") or joint_continuity.get("joint_continuity_max_delta_rad") or 0.0),
            6,
        )
        for key in PROFILE_TIME_KEYS + PROFILE_COUNT_KEYS:
            record[key] = round(float(planner_profile.get(key, 0.0)), 6)
        records.append(record)
        if debug_tree_out is not None:
            tree_data_first = debug_tree_out
        if validation_first is None:
            validation_first = result.get("validation")

    csv_path = os.path.join(support_outdir, f"failure_analysis_stage{stage}_{timestamp}.csv")
    json_path = os.path.join(support_outdir, f"failure_analysis_stage{stage}_{timestamp}.json")
    report_path = os.path.join(outdir, f"debug_report_stage{stage}_{timestamp}.md")

    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "stage",
            "seed",
            "path_found",
            "success",
            "runtime_s",
            "category",
            "joint_continuity_max_delta_rad",
        ] + PROFILE_TIME_KEYS + PROFILE_COUNT_KEYS
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k) for k in fieldnames})

    profile_means = summarize_profile_means(records)
    joint_dq_values = [float(record.get("joint_continuity_max_delta_rad", 0.0)) for record in records if record.get("path_found", 0)]
    summary = {
        "stage": stage,
        "trials": args.analysis_trials,
        "seed_start": args.analysis_seed_start,
        "max_time_per_attempt_s": args.max_time,
        "counts": counts,
        "success_rate": counts["success"] / max(1, args.analysis_trials),
        "path_found_rate": sum(record["path_found"] for record in records) / max(1, len(records)),
        "avg_runtime_s": sum(r["runtime_s"] for r in records) / max(1, len(records)),
        "avg_joint_continuity_max_delta_rad": (sum(joint_dq_values) / max(1, len(joint_dq_values)) if stage >= 2 else 0.0),
        "profile_means": profile_means,
        "validation_first": validation_first,
    }
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    artifacts: Dict[str, Optional[str]] = {
        "csv": csv_path,
        "json": json_path,
        "report": report_path,
        "failure_distribution": None,
        "success_rate": None,
        "runtime_by_seed": None,
        "tree_plot": None,
        "validation_plot": (None if validation_first is None else validation_first.get("plot_path")),
        "planner_breakdown": None,
        "profile_prof": prof_path,
        "profile_txt": prof_txt_path,
    }

    if not args.analysis_no_plot:
        failure_plot = os.path.join(support_outdir, f"failure_distribution_stage{stage}_{timestamp}.png")
        success_plot = os.path.join(support_outdir, f"stage{stage}_success_{timestamp}.png")
        runtime_plot = os.path.join(support_outdir, f"runtime_by_seed_stage{stage}_{timestamp}.png")
        breakdown_plot = os.path.join(support_outdir, f"planner_breakdown_stage{stage}_{timestamp}.png")
        if plot_failure_distribution(counts, failure_plot, stage):
            artifacts["failure_distribution"] = failure_plot
        if plot_success_rate(summary["success_rate"], success_plot, stage):
            artifacts["success_rate"] = success_plot
        if plot_runtime_by_seed(records, runtime_plot, stage):
            artifacts["runtime_by_seed"] = runtime_plot
        if plot_planner_breakdown(profile_means, breakdown_plot, stage):
            artifacts["planner_breakdown"] = breakdown_plot
        if tree_data_first is not None:
            tree_plot = os.path.join(support_outdir, f"tree_structure_stage{stage}_seed{args.analysis_seed_start}_{timestamp}.png")
            axis_limits = compute_tree_axis_limits([tree_data_first])
            if plot_tree_3d(
                tree_data_first,
                tree_plot,
                f"{stage_label(stage)} tree structure (seed {args.analysis_seed_start})",
                axis_limits=axis_limits,
            ):
                artifacts["tree_plot"] = tree_plot

    write_markdown_report(report_path, timestamp, args, stage, summary, artifacts)
    return {
        "stage": stage,
        "records": records,
        "summary": summary,
        "artifacts": artifacts,
        "tree_data_first": tree_data_first,
    }


def run_stage_summary_only(
    args,
    stage: int,
    position_res: float,
    rotation_res: float,
    support_outdir: Optional[str] = None,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    counts = {label: 0 for label in FAILURE_LABELS}
    profile_seed = args.profile_seed if args.profile_seed is not None else args.analysis_seed_start
    validation_first: Optional[Dict[str, Any]] = None

    for i in range(args.analysis_trials):
        seed = args.analysis_seed_start + i
        planner_profile: Dict[str, Any] = {}
        profiler = cProfile.Profile() if seed == profile_seed else None
        try:
            if profiler is not None:
                result = profiler.runcall(
                    run_stage_trial,
                    stage=stage,
                    grasp_json=args.grasp_json,
                    start_state_json=args.start_state,
                    end_state_json=args.end_state,
                    use_gui=False,
                    dist_metric=args.dist_metric,
                    goal_bias=args.goal_bias,
                    position_res=position_res,
                    rotation_res=rotation_res,
                    max_time=args.max_time,
                    max_iterations=args.max_iterations,
                    max_attempts=args.max_attempts,
                    endpoint_ik_attempts=args.endpoint_ik_attempts,
                    random_seed=seed,
                    enable_collision=args.floating_collision,
                    joint_continuity_threshold_rad=args.joint_continuity_threshold,
                    lock_renderer_during_search=args.lock_renderer_during_search,
                    debug_tree_out=None,
                    planner_profile_out=planner_profile,
                )
            else:
                result = run_stage_trial(
                    stage=stage,
                    grasp_json=args.grasp_json,
                    start_state_json=args.start_state,
                    end_state_json=args.end_state,
                    use_gui=False,
                    dist_metric=args.dist_metric,
                    goal_bias=args.goal_bias,
                    position_res=position_res,
                    rotation_res=rotation_res,
                    max_time=args.max_time,
                    max_iterations=args.max_iterations,
                    max_attempts=args.max_attempts,
                    endpoint_ik_attempts=args.endpoint_ik_attempts,
                    random_seed=seed,
                    enable_collision=args.floating_collision,
                    joint_continuity_threshold_rad=args.joint_continuity_threshold,
                    lock_renderer_during_search=args.lock_renderer_during_search,
                    debug_tree_out=None,
                    planner_profile_out=planner_profile,
                )
        finally:
            teardown_stage1_scene()

        if support_outdir is not None and result.get("validation") is not None:
            result["validation"]["plot_path"] = relocate_artifact_to_support(result["validation"].get("plot_path"), support_outdir)

        category = classify_result(result, planner_profile, stage, collision_enabled(stage, args.floating_collision))
        counts[category] += 1
        record = {
            "stage": stage,
            "seed": seed,
            "position_res": position_res,
            "rotation_res": rotation_res,
            "path_found": int(result.get("path_found", result["path"] is not None)),
            "success": int(result["success"]),
            "runtime_s": round(result["runtime_s"], 4),
            "category": category,
            "planner_profile": planner_profile,
            "validation": result.get("validation"),
        }
        joint_continuity = result.get("joint_continuity") or {}
        if not joint_continuity:
            joint_continuity = result.get("validation") or {}
        record["joint_continuity_max_delta_rad"] = round(
            float(joint_continuity.get("max_delta_rad") or joint_continuity.get("joint_continuity_max_delta_rad") or 0.0),
            6,
        )
        for key in PROFILE_TIME_KEYS + PROFILE_COUNT_KEYS:
            record[key] = round(float(planner_profile.get(key, 0.0)), 6)
        records.append(record)
        if validation_first is None:
            validation_first = result.get("validation")

    joint_dq_values = [float(record.get("joint_continuity_max_delta_rad", 0.0)) for record in records if record.get("path_found", 0)]
    return {
        "stage": stage,
        "records": records,
        "summary": {
            "stage": stage,
            "trials": args.analysis_trials,
            "counts": counts,
            "success_rate": counts["success"] / max(1, args.analysis_trials),
            "path_found_rate": sum(record["path_found"] for record in records) / max(1, len(records)),
            "avg_runtime_s": sum(r["runtime_s"] for r in records) / max(1, len(records)),
            "avg_joint_continuity_max_delta_rad": (sum(joint_dq_values) / max(1, len(joint_dq_values)) if stage >= 2 else 0.0),
            "profile_means": summarize_profile_means(records),
            "validation_first": validation_first,
        },
    }


def write_stage_comparison_report(
    report_path: str,
    timestamp: str,
    args,
    stage_results: Dict[int, Dict[str, Any]],
    artifacts: Dict[str, Optional[str]],
) -> None:
    lines: List[str] = []
    stages = sorted(stage_results)
    lines.append(f"# Stage Comparison Debugging Report ({timestamp})")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This report compares Stages 1, 2, and 3 across the same seed range.")
    lines.append("")
    lines.append("Run setup:")
    lines.append("")
    lines.append(f"- Trials per stage: `{args.analysis_trials}` seeds (`{args.analysis_seed_start}..{args.analysis_seed_start + args.analysis_trials - 1}`)")
    lines.append(f"- Per-attempt max time: `{args.max_time}s`")
    lines.append(f"- Dist metric: `{args.dist_metric}`")
    lines.append(f"- Position resolution: `{args.position_res} m`")
    lines.append(f"- Rotation resolution: `{args.rotation_res} rad`")
    lines.append(f"- Endpoint IK attempts: `{args.endpoint_ik_attempts}`")
    lines.append(f"- Joint continuity threshold: `{args.joint_continuity_threshold} rad`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1) Workspace Tree Visualization")
    lines.append("")
    lines.append("The first seed is rendered for each stage so the exploration footprint can be compared directly.")
    lines.append("")
    for stage in stages:
        tree_plot = stage_results[stage]["artifacts"].get("tree_plot")
        if tree_plot:
            lines.append(f"### {stage_label(stage)}")
            lines.append(f"![{stage_label(stage)} Tree]({report_relpath(report_path, tree_plot)})")
            lines.append("")
    lines.append("Observation:")
    lines.append("")
    lines.append("- Stage 1 isolates task-space exploration.")
    lines.append("- Stage 2 shows how dual-arm IK feasibility prunes the same task-space search.")
    lines.append("- Stage 3 shows the additional pruning introduced by collision checking.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2) Trajectory Validation")
    lines.append("")
    lines.append("The first-seed trajectory replay validation plot is included for each stage.")
    lines.append("")
    for stage in stages:
        validation_plot = stage_results[stage]["artifacts"].get("validation_plot")
        validation = stage_results[stage]["summary"].get("validation_first")
        if validation_plot:
            lines.append(f"### {stage_label(stage)} Validation")
            lines.append(f"![{stage_label(stage)} Validation]({report_relpath(report_path, validation_plot)})")
            if validation:
                lines.append("")
                lines.append(
                    f"- Collision-free: **{validation_status_label(validation.get('collision_free'))}**"
                    f", joint continuity: **{validation_status_label(validation.get('joint_continuity_ok'))}**"
                    f", relative transform: **{validation_status_label(validation.get('relative_transform_ok'))}**"
                )
                lines.append(f"- Joint-path source: `{validation.get('joint_path_source') or 'n/a'}`")
                if stage >= 2:
                    lines.append(f"- Max dq: `{float(validation.get('joint_continuity_max_delta_rad') or 0.0):.4f} rad`")
                lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3) Failure Distribution Analysis")
    lines.append("")
    if artifacts.get("failure_distribution_comparison"):
        lines.append(f"![Failure Distribution Comparison]({report_relpath(report_path, artifacts['failure_distribution_comparison'])})")
        lines.append("")
    lines.append("| Stage | Task-space | IK | Continuity | Collision | Success | Dominant failure |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for stage in stages:
        summary = stage_results[stage]["summary"]
        lines.append(
            f"| {stage_label(stage)} | {summary['counts']['task_space_failure']} | {summary['counts']['ik_failure']} | "
            f"{summary['counts']['continuity_failure']} | "
            f"{summary['counts']['collision_failure']} | {summary['counts']['success']} | {dominant_failure_label(summary)} |"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- Stage 1 failures are pure task-space failures.")
    lines.append("- New IK failures in Stage 2 quantify the cost of enforcing dual-arm feasibility.")
    lines.append("- New continuity failures show where seed-chained IK can find a pose path but not a smooth joint realization.")
    lines.append("- New collision failures quantify the extra cost of self/environment avoidance once IK already succeeds.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4) Per-Stage Comparison")
    lines.append("")
    if artifacts.get("success_rate_comparison"):
        lines.append(f"![Success Rate Comparison]({report_relpath(report_path, artifacts['success_rate_comparison'])})")
        lines.append("")
    if artifacts.get("runtime_comparison"):
        lines.append(f"![Runtime Comparison]({report_relpath(report_path, artifacts['runtime_comparison'])})")
        lines.append("")
    if artifacts.get("planner_breakdown_comparison"):
        lines.append(f"![Planner Breakdown Comparison]({report_relpath(report_path, artifacts['planner_breakdown_comparison'])})")
        lines.append("")
    lines.append("| Stage | Validated success | Path found | Avg runtime (s) | Avg iterations | Avg nodes | Avg poses checked | Avg IK calls | Avg collision hits | Avg max dq |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for stage in stages:
        summary = stage_results[stage]["summary"]
        means = summary["profile_means"]
        dq_text = f"{float(summary.get('avg_joint_continuity_max_delta_rad', 0.0)):.4f}" if stage >= 2 else "n/a"
        lines.append(
            f"| {stage_label(stage)} | {summary['success_rate']:.0%} | {summary['path_found_rate']:.0%} | {summary['avg_runtime_s']:.3f} | "
            f"{means.get('iterations', 0.0):.1f} | {means.get('nodes_created', 0.0):.1f} | "
            f"{means.get('poses_checked', 0.0):.1f} | {means.get('ik_calls', 0.0):.1f} | {means.get('collision_hits', 0.0):.1f} | {dq_text} |"
        )
    lines.append("")
    lines.append("Detailed stage reports:")
    lines.append("")
    for stage in stages:
        report = stage_results[stage]["artifacts"].get("report")
        if report:
            lines.append(f"- `{report_relpath(report_path, report)}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Final Answer to Debugging Goals")
    lines.append("")
    lines.append("1. **Workspace tree visualization**: Achieved. The report includes one tree image per stage for the same seed.")
    lines.append("2. **Failure distribution analysis**: Achieved. Failure categories are compared side by side across all three stages.")
    lines.append("3. **Per-stage comparison**: Achieved. Success rate, runtime, bottleneck mix, and planner timing are summarized side by side across Stages 1, 2, and 3.")
    lines.append("")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def write_resolution_sweep_report(
    report_path: str,
    timestamp: str,
    args,
    sweep_results: Sequence[Dict[str, Any]],
    artifacts: Dict[str, Optional[str]],
) -> None:
    lines: List[str] = []
    lines.append(f"# Resolution Sweep Report ({timestamp})")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This report compares Stage 1, Stage 2, and Stage 3 across multiple task-space interpolation resolutions.")
    lines.append("")
    lines.append("Run setup:")
    lines.append("")
    lines.append(f"- Trials per stage/resolution pair: `{args.analysis_trials}` seeds (`{args.analysis_seed_start}..{args.analysis_seed_start + args.analysis_trials - 1}`)")
    lines.append(f"- Dist metric: `{args.dist_metric}`")
    lines.append(f"- Endpoint IK attempts: `{args.endpoint_ik_attempts}`")
    lines.append(f"- Joint continuity threshold: `{args.joint_continuity_threshold} rad`")
    lines.append("")
    if artifacts.get("csv"):
        lines.append(f"CSV: `{report_relpath(report_path, artifacts['csv'])}`")
    if artifacts.get("json"):
        lines.append(f"JSON: `{report_relpath(report_path, artifacts['json'])}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Resolution | Stage | Validated success | Path found | Avg runtime (s) | Dominant failure | Avg continuity rejects | Avg max dq |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- | ---: | --- |")
    for sweep_result in sweep_results:
        label = sweep_result["label"]
        for stage in STAGE_ORDER:
            summary = sweep_result["stage_results"][stage]["summary"]
            lines.append(
                f"| {label} | {stage_label(stage)} | {summary['success_rate']:.0%} | {summary['path_found_rate']:.0%} | "
                f"{summary['avg_runtime_s']:.3f} | {dominant_failure_label(summary)} | "
                f"{summary['profile_means'].get('continuity_rejections', 0.0):.1f} | "
                f"{float(summary.get('avg_joint_continuity_max_delta_rad', 0.0)):.4f} |"
            )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- Stage 1 should remain largely insensitive to the continuity threshold because it does not plan in joint space.")
    lines.append("- Stage 2 indicates whether finer Cartesian interpolation is enough to recover validated smooth IK paths.")
    lines.append("- Stage 3 indicates whether continuity and collision can both be satisfied at the same resolution before a ladder-graph fallback is needed.")
    lines.append("")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def run_analysis(args) -> None:
    ensure_dir(args.analysis_outdir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stage_result = run_stage_analysis(args, args.stage, timestamp, args.analysis_outdir)
    artifacts = stage_result["artifacts"]
    logger.info(f"Saved analysis CSV: {artifacts['csv']}")
    logger.info(f"Saved analysis JSON: {artifacts['json']}")
    logger.info(f"Saved debug report: {artifacts['report']}")
    if artifacts["profile_prof"]:
        logger.info(f"Saved profile dump: {artifacts['profile_prof']}")
    if artifacts["profile_txt"]:
        logger.info(f"Saved profile text: {artifacts['profile_txt']}")


def run_stage_comparison(args) -> None:
    ensure_dir(args.analysis_outdir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    support_outdir = support_dir(args.analysis_outdir)
    stage_results: Dict[int, Dict[str, Any]] = {}
    for stage in STAGE_ORDER:
        stage_results[stage] = run_stage_analysis(args, stage, timestamp, args.analysis_outdir)

    combined_records: List[Dict[str, Any]] = []
    for stage in STAGE_ORDER:
        combined_records.extend(stage_results[stage]["records"])

    combined_csv_path = os.path.join(support_outdir, f"stage_comparison_{timestamp}.csv")
    combined_json_path = os.path.join(support_outdir, f"stage_comparison_{timestamp}.json")
    comparison_report_path = os.path.join(args.analysis_outdir, f"stage_comparison_report_{timestamp}.md")
    comparison_artifacts: Dict[str, Optional[str]] = {
        "csv": combined_csv_path,
        "json": combined_json_path,
        "report": comparison_report_path,
        "failure_distribution_comparison": None,
        "success_rate_comparison": None,
        "runtime_comparison": None,
        "planner_breakdown_comparison": None,
    }

    with open(combined_csv_path, "w", newline="") as f:
        fieldnames = [
            "stage",
            "seed",
            "path_found",
            "success",
            "runtime_s",
            "category",
            "joint_continuity_max_delta_rad",
        ] + PROFILE_TIME_KEYS + PROFILE_COUNT_KEYS
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in combined_records:
            writer.writerow({k: record.get(k) for k in fieldnames})

    with open(combined_json_path, "w") as f:
        json.dump(
            {
                "stage_summaries": {str(stage): stage_results[stage]["summary"] for stage in STAGE_ORDER},
                "records": combined_records,
            },
            f,
            indent=2,
        )

    if not args.analysis_no_plot:
        failure_plot = os.path.join(support_outdir, f"failure_distribution_comparison_{timestamp}.png")
        success_plot = os.path.join(support_outdir, f"success_rate_comparison_{timestamp}.png")
        runtime_plot = os.path.join(support_outdir, f"runtime_comparison_{timestamp}.png")
        breakdown_plot = os.path.join(support_outdir, f"planner_breakdown_comparison_{timestamp}.png")
        stage_summaries = {stage: stage_results[stage]["summary"] for stage in STAGE_ORDER}
        if plot_stage_failure_comparison(stage_summaries, failure_plot):
            comparison_artifacts["failure_distribution_comparison"] = failure_plot
        if plot_stage_success_comparison(stage_summaries, success_plot):
            comparison_artifacts["success_rate_comparison"] = success_plot
        if plot_stage_runtime_comparison(stage_summaries, runtime_plot):
            comparison_artifacts["runtime_comparison"] = runtime_plot
        if plot_stage_planner_breakdown_comparison(stage_summaries, breakdown_plot):
            comparison_artifacts["planner_breakdown_comparison"] = breakdown_plot
        shared_tree_limits = compute_tree_axis_limits(
            [
                stage_results[stage]["tree_data_first"]
                for stage in STAGE_ORDER
                if stage_results[stage]["tree_data_first"] is not None
            ]
        )
        if shared_tree_limits is not None:
            for stage in STAGE_ORDER:
                tree_data = stage_results[stage]["tree_data_first"]
                tree_plot = stage_results[stage]["artifacts"].get("tree_plot")
                if tree_data is None or tree_plot is None:
                    continue
                plot_tree_3d(
                    tree_data,
                    tree_plot,
                    f"{stage_label(stage)} tree structure (seed {args.analysis_seed_start})",
                    axis_limits=shared_tree_limits,
                )

    write_stage_comparison_report(comparison_report_path, timestamp, args, stage_results, comparison_artifacts)
    logger.info(f"Saved comparison CSV: {combined_csv_path}")
    logger.info(f"Saved comparison JSON: {combined_json_path}")
    logger.info(f"Saved comparison report: {comparison_report_path}")


def run_resolution_sweep(args) -> None:
    ensure_dir(args.analysis_outdir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    support_outdir = support_dir(args.analysis_outdir)
    resolution_pairs = parse_resolution_sweep_spec(args.resolution_sweep)
    sweep_results: List[Dict[str, Any]] = []
    combined_records: List[Dict[str, Any]] = []

    for position_res, rotation_res in resolution_pairs:
        label = resolution_label(position_res, rotation_res)
        stage_results: Dict[int, Dict[str, Any]] = {}
        logger.info(f"Running resolution sweep point {label}")
        for stage in STAGE_ORDER:
            stage_results[stage] = run_stage_summary_only(args, stage, position_res, rotation_res, support_outdir=support_outdir)
            for record in stage_results[stage]["records"]:
                record["resolution_label"] = label
                combined_records.append(record)
        sweep_results.append(
            {
                "label": label,
                "position_res": position_res,
                "rotation_res": rotation_res,
                "stage_results": stage_results,
            }
        )

    csv_path = os.path.join(support_outdir, f"resolution_sweep_{timestamp}.csv")
    json_path = os.path.join(support_outdir, f"resolution_sweep_{timestamp}.json")
    report_path = os.path.join(args.analysis_outdir, f"resolution_sweep_report_{timestamp}.md")
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "resolution_label",
            "position_res",
            "rotation_res",
            "stage",
            "seed",
            "path_found",
            "success",
            "runtime_s",
            "category",
            "joint_continuity_max_delta_rad",
        ] + PROFILE_TIME_KEYS + PROFILE_COUNT_KEYS
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in combined_records:
            writer.writerow({k: record.get(k) for k in fieldnames})

    with open(json_path, "w") as f:
        json.dump(
            {
                "resolution_pairs": [
                    {
                        "label": item["label"],
                        "position_res": item["position_res"],
                        "rotation_res": item["rotation_res"],
                        "stage_summaries": {str(stage): item["stage_results"][stage]["summary"] for stage in STAGE_ORDER},
                    }
                    for item in sweep_results
                ],
                "records": combined_records,
            },
            f,
            indent=2,
        )

    write_resolution_sweep_report(
        report_path,
        timestamp,
        args,
        sweep_results,
        {"csv": csv_path, "json": json_path, "report": report_path},
    )
    logger.info(f"Saved resolution sweep CSV: {csv_path}")
    logger.info(f"Saved resolution sweep JSON: {json_path}")
    logger.info(f"Saved resolution sweep report: {report_path}")


def parse_args():
    default_grasp_json, default_start_state, default_end_state = build_default_paths()
    parser = argparse.ArgumentParser(description="Stage 1/2/3 floating-bar RRT debug runner")
    parser.add_argument("--grasp-json", type=str, default=default_grasp_json, help="Path to grasp JSON file")
    parser.add_argument("--start-state", type=str, default=default_start_state, help="Path to start RobotCellState JSON")
    parser.add_argument("--end-state", type=str, default=default_end_state, help="Path to end RobotCellState JSON")
    parser.add_argument("--stage", choices=[1, 2, 3], type=int, default=1, help="Planning stage to analyze")
    parser.add_argument("--gui", action="store_true", help="Run with PyBullet GUI")
    parser.add_argument("--goal-bias", type=float, default=0.1, help="Goal sampling probability")
    parser.add_argument("--dist-metric", choices=["feature", "pose6d"], default="feature", help="Task-space distance metric")
    parser.add_argument("--position-res", type=float, default=0.01, help="Translation resolution used during pose extension, in meters")
    parser.add_argument("--rotation-res", type=float, default=0.025, help="Rotation resolution used during pose extension, in radians")
    parser.add_argument("--max-time", type=float, default=30.0, help="Max planning time per attempt")
    parser.add_argument("--max-iterations", type=int, default=2000, help="Max RRT iterations per attempt")
    parser.add_argument("--max-attempts", type=int, default=5, help="Random restarts")
    parser.add_argument("--endpoint-ik-attempts", type=int, default=20, help="Max random seeds used when solving endpoint IK in Stage 2/3")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed for one-shot mode")
    parser.add_argument("--joint-continuity-threshold", type=float, default=0.2, help="Maximum allowed wrapped joint delta between neighboring Stage 2/3 configurations, in radians")
    parser.add_argument(
        "--floating-collision",
        action="store_true",
        help="Enable floating-bar collision in Stage 1; Stage 3 always enables robot collision checking",
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
    parser.add_argument("--compare-stages", action="store_true", help="Run Stage 1, Stage 2, and Stage 3 batch analysis in one go and emit a comparison report")
    parser.add_argument(
        "--resolution-sweep",
        type=str,
        default="",
        help="Semicolon-separated resolution pairs like '0.05,0.1;0.03,0.07;0.02,0.05' for a multi-resolution Stage 1/2/3 sweep",
    )
    parser.add_argument("--profile-seed", type=int, default=None, help="Seed to capture with cProfile during analysis")
    parser.add_argument("--profile-top-n", type=int, default=30, help="Number of cumulative cProfile rows to include in the text report")
    parser.set_defaults(floating_collision=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analysis_trials > 0:
        if args.resolution_sweep:
            run_resolution_sweep(args)
        elif args.compare_stages:
            run_stage_comparison(args)
        else:
            run_analysis(args)
        return

    use_gui = args.gui
    debug_tree_out: Dict = {}
    result = run_stage_trial(
        stage=args.stage,
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
        endpoint_ik_attempts=args.endpoint_ik_attempts,
        random_seed=args.random_seed,
        enable_collision=args.floating_collision,
        joint_continuity_threshold_rad=args.joint_continuity_threshold,
        lock_renderer_during_search=args.lock_renderer_during_search,
        debug_tree_out=debug_tree_out,
    )

    if use_gui:
        run_visualization_loop(
            result["scene"]["bar_body"],
            result["path"],
            result["scene"]["cid"],
            robot=result["scene"]["robot"],
            arm_joints=result["scene"]["arm_joints"],
            path_confs=result["path_confs"],
        )

    teardown_stage1_scene()


if __name__ == "__main__":
    main()
