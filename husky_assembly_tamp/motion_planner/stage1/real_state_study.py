from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pybullet
import pybullet_planning as pp

from husky_assembly_tamp.motion_planner.stage1.minimal_rrt import (
    DESIGN_STUDY_BAR_SEQUENCE,
    DEFAULT_HOME_LEFT_TOOL_Z_OFFSET,
    DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
    HUSKY_DUAL_ARM_JOINT_NAMES,
    MOBILE_BASE_FROM_TOOL0_LEFT_HOME,
    auto_compute_home_bar_pose,
    build_default_paths,
    build_real_design_goal_spec,
    derive_home_start_poses_from_grasps,
    get_joint_collision_fn,
    load_robot_cell_state,
    run_stage1_trial,
    run_stage2_trial,
    run_stage3_trial,
    run_visualization_loop,
    setup_planning_scene,
    solve_endpoint_dual_arm_ik,
    teardown_planning_scene,
)
from husky_assembly_tamp.motion_planner.stage1.path_validation import import_matplotlib_pyplot
from husky_assembly_tamp.motion_planner.stage1.path_validation import DEFAULT_DENSE_JOINT_VALIDATION_STEP_RAD
from husky_assembly_tamp.motion_planner.stage1.trajectory_io import save_path_as_joint_trajectory
from husky_assembly_tamp.utils.params import DATA_DIR
from husky_assembly_tamp.utils.util import setup_logger


logger = setup_logger("stage1_real_state_study")


DEFAULT_TARGET_NAMES = ["G1", "G2", "G3", "G4", "V1", "V2", "H1", "D1", "V3"]
BAR_POSE_POSITION_WARN_TOL_M = 1e-4
BAR_POSE_ORIENTATION_WARN_TOL_RAD = 1e-3


def default_design_root() -> str:
    return os.path.join(
        DATA_DIR,
        "husky_assembly_design_study",
        "250929_New_Antenna_with_GH_RH_Packed",
    )


def reports_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "reports")


def support_dir() -> str:
    path = os.path.join(reports_dir(), "_support")
    os.makedirs(path, exist_ok=True)
    return path


def pose_to_json(pose) -> Dict[str, List[float]]:
    return {
        "position": [float(v) for v in pose[0]],
        "quaternion": [float(v) for v in pose[1]],
    }


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def save_smoothing_comparison_plot(
    *,
    out_path: str,
    target_name: str,
    path_before_smoothing: Sequence[Any],
    path_after_smoothing: Sequence[Any],
    smoothing_profile: Optional[Dict[str, Any]],
) -> Optional[str]:
    if not path_before_smoothing or not path_after_smoothing:
        return None
    plt = import_matplotlib_pyplot()
    if plt is None:
        return None

    before_xyz = np.asarray([np.asarray(pose[0], dtype=float) for pose in path_before_smoothing], dtype=float)
    after_xyz = np.asarray([np.asarray(pose[0], dtype=float) for pose in path_after_smoothing], dtype=float)

    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        before_xyz[:, 0],
        before_xyz[:, 1],
        before_xyz[:, 2],
        color="#ff7f0e",
        linewidth=1.8,
        alpha=0.85,
        label=f"before smoothing ({len(path_before_smoothing)} wp)",
    )
    ax.plot(
        after_xyz[:, 0],
        after_xyz[:, 1],
        after_xyz[:, 2],
        color="#1f77b4",
        linewidth=2.2,
        alpha=0.95,
        label=f"after smoothing ({len(path_after_smoothing)} wp)",
    )
    ax.scatter(before_xyz[0, 0], before_xyz[0, 1], before_xyz[0, 2], color="#2ca02c", s=36, label="start")
    ax.scatter(after_xyz[-1, 0], after_xyz[-1, 1], after_xyz[-1, 2], color="#d62728", s=36, label="goal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(target_name)
    cost_before = None if smoothing_profile is None else smoothing_profile.get("cost_before")
    cost_after = None if smoothing_profile is None else smoothing_profile.get("cost_after")
    subtitle = "Cost before: n/a | Cost after: n/a"
    if cost_before is not None and cost_after is not None:
        subtitle = f"Cost before: {float(cost_before):.4f} | Cost after: {float(cost_after):.4f}"
    fig.suptitle("Bar path before and after smoothing", fontsize=13)
    fig.text(0.5, 0.93, subtitle, ha="center", va="top", fontsize=10)
    ax.legend(loc="best")

    all_xyz = np.vstack([before_xyz, after_xyz])
    mins = np.min(all_xyz, axis=0)
    maxs = np.max(all_xyz, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_replay_bundle(
    *,
    scene: Dict[str, Any],
    spec: Dict[str, Any],
    joint_path: Sequence[Any],
    pose_path: Sequence[Any],
    trajectory_json_path: str,
    metadata_json_path: str,
) -> None:
    os.makedirs(os.path.dirname(trajectory_json_path), exist_ok=True)
    save_path_as_joint_trajectory(joint_path, HUSKY_DUAL_ARM_JOINT_NAMES, trajectory_json_path)
    metadata = {
        "scene_spec": {
            "mobile_base_from_tool0_left_home": pose_to_json(scene["mobile_base_from_tool0_left_home"]),
            "world_from_bar_start": pose_to_json(scene["world_from_bar_start"]),
            "start_joint_values": [float(v) for v in scene["start_joint_values"]],
            "end_joint_values": [float(v) for v in scene["end_joint_values"]],
            "world_from_bar_goal": pose_to_json(scene["world_from_bar_goal"]),
            "grasp_targets": [[pose_to_json(bar_pose), pose_to_json(tool_pose)] for bar_pose, tool_pose in spec["grasp_targets"]],
            "active_bar_mesh": to_jsonable(spec["active_bar_mesh"]),
            "built_bars": to_jsonable(spec["built_bars"]),
        },
        "pose_path": [pose_to_json(pose) for pose in pose_path],
        "start_pose": pose_to_json(scene["world_from_bar_start"]),
        "goal_pose": pose_to_json(scene["world_from_bar_goal"]),
    }
    with open(metadata_json_path, "w") as f:
        json.dump(metadata, f, indent=2)


def build_replay_command(
    *,
    trajectory_json_path: str,
    metadata_json_path: str,
    grasp_json: str,
    start_state_json: str,
    end_state_json: str,
) -> str:
    parts = [
        "cd",
        shlex.quote(os.getcwd()),
        "&&",
        shlex.quote(sys.executable),
        "-m",
        "husky_assembly_tamp.motion_planner.stage1.trajectory_replay",
        "--trajectory-json",
        shlex.quote(trajectory_json_path),
        "--metadata-json",
        shlex.quote(metadata_json_path),
        "--grasp-json",
        shlex.quote(grasp_json),
        "--start-state",
        shlex.quote(start_state_json),
        "--end-state",
        shlex.quote(end_state_json),
    ]
    return " ".join(parts)


def record_trajectory_video(
    scene: Dict[str, Any],
    joint_path: Sequence[Any],
    pose_path: Sequence[Any],
    out_path: str,
    frame_step: int = 2,
    frame_sleep: float = 0.02,
    width: int = 1024,
    height: int = 768,
) -> Optional[str]:
    if len(pose_path) == 0 or len(joint_path) == 0:
        return None

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        logger.warning("Skipping trajectory video capture because imageio is unavailable: %s", exc)
        return None

    pp.set_pose(scene["ghost_start"], scene["world_from_bar_start"])
    pp.set_pose(scene["ghost_goal"], scene["world_from_bar_goal"])
    pp.set_pose(scene["bar_body"], pose_path[0])
    pp.set_joint_positions(scene["robot"], scene["arm_joints"], joint_path[0])

    view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.6, 0.0, 0.7],
        distance=2.3,
        yaw=45.0,
        pitch=-25.0,
        roll=0.0,
        upAxisIndex=2,
    )
    projection_matrix = pybullet.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=float(width) / float(height),
        nearVal=0.02,
        farVal=6.0,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fps = max(1, int(round(1.0 / max(frame_sleep, 1e-6))))
    try:
        with imageio.get_writer(out_path, fps=fps, macro_block_size=1) as writer:
            for index, (conf, pose) in enumerate(zip(joint_path, pose_path)):
                if index % max(1, frame_step) != 0 and index != len(joint_path) - 1:
                    continue
                pp.set_joint_positions(scene["robot"], scene["arm_joints"], conf)
                pp.set_pose(scene["bar_body"], pose)
                pybullet.stepSimulation(physicsClientId=scene["cid"])
                _, _, rgba, _, _ = pybullet.getCameraImage(
                    width=width,
                    height=height,
                    viewMatrix=view_matrix,
                    projectionMatrix=projection_matrix,
                    renderer=pybullet.ER_TINY_RENDERER,
                    physicsClientId=scene["cid"],
                )
                image = np.reshape(np.asarray(rgba, dtype=np.uint8), (height, width, 4))
                writer.append_data(image[:, :, :3])
    except Exception as exc:
        logger.warning("Skipping trajectory video capture because writing %s failed: %s", out_path, exc)
        if os.path.isfile(out_path):
            os.remove(out_path)
        return None

    if not os.path.isfile(out_path):
        return None
    return out_path


def compute_common_start_context(
    grasp_json: str,
    start_state_json: str,
    end_state_json: str,
) -> Dict[str, Any]:
    return {
        "start_joint_values": np.asarray(load_robot_cell_state(start_state_json), dtype=float),
        "mobile_base_from_tool0_left_home": MOBILE_BASE_FROM_TOOL0_LEFT_HOME,
    }


def derive_start_pose_from_home_left_tool(
    spec: Dict[str, Any],
    common_start: Dict[str, Any],
    home_left_tool_offset: List[float],
    home_left_tool_local_yaw: float,
    auto_home_pose: bool = True,
    ik_validator: Optional[Callable[[Any], bool]] = None,
    num_geometric_candidates: int = 20,
) -> Dict[str, Any]:
    if len(spec["grasp_targets"]) < 2:
        raise ValueError(f"Target {spec['target_name']} requires two grasp targets to derive the home start pose.")
    world_from_bar_left, _ = spec["grasp_targets"][0]
    world_from_bar_right, _ = spec["grasp_targets"][1]

    bar_position_delta_m = float(
        np.linalg.norm(
            np.asarray(world_from_bar_left[0], dtype=float) - np.asarray(world_from_bar_right[0], dtype=float)
        )
    )
    left_quat = np.asarray(world_from_bar_left[1], dtype=float)
    right_quat = np.asarray(world_from_bar_right[1], dtype=float)
    left_quat /= np.linalg.norm(left_quat)
    right_quat /= np.linalg.norm(right_quat)
    quat_alignment = float(np.clip(abs(np.dot(left_quat, right_quat)), -1.0, 1.0))
    bar_orientation_delta_rad = float(2.0 * np.arccos(quat_alignment))
    if (
        bar_position_delta_m > BAR_POSE_POSITION_WARN_TOL_M
        or bar_orientation_delta_rad > BAR_POSE_ORIENTATION_WARN_TOL_RAD
    ):
        logger.warning(
            "Target %s grasp targets disagree on world_from_bar: position delta=%.6f m, orientation delta=%.6f rad",
            spec["target_name"],
            bar_position_delta_m,
            bar_orientation_delta_rad,
        )

    mobile_base_from_tool0_left_home = (
        np.asarray(common_start["mobile_base_from_tool0_left_home"][0], dtype=float) + np.asarray(home_left_tool_offset, dtype=float),
        common_start["mobile_base_from_tool0_left_home"][1],
    )
    if auto_home_pose:
        start_pose_context = auto_compute_home_bar_pose(
            spec["grasp_targets"],
            mobile_base_from_tool0_left=mobile_base_from_tool0_left_home,
            ik_validator=ik_validator,
            num_geometric_candidates=num_geometric_candidates,
        )
        logger.info(
            "Auto home pose for %s: flip_yaw=%.4f rad, bar_axis_theta=%.4f rad, alignment_score=%.4f",
            spec["target_name"],
            start_pose_context["chosen_flip_yaw"],
            start_pose_context["chosen_bar_axis_theta"],
            start_pose_context["alignment_score"],
        )
    else:
        if abs(float(home_left_tool_local_yaw)) > 0.0:
            mobile_base_from_tool0_left_home = pp.multiply(
                mobile_base_from_tool0_left_home,
                pp.Pose(euler=pp.Euler(yaw=float(home_left_tool_local_yaw))),
            )
        start_pose_context = derive_home_start_poses_from_grasps(
            spec["grasp_targets"],
            mobile_base_from_tool0_left=mobile_base_from_tool0_left_home,
        )
    return {
        "mobile_base_from_tool0_left_home": start_pose_context["mobile_base_from_tool0_left_start"],
        "world_from_bar_start": start_pose_context["mobile_base_from_bar_start"],
        "derived_right_tool_pose": start_pose_context["mobile_base_from_tool0_right_start"],
    }


def build_scene_spec_from_start_context(
    common_start: Dict[str, Any],
    spec: Dict[str, Any],
    start_context: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "mobile_base_from_tool0_left_home": start_context["mobile_base_from_tool0_left_home"],
        "world_from_bar_start": start_context["world_from_bar_start"],
        "start_joint_values": common_start["start_joint_values"],
        "end_joint_values": spec["robot_state"]["joint_values"],
        "world_from_bar_goal": spec["goal_pose"],
        "grasp_targets": spec["grasp_targets"],
        "active_bar_mesh": spec["active_bar_mesh"],
        "built_bars": spec["built_bars"],
    }


def summarize_result(target_name: str, spec: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    validation = result.get("validation", {})
    continuity = result.get("joint_continuity") or {}
    goal_pose = spec["goal_pose"]
    return {
        "target_name": target_name,
        "active_bar_body_name": spec["active_bar_mesh"]["body_name"],
        "bar_box_dims": [float(v) for v in spec["active_bar_mesh"]["aabb_dims"]],
        "goal_position": [float(v) for v in goal_pose[0]],
        "goal_quaternion": [float(v) for v in goal_pose[1]],
        "path_found": bool(result.get("path_found", result.get("path") is not None)),
        "success": bool(result["success"]),
        "runtime_s": float(result.get("runtime_s", 0.0)),
        "planning_time_s": float(result.get("planning_time_s", 0.0)),
        "smoothing_time_s": float(result.get("smoothing_time_s", 0.0)),
        "validation_time_s": float(result.get("validation_time_s", 0.0)),
        "waypoints": int(len(result["path"])) if result["path"] is not None else 0,
        "max_dq_rad": continuity.get("max_delta_rad"),
        "joint_continuity_ok": validation.get("joint_continuity_ok"),
        "collision_free": validation.get("collision_free"),
        "collision_reason": validation.get("failure_reason", result.get("failure_reason")),
        "validation_plot": validation.get("plot_path"),
        "video_mp4": result.get("video_mp4"),
        "trajectory_json": result.get("trajectory_json"),
        "trajectory_metadata_json": result.get("trajectory_metadata_json"),
        "replay_command": result.get("replay_command"),
        "smoothing_plot": result.get("smoothing_plot"),
    }


def write_report(report_path: str, json_relpath: str, args, common_start: Dict[str, Any], summaries: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append(f"# Real State Stage {args.stage} Study ({datetime.now().strftime('%Y%m%d_%H%M%S')})")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"This report benchmarks Stage {args.stage} against real design-study targets using:")
    lines.append(f"- common left-tool home pose from the default start state")
    lines.append(f"- per-target start bar pose derived from that home tool pose and the target grasp transform")
    lines.append(f"- per-target goal pose and grasps from `*_GraspTargets.json`")
    lines.append(f"- per-target active bar mesh from `RobotCell.json`")
    lines.append(f"- planning frame: mobile-base frame")
    lines.append(f"- support JSON: `{json_relpath}`")
    lines.append("")
    lines.append(f"- Design root: `{args.design_root}`")
    lines.append(f"- Targets: `{', '.join(args.targets)}`")
    lines.append(f"- Include built bars: `{args.include_built_bars}`")
    lines.append(f"- Built-bar collision enabled: `{args.enable_built_bar_collision}`")
    lines.append(f"- Position resolution: `{args.position_res}`")
    lines.append(f"- Rotation resolution: `{args.rotation_res}`")
    lines.append(f"- Joint continuity threshold: `{args.joint_continuity_threshold}`")
    lines.append(f"- Dense joint validation step: `{DEFAULT_DENSE_JOINT_VALIDATION_STEP_RAD}`")
    lines.append(f"- Max time: `{args.max_time}`")
    lines.append(f"- Max iterations: `{args.max_iterations}`")
    lines.append(f"- Max attempts: `{args.max_attempts}`")
    lines.append(f"- Swap grasps: `{args.swap_grasps}`")
    lines.append(f"- Home left-tool offset: `{np.round(args.home_left_tool_offset, 4).tolist()}`")
    lines.append(f"- Home left-tool local yaw: `{args.home_left_tool_local_yaw}`")
    lines.append(f"- Auto home pose: `{args.auto_home_pose}`")
    lines.append(f"- Batch targets mode: `{args.batch_targets_mode}`")
    lines.append("")
    lines.append("Common home tool poses:")
    lines.append(
        f"- Left tool home pose: `"
        f"{np.round(np.asarray(common_start['mobile_base_from_tool0_left_home'][0], dtype=float) + np.asarray(args.home_left_tool_offset, dtype=float), 4).tolist()}`"
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Target | Bar body | Bar dims (m) | Goal xyz (m) | Path found | Validated success | Planning time (s) | Smoothing time (s) | Validation time (s) | Waypoints | Max dq (rad) | Collision-free | Failure reason | Validation plot | Video MP4 | Replay command |")
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |")
    for item in summaries:
        validation_plot = item["validation_plot"]
        validation_label = os.path.relpath(validation_plot, reports_dir()) if validation_plot else "-"
        if validation_plot:
            validation_cell = f"[{validation_label}]({validation_label})"
        else:
            validation_cell = "-"
        video_mp4 = item.get("video_mp4")
        video_label = os.path.relpath(video_mp4, reports_dir()) if video_mp4 else "-"
        if video_mp4:
            video_cell = f"[{video_label}]({video_label})"
        else:
            video_cell = "-"
        replay_command = item.get("replay_command")
        replay_cell = "-" if not replay_command else f"`{replay_command}`"
        max_dq_text = "-" if item["max_dq_rad"] is None else f"{item['max_dq_rad']:.4f}"
        lines.append(
            f"| {item['target_name']} | {item['active_bar_body_name']} | "
            f"`{np.round(item['bar_box_dims'], 4).tolist()}` | "
            f"`{np.round(item['goal_position'], 4).tolist()}` | "
            f"{'PASS' if item['path_found'] else 'FAIL'} | "
            f"{'PASS' if item['success'] else 'FAIL'} | "
            f"{item['planning_time_s']:.3f} | "
            f"{item['smoothing_time_s']:.3f} | "
            f"{item['validation_time_s']:.3f} | "
            f"{item['waypoints']} | "
            f"{max_dq_text} | "
            f"{'PASS' if item['collision_free'] else 'FAIL'} | "
            f"{item['collision_reason'] or '-'} | "
            f"{validation_cell} | "
            f"{video_cell} | "
            f"{replay_cell} |"
        )
    lines.append("")
    validation_plot_items = [item for item in summaries if item.get("validation_plot")]
    if validation_plot_items:
        lines.append("## Validation Plots")
        lines.append("")
        for item in validation_plot_items:
            validation_plot = item["validation_plot"]
            validation_label = os.path.relpath(validation_plot, reports_dir())
            lines.append(f"### {item['target_name']}")
            lines.append("")
            lines.append(f"![Validation plot for {item['target_name']}]({validation_label})")
            lines.append("")
    smoothing_plot_items = [item for item in summaries if item.get("smoothing_plot")]
    if smoothing_plot_items:
        lines.append("## Smoothing Plots")
        lines.append("")
        for item in smoothing_plot_items:
            smoothing_plot = item["smoothing_plot"]
            smoothing_label = os.path.relpath(smoothing_plot, reports_dir())
            lines.append(f"### {item['target_name']}")
            lines.append("")
            lines.append(f"![Smoothing comparison for {item['target_name']}]({smoothing_label})")
            lines.append("")
    successes = sum(1 for item in summaries if item["success"])
    lines.append(f"Validated Stage {args.stage} success: `{successes} / {len(summaries)}` targets.")
    lines.append("")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def summarize_endpoint_ik_diagnosis(target_name: str, spec: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    goal_pose = spec["goal_pose"]
    return {
        "target_name": target_name,
        "active_bar_body_name": spec["active_bar_mesh"]["body_name"],
        "goal_position": [float(v) for v in goal_pose[0]],
        "start_ik_ok": result.get("start_ik_ok"),
        "goal_ik_ok": result.get("goal_ik_ok"),
        "start_collision_ok": result.get("start_collision_ok"),
        "goal_collision_ok": result.get("goal_collision_ok"),
        "failure_reason": result.get("failure_reason"),
    }


def write_endpoint_ik_report(report_path: str, json_relpath: str, args, summaries: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append(f"# Real State Endpoint IK Diagnosis ({datetime.now().strftime('%Y%m%d_%H%M%S')})")
    lines.append("")
    if args.stage == 3:
        lines.append("This report checks endpoint dual-arm IK and then validates the solved endpoint configurations with robot collision.")
    else:
        lines.append("This report checks endpoint dual-arm IK only. Robot collision is not considered.")
    lines.append("")
    lines.append(f"- Design root: `{args.design_root}`")
    lines.append(f"- Targets: `{', '.join(args.targets)}`")
    lines.append(f"- Stage: `{args.stage}`")
    lines.append(f"- Endpoint mode: `{args.diagnose_endpoint_ik}`")
    lines.append(f"- Swap grasps: `{args.swap_grasps}`")
    lines.append(f"- Home left-tool offset: `{np.round(args.home_left_tool_offset, 4).tolist()}`")
    lines.append(f"- Home left-tool local yaw: `{args.home_left_tool_local_yaw}`")
    lines.append(f"- Auto home pose: `{args.auto_home_pose}`")
    lines.append(f"- Support JSON: `{json_relpath}`")
    lines.append("")
    lines.append("| Target | Bar body | Goal xyz (m) | Start IK | Goal IK | Start collision-free | Goal collision-free | Failure reason |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for item in summaries:
        start_text = "-" if item["start_ik_ok"] is None else ("PASS" if item["start_ik_ok"] else "FAIL")
        goal_text = "-" if item["goal_ik_ok"] is None else ("PASS" if item["goal_ik_ok"] else "FAIL")
        start_collision_text = "-" if item["start_collision_ok"] is None else ("PASS" if item["start_collision_ok"] else "FAIL")
        goal_collision_text = "-" if item["goal_collision_ok"] is None else ("PASS" if item["goal_collision_ok"] else "FAIL")
        lines.append(
            f"| {item['target_name']} | {item['active_bar_body_name']} | "
            f"`{np.round(item['goal_position'], 4).tolist()}` | "
            f"{start_text} | "
            f"{goal_text} | "
            f"{start_collision_text} | "
            f"{goal_collision_text} | "
            f"{item['failure_reason'] or '-'} |"
        )
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def hold_gui_pose(scene: Dict[str, Any], endpoint_name: str, bar_pose, joint_conf: np.ndarray) -> None:
    pp.set_pose(scene["bar_body"], bar_pose)
    pp.set_pose(scene["ghost_goal"], bar_pose)
    pp.set_joint_positions(scene["robot"], scene["arm_joints"], joint_conf)
    logger.info(
        "Holding %s endpoint IK pose in GUI at bar xyz=%s. Press Enter to continue.",
        endpoint_name,
        np.round(np.asarray(bar_pose[0], dtype=float), 4).tolist(),
    )
    pp.wait_if_gui()


def evaluate_endpoint_ik(
    scene: Dict[str, Any],
    endpoint_name: str,
    bar_pose,
    seed_conf: np.ndarray,
    grasp_bar_from_right,
    rng: np.random.Generator,
    endpoint_ik_attempts: int,
    joint_collision_checker=None,
) -> Dict[str, Any]:
    conf = solve_endpoint_dual_arm_ik(
        robot=scene["robot"],
        arm_joints=scene["arm_joints"],
        tool_link_left=scene["tool_link_left"],
        tool_link_right=scene["tool_link_right"],
        bar_pose=bar_pose,
        grasp_bar_from_left=scene["grasp_bar_from_left"],
        grasp_bar_from_right=grasp_bar_from_right,
        seed_conf=seed_conf,
        rng=rng,
        max_attempts=endpoint_ik_attempts,
    )
    if conf is None:
        return {
            "endpoint_name": endpoint_name,
            "bar_pose": bar_pose,
            "conf": None,
            "ik_ok": False,
            "collision_free": None,
            "collision_output": None,
        }

    collision_free = None
    collision_output = None
    if joint_collision_checker is not None:
        collision_free = not bool(joint_collision_checker(np.asarray(conf, dtype=float), diagnosis=True))

    return {
        "endpoint_name": endpoint_name,
        "bar_pose": bar_pose,
        "conf": np.asarray(conf, dtype=float),
        "ik_ok": True,
        "collision_free": collision_free,
        "collision_output": collision_output,
    }


def validate_auto_home_start_context(
    args: argparse.Namespace,
    common_start: Dict[str, Any],
    spec: Dict[str, Any],
    scene: Dict[str, Any],
    start_context: Dict[str, Any],
    joint_collision_checker=None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    if not args.auto_home_pose or args.auto_home_ik_candidates <= 0:
        return start_context

    grasp_bar_from_right = scene["grasp_bar_from_right"]
    if grasp_bar_from_right is None:
        return start_context

    validator_rng = rng if rng is not None else np.random.default_rng(args.random_seed)

    def ik_validator(bar_pose) -> bool:
        result = evaluate_endpoint_ik(
            scene=scene,
            endpoint_name="start_auto",
            bar_pose=bar_pose,
            seed_conf=np.asarray(scene["start_joint_values"], dtype=float),
            grasp_bar_from_right=grasp_bar_from_right,
            rng=validator_rng,
            endpoint_ik_attempts=args.endpoint_ik_attempts,
            joint_collision_checker=joint_collision_checker,
        )
        return bool(result["ik_ok"] and (joint_collision_checker is None or result["collision_free"]))

    validated_context = derive_start_pose_from_home_left_tool(
        spec,
        common_start,
        args.home_left_tool_offset,
        args.home_left_tool_local_yaw,
        auto_home_pose=True,
        ik_validator=ik_validator,
        num_geometric_candidates=args.auto_home_ik_candidates,
    )
    new_bar_pose = validated_context["world_from_bar_start"]
    if not pp.is_pose_close(
        new_bar_pose,
        start_context["world_from_bar_start"],
        pos_tolerance=1e-8,
        ori_tolerance=1e-8,
    ):
        logger.info("Updated %s auto home pose after IK validation.", spec["target_name"])
    pp.set_pose(scene["bar_body"], new_bar_pose)
    pp.set_pose(scene["ghost_start"], new_bar_pose)
    if scene.get("start_text_id") is not None:
        pp.remove_debug(scene["start_text_id"])
    scene["start_text_id"] = pp.add_text(
        scene.get("start_text_label", "Start"),
        new_bar_pose[0],
        color=(0.0, 0.8, 0.0, 1.0),
    )
    grasp_marker_bodies = scene.get("grasp_marker_bodies") or []
    if grasp_marker_bodies:
        pp.set_pose(grasp_marker_bodies[0], pp.multiply(new_bar_pose, scene["grasp_bar_from_left"]))
        if scene["grasp_bar_from_right"] is not None and len(grasp_marker_bodies) >= 3:
            pp.set_pose(grasp_marker_bodies[2], pp.multiply(new_bar_pose, scene["grasp_bar_from_right"]))
    scene["world_from_bar_start"] = new_bar_pose
    scene["start_pose"] = new_bar_pose
    scene["mobile_base_from_tool0_left_home"] = validated_context["mobile_base_from_tool0_left_home"]
    scene["mobile_base_from_tool0_right_start"] = validated_context["derived_right_tool_pose"]
    return validated_context


def validate_auto_home_start_context_with_temporary_scene(
    args: argparse.Namespace,
    common_start: Dict[str, Any],
    spec: Dict[str, Any],
    start_context: Dict[str, Any],
) -> Dict[str, Any]:
    if not args.auto_home_pose or args.auto_home_ik_candidates <= 0:
        return start_context

    scene = setup_planning_scene(
        grasp_json=spec["grasp_json"],
        start_state_json=args.start_state,
        end_state_json=spec["state_json"],
        use_gui=False,
        scene_spec=build_scene_spec_from_start_context(common_start, spec, start_context),
        swap_grasps=args.swap_grasps,
    )
    try:
        joint_collision_checker = None
        if args.stage == 3:
            env_obstacles = [body for body in scene["collision_obstacles"] if body != scene["robot"]]
            joint_collision_checker = get_joint_collision_fn(
                robot=scene["robot"],
                arm_joints=scene["arm_joints"],
                obstacle_bodies=env_obstacles,
                tool_link_left=scene["tool_link_left"],
                bar_body=scene["bar_body"],
                grasp_bar_from_left=scene["grasp_bar_from_left"],
            )
        return validate_auto_home_start_context(
            args=args,
            common_start=common_start,
            spec=spec,
            scene=scene,
            start_context=start_context,
            joint_collision_checker=joint_collision_checker,
            rng=np.random.default_rng(args.random_seed),
        )
    finally:
        teardown_planning_scene()


def run_endpoint_ik_diagnosis(
    args: argparse.Namespace,
    common_start: Dict[str, Any],
    spec: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    start_context = derive_start_pose_from_home_left_tool(
        spec,
        common_start,
        args.home_left_tool_offset,
        args.home_left_tool_local_yaw,
        auto_home_pose=args.auto_home_pose,
        ik_validator=None,
        num_geometric_candidates=args.auto_home_ik_candidates,
    )
    scene_spec = build_scene_spec_from_start_context(common_start, spec, start_context)
    scene = setup_planning_scene(
        grasp_json=spec["grasp_json"],
        start_state_json=args.start_state,
        end_state_json=spec["state_json"],
        use_gui=args.gui,
        scene_spec=scene_spec,
        swap_grasps=args.swap_grasps,
    )
    try:
        grasp_bar_from_right = scene["grasp_bar_from_right"]
        if grasp_bar_from_right is None:
            return {"start_conf": None, "goal_conf": None, "failure_reason": "missing_right_grasp"}
        rng = np.random.default_rng(args.random_seed)
        joint_collision_checker = None
        if args.stage == 3:
            env_obstacles = [body for body in scene["collision_obstacles"] if body != scene["robot"]]
            joint_collision_checker = get_joint_collision_fn(
                robot=scene["robot"],
                arm_joints=scene["arm_joints"],
                obstacle_bodies=env_obstacles,
                tool_link_left=scene["tool_link_left"],
                bar_body=scene["bar_body"],
                grasp_bar_from_left=scene["grasp_bar_from_left"],
            )

        start_context = validate_auto_home_start_context(
            args=args,
            common_start=common_start,
            spec=spec,
            scene=scene,
            start_context=start_context,
            joint_collision_checker=joint_collision_checker,
            rng=rng,
        )

        selected_endpoints = []
        if mode in {"start", "both"}:
            selected_endpoints.append(
                evaluate_endpoint_ik(
                    scene=scene,
                    endpoint_name="start",
                    bar_pose=scene["world_from_bar_start"],
                    seed_conf=np.asarray(scene["start_joint_values"], dtype=float),
                    grasp_bar_from_right=grasp_bar_from_right,
                    rng=rng,
                    endpoint_ik_attempts=args.endpoint_ik_attempts,
                    joint_collision_checker=joint_collision_checker,
                )
            )
            logger.info('start')
            logger.info(selected_endpoints[-1])
        if mode in {"goal", "both"}:
            selected_endpoints.append(
                evaluate_endpoint_ik(
                    scene=scene,
                    endpoint_name="goal",
                    bar_pose=scene["world_from_bar_goal"],
                    seed_conf=np.asarray(scene["end_joint_values"], dtype=float),
                    grasp_bar_from_right=grasp_bar_from_right,
                    rng=rng,
                    endpoint_ik_attempts=args.endpoint_ik_attempts,
                    joint_collision_checker=joint_collision_checker,
                )
            )
            logger.info('start')
            logger.info(selected_endpoints[-1])

        endpoint_by_name = {item["endpoint_name"]: item for item in selected_endpoints}
        start_result = endpoint_by_name.get("start")
        goal_result = endpoint_by_name.get("goal")
        start_conf = None if start_result is None else start_result["conf"]
        goal_conf = None if goal_result is None else goal_result["conf"]
        start_collision_ok = None if start_result is None else start_result["collision_free"]
        goal_collision_ok = None if goal_result is None else goal_result["collision_free"]

        failure_reason = None
        if mode in {"start", "both"} and start_conf is None:
            failure_reason = "start_ik_failure"
        if failure_reason is None and mode in {"goal", "both"} and goal_conf is None:
            failure_reason = "goal_ik_failure"
        if failure_reason is None and args.stage == 3 and mode in {"start", "both"} and start_collision_ok is False:
            failure_reason = "start_in_collision"
        if failure_reason is None and args.stage == 3 and mode in {"goal", "both"} and goal_collision_ok is False:
            failure_reason = "goal_in_collision"

        if args.gui:
            if start_result is not None and start_result["collision_output"]:
                logger.warning("Start endpoint collision diagnosis:\n%s", start_result["collision_output"])
            if goal_result is not None and goal_result["collision_output"]:
                logger.warning("Goal endpoint collision diagnosis:\n%s", goal_result["collision_output"])

            if mode == "start" and start_conf is not None:
                hold_gui_pose(scene, "start", scene["world_from_bar_start"], np.asarray(start_conf, dtype=float))
            elif mode == "goal" and goal_conf is not None:
                hold_gui_pose(scene, "goal", scene["world_from_bar_goal"], np.asarray(goal_conf, dtype=float))
            elif mode == "both":
                if goal_result is not None and goal_result["collision_output"] and goal_conf is not None:
                    hold_gui_pose(scene, "goal", scene["world_from_bar_goal"], np.asarray(goal_conf, dtype=float))
                elif start_result is not None and start_result["collision_output"] and start_conf is not None:
                    hold_gui_pose(scene, "start", scene["world_from_bar_start"], np.asarray(start_conf, dtype=float))
                elif goal_conf is not None:
                    hold_gui_pose(scene, "goal", scene["world_from_bar_goal"], np.asarray(goal_conf, dtype=float))
                elif start_conf is not None:
                    hold_gui_pose(scene, "start", scene["world_from_bar_start"], np.asarray(start_conf, dtype=float))

        return {
            "start_conf": None if start_conf is None else np.asarray(start_conf, dtype=float),
            "goal_conf": None if goal_conf is None else np.asarray(goal_conf, dtype=float),
            "start_ik_ok": None if mode == "goal" else bool(start_conf is not None),
            "goal_ik_ok": None if mode == "start" else bool(goal_conf is not None),
            "start_collision_ok": None if args.stage != 3 or mode == "goal" else start_collision_ok,
            "goal_collision_ok": None if args.stage != 3 or mode == "start" else goal_collision_ok,
            "start_collision_diagnosis": None if start_result is None else start_result["collision_output"],
            "goal_collision_diagnosis": None if goal_result is None else goal_result["collision_output"],
            "failure_reason": failure_reason,
        }
    finally:
        teardown_planning_scene()

def parse_args() -> argparse.Namespace:
    default_grasp_json, default_start_state, default_end_state = build_default_paths()
    targets_flag_provided = any(arg == "--targets" or arg.startswith("--targets=") for arg in sys.argv[1:])
    parser = argparse.ArgumentParser(description="Stage 1/2/3 benchmark on real design-study targets")
    parser.add_argument("--design-root", type=str, default=default_design_root(), help="Path to the design-study root")
    parser.add_argument("--robot-cell-json", type=str, default=None, help="Path to RobotCell.json")
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=DEFAULT_TARGET_NAMES,
        help="Real targets to benchmark",
    )
    parser.add_argument("--grasp-json", type=str, default=default_grasp_json, help="Common-start grasp JSON")
    parser.add_argument("--start-state", type=str, default=default_start_state, help="Common-start RobotCellState JSON")
    parser.add_argument("--end-state", type=str, default=default_end_state, help="Default end RobotCellState JSON")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=3, help="Planning stage to run")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
    parser.add_argument(
        "--lock-renderer-during-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lock the PyBullet renderer while the tree is being expanded, then show the result afterward",
    )
    parser.add_argument(
        "--visualize-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When GUI is enabled, open the slider-based path viewer after planning each target",
    )
    parser.add_argument("--position-res", type=float, default=0.005, help="Pose interpolation step in meters")
    parser.add_argument("--rotation-res", type=float, default=0.025, help="Pose interpolation step in radians")
    parser.add_argument("--endpoint-ik-attempts", type=int, default=20, help="Endpoint IK retry budget")
    parser.add_argument(
        "--joint-continuity-threshold",
        type=float,
        default=DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
        help="Max allowed joint jump in radians",
    )
    parser.add_argument("--max-time", type=float, default=30.0, help="Planner time limit in seconds")
    parser.add_argument("--max-iterations", type=int, default=2000, help="Planner iteration limit")
    parser.add_argument("--max-attempts", type=int, default=5, help="Planner restart attempts")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--home-left-tool-offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, DEFAULT_HOME_LEFT_TOOL_Z_OFFSET],
        metavar=("DX", "DY", "DZ"),
        help="Offset applied to the common left-tool home pose before deriving the start bar pose",
    )
    parser.add_argument(
        "--home-left-tool-local-yaw",
        type=float,
        default=0.0,
        help="Additional yaw rotation, in radians, applied in the local frame of the home left-tool pose",
    )
    parser.add_argument(
        "--auto-home-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-compute bar-axis rotation and flip for the home pose (ignores --home-left-tool-local-yaw when enabled)",
    )
    parser.add_argument(
        "--auto-home-ik-candidates",
        type=int,
        default=20,
        help="Number of top geometric candidates to IK-validate during auto home pose computation",
    )
    parser.add_argument("--include-built-bars", action="store_true", help="Import already-built bars into the scene")
    parser.add_argument(
        "--enable-built-bar-collision",
        action="store_true",
        help="Enable collision on imported built bars",
    )
    parser.add_argument("--swap-grasps", action="store_true", help="Swap the first two grasps loaded from each target grasp JSON")
    parser.add_argument(
        "--diagnose-start-collision",
        action="store_true",
        help="Alias for `--diagnose-endpoint-ik start` in Stage 3",
    )
    parser.add_argument(
        "--diagnose-endpoint-ik",
        choices=["start", "goal", "both"],
        default=None,
        help="Diagnose endpoint IK; in Stage 3 this also reports collision-free status and collision pairs",
    )
    parser.add_argument("--video-frame-step", type=int, default=1, help="Record every Nth waypoint into batch trajectory videos")
    parser.add_argument("--video-frame-sleep", type=float, default=0.02, help="Replay frame interval used to derive batch video FPS")
    args = parser.parse_args()
    args.batch_targets_mode = not targets_flag_provided
    return args


def main() -> None:
    args = parse_args()
    common_start = compute_common_start_context(args.grasp_json, args.start_state, args.end_state)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(support_dir(), f"real_state_study_{timestamp}.json")
    report_path = os.path.join(reports_dir(), f"real_state_study_report_{timestamp}.md")
    diagnose_mode = "start" if args.diagnose_start_collision else args.diagnose_endpoint_ik

    summaries: List[Dict[str, Any]] = []
    for target_name in args.targets:
        if target_name not in DESIGN_STUDY_BAR_SEQUENCE:
            raise ValueError(f"Unknown design-study target: {target_name}")
        spec = build_real_design_goal_spec(
            design_root=args.design_root,
            target_name=target_name,
            robot_cell_json=args.robot_cell_json,
            include_built_bars=args.include_built_bars,
            enable_built_bar_collision=args.enable_built_bar_collision,
            swap_grasps=args.swap_grasps,
        )
        # TODO here line 546 and line 566 can share one start context building
        if diagnose_mode is not None:
            result = run_endpoint_ik_diagnosis(args, common_start, spec, mode=diagnose_mode)
            summaries.append(summarize_endpoint_ik_diagnosis(target_name, spec, result))
            if args.stage == 3:
                logger.info(
                    "Target %s -> start_ik_ok=%s goal_ik_ok=%s start_collision_free=%s goal_collision_free=%s",
                    target_name,
                    summaries[-1]["start_ik_ok"],
                    summaries[-1]["goal_ik_ok"],
                    summaries[-1]["start_collision_ok"],
                    summaries[-1]["goal_collision_ok"],
                )
            else:
                logger.info(
                    "Target %s -> start_ik_ok=%s goal_ik_ok=%s",
                    target_name,
                    summaries[-1]["start_ik_ok"],
                    summaries[-1]["goal_ik_ok"],
                )
        else:
            start_context = derive_start_pose_from_home_left_tool(
                spec,
                common_start,
                args.home_left_tool_offset,
                args.home_left_tool_local_yaw,
                auto_home_pose=args.auto_home_pose,
                ik_validator=None,
                num_geometric_candidates=args.auto_home_ik_candidates,
            )
            start_context = validate_auto_home_start_context_with_temporary_scene(
                args,
                common_start,
                spec,
                start_context,
            )
            scene_spec = build_scene_spec_from_start_context(common_start, spec, start_context)
            stage_runner = {
                1: run_stage1_trial,
                2: run_stage2_trial,
                3: run_stage3_trial,
            }[args.stage]
            result = stage_runner(
                grasp_json=spec["grasp_json"],
                start_state_json=args.start_state,
                end_state_json=spec["state_json"],
                use_gui=args.gui,
                position_res=args.position_res,
                rotation_res=args.rotation_res,
                endpoint_ik_attempts=args.endpoint_ik_attempts,
                joint_continuity_threshold_rad=args.joint_continuity_threshold,
                max_time=args.max_time,
                max_iterations=args.max_iterations,
                max_attempts=args.max_attempts,
                random_seed=args.random_seed,
                lock_renderer_during_search=args.lock_renderer_during_search,
                scene_spec=scene_spec,
                validation_reports_dir=support_dir(),
                swap_grasps=args.swap_grasps,
            )
            summary = summarize_result(target_name, spec, result)
            if result.get("path_before_smoothing") is not None and result.get("path") is not None:
                smoothing_plot_path = save_smoothing_comparison_plot(
                    out_path=os.path.join(
                        support_dir(),
                        f"real_state_study_stage{args.stage}_{target_name}_{timestamp}_smoothing.png",
                    ),
                    target_name=target_name,
                    path_before_smoothing=result["path_before_smoothing"],
                    path_after_smoothing=result["path"],
                    smoothing_profile=result.get("smoothing"),
                )
                summary["smoothing_plot"] = smoothing_plot_path
                if smoothing_plot_path is not None:
                    logger.info("Saved target %s smoothing comparison plot: %s", target_name, smoothing_plot_path)
            if result.get("path") is not None and result.get("path_confs") is not None:
                trajectory_json_path = os.path.join(
                    support_dir(),
                    f"real_state_study_stage{args.stage}_{target_name}_{timestamp}_trajectory.json",
                )
                trajectory_metadata_json_path = os.path.join(
                    support_dir(),
                    f"real_state_study_stage{args.stage}_{target_name}_{timestamp}_trajectory_metadata.json",
                )
                save_replay_bundle(
                    scene=result["scene"],
                    spec=spec,
                    joint_path=result["path_confs"],
                    pose_path=result["path"],
                    trajectory_json_path=trajectory_json_path,
                    metadata_json_path=trajectory_metadata_json_path,
                )
                replay_command = build_replay_command(
                    trajectory_json_path=trajectory_json_path,
                    metadata_json_path=trajectory_metadata_json_path,
                    grasp_json=spec["grasp_json"],
                    start_state_json=args.start_state,
                    end_state_json=spec["state_json"],
                )
                summary["trajectory_json"] = trajectory_json_path
                summary["trajectory_metadata_json"] = trajectory_metadata_json_path
                summary["replay_command"] = replay_command
                logger.info("Saved target %s replay trajectory: %s", target_name, trajectory_json_path)
            if (
                not args.gui
                and result.get("path") is not None
                and result.get("path_confs") is not None
            ):
                video_path = record_trajectory_video(
                    scene=result["scene"],
                    joint_path=result["path_confs"],
                    pose_path=result["path"],
                    out_path=os.path.join(
                        support_dir(),
                        f"real_state_study_stage{args.stage}_{target_name}_{timestamp}_trajectory.mp4",
                    ),
                    frame_step=args.video_frame_step,
                    frame_sleep=args.video_frame_sleep,
                )
                summary["video_mp4"] = video_path
                if video_path is not None:
                    logger.info("Saved target %s trajectory video: %s", target_name, video_path)
            summaries.append(summary)
            logger.info(
                "Target %s -> success=%s runtime=%.3fs waypoints=%d",
                target_name,
                summaries[-1]["success"],
                summaries[-1]["runtime_s"],
                summaries[-1]["waypoints"],
            )
            if args.gui and args.visualize_path:
                run_visualization_loop(
                    result["scene"]["bar_body"],
                    result["path"],
                    result["scene"]["cid"],
                    robot=result["scene"]["robot"],
                    arm_joints=result["scene"]["arm_joints"],
                    path_confs=result["path_confs"],
                    validation=result.get("validation"),
                    scene=result.get("scene"),
                )
            teardown_planning_scene()

    payload = {
        "design_root": args.design_root,
        "targets": args.targets,
        "stage": args.stage,
        "auto_home_pose": args.auto_home_pose,
        "auto_home_ik_candidates": args.auto_home_ik_candidates,
        "batch_targets_mode": args.batch_targets_mode,
        "common_start_pose": {
            "position": [
                float(v)
                for v in (
                    np.asarray(common_start["mobile_base_from_tool0_left_home"][0], dtype=float)
                    + np.asarray(args.home_left_tool_offset, dtype=float)
                )
            ],
            "quaternion": [float(v) for v in common_start["mobile_base_from_tool0_left_home"][1]],
        },
        "mode": f"endpoint_ik_{diagnose_mode}" if diagnose_mode is not None else f"stage{args.stage}_planning",
        "results": summaries,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    if diagnose_mode is not None:
        report_path = os.path.join(reports_dir(), f"real_state_endpoint_ik_report_{timestamp}.md")
        write_endpoint_ik_report(report_path, os.path.relpath(json_path, reports_dir()), args, summaries)
        logger.info("Saved real-state endpoint-IK report: %s", report_path)
    else:
        write_report(report_path, os.path.relpath(json_path, reports_dir()), args, common_start, summaries)
        logger.info("Saved real-state study report: %s", report_path)
    logger.info("Saved real-state study JSON: %s", json_path)


if __name__ == "__main__":
    main()
