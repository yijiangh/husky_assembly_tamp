from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pybullet
import pybullet_planning as pp

from husky_assembly_tamp.motion_planner.stage1.minimal_rrt import (
    DEFAULT_HOME_LEFT_TOOL_Z_OFFSET,
    DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
    GDRIVE_DATA_DIRECTORY,
    HUSKY_DUAL_ARM_JOINT_NAMES,
    MOBILE_BASE_FROM_TOOL0_LEFT_HOME,
    build_gdrive_bar_action_scene_spec,
    build_gdrive_scene_spec,
    run_stage1_trial,
    run_stage2_trial,
    run_stage3_trial,
    run_visualization_loop,
    teardown_planning_scene,
)
from husky_assembly_tamp.motion_planner.stage1.path_validation import DEFAULT_DENSE_JOINT_VALIDATION_STEP_RAD
from husky_assembly_tamp.motion_planner.stage1.trajectory_io import save_path_as_joint_trajectory
from husky_assembly_tamp.utils.params import DATA_DIR
from husky_assembly_tamp.utils.util import setup_logger


logger = setup_logger("stage1_real_state_study", file_mode="w")


# Used by --gdrive when no --targets are passed (one of --gdrive / --gdrive-bar-action is required).
DEFAULT_TARGET_NAMES = ["G1", "G2", "G3", "G4", "V1", "V2", "H1", "D1", "V3"]

# gdrive-convention defaults (2026-05+). --targets are state filenames under
# GDRIVE_DATA_DIRECTORY/<gdrive_problem>/RobotCellStates/ (--gdrive) or
# BarActions/ (--gdrive-bar-action).
GDRIVE_DEFAULT_PROBLEM = "2026-05-08_dual-arm_transfer_test"
GDRIVE_DEFAULT_TARGETS = ["B3_approach.json"]
GDRIVE_DEFAULT_BAR_ACTION_TARGETS = ["B1.json"]


def compute_gdrive_common_start() -> Dict[str, Any]:
    """Report-header context for gdrive datasets (no separate 'common start state')."""
    return {
        "mobile_base_from_tool0_left_home": MOBILE_BASE_FROM_TOOL0_LEFT_HOME,
    }


def build_gdrive_target_spec(
    state_arg: str,
    problem: Optional[str] = None,
    *,
    include_built_bars: bool = True,
    enable_built_bar_collision: bool = True,
) -> Dict[str, Any]:
    """Build a target spec from a single gdrive RobotCellState.

    All poses (goal_pose, grasp_targets, built_bars[*].pose) are in mobile-base
    frame; the husky URDF stays at world origin in pybullet.
    """
    gdrive_spec = build_gdrive_scene_spec(
        state_arg,
        problem=problem,
        include_env_bars=include_built_bars,
        include_active_extras=include_built_bars,
    )
    state_path = gdrive_spec["_gdrive_state_path"]
    target_name = os.path.splitext(os.path.basename(state_path))[0]
    if not enable_built_bar_collision and gdrive_spec["built_bars"]:
        gdrive_spec["built_bars"] = [{**b, "collision": False} for b in gdrive_spec["built_bars"]]
    return {
        "target_name": target_name,
        "goal_pose": gdrive_spec["world_from_bar_goal"],
        "grasp_targets": gdrive_spec["grasp_targets"],
        "active_bar_mesh": gdrive_spec["active_bar_mesh"],
        "built_bars": gdrive_spec["built_bars"],
        "scene_spec": gdrive_spec,
    }


def build_gdrive_bar_action_target_spec(
    action_arg: str,
    movement: str | int = "M1",
    problem: Optional[str] = None,
    *,
    include_built_bars: bool = False,
) -> Dict[str, Any]:
    """Build a target spec from one BarAction movement."""
    bar_action_spec = build_gdrive_bar_action_scene_spec(
        action_arg,
        movement=movement,
        problem=problem,
        include_built_bars=include_built_bars,
    )
    target_name = (
        f"{os.path.splitext(os.path.basename(bar_action_spec['_gdrive_bar_action_path']))[0]}"
        f"_{bar_action_spec['_gdrive_bar_action_movement']}"
    )
    return {
        "target_name": target_name,
        "goal_pose": bar_action_spec["world_from_bar_goal"],
        "grasp_targets": bar_action_spec["grasp_targets"],
        "active_bar_mesh": bar_action_spec["active_bar_mesh"],
        "built_bars": bar_action_spec["built_bars"],
        "scene_spec": bar_action_spec,
    }


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


def summarize_result(target_name: str, spec: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    validation = result.get("validation", {})
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
        "max_dq_rad": validation.get("joint_continuity_max_delta_rad"),
        "joint_continuity_ok": validation.get("joint_continuity_ok"),
        "collision_free": validation.get("collision_free"),
        "collision_reason": validation.get("failure_reason", result.get("failure_reason")),
        "validation_plot": validation.get("plot_path"),
        "video_mp4": result.get("video_mp4"),
        "trajectory_json": result.get("trajectory_json"),
        "trajectory_metadata_json": result.get("trajectory_metadata_json"),
        "replay_command": result.get("replay_command"),
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
    successes = sum(1 for item in summaries if item["success"])
    lines.append(f"Validated Stage {args.stage} success: `{successes} / {len(summaries)}` targets.")
    lines.append("")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--video-frame-step", type=int, default=1, help="Record every Nth waypoint into batch trajectory videos")
    parser.add_argument("--video-frame-sleep", type=float, default=0.02, help="Replay frame interval used to derive batch video FPS")
    # gdrive convention (2026-05+): single state file per target, no GraspTargets JSON.
    parser.add_argument(
        "--gdrive",
        action="store_true",
        help=("Use the gdrive dataset convention. --targets are state filenames "
              "(e.g. 'B3_approach.json' or 'B3_approach') under "
              "GDRIVE_DATA_DIRECTORY/<gdrive-problem>/RobotCellStates/."),
    )
    parser.add_argument(
        "--gdrive-bar-action",
        action="store_true",
        help=("Use gdrive BarAction inputs. --targets are BarAction filenames "
              "under GDRIVE_DATA_DIRECTORY/<gdrive-problem>/BarActions/."),
    )
    parser.add_argument(
        "--gdrive-problem",
        type=str,
        default=GDRIVE_DEFAULT_PROBLEM,
        help=f"Dataset directory under GDRIVE_DATA_DIRECTORY (default: {GDRIVE_DEFAULT_PROBLEM!r}).",
    )
    parser.add_argument(
        "--movement",
        type=str,
        default="M1",
        help="Movement selector for --gdrive-bar-action: index string, exact id, or substring (default: M1).",
    )
    args = parser.parse_args()
    args.batch_targets_mode = not targets_flag_provided
    if args.gdrive and args.gdrive_bar_action:
        raise ValueError("--gdrive and --gdrive-bar-action are mutually exclusive.")
    if not (args.gdrive or args.gdrive_bar_action):
        raise ValueError("Exactly one of --gdrive / --gdrive-bar-action is required.")
    if args.gdrive:
        # Default targets in gdrive mode if user didn't specify.
        if not targets_flag_provided:
            args.targets = list(GDRIVE_DEFAULT_TARGETS)
        # Normalize: accept 'B3_approach' and 'B3_approach.json'.
        args.targets = [t if t.endswith(".json") else f"{t}.json" for t in args.targets]
        # Override design_root for the report header.
        args.design_root = os.path.join(GDRIVE_DATA_DIRECTORY, args.gdrive_problem)
    if args.gdrive_bar_action:
        # Default targets in BarAction mode if user didn't specify.
        if not targets_flag_provided:
            args.targets = list(GDRIVE_DEFAULT_BAR_ACTION_TARGETS)
        args.targets = [t if t.endswith(".json") else f"{t}.json" for t in args.targets]
        args.design_root = os.path.join(GDRIVE_DATA_DIRECTORY, args.gdrive_problem)
        if args.movement.isdigit():
            args.movement = int(args.movement)
    return args


def main() -> None:
    args = parse_args()
    common_start = compute_gdrive_common_start()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(support_dir(), f"real_state_study_{timestamp}.json")
    report_path = os.path.join(reports_dir(), f"real_state_study_report_{timestamp}.md")

    summaries: List[Dict[str, Any]] = []
    for target_name in args.targets:
        if args.gdrive:
            spec = build_gdrive_target_spec(
                target_name,
                problem=args.gdrive_problem,
                include_built_bars=args.include_built_bars,
                enable_built_bar_collision=args.enable_built_bar_collision,
            )
        else:
            spec = build_gdrive_bar_action_target_spec(
                target_name,
                movement=args.movement,
                problem=args.gdrive_problem,
                include_built_bars=args.include_built_bars,
            )
        target_name = spec["target_name"]
        scene_spec = spec["scene_spec"]
        stage_runner = {
            1: run_stage1_trial,
            2: run_stage2_trial,
            3: run_stage3_trial,
        }[args.stage]
        result = stage_runner(
            scene_spec=scene_spec,
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
            validation_reports_dir=support_dir(),
        )
        summary = summarize_result(target_name, spec, result)
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
        "mode": f"stage{args.stage}_planning",
        "results": summaries,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    write_report(report_path, os.path.relpath(json_path, reports_dir()), args, common_start, summaries)
    logger.info("Saved real-state study report: %s", report_path)
    logger.info("Saved real-state study JSON: %s", json_path)


if __name__ == "__main__":
    main()
