"""Sample front-region Stage 3 goal poses, filter them with endpoint IK, and benchmark planning."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet
import pybullet_planning as pp

from husky_assembly_tamp.motion_planner.stage1.minimal_rrt import (
    DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
    DEFAULT_USE_ANGLE_NORMALIZATION,
    HUSKY_DUAL_ARM_JOINT_NAMES,
    HUSKY_DUAL_SRDF_PATH,
    HUSKY_DUAL_URDF_PATH,
    STAGE3_GRASP_MASK_LINKS,
    build_default_paths,
    build_validation_joint_path,
    get_joint_collision_fn,
    plan_pose_rrt,
    setup_stage1_scene,
    solve_endpoint_dual_arm_ik,
    teardown_stage1_scene,
)
from husky_assembly_tamp.motion_planner.stage1.path_validation import validate_stage_trajectory
from husky_assembly_tamp.motion_planner.stage1.trajectory_io import save_path_as_joint_trajectory
from husky_assembly_tamp.utils.util import setup_logger


logger = setup_logger("stage1_goal_pose_study")

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
SUPPORT_DIRNAME = "_support"

PoseLike = Tuple[np.ndarray, np.ndarray]
FullConf = np.ndarray


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def support_dir(outdir: str) -> str:
    path = os.path.join(outdir, SUPPORT_DIRNAME)
    ensure_dir(path)
    return path


def to_pose_dict(pose: PoseLike) -> Dict[str, List[float]]:
    return {
        "position": [float(v) for v in np.asarray(pose[0], dtype=float)],
        "quaternion": [float(v) for v in np.asarray(pose[1], dtype=float)],
    }


def compose_quaternions(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    _, quat = pybullet.multiplyTransforms([0.0, 0.0, 0.0], left, [0.0, 0.0, 0.0], right)
    return np.asarray(quat, dtype=float)


def quat_to_matrix(quat: Sequence[float]) -> np.ndarray:
    return np.asarray(pybullet.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)


def pose_list_to_dicts(path: Optional[Sequence[PoseLike]]) -> Optional[List[Dict[str, List[float]]]]:
    if path is None:
        return None
    return [to_pose_dict((np.asarray(pose[0], dtype=float), np.asarray(pose[1], dtype=float))) for pose in path]


def save_scene_overview(
    scene: Dict[str, Any],
    goal_pose: PoseLike,
    out_path: str,
    width: int = 1024,
    height: int = 768,
) -> str:
    import matplotlib.pyplot as plt

    pp.set_pose(scene["bar_body"], scene["start_pose"])
    pp.set_pose(scene["ghost_start"], scene["start_pose"])
    pp.set_pose(scene["ghost_goal"], goal_pose)
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
    _, _, rgba, _, _ = pybullet.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=pybullet.ER_TINY_RENDERER,
        physicsClientId=scene["cid"],
    )
    image = np.reshape(np.asarray(rgba, dtype=np.uint8), (height, width, 4))
    plt.imsave(out_path, image)
    return out_path


def record_trajectory_video(
    scene: Dict[str, Any],
    goal_pose: PoseLike,
    joint_path: Sequence[FullConf],
    pose_path: Sequence[PoseLike],
    out_path: str,
    frame_step: int = 1,
    frame_sleep: float = 0.02,
) -> Optional[str]:
    if not pose_path or not joint_path:
        return None
    if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
        return None

    pp.set_pose(scene["ghost_start"], scene["start_pose"])
    pp.set_pose(scene["ghost_goal"], goal_pose)
    pp.set_pose(scene["bar_body"], pose_path[0])
    pp.set_joint_positions(scene["robot"], scene["arm_joints"], joint_path[0])
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0, physicsClientId=scene["cid"])

    with pp.VideoSaver(out_path):
        for index, (conf, pose) in enumerate(zip(joint_path, pose_path)):
            if index % max(1, frame_step) != 0 and index != len(joint_path) - 1:
                continue
            pp.set_joint_positions(scene["robot"], scene["arm_joints"], conf)
            pp.set_pose(scene["bar_body"], pose)
            pybullet.stepSimulation(physicsClientId=scene["cid"])
            time.sleep(max(0.0, frame_sleep))
    if not os.path.isfile(out_path):
        return None
    return out_path


def choose_tilt_deg(rng: np.random.Generator, sample_index: int, max_tilt_deg: float) -> float:
    if sample_index % 3 == 0:
        return float(rng.uniform(5.0, min(20.0, max_tilt_deg)))
    if sample_index % 3 == 1:
        return float(rng.uniform(20.0, min(55.0, max_tilt_deg)))
    return float(rng.uniform(min(55.0, max_tilt_deg), max_tilt_deg))


def sample_front_goal_pose(
    base_goal_pose: PoseLike,
    rng: np.random.Generator,
    sample_index: int,
    max_tilt_deg: float,
) -> Tuple[PoseLike, Dict[str, float]]:
    base_pos = np.asarray(base_goal_pose[0], dtype=float)
    base_quat = np.asarray(base_goal_pose[1], dtype=float)
    rot = quat_to_matrix(base_quat)
    bar_axis = rot[:, 2]
    up_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    tilt_axis = np.cross(bar_axis, up_axis)
    tilt_axis_norm = float(np.linalg.norm(tilt_axis))
    if tilt_axis_norm < 1e-8:
        tilt_axis = rot[:, 0]
        tilt_axis_norm = float(np.linalg.norm(tilt_axis))
    tilt_axis = tilt_axis / max(tilt_axis_norm, 1e-8)

    tilt_deg = choose_tilt_deg(rng, sample_index, max_tilt_deg)
    tilt_rad = float(np.deg2rad(tilt_deg))
    yaw_deg = float(rng.uniform(-12.0, 12.0))
    yaw_rad = float(np.deg2rad(yaw_deg))

    tilt_quat = np.asarray(pybullet.getQuaternionFromAxisAngle(tilt_axis.tolist(), tilt_rad), dtype=float)
    yaw_quat = np.asarray(pybullet.getQuaternionFromAxisAngle(up_axis.tolist(), yaw_rad), dtype=float)
    new_quat = compose_quaternions(yaw_quat, compose_quaternions(tilt_quat, base_quat))

    raised_center_z = base_pos[2] + 0.45 * np.sin(tilt_rad)
    new_pos = np.array(
        [
            float(rng.uniform(base_pos[0] - 0.10, base_pos[0] + 0.08)),
            float(rng.uniform(base_pos[1] - 0.14, base_pos[1] + 0.14)),
            float(np.clip(raised_center_z + rng.uniform(-0.04, 0.06), 0.25, 0.95)),
        ],
        dtype=float,
    )
    return (new_pos, new_quat), {"tilt_deg": tilt_deg, "yaw_deg": yaw_deg}


def solve_start_and_collision_context(
    scene: Dict[str, Any],
    endpoint_ik_attempts: int,
    random_seed: int,
    use_angle_normalization: bool,
) -> Tuple[FullConf, Any]:
    grasp_bar_from_right = scene["grasp_bar_from_right"]
    if grasp_bar_from_right is None:
        raise ValueError("Stage 3 goal-pose study requires both left and right grasp targets.")

    rng = np.random.default_rng(random_seed)
    start_conf = solve_endpoint_dual_arm_ik(
        robot=scene["robot"],
        arm_joints=scene["arm_joints"],
        tool_link_left=scene["tool_link_left"],
        tool_link_right=scene["tool_link_right"],
        bar_pose=scene["start_pose"],
        grasp_bar_from_left=scene["grasp_bar_from_left"],
        grasp_bar_from_right=grasp_bar_from_right,
        seed_conf=scene["start_joint_values"],
        rng=rng,
        max_attempts=endpoint_ik_attempts,
        use_angle_normalization=use_angle_normalization,
    )
    if start_conf is None:
        raise RuntimeError("Stage 3 study start pose has no valid dual-arm IK solution.")

    env_obstacles = [body for body in scene["collision_obstacles"] if body != scene["robot"]]
    joint_collision_fn = get_joint_collision_fn(
        robot=scene["robot"],
        arm_joints=scene["arm_joints"],
        obstacle_bodies=env_obstacles,
        tool_link_left=scene["tool_link_left"],
        bar_body=scene["bar_body"],
        grasp_bar_from_left=scene["grasp_bar_from_left"],
    )
    if joint_collision_fn(start_conf):
        raise RuntimeError("Stage 3 study start configuration is in collision.")
    return start_conf, joint_collision_fn


def check_goal_feasibility(
    scene: Dict[str, Any],
    goal_pose: PoseLike,
    endpoint_ik_attempts: int,
    random_seed: int,
    joint_collision_fn,
    use_angle_normalization: bool,
) -> Dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    goal_conf = solve_endpoint_dual_arm_ik(
        robot=scene["robot"],
        arm_joints=scene["arm_joints"],
        tool_link_left=scene["tool_link_left"],
        tool_link_right=scene["tool_link_right"],
        bar_pose=goal_pose,
        grasp_bar_from_left=scene["grasp_bar_from_left"],
        grasp_bar_from_right=scene["grasp_bar_from_right"],
        seed_conf=scene["end_joint_values"],
        rng=rng,
        max_attempts=endpoint_ik_attempts,
        use_angle_normalization=use_angle_normalization,
    )
    if goal_conf is None:
        return {"accepted": False, "reason": "endpoint_ik_failure", "goal_conf": None}

    in_collision = bool(joint_collision_fn(goal_conf))
    if in_collision:
        return {"accepted": False, "reason": "goal_conf_in_collision", "goal_conf": None}

    return {"accepted": True, "reason": "accepted", "goal_conf": goal_conf}


def run_stage3_for_goal(
    scene: Dict[str, Any],
    candidate_id: int,
    goal_pose: PoseLike,
    start_conf: FullConf,
    goal_conf: FullConf,
    joint_collision_fn,
    args,
    random_seed: int,
    reports_dir: str,
    timestamp: str,
) -> Dict[str, Any]:
    ik_context = {
        "robot": scene["robot"],
        "arm_joints": scene["arm_joints"],
        "tool_link_left": scene["tool_link_left"],
        "tool_link_right": scene["tool_link_right"],
        "grasp_bar_from_left": scene["grasp_bar_from_left"],
        "grasp_bar_from_right": scene["grasp_bar_from_right"],
    }
    planner_profile: Dict[str, Any] = {}
    t0 = time.perf_counter()
    path, path_confs = plan_pose_rrt(
        robot=scene["robot"],
        bar_body=scene["bar_body"],
        obstacle_bodies=scene["collision_obstacles"],
        start_pose=scene["start_pose"],
        goal_pose=goal_pose,
        start_conf=start_conf,
        goal_conf=goal_conf,
        dist_metric=args.dist_metric,
        goal_sample_prob=args.goal_bias,
        position_res=args.position_res,
        rotation_res=args.rotation_res,
        random_seed=random_seed,
        max_time=args.max_time,
        max_iterations=args.max_iterations,
        max_attempts=args.max_attempts,
        enable_collision=True,
        enable_ik=True,
        ik_context=ik_context,
        joint_collision_fn=joint_collision_fn,
        joint_continuity_threshold_rad=args.joint_continuity_threshold,
        use_angle_normalization=args.use_angle_normalization,
        use_draw=False,
        debug_tree_out=None,
        profile_out=planner_profile,
    )
    runtime_s = time.perf_counter() - t0
    joint_path, joint_path_source, joint_path_reason = build_validation_joint_path(
        scene=scene,
        path=path,
        path_confs=path_confs,
        start_conf=start_conf,
        endpoint_ik_attempts=args.endpoint_ik_attempts,
        random_seed=random_seed,
    )
    validation = validate_stage_trajectory(
        stage=3,
        scene=scene,
        path=path,
        joint_path=joint_path,
        joint_path_source=joint_path_source,
        joint_path_reason=joint_path_reason,
        urdf_path=HUSKY_DUAL_URDF_PATH,
        srdf_path=HUSKY_DUAL_SRDF_PATH,
        grasp_mask_links=STAGE3_GRASP_MASK_LINKS,
        joint_continuity_threshold_rad=args.joint_continuity_threshold,
        use_angle_normalization=args.use_angle_normalization,
        reports_dir=reports_dir,
    )
    success = bool(path is not None and validation.get("joint_continuity_ok") and validation.get("collision_free"))
    artifact_prefix = f"goal_pose_study_{timestamp}_candidate{candidate_id:02d}"
    screenshot_path = save_scene_overview(
        scene=scene,
        goal_pose=goal_pose,
        out_path=os.path.join(reports_dir, f"{artifact_prefix}_overview.png"),
        width=args.capture_width,
        height=args.capture_height,
    )
    trajectory_json_path: Optional[str] = None
    metadata_json_path: Optional[str] = None
    video_path: Optional[str] = None
    if path is not None and path_confs is not None:
        trajectory_json_path = os.path.join(reports_dir, f"{artifact_prefix}_JointTrajectory.json")
        save_path_as_joint_trajectory(path_confs, HUSKY_DUAL_ARM_JOINT_NAMES, trajectory_json_path)
        metadata_json_path = os.path.join(reports_dir, f"{artifact_prefix}_metadata.json")
        with open(metadata_json_path, "w") as f:
            json.dump(
                {
                    "candidate_id": candidate_id,
                    "start_pose": to_pose_dict(scene["start_pose"]),
                    "goal_pose": to_pose_dict(goal_pose),
                    "pose_path": pose_list_to_dicts(path),
                    "joint_trajectory_json": os.path.basename(trajectory_json_path),
                    "joint_names": list(HUSKY_DUAL_ARM_JOINT_NAMES),
                    "validation_plot": (None if validation.get("plot_path") is None else os.path.basename(validation["plot_path"])),
                },
                f,
                indent=2,
            )
        if args.capture_artifacts:
            video_path = record_trajectory_video(
                scene=scene,
                goal_pose=goal_pose,
                joint_path=path_confs,
                pose_path=path,
                out_path=os.path.join(reports_dir, f"{artifact_prefix}.mp4"),
                frame_step=args.video_frame_step,
                frame_sleep=args.video_frame_sleep,
            )
    return {
        "path_found": bool(path is not None),
        "success": success,
        "runtime_s": float(runtime_s),
        "path_waypoints": 0 if path is None else len(path),
        "planner_profile": planner_profile,
        "validation": validation,
        "artifacts": {
            "overview_png": screenshot_path,
            "video_mp4": video_path,
            "trajectory_json": trajectory_json_path,
            "metadata_json": metadata_json_path,
        },
    }


def write_report(
    report_path: str,
    timestamp: str,
    args,
    scene: Dict[str, Any],
    candidate_results: Sequence[Dict[str, Any]],
    planner_results: Sequence[Dict[str, Any]],
    support_json_path: str,
) -> None:
    accepted_count = sum(int(item["accepted"]) for item in candidate_results)
    success_count = sum(int(item["planner"]["success"]) for item in planner_results)
    report_dir = os.path.dirname(report_path)
    lines: List[str] = []
    lines.append(f"# Stage 3 Front Goal Pose Study ({timestamp})")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This report keeps the current Stage 3 start pose fixed, samples front-region goal poses with increasing tilt, filters them with endpoint dual-arm IK plus collision, and runs the Stage 3 planner on the accepted subset.")
    lines.append("")
    lines.append(f"- Candidate attempts: `{args.candidate_attempts}`")
    lines.append(f"- Accepted goals requested: `{args.accepted_goals}`")
    lines.append(f"- Position resolution: `{args.position_res} m`")
    lines.append(f"- Rotation resolution: `{args.rotation_res} rad`")
    lines.append(f"- Joint continuity threshold: `{args.joint_continuity_threshold} rad`")
    lines.append(f"- Endpoint IK attempts: `{args.endpoint_ik_attempts}`")
    lines.append(f"- Random seed: `{args.random_seed}`")
    lines.append(f"- Support JSON: `{os.path.relpath(support_json_path, os.path.dirname(report_path))}`")
    lines.append("")
    lines.append("Reference poses:")
    lines.append("")
    lines.append(f"- Start pose: `{np.round(scene['start_pose'][0], 4).tolist()}`")
    lines.append(f"- Baseline goal pose: `{np.round(scene['end_pose'][0], 4).tolist()}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Feasibility Filter")
    lines.append("")
    lines.append("| Candidate | Source | Tilt (deg) | Yaw (deg) | Position xyz (m) | Endpoint IK | Accepted | Reason |")
    lines.append("| --- | --- | ---: | ---: | --- | --- | --- | --- |")
    for item in candidate_results:
        pos = item["goal_pose"]["position"]
        lines.append(
            f"| {item['candidate_id']} | {item['source']} | {item['tilt_deg']:.1f} | {item['yaw_deg']:.1f} | "
            f"`[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]` | "
            f"{'PASS' if item['reason'] != 'endpoint_ik_failure' else 'FAIL'} | "
            f"{'PASS' if item['accepted'] else 'FAIL'} | {item['reason']} |"
        )
    lines.append("")
    lines.append(f"Accepted `{accepted_count}` of `{len(candidate_results)}` sampled goals for full planning.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Stage 3 Results")
    lines.append("")
    lines.append("| Candidate | Source | Tilt (deg) | Position xyz (m) | Path found | Validated success | Runtime (s) | Waypoints | Max dq (rad) | Collision-free |")
    lines.append("| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |")
    for item in planner_results:
        planner = item["planner"]
        validation = planner["validation"]
        pos = item["goal_pose"]["position"]
        lines.append(
            f"| {item['candidate_id']} | {item['source']} | {item['tilt_deg']:.1f} | "
            f"`[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]` | "
            f"{'PASS' if planner['path_found'] else 'FAIL'} | "
            f"{'PASS' if planner['success'] else 'FAIL'} | "
            f"{planner['runtime_s']:.3f} | {planner['path_waypoints']} | "
            f"{float(validation.get('joint_continuity_max_delta_rad') or 0.0):.4f} | "
            f"{'PASS' if validation.get('collision_free') else 'FAIL'} |"
        )
    lines.append("")
    lines.append(f"Validated Stage 3 success: `{success_count} / {len(planner_results)}` accepted goals.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Goal Visuals")
    lines.append("")
    for item in planner_results:
        planner = item["planner"]
        artifacts = planner.get("artifacts") or {}
        overview_png = artifacts.get("overview_png")
        video_mp4 = artifacts.get("video_mp4")
        trajectory_json = artifacts.get("trajectory_json")
        metadata_json = artifacts.get("metadata_json")
        lines.append(f"### Candidate {item['candidate_id']} ({item['tilt_deg']:.1f} deg)")
        lines.append("")
        if overview_png:
            lines.append(f"![Candidate {item['candidate_id']} Overview]({os.path.relpath(overview_png, report_dir)})")
            lines.append("")
        if video_mp4:
            rel_video = os.path.relpath(video_mp4, report_dir)
            lines.append(f"- Video: [{rel_video}]({rel_video})")
        if trajectory_json:
            rel_trajectory = os.path.relpath(trajectory_json, report_dir)
            lines.append(f"- JointTrajectory JSON: [{rel_trajectory}]({rel_trajectory})")
        if metadata_json:
            rel_metadata = os.path.relpath(metadata_json, report_dir)
            lines.append(f"- Metadata JSON: [{rel_metadata}]({rel_metadata})")
        validation_plot = planner["validation"].get("plot_path")
        if validation_plot:
            rel_validation = os.path.relpath(validation_plot, report_dir)
            lines.append(f"- Validation plot: [{rel_validation}]({rel_validation})")
        lines.append("")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def parse_args():
    default_grasp_json, default_start_state, default_end_state = build_default_paths()
    parser = argparse.ArgumentParser(description="Stage 3 front goal-pose study")
    parser.add_argument("--grasp-json", type=str, default=default_grasp_json, help="Path to grasp JSON file")
    parser.add_argument("--start-state", type=str, default=default_start_state, help="Path to start RobotCellState JSON")
    parser.add_argument("--end-state", type=str, default=default_end_state, help="Path to baseline end RobotCellState JSON")
    parser.add_argument("--candidate-attempts", type=int, default=12, help="Number of random front-goal samples to try after the baseline goal")
    parser.add_argument("--accepted-goals", type=int, default=6, help="Maximum number of collision-free endpoint-IK goals to benchmark, including the baseline goal if accepted")
    parser.add_argument("--max-tilt-deg", type=float, default=88.0, help="Maximum sampled bar tilt away from the baseline horizontal goal")
    parser.add_argument("--goal-bias", type=float, default=0.1, help="Goal sampling probability inside Stage 3 planning")
    parser.add_argument("--dist-metric", choices=["feature", "pose6d"], default="feature", help="Task-space distance metric")
    parser.add_argument("--position-res", type=float, default=0.01, help="Translation resolution used during pose extension, in meters")
    parser.add_argument("--rotation-res", type=float, default=0.025, help="Rotation resolution used during pose extension, in radians")
    parser.add_argument("--max-time", type=float, default=30.0, help="Max planning time per attempt")
    parser.add_argument("--max-iterations", type=int, default=2000, help="Max RRT iterations per attempt")
    parser.add_argument("--max-attempts", type=int, default=5, help="Random restarts inside Stage 3 planning")
    parser.add_argument("--endpoint-ik-attempts", type=int, default=20, help="Max random seeds used when solving endpoint IK")
    parser.add_argument("--joint-continuity-threshold", type=float, default=DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD, help="Maximum allowed wrapped joint delta between neighboring configurations, in radians")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed for pose sampling and planner runs")
    parser.add_argument("--analysis-outdir", type=str, default=REPORTS_DIR, help="Output directory for reports")
    parser.add_argument("--gui", action="store_true", help="Run the study with PyBullet GUI. Required for mp4 capture via VideoSaver.")
    parser.add_argument("--capture-artifacts", action="store_true", help="Record per-goal mp4 replays and overview screenshots")
    parser.add_argument("--capture-width", type=int, default=1024, help="Overview screenshot width in pixels")
    parser.add_argument("--capture-height", type=int, default=768, help="Overview screenshot height in pixels")
    parser.add_argument("--video-frame-step", type=int, default=2, help="Record every Nth waypoint into the replay video")
    parser.add_argument("--video-frame-sleep", type=float, default=0.02, help="Sleep time between recorded frames, in seconds")
    parser.add_argument("--use-angle-normalization", action="store_true", help="Enable wrapped-angle normalization inside IK and continuity checks")
    parser.set_defaults(use_angle_normalization=DEFAULT_USE_ANGLE_NORMALIZATION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.analysis_outdir)
    support_outdir = support_dir(args.analysis_outdir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(args.analysis_outdir, f"goal_pose_study_report_{timestamp}.md")
    json_path = os.path.join(support_outdir, f"goal_pose_study_{timestamp}.json")

    scene = setup_stage1_scene(args.grasp_json, args.start_state, args.end_state, use_gui=args.gui)
    try:
        start_conf, joint_collision_fn = solve_start_and_collision_context(
            scene=scene,
            endpoint_ik_attempts=args.endpoint_ik_attempts,
            random_seed=args.random_seed,
            use_angle_normalization=args.use_angle_normalization,
        )

        rng = np.random.default_rng(args.random_seed)
        candidate_results: List[Dict[str, Any]] = []
        planner_results: List[Dict[str, Any]] = []

        baseline_pose = (
            np.asarray(scene["end_pose"][0], dtype=float).copy(),
            np.asarray(scene["end_pose"][1], dtype=float).copy(),
        )
        candidate_poses: List[Tuple[str, PoseLike, Dict[str, float]]] = [("baseline", baseline_pose, {"tilt_deg": 0.0, "yaw_deg": 0.0})]
        for sample_index in range(args.candidate_attempts):
            goal_pose, metadata = sample_front_goal_pose(scene["end_pose"], rng, sample_index, args.max_tilt_deg)
            candidate_poses.append(("sampled", goal_pose, metadata))

        accepted = 0
        for candidate_id, (source, goal_pose, metadata) in enumerate(candidate_poses):
            feasibility = check_goal_feasibility(
                scene=scene,
                goal_pose=goal_pose,
                endpoint_ik_attempts=args.endpoint_ik_attempts,
                random_seed=args.random_seed + 1000 + candidate_id,
                joint_collision_fn=joint_collision_fn,
                use_angle_normalization=args.use_angle_normalization,
            )
            candidate_record = {
                "candidate_id": candidate_id,
                "source": source,
                "tilt_deg": float(metadata["tilt_deg"]),
                "yaw_deg": float(metadata["yaw_deg"]),
                "goal_pose": to_pose_dict(goal_pose),
                "accepted": bool(feasibility["accepted"]),
                "reason": str(feasibility["reason"]),
            }
            candidate_results.append(candidate_record)

            if not feasibility["accepted"]:
                continue
            if accepted >= args.accepted_goals:
                candidate_record["accepted"] = False
                candidate_record["reason"] = "accepted_but_skipped_limit"
                continue

            planner_record = dict(candidate_record)
            planner_record["accepted"] = True
            planner_record["reason"] = "accepted"
            planner_record["planner"] = run_stage3_for_goal(
                scene=scene,
                candidate_id=candidate_id,
                goal_pose=goal_pose,
                start_conf=start_conf,
                goal_conf=np.asarray(feasibility["goal_conf"], dtype=float),
                joint_collision_fn=joint_collision_fn,
                args=args,
                random_seed=args.random_seed + 2000 + candidate_id,
                reports_dir=support_outdir,
                timestamp=timestamp,
            )
            planner_results.append(planner_record)
            accepted += 1

        with open(json_path, "w") as f:
            json.dump(
                {
                    "setup": {
                        "start_pose": to_pose_dict(scene["start_pose"]),
                        "baseline_goal_pose": to_pose_dict(scene["end_pose"]),
                        "candidate_attempts": args.candidate_attempts,
                        "accepted_goals": args.accepted_goals,
                        "position_res": args.position_res,
                        "rotation_res": args.rotation_res,
                        "joint_continuity_threshold": args.joint_continuity_threshold,
                        "random_seed": args.random_seed,
                    },
                    "candidates": candidate_results,
                    "planner_results": planner_results,
                },
                f,
                indent=2,
            )
        write_report(report_path, timestamp, args, scene, candidate_results, planner_results, json_path)
        logger.info("Saved goal-pose study report: %s", report_path)
        logger.info("Saved goal-pose study JSON: %s", json_path)
    finally:
        teardown_stage1_scene()


if __name__ == "__main__":
    main()
