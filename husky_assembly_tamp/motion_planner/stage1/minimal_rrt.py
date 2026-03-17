"""
Minimal Stage 1/2/3 floating-bar RRT.

This is a clean restart from the original design intent:
- task-space sampling
- single-tree RRT
- Stage 1: no IK in the planner loop
- Stage 2: seed-chained dual-arm IK in extend, collision still off
- Stage 3: seed-chained dual-arm IK with robot collision on
- no ladder graph
- optional floating-body collision against a fixed robot

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet
import pybullet_planning as pp
from compas_fab.robots import RobotSemantics
from compas_robots import RobotModel
from pybullet_planning.motion_planners.rrt import TreeNode, configs

from husky_assembly_tamp.motion_planner.stage1.path_validation import validate_stage_trajectory
from husky_assembly_tamp.utils.params import DATA_DIR
from husky_assembly_tamp.utils.util import calculate_pose_error, normalize_angles, setup_logger


logger = setup_logger("stage1_minimal_rrt")

HUSKY_DUAL_URDF_PATH = os.path.join(
    DATA_DIR,
    "husky_urdf/mt_husky_dual_ur5_e_moveit_config/urdf/husky_dual_ur5_e_no_base_joint_All_Calibrated.urdf",
)
HUSKY_DUAL_SRDF_PATH = os.path.join(
    DATA_DIR,
    "husky_urdf/mt_husky_dual_ur5_e_moveit_config/config/dual_arm_husky.srdf",
)
HUSKY_DUAL_ARM_JOINT_NAMES = [
    "left_ur_arm_shoulder_pan_joint",
    "left_ur_arm_shoulder_lift_joint",
    "left_ur_arm_elbow_joint",
    "left_ur_arm_wrist_1_joint",
    "left_ur_arm_wrist_2_joint",
    "left_ur_arm_wrist_3_joint",
    "right_ur_arm_shoulder_pan_joint",
    "right_ur_arm_shoulder_lift_joint",
    "right_ur_arm_elbow_joint",
    "right_ur_arm_wrist_1_joint",
    "right_ur_arm_wrist_2_joint",
    "right_ur_arm_wrist_3_joint",
]
TOOL_LINK_LEFT = "left_ur_arm_tool0"
TOOL_LINK_RIGHT = "right_ur_arm_tool0"
STAGE3_GRASP_MASK_LINKS = [
    "left_ur_arm_wrist_3_link",
    "right_ur_arm_wrist_3_link",
    TOOL_LINK_LEFT,
    TOOL_LINK_RIGHT,
]
INIT_ARM_JOINT_ANGLES = np.array([0.0, -np.pi / 2.0, 0.0, 0.0, 0.0, 0.0] * 2, dtype=float)
BAR_RADIUS = 0.015
BAR_LENGTH = 1.0
BAR_BOX_DIMS = (2.0 * BAR_RADIUS, 2.0 * BAR_RADIUS, BAR_LENGTH)
STAGE1_DEBUG_START_OFFSET = np.array([-0.5, 0.0, 0.5], dtype=float)
DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD = 0.2


PoseLike = Tuple[np.ndarray, np.ndarray]
GraspTarget = Tuple[PoseLike, PoseLike]
ArmConf = np.ndarray
FullConf = np.ndarray


def load_grasp_targets(json_path: str) -> List[GraspTarget]:
    with open(json_path) as f:
        raw = json.load(f)
    targets = []
    for item in raw:
        d = item["data"]
        world_from_bar = pp.pose_from_tform(np.array(d["world_from_bar"]["data"]["matrix"]))
        world_from_tool0 = pp.pose_from_tform(np.array(d["world_from_tool0"]["data"]["matrix"]))
        targets.append((world_from_bar, world_from_tool0))
    return targets


def load_robot_cell_state(json_path: str) -> np.ndarray:
    with open(json_path) as f:
        data = json.load(f)
    state = data["data"]
    return np.asarray(state["robot_configuration"]["data"]["joint_values"], dtype=float)


def get_bar_feature_points() -> List[np.ndarray]:
    half_width, half_depth, half_length = 0.5 * np.asarray(BAR_BOX_DIMS, dtype=float)
    return [
        np.array([sx * half_width, sy * half_depth, sz * half_length], dtype=float)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]


def pose_to_feature_vec(pose: PoseLike, feature_points: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    if not feature_points:
        return None
    pts = []
    for p_local in feature_points:
        p_world, _ = pp.multiply(pose, (p_local, [0, 0, 0, 1]))
        pts.append(np.asarray(p_world, dtype=float))
    return np.concatenate(pts, axis=0)


def pose_distance(pose1: PoseLike, pose2: PoseLike, dist_metric: str, feature_points: Sequence[np.ndarray]) -> float:
    pos1, quat1 = pose1
    pos2, quat2 = pose2
    if dist_metric == "feature":
        vec1 = pose_to_feature_vec((pos1, quat1), feature_points)
        vec2 = pose_to_feature_vec((pos2, quat2), feature_points)
        if vec1 is not None and vec2 is not None:
            return float(np.linalg.norm(vec2 - vec1))
    dx = pos2 - pos1
    rot_dist = pp.quat_angle_between(quat1, quat2)
    return float(np.linalg.norm(np.array([dx[0], dx[1], dx[2], rot_dist], dtype=float)))


def sample_pose(
    robot: int,
    goal_pose: PoseLike,
    rng: np.random.Generator,
    goal_sample_prob: float,
    workspace_xy: float,
    workspace_z: float,
) -> Tuple[PoseLike, bool]:
    if rng.random() < goal_sample_prob:
        return goal_pose, True
    base_pos, _ = pp.get_pose(robot)
    cx, cy, cz = np.asarray(base_pos, dtype=float)
    x = cx + rng.uniform(-workspace_xy / 2.0, workspace_xy / 2.0)
    y = cy + rng.uniform(-workspace_xy / 2.0, workspace_xy / 2.0)
    z_min = max(0.05, cz)
    z = rng.uniform(z_min, z_min + workspace_z)
    roll = rng.uniform(-np.pi, np.pi)
    pitch = rng.uniform(-np.pi, np.pi)
    yaw = rng.uniform(-np.pi, np.pi)
    return pp.Pose(point=[x, y, z], euler=pp.Euler(roll, pitch, yaw)), False


def nearest_node(
    nodes: List[TreeNode],
    target_pose: PoseLike,
    dist_metric: str,
    feature_points: Sequence[np.ndarray],
    feature_vecs: Dict[int, np.ndarray],
) -> TreeNode:
    if dist_metric == "feature":
        target_vec = pose_to_feature_vec(target_pose, feature_points)
        if target_vec is not None:
            return min(
                nodes,
                key=lambda node: float(np.linalg.norm(feature_vecs[id(node)] - target_vec)),
            )
    return min(nodes, key=lambda node: pose_distance(node.config, target_pose, dist_metric, feature_points))


def export_tree(nodes: List[TreeNode]) -> Dict[str, List[List[float]]]:
    id_to_idx: Dict[int, int] = {}
    points: List[List[float]] = []
    for node in nodes:
        idx = len(points)
        id_to_idx[id(node)] = idx
        pos = np.asarray(node.config[0], dtype=float).reshape(3)
        points.append([float(pos[0]), float(pos[1]), float(pos[2])])
    edges: List[List[int]] = []
    for node in nodes:
        if node.parent is None:
            continue
        pid = id(node.parent)
        cid = id(node)
        if pid in id_to_idx and cid in id_to_idx:
            edges.append([id_to_idx[pid], id_to_idx[cid]])
    return {"points": points, "edges": edges}


def get_pose_collision_fn(bar_body: int, obstacle_bodies: Sequence[int], enable_collision: bool) -> Callable[[PoseLike], bool]:
    if not enable_collision:
        return lambda pose: False
    floating_collision_fn = pp.get_floating_body_collision_fn(
        bar_body,
        obstacles=list(obstacle_bodies),
        disabled_collisions=[],
    )
    return lambda pose: bool(floating_collision_fn(pose))


def get_disabled_collisions_from_link_names(
    robot: int,
    link_name_pairs: Sequence[Tuple[str, str]],
) -> List[Tuple[int, int]]:
    disabled_pairs: List[Tuple[int, int]] = []
    for link1_name, link2_name in link_name_pairs:
        if not (pp.has_link(robot, link1_name) and pp.has_link(robot, link2_name)):
            continue
        disabled_pairs.append((pp.link_from_name(robot, link1_name), pp.link_from_name(robot, link2_name)))
    return disabled_pairs


def get_joint_collision_fn(
    robot: int,
    arm_joints: Sequence[int],
    obstacle_bodies: Sequence[int],
    tool_link_left: int,
    bar_body: int,
    grasp_bar_from_left: PoseLike,
) -> Callable[[FullConf], bool]:
    robot_model = RobotModel.from_urdf_file(HUSKY_DUAL_URDF_PATH)
    semantics = RobotSemantics.from_srdf_file(HUSKY_DUAL_SRDF_PATH, robot_model)
    disabled_collisions = get_disabled_collisions_from_link_names(robot, semantics.disabled_collisions)
    attachment = pp.Attachment(robot, tool_link_left, pp.invert(grasp_bar_from_left), bar_body)
    extra_disabled_collisions = []
    for link_name in STAGE3_GRASP_MASK_LINKS:
        if not pp.has_link(robot, link_name):
            continue
        link = pp.link_from_name(robot, link_name)
        extra_disabled_collisions.append(((robot, link), (bar_body, pp.BASE_LINK)))
    collision_fn = pp.get_collision_fn(
        robot,
        arm_joints,
        obstacles=list(obstacle_bodies),
        attachments=[attachment],
        self_collisions=True,
        disabled_collisions=disabled_collisions,
        extra_disabled_collisions=extra_disabled_collisions,
        max_distance=0.0,
    )
    return lambda conf: bool(collision_fn(np.asarray(conf, dtype=float)))


def add_profile_time(profile_out: Optional[Dict[str, Any]], key: str, dt: float) -> None:
    if profile_out is None:
        return
    profile_out[key] = float(profile_out.get(key, 0.0)) + float(dt)


def bump_profile_count(profile_out: Optional[Dict[str, Any]], key: str, inc: int = 1) -> None:
    if profile_out is None:
        return
    profile_out[key] = int(profile_out.get(key, 0)) + int(inc)


def goal_pose_reached(pose: PoseLike, goal_pose: PoseLike, position_res: float, rotation_res: float) -> bool:
    return bool(
        pp.is_pose_close(
            pose,
            goal_pose,
            pos_tolerance=max(position_res, 1e-6),
            ori_tolerance=max(rotation_res, 1e-6),
        )
    )


def solve_single_arm_ik(
    robot: int,
    arm_joints: Sequence[int],
    tool_link: int,
    full_seed_conf: FullConf,
    target_tool_pose: PoseLike,
    arm_slice: slice,
) -> Optional[FullConf]:
    seed_conf = np.asarray(full_seed_conf, dtype=float)
    pp.set_joint_positions(robot, arm_joints, seed_conf)
    result = np.asarray(
        pybullet.calculateInverseKinematics(
            robot,
            tool_link,
            target_tool_pose[0],
            target_tool_pose[1],
            maxNumIterations=1000,
            residualThreshold=1e-6,
        ),
        dtype=float,
    )
    if result.shape[0] < max(arm_slice.stop, len(seed_conf)):
        return None
    solved_conf = seed_conf.copy()
    solved_conf[arm_slice] = normalize_angles(result[arm_slice])
    pp.set_joint_positions(robot, arm_joints, solved_conf)
    pose_res = pp.get_link_pose(robot, tool_link)
    pose_err = calculate_pose_error(target_tool_pose, pose_res)
    if np.linalg.norm(pose_err) > 1e-4:
        return None
    return normalize_angles(solved_conf)


def validate_dual_arm_bar_pose(
    robot: int,
    arm_joints: Sequence[int],
    tool_link_left: int,
    tool_link_right: int,
    full_conf: FullConf,
    bar_pose: PoseLike,
    grasp_bar_from_left: PoseLike,
    grasp_bar_from_right: PoseLike,
) -> bool:
    pp.set_joint_positions(robot, arm_joints, full_conf)
    target_left = pp.multiply(bar_pose, grasp_bar_from_left)
    target_right = pp.multiply(bar_pose, grasp_bar_from_right)
    world_from_left = pp.get_link_pose(robot, tool_link_left)
    world_from_right = pp.get_link_pose(robot, tool_link_right)
    if not pp.is_pose_close(target_left, world_from_left, pos_tolerance=1e-4, ori_tolerance=1e-4):
        return False
    if not pp.is_pose_close(target_right, world_from_right, pos_tolerance=1e-4, ori_tolerance=1e-4):
        return False
    bar_from_left = pp.invert(grasp_bar_from_left)
    bar_from_right = pp.invert(grasp_bar_from_right)
    left_bar_pose = pp.multiply(world_from_left, bar_from_left)
    right_bar_pose = pp.multiply(world_from_right, bar_from_right)
    return bool(
        pp.is_pose_close(left_bar_pose, bar_pose, pos_tolerance=1e-4, ori_tolerance=1e-4)
        and pp.is_pose_close(right_bar_pose, bar_pose, pos_tolerance=1e-4, ori_tolerance=1e-4)
        and pp.is_pose_close(left_bar_pose, right_bar_pose, pos_tolerance=1e-4, ori_tolerance=1e-4)
    )


def solve_dual_arm_pose_ik(
    robot: int,
    arm_joints: Sequence[int],
    tool_link_left: int,
    tool_link_right: int,
    bar_pose: PoseLike,
    grasp_bar_from_left: PoseLike,
    grasp_bar_from_right: PoseLike,
    seed_conf: FullConf,
    profile_out: Optional[Dict[str, Any]] = None,
) -> Optional[FullConf]:
    target_left = pp.multiply(bar_pose, grasp_bar_from_left)
    target_right = pp.multiply(bar_pose, grasp_bar_from_right)
    seed_conf = normalize_angles(np.asarray(seed_conf, dtype=float))
    t0 = time.perf_counter()
    attempts = (
        ("right", "left"),
        ("left", "right"),
    )
    for order in attempts:
        conf = seed_conf.copy()
        success = True
        for arm_name in order:
            bump_profile_count(profile_out, "ik_calls")
            if arm_name == "right":
                conf_next = solve_single_arm_ik(
                    robot=robot,
                    arm_joints=arm_joints,
                    tool_link=tool_link_right,
                    full_seed_conf=conf,
                    target_tool_pose=target_right,
                    arm_slice=slice(6, 12),
                )
            else:
                conf_next = solve_single_arm_ik(
                    robot=robot,
                    arm_joints=arm_joints,
                    tool_link=tool_link_left,
                    full_seed_conf=conf,
                    target_tool_pose=target_left,
                    arm_slice=slice(0, 6),
                )
            if conf_next is None:
                success = False
                bump_profile_count(profile_out, "ik_failures")
                break
            conf = conf_next
        if success and validate_dual_arm_bar_pose(
            robot=robot,
            arm_joints=arm_joints,
            tool_link_left=tool_link_left,
            tool_link_right=tool_link_right,
            full_conf=conf,
            bar_pose=bar_pose,
            grasp_bar_from_left=grasp_bar_from_left,
            grasp_bar_from_right=grasp_bar_from_right,
        ):
            add_profile_time(profile_out, "ik_time_s", time.perf_counter() - t0)
            bump_profile_count(profile_out, "nodes_with_ik")
            return normalize_angles(conf)
    add_profile_time(profile_out, "ik_time_s", time.perf_counter() - t0)
    return None


def solve_endpoint_dual_arm_ik(
    robot: int,
    arm_joints: Sequence[int],
    tool_link_left: int,
    tool_link_right: int,
    bar_pose: PoseLike,
    grasp_bar_from_left: PoseLike,
    grasp_bar_from_right: PoseLike,
    seed_conf: FullConf,
    rng: np.random.Generator,
    max_attempts: int,
    profile_out: Optional[Dict[str, Any]] = None,
) -> Optional[FullConf]:
    for attempt in range(max(1, max_attempts)):
        if attempt == 0:
            attempt_seed = np.asarray(seed_conf, dtype=float)
        else:
            attempt_seed = rng.uniform(-np.pi, np.pi, len(seed_conf))
        conf = solve_dual_arm_pose_ik(
            robot=robot,
            arm_joints=arm_joints,
            tool_link_left=tool_link_left,
            tool_link_right=tool_link_right,
            bar_pose=bar_pose,
            grasp_bar_from_left=grasp_bar_from_left,
            grasp_bar_from_right=grasp_bar_from_right,
            seed_conf=attempt_seed,
            profile_out=profile_out,
        )
        if conf is not None:
            return conf
    return None


def extend_toward(
    nodes: List[TreeNode],
    source: TreeNode,
    target_pose: PoseLike,
    collision_fn: Callable[[PoseLike], bool],
    joint_collision_fn: Optional[Callable[[FullConf], bool]],
    draw_color: Tuple[float, float, float, float],
    use_draw: bool,
    position_res: float,
    rotation_res: float,
    dist_metric: str,
    feature_points: Sequence[np.ndarray],
    feature_vecs: Dict[int, np.ndarray],
    enable_ik: bool = False,
    node_confs: Optional[Dict[int, FullConf]] = None,
    ik_context: Optional[Dict[str, Any]] = None,
    joint_continuity_threshold_rad: Optional[float] = None,
    profile_out: Optional[Dict[str, Any]] = None,
) -> Tuple[TreeNode, bool, str]:
    current = source
    reached = True
    stop_reason = "reached"
    current_conf = None if node_confs is None else node_confs.get(id(source))
    if enable_ik and (node_confs is None or ik_context is None or current_conf is None):
        return current, False, "ik_failure"
    for pose in list(
        pp.interpolate_poses(
            source.config,
            target_pose,
            pos_step_size=max(position_res, 1e-6),
            ori_step_size=max(rotation_res, 1e-6),
        )
    )[1:]:
        bump_profile_count(profile_out, "poses_checked")
        next_conf = None
        if enable_ik:
            next_conf = solve_dual_arm_pose_ik(
                robot=ik_context["robot"],
                arm_joints=ik_context["arm_joints"],
                tool_link_left=ik_context["tool_link_left"],
                tool_link_right=ik_context["tool_link_right"],
                bar_pose=pose,
                grasp_bar_from_left=ik_context["grasp_bar_from_left"],
                grasp_bar_from_right=ik_context["grasp_bar_from_right"],
                seed_conf=current_conf,
                profile_out=profile_out,
            )
            if next_conf is None:
                reached = False
                stop_reason = "ik_failure"
                break
            if joint_continuity_threshold_rad is not None:
                step_delta = np.abs(
                    normalize_angles(np.asarray(next_conf, dtype=float) - np.asarray(current_conf, dtype=float))
                )
                if float(np.max(step_delta)) > float(joint_continuity_threshold_rad):
                    reached = False
                    stop_reason = "continuity"
                    bump_profile_count(profile_out, "continuity_rejections")
                    break
            if joint_collision_fn is not None and joint_collision_fn(next_conf):
                reached = False
                stop_reason = "collision"
                bump_profile_count(profile_out, "collision_hits")
                break
        if collision_fn(pose):
            reached = False
            stop_reason = "collision"
            bump_profile_count(profile_out, "collision_hits")
            break
        # this is the version of extension that stops at the first collision of the extend, but still add the valid interp so far into the tree
        node = TreeNode(pose, parent=current)
        nodes.append(node)
        if enable_ik and node_confs is not None and next_conf is not None:
            node_confs[id(node)] = next_conf
        bump_profile_count(profile_out, "nodes_created")
        if dist_metric == "feature":
            feature_vec = pose_to_feature_vec(pose, feature_points)
            if feature_vec is not None:
                feature_vecs[id(node)] = feature_vec
        if use_draw:
            pp.add_line(current.config[0], node.config[0], width=1.5, color=draw_color)
        current = node
        if next_conf is not None:
            current_conf = next_conf
    return current, reached, stop_reason


def summarize_joint_continuity(
    joint_path: Optional[Sequence[FullConf]],
    threshold_rad: float = DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
) -> Dict[str, Any]:
    summary = {
        "ok": None,
        "max_delta_rad": None,
        "first_bad_step": None,
        "threshold_rad": float(threshold_rad),
    }
    if joint_path is None:
        return summary
    normalized_joint_path = [normalize_angles(np.asarray(conf, dtype=float)) for conf in joint_path]
    if len(normalized_joint_path) < 2:
        summary["ok"] = True
        summary["max_delta_rad"] = 0.0
        return summary

    step_max_deltas = []
    for prev_conf, next_conf in zip(normalized_joint_path[:-1], normalized_joint_path[1:]):
        step_delta = np.abs(normalize_angles(np.asarray(next_conf, dtype=float) - np.asarray(prev_conf, dtype=float)))
        step_max_deltas.append(float(np.max(step_delta)))
    max_delta = max(step_max_deltas) if step_max_deltas else 0.0
    first_bad_step = next((idx + 1 for idx, delta in enumerate(step_max_deltas) if delta > threshold_rad), None)
    summary["ok"] = first_bad_step is None
    summary["max_delta_rad"] = float(max_delta)
    summary["first_bad_step"] = first_bad_step
    return summary


def densify_pose_path(
    path: Sequence[PoseLike],
    position_res: float,
    rotation_res: float,
) -> List[PoseLike]:
    if not path:
        return []
    dense_path: List[PoseLike] = [path[0]]
    for start_pose, end_pose in zip(path[:-1], path[1:]):
        segment = list(
            pp.interpolate_poses(
                start_pose,
                end_pose,
                pos_step_size=max(position_res, 1e-6),
                ori_step_size=max(rotation_res, 1e-6),
            )
        )
        dense_path.extend(segment[1:])
    return dense_path


def reconstruct_joint_path_for_pose_path(
    scene: Dict[str, Any],
    pose_path: Sequence[PoseLike],
    start_conf: FullConf,
    joint_collision_fn: Optional[Callable[[FullConf], bool]] = None,
    joint_continuity_threshold_rad: Optional[float] = None,
    profile_out: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[List[FullConf]], Optional[str]]:
    if not pose_path:
        return [], None
    grasp_bar_from_right = scene["grasp_bar_from_right"]
    if grasp_bar_from_right is None:
        return None, "missing_right_grasp"

    current_conf = normalize_angles(np.asarray(start_conf, dtype=float))
    joint_path = [current_conf]
    for idx, pose in enumerate(pose_path[1:], start=1):
        next_conf = solve_dual_arm_pose_ik(
            robot=scene["robot"],
            arm_joints=scene["arm_joints"],
            tool_link_left=scene["tool_link_left"],
            tool_link_right=scene["tool_link_right"],
            bar_pose=pose,
            grasp_bar_from_left=scene["grasp_bar_from_left"],
            grasp_bar_from_right=grasp_bar_from_right,
            seed_conf=current_conf,
            profile_out=profile_out,
        )
        if next_conf is None:
            return None, f"ik_failure_at_waypoint_{idx}"
        next_conf = normalize_angles(np.asarray(next_conf, dtype=float))
        if joint_continuity_threshold_rad is not None:
            step_delta = np.abs(normalize_angles(np.asarray(next_conf, dtype=float) - np.asarray(current_conf, dtype=float)))
            if float(np.max(step_delta)) > float(joint_continuity_threshold_rad):
                bump_profile_count(profile_out, "continuity_rejections")
                return None, f"continuity_at_waypoint_{idx}"
        if joint_collision_fn is not None:
            t_collision = time.perf_counter()
            in_collision = joint_collision_fn(next_conf)
            add_profile_time(profile_out, "collision_check_time_s", time.perf_counter() - t_collision)
            if in_collision:
                bump_profile_count(profile_out, "collision_hits")
                return None, f"collision_at_waypoint_{idx}"
        current_conf = next_conf
        joint_path.append(current_conf)
    return joint_path, None


def refine_pose_path_with_seed_chained_ik(
    scene: Dict[str, Any],
    path: Sequence[PoseLike],
    path_confs: Sequence[FullConf],
    start_conf: FullConf,
    base_position_res: float,
    base_rotation_res: float,
    refine_max_passes: int,
    joint_continuity_threshold_rad: Optional[float] = DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
    joint_collision_fn: Optional[Callable[[FullConf], bool]] = None,
    profile_out: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    coarse_summary = summarize_joint_continuity(path_confs)
    result: Dict[str, Any] = {
        "enabled": True,
        "attempted": False,
        "used_refined_path": False,
        "status": "skipped",
        "coarse_waypoints": len(path),
        "final_waypoints": len(path),
        "coarse_joint_continuity": coarse_summary,
        "final_joint_continuity": coarse_summary,
        "passes": [],
        "passes_run": 0,
        "path": list(path),
        "path_confs": [normalize_angles(np.asarray(conf, dtype=float)) for conf in path_confs],
    }
    if len(path) < 2 or refine_max_passes <= 0:
        result["status"] = "disabled"
        return result
    if coarse_summary.get("ok"):
        result["status"] = "already_continuous"
        return result

    best_path = list(path)
    best_confs = [normalize_angles(np.asarray(conf, dtype=float)) for conf in path_confs]
    best_summary = coarse_summary
    t_refine = time.perf_counter()

    for pass_idx in range(refine_max_passes):
        pass_position_res = max(base_position_res / (2**pass_idx), 1e-4)
        pass_rotation_res = max(base_rotation_res / (2**pass_idx), 1e-4)
        dense_path = densify_pose_path(path, pass_position_res, pass_rotation_res)
        pass_record: Dict[str, Any] = {
            "pass_index": pass_idx + 1,
            "position_res": float(pass_position_res),
            "rotation_res": float(pass_rotation_res),
            "waypoints": len(dense_path),
        }
        result["attempted"] = True
        joint_path, failure_reason = reconstruct_joint_path_for_pose_path(
            scene=scene,
            pose_path=dense_path,
            start_conf=start_conf,
            joint_collision_fn=joint_collision_fn,
            joint_continuity_threshold_rad=joint_continuity_threshold_rad,
            profile_out=profile_out,
        )
        if joint_path is None:
            pass_record["status"] = "failed"
            pass_record["failure_reason"] = failure_reason
            result["passes"].append(pass_record)
            result["status"] = f"failed:{failure_reason}"
            break

        continuity = summarize_joint_continuity(joint_path)
        pass_record["status"] = "success"
        pass_record["joint_continuity"] = continuity
        result["passes"].append(pass_record)

        best_max_delta = best_summary.get("max_delta_rad")
        candidate_max_delta = continuity.get("max_delta_rad")
        if candidate_max_delta is not None and (
            best_max_delta is None or candidate_max_delta <= best_max_delta + 1e-9
        ):
            best_path = dense_path
            best_confs = joint_path
            best_summary = continuity
            result["used_refined_path"] = len(dense_path) > len(path)
            result["status"] = "improved" if not continuity.get("ok") else "continuity_pass"
        if continuity.get("ok"):
            break

    add_profile_time(profile_out, "refinement_time_s", time.perf_counter() - t_refine)
    if profile_out is not None:
        profile_out["refinement_passes"] = len(result["passes"])
        profile_out["refinement_waypoints"] = len(best_path)
        profile_out["refinement_used"] = int(result["used_refined_path"])

    result["path"] = best_path
    result["path_confs"] = best_confs
    result["passes_run"] = len(result["passes"])
    result["final_waypoints"] = len(best_path)
    result["final_joint_continuity"] = best_summary
    return result


def update_debug_tree(
    debug_tree_out: Optional[Dict],
    success: bool,
    iterations: int,
    nodes: List[TreeNode],
    start_pose: PoseLike,
    goal_pose: PoseLike,
) -> None:
    if debug_tree_out is None:
        return
    debug_tree_out.clear()
    debug_tree_out["success"] = success
    debug_tree_out["iterations"] = iterations
    debug_tree_out["tree1"] = export_tree(nodes)
    debug_tree_out["tree2"] = {"points": [], "edges": []}
    debug_tree_out["start_pose"] = [float(v) for v in start_pose[0]]
    debug_tree_out["goal_pose"] = [float(v) for v in goal_pose[0]]


def plan_pose_rrt(
    robot: int,
    bar_body: int,
    obstacle_bodies: Sequence[int],
    start_pose: PoseLike,
    goal_pose: PoseLike,
    start_conf: Optional[FullConf] = None,
    goal_conf: Optional[FullConf] = None,
    dist_metric: str = "feature",
    goal_sample_prob: float = 0.1,
    workspace_xy: float = 2.2,
    workspace_z: float = 1.2,
    position_res: float = 0.05,
    rotation_res: float = 0.1,
    random_seed: Optional[int] = None,
    max_time: float = 30.0,
    max_iterations: int = 2000,
    max_attempts: int = 5,
    enable_collision: bool = False,
    enable_ik: bool = False,
    ik_context: Optional[Dict[str, Any]] = None,
    joint_collision_fn: Optional[Callable[[FullConf], bool]] = None,
    joint_continuity_threshold_rad: Optional[float] = None,
    use_draw: bool = True,
    debug_tree_out: Optional[Dict] = None,
    profile_out: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[List[PoseLike]], Optional[List[FullConf]]]:
    """Plan a collision-free path in SE(3) pose space using RRT.

    Args:
        robot: PyBullet body ID of the robot (used for IK and workspace sampling).
        bar_body: PyBullet body ID of the bar being manipulated (used for collision checks).
        obstacle_bodies: PyBullet body IDs of obstacles to check collisions against.
        start_pose: Starting SE(3) pose of the bar, as ((x, y, z), (qx, qy, qz, qw)).
        goal_pose: Goal SE(3) pose of the bar, same format as start_pose.
        start_conf: Full joint configuration of the robot at the start pose. Required
            when enable_ik=True (stage 2/3 planning); unused otherwise.
        goal_conf: Full joint configuration of the robot at the goal pose. Required
            when joint_collision_fn is provided.
        dist_metric: Distance metric for nearest-neighbor lookup. ``"feature"`` uses
            a feature-point projection of the bar pose; ``"pose"`` uses raw
            position/quaternion distance.
        goal_sample_prob: Probability of sampling the goal pose directly instead of
            a random workspace pose on each RRT iteration.
        workspace_xy: Half-width (metres) of the square XY workspace region sampled
            during random pose generation.
        workspace_z: Height (metres) of the workspace Z region sampled during random
            pose generation.
        position_res: Linear step size (metres) used when extending the tree and when
            testing whether the goal has been reached.
        rotation_res: Angular step size (radians) used when extending the tree and
            when testing whether the goal has been reached.
        random_seed: Seed for the NumPy RNG. Pass an integer for reproducible runs;
            ``None`` gives a non-deterministic result.
        max_time: Wall-clock time limit (seconds) per planning attempt before moving
            on to the next attempt.
        max_iterations: Maximum RRT iterations per attempt.
        max_attempts: Number of independent planning attempts before giving up.
        enable_collision: Whether to enable collision checking. In Stage 1 this
            enables floating-bar pose collision; in Stage 3 this enables joint-space
            robot collision through ``joint_collision_fn``.
        enable_ik: Whether to compute and propagate IK solutions along the tree. Must
            be True for stage-2/3 planning; requires start_conf.
        ik_context: Extra context passed to the IK solver when enable_ik=True (e.g.
            preferred joint configuration, solver settings).
        joint_collision_fn: Optional robot collision checker operating on a full
            12-DOF arm configuration. Used by Stage 3 after IK succeeds.
        use_draw: Whether to draw the RRT tree edges in the PyBullet GUI while
            planning. Disable when running headless or for speed.
        debug_tree_out: Optional dict that is populated with the final RRT tree for
            offline inspection (nodes, edges, success flag, iteration count).
        profile_out: Optional dict populated with timing and count statistics for
            each planning run (collision checks, IK calls, node counts, outcome, etc.).

    Returns:
        A tuple ``(path_poses, path_confs)``.  On success, ``path_poses`` is the list
        of SE(3) waypoint poses from start to goal and ``path_confs`` is the
        corresponding list of joint configurations (or ``None`` when enable_ik is
        False).  Returns ``(None, None)`` if no path is found within the given limits.
    """
    rng = np.random.default_rng(random_seed)
    feature_points = get_bar_feature_points()
    collision_fn = get_pose_collision_fn(bar_body, obstacle_bodies, enable_collision and not enable_ik)
    if profile_out is not None:
        profile_out.clear()
        profile_out.update(
            {
                "attempts": 0,
                "iterations": 0,
                "nodes_created": 0,
                "poses_checked": 0,
                "collision_hits": 0,
                "ik_calls": 0,
                "ik_failures": 0,
                "nodes_with_ik": 0,
                "continuity_rejections": 0,
            }
        )

    if joint_collision_fn is not None:
        if start_conf is None or goal_conf is None:
            raise ValueError("Collision-aware planning requires both start_conf and goal_conf.")
        t_collision = time.perf_counter()
        start_in_collision = joint_collision_fn(start_conf)
        add_profile_time(profile_out, "collision_check_time_s", time.perf_counter() - t_collision)
        if start_in_collision:
            logger.warning("Start configuration is in collision.")
            if profile_out is not None:
                profile_out["outcome"] = "start_in_collision"
                bump_profile_count(profile_out, "collision_hits")
            return None, None
        t_collision = time.perf_counter()
        goal_in_collision = joint_collision_fn(goal_conf)
        add_profile_time(profile_out, "collision_check_time_s", time.perf_counter() - t_collision)
        if goal_in_collision:
            logger.warning("Goal configuration is in collision.")
            if profile_out is not None:
                profile_out["outcome"] = "goal_in_collision"
                bump_profile_count(profile_out, "collision_hits")
            return None, None
    else:
        t_collision = time.perf_counter()
        start_in_collision = collision_fn(start_pose)
        add_profile_time(profile_out, "collision_check_time_s", time.perf_counter() - t_collision)
        if start_in_collision:
            logger.warning("Start pose is in floating-body collision.")
            if profile_out is not None:
                profile_out["outcome"] = "start_in_collision"
                bump_profile_count(profile_out, "collision_hits")
            return None, None
        t_collision = time.perf_counter()
        goal_in_collision = collision_fn(goal_pose)
        add_profile_time(profile_out, "collision_check_time_s", time.perf_counter() - t_collision)
        if goal_in_collision:
            logger.warning("Goal pose is in floating-body collision.")
            if profile_out is not None:
                profile_out["outcome"] = "goal_in_collision"
                bump_profile_count(profile_out, "collision_hits")
            return None, None

    best_tree: List[TreeNode] = []
    total_iterations = 0
    for attempt in range(max_attempts):
        bump_profile_count(profile_out, "attempts")
        start_time = time.time()
        root = TreeNode(start_pose)
        nodes = [root]
        node_confs: Dict[int, FullConf] = {}
        if enable_ik:
            if start_conf is None:
                raise ValueError("Stage 2/3 planning requires start_conf.")
            node_confs[id(root)] = np.asarray(start_conf, dtype=float)
        feature_vecs: Dict[int, np.ndarray] = {}
        bump_profile_count(profile_out, "nodes_created")
        if dist_metric == "feature":
            t_feature = time.perf_counter()
            root_feature = pose_to_feature_vec(start_pose, feature_points)
            add_profile_time(profile_out, "feature_time_s", time.perf_counter() - t_feature)
            if root_feature is not None:
                feature_vecs[id(root)] = root_feature

        for iteration in range(max_iterations):
            total_iterations += 1
            if profile_out is not None:
                profile_out["iterations"] = total_iterations
            if (time.time() - start_time) >= max_time:
                break
            t_sample = time.perf_counter()
            target_pose, _ = sample_pose(robot, goal_pose, rng, goal_sample_prob, workspace_xy, workspace_z)
            add_profile_time(profile_out, "sample_time_s", time.perf_counter() - t_sample)
            t_nearest = time.perf_counter()
            nearest = nearest_node(nodes, target_pose, dist_metric, feature_points, feature_vecs)
            add_profile_time(profile_out, "nearest_time_s", time.perf_counter() - t_nearest)
            t_extend = time.perf_counter()
            new_last, reached, stop_reason = extend_toward(
                nodes=nodes,
                source=nearest,
                target_pose=target_pose,
                collision_fn=collision_fn,
                joint_collision_fn=joint_collision_fn,
                draw_color=(0.85, 0.2, 0.2, 0.45),
                use_draw=use_draw,
                position_res=position_res,
                rotation_res=rotation_res,
                dist_metric=dist_metric,
                feature_points=feature_points,
                feature_vecs=feature_vecs,
                enable_ik=enable_ik,
                node_confs=node_confs,
                ik_context=ik_context,
                joint_continuity_threshold_rad=joint_continuity_threshold_rad,
                profile_out=profile_out,
            )
            add_profile_time(profile_out, "extend_tree_time_s", time.perf_counter() - t_extend)
            if not reached:
                if stop_reason == "ik_failure" and enable_ik and profile_out is not None:
                    profile_out["outcome"] = "extend_ik_failure"
                elif stop_reason == "continuity" and enable_ik and profile_out is not None:
                    profile_out["outcome"] = "extend_continuity_failure"
                elif stop_reason == "collision" and profile_out is not None:
                    profile_out["outcome"] = "extend_collision_failure"
                continue
            t_goal_dist = time.perf_counter()
            if goal_pose_reached(new_last.config, goal_pose, position_res, rotation_res):
                add_profile_time(profile_out, "goal_test_time_s", time.perf_counter() - t_goal_dist)
                update_debug_tree(debug_tree_out, True, iteration + 1, nodes, start_pose, goal_pose)
                if profile_out is not None:
                    profile_out["outcome"] = "success"
                path_nodes = new_last.retrace()
                path_poses = configs(path_nodes)
                path_confs = None
                if enable_ik:
                    path_confs = [np.asarray(node_confs[id(node)], dtype=float) for node in path_nodes]
                return path_poses, path_confs
            else:
                add_profile_time(profile_out, "goal_test_time_s", time.perf_counter() - t_goal_dist)
        best_tree = nodes
        logger.info(f"Attempt {attempt + 1}/{max_attempts}: no path found.")

    update_debug_tree(debug_tree_out, False, total_iterations, best_tree, start_pose, goal_pose)
    if profile_out is not None:
        profile_out["iterations"] = total_iterations
        profile_out["outcome"] = "task_space_failure"
    return None, None


def compute_bar_pose_from_state(
    robot: int,
    arm_joints: Sequence[int],
    tool_link: int,
    joint_values: Sequence[float],
    grasp_bar_from_tool: PoseLike,
) -> PoseLike:
    pp.set_joint_positions(robot, arm_joints, joint_values)
    world_from_tool = pp.get_link_pose(robot, tool_link)
    return pp.multiply(world_from_tool, pp.invert(grasp_bar_from_tool))


def build_default_paths() -> Tuple[str, str, str]:
    robot_cell_dir = os.path.join(DATA_DIR, "husky_assembly_design_study", "250904_transfer_path_test", "RobotCellStates")
    grasp_json = os.path.join(robot_cell_dir, "IK_test__GraspTargets.json")
    start_state = os.path.join(robot_cell_dir, "IK_test__20250905_101010_RobotCellState.json")
    end_state = os.path.join(robot_cell_dir, "IK_test__20250909_235058_RobotCellState.json")
    return grasp_json, start_state, end_state


def setup_stage1_scene(
    grasp_json: str,
    start_state_json: str,
    end_state_json: str,
    use_gui: bool = False,
) -> Dict[str, Any]:
    if not os.path.isfile(HUSKY_DUAL_URDF_PATH):
        raise FileNotFoundError(f"URDF not found: {HUSKY_DUAL_URDF_PATH}")

    cid = pp.connect(use_gui=use_gui)
    if sys.platform == "win32":
        sys.stdout = open("CONOUT$", "w")
        sys.stderr = open("CONOUT$", "w")
    if use_gui:
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 1, physicsClientId=cid)
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=2.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5],
            physicsClientId=cid,
        )

    with pp.LockRenderer():
        robot = pp.load_pybullet(HUSKY_DUAL_URDF_PATH, fixed_base=True)
    arm_joints = pp.joints_from_names(robot, HUSKY_DUAL_ARM_JOINT_NAMES)
    tool_link_left = pp.link_from_name(robot, TOOL_LINK_LEFT)
    tool_link_right = pp.link_from_name(robot, TOOL_LINK_RIGHT)
    pp.set_joint_positions(robot, arm_joints, INIT_ARM_JOINT_ANGLES)

    box_width, box_depth, box_length = BAR_BOX_DIMS
    bar_body = pp.create_box(box_width, box_depth, box_length, color=(0.8, 0.4, 0.1, 0.65))
    ghost_start = pp.create_box(box_width, box_depth, box_length, color=(0.0, 0.8, 0.0, 0.35))
    ghost_goal = pp.create_box(box_width, box_depth, box_length, color=(0.8, 0.0, 0.0, 0.35))

    grasp_targets = load_grasp_targets(grasp_json)
    if len(grasp_targets) < 1:
        raise ValueError(f"Expected at least one grasp target in {grasp_json}")
    world_from_bar_l, world_from_tool0_left = grasp_targets[0]
    grasp_bar_from_left = pp.multiply(pp.invert(world_from_bar_l), world_from_tool0_left)
    grasp_bar_from_right: Optional[PoseLike] = None
    if len(grasp_targets) >= 2:
        world_from_bar_r, world_from_tool0_right = grasp_targets[1]
        grasp_bar_from_right = pp.multiply(pp.invert(world_from_bar_r), world_from_tool0_right)

    start_joint_values = load_robot_cell_state(start_state_json)
    end_joint_values = load_robot_cell_state(end_state_json)

    start_pose_fk = compute_bar_pose_from_state(robot, arm_joints, tool_link_left, start_joint_values, grasp_bar_from_left)
    end_pose = compute_bar_pose_from_state(robot, arm_joints, tool_link_left, end_joint_values, grasp_bar_from_left)
    if grasp_bar_from_right is not None:
        start_pose_right = compute_bar_pose_from_state(
            robot, arm_joints, tool_link_right, start_joint_values, grasp_bar_from_right
        )
        end_pose_right = compute_bar_pose_from_state(
            robot, arm_joints, tool_link_right, end_joint_values, grasp_bar_from_right
        )
        if not pp.is_pose_close(start_pose_fk, start_pose_right, pos_tolerance=1e-4, ori_tolerance=1e-4):
            logger.warning("Start bar pose from left/right grasps does not match exactly; Stage 1 uses the left-arm result.")
            logger.warning(f"  left:  pos={np.round(start_pose_fk[0], 4)}, quat={np.round(start_pose_fk[1], 4)}")
            logger.warning(f"  right: pos={np.round(start_pose_right[0], 4)}, quat={np.round(start_pose_right[1], 4)}")
        if not pp.is_pose_close(end_pose, end_pose_right, pos_tolerance=1e-4, ori_tolerance=1e-4):
            logger.warning("Goal bar pose from left/right grasps does not match exactly; Stage 1 uses the left-arm result.")
            logger.warning(f"  left:  pos={np.round(end_pose[0], 4)}, quat={np.round(end_pose[1], 4)}")
            logger.warning(f"  right: pos={np.round(end_pose_right[0], 4)}, quat={np.round(end_pose_right[1], 4)}")

    start_pose = (
        np.asarray(start_pose_fk[0], dtype=float) + STAGE1_DEBUG_START_OFFSET,
        np.asarray(start_pose_fk[1], dtype=float),
    )
    logger.warning(
        "Stage 1 debug start pose is offset from the FK-consistent bar pose by %s in world coordinates. "
        "This intentionally makes the start bar pose incompatible with the start robot configuration and must be fixed later.",
        STAGE1_DEBUG_START_OFFSET.tolist(),
    )
    logger.warning(f"  FK start pose:      pos={np.round(start_pose_fk[0], 4)}, quat={np.round(start_pose_fk[1], 4)}")
    logger.warning(f"  Planning start pose: pos={np.round(start_pose[0], 4)}, quat={np.round(start_pose[1], 4)}")

    pp.set_joint_positions(robot, arm_joints, start_joint_values)
    pp.set_pose(bar_body, start_pose)
    pp.set_pose(ghost_start, start_pose)
    pp.set_pose(ghost_goal, end_pose)
    pp.add_text("Start", start_pose[0], color=(0.0, 0.8, 0.0, 1.0))
    pp.add_text("Goal", end_pose[0], color=(0.8, 0.0, 0.0, 1.0))

    collision_obstacles = [body for body in pp.get_bodies() if body not in {bar_body, ghost_start, ghost_goal}]

    return {
        "cid": cid,
        "robot": robot,
        "arm_joints": arm_joints,
        "tool_link_left": tool_link_left,
        "tool_link_right": tool_link_right,
        "bar_body": bar_body,
        "ghost_start": ghost_start,
        "ghost_goal": ghost_goal,
        "start_pose_fk": start_pose_fk,
        "start_pose": start_pose,
        "end_pose": end_pose,
        "start_joint_values": start_joint_values,
        "end_joint_values": end_joint_values,
        "grasp_bar_from_left": grasp_bar_from_left,
        "grasp_bar_from_right": grasp_bar_from_right,
        "collision_obstacles": collision_obstacles,
    }


def teardown_stage1_scene() -> None:
    pp.disconnect()


def run_stage_trial(
    stage: int,
    grasp_json: str,
    start_state_json: str,
    end_state_json: str,
    use_gui: bool = False,
    dist_metric: str = "feature",
    goal_bias: float = 0.1,
    position_res: float = 0.05,
    rotation_res: float = 0.1,
    max_time: float = 30.0,
    max_iterations: int = 2000,
    max_attempts: int = 5,
    endpoint_ik_attempts: int = 20,
    random_seed: Optional[int] = None,
    enable_collision: bool = True,
    joint_continuity_threshold_rad: Optional[float] = DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
    refine_after_plan: bool = True,
    refine_position_res: Optional[float] = None,
    refine_rotation_res: Optional[float] = None,
    refine_max_passes: int = 2,
    lock_renderer_during_search: bool = False,
    debug_tree_out: Optional[Dict] = None,
    planner_profile_out: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if stage not in {1, 2, 3}:
        raise ValueError(f"Unsupported stage: {stage}")
    scene = setup_stage1_scene(grasp_json, start_state_json, end_state_json, use_gui=use_gui)
    try:
        enable_ik = stage >= 2
        enforce_collision = stage >= 3
        rng = np.random.default_rng(random_seed)
        start_conf = None
        goal_conf = None
        joint_collision_fn = None
        if enable_ik:
            grasp_bar_from_right = scene["grasp_bar_from_right"]
            if grasp_bar_from_right is None:
                raise ValueError(f"Stage {stage} requires both left and right grasp targets.")
            t_endpoint = time.perf_counter()
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
                profile_out=planner_profile_out,
            )
            if start_conf is None:
                add_profile_time(planner_profile_out, "endpoint_ik_time_s", time.perf_counter() - t_endpoint)
                if planner_profile_out is not None:
                    planner_profile_out["outcome"] = "start_ik_failure"
                logger.warning(f"Stage {stage} start pose has no valid dual-arm IK solution.")
                return {"scene": scene, "path": None, "path_confs": None, "runtime_s": 0.0, "success": False}
            goal_conf = solve_endpoint_dual_arm_ik(
                robot=scene["robot"],
                arm_joints=scene["arm_joints"],
                tool_link_left=scene["tool_link_left"],
                tool_link_right=scene["tool_link_right"],
                bar_pose=scene["end_pose"],
                grasp_bar_from_left=scene["grasp_bar_from_left"],
                grasp_bar_from_right=grasp_bar_from_right,
                seed_conf=scene["end_joint_values"],
                rng=rng,
                max_attempts=endpoint_ik_attempts,
                profile_out=planner_profile_out,
            )
            add_profile_time(planner_profile_out, "endpoint_ik_time_s", time.perf_counter() - t_endpoint)
            if goal_conf is None:
                if planner_profile_out is not None:
                    planner_profile_out["outcome"] = "goal_ik_failure"
                logger.warning(f"Stage {stage} goal pose has no valid dual-arm IK solution.")
                return {"scene": scene, "path": None, "path_confs": None, "runtime_s": 0.0, "success": False}
            if enforce_collision:
                env_obstacles = [body for body in scene["collision_obstacles"] if body != scene["robot"]]
                joint_collision_fn = get_joint_collision_fn(
                    robot=scene["robot"],
                    arm_joints=scene["arm_joints"],
                    obstacle_bodies=env_obstacles,
                    tool_link_left=scene["tool_link_left"],
                    bar_body=scene["bar_body"],
                    grasp_bar_from_left=scene["grasp_bar_from_left"],
                )

        logger.info(f"Running minimal Stage {stage} RRT.")
        logger.info(f"  start pose: {np.round(scene['start_pose'][0], 4)}")
        logger.info(f"  goal pose:  {np.round(scene['end_pose'][0], 4)}")
        logger.info(f"  IK: {'on' if enable_ik else 'off'}")
        logger.info(f"  collision: {'on' if (enable_collision if stage == 1 else enforce_collision) else 'off'}")
        logger.info(f"  position_res: {position_res}")
        logger.info(f"  rotation_res: {rotation_res}")
        if enable_ik:
            logger.info(f"  joint continuity threshold: {joint_continuity_threshold_rad}")
            logger.info(f"  refine after plan: {'on' if refine_after_plan else 'off'}")
            if refine_after_plan:
                logger.info(f"  refine position_res: {refine_position_res if refine_position_res is not None else position_res / 2.0}")
                logger.info(f"  refine rotation_res: {refine_rotation_res if refine_rotation_res is not None else rotation_res / 2.0}")
                logger.info(f"  refine max passes: {refine_max_passes}")
        logger.info(f"  collision obstacles: {len(scene['collision_obstacles'])} bodies")
        logger.info(f"  lock renderer during search: {'on' if (use_gui and lock_renderer_during_search) else 'off'}")

        planning_kwargs = dict(
            robot=scene["robot"],
            bar_body=scene["bar_body"],
            obstacle_bodies=scene["collision_obstacles"],
            start_pose=scene["start_pose"],
            goal_pose=scene["end_pose"],
            start_conf=start_conf,
            goal_conf=goal_conf,
            dist_metric=dist_metric,
            goal_sample_prob=goal_bias,
            position_res=position_res,
            rotation_res=rotation_res,
            random_seed=random_seed,
            max_time=max_time,
            max_iterations=max_iterations,
            max_attempts=max_attempts,
            enable_collision=(enable_collision if stage == 1 else enforce_collision),
            enable_ik=enable_ik,
            ik_context={
                "robot": scene["robot"],
                "arm_joints": scene["arm_joints"],
                "tool_link_left": scene["tool_link_left"],
                "tool_link_right": scene["tool_link_right"],
                "grasp_bar_from_left": scene["grasp_bar_from_left"],
                "grasp_bar_from_right": scene["grasp_bar_from_right"],
            }
            if enable_ik
            else None,
            joint_collision_fn=joint_collision_fn,
            joint_continuity_threshold_rad=(joint_continuity_threshold_rad if enable_ik else None),
            use_draw=use_gui,
            debug_tree_out=debug_tree_out,
            profile_out=planner_profile_out,
        )
        t0 = time.perf_counter()
        if use_gui and lock_renderer_during_search:
            with pp.LockRenderer():
                path, path_confs = plan_pose_rrt(**planning_kwargs)
        else:
            path, path_confs = plan_pose_rrt(**planning_kwargs)
        runtime_s = time.perf_counter() - t0

        if path is not None:
            logger.info(f"Found Stage {stage} pose path with {len(path)} waypoints.")
            pp.set_pose(scene["bar_body"], path[-1])
            if enable_ik and path_confs:
                pp.set_joint_positions(scene["robot"], scene["arm_joints"], path_confs[-1])
        else:
            logger.warning(f"No Stage {stage} pose path found.")

        coarse_path = path
        coarse_path_confs = path_confs
        coarse_continuity = summarize_joint_continuity(path_confs) if path_confs is not None else None
        refinement: Optional[Dict[str, Any]] = None
        if enable_ik and path is not None and path_confs is not None and start_conf is not None:
            refinement = {
                "enabled": bool(refine_after_plan),
                "attempted": False,
                "used_refined_path": False,
                "status": "disabled",
                "coarse_waypoints": len(path),
                "final_waypoints": len(path),
                "coarse_joint_continuity": coarse_continuity,
                "final_joint_continuity": coarse_continuity,
                "passes": [],
            }
            if refine_after_plan:
                refinement = refine_pose_path_with_seed_chained_ik(
                    scene=scene,
                    path=path,
                    path_confs=path_confs,
                    start_conf=start_conf,
                    base_position_res=(refine_position_res if refine_position_res is not None else max(position_res / 2.0, 1e-4)),
                    base_rotation_res=(refine_rotation_res if refine_rotation_res is not None else max(rotation_res / 2.0, 1e-4)),
                    refine_max_passes=refine_max_passes,
                    joint_continuity_threshold_rad=joint_continuity_threshold_rad,
                    joint_collision_fn=joint_collision_fn,
                    profile_out=planner_profile_out,
                )
                if refinement.get("used_refined_path"):
                    path = refinement["path"]
                    path_confs = refinement["path_confs"]
                    logger.info(
                        "Refinement accepted: %s -> %s waypoints, max dq %.4f -> %.4f rad",
                        refinement.get("coarse_waypoints"),
                        refinement.get("final_waypoints"),
                        float((refinement.get("coarse_joint_continuity") or {}).get("max_delta_rad") or 0.0),
                        float((refinement.get("final_joint_continuity") or {}).get("max_delta_rad") or 0.0),
                    )
                    pp.set_pose(scene["bar_body"], path[-1])
                    pp.set_joint_positions(scene["robot"], scene["arm_joints"], path_confs[-1])
                else:
                    logger.info(
                        "Refinement result: %s (max dq %.4f -> %.4f rad)",
                        refinement.get("status"),
                        float((refinement.get("coarse_joint_continuity") or {}).get("max_delta_rad") or 0.0),
                        float((refinement.get("final_joint_continuity") or {}).get("max_delta_rad") or 0.0),
                    )

        validation_joint_path, validation_joint_path_source, validation_joint_path_reason = build_validation_joint_path(
            scene=scene,
            path=path,
            path_confs=path_confs,
            start_conf=start_conf,
            endpoint_ik_attempts=endpoint_ik_attempts,
            random_seed=random_seed,
        )
        validation = validate_stage_trajectory(
            stage=stage,
            scene=scene,
            path=path,
            joint_path=validation_joint_path,
            joint_path_source=validation_joint_path_source,
            joint_path_reason=validation_joint_path_reason,
            urdf_path=HUSKY_DUAL_URDF_PATH,
            srdf_path=HUSKY_DUAL_SRDF_PATH,
            grasp_mask_links=STAGE3_GRASP_MASK_LINKS,
        )
        log_validation_summary(validation)
        refinement_summary = None
        if refinement is not None:
            refinement_summary = {k: v for k, v in refinement.items() if k not in {"path", "path_confs"}}
        path_found = path is not None
        validated_success = path_found
        if stage >= 2:
            validated_success = validated_success and bool(validation.get("joint_continuity_ok"))
            validated_success = validated_success and bool(validation.get("collision_free"))

        return {
            "stage": stage,
            "scene": scene,
            "path": path,
            "path_confs": path_confs,
            "coarse_path": coarse_path,
            "coarse_path_confs": coarse_path_confs,
            "coarse_joint_continuity": coarse_continuity,
            "refinement": refinement_summary,
            "validation_joint_path": validation_joint_path,
            "validation_joint_path_source": validation_joint_path_source,
            "validation": validation,
            "start_conf": start_conf,
            "goal_conf": goal_conf,
            "runtime_s": runtime_s,
            "path_found": path_found,
            "success": bool(validated_success),
        }
    except Exception:
        teardown_stage1_scene()
        raise


def run_stage1_trial(**kwargs) -> Dict[str, Any]:
    return run_stage_trial(stage=1, **kwargs)


def run_stage2_trial(**kwargs) -> Dict[str, Any]:
    return run_stage_trial(stage=2, **kwargs)


def run_stage3_trial(**kwargs) -> Dict[str, Any]:
    return run_stage_trial(stage=3, **kwargs)


def build_validation_joint_path(
    scene: Dict[str, Any],
    path: Optional[Sequence[PoseLike]],
    path_confs: Optional[Sequence[FullConf]],
    start_conf: Optional[FullConf],
    endpoint_ik_attempts: int,
    random_seed: Optional[int],
) -> Tuple[Optional[List[FullConf]], Optional[str], Optional[str]]:
    if path is None:
        return None, None, "no_path"
    if path_confs is not None:
        if len(path_confs) != len(path):
            return None, "planner", "planner_joint_path_length_mismatch"
        return [normalize_angles(np.asarray(conf, dtype=float)) for conf in path_confs], "planner", None

    grasp_bar_from_right = scene["grasp_bar_from_right"]
    if grasp_bar_from_right is None:
        return None, None, "missing_right_grasp_for_validation"

    rng = np.random.default_rng(random_seed)
    if start_conf is not None:
        current_conf: Optional[FullConf] = normalize_angles(np.asarray(start_conf, dtype=float))
    else:
        current_conf = solve_endpoint_dual_arm_ik(
            robot=scene["robot"],
            arm_joints=scene["arm_joints"],
            tool_link_left=scene["tool_link_left"],
            tool_link_right=scene["tool_link_right"],
            bar_pose=path[0],
            grasp_bar_from_left=scene["grasp_bar_from_left"],
            grasp_bar_from_right=grasp_bar_from_right,
            seed_conf=scene["start_joint_values"],
            rng=rng,
            max_attempts=endpoint_ik_attempts,
        )
    if current_conf is None:
        return None, "reconstructed", "validation_start_ik_failure"

    joint_path = [current_conf]
    for idx, pose in enumerate(path[1:], start=1):
        next_conf = solve_dual_arm_pose_ik(
            robot=scene["robot"],
            arm_joints=scene["arm_joints"],
            tool_link_left=scene["tool_link_left"],
            tool_link_right=scene["tool_link_right"],
            bar_pose=pose,
            grasp_bar_from_left=scene["grasp_bar_from_left"],
            grasp_bar_from_right=grasp_bar_from_right,
            seed_conf=current_conf,
        )
        if next_conf is None:
            return None, "reconstructed", f"validation_ik_failure_at_waypoint_{idx}"
        current_conf = normalize_angles(np.asarray(next_conf, dtype=float))
        joint_path.append(current_conf)
    return joint_path, "reconstructed", None


def log_validation_summary(validation: Dict[str, Any]) -> None:
    plot_path = validation.get("plot_path")
    if plot_path:
        logger.info(f"Saved trajectory validation plot: {plot_path}")

    collision_free = validation.get("collision_free")
    if collision_free is None:
        reason = validation.get("joint_path_reason")
        if reason:
            logger.warning(f"Trajectory validation could not run robot-side checks: {reason}")
    else:
        logger.info(f"Trajectory collisions: {'pass' if collision_free else 'fail'}")
        for key, details in validation.get("collision_breakdown", {}).items():
            if details.get("count", 0) <= 0:
                continue
            logger.warning(
                "  %s collisions: %s hits (first waypoint %s)",
                key,
                details["count"],
                details["first_index"],
            )

    joint_continuity_ok = validation.get("joint_continuity_ok")
    if joint_continuity_ok is not None:
        logger.info(
            "Joint continuity: %s (max dq=%.4f rad, threshold=%.4f rad)",
            "pass" if joint_continuity_ok else "fail",
            float(validation.get("joint_continuity_max_delta_rad") or 0.0),
            float(validation.get("joint_continuity_threshold_rad") or 0.0),
        )
        if validation.get("joint_continuity_first_bad_step") is not None:
            logger.warning(f"  first joint continuity violation at waypoint {validation['joint_continuity_first_bad_step']}")

    relative_transform_ok = validation.get("relative_transform_ok")
    if relative_transform_ok is not None:
        max_axis_drift_deg = validation.get("relative_transform_max_axis_angle_deg") or {}
        logger.info(
            "End-effector relative transform drift: %s (max pos=%.6f m, max axis drift xyz=[%.3f, %.3f, %.3f] deg)",
            "pass" if relative_transform_ok else "fail",
            float(validation.get("relative_transform_max_translation_m") or 0.0),
            float(max_axis_drift_deg.get("x") or 0.0),
            float(max_axis_drift_deg.get("y") or 0.0),
            float(max_axis_drift_deg.get("z") or 0.0),
        )


def run_visualization_loop(
    bar_body: int,
    path: Sequence[PoseLike],
    cid: int,
    robot: Optional[int] = None,
    arm_joints: Optional[Sequence[int]] = None,
    path_confs: Optional[Sequence[FullConf]] = None,
) -> None:
    if path is None:
        logger.info("No path to visualize. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            return

    logger.info(f"Visualizing pose path with {len(path)} waypoints.")
    path_slider = pybullet.addUserDebugParameter("Path t", 0.0, 1.0, 0.0, physicsClientId=cid)
    current_idx = -1
    while True:
        try:
            t = pybullet.readUserDebugParameter(path_slider, physicsClientId=cid)
            idx = int(round(t * (len(path) - 1)))
            idx = max(0, min(idx, len(path) - 1))
            if idx != current_idx:
                current_idx = idx
                pp.set_pose(bar_body, path[idx])
                if robot is not None and arm_joints is not None and path_confs is not None and idx < len(path_confs):
                    pp.set_joint_positions(robot, arm_joints, path_confs[idx])
            time.sleep(0.01)
        except KeyboardInterrupt:
            return


def main() -> None:
    default_grasp_json, default_start_state, default_end_state = build_default_paths()
    parser = argparse.ArgumentParser(description="Minimal Stage 1/2/3 floating-bar RRT")
    parser.add_argument("--grasp-json", type=str, default=default_grasp_json, help="Path to grasp JSON file")
    parser.add_argument("--start-state", type=str, default=default_start_state, help="Path to start RobotCellState JSON")
    parser.add_argument("--end-state", type=str, default=default_end_state, help="Path to end RobotCellState JSON")
    parser.add_argument("--stage", choices=[1, 2, 3], type=int, default=3, help="Planning stage to run")
    parser.add_argument("--no-gui", action="store_true", help="Run without PyBullet GUI")
    parser.add_argument("--goal-bias", type=float, default=0.1, help="Goal sampling probability")
    parser.add_argument("--dist-metric", choices=["feature", "pose6d"], default="feature", help="Task-space distance metric")
    parser.add_argument("--position-res", type=float, default=0.01, help="Translation resolution used during pose extension, in meters")
    parser.add_argument("--rotation-res", type=float, default=0.025, help="Rotation resolution used during pose extension, in radians")
    parser.add_argument("--max-time", type=float, default=30.0, help="Max planning time per attempt")
    parser.add_argument("--max-iterations", type=int, default=2000, help="Max RRT iterations per attempt")
    parser.add_argument("--max-attempts", type=int, default=5, help="Random restarts")
    parser.add_argument("--endpoint-ik-attempts", type=int, default=20, help="Max random seeds used when solving endpoint IK in Stage 2/3")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--joint-continuity-threshold",
        type=float,
        default=DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
        help="Maximum allowed wrapped joint delta between neighboring Stage 2/3 configurations, in radians",
    )
    parser.add_argument(
        "--no-refine-after-plan",
        action="store_true",
        help="Disable dense post-plan seed-chained IK refinement for Stage 2/3",
    )
    parser.add_argument(
        "--refine-position-res",
        type=float,
        default=None,
        help="Initial translation resolution used during Stage 2/3 post-plan refinement, in meters",
    )
    parser.add_argument(
        "--refine-rotation-res",
        type=float,
        default=None,
        help="Initial rotation resolution used during Stage 2/3 post-plan refinement, in radians",
    )
    parser.add_argument(
        "--refine-max-passes",
        type=int,
        default=2,
        help="Maximum number of Stage 2/3 post-plan refinement passes; each pass halves the refinement resolution",
    )
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
    parser.set_defaults(floating_collision=False, lock_renderer_during_search=True)
    args = parser.parse_args()

    use_gui = not args.no_gui
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
        refine_after_plan=not args.no_refine_after_plan,
        refine_position_res=args.refine_position_res,
        refine_rotation_res=args.refine_rotation_res,
        refine_max_passes=args.refine_max_passes,
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
