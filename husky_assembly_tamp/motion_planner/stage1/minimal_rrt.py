"""
Minimal Stage 1 floating-bar RRT.

This is a clean restart from the original design intent:
- task-space only
- no IK in the planner loop
- no ladder graph
- optional floating-body collision against a fixed robot

"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet
import pybullet_planning as pp
from pybullet_planning.motion_planners.rrt import TreeNode, configs

from husky_assembly_tamp.utils.params import DATA_DIR
from husky_assembly_tamp.utils.util import setup_logger


logger = setup_logger("stage1_minimal_rrt")

HUSKY_DUAL_URDF_PATH = os.path.join(
    DATA_DIR,
    "husky_urdf/mt_husky_dual_ur5_e_moveit_config/urdf/husky_dual_ur5_e_no_base_joint_All_Calibrated.urdf",
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
INIT_ARM_JOINT_ANGLES = np.array([0.0, -np.pi / 2.0, 0.0, 0.0, 0.0, 0.0] * 2, dtype=float)
BAR_RADIUS = 0.015
BAR_LENGTH = 1.0
BAR_BOX_DIMS = (2.0 * BAR_RADIUS, 2.0 * BAR_RADIUS, BAR_LENGTH)
STAGE1_DEBUG_START_OFFSET = np.array([-0.5, 0.0, 0.5], dtype=float)


PoseLike = Tuple[np.ndarray, np.ndarray]
GraspTarget = Tuple[PoseLike, PoseLike]


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
) -> PoseLike:
    if rng.random() < goal_sample_prob:
        return goal_pose
    base_pos, _ = pp.get_pose(robot)
    cx, cy, cz = np.asarray(base_pos, dtype=float)
    x = cx + rng.uniform(-workspace_xy / 2.0, workspace_xy / 2.0)
    y = cy + rng.uniform(-workspace_xy / 2.0, workspace_xy / 2.0)
    z_min = max(0.05, cz)
    z = rng.uniform(z_min, z_min + workspace_z)
    roll = rng.uniform(-np.pi, np.pi)
    pitch = rng.uniform(-np.pi, np.pi)
    yaw = rng.uniform(-np.pi, np.pi)
    return pp.Pose(point=[x, y, z], euler=pp.Euler(roll, pitch, yaw))


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


def add_profile_time(profile_out: Optional[Dict[str, Any]], key: str, dt: float) -> None:
    if profile_out is None:
        return
    profile_out[key] = float(profile_out.get(key, 0.0)) + float(dt)


def bump_profile_count(profile_out: Optional[Dict[str, Any]], key: str, inc: int = 1) -> None:
    if profile_out is None:
        return
    profile_out[key] = int(profile_out.get(key, 0)) + int(inc)


def extend_toward(
    nodes: List[TreeNode],
    source: TreeNode,
    target_pose: PoseLike,
    collision_fn: Callable[[PoseLike], bool],
    draw_color: Tuple[float, float, float, float],
    use_draw: bool,
    position_res: float,
    rotation_res: float,
    dist_metric: str,
    feature_points: Sequence[np.ndarray],
    feature_vecs: Dict[int, np.ndarray],
    profile_out: Optional[Dict[str, Any]] = None,
) -> Tuple[TreeNode, bool]:
    current = source
    reached = True
    for pose in list(
        pp.interpolate_poses(
            source.config,
            target_pose,
            pos_step_size=max(position_res, 1e-6),
            ori_step_size=max(rotation_res, 1e-6),
        )
    )[1:]:
        bump_profile_count(profile_out, "poses_checked")
        if collision_fn(pose):
            reached = False
            bump_profile_count(profile_out, "collision_hits")
            break
        node = TreeNode(pose, parent=current)
        nodes.append(node)
        bump_profile_count(profile_out, "nodes_created")
        if dist_metric == "feature":
            feature_vec = pose_to_feature_vec(pose, feature_points)
            if feature_vec is not None:
                feature_vecs[id(node)] = feature_vec
        if use_draw:
            pp.add_line(current.config[0], node.config[0], width=1.5, color=draw_color)
        current = node
    return current, reached


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
    use_draw: bool = True,
    debug_tree_out: Optional[Dict] = None,
    profile_out: Optional[Dict[str, Any]] = None,
) -> Optional[List[PoseLike]]:
    rng = np.random.default_rng(random_seed)
    feature_points = get_bar_feature_points()
    collision_fn = get_pose_collision_fn(bar_body, obstacle_bodies, enable_collision)
    if profile_out is not None:
        profile_out.clear()
        profile_out.update(
            {
                "attempts": 0,
                "iterations": 0,
                "nodes_created": 0,
                "poses_checked": 0,
                "collision_hits": 0,
            }
        )

    t_collision = time.perf_counter()
    start_in_collision = collision_fn(start_pose)
    add_profile_time(profile_out, "collision_check_time_s", time.perf_counter() - t_collision)
    if start_in_collision:
        logger.warning("Start pose is in floating-body collision.")
        if profile_out is not None:
            profile_out["outcome"] = "start_in_collision"
        return None
    t_collision = time.perf_counter()
    goal_in_collision = collision_fn(goal_pose)
    add_profile_time(profile_out, "collision_check_time_s", time.perf_counter() - t_collision)
    if goal_in_collision:
        logger.warning("Goal pose is in floating-body collision.")
        if profile_out is not None:
            profile_out["outcome"] = "goal_in_collision"
        return None

    best_tree: List[TreeNode] = []
    total_iterations = 0
    for attempt in range(max_attempts):
        bump_profile_count(profile_out, "attempts")
        start_time = time.time()
        root = TreeNode(start_pose)
        nodes = [root]
        feature_vecs: Dict[int, np.ndarray] = {}
        bump_profile_count(profile_out, "nodes_created")
        if dist_metric == "feature":
            t_feature = time.perf_counter()
            root_feature = pose_to_feature_vec(start_pose, feature_points)
            add_profile_time(profile_out, "feature_time_s", time.perf_counter() - t_feature)
            if root_feature is not None:
                feature_vecs[id(root)] = root_feature

        t_extend = time.perf_counter()
        direct_last, direct_ok = extend_toward(
            nodes=nodes,
            source=root,
            target_pose=goal_pose,
            collision_fn=collision_fn,
            draw_color=(0.2, 0.8, 0.2, 0.6),
            use_draw=use_draw,
            position_res=position_res,
            rotation_res=rotation_res,
            dist_metric=dist_metric,
            feature_points=feature_points,
            feature_vecs=feature_vecs,
            profile_out=profile_out,
        )
        add_profile_time(profile_out, "extend_direct_time_s", time.perf_counter() - t_extend)
        if direct_ok:
            update_debug_tree(debug_tree_out, True, 0, nodes, start_pose, goal_pose)
            if profile_out is not None:
                profile_out["iterations"] = total_iterations
                profile_out["outcome"] = "success"
            return configs(direct_last.retrace())

        for iteration in range(max_iterations):
            total_iterations += 1
            if profile_out is not None:
                profile_out["iterations"] = total_iterations
            if (time.time() - start_time) >= max_time:
                break
            t_sample = time.perf_counter()
            target_pose = sample_pose(robot, goal_pose, rng, goal_sample_prob, workspace_xy, workspace_z)
            add_profile_time(profile_out, "sample_time_s", time.perf_counter() - t_sample)
            t_nearest = time.perf_counter()
            nearest = nearest_node(nodes, target_pose, dist_metric, feature_points, feature_vecs)
            add_profile_time(profile_out, "nearest_time_s", time.perf_counter() - t_nearest)
            t_extend = time.perf_counter()
            new_last, reached = extend_toward(
                nodes=nodes,
                source=nearest,
                target_pose=target_pose,
                collision_fn=collision_fn,
                draw_color=(0.85, 0.2, 0.2, 0.45),
                use_draw=use_draw,
                position_res=position_res,
                rotation_res=rotation_res,
                dist_metric=dist_metric,
                feature_points=feature_points,
                feature_vecs=feature_vecs,
                profile_out=profile_out,
            )
            add_profile_time(profile_out, "extend_tree_time_s", time.perf_counter() - t_extend)
            if not reached:
                continue
            t_goal_dist = time.perf_counter()
            if pose_distance(new_last.config, goal_pose, dist_metric, feature_points) <= max(position_res, rotation_res):
                add_profile_time(profile_out, "goal_test_time_s", time.perf_counter() - t_goal_dist)
                t_goal_extend = time.perf_counter()
                goal_last, goal_ok = extend_toward(
                    nodes=nodes,
                    source=new_last,
                    target_pose=goal_pose,
                    collision_fn=collision_fn,
                    draw_color=(0.1, 0.7, 0.1, 0.85),
                    use_draw=use_draw,
                    position_res=position_res,
                    rotation_res=rotation_res,
                    dist_metric=dist_metric,
                    feature_points=feature_points,
                    feature_vecs=feature_vecs,
                    profile_out=profile_out,
                )
                add_profile_time(profile_out, "extend_goal_time_s", time.perf_counter() - t_goal_extend)
                if goal_ok:
                    update_debug_tree(debug_tree_out, True, iteration + 1, nodes, start_pose, goal_pose)
                    if profile_out is not None:
                        profile_out["outcome"] = "success"
                    return configs(goal_last.retrace())
            else:
                add_profile_time(profile_out, "goal_test_time_s", time.perf_counter() - t_goal_dist)
        best_tree = nodes
        logger.info(f"Attempt {attempt + 1}/{max_attempts}: no path found.")

    update_debug_tree(debug_tree_out, False, total_iterations, best_tree, start_pose, goal_pose)
    if profile_out is not None:
        profile_out["iterations"] = total_iterations
        profile_out["outcome"] = "task_space_failure"
    return None


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
        "bar_body": bar_body,
        "ghost_start": ghost_start,
        "ghost_goal": ghost_goal,
        "start_pose_fk": start_pose_fk,
        "start_pose": start_pose,
        "end_pose": end_pose,
        "collision_obstacles": collision_obstacles,
    }


def teardown_stage1_scene() -> None:
    pp.disconnect()


def run_stage1_trial(
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
    random_seed: Optional[int] = None,
    enable_collision: bool = True,
    lock_renderer_during_search: bool = False,
    debug_tree_out: Optional[Dict] = None,
    planner_profile_out: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    scene = setup_stage1_scene(grasp_json, start_state_json, end_state_json, use_gui=use_gui)
    try:
        logger.info("Running minimal Stage 1 RRT.")
        logger.info(f"  start pose: {np.round(scene['start_pose'][0], 4)}")
        logger.info(f"  goal pose:  {np.round(scene['end_pose'][0], 4)}")
        logger.info(f"  floating collision: {'on' if enable_collision else 'off'}")
        logger.info(f"  position_res: {position_res}")
        logger.info(f"  rotation_res: {rotation_res}")
        logger.info(f"  floating collision obstacles: {len(scene['collision_obstacles'])} bodies")
        logger.info(f"  lock renderer during search: {'on' if (use_gui and lock_renderer_during_search) else 'off'}")

        planning_kwargs = dict(
            robot=scene["robot"],
            bar_body=scene["bar_body"],
            obstacle_bodies=scene["collision_obstacles"],
            start_pose=scene["start_pose"],
            goal_pose=scene["end_pose"],
            dist_metric=dist_metric,
            goal_sample_prob=goal_bias,
            position_res=position_res,
            rotation_res=rotation_res,
            random_seed=random_seed,
            max_time=max_time,
            max_iterations=max_iterations,
            max_attempts=max_attempts,
            enable_collision=enable_collision,
            use_draw=use_gui,
            debug_tree_out=debug_tree_out,
            profile_out=planner_profile_out,
        )
        t0 = time.perf_counter()
        if use_gui and lock_renderer_during_search:
            with pp.LockRenderer():
                path = plan_pose_rrt(**planning_kwargs)
        else:
            path = plan_pose_rrt(**planning_kwargs)
        runtime_s = time.perf_counter() - t0

        if path is not None:
            logger.info(f"Found Stage 1 pose path with {len(path)} waypoints.")
            pp.set_pose(scene["bar_body"], path[-1])
        else:
            logger.warning("No Stage 1 pose path found.")

        return {
            "scene": scene,
            "path": path,
            "runtime_s": runtime_s,
            "success": path is not None,
        }
    except Exception:
        teardown_stage1_scene()
        raise


def run_visualization_loop(bar_body: int, path: Sequence[PoseLike], cid: int) -> None:
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
            time.sleep(0.01)
        except KeyboardInterrupt:
            return


def main() -> None:
    default_grasp_json, default_start_state, default_end_state = build_default_paths()
    parser = argparse.ArgumentParser(description="Minimal Stage 1 floating-bar RRT")
    parser.add_argument("--grasp-json", type=str, default=default_grasp_json, help="Path to grasp JSON file")
    parser.add_argument("--start-state", type=str, default=default_start_state, help="Path to start RobotCellState JSON")
    parser.add_argument("--end-state", type=str, default=default_end_state, help="Path to end RobotCellState JSON")
    parser.add_argument("--no-gui", action="store_true", help="Run without PyBullet GUI")
    parser.add_argument("--goal-bias", type=float, default=0.1, help="Goal sampling probability")
    parser.add_argument("--dist-metric", choices=["feature", "pose6d"], default="feature", help="Task-space distance metric")
    parser.add_argument("--position-res", type=float, default=0.1, help="Translation resolution used during pose extension, in meters")
    parser.add_argument("--rotation-res", type=float, default=0.2, help="Rotation resolution used during pose extension, in radians")
    parser.add_argument("--max-time", type=float, default=30.0, help="Max planning time per attempt")
    parser.add_argument("--max-iterations", type=int, default=2000, help="Max RRT iterations per attempt")
    parser.add_argument("--max-attempts", type=int, default=5, help="Random restarts")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed")
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
    parser.set_defaults(floating_collision=True)
    args = parser.parse_args()

    use_gui = not args.no_gui
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
