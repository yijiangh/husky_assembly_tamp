"""
Minimal Stage 1 floating-bar RRT.

This is a clean restart from the original design intent:
- task-space only
- no IK in the planner loop
- no ladder graph
- optional floating-body collision against a fixed robot

Usage:
    python -m husky_assembly_tamp.motion_planner.stage1.minimal_rrt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet
import pybullet_planning as pp

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


@dataclass
class PoseNode:
    pose: PoseLike
    parent: Optional["PoseNode"] = None
    feature_vec: Optional[np.ndarray] = None

    def retrace(self) -> List["PoseNode"]:
        nodes: List[PoseNode] = []
        current: Optional[PoseNode] = self
        while current is not None:
            nodes.append(current)
            current = current.parent
        return list(reversed(nodes))


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


def poses_close(pose1: PoseLike, pose2: PoseLike, pos_tol: float = 1e-4, rot_tol: float = 1e-4) -> bool:
    return bool(pp.is_pose_close(pose1, pose2, pos_tolerance=pos_tol, ori_tolerance=rot_tol))


def cart_linear_interp(pose1: PoseLike, pose2: PoseLike, position_res: float, rotation_res: float) -> List[PoseLike]:
    pos1, quat1 = pose1
    pos2, quat2 = pose2
    return list(
        pp.interpolate_poses(
            (pos1, quat1),
            (pos2, quat2),
            pos_step_size=max(position_res, 1e-6),
            ori_step_size=max(rotation_res, 1e-6),
        )
    )


class FloatingBarRRT:
    def __init__(
        self,
        robot: int,
        bar_body: int,
        obstacle_bodies: Optional[Sequence[int]] = None,
        dist_metric: str = "feature",
        goal_sample_prob: float = 0.1,
        workspace_xy: float = 2.2,
        workspace_z: float = 1.2,
        position_res: float = 0.05,
        rotation_res: float = 0.1,
        random_seed: Optional[int] = None,
    ):
        self.robot = robot
        self.bar_body = bar_body
        self.obstacle_bodies = list(obstacle_bodies) if obstacle_bodies is not None else [robot]
        self.dist_metric = dist_metric
        self.goal_sample_prob = float(goal_sample_prob)
        self.workspace_xy = float(workspace_xy)
        self.workspace_z = float(workspace_z)
        self.position_res = float(position_res)
        self.rotation_res = float(rotation_res)
        self.rng = np.random.default_rng(random_seed)
        self._bar_local_feature_points = self._compute_bar_feature_points()

    def _compute_bar_feature_points(self) -> Optional[List[np.ndarray]]:
        try:
            half_width, half_depth, half_length = 0.5 * np.asarray(BAR_BOX_DIMS, dtype=float)
            return [
                np.array([sx * half_width, sy * half_depth, sz * half_length], dtype=float)
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ]
        except Exception:
            return None

    def _pose_to_feature_vec(self, pose: PoseLike) -> Optional[np.ndarray]:
        if not self._bar_local_feature_points:
            return None
        pts = []
        for p_local in self._bar_local_feature_points:
            p_world, _ = pp.multiply(pose, (p_local, [0, 0, 0, 1]))
            pts.append(np.asarray(p_world, dtype=float))
        return np.concatenate(pts, axis=0)

    def _attach_feature_vec(self, node: PoseNode) -> None:
        if self.dist_metric == "feature":
            node.feature_vec = self._pose_to_feature_vec(node.pose)

    def _pose_distance(self, pose1: PoseLike, pose2: PoseLike) -> float:
        pos1, quat1 = pose1
        pos2, quat2 = pose2
        if self.dist_metric == "feature":
            vec1 = self._pose_to_feature_vec((pos1, quat1))
            vec2 = self._pose_to_feature_vec((pos2, quat2))
            if vec1 is not None and vec2 is not None:
                return float(np.linalg.norm(vec2 - vec1))
        dx = pos2 - pos1
        rot_dist = pp.quat_angle_between(quat1, quat2)
        return float(np.linalg.norm(np.array([dx[0], dx[1], dx[2], rot_dist], dtype=float)))

    def _sample_pose(self, goal_pose: PoseLike) -> PoseLike:
        if self.rng.random() < self.goal_sample_prob:
            return goal_pose
        base_pos, _ = pp.get_pose(self.robot)
        cx, cy, cz = np.asarray(base_pos, dtype=float)
        x = cx + self.rng.uniform(-self.workspace_xy / 2.0, self.workspace_xy / 2.0)
        y = cy + self.rng.uniform(-self.workspace_xy / 2.0, self.workspace_xy / 2.0)
        z_min = max(0.05, cz)
        z = self.rng.uniform(z_min, z_min + self.workspace_z)
        roll = self.rng.uniform(-np.pi, np.pi)
        pitch = self.rng.uniform(-np.pi, np.pi)
        yaw = self.rng.uniform(-np.pi, np.pi)
        return pp.Pose(point=[x, y, z], euler=pp.Euler(roll, pitch, yaw))

    def _nearest_node(self, nodes: List[PoseNode], target_pose: PoseLike) -> PoseNode:
        if self.dist_metric == "feature":
            target_vec = self._pose_to_feature_vec(target_pose)
            if target_vec is not None:
                return min(
                    nodes,
                    key=lambda node: float(
                        np.linalg.norm(
                            (node.feature_vec if node.feature_vec is not None else self._pose_to_feature_vec(node.pose))
                            - target_vec
                        )
                    ),
                )
        return min(nodes, key=lambda node: self._pose_distance(node.pose, target_pose))

    def _export_tree(self, nodes: List[PoseNode]) -> Dict[str, List[List[float]]]:
        id_to_idx: Dict[int, int] = {}
        points: List[List[float]] = []
        for node in nodes:
            idx = len(points)
            id_to_idx[id(node)] = idx
            pos = np.asarray(node.pose[0], dtype=float).reshape(3)
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

    def _make_collision_fn(self, enable_collision: bool) -> Callable[[PoseLike], bool]:
        if not enable_collision:
            return lambda pose: False
        floating_collision_fn = pp.get_floating_body_collision_fn(
            self.bar_body,
            obstacles=self.obstacle_bodies,
            disabled_collisions=[],
        )
        return lambda pose: bool(floating_collision_fn(pose))

    def _extend_toward(
        self,
        nodes: List[PoseNode],
        source: PoseNode,
        target_pose: PoseLike,
        collision_fn: Callable[[PoseLike], bool],
        draw_color: Tuple[float, float, float, float],
        use_draw: bool,
    ) -> Tuple[PoseNode, bool]:
        current = source
        reached = True
        for pose in cart_linear_interp(source.pose, target_pose, self.position_res, self.rotation_res)[1:]:
            if collision_fn(pose):
                reached = False
                break
            node = PoseNode(pose=pose, parent=current)
            self._attach_feature_vec(node)
            nodes.append(node)
            if use_draw:
                pp.add_line(current.pose[0], node.pose[0], width=1.5, color=draw_color)
            current = node
        return current, reached

    def plan(
        self,
        start_pose: PoseLike,
        goal_pose: PoseLike,
        max_time: float = 30.0,
        max_iterations: int = 2000,
        max_attempts: int = 5,
        enable_collision: bool = False,
        use_draw: bool = True,
        debug_tree_out: Optional[Dict] = None,
    ) -> Optional[List[PoseLike]]:
        collision_fn = self._make_collision_fn(enable_collision)

        if collision_fn(start_pose):
            logger.warning("Start pose is in floating-body collision.")
            return None
        if collision_fn(goal_pose):
            logger.warning("Goal pose is in floating-body collision.")
            return None

        best_tree: List[PoseNode] = []
        total_iterations = 0
        for attempt in range(max_attempts):
            start_time = time.time()
            root = PoseNode(start_pose)
            self._attach_feature_vec(root)
            nodes = [root]

            direct_last, direct_ok = self._extend_toward(
                nodes=nodes,
                source=root,
                target_pose=goal_pose,
                collision_fn=collision_fn,
                draw_color=(0.2, 0.8, 0.2, 0.6),
                use_draw=use_draw,
            )
            if direct_ok:
                path_nodes = direct_last.retrace()
                if debug_tree_out is not None:
                    debug_tree_out.clear()
                    debug_tree_out["success"] = True
                    debug_tree_out["iterations"] = 0
                    debug_tree_out["tree1"] = self._export_tree(nodes)
                    debug_tree_out["tree2"] = {"points": [], "edges": []}
                    debug_tree_out["start_pose"] = [float(v) for v in start_pose[0]]
                    debug_tree_out["goal_pose"] = [float(v) for v in goal_pose[0]]
                return [node.pose for node in path_nodes]

            for iteration in range(max_iterations):
                total_iterations += 1
                if (time.time() - start_time) >= max_time:
                    break
                target_pose = self._sample_pose(goal_pose)
                nearest = self._nearest_node(nodes, target_pose)
                new_last, reached = self._extend_toward(
                    nodes=nodes,
                    source=nearest,
                    target_pose=target_pose,
                    collision_fn=collision_fn,
                    draw_color=(0.85, 0.2, 0.2, 0.45),
                    use_draw=use_draw,
                )
                if not reached:
                    continue
                if self._pose_distance(new_last.pose, goal_pose) <= max(self.position_res, self.rotation_res):
                    goal_last, goal_ok = self._extend_toward(
                        nodes=nodes,
                        source=new_last,
                        target_pose=goal_pose,
                        collision_fn=collision_fn,
                        draw_color=(0.1, 0.7, 0.1, 0.85),
                        use_draw=use_draw,
                    )
                    if goal_ok:
                        path_nodes = goal_last.retrace()
                        if debug_tree_out is not None:
                            debug_tree_out.clear()
                            debug_tree_out["success"] = True
                            debug_tree_out["iterations"] = iteration + 1
                            debug_tree_out["tree1"] = self._export_tree(nodes)
                            debug_tree_out["tree2"] = {"points": [], "edges": []}
                            debug_tree_out["start_pose"] = [float(v) for v in start_pose[0]]
                            debug_tree_out["goal_pose"] = [float(v) for v in goal_pose[0]]
                        return [node.pose for node in path_nodes]
            best_tree = nodes
            logger.info(f"Attempt {attempt + 1}/{max_attempts}: no path found.")

        if debug_tree_out is not None:
            debug_tree_out.clear()
            debug_tree_out["success"] = False
            debug_tree_out["iterations"] = total_iterations
            debug_tree_out["tree1"] = self._export_tree(best_tree)
            debug_tree_out["tree2"] = {"points": [], "edges": []}
            debug_tree_out["start_pose"] = [float(v) for v in start_pose[0]]
            debug_tree_out["goal_pose"] = [float(v) for v in goal_pose[0]]
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


def run_visualization_loop(bar_body: int, path: List[PoseLike], cid: int) -> None:
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
    parser.set_defaults(floating_collision=True)
    args = parser.parse_args()

    if not os.path.isfile(HUSKY_DUAL_URDF_PATH):
        raise FileNotFoundError(f"URDF not found: {HUSKY_DUAL_URDF_PATH}")

    use_gui = not args.no_gui
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

    grasp_targets = load_grasp_targets(args.grasp_json)
    if len(grasp_targets) < 1:
        raise ValueError(f"Expected at least one grasp target in {args.grasp_json}")
    world_from_bar_l, world_from_tool0_left = grasp_targets[0]
    grasp_bar_from_left = pp.multiply(pp.invert(world_from_bar_l), world_from_tool0_left)
    grasp_bar_from_right: Optional[PoseLike] = None
    if len(grasp_targets) >= 2:
        world_from_bar_r, world_from_tool0_right = grasp_targets[1]
        grasp_bar_from_right = pp.multiply(pp.invert(world_from_bar_r), world_from_tool0_right)

    start_joint_values = load_robot_cell_state(args.start_state)
    end_joint_values = load_robot_cell_state(args.end_state)

    start_pose_fk = compute_bar_pose_from_state(robot, arm_joints, tool_link_left, start_joint_values, grasp_bar_from_left)
    end_pose = compute_bar_pose_from_state(robot, arm_joints, tool_link_left, end_joint_values, grasp_bar_from_left)
    if grasp_bar_from_right is not None:
        start_pose_right = compute_bar_pose_from_state(
            robot, arm_joints, tool_link_right, start_joint_values, grasp_bar_from_right
        )
        end_pose_right = compute_bar_pose_from_state(
            robot, arm_joints, tool_link_right, end_joint_values, grasp_bar_from_right
        )
        if not poses_close(start_pose_fk, start_pose_right):
            logger.warning("Start bar pose from left/right grasps does not match exactly; Stage 1 uses the left-arm result.")
            logger.warning(f"  left:  pos={np.round(start_pose_fk[0], 4)}, quat={np.round(start_pose_fk[1], 4)}")
            logger.warning(f"  right: pos={np.round(start_pose_right[0], 4)}, quat={np.round(start_pose_right[1], 4)}")
        if not poses_close(end_pose, end_pose_right):
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

    logger.info("Running minimal Stage 1 RRT.")
    logger.info(f"  start pose: {np.round(start_pose[0], 4)}")
    logger.info(f"  goal pose:  {np.round(end_pose[0], 4)}")
    logger.info(f"  floating collision: {'on' if args.floating_collision else 'off'}")
    logger.info(f"  floating collision obstacles: {len(collision_obstacles)} bodies")

    planner = FloatingBarRRT(
        robot=robot,
        bar_body=bar_body,
        obstacle_bodies=collision_obstacles,
        dist_metric=args.dist_metric,
        goal_sample_prob=args.goal_bias,
        random_seed=args.random_seed,
    )
    debug_tree_out: Dict = {}
    path = planner.plan(
        start_pose=start_pose,
        goal_pose=end_pose,
        max_time=args.max_time,
        max_iterations=args.max_iterations,
        max_attempts=args.max_attempts,
        enable_collision=args.floating_collision,
        use_draw=use_gui,
        debug_tree_out=debug_tree_out,
    )

    if path is not None:
        logger.info(f"Found Stage 1 pose path with {len(path)} waypoints.")
        pp.set_pose(bar_body, path[-1])
    else:
        logger.warning("No Stage 1 pose path found.")

    if use_gui:
        run_visualization_loop(bar_body, path, cid)

    pp.disconnect()


if __name__ == "__main__":
    main()
