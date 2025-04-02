import math
import os
import random
import sys
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import casadi as ca
import numpy as np
import pybullet as p
import pybullet_planning as pp
import torch

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from model.scene_parse import SceneParser
from motion_planner.svsdf import SDF
from multi_tangent.collision import create_collision_bodies
from robot.robot_setup import RobotSetup
from utils.collision import init_pb
from utils.params import URDF_PATH


@dataclass
class Node:
    """RRT tree node"""

    state: np.ndarray  # Joint angle state
    parent: Optional["Node"] = None  # Parent node
    cost: float = 0.0  # Cost from root to current node


class RRTree:
    """RRT tree data structure"""

    def __init__(self, root_state: np.ndarray):
        self.root = Node(root_state)
        self.nodes = [self.root]

    def add_node(self, state: np.ndarray, parent: Node, cost: float = 0.0) -> Node:
        """Add new node to tree"""
        node = Node(state, parent, cost)
        self.nodes.append(node)
        return node

    def get_nearest_node(self, state: np.ndarray) -> Tuple[Node, float]:
        """Find nearest node to given state"""
        min_dist = float("inf")
        nearest = None

        for node in self.nodes:
            dist = np.linalg.norm(node.state - state)
            if dist < min_dist:
                min_dist = dist
                nearest = node

        return nearest, min_dist

    def get_path_to_root(self, node: Node) -> List[np.ndarray]:
        """Get path from node to root"""
        path = []
        current = node
        while current is not None:
            path.append(current.state)
            current = current.parent
        return list(reversed(path))


class TrajectoryIPGARSolver:
    def __init__(self, urdf_path: str, robot_setup: RobotSetup, extend_method_name: str = "default", tensor_args: Optional[Dict] = None, logger_level: str = "error") -> None:
        """Initialize IPGAR trajectory planner with custom BIRRT as base planner"""
        self.urdf_path = urdf_path
        self.robot_setup = robot_setup
        self.tensor_args = tensor_args
        self.logger_level = logger_level

        # BIRRT parameters
        self.step_size = 0.05  # Extension step size
        self.goal_bias = 0.20  # Goal bias probability
        self.max_iterations = 10000  # Maximum iterations
        self.goal_threshold = 0.05  # Goal threshold
        self.interpolation_steps = 5  # Path interpolation steps

        # SDF computation setup
        self.x_sym = ca.MX.sym("x", 3)
        self.q_sym = ca.MX.sym("q", 6)
        self.p_sym = ca.MX.sym("p", 3)
        sdf = SDF(self.urdf_path, self.robot_setup, self.q_sym)
        sdf_sym = sdf(self.p_sym, self.q_sym, self.x_sym)
        sdf_grad = ca.gradient(sdf_sym, self.q_sym)
        self.sdf = ca.Function("sdf", [self.p_sym, self.q_sym, self.x_sym], [sdf_sym, sdf_grad])

        self.sdf_threshold = 0.01

        # Select extension function based on name
        if extend_method_name == "default":
            self.extend_fn = self._extend_tree_default
        else:
            raise ValueError(f"Unknown extension method: {extend_method_name}")

    def _extend_tree_default(self, tree: RRTree, target_state: np.ndarray, collision_fn: Callable[[np.ndarray], bool]) -> Optional[Node]:
        """Default RRT tree extension function"""
        # Sample random state or bias towards goal
        if random.random() < self.goal_bias:
            sample = target_state
        else:
            sample = np.random.uniform(-np.pi, np.pi, 6)

        # Find nearest node
        nearest, _ = tree.get_nearest_node(sample)

        # Calculate extension direction
        direction = sample - nearest.state
        distance = np.linalg.norm(direction)

        if distance > self.step_size:
            direction = direction / distance * self.step_size
            new_state = nearest.state + direction
        else:
            new_state = sample

        # Check if new state is valid
        if not collision_fn(new_state):
            return tree.add_node(new_state, nearest)
        return None

    def _extend_tree(self, tree: RRTree, target_state: np.ndarray, collision_fn: Callable[[np.ndarray], bool], active_obstacles: List[Dict], original_path_segment: np.ndarray, pose_2d: np.ndarray) -> Optional[Node]:
        """RRT tree extension with obstacle repulsion"""
        # Sample random state or bias towards goal
        if random.random() < self.goal_bias:
            q_rand = target_state
        else:
            q_rand = np.random.uniform(-np.pi, np.pi, 6)

        # Find nearest node
        q_near_node, _ = tree.get_nearest_node(q_rand)
        q_near = q_near_node.state

        # 计算所有障碍物的合力
        v_repulsion_total = np.zeros(6)
        min_sdf_val = float("inf")
        rep_decay_rate = 10.0

        if active_obstacles:
            for obs_sphere in active_obstacles:
                obs_pos = np.array(obs_sphere["position"])
                current_sdf_val_ca, current_sdf_grad_ca = self.sdf(pose_2d, q_near, obs_pos)
                current_sdf_val = current_sdf_val_ca.toarray().item()
                current_grad = current_sdf_grad_ca.toarray().flatten()

                # 更新最小SDF值（用于自适应步长）
                if current_sdf_val < min_sdf_val:
                    min_sdf_val = current_sdf_val

                if np.linalg.norm(current_grad) > 1e-6:
                    # 归一化梯度
                    grad_norm = current_grad / np.linalg.norm(current_grad)

                    # 根据SDF值计算排斥权重
                    w_rep = np.exp(-rep_decay_rate * max(0, current_sdf_val))

                    # 累加排斥力
                    v_repulsion_total += w_rep * grad_norm

            # 如果有非零排斥力，归一化合力
            if np.linalg.norm(v_repulsion_total) > 1e-6:
                v_repulsion_total = v_repulsion_total / np.linalg.norm(v_repulsion_total)

            if min_sdf_val > self.sdf_threshold:
                min_sdf_val = self.sdf_threshold
        else:
            min_sdf_val = self.sdf_threshold

        # Calculate standard exploration direction
        v_standard_raw = q_rand - q_near
        dist_standard = np.linalg.norm(v_standard_raw)
        if dist_standard > 1e-6:
            v_standard = v_standard_raw / dist_standard
        else:
            v_standard = np.zeros(6)

        w_std = 1.0
        w_rep = 0.2

        # Combine final extension direction
        v_final_raw = w_std * v_standard + (w_rep * v_repulsion_total)
        norm_final = np.linalg.norm(v_final_raw)

        v_final = v_final_raw / norm_final

        # Generate new configuration
        q_new = q_near + v_final * self.step_size
        # q_new = np.clip(q_new, -np.pi, np.pi)

        # Check state validity
        if not collision_fn(q_new):
            # for t in np.linspace(0, 1, 5)[1:]:
            #     intermediate_state = q_near + t * (q_new - q_near)
            #     if collision_fn(intermediate_state):
            #         return None
            return tree.add_node(q_new, q_near_node)

        return None

    def _create_collision_fn(self, obstacle_bodies: List[int]) -> Callable[[np.ndarray], bool]:
        """Create PyBullet-based collision function"""
        robot_body = self.robot_setup.robot
        arm_joints = self.robot_setup.arm_joints
        attachments = [self.robot_setup.ee_attachment] + self.robot_setup.attachments
        disabled_collisions = self.robot_setup.disabled_collisions
        tool_link = self.robot_setup.tool_link
        wrist_link = pp.link_from_name(robot_body, "ur_arm_wrist_3_link")

        extra_disabled_collisions = []
        if self.robot_setup.ee_attachment is not None:
            extra_disabled_collisions.extend(
                [
                    ((robot_body, wrist_link), (self.robot_setup.ee_attachment.child, pp.BASE_LINK)),
                ]
            )

        return pp.get_collision_fn(
            robot_body,
            arm_joints,
            obstacles=obstacle_bodies,
            attachments=attachments,
            self_collisions=True,
            disabled_collisions=disabled_collisions,
            extra_disabled_collisions=extra_disabled_collisions,
            max_distance=0.0,
        )

    def _check_path_collision(self, path: np.ndarray, obstacle_bodies: List[int]) -> List[Tuple[int, int]]:
        """Check path for collisions and return collision intervals"""
        collision_fn = self._create_collision_fn(obstacle_bodies)
        collision_intervals = []
        start_idx = None

        for i, conf in enumerate(path):
            in_collision = collision_fn(conf)

            if in_collision and start_idx is None:
                start_idx = max(0, i - 1)
            elif not in_collision and start_idx is not None:
                collision_intervals.append((start_idx, i))
                start_idx = None

        if start_idx is not None:
            collision_intervals.append((start_idx, len(path) - 1))

        return collision_intervals

    def _compute_path_obstacle_sdfs(self, pose_2d: np.ndarray, path: np.ndarray, obstacles: List[Dict]) -> Dict[int, float]:
        """Compute minimum SDF values for obstacles along path"""
        min_sdfs = {}
        for obs_idx, obstacle_dict in enumerate(obstacles):
            min_sdf = float("inf")
            obs_pos = np.array(obstacle_dict["position"])
            for point in path:
                sdf_val_ca, _ = self.sdf(pose_2d, point, obs_pos)
                min_sdf = min(min_sdf, sdf_val_ca.toarray().item())
            min_sdfs[obs_idx] = min_sdf
        return min_sdfs

    def _interpolate_path(self, path: List[np.ndarray]) -> np.ndarray:
        """Interpolate path points"""
        if len(path) < 2:
            return np.array(path)

        interpolated = []
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            for j in range(self.interpolation_steps):
                t = j / self.interpolation_steps
                state = start + t * (end - start)
                interpolated.append(state)

        return np.array(interpolated)

    def birrt_plan(
        self, q_init: np.ndarray, q_target: np.ndarray, max_time: int, collision_fn: Callable[[np.ndarray], bool], extend_fn: Callable[[RRTree, np.ndarray, Callable[[np.ndarray], bool]], Optional[Node]], interpolate: bool = True
    ) -> Dict:
        """Plan path using BIRRT algorithm"""
        forward_tree = RRTree(q_init)
        backward_tree = RRTree(q_target)

        start_time = time.time()
        for _ in range(self.max_iterations):
            if time.time() - start_time > max_time:
                return {"success": False, "path": None}

            # Extend forward tree
            new_node = extend_fn(forward_tree, q_target, collision_fn)

            if new_node is not None:
                nearest_backward, dist = backward_tree.get_nearest_node(new_node.state)
                if dist < self.goal_threshold:
                    forward_path = forward_tree.get_path_to_root(new_node)
                    backward_path = backward_tree.get_path_to_root(nearest_backward)
                    path = forward_path + list(reversed(backward_path[:-1]))

                    if interpolate:
                        path = self._interpolate_path(path)
                    else:
                        path = np.array(path)

                    for state in path:
                        if collision_fn(state):
                            print("Warning: Initial path state has collision.")
                            continue

                    return {"success": True, "path": path}

            # Extend backward tree
            new_node = extend_fn(backward_tree, q_init, collision_fn)

            if new_node is not None:
                nearest_forward, dist = forward_tree.get_nearest_node(new_node.state)
                if dist < self.goal_threshold:
                    forward_path = forward_tree.get_path_to_root(nearest_forward)
                    backward_path = backward_tree.get_path_to_root(new_node)
                    path = forward_path + list(reversed(backward_path[:-1]))

                    if interpolate:
                        path = self._interpolate_path(path)
                    else:
                        path = np.array(path)

                    for state in path:
                        if collision_fn(state):
                            print("Warning: Initial path state has collision.")
                            continue

                    return {"success": True, "path": path}

        return {"success": False, "path": None}

    def plan(
        self,
        q_init: np.ndarray,
        q_target: np.ndarray,
        pose_2d: np.ndarray,
        max_time: int,
        max_attempts: int,
        element_bodies: List[int],
        grasped_element: Optional[int] = None,
        grasped_attachment: Optional[object] = None,
    ) -> Dict:
        """Main planning function using IPGAR algorithm"""
        # Initialize obstacles
        spheres = [SceneParser.approximate_cylinder(element_body, count=30) for element_body in element_bodies]
        active_obstacles = []
        # 使用字典存储remaining_obstacles，以name为键
        remaining_obstacles_dict = {}
        for sphere_list in spheres:
            for sphere in sphere_list:
                name = sphere.get("name", f"unnamed_{len(remaining_obstacles_dict)}")
                remaining_obstacles_dict[name] = sphere
        current_path = None

        # Initial path planning
        print("Step 1: Planning initial path in free space using BiRRT...")
        collision_fn = self._create_collision_fn([])  # Empty list for free space
        initial_solution = self.birrt_plan(q_init, q_target, max_time, collision_fn, self._extend_tree_default, interpolate=False)
        if not initial_solution["success"]:
            print("ERROR: Initial path planning failed even in free space.")
            return {"success": False, "path": None}
        print("Initial path found.")
        initial_path = initial_solution["path"]

        # Main planning loop
        print("Step 2: Starting Incremental Obstacle Addition and Repair Loop...")
        iteration = 0
        current_path = initial_path
        active_obstacles_body = []
        while remaining_obstacles_dict:
            iteration += 1
            print(f"\n--- Iteration {iteration}: Processing obstacles ---")
            print(f"Remaining obstacles: {len(remaining_obstacles_dict)}")

            with pp.LockRenderer():
                for body in active_obstacles_body:
                    pp.set_color(body, [0, 0, 1, 1])

            # Step 1: Calculate SVSDF for all remaining obstacles
            print("Calculating SVSDF for remaining obstacles...")
            remaining_obstacles_list = list(remaining_obstacles_dict.values())
            svsdf_values = self._compute_path_obstacle_sdfs(pose_2d, current_path, remaining_obstacles_list)

            # 将SVSDF值从索引映射到name
            svsdf_by_name = {}
            for i, obs in enumerate(remaining_obstacles_list):
                name = obs.get("name", f"unnamed_{i}")
                svsdf_by_name[name] = svsdf_values.get(i, 0)

            # Group obstacles by element
            element_obstacles = {}
            for name, obs in remaining_obstacles_dict.items():
                if "_sphere_" in name:
                    element_id = name.split("_sphere_")[0]
                    if element_id not in element_obstacles:
                        element_obstacles[element_id] = []
                    element_obstacles[element_id].append((name, obs, svsdf_by_name.get(name, 0)))

            # Step 2: Find elements where all obstacles have SVSDF > threshold
            elements_all_above_threshold = []
            for element_id, obs_list in element_obstacles.items():
                if all(sdf_val > self.sdf_threshold for _, _, sdf_val in obs_list):
                    elements_all_above_threshold.append(element_id)

            # Process elements with all obstacles above threshold
            if elements_all_above_threshold:
                print(f"Found {len(elements_all_above_threshold)} elements with all obstacles above threshold.")

                # 处理所有满足条件的元素的障碍物
                obs_to_activate = []
                for element_id in elements_all_above_threshold:
                    print(f"Preparing obstacles for element {element_id}")
                    for name, obs, sdf_val in element_obstacles[element_id]:
                        obs_to_activate.append((name, obs))

                # 添加所有障碍物
                with pp.LockRenderer():
                    for name, obs in obs_to_activate:
                        # 从字典中移除
                        del remaining_obstacles_dict[name]
                        active_obstacles.append(obs)
                        # 创建PyBullet物体
                        obs_id = pp.create_sphere(obs["radius"])
                        pp.set_point(obs_id, obs["position"])
                        pp.set_color(obs_id, [0, 0, 1, 1])
                        active_obstacles_body.append(obs_id)
                        print(f"  已激活障碍物: {name} with SVSDF > {self.sdf_threshold}")

                # Update collision function with new obstacles
                collision_fn = self._create_collision_fn(active_obstacles_body)

                # Check path for collisions
                collision_intervals = self._check_path_collision(current_path, active_obstacles_body)
                if len(collision_intervals) > 0:
                    print(f"Path has {len(collision_intervals)} collisions after adding elements, repairing...")

                    # Repair colliding segments
                    repair_result = self._repair_path(current_path, collision_intervals, collision_fn, active_obstacles_body, active_obstacles)
                    if repair_result["success"]:
                        current_path = repair_result["path"]
                        print("Path successfully repaired after adding elements.")
                    else:
                        print("ERROR: Path repair failed after adding elements.")
                        with pp.LockRenderer():
                            for body_id in active_obstacles_body:
                                pp.remove_body(body_id)
                        return {"success": False, "path": None}

            # Step 3: For remaining elements, calculate average SVSDF
            if remaining_obstacles_dict:
                print("Calculating average SVSDF for remaining elements...")
                element_averages = {}
                for element_id, obs_list in element_obstacles.items():
                    if element_id not in elements_all_above_threshold:
                        avg_sdf = np.mean([sdf_val for _, _, sdf_val in obs_list])
                        element_averages[element_id] = avg_sdf

                if element_averages:
                    # Step 4: Select element with highest average SVSDF
                    max_avg_element = max(element_averages.items(), key=lambda x: x[1])[0]
                    print(f"Selected element {max_avg_element} with average SVSDF {element_averages[max_avg_element]:.4f}")

                    # sorted_obstacles = sorted(element_obstacles[max_avg_element], key=lambda x: x[0])
                    # sorted_obstacles = sorted(element_obstacles[max_avg_element], key=lambda x: x[2], reverse=True)
                    sorted_obstacles = sorted(element_obstacles[max_avg_element], key=lambda x: x[2])

                    # Process obstacles one by one
                    for name, obs, sdf_val in sorted_obstacles:
                        with pp.LockRenderer():
                            for body in active_obstacles_body:
                                pp.set_color(body, [0, 0, 1, 1])

                        # 检查障碍物是否仍在字典中
                        if name not in remaining_obstacles_dict:
                            continue

                        # 添加到活动障碍物
                        with pp.LockRenderer():
                            # 从字典中移除
                            del remaining_obstacles_dict[name]
                            active_obstacles.append(obs)
                            # 创建PyBullet物体
                            obs_id = pp.create_sphere(obs["radius"])
                            pp.set_point(obs_id, obs["position"])
                            pp.set_color(obs_id, [0, 1, 0, 1])  # Green for normal addition
                            active_obstacles_body.append(obs_id)

                        print(f"  Activated obstacle: {name} with SVSDF {sdf_val:.4f}")

                        # Update collision function and check path
                        collision_fn = self._create_collision_fn(active_obstacles_body)
                        collision_intervals = self._check_path_collision(current_path, active_obstacles_body)

                        if len(collision_intervals) > 0:
                            print(f"  Path has {len(collision_intervals)} collisions after adding obstacle, repairing...")

                            # Repair path
                            repair_result = self._repair_path(current_path, collision_intervals, collision_fn, active_obstacles_body, active_obstacles)
                            if repair_result["success"]:
                                current_path = repair_result["path"]
                                print("  Path successfully repaired after adding obstacle.")
                            else:
                                print("  Repair failed, removing the last added obstacle.")
                                with pp.LockRenderer():
                                    pp.remove_body(active_obstacles_body.pop())
                                active_obstacles.pop()
                                # 重新添加到字典
                                remaining_obstacles_dict[name] = obs
                                # Update collision function
                                collision_fn = self._create_collision_fn(active_obstacles_body)

            print(f"Total active obstacles: {len(active_obstacles)} (PyBullet bodies: {len(active_obstacles_body)})")

        # Final path check and output
        final_collision_check = self._check_path_collision(current_path, active_obstacles_body)
        if len(final_collision_check) > 0:
            print("ERROR: Final path still has collisions!")
            with pp.LockRenderer():
                for body_id in active_obstacles_body:
                    pp.remove_body(body_id)
            return {"success": False, "path": None}

        print("\nIPGAR Planning Successful!")
        final_path_interpolated = self._interpolate_path(list(current_path))

        with pp.LockRenderer():
            for body_id in active_obstacles_body:
                pp.remove_body(body_id)

        return {"success": True, "path": final_path_interpolated}

    def _repair_path(self, path, collision_intervals, collision_fn, active_obstacles_body, active_obstacles):
        """Repair path segments with collisions"""
        print(f"Repairing {len(collision_intervals)} colliding segments...")
        repair_successful = True
        new_path_segments = []
        last_safe_idx = 0
        current_interval_idx = 0

        while current_interval_idx < len(collision_intervals):
            # 获取当前区间
            current_start, current_end = collision_intervals[current_interval_idx]
            expanded_start = current_start
            expanded_end = current_end
            repair_attempts = 0
            max_repair_attempts = 5

            while repair_attempts < max_repair_attempts:
                print(f"  Repair attempt {repair_attempts + 1}/{max_repair_attempts} " f"for segment {expanded_start} to {expanded_end}...")

                # 添加安全路径段
                if expanded_start > last_safe_idx:
                    new_path_segments.append(path[last_safe_idx : expanded_start + 1])

                # 尝试修复扩展区间
                q_start_local = path[expanded_start]
                q_end_local = path[expanded_end]

                # repair_result = self.birrt_plan(q_start_local, q_end_local, 20.0, collision_fn, self._extend_tree_default, interpolate=True)

                wrapped_extend_fn = partial(self._extend_tree, active_obstacles=active_obstacles, original_path_segment=path[expanded_start:expanded_end], pose_2d=pose_2d)
                repair_result = self.birrt_plan(q_start_local, q_end_local, 20.0, collision_fn, wrapped_extend_fn, interpolate=True)

                if repair_result["success"]:
                    print(f"  Segment repair successful after {repair_attempts + 1} attempts.")
                    repaired_segment = repair_result["path"]
                    new_path_segments.append(repaired_segment)
                    last_safe_idx = expanded_end
                    current_interval_idx += 1
                    break
                else:
                    print(f"  Repair attempt {repair_attempts + 1} failed, expanding repair region...")
                    repair_attempts += 1

                    # 如果还有下一个区间，扩展修复区间
                    if current_interval_idx + 1 < len(collision_intervals):
                        next_start, next_end = collision_intervals[current_interval_idx + 1]
                        expanded_start = min(expanded_start, next_start)
                        expanded_end = max(expanded_end, next_end)
                        current_interval_idx += 1
                    else:
                        # 如果没有下一个区间，增加当前区间的范围
                        expansion_size = 10
                        expanded_start = max(0, expanded_start - expansion_size)
                        expanded_end = min(len(path) - 1, expanded_end + expansion_size)

            if repair_attempts >= max_repair_attempts:
                print(f"  Segment repair FAILED after {max_repair_attempts} attempts with expanded regions.")
                repair_successful = False
                return {"success": False, "path": None}

        # 添加最后的安全路径段
        if last_safe_idx < len(path) - 1:
            new_path_segments.append(path[last_safe_idx:])

        if repair_successful:
            final_path = np.concatenate(new_path_segments, axis=0)
            final_path = self._shortcut_path(final_path, collision_fn)
            return {"success": True, "path": final_path}
        else:
            return {"success": False, "path": None}

    def _shortcut_path(self, path, collision_fn, iterations=100):
        """
        使用路径缩短（Shortcut Smoothing）优化路径。

        Args:
            path (np.ndarray): 原始路径，包含一系列配置点
            collision_fn (Callable): 碰撞检测函数
            iterations (int): 尝试缩短的次数

        Returns:
            np.ndarray: 优化后的路径，保证包含原始路径的起始点和终止点
        """
        if not isinstance(path, np.ndarray) or len(path) < 3:
            return path  # 路径太短，无法缩短

        print("Starting Path Shortcutting...")
        optimized_path = deepcopy(path)  # 操作副本以防万一
        n = len(optimized_path)
        
        # 如果路径已经只有3个点（起点、终点和一个中间点），直接返回
        if n <= 3:
            print("Path already minimal (length <= 3), skipping shortcutting")
            return optimized_path
        
        # 定义线段碰撞检测函数
        def is_segment_collision(q1, q2):
            """检查两点之间的直线段是否有碰撞"""
            # 在两点之间采样多个中间点进行检查
            samples = 100
            for t in np.linspace(0, 1, samples)[1:-1]:  # 排除端点
                q_interp = q1 + t * (q2 - q1)
                if collision_fn(q_interp):
                    return True
            return False

        # 保存原始的起始点和终止点，确保它们不会被修改
        start_point = path[0].copy()
        end_point = path[-1].copy()

        shortcut_count = 0
        for k in range(iterations):
            # 如果路径已经只有3个点，停止优化
            if n <= 3:
                break
                
            # 选择区间：从第1个点到倒数第2个点（保留起点和终点）
            # 确保i和j之间有足够的空间
            if n <= 4:  # 如果路径长度<=4，无法进行有效的缩短
                break
                
            i = np.random.randint(1, n - 3)  # 从1到n-4中选择，确保后面有至少3个点
            j = np.random.randint(i + 2, n - 1)  # 从i+2到n-2中选择

            q_i = optimized_path[i]
            q_j = optimized_path[j]

            # 检查 q_i 和 q_j 之间是否存在直接无碰撞路径
            if not is_segment_collision(q_i, q_j):
                # 如果无碰撞，移除中间的节点
                shortcut_count += 1
                print(f"  Shortcut {shortcut_count}: between index {i} and {j}, removing {j-i-1} configurations")
                
                # 更新路径：保留 i 之前的部分 + 直连线段 + j 之后的部分
                optimized_path = np.vstack([optimized_path[: i + 1], optimized_path[j:]])
                n = len(optimized_path)  # 更新路径长度
                
                # 确保起点和终点不变
                optimized_path[0] = start_point
                optimized_path[-1] = end_point

        # 最终检查，确保起点和终点正确
        if len(optimized_path) > 0:
            optimized_path[0] = start_point
        if len(optimized_path) > 1:
            optimized_path[-1] = end_point
            
        print(f"Path Shortcutting finished. Original length: {len(path)}, Optimized length: {len(optimized_path)}, Shortcuts found: {shortcut_count}")
        return optimized_path


if __name__ == "__main__":

    init_pb()

    scene_file = os.path.join(HERE, "model", "scenes", "cuboid_1", "task_1.yml")
    scene_parser = SceneParser(scene_file)
    scene_parser.load_scene()
    line_pts, radius_per_edge = scene_parser.get_element_info()
    bodies = create_collision_bodies(line_pts, radius_per_edge, viewer=True)

    for body in bodies:
        pp.set_color(body, [1, 0, 0, 0.25])

    start_q = np.array(scene_parser.get_robot_start_pose())
    target_q = np.array(scene_parser.get_robot_target_pose())
    pose_2d = scene_parser.get_robot_pose_2d(output_type="array")

    # start_q[0] = 20.0 / 180.0 * np.pi

    rb = RobotSetup("rb")  # 'rb' is created here, only available in main block
    rb.set_joint_positions(rb.arm_joints, start_q)
    rb.set_base_pose_2d(pose_2d[0], pose_2d[1], pose_2d[2])

    # 执行规划并获取路径
    # path = rb.plan_manipulator_path(start_q, target_q, [], bodies, max_time=600, max_iterations=10000)

    # Pass rb (RobotSetup instance) to the solver
    solver = TrajectoryIPGARSolver(URDF_PATH, rb)
    plan_result = solver.plan(start_q, target_q, pose_2d, max_time=600, max_attempts=10000, element_bodies=bodies)

    if plan_result["success"]:
        path = plan_result["path"]
    else:
        path = None

    if path is not None:
        # -------------------- 下面是使用pybullet进行可视化的代码 --------------------#
        slider = p.addUserDebugParameter("replay", 0, 1, 0)

        for body in bodies:
            pp.set_color(body, [1, 0, 0, 1])

        while True:
            slider_value = p.readUserDebugParameter(slider)
            time_idx = int(slider_value * (path.shape[0] - 1))
            joint_val = path[time_idx]
            rb.set_joint_positions(rb.arm_joints, joint_val)  # Use rb here for visualization
            time.sleep(1.0 / 60)
