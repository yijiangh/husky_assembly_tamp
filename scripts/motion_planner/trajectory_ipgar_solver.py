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
from scipy.interpolate import splev, splprep

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from model.scene_parse import SceneParser
from motion_planner.svsdf import SDF
from multi_tangent.collision import create_collision_bodies
from robot.robot_setup import RobotSetup
from utils.collision import element_collision_info, init_pb
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
    def __init__(self, urdf_path: str, robot_setup: RobotSetup, grasp_offset: List[float], extend_method_name: str = "default", tensor_args: Optional[Dict] = None, logger_level: str = "error") -> None:
        """Initialize IPGAR trajectory planner with custom BIRRT as base planner"""
        self.urdf_path = urdf_path
        self.robot_setup = robot_setup
        self.grasp_offset = grasp_offset
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
        sdf = SDF(self.urdf_path, self.robot_setup, self.q_sym, self.p_sym, self.x_sym, self.grasp_offset, element_collision_info)
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
            for t in np.linspace(0, 1, 5)[1:]:
                intermediate_state = nearest.state + t * (new_state - nearest.state)
                if collision_fn(intermediate_state):
                    return None
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
        w_rep = 0.1

        # Combine final extension direction
        v_final_raw = w_std * v_standard + w_rep * v_repulsion_total
        norm_final = np.linalg.norm(v_final_raw)

        v_final = v_final_raw / norm_final

        # Generate new configuration
        q_new = q_near + v_final * self.step_size
        # q_new = np.clip(q_new, -np.pi, np.pi)

        # Check state validity
        if not collision_fn(q_new):
            for t in np.linspace(0, 1, 5)[1:]:
                intermediate_state = q_near + t * (q_new - q_near)
                if collision_fn(intermediate_state):
                    return None
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

    def _check_path_collision(self, path: np.ndarray, collision_fn: Callable[[np.ndarray], bool]) -> List[Tuple[int, int]]:
        """Check path for collisions and return collision intervals"""
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
        print("\n========== IPGAR TRAJECTORY PLANNING ==========")

        # 定义障碍物密度序列，从低到高
        density_levels = [15, 50]
        final_path = None

        # 外层循环：逐步增加障碍物密度
        for iteration, density in enumerate(density_levels):
            print(f"\n========== DENSITY ITERATION {iteration+1}/{len(density_levels)} (count={density}) ==========")

            # 初始化障碍物
            spheres = [SceneParser.approximate_cylinder(element_body, count=density) for element_body in element_bodies]
            active_obstacles = []
            # 使用字典存储remaining_obstacles，以name为键
            remaining_obstacles_dict = {}
            for sphere_list in spheres:
                for sphere in sphere_list:
                    name = sphere.get("name", f"unnamed_{len(remaining_obstacles_dict)}")
                    remaining_obstacles_dict[name] = sphere

            # 第一次迭代时初始化路径，后续迭代使用上一次的结果
            if final_path is None:
                # 初始路径规划
                print(f"IPGAR: Step 1 - Planning initial path in free space...")
                # collision_fn = self._create_collision_fn([])  # Empty list for free space
                # initial_solution = self.birrt_plan(q_init, q_target, max_time, collision_fn, self._extend_tree_default, interpolate=False)
                # if not initial_solution["success"]:
                #     print("IPGAR: ERROR - Initial path planning failed in free space")
                #     return {"success": False, "path": None}
                # print("IPGAR: Initial path successfully found")
                # current_path = initial_solution["path"]
                
                initial_solution = self.robot_setup.plan_manipulator_path(q_init, q_target, self.robot_setup.attachments, [], max_time=10, max_iterations=10000)
                if initial_solution is None:
                    print("IPGAR: ERROR - Initial path planning failed in free space")
                    return {"success": False, "path": None}
                print("IPGAR: Initial path successfully found")
                current_path = initial_solution
                  
                # 应用路径优化流程，确保每个步骤都保留原始起点和终点
                collision_fn = self._create_collision_fn([])
                temp_path = self._shortcut_path(current_path, collision_fn, iterations=30)
                temp_path = self._resample_path_fixed_length(temp_path, fixed_length=max(100, len(temp_path)))
                # final_path = self._smooth_path_bspline(final_path, collision_fn)
                current_path = deepcopy(temp_path)
            else:
                # 使用上一次规划结果作为当前路径
                print(f"IPGAR: Using previous density level path as starting path")
                current_path = final_path

            # 主规划循环
            print("IPGAR: Step 2 - Starting Incremental Obstacle Addition and Repair process...")
            iteration_count = 0
            active_obstacles_body = []
            while remaining_obstacles_dict:
                iteration_count += 1
                print("\n---------- IPGAR Iteration {} ----------".format(iteration_count))
                print(f"IPGAR: Processing obstacles - {len(remaining_obstacles_dict)} remaining")

                with pp.LockRenderer():
                    for body in active_obstacles_body:
                        pp.set_color(body, [0, 0, 1, 1])

                # Step 1: Calculate SVSDF for all remaining obstacles
                print("IPGAR: Calculating SVSDF values for remaining obstacles...")
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
                    print(f"IPGAR: Found {len(elements_all_above_threshold)} elements with all obstacles above threshold")

                    # 处理所有满足条件的元素的障碍物
                    obs_to_activate = []
                    for element_id in elements_all_above_threshold:
                        print(f"IPGAR: Preparing obstacles for element {element_id}")
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
                            print(f"  IPGAR: Activated obstacle {name} - SVSDF > {self.sdf_threshold}")

                    # Update collision function with new obstacles
                    collision_fn = self._create_collision_fn(active_obstacles_body)

                    # Check path for collisions
                    collision_intervals = self._check_path_collision(current_path, collision_fn)
                    if len(collision_intervals) > 0:
                        print(f"IPGAR: Path has {len(collision_intervals)} collisions after adding elements, repairing...")

                        # Repair colliding segments
                        repair_result = self._repair_path(current_path, collision_intervals, collision_fn, active_obstacles_body, active_obstacles, pose_2d)
                        if repair_result["success"]:
                            current_path = repair_result["path"]
                            print("IPGAR: Path successfully repaired after adding elements")
                        else:
                            print("IPGAR: ERROR - Path repair failed after adding elements")
                            with pp.LockRenderer():
                                for body_id in active_obstacles_body:
                                    pp.remove_body(body_id)
                        # 当前密度失败，但可以尝试降低密度
                        break
                    
                if len(remaining_obstacles_dict) == 0:
                    print("IPGAR: All obstacles have been added")
                    break

                # Step 3: For remaining elements, calculate average SVSDF
                if remaining_obstacles_dict:
                    print("IPGAR: Calculating average SVSDF for remaining elements...")
                    element_averages = {}
                    for element_id, obs_list in element_obstacles.items():
                        if element_id not in elements_all_above_threshold:
                            avg_sdf = np.mean([sdf_val for _, _, sdf_val in obs_list])
                            element_averages[element_id] = avg_sdf

                    if element_averages:
                        # Step 4: Select element with highest average SVSDF
                        avg_element = max(element_averages.items(), key=lambda x: x[1])[0]
                        print(f"IPGAR: Selected element {avg_element} with average SVSDF {element_averages[avg_element]:.4f}")

                        # # Step 4: Select element with lowest average SVSDF
                        # avg_element = min(element_averages.items(), key=lambda x: x[1])[0]
                        # print(f"IPGAR: Selected element {avg_element} with average SVSDF {element_averages[avg_element]:.4f}")

                        # sorted_obstacles = sorted(element_obstacles[max_avg_element], key=lambda x: x[2], reverse=True) # SVSDF从大到小排序
                        # sorted_obstacles = sorted(element_obstacles[max_avg_element], key=lambda x: x[2]) # SVSDF从小到大排序
                        # 按照obstacle的编号(sphere_id)排序
                        sorted_obstacles = [sorted(element_obstacles[element_id], key=lambda x: int(x[0].split("_sphere_")[1]) if "_sphere_" in x[0] else 0) for element_id in element_averages.keys()]  # 按sphere_id排序

                        # Process obstacles one by one
                        while len([item for sublist in sorted_obstacles for item in sublist]) != 0:
                            
                            current_active_obs_ids = []
                            for sub_obs_list in sorted_obstacles:
                                
                                find_collision = False
                                while len(sub_obs_list) != 0:
                                    name, obs, sdf_val = sub_obs_list.pop(0)
                                    
                                    del remaining_obstacles_dict[name]
                                    active_obstacles.append(obs)
                                    # 创建PyBullet物体
                                    obs_id = pp.create_sphere(obs["radius"])
                                    pp.set_point(obs_id, obs["position"])
                                    pp.set_color(obs_id, [0, 1, 0, 1])  # Green for normal addition
                                    active_obstacles_body.append(obs_id)
                                    print(f"  IPGAR: Activated obstacle {name} with SVSDF {sdf_val:.4f}")
                                    
                                    if sdf_val > self.sdf_threshold:
                                        pp.set_color(obs_id, [0, 0, 1, 1])
                                        continue
                                    
                                    find_collision = True
                                    break
                                
                                if find_collision:
                                    current_active_obs_ids.append(obs_id)
                                    
                            if len(current_active_obs_ids) == 0:
                                continue
                            
                            # Update collision function and check path
                            with pp.LockRenderer():
                                collision_fn = self._create_collision_fn(active_obstacles_body)
                                collision_intervals = self._check_path_collision(current_path, collision_fn)       
                                    
                            if len(collision_intervals) > 0:
                                print(f"  IPGAR: Path has {len(collision_intervals)} collisions, repairing...")
                                
                                # Repair path
                                repair_result = self._repair_path(current_path, collision_intervals, collision_fn, active_obstacles_body, active_obstacles, pose_2d)
                                if repair_result["success"]:
                                    current_path = repair_result["path"]
                                    print("  IPGAR: Path successfully repaired")
                                    
                                    # Recalculate SVSDF for remaining obstacles in this element after successful repair
                                    temp_sorted_obstacles = []
                                    for sub_obs_list in sorted_obstacles:
                                        temp_svsdf_values = self._compute_path_obstacle_sdfs(pose_2d, current_path, [item[1] for item in sub_obs_list])
                                        temp_sorted_obs = []
                                        for i, item in enumerate(sub_obs_list):
                                            new_item = list(deepcopy(item))
                                            new_item[2] = temp_svsdf_values[i]
                                            print(f"      IPGAR: Updated SVSDF for {item[0]}: {item[2]:.4f} -> {new_item[2]:.4f}")
                                            temp_sorted_obs.append(tuple(new_item))
                                        temp_sorted_obstacles.append(temp_sorted_obs)
                                    sorted_obstacles = deepcopy(temp_sorted_obstacles)
                                    
                                else:
                                    print("  IPGAR: Repair failed")
                            
                            for temp_id in current_active_obs_ids:
                                pp.set_color(temp_id, [0, 0, 1, 1])
                            
                with pp.LockRenderer():
                    # 应用路径优化流程，确保每个步骤都保留原始起点和终点
                    temp_path = self._shortcut_path(current_path, collision_fn, iterations=30)
                    temp_path = self._resample_path_fixed_length(temp_path, fixed_length=max(100, len(temp_path)))
                    # final_path = self._smooth_path_bspline(final_path, collision_fn)
                    current_path = deepcopy(temp_path)

                print(f"IPGAR: Total active obstacles: {len(active_obstacles)} (PyBullet bodies: {len(active_obstacles_body)})")

            # 清理当前迭代的PyBullet物体
            with pp.LockRenderer():
                for body_id in active_obstacles_body:
                    pp.remove_body(body_id)

            # 检查当前密度迭代是否成功
            if not remaining_obstacles_dict:
                print(f"IPGAR: Density level {density} planning successful!")
                final_path = current_path
            else:
                print(f"IPGAR: Density level {density} planning incomplete, using previous result")
                # 如果当前密度失败但已有前一次成功的结果，可以继续尝试下一个密度
                if final_path is not None:
                    continue
                else:
                    # 如果第一次密度就失败且没有前一次结果，则整体规划失败
                    return {"success": False, "path": None}

        # Final path check and output
        # 使用最后一次成功规划的密度级别重新创建障碍物进行最终检查
        final_density = max([d for i, d in enumerate(density_levels) if i < len(density_levels) and (i == len(density_levels) - 1 or final_path is not None)])
        final_spheres = [SceneParser.approximate_cylinder(element_body, count=final_density) for element_body in element_bodies]
        final_obstacle_bodies = []

        with pp.LockRenderer():
            for sphere_list in final_spheres:
                for sphere in sphere_list:
                    obs_id = pp.create_sphere(sphere["radius"])
                    pp.set_point(obs_id, sphere["position"])
                    pp.set_color(obs_id, [1, 0, 0, 0.5])  # 红色半透明用于最终验证
                    final_obstacle_bodies.append(obs_id)

        final_collision_fn = self._create_collision_fn(final_obstacle_bodies)
        final_collision_check = self._check_path_collision(final_path, final_collision_fn)
        if len(final_collision_check) > 0:
            print(f"IPGAR: ERROR - Final path still has {len(final_collision_check)} collisions!")

            # 尝试最后一次修复
            print(f"IPGAR: Attempting final path repair...")
            final_repair_result = self._repair_path(final_path, final_collision_check, final_collision_fn, final_obstacle_bodies, [], pose_2d)

            if not final_repair_result["success"]:
                print("IPGAR: Final repair failed!")
                with pp.LockRenderer():
                    for body_id in final_obstacle_bodies:
                        pp.remove_body(body_id)
                return {"success": False, "path": None}

            final_path = final_repair_result["path"]
            print("IPGAR: Final path successfully repaired")

        print("\n========== IPGAR PLANNING SUCCESSFUL ==========")
        final_path_interpolated = self._interpolate_path(list(final_path))

        with pp.LockRenderer():
            for body_id in final_obstacle_bodies:
                pp.remove_body(body_id)

        return {"success": True, "path": final_path_interpolated}

    def _repair_path(self, path, collision_intervals, collision_fn, active_obstacles_body, active_obstacles, pose_2d):
        """Repair path segments with collisions"""
        print("---------- Path Repair ----------")
        print(f"Repair: Processing {len(collision_intervals)} collision segments...")
        repair_successful = True

        # 保存原始起点和终点
        start_point = path[0].copy()
        end_point = path[-1].copy()

        current_path = path.copy()  # 创建路径副本，以便在修补过程中更新
        
        # 循环直到所有碰撞区间都被处理
        while collision_intervals:
            # 获取当前区间
            current_start, current_end = collision_intervals[0]
            expanded_start = current_start
            expanded_end = current_end
            repair_attempts = 0
            max_repair_attempts = 10

            while repair_attempts < max_repair_attempts:
                print(f"  Repair: Attempt {repair_attempts + 1}/{max_repair_attempts} for segment {expanded_start}-{expanded_end}")

                # 尝试修复扩展区间
                q_start_local = current_path[expanded_start]
                q_end_local = current_path[expanded_end]
                
                max_plan_time = 10.0 + repair_attempts * 10.0
                
                # repair_result = self.birrt_plan(q_start_local, q_end_local, 30.0, collision_fn, self._extend_tree_default, interpolate=False)

                # wrapped_extend_fn = partial(self._extend_tree, active_obstacles=active_obstacles, original_path_segment=path[expanded_start:expanded_end], pose_2d=pose_2d)
                # repair_result = self.birrt_plan(q_start_local, q_end_local, max_plan_time, collision_fn, wrapped_extend_fn, interpolate=False)
                with pp.LockRenderer():
                    repair_result = self.robot_setup.plan_manipulator_path(q_start_local, q_end_local, self.robot_setup.attachments, active_obstacles_body, max_time=max_plan_time, max_iterations=10000)

                # if repair_result["success"]:
                if repair_result is not None:
                    print(f"  Repair: Success after {repair_attempts + 1} attempts")

                    # 更新路径：保留修复区间之前的部分 + 修复后的段 + 修复区间之后的部分
                    # repaired_segment = repair_result["path"]
                    repaired_segment = repair_result

                    # 构建新路径
                    new_path = []

                    # 添加前缀（如果有）
                    if expanded_start > 0:
                        new_path.append(current_path[:expanded_start])

                    # 添加修复段
                    new_path.append(repaired_segment)

                    # 添加后缀（如果有）
                    if expanded_end < len(current_path) - 1:  # 注意这里使用current_path的长度
                        new_path.append(current_path[expanded_end + 1 :])

                    # 合并路径段
                    current_path = np.vstack(new_path)

                    # 确保起点和终点保持不变
                    if len(current_path) > 0:
                        current_path[0] = start_point
                    if len(current_path) > 1:
                        current_path[-1] = end_point

                    # 重新检测碰撞区间
                    with pp.LockRenderer():
                        collision_intervals = self._check_path_collision(current_path, collision_fn)
                    print(f"  Repair: After update, {len(collision_intervals)} collision segments remain")

                    # 如果没有剩余碰撞，完成修复
                    if not collision_intervals:
                        print("  Repair: All collisions resolved")
                        break
                    else:
                        # 重置尝试计数器，开始处理新的第一个区间
                        break

                else:
                    print(f"  Repair: Attempt {repair_attempts + 1} failed, expanding repair region...")
                    repair_attempts += 1

                    # 扩展修补区间
                    expansion_size = len(current_path) // 10
                    expanded_start = max(0, expanded_start - expansion_size)
                    expanded_end = min(len(current_path) - 1, expanded_end + expansion_size)

                    # # 如果还有下一个区间，扩展修复区间
                    # if current_interval_idx + 1 < len(collision_intervals):
                    #     next_start, next_end = collision_intervals[current_interval_idx + 1]
                    #     expanded_start = min(expanded_start, next_start)
                    #     expanded_end = max(expanded_end, next_end)
                    #     current_interval_idx += 1
                    # else:
                    #     # 如果没有下一个区间，增加当前区间的范围
                    #     expansion_size = 10
                    #     expanded_start = max(0, expanded_start - expansion_size)
                    #     expanded_end = min(len(path) - 1, expanded_end + expansion_size)

            # 如果达到最大尝试次数仍未成功，则修补失败
            if repair_attempts >= max_repair_attempts:
                print(f"  Repair: FAILED after {max_repair_attempts} attempts with expanded regions")
                repair_successful = False
                return {"success": False, "path": None}

        if repair_successful:
            print("Repair: All segments successfully repaired")
            print("---------- Path Repair Complete ----------")

            return {"success": True, "path": current_path}
        else:
            print("Repair: Process failed")
            return {"success": False, "path": None}

    def _resample_path_fixed_length(self, path, fixed_length=100):
        """
        将路径重采样为固定数量的点。

        Args:
            path (np.ndarray): 原始路径
            fixed_length (int): 重采样后的路径长度

        Returns:
            np.ndarray: 重采样后的路径
        """
        if len(path) < 2:
            return path
        
        if len(path) >= fixed_length:
            return path

        print(f"Resampling: Resampling path to {fixed_length} points...")

        # 保存原始起点和终点
        start_point = path[0].copy()
        end_point = path[-1].copy()

        # 计算路径长度（各段长度之和）
        total_dist = 0
        dists = []
        for i in range(len(path) - 1):
            d = np.linalg.norm(path[i + 1] - path[i])
            total_dist += d
            dists.append(d)

        # 创建累积距离数组
        cum_dists = np.cumsum([0] + dists)
        cum_dists /= cum_dists[-1]  # 归一化到[0,1]范围

        # 在归一化距离上均匀采样
        alpha = np.linspace(0, 1, fixed_length)
        resampled_path = np.zeros((fixed_length, path.shape[1]))

        for i in range(fixed_length):
            # 找到当前alpha值所在的路径段
            idx = np.searchsorted(cum_dists, alpha[i]) - 1
            idx = max(0, min(idx, len(path) - 2))  # 确保索引有效

            # 计算在该段内的插值因子
            seg_alpha = (alpha[i] - cum_dists[idx]) / (cum_dists[idx + 1] - cum_dists[idx]) if cum_dists[idx] < cum_dists[idx + 1] else 0

            # 线性插值得到新的配置点
            resampled_path[i] = path[idx] + seg_alpha * (path[idx + 1] - path[idx])

        # 确保起点和终点保持不变
        resampled_path[0] = start_point
        resampled_path[-1] = end_point

        print(f"Resampling: Complete - Original length: {len(path)} -> New length: {fixed_length}")
        return resampled_path

    def _shortcut_path(self, path, collision_fn, iterations=5):
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

        print("---------- Path Optimization ----------")
        print("Shortcutting: Starting path optimization process...")
        optimized_path = deepcopy(path)  # 操作副本以防万一
        n = len(optimized_path)

        # 如果路径已经只有3个点（起点、终点和一个中间点），直接返回
        if n <= 3:
            print("Shortcutting: Path already minimal (length <= 3), skipping")
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
                print(f"  Shortcutting: Found shortcut {shortcut_count} - Between indices {i} and {j}, removing {j-i-1} points")

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

        print(f"Shortcutting: Complete - Original: {len(path)} points, Optimized: {len(optimized_path)} points, Shortcuts: {shortcut_count}")
        return optimized_path

    def _smooth_path_bspline(self, path, collision_fn, num_points=50, smoothing_factor=0):
        """
        使用 B 样条拟合平滑路径，并进行碰撞检测。

        Args:
            path (list): 路径，包含一系列配置点 (e.g., [[x1, y1], [x2, y2], ...]).
            collision_fn (Callable): 碰撞检测函数
            num_points (int): 在生成的样条曲线上采样用于碰撞检测和平滑路径表示的点数。
            smoothing_factor (float): B样条拟合的平滑因子 (s)。
                                        s=0: 样条曲线将通过所有原始点（插值）。
                                        s>0: 样条曲线会更平滑，但可能不通过所有原始点（逼近）。

        Returns:
            list or None: 平滑后的路径（包含 num_points 个配置点），如果样条路径与障碍物碰撞则返回 None。
        """

        # 定义线段碰撞检测函数
        def is_segment_collision(q1, q2):
            """检查两点之间的直线段是否有碰撞"""
            # 在两点之间采样多个中间点进行检查
            samples = 10
            for t in np.linspace(0, 1, samples)[1:-1]:  # 排除端点
                q_interp = q1 + t * (q2 - q1)
                if collision_fn(q_interp):
                    return True
            return False

        if len(path) < 2:
            print("B-Spline: Path too short for smoothing, returning original path")
            return path

        # 保存原始起点和终点
        start_point = path[0].copy()
        end_point = path[-1].copy()

        print("B-Spline: Starting smoothing process...")
        path_np = np.array(path)
        dims = path_np.shape[1]  # 获取配置空间的维度

        # splprep 需要将坐标按维度分开
        # tck 是包含节点向量、系数和次数的元组
        # u 是每个原始点对应的参数值
        try:
            # k 是样条次数，通常为 3 (cubic)
            tck, u = splprep([path_np[:, d] for d in range(dims)], s=smoothing_factor, k=min(3, len(path) - 1))
        except ValueError as e:
            print(f"B-Spline: Error - {e}. Path might be too simple or co-linear")
            return path  # 无法生成样条，返回原始路径

        # 在参数范围 [0, 1] 内均匀生成 num_points 个参数点
        u_new = np.linspace(u.min(), u.max(), num_points)

        # 使用 splev 计算样条曲线上这些参数点对应的配置
        # der=0 表示计算位置
        new_points_coords = splev(u_new, tck, der=0)

        # 将分开的坐标重新组合成配置点列表
        # new_points_coords 是一个包含每个维度坐标列表的元组，需要转置
        smooth_path = np.vstack(new_points_coords).T

        # 确保起点和终点保持不变
        smooth_path[0] = start_point
        smooth_path[-1] = end_point

        # --- 非常重要：检查生成的样条路径是否碰撞 ---
        print("B-Spline: Checking collisions along smoothed path...")
        for i in range(len(smooth_path) - 1):
            if is_segment_collision(smooth_path[i], smooth_path[i + 1]):
                print(f"B-Spline: Collision detected between points {i} and {i+1}")
                print("B-Spline: Smoothing failed due to collision, returning original path")
                return path

        print("B-Spline: Smoothing successful and collision-free")
        print("---------- Path Optimization Complete ----------")
        return smooth_path


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
    grasp_offset = scene_parser.get_robot_grasp_offset()

    # start_q[0] = 20.0 / 180.0 * np.pi

    rb = RobotSetup("rb")  # 'rb' is created here, only available in main block
    rb.set_joint_positions(rb.arm_joints, start_q)
    rb.set_base_pose_2d(pose_2d[0], pose_2d[1], pose_2d[2])

    line_pts_grasped = [np.array([0, 0, 0]), np.array([0, 0, 1])]
    grasped_element = create_collision_bodies(line_pts_grasped, [0.01], viewer=True)[0]
    pp.set_pose(grasped_element, pp.multiply(pp.get_link_pose(rb.robot, rb.tool_link), pp.Pose(point=grasp_offset, euler=pp.Euler(1.5708, 0, 0))))
    grasped_attachment = pp.create_attachment(rb.robot, rb.tool_link, grasped_element)
    rb.update_attachments([grasped_attachment])

    # 执行规划并获取路径
    # path = rb.plan_manipulator_path(start_q, target_q, [], bodies, max_time=600, max_iterations=10000)

    # Pass rb (RobotSetup instance) to the solver
    solver = TrajectoryIPGARSolver(URDF_PATH, rb, grasp_offset)
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
