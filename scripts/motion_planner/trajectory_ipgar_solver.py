import math
import os
import random
import sys
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

import casadi as ca
import numpy as np
import pybullet as p
import pybullet_planning as pp
import torch
from scipy.interpolate import splev, splprep
import time

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from model.scene_parse import SceneParser
from motion_planner.svsdf import SDF
from multi_tangent.collision import create_collision_bodies
from robot.robot_setup import RobotSetup
from utils.collision import element_collision_info, init_pb
from utils.params import URDF_PATH


class RRTTree:
    """RRT树数据结构，使用唯一ID标识节点以避免浮点数舍入误差"""

    def __init__(self, root_config: np.ndarray):
        """初始化RRT树

        Args:
            root_config: 根节点配置
        """
        self.next_id = 0
        self.nodes = {}  # id -> 配置
        self.parents = {}  # id -> 父节点id
        self.children = {}  # id -> 子节点id列表
        self.root_id = None  # 记录根节点ID

        # 添加根节点
        self.add_node(root_config, None)

    def add_node(self, config: np.ndarray, parent_id: Optional[int]) -> int:
        """添加节点到树中

        Args:
            config: 节点配置
            parent_id: 父节点ID，如果是根节点则为None

        Returns:
            新节点的ID
        """
        node_id = self.next_id
        self.next_id += 1

        # 存储节点配置
        self.nodes[node_id] = np.array(config)

        # 设置父子关系
        self.parents[node_id] = parent_id
        if parent_id is not None:
            if parent_id not in self.children:
                self.children[parent_id] = []
            self.children[parent_id].append(node_id)
        else:
            # 如果是根节点（没有父节点），记录根节点ID
            self.root_id = node_id

        return node_id

    def get_node_config(self, node_id: int) -> np.ndarray:
        """获取节点配置

        Args:
            node_id: 节点ID

        Returns:
            节点配置
        """
        return self.nodes[node_id]

    def get_nearest_node(self, target_config: np.ndarray, distance_fn: Callable) -> int:
        """找到距离目标配置最近的节点

        Args:
            target_config: 目标配置
            distance_fn: 距离函数

        Returns:
            最近节点的ID
        """
        min_dist = float("inf")
        nearest_id = None

        for node_id, config in self.nodes.items():
            dist = distance_fn(config, target_config)
            if dist < min_dist:
                min_dist = dist
                nearest_id = node_id

        return nearest_id

    def get_path_to_root(self, node_id: int) -> List[np.ndarray]:
        """获取从指定节点到根节点的路径

        Args:
            node_id: 起始节点ID

        Returns:
            从根节点到指定节点的配置列表
        """
        path = []
        current_id = node_id

        while current_id is not None:
            path.append(self.nodes[current_id])
            current_id = self.parents[current_id]

        # 反转路径，从根到叶
        path.reverse()
        return path

    def get_node_count(self) -> int:
        """获取树中的节点数量

        Returns:
            节点数量
        """
        return len(self.nodes)

    def get_root_node(self) -> np.ndarray:
        """获取树的根节点配置

        Returns:
            根节点配置
        """
        if self.root_id is not None:
            return self.nodes[self.root_id]
        return None

    def rewire(self, collision_fn: Callable, steps: int = 10, max_workers: int = 4) -> int:
        """根据更新后的障碍物对树进行剪枝，去除发生碰撞的分支

        Args:
            collision_fn: 碰撞检测函数
            steps: 路径检查的步数
            max_workers: 并行执行的最大工作线程数

        Returns:
            移除的节点数量
        """
        if self.root_id is None or len(self.nodes) == 0:
            return 0  # 空树，无需剪枝

        removed_count = 0
        nodes_to_remove = []

        # 首先检查根节点
        if collision_fn(self.get_root_node()):
            # 根节点碰撞，清空整棵树
            removed_count = len(self.nodes)
            self.nodes.clear()
            self.parents.clear()
            self.children.clear()
            self.root_id = None
            return removed_count

        # 使用BFS逐层检查节点
        queue = deque([self.root_id])
        visited = {self.root_id}
        
        # 定义用于并行执行的碰撞检测函数
        def check_collision_segment(node_id, child_id):
            """检查从node_id到child_id的路径是否碰撞
            
            Returns:
                Tuple[int, bool]: (子节点ID, 是否碰撞)
            """
            node_config = self.nodes[node_id]
            child_config = self.nodes[child_id]
            
            # 检查子节点本身是否与障碍物碰撞
            if collision_fn(child_config):
                return child_id, True
                
            # 检查从父节点到子节点的路径是否碰撞
            for step in range(1, steps):  # 跳过起点和终点，它们已经被单独检查
                t = step / steps
                interp_config = node_config * (1 - t) + child_config * t
                if collision_fn(interp_config):
                    return child_id, True
                    
            return child_id, False

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while queue:
                node_id = queue.popleft()
                node_config = self.nodes[node_id]

                # 获取子节点
                child_nodes = self.children.get(node_id, [])
                
                if not child_nodes:
                    continue
                    
                # 并行提交所有子节点的碰撞检测任务
                futures = []
                for child_id in child_nodes:
                    if child_id in visited:
                        continue
                    futures.append(executor.submit(check_collision_segment, node_id, child_id))
                
                # 收集结果
                for future in futures:
                    child_id, has_collision = future.result()
                    
                    if has_collision:
                        # 将该节点及其所有后代标记为需要移除
                        self._mark_branch_for_removal(child_id, nodes_to_remove, visited)
                    else:
                        # 节点和路径都无碰撞，添加到队列中继续检查
                        queue.append(child_id)
                        visited.add(child_id)

        # 移除标记的节点和分支
        for node_id in nodes_to_remove:
            self._remove_node(node_id)
            removed_count += 1

        return removed_count

    def _mark_branch_for_removal(self, node_id: int, nodes_to_remove: List[int], visited: set):
        """递归标记需要移除的分支

        Args:
            node_id: 当前节点ID
            nodes_to_remove: 需要移除的节点ID列表
            visited: 已访问节点集合，防止重复访问
        """
        if node_id in visited:
            return

        visited.add(node_id)
        nodes_to_remove.append(node_id)

        # 递归标记所有子节点
        for child_id in self.children.get(node_id, []):
            self._mark_branch_for_removal(child_id, nodes_to_remove, visited)

    def _remove_node(self, node_id: int):
        """从树中移除指定节点

        Args:
            node_id: 要移除的节点ID
        """
        # 从父节点的子节点列表中移除
        parent_id = self.parents.get(node_id)
        if parent_id is not None and parent_id in self.children:
            if node_id in self.children[parent_id]:
                self.children[parent_id].remove(node_id)

            # 如果父节点没有子节点了，清理空列表
            if not self.children[parent_id]:
                del self.children[parent_id]

        # 清理该节点的所有记录
        if node_id in self.nodes:
            del self.nodes[node_id]
        if node_id in self.parents:
            del self.parents[node_id]
        if node_id in self.children:
            del self.children[node_id]


class BiRRTPlanner:
    """双向快速随机探索树(BiRRT)规划器，使用RRTTree类管理树结构"""

    def __init__(self, robot_setup: RobotSetup, step_size: float = 0.05, goal_bias: float = 0.2, max_iterations: int = 10000, goal_threshold: float = 0.05):
        """初始化BiRRT规划器

        Args:
            robot_setup: 机器人配置
            step_size: 每次扩展的步长
            goal_bias: 朝向目标采样的概率
            max_iterations: 最大迭代次数
            goal_threshold: 目标阈值，当两棵树距离小于此值时视为连接
        """
        self.robot = robot_setup
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.max_iterations = max_iterations
        self.goal_threshold = goal_threshold

        # 保存RRT树的私有变量
        self._start_tree = None
        self._end_tree = None

    def plan(
        self,
        start_conf: np.ndarray,
        end_conf: np.ndarray,
        collision_fn: Callable,
        distance_fn: Optional[Callable] = None,
        sample_fn: Optional[Callable] = None,
        extend_fn: Optional[Callable] = None,
        max_time: float = 10.0,
        diagnosis: bool = False,
        smooth: int = 40,
        resolution: float = 1.0,
        start_tree: Optional[RRTTree] = None,
        end_tree: Optional[RRTTree] = None,
    ) -> np.ndarray:
        """规划从起始配置到目标配置的路径

        Args:
            start_conf: 起始配置
            end_conf: 目标配置
            collision_fn: 碰撞检测函数，接受一个配置，返回是否碰撞
            distance_fn: 距离函数，计算两个配置之间的距离，可选
            sample_fn: 采样函数，返回随机配置，可选
            extend_fn: 扩展函数，生成从一个配置到另一个配置的路径点，可选
            max_time: 最大规划时间，单位秒
            diagnosis: 是否启用诊断
            smooth: 路径平滑迭代次数
            resolution: 扩展分辨率
            start_tree: 可选的起始树，如果提供则使用此树
            end_tree: 可选的目标树，如果提供则使用此树

        Returns:
            规划得到的路径，numpy数组形式，若失败则返回None
        """
        start_time = time.time()

        # 检查目标配置是否有效
        if collision_fn(end_conf, diagnosis=diagnosis):
            print("目标配置处于碰撞状态")
            return None

        # 检查起始配置是否有效
        if collision_fn(start_conf, diagnosis=diagnosis):
            print("起始配置处于碰撞状态")
            return None

        # 初始化默认函数
        if distance_fn is None:
            distance_fn = pp.get_distance_fn(self.robot.robot, self.robot.arm_joints)

        if sample_fn is None:

            def get_sample_fn():
                lower, upper = pp.get_custom_limits(self.robot.robot, self.robot.arm_joints, circular_limits=pp.CIRCULAR_LIMITS)
                generator = pp.interval_generator(lower, upper)

                def fn():
                    sample = list(next(generator))
                    return tuple(sample)

                return fn

            sample_fn = get_sample_fn()

        if extend_fn is None:
            resolutions = np.array([resolution / 180.0 * np.pi for j in self.robot.arm_joints])
            extend_fn = pp.get_extend_fn(self.robot.robot, self.robot.arm_joints, resolutions=resolutions)

        # 初始化树，如果提供了树则使用，否则创建新的
        start_tree = start_tree if start_tree is not None else RRTTree(start_conf)
        end_tree = end_tree if end_tree is not None else RRTTree(end_conf)

        # 交替扩展两棵树
        start_to_end = True

        try:
            for iteration in range(self.max_iterations):
                if time.time() - start_time > max_time:
                    print(f"规划超时，已用时 {time.time() - start_time:.2f}s")
                    # 保存当前树
                    self._start_tree = start_tree
                    self._end_tree = end_tree
                    return None

                # 采样一个随机配置或朝向目标偏置
                if start_to_end and random.random() < self.goal_bias:
                    rand_conf = end_tree.get_root_node()
                elif not start_to_end and random.random() < self.goal_bias:
                    rand_conf = start_tree.get_root_node()
                else:
                    rand_conf = sample_fn()

                # 根据当前扩展方向选择树
                active_tree = start_tree if start_to_end else end_tree
                inactive_tree = end_tree if start_to_end else start_tree

                # 找到活动树中最近的节点
                nearest_id = active_tree.get_nearest_node(rand_conf, distance_fn)
                nearest_config = active_tree.get_node_config(nearest_id)

                # 获取从nearest_config到rand_conf的扩展路径
                # 关键修改：extend_fn返回一个生成器，包含从q1到q2的路径点
                extension_path = list(extend_fn(nearest_config, rand_conf))
                if len(extension_path) <= 1:  # 只有起点，没有生成新的配置
                    continue

                # 尝试扩展活动树，直到碰撞或者达到目标
                last_valid_id = nearest_id
                for i, new_config in enumerate(extension_path[1:], 1):  # 跳过第一个点(与nearest_config相同)
                    if collision_fn(new_config, diagnosis=diagnosis):
                        break  # 在这一点发生碰撞，停止扩展

                    # 将新节点添加到活动树
                    new_id = active_tree.add_node(new_config, last_valid_id)
                    last_valid_id = new_id

                    # 找到非活动树中最近的节点
                    connect_id = inactive_tree.get_nearest_node(new_config, distance_fn)
                    connect_config = inactive_tree.get_node_config(connect_id)

                    # 检查是否可以直接连接两棵树
                    if distance_fn(new_config, connect_config) < self.goal_threshold:
                        # 构建路径
                        if start_to_end:
                            path = self._extract_path(start_tree, new_id, end_tree, connect_id)
                        else:
                            path = self._extract_path(start_tree, connect_id, end_tree, new_id)

                        print(f"找到路径，迭代次数: {iteration+1}, 扩展点数: {i}, 用时: {time.time() - start_time:.2f}s")
                        print(f"树大小 - 起点树: {start_tree.get_node_count()}, 终点树: {end_tree.get_node_count()}")

                        # 保存当前树
                        self._start_tree = start_tree
                        self._end_tree = end_tree

                        # 应用路径平滑
                        if smooth > 0:
                            path = self._smooth_path(path, collision_fn, extend_fn, distance_fn, smooth, diagnosis)

                        return np.array(path)

                    # 尝试连接两棵树
                    connection_path = list(extend_fn(connect_config, new_config))

                    # 检查连接路径是否有碰撞
                    connection_valid = True
                    last_connect_id = connect_id
                    connecting_nodes = []  # 保存连接过程中添加的节点ID

                    for connect_config in connection_path[1:]:  # 跳过第一个点
                        if collision_fn(connect_config, diagnosis=diagnosis):
                            connection_valid = False
                            break  # 连接路径中发生碰撞

                        # 向inactive_tree添加新节点
                        new_connect_id = inactive_tree.add_node(connect_config, last_connect_id)
                        last_connect_id = new_connect_id
                        connecting_nodes.append(new_connect_id)

                        # 检查是否足够接近以完成连接
                        if distance_fn(connect_config, new_config) < self.goal_threshold:
                            # 构建路径
                            if start_to_end:
                                path = self._extract_path(start_tree, new_id, end_tree, new_connect_id)
                            else:
                                path = self._extract_path(start_tree, new_connect_id, end_tree, new_id)

                            print(f"找到路径(通过连接)，迭代次数: {iteration+1}, 用时: {time.time() - start_time:.2f}s")
                            print(f"树大小 - 起点树: {start_tree.get_node_count()}, 终点树: {end_tree.get_node_count()}")

                            # 保存当前树
                            self._start_tree = start_tree
                            self._end_tree = end_tree

                            # 应用路径平滑
                            if smooth > 0:
                                path = self._smooth_path(path, collision_fn, extend_fn, distance_fn, smooth, diagnosis)

                            return np.array(path)

                    if connection_valid:
                        # 如果整个连接路径都是有效的，但没有足够接近，保留最后的连接点继续尝试
                        continue

                # 切换扩展的树
                start_to_end = not start_to_end

            print(f"达到最大迭代次数 {self.max_iterations}，规划失败")
            # 保存当前树，尽管规划失败
            self._start_tree = start_tree
            self._end_tree = end_tree
            return None

        except Exception as e:
            print(f"规划过程中出现异常: {e}")
            # 保存当前树，即使发生异常
            self._start_tree = start_tree
            self._end_tree = end_tree
            return None

    def _extract_path(self, start_tree: RRTTree, start_node_id: int, end_tree: RRTTree, end_node_id: int) -> List[np.ndarray]:
        """从两棵树中提取完整路径

        Args:
            start_tree: 起点树
            start_node_id: 起点树中连接点的ID
            end_tree: 终点树
            end_node_id: 终点树中连接点的ID

        Returns:
            完整路径，配置列表
        """
        # 从起点树获取路径
        start_path = start_tree.get_path_to_root(start_node_id)

        # 从终点树获取路径并反转
        end_path = end_tree.get_path_to_root(end_node_id)
        end_path.reverse()  # 反转以获得从连接点到终点的路径

        # 连接两部分路径
        return start_path + end_path[1:]  # 去掉重复的连接点

    def _smooth_path(self, path: List[np.ndarray], collision_fn: Callable, extend_fn: Callable, distance_fn: Callable, iterations: int, diagnosis: bool) -> List[np.ndarray]:
        """对路径进行平滑处理

        Args:
            path: 原始路径
            collision_fn: 碰撞检测函数
            extend_fn: 扩展函数
            distance_fn: 距离函数
            iterations: 平滑迭代次数
            diagnosis: 是否启用诊断

        Returns:
            平滑后的路径
        """
        if len(path) <= 2:
            return path

        smoothed_path = list(path)  # 复制原始路径

        for _ in range(iterations):
            if len(smoothed_path) <= 2:
                break  # 路径已经不能再简化

            # 随机选择两个点
            i = random.randint(0, len(smoothed_path) - 1)
            j = random.randint(0, len(smoothed_path) - 1)

            if abs(i - j) <= 1:
                continue  # 跳过相邻点

            if i > j:
                i, j = j, i  # 确保 i < j

            # 检查i和j之间是否可以直接连接
            direct_path = list(extend_fn(smoothed_path[i], smoothed_path[j]))

            # 检查直接路径是否有碰撞
            path_valid = True
            for q in direct_path[1:-1]:  # 跳过起点和终点
                if collision_fn(q, diagnosis=diagnosis):
                    path_valid = False
                    break

            if path_valid:
                # 替换i和j之间的路径为直接连接
                smoothed_path = smoothed_path[: i + 1] + direct_path[1:] + smoothed_path[j + 1 :]

        return smoothed_path

    def get_trees(self) -> Tuple[RRTTree, RRTTree]:
        """获取当前保存的RRT树

        Returns:
            起点树和终点树的元组
        """
        return self._start_tree, self._end_tree

    def set_trees(self, start_tree: RRTTree, end_tree: RRTTree):
        """设置当前保存的RRT树

        Args:
            start_tree: 起点树
            end_tree: 终点树
        """
        self._start_tree = start_tree
        self._end_tree = end_tree


class TrajectoryIPGARSolver:
    def __init__(self, urdf_path: str, robot_setup: RobotSetup, grasp_offset: List[float]) -> None:
        """Initialize IPGAR trajectory planner with custom BIRRT as base planner"""
        self.urdf_path = urdf_path
        self.robot_setup = robot_setup
        self.grasp_offset = grasp_offset

        # # BIRRT parameters
        # self.step_size = 0.05  # Extension step size
        # self.goal_bias = 0.20  # Goal bias probability
        # self.max_iterations = 10000  # Maximum iterations
        # self.goal_threshold = 0.05  # Goal threshold
        # self.interpolation_steps = 5  # Path interpolation steps

        self.planner = BiRRTPlanner(robot_setup)

        # SDF computation setup
        self.x_sym = ca.MX.sym("x", 3)
        self.q_sym = ca.MX.sym("q", 6)
        self.p_sym = ca.MX.sym("p", 3)
        sdf = SDF(self.urdf_path, self.robot_setup, self.q_sym, self.p_sym, self.x_sym, self.grasp_offset, element_collision_info)
        sdf_sym = sdf(self.p_sym, self.q_sym, self.x_sym)
        sdf_grad = ca.gradient(sdf_sym, self.q_sym)
        self.sdf = ca.Function("sdf", [self.p_sym, self.q_sym, self.x_sym], [sdf_sym, sdf_grad])
        self.sdf_threshold = 0.01

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
        # 为每个障碍物定义一个计算最小SDF的函数
        def compute_min_sdf_for_obstacle(obs_idx_and_dict):
            obs_idx, obstacle_dict = obs_idx_and_dict
            min_sdf = float("inf")
            obs_pos = np.array(obstacle_dict["position"])
            for point in path:
                sdf_val_ca, _ = self.sdf(pose_2d, point, obs_pos)
                min_sdf = min(min_sdf, sdf_val_ca.toarray().item())
            return obs_idx, min_sdf
        
        # 创建障碍物索引和字典的列表
        obs_with_idx = [(obs_idx, obstacle_dict) for obs_idx, obstacle_dict in enumerate(obstacles)]
        
        # 使用线程池并行计算
        min_sdfs = {}
        max_workers = min(8, len(obstacles))  # 设置合理的工作线程数，避免创建过多线程
        
        if len(obstacles) > 0:  # 确保有障碍物需要处理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有计算任务
                future_results = [executor.submit(compute_min_sdf_for_obstacle, obs_item) for obs_item in obs_with_idx]
                
                # 收集结果
                for future in future_results:
                    obs_idx, min_sdf = future.result()
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

    def plan(self, q_init: np.ndarray, q_target: np.ndarray, pose_2d: np.ndarray, element_bodies: List[int]) -> Dict:
        """Main planning function using IPGAR algorithm"""
        print("\n========== IPGAR TRAJECTORY PLANNING ==========")

        density_levels = [50]
        final_path = None
        start_tree, end_tree = None, None
        plan_success = True

        # **************************************************************************
        # 外层循环：逐步增加障碍物密度
        # **************************************************************************

        for iteration, density in enumerate(density_levels):
            print(f"\n========== DENSITY ITERATION {iteration+1}/{len(density_levels)} (count={density}) ==========")

            # **************************************************************************
            # 1. 初始化障碍物
            # **************************************************************************

            # -------------------- 1.1 初始化障碍物 --------------------#
            spheres = [SceneParser.approximate_cylinder(element_body, count=density) for element_body in element_bodies]
            active_obstacles = []
            active_obstacles_body = []
            remaining_obstacles_dict = {}  # 使用字典存储remaining_obstacles，以name为键
            for sphere_list in spheres:
                for sphere in sphere_list:
                    name = sphere.get("name", f"unnamed_{len(remaining_obstacles_dict)}")
                    remaining_obstacles_dict[name] = sphere

            # -------------------- 1.2 初始路径规划 --------------------#
            collision_fn = self._create_collision_fn(active_obstacles_body)
            if final_path is None:
                print(f"IPGAR: Planning initial path in free space...")
                initial_solution = self.planner.plan(q_init, q_target, collision_fn, smooth=100)
                if initial_solution is None:
                    print("IPGAR: ERROR - Initial path planning failed in free space")
                    print("\n========== IPGAR PLANNING FAILED ==========")
                    plan_success = False
                    break
                print("IPGAR: Initial path successfully found")
                current_path = initial_solution
            else:
                current_path = final_path

            # **************************************************************************
            # 2. 筛选障碍物
            # **************************************************************************

            print(f"IPGAR: Processing obstacles - {len(remaining_obstacles_dict)} remaining")

            # -------------------- 2.1 Calculate SVSDF for all remaining obstacles --------------------#
            print("IPGAR: Calculating SVSDF values for remaining obstacles...")
            remaining_obstacles_list = list(remaining_obstacles_dict.values())
            svsdf_values = self._compute_path_obstacle_sdfs(pose_2d, current_path, remaining_obstacles_list)

            # -------------------- 2.2 Map SVSDF values from indices to obstacle names --------------------#
            svsdf_by_name = {}
            for i, obs in enumerate(remaining_obstacles_list):
                name = obs.get("name", f"unnamed_{i}")
                svsdf_by_name[name] = svsdf_values.get(i, float("inf"))

            # -------------------- 2.3 Group obstacles by element --------------------#
            element_obstacles = {}
            for name, obs in remaining_obstacles_dict.items():
                if "_sphere_" in name:
                    element_id = name.split("_sphere_")[0]
                    if element_id not in element_obstacles:
                        element_obstacles[element_id] = []
                    element_obstacles[element_id].append((name, obs, svsdf_by_name.get(name, 0)))

            # -------------------- 2.4 Find elements where all obstacles have SVSDF > threshold --------------------#
            elements_all_above_threshold = []
            for element_id, obs_list in element_obstacles.items():
                if all(sdf_val > self.sdf_threshold for _, _, sdf_val in obs_list):
                    elements_all_above_threshold.append(element_id)

            # -------------------- 2.5 Process elements with all obstacles above threshold --------------------#
            if elements_all_above_threshold:
                print(f"IPGAR: Found {len(elements_all_above_threshold)} elements with all obstacles above threshold {self.sdf_threshold}")
                with pp.LockRenderer():
                    for element_id in elements_all_above_threshold:
                        for name, obs, sdf_val in element_obstacles[element_id]:
                            obs_id = pp.create_sphere(obs["radius"])
                            pp.set_point(obs_id, obs["position"])
                            pp.set_color(obs_id, [0, 0, 1, 1])
                            del remaining_obstacles_dict[name]
                            active_obstacles.append(obs)
                            active_obstacles_body.append(obs_id)
                            print(f"    IPGAR: Activated obstacle {name}")
                collision_fn = self._create_collision_fn(active_obstacles_body)

            # -------------------- 2.6 Check if need to continue --------------------#
            if len(remaining_obstacles_dict) == 0:
                print("IPGAR: All obstacles have been added, continue to next density level")
                continue

            # -------------------- 2.7 Update trees --------------------#
            with pp.LockRenderer():
                print(f"IPGAR: Updating trees...")
                start_tree, end_tree = self.planner.get_trees()
                start_tree.rewire(collision_fn, steps=3)
                end_tree.rewire(collision_fn, steps=3)
                self.planner.set_trees(start_tree, end_tree)

            # **************************************************************************
            # 3. 逐个添加剩余的障碍物
            # **************************************************************************

            print(f"IPGAR: Processing obstacles - {len(remaining_obstacles_dict)} remaining")

            # -------------------- 3.1 For remaining elements, calculate average SVSDF --------------------#
            print("IPGAR: Calculating average SVSDF for remaining elements...")
            element_averages = {}
            for element_id, obs_list in element_obstacles.items():
                if element_id not in elements_all_above_threshold:
                    avg_sdf = np.mean([sdf_val for _, _, sdf_val in obs_list])
                    element_averages[element_id] = avg_sdf

            # -------------------- 3.2 Select element with highest average SVSDF --------------------#
            avg_element = max(element_averages.items(), key=lambda x: x[1])[0]
            print(f"IPGAR: Selected element {avg_element} with average SVSDF {element_averages[avg_element]:.4f}")

            # # -------------------- 3.2 Select element with lowest average SVSDF --------------------#
            # avg_element = min(element_averages.items(), key=lambda x: x[1])[0]
            # print(f"IPGAR: Selected element {avg_element} with average SVSDF {element_averages[avg_element]:.4f}")

            # -------------------- 3.3 Sort obstacles --------------------#
            # sorted_obstacles = [sorted(element_obstacles[element_id], key=lambda x: x[2], reverse=True) for element_id in element_averages.keys()] # SVSDF从大到小排序
            # sorted_obstacles = [sorted(element_obstacles[element_id], key=lambda x: x[2]) for element_id in element_averages.keys()] # SVSDF从小到大排序
            sorted_obstacles = [sorted(element_obstacles[element_id], key=lambda x: int(x[0].split("_sphere_")[1]) if "_sphere_" in x[0] else 0) for element_id in element_averages.keys()]  # 按照obstacle的编号sphere_id排序

            # -------------------- 3.4 Process obstacles one by one --------------------#
            while len([item for sublist in sorted_obstacles for item in sublist]) != 0:

                # -------------------- 3.4.1 寻找碰撞的连续障碍物列表 --------------------#
                with pp.LockRenderer():
                    find_collision = False
                    current_active_obs_ids = []
                    for sub_obs_list in sorted_obstacles:
                        current_obs_ids = []
                        # 找到第一个碰撞的障碍物
                        while len(sub_obs_list) != 0:
                            name, obs, sdf_val = sub_obs_list.pop(0)
                            obs_id = pp.create_sphere(obs["radius"])
                            pp.set_point(obs_id, obs["position"])
                            pp.set_color(obs_id, [0, 1, 0, 1])
                            del remaining_obstacles_dict[name]
                            active_obstacles.append(obs)
                            active_obstacles_body.append(obs_id)
                            current_obs_ids.append(obs_id)
                            if sdf_val <= self.sdf_threshold:
                                print(f"    IPGAR: Activated collision obstacle {name}")
                                find_collision = True
                                break
                            else:
                                print(f"    IPGAR: Activated no collision obstacle {name}")
                                pp.set_color(obs_id, [0, 0, 1, 1])
                        # 寻找后续连续的碰撞障碍物
                        while len(sub_obs_list) != 0:
                            name, obs, sdf_val = sub_obs_list[0]
                            if sdf_val <= self.sdf_threshold:
                                sub_obs_list.pop(0)
                                obs_id = pp.create_sphere(obs["radius"])
                                pp.set_point(obs_id, obs["position"])
                                pp.set_color(obs_id, [0, 1, 0, 1])
                                del remaining_obstacles_dict[name]
                                active_obstacles.append(obs)
                                active_obstacles_body.append(obs_id)
                                current_obs_ids.append(obs_id)
                                print(f"    IPGAR: Activated continuous collision obstacle {name}")
                            else:
                                break
                        current_active_obs_ids.extend(current_obs_ids)

                # -------------------- 3.4.2 当前没有碰撞，可以结束 --------------------#
                if len(remaining_obstacles_dict) == 0 and not find_collision:
                    break

                # -------------------- 3.4.3 Update collision function, check path and rewire trees --------------------#
                with pp.LockRenderer():
                    print(f"    IPGAR: Updating collision function and rewiring trees...")
                    collision_fn = self._create_collision_fn(active_obstacles_body)
                    collision_intervals = self._check_path_collision(current_path, collision_fn)
                    start_tree.rewire(collision_fn, steps=2)
                    end_tree.rewire(collision_fn, steps=2)
                    self.planner.set_trees(start_tree, end_tree)
                # -------------------- 3.4.4 路径有碰撞，进行修复 --------------------#
                if len(collision_intervals) > 0:
                    print(f"IPGAR: Path has {len(collision_intervals)} collisions, repairing...")
                    # Repair path
                    repair_result = self._repair_path(current_path, collision_intervals, collision_fn, active_obstacles_body)
                    if repair_result["success"]:
                        current_path = repair_result["path"]
                        print("IPGAR: Path successfully repaired")
                        # Recalculate SVSDF for remaining obstacles in this element after successful repair
                        temp_sorted_obstacles = []
                        for sub_obs_list in sorted_obstacles:
                            temp_svsdf_values = self._compute_path_obstacle_sdfs(pose_2d, current_path, [item[1] for item in sub_obs_list])
                            temp_sorted_obs = []
                            for i, item in enumerate(sub_obs_list):
                                new_item = list(deepcopy(item))
                                new_item[2] = temp_svsdf_values[i]
                                print(f"    IPGAR: Updated SVSDF for {item[0]}: {item[2]:.4f} -> {new_item[2]:.4f}")
                                temp_sorted_obs.append(tuple(new_item))
                            temp_sorted_obstacles.append(temp_sorted_obs)
                        sorted_obstacles = deepcopy(temp_sorted_obstacles)
                        for temp_id in current_active_obs_ids:
                            pp.set_color(temp_id, [0, 0, 1, 1])
                    else:
                        if len(remaining_obstacles_dict) == 0:
                            print("IPGAR: Repair failed but there is no remaining obstacles!")
                            print("\n========== IPGAR PLANNING FAILED ==========")
                            plan_success = False
                            break
                        else:
                            print("IPGAR: Repair failed...")
                            for temp_id in current_active_obs_ids:
                                pp.set_color(temp_id, [0, 1, 1, 1])

            # -------------------- 3.5 If plan failed, remove all obstacles and return --------------------#
            if not plan_success:
                for temp_id in active_obstacles_body:
                    pp.remove_body(temp_id)
                return {"success": False, "path": None}

            # -------------------- 3.6 If plan success, smooth path --------------------#
            with pp.LockRenderer():
                temp_path = self._shortcut_path(current_path, collision_fn, iterations=30)
                temp_path = self._resample_path_fixed_length(temp_path, fixed_length=max(100, len(temp_path)))
                current_path = deepcopy(temp_path)

            # -------------------- 3.7 Clean up PyBullet objects from current iteration --------------------#
            with pp.LockRenderer():
                for body_id in active_obstacles_body:
                    pp.remove_body(body_id)

            # -------------------- 3.8 Set iteration result --------------------#
            if current_path is not None:
                final_path = current_path
                print(f"IPGAR: Density level {density} planning successful!")
                continue
            else:
                if final_path is not None:
                    print(f"IPGAR: Density level {density} planning incomplete, using previous result!")
                    continue
                else:
                    print("IPGAR: ERROR - First density level planning failed!")
                    return {"success": False, "path": None}

        # **************************************************************************
        # 4. Final path check and output
        # **************************************************************************

        final_density = max([d for i, d in enumerate(density_levels) if i < len(density_levels) and (i == len(density_levels) - 1 or final_path is not None)])
        final_spheres = [SceneParser.approximate_cylinder(element_body, count=final_density) for element_body in element_bodies]
        final_obstacle_bodies = []

        with pp.LockRenderer():
            for sphere_list in final_spheres:
                for sphere in sphere_list:
                    obs_id = pp.create_sphere(sphere["radius"])
                    pp.set_point(obs_id, sphere["position"])
                    pp.set_color(obs_id, [1, 0, 0, 0.5])
                    final_obstacle_bodies.append(obs_id)

        final_collision_fn = self._create_collision_fn(final_obstacle_bodies)
        final_collision_check = self._check_path_collision(final_path, final_collision_fn)
        if len(final_collision_check) > 0:
            print(f"IPGAR: ERROR - Final path still has {len(final_collision_check)} collisions!")
            with pp.LockRenderer():
                for body_id in final_obstacle_bodies:
                    pp.remove_body(body_id)
            return {"success": False, "path": None}
        else:
            print("IPGAR: Final path successfully validated")

        print("\n========== IPGAR PLANNING SUCCESSFUL ==========")
        final_path = self._resample_path_fixed_length(final_path, fixed_length=max(1000, len(final_path)))
        with pp.LockRenderer():
            for body_id in final_obstacle_bodies:
                pp.remove_body(body_id)
        return {"success": True, "path": final_path}

    def _repair_path(self, path, collision_intervals, collision_fn, active_obstacles_body):
        """Repair path segments with collisions"""
        print("    ---------- Path Repair ----------")
        print(f"    Repair: Processing {len(collision_intervals)} collision segments...")
        repair_successful = True

        # 保存原始起点和终点
        start_point = path[0].copy()
        end_point = path[-1].copy()

        current_path = path.copy()  # 创建路径副本，以便在修补过程中更新

        def get_sample_fn(lower, upper):
            generator = pp.interval_generator(lower, upper)

            def fn():
                sample = list(next(generator))
                return tuple(sample)

            return fn
            
        # 创建带SVSDF斥力场的扩展函数
        def get_extend_fn(original_extend_fn, pose_2d, obstacles, repulsion_weight=1.0, attraction_weight=1.0, safety_distance=0.05):
            """
            创建一个扩展函数，结合原始扩展方向和SVSDF障碍物斥力
            
            Args:
                original_extend_fn: 原始扩展函数
                pose_2d: 机器人2D位姿
                obstacles: 障碍物列表
                repulsion_weight: 斥力权重
                attraction_weight: 吸引力(原始方向)权重
                safety_distance: 安全距离阈值，当SDF小于此值时增强斥力
                
            Returns:
                新的扩展函数
            """
            # 确保obstacles是列表
            if not isinstance(obstacles, list):
                obstacles = []
                
            # 从active_obstacles_body获取障碍物位置信息
            obstacle_positions = []
            for obs_id in active_obstacles_body:
                try:
                    pos, _ = pp.get_pose(obs_id)
                    radius = 0.01
                    obstacle_positions.append({"position": pos, "radius": radius})
                except Exception as e:
                    print(f"警告: 无法获取障碍物 {obs_id} 的位置: {e}")
            
            def svsdf_extend(q1, q2):
                """
                带SVSDF斥力的扩展函数
                
                Args:
                    q1: 起始配置
                    q2: 目标配置
                    
                Returns:
                    修改后的路径点生成器
                """
                # 首先获取原始路径点
                original_path = list(original_extend_fn(q1, q2))
                if len(original_path) <= 1:
                    # 原始扩展失败，直接返回
                    for pt in original_path:
                        yield pt
                    return
                
                # 获取扩展方向（单位向量）
                direction = q2 - q1
                direction_norm = np.linalg.norm(direction)
                if direction_norm < 1e-6:
                    # 如果方向太小，直接返回原始路径
                    for pt in original_path:
                        yield pt
                    return
                
                direction = direction / direction_norm  # 归一化
                
                # 处理路径上的每个点
                for i, q in enumerate(original_path):
                    if i == 0:  # 第一个点直接返回
                        yield q
                        continue
                        
                    # 确保使用numpy数组进行运算
                    q_np = np.array(q) if isinstance(q, tuple) else q
                    q_prev_np = np.array(original_path[i-1]) if isinstance(original_path[i-1], tuple) else original_path[i-1]
                    
                    # 计算所有障碍物对当前点的斥力
                    repulsion = np.zeros_like(q_np)
                    for obs in obstacle_positions:
                        # 计算SDF值和梯度
                        obs_pos = np.array(obs["position"])
                        sdf_val, sdf_grad = self.sdf(pose_2d, q_np, obs_pos)
                        sdf_val = float(sdf_val)
                        
                        # 当SDF值小于安全距离时增强斥力
                        if sdf_val < safety_distance:
                            # 斥力与SDF负梯度成正比，与SDF值成反比
                            force_magnitude = (safety_distance - sdf_val) / safety_distance
                            force_direction = -np.array(sdf_grad).flatten()  # 使用SDF梯度的反方向
                            
                            # 归一化斥力方向
                            force_dir_norm = np.linalg.norm(force_direction)
                            if force_dir_norm > 1e-6:
                                force_direction = force_direction / force_dir_norm
                                
                                # 累加所有障碍物的斥力
                                repulsion += force_magnitude * force_direction
                    
                    # 如果有计算得到的斥力
                    if np.linalg.norm(repulsion) > 1e-6:
                        # 归一化斥力
                        repulsion = repulsion / np.linalg.norm(repulsion)
                        
                        # 结合原始方向和斥力方向
                        combined_direction = attraction_weight * direction + repulsion_weight * repulsion
                        combined_norm = np.linalg.norm(combined_direction)
                        
                        if combined_norm > 1e-6:
                            # 归一化组合方向
                            combined_direction = combined_direction / combined_norm
                            
                            # 计算修正后的位置 - 注意使用numpy数组
                            step_size = np.linalg.norm(q_np - q_prev_np)
                            new_q = q_prev_np + step_size * combined_direction
                            
                            # 如果需要，将numpy数组转回元组
                            if isinstance(q, tuple):
                                new_q = tuple(new_q)
                                
                            yield new_q
                            continue
                
                    # 如果没有有效的斥力或组合失败，使用原始路径点
                    yield q
            
            return svsdf_extend

        # 循环直到所有碰撞区间都被处理
        while collision_intervals:
            # 获取当前区间
            current_start, current_end = collision_intervals[0]
            expanded_start = current_start
            expanded_end = current_end
            repair_attempts = 0
            max_repair_attempts = 20

            while repair_attempts < max_repair_attempts:
                print(f"    Repair: Attempt {repair_attempts + 1}/{max_repair_attempts} for segment {expanded_start}-{expanded_end}")

                # 尝试修复扩展区间
                q_start_local = current_path[expanded_start]
                q_end_local = current_path[expanded_end]

                max_plan_time = 5.0 + repair_attempts * 5.0

                # 计算当前轨迹片段上每个关节角的最大值和最小值
                joint_limits = []
                for joint_idx in range(q_start_local.shape[0]):
                    joint_values = current_path[expanded_start : expanded_end + 1, joint_idx]
                    min_val = np.min(joint_values)
                    max_val = np.max(joint_values)
                    joint_limits.append((min_val, max_val))

                # 扩展关节限制范围以增加采样空间
                expansion_factor = 0.5
                expanded_limits = []
                lower_raw, upper_raw = pp.get_custom_limits(self.robot_setup.robot, self.robot_setup.arm_joints, circular_limits=pp.CIRCULAR_LIMITS)
                for i, joint_range in enumerate(joint_limits):
                    min_val, max_val = joint_range
                    range_val = max_val - min_val
                    range_val = max(range_val * expansion_factor, (upper_raw[i] - lower_raw[i]) * (repair_attempts + 1) / max_repair_attempts * 0.5)
                    expanded_min = max(lower_raw[i], min_val - range_val)
                    expanded_max = min(upper_raw[i], max_val + range_val)
                    expanded_limits.append((expanded_min, expanded_max))

                print("    Repair: joint sample space")
                print(f"        lower limit: {[f'{limit[0]:.3f}' for limit in expanded_limits]}")
                print(f"        upper limit: {[f'{limit[1]:.3f}' for limit in expanded_limits]}")
                print(f"        percentage : {[f'{(expanded_limits[i][1] - expanded_limits[i][0])/(upper_raw[i]- lower_raw[i]) * 100.0:.3f}' for i in range(len(expanded_limits))]}%")

                # 使用扩展后的限制创建采样函数
                sample_fn = get_sample_fn(lower=np.array([limit[0] for limit in expanded_limits]), upper=np.array([limit[1] for limit in expanded_limits]))
                
                # 获取原始扩展函数
                resolutions = np.array([0.1 / 180.0 * np.pi for j in self.robot_setup.arm_joints])  # 使用较小的分辨率提高精度
                original_extend_fn = pp.get_extend_fn(self.robot_setup.robot, self.robot_setup.arm_joints, resolutions=resolutions)
                
                # 创建增强的扩展函数，结合SVSDF斥力
                # 根据修复尝试次数调整权重
                repulsion_weight = min(2.0, 0.5 + repair_attempts * 0.1)  # 随着尝试次数增加，增加斥力权重
                attraction_weight = 1.0
                
                # 获取当前的2D姿态
                pose_2d_current = pp.get_joint_positions(self.robot_setup.robot, self.robot_setup.base_joints)
                
                # 创建结合SVSDF的扩展函数
                svsdf_extend_fn = get_extend_fn(
                    original_extend_fn, 
                    pose_2d_current, 
                    active_obstacles_body,
                    repulsion_weight=repulsion_weight,
                    attraction_weight=attraction_weight,
                    safety_distance=0.01 * (1 + repair_attempts * 0.1)  # 随着尝试次数增加，扩大安全距离
                )

                with pp.LockRenderer():
                    start_tree, end_tree = self.planner.get_trees()
                    repair_path = self.planner.plan(
                        start_point, 
                        end_point, 
                        collision_fn, 
                        sample_fn=sample_fn, 
                        extend_fn=None,  # 使用增强的扩展函数
                        max_time=max_plan_time, 
                        start_tree=start_tree, 
                        end_tree=end_tree
                    )

                if repair_path is not None:
                    print(f"    Repair: Success after {repair_attempts + 1} attempts")
                    current_path = repair_path

                    # 重新检测碰撞区间
                    with pp.LockRenderer():
                        collision_intervals = self._check_path_collision(current_path, collision_fn)
                    print(f"    Repair: After update, {len(collision_intervals)} collision segments remain")

                    # 如果没有剩余碰撞，完成修复
                    if not collision_intervals:
                        print("    Repair: All collisions resolved")
                        break
                    else:
                        # 重置尝试计数器，开始处理新的第一个区间
                        break

                else:
                    print(f"    Repair: Attempt {repair_attempts + 1} failed, expanding repair region...")
                    repair_attempts += 1

                    # 扩展修补区间
                    expansion_size = len(current_path) // (max_repair_attempts * 2)
                    expanded_start = max(0, expanded_start - expansion_size)
                    expanded_end = min(len(current_path) - 1, expanded_end + expansion_size)

            # 如果达到最大尝试次数仍未成功，则修补失败
            if repair_attempts >= max_repair_attempts:
                print(f"    Repair: FAILED after {max_repair_attempts} attempts with expanded regions")
                repair_successful = False
                return {"success": False, "path": None}

        if repair_successful:
            print("    Repair: All segments successfully repaired")
            print("    ---------- Path Repair Complete ----------")

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

        print(f"Resampling: Resampling path from {len(path)} to {fixed_length} points...")

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
            samples = 360
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
                print(f"Shortcutting: Found shortcut {shortcut_count} - Between indices {i} and {j}, removing {j-i-1} points")

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

    rb = RobotSetup("rb")  # 'rb' is created here, only available in main block
    rb.set_joint_positions(rb.arm_joints, start_q)
    rb.set_base_pose_2d(pose_2d[0], pose_2d[1], pose_2d[2])

    line_pts_grasped = [np.array([0, 0, 0]), np.array([0, 0, 1])]
    grasped_element = create_collision_bodies(line_pts_grasped, [0.01], viewer=True)[0]
    pp.set_pose(grasped_element, pp.multiply(pp.get_link_pose(rb.robot, rb.tool_link), pp.Pose(point=grasp_offset, euler=pp.Euler(1.5708, 0, 0))))
    grasped_attachment = pp.create_attachment(rb.robot, rb.tool_link, grasped_element)
    rb.update_attachments([grasped_attachment])

    # Pass rb (RobotSetup instance) to the solver
    solver = TrajectoryIPGARSolver(URDF_PATH, rb, grasp_offset)

    start_time = time.time()

    plan_result = solver.plan(start_q, target_q, pose_2d, bodies)

    end_time = time.time()
    print(f"IPGAR: Time taken: {end_time - start_time:.2f} seconds")

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
