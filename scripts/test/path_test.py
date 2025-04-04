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

import casadi as ca
import numpy as np
import pybullet as p
import pybullet_planning as pp
import torch
from scipy.interpolate import splev, splprep

# 添加项目路径
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

    def rewire(self, collision_fn: Callable) -> int:
        """根据更新后的障碍物对树进行剪枝，去除发生碰撞的分支
        
        Args:
            collision_fn: 碰撞检测函数
            
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
        
        while queue:
            node_id = queue.popleft()
            node_config = self.nodes[node_id]
            
            # 获取子节点
            child_nodes = self.children.get(node_id, [])
            
            for child_id in child_nodes:
                if child_id in visited:
                    continue
                
                child_config = self.nodes[child_id]
                
                # 检查子节点本身是否与障碍物碰撞
                if collision_fn(child_config):
                    # 将该节点及其所有后代标记为需要移除
                    self._mark_branch_for_removal(child_id, nodes_to_remove, visited)
                    continue
                
                # 检查从父节点到子节点的路径是否碰撞
                path_collision = False
                
                # 生成从父节点到子节点的简单路径（这里使用线性插值）
                steps = 10  # 路径检查的步数，可以根据需要调整
                for step in range(1, steps):  # 跳过起点和终点，它们已经被单独检查
                    t = step / steps
                    interp_config = node_config * (1 - t) + child_config * t
                    if collision_fn(interp_config):
                        path_collision = True
                        break
                
                if path_collision:
                    # 路径发生碰撞，移除子节点及其分支
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


# 测试函数
def test_birrt_planner():
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

    extra_disabled_collisions = [((rb.robot, pp.link_from_name(rb.robot, "ur_arm_wrist_3_link")), (rb.ee_attachment.child, pp.BASE_LINK))]
    collision_fn = pp.get_collision_fn(rb.robot, rb.arm_joints, obstacles=bodies[:8], attachments=[rb.ee_attachment] + rb.attachments, self_collisions=True, extra_disabled_collisions=extra_disabled_collisions)
    solver = BiRRTPlanner(rb)
    path = solver.plan(start_q, target_q, collision_fn)

    # path = rb.plan_manipulator_path(start_q, target_q, rb.attachments, bodies[:8])

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


if __name__ == "__main__":
    test_birrt_planner()
