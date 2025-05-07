import colorsys
import itertools
import os
import random
import sys
import time
import warnings
from collections import deque
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pybullet as p
import pybullet_planning as pp
from ompl import base as ob
from ompl import util as ou
from pybullet_planning import Attachment
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from model.scene_parse import SceneParser
from motion_planner.pb_ompl import pb_ompl
from motion_planner.trajectory_ompl_solver import PbOMPLRobotWrapper
from multi_tangent.collision import create_collision_bodies
from robot.robot_setup import RobotSetup
from utils.collision import init_pb
from utils.params import URDF_PATH
from utils.util import PrintManager, interpolate

# 初始化PrintManager实例
printer = PrintManager()


class TopologyPlanner:
    """内部拓扑规划器类，用于高层路径规划

    基于通道信息进行拓扑规划，找出最优的通道通过顺序。
    """

    def __init__(self, robot_setup: RobotSetup, channel_info: List[Dict], bodies: List[int], eval_max_attempts: int = 50000, sample_attempts: int = 100, connect_threshold: float = 0.25, alpha: float = 1.0, beta: float = 2.0, gamma: float = 3.0):
        """初始化拓扑规划器

        Args:
            robot_setup: 机器人设置
            channel_info: 通道信息列表
            bodies: 碰撞体列表
            eval_max_attempts: 评估通道的最大尝试次数
        """
        self.robot_setup = robot_setup
        self.channel_info = channel_info
        self.bodies = bodies
        self.eval_max_attempts = eval_max_attempts
        self.sample_attempts = sample_attempts
        self.connect_threshold = connect_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        with pp.LockRenderer():
            self.channel_info = self._evaluate_channel_priority()
            self.channel_colors = self._generate_channel_colors()
            self.channel_graph = self._build_channel_graph(sample_attempts=self.sample_attempts, connect_threshold=self.connect_threshold)  # {channel_idx: {neighbor_idx: weight}}

    def _evaluate_channel_priority(self) -> List[Dict]:
        """评估各个通道的优先级

        基于可达性和几何特性计算每个通道的优先级得分。

        Returns:
            更新了优先级信息的通道列表
        """

        # 评估通道可达性
        def get_sample_fn():
            lower, upper = pp.get_custom_limits(self.robot_setup.robot, self.robot_setup.arm_joints, circular_limits=pp.CIRCULAR_LIMITS)
            generator = pp.interval_generator(lower, upper)

            def fn():
                sample = list(next(generator))
                return tuple(sample)

            return fn

        sample_fn = get_sample_fn()
        collision_fn = self.robot_setup.create_collision_fn(self.bodies)
        channel_reachability = []

        channel_body = None
        for channel in self.channel_info:
            check = 0
            attempt = 0
            if channel_body is not None:
                pp.remove_body(channel_body)
            channel_body = SceneParser.load_channel(channel)
            contact_fn = self.robot_setup.create_collision_fn([channel_body])
            while attempt < self.eval_max_attempts:
                joint_val = sample_fn()
                if not collision_fn(joint_val) and contact_fn(joint_val):
                    check += 1
                attempt += 1
            channel_reachability.append(check)
        if channel_body is not None:
            pp.remove_body(channel_body)

        # 计算可达性权重
        reachability_with_index = [(value, idx) for idx, value in enumerate(channel_reachability)]
        reachability_with_index.sort(reverse=True)
        num_channels = len(channel_reachability)
        for rank, (value, idx) in enumerate(reachability_with_index):
            weight = (num_channels - rank) / num_channels
            self.channel_info[idx]["reachability"] = value
            self.channel_info[idx]["reachability_weight"] = weight

        # 评估通道几何特性
        for channel in self.channel_info:
            channel_type = channel["type"]
            passability = SceneParser.compute_area(channel_type, channel)
            channel["passability"] = passability

        # 计算总优先级
        for channel in self.channel_info:
            channel["priority"] = self.alpha * channel["reachability_weight"] + self.beta * channel["passability"]

        return self.channel_info

    def _generate_channel_colors(self) -> List[List[float]]:
        """为每个通道生成唯一的颜色

        Returns:
            每个通道对应的颜色列表
        """
        num_channels = len(self.channel_info)
        colors = []

        # 使用HSV颜色空间生成均匀分布的颜色
        for i in range(num_channels):
            hue = i / num_channels
            saturation = 0.8
            value = 0.9
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append([r, g, b])

        return colors

    def _build_channel_graph(self, sample_attempts: int = 200, connect_threshold: float = 0.25) -> Dict:
        """构建通道之间的无向图

        对于每对通道，随机选择点尝试连接，如果连线不发生碰撞，则添加这条边。

        Args:
            sample_attempts: 每对通道之间的采样次数

        Returns:
            通道之间的无向图
        """
        init_pose = pp.get_pose(self.robot_setup.robot)
        self.robot_setup.set_base_pose(pp.Pose(point=[10, 10, 10], euler=[0, 0, 0]))

        # 初始化图结构
        graph = {i: {} for i in range(len(self.channel_info))}

        for i in range(len(self.channel_info)):
            closest_channels, closest_channel_idx = SceneParser.get_k_closest_channel(self.channel_info[i], self.channel_info, 4)

            for j_closest in range(len(closest_channels)):
                if i == closest_channel_idx[j_closest]:
                    continue
                else:
                    j = closest_channel_idx[j_closest]

                other_channel_idx = [k for k in range(len(self.channel_info)) if k != j and k != i]
                other_channel_bodies = []
                for k in other_channel_idx:
                    body_k = SceneParser.load_channel(self.channel_info[k])
                    other_channel_bodies.append(body_k)

                channel_i = self.channel_info[i]
                channel_j = self.channel_info[j]

                # 收集通道信息
                center_i = SceneParser.compute_channel_center(channel_i)
                type_i = channel_i["type"]

                center_j = SceneParser.compute_channel_center(channel_j)
                type_j = channel_j["type"]

                distance = np.linalg.norm(center_i - center_j)

                # 计算采样成功率
                success_count = 0

                for _ in range(sample_attempts):
                    point_i = SceneParser.sample_points_in_channel(type_i, channel_i, channel_i["thickness"], num_samples=1, ratio=0.5).flatten()
                    point_j = SceneParser.sample_points_in_channel(type_j, channel_j, channel_j["thickness"], num_samples=1, ratio=0.5).flatten()

                    # pp.draw_point(point_i, size=0.05)
                    # pp.draw_point(point_j, size=0.05)

                    result = p.rayTest(point_i, point_j)[0]
                    hit_body = result[0]
                    if hit_body == -1:
                        # pp.add_line(point_i, point_j, color=(0, 0, 1, 1))
                        success_count += 1
                    # else:
                    #     pp.add_line(point_i, point_j, color=(1, 0, 0, 1))

                for body in other_channel_bodies:
                    pp.remove_body(body)

                # 计算成功率作为边的权重
                success_rate = success_count / sample_attempts
                if success_rate > connect_threshold:
                    # 添加双向边
                    graph[i][j] = success_rate / distance
                    graph[j][i] = success_rate / distance

        self.robot_setup.set_base_pose(init_pose)

        return graph

    def _build_full_graph(self, start_xyz: np.ndarray, target_xyz: np.ndarray, sample_attempts: int = 200, connect_threshold: float = 0.25) -> Dict:
        """构建包含起点、终点和所有通道的完整图结构

        Args:
            start_xyz: 起点坐标
            target_xyz: 终点坐标
            sample_attempts: 每对节点之间的采样次数

        Returns:
            完整的图表示，起点为-1，终点为-2
        """
        # 获取已有的通道图
        channel_graph = self.channel_graph

        # 初始化完整图结构，包括起点(-1)和终点(-2)
        full_graph = {-1: {}, -2: {}}
        for i in range(len(self.channel_info)):
            full_graph[i] = {}
            # 将已有通道间连接复制到完整图中
            for j, weight in channel_graph[i].items():
                full_graph[i][j] = weight

        # 保存机器人初始位姿
        init_pose = pp.get_pose(self.robot_setup.robot)
        self.robot_setup.set_base_pose(pp.Pose(point=[10, 10, 10], euler=[0, 0, 0]))

        # 检测起点到各通道的连接
        for i in range(len(self.channel_info)):
            channel = self.channel_info[i]
            channel_center = SceneParser.compute_channel_center(channel)
            channel_type = channel["type"]
            distance = np.linalg.norm(channel_center - start_xyz)

            # 加载除当前通道外的其他通道，用于碰撞检测
            other_channel_idx = [k for k in range(len(self.channel_info)) if k != i]
            other_channel_bodies = []
            for k in other_channel_idx:
                body_k = SceneParser.load_channel(self.channel_info[k])
                other_channel_bodies.append(body_k)

            success_count = 0

            # 在通道上随机采样多个点，检查与起点的连接
            for _ in range(sample_attempts):
                channel_point = SceneParser.sample_points_in_channel(channel_type, channel, channel["thickness"], num_samples=1, ratio=0.5).flatten()

                # 检查起点到通道点的连接
                result = p.rayTest(start_xyz, channel_point)[0]
                hit_body = result[0]

                if hit_body == -1:
                    success_count += 1

            # 如果有成功连接，添加到图中
            success_rate = success_count / sample_attempts
            if success_rate > connect_threshold:
                full_graph[-1][i] = success_rate / distance
                full_graph[i][-1] = success_rate / distance

            # 清理其他通道的碰撞体
            for body in other_channel_bodies:
                pp.remove_body(body)

        # 检测终点到各通道的连接
        for i in range(len(self.channel_info)):
            channel = self.channel_info[i]
            channel_center = SceneParser.compute_channel_center(channel)
            channel_type = channel["type"]
            distance = np.linalg.norm(channel_center - target_xyz)

            # 加载除当前通道外的其他通道，用于碰撞检测
            other_channel_idx = [k for k in range(len(self.channel_info)) if k != i]
            other_channel_bodies = []
            for k in other_channel_idx:
                body_k = SceneParser.load_channel(self.channel_info[k])
                other_channel_bodies.append(body_k)

            success_count = 0

            # 在通道上随机采样多个点，检查与终点的连接
            for _ in range(sample_attempts):
                channel_point = SceneParser.sample_points_in_channel(channel_type, channel, channel["thickness"], num_samples=1, ratio=0.5).flatten()

                # 检查通道点到终点的连接
                result = p.rayTest(channel_point, target_xyz)[0]
                hit_body = result[0]

                if hit_body == -1:
                    success_count += 1

            # 如果有成功连接，添加到图中
            success_rate = success_count / sample_attempts
            if success_rate > connect_threshold:
                full_graph[-2][i] = success_rate / distance
                full_graph[i][-2] = success_rate / distance

            # 清理其他通道的碰撞体
            for body in other_channel_bodies:
                pp.remove_body(body)

        # 恢复机器人位姿
        self.robot_setup.set_base_pose(init_pose)

        return full_graph

    def _find_all_paths(self, graph: Dict, start_node: int, target_node: int, max_depth: int = 3, timeout: float = 5.0) -> List[List[int]]:
        """使用BFS找到图中所有低于给定深度的可行路径

        Args:
            graph: 图结构
            start_node: 起点节点索引
            target_node: 终点节点索引
            max_depth: 最大搜索深度
            timeout: 超时时间(秒)

        Returns:
            所有可行路径列表
        """
        all_paths = []
        start_time = time.time()

        # 使用队列进行BFS
        # 每个元素包含: (当前节点, 当前路径, 当前深度)
        queue = deque([(start_node, [start_node], 0)])

        while queue:
            current, path, depth = queue.popleft()

            # 达到目标节点
            if current == target_node:
                all_paths.append(path)
                continue

            # 如果已达到最大深度，不再进一步扩展
            if depth >= max_depth:
                continue

            # 扩展所有邻居节点
            for neighbor, _ in graph[current].items():
                # 避免循环
                if neighbor in path:
                    continue

                # 将邻居添加到队列中，深度加1
                new_path = path + [neighbor]
                queue.append((neighbor, new_path, depth + 1))

        return all_paths

    def _calculate_path_priorities(self, paths: List[List[int]], graph: Dict) -> List[tuple]:
        """计算每条路径的优先级，并按优先级排序

        Args:
            paths: 路径列表
            graph: 图结构

        Returns:
            排序后的 (path, priority) 元组列表
        """
        path_priorities = []

        for path in paths:
            # 跳过空路径
            if len(path) <= 1:
                continue

            # 计算路径优先级
            priority = 0.0

            # 累加各边的贡献
            for i in range(len(path) - 1):
                node_from = path[i]
                node_to = path[i + 1]

                # 获取边权重
                edge_weight = graph[node_from][node_to]

                # 计算节点优先级
                # 特殊节点（起点/终点）的优先级为1.0
                node_priority = 1.0
                if node_from >= 0:  # 通道节点
                    node_priority = self.channel_info[node_from].get("priority", 1.0)

                # 累加当前边的贡献：节点优先级 + 边权重
                priority += node_priority + edge_weight

            # 考虑最后一个节点的优先级（如果不是终点）
            if path[-1] >= 0:
                priority += self.channel_info[path[-1]].get("priority", 1.0)

            norm_priority = priority / (len(path) ** self.gamma)

            path_priorities.append((path, norm_priority))

        # 按优先级降序排序
        path_priorities.sort(key=lambda x: x[1], reverse=True)

        return path_priorities

    def plot_graph(self, graph: Dict, highlight_path: List[int] = None, start_point: np.ndarray = None, target_point: np.ndarray = None):
        """可视化连接图结构

        Args:
            graph: 图结构
            highlight_path: 要高亮显示的路径
            start_point: 起点坐标
            target_point: 终点坐标
        """
        # 移除所有调试线
        pp.remove_all_debug()

        # 存储节点位置
        node_positions = {}
        node_bodies = {}

        # 绘制通道节点
        for i in range(len(self.channel_info)):
            channel = self.channel_info[i]
            center = SceneParser.compute_channel_center(channel)

            # 将通道中心作为节点位置
            node_positions[i] = center

            sphere = pp.create_sphere(0.02)
            pp.set_point(sphere, center)

            # 绘制通道
            channel_body = SceneParser.load_channel(channel)
            pp.set_color(channel_body, self.channel_colors[i] + [0.3])  # 使用通道色彩，半透明
            node_bodies[i] = channel_body

            # 绘制节点编号
            pp.add_text(str(i), center + np.array([0, 0, 0.1]))

        # 检查图中是否有起点(-1)和终点(-2)
        if -1 in graph:
            # 找出起点连接的第一个通道，用于确定起点位置
            connected_channels = list(graph[-1].keys())
            if connected_channels:
                # 计算起点位置：使用所有连接通道的平均位置，但调整Z坐标为较低值
                connected_positions = [node_positions[ch] for ch in connected_channels if ch in node_positions]
                if connected_positions:
                    if start_point is not None:
                        start_pos = start_point
                    else:
                        avg_pos = sum(connected_positions) / len(connected_positions)
                        start_pos = np.array([avg_pos[0], avg_pos[1], min(p[2] for p in connected_positions) - 0.2])

                    # 存储起点位置
                    node_positions[-1] = start_pos

                    # 绘制起点标记
                    start_marker = pp.create_sphere(0.05, color=(0, 1, 0, 1))  # 绿色
                    pp.set_pose(start_marker, pp.Pose(point=start_pos))
                    node_bodies[-1] = start_marker

                    # 添加标签
                    pp.add_text("START", start_pos + np.array([0, 0, 0.1]))

        if -2 in graph:
            # 找出终点连接的第一个通道，用于确定终点位置
            connected_channels = list(graph[-2].keys())
            if connected_channels:
                # 计算终点位置：使用所有连接通道的平均位置，但调整Z坐标为较高值
                connected_positions = [node_positions[ch] for ch in connected_channels if ch in node_positions]
                if connected_positions:
                    if target_point is not None:
                        target_pos = target_point
                    else:
                        avg_pos = sum(connected_positions) / len(connected_positions)
                        target_pos = np.array([avg_pos[0], avg_pos[1], max(p[2] for p in connected_positions) + 0.2])

                    # 存储终点位置
                    node_positions[-2] = target_pos

                    # 绘制终点标记
                    target_marker = pp.create_sphere(0.05, color=(1, 0, 0, 1))  # 红色
                    pp.set_pose(target_marker, pp.Pose(point=target_pos))
                    node_bodies[-2] = target_marker

                    # 添加标签
                    pp.add_text("TARGET", target_pos + np.array([0, 0, 0.1]))

        # 收集所有边的权重，用于归一化
        all_weights = []
        edge_info = []  # 用于存储边的信息：(node_from, node_to, weight)
        for node_from, neighbors in graph.items():
            for node_to, weight in neighbors.items():
                if node_from in node_positions and node_to in node_positions:
                    all_weights.append(weight)
                    edge_info.append((node_from, node_to, weight))

        # 如果没有边，无需继续
        if not all_weights:
            return

        # 按照权重从大到小排序边的信息
        edge_info.sort(key=lambda x: x[2], reverse=True)

        # 创建从边到归一化权重的映射
        edge_to_normalized_weight = {}
        total_edges = len(edge_info)

        # 将排序后的边均匀映射到[1,0]范围
        for i, (node_from, node_to, weight) in enumerate(edge_info):
            # 排名第一的边映射到1，排名最后的边映射到0
            normalized_weight = 1.0 - (i / (total_edges - 1)) if total_edges > 1 else 0.5
            edge_to_normalized_weight[(node_from, node_to)] = normalized_weight

        # 绘制图中的边
        for node_from, neighbors in graph.items():
            # 跳过没有位置信息的节点
            if node_from not in node_positions:
                continue

            pos_from = node_positions[node_from]

            for node_to, weight in neighbors.items():
                # 跳过没有位置信息的节点
                if node_to not in node_positions:
                    continue

                pos_to = node_positions[node_to]

                # 获取预先计算的归一化权重
                normalized_weight = edge_to_normalized_weight.get((node_from, node_to), 0.5)

                # 从绿色(0,1,0)到蓝色(0,0,1)的颜色映射
                # 权重越高，越偏向绿色；权重越低，越偏向蓝色
                edge_color = [0.0, normalized_weight, 1.0 - normalized_weight, 1.0]  # RGBA格式：[R, G, B, A]

                # 绘制线段
                pp.add_line(pos_from, pos_to, color=edge_color, width=2)

                # 在边的中间显示权重
                mid_point = (pos_from + pos_to) / 2
                weight_text = f"{weight:.2f}"
                pp.add_text(weight_text, mid_point)

        # 高亮显示特定路径（如果提供）
        if highlight_path and len(highlight_path) > 1:
            for i in range(len(highlight_path) - 1):
                node_from = highlight_path[i]
                node_to = highlight_path[i + 1]

                # 跳过没有位置信息的节点
                if node_from not in node_positions or node_to not in node_positions:
                    continue

                pos_from = node_positions[node_from]
                pos_to = node_positions[node_to]

                # 使用醒目的黄色高亮显示
                pp.add_line(pos_from, pos_to, color=[1.0, 0.0, 0.0, 1.0], width=10)

    def plan(self, start_xyz: np.ndarray, target_xyz: np.ndarray, verbose: bool = False, verbose_level: int = 1) -> List[int]:
        """执行拓扑规划

        Args:
            start_xyz: 起点坐标
            target_xyz: 终点坐标
            verbose: 是否打印详细信息
            verbose_level: 打印详细信息级别

        Returns:
            最优通道路径
        """
        # 1. 建立完整的图
        self.full_graph = self._build_full_graph(start_xyz, target_xyz, sample_attempts=self.sample_attempts, connect_threshold=self.connect_threshold)

        # 2. 计算所有可行路径，并按照优先级排序
        self.all_paths = self._find_all_paths(self.full_graph, start_node=-1, target_node=-2)

        # 计算每条路径的优先级分数并排序
        paths_with_priority = self._calculate_path_priorities(self.all_paths, self.full_graph)

        # 打印排序后的路径
        if verbose:
            with printer.indented(verbose_level):
                printer.info("找到的路径（按优先级排序）:")
            with printer.indented(verbose_level + 1):
                for idx, (path, priority) in enumerate(paths_with_priority):
                    path_str = " -> ".join([str(node) if node >= 0 else ("起点" if node == -1 else "终点") for node in path])
                    printer.info(f"路径 {idx+1}: {path_str}, 优先级: {priority:.4f}")

        # 选择最优路径
        if paths_with_priority:
            best_path = paths_with_priority[0][0]
            return best_path
        else:
            if verbose:
                with printer.indented(verbose_level):
                    printer.info("未找到可行路径!")
            return []


if __name__ == "__main__":
    # 示例用法
    init_pb()

    scene_file = os.path.join(HERE, "model", "scenes", "shelf_1", "task_1.yml")
    # scene_file = os.path.join(HERE, "model", "scenes", "cuboid_1", "task_1.yml")
    scene_parser = SceneParser(scene_file)
    bodies = scene_parser.create_elements()

    start_q = np.array(scene_parser.get_robot_start_pose())
    target_q = np.array(scene_parser.get_robot_target_pose())
    pose_2d = scene_parser.get_robot_pose_2d(output_type="array")
    channel_info = scene_parser.get_channel_info()
    grasp_offset = scene_parser.get_robot_grasp_offset()

    rb = scene_parser.create_robot("r0")

    grasped_element, grasped_attachment = scene_parser.create_attachment(rb)
    rb.update_attachments([grasped_attachment])

    rb.set_joint_positions(rb.arm_joints, start_q)
    start_point = np.array(pp.get_point(grasped_element))

    rb.set_joint_positions(rb.arm_joints, target_q)
    target_point = np.array(pp.get_point(grasped_element))

    # 创建TAMPOR规划器并进行规划
    solver = TopologyPlanner(rb, channel_info, bodies, eval_max_attempts=1000, sample_attempts=100, connect_threshold=0.25)
    best_path = solver.plan(start_point, target_point, verbose=True, verbose_level=0)
    rb.set_base_pose_2d(10, 10, 0)
    solver.plot_graph(solver.full_graph, highlight_path=best_path, start_point=start_point, target_point=target_point)

    pp.wait_for_user()
