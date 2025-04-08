import colorsys
import itertools
import os
import random
import sys
import time
import warnings
from collections import deque
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pybullet as p
import pybullet_planning as pp
from pybullet_planning import Attachment
from scipy.spatial.transform import Rotation

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from model.scene_parse import SceneParser
from multi_tangent.collision import create_collision_bodies
from robot.robot_setup import RobotSetup
from utils.collision import init_pb
from utils.params import URDF_PATH


class TrajectoryTAMPORSolver:
    """TAMPOR (Topology-Aware Motion Planning with Obstacle Rearrangement)求解器

    将TopologyPlanner和Planner整合到一个类中，提供统一的规划接口。
    """

    def __init__(self, robot_setup: RobotSetup, channel_info: List[Dict], grasp_offset: List[float], object_size: List[float] = [1.0, 0.02], obstacle_size: List[float] = [1.0, 0.02], eval_max_attempts: int = 50000) -> None:
        """初始化TAMPOR求解器

        Args:
            robot_setup: 机器人设置对象
            channel_info: 通道信息列表
            grasp_offset: 抓取偏移量
            object_size: 物体尺寸 [长度, 宽度]
            obstacle_size: 障碍物尺寸 [长度, 宽度]
            eval_max_attempts: 评估通道优先级的最大尝试次数
        """
        self.robot_setup = robot_setup
        self.channel_info = channel_info
        self.grasp_offset = grasp_offset
        self.object_size = object_size
        self.obstacle_size = obstacle_size
        self.eval_max_attempts = eval_max_attempts

        # 创建内部规划器实例，但不立即初始化
        self.topology_planner = None
        self.path_planner = None

    def _init_topology_planner(self, bodies: List[int]) -> None:
        """初始化拓扑规划器

        Args:
            bodies: 碰撞体列表
        """
        self.topology_planner = self._TopologyPlanner(self.robot_setup, self.channel_info, bodies, self.object_size, self.eval_max_attempts)

    def _init_path_planner(self, collision_fn: Callable[[np.ndarray], bool]) -> None:
        """初始化路径规划器

        Args:
            collision_fn: 碰撞检测函数
        """
        self.path_planner = self._Planner(self.robot_setup, self.channel_info, collision_fn, self.object_size, self.obstacle_size)

    class _TopologyPlanner:
        """内部拓扑规划器类，用于高层路径规划

        基于通道信息进行拓扑规划，找出最优的通道通过顺序。
        """

        def __init__(self, robot_setup: RobotSetup, channel_info: List[Dict], bodies: List[int], object_size: List[float] = [1.0, 0.02], eval_max_attempts: int = 50000):
            """初始化拓扑规划器

            Args:
                robot_setup: 机器人设置
                channel_info: 通道信息列表
                bodies: 碰撞体列表
                object_size: 物体尺寸
                eval_max_attempts: 评估通道的最大尝试次数
            """
            self.robot_setup = robot_setup
            self.channel_info = channel_info
            self.bodies = bodies
            self.object_size = object_size
            self.eval_max_attempts = eval_max_attempts

            with pp.LockRenderer():
                self.channel_info = self._evaluate_channel_priority()
                self.channel_colors = self._generate_channel_colors()
                self.channel_graph = self._build_channel_graph()  # {channel_idx: {neighbor_idx: weight}}

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
                channel_size = channel["size"]
                channel_thickness = channel["thickness"]
                channel_type = channel["type"]
                channel_center = np.array(channel["center"])
                channel_direction = np.array(channel["direction"])

                if channel_type == "rectangle":
                    length, width = channel_size
                    if width >= self.object_size[0] and length >= self.object_size[0]:
                        angle_constraint = False
                        passability = 1.0
                    elif min(width, length) < self.object_size[1]:
                        angle_constraint = True
                        passability = float("inf")  # impossible
                    else:
                        angle_constraint = True
                        passability = width * length

                channel["angle_constraint"] = angle_constraint
                channel["passability"] = passability

            # 计算总优先级
            for channel in self.channel_info:
                channel["priority"] = 2.0 * channel["reachability_weight"] + channel["passability"]

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

        def _build_channel_graph(self, sample_attempts: int = 200) -> Dict:
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

            # 双层循环遍历所有通道对
            for i in range(len(self.channel_info)):
                for j in range(i + 1, len(self.channel_info)):
                    if i == j:
                        continue

                    other_channel_idx = [k for k in range(len(self.channel_info)) if k != i and k != j]
                    other_channel_bodies = []
                    for k in other_channel_idx:
                        body_k = SceneParser.load_channel(self.channel_info[k])
                        other_channel_bodies.append(body_k)

                    channel_i = self.channel_info[i]
                    channel_j = self.channel_info[j]

                    # 收集通道信息
                    center_i = np.array(channel_i["center"])
                    dir_i = np.array(channel_i["direction"])
                    size_i = channel_i["size"]
                    type_i = channel_i["type"]

                    center_j = np.array(channel_j["center"])
                    dir_j = np.array(channel_j["direction"])
                    size_j = channel_j["size"]
                    type_j = channel_j["type"]

                    distance = np.linalg.norm(center_i - center_j)

                    # 计算采样成功率
                    success_count = 0

                    for _ in range(sample_attempts):
                        if type_i == "rectangle":
                            length_i, width_i = size_i
                            z_axis = dir_i / np.linalg.norm(dir_i)
                            temp_x = np.array([1, 0, 0])
                            if np.abs(np.dot(temp_x, z_axis)) > 0.9:
                                temp_x = np.array([0, 1, 0])
                            y_axis = np.cross(z_axis, temp_x)
                            y_axis = y_axis / np.linalg.norm(y_axis)
                            x_axis = np.cross(y_axis, z_axis)
                            x_axis = x_axis / np.linalg.norm(x_axis)
                            dx = np.random.uniform(-length_i / 2 + 0.01, length_i / 2 - 0.01)
                            dy = np.random.uniform(-width_i / 2 + 0.01, width_i / 2 - 0.01)
                            point_i = center_i + dx * x_axis + dy * y_axis

                        if type_j == "rectangle":
                            length_j, width_j = size_j
                            z_axis = dir_j / np.linalg.norm(dir_j)
                            temp_x = np.array([1, 0, 0])
                            if np.abs(np.dot(temp_x, z_axis)) > 0.9:
                                temp_x = np.array([0, 1, 0])
                            y_axis = np.cross(z_axis, temp_x)
                            y_axis = y_axis / np.linalg.norm(y_axis)
                            x_axis = np.cross(y_axis, z_axis)
                            x_axis = x_axis / np.linalg.norm(x_axis)
                            dx = np.random.uniform(-length_j / 2 + 0.01, length_j / 2 - 0.01)
                            dy = np.random.uniform(-width_j / 2 + 0.01, width_j / 2 - 0.01)
                            point_j = center_j + dx * x_axis + dy * y_axis

                        result = p.rayTest(point_i, point_j)[0]
                        hit_body = result[0]
                        if hit_body == -1:
                            success_count += 1

                    for body in other_channel_bodies:
                        pp.remove_body(body)

                    # 计算成功率作为边的权重
                    if success_count > 0:
                        success_rate = success_count / sample_attempts
                        # 添加双向边
                        graph[i][j] = success_rate / distance
                        graph[j][i] = success_rate / distance

            self.robot_setup.set_base_pose(init_pose)

            return graph

        def _build_full_graph(self, start_xyz: np.ndarray, target_xyz: np.ndarray, sample_attempts: int = 200) -> Dict:
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
                channel_center = np.array(channel["center"])
                channel_direction = np.array(channel["direction"])
                channel_size = channel["size"]
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
                    if channel_type == "rectangle":
                        length, width = channel_size
                        z_axis = channel_direction / np.linalg.norm(channel_direction)

                        temp_x = np.array([1, 0, 0])
                        if np.abs(np.dot(temp_x, z_axis)) > 0.9:
                            temp_x = np.array([0, 1, 0])

                        y_axis = np.cross(z_axis, temp_x)
                        y_axis = y_axis / np.linalg.norm(y_axis)
                        x_axis = np.cross(y_axis, z_axis)
                        x_axis = x_axis / np.linalg.norm(x_axis)

                        # 在通道上随机采样一个点，略微缩小范围以避免边缘问题
                        dx = np.random.uniform(-length / 2 + 0.01, length / 2 - 0.01)
                        dy = np.random.uniform(-width / 2 + 0.01, width / 2 - 0.01)
                        channel_point = channel_center + dx * x_axis + dy * y_axis

                        # 检查起点到通道点的连接
                        result = p.rayTest(start_xyz, channel_point)[0]
                        hit_body = result[0]

                        if hit_body == -1:
                            success_count += 1

                # 如果有成功连接，添加到图中
                if success_count > 0:
                    success_rate = success_count / sample_attempts
                    full_graph[-1][i] = success_rate / distance
                    full_graph[i][-1] = success_rate / distance

                # 清理其他通道的碰撞体
                for body in other_channel_bodies:
                    pp.remove_body(body)

            # 检测终点到各通道的连接
            for i in range(len(self.channel_info)):
                channel = self.channel_info[i]
                channel_center = np.array(channel["center"])
                channel_direction = np.array(channel["direction"])
                channel_size = channel["size"]
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
                    if channel_type == "rectangle":
                        length, width = channel_size
                        z_axis = channel_direction / np.linalg.norm(channel_direction)

                        temp_x = np.array([1, 0, 0])
                        if np.abs(np.dot(temp_x, z_axis)) > 0.9:
                            temp_x = np.array([0, 1, 0])

                        y_axis = np.cross(z_axis, temp_x)
                        y_axis = y_axis / np.linalg.norm(y_axis)
                        x_axis = np.cross(y_axis, z_axis)
                        x_axis = x_axis / np.linalg.norm(x_axis)

                        # 在通道上随机采样一个点，略微缩小范围以避免边缘问题
                        dx = np.random.uniform(-length / 2 + 0.01, length / 2 - 0.01)
                        dy = np.random.uniform(-width / 2 + 0.01, width / 2 - 0.01)
                        channel_point = channel_center + dx * x_axis + dy * y_axis

                        # 检查通道点到终点的连接
                        result = p.rayTest(channel_point, target_xyz)[0]
                        hit_body = result[0]

                        if hit_body == -1:
                            success_count += 1

                # 如果有成功连接，添加到图中
                if success_count > 0:
                    success_rate = success_count / sample_attempts
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

                    # 累加当前边的贡献：节点优先级 * 边权重
                    priority += node_priority * edge_weight

                # 考虑最后一个节点的优先级（如果不是终点）
                if path[-1] >= 0:
                    priority += self.channel_info[path[-1]].get("priority", 1.0)

                norm_priority = priority / (len(path) ** 3.0)

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
                center = np.array(channel["center"])

                # 将通道中心作为节点位置
                node_positions[i] = center

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

                    # 边的颜色基于权重
                    # 权重越高（连接成功率越高），颜色越绿
                    edge_color = [1.0 - weight, weight, 0.0, 1.0]  # RGBA, 半透明

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
                    pp.add_line(pos_from, pos_to, color=[1.0, 1.0, 0.0, 1.0], width=6)

        def plan(self, start_xyz: np.ndarray, target_xyz: np.ndarray) -> List[int]:
            """执行拓扑规划

            Args:
                start_xyz: 起点坐标
                target_xyz: 终点坐标

            Returns:
                最优通道路径
            """
            # 1. 建立完整的图
            with pp.LockRenderer():
                self.full_graph = self._build_full_graph(start_xyz, target_xyz)

            # 2. 计算所有可行路径，并按照优先级排序
            self.all_paths = self._find_all_paths(self.full_graph, start_node=-1, target_node=-2)

            # 计算每条路径的优先级分数并排序
            paths_with_priority = self._calculate_path_priorities(self.all_paths, self.full_graph)

            # 打印排序后的路径
            print("\n找到的路径（按优先级排序）:")
            for idx, (path, priority) in enumerate(paths_with_priority):
                path_str = " -> ".join([str(node) if node >= 0 else ("起点" if node == -1 else "终点") for node in path])
                print(f"路径 {idx+1}: {path_str}, 优先级: {priority:.4f}")

            # 选择最优路径
            if paths_with_priority:
                best_path = paths_with_priority[0][0]
                return best_path
            else:
                print("未找到可行路径!")
                return []

    class _Planner:
        """内部路径规划器类，用于低层轨迹规划

        通过采样和排序关键帧，规划通过指定通道序列的路径。
        """

        def __init__(self, robot_setup: RobotSetup, channel_info: List[Dict], collision_fn: Callable[[np.ndarray], bool], object_size: List[float] = [1.0, 0.02], obstacle_size: List[float] = [1.0, 0.02]):
            """初始化路径规划器

            Args:
                robot_setup: 机器人设置
                channel_info: 通道信息列表
                collision_fn: 碰撞检测函数
                object_size: 物体尺寸
                obstacle_size: 障碍物尺寸
            """
            self.robot_setup = robot_setup
            self.channel_info = channel_info
            self.collision_fn = collision_fn
            self.object_size = object_size
            self.obstacle_size = obstacle_size

        def _generate_key_frames(self, channel_id: int, num_points: int = 10, max_attempts: int = 1000) -> List[np.ndarray]:
            """在指定通道内生成关键帧

            Args:
                channel_id: 通道ID
                num_points: 生成的关键帧数量
                max_attempts: 最大尝试次数

            Returns:
                关键帧列表，如果生成失败则返回None
            """
            key_frames = []
            channel = self.channel_info[channel_id]
            center = np.array(channel["center"])
            direction = np.array(channel["direction"])
            size = channel["size"]
            length, width = size
            thickness = channel["thickness"]

            z_axis = direction / np.linalg.norm(direction)
            temp_x = np.array([1, 0, 0])
            if np.abs(np.dot(temp_x, z_axis)) > 0.9:
                temp_x = np.array([0, 1, 0])
            y_axis = np.cross(z_axis, temp_x)
            y_axis = y_axis / np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)

            # 生成关键帧
            attempt = 0
            while attempt < max_attempts:
                attempt += 1
                dx = np.random.uniform(-length / 8, length / 8)
                dy = np.random.uniform(-width / 8, width / 8)
                dz = np.random.uniform(-thickness * 2, thickness * 2)
                roll = np.random.uniform(-np.pi, np.pi)
                pitch = np.random.uniform(-np.pi, np.pi)
                yaw = np.random.uniform(-np.pi, np.pi)
                point = center + dx * x_axis + dy * y_axis + dz * z_axis
                element_pose = pp.Pose(point=point, euler=pp.Euler(roll, pitch, yaw))  # world_from_element

                # 获取运动学解
                joint_val = self.robot_setup.get_relative_ik_solution(element_pose, q_init=np.random.uniform(-np.pi, np.pi, size=6).tolist())
                if joint_val is not None:
                    if not self.collision_fn(joint_val):
                        key_frames.append(joint_val)
                if len(key_frames) >= num_points:
                    break

            if len(key_frames) == 0:
                return None

            return key_frames

        def _sort_key_frames(self, channel_id: int, key_frames: List[np.ndarray]) -> List[np.ndarray]:
            """根据channel信息和机器人配置对key frames进行排序

            Args:
                channel_id: 通道ID
                key_frames: 生成的关键帧列表

            Returns:
                排序后的关键帧列表
            """
            if not key_frames:
                return []

            channel = self.channel_info[channel_id]
            channel_center = np.array(channel["center"])
            channel_direction = np.array(channel["direction"])
            channel_size = channel["size"]
            channel_thickness = channel["thickness"]

            # 计算每个关键帧的评分
            frame_scores = []
            for frame in key_frames:
                # 设置机器人关节角度
                self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, frame)

                # 获取末端位置
                tool_pose = pp.get_link_pose(self.robot_setup.robot, self.robot_setup.tool_link)
                tool_point = np.array(tool_pose[0])

                # 计算与通道中心的距离
                distance_to_center = np.linalg.norm(tool_point - channel_center)

                # 计算与通道方向的夹角
                tool_direction = np.array(pp.tform_from_pose(tool_pose))
                tool_direction = Rotation.from_matrix(tool_direction[:3, :3]).as_rotvec()
                alignment = np.abs(np.dot(tool_direction, channel_direction)) / (np.linalg.norm(tool_direction) * np.linalg.norm(channel_direction))

                # 检查碰撞情况
                collision_score = 0 if self.collision_fn(frame) else 1

                # 计算最终评分 (距离越近、越对齐方向、无碰撞越好)
                score = collision_score * (alignment + 1.0 / (distance_to_center + 0.1))

                frame_scores.append((frame, score))

            # 按评分降序排序
            frame_scores.sort(key=lambda x: x[1], reverse=True)

            return [frame for frame, _ in frame_scores]

        def _stratified_sampling(self, strata, sample_sizes=None, sample_fraction=None):
            """分层随机抽样的迭代器

            Args:
                strata: 包含多个子列表的列表，每个子列表是一个层
                sample_sizes: 每层要抽取的样本数量列表
                sample_fraction: 每层要抽取的样本比例

            Yields:
                每次产生一个可能的抽样结果列表
            """
            # 计算每层的抽样数量
            if sample_sizes is None and sample_fraction is None:
                sample_sizes = [1] * len(strata)

            if sample_fraction is not None:
                sample_sizes = [max(1, int(len(s) * sample_fraction)) for s in strata]

            # 对每层准备所有可能的组合
            all_combinations = []
            total_combinations = 1

            for i, stratum in enumerate(strata):
                # 确保不会抽取超过层中元素数量的样本
                size = min(sample_sizes[i], len(stratum))
                # 获取该层所有可能的组合
                stratum_combinations = list(itertools.combinations(stratum, size))
                all_combinations.append(stratum_combinations)
                total_combinations *= len(stratum_combinations)

            # 如果组合数量过大，发出警告
            if total_combinations > 10000:
                warnings.warn(f"将生成 {total_combinations} 种组合，这可能会消耗大量内存并且运行缓慢")

            # 生成所有可能的排列组合
            for combination in itertools.product(*all_combinations):
                # 将每层的组合展平成一个列表
                result = []
                for samples in combination:
                    result.extend(samples)
                yield result

        def plan(self, start_conf: np.ndarray, target_conf: np.ndarray, channel_path: List[int], max_time: float = 600.0, num_points: int = 20) -> List[np.ndarray]:
            """规划从起始配置到目标配置的路径

            Args:
                start_conf: 起始关节配置
                target_conf: 目标关节配置
                channel_path: 通道路径
                max_time: 最大规划时间(秒)
                num_points: 每个通道生成的关键帧数量

            Returns:
                规划路径，如果失败则返回None
            """
            print("Generating key frames...")

            key_frames_list = [[start_conf]]
            for channel_id in channel_path:
                if channel_id == -1 or channel_id == -2:
                    continue
                key_frames = self._generate_key_frames(channel_id, num_points=num_points)
                if key_frames is not None:
                    # 排序关键帧
                    # key_frames = self._sort_key_frames(channel_id, key_frames)
                    key_frames_list.append(key_frames)
            key_frames_list.append([target_conf])

            print("Generating all possible paths...")

            def get_sample_fn():
                lower, upper = pp.get_custom_limits(self.robot_setup.robot, self.robot_setup.arm_joints, circular_limits=pp.CIRCULAR_LIMITS)
                generator = pp.interval_generator(lower, upper)

                def fn():
                    sample = list(next(generator))
                    return tuple(sample)

                return fn

            sample_fn = get_sample_fn()
            resolutions = np.array([1.0 / 180.0 * np.pi for j in self.robot_setup.arm_joints])
            extend_fn = pp.get_extend_fn(self.robot_setup.robot, self.robot_setup.arm_joints, resolutions=resolutions)
            distance_fn = pp.get_distance_fn(self.robot_setup.robot, self.robot_setup.arm_joints)

            start_time = time.time()
            timeout = False
            for idx, temp_channel_path in enumerate(self._stratified_sampling(key_frames_list)):
                path = []
                success = True
                print(f"Generating {idx+1} / {len(list(self._stratified_sampling(key_frames_list)))} path...")
                for i in range(len(temp_channel_path) - 1):
                    print(f"    Generating {i+1} / {len(temp_channel_path) - 1} path...")
                    start_conf = temp_channel_path[i]
                    target_conf = temp_channel_path[i + 1]
                    temp_path = self.robot_setup.plan_manipulator_path(
                        start_conf, target_conf, self.robot_setup.attachments, [], collision_fn=self.collision_fn, sample_fn=sample_fn, extend_fn=extend_fn, distance_fn=distance_fn, max_time=15.0
                    )

                    current_time = time.time()
                    if current_time - start_time > max_time:
                        print(f"    Time out! {current_time - start_time:.2f} seconds")
                        success = False
                        timeout = True
                        break

                    if temp_path is not None:
                        path.append(temp_path)
                    else:
                        success = False
                        print(f"    Failed to generate path for {i+1} / {len(temp_channel_path) - 1} path...")
                        break

                if timeout:
                    return None

                if success:
                    return np.concatenate(path, axis=0)

            return None

    def plan(self, start_conf: np.ndarray, target_conf: np.ndarray, element_bodies: List[int], pose_2d: np.ndarray, grasp_attachment: Attachment) -> Dict:
        """执行完整的规划过程

        结合拓扑规划和路径规划，找到从起点到终点的最优路径。

        Args:
            start_conf: 起始关节配置
            target_conf: 目标关节配置
            element_bodies: 碰撞体列表
            pose_2d: 机器人2D姿态
            grasp_attachment: 抓取附件

        Returns:
            包含规划结果的字典 {"success": bool, "path": np.ndarray}
        """
        print("\n========== TAMPOR TRAJECTORY PLANNING ==========")

        # 保存当前机器人状态
        self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, start_conf)
        self.robot_setup.update_attachments([grasp_attachment])

        # 获取起点和终点位置
        start_point = np.array(pp.get_point(grasp_attachment.child))
        self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, target_conf)
        target_point = np.array(pp.get_point(grasp_attachment.child))

        # 恢复起始位置
        self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, start_conf)

        # 创建碰撞检测函数
        collision_fn = self.robot_setup.create_collision_fn(element_bodies)

        # 初始化规划器
        with pp.LockRenderer():
            print("TAMPOR: Initializing topology planner...")
            self._init_topology_planner(element_bodies)
            print("TAMPOR: Initializing path planner...")
            self._init_path_planner(collision_fn)

        # 1. 拓扑规划
        print("TAMPOR: Starting topology planning...")
        start_time = time.time()
        best_path = self.topology_planner.plan(start_point, target_point)
        topology_time = time.time() - start_time
        print(f"TAMPOR: Topology planning completed in {topology_time:.2f} seconds")

        if not best_path:
            print("TAMPOR: Failed to find a valid topology path")
            return {"success": False, "path": None}

        # 2. 路径规划
        print("TAMPOR: Starting path planning...")
        start_time = time.time()
        with pp.LockRenderer():
            path = self.path_planner.plan(start_conf, target_conf, best_path, num_points=100, max_time=1000)
        path_time = time.time() - start_time
        print(f"TAMPOR: Path planning completed in {path_time:.2f} seconds")

        if path is None:
            print("TAMPOR: Failed to find a valid trajectory path")
            return {"success": False, "path": None}

        print("\n========== TAMPOR PLANNING SUCCESSFUL ==========")
        print(f"TAMPOR: Total planning time: {topology_time + path_time:.2f} seconds")

        return {"success": True, "path": path}


if __name__ == "__main__":
    # 示例用法
    init_pb()

    scene_file = os.path.join(HERE, "model", "scenes", "cuboid_1", "task_1.yml")
    scene_parser = SceneParser(scene_file)
    scene_parser.load_scene()
    line_pts, radius_per_edge = scene_parser.get_element_info()
    bodies = create_collision_bodies(line_pts, radius_per_edge, viewer=True)

    start_q = np.array(scene_parser.get_robot_start_pose())
    target_q = np.array(scene_parser.get_robot_target_pose())
    pose_2d = scene_parser.get_robot_pose_2d(output_type="array")
    grasp_offset = scene_parser.get_robot_grasp_offset()
    channel_info = scene_parser.get_channel_info()

    rb = RobotSetup("rb")
    rb.set_joint_positions(rb.arm_joints, start_q)
    rb.set_base_pose_2d(pose_2d[0], pose_2d[1], pose_2d[2])

    line_pts_grasped = [np.array([0, 0, 0]), np.array([0, 0, 1])]
    grasped_element = create_collision_bodies(line_pts_grasped, [0.01], viewer=True)[0]
    pp.set_pose(grasped_element, pp.multiply(pp.get_link_pose(rb.robot, rb.tool_link), pp.Pose(point=grasp_offset, euler=pp.Euler(1.5708, 0, 0))))
    grasped_attachment = pp.create_attachment(rb.robot, rb.tool_link, grasped_element)
    rb.update_attachments([grasped_attachment])

    # 创建TAMPOR规划器并进行规划
    solver = TrajectoryTAMPORSolver(rb, channel_info, grasp_offset, eval_max_attempts=1000)
    plan_result = solver.plan(start_q, target_q, bodies, pose_2d, grasped_attachment)

    if plan_result["success"]:
        path = plan_result["path"]

        # 可视化路径
        slider = p.addUserDebugParameter("replay", 0, 1, 0)
        while True:
            slider_value = p.readUserDebugParameter(slider)
            time_idx = int(slider_value * (path.shape[0] - 1))
            joint_val = path[time_idx]
            rb.set_joint_positions(rb.arm_joints, joint_val)
            time.sleep(1.0 / 60)
    else:
        print("规划失败")
