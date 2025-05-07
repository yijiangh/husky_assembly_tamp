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


class Planner:
    """内部路径规划器类，用于低层轨迹规划

    通过采样和排序关键帧，规划通过指定通道序列的路径。
    """

    def __init__(self, robot_setup: RobotSetup, channel_info: List[Dict], grasp_pose, collision_fn: Callable[[np.ndarray], bool], verbose: bool = False):
        """初始化路径规划器

        Args:
            robot_setup: 机器人设置
            channel_info: 通道信息列表
            grasp_pose: 抓取位姿
            collision_fn: 碰撞检测函数
        """
        self.robot_setup = robot_setup
        self.channel_info = channel_info
        self.grasp_pose = grasp_pose
        self.collision_fn = collision_fn

        def plan_fn(start_conf: np.ndarray, target_conf: np.ndarray, max_time: float = 10.0, interpolate_num: int = 10000, enable_reset: bool = False, verbose: bool = False, verbose_level: int = 2):
            start_time = time.time()
            while time.time() - start_time < max_time:
                res, path = self.pb_ompl_interface.plan_start_goal(start_conf.tolist(), target_conf.tolist(), allowed_time=max_time - (time.time() - start_time), interpolate_num=interpolate_num, verbose=verbose)
                if res:
                    path_array = np.array(path)
                    if len(path_array) == 0:
                        if enable_reset:
                            self.setup_pb_ompl(verbose)
                        if verbose:
                            with printer.indented(verbose_level):
                                printer.warning("Planner returned an empty path")
                        continue
                    start_diff = np.linalg.norm(path_array[0] - start_conf)
                    goal_diff = np.linalg.norm(path_array[-1] - target_conf)
                    if start_diff > 1e-6:
                        if enable_reset:
                            self.setup_pb_ompl(verbose)
                        if verbose:
                            with printer.indented(verbose_level):
                                printer.warning(f"Path start point doesn't match input start point, difference: {start_diff}")
                        continue
                    if goal_diff > 1e-6:
                        if enable_reset:
                            self.setup_pb_ompl(verbose)
                        if verbose:
                            with printer.indented(verbose_level):
                                printer.warning(f"Path end point doesn't match input target point, difference: {goal_diff}")
                        continue
                    path_array = interpolate(path_array, max(interpolate_num, len(path_array)))
                    collision_free = True
                    for conf in path_array:
                        if self.collision_fn(conf):
                            collision_free = False
                            break
                    if collision_free:
                        return path_array
                    else:
                        if enable_reset:
                            self.setup_pb_ompl(verbose)
                        if verbose:
                            with printer.indented(verbose_level):
                                printer.warning("Path is not collision free, trying again...")
            return None

        self.planner = plan_fn

    def setup_pb_ompl(self, verbose: bool = False):
        verbose = False  # TODO: 当可以调整plan的缩进时，这里可以去掉
        self.robot = PbOMPLRobotWrapper(self.robot_setup.robot, self.robot_setup.arm_joints)
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, [])
        self.pb_ompl_interface.set_planner("RRTConnect")
        self.pb_ompl_interface.si.setStateValidityCheckingResolution(0.0005)
        if hasattr(self.pb_ompl_interface.space, "setLongestValidSegmentFraction"):
            self.pb_ompl_interface.space.setLongestValidSegmentFraction(0.0005)
        if hasattr(self.pb_ompl_interface.planner, "setRange"):
            self.pb_ompl_interface.planner.setRange(0.01)

        if not verbose:
            ou.setLogLevel(ou.LOG_ERROR)
        else:
            ou.setLogLevel(ou.LOG_DEBUG)

        def custom_is_state_valid(state):
            state_arr = np.array([state[i] for i in range(self.robot.num_dim)])
            return not self.collision_fn(state_arr)

        self.pb_ompl_interface.ss.setStateValidityChecker(ob.StateValidityCheckerFn(custom_is_state_valid))

    def _generate_key_frames(self, channel_id: int, grasps: List = None, grasp_weights: List[float] = None, num_points: int = 10, max_attempts: int = 1000) -> List[np.ndarray]:
        """在指定通道内生成关键帧

        Args:
            channel_id: 通道ID
            grasps: 可能的抓取位姿列表
            num_points: 生成的关键帧数量
            max_attempts: 最大尝试次数

        Returns:
            关键帧列表，如果生成失败则返回None
        """
        key_frames = []
        channel = self.channel_info[channel_id]
        # 生成关键帧
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            pose = SceneParser.sample_pose_in_channel(channel["type"], channel, channel["thickness"], num_samples=1, ratio=0.1).flatten()
            
            # 设置世界坐标系下元素的位姿
            world_from_element = pp.Pose(point=pose[:3].tolist(), euler=pp.Euler(pose[3], pose[4], pose[5]))
            
            # 如果提供了grasps，随机选择一个用于计算
            if grasps and len(grasps) > 0:
                if grasp_weights and len(grasp_weights) > 0:
                    grasp_pose = random.choices(grasps, weights=grasp_weights, k=1)[0]
                else:
                    grasp_pose = random.choice(grasps)
                world_from_tool = pp.multiply(world_from_element, pp.invert(grasp_pose))
            else:
                world_from_tool = pp.multiply(world_from_element, pp.invert(self.grasp_pose))

            # 获取运动学解
            joint_val = self.robot_setup.get_relative_ik_solution(world_from_tool, q_init=np.random.uniform(-np.pi, np.pi, size=6).tolist())
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
            
            # 计算两个方向向量的点积，得到cos(theta)
            cos_theta = np.dot(tool_direction, channel_direction) / (np.linalg.norm(tool_direction) * np.linalg.norm(channel_direction))
            # 确保我们处理的是锐角(取绝对值，处理可能的反向情况)
            cos_theta = abs(cos_theta)
            # 计算与90度的接近程度：sin(theta) = sqrt(1 - cos^2(theta))
            # 越接近90度，sin_theta越接近1
            sin_theta = np.sqrt(1 - cos_theta * cos_theta)
            
            # 使用sin_theta作为评分依据，越接近90度(sin_theta接近1)，评分越高
            angle_score = sin_theta

            # 检查碰撞情况
            collision_score = 0 if self.collision_fn(frame) else 1

            # 计算最终评分 (角度越接近90度、无碰撞越好)
            score = collision_score * (angle_score + 1.0)

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

    def plan(
        self,
        start_conf: np.ndarray,
        target_conf: np.ndarray,
        channel_path: List[int],
        grasps: List = None,
        grasp_weights: List[float] = None,
        max_time: float = 600.0,
        init_step_max_time: float = 200.0,
        single_plan_max_time: float = 15.0,
        num_points: int = 20,
        verbose: bool = False,
        verbose_level: int = 1,
    ) -> List[np.ndarray]:
        """规划从起始配置到目标配置的路径

        Args:
            start_conf: 起始关节配置
            target_conf: 目标关节配置
            channel_path: 通道路径
            grasps: 可能的抓取位姿列表
            max_time: 最大规划时间(秒)
            single_plan_max_time: 每个步骤的最大规划时间(秒)
            grow_max_time: 树生长的最大时间(秒)
            num_points: 每个通道生成的关键帧数量
            grow_tree_max_nodes: 每个树生长的最大节点数量
            verbose: 是否打印详细信息
            verbose_level: 打印详细信息级别

        Returns:
            规划路径，如果失败则返回None
        """

        # **************************************************************************
        # 1. 尝试直接使用pb_ompl规划
        # **************************************************************************
        self.setup_pb_ompl(verbose)
        path = self.planner(start_conf, target_conf, max_time=init_step_max_time, enable_reset=True, verbose=verbose, verbose_level=verbose_level)
        if path is not None:
            return path
        with printer.indented(verbose_level):
            printer.warning("Warning: No path found in the first step, trying to find a path in the second step...")

        init_step_time = init_step_max_time

        # **************************************************************************
        # 2. 生成关键帧
        # **************************************************************************

        if verbose:
            with printer.indented(verbose_level):
                printer.info("Generating key frames...")
        start_time = time.time()
        key_frames_list = [[start_conf]]
        for channel_id in channel_path:
            if channel_id == -1 or channel_id == -2:
                continue
            if verbose:
                with printer.indented(verbose_level + 1):
                    printer.info(f"Generating key frames for channel {channel_id}...")
            key_frames = self._generate_key_frames(channel_id, grasps=grasps, grasp_weights=grasp_weights, num_points=num_points)
            if key_frames is not None:
                # 排序关键帧
                # key_frames = self._sort_key_frames(channel_id, key_frames)
                key_frames_list.append(key_frames)
        key_frames_list.append([target_conf])
        key_frame_time = time.time() - start_time
        if verbose:
            with printer.indented(verbose_level):
                printer.info(f"Key frames generated in {key_frame_time:.2f} seconds")

        key_frames_idx = []
        for i in range(len(key_frames_list)):
            key_frames_idx.append(list(range(len(key_frames_list[i]))))

        init_step_time += key_frame_time

        # **************************************************************************
        # 3. 构建中间树
        # **************************************************************************

        intermediate_trees_time = 0.0
        init_step_time += intermediate_trees_time

        # **************************************************************************
        # 4. 生成所有可能的路径
        # **************************************************************************

        if verbose:
            with printer.indented(verbose_level):
                printer.info("Generating all possible paths...")
                if max_time < init_step_time:
                    printer.warning("Warning: No time remained!!!!")
                    return None

        start_time = time.time()
        timeout = False
        for idx, temp_channel_path in enumerate(self._stratified_sampling(key_frames_idx)):
            current_time = time.time()
            if current_time - start_time > max_time - init_step_time:
                success = False
                timeout = True
                break
            path = []
            success = True
            if verbose:
                with printer.indented(verbose_level + 1):
                    printer.info(f"Attempting path {idx+1} / {len(list(self._stratified_sampling(key_frames_idx)))}...")
            for i in range(len(temp_channel_path) - 1):
                if verbose:
                    with printer.indented(verbose_level + 2):
                        printer.info(f"Planning segment {i+1} / {len(temp_channel_path) - 1}...")
                temp_start_conf = key_frames_list[i][temp_channel_path[i]]
                temp_end_conf = key_frames_list[i + 1][temp_channel_path[i + 1]]
                self.setup_pb_ompl(verbose)
                temp_path = self.planner(temp_start_conf, temp_end_conf, max_time=min(single_plan_max_time * (i + 1), max_time - init_step_time - (time.time() - start_time)), enable_reset=True, verbose=verbose, verbose_level=verbose_level + 3)

                if temp_path is not None:
                    path.append(temp_path)
                    if verbose:
                        with printer.indented(verbose_level + 3):
                            printer.info(f"Total target conf: {target_conf}")
                            printer.info(f"Step target conf: {temp_end_conf}")
                            printer.info(f"Solution end conf: {temp_path[-1]}")
                else:
                    success = False
                    break

                current_time = time.time()
                if current_time - start_time > max_time - init_step_time:
                    success = False
                    timeout = True
                    break

            if timeout:
                if verbose:
                    with printer.indented(verbose_level + 2):
                        printer.warning(f"Time out! {current_time - start_time:.2f} seconds")
                return None

            if success:
                if verbose:
                    with printer.indented(verbose_level + 1):
                        printer.success(f"Path found! Total time: {time.time() - start_time:.2f} seconds")
                return np.concatenate(path, axis=0)

        return None
