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

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from model.scene_parse import SceneParser
from motion_planner.pb_ompl import pb_ompl
from motion_planner.tapom.topology import TopologyPlanner
from motion_planner.tapom.planner import Planner
from motion_planner.trajectory_ompl_solver import PbOMPLRobotWrapper
from multi_tangent.collision import create_collision_bodies
from robot.robot_setup import RobotSetup
from utils.collision import init_pb
from utils.params import HUSKY_URDF_PATH
from utils.util import PrintManager, interpolate


# 初始化PrintManager实例
printer = PrintManager()


class TrajectoryTAMPORSolver:
    """TAMPOR (Topology-Aware Motion Planning with Obstacle Rearrangement)求解器

    将TopologyPlanner和Planner整合到一个类中，提供统一的规划接口。
    """

    def __init__(self, robot_setup: RobotSetup, channel_info: List[Dict], grasp_pose, eval_max_attempts: int = 50000) -> None:
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
        self.grasp_pose = grasp_pose
        self.eval_max_attempts = eval_max_attempts

        # 创建内部规划器实例，但不立即初始化
        self.topology_planner = None
        self.path_planner = None

    def _init_topology_planner(self, bodies: List[int], alpha: float = 1.0, beta: float = 2.0, gamma: float = 3.0) -> None:
        """初始化拓扑规划器

        Args:
            bodies: 碰撞体列表
        """
        self.topology_planner = TopologyPlanner(self.robot_setup, self.channel_info, bodies, eval_max_attempts=self.eval_max_attempts, alpha=alpha, beta=beta, gamma=gamma)

    def _init_path_planner(self, collision_fn: Callable[[np.ndarray], bool], verbose: bool = False) -> None:
        """初始化路径规划器

        Args:
            collision_fn: 碰撞检测函数
        """
        self.path_planner = Planner(self.robot_setup, self.channel_info, self.grasp_pose, collision_fn, verbose=verbose)

    def plan(
        self,
        start_conf: np.ndarray,
        target_conf: np.ndarray,
        element_bodies: List[int],
        grasp_attachments: List[Attachment],
        grasps: List,
        grasp_weights: List[float] = None,
        max_time: float = 600.0,
        init_step_max_time: float = 200.0,
        step_max_time: float = 15.0,
        key_frame_num: int = 20,
        alpha: float = 1.0,
        beta: float = 2.0,
        gamma: float = 3.0,
        verbose: bool = False,
    ) -> Dict:
        """Execute the complete planning process

        Combine topology planning and path planning to find the optimal path from start to goal.

        Args:
            start_conf: Initial joint configuration
            target_conf: Target joint configuration
            element_bodies: List of collision bodies
            grasp_attachments: List of grasp attachments
            max_time: Maximum planning time
            init_step_max_time: Maximum initial step time
            step_max_time: Maximum step time
            key_frame_num: Number of key frames
            verbose: Whether to print verbose information

        Returns:
            Dictionary containing planning results {"success": bool, "path": np.ndarray}
        """
        if verbose:
            printer.info("\n========== TAMPOR TRAJECTORY PLANNING ==========")

        # 获取起点和终点位置
        self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, start_conf)
        # 计算所有grasp_attachment的平均位置作为起点
        start_points = []
        for attachment in grasp_attachments:
            start_points.append(np.array(pp.get_point(attachment.child)))
        start_point = np.mean(start_points, axis=0)

        self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, target_conf)
        # 计算所有grasp_attachment的平均位置作为终点
        target_points = []
        for attachment in grasp_attachments:
            target_points.append(np.array(pp.get_point(attachment.child)))
        target_point = np.mean(target_points, axis=0)

        # 恢复起始位置
        self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, start_conf)

        # 创建碰撞检测函数
        collision_fn = self.robot_setup.create_collision_fn(element_bodies)

        # 初始化规划器
        with pp.LockRenderer():
            if verbose:
                printer.info("TAMPOR: Initializing topology planner...")
            self._init_topology_planner(element_bodies, alpha, beta, gamma)
            if verbose:
                printer.info("TAMPOR: Initializing path planner...")
            self._init_path_planner(collision_fn, verbose)

        # 1. Topology Planning
        if verbose:
            printer.info("TAMPOR: Starting topology planning...")
        start_time = time.time()
        best_path = self.topology_planner.plan(start_point, target_point, verbose=verbose, verbose_level=1)
        topology_time = time.time() - start_time
        if verbose:
            printer.info(f"TAMPOR: Topology planning completed in {topology_time:.2f} seconds")

        if not best_path:
            if verbose:
                printer.warning("TAMPOR: Failed to find a valid topology path")
            return {"success": False, "path": None}

        # if best_path == [-1, 4, -2]: # cuboid_1, task_1
        # if best_path == [-1, 2, 11, -2]: # shelf_1, task_1
        #     return {"success": True, "path": best_path}
        # else:
        #     return {"success": False, "path": None}

        # 2. Path Planning
        if verbose:
            printer.info("TAMPOR: Starting path planning...")
        start_time = time.time()
        # with pp.LockRenderer():
        path = self.path_planner.plan(
            start_conf, target_conf, best_path, grasps=grasps, grasp_weights=grasp_weights, max_time=max_time - topology_time, init_step_max_time=init_step_max_time, single_plan_max_time=step_max_time, num_points=key_frame_num, verbose=verbose, verbose_level=1
        )
        path_time = time.time() - start_time
        if verbose:
            printer.info(f"TAMPOR: Path planning completed in {path_time:.2f} seconds")

        if path is None:
            if verbose:
                printer.warning("TAMPOR: Failed to find a valid trajectory path")
            return {"success": False, "path": None}

        if verbose:
            printer.info("\n========== TAMPOR PLANNING SUCCESSFUL ==========")
            printer.info(f"TAMPOR: Total planning time: {topology_time + path_time:.2f} seconds")

        return {"success": True, "path": path}


if __name__ == "__main__":
    # 示例用法
    init_pb()

    scene_file = os.path.join(HERE, "model", "scenes", "rebar_1", "task_1.yml")
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
    solver = TrajectoryTAMPORSolver(rb, channel_info, grasp_offset, eval_max_attempts=1000)
    solver._init_topology_planner(bodies)
    best_path = solver.topology_planner.plan(start_point, target_point, verbose=True, verbose_level=0)
    rb.set_base_pose_2d(10, 10, 0)
    solver.topology_planner.plot_graph(solver.topology_planner.full_graph, highlight_path=best_path, start_point=start_point, target_point=target_point)

    pp.wait_for_user()

    # plan_result = solver.plan(start_q, target_q, bodies, grasped_attachment, key_frame_num=10, max_time=600, grow_tree_max_nodes=50, verbose=True)

    # if plan_result["success"]:
    #     path = plan_result["path"]

    #     # 可视化路径
    #     slider = p.addUserDebugParameter("replay", 0, 1, 0)
    #     while True:
    #         slider_value = p.readUserDebugParameter(slider)
    #         time_idx = int(slider_value * (path.shape[0] - 1))
    #         joint_val = path[time_idx]
    #         rb.set_joint_positions(rb.arm_joints, joint_val)
    #         time.sleep(1.0 / 60)
    # else:
    #     printer.error("规划失败")
