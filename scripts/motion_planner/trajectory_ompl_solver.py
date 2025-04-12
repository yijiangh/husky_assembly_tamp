import math
import os
import sys
from copy import deepcopy
from typing import Callable, List, Union

import numpy as np
import pybullet as p

# OMPL
# 添加pb_ompl导入
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from utils.params import *
from motion_planner.pb_ompl import pb_ompl


class PbOMPLRobotWrapper(pb_ompl.PbOMPLRobot):
    """
    包装PbOMPLRobot类，使其适配我们的碰撞检测函数
    """

    def __init__(self, robot_id, joint_indices):
        self.id = robot_id
        self.joint_idx = joint_indices
        self.num_dim = len(joint_indices)
        self.state = [0] * self.num_dim
        self.joint_bounds = []
        self._set_manual_joint_bounds()

    def _set_manual_joint_bounds(self):
        # 手动设置每个关节的边界，避免从PyBullet读取
        for _ in range(self.num_dim):
            self.joint_bounds.append([-2 * math.pi, 2 * math.pi])

    def get_joint_bounds(self):
        if not self.joint_bounds:
            self._set_manual_joint_bounds()
        return self.joint_bounds

    def _is_not_fixed(self, joint_idx):
        return True  # 我们已经预先过滤了固定关节


class TrajectoryOMPLSolver:
    def __init__(
        self,
        collision_fn: Callable[[np.ndarray], bool],
        planner: str = "RRTConnect",
        robot_id: int = None,
        arm_joints: List[int] = None,
        obstacles: List[int] = None,
    ):
        self.collision_fn = collision_fn
        self.robot_id = robot_id
        self.arm_joints = arm_joints
        self.obstacles = obstacles if obstacles else []

        # 如果提供了机器人ID和关节索引，使用pb_ompl
        if robot_id is not None and arm_joints is not None:
            self.use_pb_ompl = True
            self.setup_pb_ompl(planner)
        else:
            self.use_pb_ompl = False
            self.setup_regular_ompl(planner)

    def setup_pb_ompl(self, planner_name):
        """设置pb_ompl规划器"""
        # 创建机器人包装器
        self.robot = PbOMPLRobotWrapper(self.robot_id, self.arm_joints)

        # 初始化pb_ompl接口
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles)

        # 设置规划器
        self.pb_ompl_interface.set_planner(planner_name)

        self.pb_ompl_interface.si.setStateValidityCheckingResolution(0.0005)

        if hasattr(self.pb_ompl_interface.space, "setLongestValidSegmentFraction"):
            self.pb_ompl_interface.space.setLongestValidSegmentFraction(0.0005)

        if hasattr(self.pb_ompl_interface.planner, "setRange"):
            self.pb_ompl_interface.planner.setRange(0.01)

        # 配置碰撞检测
        if self.collision_fn:
            # 替换默认的碰撞检测为自定义函数
            def custom_is_state_valid(state):
                state_arr = np.array([state[i] for i in range(self.robot.num_dim)])
                return not self.collision_fn(state_arr)

            self.pb_ompl_interface.ss.setStateValidityChecker(ob.StateValidityCheckerFn(custom_is_state_valid))

    def setup_regular_ompl(self, planner_name):
        """设置常规OMPL规划器（保留原有实现）"""
        ou.setLogLevel(ou.LOG_ERROR)

        self.space = ob.RealVectorStateSpace(6)
        self.space.setLongestValidSegmentFraction(0.01)

        # 设置每个关节的边界
        bounds = ob.RealVectorBounds(6)
        bounds.setLow(-2 * math.pi)
        bounds.setHigh(2 * math.pi)
        self.space.setBounds(bounds)

        # 创建空间信息对象
        self.si = ob.SpaceInformation(self.space)

        # 设置状态有效性检查器
        self.si.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))
        self.si.setStateValidityCheckingResolution(0.001)  # 0.005

        # 配置规划器
        if planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.si)
        elif planner_name == "AITstar":
            self.planner = og.AITstar(self.si)
        elif planner_name == "EITstar":
            self.planner = og.EITstar(self.si)
        else:
            self.planner = og.RRTConnect(self.si)

    def isStateValid(self, state):
        """
        常规OMPL的状态有效性检查
        """
        state_arr = np.zeros(6)
        for i in range(6):
            state_arr[i] = state[i]
        if self.collision_fn is not None:
            return not self.collision_fn(state_arr)
        else:
            return True

    def plan(self, start_angles: np.ndarray, goal_angles: np.ndarray, interp_num: int = 1000, time: float = 10.0) -> np.ndarray:
        """
        规划路径

        参数:
            start_angles: 起始关节角度
            goal_angles: 目标关节角度
            interp_num: 插值数量
            time: 最大规划时间（秒）

        返回:
            规划出的路径（如果成功）或None（如果失败）
        """
        if self.use_pb_ompl:
            # 使用pb_ompl进行规划
            res, path = self.pb_ompl_interface.plan_start_goal(start_angles.tolist(), goal_angles.tolist(), allowed_time=time)

            if res:
                # 将路径转换为numpy数组
                path_array = np.array(path)

                # 验证路径
                for conf in path_array:
                    if self.collision_fn(conf):
                        return None
                return np.array(path)
            else:
                return None
        else:
            # 使用原始OMPL方法
            # 定义起始状态
            start = ob.State(self.space)
            for i in range(6):
                start()[i] = start_angles[i]

            # 定义目标状态
            goal = ob.State(self.space)
            for i in range(6):
                goal()[i] = goal_angles[i]

            # 创建问题定义
            pdef = ob.ProblemDefinition(self.si)
            pdef.setStartAndGoalStates(start, goal)

            # 设置规划器的问题定义
            self.planner.setProblemDefinition(pdef)

            # 求解路径规划问题
            solved = self.planner.solve(time)

            if solved:
                path = pdef.getSolutionPath()
                num_states = path.getStateCount()

                # 添加插值
                path.interpolate(interp_num)
                num_states = path.getStateCount()

                path_array = np.zeros((num_states, 6))
                for i in range(num_states):
                    state = path.getState(i)
                    for j in range(6):
                        path_array[i, j] = state[j]

                # 验证路径
                for conf in path_array:
                    if self.collision_fn(conf):
                        return None
                return path_array
            else:
                return None
