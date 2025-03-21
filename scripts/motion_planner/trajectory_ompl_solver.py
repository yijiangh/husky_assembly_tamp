import math
import os
import sys
from typing import Callable, List, Union

import numpy as np
from copy import deepcopy

# OMPL
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from utils.params import *


class TrajectoryOMPLSolver:
    def __init__(
        self,
        collision_fn: Callable[[np.ndarray], bool],
        planner: str = "RRTConnect",
        logger_level: int = ou.LOG_ERROR,
    ):
        self.collision_fn = collision_fn

        ou.setLogLevel(logger_level)

        self.space = ob.RealVectorStateSpace(6)
        self.space.setLongestValidSegmentFraction(0.05)

        # 设置每个关节的边界
        bounds = ob.RealVectorBounds(6)
        for i in range(6):
            bounds.setLow(i, -math.pi)
            bounds.setHigh(i, math.pi)
        self.space.setBounds(bounds)

        # 创建空间信息对象
        self.si = ob.SpaceInformation(self.space)

        # 设置状态有效性检查器
        self.si.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))
        self.si.setStateValidityCheckingResolution(0.01)  # 0.005

        # 配置规划器
        if planner == "RRTConnect":
            self.planner = og.RRTConnect(self.si)
        elif planner == "AITstar":
            self.planner = og.AITstar(self.si)
        elif planner == "EITstar":
            self.planner = og.EITstar(self.si)
        else:
            self.planner = og.RRTConnect(self.si)

    def isStateValid(self, state):
        """
        Check if the state is valid (collision-free)

        Params:
            state: current joint positions (6 dimensions)

        Returns:
            bool: True if successful, False otherwise.
        """
        state_arr = np.zeros(6)
        for i in range(6):
            state_arr[i] = state[i]
        if self.collision_fn is not None:
            return not self.collision_fn(state_arr)
        else:
            return True

    def plan(
        self, start_angles: np.ndarray, goal_angles: np.ndarray, interp_num: int = 1000, time: float = 10.0
    ) -> np.ndarray:
        """
        Plan manipulator path.

        Params:
            start_angles (np.ndarray): start joint angles (6 dimensions)
            goal_angles (np.ndarray): goal joint angles (6 dimensions)
            time (float): planning time (seconds), default 10.0

        Returns:
            np.ndarray: path (joint angles) if successful, None otherwise
        """
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

            path.interpolate(num_states * interp_num)
            num_states = num_states * interp_num

            path_array = np.zeros((num_states, 6))
            for i in range(num_states):
                state = path.getState(i)
                for j in range(6):
                    path_array[i, j] = state[j]

            for conf in path_array:
                if self.collision_fn(conf):
                    return None
            return path_array
        else:
            return None
