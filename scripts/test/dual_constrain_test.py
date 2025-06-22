#!/usr/bin/env python3

import os
import sys
import numpy as np
import pybullet as p
import pybullet_planning as pp
from compas_fab.robots import RobotSemantics
from compas_fab.robots.robot import RobotModel
from tracikpy import TracIKSolver
import time  # 导入时间模块用于生成文件名和可视化控制
import argparse
import math
from functools import partial

# Import OMPL libraries
try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys

    sys.path.insert(0, join(dirname(dirname(dirname(abspath(__file__)))), "py-bindings"))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from multi_tangent.collision import create_collision_bodies
from utils.collision import init_pb
from utils.params import *
from robot.robot_setup import RobotSetup, HUSKY_URDF_PATH, HUSKY_ARM_JOINT_NAMES, HUSKY_CONTROL_JOINT_NAMES, HUSKY_TOOL0_NAME
from ConstrainedPlanningCommon import *
from utils.util import interpolate

if __name__ == "__main__":
    init_pb()
    robot = RobotSetup("r0", robot_type="husky_dual")
    pp.wait_for_user()