import os
import sys
import time
from copy import deepcopy

import numpy as np
import pybullet as p
import pybullet_planning as pp

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from multi_tangent.collision import create_collision_bodies
from ompl_manipulator_test import TrajectoryOMPLSolver
from robot.robot_setup import RobotSetup
from utils.collision import init_pb
from model.scene_parse import SceneParser

if __name__ == "__main__":

    init_pb()
    
    scene_file = os.path.join(HERE, "model", "scenes", "cuboid_1", "task_1.yml")
    scene_parser = SceneParser(scene_file)
    scene_parser.load_scene()
    line_pts, radius_per_edge = scene_parser.get_element_info()
    bodies = create_collision_bodies(line_pts, radius_per_edge, viewer=True)
    
    start_q = np.array(scene_parser.get_robot_start_pose())
    target_q = np.array(scene_parser.get_robot_target_pose())
    pose_2d = scene_parser.get_robot_pose_2d(output_type="array")
    
    rb = RobotSetup("rb")
    rb.set_joint_positions(rb.arm_joints, start_q)
    rb.set_base_pose_2d(pose_2d[0], pose_2d[1], pose_2d[2])

    # 执行规划并获取路径
    path = rb.plan_manipulator_path(start_q, target_q, [], bodies[:2], max_time=600, max_iterations=10000)

    # -------------------- 下面是使用pybullet进行可视化的代码 --------------------#
    slider = p.addUserDebugParameter("replay", 0, 1, 0)

    while True:
        slider_value = p.readUserDebugParameter(slider)
        time_idx = int(slider_value * (path.shape[0] - 1))
        joint_val = path[time_idx]
        rb.set_joint_positions(rb.arm_joints, joint_val)
        time.sleep(1.0 / 60)
