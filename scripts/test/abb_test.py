import os
import sys
import numpy as np
import pybullet as p
import pybullet_planning as pp
from compas_fab.robots import RobotSemantics
from compas_fab.robots.robot import RobotModel
from tracikpy import TracIKSolver
import time  # 导入时间模块用于生成文件名

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from multi_tangent.collision import create_collision_bodies
from utils.collision import init_pb
from utils.params import *
from robot.robot_setup import RobotSetup
from model.scene_parse import SceneParser

JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

init_pb()

# robot_setup = RobotSetup(robot_type="abb")

# # joint_sliders = []
# # for i in range(6):
# #     slider = p.addUserDebugParameter(f"joint_{i+1}", -3.14, 3.14, 0)
# #     joint_sliders.append(slider)
# # while True:
# #     joint_pos = []
# #     for i in range(6):
# #         joint_pos.append(p.readUserDebugParameter(joint_sliders[i]))
# #     robot_setup.set_joint_positions(robot_setup.arm_joints, joint_pos)
# #     time.sleep(0.01)

# q = np.array([0, 0, 0, 0, 0, 0])
# q_1 = np.array([0.0, -0.01073136, 0.16691836, 0.0, -0.156187, 0.0])
# path = robot_setup.plan_manipulator_path(q, q_1, [], [])

# replay_slider = p.addUserDebugParameter("replay", 0, 1, 0)

# while True:
#     replay_value = p.readUserDebugParameter(replay_slider)
#     q_next = path[int((len(path) - 1) * replay_value)]
#     robot_setup.set_joint_positions(robot_setup.arm_joints, q_next)
#     time.sleep(0.01)

# pp.wait_for_user()

scene_parser = SceneParser(os.path.join(HERE, "model", "scenes", "rebar_1", "task_1.yml"))

rb = scene_parser.create_robot("r0")

# 设置抓取物体
attachment_body, grasp_attachment, approximate_attachment_body, approximate_attachment = scene_parser.create_attachment(rb, approximate=True)
rb.update_attachments(grasp_attachment + approximate_attachment)
# 加载场景元素
# element_bodies, element_infos = scene_parser.create_elements(color=[1, 0, 0, 1])
element_bodies, element_infos = scene_parser.create_elements()

collision_fn = rb.create_collision_fn(element_bodies)

pose = pp.Pose(point=[2.04421, 0.186707, 1.12262], euler=pp.Euler(0, 0, 0))
tool_from_obj = pp.Pose(point=scene_parser.get_robot_base_grasp_offset(), euler=pp.Euler(*scene_parser.get_robot_base_grasp_rotation()))
q = rb.get_grasp_ik_solution(pose, tool_from_obj)
rb.set_joint_positions(rb.arm_joints, q)
print(f"q: {q}")

pp.wait_for_user()

joint_sliders = []
for i in range(6):
    slider = p.addUserDebugParameter(f"joint_{i+1}", -6.28, 6.28, q[i])
    joint_sliders.append(slider)
while True:
    joint_pos = []
    for i in range(6):
        joint_pos.append(p.readUserDebugParameter(joint_sliders[i]))
    rb.set_joint_positions(rb.arm_joints, joint_pos)
    print(f"collision_fn: {collision_fn(joint_pos)}, joint_positions: {joint_pos}")
    time.sleep(0.01)

pp.wait_for_user()
