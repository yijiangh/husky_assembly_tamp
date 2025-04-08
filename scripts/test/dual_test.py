import os
import sys

import numpy as np
import pybullet as p
import pybullet_planning as pp
from compas_fab.robots import RobotSemantics
from compas_fab.robots.robot import RobotModel
from tracikpy import TracIKSolver

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from multi_tangent.collision import create_collision_bodies
from utils.collision import init_pb
from utils.params import *

CONTROL_JOINT_NAMES = ["left_ur_arm_shoulder_pan_joint", "left_ur_arm_shoulder_lift_joint", "left_ur_arm_elbow_joint", "left_ur_arm_wrist_1_joint", "left_ur_arm_wrist_2_joint", "left_ur_arm_wrist_3_joint"]

init_pb()

robot_urdf = os.path.join(DATA_DIR, "husky_urdf", "mt_husky_dual_ur5_e_moveit_config", "urdf", "husky_dual_ur5_e.urdf")
robot_srdf = os.path.join(DATA_DIR, "husky_urdf", "mt_husky_dual_ur5_e_moveit_config", "config", "husky.srdf")
gripper_obj = os.path.join(DATA_DIR, "husky_urdf", "robotiq_85", "meshes", "static", "robotiq_85_close_20mm.obj")

robot = pp.load_pybullet(robot_urdf, fixed_base=False, cylinder=False)
# left_joints = pp.joints_from_names(robot, CONTROL_JOINT_NAMES)

# # 创建6个滑块来控制左臂关节角度
# sliders = []
# for i, joint_name in enumerate(CONTROL_JOINT_NAMES):
#     joint_index = pp.joint_from_name(robot, joint_name)
#     joint_info = pp.get_joint_info(robot, joint_index)
#     lower_limit = joint_info.jointLowerLimit
#     upper_limit = joint_info.jointUpperLimit
#     slider = p.addUserDebugParameter(
#         joint_name, 
#         lower_limit, 
#         upper_limit, 
#         pp.get_joint_position(robot, joint_index)
#     )
#     sliders.append(slider)
    
# pose = pp.get_link_pose(robot, pp.link_from_name(robot, "left_ur_arm_tool0"))
# pp.draw_pose(pose, length = 0.25)

# # 主循环中更新关节位置
# while True:
#     joint_positions = []
#     for slider in sliders:
#         joint_positions.append(p.readUserDebugParameter(slider))
#     pp.set_joint_positions(robot, left_joints, joint_positions)


line_pts_grasped = [np.array([0, 0, 0]), np.array([0, 0, 1])]
grasped_element = create_collision_bodies(line_pts_grasped, [0.01], viewer=True)[0]
# pp.set_pose(grasped_element, pp.multiply(pp.get_link_pose(rb.robot, rb.tool_link), pp.Pose(point=grasp_offset, euler=pp.Euler(1.5708, 0, 0))))
# grasped_attachment = pp.create_attachment(rb.robot, rb.tool_link, grasped_element)
# rb.update_attachments([grasped_attachment])


sliders = []
names = ["x", "y", "z", "roll", "pitch", "yaw"]
for i in range(6):
    slider = p.addUserDebugParameter(f"{names[i]}", -3.14, 3.14, 0)
    sliders.append(slider)
    
while True:
    cartesian_pos = []
    for slider in sliders:
        cartesian_pos.append(p.readUserDebugParameter(slider))
    pp.set_pose(grasped_element, pp.Pose(point=cartesian_pos[0:3], euler=pp.Euler(*cartesian_pos[3:])))

pp.wait_for_user()
