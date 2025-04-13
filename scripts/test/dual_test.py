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

LEFT_JOINT_NAMES = ["left_ur_arm_shoulder_pan_joint", "left_ur_arm_shoulder_lift_joint", "left_ur_arm_elbow_joint", "left_ur_arm_wrist_1_joint", "left_ur_arm_wrist_2_joint", "left_ur_arm_wrist_3_joint"]
RIGHT_JOINT_NAMES = ["right_ur_arm_shoulder_pan_joint", "right_ur_arm_shoulder_lift_joint", "right_ur_arm_elbow_joint", "right_ur_arm_wrist_1_joint", "right_ur_arm_wrist_2_joint", "right_ur_arm_wrist_3_joint"]
init_pb()

robot_urdf = os.path.join(DATA_DIR, "husky_urdf", "mt_husky_dual_ur5_e_moveit_config", "urdf", "husky_dual_ur5_e.urdf")
robot_srdf = os.path.join(DATA_DIR, "husky_urdf", "mt_husky_dual_ur5_e_moveit_config", "config", "husky.srdf")
gripper_obj = os.path.join(DATA_DIR, "husky_urdf", "robotiq_85", "meshes", "static", "robotiq_85_close_20mm.obj")

robot = pp.load_pybullet(robot_urdf, fixed_base=False, cylinder=False)
left_tool0_pose = pp.get_link_pose(robot, pp.link_from_name(robot, "left_ur_arm_tool0"))
left_ee = pp.create_obj(gripper_obj, scale=1)
pp.set_pose(left_ee, pp.multiply(left_tool0_pose, pp.Pose(euler=pp.Euler(yaw=-np.pi / 2))))
left_ee_attachment = pp.create_attachment(robot, pp.link_from_name(robot, "left_ur_arm_tool0"), left_ee)

right_tool0_pose = pp.get_link_pose(robot, pp.link_from_name(robot, "right_ur_arm_tool0"))
right_ee = pp.create_obj(gripper_obj, scale=1)
pp.set_pose(right_ee, pp.multiply(right_tool0_pose, pp.Pose(euler=pp.Euler(yaw=-np.pi / 2))))
right_ee_attachment = pp.create_attachment(robot, pp.link_from_name(robot, "right_ur_arm_tool0"), right_ee)

left_joints = pp.joints_from_names(robot, LEFT_JOINT_NAMES)
right_joints = pp.joints_from_names(robot, RIGHT_JOINT_NAMES)

line_pts_grasped = [np.array([0, 0, 0]), np.array([0, 0, 1])]
grasped_element = create_collision_bodies(line_pts_grasped, [0.01], viewer=True)[0]
left_box = pp.create_box(0.02, 0.02, 0.1)
right_box = pp.create_box(0.02, 0.02, 0.1)
pp.set_color(right_box, [0, 0, 1, 1])

left_solver = TracIKSolver(robot_urdf, "base_link", "left_ur_arm_tool0")
right_solver = TracIKSolver(robot_urdf, "base_link", "right_ur_arm_tool0")

box_sliders = []
names = ["left_y", "left_pitch", "right_y", "right_pitch"]
for i in range(4):
    slider = p.addUserDebugParameter(f"{names[i]}", -3.14, 3.14, 0)
    box_sliders.append(slider)
    
element_sliders = []
names = ["x", "y", "z", "roll", "pitch", "yaw"]
for i in range(6):
    slider = p.addUserDebugParameter(f"{names[i]}", -3.14, 3.14, 0)
    element_sliders.append(slider)
    
left_q_init = pp.get_joint_positions(robot, left_joints)
right_q_init = pp.get_joint_positions(robot, right_joints)

# 添加录制和停止按钮
record_button = p.addUserDebugParameter("Record", 1, 0, 0)
stop_button = p.addUserDebugParameter("Stop", 1, 0, 0)

# 录制相关变量
is_recording = False
recorded_data = []
last_record_value = 0
last_stop_value = 0

while True:
    # 检查录制按钮状态
    record_value = p.readUserDebugParameter(record_button)
    if record_value != last_record_value:
        last_record_value = record_value
        is_recording = True
        print("开始录制...")
        recorded_data = []  # 清空之前的录制数据
        
    # 检查停止按钮状态
    stop_value = p.readUserDebugParameter(stop_button)
    if stop_value != last_stop_value and is_recording:
        last_stop_value = stop_value
        is_recording = False
        print("停止录制...")
        
        # 保存录制的数据
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"recorded_data_{timestamp}.npz"
        save_path = os.path.join(HERE, "recordings", filename)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 将列表转换为数组并保存
        data_array = np.array(recorded_data, dtype=object)
        np.savez(save_path, data=data_array)
        print(f"数据已保存到: {save_path}")
    
    cartesian_pose = []
    for slider in element_sliders:
        cartesian_pose.append(p.readUserDebugParameter(slider))
    pose = pp.Pose(point=cartesian_pose[0:3], euler=pp.Euler(*cartesian_pose[3:]))
    pp.set_pose(grasped_element, pose)
    
    cartesian_offset = []
    for slider in box_sliders:
        cartesian_offset.append(p.readUserDebugParameter(slider))
    offset_pose = pp.Pose(point=[0, 0.15, cartesian_offset[0]], euler=pp.Euler(roll=1.5708, yaw=cartesian_offset[1]))
    left_box_pose = pp.multiply(pose, offset_pose)
    pp.set_pose(left_box, left_box_pose)
    
    offset_pose = pp.Pose(point=[0, 0.15, cartesian_offset[2]], euler=pp.Euler(roll=1.5708, yaw=cartesian_offset[3]))
    right_box_pose = pp.multiply(pose, offset_pose)
    pp.set_pose(right_box, right_box_pose)
    
    left_sol = left_solver.ik(pp.tform_from_pose(left_box_pose), qinit=left_q_init)
    if left_sol is not None:
        pp.set_joint_positions(robot, left_joints, left_sol)
        left_q_init = left_sol
        left_ee_attachment.assign()
        
    right_sol = right_solver.ik(pp.tform_from_pose(right_box_pose), qinit=right_q_init)
    if right_sol is not None:
        pp.set_joint_positions(robot, right_joints, right_sol)
        right_q_init = right_sol
        right_ee_attachment.assign()
        
    # 录制数据
    if is_recording and left_sol is not None and right_sol is not None:
        frame_data = {
            'box_sliders': cartesian_offset,
            'element_sliders': cartesian_pose,
            'left_sol': left_sol,
            'right_sol': right_sol
        }
        recorded_data.append(frame_data)
