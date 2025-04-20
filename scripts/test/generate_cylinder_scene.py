import argparse
import json
import os
import random
import signal
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import pybullet as p
import pybullet_planning as pp

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from model.scene_parse import SceneParser
from multi_tangent.collision import create_collision_bodies
from multi_tangent.convert import flatten_list
from robot.robot_setup import RobotSetup
from utils.collision import Element, create_couplers, init_pb
from utils.params import *
from utils.util import CounterModule, SetSeeds, PrintManager


# 初始化一个函数来创建所有滑条，方便在revert时重新创建
def create_sliders(init_pos=[0.5, 0, 0.5], init_orient=[0, 0, 0], init_length=0.3, init_radius=0.05):
    """创建所有滑块，并返回滑块ID字典"""
    sliders = {}
    
    # 主滑块 - 粗调整
    sliders['x'] = p.addUserDebugParameter("X Coarse", -2, 2, init_pos[0])
    sliders['y'] = p.addUserDebugParameter("Y Coarse", -2, 2, init_pos[1])
    sliders['z'] = p.addUserDebugParameter("Z Coarse", 0, 2, init_pos[2])
    
    # Fine sliders - for precise position adjustments (±0.1 range)
    sliders['x_fine'] = p.addUserDebugParameter("X Fine", -0.1, 0.1, 0)
    sliders['y_fine'] = p.addUserDebugParameter("Y Fine", -0.1, 0.1, 0)
    sliders['z_fine'] = p.addUserDebugParameter("Z Fine", -0.1, 0.1, 0)
    
    # Orientation sliders
    sliders['roll'] = p.addUserDebugParameter("Roll Coarse", -np.pi, np.pi, init_orient[0])
    sliders['pitch'] = p.addUserDebugParameter("Pitch Coarse", -np.pi, np.pi, init_orient[1])
    sliders['yaw'] = p.addUserDebugParameter("Yaw Coarse", -np.pi, np.pi, init_orient[2])
    
    # Fine sliders - for precise orientation adjustments (±0.1 radians, approx. ±5.7 degrees)
    sliders['roll_fine'] = p.addUserDebugParameter("Roll Fine", -0.1, 0.1, 0)
    sliders['pitch_fine'] = p.addUserDebugParameter("Pitch Fine", -0.1, 0.1, 0)
    sliders['yaw_fine'] = p.addUserDebugParameter("Yaw Fine", -0.1, 0.1, 0)
    
    # Size sliders
    sliders['length'] = p.addUserDebugParameter("Length Coarse", 0.01, 3, init_length)
    sliders['radius'] = p.addUserDebugParameter("Radius Coarse", 0.01, 0.5, init_radius)
    
    # Fine sliders - for precise size adjustments (±0.05 range)
    sliders['length_fine'] = p.addUserDebugParameter("Length Fine", -0.05, 0.05, 0)
    sliders['radius_fine'] = p.addUserDebugParameter("Radius Fine", -0.05, 0.05, 0)
    
    # Buttons
    sliders['add'] = p.addUserDebugParameter("Add Cylinder", 1, 0, 0)
    sliders['revert'] = p.addUserDebugParameter("Revert", 1, 0, 0)
    sliders['save'] = p.addUserDebugParameter("Save to JSON", 1, 0, 0)
    sliders['save_yaml'] = p.addUserDebugParameter("Save as YAML", 1, 0, 0)
    
    return sliders

# 函数：删除所有UI控件
def remove_all_ui_controls():
    """删除环境中所有的用户界面控件"""
    p.removeAllUserParameters()

# 函数：查找最新的cylinders JSON文件
def find_latest_cylinder_file():
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_cylinders")
    if not os.path.exists(save_dir):
        return None
    
    # 查找所有JSON文件
    json_files = [f for f in os.listdir(save_dir) if f.endswith('.json') and f.startswith('cylinders_')]
    
    if not json_files:
        return None
    
    # 按文件名排序（因为文件名包含时间戳）
    json_files.sort(reverse=True)  # 最新的在前面
    
    return os.path.join(save_dir, json_files[0])

# 函数：从JSON文件加载圆柱体
def load_cylinders_from_file(file_path):
    if not file_path or not os.path.exists(file_path):
        return []
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        loaded_cylinders = []
        
        # 从JSON中创建圆柱体
        for cylinder in data.get('cylinders', []):
            pos = cylinder['position']
            ori = cylinder['orientation']
            
            # 创建圆柱体
            cylinder_id = pp.create_cylinder(
                cylinder['radius'], 
                cylinder['length'],
                color=(random.random(), random.random(), random.random(), 1)
            )
            
            # 设置位置和方向
            pose = pp.Pose(
                point=[pos['x'], pos['y'], pos['z']],
                euler=[ori['roll'], ori['pitch'], ori['yaw']]
            )
            pp.set_pose(cylinder_id, pose)
            
            # 将圆柱体信息添加到列表中
            loaded_cylinders.append((
                cylinder_id, 
                [pos['x'], pos['y'], pos['z'], ori['roll'], ori['pitch'], ori['yaw']], 
                cylinder['length'], 
                cylinder['radius']
            ))
        
        print(f"从 {file_path} 加载了 {len(loaded_cylinders)} 个圆柱体")
        return loaded_cylinders
    except Exception as e:
        import traceback
        print(f"加载圆柱体数据时出错: {e}")
        traceback.print_exc()
        return []

# 函数：将当前场景保存为YAML文件
def save_scene_to_yaml(robot_setup: RobotSetup, added_cylinders):
    try:
        # 创建保存数据的目录
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_scenes")
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"scene_{timestamp}.yml")
        
        # 打开文件准备写入
        with open(save_path, 'w') as f:
            # 写入头部信息和元数据
            f.write(f"# Generated by test.py cylinder interaction tool\n")
            f.write(f"# Generated time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            scene_name = f"scene_{timestamp}"
            num_elements = len(added_cylinders)
            f.write(f'scene_name: "{scene_name}"\n')
            f.write(f"num_elements: {num_elements}\n\n")
            
            # 获取并写入机器人数据
            if robot_setup:
                # 获取关节角度
                joint_positions = list(pp.get_joint_positions(robot_setup.robot, robot_setup.arm_joints))
                # 获取机器人位姿
                robot_pos, robot_orn = p.getBasePositionAndOrientation(robot_setup.robot)
                robot_euler = p.getEulerFromQuaternion(robot_orn)
                
                # 写入机器人信息
                f.write("robot:\n")
                f.write(f"  target: {joint_positions}\n")
                f.write(f"  pose_2d: [{float(robot_pos[0])}, {float(robot_pos[1])}, {float(robot_euler[2])}]\n")
                f.write(f"  grasp_offset: [0.0, 0.0, 0.15]\n\n")
            
            # 写入圆柱体数据
            f.write("elements:\n")
            for idx, (cylinder_id, pose, length, radius) in enumerate(added_cylinders):
                # 获取位置和朝向
                pos = pose[0:3]
                euler = pose[3:6]
                # 将欧拉角转换为四元数
                quat = p.getQuaternionFromEuler(euler)
                
                # 写入元素信息
                element_id = f"element_{idx+1}"
                f.write(f'  - id: "{element_id}"\n')
                f.write(f"    position: [{float(pos[0])}, {float(pos[1])}, {float(pos[2])}]\n")
                f.write(f"    orientation: [{float(quat[0])}, {float(quat[1])}, {float(quat[2])}, {float(quat[3])}]\n")
                f.write("    shape:\n")
                f.write('      type: "cylinder"\n')
                f.write("      parameters:\n")
                f.write(f"        radius: {float(radius)}\n")
                f.write(f"        height: {float(length)}\n")
                f.write("    sphere_fit:\n")
                f.write(f"      radius: {float(radius)}\n")
                f.write(f"      count: 100\n\n")
        
        print(f"已成功保存场景到YAML文件: {save_path}")
        return True
    except Exception as e:
        import traceback
        print(f"保存场景到YAML文件时出错:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("错误详情:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--env", type=str, default="wooden_storage_shelf.stl", help="env object")
    parser.add_argument("--grasp", type=str, default="element_1", help="grasp element")
    parser.add_argument("--cylinder_file", type=str, help="Specific cylinder JSON file to load")
    args = parser.parse_args()

    init_pb()
    
    env_obj_path = os.path.join(HERE, "model", "obj", "env", args.env)
    env_obj = pp.create_obj(env_obj_path, scale=0.006)
    pp.set_pose(env_obj, pp.Pose(point=[0, 0, 0], euler=[0, 0, 0]))
    
    rb = RobotSetup("r0")
    pp.set_pose(rb.robot, pp.Pose(point=[-0.25, -0.75, 0], euler=[0, 0, 1.5708]))
    
    # 设置抓取物体
    grasp_offset = [0.0, 0.0, 0.15]
    line_pts_grasped = [np.array([0, 0, 0]), np.array([0, 0, 0.75])]
    grasped_element = create_collision_bodies(line_pts_grasped, [0.02], viewer=True)[0]
    pp.set_pose(grasped_element, pp.multiply(pp.get_link_pose(rb.robot, rb.tool_link), pp.Pose(point=grasp_offset, euler=pp.Euler(1.5708, 0, 0))))
    grasp_attachment = pp.create_attachment(rb.robot, rb.tool_link, grasped_element)
    rb.update_attachments([grasp_attachment])
    
    # **************************************************************************
    # 关节角控制
    # **************************************************************************
    
    initial_angles = [1.5708, -1.257, 1.058, -1.587, -1.455, 1.587]
    j0_slider = p.addUserDebugParameter("joint contorl j0", -2 * np.pi, 2 * np.pi, initial_angles[0])
    j1_slider = p.addUserDebugParameter("joint contorl j1", -2 * np.pi, 2 * np.pi, initial_angles[1])
    j2_slider = p.addUserDebugParameter("joint contorl j2", -2 * np.pi, 2 * np.pi, initial_angles[2])
    j3_slider = p.addUserDebugParameter("joint contorl j3", -2 * np.pi, 2 * np.pi, initial_angles[3])
    j4_slider = p.addUserDebugParameter("joint contorl j4", -2 * np.pi, 2 * np.pi, initial_angles[4])
    j5_slider = p.addUserDebugParameter("joint contorl j5", -2 * np.pi, 2 * np.pi, initial_angles[5])
    rb.set_joint_positions(rb.arm_joints, np.array(initial_angles))
    while True:
        j0 = p.readUserDebugParameter(j0_slider)
        j1 = p.readUserDebugParameter(j1_slider)
        j2 = p.readUserDebugParameter(j2_slider)
        j3 = p.readUserDebugParameter(j3_slider)
        j4 = p.readUserDebugParameter(j4_slider)
        j5 = p.readUserDebugParameter(j5_slider)
        rb.set_joint_positions(rb.arm_joints, np.array([j0, j1, j2, j3, j4, j5]))
        grasp_attachment.assign()
        time.sleep(1.0 / 240)
        
    # **************************************************************************
    # 碰撞体设置
    # **************************************************************************
    # pp.create_cylinder # This line was incomplete

    # --- Cylinder Interaction Setup ---
    # 尝试加载cylinder数据
    if args.cylinder_file:
        cylinder_file = args.cylinder_file
    else:
        cylinder_file = find_latest_cylinder_file()
    
    added_cylinders = load_cylinders_from_file(cylinder_file)

    # 使用最后一个圆柱体的初始值或默认值创建滑块
    if added_cylinders:
        # 使用最后一个圆柱体的值作为初始值
        last_id, last_pose, last_length, last_radius = added_cylinders[-1]
        sliders = create_sliders(
            init_pos=last_pose[0:3],
            init_orient=last_pose[3:6],
            init_length=last_length,
            init_radius=last_radius
        )
    else:
        # 使用默认值
        sliders = create_sliders(
            init_pos=[0.5, 0, 0.5],
            init_orient=[0, 0, 0],
            init_length=0.3,
            init_radius=0.05
        )
    
    # 获取滑块的初始值
    prev_add_button_value = p.readUserDebugParameter(sliders['add'])
    prev_revert_button_value = p.readUserDebugParameter(sliders['revert'])
    prev_save_button_value = p.readUserDebugParameter(sliders['save'])
    prev_save_yaml_value = p.readUserDebugParameter(sliders['save_yaml'])

    print("Interactive cylinder mode started. Use sliders and buttons.")
    print(f"Loaded {len(added_cylinders)} cylinders. Save button initial value: {prev_save_button_value}")

    while True:
        # Read current slider values
        # 读取主滑块和精细滑块值并组合
        x = p.readUserDebugParameter(sliders['x']) + p.readUserDebugParameter(sliders['x_fine'])
        y = p.readUserDebugParameter(sliders['y']) + p.readUserDebugParameter(sliders['y_fine'])
        z = p.readUserDebugParameter(sliders['z']) + p.readUserDebugParameter(sliders['z_fine'])
        
        roll = p.readUserDebugParameter(sliders['roll']) + p.readUserDebugParameter(sliders['roll_fine'])
        pitch = p.readUserDebugParameter(sliders['pitch']) + p.readUserDebugParameter(sliders['pitch_fine'])
        yaw = p.readUserDebugParameter(sliders['yaw']) + p.readUserDebugParameter(sliders['yaw_fine'])
        
        length = max(0.01, p.readUserDebugParameter(sliders['length']) + p.readUserDebugParameter(sliders['length_fine']))
        radius = max(0.01, p.readUserDebugParameter(sliders['radius']) + p.readUserDebugParameter(sliders['radius_fine']))

        current_pose_list = [x, y, z, roll, pitch, yaw]
        current_pose_pb = pp.Pose(point=[x, y, z], euler=[roll, pitch, yaw])

        # Read button states
        current_add_button_value = p.readUserDebugParameter(sliders['add'])
        current_revert_button_value = p.readUserDebugParameter(sliders['revert'])
        current_save_button_value = p.readUserDebugParameter(sliders['save'])
        current_save_yaml_value = p.readUserDebugParameter(sliders['save_yaml'])

        # --- Button Logic ---
        # Add Cylinder
        if current_add_button_value > prev_add_button_value:
            print("添加圆柱体...")
            try:
                # 使用随机颜色使不同的圆柱体更容易区分
                r, g, b = random.random(), random.random(), random.random()
                new_cylinder_id = pp.create_cylinder(radius, length, color=(r, g, b, 1))
                pp.set_pose(new_cylinder_id, current_pose_pb)
                added_cylinders.append((new_cylinder_id, current_pose_list, length, radius))
                print(f"已添加圆柱体 {new_cylinder_id}。总数: {len(added_cylinders)}")
            except Exception as e:
                print(f"创建或放置圆柱体时出错: {e}")

        # --- Save button logic ---
        if current_save_button_value > prev_save_button_value:
            print(f"保存按钮被点击! 当前值: {current_save_button_value}, 之前值: {prev_save_button_value}")
            prev_save_button_value = current_save_button_value
            if added_cylinders:
                try:
                    # 创建保存数据的目录
                    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_cylinders")
                    print(f"尝试创建保存目录: {save_dir}")
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # 创建带时间戳的文件名
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(save_dir, f"cylinders_{timestamp}.json")
                    print(f"将保存到文件: {save_path}")
                    
                    # 准备保存的数据
                    cylinders_data = []
                    for cylinder_id, pose, length, radius in added_cylinders:
                        cylinder_info = {
                            "id": int(cylinder_id),
                            "position": {"x": float(pose[0]), "y": float(pose[1]), "z": float(pose[2])},
                            "orientation": {"roll": float(pose[3]), "pitch": float(pose[4]), "yaw": float(pose[5])},
                            "length": float(length),
                            "radius": float(radius)
                        }
                        cylinders_data.append(cylinder_info)
                    print(f"已准备 {len(cylinders_data)} 个圆柱体的数据")
                    
                    # 添加元数据
                    save_data = {
                        "timestamp": timestamp,
                        "total_cylinders": len(added_cylinders),
                        "cylinders": cylinders_data
                    }
                    
                    # 保存到JSON文件
                    with open(save_path, 'w') as f:
                        json.dump(save_data, f, indent=4)
                    
                    print(f"已成功保存 {len(added_cylinders)} 个圆柱体到文件: {save_path}")
                except Exception as e:
                    import traceback
                    print(f"保存圆柱体数据时出错:")
                    print(f"错误类型: {type(e).__name__}")
                    print(f"错误信息: {str(e)}")
                    print("错误详情:")
                    traceback.print_exc()
            else:
                print("没有圆柱体可以保存")

        # --- Save YAML button logic ---
        if current_save_yaml_value > prev_save_yaml_value:
            print(f"YAML保存按钮被点击!")
            prev_save_yaml_value = current_save_yaml_value
            if added_cylinders:
                save_scene_to_yaml(rb, added_cylinders)
            else:
                print("没有圆柱体可以保存到YAML")

        # Revert button logic
        if current_revert_button_value > prev_revert_button_value:
            print("撤销上一个圆柱体...")
            prev_revert_button_value = current_revert_button_value  # 立即更新按钮值
            if added_cylinders:
                # 删除最新添加的圆柱体
                last_cylinder_id, _, _, _ = added_cylinders.pop()
                try:
                    p.removeBody(last_cylinder_id)
                    print(f"已移除圆柱体 {last_cylinder_id}。剩余: {len(added_cylinders)}")
                    
                    # 如果还有剩余的圆柱体，将滑块位置设置为当前最后一个圆柱体的参数
                    if added_cylinders:
                        # 获取最新的圆柱体参数
                        _, new_last_pose, new_last_length, new_last_radius = added_cylinders[-1]
                        
                        # 移除所有UI控件
                        remove_all_ui_controls()
                        
                        # 重新创建滑块，使用最后一个圆柱体的参数作为初始值
                        sliders = create_sliders(
                            init_pos=new_last_pose[0:3],
                            init_orient=new_last_pose[3:6],
                            init_length=new_last_length,
                            init_radius=new_last_radius
                        )
                        
                        # 更新按钮值
                        prev_add_button_value = p.readUserDebugParameter(sliders['add'])
                        prev_revert_button_value = p.readUserDebugParameter(sliders['revert'])
                        prev_save_button_value = p.readUserDebugParameter(sliders['save'])
                        prev_save_yaml_value = p.readUserDebugParameter(sliders['save_yaml'])
                        
                        # 注意：PyBullet不直接支持设置滑块的值，以下是模拟用户输入的方法
                        # 由于缺乏直接设置方法，这里只提示用户
                        print("已重新创建UI控件，使用最后一个圆柱体的参数:")
                        print(f"位置: x={new_last_pose[0]:.2f}, y={new_last_pose[1]:.2f}, z={new_last_pose[2]:.2f}")
                        print(f"朝向: roll={new_last_pose[3]:.2f}, pitch={new_last_pose[4]:.2f}, yaw={new_last_pose[5]:.2f}")
                        print(f"尺寸: length={new_last_length:.2f}, radius={new_last_radius:.2f}")
                    else:
                        # 如果没有剩余的圆柱体，重新创建使用默认值的滑块
                        # 移除所有UI控件
                        remove_all_ui_controls()
                        
                        sliders = create_sliders() # 使用默认值
                        
                        # 更新按钮值
                        prev_add_button_value = p.readUserDebugParameter(sliders['add'])
                        prev_revert_button_value = p.readUserDebugParameter(sliders['revert'])
                        prev_save_button_value = p.readUserDebugParameter(sliders['save'])
                        prev_save_yaml_value = p.readUserDebugParameter(sliders['save_yaml'])
                        
                        print("没有剩余的圆柱体，已重置滑块为默认值")
                except Exception as e:
                    print(f"移除圆柱体时出错: {e}")
            else:
                print("没有圆柱体可以撤销")

        # Update previous button values
        prev_add_button_value = current_add_button_value
        if current_save_button_value != prev_save_button_value:
            prev_save_button_value = current_save_button_value
        if current_save_yaml_value != prev_save_yaml_value:
            prev_save_yaml_value = current_save_yaml_value
        
        # --- Slider Logic (Update Pose of Last Cylinder) ---
        if added_cylinders:
            last_cylinder_id, last_pose_list, last_length, last_radius = added_cylinders[-1]
            # 直接设置位姿，无需比较是否改变
            # 这样保证了最灵敏的响应
            try:
                # 直接更新位置，不做额外检查以确保更好的响应性
                pp.set_pose(last_cylinder_id, current_pose_pb)
                # 更新存储的位姿
                added_cylinders[-1] = (last_cylinder_id, current_pose_list, length, radius)
            except p.error as e:
                # 处理可能的错误（比如物体已被删除）
                print(f"更新位姿时出错: {e}，从列表中移除该物体")
                added_cylinders = [c for c in added_cylinders if c[0] != last_cylinder_id]

        # Keep the simulation running and responsive
        # p.stepSimulation() # Optional: only if physics is needed
        time.sleep(1.0 / 1000.0)  # 更高的刷新率提供更流畅的交互体验