import math
import os
import random
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union

import casadi as ca
import numpy as np
import pybullet as p
import pybullet_planning as pp
import torch
from scipy.interpolate import splev, splprep

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from model.data_loader import SceneDataLoader
from model.multibrach_model import MultiPathTrajectoryNetwork
from model.scene_parse import SceneParser
from motion_planner.svsdf import SDF
from multi_tangent.collision import create_collision_bodies
from robot.robot_setup import RobotSetup
from utils.collision import element_collision_info, init_pb
from utils.params import URDF_PATH


def predict_trajectory(model, grasped_point_cloud, env_point_cloud, start_joints, target_joints, grasp_offset):
    """
    使用模型预测轨迹

    Args:
        model: 加载的模型
        grasped_point_cloud: 抓取物体的点云 [batch_size, num_points, 3+3]
        env_point_cloud: 环境点云 [batch_size, num_points, 3+3]
        start_joints: 起始关节角 [batch_size, 6]
        target_joints: 目标关节角 [batch_size, 6]
        grasp_offset: 抓握位置偏移 [batch_size, 3]

    Returns:
        np.ndarray: 预测的轨迹
    """

    # 转换为张量并添加批次维度(如果需要)
    if len(grasped_point_cloud.shape) == 2:
        grasped_point_cloud = grasped_point_cloud.unsqueeze(0)
    if len(env_point_cloud.shape) == 2:
        env_point_cloud = env_point_cloud.unsqueeze(0)
    if len(start_joints.shape) == 1:
        start_joints = start_joints.unsqueeze(0)
    if len(target_joints.shape) == 1:
        target_joints = target_joints.unsqueeze(0)
    if len(grasp_offset.shape) == 1:
        grasp_offset = grasp_offset.unsqueeze(0)
    
    # 将所有输入数据移动到与模型相同的设备上
    grasped_point_cloud = grasped_point_cloud.to(device)
    env_point_cloud = env_point_cloud.to(device)
    start_joints = start_joints.to(device)
    target_joints = target_joints.to(device)
    grasp_offset = grasp_offset.to(device)
    
    with torch.no_grad():
        # 预测轨迹
        pred_trajectory = model(grasped_point_cloud, env_point_cloud, start_joints, target_joints, grasp_offset)
    
    # 转换为numpy数组
    return pred_trajectory.cpu().numpy()


if __name__ == "__main__":
    # 设置随机种子确保结果可重现
    random.seed(42)
    np.random.seed(42)

    # 初始化PyBullet环境
    init_pb()

    # 模型路径
    model_path = "/home/jeong/summer_research/eth_ws/src/husky_assembly/scripts/model/results/trajectory_model_20250331_214519/best_model.pth"

    # 载入模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = MultiPathTrajectoryNetwork(object_feature_dim=512, env_feature_dim=512, task_feature_dim=512, traj_feature_dim=512, fusion_dim=1024, output_seq_len=2048, joint_dim=6, normal_channel=True, use_lstm=False, dropout=0.5).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 加载场景
    scene_name = "cuboid_1"
    task_name = "task_1"
    algorithm_name = "BIRRT"
    scene_file = os.path.join(HERE, "model", "scenes", scene_name, f"{task_name}.yml")

    # 使用SceneParser解析场景
    scene_parser = SceneParser(scene_file)
    scene_parser.load_scene()

    # 获取场景中的元素信息
    line_pts, radius_per_edge = scene_parser.get_element_info()
    bodies = create_collision_bodies(line_pts, radius_per_edge, viewer=True)

    # 获取机器人配置信息
    start_q = np.array(scene_parser.get_robot_start_pose())
    target_q = np.array(scene_parser.get_robot_target_pose())
    pose_2d = scene_parser.get_robot_pose_2d(output_type="array")
    grasp_offset = np.array(scene_parser.get_robot_grasp_offset())

    # 使用DataLoader加载点云数据
    data_loader = SceneDataLoader()

    # 准备点云数据
    num_points = 1024  # 环境点云的点数
    num_grasp_points = 256  # 抓取物体的点云点数

    # 创建点云数据集
    dataset = data_loader.create_dataset(scene_names=scene_name, task_names=task_name, algorithm_names=algorithm_name, num_points=num_points, num_grasp_points=num_grasp_points, normal_channel=True)

    # 如果数据集为空，则手动生成点云
    if len(dataset) == 0:
        print("警告：数据集为空，正在手动生成点云数据...")
        # 获取环境点云
        env_point_cloud, _ = data_loader._sample_points_from_elements(line_pts, radius_per_edge, num_points=num_points, normal_channel=True)

        # 获取抓取物体点云
        grasped_point_cloud, _ = data_loader._sample_points_from_elements([np.array([0, 0, 0]), np.array([0, 0, 1])], [0.01], num_points=num_grasp_points, normal_channel=True)

        # 转换为张量
        env_point_cloud = torch.from_numpy(env_point_cloud).float()
        grasped_point_cloud = torch.from_numpy(grasped_point_cloud).float()
        start_joints = torch.from_numpy(start_q).float()
        target_joints = torch.from_numpy(target_q).float()
        grasp_offset_tensor = torch.from_numpy(grasp_offset).float()
    else:
        # 从数据集获取第一个样本
        sample = dataset[2]
        env_point_cloud = sample["point_cloud"]
        grasped_point_cloud = sample["grasped_point_cloud"]
        start_joints = sample["robot_start_pose"]
        target_joints = sample["robot_target_pose"]
        grasp_offset_tensor = sample["grasp_offset"]

    # 使用模型预测轨迹
    print("正在预测轨迹...")
    predicted_trajectory = predict_trajectory(model, grasped_point_cloud, env_point_cloud, start_joints, target_joints, grasp_offset_tensor)

    # 打印轨迹信息
    print(f"预测轨迹形状: {predicted_trajectory.shape}")

    # 设置机器人
    rb = RobotSetup("rb")
    rb.set_joint_positions(rb.arm_joints, start_q)
    rb.set_base_pose_2d(pose_2d[0], pose_2d[1], pose_2d[2])

    # 创建被抓取元素
    line_pts_grasped = [np.array([0, 0, 0]), np.array([0, 0, 1])]
    grasped_element = create_collision_bodies(line_pts_grasped, [0.01], viewer=True)[0]
    pp.set_pose(grasped_element, pp.multiply(pp.get_link_pose(rb.robot, rb.tool_link), pp.Pose(point=grasp_offset, euler=pp.Euler(1.5708, 0, 0))))
    grasped_attachment = pp.create_attachment(rb.robot, rb.tool_link, grasped_element)
    rb.update_attachments([grasped_attachment])

    # 可视化预测轨迹
    print("正在可视化预测轨迹...")
    traj = predicted_trajectory[0]  # 取批次中的第一个轨迹

    pp.wait_for_user("按回车键开始可视化预测轨迹...")

    # -------------------- 下面是使用pybullet进行可视化的代码 --------------------#
    slider = p.addUserDebugParameter("replay", 0, 1, 0)

    for body in bodies:
        pp.set_color(body, [1, 0, 0, 1])

    while True:
        slider_value = p.readUserDebugParameter(slider)
        time_idx = int(slider_value * (traj.shape[0] - 1))
        joint_val = traj[time_idx]
        rb.set_joint_positions(rb.arm_joints, joint_val)  # Use rb here for visualization
        time.sleep(1.0 / 60)
