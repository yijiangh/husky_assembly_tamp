#!/usr/bin/env python3
import argparse
import glob
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pybullet as p
import pybullet_planning as pp

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from model.scene_parse import SceneParser
from multi_tangent.collision import create_collision_bodies
from robot.robot_setup import RobotSetup
from utils.collision import init_pb
from utils.params import HERE, LOG_DIR


class TrajectoryPlayer:
    """轨迹回放系统，用于读取和播放规划轨迹"""

    def __init__(self, scene_name: str, task_name: str, algorithm_name: str, time_stamp: Optional[str] = None):
        """
        初始化轨迹播放器
        
        Args:
            scene_name: 场景名称
            task_name: 任务名称
            algorithm_name: 算法名称
            time_stamp: 时间戳，如果为None则自动查找最新的
        """
        self.scene_name = scene_name
        self.task_name = task_name
        self.algorithm_name = algorithm_name
        self.time_stamp = self._find_latest_timestamp() if time_stamp is None else time_stamp
        
        # 轨迹数据
        self.trajectories = []
        self.trajectory_names = []  # 存储轨迹文件名
        self.current_trajectory_idx = 0
        self.current_frame_idx = 0
        self.is_playing = False
        self.is_paused = False
        self.play_speed = 1.0
        
        # 设置数据路径
        self.data_dir = os.path.join(HERE, "model", "data")
        self.scenes_dir = os.path.join(HERE, "model", "scenes")
        
        # 初始化PyBullet
        init_pb()
        
        # 加载场景和轨迹
        self._load_scene()
        self._load_trajectories()
        
        # 设置UI控制
        self._setup_ui()
        
        # 打印当前状态
        self._print_status()
    
    def _find_latest_timestamp(self) -> str:
        """查找logs/corner_case目录下最新的时间戳文件夹"""
        corner_case_dir = os.path.join(LOG_DIR, "corner_case")
        
        if not os.path.exists(corner_case_dir):
            raise FileNotFoundError(f"未找到轨迹目录: {corner_case_dir}")
        
        # 获取所有时间戳目录
        timestamp_dirs = [d for d in os.listdir(corner_case_dir) 
                          if os.path.isdir(os.path.join(corner_case_dir, d))]
        
        if not timestamp_dirs:
            raise FileNotFoundError(f"在 {corner_case_dir} 中未找到时间戳目录")
        
        # 按时间戳字符串排序（格式应为YYYYMMDD_HHMMSS）
        timestamp_dirs.sort(reverse=True)  # 降序排列，最新的在前
        latest_timestamp = timestamp_dirs[0]
        
        print(f"自动选择最新时间戳: {latest_timestamp}")
        return latest_timestamp
    
    def _load_scene(self):
        """加载场景配置和机器人模型"""
        print(f"\n加载场景: {self.scene_name}/{self.task_name}")
        
        # 加载场景配置
        scene_file = os.path.join(self.scenes_dir, self.scene_name, f"{self.task_name}.yml")
        if not os.path.exists(scene_file):
            raise FileNotFoundError(f"场景配置文件不存在: {scene_file}")
        
        self.parser = SceneParser(scene_file)
        self.parser._load_scene()
        
        # 获取场景元素信息
        line_pts, radius_per_edge = self.parser.get_element_info()
        self.element_bodies = create_collision_bodies(line_pts, radius_per_edge, viewer=True)
        
        # 获取机器人配置
        self.grasp_offset = self.parser.get_robot_grasp_offset()
        self.pose_2d = self.parser.get_robot_pose_2d(output_type="array")
        
        # 设置机器人
        self.rb = RobotSetup("r0")
        pp.set_pose(self.rb.robot, pp.Pose(point=[self.pose_2d[0], self.pose_2d[1], 0], euler=pp.Euler(0, 0, self.pose_2d[2])))
        
        # 设置抓取物体
        line_pts_grasped = [np.array([0, 0, 0]), np.array([0, 0, 1])]
        self.grasped_element = create_collision_bodies(line_pts_grasped, [0.01], viewer=True)[0]
        pp.set_pose(self.grasped_element, pp.multiply(pp.get_link_pose(self.rb.robot, self.rb.tool_link), pp.Pose(point=self.grasp_offset, euler=pp.Euler(1.5708, 0, 0))))
        self.grasp_attachment = pp.create_attachment(self.rb.robot, self.rb.tool_link, self.grasped_element)
        self.rb.update_attachments([self.grasp_attachment])
    
    def _load_trajectories(self):
        """加载指定路径下的所有轨迹文件"""
        # 使用新的路径结构: LOG_DIR/corner_case/time_stamp/scene/task/algorithm
        traj_dir = os.path.join(LOG_DIR, "corner_case", self.time_stamp, 
                               self.scene_name, self.task_name, self.algorithm_name)
        
        if not os.path.exists(traj_dir):
            raise FileNotFoundError(f"轨迹目录不存在: {traj_dir}")
        
        # 查找所有轨迹文件（*.npy文件，不再限定于plan_*.npy）
        traj_files = sorted(glob.glob(os.path.join(traj_dir, "*.npy")))
        
        if not traj_files:
            raise FileNotFoundError(f"在 {traj_dir} 中未找到轨迹文件")
        
        print(f"找到 {len(traj_files)} 个轨迹文件")
        
        # 定义一个函数来提取文件名中的索引
        def get_file_index(file_path):
            file_name = os.path.basename(file_path)
            file_name_no_ext = os.path.splitext(file_name)[0]
            try:
                # 尝试从文件名中提取索引
                if file_name.startswith("plan_"):
                    return int(file_name_no_ext.split("_")[1])
                else:
                    return int(file_name_no_ext)
            except (ValueError, IndexError):
                # 如果无法提取索引，默认返回最大值
                return float('inf')
        
        # 按索引排序轨迹文件
        traj_files.sort(key=get_file_index)
        
        # 加载所有轨迹
        for traj_file in traj_files:
            try:
                trajectory = np.load(traj_file)
                file_name = os.path.basename(traj_file)
                
                self.trajectories.append(trajectory)
                self.trajectory_names.append(file_name)
                
                print(f"已加载: {file_name}, 帧数: {len(trajectory)}")
            except Exception as e:
                print(f"无法加载轨迹文件 {traj_file}: {e}")
        
        if not self.trajectories:
            raise ValueError("未能加载任何轨迹文件")
    
    def _setup_ui(self):
        """设置用户界面控件"""
        # 移除所有现有控件
        p.removeAllUserParameters()
        
        # 添加控制按钮和滑块
        self.play_button = p.addUserDebugParameter("Play", 1, 0, 0)
        self.pause_button = p.addUserDebugParameter("Pause", 1, 0, 0)
        self.prev_button = p.addUserDebugParameter("Previous", 1, 0, 0)
        self.next_button = p.addUserDebugParameter("Next", 1, 0, 0)
        self.speed_slider = p.addUserDebugParameter("Speed", 1.0, 50.0, 1.0)
        self.verbose_button = p.addUserDebugParameter("Verbose", 1, 0, 0)  # 添加Verbose按钮
        
        # 存储按钮状态
        self.prev_play_value = p.readUserDebugParameter(self.play_button)
        self.prev_pause_value = p.readUserDebugParameter(self.pause_button)
        self.prev_prev_value = p.readUserDebugParameter(self.prev_button)
        self.prev_next_value = p.readUserDebugParameter(self.next_button)
        self.prev_verbose_value = p.readUserDebugParameter(self.verbose_button)
    
    def _print_status(self):
        """在终端中打印当前状态信息"""
        # 获取当前轨迹的文件名
        current_traj_name = self.trajectory_names[self.current_trajectory_idx]
        
        status_text = (f"\033[K"  # 清除当前行
                    f"Scene: {self.scene_name}, Task: {self.task_name}, Algorithm: {self.algorithm_name}\n\033[K"
                    f"Timestamp: {self.time_stamp}\n\033[K"
                    f"Trajectory: {self.current_trajectory_idx + 1}/{len(self.trajectories)} [{current_traj_name}]\n\033[K"
                    f"Frame: {self.current_frame_idx}/{len(self.trajectories[self.current_trajectory_idx])}")
        
        print(f"\r{status_text}", end="\n", flush=True)
    
    def _print_joint_angles(self):
        """打印当前机器人的关节角度"""
        # 获取当前关节角度
        joint_angles = pp.get_joint_positions(self.rb.robot, self.rb.arm_joints)
        
        # 获取关节名称
        joint_names = pp.get_joint_names(self.rb.robot, self.rb.arm_joints)
        
        print("\n当前关节角度:")
        print("-" * 30)
        for i, (name, angle) in enumerate(zip(joint_names, joint_angles)):
            # 将弧度转换为度数显示
            angle_deg = np.degrees(angle)
            print(f"{i+1}. {name}: {angle:.4f} rad ({angle_deg:.2f}°)")
        print("-" * 30)
    
    def run(self):
        """运行轨迹播放器主循环"""
        print("\n轨迹播放器已启动。使用界面按钮控制播放。")
        
        # 显示初始位置
        self._set_robot_pose(self.trajectories[self.current_trajectory_idx][0])
        
        try:
            last_status_update = time.time()
            while True:
                # 检查UI按钮状态
                self._check_ui_controls()
                
                # 播放轨迹
                if self.is_playing and not self.is_paused:
                    trajectory = self.trajectories[self.current_trajectory_idx]
                    
                    if self.current_frame_idx < len(trajectory):
                        # 设置机器人位姿
                        self._set_robot_pose(trajectory[self.current_frame_idx])
                        
                        # 更新状态显示（限制刷新频率，避免终端刷新过快）
                        current_time = time.time()
                        if current_time - last_status_update > 0.2:  # 每0.2秒更新一次状态
                            self._print_status()
                            last_status_update = current_time
                        
                        # 固定步进一帧
                        self.current_frame_idx += 1
                        
                        # 检查是否播放完毕
                        if self.current_frame_idx >= len(trajectory):
                            self.is_playing = False
                            self.current_frame_idx = 0
                            self._print_status()
                            print("播放完成")
                    
                    # 根据速度值计算延迟时间 - 速度越大，延迟越小
                    delay = 1.0 / self.play_speed * 0.01  # 基准延迟0.01秒除以速度
                    time.sleep(delay)  # 播放延迟
                else:
                    # 非播放状态的延迟
                    time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n退出轨迹播放器")
    
    def _check_ui_controls(self):
        """检查UI控件状态并响应"""
        # 检查播放按钮
        current_play_value = p.readUserDebugParameter(self.play_button)
        if current_play_value != self.prev_play_value:
            self.prev_play_value = current_play_value
            self.is_playing = True
            self.is_paused = False
            print("开始播放")
            self._print_status()
        
        # 检查暂停按钮
        current_pause_value = p.readUserDebugParameter(self.pause_button)
        if current_pause_value != self.prev_pause_value:
            self.prev_pause_value = current_pause_value
            self.is_paused = not self.is_paused
            print("暂停播放" if self.is_paused else "恢复播放")
            self._print_status()
        
        # 检查上一个按钮
        current_prev_value = p.readUserDebugParameter(self.prev_button)
        if current_prev_value != self.prev_prev_value:
            self.prev_prev_value = current_prev_value
            self._switch_trajectory(-1)
        
        # 检查下一个按钮
        current_next_value = p.readUserDebugParameter(self.next_button)
        if current_next_value != self.prev_next_value:
            self.prev_next_value = current_next_value
            self._switch_trajectory(1)
        
        # 检查Verbose按钮
        current_verbose_value = p.readUserDebugParameter(self.verbose_button)
        if current_verbose_value != self.prev_verbose_value:
            self.prev_verbose_value = current_verbose_value
            # 打印当前关节角度
            self._print_joint_angles()
        
        # 检查速度滑块
        self.play_speed = p.readUserDebugParameter(self.speed_slider)
    
    def _switch_trajectory(self, direction: int):
        """
        切换到下一个或上一个轨迹
        
        Args:
            direction: 方向，1表示下一个，-1表示上一个
        """
        new_idx = self.current_trajectory_idx + direction
        
        if 0 <= new_idx < len(self.trajectories):
            self.current_trajectory_idx = new_idx
            self.current_frame_idx = 0
            self.is_playing = False
            
            # 显示新轨迹的初始位置
            self._set_robot_pose(self.trajectories[self.current_trajectory_idx][0])
            
            print(f"切换到轨迹 {self.current_trajectory_idx + 1}/{len(self.trajectories)} [{self.trajectory_names[new_idx]}]")
            self._print_status()
    
    def _set_robot_pose(self, conf: np.ndarray):
        """
        设置机器人位姿
        
        Args:
            conf: 关节配置
        """
        self.rb.set_joint_positions(self.rb.arm_joints, conf)
        self.grasp_attachment.assign()


def main():
    """主函数，解析命令行参数并启动轨迹播放器"""
    parser = argparse.ArgumentParser(description="轨迹回放系统")
    parser.add_argument("--scene", type=str, default="cuboid_1", help="场景名称")
    parser.add_argument("--task", type=str, default="task_1", help="任务名称")
    parser.add_argument("--algorithm", type=str, default="BIRRT", help="算法名称")
    parser.add_argument("--timestamp", type=str, help="时间戳(YYYYMMDD_HHMMSS格式)，如果未提供则使用最新的")
    
    args = parser.parse_args()
    
    try:
        player = TrajectoryPlayer(args.scene, args.task, args.algorithm, args.timestamp)
        player.run()
    except KeyboardInterrupt:
        print("\n用户中断，退出程序")
    except Exception as e:
        print(f"\n发生错误: {e}")


if __name__ == "__main__":
    main()
