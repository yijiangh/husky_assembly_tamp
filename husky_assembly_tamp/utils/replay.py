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
from scipy.interpolate import CubicSpline

from husky_assembly_tamp.model.scene_parse import SceneParser
from multi_tangent.collision import create_collision_bodies
from husky_assembly_tamp.robot.robot_setup import RobotSetup
from husky_assembly_tamp.utils.collision import init_pb
from husky_assembly_tamp.utils.params import HERE, LOG_DIR


class TrajectoryPlayer:
    """轨迹回放系统，用于读取和播放规划轨迹"""

    def __init__(self, scene_name: str, task_name: str, algorithm_name: str):
        """
        初始化轨迹播放器

        Args:
            scene_name: 场景名称
            task_name: 任务名称
            algorithm_name: 算法名称
        """
        self.scene_name = scene_name
        self.task_name = task_name
        self.algorithm_name = algorithm_name

        # 轨迹数据
        self.trajectories = []
        self.trajectory_names = []  # 存储轨迹文件名
        self.current_trajectory_idx = 0
        self.current_frame_idx = 0
        self.is_playing = False
        self.is_paused = False
        self.play_speed = 1.0
        self.smoothed_trajectory = None  # 存储平滑后的轨迹

        # 设置数据路径
        self.data_dir = LOG_DIR
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
        timestamp_dirs = [d for d in os.listdir(corner_case_dir) if os.path.isdir(os.path.join(corner_case_dir, d))]

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

        # 获取场景元素信息
        self.element_bodies, self.element_infos = self.parser.create_elements(color=[1, 0, 0, 1])

        # 设置机器人
        self.rb = self.parser.create_robot("r0")

        self.grasped_element, self.grasp_attachment = self.parser.create_attachment(self.rb)
        self.rb.update_attachments(self.grasp_attachment)
        self.collision_fn = self.rb.create_collision_fn(self.element_bodies)

    def _load_trajectories(self):
        """加载指定路径下的所有轨迹文件"""
        # 使用新的路径结构: LOG_DIR/corner_case/time_stamp/scene/task/algorithm
        traj_dir = os.path.join(LOG_DIR, self.scene_name, self.task_name, self.algorithm_name)

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
                return float("inf")

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
        self.smooth_button = p.addUserDebugParameter("Smooth", 1, 0, 0)  # 添加Smooth按钮
        self.save_button = p.addUserDebugParameter("Save", 1, 0, 0)  # 添加Save按钮

        # 存储按钮状态
        self.prev_play_value = p.readUserDebugParameter(self.play_button)
        self.prev_pause_value = p.readUserDebugParameter(self.pause_button)
        self.prev_prev_value = p.readUserDebugParameter(self.prev_button)
        self.prev_next_value = p.readUserDebugParameter(self.next_button)
        self.prev_verbose_value = p.readUserDebugParameter(self.verbose_button)
        self.prev_smooth_value = p.readUserDebugParameter(self.smooth_button)
        self.prev_save_value = p.readUserDebugParameter(self.save_button)

    def _print_status(self):
        """在终端中打印当前状态信息"""
        # 获取当前轨迹的文件名
        current_traj_name = self.trajectory_names[self.current_trajectory_idx]

        status_text = (
            f"\033[K"  # 清除当前行
            f"Scene: {self.scene_name}, Task: {self.task_name}, Algorithm: {self.algorithm_name}\n\033[K"
            f"Trajectory: {self.current_trajectory_idx + 1}/{len(self.trajectories)} [{current_traj_name}]\n\033[K"
            f"Frame: {self.current_frame_idx}/{len(self.trajectories[self.current_trajectory_idx])}"
        )

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

        # 检查Smooth按钮
        current_smooth_value = p.readUserDebugParameter(self.smooth_button)
        if current_smooth_value != self.prev_smooth_value:
            self.prev_smooth_value = current_smooth_value
            self._smooth_trajectory()

        # 检查Save按钮
        current_save_value = p.readUserDebugParameter(self.save_button)
        if current_save_value != self.prev_save_value:
            self.prev_save_value = current_save_value
            self._save_smoothed_trajectory()

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

    def _smooth_trajectory(self):
        """平滑当前轨迹"""
        if not self.trajectories:
            print("没有轨迹可以平滑")
            return

        print("正在平滑轨迹...")
        trajectory = self.trajectories[self.current_trajectory_idx]
        
        try:
            # 步骤1: 轨迹shortcutting
            print("1. 正在执行轨迹shortcutting...")
            shortcut_trajectory = self._shortcut_trajectory(trajectory)
            print(f"   Shortcutting完成, 帧数: {len(shortcut_trajectory)}")
            
            # 步骤2: 插值到10000帧
            print("2. 正在插值到10000帧...")
            interpolated_trajectory = self._interpolate_trajectory(shortcut_trajectory, 10000)
            print(f"   插值完成, 帧数: {len(interpolated_trajectory)}")
            
            # # 步骤3: 平滑轨迹
            # print("3. 正在平滑轨迹...")
            # smoothed_trajectory = self._apply_smoothing(interpolated_trajectory)
            # print(f"   平滑完成, 帧数: {len(smoothed_trajectory)}")
            
            # # 步骤4: 碰撞检查
            # print("4. 正在进行碰撞检查...")
            # collision_free = self._check_trajectory_collision(smoothed_trajectory)
            
            # # 根据碰撞检查结果选择最终轨迹
            # if collision_free:
            #     print("   碰撞检查通过，使用平滑后的轨迹")
            #     final_trajectory = smoothed_trajectory
            # else:
            #     print("   碰撞检查未通过，使用插值轨迹（未平滑）")
            #     final_trajectory = interpolated_trajectory
                
            final_trajectory = interpolated_trajectory
            
            # 存储平滑后的轨迹
            self.smoothed_trajectory = final_trajectory
            
            # 临时替换当前轨迹以便预览
            self.is_playing = False
            self.current_frame_idx = 0
            original_trajectory = self.trajectories[self.current_trajectory_idx]
            self.trajectories[self.current_trajectory_idx] = self.smoothed_trajectory
            
            print(f"轨迹处理完成。原始轨迹帧数: {len(original_trajectory)}, 最终轨迹帧数: {len(self.smoothed_trajectory)}")
            print("现在可以使用Save按钮保存处理后的轨迹，或切换到其他轨迹放弃更改")
            self._print_status()
            
            # 显示处理后轨迹的初始位置
            self._set_robot_pose(self.smoothed_trajectory[0])
            
        except Exception as e:
            print(f"处理轨迹时出错: {e}")
            import traceback
            traceback.print_exc()

    def _shortcut_trajectory(self, trajectory):
        """对轨迹进行shortcutting，使用随机采样方法"""
        if len(trajectory) <= 2:
            return trajectory
            
        print("   执行随机shortcutting算法...")
        
        # 复制原始轨迹
        result = np.copy(trajectory)
        num_points = len(result)
        
        # 最大迭代次数
        max_iterations = 100
        successful_shortcuts = 0
        
        for iteration in range(max_iterations):
            if iteration % 10 == 0:
                print(f"   Shortcutting迭代: {iteration}/{max_iterations}, 成功次数: {successful_shortcuts}")
            
            # 随机选择两个点（确保起点索引小于终点索引）
            start_idx = np.random.randint(0, num_points - 2)
            end_idx = np.random.randint(start_idx + 2, num_points)
            
            # 获取这两个点的配置
            start_conf = result[start_idx]
            end_conf = result[end_idx]
            
            # 检查直线路径是否无碰撞
            if self._is_collision_free(start_conf, end_conf):
                # 如果无碰撞，则移除中间点
                new_path = []
                new_path.extend(result[:start_idx + 1])  # 包括起点
                new_path.append(end_conf)  # 添加终点
                new_path.extend(result[end_idx + 1:])  # 添加终点之后的点
                
                # 更新结果
                result = np.array(new_path)
                num_points = len(result)
                successful_shortcuts += 1
        
        print(f"   Shortcutting完成: {len(trajectory)} -> {len(result)} 帧, 成功次数: {successful_shortcuts}")
        return result

    def _interpolate_trajectory(self, trajectory, num_points):
        """将轨迹插值到指定数量的点，使用基于距离的线性插值"""
        if len(trajectory) <= 1:
            return trajectory
            
        # 计算相邻构型之间的距离
        distances = []
        total_distance = 0.0
        
        for i in range(1, len(trajectory)):
            dist = np.linalg.norm(trajectory[i] - trajectory[i-1])
            distances.append(dist)
            total_distance += dist
        
        # 基于距离创建非均匀时间参数
        t_original = [0.0]
        cumulative_dist = 0.0
        
        for dist in distances:
            cumulative_dist += dist
            t_original.append(cumulative_dist / total_distance)
        
        # 确保t_original是numpy数组
        t_original = np.array(t_original)
        
        # 创建新的时间序列，用于目标点数
        t_new = np.linspace(0, 1, num_points)
        
        # 对每个关节单独进行插值
        joint_count = trajectory.shape[1]
        interpolated = np.zeros((num_points, joint_count))
        
        for joint_idx in range(joint_count):
            joint_values = trajectory[:, joint_idx]
            
            # 使用线性插值
            interpolated[:, joint_idx] = np.interp(t_new, t_original, joint_values)
        
        print(f"   插值完成: 原始轨迹点数={len(trajectory)}, 基于距离插值后点数={num_points}")
        return interpolated

    def _apply_smoothing(self, trajectory):
        """对轨迹应用平滑处理"""
        # 创建时间序列
        t = np.linspace(0, 1, len(trajectory))
        
        # 对每个关节单独进行平滑插值
        joint_count = trajectory.shape[1]
        smoothed = np.zeros_like(trajectory)
        
        for joint_idx in range(joint_count):
            joint_values = trajectory[:, joint_idx]
            
            # 创建平滑样条
            cs = CubicSpline(t, joint_values)
            
            # 用相同数量的点重新采样
            smoothed[:, joint_idx] = cs(t)
        
        return smoothed

    def _is_collision_free(self, conf1, conf2):
        """检查两个构型之间的直线路径是否无碰撞"""
        # 获取两个构型之间的插值点数量
        # 这里简单地根据构型间距离来确定测试点数量
        distance = np.linalg.norm(conf2 - conf1)
        num_checks = 1000
        
        # 检查插值路径上的点
        for i in range(num_checks):
            t = i / (num_checks - 1)
            conf = conf1 * (1 - t) + conf2 * t
            
            # 检查碰撞
            if self.collision_fn(conf):
                return False
        
        return True

    def _check_trajectory_collision(self, trajectory):
        """检查整个轨迹是否无碰撞"""
        # 保存当前机器人状态
        original_joint_positions = pp.get_joint_positions(self.rb.robot, self.rb.arm_joints)
        
        # 为了提高效率，只检查轨迹中的部分点
        num_frames = len(trajectory)
        step = max(1, num_frames // 100)  # 最多检查100个点
        
        collision_free = True
        
        try:
            for i in range(0, num_frames, step):
                # 检查碰撞
                if self.collision_fn(trajectory[i]):
                    print(f"   在轨迹帧 {i} 处检测到碰撞")
                    collision_free = False
                    break
        finally:
            # 恢复原始位姿
            self.rb.set_joint_positions(self.rb.arm_joints, original_joint_positions)
        
        return collision_free

    def _save_smoothed_trajectory(self):
        """保存平滑后的轨迹"""
        if self.smoothed_trajectory is None:
            print("没有平滑后的轨迹可以保存，请先使用Smooth按钮")
            return
            
        try:
            # 获取当前轨迹文件名和路径
            current_traj_name = self.trajectory_names[self.current_trajectory_idx]
            file_name_no_ext = os.path.splitext(current_traj_name)[0]
            smooth_file_name = f"{file_name_no_ext}_smooth.npy"
            
            # 构建保存路径
            traj_dir = os.path.join(LOG_DIR, self.scene_name, self.task_name, self.algorithm_name)
            save_path = os.path.join(traj_dir, smooth_file_name)
            
            # 保存平滑后的轨迹
            np.save(save_path, self.smoothed_trajectory)
            
            print(f"已保存平滑后的轨迹: {smooth_file_name}")
            
            # 重新加载所有轨迹，以便包含新保存的轨迹
            self._load_trajectories()
            
            # 尝试切换到新保存的轨迹
            for i, name in enumerate(self.trajectory_names):
                if name == smooth_file_name:
                    self.current_trajectory_idx = i
                    self.current_frame_idx = 0
                    self._set_robot_pose(self.trajectories[i][0])
                    print(f"已切换到平滑后的轨迹: {smooth_file_name}")
                    self._print_status()
                    break
                    
        except Exception as e:
            print(f"保存平滑后的轨迹时出错: {e}")


def main():
    """主函数，解析命令行参数并启动轨迹播放器"""
    parser = argparse.ArgumentParser(description="轨迹回放系统")
    parser.add_argument("--scene", type=str, default="cuboid_1", help="场景名称")
    parser.add_argument("--task", type=str, default="task_1", help="任务名称")
    parser.add_argument("--algorithm", type=str, default="BIRRT", help="算法名称")

    args = parser.parse_args()

    try:
        player = TrajectoryPlayer(args.scene, args.task, args.algorithm)
        player.run()
    except KeyboardInterrupt:
        print("\n用户中断，退出程序")


if __name__ == "__main__":
    main()
