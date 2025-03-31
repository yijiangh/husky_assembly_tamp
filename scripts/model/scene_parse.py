import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple, Union

import numpy as np
import pybullet_planning as pp
import yaml
from scipy.spatial.transform import Rotation
import glob
import time
import pybullet as p
import argparse

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from multi_tangent.collision import create_collision_bodies
from utils.collision import init_pb
from utils.params import *


class SceneParser:
    """
    A class for parsing and visualizing 3D scene data from YAML files.

    This class handles loading scene data, approximating geometric shapes with spheres,
    and visualizing the scene using PyBullet.
    """

    def __init__(self, scene_file: str):
        """
        Initialize the SceneParser with a scene file path.

        Args:
            scene_file (str): Path to the YAML scene file
        """
        self.scene_file = scene_file
        self.scene_data = None
        self.channels_info = []  # Store channel information
        self.robot_info = None  # Store robot information

    def load_scene(self):
        """
        Load and parse the scene data from the YAML file.
        Converts the raw data into a SimpleNamespace object for easier access.
        """
        with open(self.scene_file, "r") as file:
            raw_data = yaml.safe_load(file)
            self.scene_data = self._convert_to_namespace(raw_data)
            # Extract robot information
            if hasattr(self.scene_data, "robot"):
                self.robot_info = self.scene_data.robot
            else:
                raise ValueError("Robot information not found in scene file")

    def _convert_to_namespace(self, data: Dict) -> SimpleNamespace:
        """
        Recursively convert a dictionary to a SimpleNamespace object.

        Args:
            data (Dict): Dictionary data to convert

        Returns:
            SimpleNamespace: Converted data structure
        """
        if isinstance(data, dict):
            return SimpleNamespace(**{k: self._convert_to_namespace(v) for k, v in data.items()})
        elif isinstance(data, list):
            return [self._convert_to_namespace(item) for item in data]
        else:
            return data

    def approximate_elements_with_spheres(self) -> List[Dict]:
        """
        Approximate geometric elements in the scene with spheres.

        Returns:
            List[Dict]: List of approximated elements with sphere properties
        """
        if not self.scene_data:
            raise ValueError("Scene data is not loaded. Call load_scene() first.")

        approximated_elements = []
        for element in self.scene_data.elements:
            shape_type = element.shape.type
            if shape_type == "cylinder":
                approximated_elements.extend(self._approximate_cylinder(element))
            elif shape_type == "cuboid":
                approximated_elements.extend(self._approximate_cuboid(element))
            elif shape_type == "sphere":
                approximated_elements.append(element)  # No approximation needed
        return approximated_elements

    def _approximate_cylinder(self, element: SimpleNamespace) -> List[Dict]:
        """
        Approximate a cylinder with a series of spheres along its axis.

        Args:
            element (SimpleNamespace): Cylinder element to approximate

        Returns:
            List[Dict]: List of spheres approximating the cylinder
        """
        # Get cylinder center point and parameters
        center = np.array(element.position)
        height = element.shape.parameters.height
        radius = element.shape.parameters.radius
        orientation = element.orientation  # Assuming quaternion
        sphere_radius = element.sphere_fit.radius
        count = element.sphere_fit.count

        # Calculate rotated direction vector
        base_direction = np.array([0, 0, 1])  # Default cylinder axis
        rotation_matrix = self._quaternion_to_rotation_matrix(orientation)
        direction = rotation_matrix @ base_direction
        direction = direction / np.linalg.norm(direction)  # Normalize direction vector

        # Calculate cylinder start and end points (relative to center)
        half_height = height / 2
        start = center - direction * half_height
        end = center + direction * half_height

        # Generate sphere positions through linear interpolation
        spheres = []
        for i in range(count):
            t = i / (count - 1) if count > 1 else 0.5
            position = (1 - t) * start + t * end
            spheres.append({"id": f"{element.id}_sphere_{i+1}", "position": position.tolist(), "radius": sphere_radius})
        return spheres

    def _approximate_cuboid(self, element: SimpleNamespace) -> List[Dict]:
        """
        Approximate a cuboid with spheres (placeholder method).

        Args:
            element (SimpleNamespace): Cuboid element to approximate

        Returns:
            List[Dict]: List of spheres approximating the cuboid
        """
        # Placeholder for cuboid approximation
        return []

    def _quaternion_to_rotation_matrix(self, q: List[float]) -> np.ndarray:
        """
        Convert a quaternion to a 3x3 rotation matrix.

        Args:
            q (List[float]): Quaternion [x, y, z, w]

        Returns:
            np.ndarray: 3x3 rotation matrix
        """
        x, y, z, w = q
        return np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
            ]
        )

    def visualize_scene(
        self,
        scene_name: str,
        task_name: str,
        algorithm_name: str,
        plan_id: int = None,  # 修改为可选参数
        enable_spheres=True,
        enable_channels=True,
    ):
        """
        Visualize the scene using PyBullet.
        Creates collision bodies for cylinders and spheres, and visualizes channels.

        Args:
            scene_name: 场景名称
            task_name: 任务名称
            algorithm_name: 算法名称
            plan_id: 轨迹编号，如果为None则可视化所有轨迹
            enable_spheres: 是否显示球体近似
            enable_channels: 是否显示通道
        """
        # Initialize PyBullet
        init_pb()

        # Prepare line data for create_collision_bodies
        line_pts_flattened = []
        radius_per_edge = []

        # Extract line data from scene elements
        for element in self.scene_data.elements:
            if element.shape.type == "cylinder":
                # Get cylinder center point and parameters
                center = np.array(element.position)
                height = element.shape.parameters.height
                radius = element.shape.parameters.radius
                orientation = element.orientation

                # Calculate rotated direction vector
                base_direction = np.array([0, 0, 1])  # Default cylinder axis
                rotation_matrix = self._quaternion_to_rotation_matrix(orientation)
                direction = rotation_matrix @ base_direction
                direction = direction / np.linalg.norm(direction)  # Normalize direction vector

                # Calculate cylinder start and end points (relative to center)
                half_height = height / 2
                start = center - direction * half_height
                end = center + direction * half_height

                # Add to line data
                line_pts_flattened.extend([start, end])
                radius_per_edge.append(radius)

        # Create elements using create_collision_bodies
        with pp.LockRenderer():
            element_bodies = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)

        # Create sphere approximations using pp.create_sphere
        if enable_spheres:
            sphere_bodies = []
            with pp.LockRenderer():
                for sphere in self.approximate_elements_with_spheres():
                    sphere_body = pp.create_sphere(
                        radius=sphere["radius"], color=(0, 1, 0, 0.5)
                    )  # Green semi-transparent
                    pp.set_pose(sphere_body, pp.Pose(point=sphere["position"]))
                    sphere_bodies.append(sphere_body)

        if enable_channels:
            # Visualize channels
            if hasattr(self.scene_data, "channels_info"):
                channel_bodies = []
                with pp.LockRenderer():
                    for channel in self.scene_data.channels_info:
                        # Get channel parameters
                        channel_center = np.array(channel.center)
                        channel_dir = np.array(channel.direction)
                        channel_type = channel.type
                        channel_size = channel.size
                        channel_thickness = channel.thickness

                        # Calculate channel dimensions
                        if channel_type == "ellipse":
                            a = channel_size[0]  # Major axis
                            b = channel_size[1]  # Minor axis
                            radius = min(a, b) / 2
                            height = channel_thickness
                        else:  # rectangle
                            width = channel_size[0]
                            height = channel_size[1]
                            radius = min(width, height) / 2
                            height = channel_thickness

                        # Create flat transparent cylinder
                        cylinder_body = pp.create_cylinder(radius=radius, height=height, color=(0, 1, 1, 0.3))

                        # Build coordinate system based on channel_dir
                        z_axis = channel_dir / np.linalg.norm(channel_dir)

                        # Choose any vector not parallel to z-axis as temporary x-axis
                        temp_x = np.array([1, 0, 0])
                        if np.abs(np.dot(temp_x, z_axis)) > 0.9:  # If too close to parallel
                            temp_x = np.array([0, 1, 0])

                        # Calculate y-axis
                        y_axis = np.cross(z_axis, temp_x)
                        y_axis = y_axis / np.linalg.norm(y_axis)

                        # Calculate x-axis
                        x_axis = np.cross(y_axis, z_axis)
                        x_axis = x_axis / np.linalg.norm(x_axis)

                        # Build rotation matrix
                        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

                        # Convert to Euler angles
                        rotation = Rotation.from_matrix(rotation_matrix)
                        rotation_euler = rotation.as_euler("xyz", degrees=False).tolist()

                        # Set cylinder position and orientation
                        pp.set_pose(
                            cylinder_body,
                            pp.Pose(point=channel_center, euler=rotation_euler),
                        )
                        channel_bodies.append(cylinder_body)

                        # Create channel direction indicator line
                        line_body = pp.add_line(
                            channel_center, channel_center + channel_dir * 0.25, color=(0, 1, 1, 1), width=4
                        )
                        channel_bodies.append(line_body)

        # 添加轨迹复现功能
        data_dir = os.path.join(HERE, "model", "data")
        task_dir = os.path.join(data_dir, scene_name, task_name)
        alg_dir = os.path.join(task_dir, algorithm_name)

        if not os.path.exists(alg_dir):
            print(f"\n未找到算法目录 {alg_dir}")
            return

        # 获取所有轨迹文件
        traj_files = sorted(glob.glob(os.path.join(alg_dir, "plan_*.npy")))
        if not traj_files:
            print(f"\n未找到任何轨迹数据")
            return

        # 如果指定了plan_id，只处理该轨迹
        if plan_id is not None:
            traj_file = os.path.join(alg_dir, f"plan_{plan_id}.npy")
            if not os.path.exists(traj_file):
                print(f"\n未找到轨迹数据 {traj_file}")
                return
            traj_files = [traj_file]

        # 创建DataLoader实例
        from model.data_loader import SceneDataLoader
        data_loader = SceneDataLoader()

        # 加载轨迹数据并进行插值
        trajectories = data_loader.load_trajectories(
            scene_name=scene_name,
            task_name=task_name,
            algorithm_name=algorithm_name,
            target_length=5000
        )

        # 获取机器人实例
        from robot.robot_setup import RobotSetup
        rb = RobotSetup("r0")
        pp.set_pose(rb.robot, pp.Pose(point=(-0.5, 0.5, 0), euler=pp.Euler(0, 0, 0)))

        # 创建被抓取的元素
        line_pts_grasped = [np.array([0, 0, 0]), np.array([0, 0, 1])]
        grasped_element = create_collision_bodies(line_pts_grasped, [0.01], viewer=True)[0]

        # 设置被抓取元素的位姿并创建附着关系
        pp.set_pose(
            grasped_element,
            pp.multiply(
                pp.get_link_pose(rb.robot, rb.tool_link),
                pp.Pose(point=(0, 0.1, 0.15), euler=pp.Euler(1.5708, 0, 0)),
            ),
        )
        grasp_attachment = pp.create_attachment(rb.robot, rb.tool_link, grasped_element)
        
        pp.wait_for_user("按回车键开始回放...")

        # 遍历所有需要可视化的轨迹
        for i, traj_file in enumerate(traj_files):
            current_plan_id = int(os.path.basename(traj_file).split('_')[1].split('.')[0])
            
            print("\n" + "="*50)
            print(f"当前轨迹 ({i+1}/{len(traj_files)}):")
            print(f"场景: {scene_name}")
            print(f"任务: {task_name}")
            print(f"算法: {algorithm_name}")
            print(f"轨迹ID: {current_plan_id}")
            print("="*50)

            if current_plan_id >= len(trajectories):
                print(f"轨迹ID {current_plan_id} 超出范围")
                continue

            trajectory = trajectories[current_plan_id]

            # 添加回放控制滑块
            p.removeAllUserParameters()
            replay_slider = p.addUserDebugParameter("replay", 0, 1, 0)
            continue_button = p.addUserDebugParameter("continue", 1, 0, 0)
            prev_continue_button_value = p.readUserDebugParameter(continue_button)

            while True:
                # 获取回放进度
                replay = p.readUserDebugParameter(replay_slider)
                current_continue_button_value = p.readUserDebugParameter(continue_button)

                # 根据进度设置机器人位置
                idx = int(replay * (len(trajectory) - 1))
                conf = trajectory[idx]
                rb.set_joint_positions(rb.arm_joints, conf)
                grasp_attachment.assign()

                time.sleep(1.0 / 240)

                # 检查是否继续下一个轨迹
                if current_continue_button_value > prev_continue_button_value:
                    break
                prev_continue_button_value = current_continue_button_value

        print("\n所有轨迹回放完成!")

    def get_robot_start_pose(self) -> List[float]:
        """
        Get the robot's start joint configuration.

        Returns:
            List[float]: Robot's start joint angles [j1, j2, j3, j4, j5, j6]
        """
        if not self.robot_info:
            raise ValueError("Robot information not loaded")
        return self.robot_info.start

    def get_robot_target_pose(self) -> List[float]:
        """
        Get the robot's target joint configuration.

        Returns:
            List[float]: Robot's target joint angles [j1, j2, j3, j4, j5, j6]
        """
        if not self.robot_info:
            raise ValueError("Robot information not loaded")
        return self.robot_info.target

    def get_robot_grasp_offset(self) -> List[float]:
        """
        Get the robot's grasp offset in tool_link frame.

        Returns:
            List[float]: Grasp offset [x, y, z] in tool_link frame
        """
        if not self.robot_info:
            raise ValueError("Robot information not loaded")
        return self.robot_info.grasp_offset

    def get_robot_pose_2d(self, output_type: str = "list") -> Union[List[float], np.ndarray]:
        """
        Get the robot's 2D pose.

        Args:
            output_type (str): The type of output to return. Can be "list" or "array".

        Returns:
            List[float]: Robot's 2D pose [x, y, yaw]
        """
        if not self.robot_info:
            raise ValueError("Robot information not loaded")
        if output_type == "list":
            return self.robot_info.pose_2d
        elif output_type == "array":
            return np.array(self.robot_info.pose_2d)
        else:
            raise ValueError(f"Invalid output type: {output_type}")

    def get_element_info(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract line points and radii information from scene elements.

        Returns:
            Tuple[List[np.ndarray], List[float]]: A tuple containing:
                - List of line points (each point is a numpy array)
                - List of radii for each line segment
        """
        if not self.scene_data:
            raise ValueError("Scene data is not loaded. Call load_scene() first.")

        line_pts_flattened = []
        radius_per_edge = []

        # Extract line data from scene elements
        for element in self.scene_data.elements:
            if element.shape.type == "cylinder":
                # Get cylinder center point and parameters
                center = np.array(element.position)
                height = element.shape.parameters.height
                radius = element.shape.parameters.radius
                orientation = element.orientation

                # Calculate rotated direction vector
                base_direction = np.array([0, 0, 1])  # Default cylinder axis
                rotation_matrix = self._quaternion_to_rotation_matrix(orientation)
                direction = rotation_matrix @ base_direction
                direction = direction / np.linalg.norm(direction)  # Normalize direction vector

                # Calculate cylinder start and end points (relative to center)
                half_height = height / 2
                start = center - direction * half_height
                end = center + direction * half_height

                # Add to line data
                line_pts_flattened.extend([start, end])
                radius_per_edge.append(radius)

        return line_pts_flattened, radius_per_edge


def reorganize_tasks(scene_name: str):
    """
    重新排布指定场景的任务编号，同时更新scenes和data目录中的文件

    Args:
        scene_name: 场景名称
    """
    scenes_dir = os.path.join(HERE, "model", "scenes", scene_name)
    data_dir = os.path.join(HERE, "model", "data", scene_name)

    if not os.path.exists(scenes_dir) or not os.path.exists(data_dir):
        print(f"场景 {scene_name} 的目录不存在")
        return

    # 获取所有任务文件并按数字顺序排序
    task_files = glob.glob(os.path.join(scenes_dir, "*.yml"))

    # 自定义排序函数，提取task_后的数字进行排序
    def get_task_number(filename):
        basename = os.path.splitext(os.path.basename(filename))[0]
        try:
            return int(basename.split("_")[1])
        except (IndexError, ValueError):
            return float("inf")  # 对于非标准命名的文件排在最后

    task_files.sort(key=get_task_number)

    # 创建新旧任务名称映射
    task_mapping = {}  # {old_name: new_name}
    current_number = 1

    for task_file in task_files:
        old_name = os.path.splitext(os.path.basename(task_file))[0]
        new_name = f"task_{current_number}"
        task_mapping[old_name] = new_name
        current_number += 1

    print("\n任务重命名映射:")
    for old, new in task_mapping.items():
        print(f"{old} -> {new}")

    # 更新scenes目录
    print("\n更新scenes目录...")
    for old_name, new_name in task_mapping.items():
        old_file = os.path.join(scenes_dir, f"{old_name}.yml")
        new_file = os.path.join(scenes_dir, f"{new_name}.yml")

        if old_name == new_name:
            continue

        # 读取并更新yml文件内容
        with open(old_file, "r") as f:
            content = f.read()

        # 更新scene_name字段
        content = content.replace(f'scene_name: "{scene_name}_{old_name}"', f'scene_name: "{scene_name}_{new_name}"')

        # 如果新旧文件名不同，直接重命名文件
        if old_file != new_file:
            # 先写入更新后的内容到原文件
            with open(old_file, "w") as f:
                f.write(content)
            # 然后重命名文件
            os.rename(old_file, new_file)

    # 更新data目录
    print("\n更新data目录...")
    for old_name, new_name in task_mapping.items():
        old_dir = os.path.join(data_dir, old_name)
        new_dir = os.path.join(data_dir, new_name)

        if old_name == new_name:
            continue

        if os.path.exists(old_dir):
            # 如果新目录已存在，先删除
            if os.path.exists(new_dir):
                import shutil
                shutil.rmtree(new_dir)
            # 重命名目录
            os.rename(old_dir, new_dir)

    print("\n重排布完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="场景工具")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["visualize", "reorganize"],
        help="工具模式: visualize(可视化) 或 reorganize(重排布)",
    )

    # 可视化模式的参数
    vis_group = parser.add_argument_group("可视化参数")
    vis_group.add_argument("--scene", type=str, help="场景名称")
    vis_group.add_argument("--task", type=str, help="任务名称")
    vis_group.add_argument("--algorithm", type=str, help="算法名称")
    vis_group.add_argument("--plan-id", type=int, help="轨迹编号，不指定则可视化所有轨迹")
    vis_group.add_argument("--enable-spheres", action="store_true", help="显示球体近似")
    vis_group.add_argument("--enable-channels", action="store_true", help="显示通道")

    # 重排布模式的参数
    reorg_group = parser.add_argument_group("重排布参数")
    reorg_group.add_argument("--target-scene", type=str, help="要重排布的场景名称")

    args = parser.parse_args()

    if args.mode == "visualize":
        # 修改参数检查
        if not all([args.scene, args.task, args.algorithm]):
            parser.error("可视化模式需要指定 --scene, --task 和 --algorithm")

        # 构建场景文件路径
        scene_file = os.path.join(HERE, "model", "scenes", args.scene, f"{args.task}.yml")

        scene_parser = SceneParser(scene_file)
        scene_parser.load_scene()
        scene_parser.visualize_scene(
            scene_name=args.scene,
            task_name=args.task,
            algorithm_name=args.algorithm,
            plan_id=args.plan_id,
            enable_spheres=args.enable_spheres,
            enable_channels=args.enable_channels,
        )
    else:  # reorganize mode
        if not args.target_scene:
            parser.error("重排布模式需要指定 --target-scene")
        reorganize_tasks(args.target_scene)
