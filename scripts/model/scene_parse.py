import os
import sys
from types import SimpleNamespace
from typing import Dict, List

import pybullet_planning as pp
from scipy.spatial.transform import Rotation

import numpy as np
import yaml

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from multi_tangent.collision import create_collision_bodies
from utils.collision import init_pb
from utils.params import *


class SceneParser:
    def __init__(self, scene_file: str):
        self.scene_file = scene_file
        self.scene_data = None
        self.channels_info = []  # 存储通道信息

    def load_scene(self):
        with open(self.scene_file, "r") as file:
            raw_data = yaml.safe_load(file)
            self.scene_data = self._convert_to_namespace(raw_data)

    def _convert_to_namespace(self, data: Dict) -> SimpleNamespace:
        if isinstance(data, dict):
            return SimpleNamespace(**{k: self._convert_to_namespace(v) for k, v in data.items()})
        elif isinstance(data, list):
            return [self._convert_to_namespace(item) for item in data]
        else:
            return data

    def approximate_elements_with_spheres(self) -> List[Dict]:
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
        # 获取圆柱体的中心点和参数
        center = np.array(element.position)
        height = element.shape.parameters.height
        radius = element.shape.parameters.radius
        orientation = element.orientation  # Assuming quaternion
        sphere_radius = element.sphere_fit.radius
        count = element.sphere_fit.count

        # 计算旋转后的方向向量
        base_direction = np.array([0, 0, 1])  # 默认圆柱体轴
        rotation_matrix = self._quaternion_to_rotation_matrix(orientation)
        direction = rotation_matrix @ base_direction
        direction = direction / np.linalg.norm(direction)  # 归一化方向向量

        # 计算圆柱体的起点和终点（以中心点为基准）
        half_height = height / 2
        start = center - direction * half_height
        end = center + direction * half_height

        # 线性插值生成球体位置
        spheres = []
        for i in range(count):
            t = i / (count - 1) if count > 1 else 0.5
            position = (1 - t) * start + t * end
            spheres.append({"id": f"{element.id}_sphere_{i+1}", "position": position.tolist(), "radius": sphere_radius})
        return spheres

    def _approximate_cuboid(self, element: SimpleNamespace) -> List[Dict]:
        # Placeholder for cuboid approximation
        return []

    def _quaternion_to_rotation_matrix(self, q: List[float]) -> np.ndarray:
        x, y, z, w = q
        return np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
            ]
        )

    def visualize_scene(self):
        # 初始化PyBullet
        init_pb()

        # 准备线段数据用于create_collision_bodies
        line_pts_flattened = []
        radius_per_edge = []

        # 从场景元素中提取线段数据
        for element in self.scene_data.elements:
            if element.shape.type == "cylinder":
                # 获取圆柱体的中心点和参数
                center = np.array(element.position)
                height = element.shape.parameters.height
                radius = element.shape.parameters.radius
                orientation = element.orientation

                # 计算旋转后的方向向量
                base_direction = np.array([0, 0, 1])  # 默认圆柱体轴
                rotation_matrix = self._quaternion_to_rotation_matrix(orientation)
                direction = rotation_matrix @ base_direction
                direction = direction / np.linalg.norm(direction)  # 归一化方向向量

                # 计算圆柱体的起点和终点（以中心点为基准）
                half_height = height / 2
                start = center - direction * half_height
                end = center + direction * half_height

                # 添加到线段数据
                line_pts_flattened.extend([start, end])
                radius_per_edge.append(radius)

        # 使用create_collision_bodies创建元素
        with pp.LockRenderer():
            element_bodies = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)

        # 使用pp.create_sphere创建球体近似
        sphere_bodies = []
        with pp.LockRenderer():
            for sphere in self.approximate_elements_with_spheres():
                sphere_body = pp.create_sphere(radius=sphere["radius"], color=(0, 1, 0, 0.5))  # 绿色半透明
                pp.set_pose(sphere_body, pp.Pose(point=sphere["position"]))
                sphere_bodies.append(sphere_body)

        # 可视化通道
        if hasattr(self.scene_data, "channels_info"):
            channel_bodies = []
            with pp.LockRenderer():
                for channel in self.scene_data.channels_info:
                    # 获取通道参数
                    channel_center = np.array(channel.center)
                    channel_dir = np.array(channel.direction)
                    channel_type = channel.type
                    channel_size = channel.size
                    channel_thickness = channel.thickness

                    # 计算通道的尺寸
                    if channel_type == "ellipse":
                        a = channel_size[0]  # 长轴
                        b = channel_size[1]  # 短轴
                        radius = min(a, b) / 2
                        height = channel_thickness
                    else:  # rectangle
                        width = channel_size[0]
                        height = channel_size[1]
                        radius = min(width, height) / 2
                        height = channel_thickness

                    # 创建扁平的透明圆柱体
                    cylinder_body = pp.create_cylinder(radius=radius, height=height, color=(0, 1, 1, 0.3))

                    # 构建基于channel_dir的坐标系
                    z_axis = channel_dir / np.linalg.norm(channel_dir)

                    # 选择任意一个不与z轴平行的向量作为临时x轴
                    temp_x = np.array([1, 0, 0])
                    if np.abs(np.dot(temp_x, z_axis)) > 0.9:  # 如果太接近平行
                        temp_x = np.array([0, 1, 0])

                    # 计算y轴
                    y_axis = np.cross(z_axis, temp_x)
                    y_axis = y_axis / np.linalg.norm(y_axis)

                    # 计算x轴
                    x_axis = np.cross(y_axis, z_axis)
                    x_axis = x_axis / np.linalg.norm(x_axis)

                    # 构建旋转矩阵
                    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

                    # 转换为欧拉角
                    rotation = Rotation.from_matrix(rotation_matrix)
                    rotation_euler = rotation.as_euler("xyz", degrees=False).tolist()

                    # 设置圆柱体的位置和方向
                    pp.set_pose(
                        cylinder_body,
                        pp.Pose(point=channel_center, euler=rotation_euler),
                    )
                    channel_bodies.append(cylinder_body)

                    # 创建通道方向指示线
                    line_body = pp.add_line(
                        channel_center, channel_center + channel_dir * 0.25, color=(0, 1, 1, 1), width=4
                    )
                    channel_bodies.append(line_body)

        print(f"场景可视化完成: {len(element_bodies)} 个元素, {len(sphere_bodies)} 个球体近似")
        if hasattr(self.scene_data, "channels_info"):
            print(f"通道数量: {len(self.scene_data.channels_info)}")

        # 保持GUI运行
        pp.wait_for_user("Press Enter to exit...")


if __name__ == "__main__":
    parser = SceneParser("/home/jeong/summer_research/eth_ws/src/husky_assembly/scripts/model/scenes/cuboid_1.yml")
    parser.load_scene()
    parser.visualize_scene()
