import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple, Union

import numpy as np
import pybullet_planning as pp
import yaml
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
import glob
import time
import pybullet as p
import argparse
import random

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from multi_tangent.collision import create_collision_bodies
from utils.collision import init_pb
from utils.params import *
from robot.robot import RobotSetup


class SceneParser:
    """
    A class for parsing and visualizing 3D scene data from YAML files.

    This class handles loading scene data, approximating geometric shapes with spheres,
    and visualizing the scene using PyBullet.
    """

    @staticmethod
    def quaternion_to_rotation_matrix(q: List[float]) -> np.ndarray:
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

    @staticmethod
    def approximate_cylinder(element_body: int, count: int = 100) -> List[Dict]:
        """
        Approximate a sphere with a series of spheres along its axis.
        """
        # 获取圆柱体的中心点和方向
        center = pp.get_pose(element_body)[0]
        orientation = pp.get_pose(element_body)[1]
        base_direction = np.array([0, 0, 1])
        rotation_matrix = SceneParser.quaternion_to_rotation_matrix(orientation)
        direction = rotation_matrix @ base_direction
        direction = direction / np.linalg.norm(direction)  # Normalize direction vector
        radius = 0.01
        height = 1.0

        half_height = height / 2
        start = center - direction * half_height
        end = center + direction * half_height

        spheres = []
        for i in range(count):
            t = i / (count - 1) if count > 1 else 0.5
            position = (1 - t) * start + t * end
            spheres.append({"name": f"{element_body}_sphere_{i+1}", "position": position.tolist(), "radius": radius})
        return spheres

    @staticmethod
    def fit_plane_to_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用最小二乘法拟合点云的平面，计算平面法向量和中心点

        Args:
            points: 形状为(N, 3)的点云数组

        Returns:
            Tuple[np.ndarray, np.ndarray]: (平面法向量, 中心点)
        """
        # 计算中心点
        center = np.mean(points, axis=0)

        # 将点云中心化
        centered_points = points - center

        # 使用SVD进行平面拟合
        _, _, vh = np.linalg.svd(centered_points, full_matrices=False)

        # 取SVD结果中的最小特征值对应的向量作为平面法向量
        normal = vh[2, :]

        # 确保法向量朝向z轴正方向（如果接近z轴）
        if normal[2] < 0:
            normal = -normal

        # 法向量归一化
        normal = normal / np.linalg.norm(normal)

        return normal, center

    @staticmethod
    def project_points_to_plane(points: np.ndarray, normal: np.ndarray, center: np.ndarray) -> np.ndarray:
        """
        将点云投影到平面上

        Args:
            points: 形状为(N, 3)的点云数组
            normal: 平面法向量
            center: 平面中心点

        Returns:
            np.ndarray: 投影后的点云数组
        """
        projected_points = []
        for point in points:
            # 计算点到平面的有符号距离
            v = point - center
            dist = np.dot(v, normal)

            # 计算投影点
            projected_point = point - dist * normal
            projected_points.append(projected_point)

        return np.array(projected_points)

    @staticmethod
    def sample_points_in_polgon(points: np.ndarray, num_samples: int = 100, ratio: float = 1.0) -> np.ndarray:
        """
        在3D多边形内采样点，使用三角分割方法

        Args:
            points: 形状为(N, 3)的点云数组，表示多边形顶点
            num_samples: 需要采样的点数量
            ratio: 缩放比例(0~1)，将多边形顶点到中心的距离缩放为原来的ratio倍

        Returns:
            np.ndarray: 形状为(num_samples, 3)的采样点数组
        """
        if len(points) < 3:
            raise ValueError("需要至少3个点来定义一个多边形")

        # 验证ratio参数范围
        if ratio < 0 or ratio > 1:
            raise ValueError("ratio参数必须在0到1之间")

        # 1. 拟合平面
        normal, center = SceneParser.fit_plane_to_points(points)

        # 2. 将点投影到平面上
        projected_points = SceneParser.project_points_to_plane(points, normal, center)

        # 3. 根据ratio参数缩放多边形
        if ratio < 1.0:
            scaled_points = []
            for point in projected_points:
                # 从中心点到顶点的向量
                vector = point - center
                # 缩放向量
                scaled_vector = vector * ratio
                # 计算新的点坐标
                scaled_point = center + scaled_vector
                scaled_points.append(scaled_point)
            projected_points = np.array(scaled_points)

        # 4. 创建平面坐标系（基于法向量）
        # 选择一个与法向量不平行的向量作为参考
        ref = np.array([1, 0, 0])
        if np.abs(np.dot(ref, normal)) > 0.9:
            ref = np.array([0, 1, 0])

        # 创建平面坐标系的两个轴
        x_axis = np.cross(normal, ref)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(normal, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # 5. 将3D点投影到2D坐标系
        points_2d = []
        for point in projected_points:
            # 计算相对于中心点的向量
            v = point - center
            # 计算在x轴和y轴上的投影
            x = np.dot(v, x_axis)
            y = np.dot(v, y_axis)
            points_2d.append([x, y])

        points_2d = np.array(points_2d)

        # 6. 使用Delaunay三角剖分
        from scipy.spatial import Delaunay

        tri = Delaunay(points_2d)

        # 7. 采样点
        samples = []
        triangles_areas = []

        # 计算每个三角形的面积
        for simplex in tri.simplices:
            # 获取三角形的顶点
            triangle = points_2d[simplex]
            # 计算三角形面积
            area = 0.5 * np.abs(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
            triangles_areas.append(area)

        # 归一化面积作为概率
        total_area = sum(triangles_areas)
        if total_area == 0:
            raise ValueError("多边形总面积为零")

        probabilities = [area / total_area for area in triangles_areas]

        # 按面积比例在三角形中采样
        for _ in range(num_samples):
            # 随机选择一个三角形，大三角形被选中的概率更高
            tri_idx = np.random.choice(len(tri.simplices), p=probabilities)
            simplex = tri.simplices[tri_idx]
            triangle = points_2d[simplex]

            # 在三角形内均匀采样
            # 参数化采样方法: p = (1-sqrt(r1))*p1 + sqrt(r1)*(1-r2)*p2 + sqrt(r1)*r2*p3
            r1 = np.random.random()
            r2 = np.random.random()

            # 计算采样点
            sample = (1 - np.sqrt(r1)) * triangle[0] + np.sqrt(r1) * (1 - r2) * triangle[1] + np.sqrt(r1) * r2 * triangle[2]

            # 将2D采样点转换回3D空间
            sample_3d = center + sample[0] * x_axis + sample[1] * y_axis
            samples.append(sample_3d)

        return np.array(samples)

    @staticmethod
    def sample_points_in_channel(shape_type: str, params: Dict, thickness: float, num_samples: int = 100, ratio: float = 1.0) -> np.ndarray:
        """
        在不同类型的3D通道内采样点

        Args:
            shape_type: 形状类型，可以是 "rectangle", "circle" 或 "polygon"
            params: 包含计算所需参数的字典
                - rectangle: 需包含 "size" 键，值为 [length, width]、"center" 和 "direction"
                - circle: 需包含 "size" 键，值为 radius、"center" 和 "direction"
                - polygon: 需包含 "points" 键，值为多边形的顶点数组
            thickness: 通道厚度
            num_samples: 需要采样的点数量
            ratio: 缩放比例(0~1)，将多边形顶点到中心的距离缩放为原来的ratio倍

        Returns:
            np.ndarray: 形状为(num_samples, 3)的采样点数组
        """
        # 验证ratio参数范围
        if ratio < 0 or ratio > 1:
            raise ValueError("ratio参数必须在0到1之间")

        if shape_type == "rectangle":
            # 检查必要参数
            if "size" not in params or "center" not in params or "direction" not in params:
                raise ValueError("Rectangle type requires 'size', 'center', and 'direction' parameters")

            # 获取参数
            size = params["size"]
            center = np.array(params["center"])
            direction = np.array(params["direction"])

            if len(size) < 2:
                raise ValueError("Rectangle size should have at least 2 values [length, width]")

            length, width = size[0], size[1]

            # 计算缩放后的长宽
            scaled_length = length * ratio
            scaled_width = width * ratio

            # 创建矩形的顶点
            # 首先计算法向量和局部坐标系
            normal = direction / np.linalg.norm(direction)

            # 计算矩形的局部坐标系
            temp_x = np.array([1, 0, 0])
            if np.abs(np.dot(temp_x, normal)) > 0.9:
                temp_x = np.array([0, 1, 0])
            local_y = np.cross(normal, temp_x)
            local_y = local_y / np.linalg.norm(local_y)
            local_x = np.cross(local_y, normal)
            local_x = local_x / np.linalg.norm(local_x)

            # 创建矩形顶点（以中心为原点的局部坐标系中）
            half_length = scaled_length / 2
            half_width = scaled_width / 2

            # 矩形的四个顶点
            points = [
                center + local_x * half_length + local_y * half_width,
                center + local_x * half_length - local_y * half_width,
                center - local_x * half_length - local_y * half_width,
                center - local_x * half_length + local_y * half_width,
            ]
            points = np.array(points)

            # 使用多边形的采样方法
            planar_samples = SceneParser.sample_points_in_polgon(points, num_samples, ratio=1.0)  # 已经缩放过了，所以ratio=1.0

        elif shape_type == "circle":
            # 检查必要参数
            if "size" not in params or "center" not in params or "direction" not in params:
                raise ValueError("Circle type requires 'size', 'center', and 'direction' parameters")

            # 获取参数
            radius = params["size"]
            center = np.array(params["center"])
            direction = np.array(params["direction"])

            if isinstance(radius, list) and len(radius) > 0:
                radius = radius[0]  # 如果是列表，取第一个元素

            # 计算缩放后的半径
            scaled_radius = radius * ratio

            # 生成圆周上的点
            normal = direction / np.linalg.norm(direction)

            # 计算圆的局部坐标系
            temp_x = np.array([1, 0, 0])
            if np.abs(np.dot(temp_x, normal)) > 0.9:
                temp_x = np.array([0, 1, 0])
            local_y = np.cross(normal, temp_x)
            local_y = local_y / np.linalg.norm(local_y)
            local_x = np.cross(local_y, normal)
            local_x = local_x / np.linalg.norm(local_x)

            # 生成圆周上的点（多边形近似）
            num_circle_points = 16  # 用16个点近似圆
            circle_points = []
            for i in range(num_circle_points):
                angle = 2 * np.pi * i / num_circle_points
                x = scaled_radius * np.cos(angle)
                y = scaled_radius * np.sin(angle)
                point = center + local_x * x + local_y * y
                circle_points.append(point)

            circle_points = np.array(circle_points)

            # 使用多边形的采样方法
            planar_samples = SceneParser.sample_points_in_polgon(circle_points, num_samples, ratio=1.0)  # 已经缩放过了，所以ratio=1.0

        elif shape_type == "polygon":
            # 检查必要参数
            if "points" not in params:
                raise ValueError("Polygon type requires 'points' parameter")

            points = np.array(params["points"])
            if len(points) < 3:
                raise ValueError("需要至少3个点来定义一个多边形")

            # 使用原有的多边形采样方法
            planar_samples = SceneParser.sample_points_in_polgon(points, num_samples, ratio=ratio)

        else:
            raise ValueError(f"不支持的形状类型: {shape_type}")

        # 计算平面法向量（对所有形状都通用）
        if shape_type in ["rectangle", "circle"]:
            normal = direction / np.linalg.norm(direction)
        else:  # polygon
            normal, _ = SceneParser.fit_plane_to_points(points)

        # 在法向方向上随机偏移生成体积样本
        volume_samples = []
        half_thickness = thickness / 2.0

        for point in planar_samples:
            # 生成[-half_thickness, half_thickness]范围内的随机偏移
            offset = np.random.uniform(-half_thickness, half_thickness)
            # 沿法向方向偏移点
            volume_point = point + offset * normal
            volume_samples.append(volume_point)

        return np.array(volume_samples)

    @staticmethod
    def sample_pose_in_channel(shape_type: str, params: Dict, thickness: float, num_samples: int = 100, ratio: float = 1.0) -> np.ndarray:
        """
        在不同类型的3D通道内采样位姿（位置+姿态）

        Args:
            shape_type: 形状类型，可以是 "rectangle", "circle" 或 "polygon"
            params: 包含计算所需参数的字典
            thickness: 通道厚度
            num_samples: 需要采样的点数量
            ratio: 缩放比例(0~1)，将多边形顶点到中心的距离缩放为原来的ratio倍

        Returns:
            np.ndarray: 形状为(num_samples, 6)的采样位姿数组，每行包含[x, y, z, roll, pitch, yaw]
        """
        # 首先采样位置点
        positions = SceneParser.sample_points_in_channel(shape_type, params, thickness, num_samples, ratio)

        # 为每个位置生成随机姿态（欧拉角）
        orientations = np.random.uniform(-np.pi, np.pi, size=(num_samples, 3))  # [roll, pitch, yaw]

        # 组合位置和姿态
        poses = np.zeros((num_samples, 6))
        poses[:, :3] = positions  # 位置 [x, y, z]
        poses[:, 3:] = orientations  # 姿态 [roll, pitch, yaw]

        return poses

    @staticmethod
    def compute_area(shape_type: str, params: Dict) -> float:
        """
        根据不同的形状类型计算面积

        Args:
            shape_type: 形状类型，可以是 "rectangle", "circle" 或 "polygon"
            params: 包含计算所需参数的字典
                - rectangle: 需包含 "size" 键，值为 [length, width]
                - circle: 需包含 "size" 键，值为 radius
                - polygon: 需包含 "points" 键，值为多边形的顶点数组

        Returns:
            float: 形状的面积
        """
        if shape_type == "rectangle":
            # 检查必要参数
            if "size" not in params:
                raise ValueError("Rectangle type requires 'size' parameter")

            # 获取长度和宽度
            size = params["size"]
            if len(size) < 2:
                raise ValueError("Rectangle size should have at least 2 values [length, width]")

            length, width = size[0], size[1]
            return length * width

        elif shape_type == "circle":
            # 检查必要参数
            if "size" not in params:
                raise ValueError("Circle type requires 'size' parameter")

            # 获取半径
            radius = params["size"]
            if isinstance(radius, list) and len(radius) > 0:
                radius = radius[0]  # 如果是列表，取第一个元素

            return np.pi * radius**2

        elif shape_type == "polygon":
            # 检查必要参数
            if "points" not in params:
                raise ValueError("Polygon type requires 'points' parameter")

            points = np.array(params["points"])
            if len(points) < 3:
                raise ValueError("需要至少3个点来定义一个多边形")

            # 1. 拟合平面
            normal, center = SceneParser.fit_plane_to_points(points)

            # 2. 将点投影到平面上
            projected_points = SceneParser.project_points_to_plane(points, normal, center)

            # 3. 创建平面坐标系（基于法向量）
            # 选择一个与法向量不平行的向量作为参考
            ref = np.array([1, 0, 0])
            if np.abs(np.dot(ref, normal)) > 0.9:
                ref = np.array([0, 1, 0])

            # 创建平面坐标系的两个轴
            x_axis = np.cross(normal, ref)
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(normal, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)

            # 4. 将3D点投影到2D坐标系
            points_2d = []
            for point in projected_points:
                # 计算相对于中心点的向量
                v = point - center
                # 计算在x轴和y轴上的投影
                x = np.dot(v, x_axis)
                y = np.dot(v, y_axis)
                points_2d.append([x, y])

            points_2d = np.array(points_2d)

            # 5. 使用Delaunay三角剖分
            from scipy.spatial import Delaunay

            tri = Delaunay(points_2d)

            # 6. 计算多边形面积
            total_area = 0.0

            # 累加所有三角形的面积
            for simplex in tri.simplices:
                # 获取三角形的顶点
                triangle = points_2d[simplex]
                # 计算三角形面积
                area = 0.5 * np.abs(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
                total_area += area

            return total_area

        else:
            raise ValueError(f"不支持的形状类型: {shape_type}")

    @staticmethod
    def load_channel(channel: Dict):
        channel_type = channel["type"]
        channel_thickness = channel["thickness"]

        if channel_type == "rectangle":
            channel_center = np.array(channel["center"])
            channel_direction = np.array(channel["direction"])
            channel_size = channel["size"]
            channel_body = pp.create_box(channel_size[0], channel_size[1], channel_thickness)

            # 计算方向轴和旋转矩阵
            channel_z_axis = channel_direction / np.linalg.norm(channel_direction)
            temp_x = np.array([1, 0, 0])
            if np.abs(np.dot(temp_x, channel_z_axis)) > 0.9:
                temp_x = np.array([0, 1, 0])
            channel_y_axis = np.cross(channel_z_axis, temp_x)
            channel_y_axis = channel_y_axis / np.linalg.norm(channel_y_axis)
            channel_x_axis = np.cross(channel_y_axis, channel_z_axis)
            channel_x_axis = channel_x_axis / np.linalg.norm(channel_x_axis)
            rotation_matrix = np.column_stack([channel_x_axis, channel_y_axis, channel_z_axis])
            rotation = Rotation.from_matrix(rotation_matrix)
            rotation_euler = rotation.as_euler("xyz", degrees=False).tolist()

            pp.set_pose(channel_body, pp.Pose(point=channel_center, euler=rotation_euler))

        elif channel_type == "circle":
            channel_center = np.array(channel["center"])
            channel_direction = np.array(channel["direction"])
            channel_size = channel["size"]
            channel_body = pp.create_cylinder(channel_size[0], channel_thickness)

            # 计算方向轴和旋转矩阵
            channel_z_axis = channel_direction / np.linalg.norm(channel_direction)
            temp_x = np.array([1, 0, 0])
            if np.abs(np.dot(temp_x, channel_z_axis)) > 0.9:
                temp_x = np.array([0, 1, 0])
            channel_y_axis = np.cross(channel_z_axis, temp_x)
            channel_y_axis = channel_y_axis / np.linalg.norm(channel_y_axis)
            channel_x_axis = np.cross(channel_y_axis, channel_z_axis)
            channel_x_axis = channel_x_axis / np.linalg.norm(channel_x_axis)
            rotation_matrix = np.column_stack([channel_x_axis, channel_y_axis, channel_z_axis])
            rotation = Rotation.from_matrix(rotation_matrix)
            rotation_euler = rotation.as_euler("xyz", degrees=False).tolist()

            pp.set_pose(channel_body, pp.Pose(point=channel_center, euler=rotation_euler))

        elif channel_type == "polygon":
            # 处理凸多边形类型的channel
            if "points" not in channel:
                raise ValueError("Polygon channel type requires 'points' parameter")

            # 获取点列表并验证
            points = np.array(channel["points"])
            if len(points) < 3:  # 至少需要3个点才能形成多边形
                raise ValueError(f"Polygon channel requires at least 3 points, got {len(points)}")

            # 检查是否提供了center和direction，如果没有则自动计算
            if "center" not in channel or "direction" not in channel:
                # 使用最小二乘法拟合平面和法向量
                normal, center = SceneParser.fit_plane_to_points(points)

                # 将点投影到拟合平面上
                projected_points = SceneParser.project_points_to_plane(points, normal, center)

                # 使用投影后的点和拟合的法向量创建多边形
                channel_body = SceneParser.create_polygon_channel(projected_points, center, normal, channel_thickness)
            else:
                # 使用用户提供的center和direction
                channel_center = np.array(channel["center"])
                channel_direction = np.array(channel["direction"])

                # 计算方向轴和旋转矩阵
                channel_z_axis = channel_direction / np.linalg.norm(channel_direction)
                temp_x = np.array([1, 0, 0])
                if np.abs(np.dot(temp_x, channel_z_axis)) > 0.9:
                    temp_x = np.array([0, 1, 0])
                channel_y_axis = np.cross(channel_z_axis, temp_x)
                channel_y_axis = channel_y_axis / np.linalg.norm(channel_y_axis)
                channel_x_axis = np.cross(channel_y_axis, channel_z_axis)
                channel_x_axis = channel_x_axis / np.linalg.norm(channel_x_axis)
                rotation_matrix = np.column_stack([channel_x_axis, channel_y_axis, channel_z_axis])

                # 创建凸多边形
                channel_body = SceneParser.create_polygon_channel(points, channel_center, rotation_matrix, channel_thickness)
        else:
            raise ValueError(f"Unknown channel type: {channel_type}")

        pp.set_color(channel_body, [0, 1, 1, 0.5])
        return channel_body

    @staticmethod
    def create_polygon_channel(points: np.ndarray, center: np.ndarray, normal_or_matrix, thickness: float) -> int:
        """
        通过一组3D点创建凸多边形channel

        Args:
            points: 3D点的列表，定义凸多边形的顶点
            center: 多边形中心
            normal_or_matrix: 平面法向量或旋转矩阵
            thickness: 厚度

        Returns:
            int: 生成的凸多边形物体ID
        """
        try:
            # 检查第三个参数是法向量还是旋转矩阵
            if normal_or_matrix.shape == (3,):  # 是法向量
                normal = normal_or_matrix
                # 计算旋转矩阵
                z_axis = normal / np.linalg.norm(normal)
                temp_x = np.array([1, 0, 0])
                if np.abs(np.dot(temp_x, z_axis)) > 0.9:
                    temp_x = np.array([0, 1, 0])
                y_axis = np.cross(z_axis, temp_x)
                y_axis = y_axis / np.linalg.norm(y_axis)
                x_axis = np.cross(y_axis, z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
                rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
            else:  # 是旋转矩阵
                rotation_matrix = normal_or_matrix

            # 计算旋转矩阵的逆矩阵
            rotation_inverse = np.linalg.inv(rotation_matrix)
            local_points = []

            for point in points:
                # 将点相对于中心偏移
                centered_point = point - center
                # 转换到局部坐标系
                local_point = rotation_inverse @ centered_point
                # 只保留x和y坐标（z方向上的坐标被压缩到一个平面）
                local_points.append([local_point[0], local_point[1]])

            # 计算2D凸包
            if len(local_points) > 2:  # ConvexHull需要至少3个点
                hull = ConvexHull(local_points)
                # 获取凸包顶点的顺序
                hull_vertices = [local_points[i] for i in hull.vertices]
            else:
                # 如果只有两个点，则直接使用
                hull_vertices = local_points

            # 检查特殊情况：如果点集近似矩形，直接使用pp.create_box更可靠
            if len(hull_vertices) == 4:
                # 检查是否接近矩形
                is_rectangle = True
                for i in range(4):
                    next_i = (i + 1) % 4
                    next_next_i = (i + 2) % 4

                    v1 = np.array(hull_vertices[next_i]) - np.array(hull_vertices[i])
                    v2 = np.array(hull_vertices[next_next_i]) - np.array(hull_vertices[next_i])

                    # 检查相邻边是否近似垂直
                    dot_product = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    if abs(dot_product) > 0.1:  # 允许一定的误差
                        is_rectangle = False
                        break

                if is_rectangle:
                    # 计算矩形尺寸
                    min_x = min(p[0] for p in hull_vertices)
                    max_x = max(p[0] for p in hull_vertices)
                    min_y = min(p[1] for p in hull_vertices)
                    max_y = max(p[1] for p in hull_vertices)

                    width = max_x - min_x
                    length = max_y - min_y

                    # 创建立方体
                    box_id = pp.create_box(width, length, thickness)
                    pp.set_pose(box_id, pp.Pose(point=center, euler=Rotation.from_matrix(rotation_matrix).as_euler("xyz")))
                    pp.set_color(box_id, [0, 1, 1, 0.5])
                    return box_id

            # 对于一般情况，创建自定义多边形
            # 确保凸包顶点是顺时针排序的（从外部看时逆时针，这对于正确的法线方向很重要）
            # 计算凸包的质心
            hull_center = np.mean(hull_vertices, axis=0)

            # 根据与质心的角度排序顶点 - 确保逆时针排序
            def angle_with_center(point):
                return np.arctan2(point[1] - hull_center[1], point[0] - hull_center[0])

            sorted_vertices = sorted(hull_vertices, key=angle_with_center)

            # 创建顶点集合
            verts = []
            # 底面顶点 (z = -thickness/2)
            bottom_verts_start = 0
            for x, y in sorted_vertices:
                verts.append((x, y, -thickness / 2))

            # 顶面顶点 (z = +thickness/2)
            top_verts_start = len(sorted_vertices)
            for x, y in sorted_vertices:
                verts.append((x, y, thickness / 2))

            # 直接使用PyBullet创建两个凸多边形（顶面和底面），然后再添加侧面
            num_verts = len(sorted_vertices)
            indices = []

            # 底面 - 三角形扇形，顶点顺序确保法线朝外（朝向-z）
            for i in range(1, num_verts - 1):
                indices.extend([0, i + 1, i])  # 逆时针顺序，法线朝外

            # 顶面 - 三角形扇形，顶点顺序确保法线朝外（朝向+z）
            top_offset = num_verts
            for i in range(1, num_verts - 1):
                indices.extend([top_offset, top_offset + i, top_offset + i + 1])  # 顺时针顺序，法线朝外

            # 侧面 - 每个侧面由两个三角形组成
            for i in range(num_verts):
                next_i = (i + 1) % num_verts
                # 添加两个三角形，确保法线指向外部
                # 注意顶点顺序：从外部观察时应为顺时针
                indices.extend([i, next_i, i + num_verts])  # 第一个三角形
                indices.extend([next_i, next_i + num_verts, i + num_verts])  # 第二个三角形

            # 创建凸多边形mesh
            visual_mesh_data = p.createVisualShape(shapeType=p.GEOM_MESH, vertices=verts, indices=indices, meshScale=[1, 1, 1], rgbaColor=[0, 1, 1, 0.5], flags=p.GEOM_FORCE_CONCAVE_TRIMESH)  # 强制使用凹面网格，不要进行凸分解

            collision_mesh_data = p.createCollisionShape(shapeType=p.GEOM_MESH, vertices=verts, indices=indices, meshScale=[1, 1, 1], flags=p.GEOM_FORCE_CONCAVE_TRIMESH)  # 强制使用凹面网格，不要进行凸分解

            # 创建多边形体
            body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_mesh_data, baseVisualShapeIndex=visual_mesh_data, basePosition=center, baseOrientation=Rotation.from_matrix(rotation_matrix).as_quat())

            return body_id

        except Exception as e:
            print(f"Error creating polygon channel: {e}")
            import traceback

            traceback.print_exc()
            # 如果创建多边形失败，退回到创建简单的盒子
            print("Falling back to a simple box shape")
            # 估计多边形的大小
            max_dimension = np.max([np.linalg.norm(p - center) for p in points]) * 2
            box_id = pp.create_box(max_dimension, max_dimension, thickness)

            # 检查normal_or_matrix的类型
            if normal_or_matrix.shape == (3,):  # 是法向量
                normal = normal_or_matrix
                # 计算旋转矩阵
                z_axis = normal / np.linalg.norm(normal)
                temp_x = np.array([1, 0, 0])
                if np.abs(np.dot(temp_x, z_axis)) > 0.9:
                    temp_x = np.array([0, 1, 0])
                y_axis = np.cross(z_axis, temp_x)
                y_axis = y_axis / np.linalg.norm(y_axis)
                x_axis = np.cross(y_axis, z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
                rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
            else:  # 是旋转矩阵
                rotation_matrix = normal_or_matrix

            pp.set_pose(box_id, pp.Pose(point=center, euler=Rotation.from_matrix(rotation_matrix).as_euler("xyz")))
            return box_id

    @staticmethod
    def load_channels(channels_info: List[Dict]):
        channel_bodies = []
        for channel_info in channels_info:
            channel_bodies.append(SceneParser.load_channel(channel_info))
        return channel_bodies

    @staticmethod
    def compute_channel_center(channel_info: Dict) -> np.ndarray:
        """
        计算通道的中心点
        """
        if channel_info["type"] == "rectangle":
            return np.array(channel_info["center"])
        elif channel_info["type"] == "circle":
            return np.array(channel_info["center"])
        elif channel_info["type"] == "polygon":
            # 检查是否已经有中心点
            if "center" in channel_info:
                return np.array(channel_info["center"])

            # 如果没有中心点，需要通过拟合平面和投影计算
            if "points" not in channel_info:
                raise ValueError("Polygon channel requires 'points' attribute to compute center")

            points = np.array(channel_info["points"])
            if len(points) < 3:
                raise ValueError("需要至少3个点来定义一个多边形")

            # 拟合平面
            normal, center = SceneParser.fit_plane_to_points(points)

            # 将点投影到平面上
            projected_points = SceneParser.project_points_to_plane(points, normal, center)

            # 计算投影点的均值作为中心点
            return np.mean(projected_points, axis=0)
        else:
            raise ValueError(f"Unsupported channel type: {channel_info['type']}")

    @staticmethod
    def get_k_closest_channel(channel_info: Dict, channels_info: List[Dict], k: int) -> Tuple[List[Dict], List[int]]:
        """
        获取k个最近的通道

        Args:
            channel_info: 参考通道
            channels_info: 所有通道的列表
            k: 要返回的最近通道数量

        Returns:
            Tuple[List[Dict], List[int]]: 包含k个最近的通道和它们在原始列表中的索引
        """
        center = SceneParser.compute_channel_center(channel_info)

        # 创建(索引, 通道)对，排除当前通道
        indexed_channels = [(i, ch) for i, ch in enumerate(channels_info) if ch != channel_info]
        if not indexed_channels:
            return [], []  # 如果没有其他通道，返回空列表

        # 计算当前通道中心到其他所有通道中心的距离
        distances = [np.linalg.norm(center - SceneParser.compute_channel_center(ch)) for _, ch in indexed_channels]

        # 按距离排序并返回前k个最近的通道及其索引
        sorted_items = sorted(zip(distances, indexed_channels), key=lambda x: x[0])

        # 确保k不超过可用通道数量
        k_valid = min(k, len(sorted_items))

        # 分离结果为通道列表和索引列表
        closest_channels = []
        original_indices = []

        for i in range(k_valid):
            idx, channel = sorted_items[i][1]
            closest_channels.append(channel)
            original_indices.append(idx)

        return closest_channels, original_indices

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
        self._load_scene()

    def _load_scene(self):
        """
        Load and parse the scene data from the YAML file.
        Converts the raw data into a SimpleNamespace object for easier access.
        """
        with open(self.scene_file, "r", encoding="utf-8") as file:
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

    def _approximate_elements_with_spheres(self) -> List[Dict]:
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
        rotation_matrix = SceneParser.quaternion_to_rotation_matrix(orientation)
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
        return self.robot_info.grasp.offset

    def get_robot_grasp_approximate_offset(self) -> List[float]:
        """
        Get the robot's grasp approximate offset in tool_link frame.
        """
        if not self.robot_info:
            raise ValueError("Robot information not loaded")
        return self.robot_info.grasp.approximate.offset

    def get_robot_grasp_type(self) -> str:
        """
        Get the robot's grasp type.

        Returns:
            str: Grasp type
        """
        if not self.robot_info:
            raise ValueError("Robot information not loaded")
        return self.robot_info.grasp.type

    def get_robot_grasp_rotation(self) -> List[float]:
        """
        Get the robot's grasp rotation.

        Returns:
            List[float]: Grasp rotation [x, y, z]
        """
        if not self.robot_info:
            raise ValueError("Robot information not loaded")
        return self.robot_info.grasp.rotation

    def get_robot_grasp_approximate_rotation(self) -> List[float]:
        """
        Get the robot's grasp approximate rotation.

        Returns:
            List[float]: Grasp approximate rotation [x, y, z]
        """
        if not self.robot_info:
            raise ValueError("Robot information not loaded")
        return self.robot_info.grasp.approximate.rotation

    def get_robot_grasp_file(self) -> str:
        """
        Get the robot's grasp file.
        """
        if not self.robot_info:
            raise ValueError("Robot information not loaded")
        return os.path.join(HERE, "model", "obj", "grasp", self.robot_info.grasp.file)

    def get_robot_grasp_size(self) -> Union[List[float], float]:
        """
        Get the robot's grasp size.
        """
        if not self.robot_info:
            raise ValueError("Robot information not loaded")
        return self.robot_info.grasp.size

    def get_robot_grasp_approximate(self) -> Dict:
        """
        Get the robot's grasp approximate.

        Returns:
            Dict: Dictionary containing grasp approximation information
        """
        if not self.robot_info:
            raise ValueError("Robot information not loaded")
        approximate_info = self.robot_info.grasp.approximate.__dict__
        return approximate_info

    def get_robot_grasp_pose(self):
        """
        Get the robot's grasp pose.
        """
        return pp.Pose(point=self.get_robot_grasp_offset(), euler=self.get_robot_grasp_rotation())

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
                rotation_matrix = SceneParser.quaternion_to_rotation_matrix(orientation)
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

    def get_channel_info(self) -> List[Dict]:
        """
        Get channel information from scene elements.

        Returns:
            List[Dict]: A list of dictionaries containing channel information
        """
        if not self.scene_data:
            raise ValueError("Scene data is not loaded. Call load_scene() first.")

        channel_info = []
        if hasattr(self.scene_data, "channels_info"):
            for channel in self.scene_data.channels_info:
                channel_data = {
                    "type": channel.type,
                    "thickness": channel.thickness,
                }

                # 添加特定类型channel的属性
                if channel.type in ["rectangle", "circle"]:
                    channel_data["center"] = channel.center
                    channel_data["direction"] = channel.direction
                    channel_data["size"] = channel.size
                elif channel.type == "polygon":
                    if hasattr(channel, "points"):
                        channel_data["points"] = channel.points
                    else:
                        raise ValueError(f"Polygon channel requires 'points' attribute but none was found")

                    if hasattr(channel, "center"):
                        channel_data["center"] = channel.center
                    if hasattr(channel, "direction"):
                        channel_data["direction"] = channel.direction

                channel_info.append(channel_data)

        return channel_info

    def create_elements(self, color=None) -> Tuple[List[int], Dict]:
        """
        创建并返回场景中定义的所有圆柱体元素的PyBullet物理对象列表。

        Args:
            color (Tuple, optional): 圆柱体的颜色，格式为 (r, g, b, a)。默认为None，将使用随机颜色。

        Returns:
            List[int]: 创建的圆柱体物理对象ID列表
        """
        if not self.scene_data:
            raise ValueError("Scene data is not loaded. Call load_scene() first.")

        element_bodies = []
        element_infos = {}

        for element in self.scene_data.elements:
            if element.shape.type == "cylinder":
                # 提取圆柱体参数
                position = element.position
                orientation = element.orientation  # 四元数 [x, y, z, w]
                euler = pp.euler_from_quat(orientation)
                radius = element.shape.parameters.radius
                height = element.shape.parameters.height

                # 为每个元素创建随机颜色（如果未指定颜色）
                if color is None:
                    element_color = (random.random(), random.random(), random.random(), 1)
                else:
                    element_color = color

                # 创建圆柱体
                cylinder_id = pp.create_cylinder(radius, height, color=element_color)

                # 设置位置和方向
                pp.set_pose(cylinder_id, pp.Pose(point=position, euler=euler))

                # 保存圆柱体ID
                element_bodies.append(cylinder_id)
                element_infos[cylinder_id] = {"position": position, "orientation": orientation, "shape_type": "cylinder", "shape_parameters": {"radius": radius, "height": height}}

            # 注意：如果需要支持其他形状如立方体或球体，可以在这里添加更多的条件分支

        return element_bodies, element_infos

    def create_attachment(self, robot: RobotSetup, approximate: bool = False) -> Union[Tuple[int, pp.Attachment], Tuple[int, pp.Attachment, int, pp.Attachment]]:
        """
        创建并返回场景中定义的所有连接器元素的PyBullet物理对象列表。

        Args:
            robot (RobotSetup): 机器人对象
            approximate (bool, optional): 是否使用近似形状. Defaults to False.

        Returns:
            Union[Tuple[int, pp.Attachment], Tuple[int, pp.Attachment, int, pp.Attachment]]: 创建的连接器物理对象ID列表
        """
        if not self.scene_data:
            raise ValueError("Scene data is not loaded. Call load_scene() first.")

        if self.get_robot_grasp_type() == "element":
            attachment_body = pp.create_cylinder(self.get_robot_grasp_size()[0], self.get_robot_grasp_size()[1])
            delta_pose = pp.Pose(point=self.get_robot_grasp_offset(), euler=self.get_robot_grasp_rotation())
            pose = pp.multiply(pp.get_link_pose(robot.robot, robot.tool_link), delta_pose)
            pp.set_pose(attachment_body, pose)
            attachment = pp.create_attachment(robot.robot, robot.tool_link, attachment_body)
        elif self.get_robot_grasp_type() == "mesh":
            attachment_body = pp.create_obj(self.get_robot_grasp_file(), scale=self.get_robot_grasp_size())
            delta_pose = pp.Pose(point=self.get_robot_grasp_offset(), euler=self.get_robot_grasp_rotation())
            pose = pp.multiply(pp.get_link_pose(robot.robot, robot.tool_link), delta_pose)
            pp.set_pose(attachment_body, pose)
            attachment = pp.create_attachment(robot.robot, robot.tool_link, attachment_body)
        elif self.get_robot_grasp_type() == "box":
            attachment_body = pp.create_box(*self.get_robot_grasp_size())
            delta_pose = pp.Pose(point=self.get_robot_grasp_offset(), euler=self.get_robot_grasp_rotation())
            pose = pp.multiply(pp.get_link_pose(robot.robot, robot.tool_link), delta_pose)
            pp.set_pose(attachment_body, pose)
            attachment = pp.create_attachment(robot.robot, robot.tool_link, attachment_body)
        else:
            raise ValueError(f"Unknown grasp type: {self.get_robot_grasp_type()}")

        if approximate:
            approximate_info = self.get_robot_grasp_approximate()
            if approximate_info["type"] == "cylinder":
                approximate_attachment_body = pp.create_cylinder(approximate_info["radius"], approximate_info["height"], color=[1, 0, 0, 1])
            elif approximate_info["type"] == "sphere":
                approximate_attachment_body = pp.create_sphere(approximate_info["radius"], color=[1, 0, 0, 1])
            elif approximate_info["type"] == "box":
                approximate_attachment_body = pp.create_box(*approximate_info["size"])
            pp.set_color(approximate_attachment_body, [1, 0, 0, 0.75])
            approximate_delta_pose = pp.Pose(point=self.get_robot_grasp_approximate_offset(), euler=self.get_robot_grasp_approximate_rotation())
            approximate_pose = pp.multiply(pp.get_link_pose(robot.robot, robot.tool_link), approximate_delta_pose)
            pp.set_pose(approximate_attachment_body, approximate_pose)
            approximate_attachment = pp.create_attachment(robot.robot, robot.tool_link, approximate_attachment_body)
            return attachment_body, attachment, approximate_attachment_body, approximate_attachment

        return attachment_body, attachment

    def create_robot(self, name: str) -> RobotSetup:
        """
        创建并返回机器人对象。

        Returns:
            RobotSetup: 创建的机器人对象
        """
        rb = RobotSetup(name)
        rb.set_base_pose_2d(*self.get_robot_pose_2d())
        rb.set_joint_positions(rb.arm_joints, self.get_robot_start_pose())
        return rb


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
    vis_group.add_argument("--scene", type=str, default="cuboid_1", help="场景名称")
    vis_group.add_argument("--task", type=str, default="task_1", help="任务名称")
    vis_group.add_argument("--enable-spheres", action="store_true", help="显示球体近似")
    vis_group.add_argument("--enable-channels", action="store_true", help="显示通道")
    # 重排布模式的参数
    reorg_group = parser.add_argument_group("重排布参数")
    reorg_group.add_argument("--target-scene", type=str, help="要重排布的场景名称")

    args = parser.parse_args()

    if args.mode == "visualize":
        # 修改参数检查
        if not all([args.scene, args.task]):
            parser.error("可视化模式需要指定 --scene 和 --task")
        # 构建场景文件路径
        scene_file = os.path.join(HERE, "model", "scenes", args.scene, f"{args.task}.yml")

        init_pb()
        scene_parser = SceneParser(scene_file)

        with pp.LockRenderer():
            robot = scene_parser.create_robot("r0")
            element_bodies = scene_parser.create_elements()
            attachment_body, attachment = scene_parser.create_attachment(robot)
            robot.update_attachments([attachment])
            channel_bodies = SceneParser.load_channels(scene_parser.get_channel_info())
            points = SceneParser.sample_points_in_channel(scene_parser.get_channel_info()[0]["type"], scene_parser.get_channel_info()[0], scene_parser.get_channel_info()[0]["thickness"], ratio=0.5)
            # points = SceneParser.sample_points_in_polgon(scene_parser.get_channel_info()[7]["points"])
            for point in points:
                pp.draw_point(point, size=0.05)

        start_pose = scene_parser.get_robot_start_pose()
        target_pose = scene_parser.get_robot_target_pose()

        # 添加按钮来切换机器人姿态
        start_button = p.addUserDebugParameter("start", 1, 0, 1)
        target_button = p.addUserDebugParameter("target", 1, 0, 1)

        # 初始化按钮状态
        prev_start_button_value = p.readUserDebugParameter(start_button)
        prev_target_button_value = p.readUserDebugParameter(target_button)

        while True:
            start_button_value = p.readUserDebugParameter(start_button)
            target_button_value = p.readUserDebugParameter(target_button)

            if start_button_value != prev_start_button_value:
                robot.set_joint_positions(robot.arm_joints, start_pose)
                prev_start_button_value = start_button_value
            if target_button_value != prev_target_button_value:
                robot.set_joint_positions(robot.arm_joints, target_pose)
                prev_target_button_value = target_button_value
            time.sleep(1.0 / 240)

    else:  # reorganize mode
        if not args.target_scene:
            parser.error("重排布模式需要指定 --target-scene")
        reorganize_tasks(args.target_scene)
