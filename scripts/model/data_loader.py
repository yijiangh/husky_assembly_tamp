import glob
import os
import sys
from typing import Dict, List, Optional, Tuple, Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from model.scene_parse import SceneParser
from utils.params import *


class PointCloudDataset(Dataset):
    """
    用于加载点云和轨迹数据的PyTorch Dataset
    """

    def __init__(
        self,
        scene_names: Union[str, List[str]],
        task_names: Union[str, List[str]],
        algorithm_names: Union[str, List[str]],
        data_loader=None,
        num_points: int = 1024,
        normal_channel: bool = True,
        trajectory_length: Optional[int] = None,
        transform: Optional[Callable] = None,
    ):
        """
        初始化点云数据集

        Args:
            scene_names: 场景名称或列表
            task_names: 任务名称或列表
            algorithm_names: 算法名称或列表
            data_loader: 已有的DataLoader实例，如果为None则创建新实例
            num_points: 每个元素采样的点数
            normal_channel: 是否包含法向量
            trajectory_length: 轨迹目标长度，如果指定则对轨迹进行重新插值
            transform: 应用于点云的变换函数
        """
        self.data_loader = data_loader if data_loader else SceneDataLoader()
        self.num_points = num_points
        self.normal_channel = normal_channel
        self.trajectory_length = trajectory_length
        self.transform = transform

        # 将单个字符串转换为列表
        if isinstance(scene_names, str):
            scene_names = [scene_names]
        if isinstance(task_names, str):
            task_names = [task_names]
        if isinstance(algorithm_names, str):
            algorithm_names = [algorithm_names]

        self.scene_names = scene_names
        self.task_names = task_names
        self.algorithm_names = algorithm_names

        # 存储所有数据源信息
        self.data_sources = []

        # 收集所有匹配的数据
        for scene in scene_names:
            tasks = task_names if task_names else self.data_loader.list_tasks_for_scene(scene)
            for task in tasks:
                algorithms = (
                    algorithm_names if algorithm_names else self.data_loader.list_algorithms_for_task(scene, task)
                )
                for algorithm in algorithms:
                    trajectories = self.data_loader.load_trajectories(scene, task, algorithm)
                    if trajectories:
                        self.data_sources.append(
                            {"scene": scene, "task": task, "algorithm": algorithm, "count": len(trajectories)}
                        )

        # 计算数据总数
        self.total_count = sum(source["count"] for source in self.data_sources)

        # 缓存每个场景的点云，避免重复计算
        self.point_cloud_cache = {}

    def __len__(self):
        """返回数据集大小"""
        return self.total_count

    def __getitem__(self, idx):
        """获取单个数据样本"""
        # 找到对应的数据源
        current_idx = idx
        source = None
        for s in self.data_sources:
            if current_idx < s["count"]:
                source = s
                break
            current_idx -= s["count"]

        if source is None:
            raise IndexError(f"索引 {idx} 超出了数据集范围")

        scene = source["scene"]
        task = source["task"]
        algorithm = source["algorithm"]

        # 从缓存加载或生成点云
        scene_task_key = f"{scene}_{task}"
        if scene_task_key not in self.point_cloud_cache:
            # 从数据源加载点云
            parser = self.data_loader.load_scene_config(scene, task)
            line_pts, radius_per_edge = parser.get_element_info()

            # 生成点云数据和元素标签
            point_cloud, element_labels = self.data_loader._sample_points_from_elements(
                line_pts, radius_per_edge, num_points=self.num_points, normal_channel=self.normal_channel
            )
            self.point_cloud_cache[scene_task_key] = (point_cloud, element_labels)
        else:
            point_cloud, element_labels = self.point_cloud_cache[scene_task_key]

        # 加载轨迹数据，使用trajectory_length参数
        trajectories = self.data_loader.load_trajectories(scene, task, algorithm, target_length=self.trajectory_length)
        trajectory = trajectories[current_idx]

        # 应用变换（如果有）
        if self.transform:
            point_cloud = self.transform(point_cloud)

        # 转换为tensor
        point_cloud_tensor = torch.FloatTensor(point_cloud)
        element_labels_tensor = torch.LongTensor(element_labels)
        trajectory_tensor = torch.FloatTensor(trajectory)

        return {
            "point_cloud": point_cloud_tensor,
            "element_labels": element_labels_tensor,
            "trajectory": trajectory_tensor,
            "scene": scene,
            "task": task,
            "algorithm": algorithm,
        }

    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=4, **kwargs):
        """
        获取PyTorch DataLoader

        Args:
            batch_size: 批次大小
            shuffle: 是否随机打乱数据
            num_workers: 加载数据的工作线程数
            **kwargs: 传递给torch.utils.data.DataLoader的额外参数

        Returns:
            torch.utils.data.DataLoader: PyTorch数据加载器
        """
        return TorchDataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)


class SceneDataLoader:
    """
    加载场景配置和机械臂轨迹数据的数据加载器

    数据来源：
    - 轨迹数据：data/${scene_name}/${task_name}/${algorithm_name}/plan_{id}.npy
    - 场景配置：scenes/${scene_name}/${task_name}.yml
    """

    def __init__(self):
        """
        初始化数据加载器
        """
        self.data_dir = os.path.join(HERE, "model", "data")
        self.scenes_dir = os.path.join(HERE, "model", "scenes")

    def list_available_scenes(self) -> List[str]:
        """
        列出所有可用的场景

        Returns:
            List[str]: 场景名称列表
        """
        return [os.path.basename(f) for f in glob.glob(os.path.join(self.scenes_dir, "*")) if os.path.isdir(f)]

    def list_tasks_for_scene(self, scene_name: str) -> List[str]:
        """
        列出指定场景的所有任务

        Args:
            scene_name: 场景名称

        Returns:
            List[str]: 任务名称列表
        """
        yaml_files = glob.glob(os.path.join(self.scenes_dir, scene_name, "*.yml"))
        return [os.path.splitext(os.path.basename(f))[0] for f in yaml_files]

    def list_algorithms_for_task(self, scene_name: str, task_name: str) -> List[str]:
        """
        列出指定场景和任务的所有算法

        Args:
            scene_name: 场景名称
            task_name: 任务名称

        Returns:
            List[str]: 算法名称列表
        """
        task_dir = os.path.join(self.data_dir, scene_name, task_name)
        if not os.path.exists(task_dir):
            return []
        return [os.path.basename(f) for f in glob.glob(os.path.join(task_dir, "*")) if os.path.isdir(f)]

    def load_trajectories(
        self, scene_name: str, task_name: str, algorithm_name: str, target_length: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        加载指定场景、任务和算法的所有轨迹数据，并可选择重新插值到指定长度

        Args:
            scene_name: 场景名称
            task_name: 任务名称
            algorithm_name: 算法名称
            target_length: 目标轨迹长度，如果指定则对轨迹进行重新插值

        Returns:
            List[np.ndarray]: 轨迹数据列表
        """
        alg_dir = os.path.join(self.data_dir, scene_name, task_name, algorithm_name)
        if not os.path.exists(alg_dir):
            return []

        trajectory_files = sorted(glob.glob(os.path.join(alg_dir, "plan_*.npy")))
        raw_trajectories = []

        # 先加载所有原始轨迹，并找出最长的轨迹长度
        max_length = 0
        for traj_file in trajectory_files:
            try:
                trajectory = np.load(traj_file)
                raw_trajectories.append(trajectory)
                max_length = max(max_length, len(trajectory))
            except Exception as e:
                print(f"无法加载轨迹文件 {traj_file}: {e}")

        # 确定实际使用的插值长度
        actual_target_length = target_length
        if target_length is not None and max_length > target_length:
            print(
                f"Warning: Maximum trajectory length ({max_length}) is greater than target length ({target_length}), will use maximum length as interpolation target"
            )
            actual_target_length = max_length

        # 进行插值处理
        trajectories = []
        for trajectory in raw_trajectories:
            if actual_target_length is not None and len(trajectory) != actual_target_length:
                trajectory = self._interpolate_trajectory(trajectory, actual_target_length)
            trajectories.append(trajectory)

        return trajectories

    def _interpolate_trajectory(self, trajectory: np.ndarray, target_length: int) -> np.ndarray:
        """
        将轨迹重新插值到指定长度，确保保留原始轨迹中的所有点

        Args:
            trajectory: 原始轨迹数据，形状为 [N, D]，其中 N 是时间步数，D 是每个步骤的维度
            target_length: 目标轨迹长度

        Returns:
            np.ndarray: 重新插值后的轨迹，形状为 [target_length, D]
        """
        # 原始轨迹长度和维度
        orig_length, dims = trajectory.shape

        # 如果目标长度小于原始长度，则需要进行下采样
        if target_length <= orig_length:
            # 选择等间隔的索引
            indices = np.round(np.linspace(0, orig_length - 1, target_length)).astype(int)
            return trajectory[indices]

        # 创建新的轨迹数组，初始化为零
        new_trajectory = np.zeros((target_length, dims))

        # 首先确保原始轨迹中的所有点都被保留
        # 计算我们需要保留原始点的索引
        orig_indices_in_new = np.round(np.linspace(0, target_length - 1, orig_length)).astype(int)

        # 将原始点放到新轨迹中
        for i, idx in enumerate(orig_indices_in_new):
            new_trajectory[idx] = trajectory[i]

        # 对于新轨迹中的每个尚未赋值的位置，通过插值填充
        # 创建一个掩码来标记哪些位置已经被赋值
        mask = np.zeros(target_length, dtype=bool)
        mask[orig_indices_in_new] = True

        # 为未赋值的位置创建插值
        for i in range(target_length):
            if not mask[i]:
                # 找到左右最近的已知点
                left_idx = np.max(orig_indices_in_new[orig_indices_in_new < i]) if any(orig_indices_in_new < i) else 0
                right_idx = (
                    np.min(orig_indices_in_new[orig_indices_in_new > i])
                    if any(orig_indices_in_new > i)
                    else target_length - 1
                )

                # 如果左右索引相同，则无法插值，使用最近的点
                if left_idx == right_idx:
                    new_trajectory[i] = new_trajectory[left_idx]
                    continue

                # 计算插值权重
                left_orig_idx = np.where(orig_indices_in_new == left_idx)[0][0]
                right_orig_idx = np.where(orig_indices_in_new == right_idx)[0][0]

                weight = (i - left_idx) / (right_idx - left_idx)

                # 线性插值
                new_trajectory[i] = (1 - weight) * trajectory[left_orig_idx] + weight * trajectory[right_orig_idx]

        return new_trajectory

    def load_scene_config(self, scene_name: str, task_name: str) -> SceneParser:
        """
        加载并解析场景配置

        Args:
            scene_name: 场景名称
            task_name: 任务名称

        Returns:
            SceneParser: 包含场景配置的场景解析器
        """
        scene_file = os.path.join(self.scenes_dir, scene_name, f"{task_name}.yml")
        if not os.path.exists(scene_file):
            raise FileNotFoundError(f"场景配置文件不存在: {scene_file}")

        parser = SceneParser(scene_file)
        parser.load_scene()
        return parser

    def prepare_point_cloud_data(
        self,
        scene_name: str,
        task_name: str,
        algorithm_name: str,
        normal_channel: bool = True,
        num_points: int = 1024,
        trajectory_length: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        准备用于PointNet模型的点云数据、元素标签和对应的轨迹

        Args:
            scene_name: 场景名称
            task_name: 任务名称
            algorithm_name: 算法名称
            normal_channel: 是否包含法向量通道
            num_points: 每个元素要采样的点数
            trajectory_length: 轨迹目标长度，如果指定则对轨迹进行重新插值

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]: 点云数据、元素标签和对应的轨迹
        """
        # 加载轨迹
        trajectories = self.load_trajectories(scene_name, task_name, algorithm_name, trajectory_length)
        if not trajectories:
            return [], [], []

        # 加载场景配置
        parser = self.load_scene_config(scene_name, task_name)

        # 获取场景元素信息
        line_pts, radius_per_edge = parser.get_element_info()

        # 从元素中采样点（包含法向量计算和元素标签）
        point_cloud, element_labels = self._sample_points_from_elements(
            line_pts, radius_per_edge, num_points=num_points, normal_channel=normal_channel
        )

        # 为每个轨迹创建相同的点云和标签
        point_clouds = [point_cloud] * len(trajectories)
        element_labels_list = [element_labels] * len(trajectories)

        return point_clouds, element_labels_list, trajectories

    def _sample_points_from_elements(
        self,
        line_pts: List[np.ndarray],
        radius_per_edge: List[float],
        num_points: int = 1024,
        normal_channel: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从元素圆柱体表面随机采样点，并添加元素标签

        Args:
            line_pts: 线段点列表
            radius_per_edge: 每个线段的半径
            num_points: 每个元素要采样的点数
            normal_channel: 是否计算法向量

        Returns:
            Tuple[np.ndarray, np.ndarray]: 点云数据和元素标签
        """
        # 从圆柱体表面采样点
        sampled_points = []
        normals = [] if normal_channel else None
        element_labels = []  # 存储每个点所属的元素索引

        # 对每个圆柱体元素采样
        num_elements = len(radius_per_edge)
        for element_idx in range(num_elements):
            i = element_idx * 2  # 每个元素由两个点定义
            if i + 1 >= len(line_pts):
                break

            start, end = line_pts[i], line_pts[i + 1]
            radius = radius_per_edge[element_idx]

            # 计算圆柱体轴向方向
            direction = end - start
            if np.linalg.norm(direction) < 1e-6:  # 避免零长度情况
                continue
            direction = direction / np.linalg.norm(direction)

            # 创建圆柱体坐标系
            if abs(direction[0]) < 0.9:
                v = np.array([1.0, 0.0, 0.0])
            else:
                v = np.array([0.0, 1.0, 0.0])

            base1 = np.cross(direction, v)
            base1 = base1 / np.linalg.norm(base1)

            base2 = np.cross(direction, base1)
            base2 = base2 / np.linalg.norm(base2)

            # 对这个元素采样num_points个点
            for _ in range(num_points):
                # 沿轴线随机位置
                t = np.random.uniform(0, 1)
                center = start * (1 - t) + end * t

                # 在圆周上的随机角度
                theta = np.random.uniform(0, 2 * np.pi)

                # 计算圆柱体表面的点
                radial_vec = np.cos(theta) * base1 + np.sin(theta) * base2
                surface_point = center + radius * radial_vec
                sampled_points.append(surface_point)

                # 添加元素标签
                element_labels.append(element_idx)

                # 如果需要法向量，则计算法向量
                if normal_channel:
                    normals.append(radial_vec)  # 法向量就是从轴到表面的方向

        # 转换为numpy数组
        points_array = np.array(sampled_points)
        labels_array = np.array(element_labels)

        # 如果需要法向量，则拼接点和法向量
        if normal_channel and points_array.size > 0:
            normals_array = np.array(normals)
            return np.hstack((points_array, normals_array)), labels_array

        return points_array, labels_array

    def visualize_point_cloud(self, scene_name: str, task_name: str, num_points: int = 1024, show_normals: bool = True):
        """
        读取场景文件，生成点云数据并使用matplotlib可视化

        Args:
            scene_name: 场景名称
            task_name: 任务名称
            num_points: 每个元素要采样的点数
            show_normals: 是否显示法向量
        """
        # 加载场景
        parser = self.load_scene_config(scene_name, task_name)

        # 获取元素信息
        line_pts, radius_per_edge = parser.get_element_info()

        # 采样点云数据
        point_cloud, _ = self._sample_points_from_elements(
            line_pts, radius_per_edge, num_points=num_points, normal_channel=True
        )

        # 创建3D图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # 分离点坐标和法向量
        points = point_cloud[:, :3]
        normals = point_cloud[:, 3:] if point_cloud.shape[1] > 3 else None

        # 计算用于绘制的颜色
        num_elements = len(radius_per_edge)
        colors = []

        # 为每个元素分配独特的颜色
        for i in range(0, len(line_pts), 2):
            if i + 1 >= len(line_pts):
                break

            element_index = i // 2
            # 创建颜色循环以区分不同元素
            color = plt.cm.tab20(element_index % 20)
            # 为当前元素的所有点指定相同颜色
            colors.extend([color] * num_points)

        # 绘制点云
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, marker="o", s=5, alpha=0.8)

        # 可选地绘制法向量
        if show_normals and normals is not None:
            # 为了清晰可见，只显示部分法向量
            skip = 10  # 每10个点显示一个法向量
            for i in range(0, len(points), skip):
                # 绘制短向量表示法向量方向
                ax.quiver(
                    points[i, 0],
                    points[i, 1],
                    points[i, 2],
                    normals[i, 0],
                    normals[i, 1],
                    normals[i, 2],
                    color="red",
                    length=0.02,
                    normalize=True,
                )

        # 绘制元素中心线
        for i in range(0, len(line_pts), 2):
            if i + 1 >= len(line_pts):
                break

            start, end = line_pts[i], line_pts[i + 1]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], "k-", linewidth=1, alpha=0.5)

        # Set figure properties
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Point Cloud Visualization: {scene_name}/{task_name}")

        # 设置轴比例相等，以保持正确的形状
        max_range = (
            np.array(
                [
                    points[:, 0].max() - points[:, 0].min(),
                    points[:, 1].max() - points[:, 1].min(),
                    points[:, 2].max() - points[:, 2].min(),
                ]
            ).max()
            / 2.0
        )

        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()
        plt.show()

    def create_dataset(
        self,
        scene_names: Union[str, List[str]],
        task_names: Union[str, List[str]] = None,
        algorithm_names: Union[str, List[str]] = None,
        num_points: int = 1024,
        normal_channel: bool = True,
        trajectory_length: Optional[int] = None,
        transform: Optional[Callable] = None,
    ) -> PointCloudDataset:
        """
        创建用于PyTorch的点云数据集

        Args:
            scene_names: 场景名称或列表
            task_names: 任务名称或列表，如果为None则使用所有任务
            algorithm_names: 算法名称或列表，如果为None则使用所有算法
            num_points: 每个元素采样的点数
            normal_channel: 是否包含法向量
            trajectory_length: 轨迹目标长度，如果指定则对轨迹进行重新插值
            transform: 应用于点云的变换函数

        Returns:
            PointCloudDataset: PyTorch数据集
        """
        return PointCloudDataset(
            scene_names=scene_names,
            task_names=task_names,
            algorithm_names=algorithm_names,
            data_loader=self,
            num_points=num_points,
            normal_channel=normal_channel,
            trajectory_length=trajectory_length,
            transform=transform,
        )


if __name__ == "__main__":
    loader = SceneDataLoader()
    # loader.visualize_point_cloud("cuboid_1", "task_1", show_normals=False)
    data = loader.prepare_point_cloud_data("cuboid_1", "task_1", "BIRRT")
