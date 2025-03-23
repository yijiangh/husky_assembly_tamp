import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
import os
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BallCloudDataset(Dataset):
    """
    数据集类，处理由小球组成的障碍物场景
    每个场景包含障碍物小球表示和通道信息
    """

    def __init__(self, data_path=None, num_samples=1000, generate_samples=True):
        """
        Params:
            data_path: 数据文件路径，如果为None则生成随机数据
            num_samples: 生成的样本数量
            generate_samples: 是否生成随机样本数据
        """
        self.data_path = data_path
        self.samples = []

        if data_path and os.path.exists(data_path):
            logger.info(f"Loading data from {data_path}")
            self.samples = torch.load(data_path)
        elif generate_samples:
            logger.info(f"Generating {num_samples} random samples")
            self.samples = self._generate_random_samples(num_samples)

    def _generate_random_samples(self, num_samples):
        """
        生成随机场景样本 - 创建结构化的通道边界
        """
        samples = []

        # 添加进度条
        pbar = tqdm(range(num_samples), desc="Generate scene samples")
        for _ in pbar:
            # 决定场景中通道的数量
            num_channels = np.random.randint(1, 21)
            all_obstacles = []

            # 生成多个通道
            channels_info = []
            for channel_idx in range(num_channels):
                # 随机决定通道类型: 椭圆形或矩形
                channel_type = np.random.choice(["ellipse", "rectangle"])

                # 生成通道中心
                channel_center = torch.randn(3) * 2.0

                # 生成随机方向向量
                channel_dir = F.normalize(torch.randn(3), dim=0)

                # 创建通道坐标系
                z_axis = channel_dir
                x_axis = F.normalize(
                    torch.linalg.cross(
                        z_axis, torch.tensor([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 else torch.tensor([0.0, 1.0, 0.0])
                    ),
                    dim=0,
                )
                y_axis = F.normalize(torch.linalg.cross(z_axis, x_axis), dim=0)

                # 通道参数
                if channel_type == "ellipse":
                    # 生成椭圆的长轴和短轴
                    # 长轴范围 [0.5, 1.0]
                    # 短轴范围 [0.25, 0.5]
                    channel_size = torch.tensor(
                        [torch.rand(1).item() * 0.5 + 0.5, torch.rand(1).item() * 0.25 + 0.25]
                    )  # 长轴和短轴
                else:  # rectangle
                    channel_size = torch.rand(2) * 0.75 + 0.25  # 矩形通道长宽 [0.25, 1.0]

                # 通道轮廓点数量
                num_boundary_points = np.random.randint(75, 225)

                # 通道小球半径
                ball_radius = torch.rand(1).item() * 0.09 + 0.01  # [0.01, 0.1]

                # 存储通道边界小球
                boundary_obstacles = []

                # 根据通道类型生成边界点
                if channel_type == "ellipse":
                    # 椭圆通道
                    a = channel_size[0]  # 长轴
                    b = channel_size[1]  # 短轴

                    for i in range(num_boundary_points):
                        angle = (i / num_boundary_points) * 2 * np.pi
                        # 椭圆上的点
                        local_pos = (
                            a * torch.cos(torch.tensor(angle)) * x_axis + b * torch.sin(torch.tensor(angle)) * y_axis
                        )

                        # 添加一些随机扰动
                        if np.random.random() > 0.4:  # 60%的几率添加Z轴扰动
                            z_distortion = (torch.rand(1) - 0.5) * 0.7  # [-0.35, 0.35]
                            local_pos = local_pos + z_distortion * z_axis

                        # 全局坐标
                        pos = channel_center + local_pos

                        # 添加小球
                        obstacle = torch.cat([pos, ball_radius * torch.ones(1)])
                        boundary_obstacles.append(obstacle)

                else:  # rectangle
                    # 矩形通道
                    width = channel_size[0]
                    height = channel_size[1]

                    # 矩形四条边的点分布
                    sides = 4
                    points_per_side = num_boundary_points // sides

                    for side in range(sides):
                        for j in range(points_per_side):
                            progress = j / points_per_side

                            if side == 0:  # 上边
                                local_pos = (width * (progress - 0.5)) * x_axis + (height / 2) * y_axis
                            elif side == 1:  # 右边
                                local_pos = (width / 2) * x_axis + (height * (0.5 - progress)) * y_axis
                            elif side == 2:  # 下边
                                local_pos = (width * (0.5 - progress)) * x_axis - (height / 2) * y_axis
                            else:  # 左边
                                local_pos = -(width / 2) * x_axis + (height * (progress - 0.5)) * y_axis

                            # 添加一些随机扰动
                            if np.random.random() > 0.3:  # 70%的几率添加扰动
                                z_distortion = (torch.rand(1) - 0.5) * 0.1  # [-0.05, 0.05]
                                local_pos = local_pos + z_distortion * z_axis

                                # 微小的x-y扰动
                                xy_distortion = (torch.rand(2) - 0.5) * 0.05  # [-0.05, 0.05]
                                local_pos = local_pos + xy_distortion[0] * x_axis + xy_distortion[1] * y_axis

                            # 全局坐标
                            pos = channel_center + local_pos

                            # 添加小球
                            obstacle = torch.cat([pos, ball_radius * torch.ones(1)])
                            boundary_obstacles.append(obstacle)

                # 将边界小球转换为张量并添加到障碍物列表
                boundary_obstacles = torch.stack(boundary_obstacles)
                all_obstacles.append(boundary_obstacles)

                # 保存通道信息
                channels_info.append(
                    {
                        "center": channel_center,
                        "direction": channel_dir,
                        "type": channel_type,
                        "size": channel_size,
                        "thickness": ball_radius * 2,
                    }
                )

            all_obstacles = torch.cat(all_obstacles, dim=0)

            sample = {
                "obstacles": all_obstacles,
                "channels_info": channels_info,  # 保存所有通道信息以便调试
            }

            samples.append(sample)

            # 更新进度条描述
            pbar.set_description(f"Generate scene samples (Channels: {num_channels})")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 获取通道信息
        channels_info = sample["channels_info"]

        # 提取通道特征
        channel_centers = (
            torch.stack([info["center"] for info in channels_info]) if channels_info else torch.zeros((1, 3))
        )
        channel_directions = (
            torch.stack([info["direction"] for info in channels_info]) if channels_info else torch.zeros((1, 3))
        )

        # 处理尺寸信息 - 确保所有尺寸都是二维张量
        channel_sizes = (
            torch.stack(
                [
                    info["size"] if isinstance(info["size"], torch.Tensor) else torch.tensor(info["size"])
                    for info in channels_info
                ]
            )
            if channels_info
            else torch.zeros((1, 2))
        )

        # 扩展返回的样本
        enhanced_sample = {
            "obstacles": sample["obstacles"],
            "channels_info": channels_info,
            "channel_centers": channel_centers,  # 中心点位置
            "channel_directions": channel_directions,  # 法向量
            "channel_sizes": channel_sizes,  # 尺寸
            "num_channels": len(channels_info),  # 通道数量
        }

        return enhanced_sample
