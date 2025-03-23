import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple
import pytorch3d.ops as ops
import os
import logging
from datetime import datetime


class SetAbstraction(nn.Module):
    """PointNet++ 的集合抽象层"""

    def __init__(self, npoint: int, radius: float, nsample: int, in_channel: int, mlp: List[int]):
        super(SetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        self.mlp = nn.ModuleList()
        last_channel = in_channel + 3  # +3 用于相对坐标

        for out_channel in mlp:
            self.mlp.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp.append(nn.BatchNorm2d(out_channel))
            self.mlp.append(nn.ReLU())
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入:
            xyz: (B, N, 3) 输入点云坐标
            points: (B, N, C) 输入点云特征
        返回:
            new_xyz: (B, npoint, 3) 采样点坐标
            new_points: (B, npoint, \sum_k mlp[k][-1]) 特征
        """
        xyz = xyz.contiguous()
        # FPS采样
        new_xyz, fps_idx = ops.sample_farthest_points(xyz, K=self.npoint)

        # 球查询
        res = ops.ball_query(new_xyz, xyz, K=self.nsample, radius=self.radius, return_nn=True)
        idx, grouped_xyz = res.idx, res.knn

        # 处理-1索引
        B, npoint, K = idx.shape
        mask = idx == -1  # [B, npoint, K]
        idx[mask] = 0  # 将-1替换为0

        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)  # 相对坐标

        if points is not None:
            grouped_points = ops.knn_gather(points, idx)
            # 将mask应用到特征上
            if mask.any():
                grouped_points[mask] = 0
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz
            if mask.any():
                grouped_points[mask] = 0

        grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, C+3, nsample, npoint]

        # MLP处理
        for layer in self.mlp:
            grouped_points = layer(grouped_points)

        # 最大池化
        new_points = torch.max(grouped_points, dim=2)[0]  # [B, C, npoint]
        new_points = new_points.permute(0, 2, 1)  # [B, npoint, C]

        return new_xyz, new_points


class ChannelCounter(nn.Module):
    """带有通道特征增强的通道数量识别模型"""

    def __init__(self, max_channels: int = 20):
        super(ChannelCounter, self).__init__()

        # 点云处理部分 - 降低维度
        self.sa1 = SetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=1, mlp=[32, 64, 128])
        self.sa2 = SetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp=[128, 256, 512])
        self.sa3 = SetAbstraction(npoint=64, radius=0.8, nsample=128, in_channel=512, mlp=[256, 512, 1024])

        # 通道特征处理部分
        self.center_encoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128))
        self.direction_encoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128))
        self.size_encoder = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 64))

        # 通道特征注意力机制
        self.center_attention = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.direction_attention = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.size_attention = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))

        # 特征融合 - 调整维度以匹配新的点云特征维度
        self.fusion = nn.Sequential(
            nn.Linear(1024 + 128 + 128 + 64, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1)
        )

        # 分类器
        self.fc1 = nn.Linear(512, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, max_channels + 1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, points, channel_centers=None, channel_directions=None, channel_sizes=None):
        """
        输入:
            points: (B, N, 4) 点云数据，包含xyz坐标和球体半径
            channel_centers: (B, C, 3) 通道中心点位置
            channel_directions: (B, C, 3) 通道法向量
            channel_sizes: (B, C, 2) 通道尺寸
        """
        batch_size = points.shape[0]

        # 处理点云数据
        xyz = points[:, :, :3]
        features = points[:, :, 3:]

        scene_centers = torch.mean(points[:, :, :3], dim=1, keepdim=True)

        # 对点云和通道中心进行归一化
        normalized_xyz = xyz - scene_centers

        # xyz, features = self.sa1(xyz, features)
        # xyz, features = self.sa2(xyz, features)
        # xyz, features = self.sa3(xyz, features)

        # 使用归一化后的坐标进行特征提取
        normalized_xyz, features = self.sa1(normalized_xyz, features)
        normalized_xyz, features = self.sa2(normalized_xyz, features)
        normalized_xyz, features = self.sa3(normalized_xyz, features)

        # 全局点云特征
        point_features = features.mean(dim=1)  # [B, 1024]

        # 如果提供了通道特征，则进行处理和融合
        if channel_centers is not None and channel_directions is not None and channel_sizes is not None:
            normalized_centers = channel_centers - scene_centers

            # 编码所有通道的特征
            # centers_encoded = self.center_encoder(channel_centers.view(-1, 3)).view(batch_size, -1, 128)
            centers_encoded = self.center_encoder(normalized_centers.view(-1, 3)).view(batch_size, -1, 128)
            directions_encoded = self.direction_encoder(channel_directions.view(-1, 3)).view(batch_size, -1, 128)
            sizes_encoded = self.size_encoder(channel_sizes.view(-1, 2)).view(batch_size, -1, 64)

            # 计算注意力权重
            center_weights = self.center_attention(centers_encoded)  # [B, C, 1]
            center_weights = F.softmax(center_weights, dim=1)

            direction_weights = self.direction_attention(directions_encoded)  # [B, C, 1]
            direction_weights = F.softmax(direction_weights, dim=1)

            size_weights = self.size_attention(sizes_encoded)  # [B, C, 1]
            size_weights = F.softmax(size_weights, dim=1)

            # 加权求和替代简单平均
            centers_feature = torch.sum(centers_encoded * center_weights, dim=1)  # [B, 128]
            directions_feature = torch.sum(directions_encoded * direction_weights, dim=1)  # [B, 128]
            sizes_feature = torch.sum(sizes_encoded * size_weights, dim=1)  # [B, 64]

            # 融合所有特征
            combined_features = torch.cat([point_features, centers_feature, directions_feature, sizes_feature], dim=1)
            x = self.fusion(combined_features)
        else:
            # 如果没有通道特征，只使用点云特征
            x = F.relu(self.ln1(self.fc1(point_features)))

        # 最终分类层
        x = self.dropout(x)
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.fc2(x)

        return x


def evaluate_model(model, dataloader, criterion, device, batch_size, logger=None):
    """在测试集上评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    current_batch = {"obstacles": [], "targets": []}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            obstacles = batch["obstacles"][0].to(device)
            num_channels = len(batch["channels_info"])
            target = num_channels - 1

            current_batch["obstacles"].append(obstacles)
            current_batch["targets"].append(target)

            if len(current_batch["obstacles"]) == batch_size or batch_idx == len(dataloader) - 1:
                max_points = max(obs.shape[0] for obs in current_batch["obstacles"])
                padded_obstacles = []

                for obs in current_batch["obstacles"]:
                    num_points = obs.shape[0]
                    if num_points < max_points:
                        padding = torch.zeros((max_points - num_points, 4), device=device)
                        padded_obs = torch.cat([obs, padding], dim=0)
                    else:
                        padded_obs = obs
                    padded_obstacles.append(padded_obs)

                batch_obstacles = torch.stack(padded_obstacles).to(device)
                batch_targets = torch.tensor(current_batch["targets"], device=device)

                output = model(batch_obstacles)
                loss = criterion(output, batch_targets)

                total_loss += loss.item()

                pred = output.argmax(dim=1)
                correct += (pred == batch_targets).sum().item()
                total += len(batch_targets)

                current_batch = {"obstacles": [], "targets": []}

    avg_loss = total_loss * batch_size / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy
