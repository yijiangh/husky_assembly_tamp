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
    """基于PointNet++的通道数量识别模型"""

    def __init__(self, max_channels: int = 20):
        super(ChannelCounter, self).__init__()

        self.sa1 = SetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=1, mlp=[64, 128, 256])  # 增加通道数

        self.sa2 = SetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=256, mlp=[256, 512, 1024])

        self.sa3 = SetAbstraction(npoint=64, radius=0.8, nsample=128, in_channel=1024, mlp=[512, 1024, 2048])

        self.fc1 = nn.Linear(2048, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, max_channels + 1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, points):
        """
        输入:
            points: (B, N, 4) 点云数据，包含xyz坐标和球体半径
        """
        xyz = points[:, :, :3]
        features = points[:, :, 3:]

        xyz, features = self.sa1(xyz, features)
        xyz, features = self.sa2(xyz, features)
        xyz, features = self.sa3(xyz, features)

        x = features.mean(dim=1)  # 全局特征

        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def evaluate_model(model, dataloader, criterion, device, batch_size):
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
