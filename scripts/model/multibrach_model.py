import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model.pointnet.pointnet import PointNet


class PositionalEncoding(nn.Module):
    """Transformer中的位置编码"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # 注册为非可训练参数
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1)]
        return x


class TaskBranch(nn.Module):
    """任务配置特征提取分支"""

    def __init__(self, input_dim=15, hidden_dims=[128, 256], output_dim=256):
        """
        Args:
            input_dim: 输入维度 (起始关节角[6] + 目标关节角[6] + 抓握位置[3])
            hidden_dims: 隐藏层维度列表
            output_dim: 输出特征维度
        """
        super(TaskBranch, self).__init__()

        self.layers = nn.ModuleList()
        prev_dim = input_dim

        # 构建隐藏层
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        # 输出层
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, start_joints, target_joints, grasp_offset):
        """
        Args:
            start_joints: 起始关节角 [batch_size, 6]
            target_joints: 目标关节角 [batch_size, 6]
            grasp_offset: 抓握位置偏移 [batch_size, 3]
        """
        # 连接所有输入
        x = torch.cat([start_joints, target_joints, grasp_offset], dim=1)

        # 前向传播
        for layer in self.layers:
            x = F.relu(layer(x))

        # 输出层
        x = self.output_layer(x)
        return x


class TrajectoryBranch(nn.Module):
    """轨迹分支，使用Transformer编码器处理轨迹序列"""

    def __init__(self, input_dim=6, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, output_dim=512):
        """
        Args:
            input_dim: 轨迹点维度 (6个关节角)
            d_model: Transformer模型维度
            nhead: 多头注意力的头数
            num_layers: Transformer编码器层数
            dim_feedforward: 前馈网络维度
            output_dim: 输出特征维度
        """
        super(TrajectoryBranch, self).__init__()

        # 输入映射层
        self.input_map = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层，聚合轨迹特征
        self.output_map = nn.Linear(d_model, output_dim)

    def forward(self, trajectory):
        """
        Args:
            trajectory: 轨迹序列 [batch_size, seq_len, 6]
        """
        # 输入映射
        x = self.input_map(trajectory)

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer编码
        x = self.transformer_encoder(x)

        # 聚合轨迹特征 (取序列的平均值)
        x = torch.mean(x, dim=1)

        # 输出映射
        x = self.output_map(x)
        return x


class TrajectoryDecoder(nn.Module):
    """轨迹解码器，使用Transformer解码器生成轨迹"""

    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=1024, output_dim=6, max_seq_len=256):
        """
        Args:
            d_model: Transformer模型维度
            nhead: 多头注意力的头数
            num_layers: Transformer解码器层数
            dim_feedforward: 前馈网络维度
            output_dim: 输出维度 (关节角数量)
            max_seq_len: 最大序列长度
        """
        super(TrajectoryDecoder, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出映射
        self.output_map = nn.Linear(d_model, output_dim)

        # 解码器输入 (可学习的初始序列)
        self.query_embed = nn.Embedding(max_seq_len, d_model)

    def forward(self, memory):
        """
        Args:
            memory: 融合的特征向量 [batch_size, d_model]
        """
        batch_size = memory.size(0)

        # 扩展记忆，将特征复制到序列长度
        memory = memory.unsqueeze(1).repeat(1, self.max_seq_len, 1)

        # 获取查询嵌入并扩展batch维度
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # 应用位置编码
        query_embed = self.pos_encoder(query_embed)

        # Transformer解码
        output = self.transformer_decoder(query_embed, memory)

        # 输出映射
        trajectory = self.output_map(output)

        return trajectory


class MultiPathTrajectoryNetwork(nn.Module):
    """多分支轨迹生成网络"""

    def __init__(
        self,
        object_feature_dim=256,
        env_feature_dim=512,
        task_feature_dim=256,
        traj_feature_dim=512,
        fusion_dim=512,
        output_seq_len=1024,
        joint_dim=6,
    ):
        """
        Args:
            object_feature_dim: 物体特征维度
            env_feature_dim: 环境特征维度
            task_feature_dim: 任务特征维度
            traj_feature_dim: 轨迹特征维度
            fusion_dim: 融合特征维度
            output_seq_len: 输出轨迹长度
            joint_dim: 关节维度
        """
        super(MultiPathTrajectoryNetwork, self).__init__()

        # 物体分支 (PointNet)
        self.object_branch = PointNet(object_feature_dim, normal_channel=True)

        # 环境分支 (PointNet)
        self.env_branch = PointNet(env_feature_dim, normal_channel=True)

        # 任务分支
        self.task_branch = TaskBranch(input_dim=15, output_dim=task_feature_dim)

        # 轨迹分支
        self.traj_branch = TrajectoryBranch(input_dim=joint_dim, output_dim=traj_feature_dim)

        # 特征融合层
        total_feature_dim = object_feature_dim + env_feature_dim + task_feature_dim + traj_feature_dim
        self.fusion_layer = nn.Sequential(nn.Linear(total_feature_dim, 1024), nn.ReLU(), nn.Linear(1024, fusion_dim))

        # 轨迹解码器
        self.decoder = TrajectoryDecoder(d_model=fusion_dim, output_dim=joint_dim, max_seq_len=output_seq_len)

    def forward(
        self, grasped_point_cloud, env_point_cloud, start_joints, target_joints, grasp_offset, input_trajectory=None
    ):
        """
        Args:
            grasped_point_cloud: 抓握物体点云 [batch_size, num_points, 3+3]
            env_point_cloud: 环境点云 [batch_size, num_points, 3+3]
            start_joints: 起始关节角 [batch_size, 6]
            target_joints: 目标关节角 [batch_size, 6]
            grasp_offset: 抓握位置偏移 [batch_size, 3]
            input_trajectory: 输入轨迹 [batch_size, seq_len, 6]
        """
        # 处理抓握物体点云
        grasped_point_cloud = grasped_point_cloud.permute(0, 2, 1)  # [B, 6, N]
        _, object_features = self.object_branch(grasped_point_cloud)
        object_features = object_features.view(object_features.size(0), -1)  # Flatten

        # 处理环境点云
        env_point_cloud = env_point_cloud.permute(0, 2, 1)  # [B, 6, N]
        _, env_features = self.env_branch(env_point_cloud)
        env_features = env_features.view(env_features.size(0), -1)  # Flatten

        # 处理任务信息
        task_features = self.task_branch(start_joints, target_joints, grasp_offset)

        # 处理输入轨迹 (如果提供)
        if input_trajectory is not None:
            traj_features = self.traj_branch(input_trajectory)
        else:
            # 如果没有提供输入轨迹，使用零向量
            batch_size = grasped_point_cloud.size(0)
            traj_features = torch.zeros(batch_size, 512, device=grasped_point_cloud.device)

        # 融合特征
        combined_features = torch.cat([object_features, env_features, task_features, traj_features], dim=1)
        fused_features = self.fusion_layer(combined_features)

        # 解码轨迹
        trajectory = self.decoder(fused_features)

        return trajectory


# 损失函数
class FullTrajectoryLoss(nn.Module):
    """
    完整轨迹生成的损失函数，使用DTW轨迹趋势损失衡量轨迹相似度
    """

    def __init__(self):
        super(FullTrajectoryLoss, self).__init__()

    def forward(self, pred_trajectory, full_trajectory):
        """
        Args:
            pred_trajectory: 预测轨迹 [batch_size, seq_len, joint_dim]
            full_trajectory: 完整目标轨迹 [batch_size, full_seq_len, joint_dim]
        """
        # 初始化DTW损失
        dtw_loss = torch.tensor(0.0, device=pred_trajectory.device)

        # 插值预测轨迹到完整轨迹长度
        interpolated_pred = self._interpolate_trajectory(pred_trajectory, full_trajectory.size(1))

        # 计算DTW损失
        loss = self._compute_dtw_batch(interpolated_pred, full_trajectory)

        return loss

    def _interpolate_trajectory(self, trajectory, target_length):
        """
        将轨迹插值到目标长度

        Args:
            trajectory: 原始轨迹 [batch_size, seq_len, joint_dim]
            target_length: 目标长度

        Returns:
            插值后的轨迹 [batch_size, target_length, joint_dim]
        """
        batch_size, seq_len, joint_dim = trajectory.shape
        device = trajectory.device

        # 创建源索引和目标索引
        orig_idx = torch.linspace(0, 1, seq_len, device=device)
        target_idx = torch.linspace(0, 1, target_length, device=device)

        interpolated = []

        # 对每个batch单独处理
        for b in range(batch_size):
            joints_interp = []

            # 对每个关节角度单独进行插值
            for j in range(joint_dim):
                # 提取当前batch和关节的值
                joint_vals = trajectory[b, :, j]

                # 使用线性插值
                interp_vals = (
                    torch.nn.functional.interpolate(
                        joint_vals.unsqueeze(0).unsqueeze(0), size=target_length, mode="linear", align_corners=True
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

                joints_interp.append(interp_vals)

            # 合并所有关节的插值结果
            joints_interp = torch.stack(joints_interp, dim=1)
            interpolated.append(joints_interp)

        # 合并所有batch的结果
        return torch.stack(interpolated, dim=0)

    def _compute_dtw_distance(self, seq1, seq2):
        """
        计算两个序列之间的DTW距离

        Args:
            seq1: 第一个序列 [seq_len1, joint_dim]
            seq2: 第二个序列 [seq_len2, joint_dim]

        Returns:
            DTW距离
        """
        n, m = seq1.size(0), seq2.size(0)
        device = seq1.device

        # 计算成本矩阵 (欧氏距离的平方)
        cost_matrix = torch.zeros((n, m), device=device)
        for i in range(n):
            for j in range(m):
                cost_matrix[i, j] = torch.sum((seq1[i] - seq2[j]) ** 2)

        # 计算累积距离矩阵
        dtw_matrix = torch.zeros((n, m), device=device)
        dtw_matrix[0, 0] = cost_matrix[0, 0]

        for i in range(1, n):
            dtw_matrix[i, 0] = dtw_matrix[i - 1, 0] + cost_matrix[i, 0]

        for j in range(1, m):
            dtw_matrix[0, j] = dtw_matrix[0, j - 1] + cost_matrix[0, j]

        for i in range(1, n):
            for j in range(1, m):
                dtw_matrix[i, j] = cost_matrix[i, j] + torch.min(
                    torch.stack([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
                )

        # 返回标准化的DTW距离
        return dtw_matrix[-1, -1] / (n + m)

    def _compute_dtw_batch(self, batch_seq1, batch_seq2):
        """
        批量计算DTW距离

        Args:
            batch_seq1: 第一批序列 [batch_size, seq_len1, joint_dim]
            batch_seq2: 第二批序列 [batch_size, seq_len2, joint_dim]

        Returns:
            批量DTW距离的平均值
        """
        batch_size = batch_seq1.size(0)
        dtw_distances = []

        for b in range(batch_size):
            dtw_dist = self._compute_dtw_distance(batch_seq1[b], batch_seq2[b])
            dtw_distances.append(dtw_dist)

        return torch.mean(torch.stack(dtw_distances))
