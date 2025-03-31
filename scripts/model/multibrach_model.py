import math

import numpy as np
import pysdtw
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """轨迹分支，使用Transformer或LSTM编码器处理轨迹序列"""

    def __init__(
        self,
        input_dim=6,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        output_dim=512,
        use_lstm=False,
        dropout=0.4,
    ):
        """
        Args:
            input_dim: 轨迹点维度 (6个关节角)
            d_model: 模型维度
            nhead: 多头注意力的头数 (仅用于Transformer)
            num_layers: 编码器层数
            dim_feedforward: 前馈网络维度 (仅用于Transformer)
            output_dim: 输出特征维度
            use_lstm: 是否使用LSTM替代Transformer
            dropout: Dropout比例
        """
        super(TrajectoryBranch, self).__init__()
        self.output_dim = output_dim
        self.use_lstm = use_lstm

        # 输入映射层
        self.input_map = nn.Linear(input_dim, d_model)

        if use_lstm:
            # LSTM编码器，添加dropout参数
            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            # 添加输出dropout
            self.dropout = nn.Dropout(dropout)
            # 由于双向LSTM，最终维度是d_model*2
            self.output_map = nn.Linear(d_model * 2, output_dim)
        else:
            # 位置编码
            self.pos_encoder = PositionalEncoding(d_model)

            # Transformer编码器层，添加dropout参数
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                dropout=dropout,  # 添加dropout
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # 添加输出dropout
            self.dropout = nn.Dropout(dropout)
            # 输出层
            self.output_map = nn.Linear(d_model, output_dim)

    def forward(self, trajectory):
        """
        Args:
            trajectory: 轨迹序列 [batch_size, seq_len, 6]
        """
        # 输入映射
        x = self.input_map(trajectory)

        if self.use_lstm:
            # LSTM编码
            output, (hidden, _) = self.lstm(x)

            # 使用最后时刻的隐藏状态
            # 对于双向LSTM，获取两个方向的最后状态并拼接
            # hidden形状: [num_layers * num_directions, batch_size, hidden_size]
            last_hidden = output[:, -1, :]  # 使用最后时刻的输出，形状: [batch_size, hidden_size*2]

            # 在输出层之前应用dropout
            last_hidden = self.dropout(last_hidden)

            # 输出映射
            x = self.output_map(last_hidden)
        else:
            # 位置编码
            x = self.pos_encoder(x)

            # Transformer编码
            x = self.transformer_encoder(x)

            # 聚合特征前应用dropout
            x = self.dropout(x)

            # 聚合轨迹特征 (取序列的平均值)
            x = torch.mean(x, dim=1)

            # 输出映射
            x = self.output_map(x)

        return x


class TrajectoryDecoder(nn.Module):
    """轨迹解码器，使用Transformer解码器生成轨迹"""

    def __init__(
        self, d_model=512, nhead=8, num_layers=6, dim_feedforward=1024, output_dim=6, max_seq_len=256, dropout=0.4
    ):
        """
        Args:
            d_model: Transformer模型维度
            nhead: 多头注意力的头数
            num_layers: Transformer解码器层数
            dim_feedforward: 前馈网络维度
            output_dim: 输出维度 (关节角数量)
            max_seq_len: 最大序列长度
            dropout: Dropout比例
        """
        super(TrajectoryDecoder, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer解码器层，添加dropout参数
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,  # 添加dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出映射
        self.output_map = nn.Linear(d_model, output_dim)

        # 解码器输入 (可学习的初始序列)
        self.query_embed = nn.Embedding(max_seq_len, d_model)

        # 添加输出dropout
        self.dropout = nn.Dropout(dropout)

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

        # 在输出层之前应用dropout
        output = self.dropout(output)

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
        normal_channel=True,
        use_lstm=False,
        dropout=0.5,  # 添加dropout参数
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
            normal_channel: 是否使用法向量
            use_lstm: 是否使用LSTM替代Transformer
            dropout: Dropout比例
        """
        super(MultiPathTrajectoryNetwork, self).__init__()

        # 物体分支 (PointNet)
        self.object_branch = PointNet(1, global_feature_dim=object_feature_dim, normal_channel=normal_channel)

        # 环境分支 (PointNet)
        self.env_branch = PointNet(1, global_feature_dim=env_feature_dim, normal_channel=normal_channel)

        # 任务分支
        self.task_branch = TaskBranch(input_dim=15, output_dim=task_feature_dim)

        # 轨迹分支
        self.traj_branch = TrajectoryBranch(
            input_dim=joint_dim,
            d_model=traj_feature_dim,
            nhead=4,
            output_dim=traj_feature_dim,
            use_lstm=use_lstm,
            dropout=dropout,  # 传递dropout参数
        )

        # 特征融合层
        total_feature_dim = object_feature_dim + env_feature_dim + task_feature_dim + traj_feature_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),  # 添加dropout
            nn.Linear(1024, fusion_dim),
        )

        # 轨迹解码器
        self.decoder = TrajectoryDecoder(
            d_model=fusion_dim,
            nhead=4,
            output_dim=joint_dim,
            max_seq_len=output_seq_len,
            dropout=dropout,  # 传递dropout参数
        )

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
        batch_size = grasped_point_cloud.size(0)
        # 处理抓握物体点云
        grasped_point_cloud = grasped_point_cloud.permute(0, 2, 1)  # [B, 6, N]
        _, object_features = self.object_branch(grasped_point_cloud)
        object_features = object_features.reshape(batch_size, -1)

        # 处理环境点云
        env_point_cloud = env_point_cloud.permute(0, 2, 1)  # [B, 6, N]
        _, env_features = self.env_branch(env_point_cloud)
        env_features = env_features.reshape(batch_size, -1)

        # 处理任务信息
        task_features = self.task_branch(start_joints, target_joints, grasp_offset)

        # 处理输入轨迹
        if input_trajectory is not None:
            traj_features = self.traj_branch(input_trajectory)
        else:
            # 如果没有提供输入轨迹，使用零向量
            batch_size = grasped_point_cloud.size(0)
            traj_features = torch.zeros(batch_size, self.traj_branch.output_dim, device=grasped_point_cloud.device)

        # 融合特征 - 确保所有特征都是[B, D]形状
        combined_features = torch.cat([object_features, env_features, task_features, traj_features], dim=1)
        fused_features = self.fusion_layer(combined_features)

        # 解码轨迹
        trajectory = self.decoder(fused_features)
        
        return trajectory


# 损失函数
class HybridTrajectoryLoss(nn.Module):
    """
    混合轨迹损失函数，结合SoftDTW与其他损失函数
    """
    def __init__(self, gamma=1.0, use_cuda=True, alpha_dtw=1.0, alpha_l1=0.5, alpha_l2=0.1):
        """
        初始化混合损失函数
        
        Args:
            gamma: SoftDTW平滑参数
            use_cuda: 是否使用CUDA
            alpha_dtw: SoftDTW损失权重
            alpha_l1: L1损失权重
            alpha_l2: L2损失权重
        """
        super(HybridTrajectoryLoss, self).__init__()
        self.fun = pysdtw.distance.pairwise_l2_squared
        self.sdtw = pysdtw.SoftDTW(gamma=gamma, dist_func=self.fun, use_cuda=use_cuda)
        self.alpha_dtw = alpha_dtw
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
        
    def forward(self, pred_trajectory, full_trajectory):
        """
        计算混合损失
        
        Args:
            pred_trajectory: 预测轨迹 [batch_size, seq_len, joint_dim]
            full_trajectory: 目标轨迹 [batch_size, seq_len, joint_dim]
            
        Returns:
            torch.Tensor: 标量损失值
        """
        # DTW损失
        batch_dtw_losses = self.sdtw(pred_trajectory, full_trajectory)
        dtw_loss = batch_dtw_losses.mean() / max(pred_trajectory.shape[1], full_trajectory.shape[1])
        
        # L1损失 - 绝对位置差异
        l1_loss = F.l1_loss(pred_trajectory, full_trajectory)
        
        # L2损失 - 平方位置差异
        l2_loss = F.mse_loss(pred_trajectory, full_trajectory)
        
        # 键控点损失 - 直接计算关键点损失而不使用cat操作
        start_point_loss = F.mse_loss(pred_trajectory[:, 0, :], full_trajectory[:, 0, :])
        mid_idx = pred_trajectory.shape[1] // 2
        mid_point_loss = F.mse_loss(
            pred_trajectory[:, mid_idx, :], 
            full_trajectory[:, mid_idx, :]
        )
        end_point_loss = F.mse_loss(pred_trajectory[:, -1, :], full_trajectory[:, -1, :])
        
        keypoints_loss = start_point_loss + mid_point_loss + end_point_loss
        
        # 组合损失
        total_loss = (self.alpha_dtw * dtw_loss + 
                      self.alpha_l1 * l1_loss + 
                      self.alpha_l2 * l2_loss + 
                      0.5 * keypoints_loss)
        
        return total_loss
