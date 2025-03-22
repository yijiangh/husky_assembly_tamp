import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
import os
import logging

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

        for _ in range(num_samples):
            # 决定场景中通道的数量
            num_channels = np.random.randint(1, 20)
            all_obstacles = []

            # 生成多个通道
            channels_info = []
            for channel_idx in range(num_channels):
                # 随机决定通道类型: 圆形、椭圆形、矩形或不规则
                channel_type = np.random.choice(["circle", "ellipse", "rectangle", "irregular"])

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
                channel_size = torch.rand(1) * 1.0 + 1.0  # 通道大小 [1.0, 2.0]
                # 根据通道类型设置尺寸
                if channel_type == "circle":
                    channel_size = torch.rand(1) * 1.0 + 1.0  # 圆形通道半径 [1.0, 2.0]
                elif channel_type == "rectangle":
                    channel_size = torch.rand(2) * 1.0 + 1.0  # 矩形通道长宽 [1.0, 2.0]
                elif channel_type == "ellipse":
                    channel_size = torch.stack(
                        [torch.rand(1) * 1.0 + 1.0, torch.rand(1) * 0.5 + 0.5]  # 长轴 [1.0, 2.0]  # 短轴 [0.5, 1.0]
                    )
                else:  # irregular
                    channel_size = torch.rand(1) * 1.0 + 1.0  # 不规则通道基准尺寸 [1.0, 2.0]

                # 通道轮廓点数量
                num_boundary_points = np.random.randint(20, 40)

                # 通道小球半径
                ball_radius = torch.rand(1) * 0.15 + 0.1  # [0.1, 0.25]

                # 存储通道边界小球
                boundary_obstacles = []

                # 根据通道类型生成边界点
                if channel_type == "circle":
                    # 圆形通道
                    for i in range(num_boundary_points):
                        angle = (i / num_boundary_points) * 2 * np.pi
                        # 圆周上的点
                        local_pos = channel_size * (
                            torch.cos(torch.tensor(angle)) * x_axis + torch.sin(torch.tensor(angle)) * y_axis
                        )

                        # 添加一些随机扰动使通道不完全平面
                        if np.random.random() > 0.5:  # 50%的几率添加Z轴扰动
                            z_distortion = (torch.rand(1) - 0.5) * 0.5  # [-0.25, 0.25]
                            local_pos = local_pos + z_distortion * z_axis

                        # 全局坐标
                        pos = channel_center + local_pos

                        # 添加小球
                        obstacle = torch.cat([pos, ball_radius * torch.ones(1)])
                        boundary_obstacles.append(obstacle)

                elif channel_type == "ellipse":
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

                elif channel_type == "rectangle":
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
                                z_distortion = (torch.rand(1) - 0.5) * 0.6  # [-0.3, 0.3]
                                local_pos = local_pos + z_distortion * z_axis

                                # 微小的x-y扰动
                                xy_distortion = (torch.rand(2) - 0.5) * 0.2  # [-0.1, 0.1]
                                local_pos = local_pos + xy_distortion[0] * x_axis + xy_distortion[1] * y_axis

                            # 全局坐标
                            pos = channel_center + local_pos

                            # 添加小球
                            obstacle = torch.cat([pos, ball_radius * torch.ones(1)])
                            boundary_obstacles.append(obstacle)

                else:  # irregular
                    # 不规则形状通道 - 使用参数方程生成
                    for i in range(num_boundary_points):
                        angle = (i / num_boundary_points) * 2 * np.pi

                        # 使用三角函数生成不规则形状
                        r = channel_size * (
                            0.7 + 0.3 * torch.sin(torch.tensor(angle * 3)) + 0.2 * torch.cos(torch.tensor(angle * 5))
                        )

                        local_pos = (
                            r * torch.cos(torch.tensor(angle)) * x_axis + r * torch.sin(torch.tensor(angle)) * y_axis
                        )

                        # 添加Z轴波浪效果
                        z_wave = 0.4 * torch.sin(torch.tensor(angle * 2)) * z_axis
                        local_pos = local_pos + z_wave

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
                        "size": (
                            channel_size.tolist() if isinstance(channel_size, torch.Tensor) else channel_size.item()
                        ),
                        "thickness": ball_radius.item() * 2,
                    }
                )

            # 添加一些随机背景障碍物
            n_background = np.random.randint(10, 30)
            background_obstacles = []

            for _ in range(n_background):
                pos = torch.randn(3) * 3.0  # 随机位置
                radius = torch.rand(1) * 0.3 + 0.1  # 随机半径 [0.1, 0.4]
                background_obstacles.append(torch.cat([pos, radius]))

            background_obstacles = torch.stack(background_obstacles)

            # 合并所有障碍物
            all_obstacles.append(background_obstacles)
            all_obstacles = torch.cat(all_obstacles, dim=0)

            # 随机选择一个通道作为主通道
            main_channel_idx = np.random.randint(0, num_channels)
            main_channel = channels_info[main_channel_idx]

            # 自由空间采样点和标签
            free_space_samples = torch.randn(100, 3) * 3.0
            free_space_labels = torch.ones(100, dtype=torch.bool)

            # 检查点是否在任何障碍物内
            for i in range(100):
                point = free_space_samples[i]
                for j in range(all_obstacles.size(0)):
                    obs_pos = all_obstacles[j, :3]
                    obs_radius = all_obstacles[j, 3]
                    if torch.norm(point - obs_pos) < obs_radius:
                        free_space_labels[i] = False
                        break

            sample = {
                "obstacles": all_obstacles,
                "channel_pos": main_channel["center"],
                "channel_dir": main_channel["direction"],
                "free_space_samples": free_space_samples,
                "free_space_labels": free_space_labels,
                "channels_info": channels_info,  # 保存所有通道信息以便调试
            }

            samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class BallsToTensor:
    """
    将不同长度的小球集合转换为固定长度的张量
    """

    def __init__(self, max_balls=200, padding_value=0.0):
        self.max_balls = max_balls
        self.padding_value = padding_value

    def __call__(self, batch):
        batch_size = len(batch)

        # 创建固定大小的张量进行填充
        all_obstacles = torch.full((batch_size, self.max_balls, 4), self.padding_value)
        channel_pos = torch.stack([item["channel_pos"] for item in batch])
        channel_dir = torch.stack([item["channel_dir"] for item in batch])

        # 填充障碍物数据
        for i, item in enumerate(batch):
            obstacles = item["obstacles"]
            n = min(obstacles.shape[0], self.max_balls)
            all_obstacles[i, :n, :] = obstacles[:n, :]

        return {"obstacles": all_obstacles, "channel_pos": channel_pos, "channel_dir": channel_dir}


# **************************************************************************
# PointNet++相关
# **************************************************************************


def square_distance(src, dst):
    """
    计算两组点之间的平方欧氏距离
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    根据索引从点集中获取特定点
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    最远点采样算法
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    球查询: 找到每个中心点半径范围内的所有点
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    sqrdists = square_distance(new_xyz, xyz)
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, S, 1)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


class SetAbstraction(nn.Module):
    """
    PointNet++的集合抽象层
    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super(SetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Params:
            xyz: 点坐标 [B, N, 3]
            points: 点特征 [B, N, D]

        Returns:
            new_xyz: 采样后的点坐标 [B, npoint, 3]
            new_points: 新点特征 [B, npoint, mlp[-1]]
        """
        B, N, C = xyz.shape

        # 采样npoint个中心点
        if self.npoint is not None:
            idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, idx)
        else:
            new_xyz = xyz

        # 球查询和分组
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, 3]
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, -1, 1, C)

        if points is not None:
            grouped_points = index_points(points, idx)  # [B, npoint, nsample, D]
            grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz_norm

        # 通过MLP处理
        grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D+3, nsample, npoint]

        for i, conv in enumerate(self.mlp_convs):
            grouped_points = F.relu(self.mlp_bns[i](conv(grouped_points)))

        # 最大池化
        new_points = torch.max(grouped_points, 2)[0]  # [B, D', npoint]
        new_points = new_points.permute(0, 2, 1)

        return new_xyz, new_points


class ObstaclePointNet(nn.Module):
    """
    处理障碍物小球的PointNet++模型
    """

    def __init__(self, normal_channel=False):
        super(ObstaclePointNet, self).__init__()

        # 初始特征维度：坐标(3)+半径(1)=4
        in_channel = 4

        # 多尺度的集合抽象层
        # 第一层：捕获局部邻域关系
        self.sa1 = SetAbstraction(
            npoint=128,  # 采样128个点
            radius=0.3,  # 小搜索半径，捕获局部关系
            nsample=32,  # 每组最多包含32个点
            in_channel=in_channel,
            mlp=[64, 64, 128],  # MLP通道数
        )

        # 第二层：捕获中等尺度的关系
        self.sa2 = SetAbstraction(
            npoint=64,  # 继续采样64个点
            radius=0.6,  # 中等搜索半径
            nsample=64,  # 每组最多包含64个点
            in_channel=128 + 3,  # 前一层特征(128) + 坐标(3)
            mlp=[128, 128, 256],
        )

        # 第三层：全局特征提取
        self.sa3 = SetAbstraction(
            npoint=None,  # 全局特征
            radius=None,  # 全局范围
            nsample=None,  # 全部点
            in_channel=256 + 3,  # 前一层特征(256) + 坐标(3)
            mlp=[256, 512, 1024],
        )

        # 特征解码器
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, 128)

    def forward(self, obstacles):
        """
        Params:
            obstacles: [B, N, 4] - 每个点包含 [x, y, z, r]

        Returns:
            x: [B, 128] - 场景特征向量
        """
        B, N, _ = obstacles.shape

        # 提取坐标和半径
        xyz = obstacles[:, :, 0:3].contiguous()
        features = obstacles.clone()  # 使用完整特征 [x, y, z, r]

        # 层次化特征提取
        xyz1, features1 = self.sa1(xyz, features)
        xyz2, features2 = self.sa2(xyz1, features1)
        _, features3 = self.sa3(xyz2, features2)

        # 全局特征向量
        x = features3.view(B, 1024)

        # 特征解码
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x


class ChannelPredictor(nn.Module):
    """
    预测场景中的最佳通道位置和方向
    """

    def __init__(self):
        super(ChannelPredictor, self).__init__()
        self.obstacle_encoder = ObstaclePointNet()

        # 预测最可能的通道入口点和方向
        self.pos_predictor = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))  # 3D坐标

        self.dir_predictor = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))  # 3D方向向量

    def forward(self, obstacles):
        # 提取场景特征
        features = self.obstacle_encoder(obstacles)

        # 预测通道位置和方向
        position = self.pos_predictor(features)
        direction = self.dir_predictor(features)
        direction = F.normalize(direction, dim=1)  # 归一化方向向量

        return position, direction


def channel_prediction_loss(pred_pos, pred_dir, gt_pos, gt_dir):
    """
    通道预测损失函数
    """
    pos_loss = F.mse_loss(pred_pos, gt_pos)
    # 方向损失（考虑方向和反方向等价）
    cos_sim = F.cosine_similarity(pred_dir, gt_dir, dim=1)
    dir_loss = torch.mean(1 - torch.abs(cos_sim))

    return pos_loss + dir_loss


def train_channel_predictor(model, data_loader, optimizer, epochs=100, save_path="models/channel_predictor.pt"):
    """
    训练通道预测模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_idx, batch in enumerate(data_loader):
            obstacles = batch["obstacles"].to(device)
            gt_pos = batch["channel_pos"].to(device)
            gt_dir = batch["channel_dir"].to(device)

            optimizer.zero_grad()

            pred_pos, pred_dir = model(obstacles)
            loss = channel_prediction_loss(pred_pos, pred_dir, gt_pos, gt_dir)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(data_loader)
        logger.info(f"Epoch {epoch+1}/{epochs} complete, Avg Loss: {avg_loss:.4f}")

        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")

    # 保存最终模型
    torch.save(model.state_dict(), save_path)
    logger.info(f"Final model saved to {save_path}")

    return model


def visualize_point_cloud(obstacles, pred_pos=None, pred_dir=None, gt_pos=None, gt_dir=None, save_path=None):
    """
    可视化点云和预测的通道
    """
    if isinstance(obstacles, torch.Tensor):
        obstacles = obstacles.detach().cpu().numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制障碍物点 (红色)
    ax.scatter(
        obstacles[:, 0],
        obstacles[:, 1],
        obstacles[:, 2],
        c="r",
        marker="o",
        s=obstacles[:, 3] * 100,
        alpha=0.5,
        label="Obstacles",
    )

    # 如果有预测通道，绘制通道
    if pred_pos is not None and pred_dir is not None:
        if isinstance(pred_pos, torch.Tensor):
            pred_pos = pred_pos.detach().cpu().numpy()
        if isinstance(pred_dir, torch.Tensor):
            pred_dir = pred_dir.detach().cpu().numpy()

        # 绘制预测通道位置和方向
        start = pred_pos
        ax.quiver(
            start[0],
            start[1],
            start[2],
            pred_dir[0],
            pred_dir[1],
            pred_dir[2],
            length=2.0,
            normalize=False,
            color="b",
            linewidth=2,
            label="Predicted Channel",
        )

    # 如果有真实通道，绘制通道
    if gt_pos is not None and gt_dir is not None:
        if isinstance(gt_pos, torch.Tensor):
            gt_pos = gt_pos.detach().cpu().numpy()
        if isinstance(gt_dir, torch.Tensor):
            gt_dir = gt_dir.detach().cpu().numpy()

        # 绘制真实通道位置和方向
        start = gt_pos
        ax.quiver(
            start[0],
            start[1],
            start[2],
            gt_dir[0],
            gt_dir[1],
            gt_dir[2],
            length=2.0,
            normalize=False,
            color="g",
            linewidth=2,
            label="Ground Truth Channel",
        )

    # 设置轴标签和图例
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Obstacle Point Cloud with Channel")
    ax.legend()

    # 保存或显示图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_channel_scene(sample, save_path=None, show_all_channels=True):
    """
    增强的可视化函数，显示通道结构
    """
    obstacles = sample["obstacles"]
    main_pos = sample["channel_pos"]
    main_dir = sample["channel_dir"]

    if isinstance(obstacles, torch.Tensor):
        obstacles = obstacles.detach().cpu().numpy()
    if isinstance(main_pos, torch.Tensor):
        main_pos = main_pos.detach().cpu().numpy()
    if isinstance(main_dir, torch.Tensor):
        main_dir = main_dir.detach().cpu().numpy()

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制障碍物点 - 使用颜色映射显示深度
    p = ax.scatter(
        obstacles[:, 0],
        obstacles[:, 1],
        obstacles[:, 2],
        c=obstacles[:, 2],  # 使用Z坐标作为颜色
        cmap="viridis",
        marker="o",
        s=obstacles[:, 3] * 150,  # 球体大小
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
    )
    fig.colorbar(p, ax=ax, label="Z axis")

    # 绘制主通道
    ax.quiver(
        main_pos[0],
        main_pos[1],
        main_pos[2],
        main_dir[0],
        main_dir[1],
        main_dir[2],
        length=2.0,
        normalize=False,
        color="r",
        linewidth=3,
        label="main channel direction",
    )

    # 如果有channels_info并且show_all_channels为True，显示所有通道
    if "channels_info" in sample and show_all_channels:
        for i, channel in enumerate(sample["channels_info"]):
            center = channel["center"]
            direction = channel["direction"]
            ch_type = channel["type"]

            if isinstance(center, torch.Tensor):
                center = center.detach().cpu().numpy()
            if isinstance(direction, torch.Tensor):
                direction = direction.detach().cpu().numpy()

            # 跳过主通道（已经用红色箭头表示）
            is_main = np.allclose(center, main_pos) and np.allclose(direction, main_dir)
            if not is_main:
                ax.quiver(
                    center[0],
                    center[1],
                    center[2],
                    direction[0],
                    direction[1],
                    direction[2],
                    length=1.5,
                    normalize=False,
                    color="blue",
                    linewidth=2,
                    label=f"channel{i+1} ({ch_type})",
                )

    # 设置轴标签和图例
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Channels and Obstacles Visualization")

    # 自动调整轴刻度和视角
    ax.set_box_aspect([1, 1, 1])  # 等比例

    # 添加图例（但过滤掉重复的标签）
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best")

    # 保存或显示图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        logger.info(f"Visualization saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()

    plt.close()


def test_model(model, test_loader, num_visualizations=5):
    """
    测试模型并可视化结果
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Testing on {device}")

    model = model.to(device)
    model.eval()

    total_loss = 0.0
    pos_errors = []
    dir_errors = []

    os.makedirs("visualizations", exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            obstacles = batch["obstacles"].to(device)
            gt_pos = batch["channel_pos"].to(device)
            gt_dir = batch["channel_dir"].to(device)

            pred_pos, pred_dir = model(obstacles)
            loss = channel_prediction_loss(pred_pos, pred_dir, gt_pos, gt_dir)

            # 计算位置误差
            pos_error = torch.norm(pred_pos - gt_pos, dim=1).mean().item()

            # 计算方向角度误差 (弧度)
            cos_sim = torch.abs(F.cosine_similarity(pred_dir, gt_dir, dim=1))
            angle_error = torch.acos(torch.clamp(cos_sim, -1.0, 1.0)).mean().item()

            total_loss += loss.item()
            pos_errors.append(pos_error)
            dir_errors.append(angle_error)

            # 可视化一些结果
            if batch_idx < num_visualizations:
                for i in range(min(2, obstacles.size(0))):
                    # 过滤掉填充的小球 (半径为0)
                    valid_mask = obstacles[i, :, 3] > 0
                    valid_obstacles = obstacles[i, valid_mask]

                    visualize_point_cloud(
                        valid_obstacles.cpu(),
                        pred_pos[i].cpu(),
                        pred_dir[i].cpu(),
                        gt_pos[i].cpu(),
                        gt_dir[i].cpu(),
                        save_path=f"visualizations/test_sample_{batch_idx}_{i}.png",
                    )

    avg_loss = total_loss / len(test_loader)
    avg_pos_error = sum(pos_errors) / len(pos_errors)
    avg_dir_error = sum(dir_errors) / len(dir_errors) * 180 / np.pi  # 转换为角度

    logger.info(f"Test Results:")
    logger.info(f"Average Loss: {avg_loss:.4f}")
    logger.info(f"Average Position Error: {avg_pos_error:.4f}")
    logger.info(f"Average Direction Error: {avg_dir_error:.2f} degrees")

    return avg_loss, avg_pos_error, avg_dir_error


def main():
    # 创建数据集
    logger.info("Creating datasets")
    dataset = BallCloudDataset(num_samples=1000, generate_samples=True)

    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=BallsToTensor(max_balls=200))

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=BallsToTensor(max_balls=200))

    # 创建模型
    logger.info("Creating model")
    model = ChannelPredictor()

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    logger.info("Starting training")
    model = train_channel_predictor(model, train_loader, optimizer, epochs=50, save_path="models/channel_predictor.pt")

    # 测试模型
    logger.info("Testing model")
    test_model(model, test_loader)

    logger.info("Done!")


def test_channel_generation(num_scenes=5, save_dir="visualizations/channel_scenes"):
    """测试通道生成并可视化结果"""
    os.makedirs(save_dir, exist_ok=True)

    # 创建数据集生成器
    dataset = BallCloudDataset(num_samples=num_scenes, generate_samples=True)

    # 可视化每个场景
    for i in range(num_scenes):
        sample = dataset[i]

        # 计算有多少种不同类型的通道
        channel_types = [ch["type"] for ch in sample["channels_info"]]
        type_counts = {t: channel_types.count(t) for t in set(channel_types)}
        type_info = ", ".join([f"{t}: {c}" for t, c in type_counts.items()])

        logger.info(f"Scene {i+1} - {len(sample['channels_info'])} channels ({type_info})")

        # 可视化场景
        save_path = f"{save_dir}/channel_scene_{i+1}.png"
        visualize_channel_scene(sample=sample, show_all_channels=True)

        logger.info(f"Scene {i+1} visualized and saved to {save_path}")


if __name__ == "__main__":
    # main()
    test_channel_generation(num_scenes=10)
