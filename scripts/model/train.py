import argparse
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# 添加项目根目录到路径
HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from model.data_loader import SceneDataLoader
from eth_ws.src.husky_assembly.scripts.model.multibrach_model import MultiPathTrajectoryNetwork, FullTrajectoryLoss


def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, data_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0

    for batch_idx, data in enumerate(data_loader):
        # 从批次数据中提取输入
        grasped_point_cloud = data["grasped_point_cloud"].to(device)
        env_point_cloud = data["point_cloud"].to(device)
        start_joints = data["robot_start_pose"].to(device)
        target_joints = data["robot_target_pose"].to(device)
        grasp_offset = data["grasp_offset"].to(device)
        full_trajectory = data["trajectory"].to(device)

        # 前向传播
        optimizer.zero_grad()
        pred_trajectory = model(grasped_point_cloud, env_point_cloud, start_joints, target_joints, grasp_offset)

        # 计算损失
        loss = criterion(pred_trajectory, full_trajectory)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 累积损失
        total_loss += loss.item()

        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(data_loader)}, Loss: {loss.item():.6f}")

    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def validate(model, data_loader, criterion, device):
    """验证模型性能"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data in data_loader:
            # 从批次数据中提取输入
            grasped_point_cloud = data["grasped_point_cloud"].to(device)
            env_point_cloud = data["point_cloud"].to(device)
            start_joints = data["robot_start_pose"].to(device)
            target_joints = data["robot_target_pose"].to(device)
            grasp_offset = data["grasp_offset"].to(device)
            full_trajectory = data["trajectory"].to(device)

            # 前向传播
            pred_trajectory = model(grasped_point_cloud, env_point_cloud, start_joints, target_joints, grasp_offset)

            # 计算损失
            loss = criterion(pred_trajectory, full_trajectory)

            # 累积损失
            total_loss += loss.item()

    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def visualize_trajectory(model, data_loader, device, num_samples=5):
    """可视化生成的轨迹"""
    model.eval()
    plt.figure(figsize=(12, 8))

    for i, data in enumerate(data_loader):
        if i >= num_samples:
            break

        # 从批次数据中提取输入
        grasped_point_cloud = data["grasped_point_cloud"].to(device)
        env_point_cloud = data["point_cloud"].to(device)
        start_joints = data["robot_start_pose"].to(device)
        target_joints = data["robot_target_pose"].to(device)
        grasp_offset = data["grasp_offset"].to(device)
        full_trajectory = data["trajectory"].to(device)

        # 生成轨迹
        with torch.no_grad():
            pred_trajectory = model(grasped_point_cloud, env_point_cloud, start_joints, target_joints, grasp_offset)

        # 转换为numpy数组
        pred_np = pred_trajectory[0].cpu().numpy()
        true_np = full_trajectory[0].cpu().numpy()

        # 绘制每个关节角度的轨迹
        for j in range(6):
            plt.subplot(num_samples, 6, i * 6 + j + 1)
            plt.plot(true_np[:, j], "b-", label="Ground Truth")
            plt.plot(pred_np[:, j], "r--", label="Predicted")
            if i == 0:
                plt.title(f"Joint {j+1}")
            if j == 0:
                plt.ylabel(f"Sample {i+1}")
            if i == num_samples - 1 and j == 0:
                plt.legend()

    plt.tight_layout()
    plt.savefig("trajectory_comparison.png")
    plt.close()


def main(args):
    # 设置随机种子
    set_seed(args.seed)

    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.output_dir, f"trajectory_model_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    # 加载数据
    print("Loading dataset...")
    data_loader = SceneDataLoader()
    dataset = data_loader.create_dataset(
        scene_names=args.scenes,
        task_names=args.tasks,
        algorithm_names=args.algorithms,
        num_points=args.num_points,
        num_grasp_points=args.num_grasp_points,
        normal_channel=args.normal_channel,
        trajectory_length=args.trajectory_length,
    )

    # 划分数据集
    dataset_size = len(dataset)
    val_size = int(dataset_size * args.val_ratio)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Dataset size: Total={dataset_size}, Train={train_size}, Val={val_size}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 初始化模型
    print("Initializing model...")
    model = MultiPathTrajectoryNetwork(
        object_feature_dim=args.object_feature_dim,
        env_feature_dim=args.env_feature_dim,
        task_feature_dim=args.task_feature_dim,
        traj_feature_dim=args.traj_feature_dim,
        fusion_dim=args.fusion_dim,
        output_seq_len=args.trajectory_length,
        joint_dim=6,  # 假设有6个关节
    ).to(device)

    # 初始化损失函数和优化器
    criterion = FullTrajectoryLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    # 训练日志
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    # 训练循环
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # 更新学习率
        scheduler.step(val_loss)

        # 打印结果
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(result_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                model_path,
            )
            print(f"Saved best model to {model_path}")

        # 定期保存检查点
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(result_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )

    # 保存最终模型
    final_model_path = os.path.join(result_dir, "final_model.pth")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_losses[-1],
            "val_loss": val_losses[-1],
        },
        final_model_path,
    )
    print(f"Saved final model to {final_model_path}")

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(result_dir, "loss_curve.png"))

    # 可视化轨迹结果
    visualize_trajectory(model, val_loader, device)
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the multi-path trajectory network")

    # 数据相关参数
    parser.add_argument("--scenes", type=str, nargs="+", default=["cuboid_1"], help="Scene names to load")
    parser.add_argument("--tasks", type=str, nargs="+", default=["task_1"], help="Task names to load")
    parser.add_argument("--algorithms", type=str, nargs="+", default=["cuRobo"], help="Algorithm names to load")
    parser.add_argument("--num-points", type=int, default=1024, help="Number of points in point cloud")
    parser.add_argument("--num-grasp-points", type=int, default=256, help="Number of points for grasped object")
    parser.add_argument("--normal-channel", action="store_true", help="Use normal channel in point cloud")
    parser.add_argument("--trajectory-length", type=int, default=256, help="Length of trajectory sequence")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation set ratio")

    # 模型相关参数
    parser.add_argument("--object-feature-dim", type=int, default=256, help="Object branch feature dimension")
    parser.add_argument("--env-feature-dim", type=int, default=512, help="Environment branch feature dimension")
    parser.add_argument("--task-feature-dim", type=int, default=256, help="Task branch feature dimension")
    parser.add_argument("--traj-feature-dim", type=int, default=512, help="Trajectory branch feature dimension")
    parser.add_argument("--fusion-dim", type=int, default=512, help="Fusion layer output dimension")

    # 训练相关参数
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--save-interval", type=int, default=10, help="Epoch interval to save checkpoint")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(args)
