import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# 添加项目根目录到路径
HERE = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(HERE)

# 导入自定义模块
from model.data_loader import SceneDataLoader as CustomDataLoader
from model.pointnet.pointnet import PointNet, to_categorical, weights_init
from utils.params import *


# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 定义训练函数
def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, data in enumerate(train_loader):
        points = data["point_cloud"].to(device)
        targets = data["element_labels"].to(device)  # 形状为 [B, N]，其中N是点的数量
        batch_size = points.size(0)

        # 转置点云以匹配PointNet期望的输入格式 [B, N, C] -> [B, C, N]
        points = points.permute(0, 2, 1)

        # 前向传播
        optimizer.zero_grad()
        pred, _ = model(points)

        targets_onehot = to_categorical(targets, 12)

        pred_flat = pred.reshape(-1, 12)
        targets_onehot_flat = targets_onehot.reshape(-1, 12)

        loss = criterion(pred_flat, targets_onehot_flat)
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = pred.max(-1)
        total += targets.size(0) * targets.size(1)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            print(
                f"Epoch: {epoch} [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {total_loss/(batch_idx+1):.4f} "
                f"Acc: {100.*correct/total:.2f}%"
            )

    return total_loss / len(train_loader), 100.0 * correct / total


# 定义测试函数
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            points = data["point_cloud"].to(device)
            targets = data["element_labels"].to(device)  # 形状为 [B, N]，其中N是点的数量
            batch_size = points.size(0)

            # 转置点云以匹配PointNet期望的输入格式 [B, N, C] -> [B, C, N]
            points = points.permute(0, 2, 1)

            # 前向传播
            pred, _ = model(points)

            targets_onehot = to_categorical(targets, 12)

            pred_flat = pred.reshape(-1, 12)
            targets_flat = targets.reshape(-1)
            targets_onehot_flat = targets_onehot.reshape(-1, 12)

            loss = criterion(pred_flat, targets_onehot_flat)

            # 统计
            total_loss += loss.item()
            _, predicted = pred.max(-1)  # 形状为 [B, N]
            predicted_flat = predicted.reshape(-1)  # 将预测结果展平为 [B*N]

            total += targets.size(0) * targets.size(1)
            correct += predicted.eq(targets).sum().item()

            # 收集所有预测和标签，用于计算混淆矩阵
            all_preds.append(predicted_flat.cpu().numpy())
            all_targets.append(targets_flat.cpu().numpy())

    # 合并所有批次的预测和标签
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_targets, all_preds)

    print(f"Test Loss: {total_loss/len(test_loader):.4f} " f"Acc: {100.*correct/total:.2f}%")

    return total_loss / len(test_loader), 100.0 * correct / total, conf_matrix


# 可视化点云分类结果
def visualize_results(model, test_loader, device, num_samples=1):
    """可视化模型分类结果"""
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i >= num_samples:
                break

            points = data["point_cloud"].to(device)
            targets = data["element_labels"].to(device)

            # 获取原始点云
            points_np = points[0].cpu().numpy()

            # 转置点云以匹配PointNet期望的输入格式
            points_tensor = points.permute(0, 2, 1)

            # 预测
            pred, _ = model(points_tensor)

            # 获取每个点的预测类别
            _, predicted = pred.max(2)  # 改为沿着第三个维度取最大值，因为输出为[B, N, C]

            # 转换为NumPy数组以便可视化
            targets_np = targets[0].cpu().numpy()
            predicted_np = predicted[0].cpu().numpy()  # 取第一个样本

            # 计算正确率
            correct = predicted_np == targets_np
            accuracy = correct.sum() / len(correct) * 100

            # 可视化
            fig = plt.figure(figsize=(15, 5))

            # 绘制原始标签
            ax1 = fig.add_subplot(131, projection="3d")
            if points_np.shape[1] > 3:  # 如果点云包含法向量
                pc_xyz = points_np[:, :3]  # 只使用xyz坐标
            else:
                pc_xyz = points_np

            scatter = ax1.scatter(pc_xyz[:, 0], pc_xyz[:, 1], pc_xyz[:, 2], c=targets_np, cmap="tab10", marker=".")
            ax1.set_title("Ground Truth")
            fig.colorbar(scatter, ax=ax1)

            # 绘制预测标签
            ax2 = fig.add_subplot(132, projection="3d")
            scatter = ax2.scatter(pc_xyz[:, 0], pc_xyz[:, 1], pc_xyz[:, 2], c=predicted_np, cmap="tab10", marker=".")
            ax2.set_title("Predictions")
            fig.colorbar(scatter, ax=ax2)

            # 绘制正确与错误的点
            ax3 = fig.add_subplot(133, projection="3d")
            scatter = ax3.scatter(pc_xyz[:, 0], pc_xyz[:, 1], pc_xyz[:, 2], c=correct, cmap="RdYlGn", marker=".")
            ax3.set_title(f"Classification Results (Accuracy: {accuracy:.2f}%)")

            plt.tight_layout()
            plt.savefig(f"classification_result.png")
            plt.show()


def main(args):
    set_seed(args.seed)

    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    data_loader = CustomDataLoader()

    # 创建数据集
    dataset = data_loader.create_dataset(
        scene_names="cuboid_1",
        task_names="task_1",
        algorithm_names="BIRRT",
        num_points=args.num_points,
        normal_channel=args.normal_channel,
        trajectory_length=2048,
    )

    # 如果数据集为空，返回
    if len(dataset) == 0:
        print("Error: Dataset is empty")
        return

    print(f"Dataset size: {len(dataset)}")

    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 初始化PointNet模型
    feature_dim = 12  # 二分类问题
    model = PointNet(feature_dim, normal_channel=args.normal_channel).to(device)
    model.apply(weights_init)

    # 使用加权损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 训练模型
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch)
        test_loss, test_acc, _ = test(model, test_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 保存模型
        # if epoch % 10 == 0:
        #     torch.save(model.state_dict(), f"pointnet_epoch_{epoch}.pth")

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(test_accs, label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()

    # 在测试集上进行最终评估
    _, _, conf_matrix = test(model, test_loader, criterion, device)

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()

    # 可视化点云分类结果
    visualize_results(model, test_loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--num_points", type=int, default=1024, help="points per cylinder")
    parser.add_argument("--normal_channel", type=bool, default=True, help="use normal channel")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    main(args)
