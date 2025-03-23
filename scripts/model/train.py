import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pytorch3d.ops as ops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from model.channel_identification import ChannelCounter
from model.dataset import BallCloudDataset


def train(
    dataset, model, epochs=100, batch_size=16, learning_rate=0.001, val_split=0.2, log_dir="logs", model_dir="models"
):
    """
    训练通道数量识别模型 - 使用随机采样批处理

    Args:
        dataset: 数据集
        model: 模型
        epochs: 训练轮数
        batch_size: 批大小
        learning_rate: 学习率
        val_split: 测试集比例
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 创建日志目录和文件
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")

    # 检查日志文件是否存在，如果存在则删除
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
            print(f"Deleted existing log file: {log_file}")
        except Exception as e:
            print(f"Error deleting log file: {e}")

    # 配置日志
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(formatter)
    # console_handler.setLevel(logging.INFO)

    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)

    # 记录初始信息
    logger.info("=" * 50)
    logger.info("Training started")
    logger.info(f"Log file saved to: {log_file}")

    # 划分训练集和测试集
    dataset_size = len(dataset)
    test_size = int(dataset_size * val_split)
    train_size = dataset_size - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Start training - Device: {device}")
    logger.info(f"Train set size: {train_size}, Test set size: {test_size}")
    logger.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}")

    best_test_accuracy = 0

    for epoch in range(epochs):
        model.train()
        correct = 0
        count = 0

        # 随机采样batch_size个数据
        batch_indices = torch.randint(len(train_dataset), (batch_size,))
        current_batch = {"obstacles": [], "targets": [], "centers": [], "directions": [], "sizes": []}

        for idx in batch_indices:
            sample = train_dataset[idx]
            obstacles = sample["obstacles"].to(device)
            num_channels = sample["num_channels"]
            target = num_channels - 1

            # 获取通道特征
            centers = sample["channel_centers"].to(device)
            directions = sample["channel_directions"].to(device)
            sizes = sample["channel_sizes"].to(device)

            current_batch["obstacles"].append(obstacles)
            current_batch["targets"].append(target)
            current_batch["centers"].append(centers)
            current_batch["directions"].append(directions)
            current_batch["sizes"].append(sizes)

        # 处理批次数据
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

        # 处理通道特征 - 使用填充确保批次内所有样本有相同维度
        max_channels = max(centers.shape[0] for centers in current_batch["centers"])

        padded_centers = []
        padded_directions = []
        padded_sizes = []

        for batch_idx in range(len(current_batch["centers"])):
            centers = current_batch["centers"][batch_idx]
            directions = current_batch["directions"][batch_idx]
            sizes = current_batch["sizes"][batch_idx]

            # 填充通道特征
            num_channels = centers.shape[0]
            if num_channels < max_channels:
                pad_size_centers = (0, 0, 0, max_channels - num_channels)
                pad_size_directions = (0, 0, 0, max_channels - num_channels)
                pad_size_sizes = (0, 0, 0, max_channels - num_channels)

                centers = F.pad(centers, pad_size_centers)
                directions = F.pad(directions, pad_size_directions)
                sizes = F.pad(sizes, pad_size_sizes)

            padded_centers.append(centers)
            padded_directions.append(directions)
            padded_sizes.append(sizes)

        batch_centers = torch.stack(padded_centers).to(device)
        batch_directions = torch.stack(padded_directions).to(device)
        batch_sizes = torch.stack(padded_sizes).to(device)
        batch_targets = torch.tensor(current_batch["targets"], device=device)

        # 前向传播和反向传播
        optimizer.zero_grad()
        output = model(batch_obstacles, batch_centers, batch_directions, batch_sizes)
        loss = criterion(output, batch_targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # 计算准确率
        pred = output.argmax(dim=1)
        correct = (pred == batch_targets).sum().item()
        count = len(batch_targets)

        # 计算平均损失和准确率
        train_loss = loss.item()
        train_accuracy = 100 * correct / count

        # 在测试集上评估
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)

        # 记录训练和测试结果
        logger.info(f"Epoch {epoch+1}/{epochs}:")
        logger.info(f"Train set - loss: {train_loss:.4f}, accuracy: {train_accuracy:.2f}%")
        logger.info(f"Test set - loss: {test_loss:.4f}, accuracy: {test_accuracy:.2f}%")
        logger.info(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # 保存最佳模型
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            model_save_path = os.path.join(model_dir, f"best_channel_identification_{timestamp}.pth")
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Save best model, test set accuracy: {test_accuracy:.2f}%")

        logger.info("-" * 50)

    logger.info("Training completed")
    logger.info(f"Best test set accuracy: {best_test_accuracy:.2f}%")

    model_save_path = os.path.join(model_dir, f"channel_identification_{timestamp}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
    return model


def evaluate_model(model, dataloader, criterion, device, logger=None):
    """在测试集上评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            # 获取单个样本数据
            obstacles = batch["obstacles"][0].to(device)
            num_channels = batch["num_channels"]
            target = num_channels - 1

            # 获取通道特征
            centers = batch["channel_centers"].to(device)
            directions = batch["channel_directions"].to(device)
            sizes = batch["channel_sizes"].to(device)

            # 直接使用单个样本数据进行预测
            output = model(obstacles.unsqueeze(0), centers.unsqueeze(0), directions.unsqueeze(0), sizes.unsqueeze(0))

            # 计算损失
            target_tensor = torch.tensor([target], device=device)
            loss = criterion(output, target_tensor)
            total_loss += loss.item()

            # 计算准确率
            pred = output.argmax(dim=1)
            correct += (pred == target_tensor).sum().item()
            total += 1

            # 记录单个样本的测试结果
            if logger:
                logger.info(f"Sample prediction: {pred.item()}, target: {target}, " f"correct: {pred.item() == target}")

    # 计算平均损失和总体准确率
    avg_loss = total_loss / total
    accuracy = 100 * correct / total

    if logger:
        logger.info(f"Overall test set - loss: {avg_loss:.4f}, accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train channel counter")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of training samples")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Model save directory")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")

    args = parser.parse_args()

    if not args.eval:

        # 创建保存目录
        os.makedirs(args.save_dir, exist_ok=True)

        # 生成数据集
        dataset = BallCloudDataset(num_samples=args.num_samples, generate_samples=True)

        # 创建模型
        model = ChannelCounter(max_channels=20)  # 最大支持20个通道

        # 训练模型
        model = train(
            dataset=dataset,
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            model_dir=args.save_dir,
        )

    else:
        # 配置日志
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        # 配置根日志记录器
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # 加载模型
        model = ChannelCounter(max_channels=20)

        latest_checkpoint = "channel_identification_20250324_003041.pth"
        checkpoint_path = os.path.join(args.save_dir, latest_checkpoint)

        # 加载模型权重
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        print(f"Load model from: {checkpoint_path}")

        # 创建测试数据集
        test_dataset = BallCloudDataset(num_samples=50, generate_samples=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        criterion = nn.CrossEntropyLoss()

        # 评估模型
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device, logger=logger)
        print(f"Test set - loss: {test_loss:.4f}, accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
