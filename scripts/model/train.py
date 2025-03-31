import argparse
import multiprocessing
import os
import sys
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from torch.utils.data import DataLoader, random_split

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from model.data_loader import SceneDataLoader
from model.multibrach_model import (HybridTrajectoryLoss,
                                    MultiPathTrajectoryNetwork)


def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, data_loader, optimizer, criterion, device, stage=1):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0

    # Use rich to create progress display
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]Stage {stage} Training[/bold blue]"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeElapsedColumn(),
        TextColumn("Remaining:"),
        TimeRemainingColumn(),
        TextColumn("{task.fields[loss]}"),
        expand=True,
    ) as progress:
        task = progress.add_task("[bold]Training...", total=len(data_loader), loss="")

        for batch_idx, data in enumerate(data_loader):
            # 确保这些数据加载正确并转移到正确的设备上
            grasped_point_cloud = data["grasped_point_cloud"].to(device)
            env_point_cloud = data["point_cloud"].to(device)
            start_joints = data["robot_start_pose"].to(device)
            target_joints = data["robot_target_pose"].to(device)
            grasp_offset = data["grasp_offset"].to(device)
            full_trajectory = data["trajectory"].to(device)

            # 前向传播
            optimizer.zero_grad()
            pred_trajectory = model(
                grasped_point_cloud,
                env_point_cloud,
                start_joints,
                target_joints,
                grasp_offset,
                input_trajectory=full_trajectory,
            )

            # 计算损失
            loss = criterion(pred_trajectory, full_trajectory)
            if torch.isnan(loss):
                print("Warning: NaN loss detected! Skipping this batch.")
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            # 反向传播和优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # 累积损失
            total_loss += loss.item()

            # 更新进度条显示当前批次的损失值
            current_loss = loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress.update(
                task,
                advance=1,
                loss=f"[bold]Current Loss:[/bold] [yellow]{current_loss:.6f}[/yellow] [bold]Average Loss:[/bold] [green]{avg_loss:.6f}[/green]",
            )

    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def validate(model, data_loader, criterion, device, stage=1):
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
            pred_trajectory = model(
                grasped_point_cloud,
                env_point_cloud,
                start_joints,
                target_joints,
                grasp_offset,
                input_trajectory=full_trajectory,
            )

            # 计算损失
            loss = criterion(pred_trajectory, full_trajectory)

            # 累积损失
            total_loss += loss.item()

    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    return avg_loss


# 第一阶段损失函数：基于L1和L2损失进行粗略学习
class Stage1Loss(nn.Module):
    """第一阶段损失函数：主要使用L1和L2损失"""

    def __init__(self):
        super(Stage1Loss, self).__init__()

    def forward(self, pred_trajectory, full_trajectory):
        # 计算L1损失
        l1_loss = F.l1_loss(pred_trajectory, full_trajectory)

        # 计算L2损失
        l2_loss = F.mse_loss(pred_trajectory, full_trajectory)

        # 单独计算首尾点损失，避免使用cat操作
        start_point_loss = F.mse_loss(pred_trajectory[:, 0, :], full_trajectory[:, 0, :])
        end_point_loss = F.mse_loss(pred_trajectory[:, -1, :], full_trajectory[:, -1, :])

        # 加强首尾点损失权重
        keypoints_loss = (start_point_loss + end_point_loss) * 5.0

        # 综合损失
        total_loss = l1_loss * 0.5 + l2_loss + keypoints_loss

        return total_loss


def train_two_stage(args):
    """分阶段训练主函数"""
    # 设置随机种子
    set_seed(args.seed)

    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(HERE, "model", args.output_dir, f"trajectory_model_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    # 创建日志文件
    log_file = os.path.join(result_dir, "training.log")

    def log_message(message):
        """仅将消息写入日志文件"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_line)

    # 加载数据
    print("Loading dataset...")
    data_loader = SceneDataLoader()

    # 获取指定场景下的所有任务
    all_tasks = []
    all_algorithms = []
    for scene in args.scenes:
        tasks = data_loader.list_tasks_for_scene(scene)
        all_tasks.extend([(scene, task) for task in tasks])
        # 对每个任务获取可用的算法
        for task in tasks:
            algorithms = data_loader.list_algorithms_for_task(scene, task)
            all_algorithms.extend(algorithms)
    # 去重算法名称
    all_algorithms = list(set(all_algorithms))

    # 创建数据集
    dataset = data_loader.create_dataset(
        scene_names=args.scenes,
        task_names=None,
        algorithm_names=None,
        num_points=args.num_points,
        num_grasp_points=args.num_grasp_points,
        normal_channel=args.normal_channel,
        trajectory_length=args.trajectory_length,
        add_noise=args.add_noise,
        noise_ratio=args.noise_ratio,
        noise_scale=args.noise_scale,
    )

    # 划分数据集
    dataset_size = len(dataset)
    if dataset_size == 0:
        print("Error: Dataset is empty. Exiting.")
        sys.exit(1)

    val_size = int(dataset_size * args.val_ratio)
    train_size = dataset_size - val_size
    
    # Ensure validation set is not empty if dataset is small
    if val_size == 0 and train_size > 0:
        val_size = 1
        train_size -= 1
        warnings.warn("Validation set size was 0, adjusted to 1.")

    if train_size <= 0:
         print(f"Error: Training set size is {train_size}. Not enough data. Exiting.")
         sys.exit(1)

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 记录训练配置到日志文件
    log_message("\n====== Training Configuration ======")
    log_message(f"Device: {device}")
    log_message(f"Scenes: {args.scenes}")
    log_message("Tasks:")
    for scene, task in all_tasks:
        log_message(f"- {scene}/{task}")
    log_message(f"Algorithms: {all_algorithms}")
    log_message(f"Dataset Size: Total={dataset_size}, Train={train_size}, Val={val_size}")
    log_message(f"Batch Size: {args.batch_size}")
    log_message(f"Learning Rate: {args.lr}")
    log_message(f"Dropout Rate: {args.dropout}")
    log_message(f"Stage 1 Epochs: {'Default (1/3 total)' if args.stage1_epochs <= 0 else args.stage1_epochs}")
    log_message(f"Stage 2 Epochs: {'Default (remaining)' if args.stage2_epochs <= 0 else args.stage2_epochs}")
    log_message(f"Output Sequence Length: {args.output_seq_len}")
    log_message(f"Trajectory Length (Input/Target): {args.trajectory_length}")
    log_message(f"Normal Channel: {args.normal_channel}")
    log_message(f"Use LSTM: {args.use_lstm}")
    log_message(f"Add Noise: {args.add_noise}")
    if args.add_noise:
        log_message(f"  Noise Ratio: {args.noise_ratio}")
        log_message(f"  Noise Scale: {args.noise_scale}")
    log_message("====== Starting Training ======\n")

    # Print config to console
    console = Console()
    console.print("\n[bold]Dataset Information:[/bold]")
    console.print(f"Scenes: {args.scenes}")
    console.print("Tasks:")
    for scene, task in all_tasks:
        console.print(f"- {scene}/{task}")
    console.print(f"Algorithms: {all_algorithms}")
    console.print(f"Dataset size: Total={dataset_size}, Train={train_size}, Val={val_size}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # 模型权重初始化函数
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    # ============================== 第一阶段训练 ==============================
    console.print(Panel("[bold yellow]Stage 1 Training: Learning Shape with L1/L2 Loss[/bold yellow]", expand=False))
    log_message("[Stage 1] Starting training: Learning shape with L1/L2 loss")

    # 初始化模型
    print("Initializing model for Stage 1...")
    model = MultiPathTrajectoryNetwork(
        object_feature_dim=args.object_feature_dim,
        env_feature_dim=args.env_feature_dim,
        task_feature_dim=args.task_feature_dim,
        traj_feature_dim=args.traj_feature_dim,
        fusion_dim=args.fusion_dim,
        output_seq_len=args.output_seq_len,
        joint_dim=6,
        normal_channel=args.normal_channel,
        use_lstm=args.use_lstm,
        dropout=0.3,  # 第一阶段使用较小的dropout
    ).to(device)

    # 应用权重初始化
    model.apply(weights_init)

    # 第一阶段损失函数：使用L1和L2损失
    stage1_criterion = Stage1Loss().to(device)

    # 第一阶段优化器
    stage1_optimizer = optim.Adam(model.parameters(), lr=args.lr * 2, weight_decay=1e-5)

    # 学习率调度器
    stage1_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        stage1_optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # 训练日志
    stage1_train_losses = []
    stage1_val_losses = []

    # 第一阶段训练循环
    best_stage1_val_loss = float("inf")
    patience_stage1 = 10
    patience_counter_stage1 = 0
    stage1_epochs = args.stage1_epochs if args.stage1_epochs > 0 else args.epochs // 3

    for epoch in range(stage1_epochs):
        # 控制台输出保持不变
        console.print(Panel(f"[yellow]Stage 1 - Epoch {epoch+1}/{stage1_epochs}[/yellow]", expand=False))

        # 训练
        train_loss = train_epoch(model, train_loader, stage1_optimizer, stage1_criterion, device, stage=1)
        stage1_train_losses.append(train_loss)

        # 验证
        val_loss = validate(model, val_loader, stage1_criterion, device, stage=1)
        stage1_val_losses.append(val_loss)

        # 更新学习率
        stage1_scheduler.step(val_loss)

        # 记录到日志文件
        current_lr_stage1 = stage1_optimizer.param_groups[0]['lr']
        log_message(
            f"Stage 1 - Epoch {epoch+1}/{stage1_epochs} - "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
            f"LR: {current_lr_stage1:.6e}"
        )

        # 控制台输出保持不变
        console.print(
            f"[bold]Stage 1 - Epoch {epoch+1}/{stage1_epochs}[/bold] - "
            f"Train Loss: [yellow]{train_loss:.6f}[/yellow], "
            f"Val Loss: [green]{val_loss:.6f}[/green], "
            f"LR: [blue]{current_lr_stage1:.6e}[/blue]"
        )

        if val_loss < best_stage1_val_loss:
            best_stage1_val_loss = val_loss
            patience_counter_stage1 = 0
            # 保存最佳阶段1模型
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": stage1_optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                os.path.join(result_dir, "stage1_best_model.pth"),
            )
            log_message(f"Stage 1 best model saved with validation loss: {val_loss:.6f}")
            console.print(f"[green]Stage 1 best model saved. Validation Loss: {val_loss:.6f}[/green]")
        else:
            patience_counter_stage1 += 1
            if patience_counter_stage1 >= patience_stage1:
                log_message(f"Stage 1 early stopping triggered at epoch {epoch+1}")
                console.print(f"[bold red]Stage 1 early stopping at epoch {epoch+1}.[/bold red]")
                break

    # 绘制第一阶段训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(stage1_train_losses, label="Stage 1 - Train Loss")
    plt.plot(stage1_val_losses, label="Stage 1 - Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Stage 1 - Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "stage1_loss_curve.png"))
    plt.close()

    # 加载第一阶段最佳模型
    if os.path.exists(os.path.join(result_dir, "stage1_best_model.pth")):
        checkpoint = torch.load(os.path.join(result_dir, "stage1_best_model.pth"))
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best Stage 1 model (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.6f}) for Stage 2 training.")
        log_message(f"Loaded best Stage 1 model (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.6f})")
    else:
        print("Warning: Best Stage 1 model checkpoint not found. Proceeding with the current model state for Stage 2.")
        log_message("Warning: Best Stage 1 model checkpoint not found.")

    # ============================== 第二阶段训练 ==============================
    console.print(Panel("[bold green]Stage 2 Training: Fine-tuning with Hybrid DTW Loss[/bold green]", expand=False))
    log_message("\n[Stage 2] Starting training: Fine-tuning with Hybrid DTW loss")

    # 第二阶段模型可以增加dropout防止过拟合
    model.traj_branch.dropout.p = args.dropout
    if hasattr(model.decoder, "dropout"):
        model.decoder.dropout.p = args.dropout
    if hasattr(model.fusion_layer, '2') and isinstance(model.fusion_layer[2], nn.Dropout):
         model.fusion_layer[2].p = args.dropout

    # 第二阶段使用混合损失函数，加强DTW权重
    stage2_criterion = HybridTrajectoryLoss(
        gamma=0.1, alpha_dtw=1.5, alpha_l1=0.3, alpha_l2=0.1,
        use_cuda= (device.type == 'cuda')
    ).to(device)

    # 第二阶段优化器
    stage2_optimizer = optim.AdamW(
        model.parameters(), lr=args.lr * 0.5, weight_decay=args.weight_decay, amsgrad=True
    )

    # 使用余弦退火学习率调度
    stage2_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(stage2_optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # 训练日志
    stage2_train_losses = []
    stage2_val_losses = []

    # 第二阶段训练循环
    best_stage2_val_loss = float("inf")
    patience_stage2 = 15
    patience_counter_stage2 = 0
    completed_stage1_epochs = len(stage1_train_losses)
    stage2_epochs = args.stage2_epochs if args.stage2_epochs > 0 else args.epochs - completed_stage1_epochs
    if stage2_epochs <= 0:
        stage2_epochs = args.epochs // 3 * 2

    for epoch in range(stage2_epochs):
        # 控制台输出保持不变
        console.print(Panel(f"[green]Stage 2 - Epoch {epoch+1}/{stage2_epochs}[/green]", expand=False))

        # 训练
        train_loss = train_epoch(model, train_loader, stage2_optimizer, stage2_criterion, device, stage=2)
        stage2_train_losses.append(train_loss)

        # 验证
        val_loss = validate(model, val_loader, stage2_criterion, device, stage=2)
        stage2_val_losses.append(val_loss)

        # 更新学习率
        stage2_scheduler.step()

        # 记录到日志文件
        current_lr_stage2 = stage2_scheduler.get_last_lr()[0]
        log_message(
            f"Stage 2 - Epoch {epoch+1}/{stage2_epochs} - "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
            f"LR: {current_lr_stage2:.6e}"
        )

        # 控制台输出保持不变
        console.print(
            f"[bold]Stage 2 - Epoch {epoch+1}/{stage2_epochs}[/bold] - "
            f"Train Loss: [yellow]{train_loss:.6f}[/yellow], "
            f"Val Loss: [green]{val_loss:.6f}[/green], "
            f"LR: [blue]{current_lr_stage2:.6e}[/blue]"
        )

        if val_loss < best_stage2_val_loss:
            best_stage2_val_loss = val_loss
            patience_counter_stage2 = 0
            # 保存最佳阶段2模型
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": stage2_optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                os.path.join(result_dir, "stage2_best_model.pth"),
            )
            log_message(f"Stage 2 best model saved with validation loss: {val_loss:.6f}")
            console.print(f"[green]Stage 2 best model saved. Validation Loss: {val_loss:.6f}[/green]")
        else:
            patience_counter_stage2 += 1
            if patience_counter_stage2 >= patience_stage2:
                log_message(f"Stage 2 early stopping triggered at epoch {epoch+1}")
                console.print(f"[bold red]Stage 2 early stopping at epoch {epoch+1}.[/bold red]")
                break

    # 绘制第二阶段训练曲线
    if stage2_train_losses:
        plt.figure(figsize=(10, 5))
        plt.plot(stage2_train_losses, label="Stage 2 - Train Loss")
        plt.plot(stage2_val_losses, label="Stage 2 - Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Stage 2 - Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(result_dir, "stage2_loss_curve.png"))
        plt.close()

    # 绘制整体训练曲线
    if stage1_train_losses or stage2_train_losses:
        plt.figure(figsize=(12, 6))
        if stage1_train_losses:
            plt.plot(range(1, len(stage1_train_losses) + 1), stage1_train_losses, label="Stage 1 - Train Loss", color="blue", linestyle="-")
            plt.plot(range(1, len(stage1_val_losses) + 1), stage1_val_losses, label="Stage 1 - Val Loss", color="blue", linestyle="--")
            plt.axvline(x=len(stage1_train_losses), color="black", linestyle=":", alpha=0.7, label='Stage Transition')

        if stage2_train_losses:
            x_offset = len(stage1_train_losses)
            x_stage2 = list(range(x_offset + 1, x_offset + 1 + len(stage2_train_losses)))
            plt.plot(x_stage2, stage2_train_losses, label="Stage 2 - Train Loss", color="green", linestyle="-")
            plt.plot(x_stage2, stage2_val_losses, label="Stage 2 - Val Loss", color="green", linestyle="--")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Combined Two-Stage Training Loss Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(result_dir, "combined_loss_curve.png"))
        plt.close()

    # 比较并选择最佳模型
    final_best_model_path = os.path.join(result_dir, "best_model.pth")
    stage1_best_path = os.path.join(result_dir, "stage1_best_model.pth")
    stage2_best_path = os.path.join(result_dir, "stage2_best_model.pth")
    selected_stage = "None"

    # Handle cases where one or both stages didn't save a model
    stage1_exists = os.path.exists(stage1_best_path) # Define before use
    stage2_exists = os.path.exists(stage2_best_path) # Define before use

    import shutil
    if stage1_exists and stage2_exists:
        if best_stage1_val_loss < best_stage2_val_loss:
            log_message(f"Stage 1 model performed better (Loss: {best_stage1_val_loss:.6f} vs {best_stage2_val_loss:.6f}). Copying as final model.")
            shutil.copy(stage1_best_path, final_best_model_path)
            selected_stage = "Stage 1"
        else:
            log_message(f"Stage 2 model performed better (Loss: {best_stage2_val_loss:.6f} vs {best_stage1_val_loss:.6f}). Copying as final model.")
            shutil.copy(stage2_best_path, final_best_model_path)
            selected_stage = "Stage 2"
    elif stage1_exists:
         log_message(f"Only Stage 1 model available (Loss: {best_stage1_val_loss:.6f}). Copying as final model.")
         shutil.copy(stage1_best_path, final_best_model_path)
         selected_stage = "Stage 1"
    elif stage2_exists:
         log_message(f"Only Stage 2 model available (Loss: {best_stage2_val_loss:.6f}). Copying as final model.")
         shutil.copy(stage2_best_path, final_best_model_path)
         selected_stage = "Stage 2"
    else:
         log_message("No best models saved from either stage. Saving the final state of the model.")
         torch.save({
             'model_state_dict': model.state_dict(),
         }, final_best_model_path)
         selected_stage = "Final State (No Best Checkpoint)"


    # 记录最终结果
    log_message("\n====== Training Finished ======")
    log_message(
        f"Final Results:\n"
        f"Best Stage 1 Validation Loss: {best_stage1_val_loss if stage1_exists else 'N/A'}\n"
        f"Best Stage 2 Validation Loss: {best_stage2_val_loss if stage2_exists else 'N/A'}\n"
        f"Selected Model from: {selected_stage}"
    )

    # 保持原有的控制台输出
    console.print(
        f"\n[bold]====== Training Finished ======[/bold]\n"
        f"Best Stage 1 Validation Loss: [yellow]{best_stage1_val_loss if stage1_exists else 'N/A'}[/yellow]\n"
        f"Best Stage 2 Validation Loss: [green]{best_stage2_val_loss if stage2_exists else 'N/A'}[/green]\n"
        f"Selected Model from: [bold cyan]{selected_stage}[/bold cyan]\n"
        f"Results saved to: {result_dir}"
    )

    # 保存训练损失历史
    loss_history = {
        "stage1_train_losses": stage1_train_losses,
        "stage1_val_losses": stage1_val_losses,
        "stage2_train_losses": stage2_train_losses,
        "stage2_val_losses": stage2_val_losses,
    }
    np.save(os.path.join(result_dir, "loss_history.npy"), loss_history)
    log_message(f"Loss history saved to loss_history.npy")

    return result_dir


def main(args):
    if args.two_stage:
        # 使用两阶段训练
        return train_two_stage(args)
    else:
        # 使用原始单阶段训练方法（保留原始代码）
        # ... 原始main函数代码 ...
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the multi-path trajectory network")

    # Data related arguments
    parser.add_argument("--scenes", type=str, nargs="+", default=["cuboid_1"], help="Scene names to load data from.")
    parser.add_argument("--num-points", type=int, default=1024, help="Number of points per element in the environment point cloud.")
    parser.add_argument("--num-grasp-points", type=int, default=256, help="Number of points for the grasped object point cloud.")
    parser.add_argument("--trajectory-length", type=int, default=2048, help="Target length for input/output trajectory sequences after interpolation.")
    parser.add_argument("--output-seq-len", type=int, default=2048, help="Length of the trajectory sequence generated by the decoder.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Ratio of the dataset to use for validation (e.g., 0.1 for 10%).")
    parser.add_argument("--no-normal-channel", action="store_false", dest="normal_channel", help="Disable using normal vectors in point clouds.")
    parser.set_defaults(normal_channel=True)
    parser.add_argument("--add-noise", action="store_false", help="Add random noise to point cloud during data loading.")
    parser.add_argument("--noise-ratio", type=float, default=0.05, help="Ratio of points to add noise to if --add-noise is used.")
    parser.add_argument("--noise-scale", type=float, default=0.01, help="Scale (standard deviation) of the noise if --add-noise is used.")

    # Model related arguments
    parser.add_argument("--object-feature-dim", type=int, default=512, help="Feature dimension from the object branch (PointNet).")
    parser.add_argument("--env-feature-dim", type=int, default=512, help="Feature dimension from the environment branch (PointNet).")
    parser.add_argument("--task-feature-dim", type=int, default=512, help="Feature dimension from the task branch (MLP).")
    parser.add_argument("--traj-feature-dim", type=int, default=512, help="Feature dimension from the trajectory branch (Transformer/LSTM).")
    parser.add_argument("--fusion-dim", type=int, default=1024, help="Output dimension of the feature fusion layer.")
    parser.add_argument("--use-lstm", action="store_true", help="Use LSTM instead of Transformer in the trajectory branch (potentially saves memory).")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for regularization in various model parts.")


    # Training related arguments
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training and validation.")
    parser.add_argument("--epochs", type=int, default=120, help="Total number of training epochs across both stages.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for optimizer.")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save training results (logs, models, plots).")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers for data loading.")
    parser.add_argument("--seed", type=int, default=531, help="Random seed for reproducibility.")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA, force use of CPU.")

    # Staged training arguments
    parser.add_argument("--two-stage", action="store_false", help="Enable two-stage training strategy.")
    parser.add_argument("--stage1-epochs", type=int, default=0, help="Number of epochs for Stage 1. If 0, defaults to 1/3 of total epochs.")
    parser.add_argument("--stage2-epochs", type=int, default=0, help="Number of epochs for Stage 2. If 0, defaults to remaining epochs after Stage 1.")

    args = parser.parse_args()

    # Ensure output sequence length matches trajectory length if not specified otherwise
    if args.output_seq_len != args.trajectory_length:
         warnings.warn(f"Output sequence length ({args.output_seq_len}) differs from trajectory length ({args.trajectory_length}). Ensure this is intended.")

    if args.two_stage:
        train_two_stage(args)
    else:
        # Placeholder for single-stage training if needed in the future
        print("Error: Single-stage training not implemented in this script. Use --two-stage flag.")
        pass
