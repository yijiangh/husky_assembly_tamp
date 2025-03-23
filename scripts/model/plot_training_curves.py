import matplotlib.pyplot as plt
import numpy as np
import os
import re


def parse_training_log(log_file):
    """
    解析训练日志文件，提取训练和测试的损失值及准确率

    Args:
        log_file: 训练日志文件路径

    Returns:
        dict: 包含训练和测试指标的字典
    """
    metrics = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "epochs": []}

    with open(log_file, "r") as f:
        current_epoch = None
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 匹配epoch行
            epoch_match = re.search(r"Epoch (\d+)/\d+:", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                metrics["epochs"].append(current_epoch)
                continue

            # 匹配训练集指标
            train_match = re.search(r"Train set - loss: ([\d.]+), accuracy: ([\d.]+)%", line)
            if train_match:
                loss = float(train_match.group(1))
                acc = float(train_match.group(2))
                metrics["train_loss"].append(loss)
                metrics["train_acc"].append(acc)
                continue

            # 匹配测试集指标
            test_match = re.search(r"Test set - loss: ([\d.]+), accuracy: ([\d.]+)%", line)
            if test_match:
                loss = float(test_match.group(1))
                acc = float(test_match.group(2))
                metrics["test_loss"].append(loss)
                metrics["test_acc"].append(acc)
                continue

    return metrics


def moving_average(data, window_size=5):
    """
    对数据进行移动平均滤波

    Args:
        data: 输入数据列表
        window_size: 滑动窗口大小

    Returns:
        list: 平滑后的数据
    """
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode="valid")


def plot_training_curves(log_file, save_dir=None, window_size=5):
    """
    绘制训练过程中的损失和准确率曲线

    Args:
        log_file: 训练日志文件路径
        save_dir: 保存图像的目录
        window_size: 移动平均窗口大小
    """
    metrics = parse_training_log(log_file)

    # 对数据进行平滑处理
    smooth_train_loss = moving_average(metrics["train_loss"], window_size)
    smooth_test_loss = moving_average(metrics["test_loss"], window_size)
    smooth_train_acc = moving_average(metrics["train_acc"], window_size)
    smooth_test_acc = moving_average(metrics["test_acc"], window_size)

    # 调整epochs以匹配平滑后的数据长度
    epochs = metrics["epochs"][window_size - 1 :]

    # 创建图形
    plt.figure(figsize=(15, 10))

    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(metrics["epochs"], metrics["train_loss"], "b-", alpha=0.2, label="Raw Training Loss")
    plt.plot(metrics["epochs"], metrics["test_loss"], "r-", alpha=0.2, label="Raw Validation Loss")
    plt.plot(epochs, smooth_train_loss, "b-", label="Smoothed Training Loss")
    plt.plot(epochs, smooth_test_loss, "r-", label="Smoothed Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(2, 1, 2)
    plt.plot(metrics["epochs"], metrics["train_acc"], "b-", alpha=0.2, label="Raw Training Accuracy")
    plt.plot(metrics["epochs"], metrics["test_acc"], "r-", alpha=0.2, label="Raw Validation Accuracy")
    plt.plot(epochs, smooth_train_acc, "b-", label="Smoothed Training Accuracy")
    plt.plot(epochs, smooth_test_acc, "r-", label="Smoothed Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # 保存或显示图像
    if save_dir:
        # 从日志文件名中提取时间后缀
        log_filename = os.path.basename(log_file)
        timestamp = ""
        if "_" in log_filename:
            timestamp = log_filename.split("_", 1)[1].split("_", 1)[1].rsplit(".", 1)[0]

        # 创建保存目录
        save_filename = f"training_curves_{timestamp}.png"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, save_filename))
        print(f"Training curves saved to {os.path.join(save_dir, save_filename)}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # 设置日志文件路径
    log_file = "logs/training_log_20250324_013315.txt"

    # 绘制训练曲线
    plot_training_curves(log_file, save_dir="logs")
