#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Plot planning algorithm performance from log file')
    parser.add_argument('--timestamp', type=str, default=None, 
                        help='Timestamp of the log directory to plot (format: YYYYMMDD_HHMMSS)')
    args = parser.parse_args()

    # 获取时间戳
    if args.timestamp:
        time_stamp = args.timestamp
    else:
        # 如果未提供时间戳，查找最新的时间戳文件夹
        corner_case_dir = os.path.join(os.path.dirname(__file__), 'logs/corner_case')
        timestamp_dirs = [d for d in os.listdir(corner_case_dir) 
                        if os.path.isdir(os.path.join(corner_case_dir, d))]
        
        if not timestamp_dirs:
            print("错误: 未找到任何时间戳目录")
            return
            
        # 按时间戳字符串排序（格式为YYYYMMDD_HHMMSS）
        timestamp_dirs.sort(reverse=True)  # 降序排列，最新的在前
        time_stamp = timestamp_dirs[0]
        print(f"未指定时间戳，使用最新的时间戳: {time_stamp}")
    
    # 构建log文件的路径
    log_dir = os.path.join(os.path.dirname(__file__), 'logs/corner_case', time_stamp)
    log_path = os.path.join(log_dir, 'log.json')
    
    if not os.path.exists(log_path):
        print(f"错误: 未找到日志文件 {log_path}")
        return
    
    # 读取JSON文件
    with open(log_path, 'r') as f:
        data = json.load(f)

    # 解析数据
    algorithms = list(data.keys())
    success_rates = []
    success_rates_std = []
    planning_times = []
    planning_times_std = []

    for alg in algorithms:
        # 计算成功率
        successes = [entry[1] for entry in data[alg]]
        success_rate = np.mean(successes) * 100  # 转换为百分比
        success_rates.append(success_rate)
        
        # 计算成功率的标准差
        n = len(successes)
        if n > 1:
            std = np.std(successes, ddof=1) * 100 / np.sqrt(n)  # 标准误差，转换为百分比
        else:
            std = 0
        success_rates_std.append(std)
        
        # 计算平均规划时间
        planning_time = [entry[2] for entry in data[alg]]
        mean_time = np.mean(planning_time)
        planning_times.append(mean_time)
        
        # 计算规划时间的标准差
        if n > 1:
            time_std = np.std(planning_time, ddof=1) / np.sqrt(n)  # 标准误差
        else:
            time_std = 0
        planning_times_std.append(time_std)

    # 为每个算法定义颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']  # 蓝色, 橙色, 绿色, 红色, 紫色, 棕色, 粉色, 灰色

    # 设置图像
    plt.figure(figsize=(12, 5))

    # 绘制成功率柱状图
    plt.subplot(1, 2, 1)
    x = np.arange(len(algorithms))
    width = 0.5
    bars = plt.bar(x, success_rates, width, yerr=success_rates_std, capsize=5, color=colors)
    plt.ylabel('Success Rate (%)')
    plt.title(f'Success Rate Comparison ({time_stamp})')
    plt.xticks(x, algorithms)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加算法名称的图例
    plt.legend(bars, algorithms, title="Algorithms")

    # 绘制规划时间柱状图
    plt.subplot(1, 2, 2)
    bars = plt.bar(x, planning_times, width, yerr=planning_times_std, capsize=5, color=colors)
    plt.ylabel('Planning Time (seconds)')
    plt.title(f'Average Planning Time Comparison ({time_stamp})')
    plt.xticks(x, algorithms)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加算法名称的图例
    plt.legend(bars, algorithms, title="Algorithms")

    plt.tight_layout()
    
    # 将图像保存到对应的时间戳文件夹
    output_file = os.path.join(log_dir, 'algorithm_comparison.png')
    plt.savefig(output_file, dpi=300)
    print(f"图表已保存到: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    main()
