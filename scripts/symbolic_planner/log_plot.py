import os
import json
import matplotlib.pyplot as plt
import numpy as np


# 提取文件夹信息并返回相关部分
def extract_folder_info(folder_path):
    folder_name = os.path.basename(folder_path)
    structure_name, algorithm1, _, algorithm2 = folder_name.split("-")
    return structure_name, algorithm1, algorithm2


# 读取json文件并返回数据
def load_json_logs(folder):
    logs = []
    for filename in os.listdir(folder):
        if filename.endswith(".log"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                logs.append(data)
    return logs


# 提取所有日志文件中的place部分失败计数，并累积每个日志中的失败类型
def extract_failures(folder):
    failure_types = [
        "attach ik failure",
        "pre attach ik failure",
        "pre attach collision failure",
        "post attach ik failure",
        "post attach collision failure",
        "back plan failure",
    ]
    failure_counts = {ftype: 0 for ftype in failure_types}
    log_count = 0

    # 读取所有日志文件
    logs = load_json_logs(folder)
    log_count = len(logs)

    if log_count == 0:
        return failure_counts  # 防止除零错误

    for log in logs:
        place_data = log.get("place", {})
        for ftype in failure_types:
            failure_counts[ftype] += place_data.get(ftype, 0)

    # 计算每种失败的平均值
    failure_averages = {ftype: failure_counts[ftype] / log_count for ftype in failure_types}

    return failure_averages


# 生成对比图的标题和题注
def create_plot_title(structure_name, algorithm1, algorithm2):
    title = f"Failure Comparison between {algorithm1} and {algorithm2}"
    caption = f"Comparison of failure types for {structure_name} using {algorithm1} vs {algorithm2}."
    return title, caption


# 主程序
def main():
    # 根文件夹路径
    root_folder = "/home/jeong/summer_research/eth/husky_assembly/scripts/symbolic_planner/logs/one_tet_MT_contact-trac-vs-pinocchio"

    # 提取结构名和算法名
    structure_name, algorithm1, algorithm2 = extract_folder_info(root_folder)

    # 读取子文件夹路径
    folder_algorithm1 = os.path.join(root_folder, algorithm1)
    folder_algorithm2 = os.path.join(root_folder, algorithm2)

    # 提取失败数据
    failures_algorithm1 = extract_failures(folder_algorithm1)
    failures_algorithm2 = extract_failures(folder_algorithm2)

    # 生成图表标题和题注
    plot_title, plot_caption = create_plot_title(structure_name, algorithm1, algorithm2)

    # 绘制对比图
    failure_types = list(failures_algorithm1.keys())
    values_algorithm1 = [failures_algorithm1[ftype] for ftype in failure_types]
    values_algorithm2 = [failures_algorithm2[ftype] for ftype in failure_types]

    x = np.arange(len(failure_types))  # 失败类型的数量

    # 绘制柱状图
    width = 0.35  # 柱的宽度
    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width / 2, values_algorithm1, width, label=algorithm1)
    rects2 = ax.bar(x + width / 2, values_algorithm2, width, label=algorithm2)

    # 添加文本标签、标题和图例
    ax.set_ylabel("Failure Counts")
    ax.set_title(plot_title)
    ax.set_xticks(x)
    ax.set_xticklabels(failure_types, rotation=45, ha="right")
    ax.legend()

    # 添加图注
    plt.figtext(0.5, -0.05, plot_caption, wrap=True, horizontalalignment="center", fontsize=12)

    # 自动显示柱状图顶部的数值
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.show()


# 执行主程序
if __name__ == "__main__":
    main()
