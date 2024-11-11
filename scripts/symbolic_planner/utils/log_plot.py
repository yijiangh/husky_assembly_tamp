import json
import os

import matplotlib.pyplot as plt
import numpy as np


# 提取主文件夹信息，识别结构名和compare模块
def extract_folder_info(folder_path):
    folder_name = os.path.basename(folder_path)

    # structure_name从第一个'-'之前提取
    structure_name = folder_name.split("-")[0]

    # compare模块从第一个'-'之后提取
    compare_module = folder_name.split("-")[1]

    return structure_name, compare_module


# 读取json文件并返回数据
def load_json_logs(folder):
    logs = []
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                logs.append(data)
    return logs


# 计算每个文件夹中的失败计数平均值
def extract_failures(folder):
    types_all = [["place failure", "post attach failure", "back failure"], ["pick failure"], ["transfer failure"], ["back failure"], ["total time"]]
    modules_all = ["place", "pick", "transfer", "back", "others"]

    counts = {}
    counts_raw = {}
    for module, types in zip(modules_all, types_all):
        counts.update({module + "/" + ftype: 0 for ftype in types})
        counts_raw.update({module + "/" + ftype: [] for ftype in types})

    log_count = 0

    # 读取所有日志文件
    logs = load_json_logs(folder)
    log_count = len(logs)

    if log_count == 0:
        return counts, counts_raw  # 防止除零错误

    for log in logs:
        for module, types in zip(modules_all, types_all):
            data = log.get(module, {})
            for ftype in types:
                counts[module + "/" + ftype] += data.get(ftype, 0)
                counts_raw[module + "/" + ftype].append(data.get(ftype, 0))

    # 计算每种失败的平均值
    averages = {ftype: counts[ftype] / log_count for ftype in counts.keys()}

    return averages, counts_raw


# 生成对比图的标题和题注
def create_plot_title(structure_name, compare_module, algorithms):
    algorithm_list = ", ".join(algorithms)
    title = f"Comparison (Averages) using {structure_name} for {compare_module} module"
    # caption = f'Comparison of average failure types for {structure_name} in the {compare_module} module using {algorithm_list}.'
    caption = ""
    return title, caption


# 主程序
def main():
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    log_name = "triangle_reciprocal_MT_contact-assembled_in_place"
    # 根文件夹路径
    root_folder = os.path.join(log_dir, log_name)

    # 提取结构名和compare模块
    structure_name, compare_module = extract_folder_info(root_folder)

    # 获取所有子文件夹（算法）
    algorithms = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    algorithms.sort()

    # 提取所有算法的失败数据
    data_avg = {}
    data_raw = {}
    for algorithm in algorithms:
        folder_algorithm = os.path.join(root_folder, algorithm)
        data_avg[algorithm], data_raw[algorithm] = extract_failures(folder_algorithm)

    # 生成图表标题和题注
    plot_title, plot_caption = create_plot_title(structure_name, compare_module, algorithms)

    # 绘制对比图
    names = list(data_avg[algorithms[0]].keys())
    x = np.arange(len(names))  # 失败类型的数量

    width = 0.8 / len(algorithms)  # 动态设置柱的宽度
    fig, ax = plt.subplots(figsize=(18, 12))

    for i, algorithm in enumerate(algorithms):
        values = [data_avg[algorithm][ftype] for ftype in names]
        values_raw = [data_raw[algorithm][ftype] for ftype in names]
        rects = ax.bar(x + i * width - (len(algorithms) - 1) * width / 2, values, width, label=algorithm)

        # 自动显示柱状图顶部的数值
        def autolabel(rects, values_raw):
            for rect, values in zip(rects, values_raw):
                height = rect.get_height()
                ax.annotate(
                    f"{height:.2f}",  # 保留两位小数
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )
                ax.annotate(
                    f"{len(values)}",  # 保留两位小数
                    xy=(rect.get_x() + rect.get_width() / 2, height / 2),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )
                for value in values:
                    ax.plot(rect.get_x() + rect.get_width() / 2, value, "o", color="black")

        autolabel(rects, values_raw)

    # 添加文本标签、标题和图例
    ax.set_ylabel("Average Failure Counts")
    ax.set_title(plot_title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()

    # 添加图注
    plt.figtext(0.5, -0.05, plot_caption, wrap=True, horizontalalignment="center", fontsize=12)

    plt.tight_layout()

    fig_name = "result.png"
    fig_path = os.path.join(root_folder, fig_name)
    plt.savefig(fig_path)

    plt.show()


# 执行主程序
if __name__ == "__main__":
    main()
