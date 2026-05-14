import json
import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from typing import Dict, List, Set, Tuple, Union
import random

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pybullet_planning as pp
from pybullet_planning import Attachment
from pybullet_planning.utils import CIRCULAR_LIMITS, DEFAULT_RESOLUTION, MAX_DISTANCE
from termcolor import colored, cprint
from .params import PROJECT_DIR, LOG_DIR


# ---------------------------------------------------------------------------
# Colorful Logging Setup
# ---------------------------------------------------------------------------

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',      # Reset
        'BOLD': '\033[1m',       # Bold
        'DIM': '\033[2m',        # Dim
    }
    
    def format(self, record):
        # Add color based on log level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        dim = self.COLORS['DIM']
        
        # Format timestamp in dim
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        
        # Build the colored message
        formatted = f"{dim}[{timestamp}]{reset} {color}{record.levelname:8}{reset} {record.getMessage()}"
        return formatted


class FileFormatter(logging.Formatter):
    """Plain formatter for file output (no ANSI codes)."""
    
    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        return f"[{timestamp}] {record.levelname:8} {record.getMessage()}"


def setup_logger(name: str = "husky_assembly", log_dir: str = None, level: int = logging.DEBUG, file_mode: str = "a") -> logging.Logger:
    """Set up logger with both console (colored) and file handlers.

    Args:
        name: Logger name (used for both the logger and log filename prefix)
        log_dir: Directory for log files. Defaults to LOG_DIR from params.
        level: Logging level. Defaults to DEBUG.
        file_mode: FileHandler mode. 'a' to append (default), 'w' to overwrite each run.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir is None:
        log_dir = LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    
    # log_filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filename = f"{name}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    file_handler = logging.FileHandler(log_path, mode=file_mode, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(FileFormatter())
    logger.addHandler(file_handler)
    
    logger.info(f"Log file: {log_path}")
    
    return logger


def reinit_logger_stream(logger: logging.Logger, stream=None) -> None:
    """Reinitialize the console handler's stream (useful after pybullet GUI on Windows).
    
    Args:
        logger: Logger instance to update.
        stream: New stream to use. Defaults to sys.stdout.
    """
    if stream is None:
        stream = sys.stdout
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.stream = stream


###########################################


def get_custom_limits(robot, custom_limits=None):
    """[summary]

    Returns
    -------
    [type]
        {joint index : (lower limit, upper limit)}
    """
    custom_limits = custom_limits or {}
    limits = {pp.joint_from_name(robot, joint): limits for joint, limits in custom_limits.items()}
    return limits


###########################################


def normalize_angles(angles, low: float = -np.pi, high: float = np.pi):
    """
    Normalize an iterable of angles to the range (low, high].

    Supports tuple, list, and np.ndarray. The return type matches the input type.
    """
    span = high - low
    if not np.isfinite(span) or span <= 0:
        raise ValueError("Invalid angle range: 'high' must be greater than 'low'.")

    array_like = np.asarray(angles, dtype=float)
    shifted = np.fmod(array_like - low, span)
    normalized = np.where(shifted <= 0, shifted + high, shifted + low)

    if isinstance(angles, np.ndarray):
        return normalized
    if isinstance(angles, tuple):
        return tuple(normalized.tolist())
    if isinstance(angles, list):
        return normalized.tolist()
    # Fallback: return numpy array if an unexpected type is provided
    return normalized


def normalize_angle(angle, low: float = -np.pi, high: float = np.pi):
    """Normalize a single angle to the range (low, high]."""
    span = high - low
    if not np.isfinite(span) or span <= 0:
        raise ValueError("Invalid angle range: 'high' must be greater than 'low'.")

    shifted = np.fmod(angle - low, span)
    if shifted <= 0:
        return shifted + high
    return shifted + low


def angles_distance(angles1, angles2):
    """
    Compute the Euclidean norm of directed circular differences between two angle vectors.

    The per-joint difference is the signed minimal angle delta δ such that
    normalize_angle(angle2 + δ) == normalize_angle(angle1).
    """
    a1 = np.asarray(angles1, dtype=float)
    a2 = np.asarray(angles2, dtype=float)
    diff = a2 - a1
    diff = normalize_angles(diff)
    return diff


def angle_distance(angle1, angle2):
    """Compute the signed minimal circular difference δ with normalize_angle(angle2 + δ) == normalize_angle(angle1)."""
    diff = angle1 - angle2
    return normalize_angle(diff)


def angle_between_unit_vectors(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    if u_norm > 0.0:
        u = u / u_norm
    if v_norm > 0.0:
        v = v / v_norm
    dot = float(np.dot(u, v))
    dot = max(-1.0, min(1.0, dot))
    return float(np.arccos(dot))


def calculate_pose_error(pose1, pose2) -> np.ndarray:
    R1 = pp.tform_from_pose(pose1)[:3, :3]
    R2 = pp.tform_from_pose(pose2)[:3, :3]
    p1 = np.array(pose1[0], dtype=float)
    p2 = np.array(pose2[0], dtype=float)
    position_err = p1 - p2
    x_err = angle_between_unit_vectors(R1[:, 0], R2[:, 0])
    y_err = angle_between_unit_vectors(R1[:, 1], R2[:, 1])
    z_err = angle_between_unit_vectors(R1[:, 2], R2[:, 2])
    return np.concatenate([position_err, np.array([x_err, y_err, z_err])])


###########################################


def interpolate(trajectory: np.ndarray, target_length: int) -> np.ndarray:
    """
    重新插值轨迹到指定长度，确保原始轨迹中的所有点都被保留

    Args:
        trajectory: 原始轨迹数据，形状为 [N, D]，其中 N 是时间步数，D 是每步的维度
        target_length: 目标轨迹长度

    Returns:
        np.ndarray: 重新插值后的轨迹，形状为 [target_length, D]
    """
    # 原始轨迹长度和维度
    orig_length, dims = trajectory.shape

    # 如果目标长度小于或等于原始长度，需要进行降采样
    if target_length <= orig_length:
        # 选择等间隔的点
        indices = np.round(np.linspace(0, orig_length - 1, target_length)).astype(int)
        return trajectory[indices]

    # 创建新轨迹数组，初始化为零
    new_trajectory = np.zeros((target_length, dims))

    # 首先确保原始轨迹中的所有点都被保留
    # 计算原始点在新轨迹中的索引
    orig_indices_in_new = np.round(np.linspace(0, target_length - 1, orig_length)).astype(int)

    # 将原始点放入新轨迹
    for i, idx in enumerate(orig_indices_in_new):
        new_trajectory[idx] = trajectory[i]

    # 创建掩码标记哪些位置已分配值
    mask = np.zeros(target_length, dtype=bool)
    mask[orig_indices_in_new] = True

    # 为没有分配值的位置创建插值
    for i in range(target_length):
        if not mask[i]:
            # 查找两侧最近的已知点
            left_idx = np.max(orig_indices_in_new[orig_indices_in_new < i]) if any(orig_indices_in_new < i) else 0
            right_idx = np.min(orig_indices_in_new[orig_indices_in_new > i]) if any(orig_indices_in_new > i) else target_length - 1

            # 如果左右索引相同，无法进行插值，使用最近点
            if left_idx == right_idx:
                new_trajectory[i] = new_trajectory[left_idx]
                continue

            # 计算插值权重
            left_orig_idx = np.where(orig_indices_in_new == left_idx)[0][0]
            right_orig_idx = np.where(orig_indices_in_new == right_idx)[0][0]

            weight = (i - left_idx) / (right_idx - left_idx)

            # 线性插值
            new_trajectory[i] = (1 - weight) * trajectory[left_orig_idx] + weight * trajectory[right_orig_idx]

    return new_trajectory


class CounterValue:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.value = 0
        self.last_update = 0
        parent.values[name] = self

    def increment(self, value=1):
        self.last_update = value
        self.value += value

    def update(self, value):
        self.value = value


class CounterModule:
    def __init__(self, root=None, name=None, parent=None):
        if root is None:
            self.modules = {}
            self.name = "root"
        else:
            self.modules = root.modules

        self.name = name
        self.parent = parent
        self.children = []
        self.values = {}

        if name is not None:
            if name not in self.modules:
                self.modules[name] = self
            if parent:
                parent.children.append(self)

    def create_handle(self, name):
        return CounterModule(root=self, name=name, parent=self)

    def add_counter_value(self, name):
        if name in self.values:
            return self.values[name]
        else:
            counter_value = CounterValue(name, self)
            return counter_value

    def plot(self):
        def collect_data(module, collected=None):
            if collected is None:
                collected = defaultdict(lambda: defaultdict(int))
            if module.name is not None:
                for value in module.values:
                    collected[module.name][value.name] += value.value
            for child in module.children:
                collect_data(child, collected)
            return collected

        data = collect_data(self)

        handles = list(data.keys())
        value_labels = list(set(vname for handle_values in data.values() for vname in handle_values.keys()))

        # 获取颜色映射
        color_map = dict(zip(value_labels, plt.cm.viridis(np.linspace(0, 1, len(value_labels)))))

        bar_width = 0.8 / len(value_labels)
        index = np.arange(len(handles))

        plt.figure(figsize=(10, 5))

        for i, value_name in enumerate(value_labels):
            heights = [data[handle].get(value_name, 0) for handle in handles]
            bar_positions = index + i * bar_width
            plt.bar(bar_positions, heights, bar_width, color=color_map[value_name], label=value_name)

            for j, height in enumerate(heights):
                if height > 0:
                    plt.text(bar_positions[j], height + 0.1, str(height), ha="center")

        plt.xlabel("Handle Names")
        plt.ylabel("Counts")
        plt.title("Counter Module with Side-by-Side Values per Handle")
        plt.xticks(index + bar_width * (len(value_labels) - 1) / 2, handles)
        plt.legend(title="Value Names")
        plt.show()

    def reset(self):
        for module in self.modules.values():
            for value in module.values:
                value.value = 0

    def save(self, path, filename):
        data_to_save = {name: {value.name: value.value for value in module.values.values()} for name, module in self.modules.items()}
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, filename)
        with open(file_path, "w") as file:
            json.dump(data_to_save, file)
        print(f"Counter values saved to {filename}")


class TermPrint(object):
    last_empty_line = False

    def __init__(self) -> None:
        pass

    @classmethod
    def print(cls, text: str, color: str = "white", blank_f: bool = False, blank_b: bool = False):
        if blank_f and not cls.last_empty_line:
            print("")
        cprint(text, color)
        if blank_b:
            print("")
            cls.last_empty_line = True
        else:
            cls.last_empty_line = False


class PrintManager:
    """打印控制模块，用于统一管理终端输出并控制缩进级别"""

    # 预定义的颜色映射，便于使用不同颜色打印不同类型的消息
    COLORS = {"info": "white", "success": "green", "warning": "yellow", "error": "red", "debug": "cyan", "highlight": "magenta"}

    # 单例模式，确保只有一个打印管理器实例
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PrintManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, indent_size: int = 4, tab_char: str = " ", use_color: bool = True, default_color: str = "white"):
        """
        初始化打印管理器

        Args:
            indent_size: 每级缩进的空格数
            tab_char: 缩进使用的字符
            use_color: 是否使用彩色输出
            default_color: 默认输出颜色
        """
        # 避免重复初始化
        if hasattr(self, "indent_level"):
            return

        self.indent_size = indent_size
        self.tab_char = tab_char
        self.indent_level = 0
        self.use_color = use_color
        self.default_color = default_color

    def _get_indent(self, level: int = None) -> str:
        """
        获取当前缩进级别下的缩进字符串

        Args:
            level: 指定的缩进级别，若不指定则使用当前级别

        Returns:
            缩进字符串
        """
        if level is None:
            level = self.indent_level
        return self.tab_char * self.indent_size * level

    def print(self, message: str, indent_level: int = None, color: str = None, end: str = "\n", flush: bool = True):
        """
        按照指定的缩进级别打印消息

        Args:
            message: 要打印的消息
            indent_level: 指定的缩进级别，若不指定则使用当前级别
            color: 指定的颜色，若不指定则使用默认颜色
            end: 行结束符
            flush: 是否立即刷新输出
        """
        if indent_level is None:
            indent_level = self.indent_level

        if color is None:
            color = self.default_color

        indent_str = self._get_indent(indent_level)
        formatted_message = f"{indent_str}{message}"

        if self.use_color:
            cprint(formatted_message, color, end=end, flush=flush)
        else:
            print(formatted_message, end=end, flush=flush)

    def info(self, message: str, indent_level: int = None):
        """打印普通信息"""
        self.print(message, indent_level, self.COLORS["info"])

    def success(self, message: str, indent_level: int = None):
        """打印成功信息"""
        self.print(message, indent_level, self.COLORS["success"])

    def warning(self, message: str, indent_level: int = None):
        """打印警告信息"""
        self.print(message, indent_level, self.COLORS["warning"])

    def error(self, message: str, indent_level: int = None):
        """打印错误信息"""
        self.print(message, indent_level, self.COLORS["error"])

    def debug(self, message: str, indent_level: int = None):
        """打印调试信息"""
        self.print(message, indent_level, self.COLORS["debug"])

    def highlight(self, message: str, indent_level: int = None):
        """打印高亮信息"""
        self.print(message, indent_level, self.COLORS["highlight"])

    def indent(self, levels: int = 1):
        """增加缩进级别"""
        self.indent_level += levels
        return self

    def dedent(self, levels: int = 1):
        """减少缩进级别"""
        self.indent_level = max(0, self.indent_level - levels)
        return self

    def reset_indent(self):
        """重置缩进级别为0"""
        self.indent_level = 0
        return self

    def set_indent(self, level: int):
        """直接设置缩进级别"""
        self.indent_level = max(0, level)
        return self

    @contextmanager
    def indented(self, levels: int = 1):
        """
        临时增加缩进级别的上下文管理器

        Args:
            levels: 增加的缩进级别数量

        示例:
            printer = PrintManager()
            printer.info("主层级消息")
            with printer.indented():
                printer.info("缩进一级的消息")
                with printer.indented(2):
                    printer.info("缩进三级的消息")
            printer.info("回到主层级")
        """
        self.indent(levels)
        try:
            yield self
        finally:
            self.dedent(levels)


def flatten(nested_list):
    result = []
    for element in nested_list:
        if isinstance(element, list):
            result.extend(flatten(element))
        else:
            result.append(element)
    return result


def timeit_decorator_counter(counter_name: str = "", verbose: bool = False, output_time: bool = False):

    def timeit_decorator(func):
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()

            if verbose:
                cprint(
                    f"\n============================================================\nFunction '{func.__name__}' executed in {end_time - start_time:.6f} seconds!\n============================================================\n",
                    "cyan",
                )

            if counter_name != "":
                others_handle: CounterModule = getattr(self, counter_name)
                time_handle: CounterValue = others_handle.add_counter_value("total time")
                time_handle.increment(end_time - start_time)

            if output_time:
                return result, end_time - start_time
            else:
                return result

        return wrapper

    return timeit_decorator


def closest_points_between_segments(seg1: List[Union[List[float], np.ndarray]], seg2: List[Union[List[float], np.ndarray]]) -> Tuple[List[float], List[float]]:
    """
    Calculate the endpoints of the common perpendicular line between two line segments.

    Params:
        seg1 ([[x1, y1, z1], [x2, y2, z2]]): segment 1
        seg2 ([[x1, y1, z1], [x2, y2, z2]]): segment 2

    Returns:
        [x1, y1, z1]: point on segment 1
        [x2, y2, z2]: point on segment 1

    """
    p1, q1 = np.array(seg1[0]), np.array(seg1[1])
    p2, q2 = np.array(seg2[0]), np.array(seg2[1])

    d1 = q1 - p1
    d2 = q2 - p2

    r = p1 - p2

    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, r)
    e = np.dot(d2, r)

    denom = a * c - b * b
    if denom == 0:
        raise ValueError("Segments are parallel!")

    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom

    s = np.clip(s, 0, 1)
    t = np.clip(t, 0, 1)

    closest_point_on_seg1: np.ndarray = p1 + s * d1
    closest_point_on_seg2: np.ndarray = p2 + t * d2

    return closest_point_on_seg1.tolist(), closest_point_on_seg2.tolist()


@contextmanager
def HideOutput():
    """
    A context manager to suppress all standard output (stdout) and standard error (stderr).
    """
    with open(os.devnull, "w") as fnull:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            sys.stdout = fnull  # 重定向标准输出
            sys.stderr = fnull  # 重定向标准错误
            yield
        finally:
            sys.stdout = original_stdout  # 恢复标准输出
            sys.stderr = original_stderr  # 恢复标准错误


def SetSeeds(seed=24):
    random.seed(seed)
    np.random.seed(seed)
    pp.set_numpy_seed(seed)
    pp.set_random_seed(seed)
