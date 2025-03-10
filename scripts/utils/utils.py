import json
import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from typing import Dict, List, Set, Tuple, Union
import random

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pybullet_planning as pp
from pybullet_planning import Attachment
from pybullet_planning.utils import CIRCULAR_LIMITS, DEFAULT_RESOLUTION, MAX_DISTANCE
from termcolor import colored, cprint
from utils.params import PROJECT_DIR

HUSKYU_JOINT_NAMES = [
    "ur_arm_shoulder_pan_joint",
    "ur_arm_shoulder_lift_joint",
    "ur_arm_elbow_joint",
    "ur_arm_wrist_1_joint",
    "ur_arm_wrist_2_joint",
    "ur_arm_wrist_3_joint",
]


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


def normalize_angles(angles):
    for i in range(len(angles)):
        angles[i] = normalize_angle(angles[i])
    return angles


def normalize_angle(angle):
    angle = np.fmod(angle + np.pi, 2 * np.pi)
    if angle <= 0:
        return angle + np.pi
    else:
        return angle - np.pi


def angles_distance(angles1, angles2):
    diff = angles1 - angles2
    diff = normalize_angles(diff)
    return np.linalg.norm(diff)


def angle_distance(angle1, angle2):
    diff = angle1 - angle2
    diff = normalize_angle(diff)
    return diff


###########################################


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
        data_to_save = {
            name: {value.name: value.value for value in module.values.values()} for name, module in self.modules.items()
        }
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


def closest_points_between_segments(
    seg1: List[Union[List[float], np.ndarray]], seg2: List[Union[List[float], np.ndarray]]
) -> Tuple[List[float], List[float]]:
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
