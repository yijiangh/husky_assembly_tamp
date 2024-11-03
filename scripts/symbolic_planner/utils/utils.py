import json
import logging
import os
import time
from collections import defaultdict
from functools import partial
from typing import Dict, List, Set, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pybullet_planning as pp
import utils.load_pddlstream as load_pddlstream
from pddlstream.algorithms.algorithm import parse_problem
from pddlstream.algorithms.downward import get_problem, task_from_domain_problem
from pddlstream.language.constants import Action, DurativeAction, FunctionAction, StreamAction, is_plan
from pddlstream.language.conversion import obj_from_pddl
from pddlstream.utils import str_from_object
from pybullet_planning import Attachment
from pybullet_planning.utils import CIRCULAR_LIMITS, DEFAULT_RESOLUTION, MAX_DISTANCE
from termcolor import colored, cprint
from utils.load_pddlstream import HERE

HUSKYU_JOINT_NAMES = [
    "ur_arm_shoulder_pan_joint",
    "ur_arm_shoulder_lift_joint",
    "ur_arm_elbow_joint",
    "ur_arm_wrist_1_joint",
    "ur_arm_wrist_2_joint",
    "ur_arm_wrist_3_joint",
]

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


###########################################
# borrowed from: https://github.com/compas-dev/compas_fab/blob/3efe608c07dc5b08653ee4132a780a3be9fb93af/src/compas_fab/backends/pybullet/utils.py#L83
def get_logger(name):
    logger = logging.getLogger(name)

    try:
        from colorlog import ColoredFormatter

        formatter = ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red",
            },
        )
    except ImportError:
        formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


def notify(msg):
    print(msg)


LOGGER = get_logger("robarch_pddl")

###########################################


def print_pddl_task_object_names(pddl_problem):
    evaluations, goal_exp, domain, externals = parse_problem(pddl_problem, unit_costs=True)
    problem = get_problem(evaluations, goal_exp, domain, unit_costs=True)
    task = task_from_domain_problem(domain, problem)
    LOGGER.debug("=" * 10)
    for task_obj, pddl_object in sorted(
        zip(task.objects, map(lambda x: obj_from_pddl(x.name), task.objects)),
        key=lambda x: int(x[0].name.split("v")[1]),
    ):
        LOGGER.debug("{} : {}".format(task_obj.name, colored_str_from_object(pddl_object.value)))
    LOGGER.debug("=" * 10)


def contains_number(value):
    for character in value:
        if character.isdigit():
            return True
    return False


def colored_str_from_object(obj, show_details=False):
    if not show_details:
        # if isinstance(obj, Frame):
        #     return '(frm)'
        # elif isinstance(obj, Transformation):
        #     return '(tf)'
        # elif isinstance(obj, Configuration):
        #     return colored('(conf)', 'yellow')
        if isinstance(obj, Action):
            return colored(obj, "yellow")

    str_rep = str_from_object(obj)
    if contains_number(str_rep):
        return colored(str_rep, "blue")
    else:
        return colored(str_rep, "red")


def print_itj_pddl_plan(plan, show_details=False):
    if not is_plan(plan):
        return
    step = 1
    color_print_fn = partial(colored_str_from_object, show_details=show_details)
    for action in plan:
        if isinstance(action, DurativeAction):
            name, args, start, duration = action
            LOGGER.info(
                "{:.2f} - {:.2f}) {} {}".format(start, start + duration, name, " ".join(map(str_from_object, args)))
            )
        elif isinstance(action, Action):
            name, args = action
            LOGGER.info("{:2}) {} {}".format(step, colored(name, "green"), " ".join(map(color_print_fn, args))))
            step += 1
        elif isinstance(action, StreamAction):
            name, inputs, outputs = action
            LOGGER.info(
                "    {}({})->({})".format(
                    name, ", ".join(map(str_from_object, inputs)), ", ".join(map(str_from_object, outputs))
                )
            )
        elif isinstance(action, FunctionAction):
            name, inputs = action
            LOGGER.info("    {}({})".format(name, ", ".join(map(str_from_object, inputs))))
        else:
            raise NotImplementedError(action)


def pddl_plan_to_string(plan):
    plan_string_lines = []
    step = 1
    for action in plan:
        if isinstance(action, DurativeAction):
            name, args, start, duration = action
            plan_string_lines.append(
                "{:.2f} - {:.2f}) {} {}".format(start, start + duration, name, " ".join(map(str_from_object, args)))
            )
        elif isinstance(action, Action):
            name, args = action
            plan_string_lines.append("{:3}: {} {}".format(step, name, " ".join(map(str_from_object, args))))
            step += 1
        elif isinstance(action, StreamAction):
            name, inputs, outputs = action
            plan_string_lines.append(
                "    {}({})->({})".format(
                    name, ", ".join(map(str_from_object, inputs)), ", ".join(map(str_from_object, outputs))
                )
            )
        elif isinstance(action, FunctionAction):
            name, inputs = action
            plan_string_lines.append("    {}({})".format(name, ", ".join(map(str_from_object, inputs))))
        else:
            raise NotImplementedError(action)
    return plan_string_lines


def pddl_plan_to_dict(plan):
    seq_n = 0  # Increment after the assembly of each beam
    act_n = 0  # Increment after every action , resets after new beam
    actions = []
    for action in plan:
        if isinstance(action, Action):
            action_name, args = action
            actions.append({"act_n": act_n, "action_name": action_name, "args": args})
            act_n += 1
    return actions

    # sequence = {'seq_n': seq_n, 'actions': []}
    # for action in plan:
    #     if isinstance(action, Action):
    #         action_name, args = action
    #         sequence['actions'].append({'act_n': act_n, 'action_name': action_name, 'args': args})
    #         act_n += 1
    #         if action_name.startswith('assemble_beam'):
    #             seq_n += 1
    #             act_n = 0
    #             sequence['beam_id'] = args[0]
    #             sequences.append(sequence)
    #             sequence = {'seq_n': seq_n, 'actions': []}
    # if len(sequence['actions']) > 0:
    #     sequences[-1]['actions'].extend(sequence['actions'])
    # return sequences


def save_plan_text(plan, pddl_folder, file_name):
    # Create folder if not exists
    if not os.path.exists(pddl_folder):
        os.makedirs(pddl_folder)

    # Save plan to file
    file_output_path = os.path.join(HERE, pddl_folder, file_name)
    with open(file_output_path, "w") as f:
        for line in pddl_plan_to_string(plan):
            f.write(line + "\n")


def save_plan_dict(plan, pddl_folder, file_name):
    # Create folder if not exists
    if not os.path.exists(pddl_folder):
        os.makedirs(pddl_folder)

    # Save plan to file
    file_output_path = os.path.join(HERE, pddl_folder, file_name)
    action_dict = pddl_plan_to_dict(plan)
    import json

    from compas.data import DataEncoder

    with open(file_output_path, "w") as f:
        json.dump(action_dict, f, indent=4, cls=DataEncoder)


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
        parent.values[name] = self

    def increment(self, value=1):
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
