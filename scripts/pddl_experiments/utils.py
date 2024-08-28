import logging
import os
from functools import partial

import load_pddlstream
import numpy as np
import pybullet_planning as pp
from load_pddlstream import HERE
from pddlstream.algorithms.algorithm import parse_problem
from pddlstream.algorithms.downward import get_problem, task_from_domain_problem
from pddlstream.language.constants import Action, DurativeAction, FunctionAction, StreamAction, is_plan
from pddlstream.language.conversion import obj_from_pddl
from pddlstream.utils import str_from_object
from termcolor import colored

HUSKYU_JOINT_NAMES = [
    "ur_arm_shoulder_pan_joint",
    "ur_arm_shoulder_lift_joint",
    "ur_arm_elbow_joint",
    "ur_arm_wrist_1_joint",
    "ur_arm_wrist_2_joint",
    "ur_arm_wrist_3_joint",
]


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


def plan_transit_motion(
    robot, end_conf, attachments, obstacles, debug=False, disabled_collisions=None, coarse_waypoints=False
):
    custom_limits = get_custom_limits(robot, {})
    resolutions = np.ones(6) * 0.05
    disabled_collisions = disabled_collisions or {}
    extra_disabled_collisions = [
        ((robot, pp.link_from_name(robot, "ur_arm_wrist_3_link")), (attachments[0].child, pp.BASE_LINK)),
    ]

    movable_joints = pp.joints_from_names(robot, HUSKYU_JOINT_NAMES)
    sample_fn = pp.get_sample_fn(robot, movable_joints, custom_limits=custom_limits)
    distance_fn = pp.get_distance_fn(robot, movable_joints)  # , weights=weights)
    extend_fn = pp.get_extend_fn(robot, movable_joints, resolutions=resolutions)

    transit_collision_fn = pp.get_collision_fn(
        robot,
        movable_joints,
        obstacles=obstacles,
        attachments=attachments,
        self_collisions=1,
        disabled_collisions=disabled_collisions,
        extra_disabled_collisions=extra_disabled_collisions,
        custom_limits=custom_limits,
        max_distance=0.01,
    )

    transit_path = None
    with pp.WorldSaver():
        with pp.LockRenderer(True):
            # * plan transit motion from current conf to pregrasp conf
            start_conf = pp.get_joint_positions(robot, movable_joints)
            # print('start conf: ', start_conf)

            # new_collision_fn = lambda q, diagnosis=False: collision_fn(q, diagnosis=True)
            if pp.check_initial_end(start_conf, end_conf, transit_collision_fn, diagnosis=debug):
                transit_path = pp.solve_motion_plan(
                    start_conf,
                    end_conf,
                    distance_fn,
                    sample_fn,
                    extend_fn,
                    transit_collision_fn,
                    algorithm="birrt",
                    max_time=20,
                    max_iterations=30,
                    smooth=20,
                    diagnosis=debug,
                    coarse_waypoints=coarse_waypoints,
                )
            else:
                notify("initial and end conf not valid")
            if transit_path is None:
                notify("transit path not found")
            else:
                notify("transit path found: transit {} pts".format(len(transit_path)))

    return transit_path


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
