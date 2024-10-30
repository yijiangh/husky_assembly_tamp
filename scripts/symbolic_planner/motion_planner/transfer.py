import os
import sys
from typing import Callable, Dict, List, Set, Tuple, Union

import numpy as np
import pybullet_planning as pp
from termcolor import cprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pybullet_planning import Attachment
from robot.robot_setup import RobotSetup
from utils.collision import Element
from utils.utils import CounterModule, normalize_angles, get_custom_limits

# collision check threshold
MAX_DISTANCE = 0.0

# collision check enable
ENABLE_SELF_COLLISIONS = True


def compute_transfer_path(
    robot_setup: RobotSetup,
    start_conf: np.ndarray,
    target_conf: np.ndarray,
    grasp_attachment: Attachment,
    obstacles: Set[int],
    unassambled_element_obstacles: Set[int],
    counter: CounterModule = None,
    path_plan_max_attempt: int = 1,
    verbose: bool = False,
    diagnosis: bool = False,
    teleops: bool = False,
) -> Tuple[List[np.ndarray], List[bool]]:
    """
    Compute transfer path (mobile manipulator conf): pre_pick_pose --> pregrasp.

    Params:
        robot_setup (RobotSetup): RobotSetup instance
        start_conf (np.ndarray): start conf of robot
        target_conf (np.ndarray): target conf of robot
        grasp_attachment (Attachment): attachment of grasping current element
        obstacles (Set[index]): fixed obstacles + assembled elements
        unassambled_element_obstacles (Set[index], [not used]): (not used)
        counter (CounterModule | None, None): counter module to count failures
        path_plan_max_attempt (int, 1): number of attempts for manipulator path planner
        verbose (bool, False): whether print debug information
        diagnosis (bool, False): whether stop and display it in pybullet if a collision is detected
        teleops (bool, False, [not used]): (not used) whether use interpolation or path plan to fill the middle point

    Returns:
        command ([np.ndarray]): pick path (mobile manipulator conf): pre_pick_pose --> pregrasp
        mask ([bool]): whether to attach the current element to the gripper
    """

    # -------------------- init counter module --------------------#
    transfer_plan_val = counter.add_counter_value("transfer plan failure")
    transfer_collision_val = counter.add_counter_value("transfer collision failure")
    transfer_val = counter.add_counter_value("transfer failure")

    # -------------------- init collision checker --------------------#
    extra_disabled_collisions = [
        (
            (robot_setup.robot, pp.link_from_name(robot_setup.robot, "ur_arm_wrist_3_link")),
            (robot_setup.ee_attachment.child, pp.BASE_LINK),
        ),
    ]
    collision_fn = pp.get_collision_fn(
        robot_setup.robot,
        robot_setup.arm_joints,
        obstacles=obstacles,
        attachments=[robot_setup.ee_attachment, grasp_attachment] + robot_setup.attachments,
        self_collisions=ENABLE_SELF_COLLISIONS,
        disabled_collisions=robot_setup.disabled_collisions,
        extra_disabled_collisions=extra_disabled_collisions,
        custom_limits=get_custom_limits(robot_setup.robot, {}),
        max_distance=MAX_DISTANCE,
    )

    # -------------------- check base conf --------------------#
    start_base_conf = start_conf[:3]
    target_base_conf = target_conf[:3]
    if np.linalg.norm(start_base_conf - target_base_conf) >= 0.1:
        raise RuntimeError("start pose of base must euqal to target pose of base!")

    # -------------------- get start and target conf --------------------#
    start_arm_conf = start_conf[3:]
    target_arm_conf = target_conf[3:]

    # -------------------- loop: compute transfer path --------------------#
    for _ in range(path_plan_max_attempt):

        # -------------------- assign attachment --------------------#
        grasp_attachment.assign()

        # -------------------- plan --------------------#
        transfer_path = robot_setup.plan_manipulator_path(
            start_arm_conf,
            target_arm_conf,
            [grasp_attachment] + robot_setup.attachments,
            obstacles,
        )
        if transfer_path is None:
            if verbose:
                cprint("transfer plan failure", "red")
            transfer_plan_val.increment()
            transfer_val.increment()
            continue

        transfer_path = [normalize_angles(conf) for conf in transfer_path]
        transfer_confs = [np.hstack((start_base_conf, conf)) for conf in transfer_path]

        # -------------------- collision check --------------------#
        fail_flag = False
        for transfer_conf in transfer_confs:
            if collision_fn(transfer_conf[3:], diagnosis):
                if verbose:
                    cprint("transfer collision failure", "red")
                transfer_collision_val.increment()
                transfer_val.increment()
                fail_flag = True
                break
        if fail_flag:
            continue

        return transfer_confs, [True] * len(transfer_confs)

    return None, None


def get_transfer_gen_fn(
    robot_setup: RobotSetup,
    element_from_index: Dict,
    fixed_obstacles: List[int],
    max_attempts: int = 10,
    collisions: bool = True,
    allow_failure: bool = False,
    verbose: bool = False,
    teleops: bool = False,
) -> Callable[
    [
        int,
        np.ndarray,
        np.ndarray,
        Attachment,
        List[int],
        List[int],
        List[Attachment],
        Union[CounterModule, None],
        bool,
    ],
    Tuple[List[np.ndarray], List[bool]],
]:
    """
    Generate transfer motion planner function.

    Params:
        robot_setup (RobotSetup): RobotSetup instance
        element_from_index ({index: Element}): element dict
        fixed_obstacles ([int]): list of id in pybullet
        max_attempts (int, 10): attempts to generate transfer motion
        collisions (bool, True): whether consider collision
        allow_failure (bool, False): yield (None * 2), False: return (None * 2) and raise an error
        verbose (bool, False): whether print debug information
        teleops (bool, False): whether to interpolate the intermediate paths

    Returns:
        Callable: gen_fn(index, start_conf, tar_conf, grasp_attachment, assembled, unassembled, attachments, counter, diagnosis)
    """

    def gen_fn(
        index: int,
        start_conf: np.ndarray,
        target_conf: np.ndarray,
        grasp_attachment: Attachment,
        assembled: List[int] = [],
        unassembled: List[int] = [],
        attachments: List[Attachment] = [],
        counter: Union[CounterModule, None] = None,
        diagnosis: bool = False,
    ):
        """
        Generate transfer motion and return path.

        Params:
            index (int): the index of element that needs to assemble
            start_conf (np.ndarray): start conf of robot
            target_conf (np.ndarray): target conf of robot
            grasp_attachment (Attachment): attachment of grasping current element
            assembled ([int], []): indices of assembled elements
            unassembled ([int], []): indices of unassembled elements (excluding the current element)
            attachments ([Attachment], []): list of attachments bound to the robot (excluding the current element)
            counter (CounterModule, None): counter module
            diagnosis (bool, False): whether stop and display it in pybullet if a collision is detected

        Returns:
            command ([np.ndarray]): robot confs
            grasp_mask ([bool]): whether to attach the current element to the gripper
        """
        cur_element: Element = element_from_index[index]

        # -------------------- update current attachments --------------------#
        robot_setup.update_attachments(attachments)

        # -------------------- set obstacles --------------------#
        element_obstacles = set({element_from_index[e].body for e in list(assembled)})
        # unassambled_element_obstacles = set({element_from_index[e].body for e in list(unassembled)})

        obstacles = set(fixed_obstacles) | element_obstacles
        if not collisions:
            obstacles = set()

        # -------------------- loop: plan transfer motion --------------------#
        for attempt in range(max_attempts):
            if verbose:
                cprint(f"transfer attempt: {attempt}", "yellow")

            # -------------------- compute transfer path --------------------#
            command, mask = compute_transfer_path(
                robot_setup,
                start_conf,
                target_conf,
                grasp_attachment,
                obstacles,
                None,
                counter=counter,
                verbose=verbose,
                diagnosis=diagnosis,
                teleops=teleops,
            )
            if command is None:
                continue

            cprint("Transfer E#{} | Attempts: {} | Command: {}".format(index, attempt, len(command)), "green")

            yield command, mask
            break

        # -------------------- fail --------------------#
        cprint("Transfer E#{} | Attempts: {} | Max attempts exceeded!".format(index, max_attempts), "red")

        if allow_failure:
            yield None, None
        else:
            return None, None

    return gen_fn
