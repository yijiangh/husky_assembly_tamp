import copy
import os
import sys
from typing import Callable, Dict, List, Set, Tuple, Union

import numpy as np
import pybullet_planning as pp
from termcolor import cprint
from utils.params import *

sys.path.append(HERE)

from pybullet_planning import Attachment, interpolate_poses
from robot.robot_setup import HUSKY_INIT_ARM_JOINT_ANGLES, RobotSetup
from utils.collision import Element
from utils.util import CounterModule, TermPrint, angles_distance, normalize_angles

# place retreat
RETREAT_DISTANCE = 0.07

# pregrasp/retreat interpolation step
POS_STEP_SIZE = 0.005
ORI_STEP_SIZE = np.pi / 128

# collision check threshold
MAX_DISTANCE = 0

# collision check enable
ENABLE_SELF_COLLISIONS = True


def compute_pick_path(
    robot_setup: RobotSetup,
    index: int,
    grasp: Tuple[Tuple[float], Tuple[float]],
    element_from_index: Dict,
    obstacles: Set[int],
    unassambled_element_obstacles: Set[int],
    counter: Union[CounterModule, None] = None,
    retreat_dist: float = RETREAT_DISTANCE,
    ik_search_max_attempt: int = 1,
    path_plan_max_attempt: int = 1,
    verbose: bool = False,
    diagnosis: bool = False,
    teleops: bool = False,
) -> Tuple[List[np.ndarray], List[bool]]:
    """
    Compute pick path (mobile manipulator conf): home_pose --> pick_pose --> post_pick_pose.

    Params:
        robot_setup (RobotSetup): RobotSetup instance
        index (int): index of current element
        grasp (pp.Pose): grasp pose (gripper_from_body)
        element_from_index ({index: Element}): dict of elements
        obstacles (Set[index]): fixed obstacles + assembled elements
        unassambled_element_obstacles: pass
        counter (CounterModule | None, None): counter module to count failures
        retreat_dist (float, RETREAT_DISTANCE): retreat distance after attach_pose (goal_pose)
        ik_search_max_attempt (int, 1): number of attempts for searching ik solution
        path_plan_max_attempt (int, 1): number of attempts for manipulator path planner
        verbose (bool, False): whether print debug information
        diagnosis (bool, False): whether stop and display it in pybullet if a collision is detected
        teleops (bool, False, [not used]): whether use interpolation or path plan to fill the middle point

    Returns:
        command ([np.ndarray]): pick path (mobile manipulator conf): home_pose --> pick_pose --> post_pick_pose
        mask ([bool]): whether to attach the current element to the gripper
    """
    cur_element: Element = element_from_index[index]

    # -------------------- init counter module --------------------#
    pick_ik_val = counter.add_counter_value("pick ik failure")
    pick_collision_val = counter.add_counter_value("pick collision failure")

    pre_pick_ik_val = counter.add_counter_value("pre pick ik failure")
    # pre_pick_plan_val = counter.add_counter_value("pre pick plan failure")
    # pre_pick_collision_val = counter.add_counter_value("pre pick collision failure")
    pre_pick_val = counter.add_counter_value("pre pick failure")

    approach_val = counter.add_counter_value("approach failure")

    pick_val = counter.add_counter_value("pick failure")

    # -------------------- generate new grasp (gripper_from_body) --------------------#
    grasp_temp = pp.Pose([0, 0, 0], pp.Euler(roll=np.pi / 2, pitch=0, yaw=0))
    grasp = (grasp[0], grasp_temp[1])  # girpper_from_body

    # -------------------- init element pose --------------------#
    aboard_attachment = robot_setup.create_aboard_attachment(cur_element.body)
    aboard_attachment.assign()

    # -------------------- init pose --------------------#
    bar_pose = pp.get_pose(cur_element.body)  # world_from_body
    pick_pose = pp.multiply(bar_pose, pp.invert(grasp))  # world_from_gripper
    pick_tool0_pose = pp.multiply(pick_pose, pp.invert(robot_setup.tool0_from_ee))  # world_from_tool0
    # print(pick_tool0_pose)

    # -------------------- init variables --------------------#
    robot_init_conf = pp.get_joint_positions(robot_setup.robot, robot_setup.control_joints)
    robot_base_conf = robot_init_conf[:3]

    # -------------------- init extra_disabled_collisions --------------------#
    extra_disabled_collisions = [
        (
            (robot_setup.robot, pp.link_from_name(robot_setup.robot, "ur_arm_wrist_3_link")),
            (robot_setup.ee_attachment.child, pp.BASE_LINK),
        ),
        (
            (robot_setup.ee_attachment.child, pp.BASE_LINK),
            (cur_element.body, pp.BASE_LINK),
        ),
    ]

    # **************************************************************************
    # pick
    # **************************************************************************

    # -------------------- init collision checker fn without grasp attachment --------------------#
    collision_fn = pp.get_collision_fn(
        robot_setup.robot,
        robot_setup.control_joints,
        obstacles=obstacles | set([cur_element.body]),
        attachments=[robot_setup.ee_attachment] + robot_setup.attachments,
        self_collisions=ENABLE_SELF_COLLISIONS,
        disabled_collisions=robot_setup.disabled_collisions,
        extra_disabled_collisions=extra_disabled_collisions,
        max_distance=MAX_DISTANCE,
    )

    # -------------------- generate ik solution for pick --------------------#
    fail_flag = True
    robot_joint_conf_last = HUSKY_INIT_ARM_JOINT_ANGLES.tolist()
    for ik_search_num in range(ik_search_max_attempt):
        pick_joint_conf = robot_setup.get_relative_ik_solution(pick_tool0_pose, robot_joint_conf_last)
        if pick_joint_conf is None:
            if verbose:
                print("    pick ik not found")
            continue
        pick_joint_conf = normalize_angles(pick_joint_conf)
        pick_conf = np.hstack((robot_base_conf, pick_joint_conf))
        fail_flag = False
        break
    if fail_flag:
        if verbose:
            cprint("pick ik failure", "red")
        pick_ik_val.increment()
        pick_val.increment()
        return None, None

    # -------------------- collision check --------------------#
    if collision_fn(pick_conf, diagnosis=diagnosis):
        if verbose:
            cprint("pick ik collision failure", "red")
        pick_collision_val.increment()
        pick_val.increment()
        return None, None

    # **************************************************************************
    # pre pick
    # **************************************************************************

    # -------------------- generate post attach pose (retreat) --------------------#
    retreat_delta_point = tuple((np.array([0, 0, -1]) * retreat_dist).tolist())
    retreat_delta_pose = pp.Pose(point=retreat_delta_point, euler=pp.Euler(roll=0, pitch=0, yaw=0))
    pre_tool0_pose = pp.multiply(pick_tool0_pose, retreat_delta_pose)
    pre_tool0_poses = list(
        interpolate_poses(pre_tool0_pose, pick_tool0_pose, pos_step_size=POS_STEP_SIZE, ori_step_size=ORI_STEP_SIZE)
    )  # pre pick --> pick

    # -------------------- inversely generate pre pick confs --------------------#
    pre_pick_confs = [pick_conf]
    fail_flag = False
    robot_joint_conf_last = pick_joint_conf
    pose_last = pick_pose
    for temp_tool0_pose in pre_tool0_poses[::-1][1:]:
        inner_fail_flag = True
        for ik_search_num in range(ik_search_max_attempt):
            pre_pick_joint_conf = robot_setup.get_relative_ik_solution(temp_tool0_pose, robot_joint_conf_last.tolist())
            if pre_pick_joint_conf is None:
                if verbose:
                    print("    pre pick ik not found")
                continue
            pre_pick_joint_conf = normalize_angles(pre_pick_joint_conf)
            if angles_distance(pre_pick_joint_conf, robot_joint_conf_last) >= np.pi / 2:
                if verbose:
                    print(
                        "    pre pick ik interval too large:\n",
                        "       next: ",
                        pre_pick_joint_conf,
                        "\n",
                        "       last: ",
                        robot_joint_conf_last,
                        "\n",
                        "       diff: ",
                        angles_distance(pre_pick_joint_conf, robot_joint_conf_last),
                        "\n",
                        "       next pose:",
                        temp_tool0_pose,
                        "\n",
                        "       last pose:",
                        pose_last,
                    )
                continue
            pre_pick_confs = [np.hstack((robot_base_conf, pre_pick_joint_conf))] + pre_pick_confs
            robot_joint_conf_last = pre_pick_joint_conf
            pose_last = temp_tool0_pose
            inner_fail_flag = False
            break
        # check whether to exit early. If not, the solution fails.
        if inner_fail_flag:
            if verbose:
                cprint("pre pick ik failure", "red")
            pre_pick_ik_val.increment()
            pre_pick_val.increment()
            pick_val.increment()
            fail_flag = True
            break
    if fail_flag:
        return None, None

    # **************************************************************************
    # post pick
    # **************************************************************************

    post_pick_confs = copy.deepcopy(pre_pick_confs)[::-1]

    # **************************************************************************
    # home pose to pre pick
    # **************************************************************************

    # -------------------- from home pose to pre pick --------------------#
    approach_confs = []
    fail_flag = True
    for plan_attempt in range(path_plan_max_attempt):
        approach_arm_path = robot_setup.plan_manipulator_path(
            robot_setup.arm_init_angles,
            pre_pick_confs[0][3:],
            attachments=robot_setup.attachments,
            obstacles=obstacles,
        )
        if approach_arm_path is None:
            if verbose:
                print("    approach plan not found")
            continue

        approach_arm_path = [normalize_angles(conf) for conf in approach_arm_path]
        approach_confs = [np.hstack((robot_base_conf, conf)) for conf in approach_arm_path]

        inner_fail_flag = False
        for conf in approach_confs:
            if collision_fn(conf, diagnosis=diagnosis):
                if verbose:
                    print("    approach collision not pass")
                inner_fail_flag = True
                break
        if inner_fail_flag:
            continue
        fail_flag = False
        break
    if fail_flag:
        if verbose:
            cprint("approach plan failure", "red")
        approach_val.increment()
        pick_val.increment()
        return None, None

    # -------------------- return command, mask --------------------#
    return approach_confs + pre_pick_confs + post_pick_confs, [False] * len(approach_confs) + [False] * len(
        pre_pick_confs
    ) + [True] * len(post_pick_confs)


def get_pick_gen_fn(
    robot_setup: RobotSetup,
    element_from_index: Dict,
    fixed_obstacles: List[int],
    max_attempts: int = 10,
    collisions: bool = True,
    allow_failure: bool = False,
    verbose: bool = False,
    teleops: bool = False,
) -> Callable[
    [int, Tuple[Tuple[float], Tuple[float]], List[int], List[int], List[Attachment], CounterModule, bool],
    Tuple[List[np.ndarray], List[bool]],
]:
    """
    Generate pick motion planner function.

    Params:
        robot_setup (RobotSetup): RobotSetup instance
        element_from_index ({index: Element}): element dict
        fixed_obstacles ([int]): list of id in pybullet
        max_attempts (int, 100): attempts to generate place motion
        collisions (bool, True): whether consider collision
        allow_failure (bool, False): yield (None * 2), False: return (None * 2) and raise an error
        verbose (bool, False): whether print debug information
        teleops (bool, False): whether to interpolate the intermediate paths

    Returns:
        Callable: gen_fn(index, grasp, assembled, unassembled, attachments, counter, diagnosis)
    """

    def gen_fn(
        index: int,
        grasp: Tuple[Tuple[float], Tuple[float]],
        assembled: List[int] = [],
        unassembled: List[int] = [],
        attachments: List[Attachment] = [],
        other_obstacles: List[int] = [],
        counter: CounterModule = None,
        diagnosis: bool = False,
    ):
        """
        Generate pick motion and return path.

        Params:
            index (int): the index of element that needs to assemble
            grasp (pp.Pose): ((x, y, z), (x, y, z, w)), gripper_from_body
            assembled ([int], []): indices of assembled elements
            unassembled ([int], []): indices of unassembled elements (excluding the current element)
            attachments ([Attachment], []): list of attachments bound to the robot (excluding the current element)
            other_obstacles ([int], []): other obstacles, e.g. other robots
            counter (CounterModule, None): counter module
            diagnosis (bool, False): whether stop and display it in pybullet if a collision is detected

        Returns:
            command ([np.ndarray]): robot confs
            grasp_mask ([bool]): whether to attach the current element to the gripper
            grasp_attach (Attachment): the attachment between the current element and the gripper
            grasp (pp.Pose): gripper_from_body
            pregrasp (pp.Pose): world_from_element, used in transfer motion plan as target conf
        """
        cur_element: Element = element_from_index[index]

        # -------------------- update current attachments --------------------#
        robot_setup.update_attachments(attachments)
        # robot_arm_init_conf = pp.get_joint_positions(robot_setup.robot, robot_setup.arm_joints)

        # -------------------- set obstacles --------------------#
        element_obstacles = set({element_from_index[e].body for e in list(assembled)})
        unassambled_element_obstacles = set({element_from_index[e].body for e in list(unassembled)})

        obstacles = set(fixed_obstacles) | set(other_obstacles) | element_obstacles | unassambled_element_obstacles
        if not collisions:
            obstacles = set()

        # -------------------- loop: plan pick motion --------------------#
        for attempt in range(max_attempts):
            if verbose:
                cprint(f"pick attempt: {attempt}", "yellow")

            # -------------------- compute pick path --------------------#
            command, mask = compute_pick_path(
                robot_setup,
                index,
                grasp,
                element_from_index,
                obstacles,
                None,
                counter=counter,
                verbose=verbose,
                diagnosis=diagnosis,
                teleops=teleops,
            )
            if command is None:
                continue

            TermPrint.print("Pick E#{} | Attempts: {} | Command: {}".format(index, attempt, len(command)), "green")

            yield command, mask
            break

        # -------------------- fail --------------------#
        TermPrint.print("Pick E#{} | Attempts: {} | Max attempts exceeded!".format(index, max_attempts), "red")

        if allow_failure:
            yield None, None
        else:
            return None, None

    return gen_fn
