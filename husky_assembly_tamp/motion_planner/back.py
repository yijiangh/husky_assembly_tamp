import os
import sys
from functools import partial
from typing import Callable, Dict, List, Set, Tuple, Union

import numpy as np
import pybullet_planning as pp
from termcolor import cprint
from husky_assembly_tamp.utils.params import *

from pybullet_planning import Attachment, interpolate_poses
from husky_assembly_tamp.robot.robot_setup import HUSKY_INIT_ARM_JOINT_ANGLES, RobotSetup
from husky_assembly_tamp.sampler.grasp_sampler import grasp_sampler
from husky_assembly_tamp.sampler.mobile_base_sampler import robot_pose_sampler
from husky_assembly_tamp.utils.collision import Element
from husky_assembly_tamp.utils.util import CounterModule, TermPrint, angles_distance, normalize_angles

# place retreat
RETREAT_DISTANCE = 0.035

# retreat interpolation step
POS_STEP_SIZE = 0.005
ORI_STEP_SIZE = np.pi / 128

# collision check threshold
MAX_DISTANCE = 0.0

# collision check enable
ENABLE_SELF_COLLISIONS = True


def compute_back_path(
    robot_setup: RobotSetup,
    index: int,
    start_conf: np.ndarray,
    grasp: Tuple[Tuple[float], Tuple[float]],
    assembled: List[int],
    element_from_index: dict,
    obstacles: Set[int],
    counter: Union[CounterModule, None] = None,
    retreat_dist: float = RETREAT_DISTANCE,
    max_attempt: int = 2,
    ik_search_max_attempt: int = 1,
    path_plan_max_attempt: int = 1,
    verbose: bool = False,
    diagnosis: bool = False,
    teleops: bool = False,
) -> Tuple[List[np.ndarray], List[bool]]:
    """
    Compute back path (mobile manipulator conf): attach_pose (goal_pose) --> home_pose.

    Params:
        robot_setup (RobotSetup): RobotSetup instance
        index (int): index of current element
        start_conf (np.ndarray): start conf of the back motion
        grasp (pp.Pose): grasp pose (gripper_from_body)
        assembled ([index], [not used]): indices of assembled elements including current element
        element_from_index ({index: Element}): dict of elements
        obstacles (Set[index]): fixed obstacles + assembled elements + current element + other robots
        counter (CounterModule | None, None): counter module to count failures
        retreat_dist (float, RETREAT_DISTANCE): retreat distance after attach_pose (goal_pose)
        max_attempt (int, 2): number of attempts for current pregrasp_poses and grasp
        ik_search_max_attempt (int, 1): number of attempts for searching ik solution
        path_plan_max_attempt (int, 1): number of attempts for manipulator path planner
        verbose (bool, False): whether print debug information
        diagnosis (bool, False): whether stop and display it in pybullet if a collision is detected
        teleops (bool, False, [not used]): whether use interpolation or path plan to fill the middle point

    Returns:
        command ([np.ndarray]): place path (mobile manipulator conf): attach_pose (goal_pose) --> home_pose
        mask ([bool]): whether to attach the current element to the gripper
    """

    cur_element: Element = element_from_index[index]

    # -------------------- init counter module --------------------#
    post_attach_ik_val = counter.add_counter_value("post attach ik failure")
    post_attach_collision_val = counter.add_counter_value("post attach collision failure")
    post_attach_val = counter.add_counter_value("post attach failure")

    back_plan_val = counter.add_counter_value("back plan failure")

    back_val = counter.add_counter_value("back failure")

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

    # -------------------- generate pose --------------------#
    attach_pose = pp.multiply(cur_element.goal_pose, pp.invert(grasp))  # world_from_gripper
    attach_tool0_pose = pp.multiply(attach_pose, pp.invert(robot_setup.tool0_from_ee))  # world_from_tool0

    # -------------------- loop: find a solution of post place --------------------#
    for _ in range(max_attempt):

        # **************************************************************************
        # post attach (retreat)
        # **************************************************************************

        # -------------------- set current element to goal pose and calculate collision --------------------#
        pp.set_pose(cur_element.body, cur_element.goal_pose)

        # -------------------- init collision checker --------------------#
        post_collision_fn = pp.get_collision_fn(
            robot_setup.robot,
            robot_setup.control_joints,
            obstacles=obstacles,
            attachments=[robot_setup.ee_attachment] + robot_setup.attachments,
            self_collisions=ENABLE_SELF_COLLISIONS,
            disabled_collisions=robot_setup.disabled_collisions,
            extra_disabled_collisions=extra_disabled_collisions,
            max_distance=MAX_DISTANCE,
        )

        # -------------------- generate post attach pose (retreat) --------------------#
        retreat_delta_point = tuple((np.array([0, 0, -1]) * retreat_dist).tolist())
        retreat_delta_pose = pp.Pose(point=retreat_delta_point, euler=pp.Euler(roll=0, pitch=0, yaw=0))
        retreat_pose = pp.multiply(attach_tool0_pose, retreat_delta_pose)
        post_tool0_poses = list(
            interpolate_poses(attach_tool0_pose, retreat_pose, pos_step_size=POS_STEP_SIZE, ori_step_size=ORI_STEP_SIZE)
        )

        # -------------------- generate post attach confs --------------------#
        post_attach_confs = []
        fail_flag = False
        robot_joint_conf_last = start_conf[3:]
        for temp_index, posttool0_pose in enumerate(post_tool0_poses):
            inner_fail_flag = True
            for ik_search_num in range(ik_search_max_attempt):
                post_attach_joint_conf = robot_setup.get_relative_ik_solution(
                    posttool0_pose, robot_joint_conf_last.tolist()
                )
                if post_attach_joint_conf is None:
                    if verbose:
                        print(f"    post attach ik not found at step {temp_index}/{len(post_tool0_poses)-1}")
                    continue
                post_attach_joint_conf = normalize_angles(post_attach_joint_conf)
                if angles_distance(post_attach_joint_conf, robot_joint_conf_last) >= np.pi / 2:
                    if verbose:
                        print(
                            "    post attach ik interval too large:\n",
                            "       next: ",
                            post_attach_joint_conf,
                            "\n",
                            "       last: ",
                            robot_joint_conf_last,
                            "\n",
                            "       diff: ",
                            angles_distance(post_attach_joint_conf, robot_joint_conf_last),
                        )
                    continue
                post_attach_confs.append(np.hstack((start_conf[:3], post_attach_joint_conf)))
                robot_joint_conf_last = post_attach_joint_conf
                inner_fail_flag = False
                break
            if inner_fail_flag:
                if verbose:
                    cprint("post attach ik failure", "red")
                post_attach_ik_val.increment()
                post_attach_val.increment()
                back_val.increment()
                fail_flag = True
                break
        if fail_flag:
            continue

        # -------------------- post attach collision check --------------------#
        fail_flag = False
        for post_attach_conf in post_attach_confs:
            if post_collision_fn(post_attach_conf, diagnosis=diagnosis):
                if verbose:
                    cprint("post attach collision failure", "red")
                post_attach_collision_val.increment()
                post_attach_val.increment()
                back_val.increment()
                fail_flag = True
                break
        if fail_flag:
            continue

        # **************************************************************************
        # back to home pose
        # **************************************************************************

        # -------------------- from post attach to home pose --------------------#
        back_confs = []
        fail_flag = True
        for plan_attempt in range(path_plan_max_attempt):
            back_arm_path = robot_setup.plan_manipulator_path(
                post_attach_confs[-1][3:],
                robot_setup.arm_init_angles,
                attachments=robot_setup.attachments,
                obstacles=obstacles,
            )
            if back_arm_path is None:
                if verbose:
                    print("    back plan not found")
                continue

            back_arm_path = [normalize_angles(conf) for conf in back_arm_path]
            back_confs = [np.hstack((post_attach_confs[-1][:3], conf)) for conf in back_arm_path]

            inner_fail_flag = False
            for back_conf in back_confs:
                if post_collision_fn(back_conf, diagnosis=diagnosis):
                    if verbose:
                        print("    back collision not pass")
                    inner_fail_flag = True
                    break
            if inner_fail_flag:
                continue
            fail_flag = False
            break
        if fail_flag:
            if verbose:
                cprint("back plan failure", "red")
            back_plan_val.increment()
            back_val.increment()
            continue

        # -------------------- return command, mask, grasp_attach, grasp, pregrasp --------------------#
        command = post_attach_confs + back_confs
        mask = [False] * len(post_attach_confs) + [False] * len(back_confs)
        return command, mask

    return None, None


def get_back_gen_fn(
    robot_setup: RobotSetup,
    element_from_index: Dict,
    fixed_obstacles: List[int],
    max_attempts: int = 10,
    collisions: bool = True,
    allow_failure: bool = False,
    verbose: bool = False,
    teleops: bool = False,
) -> Callable[
    [int, List[int], List[int], List[Attachment], CounterModule, bool],
    Tuple[
        List[np.ndarray], List[bool], Attachment, Tuple[Tuple[float], Tuple[float]], Tuple[Tuple[float], Tuple[float]]
    ],
]:
    """
    Generate back motion planner function.

    Params:
        robot_setup (RobotSetup): RobotSetup instance
        element_from_index ({index: Element}): element dict
        fixed_obstacles ([int]): list of id in pybullet
        max_attempts (int, 10): attempts to generate place motion
        collisions (bool, True): whether consider collision
        allow_failure (bool, False): yield (None * 5), False: return (None * 5) and raise an error
        verbose (bool, False): whether print debug information
        teleops (bool, False): whether to interpolate the intermediate paths

    Returns:
        Callable: gen_fn(index, assembled, unassembled, attachments, counter, diagnosis)
    """

    def gen_fn(
        index: int,
        start_conf: np.ndarray,
        grasp: Tuple[Tuple[float], Tuple[float]],
        assembled: List[int] = [],
        unassembled: List[int] = [],
        attachments: List[Attachment] = [],
        other_obstacles: List[int] = [],
        counter: CounterModule = None,
        diagnosis: bool = False,
    ):
        """
        Generate back motion and return path.

        Params:
            index (int): the index of element that needs to assemble
            start_conf (np.ndarray): start conf of the back motion
            grasp (pp.Pose): grasp pose (gripper_from_body)
            assembled ([int], []): indices of assembled elements including current element
            unassembled ([int], []): indices of unassembled elements (excluding the current element)
            attachments ([Attachment], []): list of attachments bound to the robot (excluding the current element)
            other_obstacles ([int], []): other obstacles, e.g. other robots
            counter (CounterModule, None): counter module
            diagnosis (bool, False): whether stop and display it in pybullet if a collision is detected

        Returns:
            command ([np.ndarray]): robot confs
            grasp_mask ([bool]): whether to attach the current element to the gripper
        """
        cur_element: Element = element_from_index[index]

        # -------------------- update current attachments --------------------#
        robot_setup.update_attachments(attachments)

        robot_setup.set_joint_positions(robot_setup.control_joints, start_conf)

        # -------------------- init current element to goal_pose --------------------#
        pp.set_pose(cur_element.body, cur_element.goal_pose)

        # -------------------- set obstacles --------------------#
        element_obstacles = set({element_from_index[e].body for e in list(assembled)})
        # unassambled_element_obstacles = set({element_from_index[e].body for e in list(unassembled)})

        obstacles = set(fixed_obstacles) | set(other_obstacles) | element_obstacles
        if not collisions:
            obstacles = set()

        # -------------------- loop: plan post place motion --------------------#
        for attempt in range(max_attempts):
            if verbose:
                cprint(f"back attempt: {attempt}", "yellow")

            # -------------------- compute post place path --------------------#
            command, mask = compute_back_path(
                robot_setup,
                index,
                start_conf,
                grasp,
                assembled,
                element_from_index,
                obstacles,
                counter=counter,
                verbose=verbose,
                diagnosis=diagnosis,
                teleops=teleops,
            )

            if command is None:
                continue

            TermPrint.print("Back E#{} | Attempts: {} | Command: {}".format(index, attempt, len(command)), "green")

            # -------------------- back to init position --------------------#
            robot_setup.set_joint_positions(robot_setup.arm_joints, HUSKY_INIT_ARM_JOINT_ANGLES)

            yield command, mask
            break

        # -------------------- fail --------------------#
        TermPrint.print("Back E#{} | Attempts: {} | Max attempts exceeded!".format(index, max_attempts), "red")

        if allow_failure:
            yield None, None
        else:
            return None, None

    return gen_fn
