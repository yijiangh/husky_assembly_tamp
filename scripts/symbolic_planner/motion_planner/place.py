import os
import random
import sys
import time
from functools import partial
from itertools import islice
from typing import Callable, List, Set, Tuple, Dict, Union

import numpy as np
import pybullet_planning as pp
from termcolor import cprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pybullet_planning import Attachment, interpolate_poses
from robot.robot_setup import RobotSetup
from sampler.grasp_sampler import grasp_sampler
from sampler.mobile_base_sampler import robot_pose_sampler
from utils.collision import Element
from utils.utils import CounterModule, angles_distance, normalize_angles

# place retreat
RETREAT_DISTANCE = 0.07

# pregrasp attempts
PREGRASP_MAX_ATTEMPTS = 100

# pregrasp delta sample
EPSILON = 0.05
ANGLE = np.pi / 6

# pregrasp/retreat interpolation step
POS_STEP_SIZE = 0.005
ORI_STEP_SIZE = np.pi / 128

# collision check threshold
MAX_DISTANCE = 0.01

# collision check enable
ENABLE_SELF_COLLISIONS = True


# ------------------------------------------------------------ pregrasp ------------------------------------------------------------#
def get_delta_pose_generator(epsilon=EPSILON, angle=ANGLE):
    """sample generator for an infinitesimal \delta X \in SE(3)
    This is used as the pose difference between the pre-detach pose and the detach pose.

    Parameters
    ----------
    epsilon : [type]
        [description]
    angle : [type], optional
        [description], by default np.pi/2

    Yields
    -------
    Pose
    """
    lower = [-epsilon] * 3 + [-angle] * 3
    upper = [epsilon] * 3 + [angle] * 3
    for [x, y, z, roll, pitch, yaw] in pp.interval_generator(lower, upper):  # halton?
        pose = pp.Pose(point=[x, y, z], euler=pp.Euler(roll=roll, pitch=pitch, yaw=yaw))
        yield pose


def get_pregrasp_gen_fn(
    element_from_index,
    fixed_obstacles,
    max_attempts=PREGRASP_MAX_ATTEMPTS,
    collision=True,
    teleops=False,
):
    """sample generator for a path \tao \subset SE(3) between the pre-detach pose and the goal pose of ab element.

    Parameters
    ----------
    element_from_index : [type]
        [description]
    fixed_obstacles : [type]
        [description]
    max_attempts : [type], optional
        the number of sampling trails, by default PREGRASP_MAX_ATTEMPTS
    collision : bool, optional
        [description], by default True
    teleops : bool, optional
        skip the interpolation between the key poses, by default False

    Returns
    -------
    [type]
        [description]

    Yields
    -------
    a list of Pose
        element body poses
    """
    pre_grasp_pose_gen = get_delta_pose_generator()
    # pose_gen = get_single_axis_delta_pose_generator(axis=[1, 0, 0])

    def gen_fn(index, printed, diagnosis=False):
        element: Element = element_from_index[index]
        body = element.body
        # body_init_pose = element.init_pose
        body_tar_pose = element.goal_pose  # world from tar_pose

        # -------------------- Set obstacles --------------------#
        # element_obstacles = get_element_body_in_goal_pose(element_from_index, printed)
        # element_obstacles = set({})
        element_obstacles = set({element_from_index[e].body for e in list(printed)})

        obstacles = set(fixed_obstacles) | element_obstacles
        if not collision:
            obstacles = set()
        ee_collision_fn = pp.get_floating_body_collision_fn(body, obstacles, max_distance=MAX_DISTANCE)

        # -------------------- Find path from pre_grasp_pose to attach_pose --------------------#
        for _ in range(max_attempts):
            # attach_delta_pose = next(attach_pose_gen)  # tar_pose from attach_pose
            pre_grasp_delta_pose = next(pre_grasp_pose_gen)  # tar_pose from pre_grasp_pose
            # attach_pose = multiply(body_tar_pose, attach_delta_pose)
            attach_pose = body_tar_pose
            pre_grasp_pose = pp.multiply(attach_pose, pre_grasp_delta_pose)
            is_colliding = False
            if not teleops:
                offset_path = list(
                    interpolate_poses(
                        pre_grasp_pose, attach_pose, pos_step_size=POS_STEP_SIZE, ori_step_size=ORI_STEP_SIZE
                    )
                )
            else:
                offset_path = [pre_grasp_pose, attach_pose]
            for p in offset_path:  # [:-1]
                # TODO: if colliding at the world_from_bar pose, use local velocity + normal check
                # TODO: normal can be derived from
                if ee_collision_fn(p, diagnosis=diagnosis):
                    # if element_robot_collision_fnpose2conf(p)):
                    is_colliding = True
                    break
            if not is_colliding:
                yield offset_path
                break

        else:
            yield None

    return gen_fn


# ------------------------------------------------------------ compute place path ------------------------------------------------------------#
def compute_place_path(
    robot_setup: RobotSetup,
    index: int,
    assembled: List[int],
    element_from_index: dict,
    obstacles: Set[int],
    base_pose_sampler: Callable[[], Tuple[np.ndarray, float]],
    grasp_sampler: Callable[[np.ndarray, float], Tuple[Tuple[float], Tuple[float]]],
    pregrasp_gen_fn: Callable[[int, List[int], bool], List[Tuple[Tuple[float], Tuple[float]]]],
    counter: Union[CounterModule, None] = None,
    retreat_dist: float = RETREAT_DISTANCE,
    max_attempt: int = 10,
    ik_search_max_attempt: int = 1,
    path_plan_max_attempt: int = 1,
    verbose: bool = False,
    diagnosis: bool = False,
    teleops: bool = False,
) -> Tuple[List[np.ndarray], List[bool], Attachment]:
    """
    Compute place path (mobile manipulator conf): pregrasp --> attach_pose (goal_pose) --> home_pose.

    Params:
        robot_setup (RobotSetup): RobotSetup instance
        index (int): index of current element
        assembled ([index], [not used]): indices of assembled elements
        element_from_index ({index: Element}): dict of elements
        obstacles (Set[index]): fixed obstacles + assembled elements
        base_pose_sampler (function): [] --> Tuple[np.ndarray, float] ([x, y, z], yaw)
        grasp_sampler (function): [np.ndarray, float] ([x, y, z], yaw) --> Tuple[Tuple[float], Tuple[float]] (pp.Pose, gripper_from_body)
        pregrasp_gen_fn (function): [int, List[int], bool] (index, assembled, diagnosis) --> List[Tuple[Tuple[float], Tuple[float]]] ([pp.Pose], world_from_body),
        counter (CounterModule | None, None): counter module to count failures
        retreat_dist (float, RETREAT_DISTANCE): retreat distance after attach_pose (goal_pose)
        max_attempt (int, 10): number of attempts for current pregrasp_poses and grasp
        ik_search_max_attempt (int, 1): number of attempts for searching ik solution
        path_plan_max_attempt (int, 1): number of attempts for manipulator path planner
        verbose (bool, False): whether print debug information
        diagnosis (bool, False): whether stop and display it in pybullet if a collision is detected
        teleops (bool, False, [not used]): whether use interpolation or path plan to fill the middle point

    Returns:
        command ([np.ndarray]): place path (mobile manipulator conf): pregrasp --> attach_pose (goal_pose) --> home_pose
        mask ([bool]): whether to attach the current element to the gripper
        grasp_attach (Attachment): the attachment between the current element and the gripper
        grasp (pp.Pose): grasp pose (gripper_from_body)
        pregrasp (pp.Pose): pregrasp pose (world_from_body)
    """

    cur_element: Element = element_from_index[index]

    # -------------------- init counter module --------------------#
    attach_ik_val = counter.add_counter_value("attach ik failure")
    attach_val = counter.add_counter_value("attach failure")

    pre_attach_ik_val = counter.add_counter_value("pre attach ik failure")
    pre_attach_collision_val = counter.add_counter_value("pre attach collision failure")
    pre_attach_plan_val = counter.add_counter_value("pre attach plan failure")
    pre_attach_val = counter.add_counter_value("pre attach failure")

    post_attach_ik_val = counter.add_counter_value("post attach ik failure")
    post_attach_collision_val = counter.add_counter_value("post attach collision failure")
    post_attach_val = counter.add_counter_value("post attach failure")

    back_plan_val = counter.add_counter_value("back plan failure")
    back_val = counter.add_counter_value("back failure")

    # -------------------- init variables --------------------#
    bar_tar_pose = cur_element.goal_pose  # world_from_body

    # if verbose:
    #     pp.remove_all_debug()
    #     pp.draw_pose(attach_pose, length=0.4)
    #     pp.draw_point(cur_element.axis_endpoints[0], size=0.2)

    # -------------------- init attachment of current element --------------------#
    grasp_attachment = None

    # -------------------- loop: find a solution of place --------------------#
    for _ in range(max_attempt):

        # pp.wait_for_user()

        # **************************************************************************
        # base pose
        # **************************************************************************

        base_pose_tup = base_pose_sampler()
        if base_pose_tup is None:
            if verbose:
                cprint("base pose sample failure", "red")
            # TODO: 添加计数模块
            continue

        # **************************************************************************
        # grasp, gripper_from_body
        # **************************************************************************

        grasp = grasp_sampler(base_pose_tup[0], base_pose_tup[1])

        # **************************************************************************
        # attach
        # **************************************************************************

        # -------------------- generate pose --------------------#
        attach_pose = pp.multiply(bar_tar_pose, pp.invert(grasp))  # world_from_gripper
        attach_tool0_pose = pp.multiply(attach_pose, pp.invert(robot_setup.tool0_from_ee))  # world_from_tool0

        # -------------------- generate attach conf --------------------#
        robot_base_conf = np.hstack((base_pose_tup[0][:2], np.array([base_pose_tup[1]])))  # np.array([x, y, yaw])
        robot_setup.set_joint_positions(robot_setup.base_joints, robot_base_conf)  # update pose2d in pybullet
        robot_joint_attach_conf = robot_setup.get_relative_ik_solution(attach_tool0_pose)
        if robot_joint_attach_conf is None:
            if verbose:
                cprint("attach ik failure", "red")
            attach_ik_val.increment()
            attach_val.increment()
            continue
        robot_joint_attach_conf = normalize_angles(robot_joint_attach_conf)
        robot_attach_conf = np.hstack((robot_base_conf, robot_joint_attach_conf))
        # pre_attach_confs = [robot_attach_conf]  # [conf], pregrasp --> attach_pose

        # -------------------- create attachment of current element --------------------#
        # if grasp_attachment is None:
        pp.set_pose(cur_element.body, cur_element.goal_pose)
        robot_setup.set_joint_positions(robot_setup.control_joints, robot_attach_conf)
        grasp_attachment = pp.create_attachment(robot_setup.robot, robot_setup.tool_link, cur_element.body)

        # **************************************************************************
        # pre attach (birrt)
        # **************************************************************************

        # # -------------------- init collision checker --------------------#
        # collision_fn = pp.get_collision_fn(
        #     robot_setup.robot,
        #     robot_setup.control_joints,
        #     obstacles=obstacles,
        #     attachments=[grasp_attachment] + robot_setup.attachments,
        #     self_collisions=ENABLE_SELF_COLLISIONS,
        #     disabled_collisions=robot_setup.disabled_collisions,
        #     max_distance=MAX_DISTANCE,
        # )

        # # -------------------- generate delta pose and calculate pre_grasp --------------------#
        # pre_grasp_pose_gen = get_delta_pose_generator()
        # pre_grasp_delta_pose = next(pre_grasp_pose_gen)  # tar_from_pre_grasp
        # pre_grasp_pose = pp.multiply(cur_element.goal_pose, pre_grasp_delta_pose)  # world_from_body

        # # -------------------- generate pre grasp conf --------------------#
        # pre_attach_pose = pp.multiply(pre_grasp_pose, pp.invert(grasp))  # world_from_gripper
        # pre_tool0_pose = pp.multiply(pre_attach_pose, pp.invert(robot_setup.tool0_from_ee))  # world_from_tool0
        # robot_joint_pre_grasp_conf = robot_setup.get_relative_ik_solution(pre_tool0_pose)
        # if robot_joint_pre_grasp_conf is None:
        #     if verbose:
        #         cprint("pre attach ik failure", "red")
        #     pre_attach_ik_val.increment()
        #     pre_attach_val.increment()
        #     continue

        # # -------------------- from pre attach to attach --------------------#
        # pre_attach_confs = []
        # fail_flag = True
        # for plan_attempt in range(path_plan_max_attempt):
        #     pre_attach_path = robot_setup.plan_manipulator_path(
        #         robot_joint_pre_grasp_conf,
        #         robot_joint_attach_conf,
        #         robot_setup.attachments + [grasp_attachment],
        #         obstacles,
        #         sub_way_points=True,
        #     )
        #     if pre_attach_path is None:
        #         if verbose:
        #             print("    pre attach plan not found")
        #         continue

        #     pre_attach_path = [normalize_angles(conf) for conf in pre_attach_path]
        #     pre_attach_confs = [np.hstack((robot_base_conf, conf)) for conf in pre_attach_path] + [robot_attach_conf]

        #     inner_fail_flag = False
        #     for temp_conf in pre_attach_confs:
        #         if collision_fn(temp_conf, diagnosis):
        #             if verbose:
        #                 print("    pre attach collision not pass")
        #             inner_fail_flag = True
        #             break
        #     if inner_fail_flag:
        #         continue
        #     fail_flag = False
        #     break
        # if fail_flag:
        #     if verbose:
        #         cprint("pre attach plan failure", "red")
        #     pre_attach_plan_val.increment()
        #     pre_attach_val.increment()
        #     continue

        # **************************************************************************
        # pre attach (interpolation)
        # **************************************************************************

        # -------------------- generate pregrasp path: (world_from_body) pregrasp --> goal_pose --------------------#
        pregrasp_poses = next(pregrasp_gen_fn(index, assembled, diagnosis=diagnosis))
        if pregrasp_poses is None:
            if verbose:
                cprint("pregrasp failure", "red")
            # pregrasp_val.increment()
            continue
        pre_grasp_pose = pregrasp_poses[0]

        # -------------------- generate pre attach poses --------------------#
        pre_attach_poses = [
            pp.multiply(bar_pose, pp.invert(grasp)) for bar_pose in pregrasp_poses
        ]  # world_from_gripper (world_from_ee)
        pre_tool0_poses = [
            pp.multiply(temp_pose, pp.invert(robot_setup.tool0_from_ee)) for temp_pose in pre_attach_poses
        ]  # world_from_tool0
        # attach_pose = pre_attach_poses[-1]  # world_from_gripper (world_from_ee)

        # -------------------- init collision checker --------------------#
        collision_fn = pp.get_collision_fn(
            robot_setup.robot,
            robot_setup.control_joints,
            obstacles=obstacles,
            attachments=[grasp_attachment] + robot_setup.attachments,
            self_collisions=ENABLE_SELF_COLLISIONS,
            disabled_collisions=robot_setup.disabled_collisions,
            max_distance=MAX_DISTANCE,
        )

        # -------------------- inversely generate pre attach confs excluding attach_pose --------------------#
        pre_attach_confs = [robot_attach_conf]
        fail_flag = False
        robot_joint_conf_last = robot_joint_attach_conf
        pose_last = attach_tool0_pose
        for pre_tool0_pose in pre_tool0_poses[::-1][1:]:
            inner_fail_flag = True
            for ik_search_num in range(ik_search_max_attempt):
                pre_attach_joint_conf = robot_setup.get_relative_ik_solution(
                    pre_tool0_pose, robot_joint_conf_last.tolist()
                )
                if pre_attach_joint_conf is None:
                    if verbose:
                        print("    pre attach ik not found")
                    continue
                pre_attach_joint_conf = normalize_angles(pre_attach_joint_conf)
                if angles_distance(pre_attach_joint_conf, robot_joint_conf_last) >= np.pi / 2:
                    if verbose:
                        print(
                            "    pre attach ik interval too large:\n",
                            "       next: ",
                            pre_attach_joint_conf,
                            "\n",
                            "       last: ",
                            robot_joint_conf_last,
                            "\n",
                            "       diff: ",
                            angles_distance(pre_attach_joint_conf, robot_joint_conf_last),
                            "\n",
                            "       next pose:",
                            pre_tool0_pose,
                            "\n",
                            "       last pose:",
                            pose_last,
                        )
                    continue
                pre_attach_confs = [np.hstack((robot_base_conf, pre_attach_joint_conf))] + pre_attach_confs
                robot_joint_conf_last = pre_attach_joint_conf
                pose_last = pre_tool0_pose
                inner_fail_flag = False
                break
            # check whether to exit early. If not, the solution fails.
            if inner_fail_flag:
                if verbose:
                    cprint("pre attach ik failure", "red")
                pre_attach_ik_val.increment()
                pre_attach_val.increment()
                fail_flag = True
                break
        if fail_flag:
            continue

        # -------------------- pre attach collision check --------------------#
        fail_flag = False
        for pre_attach_conf in pre_attach_confs:
            if collision_fn(pre_attach_conf, diagnosis=diagnosis):
                if verbose:
                    cprint("pre attach collision failure", "red")
                pre_attach_collision_val.increment()
                pre_attach_val.increment()
                fail_flag = True
                break
        if fail_flag:
            break

        # **************************************************************************
        # post attach (retreat)
        # **************************************************************************

        # -------------------- set current element to goal pose and calculate collision --------------------#
        pp.set_pose(cur_element.body, cur_element.goal_pose)

        # -------------------- init collision checker --------------------#
        collision_fn_without_grasp = pp.get_collision_fn(
            robot_setup.robot,
            robot_setup.control_joints,
            obstacles=obstacles | set([cur_element.body]),
            attachments=robot_setup.attachments,
            self_collisions=ENABLE_SELF_COLLISIONS,
            disabled_collisions=robot_setup.disabled_collisions,
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
        robot_joint_conf_last = robot_joint_attach_conf
        for posttool0_pose in post_tool0_poses:
            inner_fail_flag = True
            for ik_search_num in range(ik_search_max_attempt):
                post_attach_joint_conf = robot_setup.get_relative_ik_solution(
                    posttool0_pose, robot_joint_conf_last.tolist()
                )
                if post_attach_joint_conf is None:
                    continue
                post_attach_joint_conf = normalize_angles(post_attach_joint_conf)
                if angles_distance(post_attach_joint_conf, robot_joint_conf_last) >= np.pi / 2:
                    continue
                post_attach_confs.append(np.hstack((robot_base_conf, post_attach_joint_conf)))
                robot_joint_conf_last = post_attach_joint_conf
                inner_fail_flag = False
                break
            if inner_fail_flag:
                if verbose:
                    cprint("post attach ik failure", "red")
                post_attach_ik_val.increment()
                post_attach_val.increment()
                fail_flag = True
                break
        if fail_flag:
            continue

        # -------------------- post attach collision check --------------------#
        fail_flag = False
        for post_attach_conf in post_attach_confs:
            if collision_fn_without_grasp(post_attach_conf, diagnosis=diagnosis):
                if verbose:
                    cprint("post attach collision failure", "red")
                post_attach_collision_val.increment()
                post_attach_val.increment()
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
                if collision_fn_without_grasp(back_conf, diagnosis=diagnosis):
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
        command = pre_attach_confs + post_attach_confs + back_confs
        mask = [True] * len(pre_attach_confs) + [False] * len(post_attach_confs) + [False] * len(back_confs)
        return command, mask, grasp_attachment, grasp, pre_grasp_pose

    return None, None, None, None, None


def get_place_gen_fn(
    robot_setup: RobotSetup,
    element_from_index: Dict,
    fixed_obstacles: List[int],
    max_attempts: int = 100,
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
    Generate place motion planner function.

    Params:
        robot_setup (RobotSetup): RobotSetup instance
        element_from_index ({index: Element}): element dict
        fixed_obstacles ([int]): list of id in pybullet
        max_attempts (int, 100): attempts to generate place motion
        collisions (bool, True): whether consider collision
        allow_failure (bool, False): yield (None * 5), False: return (None * 5) and raise an error
        verbose (bool, False): whether print debug information
        teleops (bool, False): whether to interpolate the intermediate paths

    Returns:
        Callable: gen_fn(index, assembled, unassembled, attachments, counter, diagnosis)
    """

    # pregrasp sampler
    pregrasp_gen_fn = get_pregrasp_gen_fn(element_from_index, fixed_obstacles, collision=collisions, teleops=teleops)

    def gen_fn(
        index: int,
        assembled: List[int] = [],
        unassembled: List[int] = [],
        attachments: List[Attachment] = [],
        counter: CounterModule = None,
        diagnosis: bool = False,
    ):
        """
        Generate place motion and return path.

        Params:
            index (int): the index of element that needs to assemble
            assembled ([int], []): indices of assembled elements
            unassembled ([int], []): indices of unassembled elements (excluding the current element)
            attachments ([Attachment], []): list of attachments bound to the robot (excluding the current element)
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
        robot_arm_init_conf = pp.get_joint_positions(robot_setup.robot, robot_setup.arm_joints)

        # -------------------- init current element to goal_pose --------------------#
        pp.set_pose(cur_element.body, cur_element.goal_pose)

        # -------------------- set obstacles --------------------#
        element_obstacles = set({element_from_index[e].body for e in list(assembled)})
        # unassambled_element_obstacles = set({element_from_index[e].body for e in list(unassembled)})

        obstacles = set(fixed_obstacles) | element_obstacles
        if not collisions:
            obstacles = set()

        # -------------------- generate vertices and edges for assembled structure + current element --------------------#
        edges = []
        vertices = []
        for i in assembled:
            list_item = [array_item.tolist() for array_item in element_from_index[i].axis_endpoints]
            edges.append(list_item)
            vertices.extend(list_item)
        list_item = [array_item.tolist() for array_item in cur_element.axis_endpoints]
        edges.append(list_item)
        vertices.extend(list_item)

        # -------------------- init pose sampler function --------------------#
        pose_sampler = partial(
            robot_pose_sampler,
            vertices=vertices,
            edges=edges,
            target_edge=cur_element.axis_endpoints,
            sample_max_distance=1.25,  # dist in 2d plane, log name 1
            safety_distance=0.75,  # safty dist in 2d plane, log name 3
            reach_distance=1.25,  # dist in 3d space, log name 2
            sampling_number=200,
        )

        # -------------------- loop: plan place motion --------------------#
        for attempt in range(max_attempts):
            if verbose:
                cprint(f"place attempt: {attempt}", "yellow")

            # -------------------- grasp pose sampler --------------------#
            grasp_sampler_fun = partial(
                grasp_sampler,
                index=index,
                assembled=assembled,
                element_from_index=element_from_index,
                sample_range=0.10,
                grasp_method="robot",
                redirect_method="robot",
            )

            # -------------------- compute place path --------------------#
            command, mask, grasp_attachment, grasp, pregrasp = compute_place_path(
                robot_setup,
                index,
                assembled,
                element_from_index,
                obstacles,
                pose_sampler,
                grasp_sampler_fun,
                pregrasp_gen_fn,
                counter=counter,
                verbose=verbose,
                diagnosis=diagnosis,
                teleops=teleops,
            )

            if command is None:
                continue

            cprint("Place E#{} | Attempts: {} | Command: {}".format(index, attempt, len(command)), "green")

            # -------------------- back to init position --------------------#
            robot_setup.set_joint_positions(robot_setup.arm_joints, robot_arm_init_conf)

            yield command, mask, grasp_attachment, grasp, pregrasp
            break

        # -------------------- fail --------------------#
        if verbose:
            cprint("E#{} | Attempts: {} | Max attempts exceeded!".format(index, max_attempts), "red")

        if allow_failure:
            yield None, None, None, None, None
        else:
            return None, None, None, None, None

    return gen_fn
