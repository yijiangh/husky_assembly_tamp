import random
from functools import partial
from itertools import islice
from typing import List, Tuple, Set

import numpy as np
import pybullet_planning as pp
from collision import Element, Grasp, create_couplers, init_pb
from grasp_redirector import preview_point_calculation, redirector
from mobile_base_controller import Stanley, State
from mobile_base_planner import RRTStar, fill_yaw_angle
from mobile_base_sampler import robot_pose_sampler
from pybullet_planning import Attachment, Euler, Point, Pose, get_distance, interpolate_poses, invert, multiply
from robot_setup import INIT_ARM_JOINT_ANGLES, RobotSetup
from scipy.spatial.transform import Rotation as R
from termcolor import cprint
from utils import CounterModule, CounterValue, angles_distance, normalize_angles

# place retreat
RETREAT_DISTANCE = 0.07

# pregrasp attempts
PREGRASP_MAX_ATTEMPTS = 100

# pregrasp delta sample
EPSILON = 0.25
ANGLE = np.pi

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
        pose = Pose(point=[x, y, z], euler=Euler(roll=roll, pitch=pitch, yaw=yaw))
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
            pre_grasp_pose = multiply(attach_pose, pre_grasp_delta_pose)
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
    pregrasp_poses: List[Tuple],
    grasp: Tuple,
    index: int,
    assambled: List[int],
    element_from_index: dict,
    obstacles: Set[int],
    pose_sampler,
    counter: CounterModule = None,
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
        pregrasp_poses ([pp.Pose]): pregrasp poses (world_from_body), pregrasp --> attach_pose (goal_pose)
        grasp (pp.Pose): grasp pose (gripper_from_body)
        index (int): index of current element
        assambled ([index]): indices of assembled elements
        element_from_index ({index: Element}): dict of elements
        obstacles (Set[index]): fixed obstacles + assembled elements
        pose_sampler (function): pose_sampler(attach_point=np.ndarray)
        counter (CounterModule, None): counter module to count failures
        retreat_dist (float, RETREAT_DISTANCE): retreat distance after attach_pose (goal_pose)
        max_attempt (int, 10): number of attempts for current pregrasp_poses and grasp
        ik_search_max_attempt (int, 5): number of attempts for searching ik solution
        path_plan_max_attempt (int, 5): number of attempts for manipulator path planner
        verbose (bool, False): whether print debug information
        diagnosis (bool, False): whether stop and display it in pybullet if a collision is detected
        teleops (bool, False): whether use interpolation or path plan to fill the middle point

    Returns:
        command ([np.ndarray]): place path (mobile manipulator conf): pregrasp --> attach_pose (goal_pose) --> home_pose
        mask ([bool]): whether to attach the current element to the gripper
        grasp_attach (Attachment): the attachment between the current element and the gripper
    """

    cur_element: Element = element_from_index[index]

    # -------------------- init counter module --------------------#
    attach_ik_val = counter.add_counter_value("attach ik failure")
    pre_attach_ik_val = counter.add_counter_value("pre attach ik failure")
    pre_attach_collision_val = counter.add_counter_value("pre attach collision failure")
    post_attach_ik_val = counter.add_counter_value("post attach ik failure")
    post_attach_collision_val = counter.add_counter_value("post attach collision failure")
    back_plan_val = counter.add_counter_value("back plan failure")

    # -------------------- generate pre_attach poses --------------------#
    pre_attach_poses = [
        multiply(bar_pose, invert(grasp)) for bar_pose in pregrasp_poses
    ]  # world_from_gripper (world_from_ee)
    pre_tool0_poses = [
        pp.multiply(temp_pose, pp.invert(robot_setup.tool0_from_ee)) for temp_pose in pre_attach_poses
    ]  # world_from_tool0
    attach_pose = pre_attach_poses[0]  # world_from_gripper (world_from_ee)

    # -------------------- init attachment of current element --------------------#
    grasp_attachment = None

    # -------------------- loop: find a solution of place --------------------#
    for _ in range(max_attempt):

        # -------------------- generate robot base pose --------------------#
        base_pose_tup = pose_sampler(attach_point=np.array(attach_pose[0]))  # (np.array([x, y, z]), yaw)
        if base_pose_tup is None:
            continue

        # **************************************************************************
        # attach
        # **************************************************************************

        # -------------------- generate attach conf --------------------#
        robot_base_conf = np.hstack((base_pose_tup[0][:2], np.array([base_pose_tup[1]])))  # np.array([x, y, yaw])
        robot_setup.set_joint_positions(robot_setup.base_joints, robot_base_conf)  # update pose2d in pybullet
        robot_joint_attach_conf = robot_setup.get_relative_ik_solution(pre_tool0_poses[-1])
        if robot_joint_attach_conf is None:
            if verbose:
                print("attach ik failure.")
            attach_ik_val.increment()
            continue
        robot_joint_attach_conf = normalize_angles(robot_joint_attach_conf)
        robot_attach_conf = np.hstack((robot_base_conf, robot_joint_attach_conf))
        pre_attach_confs = [robot_attach_conf]  # [conf], pregrasp --> attach_pose

        # -------------------- create attachment of current element --------------------#
        if grasp_attachment is None:
            pp.set_pose(cur_element.body, cur_element.goal_pose)
            robot_setup.set_joint_positions(robot_setup.control_joints, robot_attach_conf)
            grasp_attachment = pp.create_attachment(robot_setup.robot, robot_setup.tool_link, cur_element.body)

        # **************************************************************************
        # pre attach
        # **************************************************************************

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
        fail_flag = False
        robot_joint_conf_last = robot_joint_attach_conf
        for pre_tool0_pose in pre_tool0_poses[::-1][1:]:
            inner_fail_flag = True
            for ik_search_num in range(ik_search_max_attempt):
                pre_attach_joint_conf = robot_setup.get_relative_ik_solution(
                    pre_tool0_pose, robot_joint_conf_last.tolist()
                )
                if pre_attach_joint_conf is None:
                    if verbose:
                        print("    pre attach ik not found.")
                    continue
                pre_attach_joint_conf = normalize_angles(pre_attach_joint_conf)
                if angles_distance(pre_attach_joint_conf, robot_joint_conf_last) >= np.pi / 2:
                    if verbose:
                        print("    pre attach ik interval too large: ", pre_attach_joint_conf, robot_joint_conf_last)
                    continue
                pre_attach_confs = [np.hstack((robot_base_conf, pre_attach_joint_conf))] + pre_attach_confs
                robot_joint_conf_last = pre_attach_joint_conf
                inner_fail_flag = False
                break
            # check whether to exit early. If not, the solution fails.
            if inner_fail_flag:
                if verbose:
                    print("pre attach ik failure.")
                pre_attach_ik_val.increment()
                fail_flag = True
                break
        if fail_flag:
            continue

        # -------------------- pre attach collision check --------------------#
        fail_flag = False
        for pre_attach_conf in pre_attach_confs:
            if collision_fn(pre_attach_conf, diagnosis):
                if verbose:
                    print("pre attach collision failure.")
                pre_attach_collision_val.increment()
                fail_flag = True
                break
        if fail_flag:
            break

        # **************************************************************************
        # post attach (retreat)
        # **************************************************************************

        # -------------------- init collision checker --------------------#
        collision_fn_without_grasp = pp.get_collision_fn(
            robot_setup.robot,
            robot_setup.control_joints,
            obstacles=obstacles,
            attachments=robot_setup.attachments,
            self_collisions=ENABLE_SELF_COLLISIONS,
            disabled_collisions=robot_setup.disabled_collisions,
            max_distance=MAX_DISTANCE,
        )

        # -------------------- generate post attach pose (retreat) --------------------#
        retreat_delta_point = tuple((np.array([0, 0, -1]) * retreat_dist).tolist())
        retreat_delta_pose = Pose(point=retreat_delta_point, euler=Euler(roll=0, pitch=0, yaw=0))
        retreat_pose = multiply(pre_tool0_poses[-1], retreat_delta_pose)
        post_tool0_poses = list(
            interpolate_poses(pre_tool0_poses[-1], retreat_pose, pos_step_size=POS_STEP_SIZE, ori_step_size=ORI_STEP_SIZE)
        )

        # -------------------- generate post attach confs --------------------#
        # post_tool0_poses = [
        #     pp.multiply(temp_pose, pp.invert(robot_setup.tool0_from_ee)) for temp_pose in post_attach_poses
        # ]
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
                    print("post attach ik failure.")
                post_attach_ik_val.increment()
                fail_flag = True
                break
        if fail_flag:
            continue

        # -------------------- post attach collision check --------------------#
        fail_flag = False
        for post_attach_conf in post_attach_confs:
            if collision_fn_without_grasp(post_attach_conf, diagnosis):
                if verbose:
                    print("post attach collision failure.")
                post_attach_collision_val.increment()
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
                    print("    back plan not found.")
                continue

            back_arm_path = [normalize_angles(conf) for conf in back_arm_path]
            back_confs = [np.hstack((post_attach_confs[-1][:3], conf)) for conf in back_arm_path]

            inner_fail_flag = False
            for back_conf in back_confs:
                if collision_fn_without_grasp(back_conf, diagnosis):
                    if verbose:
                        print("    back collision not pass.")
                    inner_fail_flag = True
                    break
            if inner_fail_flag:
                continue
            fail_flag = False
            break
        if fail_flag:
            if verbose:
                print("back plan failure.")
            back_plan_val.increment()
            continue

        # -------------------- return command, mask, grasp_attach --------------------#
        command = pre_attach_confs + post_attach_confs + back_confs
        mask = [True] * len(pre_attach_confs) + [False] * len(post_attach_confs) + [False] * len(back_confs)
        return command, mask, grasp_attachment

        # -------------------- trash --------------------#

        ## robot = robot_setup.robot
        ## tool0_from_ee = robot_setup.tool0_from_ee
        ## ik_solver = robot_setup.ik_solver
        ## control_joints = robot_setup.control_joints

        ## element: Element = element_from_index[index]
        ## body = element.body

        ## pre_tool0_poses_rev = pre_tool0_poses[::-1]
        ## pp.draw_pose(attach_pose, length=0.4)

        ## pp.draw_pose(grasp, length=0.3)
        ## pp.draw_pose(attach_pose, length=0.3)

        # -------------------- pre attach confs generation based on pose sampler --------------------#

        # robot_setup.set_base_pose_2d(*robot_base_conf)

        # pp.wait_for_user()

        # -------------------- pre attach confs generation --------------------#
        # robot_init_conf = ik_solver.ik(pp.tform_from_pose(pre_tool0_poses_rev[0]))
        # if robot_init_conf is None:
        #     if verbose:
        #         print("init attach ik failure.")
        #     continue
        # robot_base_conf = robot_init_conf[:3]
        # robot_joint_init_conf = robot_init_conf[3:]
        # robot_joint_init_conf = normalize_angles(robot_joint_init_conf)
        # pre_attach_confs = [np.hstack((robot_base_conf, robot_joint_init_conf))]

        ## robot_setup.set_joint_positions(control_joints, robot_attach_conf)

        # -----------------------------------------------------------------------------------------------------------------------------------------------

        # # pp.set_pose(body, body_tar_pose)
        # # pp.set_joint_positions(robot, control_joints, post_attach_confs[0])
        # # robot_setup.ee_attachment.assign()
        # # # draw poses in the interpolate list
        # # for pre_attach_pose in pre_attach_poses:
        # #     pp.draw_pose(pre_attach_pose)
        # # for post_attach_pose in post_attach_poses:
        # #     pp.draw_pose(post_attach_pose)
        # # pp.draw_pose(attach_pose, length=0.2)  # pose of contact point on the bar
        # a = 1
        # return (
        #     pre_attach_confs + post_attach_confs + back_confs,
        #     [1] * (len(pre_attach_confs) + 1) + [0] * len(post_attach_confs[1:] + back_confs),
        #     cur_bar_attachment,
        # )

    return None, None, None

    # world_from_tool0 = pp.multiply(attach_pose, pp.invert(tool0_from_ee))  # pose of end joint of husky
    # attach_conf = ik_solver.ik(pp.tform_from_pose(world_from_tool0))


def get_place_gen_fn(
    robot_setup: RobotSetup,
    element_from_index: dict,
    fixed_obstacles: List[int],
    max_attempts: int = 10,
    max_grasp: int = 400,
    collisions: bool = True,
    allow_failure: bool = False,
    verbose: bool = False,
    teleops: bool = False,
):
    """
    Generate place motion planner function.

    Params:
        robot_setup (RobotSetup): RobotSetup instance
        element_from_index ({index: Element}): element dict
        fixed_obstacles ([int]): list of id in pybullet
        max_attempts (int, 10): repeat num for a single grasp
        max_grasp (int, 400): the number of attempts to generate grasp
        collisions (bool, True): whether consider collision
        allow_failure (bool, False): yield (None * 5), False: return (None * 5) and raise an error
        verbose (bool, False): whether print debug information
        teleops (bool, False): whether to interpolate the intermediate paths

    Returns:
        gen_fn: gen_fn(element, assembled, unassembled, attachments, counter, diagnosis)
    """

    # pregrasp sampler
    pregrasp_gen_fn = get_pregrasp_gen_fn(element_from_index, fixed_obstacles, collision=collisions, teleops=teleops)

    def gen_fn(
        index: int,
        assembled: List[int] = [],
        unassembled: List[int] = [],
        attachments: List[int] = [],
        counter: CounterModule = None,
        diagnosis: bool = False,
    ):
        """
        Generate place motion and return path.

        Params:
            element (int): the index of element that needs to assemble
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
            sample_max_distance=1.0,  # dist in 2d plane
            safety_distance=0.5,  # safty dist in 2d plane
            reach_distance=1.5,  # dist in 3d space
            sampling_number=200,
        )

        # -------------------- genegrate grasp: gripper_from_body --------------------#
        grasp_gen = pp.get_side_cylinder_grasps(cur_element.body, safety_margin_length=0.25)

        # -------------------- loop: traversing the grasp --------------------#
        for attempt, grasp in enumerate(islice(grasp_gen, max_grasp)):
            if verbose:
                print("attempt: ", attempt)

            # -------------------- loop: try to find a solution of current grasp --------------------#
            for _ in range(max_attempts):
                # -------------------- generate pregrasp path: (world_from_body) pregrasp --> goal_pose --------------------#
                # TODO: 这里可能有改进方案，即plan一条从结构外部到goal_pose的path
                pregrasp_poses = next(pregrasp_gen_fn(index, assembled, diagnosis=diagnosis))
                if not pregrasp_poses:
                    if verbose:
                        print("pregrasp failure.")
                    continue

                # -------------------- calculate preview point and regenerate grasp --------------------#
                preview_point = preview_point_calculation(assembled + [index], element_from_index)
                # pp.draw_point(preview_point, size=0.1)
                attach_pose = multiply(pregrasp_poses[-1], invert(grasp))
                attach_pose = redirector(
                    cur_element.axis_endpoints[0],
                    cur_element.axis_endpoints[1],
                    attach_pose,
                    preview_point,
                )
                grasp = multiply(invert(attach_pose), pregrasp_poses[-1])
                # pp.draw_pose(attach_pose, length=0.25)

                # -------------------- compute place path --------------------#
                command, mask, grasp_attachment = compute_place_path(
                    robot_setup,
                    pregrasp_poses,
                    grasp,
                    index,
                    assembled,
                    element_from_index,
                    obstacles,
                    pose_sampler,
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

                yield command, mask, grasp_attachment, grasp, pregrasp_poses[0]
                break

        if verbose:
            cprint("E#{} | Attempts: {} | Max attempts exceeded!".format(index, max_grasp), "red")

        if allow_failure:
            yield None, None, None, None, None
        else:
            return None, None, None, None, None

    return gen_fn
