import random
from itertools import islice
from typing import Tuple

import numpy as np
import pybullet_planning as pp
from collision import Element, Grasp, create_couplers, init_pb
from pybullet_planning import Attachment, Euler, Point, Pose, get_distance, interpolate_poses, invert, multiply
from robot_setup import RobotSetup
from scipy.spatial.transform import Rotation as R
from termcolor import cprint

##############################

ENABLE_SELF_COLLISIONS = True
assert ENABLE_SELF_COLLISIONS
IK_MAX_ATTEMPTS = 1
PREGRASP_MAX_ATTEMPTS = 100
GRASP_MAX_ATTEMPTS = 500

ALLOWABLE_BAR_COLLISION_DEPTH = 1e-3

# pregrasp delta sample
EPSILON = 0.3
ANGLE = np.pi / 3

# pregrasp interpolation
POS_STEP_SIZE = 0.005
ORI_STEP_SIZE = np.pi / 128

RETREAT_DISTANCE = 0.07
MAX_DISTANCE = 0.0
# MAX_DISTANCE = 0.07


def get_bar_grasp_gen_fn(element_from_index, tool_pose=pp.unit_pose(), reverse_grasp=False, safety_margin_length=0.0):
    """[summary]

    # converted from https://pybullet-planning.readthedocs.io/en/latest/reference/generated/pybullet_planning.primitives.grasp_gen.get_side_cylinder_grasps.html
    # to get rid of the rotation around the local z axis

    Parameters
    ----------
    element_from_index : [type]
        [description]
    tool_pose : [type], optional
        [description], by default unit_pose()
    reverse_grasp : bool, optional
        [description], by default False
    safety_margin_length : float, optional
        the length of the no-grasp region on the bar's two ends, by default 0.0

    Returns
    -------
    [type]
        [description]

    Yields
    -------
    [type]
        [description]
    """

    # rotate the cylinder's frame to make x axis align with the longitude axis
    longitude_x = Pose(euler=Euler(pitch=np.pi / 2))

    def gen_fn(index):
        # can get from aabb as well
        bar_length = get_distance(*element_from_index[index].axis_endpoints)
        while True:
            # translation along the longitude axis
            slide_dist = random.uniform(-bar_length / 2 + safety_margin_length, bar_length / 2 - safety_margin_length)
            translate_along_x_axis = Pose(point=Point(slide_dist, 0, 0))

            for j in range(1 + reverse_grasp):
                # the base pi/2 is to make y align with the longitude axis, conforming to the convention (see image in the doc)
                # flip the gripper, gripper symmetry
                rotate_around_z = Pose(euler=[0, 0, np.pi / 2 + j * np.pi])

                object_from_gripper = multiply(longitude_x, translate_along_x_axis, rotate_around_z, tool_pose)
                yield Grasp(index, invert(object_from_gripper))

    return gen_fn


######################################


def get_element_body_in_goal_pose(element_from_index, printed):
    # for e in list(printed):
    #     pp.set_pose(element_from_index[e].body, element_from_index[e].goal_pose)
    return {element_from_index[e].body for e in list(printed)}


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


def get_single_axis_delta_pose_generator(epsilon=EPSILON, angle=ANGLE, axis=[0, 0, 1]):
    axis_np = np.array(axis)
    trans_lower_np = np.array([-epsilon] * 3) * axis_np
    angle_lower_np = np.array([-angle] * 3) * axis_np
    trans_upper_np = np.array([epsilon] * 3) * axis_np
    angle_upper_np = np.array([angle] * 3) * axis_np

    lower = np.hstack((trans_lower_np, angle_lower_np)).tolist()
    upper = np.hstack((trans_upper_np, angle_upper_np)).tolist()
    for [x, y, z, roll, pitch, yaw] in pp.interval_generator(lower, upper):  # halton?
        pose = Pose(point=[x, y, z], euler=Euler(roll=roll, pitch=pitch, yaw=yaw))
        yield pose


def get_single_axis_delta_angle_pose_generator(retreat_dist: float, angle=ANGLE, axis=[0, 0, 1]):
    axis_np = np.array(axis)
    retreat_vector = np.array([retreat_dist] * 3) * (1 - axis_np) / np.sqrt(2)
    angle_lower_np = np.array([-angle] * 3) * axis_np
    angle_upper_np = np.array([angle] * 3) * axis_np
    angle_lower = angle_lower_np.tolist()
    angle_upper = angle_upper_np.tolist()

    for [roll, pitch, yaw] in pp.interval_generator(angle_lower, angle_upper):
        rot_vector = np.array([roll, pitch, yaw]) * axis_np
        rotation = R.from_rotvec(rot_vector)
        point = rotation.apply(retreat_vector).tolist()
        pose = Pose(point=point, euler=Euler(roll=0, pitch=0, yaw=0))
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


def compute_place_path(
    robot_setup: RobotSetup,
    pregrasp_poses,
    grasp,
    index,
    element_from_index,
    obstacles,
    retreat_dist,
    verbose=False,
    diagnosis=False,
    teleops=False,
    gantry_sample_fn=None,
    max_attempt=50,
    ik_search_max_attempt=10,
    path_plan_max_attempt=15,
) -> Tuple[list[np.ndarray], int, Attachment]:
    """Give the grasp and EE workspace poses, compute cartesian planning for pre-detach ~ detach ~ post-detach process."""
    # retreat_pose_gen = get_single_axis_delta_angle_pose_generator(retreat_dist, axis=rotate_axis)

    robot = robot_setup.robot
    tool0_from_ee = robot_setup.tool0_from_ee
    ik_solver = robot_setup.ik_solver
    control_joints = robot_setup.control_joints

    element: Element = element_from_index[index]
    body = element.body

    # -------------------- pre attach poses generation --------------------#
    pre_attach_poses = [multiply(bar_pose, invert(grasp)) for bar_pose in pregrasp_poses]
    pre_tool0_poses = [pp.multiply(temp_attach_pose, pp.invert(tool0_from_ee)) for temp_attach_pose in pre_attach_poses]
    pre_tool0_poses_rev = pre_tool0_poses[::-1]
    attach_pose = pre_attach_poses[-1]

    for _ in range(max_attempt):

        # -------------------- pre attach confs generation --------------------#
        robot_init_conf = ik_solver.ik(pp.tform_from_pose(pre_tool0_poses_rev[0]))
        if robot_init_conf is None:
            if verbose:
                print("init attach ik failure.")
            continue
        robot_base_conf = robot_init_conf[:3]
        robot_joint_init_conf = robot_init_conf[3:]
        robot_joint_init_conf = conf_refine(robot_joint_init_conf)
        pre_attach_confs = [np.hstack((robot_base_conf, robot_joint_init_conf))]

        robot_setup.set_joint_positions(control_joints, robot_init_conf)

        fail_flag = False
        robot_joint_conf_last = robot_joint_init_conf
        for pre_tool0_pose in pre_tool0_poses_rev[1:]:
            for ik_search_num in range(ik_search_max_attempt):
                pre_attach_joint_conf = robot_setup.get_relative_ik_solution(
                    pre_tool0_pose, robot_joint_conf_last.tolist()
                )
                if pre_attach_joint_conf is None:
                    continue
                pre_attach_joint_conf = conf_refine(pre_attach_joint_conf)
                if np.linalg.norm(pre_attach_joint_conf - robot_joint_conf_last) >= np.pi / 2:
                    continue
                pre_attach_confs.append(np.hstack((robot_base_conf, pre_attach_joint_conf)))
                robot_joint_conf_last = pre_attach_joint_conf
                break
            if ik_search_num == ik_search_max_attempt - 1:
                if verbose:
                    print("pre attach ik failure.")
                fail_flag = True
                break
        if fail_flag:
            continue
        pre_attach_confs = pre_attach_confs[::-1]

        # -------------------- collision checker --------------------#
        cur_bar_attachment = pp.create_attachment(robot, robot_setup.tool_link, body)
        collision_fn = pp.get_collision_fn(
            robot,
            robot_setup.control_joints,
            obstacles=obstacles,
            attachments=[cur_bar_attachment],
            self_collisions=ENABLE_SELF_COLLISIONS,
            disabled_collisions=robot_setup.disabled_collisions,
            max_distance=MAX_DISTANCE,
        )
        collision_fn_no_attachment = pp.get_collision_fn(
            robot,
            robot_setup.control_joints,
            obstacles=obstacles,
            attachments=[],
            self_collisions=ENABLE_SELF_COLLISIONS,
            disabled_collisions=robot_setup.disabled_collisions,
            max_distance=MAX_DISTANCE,
        )

        # -------------------- pre attach collision check --------------------#
        fail_flag = False
        for pre_attach_conf in pre_attach_confs:
            if collision_fn(pre_attach_conf, diagnosis):
                if verbose:
                    print("pre attach collision failure.")
                fail_flag = True
                break
            # break
        if fail_flag:
            break

        # -------------------- retreat pose generation --------------------#
        retreat_delta_point = tuple((np.array([0, 0, -1]) * retreat_dist).tolist())
        retreat_delta_pose = Pose(point=retreat_delta_point, euler=Euler(roll=0, pitch=0, yaw=0))

        retreat_pose = multiply(attach_pose, retreat_delta_pose)
        post_attach_poses = list(
            interpolate_poses(attach_pose, retreat_pose, pos_step_size=POS_STEP_SIZE, ori_step_size=ORI_STEP_SIZE)
        )

        # -------------------- post attach IK --------------------#
        post_tool0_poses = [
            pp.multiply(temp_attach_pose, pp.invert(tool0_from_ee)) for temp_attach_pose in post_attach_poses
        ]
        post_attach_confs = []

        fail_flag = False
        robot_joint_conf_last = robot_joint_init_conf
        for post_tool0_pose in post_tool0_poses:
            for ik_search_num in range(ik_search_max_attempt):
                post_attach_joint_conf = robot_setup.get_relative_ik_solution(
                    post_tool0_pose, robot_joint_conf_last.tolist()
                )
                if post_attach_joint_conf is None:
                    continue
                post_attach_joint_conf = conf_refine(post_attach_joint_conf)
                if np.linalg.norm(post_attach_joint_conf - robot_joint_conf_last) >= np.pi / 2:
                    continue
                post_attach_confs.append(np.hstack((robot_base_conf, post_attach_joint_conf)))
                robot_joint_conf_last = post_attach_joint_conf
                break
            if ik_search_num == ik_search_max_attempt - 1:
                if verbose:
                    print("post attach ik failure.")
                fail_flag = True
                break
        if fail_flag:
            continue

        # -------------------- post attach collision check --------------------#
        fail_flag = False
        for post_attach_conf in post_attach_confs:
            if collision_fn(post_attach_conf, diagnosis):
                if verbose:
                    print("post attach collision failure.")
                fail_flag = True
                break
        if fail_flag:
            continue

        # -------------------- from post attach to init conf --------------------#
        fail_flag = True
        for plan_attempt in range(path_plan_max_attempt):
            back_arm_path = robot_setup.plan_manipulator_path(
                post_attach_confs[-1][3:],
                robot_setup.arm_init_angles,
                attachments=[],
                obstacles=obstacles,
            )
            if back_arm_path is None:
                if verbose:
                    print("back plan failure.")
                continue

            back_arm_path = [conf_refine(conf) for conf in back_arm_path]
            back_confs = [np.hstack((post_attach_confs[-1][:3], conf)) for conf in back_arm_path]

            inner_fail_flag = False
            for back_conf in back_confs:
                if collision_fn_no_attachment(back_conf, diagnosis):
                    if verbose:
                        print("back collision failure.")
                    inner_fail_flag = True
                    break
            if inner_fail_flag:
                continue

            fail_flag = False
            break
        if fail_flag:
            continue

        # pp.set_pose(body, body_tar_pose)
        # pp.set_joint_positions(robot, control_joints, post_attach_confs[0])
        # robot_setup.ee_attachment.assign()
        # # draw poses in the interpolate list
        # for pre_attach_pose in pre_attach_poses:
        #     pp.draw_pose(pre_attach_pose)
        # for post_attach_pose in post_attach_poses:
        #     pp.draw_pose(post_attach_pose)
        # pp.draw_pose(attach_pose, length=0.2)  # pose of contact point on the bar
        return (
            pre_attach_confs + post_attach_confs + back_confs,
            [1] * (len(pre_attach_confs) + 1) + [0] * len(post_attach_confs[1:] + back_confs),
            cur_bar_attachment,
        )

    return None, None, None

    # world_from_tool0 = pp.multiply(attach_pose, pp.invert(tool0_from_ee))  # pose of end joint of husky
    # attach_conf = ik_solver.ik(pp.tform_from_pose(world_from_tool0))


def compute_pick_path(
    robot_setup: RobotSetup,
    grasp,
    element_index: int,
    element_from_index,
    obstacles,
    unassambled_element_obstacles,
    retreat_dist=RETREAT_DISTANCE,
    verbose=False,
    diagnosis=False,
    teleops=False,
    ik_search_max_attempt=10,
    path_plan_max_attempt=50,
) -> Tuple[list[np.ndarray], list, Attachment]:
    """
    @brief: generate path: init --> pre_pick_pose --> grasp_pick_pose --> post_pick_pose\n
    ---
    @param:\n
        grasp: gripper from body\n
    ---
    @return:\n
    """
    element: Element = element_from_index[element_index]
    body = element.body
    body_init_pose = element.init_pose  # world from body
    # body_init_pose_r = robot_setup.get_relative_pose(body_init_pose) # arm base from body
    grasp_pick_pose = multiply(body_init_pose, invert(grasp))  # world from grasp

    # print("\n\nview 2: body_init_pose\n", body_init_pose)

    robot_init_conf = pp.get_joint_positions(robot_setup.robot, robot_setup.control_joints)
    robot_base_init_conf = robot_init_conf[:3]
    robot_arm_init_conf = robot_init_conf[3:]

    # print("\n\nview 3: robot_init_conf\n", robot_init_conf)

    # -------------------- approach --------------------#
    approach_delta_point = tuple((np.array([0, 0, -1]) * retreat_dist).tolist())
    approach_delta_pose = Pose(point=approach_delta_point, euler=Euler(roll=0, pitch=0, yaw=0))  # grasp from approach
    approach_pick_pose = multiply(grasp_pick_pose, approach_delta_pose)  # world from approach
    approach_tool0_pose = multiply(approach_pick_pose, invert(robot_setup.tool0_from_ee))  # world from tool0

    # print("\n\nview 4: approach_pick_pose\n", approach_pick_pose)
    # pp.draw_pose(approach_pick_pose, length=0.5)

    # -------------------- collision checker without bar --------------------#
    collision_fn_without_attach = pp.get_collision_fn(
        robot_setup.robot,
        robot_setup.control_joints,
        obstacles=obstacles | unassambled_element_obstacles,
        attachments=[],
        self_collisions=ENABLE_SELF_COLLISIONS,
        disabled_collisions=robot_setup.disabled_collisions,
        max_distance=MAX_DISTANCE,
    )

    for _ in range(path_plan_max_attempt):
        fail_flag = True
        for ik_search_num in range(ik_search_max_attempt):
            approach_pick_arm_conf = robot_setup.get_relative_ik_solution(approach_tool0_pose, robot_arm_init_conf)
            if approach_pick_arm_conf is None:
                if verbose:
                    print("approach ik failure.")
                continue
            approach_pick_arm_conf = conf_refine(approach_pick_arm_conf)

            approach_pick_conf = np.hstack((robot_base_init_conf, approach_pick_arm_conf))
            if collision_fn_without_attach(approach_pick_conf, diagnosis):
                if verbose:
                    print("approach collision failure.")
                continue
            fail_flag = False
            break
        if fail_flag:
            continue
        # -------------------- from init to approach --------------------#
        pre_pick_path = robot_setup.plan_manipulator_path(
            robot_arm_init_conf, approach_pick_arm_conf, attachments=[], obstacles=obstacles
        )
        if pre_pick_path is None:
            if verbose:
                print("pre pick plan failure.")
            continue

        pre_pick_path = [conf_refine(conf) for conf in pre_pick_path]
        pre_pick_confs = [np.hstack((robot_base_init_conf, conf)) for conf in pre_pick_path]
        # print("\n\nview 5.1: pre_pick_path\n", pre_pick_path)
        # -------------------- pre pick path collision check --------------------#
        fail_flag = False
        for pre_pick_conf in pre_pick_confs:
            if collision_fn_without_attach(pre_pick_conf, diagnosis):
                if verbose:
                    print("pre pick collision failure.")
                fail_flag = True
                break
        if fail_flag:
            continue
        # -------------------- from pre pick to pick --------------------#
        approach_pick_poses = list(
            interpolate_poses(
                approach_pick_pose, grasp_pick_pose, pos_step_size=POS_STEP_SIZE, ori_step_size=ORI_STEP_SIZE
            )
        )  # world from ee
        # -------------------- approach IK --------------------#
        approach_tool0_poses = [
            pp.multiply(temp_pose, pp.invert(robot_setup.tool0_from_ee)) for temp_pose in approach_pick_poses
        ]
        approach_confs = []
        fail_flag = False
        robot_joint_conf_last = pre_pick_path[-1]
        for approach_tool0_pose in approach_tool0_poses:
            for ik_search_num in range(ik_search_max_attempt):
                appraoch_joint_conf = robot_setup.get_relative_ik_solution(
                    approach_tool0_pose, robot_joint_conf_last.tolist()
                )
                if appraoch_joint_conf is None:
                    continue
                appraoch_joint_conf = conf_refine(appraoch_joint_conf)
                if np.linalg.norm(appraoch_joint_conf - robot_joint_conf_last) >= np.pi / 2:
                    continue
                approach_confs.append(np.hstack((robot_base_init_conf, appraoch_joint_conf)))
                robot_joint_conf_last = appraoch_joint_conf
                break
            if ik_search_num == ik_search_max_attempt - 1:
                if verbose:
                    print("approach ik failure.")
                fail_flag = True
                break
        if fail_flag:
            continue
        # # -------------------- collision checker with bar --------------------#
        # pp.set_pose(body, body_init_pose)
        # # pp.set_joint_positions(robot_setup.robot, robot_setup.control_joints, approach_confs[-1])
        # robot_setup.set_joint_positions(robot_setup.control_joints, approach_confs[-1])
        # cur_bar_attachment = pp.create_attachment(robot_setup.robot, robot_setup.tool_link, body)
        # collision_fn = pp.get_collision_fn(
        #     robot_setup.robot,
        #     robot_setup.control_joints,
        #     obstacles=obstacles,
        #     attachments=[cur_bar_attachment],
        #     # attachments=[],
        #     self_collisions=ENABLE_SELF_COLLISIONS,
        #     # self_collisions=False,tool_link
        #     disabled_collisions=robot_setup.disabled_collisions,
        #     # custom_limits=get_custom_limits(robot),
        #     max_distance=MAX_DISTANCE,
        # )
        # # -------------------- approach collision check --------------------#
        # fail_flag = False
        # for approach_conf in approach_confs:
        #     # print(">>>> ", post_attach_conf)
        #     if collision_fn(approach_conf, diagnosis):
        #         # if collision_fn(post_attach_conf, True):
        #         if verbose:
        #             print("approach collision failure.")
        #         fail_flag = True
        #         break
        #     # break
        # if fail_flag:
        #     continue

        retreat_confs = approach_confs[::-1]

        # -------------------- set position --------------------#
        # pp.set_pose(body, body_init_pose)
        # pp.set_joint_positions(robot_setup.robot, robot_setup.control_joints, approach_confs[-1])
        # robot_setup.ee_attachment.assign()

        return pre_pick_confs + approach_confs + retreat_confs, [0] * len(pre_pick_confs + approach_confs) + [1] * len(
            retreat_confs
        )

    return None, None


def compute_transfer_path(
    robot_setup: RobotSetup,
    body_attachment: Attachment,
    start,
    target,
    obstacles,
    unassambled_element_obstacles,
    verbose=False,
    diagnosis=False,
    teleops=False,
    path_plan_max_attempt=50,
) -> Tuple[list[np.ndarray]]:
    """
    @brief: generate path: post_pick_pose --> pre_grasp_pose\n
    ---
    @param:\n
        start: start conf\n
        target: target conf\n
    ---
    @return:\n
    """
    collision_fn = pp.get_collision_fn(
        robot_setup.robot,
        robot_setup.control_joints,
        obstacles=obstacles | unassambled_element_obstacles,
        attachments=[body_attachment],
        self_collisions=ENABLE_SELF_COLLISIONS,
        disabled_collisions=robot_setup.disabled_collisions,
        max_distance=MAX_DISTANCE,
    )
    start_base_conf = start[:3]
    target_base_conf = target[:3]
    if np.linalg.norm(start_base_conf - target_base_conf) >= 0.1:
        raise RuntimeError("start pose of base must euqal to target pose of base!")
    start_arm_conf = start[3:]
    target_arm_conf = target[3:]

    for _ in range(path_plan_max_attempt):
        transfer_path = robot_setup.plan_manipulator_path(
            start_arm_conf,
            target_arm_conf,
            attachments=[body_attachment],
            obstacles=obstacles | unassambled_element_obstacles,
        )
        if transfer_path is None:
            if verbose:
                print("transfer plan failure.")
            continue

        transfer_path = [conf_refine(conf) for conf in transfer_path]
        transfer_confs = [np.hstack((start_base_conf, conf)) for conf in transfer_path]

        fail_flag = False
        for transfer_conf in transfer_confs:
            if collision_fn(transfer_conf, diagnosis):
                if verbose:
                    print("transfer collision failure.")
                fail_flag = True
                break
        if fail_flag:
            continue

        return transfer_confs

    return None


def conf_refine(conf):
    for i in range(len(conf)):
        while conf[i] > np.pi:
            conf[i] -= np.pi * 2
        while conf[i] <= -np.pi:
            conf[i] += np.pi * 2
    return conf


###############################


def get_place_gen_fn(
    robot_setup: RobotSetup,
    element_from_index,
    fixed_obstacles,
    collisions=True,
    max_attempts=10,
    max_grasp=1000,
    allow_failure=False,
    verbose=False,
    teleops=False,
):

    # conditioned sampler
    pregrasp_gen_fn = get_pregrasp_gen_fn(
        element_from_index, fixed_obstacles, collision=collisions, teleops=teleops
    )  # max_attempts=max_attempts,

    def gen_fn(element, assembled=[], unassembled=[], attachments=[], diagnosis=False):
        # for bar_id in assembled:
        #     bar_id_pyb = element_from_index[bar_id].index
        #     pp.set_color(bar_id_pyb, pp.GREEN)
        robot_setup.update_attachments(attachments)
        robot_arm_init_conf = pp.get_joint_positions(robot_setup.robot, robot_setup.arm_joints)

        pp.set_pose(element_from_index[element].body, element_from_index[element].goal_pose)

        element_obstacles = set({element_from_index[e].body for e in list(assembled)})
        unassambled_element_obstacles = set({element_from_index[e].body for e in list(unassembled)})

        obstacles = set(fixed_obstacles) | element_obstacles | unassambled_element_obstacles
        if not collisions:
            obstacles = set()

        grasp_gen = pp.get_side_cylinder_grasps(element_from_index[element].body, safety_margin_length=0.5)
        # keep track of sampled traj, prune newly sampled one with more collided element
        for attempt, grasp in enumerate(islice(grasp_gen, max_grasp)):
            print("attempt: ", attempt)
            # * ik iterations, usually 1 is enough
            for _ in range(max_attempts):
                # ! when used in pddlstream (except incremental_sm), the pregrasp sampler assumes no elements assembled at all time
                pregrasp_poses = next(pregrasp_gen_fn(element, assembled, diagnosis=diagnosis))
                if not pregrasp_poses:
                    if verbose:
                        print("pregrasp failure.")
                    continue

                command, grasp_mask, grasp_attach = compute_place_path(
                    robot_setup,
                    pregrasp_poses,
                    grasp,
                    element,
                    element_from_index,
                    obstacles,
                    retreat_dist=RETREAT_DISTANCE,
                    verbose=verbose,
                    diagnosis=diagnosis,
                    teleops=teleops,
                )

                if command is None:
                    continue

                # trajectories.append(command)
                # if command not in trajectories:
                #     continue

                # if verbose:
                cprint("Place E#{} | Attempts: {} | Command: {}".format(element, attempt, len(command)), "green")

                robot_setup.set_joint_positions(robot_setup.arm_joints, robot_arm_init_conf)

                yield command, grasp_mask, grasp_attach, grasp, pregrasp_poses[0]
                break
        else:
            if verbose:
                cprint("E#{} | Attempts: {} | Max attempts exceeded!".format(element, max_grasp), "red")

            if allow_failure:
                yield None, None, None
            else:
                return

    return gen_fn


def get_pick_gen_fn(
    robot_setup: RobotSetup,
    element_from_index,
    fixed_obstacles,
    collisions=True,
    max_attempts=20,
    allow_failure=False,
    verbose=False,
    teleops=False,
):
    def gen_fn(element_index: int, grasp_raw, assembled=[], unassembled=[], attachments=[], diagnosis=False):
        robot_setup.update_attachments(attachments)
        pick_element: Element = element_from_index[element_index]
        # -------------------- obstacles --------------------#
        assambled_element_obstacles = set({element_from_index[e].body for e in list(assembled)})
        unassambled_element_obstacles = set({element_from_index[e].body for e in list(unassembled)})

        obstacles = set(fixed_obstacles) | assambled_element_obstacles
        if not collisions:
            obstacles = set()

        # -------------------- update init pose --------------------#
        element = Element(
            pick_element.index,
            pick_element.body,
            pp.get_pose(pick_element.body),
            pick_element.goal_pose,
            pick_element.axis_endpoints,
        )
        element_from_index[element_index] = element

        grasp_temp = Pose([0, 0, 0], Euler(roll=np.pi / 2, pitch=0, yaw=0))
        grasp = (grasp_raw[0], grasp_temp[1])

        for attempt in range(max_attempts):
            print("attempt: ", attempt)

            command, grasp_mask = compute_pick_path(
                robot_setup,
                grasp,
                element_index,
                element_from_index,
                obstacles,
                unassambled_element_obstacles,
                retreat_dist=RETREAT_DISTANCE,
                verbose=verbose,
                diagnosis=diagnosis,
                teleops=teleops,
            )
            if command is None:
                continue

            cprint("Pick E#{} | Attempts: {} | Command: {}".format(element_index, attempt, len(command)), "green")

            yield command, grasp_mask
            break
        else:
            if verbose:
                cprint("E#{} | Attempts: {} | Max attempts exceeded!".format(element_index, max_attempts), "red")

            if allow_failure:
                yield None, None
            else:
                return

    return gen_fn


def get_transfer_gen_fn(
    robot_setup: RobotSetup,
    element_from_index,
    fixed_obstacles,
    collisions=True,
    max_attempts=50,
    allow_failure=False,
    verbose=False,
    teleops=False,
):
    def gen_fn(
        element_index: int,
        body_attachment,
        start_conf,
        tar_conf,
        assembled=[],
        unassembled=[],
        attachments=[],
        diagnosis=False,
    ):
        robot_setup.update_attachments(attachments)
        # -------------------- obstacles --------------------#
        assambled_element_obstacles = set({element_from_index[e].body for e in list(assembled)})
        unassambled_element_obstacles = set({element_from_index[e].body for e in list(unassembled)})

        obstacles = set(fixed_obstacles) | assambled_element_obstacles
        if not collisions:
            obstacles = set()

        for attempt in range(max_attempts):
            print("attempt: ", attempt)

            command = compute_transfer_path(
                robot_setup,
                body_attachment,
                start_conf,
                tar_conf,
                obstacles,
                unassambled_element_obstacles,
                verbose=verbose,
                diagnosis=diagnosis,
                teleops=teleops,
            )
            if command is None:
                continue

            cprint("Transfer E#{} | Attempts: {} | Command: {}".format(element_index, attempt, len(command)), "green")

            yield command, [1] * len(command)
            break
        else:
            if verbose:
                cprint("E#{} | Attempts: {} | Max attempts exceeded!".format(element_index, max_attempts), "red")

            if allow_failure:
                yield None, None
            else:
                return

    return gen_fn


if __name__ == "__main__":
    init_pb()
    robot_setup0 = RobotSetup("r0")
    robot_pose = pp.get_pose(robot_setup0.robot)
    print(">>> view 1: robot pose", robot_pose)

    new_robot_pose = Pose((1.5, 1.5, 0), Euler(0, 0, np.pi / 2))
    pp.draw_pose(new_robot_pose, length=1.5)
    pp.set_pose(robot_setup0.robot, new_robot_pose)
    robot_setup0.ee_attachment.assign()
    robot_pose = pp.get_pose(robot_setup0.robot)
    print(">>> view 2: new robot pose", robot_pose)

    world_pose = ((-1, 0, 0), (0, 0, 0, 1))
    pp.draw_pose(world_pose, length=0.25)
    relative_pose = robot_setup0.get_relative_pose(world_pose)
    print(">>> view 3: relative pose", relative_pose)

    goal_pose = Pose((1.5, 2.0, 1.5), Euler(0, np.pi / 2, 0))
    pp.draw_pose(goal_pose, length=0.5)
    goal_relative_pose = robot_setup0.get_relative_pose(goal_pose)
    joint_conf = robot_setup0.get_relative_ik_solution(goal_relative_pose)
    print(">>> view 4.1: joint conf", joint_conf)
    conf = robot_setup0.ik_solver.ik(pp.tform_from_pose(goal_pose))
    print(">>> view 4.2: conf", conf)

    arm_base_pose = pp.get_link_pose(robot_setup0.robot, pp.link_from_name(robot_setup0.robot, "ur_arm_base_link"))
    print(">>> view 5: arm base pose", arm_base_pose)

    while True:
        pass
