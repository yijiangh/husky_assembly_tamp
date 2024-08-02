import numpy as np
import random
from itertools import islice
import pybullet_planning as pp
from termcolor import cprint

from pybullet_planning import multiply, Pose, Euler, Point, invert, interpolate_poses, get_distance

from robot_setup import RobotSetup
from collision import Grasp

##############################

ENABLE_SELF_COLLISIONS = True
assert ENABLE_SELF_COLLISIONS
IK_MAX_ATTEMPTS = 1
PREGRASP_MAX_ATTEMPTS = 100
GRASP_MAX_ATTEMPTS = 100

ALLOWABLE_BAR_COLLISION_DEPTH = 1e-3

# pregrasp delta sample
EPSILON = 0.05
ANGLE = np.pi/3

# pregrasp interpolation
POS_STEP_SIZE = 0.01
ORI_STEP_SIZE = np.pi/18

RETREAT_DISTANCE = 0.07
MAX_DISTANCE = 0.0

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
    longitude_x = Pose(euler=Euler(pitch=np.pi/2))
    def gen_fn(index):
        # can get from aabb as well
        bar_length = get_distance(*element_from_index[index].axis_endpoints)
        while True:
            # translation along the longitude axis
            slide_dist = random.uniform(-bar_length/2+safety_margin_length, bar_length/2-safety_margin_length)
            translate_along_x_axis = Pose(point=Point(slide_dist,0,0))

            for j in range(1 + reverse_grasp):
                # the base pi/2 is to make y align with the longitude axis, conforming to the convention (see image in the doc)
                # flip the gripper, gripper symmetry
                rotate_around_z = Pose(euler=[0, 0, np.pi/2 + j * np.pi])

                object_from_gripper = multiply(longitude_x, translate_along_x_axis, \
                    rotate_around_z, tool_pose)
                yield Grasp(index, invert(object_from_gripper)),
    return gen_fn

######################################

def get_element_body_in_goal_pose(element_from_index, printed):
    for e in list(printed):
        pp.set_pose(element_from_index[e].body, element_from_index[e].goal_pose)
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
    lower = [-epsilon]*3 + [-angle]*3
    upper = [epsilon]*3 + [angle]*3
    for [x, y, z, roll, pitch, yaw] in pp.interval_generator(lower, upper): # halton?
        pose = Pose(point=[x,y,z], euler=Euler(roll=roll, pitch=pitch, yaw=yaw))
        yield pose

def get_pregrasp_gen_fn(element_from_index, fixed_obstacles, max_attempts=PREGRASP_MAX_ATTEMPTS, collision=True, teleops=False):
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
    pose_gen = get_delta_pose_generator()

    def gen_fn(index, pose, printed, diagnosis=False):
        body = element_from_index[index].body
        pp.set_pose(body, pose)

        # element_obstacles = {element_from_index[e].body for e in list(printed)}
        element_obstacles = get_element_body_in_goal_pose(element_from_index, printed)
        obstacles = set(fixed_obstacles) | element_obstacles
        if not collision:
            obstacles = set()
        ee_collision_fn = pp.get_floating_body_collision_fn(body, obstacles, max_distance=MAX_DISTANCE)

        for _ in range(max_attempts):
            delta_pose = next(pose_gen)
            offset_pose = multiply(pose, delta_pose)
            is_colliding = False
            if not teleops:
                offset_path = list(interpolate_poses(offset_pose, pose, pos_step_size=POS_STEP_SIZE, ori_step_size=ORI_STEP_SIZE))
            else:
                offset_path = [offset_pose, pose]
            for p in offset_path: # [:-1]
                # TODO: if colliding at the world_from_bar pose, use local velocity + normal check
                # TODO: normal can be derived from
                if ee_collision_fn(p, diagnosis=diagnosis):
                # if element_robot_collision_fnpose2conf(p)):
                    is_colliding = True
                    break
            if not is_colliding:
                yield offset_path,
                break
        else:
            yield None,
    return gen_fn

def compute_place_path(robot_setup, pregrasp_poses, grasp, index, element_from_index, obstacles,
                       verbose=False, diagnosis=False, retreat_vector=np.array([0, 0, -1]), teleops=False, gantry_sample_fn=None):
    """Give the grasp and EE workspace poses, compute cartesian planning for pre-detach ~ detach ~ post-detach process.
    """
    robot = robot_setup.robot
    tool0_from_ee = robot_setup.tool0_from_ee
    ik_solver = robot_setup.ik_solver
    control_joints = robot_setup.control_joints

    body = element_from_index[index].body
    pre_attach_poses = [multiply(bar_pose, invert(grasp)) for bar_pose in pregrasp_poses]
    attach_pose = pre_attach_poses[-1]
    pre_attach_pose = pre_attach_poses[0]
    post_attach_pose = multiply(attach_pose, (retreat_vector, pp.unit_quat()))
    post_attach_poses = list(interpolate_poses(attach_pose, post_attach_pose, pos_step_size=POS_STEP_SIZE, ori_step_size=ORI_STEP_SIZE))

    # * attach IK
    world_from_tool0 = pp.multiply(attach_pose, pp.invert(tool0_from_ee))
    pp.draw_pose(world_from_tool0)

    attach_conf = ik_solver.ik(pp.tform_from_pose(world_from_tool0))

    if (attach_conf is None):
        if verbose : print('attach ik failure.')
        # handles = draw_pose(attach_pose)
        # wait_if_gui()
        # remove_handles(handles)
        return None

    pp.set_joint_positions(robot, control_joints, attach_conf)
    robot_setup.ee_attachment.assign()
    pp.set_pose(body, pregrasp_poses[-1])
    attachment = pp.create_attachment(robot, robot_setup.tool_link, body)

    # attachment is assumed to be empty here, since pregrasp sampler guarantees that
    collision_fn = pp.get_collision_fn(robot, robot_setup.control_joints, obstacles=obstacles, attachments=
                                       [attachment],
                                        self_collisions=ENABLE_SELF_COLLISIONS,
                                        disabled_collisions=robot_setup.disabled_collisions,
                                        # custom_limits=get_custom_limits(robot),
                                        max_distance=MAX_DISTANCE)

    if collision_fn(attach_conf, diagnosis):
        if verbose : print('attach collision failure.')
        return None
    # set_color(body, GREEN)
    # wait_if_gui()

    return [attach_conf]

###############################

def get_place_gen_fn(robot_setup: RobotSetup, element_from_index, fixed_obstacles, 
                     collisions=True,
                     max_attempts=IK_MAX_ATTEMPTS, 
                     max_grasp=GRASP_MAX_ATTEMPTS, 
                     allow_failure=False, 
                     verbose=False, 
                     teleops=False):
    if not collisions:
        precompute_collisions = False

    robot = robot_setup.robot
    tool0_from_ee = robot_setup.tool0_from_ee

    # conditioned sampler
    pregrasp_gen_fn = get_pregrasp_gen_fn(element_from_index, fixed_obstacles, collision=collisions, teleops=teleops) # max_attempts=max_attempts,

    retreat_distance = RETREAT_DISTANCE
    retreat_vector = retreat_distance*np.array([0, 0, -1])

    def gen_fn(element, assembled=[], diagnosis=False):
        element_obstacles = get_element_body_in_goal_pose(element_from_index, assembled)
        obstacles = set(fixed_obstacles) | element_obstacles
        if not collisions:
            obstacles = set()
        elements_order = [e for e in element_from_index if (e != element) and (element_from_index[e].body not in obstacles)]

        grasp_gen = pp.get_side_cylinder_grasps(element_from_index[element].body, safety_margin_length=0.5)
        # keep track of sampled traj, prune newly sampled one with more collided element
        element_goal_pose = element_from_index[element].goal_pose
        trajectories = []
        for attempt, grasp in enumerate(islice(grasp_gen, max_grasp)):
            # * ik iterations, usually 1 is enough
            for _ in range(max_attempts):
                # ! when used in pddlstream (except incremental_sm), the pregrasp sampler assumes no elements assembled at all time
                pregrasp_poses, = next(pregrasp_gen_fn(element, element_goal_pose, assembled, diagnosis=diagnosis))
                if not pregrasp_poses:
                    if verbose : print('pregrasp failure.')
                    continue

                command = compute_place_path(robot_setup, pregrasp_poses, grasp, element, element_from_index, obstacles, verbose=verbose, diagnosis=diagnosis, retreat_vector=retreat_vector, teleops=teleops)
                if command is None:
                    continue

                trajectories.append(command)
                if command not in trajectories:
                    continue

                # if verbose:
                cprint('Place E#{} | Attempts: {} | Trajectories: {}'.format(element, attempt, len(trajectories)), 'green')

                yield command,
                break
        else:
            if verbose:
                cprint('E#{} | Attempts: {} | Max attempts exceeded!'.format(element, max_grasp), 'red')

            if allow_failure:
                yield None,
            else:
                return
    return gen_fn