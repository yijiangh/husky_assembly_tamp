import sys, os, argparse
import time
import random
import socket, json
import struct
from threading import Thread
from plyer import notification

import numpy as np
import pybullet_planning as pp
import pybullet as p

from husky_assembly.optitrack.NatNetClient import NatNetClient
from husky_assembly.optitrack.Utils import print_configuration
from husky_assembly import DATA_DIRECTORY
from tracikpy import TracIKSolver

from compas.robots import RobotModel
from compas_fab.robots import RobotSemantics
from compas_fab.robots import Robot as RobotClass

LOCAL_SERVER = True
CLIENT_IP = '192.168.0.7' # Set to your own IP
LOCAL_SERVER_IP = 'localhost' # '127.0.0.1'
MOCAP_IP = '192.168.0.117'

HUSKY_IP = '192.168.0.113'
HUSKY_UDP_PORT = 65432
HUSKY_TCP_PORT = 12345

HERE = os.path.dirname(__file__)

yup_tform = np.eye(4)
yup_tform[:3,0] = [0, 1, 0]
yup_tform[:3,1] = [0, 0, 1]
yup_tform[:3,2] = [1, 0, 0]
zup_from_yup = pp.pose_from_tform(yup_tform)

HUSKYU_JOINT_NAMES = ['x', 'y', 'theta', 
                      "ur_arm_shoulder_pan_joint", 
                      "ur_arm_shoulder_lift_joint",
                      "ur_arm_elbow_joint", 
                      "ur_arm_wrist_1_joint", 
                      "ur_arm_wrist_2_joint", 
                      "ur_arm_wrist_3_joint" ]
JOINT_JUMP_THRESHOLD = np.pi/3
POS_STEP_SIZE = 0.01
ORI_STEP_SIZE = np.pi/18

name_from_mocap_id = {
    1028 : 'husky0804',
    1011 : 'bar',
    1029 : 'greybox',
}

# goal registar for optitrack to overwrite
rigid_body_poses = {}
arm_joint_state = {}

###################

# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
def receive_rigid_body_frame( new_id, position, rotation ):
    global rigid_body_poses
    rigid_body_poses[new_id] = (position, rotation)
    # print( "Received frame for rigid body", new_id )
    # print( "Received frame for rigid body", new_id," ",position," ",rotation )

def receive_joint_state(socket_server):
    # receive the message from socket and translate them into ROS messages
    global arm_joint_state
    data, _ = socket_server.recvfrom(65507)
    try:
        arm_joint_state = json.loads(data.decode("utf-8"))
    except Exception as e:
        print("ERROR: unable to decode the received joint state")
        print(e)

def socket_recv_thread(socket_server, stop):
    while not stop():
        try:
            receive_joint_state(socket_server)
        except socket.error as msg:
            if stop():
                # print("ERROR: command socket access error occurred:\n  %s" %msg)
                print("shutting down joint data receiving thread")

def send_base_arm_trajectory_command(socket_server, joint_names, joint_positions, time_steps):
    # check if jointPositions and timeSteps have the same size
    if (len(joint_positions) != len(time_steps)):
        print("Error: jointPositions and timeSteps have different sizes")
        return

    traj = [] # create an empty array
    for i in range(len(joint_positions)):
        if (len(joint_positions[i]) != 8):
            print("Error: jointPositions[" + str(i) + "] has length " + str(len(joint_positions[i])) + " instead of 8")
            return
        traj_point = {}
        traj_point["xVel"] = joint_positions[i][0]
        traj_point["angVel"] = joint_positions[i][1]
        traj_point["q1"] = joint_positions[i][2]
        traj_point["q2"] = joint_positions[i][3]
        traj_point["q3"] = joint_positions[i][4]
        traj_point["q4"] = joint_positions[i][5]
        traj_point["q5"] = joint_positions[i][6]
        traj_point["q6"] = joint_positions[i][7]
        traj_point["time_from_start"] = time_steps[i]
        traj.append(traj_point)

    j = {}
    j["trajectory"] = traj
    j["joint_names"] = joint_names

    j_file = json.dumps(j)
    encoded_json = j_file.encode('utf-8')
    msg = struct.pack('>I', len(encoded_json)) + encoded_json
    print("***************************")
    print("Sending goal trajectory with pts = " + str(len(joint_positions)) + " and duration = " + str(time_steps[-1]))
    print('Data size = %d' % len(encoded_json))

    try:
        socket_server.sendall(msg)
    except socket.error as e:
        print("error while sending: %s" %e)

def send_gripper_command(socket_server, gripper_pos):
    assert gripper_pos >= 0 and gripper_pos <= 255
    j = {}
    j["gripper_pos"] = gripper_pos

    j_file = json.dumps(j)
    encoded_json = j_file.encode('utf-8')
    msg = struct.pack('>I', len(encoded_json)) + encoded_json
    print("***************************")
    print("Sending goal gripper pose = " + str(gripper_pos))
    print('Data size = %d' % len(encoded_json))

    try:
        socket_server.sendall(msg)
    except socket.error as e:
        print("error while sending: %s" %e)

########################

def load_robot(ik_from_arm_base=True):
    robot_urdf = os.path.join(DATA_DIRECTORY,'husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf')
    robot_srdf = os.path.join(DATA_DIRECTORY, 'husky_urdf/mt_husky_moveit_config/config/husky.srdf')
    # gripper_obj = os.path.join(DATA_DIRECTORY,'husky_urdf/robotiq_85/meshes/static/robotiq_85_close_20mm.obj')
    gripper_obj = os.path.join(DATA_DIRECTORY,'husky_urdf/robotiq_85/meshes/static/robotiq_85_open.obj')
    # robot_urdf = os.path.join(HERE,'robotiq_85/urdf/robotiq_85_gripper_simple.urdf')
    # robot_urdf = os.path.join(HERE,'mt_husky_dual_ur5_e_moveit_config/urdf/husky_dual_ur5_e.urdf')
    # print(robot_urdf)
    assert os.path.exists(robot_urdf)
    assert os.path.exists(gripper_obj)

    move_group = 'manipulator'
    robot_model = RobotModel.from_urdf_file(robot_urdf)
    robot_semantics = RobotSemantics.from_srdf_file(robot_srdf, robot_model)
    cp_robot = RobotClass(robot_model, semantics=robot_semantics)

    robot = pp.load_pybullet(robot_urdf, fixed_base=False, cylinder=False)

    # TODO get tool def from SRDF
    if not ik_from_arm_base:
        ik_solver = TracIKSolver(robot_urdf, "world_link", "ur_arm_tool0")
    else:
        ik_solver = TracIKSolver(robot_urdf, "ur_arm_base_link", "ur_arm_tool0")
    # pp.camera_focus_on_body(robot)

    # get disabled collision pairs from SRDF
    disabled_self_collision_link_names = robot_semantics.disabled_collisions
    disabled_collisions = get_disabled_collisions(robot, disabled_self_collision_link_names) 

    tool0_pose = pp.get_link_pose(robot, pp.link_from_name(robot, 'ur_arm_tool0'))
    # pp.draw_pose(tool0_pose)
    ee = pp.create_obj(gripper_obj) 
    pp.set_pose(ee, pp.multiply(tool0_pose, pp.Pose(euler=pp.Euler(yaw=-np.pi/2))))
    ee_attachment = pp.create_attachment(robot, pp.link_from_name(robot, 'ur_arm_tool0'), ee)

    # tool0_from_ee = pp.Pose(euler=pp.Euler(yaw=-np.pi/2), point=[0,0,0.138])
    # tcp_pose = pp.multiply(tool0_pose, tool0_from_ee)
    # tcp_pose = pp.get_link_pose(robot, pp.link_from_name(robot, 'central_tcp'))
    # pp.draw_pose(tcp_pose)

    return robot, ee_attachment, ik_solver, disabled_collisions

def get_disabled_collisions(robot, disabled_self_collision_link_names):
    """get robot's link-link tuples disabled from collision checking

    Returns
    -------
    set of int-tuples
        int for link index in pybullet
    """
    return {tuple(pp.link_from_name(robot, link)
                  for link in pair if pp.has_link(robot, link))
                  for pair in disabled_self_collision_link_names}

def get_custom_limits(robot, custom_limits=None):
    """[summary]

    Returns
    -------
    [type]
        {joint index : (lower limit, upper limit)}
    """
    custom_limits = custom_limits or {}
    limits = {pp.joint_from_name(robot, joint): limits
              for joint, limits in custom_limits.items()}
    return limits

def check_path(joints, path, collision_fn=None, jump_threshold=None, diagnosis=False):
    """return False if path is not valid
    """
    joint_jump_thresholds = jump_threshold or [JOINT_JUMP_THRESHOLD for jt in joints]
    for jt1, jt2 in zip(path[:-1], path[1:]):
        delta_j = np.abs(np.array(jt1) - np.array(jt2))
        if any(delta_j > np.array(joint_jump_thresholds)):
            return False
    if collision_fn is not None:
        for q in path:
            if collision_fn(q, diagnosis):
                return False
    return True

def plan_pickup_motion(robot, ik_solver, bar_body, attachments, obstacles, debug=False, ik_from_arm_base=True, disabled_collisions=None):
    # plan a transit motion from init conf to pick_approach conf  
    custom_limits = get_custom_limits(robot, {})
    resolutions = np.ones(6) * 0.05
    disabled_collisions = disabled_collisions or {}
    extra_disabled_collisions = [
        ((robot, pp.link_from_name(robot, 'ur_arm_wrist_3_link')), 
         (attachments[0].child, pp.BASE_LINK)), 
         # pp.link_from_name(ee_body, 'robotiq_85_base_link'))),
        ]

    # joints = pp.get_movable_joints(robot)
    first_joint_id = 3 if ik_from_arm_base else 0
    movable_joints = pp.joints_from_names(robot, HUSKYU_JOINT_NAMES[first_joint_id:])

    sample_fn = pp.get_sample_fn(robot, movable_joints, custom_limits=custom_limits)
    distance_fn = pp.get_distance_fn(robot, movable_joints) #, weights=weights)
    extend_fn = pp.get_extend_fn(robot, movable_joints, resolutions=resolutions)
    transit_collision_fn = pp.get_collision_fn(robot, movable_joints, obstacles=obstacles,
                                                attachments=attachments, 
                                                self_collisions=1,
                                                disabled_collisions=disabled_collisions, extra_disabled_collisions=extra_disabled_collisions,
                                                custom_limits=custom_limits, 
                                                max_distance=0)
    extra_disabled_collisions += [
        ((bar_body, pp.BASE_LINK), 
         (attachments[0].child, pp.BASE_LINK)),
    ]
    approach_collision_fn = pp.get_collision_fn(robot, movable_joints, obstacles=obstacles,
                                                attachments=attachments, 
                                                self_collisions=1,
                                                disabled_collisions=disabled_collisions, extra_disabled_collisions=extra_disabled_collisions,
                                                custom_limits=custom_limits, 
                                                max_distance=0)
    grasp_gen = pp.get_side_cylinder_grasps(bar_body, safety_margin_length=0.5)

    if ik_from_arm_base:
        # world_from_base = pp.pose_from_pose2d(current_base_conf)
        # world_from_arm_base = pp.multiply(world_from_base, base_from_arm_base)
        # * because the base position is controlled by the joystick and updated outside this function
        # we can directly get the arm base link here
        world_from_arm_base = pp.get_link_pose(robot, pp.link_from_name(robot, "ur_arm_base_link"))
        # pp.set_color(robot, [0.5,0.5,0.5, 0.1])
    else:
        world_from_arm_base = pp.unit_pose()
    # pp.draw_pose(world_from_arm_base)

    world_from_object = pp.get_pose(bar_body)
    # tool0_from_ee = pp.Pose(euler=pp.Euler(yaw=-np.pi/2), point=[0,0,0.138])
    tool0_from_ee = pp.Pose(point=[0,0,0.138])

    # * sample grasp and IK, and plan for approach motion
    grasp_attempts = 50
    attach_conf = None
    path = None
    start_conf = pp.get_joint_positions(robot, movable_joints)
    with pp.WorldSaver():
        with pp.LockRenderer():
            for g_id in range(grasp_attempts):
                print('Grasp attempt #{}/{}'.format(g_id, grasp_attempts))
                gripper_from_object = next(grasp_gen)
                world_from_tcp_pose = pp.multiply(world_from_object, pp.invert(gripper_from_object))

                arm_base_from_tcp_pose = pp.multiply(pp.invert(world_from_arm_base), world_from_tcp_pose)
                arm_base_from_tool0 = pp.multiply(arm_base_from_tcp_pose, pp.invert(tool0_from_ee))
                # pp.draw_pose(pp.multiply(world_from_arm_base, arm_base_from_tcp_pose))

                attach_conf = ik_solver.ik(pp.tform_from_pose(arm_base_from_tool0))
                if attach_conf is not None and not approach_collision_fn(attach_conf, diagnosis=debug):
                    # print("solved conf: ", conf)
                    # print("grasp: ", gripper_from_object)

                    # * plan pregrasp motion
                    # move world_from_tool0 in the minus z direction for 0.1m
                    tool0_from_pregrasp = pp.Pose(point=[0,0,-0.1])
                    arm_base_from_pregrasp = pp.multiply(arm_base_from_tool0, tool0_from_pregrasp)

                    approach_path = []
                    pregrasp_poses = list(pp.interpolate_poses(arm_base_from_tool0, arm_base_from_pregrasp, pos_step_size=POS_STEP_SIZE, ori_step_size=ORI_STEP_SIZE))
                    prev_conf = attach_conf
                    for fpose in pregrasp_poses:
                        # pp.draw_pose(fpose)
                        attach_conf = ik_solver.ik(pp.tform_from_pose(fpose), qinit=prev_conf)
                        if attach_conf is None or approach_collision_fn(attach_conf, diagnosis=debug):
                            notify('ik can\'t find an ik solution for approaching')
                            break
                        else:
                            approach_path.append(attach_conf)

                    if len(approach_path) != len(pregrasp_poses) or \
                        not check_path(movable_joints, approach_path, jump_threshold=JOINT_JUMP_THRESHOLD):
                        continue
                    else:
                        print('Pregrasp path found: {} pts'.format(len(approach_path)))
                        # * plan transit motion from current conf to pregrasp conf
                        end_conf = approach_path[-1]
                        # print('start conf: ', start_conf)
                        transit_path = None

                        if pp.check_initial_end(start_conf, end_conf, transit_collision_fn, diagnosis=debug):
                            transit_path = pp.solve_motion_plan(start_conf, end_conf, 
                                                        distance_fn, sample_fn, extend_fn,
                                                        transit_collision_fn,
                                                        algorithm='birrt', 
                                                        max_time=10, 
                                                        max_iterations=20, 
                                                        smooth=20, diagnosis=debug,
                                                        coarse_waypoints=False,
                                                        ) 
                        else:
                            notify('initial and end confs for transit motion are not valid')

                        if transit_path is None:
                            # notify('transit path not found')
                            # return approach_path[::-1]
                            # return None
                            # path = approach_path[::-1]
                            continue
                        else:
                            notify('transit path found: transit {} pts'.format(len(transit_path)))
                        path = transit_path + approach_path[::-1]
                        break
            else:
                notify("no ik solution after {} grasp attempts".format(grasp_attempts))

    return path

def plan_transit_motion(robot, end_conf, attachments, obstacles, debug=False, ik_from_arm_base=True, disabled_collisions=None):
    custom_limits = get_custom_limits(robot, {})
    resolutions = np.ones(6) * 0.05
    disabled_collisions = disabled_collisions or {}
    extra_disabled_collisions = [
        ((robot, pp.link_from_name(robot, 'ur_arm_wrist_3_link')), 
         (attachments[0].child, pp.BASE_LINK)), 
        ]

    first_joint_id = 3 if ik_from_arm_base else 0
    movable_joints = pp.joints_from_names(robot, HUSKYU_JOINT_NAMES[first_joint_id:])

    sample_fn = pp.get_sample_fn(robot, movable_joints, custom_limits=custom_limits)
    distance_fn = pp.get_distance_fn(robot, movable_joints) #, weights=weights)
    extend_fn = pp.get_extend_fn(robot, movable_joints, resolutions=resolutions)

    transit_collision_fn = pp.get_collision_fn(robot, movable_joints, obstacles=obstacles,
                                                attachments=attachments, 
                                                self_collisions=1,
                                                disabled_collisions=disabled_collisions, extra_disabled_collisions=extra_disabled_collisions,
                                                custom_limits=custom_limits, 
                                                max_distance=0)

    transit_path = None
    with pp.WorldSaver():
        with pp.LockRenderer(True):
            # * plan transit motion from current conf to pregrasp conf
            start_conf = pp.get_joint_positions(robot, movable_joints)
            # print('start conf: ', start_conf)

            # new_collision_fn = lambda q, diagnosis=False: collision_fn(q, diagnosis=True)
            if pp.check_initial_end(start_conf, end_conf, transit_collision_fn, diagnosis=debug):
                transit_path = pp.solve_motion_plan(start_conf, end_conf, 
                                            distance_fn, sample_fn, extend_fn,
                                            transit_collision_fn,
                                            algorithm='birrt', 
                                            max_time=10, 
                                            max_iterations=20, 
                                            smooth=20, diagnosis=debug,
                                            coarse_waypoints=False,
                                            ) 
            else:
                notify('initial and end conf not valid')
            if transit_path is None:
                notify('transit path not found')
            else:
                notify('transit path found: transit {} pts'.format(len(transit_path)))

    return transit_path

def notify(msg):
    print(msg)
    notification.notify(
        title='husky_assembly',
        message=msg,
        app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
        timeout=2,  # seconds
    )

def align_joint_conf_by_joint_names(source_joint_names, target_conf, target_joint_names):
    return [target_conf[target_joint_names.index(joint_name)] for joint_name in source_joint_names]

def save_joint_state_to_json():
    global arm_joint_state
    file_path = os.path.join(HERE, 'arm_joint_state.json')
    with open(file_path, 'w') as f:
        json.dump(arm_joint_state, f, indent=4)
    notify('Arm joint state saved to {}'.format(file_path))

def read_saved_joint_state_from_json():
    file_path = os.path.join(HERE, 'arm_joint_state.json')
    if not os.path.exists(file_path):
        notify('no saved arm joint state found at {}'.format(file_path))
        return None
    with open(file_path, 'r') as f:
        saved_arm_joint_state = json.load(f)
    notify('Saved arm joint state read from {}'.format(file_path))
    return saved_arm_joint_state

#####################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teleopt_target', action='store_true',
                        help='')
    parser.add_argument('--connect_to_mocap', action='store_true',
                        help='connect to mocap.')
    parser.add_argument('--connect_to_hw', action='store_true',
                        help='connect to robot hardware.')
    parser.add_argument('--debug', action='store_true',
                        help='')
    args = parser.parse_args()

    # * create a new NatNet client
    if args.connect_to_mocap:
        mocap_client = NatNetClient()
        mocap_client.set_client_address(CLIENT_IP)
        mocap_client.set_server_address(MOCAP_IP)
        mocap_client.set_use_multicast(False)
        mocap_client.print_level = 0
        # Configure the streaming client to call our rigid body handler on the emulator to send data out.
        mocap_client.rigid_body_listener = receive_rigid_body_frame

    # * create a new Husky client
    if args.connect_to_hw:
        jt_socket_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        jt_socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # ! on the husky side, we set it to always send to the same port to the host
        jt_socket_server.bind((CLIENT_IP if not LOCAL_SERVER else LOCAL_SERVER_IP, HUSKY_UDP_PORT))
        jt_socket_server.settimeout(0.001)

        # * a thread to receive data from the husky
        stop_thread = False
        joint_state_stream_thread = Thread(target=socket_recv_thread, args=(jt_socket_server, lambda : stop_thread))
        joint_state_stream_thread.daemon = True
        joint_state_stream_thread.start()

        # # * TCP socket to send trajectory to the husky
        traj_socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        traj_socket_client.connect((HUSKY_IP if not LOCAL_SERVER else LOCAL_SERVER_IP, HUSKY_TCP_PORT))

    # * start pybullet simulator
    pp.connect(use_gui=True, shadows=True, color=[0.9, 0.9, 1.0])
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=pp.CLIENT)
    # pp.set_camera(np.deg2rad(92.0), np.deg2rad(-85), 5.20)
    pp.set_camera(92.0, -85, 5.20)

    # * Control UI
    traj_param_slider = p.addUserDebugParameter("trajectory playback", 0.0, 1.0, 0.0)
    #  For a button, the value of getUserDebugParameter for a button increases 1 at each button press.
    plan_button = p.addUserDebugParameter("plan", 1, 0, 0)
    prev_plan_button_value = p.readUserDebugParameter(plan_button)

    reverse_traj_button = p.addUserDebugParameter("reverse traj", 1, 0, 0)
    prev_reverse_value = p.readUserDebugParameter(reverse_traj_button)

    save_state_button = p.addUserDebugParameter("save arm joint state", 1, 0, 0)
    prev_save_state_button = p.readUserDebugParameter(save_state_button)

    plan_to_saved_state_button = p.addUserDebugParameter("plan to saved state", 1, 0, 0)
    prev_plan_to_saved_state_button = p.readUserDebugParameter(plan_to_saved_state_button)

    execute_button = p.addUserDebugParameter("execute", 1, 0, 0)
    prev_execute_button_value = p.readUserDebugParameter(execute_button)

    open_gripper_button = p.addUserDebugParameter("open gripper", 1, 0, 0)
    prev_open_gripper_value = p.readUserDebugParameter(open_gripper_button)

    close_gripper_button = p.addUserDebugParameter("close gripper", 1, 0, 0)
    prev_close_gripper_value = p.readUserDebugParameter(close_gripper_button)

    # TODO slider to control the time gap between each trajectory point

    ik_from_arm_base = 1
    teleop_target = args.teleopt_target

    # * load all robots and objects
    pp.draw_pose(pp.unit_pose(), 0.5)
    with pp.LockRenderer():
        p.loadMJCF(os.path.join(HERE, "plane.xml"))
        plane = pp.create_plane(color=[0.9, 0.9, 1.0, 0.0])
        bar = pp.create_cylinder(radius=0.01, height=1.0, color=pp.BROWN)
        box = pp.create_box(0.6, 0.4, 0.45, color=pp.apply_alpha(pp.GREY, 1))

        with pp.HideOutput():
            robot, ee_attachment, ik_solver, disabled_collisions = load_robot(ik_from_arm_base)

            # ! a shadow robot for displaying the trajectory
            shadow_robot, shadow_ee_attachment, _, _ = load_robot()
            shadow_color = [0.5, 0.5, 0.5, 0.7]
            pp.set_color(shadow_robot, shadow_color)
            pp.set_color(shadow_ee_attachment.child, shadow_color)

            goal_robot = None
            goal_ee_attachment = None
            if teleop_target:
                goal_robot, goal_ee_attachment, _, _ = load_robot()
                goal_color = [0, 0, 1.0, 0.7]
                pp.set_color(goal_robot, goal_color)

    first_joint_id = 3 if ik_from_arm_base else 0
    planned_joint_names = HUSKYU_JOINT_NAMES[first_joint_id:]
    planned_joints = pp.joints_from_names(robot, planned_joint_names)
    base_cmd_names = ['xVel', 'angVel']

    recorded_conf = (0,0,0,-1.3227758407592773, -1.5873312950134277, -1.5211923122406006, 0.0, 0.0, 0.0)
    if not args.connect_to_mocap :
        # a recorded pose for debuggging purpose
        temp_bar_pose = ((-1.062444806098938, 0.19626910984516144, 0.6585784554481506), (0.8137449622154236, -0.1838780641555786, 0.5276390910148621, -0.1600152850151062))
        pp.set_pose(bar, temp_bar_pose)

        pp.set_joint_positions(robot, pp.joints_from_names(robot, HUSKYU_JOINT_NAMES), recorded_conf)
        pp.set_joint_positions(shadow_robot, pp.joints_from_names(robot, HUSKYU_JOINT_NAMES), recorded_conf)
        # ee_attachment.assign()
        # shadow_ee_attachment.assign()
    else:
        if arm_joint_state:
            recorded_conf = align_joint_conf_by_joint_names(planned_joint_names, arm_joint_state['position'], arm_joint_state['name'])

    if teleop_target:
        joint_sliders = []
        for j, initial_v in zip(planned_joints, recorded_conf):
            lower, upper = pp.get_joint_limits(robot, j)
            joint_sliders.append(p.addUserDebugParameter(pp.get_joint_name(robot, j).decode("utf-8"), 
                                                         lower, upper, initial_v))
            pp.set_joint_position(shadow_robot, j, initial_v)
            pp.set_joint_position(goal_robot, j, initial_v)
        prev_joint_slider_values = [p.readUserDebugParameter(js) for js in joint_sliders]

    obstacles = [plane, box]

    rb_from_name = {
        'bar': bar,
        'husky0804': robot,
        'greybox': box,
    }

    # try:
    if True:
        # Start up the streaming client now that the callbacks are set up.
        # This will run perpetually, and operate on a separate thread.
        is_looping = False
        if args.connect_to_mocap:
            is_running = mocap_client.run()
            print_configuration(mocap_client)
            print("\n")
            if not is_running:
                print("ERROR: Could not start streaming client.")
                try:
                    sys.exit(1)
                except SystemExit:
                    print("...")
                finally:
                    print("exiting")

            is_looping = True
            time.sleep(1)
            if not mocap_client.connected():
                print("ERROR: Could not connect properly.  Check that Motive streaming is on.")
                try:
                    sys.exit(2)
                except SystemExit:
                    print("...")
                finally:
                    print("exiting")

        is_looping = is_looping | ~args.connect_to_mocap

        prev_handle = []
        planned_trajectory = None
        base_joints = pp.joints_from_names(robot, HUSKYU_JOINT_NAMES[:3])
        while is_looping:
            if prev_handle:
                pp.remove_handles(prev_handle)
                prev_handle = []

            # * mocap position update
            if args.connect_to_mocap:
                for mocap_id, name in name_from_mocap_id.items():
                    if mocap_id in rigid_body_poses:
                        yup_from_rb = rigid_body_poses[mocap_id]
                        zup_from_rb = pp.multiply(zup_from_yup, yup_from_rb)
                        if name == 'husky0804':
                            yup_tform = pp.tform_from_pose(zup_from_rb)
                            zup_tform = np.copy(yup_tform)
                            zup_tform[:3,0] = yup_tform[:3,2]
                            zup_tform[:3,1] = yup_tform[:3,0]
                            zup_tform[:3,2] = yup_tform[:3,1]
                            zup_from_rb = pp.pose_from_tform(zup_tform)
                            # estimation has some noise in roll and pitch
                            base_conf = pp.pose2d_from_pose(zup_from_rb, tolerance=2e-2)
                            pp.set_joint_positions(robot, base_joints, base_conf)
                            pp.set_joint_positions(shadow_robot, base_joints, base_conf)
                            if teleop_target:
                                pp.set_joint_positions(goal_robot, base_joints, base_conf)
                        else:
                            rb = rb_from_name[name]
                            # TODO change to set_joint_positions
                            pp.set_pose(rb, zup_from_rb)
                        # prev_handle.extend(pp.draw_pose(zup_from_rb))

            # * set the husky arm joint positions to the slider value
            # only update when the slider value changes
            if teleop_target:
                current_joint_slider_values = [p.readUserDebugParameter(js) for js in joint_sliders]
                if current_joint_slider_values != prev_joint_slider_values:
                    pp.set_joint_positions(goal_robot, planned_joints, current_joint_slider_values)
                    goal_ee_attachment.assign()

                prev_joint_slider_values = current_joint_slider_values

            # * joint state update
            if args.connect_to_hw:
                if arm_joint_state:
                    joints = pp.joints_from_names(robot, arm_joint_state['name'])
                    pp.set_joint_positions(robot, joints, arm_joint_state['position'])

            ee_attachment.assign()
            shadow_ee_attachment.assign()
            # tcp_pose = pp.get_link_pose(robot, pp.link_from_name(robot, 'bar_tcp'))
            # prev_handle.extend(pp.draw_pose(tcp_pose))

            current_plan_button_reading = p.readUserDebugParameter(plan_button)
            if current_plan_button_reading > prev_plan_button_value:
                if teleop_target:
                    planned_trajectory = plan_transit_motion(robot, current_joint_slider_values, 
                                                             [ee_attachment], 
                                                             obstacles + [bar], 
                                                             debug=args.debug, ik_from_arm_base=ik_from_arm_base, disabled_collisions=disabled_collisions)
                else:
                    # * plan the grasp, IK, and pick-up motion
                    planned_trajectory = plan_pickup_motion(robot, ik_solver, bar, 
                                                            [ee_attachment], obstacles + [bar], 
                                                            ik_from_arm_base=ik_from_arm_base, 
                                                            disabled_collisions=disabled_collisions,
                                                            debug=args.debug)
            prev_plan_button_value = current_plan_button_reading

            current_plan_to_saved_state_button_reading = p.readUserDebugParameter(plan_to_saved_state_button)
            if current_plan_to_saved_state_button_reading > prev_plan_to_saved_state_button:
                saved_joint_state = read_saved_joint_state_from_json()
                if saved_joint_state:
                    saved_conf = align_joint_conf_by_joint_names(planned_joint_names, saved_joint_state['position'], saved_joint_state['name'])
                    planned_trajectory = plan_pickup_motion(robot, ik_solver, bar, 
                                                            [ee_attachment], obstacles + [bar], 
                                                            ik_from_arm_base=ik_from_arm_base, 
                                                            disabled_collisions=disabled_collisions,
                                                            debug=args.debug,
                                                            teleop_goal_conf=saved_conf)
            prev_plan_to_saved_state_button = current_plan_to_saved_state_button_reading

            current_save_state_reading = p.readUserDebugParameter(save_state_button)
            if current_save_state_reading > prev_save_state_button:
                save_joint_state_to_json()
            prev_save_state_button = current_save_state_reading

            if planned_trajectory:
                # set the shadow robot to the slider value
                traj_param_value = p.readUserDebugParameter(traj_param_slider)
                traj_idx = int(traj_param_value * (len(planned_trajectory) - 1))
                traj_pose = planned_trajectory[traj_idx]
                pp.set_joint_positions(shadow_robot, planned_joints, traj_pose)
                shadow_ee_attachment.assign()

            current_reverse_value = p.readUserDebugParameter(reverse_traj_button)
            if current_reverse_value > prev_reverse_value:
                planned_trajectory = planned_trajectory[::-1]
                prev_reverse_value = current_reverse_value
                
            # * if execute button is pressed, execute the planned trajectory
            current_execute_button_reading = p.readUserDebugParameter(execute_button)
            if planned_trajectory and current_execute_button_reading > prev_execute_button_value:
                # padd each trajectory point in planned_trajectory with two zeros at the beginning
                padded_traj = np.concatenate([np.zeros((len(planned_trajectory), 2)), np.array(planned_trajectory)], axis=1)

                # time_from_start = np.linspace(0.0, 5.0, len(planned_trajectory))
                time_from_start = [i * 0.2 for i in range(len(planned_trajectory))]

                send_base_arm_trajectory_command(traj_socket_client, base_cmd_names + planned_joint_names, padded_traj, time_from_start)
            prev_execute_button_value = current_execute_button_reading

            # * open gripper execution
            current_openg_button_reading = p.readUserDebugParameter(open_gripper_button)
            if current_openg_button_reading > prev_open_gripper_value:
                send_gripper_command(traj_socket_client, 255)
            prev_open_gripper_value = current_openg_button_reading

            # * close gripper execution
            current_closeg_button_reading = p.readUserDebugParameter(close_gripper_button)
            if current_closeg_button_reading > prev_close_gripper_value:
                # diameter of the bar is 0.02
                send_gripper_command(traj_socket_client, 0.02)
            prev_close_gripper_value = current_closeg_button_reading

            time.sleep(0.01)

            # pp.wait_if_gui()

    # except KeyboardInterrupt:
    #     print('\n! Received keyboard interrupt, quitting threads.\n')

    # finally:
    #     stop_thread = True

    #     if args.connect_to_hw:
    #         joint_state_server.close()
    #         joint_state_stream_thread.join()

    #     if args.connect_to_mocap:
    #         mocap_client.shutdown()

    #     if pp.is_connected():
    #         pp.disconnect()
    #     sys.exit()


if __name__ == "__main__":
    main()