import sys, os, argparse
import time
import random
import socket, json
from threading import Thread

import numpy as np
import pybullet_planning as pp
import pybullet as p

from husky_assembly.optitrack.NatNetClient import NatNetClient
from husky_assembly.optitrack.Utils import print_configuration
from husky_assembly import DATA_DIRECTORY
from tracikpy import TracIKSolver

# from compas.robots import RobotModel
# from compas_fab.robots import RobotSemantics
# from compas_fab.robots import Robot as RobotClass

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
POS_STEP_SIZE = 0.001
ORI_STEP_SIZE = np.pi/18

name_from_mocap_id = {
    1028 : 'husky0804',
    1011 : 'bar',
}

# goal registar for optitrack to overwrite
rigid_body_poses = {}
arm_joint_state = {}

# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
def receive_rigid_body_frame( new_id, position, rotation ):
    global rigid_body_poses
    rigid_body_poses[new_id] = (position, rotation)
    # print( "Received frame for rigid body", new_id )
    # print( "Received frame for rigid body", new_id," ",position," ",rotation )

def receive_joint_state(socket_server):
    # receive the message from socket and translate them into ROS messages
    global arm_joint_state
    data, CLIENT_IP = socket_server.recvfrom(65507)
    arm_joint_state = json.loads(data.decode("utf-8"))

def socket_recv_thread(socket_server, stop):
    while not stop():
        try:
            receive_joint_state(socket_server)
        except socket.error as msg:
            if stop():
                # print("ERROR: command socket access error occurred:\n  %s" %msg)
                print("shutting down joint data receiving thread")

########################

def load_robot(ik_from_arm_base=True):
    robot_urdf = os.path.join(DATA_DIRECTORY,'husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf')
    gripper_obj = os.path.join(DATA_DIRECTORY,'husky_urdf/robotiq_85/meshes/static/robotiq_85_close_20mm.obj')
    # robot_urdf = os.path.join(HERE,'robotiq_85/urdf/robotiq_85_gripper_simple.urdf')
    # robot_urdf = os.path.join(HERE,'mt_husky_dual_ur5_e_moveit_config/urdf/husky_dual_ur5_e.urdf')
    # print(robot_urdf)
    assert os.path.exists(robot_urdf)
    assert os.path.exists(gripper_obj)
    
    robot = pp.load_pybullet(robot_urdf, fixed_base=False, cylinder=True)

    # TODO get tool def from SRDF
    if not ik_from_arm_base:
        ik_solver = TracIKSolver(robot_urdf, "world_link", "ur_arm_tool0")
    else:
        ik_solver = TracIKSolver(robot_urdf, "ur_arm_base_link", "ur_arm_tool0")
    # pp.camera_focus_on_body(robot)

    # TODO get disabled collision pairs from SRDF
    # pp.dump_body(robot)
    # base_bb = pp.create_box(0.9864, 0.6851, 0.3767)
    # pp.set_pose(base_bb, pp.Pose(point=[0,0,0.18835]))

    # joints = pp.get_movable_joints(robot)
    # for j in joints:
    #     child_link = pp.child_link_from_joint(j)
    #     # print('Joint {} - Child {}'.format(pp.get_joint_name(robot, j), pp.get_link_name(robot, child_link)))
    #     link_pose = pp.get_link_pose(robot, child_link)
    #     pp.draw_pose(link_pose)

    tool0_pose = pp.get_link_pose(robot, pp.link_from_name(robot, 'ur_arm_tool0'))
    # pp.draw_pose(tool0_pose)

    ee = pp.create_obj(gripper_obj) 
    pp.set_pose(ee, pp.multiply(tool0_pose, pp.Pose(euler=pp.Euler(yaw=-np.pi/2))))
    ee_attachment = pp.create_attachment(robot, pp.link_from_name(robot, 'ur_arm_tool0'), ee)

    # tool0_from_ee = pp.Pose(euler=pp.Euler(yaw=-np.pi/2), point=[0,0,0.138])
    # tcp_pose = pp.multiply(tool0_pose, tool0_from_ee)
    # tcp_pose = pp.get_link_pose(robot, pp.link_from_name(robot, 'central_tcp'))
    # pp.draw_pose(tcp_pose)

    return robot, ee_attachment, ik_solver

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

def plan_pickup_motion(robot, ik_solver, current_conf, bar_body, attachments, obstacles, debug=False, ik_from_arm_base=True):
    # plan a transit motion from init conf to pick_approach conf  
    custom_limits = get_custom_limits(robot, {})
    resolutions = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    disabled_collisions = {}
    # extra_disabled_collisions = {}
    extra_disabled_collisions =[
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
    collision_fn = pp.get_collision_fn(robot, movable_joints, obstacles=obstacles,
                                       attachments=attachments, 
                                       self_collisions=True,
                                       disabled_collisions=disabled_collisions, extra_disabled_collisions=extra_disabled_collisions,
                                       custom_limits=custom_limits, 
                                       max_distance=0)
    grasp_gen = pp.get_side_cylinder_grasps(bar_body)

    if ik_from_arm_base:
        # world_from_base = pp.pose_from_pose2d(current_base_conf)
        # world_from_arm_base = pp.multiply(world_from_base, base_from_arm_base)
        # * because the base position is controlled by the joystick and updated outside this function
        # we can directly get the arm base link here
        world_from_arm_base = pp.get_link_pose(robot, pp.link_from_name(robot, "ur_arm_base_link"))
        # pp.set_color(robot, [0.5,0.5,0.5, 0.1])
    else:
        world_from_arm_base = pp.unit_pose()
    pp.draw_pose(world_from_arm_base)

    world_from_object = pp.get_pose(bar_body)
    # tool0_from_ee = pp.Pose(euler=pp.Euler(yaw=-np.pi/2), point=[0,0,0.138])
    tool0_from_ee = pp.Pose(point=[0,0,0.138])

    # * sample grasp and IK
    conf = None
    with pp.WorldSaver():
        for _ in range(50):
            gripper_from_object = next(grasp_gen)
            world_from_tcp_pose = pp.multiply(world_from_object, pp.invert(gripper_from_object))

            arm_base_from_tcp_pose = pp.multiply(pp.invert(world_from_arm_base), world_from_tcp_pose)
            arm_base_from_tool0 = pp.multiply(arm_base_from_tcp_pose, pp.invert(tool0_from_ee))
            # pp.draw_pose(pp.multiply(world_from_arm_base, arm_base_from_tcp_pose))

            conf = ik_solver.ik(pp.tform_from_pose(arm_base_from_tool0))
            if conf is not None and not collision_fn(conf, diagnosis=debug):
                # print("solved conf: ", conf)
                # print("grasp: ", gripper_from_object)
                break
        else:
            print("no ik solution")
            return

    # * update robot state in sim
    pp.set_joint_positions(robot, movable_joints, conf)
    for attachment in attachments:
        attachment.assign()

    # path = []
    # base_pose = pp.pose_from_pose2d(conf[:3])
    # print("converted base conf: ", pp.pose2d_from_pose(base_pose))

        # pp.wait_if_gui()

    # with pp.LockRenderer():
    #     if pp.check_initial_end(start_conf, end_conf, collision_fn, diagnosis=debug):
    #         transit_path = pp.birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn,
    #                      restarts=50, iterations=100, smooth=True, max_time=10)
    #     assert transit_path
    #     path.append(transit_path)

    #     pp.set_joint_positions(robot, joints, PICKUP_CONF)
    #     pickup_pose = pp.get_link_pose(robot, tool_attach_link)

    #     pp.set_joint_positions(robot, joints, PICKUP_APPROACH_CONF)
    #     offset_pose = pp.get_link_pose(robot, tool_attach_link)

    #     approach_path = None
    #     pickup_poses = list(pp.interpolate_poses(offset_pose, pickup_pose, pos_step_size=POS_STEP_SIZE, ori_step_size=ORI_STEP_SIZE))
    #     approach_path = []
    #     for fpose in pickup_poses:
    #         pp.draw_pose(fpose)
    #         pb_q = pp.inverse_kinematics(robot, tool_attach_link, fpose)
    #         if pb_q is None:
    #             print('pb ik can\'t find an ik solution')
    #             pp.wait_for_user('Check pose, IK failed.')
    #         else:
    #             approach_path.append(pb_q)

    #     if not check_path(joints, approach_path, collision_fn=collision_fn, jump_threshold=None, diagnosis=args.debug):
    #         approach_path = None
    #     assert approach_path is not None
    #     path.append(approach_path)
    #     path.append(approach_path[::-1])
    #     path.append(transit_path[::-1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable_mocap_tracking', action='store_true',
                        help='Disable mocap connection.')
    parser.add_argument('--disable_joint_tracking', action='store_true',
                        help='Disable mocap connection.')
    parser.add_argument('--debug', action='store_true',
                        help='')
    args = parser.parse_args()

    # * create a new NatNet client
    if not args.disable_mocap_tracking:
        CLIENT_IP = '192.168.0.7' # Set to your own IP
        MOCAP_IP = '192.168.0.117'
        HUSKY_IP = '192.168.131.9'
        mocap_client = NatNetClient()
        mocap_client.set_client_address(CLIENT_IP)
        mocap_client.set_server_address(MOCAP_IP)
        mocap_client.set_use_multicast(False)
        mocap_client.print_level = 0
        # Configure the streaming client to call our rigid body handler on the emulator to send data out.
        mocap_client.rigid_body_listener = receive_rigid_body_frame

    # * create a new Husky client
    if not args.disable_joint_tracking:
        # HOST = '192.168.0.113'  # Standard loopback interface address (localhost)
        PORT = 65432  # Port to listen on (non-privileged ports are > 1023)
        joint_state_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        joint_state_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # on the husky side, we set it to always send to the same port to the host
        joint_state_server.bind((CLIENT_IP, PORT))
        joint_state_server.settimeout(0.001)

        # * a thread to receive data from the husky
        stop_thread = False
        joint_state_stream_thread = Thread(target=socket_recv_thread, args=(joint_state_server, lambda : stop_thread))
        joint_state_stream_thread.daemon = True
        joint_state_stream_thread.start()

    # * start pybullet simulator
    pp.connect(use_gui=True, shadows=True, color=[0.9, 0.9, 1.0])
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=pp.CLIENT)

    # * Control UI
    traj_param_slider = p.addUserDebugParameter("trajectory playback", 0.0, 1.0, 0.0)
    #  For a button, the value of getUserDebugParameter for a button increases 1 at each button press.
    plan_button = p.addUserDebugParameter("plan", 1, 0, 0)
    execute_button = p.addUserDebugParameter("execute", 1, 0, 0)
    prev_plan_button_value = p.readUserDebugParameter(plan_button)
    prev_execute_button_value = p.readUserDebugParameter(execute_button)
    joint_sliders = []
    joint_sliders.append(p.addUserDebugParameter(HUSKYU_JOINT_NAMES[0], -3, 3, 0))
    joint_sliders.append(p.addUserDebugParameter(HUSKYU_JOINT_NAMES[1], -3, 3, 0))
    joint_sliders.append(p.addUserDebugParameter(HUSKYU_JOINT_NAMES[2], 0, np.pi*2, 0))

    ik_from_arm_base = 1

    # * load all robots and objects
    pp.draw_pose(pp.unit_pose(), 1.0)
    with pp.LockRenderer():
        p.loadMJCF(os.path.join(HERE, "plane.xml"))
        # pp.create_plane(color=[0.9, 0.9, 1.0])
        bar = pp.create_cylinder(radius=0.01, height=1.0)
        with pp.HideOutput():
            robot, ee_attachment, ik_solver = load_robot(ik_from_arm_base)

            # ! a shadow robot for displaying the trajectory
            # shadow_robot, shadow_ee_attachment, _ = load_robot()
            # shadow_color = [0.5, 0.5, 0.5, 0.3]
            # pp.set_color(shadow_robot, shadow_color)
            # pp.set_color(shadow_ee_attachment.child, shadow_color)

    obstacles = []

    rb_from_name = {
        'bar': bar,
        'husky0804': robot,
    }

    if args.disable_mocap_tracking:
        # a recorded pose for debuggging purpose
        temp_bar_pose = ((-1.062444806098938, 0.19626910984516144, 0.6585784554481506), (0.8137449622154236, -0.1838780641555786, 0.5276390910148621, -0.1600152850151062))
        pp.set_pose(bar, temp_bar_pose)
        # base_conf = (-1.8195197415944886, -0.00027021797075961755, -1.2547157015094492)
        # pp.set_joint_positions(robot, pp.joints_from_names(robot, HUSKYU_JOINT_NAMES[:3]), base_conf)

# solved conf:  [-1.81951974e+00 -2.70217971e-04 -5.86842433e+37 -5.42704427e+00
#  -1.67372424e+00 -1.34852926e+00  2.65833299e+00  4.13893596e+00
#   1.78973138e+00]
# grasp:  ((0.0, 0.4418979585170746, 0.0), (-0.6446115374565125, 0.2906475067138672, 0.2906475067138672, 0.6446115374565125))
# converted base conf:  (-1.8195197415944886, -0.00027021797075961755, -1.2547157015094492)
# 

    # print(pp.get_joint_positions(robot, pp.get_movable_joints(robot)))
    # pp.wait_if_gui()
    # sys.exit(0)

    # try:
    if True:
        # Start up the streaming client now that the callbacks are set up.
        # This will run perpetually, and operate on a separate thread.
        is_looping = False
        if not args.disable_mocap_tracking:
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

        is_looping = is_looping | args.disable_mocap_tracking

        prev_handle = []
        planned_trajectory = None
        base_joints = pp.joints_from_names(robot, HUSKYU_JOINT_NAMES[:3])
        while is_looping:
            if prev_handle:
                pp.remove_handles(prev_handle)
                prev_handle = []

            # * mocap position update
            if not args.disable_mocap_tracking:
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

                        rb = rb_from_name[name]
                        pp.set_pose(rb, zup_from_rb)
                        prev_handle.extend(pp.draw_pose(zup_from_rb))
            else:
                # * set the husky base joint positions to the slider value
                if ik_from_arm_base:
                    base_values = [p.readUserDebugParameter(bj) for bj in joint_sliders]
                    pp.set_joint_positions(robot, base_joints, base_values)

            # * joint state update
            if not args.disable_joint_tracking:
                if arm_joint_state:
                    joints = pp.joints_from_names(robot, arm_joint_state['name'])
                    pp.set_joint_positions(robot, joints, arm_joint_state['position'])

            ee_attachment.assign()
            tcp_pose = pp.get_link_pose(robot, pp.link_from_name(robot, 'bar_tcp'))
            prev_handle.extend(pp.draw_pose(tcp_pose))

            current_plan_button_reading = p.readUserDebugParameter(plan_button)
            if current_plan_button_reading > prev_plan_button_value:
                # * plan the grasp, IK, and pick-up motion
                plan_pickup_motion(robot, ik_solver, None, bar, [ee_attachment], obstacles, 
                                   ik_from_arm_base=ik_from_arm_base, debug=args.debug)
                prev_plan_button_value = current_plan_button_reading

            if planned_trajectory:
                # set the shadow robot to the slider value
                traj_param_value = p.readUserDebugParameter(traj_param_slider)
                traj_idx = int(traj_param_value * (len(planned_trajectory) - 1))
                traj_pose = planned_trajectory[traj_idx]
                # pp.set_joint_positions(shadow_robot, pp.get_movable_joints(shadow_robot), traj_pose)

            # else:
            #     # hide the shadow robot
            #     pp.set_pose(shadow_robot, husky_pose)
            #     if arm_joint_state:
            #         joints = pp.joints_from_names(shadow_robot, arm_joint_state['name'])
            #         pp.set_joint_positions(shadow_robot, joints, arm_joint_state['position'])
            #     shadow_ee_attachment.assign()

            time.sleep(0.01)

            # pp.wait_if_gui()

    # except KeyboardInterrupt:
    #     print('\n! Received keyboard interrupt, quitting threads.\n')

    # finally:
    #     stop_thread = True

    #     if not args.disable_joint_tracking:
    #         joint_state_server.close()
    #         joint_state_stream_thread.join()

    #     if not args.disable_joint_tracking:
    #         mocap_client.shutdown()

    #     if pp.is_connected():
    #         pp.disconnect()
    #     sys.exit()


if __name__ == "__main__":
    main()