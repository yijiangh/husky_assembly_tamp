import sys, os, argparse
import numpy as np
import pybullet_planning as pp
import pybullet as p
import sys, os, argparse
import time
import socket, json
from threading import Thread
from collections import defaultdict

from husky_assembly.optitrack.NatNetClient import NatNetClient
from husky_assembly.optitrack.Utils import print_configuration
from husky_assembly import DATA_DIRECTORY

from single_bar_grasp import HUSKYU_JOINT_NAMES, CLIENT_IP, MOCAP_IP, HUSKY_TCP_PORT, HUSKY_UDP_PORT, LOCAL_SERVER_IP, LOCAL_SERVER, HERE, HUSKY_IP, \
    zup_from_yup, load_robot, socket_recv_thread, \
    receive_rigid_body_frame, rigid_body_poses, arm_joint_state, align_joint_conf_by_joint_names, \
    plan_transit_motion, send_base_arm_trajectory_command

name_from_mocap_id = {
    1028 : 'husky0804',
    # 1011 : 'bar',
    # 1030 : 'foundation_bar',
    # 1029 : 'greybox',
    1032 : 'ur_shoulder_link',
}

UR_BASE_LINK_NAME = 'ur_arm_base_link'
UR_SHOULDER_LINK_NAME = 'ur_arm_shoulder_link'
CART_BASE_ID = 1028
SHOULDER_ID = 1032

def append_mocap_data(recorded_data, robot):
    global rigid_body_poses

    world_from_cart_marker = rigid_body_poses[CART_BASE_ID]
    world_from_shoulder_marker = rigid_body_poses[SHOULDER_ID]
    # joint state should have been updated by the joint state listening thread
    base_from_shoulder = pp.get_relative_pose(robot, pp.link_from_name(robot, UR_SHOULDER_LINK_NAME), pp.link_from_name(robot, UR_BASE_LINK_NAME))

    recorded_data['robot'].append({'world_from_cart_marker': world_from_cart_marker,
                                   'world_from_shoulder_marker': world_from_shoulder_marker,
                                   'base_from_shoulder': base_from_shoulder,})
    return recorded_data

def display_frames(robot, base_from_cart_marker, shoulder_from_marker):
    global rigid_body_poses

    # world_from_base_link = pp.get_link_pose(robot, pp.link_from_name(robot, UR_BASE_LINK_NAME))
    # world_from_shoulder = pp.get_link_pose(robot, pp.link_from_name(robot, UR_SHOULDER_LINK_NAME))

    world_from_cart_marker = rigid_body_poses[CART_BASE_ID]
    world_from_shoulder_marker = rigid_body_poses[SHOULDER_ID]

    # pp.draw_pose(world_from_cart_marker, length=0.2)
    pp.draw_pose(pp.multiply(zup_from_yup, world_from_cart_marker), length=0.2)
    pp.draw_pose(pp.multiply(zup_from_yup, world_from_shoulder_marker), length=0.2)

    shoulder_from_arm_base = pp.get_relative_pose(robot, pp.link_from_name(robot, UR_BASE_LINK_NAME), pp.link_from_name(robot, UR_SHOULDER_LINK_NAME))

    # * goal is to deduce world_from_arm_base from the marker poses
    # zup_from_rb = pp.multiply(zup_from_yup, yup_from_rb)
    world_from_base_pose1 = pp.multiply(zup_from_yup, world_from_cart_marker, pp.invert(base_from_cart_marker))
    world_from_base_pose2 = pp.multiply(zup_from_yup, world_from_shoulder_marker, pp.invert(shoulder_from_marker), shoulder_from_arm_base)

    pp.draw_pose(world_from_base_pose1, length=0.2)
    pp.add_text('cart-p', world_from_base_pose1[0])

    pp.draw_pose(world_from_base_pose2, length=0.2)
    pp.add_text('shoulder-p', world_from_base_pose2[0])

    print('Step')

def finish_recording_and_compute(recorded_data):
    return None, None

def main():
    parser = argparse.ArgumentParser()
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

    # # * y-up to be consistent with mocap
    # p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1, physicsClientId=pp.CLIENT)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=pp.CLIENT)
    # pp.set_camera(np.deg2rad(92.0), np.deg2rad(-85), 5.20)
    # pp.set_camera(92.0, -85, 5.20)

    # * Control UI
    traj_param_slider = p.addUserDebugParameter("trajectory playback", 0.0, 1.0, 0.0)
    #  For a button, the value of getUserDebugParameter for a button increases 1 at each button press.
    plan_button = p.addUserDebugParameter("plan", 1, 0, 0)
    prev_plan_button_value = p.readUserDebugParameter(plan_button)

    reverse_traj_button = p.addUserDebugParameter("reverse traj", 1, 0, 0)
    prev_reverse_value = p.readUserDebugParameter(reverse_traj_button)

    execute_button = p.addUserDebugParameter("execute", 1, 0, 0)
    prev_execute_button_value = p.readUserDebugParameter(execute_button)

    record_button = p.addUserDebugParameter("record data point", 1, 0, 0)
    prev_record_value = p.readUserDebugParameter(record_button)

    finish_rec_button = p.addUserDebugParameter("finish record and compute", 1, 0, 0)
    prev_finish_rec_value = p.readUserDebugParameter(finish_rec_button)

    # slider to control the time gap between each trajectory point
    dt_slider = p.addUserDebugParameter("traj dt", 0.04, 0.2, 0.2)

    ik_from_arm_base = 1

    # * load all robots and objects
    pp.draw_pose(pp.unit_pose(), 0.5)
    with pp.LockRenderer():
        p.loadMJCF(os.path.join(HERE, "plane.xml"))
        plane = pp.create_plane(color=[0.9, 0.9, 1.0, 0.0])

        # bar = pp.create_cylinder(radius=0.01, height=1.0, color=pp.BROWN)
        # foundation_bar = pp.create_cylinder(radius=0.01, height=0.75, color=pp.BROWN)
        # box = pp.create_box(0.6, 0.4, 0.45, color=pp.apply_alpha(pp.GREY, 1))

        with pp.HideOutput():
            alpha = 0.1
            robot, ee_attachment, ik_solver, disabled_collisions = load_robot(ik_from_arm_base)
            state_color = [0.1, 0.4, 0.2, alpha]
            pp.set_color(robot, state_color)

            # ! a shadow robot for displaying the trajectory
            shadow_robot, shadow_ee_attachment, _, _ = load_robot()
            shadow_color = [0.5, 0.5, 0.5, alpha]
            pp.set_color(shadow_robot, shadow_color)
            pp.set_color(shadow_ee_attachment.child, shadow_color)

            goal_robot = None
            goal_ee_attachment = None

            goal_robot, goal_ee_attachment, _, _ = load_robot()
            goal_color = [0, 0.2, 0.5, alpha]
            pp.set_color(goal_robot, goal_color)

    first_joint_id = 3 if ik_from_arm_base else 0
    planned_joint_names = HUSKYU_JOINT_NAMES[first_joint_id:]
    planned_joints = pp.joints_from_names(robot, planned_joint_names)
    base_cmd_names = ['xVel', 'angVel']

    # recorded_conf = (0,0,0,-1.3227758407592773, -1.5873312950134277, -1.5211923122406006, 0.0, 0.0, 0.0)
    recorded_conf = np.zeros(len(HUSKYU_JOINT_NAMES))
    if not args.connect_to_mocap :
        # a recorded pose for debuggging purpose
        # temp_bar_pose = ((-1.062444806098938, 0.19626910984516144, 0.6585784554481506), (0.8137449622154236, -0.1838780641555786, 0.5276390910148621, -0.1600152850151062))
        # pp.set_pose(bar, temp_bar_pose)

        pp.set_joint_positions(robot, pp.joints_from_names(robot, HUSKYU_JOINT_NAMES), recorded_conf)
        pp.set_joint_positions(shadow_robot, pp.joints_from_names(robot, HUSKYU_JOINT_NAMES), recorded_conf)
        # ee_attachment.assign()
        # shadow_ee_attachment.assign()
    else:
        if arm_joint_state:
            recorded_conf = align_joint_conf_by_joint_names(planned_joint_names, arm_joint_state['position'], arm_joint_state['name'])

    joint_sliders = []
    for j, initial_v in zip(planned_joints, recorded_conf):
        lower, upper = pp.get_joint_limits(robot, j)
        joint_sliders.append(p.addUserDebugParameter(pp.get_joint_name(robot, j).decode("utf-8"), 
                                                     lower, upper, initial_v))
        pp.set_joint_position(shadow_robot, j, initial_v)
        pp.set_joint_position(goal_robot, j, initial_v)
    prev_joint_slider_values = [p.readUserDebugParameter(js) for js in joint_sliders]

    obstacles = [plane]
    pp.camera_focus_on_body(robot)

    rb_from_name = {
        'husky0804': robot,
        # 'bar': bar,
        # 'greybox': box,
        # 'foundation_bar': foundation_bar,
    }


    recorded_data = defaultdict(list)

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
                sys.exit(1)

            is_looping = True
            time.sleep(1)
            if not mocap_client.connected():
                print("ERROR: Could not connect properly.  Check that Motive streaming is on.")
                sys.exit(2)

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

                            # base_conf = pp.pose2d_from_pose(yup_from_rb, tolerance=2e-2)
                            # x, _, z = yup_from_rb[0]
                            # roll, pitch, yaw = pp.euler_from_quat(yup_from_rb[1])
                            # base_conf = (z, x, yaw)

                            pp.set_joint_positions(robot, base_joints, base_conf)
                            pp.set_joint_positions(shadow_robot, base_joints, base_conf)
                            pp.set_joint_positions(goal_robot, base_joints, base_conf)

                        # elif name in rb_from_name:
                        #     rb = rb_from_name[name]
                        #     # TODO change to set_joint_positions
                        #     pp.set_pose(rb, zup_from_rb)
                        # prev_handle.extend(pp.draw_pose(zup_from_rb))

            # * set the husky arm joint positions to the slider value
            # only update when the slider value changes
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
                planned_trajectory = plan_transit_motion(robot, current_joint_slider_values, 
                                                         [ee_attachment], 
                                                         obstacles, 
                                                         debug=args.debug, ik_from_arm_base=ik_from_arm_base, disabled_collisions=disabled_collisions)

            prev_plan_button_value = current_plan_button_reading

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

                dt_value = p.readUserDebugParameter(dt_slider)
                time_from_start = [i * dt_value for i in range(len(planned_trajectory))]
                send_base_arm_trajectory_command(traj_socket_client, base_cmd_names + planned_joint_names, padded_traj, time_from_start)
            prev_execute_button_value = current_execute_button_reading

            # * record data
            current_rec_reading = p.readUserDebugParameter(record_button)
            if current_rec_reading > prev_record_value:
                recorded_data = append_mocap_data(recorded_data, robot)
                print('current data len: ', len(recorded_data['robot']))
            prev_record_value = current_rec_reading

            # * finish recording and compute
            current_finish_rec_reading = p.readUserDebugParameter(finish_rec_button)
            if current_finish_rec_reading > prev_finish_rec_value:
                # base_from_cart_marker, shoulder_from_marker = finish_recording_and_compute(recorded_data)
                world_from_cart_base_marker = rigid_body_poses[CART_BASE_ID]
                world_from_arm_base = pp.get_link_pose(robot, pp.link_from_name(robot, UR_BASE_LINK_NAME))
                pp.draw_pose(world_from_arm_base, length=0.1)
                init_base_from_cart_marker = pp.multiply(pp.invert(world_from_arm_base), 
                                                         pp.multiply(zup_from_yup, world_from_cart_base_marker))

                # init_base_from_cart_marker = pp.get_relative_pose(robot, pp.link_from_name(robot, UR_BASE_LINK_NAME), pp.link_from_name(robot, 'base_footprint'))

                # init_base_from_cart_marker = pp.get_relative_pose(robot, pp.link_from_name(robot, UR_BASE_LINK_NAME), pp.link_from_name(robot, 'base_footprint'))
                init_shoulder_from_marker = pp.Pose(point=[0.0, 0.0, 0.06])

                display_frames(robot, init_base_from_cart_marker, init_shoulder_from_marker)

                ## clear all recorded data
                recorded_data = defaultdict(list)

            prev_finish_rec_value = current_finish_rec_reading


            time.sleep(0.01)


if __name__ == "__main__":
    main()