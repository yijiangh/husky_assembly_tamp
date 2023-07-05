import sys, os
import time
import random
import socket, json
from threading import Thread

import numpy as np
import pybullet_planning as pp
import pybullet as p

from husky_assembly.optitrack.NatNetClient import NatNetClient
import husky_assembly.optitrack.DataDescriptions as DataDescriptions
import husky_assembly.optitrack.MoCapData as MoCapData
from husky_assembly.optitrack.Utils import print_configuration
from husky_assembly import DATA_DIRECTORY
from husky_assembly.husky_client import HuskyClient

HERE = os.path.dirname(__file__)

yup_tform = np.eye(4)
yup_tform[:3,0] = [0, 1, 0]
yup_tform[:3,1] = [0, 0, 1]
yup_tform[:3,2] = [1, 0, 0]
# yup_from_zup_tform = np.linalg.inv(yup_tform)
zup_from_yup = pp.pose_from_tform(yup_tform)
yup_from_zup = pp.invert(zup_from_yup)

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

def socket_recv_thread(socket_server):
    while True:
        try:
            receive_joint_state(socket_server)
        except socket.timeout:
            pass

########################

def get_bar_grasp_gen_fn(bar_length, tool_pose=pp.unit_pose(), reverse_grasp=False, safety_margin_length=0.0):
    """[summary]

    # converted from https://pybullet-planning.readthedocs.io/en/latest/reference/generated/pybullet_planning.primitives.grasp_gen.get_side_cylinder_grasps.html
    # to get rid of the rotation around the local z axis

    """

    # rotate the cylinder's frame to make x axis align with the longitude axis
    longitude_x = pp.Pose(euler=pp.Euler(pitch=np.pi/2))
    def gen_fn():
        while True:
            # translation along the longitude axis
            slide_dist = random.uniform(-bar_length/2+safety_margin_length, bar_length/2-safety_margin_length)
            translate_along_x_axis = pp.Pose(point=pp.Point(slide_dist,0,0))

            for j in range(1 + reverse_grasp):
                # the base pi/2 is to make y align with the longitude axis, conforming to the convention (see image in the doc)
                # flip the gripper, gripper symmetry
                rotate_around_z = pp.Pose(euler=[0, 0, np.pi/2 + j * np.pi])

                object_from_gripper = pp.multiply(longitude_x, translate_along_x_axis, \
                    rotate_around_z, tool_pose)
                yield pp.invert(object_from_gripper)
    return gen_fn

def load_robot():
    robot_urdf = os.path.join(DATA_DIRECTORY,'husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf')
    gripper_obj = os.path.join(DATA_DIRECTORY,'husky_urdf/robotiq_85/meshes/static/robotiq_85_close_20mm.obj')
    # robot_urdf = os.path.join(HERE,'robotiq_85/urdf/robotiq_85_gripper_simple.urdf')
    # robot_urdf = os.path.join(HERE,'mt_husky_dual_ur5_e_moveit_config/urdf/husky_dual_ur5_e.urdf')
    # print(robot_urdf)
    assert os.path.exists(robot_urdf)
    assert os.path.exists(gripper_obj)
    
    robot = pp.load_pybullet(robot_urdf, fixed_base=False, cylinder=True)
    # pp.clone_body(robot, collision=True, visual=False)
    # pp.camera_focus_on_body(robot)

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
    pp.draw_pose(tool0_pose)

    ee = pp.create_obj(gripper_obj) 
    pp.set_pose(ee, pp.multiply(tool0_pose, pp.Pose(euler=pp.Euler(yaw=-np.pi/2))))
    ee_attachment = pp.create_attachment(robot, pp.link_from_name(robot, 'ur_arm_tool0'), ee)

    # tool0_from_ee = pp.Pose(euler=pp.Euler(yaw=-np.pi/2), point=[0,0,0.138])
    # tcp_pose = pp.multiply(tool0_pose, tool0_from_ee)
    # tcp_pose = pp.get_link_pose(robot, pp.link_from_name(robot, 'central_tcp'))
    # pp.draw_pose(tcp_pose)

    return robot, ee_attachment

def main():
    # * create a new NatNet client
    optionsDict = {}
    optionsDict["serverAddress"] = "192.168.0.117" # optitrack server address
    optionsDict["clientAddress"] = "192.168.0.180" # this machine's address
    optionsDict["use_multicast"] = True
    streaming_client = NatNetClient()
    streaming_client.set_client_address(optionsDict["clientAddress"])
    streaming_client.set_server_address(optionsDict["serverAddress"])
    streaming_client.set_use_multicast(optionsDict["use_multicast"])
    streaming_client.print_level = 0
    # Configure the streaming client to call our rigid body handler on the emulator to send data out.
    streaming_client.rigid_body_listener = receive_rigid_body_frame

    # * create a new Husky client
    # HOST = '192.168.0.113'  # Standard loopback interface address (localhost)
    PORT = 65432  # Port to listen on (non-privileged ports are > 1023)
    CLIENT_IP = '192.168.0.180' # Set to your own IP
    stream_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    stream_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # on the husky side, we set it to always send to the same port to the host
    stream_server.bind((CLIENT_IP, PORT))
    stream_server.settimeout(0.001)
    stream_thread = Thread(target=socket_recv_thread, args=(stream_server,))
    stream_thread.daemon = True
    stream_thread.start()

    # * start pybullet simulator
    pp.connect(use_gui=True, shadows=True, color=[0.9, 0.9, 1.0])
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=pp.CLIENT)
    traj_param = p.addUserDebugParameter("trajectory playback", 0.0, 1.0, 0.0)
    plan = p.addUserDebugParameter("plan", 1, 0, 0)
    execute = p.addUserDebugParameter("execute", 1, 0, 1)
    pp.draw_pose(pp.unit_pose(), 1.0)
    with pp.LockRenderer():
        p.loadMJCF(os.path.join(HERE, "plane.xml"))
        # pp.create_plane(color=[0.9, 0.9, 1.0])
        bar = pp.create_cylinder(radius=0.01, height=1.0)
        with pp.HideOutput():
            robot, ee_attachment = load_robot()
    rb_from_name = {
        'bar': bar,
        'husky0804': robot,
    }
    # print(pp.get_joint_positions(robot, pp.get_movable_joints(robot)))
    # pp.wait_if_gui()
    # sys.exit(0)

    # Start up the streaming client now that the callbacks are set up.
    # This will run perpetually, and operate on a separate thread.
    try:
        is_running = streaming_client.run()
        print_configuration(streaming_client)
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
        if not streaming_client.connected():
            print("ERROR: Could not connect properly.  Check that Motive streaming is on.")
            try:
                sys.exit(2)
            except SystemExit:
                print("...")
            finally:
                print("exiting")

        # joint_state_data = socketRecvMessage(socket_server)

        prev_handle = []
        while is_looping:
            if prev_handle:
                pp.remove_handles(prev_handle)
                prev_handle = []

            # * mocap position update
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
                    if name == 'husky0804':
                        ee_attachment.assign()

                    prev_handle.extend(pp.draw_pose(zup_from_rb))

            # * joint state update
            # arm_joint_state = socketRecvMessage(socket_server)
            if arm_joint_state:
                joints = pp.joints_from_names(robot, arm_joint_state['name'])
                pp.set_joint_positions(robot, joints, arm_joint_state['position'])

            time.sleep(0.01)

            # bar_length = 0.5
            # bar_body = pp.create_cylinder(0.01, bar_length, mass=pp.STATIC_MASS)
            # grasp_gen = get_bar_grasp_gen_fn(bar_length)

            # for _ in range(10):
            #     gripper_from_object = next(grasp_gen())
            #     world_from_object = pp.multiply(tcp_pose, gripper_from_object)
            #     pp.set_pose(bar_body, world_from_object)
            #     pp.wait_if_gui() 

            # pp.wait_if_gui()

    except KeyboardInterrupt:
        print('\n! Received keyboard interrupt, quitting threads.\n')

    finally:
        streaming_client.shutdown()
        stream_server.close()
        # attempt to join the threads back.
        stream_thread.join()
        pp.disconnect()

        sys.exit()


if __name__ == "__main__":
    main()