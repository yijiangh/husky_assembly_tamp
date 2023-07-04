import sys, os
import time
from husky_assembly.optitrack.NatNetClient import NatNetClient
import husky_assembly.optitrack.DataDescriptions as DataDescriptions
import husky_assembly.optitrack.MoCapData as MoCapData
from husky_assembly.optitrack.Utils import print_configuration
import numpy as np
import pybullet_planning as pp
import pybullet as p

HERE = os.path.dirname(__file__)

yup_tform = np.eye(4)
yup_tform[:3,0] = [0, 1, 0]
yup_tform[:3,1] = [0, 0, 1]
yup_tform[:3,2] = [1, 0, 0]
# yup_from_zup_tform = np.linalg.inv(yup_tform)
zup_from_yup = pp.pose_from_tform(yup_tform)
yup_from_zup = pp.invert(zup_from_yup)

rigid_body_poses = {}

# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
def receive_rigid_body_frame( new_id, position, rotation ):
    global rigid_body_poses
    rigid_body_poses[new_id] = (position, rotation)
    # print( "Received frame for rigid body", new_id )
    # print( "Received frame for rigid body", new_id," ",position," ",rotation )

# write a script to transform y-up coordinate system to z-up coordinate system

########################

def main():
    # create a new NatNet client
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
    # streaming_client.new_frame_listener = receive_new_frame
    streaming_client.rigid_body_listener = receive_rigid_body_frame

    pp.connect(use_gui=True, shadows=True, color=[0.9, 0.9, 1.0])
    # pp.create_plane(color=[0.9, 0.9, 1.0])
    p.loadMJCF(os.path.join(HERE, "plane.xml"))

    pp.draw_pose(pp.unit_pose())

    # * create a box for the rigid body
    mocap_id = 1011
    # mocap_id = 1003
    # rb = pp.create_box(0.1, 0.1, 0.1)
    rb = pp.create_cylinder(radius=0.01, height=1.0)

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

        prev_handle = []
        while is_looping:
            if mocap_id in rigid_body_poses:
                yup_from_rb = rigid_body_poses[mocap_id]
                zup_from_rb = pp.multiply(zup_from_yup, yup_from_rb)

                pp.set_pose(rb, zup_from_rb)
                if prev_handle:
                    pp.remove_handles(prev_handle)
                prev_handle = pp.draw_pose(zup_from_rb)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print('\n! Received keyboard interrupt, quitting threads.\n')

    finally:
        streaming_client.shutdown()
        # pp.disconnect()

        sys.exit()


if __name__ == "__main__":
    main()