import sys, time
from husky_assembly.optitrack.NatNetClient import NatNetClient
from husky_assembly.optitrack.Utils import print_configuration

CLIENT_IP = '192.168.0.7' # Set to your own IP
MOCAP_IP = '192.168.0.117'

def receive_rigid_body_frame( new_id, position, rotation ):
    # global rigid_body_poses
    # rigid_body_poses[new_id] = (position, rotation)
    print( "Received frame for rigid body", new_id )
    # print( "Received frame for rigid body", new_id," ",position," ",rotation )
    pass

def receive_labeled_marker_frame( labeled_marker_from_model_id ):
    # global rigid_body_poses
    # rigid_body_poses[new_id] = (position, rotation)
    # print( "Received frame for labeled markers: {} sets".format(len(labeled_marker_from_model_id)))
    print( "Received frame for labeled markers: {}".format(labeled_marker_from_model_id))
    pass

mocap_client = NatNetClient()
mocap_client.set_client_address(CLIENT_IP)
mocap_client.set_server_address(MOCAP_IP)
mocap_client.set_use_multicast(False)
mocap_client.print_level = 1
# Configure the streaming client to call our rigid body handler on the emulator to send data out.
mocap_client.rigid_body_listener = receive_rigid_body_frame
mocap_client.labeled_marker_listener = receive_labeled_marker_frame

is_looping = False
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

# time.sleep(1)
is_looping = False
mocap_client.shutdown()
sys.exit(0)
