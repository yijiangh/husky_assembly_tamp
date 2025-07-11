import os, time
from compas_fab.robots import RobotSemantics, RobotCell, RobotCellState, FrameTarget, TargetMode
from compas_fab.backends import PyBulletClient, PyBulletPlanner
import pybullet_planning as pp
from compas import json_load
from conversions import pose_from_frame, frame_from_pose

HERE = os.path.dirname(__file__)

design_study_path = os.path.join(HERE, "..", "..", "data", "husky_assembly_design_study")
design_case = "250707_RobotX_box_demo"
robot_cell_json_path = os.path.join(
    design_study_path,
    design_case,
    "RobotCell.json"
)
with open(robot_cell_json_path, "r") as f:
    robot_cell = json_load(f)

robot_cell_state_path = os.path.join(
    design_study_path,
    design_case,
    "RobotCellStates",
    "robotx_box_A3-A_RobotCellState.json"
)
with open(robot_cell_state_path, "r") as f:
    robot_cell_state = json_load(f)

# --- 5. Run IK using PyBullet ---
use_gui = True
with PyBulletClient(connection_type="gui" if use_gui else "direct", verbose=False) as client:
    pp.CLIENTS[client.client_id] = use_gui
    planner = PyBulletPlanner(client)
    # pp.create_plane(color=pp.GREY)

    start = time.time()
    with pp.LockRenderer(0):
        planner.set_robot_cell(robot_cell)
        planner.set_robot_cell_state(robot_cell_state)
    print("Setting robot cell and state took {:.3f} seconds".format(time.time() - start))

    # Robot base pose need to be manually updated
    robot_base_frame = robot_cell_state.robot_base_frame
    robot_base_pose = pose_from_frame(robot_base_frame)
    pp.set_pose(client.robot_puid, robot_base_pose)

    # ! IK targets are saved in the file called 'robotx_box_A0-G_GraspTargets.json'
    # for parsing, see load_grasp_targets.py

    # pose = pose_from_frame(frame_WCF)
    # pp.draw_pose(pose)
    pp.wait_if_gui()

print("Done.") 