#!/usr/bin/env python3
import os
import sys
from math import pi

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import moveit_commander
import moveit_msgs.msg
import rospy
from moveit_msgs.msg import RobotTrajectory, DisplayTrajectory
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from utils.params import *

package_path = os.environ.get("ROS_PACKAGE_PATH", "").split(os.pathsep)
if PACKAGE_DIRECTORY not in package_path:
    package_path.append(PACKAGE_DIRECTORY)
os.environ["ROS_PACKAGE_PATH"] = os.pathsep.join(package_path)

if __name__ == "__main__":

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("move_group_python_interface_tutorial", anonymous=True)

    group_name = "arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    robot = moveit_commander.RobotCommander()
    current_state = robot.get_current_state()

    display_trajectory_publisher = rospy.Publisher(
        "/move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory, queue_size=20
    )

    user_trajectory = [[0, var / 180 * pi, 0, 0, 0, 0] for var in range(-90, 0, 1)]
    user_trajectory.append([0, 0, 0, 0, 0, 0])

    trajectory = JointTrajectory()
    trajectory.joint_names = [
        "ur_arm_shoulder_pan_joint",
        "ur_arm_shoulder_lift_joint",
        "ur_arm_elbow_joint",
        "ur_arm_wrist_1_joint",
        "ur_arm_wrist_2_joint",
        "ur_arm_wrist_3_joint",
    ]
    trajectory.header = Header(stamp=rospy.Time.now())

    time_step = 1.0 / 30
    for i, angles in enumerate(user_trajectory):
        point = JointTrajectoryPoint()
        point.positions = angles
        point.time_from_start = rospy.Duration(i * time_step)
        trajectory.points.append(point)

    robot_trajectory = RobotTrajectory()
    robot_trajectory.joint_trajectory = trajectory

    display_trajectory = DisplayTrajectory()
    display_trajectory.trajectory_start = current_state
    display_trajectory.trajectory.append(robot_trajectory)

    pub = rospy.Publisher('/move_group/display_planned_path', DisplayTrajectory, queue_size=10)
    rospy.sleep(1)
    pub.publish(display_trajectory)

    rospy.spin()

    # **************************************************************************
    # open pybullet to visualize
    # **************************************************************************

    # import os
    # import sys
    # import pybullet_planning as pp
    # import numpy as np

    # HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    # sys.path.append(HERE)

    # import utils.load_multi_tangent as load_multi_tangent
    # from multi_tangent.collision import create_collision_bodies
    # from multi_tangent.convert import flatten_list
    # from robot.robot_setup import RobotSetup
    # from utils.collision import Element, create_couplers, init_pb

    # urdf_path = (
    #     "/home/jeong/summer_research/eth/husky_assembly/data/husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf"
    # )

    # init_pb()

    # rb = RobotSetup("r0")
    # rb.set_joint_positions(rb.arm_joints, np.array([0] * 6))

    # pp.draw_pose(pp.get_pose(rb.ee_attachment.child))

    # pp.wait_for_user()
