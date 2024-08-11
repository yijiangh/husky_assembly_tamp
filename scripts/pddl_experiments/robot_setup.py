import os
import sys

import numpy as np
import pybullet_planning as pp

HERE = os.path.dirname(__file__)
husky_assembly_path = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
sys.path.append(husky_assembly_path)

from compas_fab.robots.robot import RobotModel
from compas_fab.robots import Robot as RobotClass
from compas_fab.robots import RobotSemantics
from husky_assembly import DATA_DIRECTORY
from tracikpy import TracIKSolver

TOOL0_FROM_EE = pp.Pose(point=[0, 0, 0.160])
CONTROL_JOINT_NAMES = [
    "x",
    "y",
    "theta",
    "ur_arm_shoulder_pan_joint",
    "ur_arm_shoulder_lift_joint",
    "ur_arm_elbow_joint",
    "ur_arm_wrist_1_joint",
    "ur_arm_wrist_2_joint",
    "ur_arm_wrist_3_joint",
]
ARM_CONTROL_JOINT_NAMES = [
    "ur_arm_shoulder_pan_joint",
    "ur_arm_shoulder_lift_joint",
    "ur_arm_elbow_joint",
    "ur_arm_wrist_1_joint",
    "ur_arm_wrist_2_joint",
    "ur_arm_wrist_3_joint",
]

########################


class RobotSetup(object):
    def __init__(self, robot_name="r0"):
        self.name = robot_name
        robot, ee_attachment, ik_solver, ik_solver_relative, disabled_collisions = self._load_robot()
        self.robot = robot
        self.ik_solver = ik_solver
        self.ik_solver_relative = ik_solver_relative
        self.ee_attachment = ee_attachment
        self.disabled_collisions = disabled_collisions
        self.tool0_from_ee = TOOL0_FROM_EE
        self.tool_link = pp.link_from_name(robot, "ur_arm_tool0")

        self.control_joints = pp.joints_from_names(robot, CONTROL_JOINT_NAMES)

    def _load_robot(self, ik_from_arm_base=False):
        robot_urdf = os.path.join(DATA_DIRECTORY, "husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf")
        robot_srdf = os.path.join(DATA_DIRECTORY, "husky_urdf/mt_husky_moveit_config/config/husky.srdf")

        gripper_obj = os.path.join(DATA_DIRECTORY, "husky_urdf/robotiq_85/meshes/static/robotiq_85_close_20mm.obj")
        gripper_scale = 1

        assert os.path.exists(robot_urdf)
        assert os.path.exists(gripper_obj)

        move_group = "manipulator"
        robot_model = RobotModel.from_urdf_file(robot_urdf)
        robot_semantics = RobotSemantics.from_srdf_file(robot_srdf, robot_model)
        # cp_robot = RobotClass(robot_model, semantics=robot_semantics)

        robot = pp.load_pybullet(robot_urdf, fixed_base=False, cylinder=False)

        if not ik_from_arm_base:
            ik_solver = TracIKSolver(robot_urdf, "world_link", "ur_arm_tool0")
            ik_solver_relative = TracIKSolver(robot_urdf, "ur_arm_base_link", "ur_arm_tool0")
        else:
            ik_solver = TracIKSolver(robot_urdf, "ur_arm_base_link", "ur_arm_tool0")
            ik_solver_relative = None
        # pp.camera_focus_on_body(robot)

        # get disabled collision pairs from SRDF
        disabled_self_collision_link_names = robot_semantics.disabled_collisions
        disabled_collisions = self.get_disabled_collisions_from_link_names(robot, disabled_self_collision_link_names)

        tool0_pose = pp.get_link_pose(robot, pp.link_from_name(robot, "ur_arm_tool0"))
        ee = pp.create_obj(gripper_obj, scale=gripper_scale)
        pp.set_pose(ee, pp.multiply(tool0_pose, pp.Pose(euler=pp.Euler(yaw=-np.pi / 2))))

        ee_attachment = pp.create_attachment(robot, pp.link_from_name(robot, "ur_arm_tool0"), ee)

        return robot, ee_attachment, ik_solver, ik_solver_relative, disabled_collisions

    def get_disabled_collisions_from_link_names(self, robot, disabled_self_collision_link_names):
        """get robot's link-link tuples disabled from collision checking

        Returns
        -------
        set of int-tuples
            int for link index in pybullet
        """
        return {
            tuple(pp.link_from_name(robot, link) for link in pair if pp.has_link(robot, link))
            for pair in disabled_self_collision_link_names
        }

    def get_custom_limits(self, robot, custom_limits=None):
        """[summary]

        Returns
        -------
        [type]
            {joint index : (lower limit, upper limit)}
        """
        custom_limits = custom_limits or {}
        limits = {pp.joint_from_name(robot, joint): limits for joint, limits in custom_limits.items()}
        return limits
    
    def get_relative_pose(self, world_tool_pose, link_name="ur_arm_base_link"):
        link_pose = pp.get_link_pose(self.robot, pp.link_from_name(self.robot, link_name))
        # pp.draw_pose(link_pose, length=0.5)
        # cur_pose = pp.get_pose(self.robot) # world from baselink

        # world_tool0_pose = pp.multiply(world_attach_pose, self.tool0_from_ee)
        return pp.multiply(pp.invert(link_pose), world_tool_pose)
    
    def get_relative_ik_solution(self, tool_pose_world, q_init = None):
        tool_pose_relative = self.get_relative_pose(tool_pose_world)
        conf = self.ik_solver_relative.ik(pp.tform_from_pose(tool_pose_relative), qinit=q_init)
        return conf

