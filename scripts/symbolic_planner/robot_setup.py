import os
import sys

import numpy as np
import pybullet_planning as pp
from pybullet_planning import Attachment, Euler, Point, Pose, get_distance, interpolate_poses, invert, multiply


HERE = os.path.dirname(__file__)
husky_assembly_path = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
sys.path.append(husky_assembly_path)

from compas_fab.robots.robot import RobotModel
from compas_fab.robots import Robot as RobotClass
from compas_fab.robots import RobotSemantics
from husky_assembly import DATA_DIRECTORY
from tracikpy import TracIKSolver
from utils import HUSKYU_JOINT_NAMES, plan_transit_motion


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
BASE_CONTROL_JOINT_NAMES = ["x", "y", "theta"]
INIT_ARM_JOINT_ANGLES = np.array([0, -np.pi / 2, 0, 0, 0, 0])

########################


class RobotSetup(object):
    def __init__(self, robot_name="r0", attachments=[]):
        self.name = robot_name
        robot, ee_attachment, ik_solver, ik_solver_relative, disabled_collisions = self._load_robot()
        self.robot = robot
        self.ik_solver = ik_solver
        self.ik_solver_relative = ik_solver_relative
        self.ee_attachment = ee_attachment
        self.attachments = attachments
        self.disabled_collisions = disabled_collisions
        self.tool0_from_ee = TOOL0_FROM_EE
        self.tool_link = pp.link_from_name(robot, "ur_arm_tool0")

        self.control_joints = pp.joints_from_names(robot, CONTROL_JOINT_NAMES)
        self.arm_joints = pp.joints_from_names(robot, HUSKYU_JOINT_NAMES)
        self.base_joints = pp.joints_from_names(robot, BASE_CONTROL_JOINT_NAMES)
        self.arm_init_angles = INIT_ARM_JOINT_ANGLES
        self.set_joint_positions(self.arm_joints, self.arm_init_angles)

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

    def get_relative_pose(self, pose_world, link_name="ur_arm_base_link"):
        link_pose = pp.get_link_pose(self.robot, pp.link_from_name(self.robot, link_name))
        return pp.multiply(pp.invert(link_pose), pose_world)

    def get_relative_ik_solution(self, tool_pose_world, q_init=None):
        tool_pose_relative = self.get_relative_pose(tool_pose_world)
        conf = self.ik_solver_relative.ik(pp.tform_from_pose(tool_pose_relative), qinit=q_init)
        self.ee_attachment.assign()
        return conf

    def plan_manipulator_path(
        self, init_q, target_q, attachments, obstacles, sub_way_points=False, way_points_max_num=15
    ):
        # pp.set_joint_positions(self.robot, self.arm_joints, init_q)
        self.set_joint_positions(self.arm_joints, init_q)
        self.ee_attachment.assign()
        for att in attachments:
            att.assign()

        # print(">>> short path")
        planned_path_coarse = plan_transit_motion(
            self.robot,
            target_q,
            [self.ee_attachment] + attachments,
            obstacles,
            debug=False,
            disabled_collisions=self.disabled_collisions,
            coarse_waypoints=sub_way_points,
        )
        self.ee_attachment.assign()

        if planned_path_coarse is not None and planned_path_coarse != False:
            planned_path_coarse = [np.array(conf) for conf in planned_path_coarse]
        else:
            return None
        if not sub_way_points:
            return planned_path_coarse
        if len(planned_path_coarse) >= way_points_max_num:
            step = int(len(planned_path_coarse) / way_points_max_num)
            temp = [planned_path_coarse[i : i + step] for i in range(0, len(planned_path_coarse), step)]
            way_points = [conf[0] for conf in temp]
            way_points.append(np.array(target_q))
        else:
            way_points = planned_path_coarse.copy()

        # print(">>> long path")
        planned_path = []
        for idx, conf in enumerate(way_points[:-1]):
            self.set_joint_positions(self.arm_joints, conf)
            next_conf = way_points[idx + 1]
            planned_path_segment = plan_transit_motion(
                self.robot,
                next_conf,
                [self.ee_attachment] + attachments,
                obstacles,
                disabled_collisions=self.disabled_collisions,
            )
            self.ee_attachment.assign()
            if planned_path_segment is None:
                return None
            planned_path.extend(planned_path_segment)

        if planned_path is not None:
            planned_path = [np.array(conf) for conf in planned_path]
        return planned_path

    def set_base_pose(self, pose):
        pp.set_pose(self.robot, pose)
        self.ee_attachment.assign()
        for attachment in self.attachments:
            attachment: Attachment
            attachment.assign()

    def set_base_pose_2d(self, x, y, yaw=0.0):
        pose = pp.Pose(point=[x, y, 0], euler=pp.Euler(0, 0, yaw))
        pp.set_pose(self.robot, pose)
        self.ee_attachment.assign()
        for attachment in self.attachments:
            attachment: Attachment
            attachment.assign()

    def set_joint_positions(self, control_joints, conf):
        pp.set_joint_positions(self.robot, control_joints, conf)
        self.ee_attachment.assign()
        for attachment in self.attachments:
            attachment: Attachment
            attachment.assign()

    def update_attachments(self, attachments):
        self.attachments = attachments
