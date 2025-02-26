import os
import sys
import xml.etree.ElementTree as ET
from functools import partial
from typing import Dict, List, Set, Tuple, Union

import casadi as ca
import numpy as np
from utils.params import *

husky_assembly_path = os.path.join(PROJECT_DIR, "src")
sys.path.append(husky_assembly_path)
sys.path.append(PROJECT_DIR)

import pybullet_planning as pp
from compas_fab.robots import Robot as RobotClass
from compas_fab.robots import RobotSemantics
from compas_fab.robots.robot import RobotModel
from pybullet_planning import Attachment, Euler, Point, Pose, get_distance, interpolate_poses, invert, multiply
from solver.ik_pinocchio_solver import PinocchioSolver
from tracikpy import TracIKSolver
from utils.utils import HUSKYU_JOINT_NAMES, get_custom_limits

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

if PICK_DIRECTION == "left":
    ONBOARD_POSE = [0.0, -0.5, 0.5, -np.pi / 2, 0.0, np.pi / 2]  # [x, y, z, r, p, y]
    ONBOARD_LINK = "ur_arm_base_link"
elif PICK_DIRECTION == "behind":
    ONBOARD_POSE = [0.4, 0.0, 0.5, -np.pi / 2, 0.0, 0.0]  # [x, y, z, r, p, y]
    ONBOARD_LINK = "ur_arm_base_link"


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
        # cloned_b1 = pp.clone_body(robot, links=[7], collision=True, visual=False)

        if not ik_from_arm_base:
            trac_ik_solver = TracIKSolver(robot_urdf, "world_link", "ur_arm_tool0")
            trac_ik_solver_relative = TracIKSolver(robot_urdf, "ur_arm_base_link", "ur_arm_tool0")
            ik_solver = trac_ik_solver.ik
            ik_solver_relative = trac_ik_solver_relative.ik
            self.tracik_ik_solver = trac_ik_solver.ik
            self.tracik_ik_solver_relative = trac_ik_solver_relative.ik

            pinocchio_solver = PinocchioSolver(robot_urdf)
            ik_solver = partial(pinocchio_solver.ik, base_name="world_link", tip_name="ur_arm_tool0", relative=False)
            ik_solver_relative = partial(
                pinocchio_solver.ik, base_name="ur_arm_base_link", tip_name="ur_arm_tool0", relative=True
            )
            self.pinocchio_ik_solver = ik_solver
            self.pinocchio_ik_solver_relative = ik_solver_relative
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

    def get_relative_pose(self, pose_world, link_name="ur_arm_base_link") -> Tuple[Tuple[float], Tuple[float]]:
        link_pose = pp.get_link_pose(self.robot, pp.link_from_name(self.robot, link_name))
        return pp.multiply(pp.invert(link_pose), pose_world)

    def get_relative_ik_solution(
        self,
        tool_pose_world: Tuple[Tuple[float], Tuple[float]],
        q_init: Union[List[float], None] = None,
        solver: str = "pinocchio",
    ) -> np.ndarray:
        """
        Calculate ik solution of manipulator.

        Params:
            tool_pose_world (Tuple[Tuple[float], Tuple[float]]): pp.Pose, world_from_tool0
            q_init ([float] | None, None): conf of manipulator as initial guess
            solver (str, "pinocchio"): pinocchio/tracik

        Returns:
            q (np.ndarray): ik solution of conf
        """
        tool_pose_relative = self.get_relative_pose(tool_pose_world)
        if solver == "pinocchio":
            conf = self.pinocchio_ik_solver_relative(pp.tform_from_pose(tool_pose_relative), qinit=q_init)
        else:
            conf = self.tracik_ik_solver_relative(pp.tform_from_pose(tool_pose_relative), qinit=q_init)
        self.ee_attachment.assign()
        return conf

    def plan_manipulator_path(
        self,
        init_q: np.ndarray,
        target_q: np.ndarray,
        attachments: List[Attachment],
        obstacles: Set[int],
        sub_way_points: bool = False,
        way_points_max_num: int = 15,
    ) -> List[np.ndarray]:
        """
        Plan manipulator path from init_q to target_q with attachments.

        Params:
            init_q (np.ndarray): start conf
            target_q (np.ndarray): target conf
            attachments ([Attachment]): Attachments on the robot including base and manipulator, excluding ee_attachment
            obstacles (Set[int]): fixed obstacles + assembled elements
            sub_way_points (bool, False, [not used]): whether generate intermediate points
            way_points_max_num (int, 15, [not used]): max num of intermediate points

        Returns:
            path ([np.ndarray]): manipulator confs from start to target
        """
        self.set_joint_positions(self.arm_joints, init_q)
        self.ee_attachment.assign()
        for att in attachments:
            att.assign()

        base_conf: Tuple = pp.get_joint_positions(self.robot, self.base_joints)
        init_conf = np.hstack((np.array(base_conf), init_q))
        target_conf = np.hstack((np.array(base_conf), target_q))
        frozen_joints = list(self.base_joints)
        frozen_values = [init_conf[self.control_joints.index(joint_id)] for joint_id in frozen_joints]

        planned_path = self.plan_manipulator_motion(
            init_conf,
            target_conf,
            [self.ee_attachment] + attachments,
            obstacles,
            disabled_collisions=self.disabled_collisions,
            frozen_joints=frozen_joints,
            frozen_values=frozen_values,
            diagnosis=False,
        )
        self.ee_attachment.assign()

        if planned_path is not None:
            planned_path = [np.array(conf)[3:] for conf in planned_path]
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

    def create_aboard_attachment(self, body: int) -> Attachment:
        """
        Create attachment on the robot.

        Params:
            body (int): index in the PyBullet

        Returns:
            Attachment: attachment between robot and body
        """
        ipad_link_pose = pp.get_link_pose(self.robot, pp.link_from_name(self.robot, ONBOARD_LINK))
        delta_pose = Pose(
            point=ONBOARD_POSE[:3], euler=Euler(roll=ONBOARD_POSE[3], pitch=ONBOARD_POSE[4], yaw=ONBOARD_POSE[5])
        )
        bar_pose = multiply(ipad_link_pose, delta_pose)
        pp.set_pose(body, bar_pose)
        # pp.draw_pose(bar_pose, length=0.3)
        attachment = pp.create_attachment(self.robot, pp.link_from_name(self.robot, ONBOARD_LINK), body)
        return attachment

    def plan_manipulator_motion(
        self,
        start_conf: np.ndarray,
        end_conf: np.ndarray,
        attachments: List[Attachment],
        obstacles: Set[int],
        disabled_collisions: Dict = {},
        frozen_joints: List[int] = [],
        frozen_values: List[float] = [],
        coarse_waypoints: bool = False,
        diagnosis: bool = False,
    ) -> Union[List[Tuple[float]], None]:
        """
        Plan manipulator path from current conf to end conf.

        Params:
            start_conf (np.ndarray): start conf of manipulator and base
            end_conf (np.ndarray): target conf of manipulator and base
            attachments ([Attachment]): Attachments on the robot including base, manipulator and ee_attachment
            obstacles (Set[int]): fixed obstacles + assembled elements
            disabled_collisions (Dict, {}): disabled collisions
            frozen_joints ([int], []): id of joints need to freeze
            frozen_values ([float], []): value of joints need to freeze
            coarse_waypoints (bool, False): whether generate sparse points
            diagnosis (bool, True): whether stop and display it in pybullet if a collision is detected

        Returns:
            [(conf_1), (conf_2), ..., (conf_n)] | None: path of manipulator and base
        """

        def get_sample_fn(body, joints, frozen_joints, frozen_values, custom_limits={}, **kwargs):
            lower_limits, upper_limits = pp.get_custom_limits(
                body, joints, custom_limits, circular_limits=pp.CIRCULAR_LIMITS
            )
            generator = pp.interval_generator(lower_limits, upper_limits, **kwargs)

            def fn():
                sample = list(next(generator))
                for id, value in zip(frozen_joints, frozen_values):
                    sample[id] = value
                return tuple(sample)

            return fn

        # -------------------- init params --------------------#
        custom_limits = get_custom_limits(self.robot, {})
        resolutions = np.array(list(map(lambda id: 1.0 if id in frozen_joints else np.pi / 180.0, self.control_joints)))
        disabled_collisions = disabled_collisions or {}
        extra_disabled_collisions = [
            ((self.robot, pp.link_from_name(self.robot, "ur_arm_wrist_3_link")), (attachments[0].child, pp.BASE_LINK)),
        ]

        # -------------------- init functions --------------------#
        # sample_fn = pp.get_sample_fn(self.robot, self.control_joints, custom_limits=custom_limits)
        sample_fn = get_sample_fn(
            self.robot,
            self.control_joints,
            frozen_joints=frozen_joints,
            frozen_values=frozen_values,
            custom_limits=custom_limits,
        )
        distance_fn = pp.get_distance_fn(self.robot, self.control_joints)
        extend_fn = pp.get_extend_fn(self.robot, self.control_joints, resolutions=resolutions)

        transit_collision_fn = pp.get_collision_fn(
            self.robot,
            self.control_joints,
            obstacles=obstacles,
            attachments=attachments,
            self_collisions=True,
            disabled_collisions=disabled_collisions,
            extra_disabled_collisions=extra_disabled_collisions,
            custom_limits=custom_limits,
            max_distance=0.0,
        )

        transit_collision_fn_debug = partial(transit_collision_fn, diagnosis=diagnosis)

        transit_path = None
        with pp.WorldSaver():
            # if pp.check_initial_end(start_conf, end_conf, transit_collision_fn, diagnosis=debug):
            if not transit_collision_fn(end_conf, diagnosis=diagnosis):
                transit_path = pp.solve_motion_plan(
                    start_conf,
                    end_conf,
                    distance_fn,
                    sample_fn,
                    extend_fn,
                    transit_collision_fn_debug,
                    algorithm="birrt",
                    max_time=10,
                    max_iterations=40,
                    smooth=40,
                    diagnosis=diagnosis,
                    coarse_waypoints=coarse_waypoints,
                )
            else:
                print("end collision not pass")

        if isinstance(transit_path, bool):
            transit_path = None

        return transit_path

    @staticmethod
    def parse_urdf(urdf_path: str) -> Dict:
        """
        Parse URDF file and extract joint info.

        Params:
            urdf_path (str): path of urdf file

        Returns:
            Dict: joint info
        """
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        joints = {}
        for joint in root.findall("joint"):
            name = joint.get("name")
            joint_type = joint.get("type")

            origin = joint.find("origin")
            if origin is not None:
                xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
                rpy = [float(r) for r in origin.get("rpy", "0 0 0").split()]
            else:
                xyz, rpy = [0, 0, 0], [0, 0, 0]

            axis = joint.find("axis")
            if axis is not None:
                axis = [float(a) for a in axis.get("xyz", "1 0 0").split()]
            else:
                axis = [1, 0, 0]

            joints[name] = {"type": joint_type, "origin": {"xyz": xyz, "rpy": rpy}, "axis": axis}

        return joints

    @staticmethod
    def rpy_2_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Compute matrix given by rpy.

        Params:
            roll (float): roll
            pitch (float): pitch
            yaw (float): yaw

        Returns:
            np.ndarray: 3x3 matrix
        """
        Rx = np.zeros((3, 3))

        Rx[0, 0] = 1
        Rx[0, 1] = 0
        Rx[0, 2] = 0

        Rx[1, 0] = 0
        Rx[1, 1] = np.cos(roll)
        Rx[1, 2] = -np.sin(roll)

        Rx[2, 0] = 0
        Rx[2, 1] = np.sin(roll)
        Rx[2, 2] = np.cos(roll)

        Ry = np.zeros((3, 3))

        Ry[0, 0] = np.cos(pitch)
        Ry[0, 1] = 0
        Ry[0, 2] = np.sin(pitch)

        Ry[1, 0] = 0
        Ry[1, 1] = 1
        Ry[1, 2] = 0

        Ry[2, 0] = -np.sin(pitch)
        Ry[2, 1] = 0
        Ry[2, 2] = np.cos(pitch)

        Rz = np.zeros((3, 3))

        Rz[0, 0] = np.cos(yaw)
        Rz[0, 1] = -np.sin(yaw)
        Rz[0, 2] = 0

        Rz[1, 0] = np.sin(yaw)
        Rz[1, 1] = np.cos(yaw)
        Rz[1, 2] = 0

        Rz[2, 0] = 0
        Rz[2, 1] = 0
        Rz[2, 2] = 1

        return Rz @ Ry @ Rx

    @staticmethod
    def skew(v: ca.MX) -> ca.MX:
        """
        Generate skew-symmetric matrix given by axis.

        Params:
            v (ca.MX): vector of axis

        Returns:
            ca.MX: skew-symmetric matrix
        """
        assert v.size1() == 3, "输入向量必须是三维的"

        x = v[0]
        y = v[1]
        z = v[2]

        skew_matrix = ca.MX(3, 3)

        skew_matrix[0, 0] = 0
        skew_matrix[0, 1] = -z
        skew_matrix[0, 2] = y

        skew_matrix[1, 0] = z
        skew_matrix[1, 1] = 0
        skew_matrix[1, 2] = -x

        skew_matrix[2, 0] = -y
        skew_matrix[2, 1] = x
        skew_matrix[2, 2] = 0

        return skew_matrix

    @staticmethod
    def expm(A: ca.MX, n_terms: int = 20) -> ca.MX:
        """
        Compute the exponential exp(A) of the matrix A using Taylor series expansion.

        Params:
            A (ca.MX): matrix
            n_terms (int, 20): number of expanded items

        Returns:
            ca.MX: matrix exp(A)
        """
        if A.size1() != A.size2():
            raise ValueError("矩阵必须是方阵")

        exp_A = ca.MX.eye(A.size1())
        A_power = ca.MX.eye(A.size1())
        factorial = 1

        for n in range(1, n_terms + 1):
            A_power = A_power @ A
            factorial *= n
            exp_A += A_power / factorial

        return exp_A

    @staticmethod
    def transform_matrix(
        xyz: List[float], rpy: List[float], axis: List[float], q_val: ca.MX, joint_type: str
    ) -> ca.MX:
        """
        Construct transformation matrix according to joint type.

        Params:
            xyz (List[float]): xyz
            rpy (List[float]): rpy
            axis (List[float]): axis
            q_val (ca.MX): symbolic joint angle
            joint_type (str): revolute/prismatic

        Returns:
            ca.MX: 4x4 matrix
        """
        T = ca.MX.eye(4)
        T[:3, :3] = RobotSetup.rpy_2_matrix(*rpy)
        T[:3, 3] = xyz

        if joint_type == "revolute":
            R_joint = ca.MX.eye(4)
            R_joint[:3, :3] = RobotSetup.expm(q_val * RobotSetup.skew(ca.MX(axis)))
            return T @ R_joint
        elif joint_type == "prismatic":
            P_joint = ca.MX.eye(4)
            P_joint[:3, 3] = ca.MX(axis) * q_val
            return T @ P_joint
        else:
            return T

    @staticmethod
    def symbolic_forward(
        urdf_path: str,
        joint_name_list: List[str],
        control_joint_name_list: List[str],
        q: Union[ca.MX, None] = None,
        output_type: str = "function",
    ):
        """
        Creates symbolic forward kinematics equations given a URDF file path and a list of joint names.

        Params:
            urdf_path (str): urdf path of robot
            joint_name_list (List[str]): name list of manipulator including all redundant joints
            control_joint_name_list (List[str]): name list of controlled joints
            q (ca.MX | None, None): joint variables
            output_type (str, "function"): "function"/"matrix"

        Returns:
            ca.Function: [q] --> np.ndarray (4x4)
        """
        joints = RobotSetup.parse_urdf(urdf_path)
        if q is None or len(control_joint_name_list) == 0:
            q = ca.MX.sym("q", len(control_joint_name_list))
        T = ca.MX.eye(4)
        for i, joint_name in enumerate(joint_name_list):
            joint = joints[joint_name]
            if joint_name in control_joint_name_list:
                i_q = control_joint_name_list.index(joint_name)
                joint_T = RobotSetup.transform_matrix(
                    joint["origin"]["xyz"], joint["origin"]["rpy"], joint["axis"], q[i_q], joint["type"]
                )
            else:
                joint_T = RobotSetup.transform_matrix(
                    joint["origin"]["xyz"], joint["origin"]["rpy"], joint["axis"], 0, joint["type"]
                )
            T = T @ joint_T

        if output_type == "function":
            fk_function = ca.Function("forward_kinematics", [q], [T])
            return fk_function
        elif output_type == "matrix":
            return T
        else:
            fk_function = ca.Function("forward_kinematics", [q], [T])
            return fk_function
