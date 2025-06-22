import os
import sys
import xml.etree.ElementTree as ET
from functools import partial
from typing import Dict, List, Set, Tuple, Union, Callable

import casadi as ca
import numpy as np
from utils.params import DATA_DIR, PICK_DIRECTION, PROJECT_DIR

HUSKY_ASSEMBLY_PATH = os.path.join(PROJECT_DIR, "src")
sys.path.extend([HUSKY_ASSEMBLY_PATH, PROJECT_DIR])

import pybullet_planning as pp
from compas_fab.robots import RobotSemantics
from compas_fab.robots.robot import RobotModel
from pybullet_planning import Attachment, Euler, Point, Pose, multiply
from solver.ik_pinocchio_solver import PinocchioSolver
from utils.util import HUSKY_ARM_JOINT_NAMES
from utils.params import URDF_PATH
from utils.utils_casadi import eval

# Constants
TOOL0_FROM_EE_POSE = pp.Pose(point=[0, 0, 0.160])
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

# Onboard configuration based on pick direction
ONBOARD_POSE = [0.0, -0.5, 0.5, -np.pi / 2, 0.0, np.pi / 2] if PICK_DIRECTION == "left" else [0.4, 0.0, 0.5, -np.pi / 2, 0.0, 0.0]
ONBOARD_LINK = "ur_arm_base_link"


class RobotSetup:
    """Handles robot setup, kinematics, and motion planning using Pinocchio IK solver."""

    MANIPULATOR_CONTROL_JOINT_NAMES = [
        "ur_arm_shoulder_pan_joint",
        "ur_arm_shoulder_lift_joint",
        "ur_arm_elbow_joint",
        "ur_arm_wrist_1_joint",
        "ur_arm_wrist_2_joint",
        "ur_arm_wrist_3_joint",
    ]

    MANIPULATOR_REDUCED_MODEL_JOINT_NAMES = [
        "ur_arm_base_link-base_fixed_joint",
        "ur_arm_shoulder_pan_joint",
        "ur_arm_shoulder_lift_joint",
        "ur_arm_elbow_joint",
        "ur_arm_wrist_1_joint",
        "ur_arm_wrist_2_joint",
        "ur_arm_wrist_3_joint",
        "ur_arm_wrist_3-flange",
        "ur_arm_flange-tool0",
        "tool0-bar_tcp_fixed_joint",
    ]

    BASE_CONTROL_JOINT_NAMES = []

    BASE_REDUCED_MODEL_JOINT_NAMES = [
        "base_footprint_joint",
        "top_plate_joint",
        "top_plate_front_joint",
        "arm_mount_joint",
        # "ur_arm_base_link-base_fixed_joint",
    ]

    def __init__(self, robot_name: str = "r0", attachments: List[Attachment] = None):
        """Initialize the RobotSetup instance.

        Params:
            robot_name: Name of the robot (default: "r0").
            attachments: List of attachments (default: None).
        """
        self.name = robot_name
        self.attachments = attachments or []
        self._setup_robot()

    def _setup_robot(self) -> None:
        """Load robot model and initialize components with Pinocchio IK solver."""
        robot_data = self._load_robot()
        self.robot = robot_data["robot"]
        self.ee_attachment = robot_data["ee_attachment"]
        self.ik_solver_relative = robot_data["ik_solver_relative"]
        self.disabled_collisions = robot_data["disabled_collisions"]
        self.tool0_from_ee = TOOL0_FROM_EE_POSE
        self.tool_link = pp.link_from_name(self.robot, "ur_arm_tool0")

        self.control_joints = pp.joints_from_names(self.robot, CONTROL_JOINT_NAMES)
        self.arm_joints = pp.joints_from_names(self.robot, HUSKY_ARM_JOINT_NAMES)
        self.base_joints = pp.joints_from_names(self.robot, BASE_CONTROL_JOINT_NAMES)
        self.arm_init_angles = INIT_ARM_JOINT_ANGLES
        self.set_joint_positions(self.arm_joints, self.arm_init_angles)

        base_from_connect_sym = RobotSetup.symbolic_forward(URDF_PATH, self.BASE_REDUCED_MODEL_JOINT_NAMES, self.BASE_CONTROL_JOINT_NAMES, output_type="matrix")
        self.base_from_connect = eval("base_from_connect", base_from_connect_sym, [], [])

    def _load_robot(self) -> Dict:
        """Load robot URDF and configure Pinocchio IK solver for relative kinematics.

        Returns:
            Dict containing robot, ee_attachment, ik_solver_relative, and disabled_collisions.

        Raises:
            FileNotFoundError: If required files are missing.
        """
        robot_urdf = os.path.join(DATA_DIR, "husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf")
        robot_srdf = os.path.join(DATA_DIR, "husky_urdf/mt_husky_moveit_config/config/husky.srdf")
        gripper_obj = os.path.join(DATA_DIR, "husky_urdf/robotiq_85/meshes/static/robotiq_85_close_20mm.obj")
        # gripper_obj = os.path.join(DATA_DIR, "husky_urdf/robotiq_85/meshes/static/robotiq_85_open.obj")

        if not all(os.path.exists(path) for path in [robot_urdf, robot_srdf, gripper_obj]):
            raise FileNotFoundError("Required robot or gripper files not found.")

        robot = pp.load_pybullet(robot_urdf, fixed_base=False, cylinder=False)
        robot_model = RobotModel.from_urdf_file(robot_urdf)
        semantics = RobotSemantics.from_srdf_file(robot_srdf, robot_model)
        disabled_collisions = self.get_disabled_collisions_from_link_names(robot, semantics.disabled_collisions)

        # Configure Pinocchio IK solver for relative IK only
        pinocchio_solver = PinocchioSolver(robot_urdf)
        ik_solver_relative = partial(pinocchio_solver.ik, tip_name="ur_arm_tool0")

        tool0_pose = pp.get_link_pose(robot, pp.link_from_name(robot, "ur_arm_tool0"))
        ee = pp.create_obj(gripper_obj, scale=1)
        pp.set_pose(ee, pp.multiply(tool0_pose, pp.Pose(euler=pp.Euler(yaw=-np.pi / 2))))
        ee_attachment = pp.create_attachment(robot, pp.link_from_name(robot, "ur_arm_tool0"), ee)

        return {
            "robot": robot,
            "ee_attachment": ee_attachment,
            "ik_solver_relative": ik_solver_relative,
            "disabled_collisions": disabled_collisions,
        }

    def set_joint_positions(self, control_joints: List[int], conf: np.ndarray) -> None:
        """Set joint positions and update attachments.

        Params:
            control_joints: List of joint indices.
            conf: Joint configuration array.
        """
        pp.set_joint_positions(self.robot, control_joints, conf)
        self.ee_attachment.assign()
        for attachment in self.attachments:
            attachment.assign()

    def get_disabled_collisions_from_link_names(self, robot: int, link_names: Set[Tuple[str, str]]) -> Set[Tuple[int, int]]:
        """Get link pairs disabled from collision checking.

        Params:
            robot: PyBullet robot ID.
            link_names: Set of link name pairs to disable.

        Returns:
            Set of tuples containing link indices.
        """
        return {tuple(pp.link_from_name(robot, link) for link in pair if pp.has_link(robot, link)) for pair in link_names}

    def get_relative_pose(self, pose_world: Tuple, link_name: str = "ur_arm_base_link") -> Tuple:
        """Calculate pose relative to a specified link.

        Params:
            pose_world: World frame pose as (position, orientation).
            link_name: Name of the reference link (default: "ur_arm_base_link").

        Returns:
            Relative pose as (position, orientation).
        """
        link_pose = pp.get_link_pose(self.robot, pp.link_from_name(self.robot, link_name))
        return pp.multiply(pp.invert(link_pose), pose_world)

    def get_relative_ik_solution(self, world_from_tool: Tuple, q_init: List[float] = None) -> np.ndarray:
        """Calculate inverse kinematics solution relative to base using Pinocchio.

        Params:
            tool_pose_world: Tool pose in world frame.
            q_init: Initial joint configuration guess (default: None).

        Returns:
            Joint configuration solving the IK problem.
        """
        world_from_connect = pp.multiply(pp.get_pose(self.robot), pp.pose_from_tform(self.base_from_connect))
        connect_from_tool = pp.multiply(pp.invert(world_from_connect), world_from_tool)
        tform = pp.tform_from_pose(connect_from_tool)
        conf = self.ik_solver_relative(tform, qinit=q_init)
        self.ee_attachment.assign()
        return conf

    def plan_manipulator_path(self, init_q: np.ndarray, target_q: np.ndarray, attachments: List[Attachment], obstacles: Set[int], **kwargs) -> np.ndarray:
        """Plan a manipulator path from initial to target configuration.

        Params:
            init_q: Initial joint configuration.
            target_q: Target joint configuration.
            attachments: List of attachments excluding ee_attachment.
            obstacles: Set of obstacle IDs.

        Returns:
            List of joint configurations forming the path, or None if planning fails.
        """
        self.set_joint_positions(self.arm_joints, init_q)
        for att in [self.ee_attachment] + attachments:
            att.assign()

        path = self.plan_manipulator_motion(init_q, target_q, [self.ee_attachment] + attachments, obstacles, disabled_collisions=self.disabled_collisions, **kwargs)
        return np.array([np.array(conf) for conf in path]) if path else None

    def set_base_pose(self, pose: Pose) -> None:
        """Set the robot's base pose and update attachments.

        Params:
            pose: Base pose to set.
        """
        pp.set_pose(self.robot, pose)
        self.ee_attachment.assign()
        for attachment in self.attachments:
            attachment.assign()

    def set_base_pose_2d(self, x: float, y: float, yaw: float = 0.0) -> None:
        """Set the robot's base pose in 2D and update attachments.

        Params:
            x: X-coordinate.
            y: Y-coordinate.
            yaw: Yaw angle in radians (default: 0.0).
        """
        pose = pp.Pose(point=[x, y, 0], euler=pp.Euler(yaw=yaw))
        self.set_base_pose(pose)

    def update_attachments(self, attachments: List[Attachment]) -> None:
        """Update the list of attachments.

        Params:
            attachments: New list of attachments.
        """
        self.attachments = attachments

    def create_aboard_attachment(self, body: int) -> Attachment:
        """Create an attachment on the robot at the onboard link.

        Params:
            body: PyBullet body ID to attach.

        Returns:
            Attachment object linking the robot and body.
        """
        link_pose = pp.get_link_pose(self.robot, pp.link_from_name(self.robot, ONBOARD_LINK))
        delta_pose = Pose(point=ONBOARD_POSE[:3], euler=Euler(*ONBOARD_POSE[3:]))
        body_pose = multiply(link_pose, delta_pose)
        pp.set_pose(body, body_pose)
        return pp.create_attachment(self.robot, pp.link_from_name(self.robot, ONBOARD_LINK), body)

    def plan_manipulator_motion(self, start_conf: np.ndarray, end_conf: np.ndarray, attachments: List[Attachment], obstacles: Set[int], **kwargs) -> Union[List[Tuple[float]], None]:
        """Plan a motion path for the manipulator.

        Params:
            start_conf: Starting configuration.
            end_conf: Target configuration.
            attachments: List of attachments.
            obstacles: Set of obstacle IDs.
            **kwargs: Additional options (e.g., disabled_collisions, frozen_joints).

        Returns:
            List of configurations forming the path, or None if planning fails.
        """
        disabled_collisions = kwargs.get("disabled_collisions", {})
        frozen_joints = kwargs.get("frozen_joints", [])
        frozen_values = kwargs.get("frozen_values", [])
        coarse_waypoints = kwargs.get("coarse_waypoints", False)
        diagnosis = kwargs.get("diagnosis", False)
        max_time = kwargs.get("max_time", 10)
        max_iterations = kwargs.get("max_iterations", 10000)
        smooth = kwargs.get("smooth", 40)
        resolution = kwargs.get("resolution", 1.0)

        resolutions = np.array([resolution if j in frozen_joints else resolution / 180.0 * np.pi for j in self.arm_joints])

        def get_sample_fn():
            lower, upper = pp.get_custom_limits(self.robot, self.arm_joints, circular_limits=pp.CIRCULAR_LIMITS)
            generator = pp.interval_generator(lower, upper)

            def fn():
                sample = list(next(generator))
                for idx, val in zip(frozen_joints, frozen_values):
                    sample[idx] = val
                return tuple(sample)

            return fn

        default_sample_fn = get_sample_fn()
        default_distance_fn = pp.get_distance_fn(self.robot, self.arm_joints)
        default_extend_fn = pp.get_extend_fn(self.robot, self.arm_joints, resolutions=resolutions)
        default_collision_fn = self.create_collision_fn(obstacles)

        sample_fn = kwargs.get("sample_fn", default_sample_fn)
        distance_fn = kwargs.get("distance_fn", default_distance_fn)
        extend_fn = kwargs.get("extend_fn", default_extend_fn)
        collision_fn = kwargs.get("collision_fn", default_collision_fn)

        with pp.WorldSaver():
            if not collision_fn(end_conf, diagnosis=diagnosis):
                return pp.solve_motion_plan(
                    start_conf,
                    end_conf,
                    distance_fn,
                    sample_fn,
                    extend_fn,
                    partial(collision_fn, diagnosis=diagnosis),
                    algorithm="birrt",
                    max_time=max_time,
                    max_iterations=max_iterations,
                    smooth=smooth,
                    diagnosis=diagnosis,
                    coarse_waypoints=coarse_waypoints,
                )
            print("End configuration in collision.")
            return None

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
    def transform_matrix(xyz: List[float], rpy: List[float], axis: List[float], q_val: ca.MX, joint_type: str) -> ca.MX:
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
                joint_T = RobotSetup.transform_matrix(joint["origin"]["xyz"], joint["origin"]["rpy"], joint["axis"], q[i_q], joint["type"])
            else:
                joint_T = RobotSetup.transform_matrix(joint["origin"]["xyz"], joint["origin"]["rpy"], joint["axis"], 0, joint["type"])
            T = T @ joint_T

        if output_type == "function":
            fk_function = ca.Function("forward_kinematics", [q], [T])
            return fk_function
        elif output_type == "matrix":
            return T
        else:
            fk_function = ca.Function("forward_kinematics", [q], [T])
            return fk_function

    def create_collision_fn(self, obstacle_bodies: List[int]) -> Callable[[np.ndarray], bool]:
        """Create PyBullet-based collision function"""
        robot_body = self.robot
        arm_joints = self.arm_joints
        attachments = [self.ee_attachment] + self.attachments
        disabled_collisions = self.disabled_collisions
        tool_link = self.tool_link
        wrist_link = pp.link_from_name(robot_body, "ur_arm_wrist_3_link")

        extra_disabled_collisions = []
        if self.ee_attachment is not None:
            extra_disabled_collisions.extend(
                [
                    ((robot_body, wrist_link), (self.ee_attachment.child, pp.BASE_LINK)),
                ]
            )

        grasped_collision_fn_list = []
        for attachment in self.attachments:
            grasped_collision_fn_list.append(pp.get_floating_body_collision_fn(attachment.child, obstacles=obstacle_bodies + [self.robot]))
        robot_collision_fn = pp.get_collision_fn(
            robot_body, arm_joints, obstacles=obstacle_bodies, attachments=attachments, self_collisions=True, disabled_collisions=disabled_collisions, extra_disabled_collisions=extra_disabled_collisions, max_distance=0.0
        )

        def collision_fn(joint_conf, diagnosis=False):
            """
            检查给定关节配置是否发生碰撞

            Args:
                joint_conf (np.ndarray): 关节配置
                diagnosis (bool, False): 是否返回诊断信息

            Returns:
                bool: 如果有碰撞返回True，否则返回False
            """
            robot_collision = robot_collision_fn(joint_conf, diagnosis=diagnosis)
            self.set_joint_positions(arm_joints, joint_conf)
            grasped_collision = False
            for idx, grasped_collision_fn in enumerate(grasped_collision_fn_list):
                pose = pp.get_pose(self.attachments[idx].child)
                grasped_collision = grasped_collision or grasped_collision_fn(pose, diagnosis=diagnosis)
            return grasped_collision or robot_collision

        return collision_fn
