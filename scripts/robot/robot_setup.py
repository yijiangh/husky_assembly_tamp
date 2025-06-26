import os
import sys
import xml.etree.ElementTree as ET
from functools import partial
from typing import Callable, Dict, List, Set, Tuple, Union

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

# from utils.util import HUSKY_ARM_JOINT_NAMES
from utils.params import URDF_PATH
from utils.utils_casadi import eval

# Husky specific constants
HUSKY_TOOL0_FROM_EE_POSE = pp.Pose(point=[0, 0, 0.160])
HUSKY_TOOL0_NAME = "ur_arm_tool0"
HUSKY_INIT_ARM_JOINT_ANGLES = np.array([0, -np.pi / 2, 0, 0, 0, 0])
HUSKY_ONBOARD_LINK = "ur_arm_base_link"
HUSKY_ONBOARD_POSE = [0.0, -0.5, 0.5, -np.pi / 2, 0.0, np.pi / 2] if PICK_DIRECTION == "left" else [0.4, 0.0, 0.5, -np.pi / 2, 0.0, 0.0]
HUSKY_GRASP_MASK_LINKS = ["ur_arm_wrist_3_link"]

# Husky joint names
HUSKY_CONTROL_JOINT_NAMES = ["x", "y", "theta", "ur_arm_shoulder_pan_joint", "ur_arm_shoulder_lift_joint", "ur_arm_elbow_joint", "ur_arm_wrist_1_joint", "ur_arm_wrist_2_joint", "ur_arm_wrist_3_joint"]
HUSKY_ARM_JOINT_NAMES = ["ur_arm_shoulder_pan_joint", "ur_arm_shoulder_lift_joint", "ur_arm_elbow_joint", "ur_arm_wrist_1_joint", "ur_arm_wrist_2_joint", "ur_arm_wrist_3_joint"]
HUSKY_BASE_CONTROL_JOINT_NAMES = ["x", "y", "theta"]
HUSKY_BASE_REDUCED_MODEL_JOINT_NAMES = ["base_footprint_joint", "top_plate_joint", "top_plate_front_joint", "arm_mount_joint"]

# Husky file paths
HUSKY_URDF_PATH = os.path.join(DATA_DIR, "husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf")
HUSKY_SRDF_PATH = os.path.join(DATA_DIR, "husky_urdf/mt_husky_moveit_config/config/husky.srdf")
HUSKY_GRIPPER_OBJ = os.path.join(DATA_DIR, "husky_urdf/robotiq_85/meshes/static/robotiq_85_close_20mm.obj")

# Constants for ABB
ABB_TOOL0_NAME = "eef_tcp_frame"
ABB_INIT_ARM_JOINT_ANGLES = np.array([0, 0, 0, 0, 0, 0])
ABB_ONBOARD_LINK = "world_link"
ABB_ONBOARD_POSE = [0, 0, 0, 0, 0, 0]
ABB_GRASP_MASK_LINKS = ["eef_tcp_frame", "eef_base_link"]

# ABB joint names
ABB_JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
ABB_BASE_CONTROL_JOINT_NAMES = []
ABB_CONTROL_JOINT_NAMES = ABB_JOINT_NAMES
ABB_BASE_REDUCED_MODEL_JOINT_NAMES = []

# ABB file paths
ABB_URDF_PATH = os.path.join(DATA_DIR, "abb_irb4600_40_255/urdf/ECL_robot1_with_track.urdf")
ABB_SRDF_PATH = os.path.join(DATA_DIR, "abb_irb4600_40_255/srdf/ECL_robot1_with_track.srdf")

HUSKY_DUAL_TOOL0_LEFT = "left_ur_arm_tool0"
HUSKY_DUAL_TOOL0_RIGHT = "right_ur_arm_tool0"
HUSKY_DUAL_ARM_JOINT_NAMES_LEFT = ["left_ur_arm_shoulder_pan_joint", "left_ur_arm_shoulder_lift_joint", "left_ur_arm_elbow_joint", "left_ur_arm_wrist_1_joint", "left_ur_arm_wrist_2_joint", "left_ur_arm_wrist_3_joint"]
HUSKY_DUAL_ARM_JOINT_NAMES_RIGHT = ["right_ur_arm_shoulder_pan_joint", "right_ur_arm_shoulder_lift_joint", "right_ur_arm_elbow_joint", "right_ur_arm_wrist_1_joint", "right_ur_arm_wrist_2_joint", "right_ur_arm_wrist_3_joint"]
HUSKY_DUAL_BASE_CONTROL_JOINT_NAMES = []
HUSKY_DUAL_CONTROL_JOINT_NAMES_LEFT = HUSKY_BASE_CONTROL_JOINT_NAMES + HUSKY_DUAL_ARM_JOINT_NAMES_LEFT
HUSKY_DUAL_CONTROL_JOINT_NAMES_RIGHT = HUSKY_BASE_CONTROL_JOINT_NAMES + HUSKY_DUAL_ARM_JOINT_NAMES_RIGHT
HUSKY_DUAL_ONBOARD_LINK_LEFT = "left_ur_arm_base_link"
HUSKY_DUAL_ONBOARD_LINK_RIGHT = "right_ur_arm_base_link"
HUSKY_DUAL_GRASP_MASK_LINKS_LEFT = ["left_ur_arm_wrist_3_link"]
HUSKY_DUAL_GRASP_MASK_LINKS_RIGHT = ["right_ur_arm_wrist_3_link"]
HUSKY_DUAL_URDF_PATH = os.path.join(DATA_DIR, "husky_urdf/mt_husky_dual_ur5_e_moveit_config/urdf/husky_dual_ur5_e.urdf")
HUSKY_DUAL_SRDF_PATH = os.path.join(DATA_DIR, "husky_urdf/mt_husky_dual_ur5_e_moveit_config/config/husky.srdf")

# Combined joints for dual-arm system
HUSKY_DUAL_ARM_JOINT_NAMES = HUSKY_DUAL_ARM_JOINT_NAMES_LEFT + HUSKY_DUAL_ARM_JOINT_NAMES_RIGHT

# Combine base control joints with dual-arm joints
HUSKY_DUAL_CONTROL_JOINT_NAMES = HUSKY_DUAL_BASE_CONTROL_JOINT_NAMES + HUSKY_DUAL_ARM_JOINT_NAMES

# New constant for dual arms
HUSKY_DUAL_INIT_ARM_JOINT_ANGLES = np.tile(HUSKY_INIT_ARM_JOINT_ANGLES, 2)


class RobotSetup:
    """Handles robot setup, kinematics, and motion planning using Pinocchio IK solver."""

    def __init__(
        self,
        robot_name: str = "r0",
        attachments: List[Attachment] = None,
        robot_type: str = "husky",
    ):
        """Initialize the RobotSetup instance.

        Params:
            robot_name: Name of the robot (default: "r0").
            attachments: List of attachments (default: None).
            robot_type: Type of robot to setup ("husky", "abb", or "husky_dual", default: "husky").
        """
        self.name = robot_name
        self.attachments = attachments or []

        # Remember the robot type for conditional logic in setup
        self.robot_type = robot_type

        # Initialize robot parameters
        self._init_robot_params(robot_type)

        # Set up the robot
        self._setup_robot()

    def _init_robot_params(self, robot_type: str) -> None:
        """Initialize robot-specific parameters based on robot type.

        Params:
            robot_type: Type of robot ("husky" or "abb")
        """
        self.robot_params = {}

        # Common parameters with default values
        self.robot_params["tool0_from_ee"] = np.eye(4)
        self.robot_params["base_from_connect"] = np.eye(4)

        if robot_type == "husky":
            self.robot_params["urdf_path"] = HUSKY_URDF_PATH
            self.robot_params["srdf_path"] = HUSKY_SRDF_PATH
            self.robot_params["gripper_obj"] = HUSKY_GRIPPER_OBJ
            self.robot_params["tool0_name"] = HUSKY_TOOL0_NAME
            self.robot_params["joint_names"] = HUSKY_ARM_JOINT_NAMES
            self.robot_params["init_angles"] = HUSKY_INIT_ARM_JOINT_ANGLES
            self.robot_params["tool0_from_ee"] = HUSKY_TOOL0_FROM_EE_POSE
            self.robot_params["onboard_link"] = HUSKY_ONBOARD_LINK
            self.robot_params["onboard_pose"] = HUSKY_ONBOARD_POSE
            self.robot_params["base_control_joint_names"] = HUSKY_BASE_CONTROL_JOINT_NAMES
            self.robot_params["control_joint_names"] = HUSKY_CONTROL_JOINT_NAMES
            self.robot_params["base_reduced_model_joint_names"] = HUSKY_BASE_REDUCED_MODEL_JOINT_NAMES
            self.robot_params["grasp_mask_links"] = HUSKY_GRASP_MASK_LINKS

            # Pre-calculate base_from_connect for Husky
            base_from_connect_sym = RobotSetup.symbolic_forward(URDF_PATH, HUSKY_BASE_REDUCED_MODEL_JOINT_NAMES, [], output_type="matrix")
            self.robot_params["base_from_connect"] = eval("base_from_connect", base_from_connect_sym, [], [])

        elif robot_type == "abb":
            self.robot_params["urdf_path"] = ABB_URDF_PATH
            self.robot_params["srdf_path"] = ABB_SRDF_PATH
            self.robot_params["gripper_obj"] = None
            self.robot_params["tool0_name"] = ABB_TOOL0_NAME
            self.robot_params["joint_names"] = ABB_JOINT_NAMES
            self.robot_params["init_angles"] = ABB_INIT_ARM_JOINT_ANGLES
            self.robot_params["onboard_link"] = ABB_ONBOARD_LINK
            self.robot_params["onboard_pose"] = ABB_ONBOARD_POSE
            self.robot_params["base_control_joint_names"] = ABB_BASE_CONTROL_JOINT_NAMES
            self.robot_params["control_joint_names"] = ABB_CONTROL_JOINT_NAMES
            self.robot_params["base_reduced_model_joint_names"] = ABB_BASE_REDUCED_MODEL_JOINT_NAMES
            self.robot_params["grasp_mask_links"] = ABB_GRASP_MASK_LINKS
            # ABB doesn't need base_from_connect calculation

        elif robot_type == "husky_dual":
            self.robot_params["urdf_path"] = HUSKY_DUAL_URDF_PATH
            self.robot_params["srdf_path"] = HUSKY_DUAL_SRDF_PATH
            self.robot_params["gripper_obj"] = HUSKY_GRIPPER_OBJ
            self.robot_params["tool0_name_right"] = HUSKY_DUAL_TOOL0_RIGHT
            self.robot_params["tool0_name_left"] = HUSKY_DUAL_TOOL0_LEFT
            self.robot_params["joint_names"] = HUSKY_DUAL_ARM_JOINT_NAMES
            self.robot_params["right_joint_names"] = HUSKY_DUAL_ARM_JOINT_NAMES_RIGHT
            self.robot_params["left_joint_names"] = HUSKY_DUAL_ARM_JOINT_NAMES_LEFT
            self.robot_params["init_angles"] = HUSKY_DUAL_INIT_ARM_JOINT_ANGLES
            self.robot_params["tool0_from_ee"] = HUSKY_TOOL0_FROM_EE_POSE
            self.robot_params["onboard_link_right"] = HUSKY_DUAL_ONBOARD_LINK_RIGHT
            self.robot_params["onboard_link_left"] = HUSKY_DUAL_ONBOARD_LINK_LEFT
            self.robot_params["onboard_pose"] = HUSKY_ONBOARD_POSE
            self.robot_params["base_control_joint_names"] = HUSKY_DUAL_BASE_CONTROL_JOINT_NAMES
            self.robot_params["control_joint_names"] = HUSKY_DUAL_CONTROL_JOINT_NAMES
            self.robot_params["base_reduced_model_joint_names"] = HUSKY_BASE_REDUCED_MODEL_JOINT_NAMES
            self.robot_params["grasp_mask_links"] = HUSKY_DUAL_GRASP_MASK_LINKS_LEFT + HUSKY_DUAL_GRASP_MASK_LINKS_RIGHT

            # Skip symbolic_forward computation for dual-arm variant for now
            self.robot_params["base_from_connect"] = np.eye(4)
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")

    def _setup_robot(self) -> None:
        """Load robot model and initialize components with Pinocchio IK solver."""
        # Load the robot data
        robot_data = self._load_robot()

        # Set up the robot and its components
        self.robot = robot_data["robot"]
        self.ee_attachment = robot_data["ee_attachment"]
        if self.robot_type == "husky_dual":
            self.left_ee_attachment = robot_data["left_ee_attachment"]
            self.right_ee_attachment = robot_data["right_ee_attachment"]
            self.ik_solver_relative_left = robot_data["ik_solver_relative_left"]
            self.ik_solver_relative_right = robot_data["ik_solver_relative_right"]
            self.ik_solver_relative = self.ik_solver_relative_left  # Default to left for backward compatibility
        else:
            self.ik_solver_relative = robot_data["ik_solver_relative"]
        self.disabled_collisions = robot_data["disabled_collisions"]

        # Set up robot links and joints
        if self.robot_type == "husky_dual":
            self.tool_link_left = pp.link_from_name(self.robot, self.robot_params["tool0_name_left"])
            self.tool_link_right = pp.link_from_name(self.robot, self.robot_params["tool0_name_right"])
            self.tool_link = self.tool_link_left  # Default to left for backward compatibility
            self.arm_joints_left = pp.joints_from_names(self.robot, self.robot_params["left_joint_names"])
            self.arm_joints_right = pp.joints_from_names(self.robot, self.robot_params["right_joint_names"])
            self.arm_joints = self.arm_joints_left + self.arm_joints_right
        else:
            self.tool_link = pp.link_from_name(self.robot, self.robot_params["tool0_name"])
            self.arm_joints = pp.joints_from_names(self.robot, self.robot_params["joint_names"])
        self.arm_init_angles = self.robot_params["init_angles"]

        # Set up base joints if they exist
        if self.robot_params["base_control_joint_names"]:
            self.base_joints = pp.joints_from_names(self.robot, self.robot_params["base_control_joint_names"])
        else:
            self.base_joints = []

        # Set up control joints if they exist
        if self.robot_params["control_joint_names"]:
            self.control_joints = pp.joints_from_names(self.robot, self.robot_params["control_joint_names"])
        else:
            self.control_joints = []

        # Initialize joint positions
        self.set_joint_positions(self.arm_joints, self.arm_init_angles)

    def _load_robot(self) -> Dict:
        """Load robot URDF and configure Pinocchio IK solver for relative kinematics.

        Returns:
            Dict containing robot, ee_attachment(s), ik_solver_relative(s), and disabled_collisions.
            For dual-arm robots, also includes left_ee_attachment, right_ee_attachment, 
            ik_solver_relative_left, and ik_solver_relative_right.

        Raises:
            FileNotFoundError: If required files are missing.
        """
        # Get robot parameters
        robot_urdf = self.robot_params["urdf_path"]
        robot_srdf = self.robot_params["srdf_path"]
        gripper_obj = self.robot_params["gripper_obj"]
        
        # Handle tool0_name for different robot types
        if self.robot_type == "husky_dual":
            # For dual-arm, we don't use a single tool0_name
            tool0_name = None
        else:
            tool0_name = self.robot_params["tool0_name"]

        # Check if files exist
        if not os.path.exists(robot_urdf) or not os.path.exists(robot_srdf):
            raise FileNotFoundError("Required robot files not found.")

        # Load robot in PyBullet
        robot = pp.load_pybullet(robot_urdf, fixed_base=False, cylinder=False)

        # By default, populate empty placeholders
        disabled_collisions = set()
        ik_solver_relative = lambda *args, **kwargs: None  # default no-op solver

        robot_model = RobotModel.from_urdf_file(robot_urdf)
        semantics = RobotSemantics.from_srdf_file(robot_srdf, robot_model)
        disabled_collisions = self.get_disabled_collisions_from_link_names(robot, semantics.disabled_collisions)
        if self.robot_type != "husky_dual":
            pinocchio_solver = PinocchioSolver(
                robot_urdf,
                manipulator_joint_names=self.robot_params["joint_names"],
                control_joint_names=self.robot_params["control_joint_names"],
                base_link_name=self.robot_params["onboard_link"],
            )
            ik_solver_relative = partial(pinocchio_solver.ik, tip_name=tool0_name)
        else:
            pinocchio_solver_right = PinocchioSolver(
                robot_urdf,
                manipulator_joint_names=self.robot_params["joint_names"],
                control_joint_names=self.robot_params["right_joint_names"],
                base_link_name=self.robot_params["onboard_link_right"]
            )
            pinocchio_solver_left = PinocchioSolver(
                robot_urdf,
                manipulator_joint_names=self.robot_params["joint_names"],
                control_joint_names=self.robot_params["left_joint_names"],
                base_link_name=self.robot_params["onboard_link_left"]
            )
            ik_solver_relative_left = partial(pinocchio_solver_left.ik, tip_name=self.robot_params["tool0_name_left"])
            ik_solver_relative_right = partial(pinocchio_solver_right.ik, tip_name=self.robot_params["tool0_name_right"])

        # Create ee attachment if gripper exists
        ee_attachment = None
        left_ee_attachment = None
        right_ee_attachment = None
        
        if gripper_obj and os.path.exists(gripper_obj):
            if self.robot_type == "husky_dual":
                # Create left ee attachment
                left_tool0_pose = pp.get_link_pose(robot, pp.link_from_name(robot, self.robot_params["tool0_name_left"]))
                left_ee = pp.create_obj(gripper_obj, scale=1)
                pp.set_pose(left_ee, pp.multiply(left_tool0_pose, pp.Pose(euler=pp.Euler(yaw=-np.pi / 2))))
                left_ee_attachment = pp.create_attachment(robot, pp.link_from_name(robot, self.robot_params["tool0_name_left"]), left_ee)
                
                # Create right ee attachment
                right_tool0_pose = pp.get_link_pose(robot, pp.link_from_name(robot, self.robot_params["tool0_name_right"]))
                right_ee = pp.create_obj(gripper_obj, scale=1)
                pp.set_pose(right_ee, pp.multiply(right_tool0_pose, pp.Pose(euler=pp.Euler(yaw=-np.pi / 2))))
                right_ee_attachment = pp.create_attachment(robot, pp.link_from_name(robot, self.robot_params["tool0_name_right"]), right_ee)
                
                # For backward compatibility, set ee_attachment to left
                ee_attachment = left_ee_attachment
            else:
                tool0_pose = pp.get_link_pose(robot, pp.link_from_name(robot, tool0_name))
                ee = pp.create_obj(gripper_obj, scale=1)
                pp.set_pose(ee, pp.multiply(tool0_pose, pp.Pose(euler=pp.Euler(yaw=-np.pi / 2))))
                ee_attachment = pp.create_attachment(robot, pp.link_from_name(robot, tool0_name), ee)

        if self.robot_type == "husky_dual":
            return {
                "robot": robot,
                "ee_attachment": ee_attachment,
                "left_ee_attachment": left_ee_attachment,
                "right_ee_attachment": right_ee_attachment,
                "ik_solver_relative_left": ik_solver_relative_left,
                "ik_solver_relative_right": ik_solver_relative_right,
                "disabled_collisions": disabled_collisions,
            }
        else:
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
        
        # Update ee attachments
        if self.robot_type == "husky_dual":
            if self.left_ee_attachment:
                self.left_ee_attachment.assign()
            if self.right_ee_attachment:
                self.right_ee_attachment.assign()
        else:
            if self.ee_attachment:
                self.ee_attachment.assign()
                
        # Update other attachments
        for attachment in self.attachments:
            attachment.assign()

    def set_left_arm_joint_positions(self, conf: np.ndarray) -> None:
        """Set left arm joint positions (dual-arm robots only).
        
        Params:
            conf: Joint configuration array for left arm.
        """
        if self.robot_type != "husky_dual":
            raise ValueError("Left arm joint setting is only available for dual-arm robots")
        pp.set_joint_positions(self.robot, self.arm_joints_left, conf)
        if self.left_ee_attachment:
            self.left_ee_attachment.assign()
        for attachment in self.attachments:
            attachment.assign()

    def set_right_arm_joint_positions(self, conf: np.ndarray) -> None:
        """Set right arm joint positions (dual-arm robots only).
        
        Params:
            conf: Joint configuration array for right arm.
        """
        if self.robot_type != "husky_dual":
            raise ValueError("Right arm joint setting is only available for dual-arm robots")
        pp.set_joint_positions(self.robot, self.arm_joints_right, conf)
        if self.right_ee_attachment:
            self.right_ee_attachment.assign()
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

    def get_relative_pose(self, pose_world: Tuple, link_name: str = None, arm_side: str = None) -> Tuple:
        """Calculate pose relative to a specified link.

        Params:
            pose_world: World frame pose as (position, orientation).
            link_name: Name of the reference link (default: robot's onboard link).
            arm_side: For dual-arm robots, specify "left" or "right" to use corresponding onboard link.

        Returns:
            Relative pose as (position, orientation).
        """
        if link_name is None:
            # Select appropriate onboard link based on robot type and arm_side
            if self.robot_type == "husky_dual":
                if arm_side == "right":
                    link_name = self.robot_params["onboard_link_right"]
                else:  # Default to left if not specified or if specified as "left"
                    link_name = self.robot_params["onboard_link_left"]
            else:
                link_name = self.robot_params["onboard_link"]

        link_pose = pp.get_link_pose(self.robot, pp.link_from_name(self.robot, link_name))
        return pp.multiply(pp.invert(link_pose), pose_world)

    def get_left_relative_pose(self, pose_world: Tuple) -> Tuple:
        """Calculate pose relative to left arm base link (dual-arm robots only).

        Params:
            pose_world: World frame pose as (position, orientation).

        Returns:
            Relative pose as (position, orientation).
        """
        if self.robot_type != "husky_dual":
            raise ValueError("Left relative pose is only available for dual-arm robots")
        return self.get_relative_pose(pose_world, arm_side="left")

    def get_right_relative_pose(self, pose_world: Tuple) -> Tuple:
        """Calculate pose relative to right arm base link (dual-arm robots only).

        Params:
            pose_world: World frame pose as (position, orientation).

        Returns:
            Relative pose as (position, orientation).
        """
        if self.robot_type != "husky_dual":
            raise ValueError("Right relative pose is only available for dual-arm robots")
        return self.get_relative_pose(pose_world, arm_side="right")

    def get_relative_ik_solution(self, world_from_tool: Tuple, q_init: List[float] = None, arm_side: str = None) -> np.ndarray:
        """Calculate inverse kinematics solution relative to base using Pinocchio.

        Params:
            world_from_tool: Tool pose in world frame.
            q_init: Initial joint configuration guess (default: None).
            arm_side: For dual-arm robots, specify "left" or "right" (default: None uses left or single arm).

        Returns:
            Joint configuration solving the IK problem.
        """
        # Use the pre-calculated base_from_connect (for all robot types)
        world_from_connect = pp.multiply(pp.get_pose(self.robot), pp.pose_from_tform(self.robot_params["base_from_connect"]))
        connect_from_tool = pp.multiply(pp.invert(world_from_connect), world_from_tool)
        tform = pp.tform_from_pose(connect_from_tool)

        # Select appropriate solver based on arm_side for dual-arm robots
        if self.robot_type == "husky_dual":
            if arm_side == "right":
                solver = self.ik_solver_relative_right
            else:  # Default to left if not specified or if specified as "left"
                solver = self.ik_solver_relative_left
        else:
            solver = self.ik_solver_relative

        conf = solver(tform, qinit=q_init)
        
        # Update ee attachments
        if self.robot_type == "husky_dual":
            if arm_side == "right" and self.right_ee_attachment:
                self.right_ee_attachment.assign()
            elif self.left_ee_attachment:  # Default to left or explicitly left
                self.left_ee_attachment.assign()
        else:
            if self.ee_attachment:
                self.ee_attachment.assign()

        if conf is not None:
            # 将关节角度规范化到 [-pi, pi] 范围内
            conf = np.array([(angle + np.pi) % (2 * np.pi) - np.pi for angle in conf])
        return conf

    def get_grasp_ik_solution(self, world_from_object: Tuple, tool_from_obj: Tuple, q_init: List[float] = None, arm_side: str = None) -> np.ndarray:
        """Calculate inverse kinematics solution relative to base using Pinocchio.

        Params:
            world_from_object: Object pose in world frame.
            tool_from_obj: Tool pose relative to object.
            q_init: Initial joint configuration guess (default: None).
            arm_side: For dual-arm robots, specify "left" or "right" (default: None uses left or single arm).
        """
        world_from_tool = pp.multiply(world_from_object, pp.invert(tool_from_obj))
        return self.get_relative_ik_solution(world_from_tool, q_init, arm_side)

    def get_left_arm_ik_solution(self, world_from_tool: Tuple, q_init: List[float] = None) -> np.ndarray:
        """Calculate IK solution for left arm (dual-arm robots only).
        
        Params:
            world_from_tool: Tool pose in world frame.
            q_init: Initial joint configuration guess (default: None).
            
        Returns:
            Joint configuration solving the IK problem for left arm.
        """
        if self.robot_type != "husky_dual":
            raise ValueError("Left arm IK is only available for dual-arm robots")
        return self.get_relative_ik_solution(world_from_tool, q_init, arm_side="left")

    def get_right_arm_ik_solution(self, world_from_tool: Tuple, q_init: List[float] = None) -> np.ndarray:
        """Calculate IK solution for right arm (dual-arm robots only).
        
        Params:
            world_from_tool: Tool pose in world frame.
            q_init: Initial joint configuration guess (default: None).
            
        Returns:
            Joint configuration solving the IK problem for right arm.
        """
        if self.robot_type != "husky_dual":
            raise ValueError("Right arm IK is only available for dual-arm robots")
        return self.get_relative_ik_solution(world_from_tool, q_init, arm_side="right")

    def get_left_ee_attachment(self):
        """Get left end-effector attachment (dual-arm robots only).
        
        Returns:
            Left end-effector attachment object.
        """
        if self.robot_type != "husky_dual":
            raise ValueError("Left EE attachment is only available for dual-arm robots")
        return self.left_ee_attachment

    def get_right_ee_attachment(self):
        """Get right end-effector attachment (dual-arm robots only).
        
        Returns:
            Right end-effector attachment object.
        """
        if self.robot_type != "husky_dual":
            raise ValueError("Right EE attachment is only available for dual-arm robots")
        return self.right_ee_attachment

    def get_ee_attachment(self, arm_side: str = None):
        """Get end-effector attachment for specified arm.
        
        Params:
            arm_side: For dual-arm robots, specify "left" or "right" (default: None uses left or single arm).
            
        Returns:
            End-effector attachment object.
        """
        if self.robot_type == "husky_dual":
            if arm_side == "right":
                return self.right_ee_attachment
            else:  # Default to left if not specified or if specified as "left"
                return self.left_ee_attachment
        else:
            return self.ee_attachment

    def get_tool_link(self, arm_side: str = None):
        """Get tool link for specified arm.
        
        Params:
            arm_side: For dual-arm robots, specify "left" or "right" (default: None uses left or single arm).
            
        Returns:
            Tool link ID.
        """
        if self.robot_type == "husky_dual":
            if arm_side == "right":
                return self.tool_link_right
            else:  # Default to left if not specified or if specified as "left"
                return self.tool_link_left
        else:
            return self.tool_link

    def get_arm_joints(self, arm_side: str = None):
        """Get arm joints for specified arm.
        
        Params:
            arm_side: For dual-arm robots, specify "left" or "right" (default: None uses all joints or single arm).
            
        Returns:
            List of joint indices.
        """
        if self.robot_type == "husky_dual":
            if arm_side == "right":
                return self.arm_joints_right
            elif arm_side == "left":
                return self.arm_joints_left
            else:  # Return all arm joints if no specific side specified
                return self.arm_joints
        else:
            return self.arm_joints

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

        # Create a list of all attachments
        attachments_list = []
        if self.robot_type == "husky_dual":
            if self.left_ee_attachment:
                attachments_list.append(self.left_ee_attachment)
            if self.right_ee_attachment:
                attachments_list.append(self.right_ee_attachment)
        else:
            if self.ee_attachment:
                attachments_list.append(self.ee_attachment)
        attachments_list.extend(attachments)

        # Ensure all attachments are assigned
        for att in attachments_list:
            att.assign()

        path = self._plan_manipulator_motion(init_q, target_q, attachments_list, obstacles, disabled_collisions=self.disabled_collisions, **kwargs)
        return np.array([np.array(conf) for conf in path]) if path else None

    def set_base_pose(self, pose: Pose) -> None:
        """Set the robot's base pose and update attachments.

        Params:
            pose: Base pose to set.
        """
        pp.set_pose(self.robot, pose)
        
        # Update ee attachments
        if self.robot_type == "husky_dual":
            if self.left_ee_attachment:
                self.left_ee_attachment.assign()
            if self.right_ee_attachment:
                self.right_ee_attachment.assign()
        else:
            if self.ee_attachment:
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

    def create_aboard_attachment(self, body: int, arm_side: str = None) -> Attachment:
        """Create an attachment on the robot at the onboard link.

        Params:
            body: PyBullet body ID to attach.
            arm_side: For dual-arm robots, specify "left" or "right" (default: None uses left or single arm).

        Returns:
            Attachment object linking the robot and body.
        """
        # Select appropriate onboard link based on robot type and arm_side
        if self.robot_type == "husky_dual":
            if arm_side == "right":
                onboard_link_name = self.robot_params["onboard_link_right"]
            else:  # Default to left if not specified or if specified as "left"
                onboard_link_name = self.robot_params["onboard_link_left"]
        else:
            onboard_link_name = self.robot_params["onboard_link"]

        link_pose = pp.get_link_pose(self.robot, pp.link_from_name(self.robot, onboard_link_name))
        delta_pose = Pose(point=self.robot_params["onboard_pose"][:3], euler=Euler(*self.robot_params["onboard_pose"][3:]))
        body_pose = multiply(link_pose, delta_pose)
        pp.set_pose(body, body_pose)
        return pp.create_attachment(self.robot, pp.link_from_name(self.robot, onboard_link_name), body)

    def create_left_aboard_attachment(self, body: int) -> Attachment:
        """Create an attachment on the left arm onboard link (dual-arm robots only).

        Params:
            body: PyBullet body ID to attach.

        Returns:
            Attachment object linking the robot and body.
        """
        if self.robot_type != "husky_dual":
            raise ValueError("Left aboard attachment is only available for dual-arm robots")
        return self.create_aboard_attachment(body, arm_side="left")

    def create_right_aboard_attachment(self, body: int) -> Attachment:
        """Create an attachment on the right arm onboard link (dual-arm robots only).

        Params:
            body: PyBullet body ID to attach.

        Returns:
            Attachment object linking the robot and body.
        """
        if self.robot_type != "husky_dual":
            raise ValueError("Right aboard attachment is only available for dual-arm robots")
        return self.create_aboard_attachment(body, arm_side="right")

    def _plan_manipulator_motion(self, start_conf: np.ndarray, end_conf: np.ndarray, attachments: List[Attachment], obstacles: Set[int], **kwargs) -> Union[List[Tuple[float]], None]:
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

    def create_collision_fn(self, obstacle_bodies: List[int] = []) -> Callable[[np.ndarray], bool]:
        """Create PyBullet-based collision function"""
        robot_body = self.robot
        arm_joints = self.arm_joints
        attachments = []
        if self.robot_type == "husky_dual":
            if self.left_ee_attachment:
                attachments.append(self.left_ee_attachment)
            if self.right_ee_attachment:
                attachments.append(self.right_ee_attachment)
        else:
            if self.ee_attachment:
                attachments.append(self.ee_attachment)
        attachments.extend(self.attachments)
        disabled_collisions = self.disabled_collisions

        # Get tool link and wrist link based on robot type
        extra_disabled_collisions = []
        for link_name in self.robot_params["grasp_mask_links"]:
            link = pp.link_from_name(robot_body, link_name)
            for attachment in attachments:
                extra_disabled_collisions.extend(
                    [
                        ((robot_body, link), (attachment.child, pp.BASE_LINK)),
                    ]
                )

        grasped_collision_fn_list = []
        for attachment in self.attachments:
            grasped_collision_fn_list.append(pp.get_floating_body_collision_fn(attachment.child, obstacles=obstacle_bodies + [self.robot], disabled_collisions=extra_disabled_collisions))

        robot_collision_fn = pp.get_collision_fn(
            robot_body, arm_joints, obstacles=obstacle_bodies, attachments=attachments, self_collisions=True, disabled_collisions=disabled_collisions, extra_disabled_collisions=extra_disabled_collisions, max_distance=0.0
        )

        def collision_fn(joint_conf, diagnosis=False):
            """Check if a given joint configuration results in a collision.

            Args:
                joint_conf (np.ndarray): Joint configuration.
                diagnosis (bool, False): Whether to return diagnosis information.

            Returns:
                bool: True if there is a collision, False otherwise.
            """
            robot_collision = robot_collision_fn(joint_conf, diagnosis=diagnosis)
            self.set_joint_positions(arm_joints, joint_conf)
            grasped_collision = False
            for idx, grasped_collision_fn in enumerate(grasped_collision_fn_list):
                pose = pp.get_pose(self.attachments[idx].child)
                grasped_collision = grasped_collision or grasped_collision_fn(pose, diagnosis=diagnosis)
            return grasped_collision or robot_collision

        return collision_fn

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
        assert v.size1() == 3, "Input vector must be three-dimensional"

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
            raise ValueError("Matrix must be square")

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
