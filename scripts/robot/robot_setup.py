import os
import sys
from functools import partial
from typing import Dict, List, Set, Tuple, Union

import numpy as np
from utils.params import DATA_DIRECTORY, PICK_DIRECTION, PROJECT_DIR

HUSKY_ASSEMBLY_PATH = os.path.join(PROJECT_DIR, "src")
sys.path.extend([HUSKY_ASSEMBLY_PATH, PROJECT_DIR])

import pybullet_planning as pp
from compas_fab.robots import RobotSemantics
from compas_fab.robots.robot import RobotModel
from pybullet_planning import Attachment, Euler, Point, Pose, multiply
from solver.ik_pinocchio_solver import PinocchioSolver
from utils.utils import HUSKYU_JOINT_NAMES

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
ONBOARD_POSE = (
    [0.0, -0.5, 0.5, -np.pi / 2, 0.0, np.pi / 2] if PICK_DIRECTION == "left" else [0.4, 0.0, 0.5, -np.pi / 2, 0.0, 0.0]
)
ONBOARD_LINK = "ur_arm_base_link"


class RobotSetup:
    """Handles robot setup, kinematics, and motion planning using Pinocchio IK solver."""

    def __init__(self, robot_name: str = "r0", attachments: List[Attachment] = None):
        """Initialize the RobotSetup instance.

        Args:
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
        self.arm_joints = pp.joints_from_names(self.robot, HUSKYU_JOINT_NAMES)
        self.base_joints = pp.joints_from_names(self.robot, BASE_CONTROL_JOINT_NAMES)
        self.arm_init_angles = INIT_ARM_JOINT_ANGLES
        self.set_joint_positions(self.arm_joints, self.arm_init_angles)

    def _load_robot(self) -> Dict:
        """Load robot URDF and configure Pinocchio IK solver for relative kinematics.

        Returns:
            Dict containing robot, ee_attachment, ik_solver_relative, and disabled_collisions.

        Raises:
            FileNotFoundError: If required files are missing.
        """
        robot_urdf = os.path.join(DATA_DIRECTORY, "husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf")
        robot_srdf = os.path.join(DATA_DIRECTORY, "husky_urdf/mt_husky_moveit_config/config/husky.srdf")
        gripper_obj = os.path.join(DATA_DIRECTORY, "husky_urdf/robotiq_85/meshes/static/robotiq_85_close_20mm.obj")

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

        Args:
            control_joints: List of joint indices.
            conf: Joint configuration array.
        """
        pp.set_joint_positions(self.robot, control_joints, conf)
        self.ee_attachment.assign()
        for attachment in self.attachments:
            attachment.assign()

    def get_disabled_collisions_from_link_names(
        self, robot: int, link_names: Set[Tuple[str, str]]
    ) -> Set[Tuple[int, int]]:
        """Get link pairs disabled from collision checking.

        Args:
            robot: PyBullet robot ID.
            link_names: Set of link name pairs to disable.

        Returns:
            Set of tuples containing link indices.
        """
        return {
            tuple(pp.link_from_name(robot, link) for link in pair if pp.has_link(robot, link)) for pair in link_names
        }

    def get_relative_pose(self, pose_world: Tuple, link_name: str = "ur_arm_base_link") -> Tuple:
        """Calculate pose relative to a specified link.

        Args:
            pose_world: World frame pose as (position, orientation).
            link_name: Name of the reference link (default: "ur_arm_base_link").

        Returns:
            Relative pose as (position, orientation).
        """
        link_pose = pp.get_link_pose(self.robot, pp.link_from_name(self.robot, link_name))
        return pp.multiply(pp.invert(link_pose), pose_world)

    def get_relative_ik_solution(self, tool_pose_world: Tuple, q_init: List[float] = None) -> np.ndarray:
        """Calculate inverse kinematics solution relative to base using Pinocchio.

        Args:
            tool_pose_world: Tool pose in world frame.
            q_init: Initial joint configuration guess (default: None).

        Returns:
            Joint configuration solving the IK problem.
        """
        tool_pose_relative = self.get_relative_pose(tool_pose_world)
        tform = pp.tform_from_pose(tool_pose_relative)
        conf = self.ik_solver_relative(tform, qinit=q_init)
        self.ee_attachment.assign()
        return conf

    def plan_manipulator_path(
        self, init_q: np.ndarray, target_q: np.ndarray, attachments: List[Attachment], obstacles: Set[int]
    ) -> List[np.ndarray]:
        """Plan a manipulator path from initial to target configuration.

        Args:
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

        base_conf = pp.get_joint_positions(self.robot, self.base_joints)
        init_conf = np.hstack((base_conf, init_q))
        target_conf = np.hstack((base_conf, target_q))
        frozen_joints = self.base_joints
        frozen_values = [init_conf[self.control_joints.index(j)] for j in frozen_joints]

        path = self.plan_manipulator_motion(
            init_conf,
            target_conf,
            [self.ee_attachment] + attachments,
            obstacles,
            disabled_collisions=self.disabled_collisions,
            frozen_joints=frozen_joints,
            frozen_values=frozen_values,
        )
        return [np.array(conf)[3:] for conf in path] if path else None

    def set_base_pose(self, pose: Pose) -> None:
        """Set the robot's base pose and update attachments.

        Args:
            pose: Base pose to set.
        """
        pp.set_pose(self.robot, pose)
        self.ee_attachment.assign()
        for attachment in self.attachments:
            attachment.assign()

    def set_base_pose_2d(self, x: float, y: float, yaw: float = 0.0) -> None:
        """Set the robot's base pose in 2D and update attachments.

        Args:
            x: X-coordinate.
            y: Y-coordinate.
            yaw: Yaw angle in radians (default: 0.0).
        """
        pose = pp.Pose(point=[x, y, 0], euler=pp.Euler(yaw=yaw))
        self.set_base_pose(pose)

    def update_attachments(self, attachments: List[Attachment]) -> None:
        """Update the list of attachments.

        Args:
            attachments: New list of attachments.
        """
        self.attachments = attachments

    def create_aboard_attachment(self, body: int) -> Attachment:
        """Create an attachment on the robot at the onboard link.

        Args:
            body: PyBullet body ID to attach.

        Returns:
            Attachment object linking the robot and body.
        """
        link_pose = pp.get_link_pose(self.robot, pp.link_from_name(self.robot, ONBOARD_LINK))
        delta_pose = Pose(point=ONBOARD_POSE[:3], euler=Euler(*ONBOARD_POSE[3:]))
        body_pose = multiply(link_pose, delta_pose)
        pp.set_pose(body, body_pose)
        return pp.create_attachment(self.robot, pp.link_from_name(self.robot, ONBOARD_LINK), body)

    def plan_manipulator_motion(
        self, start_conf: np.ndarray, end_conf: np.ndarray, attachments: List[Attachment], obstacles: Set[int], **kwargs
    ) -> Union[List[Tuple[float]], None]:
        """Plan a motion path for the manipulator.

        Args:
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

        def get_sample_fn():
            lower, upper = pp.get_custom_limits(self.robot, self.control_joints, circular_limits=pp.CIRCULAR_LIMITS)
            generator = pp.interval_generator(lower, upper)

            def fn():
                sample = list(next(generator))
                for idx, val in zip(frozen_joints, frozen_values):
                    sample[idx] = val
                return tuple(sample)

            return fn

        sample_fn = get_sample_fn()
        distance_fn = pp.get_distance_fn(self.robot, self.control_joints)
        resolutions = np.array([1.0 if j in frozen_joints else np.pi / 180.0 for j in self.control_joints])
        extend_fn = pp.get_extend_fn(self.robot, self.control_joints, resolutions=resolutions)
        collision_fn = pp.get_collision_fn(
            self.robot,
            self.control_joints,
            obstacles=obstacles,
            attachments=attachments,
            self_collisions=True,
            disabled_collisions=disabled_collisions,
            extra_disabled_collisions=[
                (
                    (self.robot, pp.link_from_name(self.robot, "ur_arm_wrist_3_link")),
                    (attachments[0].child, pp.BASE_LINK),
                )
            ],
        )

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
                    max_time=10,
                    max_iterations=40,
                    smooth=40,
                    diagnosis=diagnosis,
                    coarse_waypoints=coarse_waypoints,
                )
            print("End configuration in collision.")
            return None
