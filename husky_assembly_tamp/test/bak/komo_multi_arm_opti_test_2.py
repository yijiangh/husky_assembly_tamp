import numpy as np
import pybullet_planning as pp
import robotic as ry
import json
import os
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field


def horizontal_cylinder_quaternion(direction: np.ndarray) -> List[float]:
    """Calculate quaternion for a horizontal cylinder aligned with the given direction."""
    dir_norm = direction / np.linalg.norm(direction)
    z_axis = np.array([0, 0, 1])
    rot_axis = np.cross(z_axis, dir_norm)
    if np.linalg.norm(rot_axis) < 1e-6:
        rot_axis = np.array([1, 0, 0])
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    angle = np.pi / 2
    return [np.cos(angle / 2), rot_axis[0] * np.sin(angle / 2), rot_axis[1] * np.sin(angle / 2), rot_axis[2] * np.sin(angle / 2)]


def normalize_vector(vec: np.ndarray, default: Optional[np.ndarray] = None) -> np.ndarray:
    """Normalize a vector, returning default if the vector is too small."""
    if default is None:
        default = np.array([1, 0, 0])
    return vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 1e-6 else default


def generate_random_initial_state(config: ry.Config) -> np.ndarray:
    """Generate a random initial joint state within joint limits."""
    low, high = config.getJointLimits()
    q_random = np.random.uniform(low, high, size=len(low))
    return q_random


@dataclass
class CylinderElement:
    """
    Represents a cylindrical element with position, orientation, and geometric properties.

    Attributes:
        name: Unique identifier for the element
        position: 3D position [x, y, z]
        quaternion: Orientation as quaternion [w, x, y, z]
        length: Length of the cylinder (along its axis)
        radius: Radius of the cylinder
        color: RGB or RGBA color
        contact: Whether this element participates in collision detection
        direction: Direction vector of the cylinder axis (for horizontal cylinders)
    """

    name: str
    position: np.ndarray
    quaternion: List[float] = field(default_factory=lambda: [1, 0, 0, 0])
    length: float = 1.0
    radius: float = 0.01
    color: List[float] = field(default_factory=lambda: [1, 1, 1])
    contact: bool = True
    direction: Optional[np.ndarray] = None

    def __post_init__(self):
        self.position = np.array(self.position)
        if self.direction is not None:
            self.direction = np.array(self.direction)

    @property
    def size(self) -> List[float]:
        """Return cylinder size as [length, radius] for ry.ST.cylinder."""
        return [self.length, self.radius]

    def add_to_config(self, config: ry.Config) -> ry.Frame:
        """
        Add this element to a ry.Config.

        Args:
            config: The ry.Config to add the element to

        Returns:
            The created frame
        """
        frame = config.addFrame(self.name)
        frame.setShape(ry.ST.cylinder, self.size)
        frame.setPosition(self.position.tolist())
        frame.setQuaternion(self.quaternion)
        frame.setColor(self.color)
        if self.contact:
            frame.setContact(1)
        return frame

    @classmethod
    def create_vertical(
        cls,
        name: str,
        position: np.ndarray,
        length: float = 1.0,
        radius: float = 0.01,
        color: Optional[List[float]] = None,
        contact: bool = True,
    ) -> "CylinderElement":
        """
        Create a vertical cylinder element (aligned with Z-axis).

        Args:
            name: Element name
            position: 3D position
            length: Cylinder length
            radius: Cylinder radius
            color: Color (default white)
            contact: Enable collision detection
        """
        if color is None:
            color = [1, 1, 1]
        return cls(
            name=name,
            position=np.array(position),
            quaternion=[1, 0, 0, 0],
            length=length,
            radius=radius,
            color=color,
            contact=contact,
            direction=np.array([0, 0, 1]),
        )

    @classmethod
    def create_horizontal(
        cls,
        name: str,
        position: np.ndarray,
        direction: np.ndarray,
        length: float = 1.0,
        radius: float = 0.01,
        color: Optional[List[float]] = None,
        contact: bool = True,
    ) -> "CylinderElement":
        """
        Create a horizontal cylinder element aligned with the given direction.

        Args:
            name: Element name
            position: 3D position
            direction: Direction vector for the cylinder axis
            length: Cylinder length
            radius: Cylinder radius
            color: Color (default white)
            contact: Enable collision detection
        """
        if color is None:
            color = [1, 1, 1]
        direction = np.array(direction)
        quaternion = horizontal_cylinder_quaternion(direction)
        return cls(
            name=name,
            position=np.array(position),
            quaternion=quaternion,
            length=length,
            radius=radius,
            color=color,
            contact=contact,
            direction=direction,
        )

    def get_end_position(self, target_pos: np.ndarray) -> np.ndarray:
        """
        Get the endpoint of the cylinder that is closer to the target position.

        Args:
            target_pos: Target position to determine which end

        Returns:
            Position of the cylinder end closer to target
        """
        if self.direction is None:
            raise ValueError("Direction not set for this element")

        dir_norm = normalize_vector(self.direction)
        half_length = self.length / 2

        dir_to_target = np.array([target_pos[0] - self.position[0], target_pos[1] - self.position[1], 0])
        dir_to_target_norm = normalize_vector(dir_to_target, dir_norm)
        dot = np.dot(dir_norm, dir_to_target_norm)

        return self.position + dir_norm * half_length * (1 if dot > 0 else -1)

    def get_other_end_position(self, target_pos: np.ndarray) -> np.ndarray:
        """
        Get the endpoint of the cylinder that is farther from the target position.

        Args:
            target_pos: Target position to determine which end

        Returns:
            Position of the cylinder end farther from target
        """
        if self.direction is None:
            raise ValueError("Direction not set for this element")

        dir_norm = normalize_vector(self.direction)
        half_length = self.length / 2

        dir_to_target = np.array([target_pos[0] - self.position[0], target_pos[1] - self.position[1], 0])
        dir_to_target_norm = normalize_vector(dir_to_target, dir_norm)
        dot = np.dot(dir_norm, dir_to_target_norm)

        return self.position - dir_norm * half_length * (1 if dot > 0 else -1)


class GeometryCalculator:
    """Utility class for geometric calculations related to element positioning."""

    @staticmethod
    def calculate_edge_direction(v_start: np.ndarray, v_end: np.ndarray) -> np.ndarray:
        """Calculate the direction vector of an edge between two vertices (XY plane)."""
        return np.array([v_end[0] - v_start[0], v_end[1] - v_start[1], 0])

    @staticmethod
    def calculate_edge_midpoint(v_start: np.ndarray, v_end: np.ndarray, z_pos: float) -> np.ndarray:
        """Calculate the midpoint of an edge at a specified Z position."""
        return np.array([(v_start[0] + v_end[0]) / 2, (v_start[1] + v_end[1]) / 2, z_pos])

    @staticmethod
    def calculate_protrusion_offset(edge_mid: np.ndarray, protrusion_target: np.ndarray, offset_distance: float) -> np.ndarray:
        """Calculate position offset towards a protrusion target."""
        protrusion_dir = np.array([protrusion_target[0] - edge_mid[0], protrusion_target[1] - edge_mid[1], 0])
        protrusion_dir = normalize_vector(protrusion_dir) * offset_distance
        return edge_mid + protrusion_dir

    @staticmethod
    def calculate_horizontal_element_position(
        v_start: np.ndarray,
        v_end: np.ndarray,
        protrusion_target: np.ndarray,
        z_pos: float,
        protrusion_offset: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate position and direction for a horizontal element.

        Returns:
            Tuple of (element_position, edge_direction)
        """
        edge_dir = GeometryCalculator.calculate_edge_direction(v_start, v_end)
        edge_mid = GeometryCalculator.calculate_edge_midpoint(v_start, v_end, z_pos)
        element_pos = GeometryCalculator.calculate_protrusion_offset(edge_mid, protrusion_target, protrusion_offset)
        return element_pos, edge_dir

    @staticmethod
    def calculate_vertical_element_position(
        element_end: np.ndarray,
        initial_v_pos: np.ndarray,
        vertical_distance: float,
        z_pos: float,
    ) -> np.ndarray:
        """Calculate position for a vertical element near a horizontal element's end."""
        dir_to_v = np.array([initial_v_pos[0] - element_end[0], initial_v_pos[1] - element_end[1], 0])
        dir_to_v_norm = normalize_vector(dir_to_v)
        v_pos = (element_end + dir_to_v_norm * vertical_distance).tolist()
        v_pos[2] = z_pos
        return np.array(v_pos)


class RobotPositionCalculator:
    """Utility class for calculating robot base positions relative to elements."""

    @staticmethod
    def calculate_position_perpendicular(element_pos: np.ndarray, edge_dir: np.ndarray, distance: float) -> np.ndarray:
        """
        Calculate robot position perpendicular to an element's edge direction.

        Args:
            element_pos: Position of the element
            edge_dir: Direction of the element's edge
            distance: Distance from element (positive = one side, negative = other side)

        Returns:
            Robot base position [x, y, z]
        """
        edge_dir_norm = normalize_vector(edge_dir)
        perp_dir = np.array([-edge_dir_norm[1], edge_dir_norm[0], 0])
        element_xy = np.array([element_pos[0], element_pos[1], 0.0])
        robot_pos = element_xy + perp_dir * distance
        return robot_pos

    @staticmethod
    def calculate_pose_toward_target(
        element_pos: np.ndarray,
        edge_dir: np.ndarray,
        distance: float,
        look_at: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Calculate a base pose (position + yaw quaternion) that faces a target.

        Args:
            element_pos: Position of the element used for perpendicular offset
            edge_dir: Edge direction of the element
            distance: Perpendicular offset distance
            look_at: Optional target position to face (defaults to element_pos)

        Returns:
            Tuple of (position, quaternion) where quaternion is [w, x, y, z]
        """
        position = RobotPositionCalculator.calculate_position_perpendicular(element_pos, edge_dir, distance)
        if look_at is None:
            look_at = element_pos

        heading = np.array([look_at[0] - position[0], look_at[1] - position[1], 0])
        heading_norm = normalize_vector(heading, default=np.array([1, 0, 0]))
        yaw = np.arctan2(heading_norm[1], heading_norm[0])
        quaternion = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        return position, quaternion

    @staticmethod
    def apply_base_poses(config: ry.Config, base_frame_names: List[str], base_poses: List[Tuple[np.ndarray, List[float]]]) -> None:
        """Set position + orientation for each base frame."""
        for frame_name, (position, quaternion) in zip(base_frame_names, base_poses):
            frame = config.getFrame(frame_name)
            pos_list = position.tolist() if isinstance(position, np.ndarray) else position
            frame.setPosition(pos_list)
            frame.setQuaternion(quaternion)

    @staticmethod
    def add_robot_to_config(
        config: ry.Config,
        robot_file: str,
        prefix: str,
        position: np.ndarray,
        quaternion: Optional[List[float]] = None,
    ) -> ry.Frame:
        """
        Add a robot to the configuration.

        Args:
            config: The ry.Config to add the robot to
            robot_file: Path to the robot description file
            prefix: Prefix for robot frame names
            position: Robot base position
            quaternion: Robot base orientation (default: [1, 0, 0, 1])

        Returns:
            The robot base frame
        """
        if quaternion is None:
            quaternion = [1, 0, 0, 1]
        pos_list = position.tolist() if isinstance(position, np.ndarray) else position
        return config.addFile(robot_file, prefix).setPosition(pos_list).setQuaternion(quaternion)


def create_horizontal_element(
    config: ry.Config,
    name: str,
    v_start: np.ndarray,
    v_end: np.ndarray,
    protrusion_target: np.ndarray,
    z_pos: float,
    color: List[float],
    length: float = 1.0,
    radius: float = 0.01,
    protrusion_offset: float = 0.15,
    contact: bool = True,
) -> CylinderElement:
    """
    Create a horizontal cylinder element and add it to the config.

    Args:
        config: The ry.Config to add the element to
        name: Element name
        v_start: Start vertex of the edge
        v_end: End vertex of the edge
        protrusion_target: Target point for protrusion direction
        z_pos: Z position
        color: Element color
        length: Cylinder length
        radius: Cylinder radius
        protrusion_offset: Distance to offset from edge midpoint
        contact: Enable collision detection

    Returns:
        CylinderElement object
    """
    element_pos, edge_dir = GeometryCalculator.calculate_horizontal_element_position(v_start, v_end, protrusion_target, z_pos, protrusion_offset)

    element = CylinderElement.create_horizontal(
        name=name,
        position=element_pos,
        direction=edge_dir,
        length=length,
        radius=radius,
        color=color,
        contact=contact,
    )
    element.add_to_config(config)
    return element


def create_vertical_element(
    config: ry.Config,
    name: str,
    position: np.ndarray,
    color: List[float],
    length: float = 1.0,
    radius: float = 0.01,
    contact: bool = True,
) -> CylinderElement:
    """
    Create a vertical cylinder element and add it to the config.

    Args:
        config: The ry.Config to add the element to
        name: Element name
        position: 3D position
        color: Element color
        length: Cylinder length
        radius: Cylinder radius
        contact: Enable collision detection

    Returns:
        CylinderElement object
    """
    element = CylinderElement.create_vertical(
        name=name,
        position=position,
        length=length,
        radius=radius,
        color=color,
        contact=contact,
    )
    element.add_to_config(config)
    return element


class SingleKeyFrameSolver:
    """
    A solver for multi-phase keyframe optimization using KOMO.

    This class encapsulates the logic for solving inverse kinematics problems
    with gripper constraints for multiple robots across multiple phases.

    Supports nested lists for robot_names and target_names to represent
    multiple robots grasping the same target. For example:
        robot_names_phases = [
            [["r1_gripper", "r2_gripper"], "r3_gripper"],  # Phase 1
            [["r1_gripper", "r2_gripper"], "r3_gripper"],  # Phase 2
        ]
        target_names_phases = [
            [["element_4", "element_4"], "element_5"],  # Phase 1
            [["element_4", "element_4"], "element_6"],  # Phase 2
        ]
    This means in Phase 1: r1+r2 grasp element_4, r3 grasps element_5.
    In Phase 2: r1+r2 grasp element_4, r3 grasps element_6.
    """

    def __init__(
        self,
        config: ry.Config,
        robot_names_phases: List[List[Union[str, List[str]]]],
        target_names_phases: List[List[Union[str, List[str]]]],
        joint_weight: float = 0.1,
        gripper_weight: float = 5.11,
        position_rel_z_bounds: Tuple[float, float] = (0.4, -0.4),
        constraint_eps: float = 1e-3,
        max_attempts: int = 10,
        damping: Optional[float] = None,
        wolfe: Optional[float] = None,
        random_init_mult: float = 3.0,
        random_init_offset: float = -1.5,
        freeze_arm_joints: bool = False,
        x_home: Optional[np.ndarray] = None,
        arm_joint_indices: Optional[List[List[int]]] = None,
        group_distance_constraints: Optional[List[Optional[np.ndarray]]] = None,
        distance_frame_names: Optional[List[Union[str, List[str]]]] = None,
        distance_weight: float = 1.0,
        collision_weight: float = 1.0,
        pose_rel_constraints: Optional[List[Tuple[str, str, List[float]]]] = None,
        pose_rel_weight: float = 1.0,
    ):
        """
        Initialize the SingleKeyFrameSolver.

        Args:
            config: The ry.Config object containing the robot and environment setup
            robot_names_phases: List of robot gripper frame names per phase.
                         Each phase has a list that can contain nested lists for dual-arm grasping.
                         e.g., [[["r1_gripper", "r2_gripper"], "r3_gripper"], [...]]
            target_names_phases: List of target element frame names per phase.
                          Structure must match robot_names_phases.
                          e.g., [[["element_4", "element_4"], "element_5"], [...]]
            joint_weight: Weight for joint state regularization objective
            gripper_weight: Weight for gripper constraint objectives
            position_rel_z_bounds: Tuple of (upper, lower) bounds for relative position Z constraint
            constraint_eps: Tolerance for constraint satisfaction checking
            max_attempts: Maximum number of optimization attempts
            damping: Damping parameter for NLP solver (optional)
            wolfe: Wolfe parameter for NLP solver (optional)
            random_init_mult: Multiplier for random initialization
            random_init_offset: Offset for random initialization
            freeze_arm_joints: Whether to freeze arm joint angles to home position
            x_home: Home joint state for freezing arm joints (required if freeze_arm_joints=True)
            arm_joint_indices: List of lists, each containing joint indices for each robot's arm (required if freeze_arm_joints=True)
            group_distance_constraints: List of upper triangular matrices defining distance constraints
                                        within each group. For a group with n robots, provide an n×n matrix
                                        where element [i,j] (i<j) specifies the target distance between
                                        robot i and robot j. Use -1 to indicate no constraint between two robots.
                                        None matrix means no constraints for that group.
                                        Example for 2 robots: np.array([[-1, 1.0], [-1, -1]])
                                        means distance between robot 0 and 1 should be 1.0
            distance_frame_names: List of frame names (grouped like robot_names) used for distance constraints.
                                  e.g., [["r1_panda_link0", "r2_panda_link0"], "r3_panda_link0"] for base link distances.
                                  If None, uses robot_names from first phase for distance calculation.
            distance_weight: Weight for distance constraint objectives
            collision_weight: Weight for collision avoidance constraint
            pose_rel_constraints: List of directed constraints (frame_i, frame_j, target_pose7).
                                  Position uses target_pose7[:3] via positionRel; rotation uses scalarProductXX to align x-axes.
            pose_rel_weight: Weight for poseRel constraints
        """
        self.config = config
        self.robot_names_phases = robot_names_phases
        self.target_names_phases = target_names_phases
        self.num_phases = len(robot_names_phases)
        self.joint_weight = joint_weight
        self.gripper_weight = gripper_weight
        self.position_rel_z_bounds = position_rel_z_bounds
        self.constraint_eps = constraint_eps
        self.max_attempts = max_attempts
        self.damping = damping
        self.wolfe = wolfe
        self.random_init_mult = random_init_mult
        self.random_init_offset = random_init_offset
        self.freeze_arm_joints = freeze_arm_joints
        self.x_home = x_home
        self.arm_joint_indices = arm_joint_indices
        self.group_distance_constraints = group_distance_constraints
        self.distance_frame_names = distance_frame_names if distance_frame_names is not None else robot_names_phases[0]
        self.distance_weight = distance_weight
        self.collision_weight = collision_weight
        self.pose_rel_constraints = pose_rel_constraints
        self.pose_rel_weight = pose_rel_weight

        # Validate number of phases match
        if len(robot_names_phases) != len(target_names_phases):
            raise ValueError(f"Number of phases in robot_names ({len(robot_names_phases)}) must match target_names ({len(target_names_phases)})")

        # Validate structure matching for each phase
        for phase_idx, (robot_names, target_names) in enumerate(zip(robot_names_phases, target_names_phases)):
            if len(robot_names) != len(target_names):
                raise ValueError(f"Phase {phase_idx+1}: Number of robot groups ({len(robot_names)}) must match number of target groups ({len(target_names)})")

            for i, (robots, targets) in enumerate(zip(robot_names, target_names)):
                robots_is_list = isinstance(robots, list)
                targets_is_list = isinstance(targets, list)
                if robots_is_list != targets_is_list:
                    raise ValueError(f"Phase {phase_idx+1}, Group {i}: robot_names and target_names must have matching structure (both list or both str)")
                if robots_is_list and len(robots) != len(targets):
                    raise ValueError(f"Phase {phase_idx+1}, Group {i}: nested lists must have same length ({len(robots)} vs {len(targets)})")

        # Build flattened robot-target pairs for each phase
        self._robot_target_pairs_by_phase = self._build_robot_target_pairs_by_phase()

        # Build group distance constraint pairs
        self._group_distance_pairs = self._build_group_distance_pairs()

        if freeze_arm_joints:
            if x_home is None:
                raise ValueError("x_home must be provided when freeze_arm_joints=True")
            if arm_joint_indices is None:
                raise ValueError("arm_joint_indices must be provided when freeze_arm_joints=True")
            # Note: arm_joint_indices should match the total number of robots (flattened from first phase)
            total_robots = len(self._robot_target_pairs_by_phase[0])
            if len(arm_joint_indices) != total_robots:
                raise ValueError(f"Number of arm_joint_indices ({len(arm_joint_indices)}) must match total number of robots ({total_robots})")

    def _build_robot_target_pairs_by_phase(self) -> List[List[Tuple[str, str]]]:
        """
        Build a flattened list of (robot_name, target_name) pairs for each phase.

        Returns:
            List of lists, where each inner list contains (robot_name, target_name) tuples for that phase
        """
        pairs_by_phase = []
        for robot_names, target_names in zip(self.robot_names_phases, self.target_names_phases):
            pairs = []
            for robots, targets in zip(robot_names, target_names):
                if isinstance(robots, list):
                    # Nested list case: multiple robots for potentially same/different targets
                    for robot, target in zip(robots, targets):
                        pairs.append((robot, target))
                else:
                    # Simple case: single robot-target pair
                    pairs.append((robots, targets))
            pairs_by_phase.append(pairs)
        return pairs_by_phase

    def _build_group_distance_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Build a list of (frame_i, frame_j, target_distance) tuples from group distance constraints.

        Uses upper triangular matrix format where element [i,j] (i<j) specifies the target
        distance between frame i and frame j within the same group.
        Use -1 to indicate no constraint between two frames.

        Returns:
            List of (frame_i_name, frame_j_name, target_distance) tuples
        """
        pairs = []
        if self.group_distance_constraints is None:
            return pairs

        for group_idx, frames in enumerate(self.distance_frame_names):
            if not isinstance(frames, list):
                # Single frame in group, no distance constraints possible
                continue

            if group_idx >= len(self.group_distance_constraints):
                continue

            distance_matrix = self.group_distance_constraints[group_idx]
            if distance_matrix is None:
                continue

            n_frames = len(frames)
            # Extract upper triangular pairs (i < j)
            for i in range(n_frames):
                for j in range(i + 1, n_frames):
                    if i < distance_matrix.shape[0] and j < distance_matrix.shape[1]:
                        dist = distance_matrix[i, j]
                        # Skip if distance is -1 (no constraint) or negative
                        if dist >= 0:
                            pairs.append((frames[i], frames[j], float(dist)))

        return pairs

    def check_gripper_constraints(self, keyframes: np.ndarray) -> Tuple[bool, List, List]:
        """
        Manually check gripper constraints using eval for all phases.
        Constraints with weight=0 are skipped.

        Args:
            keyframes: Keyframes array with shape (num_phases, num_joints)

        Returns:
            Tuple of (all_satisfied, eq_constraints, ineq_constraints)
        """
        eq_constraints = []
        ineq_constraints = []

        # Check constraints for each phase
        for phase_idx in range(self.num_phases):
            q_state = keyframes[phase_idx]
            self.config.setJointState(q_state)
            self.config.computeCollisions()

            # Check constraints for each robot-target pair in this phase (skip if gripper_weight == 0)
            if self.gripper_weight != 0:
                for robot_name, target_name in self._robot_target_pairs_by_phase[phase_idx]:
                    scalar_product_xy = self.config.eval(ry.FS.scalarProductXY, [target_name, robot_name])
                    scalar_product_yy = self.config.eval(ry.FS.scalarProductYY, [target_name, robot_name])
                    val_xy = float(scalar_product_xy[0][0])
                    val_yy = float(scalar_product_yy[0][0])
                    eq_constraints.append((f"phase{phase_idx+1}_scalarProductXY_{robot_name}_{target_name}", val_xy, 0))
                    eq_constraints.append((f"phase{phase_idx+1}_scalarProductYY_{robot_name}_{target_name}", val_yy, 0))

                    position_rel = self.config.eval(ry.FS.positionRel, [robot_name, target_name])
                    pos_rel = position_rel[0]
                    val_pos_z = float(pos_rel[2])
                    upper, lower = self.position_rel_z_bounds
                    ineq_constraints.append((f"phase{phase_idx+1}_positionRel_{robot_name}_{target_name}_z", val_pos_z, upper, lower))

            # Check group distance constraints (skip if distance_weight == 0)
            # if self.distance_weight != 0:
            #     for robot_i, robot_j, target_dist in self._group_distance_pairs:
            #         dist_val = self.config.eval(ry.FS.distance, [robot_i, robot_j])
            #         actual_dist = float(dist_val[0][0])
            #         eq_constraints.append((f"phase{phase_idx+1}_distance_{robot_i}_{robot_j}", actual_dist, -target_dist))

            accumulated_collisions = self.config.eval(ry.FS.accumulatedCollisions, [])
            val_collisions = float(accumulated_collisions[0][0])
            eq_constraints.append((f"phase{phase_idx+1}_accumulatedCollisions", val_collisions, 0))

        all_eq_satisfied = all(abs(val - target) < self.constraint_eps for _, val, target in eq_constraints)
        all_ineq_satisfied = all(val <= upper and val >= lower for _, val, upper, lower in ineq_constraints)

        return all_eq_satisfied and all_ineq_satisfied, eq_constraints, ineq_constraints

    def solve_komo_problem(self, komo: ry.KOMO, initial_state: Optional[np.ndarray] = None, view: bool = False) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        """
        Solve a KOMO problem with multiple attempts.

        Args:
            komo: The KOMO problem to solve
            initial_state: Initial joint state (optional)
            view: Whether to visualize the solution

        Returns:
            Tuple of (retval_dict, keyframes)
        """
        for num_attempt in range(self.max_attempts):

            def _init_with_path_or_const(state: np.ndarray):
                """Initialize KOMO with a per-phase path if 2D, else constant."""
                state_arr = np.asarray(state)
                if state_arr.ndim == 2:
                    komo.initWithPath(state_arr)
                else:
                    komo.initWithConstant(state_arr)

            if num_attempt == 0 and initial_state is not None:
                _init_with_path_or_const(initial_state)
            else:
                x_init = generate_random_initial_state(self.config)
                # Tile the random state across phases so each phase has a seed
                x_path = np.tile(x_init, (self.num_phases, 1)) if self.num_phases > 1 else x_init
                _init_with_path_or_const(x_path)

            solver = ry.NLP_Solver(komo.nlp(), verbose=0)

            # if self.damping is not None:
            #     solver.setOptions(damping=self.damping)
            # if self.wolfe is not None:
            #     solver.setOptions(wolfe=self.wolfe)

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # This is generated by AI, not sure if it is correct.
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # NLP_Solver.setOptions() parameters:
            #
            # Stopping criteria:
            #   verbose (int, default=1): Verbosity level for solver output (0=silent, higher=more info)
            #   stopTolerance (float, default=0.01): Convergence tolerance on step size Delta
            #       Optimization stops when |x - x'| < stopTolerance
            #   stopFTolerance (float, default=-1): Tolerance on objective function change
            #       If >= 0, stops when |f - f'| < stopFTolerance (-1 means disabled)
            #   stopGTolerance (float, default=-1): Tolerance on gradient norm
            #       If >= 0, stops when ||grad|| < stopGTolerance (-1 means disabled)
            #   stopEvals (int, default=1000): Maximum number of function evaluations allowed
            #   stopInners (int, default=1000): Maximum inner loop iterations (within one penalty level)
            #   stopOuters (int, default=1000): Maximum outer loop iterations (penalty parameter updates)
            #
            # Step size control:
            #   stepMax (float, default=0.2): Maximum step size per iteration
            #       Larger values allow bigger jumps but may cause instability
            #   stepInc (float, default=1.5): Factor to increase step size after successful step
            #   stepDec (float, default=0.5): Factor to decrease step size after failed step
            #
            # Regularization:
            #   damping (float, default=1.0): Damping/regularization parameter (Levenberg-Marquardt style)
            #       Higher = more stable but slower convergence; Lower = more aggressive optimization
            #
            # Line search:
            #   wolfe (float, default=0.01): Wolfe condition parameter for line search
            #       Smaller values make line search conditions more lenient
            #
            # Augmented Lagrangian parameters (for constrained optimization):
            #   muInit (float, default=1.0): Initial penalty coefficient for constraints
            #       Smaller = start with looser constraint enforcement
            #   muInc (float, default=5.0): Factor to increase mu after each outer iteration
            #       Smaller = slower constraint tightening, more stable
            #   muMax (float, default=10000.0): Maximum penalty coefficient
            #   muLBInit (float, default=0.1): Initial penalty for lower bound constraints
            #   muLBDec (float, default=0.2): Factor to decrease muLB
            #   lambdaMax (float, default=-1): Maximum Lagrange multiplier magnitude (-1 = no limit)

            solver.setOptions(
                stopEvals=5000,
                stopTolerance=1e-6,
                stepMax=0.5,
                damping=0.1,
                stepInc=2.0,
                stepDec=0.3,
                wolfe=0.001,
                muInit=0.1,
                muInc=2.0,
                muMax=100000.0,
            )

            retval = solver.solve(verbose=0)
            retval = retval.dict()

            if view:
                print(retval)
                komo.view(True, "IK solution")

            if retval["feasible"]:
                keyframes = komo.getPath()
                if keyframes is not None and len(keyframes) >= self.num_phases:  # TODO: Check if this is correct.
                    is_feasible, eq_vals, ineq_vals = self.check_gripper_constraints(keyframes)
                    if is_feasible:
                        return retval, keyframes

        return retval, None

    def solve(self, initial_state: np.ndarray, view: bool = False) -> Tuple[Any, Optional[ry.KOMO]]:
        """
        Solve the multi-phase optimization problem for a given initial state.

        Args:
            initial_state: Initial joint state
            view: Whether to visualize the solution

        Returns:
            Tuple of (RetWrapper result, komo object)
        """
        # Apply joint freezing if enabled
        if self.freeze_arm_joints:
            init_arr = np.asarray(initial_state).copy()
            if init_arr.ndim == 1:
                for robot_idx, joint_indices in enumerate(self.arm_joint_indices):
                    for joint_idx in joint_indices:
                        if joint_idx < len(self.x_home):
                            init_arr[joint_idx] = self.x_home[joint_idx]
            elif init_arr.ndim == 2:
                for phase_idx in range(init_arr.shape[0]):
                    for robot_idx, joint_indices in enumerate(self.arm_joint_indices):
                        for joint_idx in joint_indices:
                            if joint_idx < len(self.x_home):
                                init_arr[phase_idx, joint_idx] = self.x_home[joint_idx]
            initial_state = init_arr

        # Create multi-phase KOMO: phases=num_phases, slicesPerPhase=1, kOrder=1 to allow cross-phase equality
        komo = ry.KOMO(self.config, phases=self.num_phases, slicesPerPhase=1, kOrder=1, enableCollisions=True)

        # Add joint state regularization objective (applies to all phases)
        # komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [self.joint_weight], initial_state)

        # Add gripper constraints for each phase
        for phase_idx in range(self.num_phases):
            phase_time = [phase_idx + 1]  # KOMO phases are 1-indexed: [1], [2], etc.

            # Add gripper constraints for each robot-target pair in this phase
            for robot_name, target_name in self._robot_target_pairs_by_phase[phase_idx]:
                # Alignment constraints (scalar products)
                komo.addObjective(phase_time, ry.FS.scalarProductXY, [target_name, robot_name], ry.OT.eq, [self.gripper_weight], [0])
                komo.addObjective(phase_time, ry.FS.scalarProductYY, [target_name, robot_name], ry.OT.eq, [self.gripper_weight], [0])

                # Position constraints (inequality)
                upper, lower = self.position_rel_z_bounds
                komo.addObjective(phase_time, ry.FS.positionRel, [robot_name, target_name], ry.OT.ineq, [self.gripper_weight], [0, 0, upper])
                komo.addObjective(phase_time, ry.FS.positionRel, [robot_name, target_name], ry.OT.ineq, [-self.gripper_weight], [0, 0, lower])

        # # Add group distance constraints (applies to all phases)
        # for robot_i, robot_j, target_dist in self._group_distance_pairs:
        #     # komo.addObjective([], ry.FS.distance, [robot_i, robot_j], ry.OT.eq, [self.distance_weight], [-target_dist])
        #     komo.addObjective([], ry.FS.distance, [robot_i, robot_j], ry.OT.sos, [self.distance_weight], [-target_dist])  # TODO: 换成ry.FS.poseRel

        # Directed pose/heading constraints (applies to all phases)
        # Position via positionRel; rotation via scalarProductXX (x-axes alignment)
        if self.pose_rel_constraints:
            for frame_i, frame_j, target_pose in self.pose_rel_constraints:
                pos_target = target_pose[:3]
                pos_weight = [self.pose_rel_weight] * 3
                komo.addObjective([], ry.FS.positionRel, [frame_i, frame_j], ry.OT.sos, pos_weight, pos_target)
                komo.addObjective([], ry.FS.scalarProductXX, [frame_i, frame_j], ry.OT.sos, [self.pose_rel_weight], [1.0])

        # Collision avoidance (applies to all phases)
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [self.collision_weight])

        # Enforce r3 joint states equal between Phase 1 and Phase 2 (soft equality on velocity for r3 joints)
        all_joint_names = self.config.getJointNames()
        r3_joint_indices = [i for i, name in enumerate(all_joint_names) if name.startswith("r3_")]
        if len(r3_joint_indices) > 0:
            weight = np.zeros(len(all_joint_names))
            weight[r3_joint_indices] = self.joint_weight * 10  # stronger weight for equality
            # order=1 enforces zero velocity at the phase transition (q2 - q1 = 0) for selected joints
            komo.addObjective([2], ry.FS.qItself, [], ry.OT.eq, weight.tolist(), order=1)

        ret_dict, keyframes = self.solve_komo_problem(komo, initial_state=initial_state, view=view)

        class RetWrapper:
            def __init__(self, ret_dict, keyframes, num_phases):
                self.feasible = ret_dict.get("feasible", False) and keyframes is not None
                self.eq = ret_dict.get("eq", float("inf"))
                self.ineq = ret_dict.get("ineq", float("inf"))
                self.sos = ret_dict.get("sos", float("inf"))
                self.keyframes = keyframes
                self.num_phases = num_phases

        ret = RetWrapper(ret_dict, keyframes, self.num_phases)

        return ret, komo


print("The path where model files are pre-installed:\n", ry.raiPath(""))

C = ry.Config()

# Element configuration constants
CYLINDER_LENGTH = 1.0
CYLINDER_RADIUS = 0.01
PROTRUSION_OFFSET = 0.15
VERTICAL_DISTANCE = 0.04
VERTICAL_Z = 0.5
HORIZONTAL_Z = [0.75, 0.77, 0.79]
ROBOT_DISTANCE = -1

# Vertex positions for the triangular structure
v1_pos = np.array([0.5, 0.0, VERTICAL_Z])
v2_pos = np.array([-0.25, 0.433, VERTICAL_Z])
v3_pos = np.array([-0.25, -0.433, VERTICAL_Z])


if __name__ == "__main__":
    # Create horizontal elements (beams) using the new class-based approach
    # element_6 has contact=True so robots avoid collision in both phases
    element_4 = create_horizontal_element(C, "element_4", v2_pos, v3_pos, v1_pos, HORIZONTAL_Z[0], [1, 1, 0], length=CYLINDER_LENGTH, radius=CYLINDER_RADIUS, protrusion_offset=PROTRUSION_OFFSET, contact=True)
    element_5 = create_horizontal_element(C, "element_5", v3_pos, v1_pos, v2_pos, HORIZONTAL_Z[1], [1, 0, 1], length=CYLINDER_LENGTH, radius=CYLINDER_RADIUS, protrusion_offset=PROTRUSION_OFFSET, contact=True)
    element_6 = create_horizontal_element(C, "element_6", v1_pos, v2_pos, v3_pos, HORIZONTAL_Z[2], [0, 1, 1], length=CYLINDER_LENGTH, radius=CYLINDER_RADIUS, protrusion_offset=PROTRUSION_OFFSET, contact=True)

    # Calculate element endpoints for vertical element positioning
    element_4_end = element_4.get_end_position(v1_pos)
    element_5_end = element_5.get_end_position(v2_pos)
    element_6_other_end = element_6.get_other_end_position(v3_pos)

    # Calculate vertical element positions
    v1_final = GeometryCalculator.calculate_vertical_element_position(element_4_end, v1_pos, VERTICAL_DISTANCE, VERTICAL_Z)
    v2_final = GeometryCalculator.calculate_vertical_element_position(element_5_end, v2_pos, VERTICAL_DISTANCE, VERTICAL_Z)
    v3_final = GeometryCalculator.calculate_vertical_element_position(element_6_other_end, v3_pos, VERTICAL_DISTANCE, VERTICAL_Z)

    # Create vertical elements (columns)
    element_1 = create_vertical_element(C, "element_1", v1_final, [1, 0, 0], length=CYLINDER_LENGTH, radius=CYLINDER_RADIUS, contact=True)
    element_2 = create_vertical_element(C, "element_2", v2_final, [0, 1, 0], length=CYLINDER_LENGTH, radius=CYLINDER_RADIUS, contact=True)
    element_3 = create_vertical_element(C, "element_3", v3_final, [0, 0, 1], length=CYLINDER_LENGTH, radius=CYLINDER_RADIUS, contact=True)

    # Calculate robot base poses (position + yaw quaternion) for each phase
    # Phase 1: r1+r2 face element_4, r3 faces element_5
    robot_base_poses_phase1 = [
        RobotPositionCalculator.calculate_pose_toward_target(element_4.position, element_4.direction, ROBOT_DISTANCE, element_4.position),
        RobotPositionCalculator.calculate_pose_toward_target(element_4.position, element_4.direction, ROBOT_DISTANCE + 0.5, element_4.position),
        RobotPositionCalculator.calculate_pose_toward_target(element_5.position, element_5.direction, ROBOT_DISTANCE, element_5.position),
    ]

    # Phase 2: r1+r2 face element_6, r3 continues to face element_5
    robot_base_poses_phase2 = [
        RobotPositionCalculator.calculate_pose_toward_target(element_6.position, element_6.direction, ROBOT_DISTANCE, element_6.position),
        RobotPositionCalculator.calculate_pose_toward_target(element_6.position, element_6.direction, ROBOT_DISTANCE + 0.5, element_6.position),
        RobotPositionCalculator.calculate_pose_toward_target(element_5.position, element_5.direction, ROBOT_DISTANCE, element_5.position),
    ]

    robot_base_poses_phases = [robot_base_poses_phase1, robot_base_poses_phase2]

    # Add robots to configuration (using Phase 1 poses initially)
    # r1_base_frame = RobotPositionCalculator.add_robot_to_config(
    #     C, ry.raiPath("panda/panda.g"), "r1_", robot_base_poses_phase1[0][0], robot_base_poses_phase1[0][1]
    # )
    # r2_base_frame = RobotPositionCalculator.add_robot_to_config(
    #     C, ry.raiPath("panda/panda.g"), "r2_", robot_base_poses_phase1[1][0], robot_base_poses_phase1[1][1]
    # )
    # r3_base_frame = RobotPositionCalculator.add_robot_to_config(
    #     C, ry.raiPath("panda/panda.g"), "r3_", robot_base_poses_phase1[2][0], robot_base_poses_phase1[2][1]
    # )
    r1_base_frame = RobotPositionCalculator.add_robot_to_config(C, ry.raiPath("panda/panda.g"), "r1_", [0, 0, 0], [1, 0, 0, 0])
    r2_base_frame = RobotPositionCalculator.add_robot_to_config(C, ry.raiPath("panda/panda.g"), "r2_", [0, 0, 0], [1, 0, 0, 0])
    r3_base_frame = RobotPositionCalculator.add_robot_to_config(C, ry.raiPath("panda/panda.g"), "r3_", [0, 0, 0], [1, 0, 0, 0])

    base1 = C.getFrame("r1_panda_link0")
    base2 = C.getFrame("r2_panda_link0")
    base3 = C.getFrame("r3_panda_link0")
    # base1.setJoint(ry.JT.transXY, [-0.25, -0.25, 0.25, 0.25])
    # base2.setJoint(ry.JT.transXY, [-0.25, -0.25, 0.25, 0.25])
    # base3.setJoint(ry.JT.transXY, [-0.25, -0.25, 0.25, 0.25])
    base1.setJoint(ry.JT.transXYPhi, [-5, -5, -np.pi, 5, 5, np.pi])
    base2.setJoint(ry.JT.transXYPhi, [-5, -5, -np.pi, 5, 5, np.pi])
    base3.setJoint(ry.JT.transXYPhi, [-5, -5, -np.pi, 5, 5, np.pi])
    base_frame_names = ["r1_panda_link0", "r2_panda_link0", "r3_panda_link0"]
    initial_base_positions = [base1.getPosition(), base2.getPosition(), base3.getPosition()]
    initial_base_quaternions = [base1.getQuaternion(), base2.getQuaternion(), base3.getQuaternion()]

    all_joint_names = C.getJointNames()
    base_joint_indices = []
    for i, name in enumerate(all_joint_names):
        if any(base_name in name for base_name in base_frame_names):
            base_joint_indices.append(i)

    C.view()

    # pp.wait_for_user()

    def draw_pose(config, frame_name, pose_name_prefix, length=0.1):
        """Draw pose using marker frames"""
        frame = config.getFrame(frame_name)
        pos = frame.getPosition()
        quat = frame.getQuaternion()

        config.addFrame(f"{pose_name_prefix}_marker").setShape(ry.ST.marker, [length]).setPosition(pos).setQuaternion(quat).setColor([1, 1, 0])

    # Ensure the config reflects the Phase 1 base poses before capturing home
    RobotPositionCalculator.apply_base_poses(C, base_frame_names, robot_base_poses_phase1)

    x_home = C.getJointState()

    def build_base_pose_path(poses_by_phase: List[List[Tuple[np.ndarray, List[float]]]], template_state: np.ndarray) -> np.ndarray:
        """Build a per-phase joint state path that encodes phase-specific base poses."""
        path = np.tile(template_state, (len(poses_by_phase), 1))
        for phase_idx, phase_poses in enumerate(poses_by_phase):
            for robot_idx, (pos, quat) in enumerate(phase_poses):
                yaw = 2 * np.arctan2(quat[3], quat[0])  # Recover yaw from [w, x, y, z]
                base_offset = robot_idx * 10  # 3 base DOF followed by 7 arm DOF
                path[phase_idx, base_offset : base_offset + 3] = [pos[0], pos[1], yaw]
        return path

    # Solver configuration parameters (shared)
    joint_weight = 0.1 # bak
    # joint_weight = 0
    gripper_weight = 5.11 # bak
    # gripper_weight = 1
    position_rel_z_bounds = (0.45, -0.45)
    constraint_eps = 1e-3
    max_attempts = 1
    freeze_arm_joints = True
    distance_weight = 0
    collision_weight = 10
    pose_rel_weight = 10

    # Arm joint indices for freezing (assuming 10 DOF per robot: 3 base + 7 arm)
    arm_joint_indices = [
        list(range(0 * 10 + 3, 0 * 10 + 3 + 7)),  # Robot 1 arm joints
        list(range(1 * 10 + 3, 1 * 10 + 3 + 7)),  # Robot 2 arm joints
        list(range(2 * 10 + 3, 2 * 10 + 3 + 7)),  # Robot 3 arm joints
    ]

    # Joint indices for each robot (base + arm = 10 DOF each)
    robot_1_joint_indices = list(range(0 * 10, 0 * 10 + 10))
    robot_2_joint_indices = list(range(1 * 10, 1 * 10 + 10))
    robot_3_joint_indices = list(range(2 * 10, 2 * 10 + 10))
    robot_group_joint_indices = robot_1_joint_indices + robot_2_joint_indices  # r1 + r2

    # Distance frame names (grouped like robot_names, using base link frames)
    distance_frame_names = [["r1_panda_link0", "r2_panda_link0"], "r3_panda_link0"]

    # Group distance constraints (upper triangular matrix, -1 means no constraint)
    group_distance_constraints = [
        np.array([[-1, 0.5], [-1, -1]]),  # r1-r2 base distance = 0.3 (kept for compatibility, weight=0)
        None,  # Group 1 (r3): single robot, no distance constraint
    ]

    # Directed poseRel constraints (frame_i in frame_j coordinates)
    pose_rel_constraints = [
        ("r2_panda_link0", "r1_panda_link0", [0.0, 0.5, 0.0, 1, 0, 0, 0]),
    ]

    # =========================================================================
    # Multi-Phase Setup: Two phases with different r3 targets
    # Phase 1: r1+r2 -> element_4, r3 -> element_5
    # Phase 2: r1+r2 -> element_6, r3 -> element_5
    # =========================================================================
    robot_names_phase1 = [["r1_gripper", "r2_gripper"], "r3_gripper"]
    target_names_phase1 = [["element_4", "element_4"], "element_5"]

    robot_names_phase2 = [["r1_gripper", "r2_gripper"], "r3_gripper"]
    target_names_phase2 = [["element_6", "element_6"], "element_5"]

    # Combine into multi-phase format
    robot_names_phases = [robot_names_phase1, robot_names_phase2]
    target_names_phases = [target_names_phase1, target_names_phase2]

    solver = SingleKeyFrameSolver(
        config=C,
        robot_names_phases=robot_names_phases,
        target_names_phases=target_names_phases,
        joint_weight=joint_weight,
        gripper_weight=gripper_weight,
        position_rel_z_bounds=position_rel_z_bounds,
        constraint_eps=constraint_eps,
        max_attempts=max_attempts,
        freeze_arm_joints=freeze_arm_joints,
        x_home=x_home,
        arm_joint_indices=arm_joint_indices,
        group_distance_constraints=group_distance_constraints,
        distance_frame_names=distance_frame_names,
        distance_weight=distance_weight,
        collision_weight=collision_weight,
        pose_rel_constraints=pose_rel_constraints,
        pose_rel_weight=pose_rel_weight,
    )

    num_initial_states = 100
    print(f"Generating {num_initial_states} random initial states")
    print(f"Joint weight: {joint_weight}")
    print(f"Gripper weight: {gripper_weight}")
    print(f"Freeze arm joints: {freeze_arm_joints}")
    print(f"\nMulti-phase KOMO optimization (single solve, no alternating):")
    print(f"  Phase 1: Robot group -> element_4, Single robot -> element_5")
    print(f"  Phase 2: Robot group -> element_6, Single robot -> element_5")

    np.random.seed(42)

    # Generate initial states
    # Precompute a per-phase base-pose path so each phase starts near its target
    base_pose_path_template = build_base_pose_path(robot_base_poses_phases, x_home)

    initial_states = []
    for i in range(num_initial_states):
        q_path = base_pose_path_template.copy()

        # Add small noise on base joints to diversify seeds while keeping phase-specific targets
        if len(base_joint_indices) > 0:
            noise = np.random.uniform(-0.5, 0.5, size=(q_path.shape[0], len(base_joint_indices)))
            for phase_idx in range(q_path.shape[0]):
                q_path[phase_idx, base_joint_indices] += noise[phase_idx]

        initial_states.append(q_path)
        print(f"Initial state path {i+1}: generated with phase-specific bases")

    pp.wait_for_user()

    all_results = []

    for state_idx, initial_state in enumerate(initial_states):
        print(f"\n{'='*60}")
        print(f"Initial State {state_idx + 1}/{num_initial_states}")
        print(f"{'='*60}")

        # Solve multi-phase optimization in a single call
        ret, komo = solver.solve(initial_state, view=False)

        if ret.feasible:
            # Extract keyframes for each phase
            q_phase1 = ret.keyframes[0]
            q_phase2 = ret.keyframes[1]

            print(f"  ✓ Multi-phase optimization succeeded (eq={ret.eq:.3e})")

            # Record results
            result = {
                "state_idx": state_idx,
                "initial_state": initial_state.tolist(),
                "feasible": True,
                "eq": ret.eq,
                "ineq": ret.ineq,
                "sos": ret.sos,
                "q_phase1": q_phase1.tolist(),
                "q_phase2": q_phase2.tolist(),
            }
        else:
            print(f"  ✗ Multi-phase optimization failed")

            # Record results
            result = {
                "state_idx": state_idx,
                "initial_state": initial_state.tolist(),
                "feasible": False,
                "eq": ret.eq,
                "ineq": ret.ineq,
                "sos": ret.sos,
                "q_phase1": None,
                "q_phase2": None,
            }

        all_results.append(result)

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    feasible_count = sum(1 for r in all_results if r["feasible"])
    print(f"Feasible: {feasible_count}/{num_initial_states}")

    # Collect all feasible configurations
    all_feasible = [r for r in all_results if r["feasible"]]

    output_dir = "komo_results"
    os.makedirs(output_dir, exist_ok=True)

    if len(all_feasible) > 0:
        print(f"\nSaving {len(all_feasible)} feasible configurations...")

        results_file = os.path.join(output_dir, "multi_phase_configs.json")
        with open(results_file, "w") as f:
            json.dump(all_feasible, f, indent=2)
        print(f"Saved to {results_file}")

    if len(all_feasible) > 0:
        print("\nFeasible configurations summary:")
        for r in all_feasible:
            print(f"  State {r['state_idx']+1}: eq={r['eq']:.3e}, sos={r['sos']:.3e}")

        print(f"\n{'='*60}")
        print(f"Viewing all {len(all_feasible)} feasible configurations")
        print(f"{'='*60}")

        for idx, result in enumerate(all_feasible):
            print(f"\n{'='*60}")
            print(f"Configuration {idx + 1}/{len(all_feasible)} (State {result['state_idx'] + 1})")
            print(f"eq={result['eq']:.3e}, sos={result['sos']:.3e}")
            print(f"{'='*60}")

            # Show Phase 1 result
            print(f"\n--- Phase 1 Result (r3 -> element_5) ---")
            q_phase1 = np.array(result["q_phase1"])
            C.setJointState(q_phase1)
            C.view()

            # print(f"  Distance constraints:")
            # for robot_i, robot_j, target_dist in solver._group_distance_pairs:
            #     actual_dist = C.eval(ry.FS.distance, [robot_i, robot_j])[0][0]
            #     print(f"    {robot_i} - {robot_j}: target={target_dist:.3f}, actual={actual_dist:.3f}")

            print(f"  PoseRel constraints (positionRel + scalarProductXX):")
            for frame_i, frame_j, target_pose in solver.pose_rel_constraints:
                actual_pos = C.eval(ry.FS.positionRel, [frame_i, frame_j])[0]
                actual_align = C.eval(ry.FS.scalarProductXX, [frame_i, frame_j])[0][0]
                print(f"    {frame_i} - {frame_j}: target_pos={target_pose[:3]}, actual_pos={actual_pos.tolist()}, alignXX={actual_align:.3f}")

            pp.wait_for_user()

            # Show Phase 2 result
            print(f"\n--- Phase 2 Result (r3 -> element_6) ---")
            q_phase2 = np.array(result["q_phase2"])
            C.setJointState(q_phase2)
            C.view()

            # print(f"  Distance constraints:")
            # for robot_i, robot_j, target_dist in solver._group_distance_pairs:
            #     actual_dist = C.eval(ry.FS.distance, [robot_i, robot_j])[0][0]
            #     print(f"    {robot_i} - {robot_j}: target={target_dist:.3f}, actual={actual_dist:.3f}")

            print(f"  PoseRel constraints (positionRel + scalarProductXX):")
            for frame_i, frame_j, target_pose in solver.pose_rel_constraints:
                actual_pos = C.eval(ry.FS.positionRel, [frame_i, frame_j])[0]
                actual_align = C.eval(ry.FS.scalarProductXX, [frame_i, frame_j])[0][0]
                print(f"    {frame_i} - {frame_j}: target_pos={target_pose[:3]}, actual_pos={actual_pos.tolist()}, alignXX={actual_align:.3f}")

            pp.wait_for_user()
    else:
        print("\nNo feasible configurations found!")
        C.view()
        pp.wait_for_user()
