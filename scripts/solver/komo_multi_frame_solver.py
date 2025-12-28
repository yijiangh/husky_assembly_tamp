"""
Multi-phase KOMO solver for multi-robot grasp tasks.

This module provides classes and utilities for solving inverse kinematics problems
with gripper constraints for multiple robots across multiple phases.
"""

import numpy as np
import robotic as ry
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field


# ============================================================================
# Utility Functions
# ============================================================================


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


# ============================================================================
# Geometry and Element Classes
# ============================================================================


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


# ============================================================================
# Constraint Management
# ============================================================================


@dataclass
class Constraint:
    """
    Represents a constraint to be checked after optimization and automatically added to KOMO.

    Attributes:
        name: Unique identifier for the constraint
        constraint_type: Type of constraint ('eq' for equality, 'ineq' for inequality, 'sos' for soft objective)
        feature_type: The ry.FS feature type (e.g., ry.FS.scalarProductXY)
        frames: List of frame names for the feature evaluation
        phase_idx: Phase index (None if applies to all phases). KOMO phases are 1-indexed, so phase_idx 0 -> [1], 1 -> [2], etc.
        objective_type: The ry.OT objective type (ry.OT.eq, ry.OT.ineq, ry.OT.sos)
        weight: Weight of the constraint (constraints with weight=0 are skipped)
        target: Target value(s) for the constraint. For eq: scalar or list, for ineq: list with bounds, for sos: target value(s)
        order: Order for velocity constraints (default 0)
    """

    name: str
    constraint_type: str  # 'eq' or 'ineq' or 'sos'
    feature_type: Any  # ry.FS type
    frames: List[str]
    objective_type: Any  # ry.OT type
    weight: float = 1.0
    phase_idx: Optional[int] = None  # None means applies to all phases
    target: Optional[Union[float, List[float]]] = None
    order: int = 0

    def should_check(self) -> bool:
        """Return True if this constraint should be checked (weight != 0)."""
        if isinstance(self.weight, list):
            # For list weights, check if any element is non-zero
            return any(abs(w) > 1e-10 for w in self.weight)
        else:
            return abs(self.weight) > 1e-10

    def get_phase_time(self) -> List[int]:
        """Get phase time list for KOMO (1-indexed). Empty list means all phases."""
        if self.phase_idx is None:
            return []
        return [self.phase_idx + 1]  # KOMO phases are 1-indexed

    def get_target_value(self) -> float:
        """Get target value for equality constraint checking."""
        if isinstance(self.target, list):
            return self.target[0] if len(self.target) > 0 else 0.0
        return self.target if self.target is not None else 0.0

    # Additional attribute for constraint checking (not used in KOMO addition)
    bounds: Optional[Tuple[float, float]] = None  # (upper, lower) for inequality constraints

    def get_bounds(self) -> Optional[Tuple[float, float]]:
        """Get bounds for inequality constraint checking."""
        return self.bounds


class ConstraintManager:
    """Manages constraints registered during KOMO problem setup."""

    def __init__(self, config: ry.Config, constraint_eps: float = 1e-3):
        """
        Initialize the constraint manager.

        Args:
            config: The ry.Config object for constraint evaluation
            constraint_eps: Tolerance for constraint satisfaction checking
        """
        self.config = config
        self.constraint_eps = constraint_eps
        self.constraints: List[Constraint] = []

    def register(self, constraint: Constraint, komo: Optional[ry.KOMO] = None) -> None:
        """
        Register a constraint and optionally add it to KOMO immediately.

        Args:
            constraint: The constraint to register
            komo: Optional KOMO object. If provided, the constraint will be automatically added.
        """
        self.constraints.append(constraint)
        if komo is not None:
            self._add_constraint_to_komo(constraint, komo)

    def add_all_to_komo(self, komo: ry.KOMO) -> None:
        """
        Add all registered constraints to the KOMO problem.

        Args:
            komo: The KOMO object to add constraints to
        """
        for constraint in self.constraints:
            if constraint.should_check():
                self._add_constraint_to_komo(constraint, komo)

    def _add_constraint_to_komo(self, constraint: Constraint, komo: ry.KOMO) -> None:
        """
        Add a single constraint to KOMO.

        Args:
            constraint: The constraint to add
            komo: The KOMO object
        """
        phase_time = constraint.get_phase_time()
        weight_list = [constraint.weight] if not isinstance(constraint.weight, list) else constraint.weight

        # Prepare target value
        if constraint.target is None:
            target = [0]
        elif isinstance(constraint.target, list):
            target = constraint.target
        else:
            target = [constraint.target]

        # Add objective to KOMO
        if constraint.order > 0:
            komo.addObjective(phase_time, constraint.feature_type, constraint.frames, constraint.objective_type, weight_list, target, order=constraint.order)
        else:
            komo.addObjective(phase_time, constraint.feature_type, constraint.frames, constraint.objective_type, weight_list, target)

    def check_all(self, keyframes: np.ndarray) -> Tuple[bool, List, List]:
        """
        Check all registered constraints using eval for all phases.

        Args:
            keyframes: Keyframes array with shape (num_phases, num_joints)

        Returns:
            Tuple of (all_satisfied, eq_constraints, ineq_constraints)
            where eq_constraints is list of (name, actual_value, target_value)
            and ineq_constraints is list of (name, actual_value, upper_bound, lower_bound)
        """
        eq_constraints = []
        ineq_constraints = []

        num_phases = keyframes.shape[0] if keyframes.ndim == 2 else 1

        # Check constraints for each phase
        for phase_idx in range(num_phases):
            q_state = keyframes[phase_idx] if keyframes.ndim == 2 else keyframes
            self.config.setJointState(q_state)
            self.config.computeCollisions()

            # Check constraints that apply to this phase
            for constraint in self.constraints:
                if not constraint.should_check():
                    continue

                # Check if constraint applies to this phase
                if constraint.phase_idx is not None and constraint.phase_idx != phase_idx:
                    continue

                # Skip checking constraints that cannot be evaluated (e.g., qItself with base frames)
                # These are velocity/equality constraints that are better validated through optimization results
                if constraint.feature_type == ry.FS.qItself and constraint.order > 0:
                    continue  # Skip velocity constraints as they're validated by KOMO optimization

                # Evaluate the feature
                try:
                    result = self.config.eval(constraint.feature_type, constraint.frames)

                    # Extract feature value from tuple (phi, J)
                    phi = result[0] if isinstance(result, tuple) else result

                    # Handle different result shapes
                    if constraint.feature_type == ry.FS.positionRel:
                        actual_value = float(phi[2])  # z-component for position constraints
                    else:
                        actual_value = float(phi[0] if hasattr(phi, "__getitem__") else phi)
                except Exception as e:
                    # If evaluation fails, skip this constraint
                    # Some constraints (like qItself with base frames) cannot be evaluated directly
                    if "is not a joint or pathDof" in str(e) or "dim_phi" in str(e):
                        continue  # Silently skip constraints that cannot be evaluated
                    print(f"Warning: Failed to evaluate constraint {constraint.name}: {e}")
                    continue

                if constraint.constraint_type == "eq":
                    target = constraint.get_target_value()
                    eq_constraints.append((constraint.name, actual_value, target))
                elif constraint.constraint_type == "ineq":
                    bounds = constraint.get_bounds()
                    if bounds is not None:
                        upper, lower = bounds
                        ineq_constraints.append((constraint.name, actual_value, upper, lower))
                # Note: 'sos' constraints are soft objectives and are typically checked less strictly

        all_eq_satisfied = all(abs(val - target) < self.constraint_eps for _, val, target in eq_constraints)
        all_ineq_satisfied = all(val <= upper and val >= lower for _, val, upper, lower in ineq_constraints)

        return all_eq_satisfied and all_ineq_satisfied, eq_constraints, ineq_constraints
    
    def check_collisions(self, keyframes: np.ndarray) -> List[List[str]]:
        """
        Find all collision pairs in each phase and return as a list of [frame1, frame2] pairs.
        
        Args:
            keyframes: Keyframes array with shape (num_phases, num_joints) or (num_joints,)
        
        Returns:
            List of collision pairs, where each pair is [frame1_name, frame2_name]
        """
        collision_pairs = []
        num_phases = keyframes.shape[0] if keyframes.ndim == 2 else 1
        
        # Check collisions for each phase
        for phase_idx in range(num_phases):
            collisions_this_phase = []
            q_state = keyframes[phase_idx] if keyframes.ndim == 2 else keyframes
            self.config.setJointState(q_state)
            self.config.computeCollisions()
            
            # Get collision pairs (returns list of (frame1_name, frame2_name, distance) tuples)
            collisions = self.config.getCollisions(belowMargin=0.0)
            
            # Extract frame names from collision tuples (frame1 and frame2 are already strings)
            for frame1_name, frame2_name, _ in collisions:
                collisions_this_phase.append([frame1_name, frame2_name])
                
            collision_pairs.append(collisions_this_phase)
        
        return collision_pairs
    
    @staticmethod
    def check_collisions_static(C: ry.Config, keyframes: np.ndarray) -> List[List[str]]:
        """
        Find all collision pairs in each phase and return as a list of [frame1, frame2] pairs.
        
        Args:
            keyframes: Keyframes array with shape (num_phases, num_joints) or (num_joints,)
        
        Returns:
            List of collision pairs, where each pair is [frame1_name, frame2_name]
        """
        collision_pairs = []
        num_phases = keyframes.shape[0] if keyframes.ndim == 2 else 1
        
        # Check collisions for each phase
        for phase_idx in range(num_phases):
            collisions_this_phase = []
            q_state = keyframes[phase_idx] if keyframes.ndim == 2 else keyframes
            C.setJointState(q_state)
            C.computeCollisions()
            
            # Get collision pairs (returns list of (frame1_name, frame2_name, distance) tuples)
            collisions = C.getCollisions(belowMargin=0.0)
            
            # Extract frame names from collision tuples (frame1 and frame2 are already strings)
            for frame1_name, frame2_name, _ in collisions:
                collisions_this_phase.append([frame1_name, frame2_name])
                
            collision_pairs.append(collisions_this_phase)
        
        return collision_pairs


# ============================================================================
# Main Solver Class
# ============================================================================


class MultiPhaseKomoSolver:
    """
    Multi-phase KOMO solver for multi-robot grasp tasks (single keyframe per phase).

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
        damping: Optional[float] = None,
        wolfe: Optional[float] = None,
        freeze_arm_joints: bool = False,
        x_home: Optional[np.ndarray] = None,
        arm_joint_indices: Optional[List[List[int]]] = None,
        collision_weight: float = 1.0,
        pose_rel_constraints: Optional[List[Tuple[str, str, List[float]]]] = None,
        pose_rel_weight: float = 1.0,
        enable_constraint_verification: bool = True,
        baselink_names_phases: Optional[List[List[Union[str, List[str]]]]] = None,
        baselink_distance_weight: float = 1.0,
        baselink_distance_target: float = 10.0,
        phase_switch_robots: List[str] = [],
        phase_switch_weight: float = 1.0,
    ):
        """
        Initialize the solver.

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
            damping: Damping parameter for NLP solver (optional)
            wolfe: Wolfe parameter for NLP solver (optional)
            freeze_arm_joints: Whether to freeze arm joint angles to home position
            x_home: Home joint state for freezing arm joints (required if freeze_arm_joints=True)
            arm_joint_indices: List of lists, each containing joint indices for each robot's arm (required if freeze_arm_joints=True)
            collision_weight: Weight for collision avoidance constraint
            pose_rel_constraints: List of directed constraints (frame_i, frame_j, target_pose7).
                                  Position uses target_pose7[:3] via positionRel; rotation uses scalarProductXX to align x-axes.
            pose_rel_weight: Weight for poseRel constraints
            enable_constraint_verification: Whether to perform secondary constraint checking after optimization
            baselink_names_phases: Optional list of baselink frame names per phase.
                                  Structure must match robot_names_phases.
                                  e.g., [[["r1_base_footprint", "r1_base_footprint"], "r2_base_footprint"], [...]]
                                  If None, baselink distance constraints are skipped.
            baselink_distance_weight: Weight for soft constraint maximizing distance between baselink and element
            baselink_distance_target: Target distance value for baselink-element distance constraint (large value to maximize distance)
        """
        self.config = config
        self.robot_names_phases = robot_names_phases
        self.target_names_phases = target_names_phases
        self.num_phases = len(robot_names_phases)
        self.joint_weight = joint_weight
        self.gripper_weight = gripper_weight
        self.position_rel_z_bounds = position_rel_z_bounds
        self.constraint_eps = constraint_eps
        self.damping = damping
        self.wolfe = wolfe
        self.freeze_arm_joints = freeze_arm_joints
        self.x_home = x_home
        self.arm_joint_indices = arm_joint_indices
        self.collision_weight = collision_weight
        self.pose_rel_constraints = pose_rel_constraints
        self.pose_rel_weight = pose_rel_weight
        self.enable_constraint_verification = enable_constraint_verification
        self.baselink_names_phases = baselink_names_phases
        self.baselink_distance_weight = baselink_distance_weight
        self.baselink_distance_target = baselink_distance_target
        self.phase_switch_robots = phase_switch_robots
        self.phase_switch_weight = phase_switch_weight

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

        # Build flattened baselink-target pairs for each phase (if baselink names provided)
        if baselink_names_phases is not None:
            # Validate baselink_names_phases structure matches robot_names_phases
            if len(baselink_names_phases) != len(robot_names_phases):
                raise ValueError(f"Number of phases in baselink_names ({len(baselink_names_phases)}) must match robot_names ({len(robot_names_phases)})")
            
            for phase_idx, (robot_names, baselink_names) in enumerate(zip(robot_names_phases, baselink_names_phases)):
                if len(baselink_names) != len(robot_names):
                    raise ValueError(f"Phase {phase_idx+1}: Number of baselink groups ({len(baselink_names)}) must match number of robot groups ({len(robot_names)})")
                
                for i, (robots, baselinks) in enumerate(zip(robot_names, baselink_names)):
                    robots_is_list = isinstance(robots, list)
                    baselinks_is_list = isinstance(baselinks, list)
                    if robots_is_list != baselinks_is_list:
                        raise ValueError(f"Phase {phase_idx+1}, Group {i}: robot_names and baselink_names must have matching structure (both list or both str)")
                    if robots_is_list and len(robots) != len(baselinks):
                        raise ValueError(f"Phase {phase_idx+1}, Group {i}: nested lists must have same length ({len(robots)} vs {len(baselinks)})")
            
            self._baselink_target_pairs_by_phase = self._build_baselink_target_pairs_by_phase()
        else:
            self._baselink_target_pairs_by_phase = None

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

    def _build_baselink_target_pairs_by_phase(self) -> List[List[Tuple[str, str]]]:
        """
        Build a flattened list of (baselink_name, target_name) pairs for each phase.
        Duplicate pairs within each phase are removed.
        
        Returns:
            List of lists, where each inner list contains unique (baselink_name, target_name) tuples for that phase
        """
        pairs_by_phase = []
        for baselink_names, target_names in zip(self.baselink_names_phases, self.target_names_phases):
            pairs = []
            seen_pairs = set()  # Track seen pairs to avoid duplicates
            for baselinks, targets in zip(baselink_names, target_names):
                if isinstance(baselinks, list):
                    # Nested list case: multiple baselinks for potentially same/different targets
                    for baselink, target in zip(baselinks, targets):
                        pair = (baselink, target)
                        if pair not in seen_pairs:
                            pairs.append(pair)
                            seen_pairs.add(pair)
                else:
                    # Simple case: single baselink-target pair
                    pair = (baselinks, targets)
                    if pair not in seen_pairs:
                        pairs.append(pair)
                        seen_pairs.add(pair)
            pairs_by_phase.append(pairs)
        return pairs_by_phase

    def check_constraints(self, constraint_manager: ConstraintManager, keyframes: np.ndarray) -> Tuple[bool, List, List]:
        """
        Check all constraints using the constraint manager.

        Args:
            constraint_manager: The ConstraintManager containing all registered constraints
            keyframes: Keyframes array with shape (num_phases, num_joints)

        Returns:
            Tuple of (all_satisfied, eq_constraints, ineq_constraints)
        """
        # collisions = constraint_manager.check_collisions(keyframes)
        # print(f"Collisions: {collisions}")
        return constraint_manager.check_all(keyframes)

    def solve_komo_problem(self, komo: ry.KOMO, constraint_manager: Optional[ConstraintManager] = None, initial_state: Optional[np.ndarray] = None, view: bool = False) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        """
        Solve a KOMO problem (single attempt).

        Args:
            komo: The KOMO problem to solve
            constraint_manager: Optional ConstraintManager for constraint verification
            initial_state: Initial joint state (optional)
            view: Whether to visualize the solution

        Returns:
            Tuple of (retval_dict, keyframes)
        """

        def _init_with_path_or_const(state: np.ndarray):
            """Initialize KOMO with a per-phase path if 2D, else constant."""
            state_arr = np.asarray(state)
            if state_arr.ndim == 2:
                komo.initWithPath(state_arr)
            else:
                komo.initWithConstant(state_arr)

        if initial_state is not None:
            _init_with_path_or_const(initial_state)
        else:
            x_init = generate_random_initial_state(self.config)
            x_path = np.tile(x_init, (self.num_phases, 1)) if self.num_phases > 1 else x_init
            _init_with_path_or_const(x_path)

        solver = ry.NLP_Solver(komo.nlp(), verbose=0)

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
            
        is_feasible_tmp, eq_vals, ineq_vals = self.check_constraints(constraint_manager, komo.getPath())

        if retval["feasible"]:
            keyframes = komo.getPath()
            if keyframes is not None and len(keyframes) >= self.num_phases:
                # Only check constraints if verification is enabled and constraint_manager is provided
                if self.enable_constraint_verification and constraint_manager is not None:
                    is_feasible, eq_vals, ineq_vals = self.check_constraints(constraint_manager, keyframes)
                    if is_feasible:
                        return retval, keyframes
                else:
                    # If verification is disabled, accept the solution directly
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

        # Create constraint manager for constraint verification
        constraint_manager = ConstraintManager(self.config, self.constraint_eps)

        # Register and add gripper constraints for each phase
        for phase_idx in range(self.num_phases):
            # Add gripper constraints for each robot-target pair in this phase
            for robot_name, target_name in self._robot_target_pairs_by_phase[phase_idx]:
                # Alignment constraints (scalar products) - equality
                constraint_manager.register(
                    Constraint(
                        name=f"phase{phase_idx+1}_scalarProductXY_{robot_name}_{target_name}",
                        constraint_type="eq",
                        feature_type=ry.FS.scalarProductXY,
                        frames=[target_name, robot_name],
                        objective_type=ry.OT.eq,
                        phase_idx=phase_idx,
                        target=0.0,
                        weight=[self.gripper_weight],  # Weight should be a list
                    ),
                    komo,
                )

                constraint_manager.register(
                    Constraint(
                        name=f"phase{phase_idx+1}_scalarProductYY_{robot_name}_{target_name}",
                        constraint_type="eq",
                        feature_type=ry.FS.scalarProductYY,
                        frames=[target_name, robot_name],
                        objective_type=ry.OT.eq,
                        phase_idx=phase_idx,
                        target=0.0,
                        weight=[self.gripper_weight],  # Weight should be a list
                    ),
                    komo,
                )

                # Position constraints (inequality) - need two constraints for upper and lower bounds
                upper, lower = self.position_rel_z_bounds
                # Upper bound constraint
                constraint_manager.register(
                    Constraint(
                        name=f"phase{phase_idx+1}_positionRel_{robot_name}_{target_name}_upper",
                        constraint_type="ineq",
                        feature_type=ry.FS.positionRel,
                        frames=[robot_name, target_name],
                        objective_type=ry.OT.ineq,
                        phase_idx=phase_idx,
                        target=[0, 0, upper],
                        weight=[self.gripper_weight],  # Weight should be a list to match original code
                        bounds=(upper, lower),  # Store bounds for checking
                    ),
                    komo,
                )
                # Lower bound constraint (negative weight for flipped inequality)
                constraint_manager.register(
                    Constraint(
                        name=f"phase{phase_idx+1}_positionRel_{robot_name}_{target_name}_lower",
                        constraint_type="ineq",
                        feature_type=ry.FS.positionRel,
                        frames=[robot_name, target_name],
                        objective_type=ry.OT.ineq,
                        phase_idx=phase_idx,
                        target=[0, 0, lower],
                        weight=[-self.gripper_weight],  # Negative weight for lower bound
                        bounds=(upper, lower),  # Store bounds for checking
                    ),
                    komo,
                )

                # Register a single constraint for checking (checks z-component within bounds)
                constraint_manager.register(
                    Constraint(
                        name=f"phase{phase_idx+1}_positionRel_{robot_name}_{target_name}_z",
                        constraint_type="ineq",
                        feature_type=ry.FS.positionRel,
                        frames=[robot_name, target_name],
                        objective_type=ry.OT.ineq,  # Not actually used for this check-only constraint
                        phase_idx=phase_idx,
                        target=None,
                        weight=0.0,  # Don't add to KOMO, only for checking
                        bounds=(upper, lower),
                    )
                )

        # Soft constraints: maximize distance between baselink and element for each phase
        if self._baselink_target_pairs_by_phase is not None and self.baselink_distance_weight > 0:
            for phase_idx in range(self.num_phases):
                for baselink_name, target_name in self._baselink_target_pairs_by_phase[phase_idx]:
                    try:
                        # Verify baselink frame exists in config
                        self.config.getFrame(baselink_name)
                        
                        # a = self.config.eval(ry.FS.distance, [baselink_name, target_name])
                        
                        constraint_manager.register(
                            Constraint(
                                name=f"phase{phase_idx+1}_baselink_distance_{baselink_name}_{target_name}",
                                constraint_type="sos",
                                feature_type=ry.FS.distance,
                                frames=[baselink_name, target_name],
                                objective_type=ry.OT.sos,
                                phase_idx=phase_idx,
                                target=-self.baselink_distance_target,
                                weight=[self.baselink_distance_weight],
                            ),
                            komo,
                        )
                    except Exception:
                        # If baselink frame doesn't exist, skip this constraint
                        # This allows the solver to work even if some baselink frames are missing
                        pass

        # Directed pose/heading constraints (applies to all phases)
        # Position via positionRel; rotation via scalarProductXX (x-axes alignment)
        # Note: These are sos (soft) objectives, but we can still register them for verification
        if self.pose_rel_constraints:
            for frame_i, frame_j, target_pose in self.pose_rel_constraints:
                pos_target = target_pose[:3]
                pos_weight = [self.pose_rel_weight] * 3
                constraint_manager.register(
                    Constraint(
                        name=f"poseRel_position_{frame_i}_{frame_j}",
                        constraint_type="sos",
                        feature_type=ry.FS.positionRel,
                        frames=[frame_i, frame_j],
                        objective_type=ry.OT.sos,
                        phase_idx=None,  # Applies to all phases
                        target=pos_target,
                        weight=pos_weight,
                    ),
                    komo,
                )

                constraint_manager.register(
                    Constraint(
                        name=f"poseRel_scalarProductXX_{frame_i}_{frame_j}",
                        constraint_type="sos",
                        feature_type=ry.FS.scalarProductXX,
                        frames=[frame_i, frame_j],
                        objective_type=ry.OT.sos,
                        phase_idx=None,  # Applies to all phases
                        target=1.0,
                        weight=[self.pose_rel_weight],  # Weight should be a list
                    ),
                    komo,
                )

        # Collision avoidance (applies to all phases)
        # Register once for all phases, but check for each phase
        constraint_manager.register(
            Constraint(
                name="accumulatedCollisions_all",
                constraint_type="eq",
                feature_type=ry.FS.accumulatedCollisions,
                frames=[],
                objective_type=ry.OT.eq,
                phase_idx=None,  # Applies to all phases
                target=0.0,
                weight=[self.collision_weight],  # Weight should be a list
            ),
            komo,
        )

        # Also register per-phase constraints for checking
        for phase_idx in range(self.num_phases):
            constraint_manager.register(
                Constraint(
                    name=f"phase{phase_idx+1}_accumulatedCollisions",
                    constraint_type="eq",
                    feature_type=ry.FS.accumulatedCollisions,
                    frames=[],
                    objective_type=ry.OT.eq,  # Not actually used, only for checking
                    phase_idx=phase_idx,
                    target=0.0,
                    weight=0.0,  # Don't add to KOMO, only for checking
                )
            )

        # Enforce r3 joint states equal between Phase 1 and Phase 2 (soft equality on velocity for r3 joints)
        all_joint_names = self.config.getJointNames()
        phase_switch_joint_indices = []
        for robot_name in self.phase_switch_robots:
            phase_switch_joint_indices.extend([i for i, name in enumerate(all_joint_names) if name.startswith(robot_name + "_")])
        if len(phase_switch_joint_indices) > 0:
            weight = np.zeros(len(all_joint_names))
            weight[phase_switch_joint_indices] = self.phase_switch_weight  # stronger weight for equality
            # order=1 enforces zero velocity at the phase transition (q2 - q1 = 0) for selected joints
            constraint_manager.register(
                Constraint(name=f"phase_switch_joint_equality_{robot_name}", constraint_type="eq", feature_type=ry.FS.qItself, frames=[], objective_type=ry.OT.eq, phase_idx=1, target=0.0, weight=weight.tolist(), order=1),  # Phase 2 (index 1, KOMO phase 2)
                komo,
            )

        ret_dict, keyframes = self.solve_komo_problem(komo, constraint_manager=constraint_manager, initial_state=initial_state, view=view)

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


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Utility functions
    "horizontal_cylinder_quaternion",
    "normalize_vector",
    "generate_random_initial_state",
    # Geometry classes
    "CylinderElement",
    "GeometryCalculator",
    "RobotPositionCalculator",
    # Element creation functions
    "create_horizontal_element",
    "create_vertical_element",
    # Constraint management
    "Constraint",
    "ConstraintManager",
    # Main solver
    "MultiPhaseKomoSolver",
]
