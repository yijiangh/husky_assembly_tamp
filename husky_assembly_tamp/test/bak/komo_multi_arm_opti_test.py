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
    A solver for single keyframe optimization using KOMO.

    This class encapsulates the logic for solving inverse kinematics problems
    with gripper constraints for multiple robots.

    Supports nested lists for robot_names and target_names to represent
    multiple robots grasping the same target. For example:
        robot_names = [["r1_gripper", "r2_gripper"], "r3_gripper"]
        target_names = [["element_1", "element_1"], "element_2"]
    This means r1 and r2 both grasp element_1, while r3 grasps element_2.
    """

    def __init__(
        self,
        config: ry.Config,
        robot_names: List[Union[str, List[str]]],
        target_names: List[Union[str, List[str]]],
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
    ):
        """
        Initialize the SingleKeyFrameSolver.

        Args:
            config: The ry.Config object containing the robot and environment setup
            robot_names: List of robot gripper frame names, can be nested lists for dual-arm grasping.
                         e.g., ["r1_gripper", "r2_gripper"] or [["r1_gripper", "r2_gripper"], "r3_gripper"]
            target_names: List of target element frame names, structure must match robot_names.
                          e.g., ["element_1", "element_2"] or [["element_1", "element_1"], "element_2"]
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
                                  If None, uses robot_names for distance calculation.
            distance_weight: Weight for distance constraint objectives
            collision_weight: Weight for collision avoidance constraint
        """
        self.config = config
        self.robot_names = robot_names
        self.target_names = target_names
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
        self.distance_frame_names = distance_frame_names if distance_frame_names is not None else robot_names
        self.distance_weight = distance_weight
        self.collision_weight = collision_weight

        # Validate structure matching
        if len(robot_names) != len(target_names):
            raise ValueError(f"Number of robot groups ({len(robot_names)}) must match number of target groups ({len(target_names)})")

        # Validate nested structure
        for i, (robots, targets) in enumerate(zip(robot_names, target_names)):
            robots_is_list = isinstance(robots, list)
            targets_is_list = isinstance(targets, list)
            if robots_is_list != targets_is_list:
                raise ValueError(f"Group {i}: robot_names and target_names must have matching structure (both list or both str)")
            if robots_is_list and len(robots) != len(targets):
                raise ValueError(f"Group {i}: nested lists must have same length ({len(robots)} vs {len(targets)})")

        # Build flattened robot-target pairs
        self._robot_target_pairs = self._build_robot_target_pairs()

        # Build group distance constraint pairs
        self._group_distance_pairs = self._build_group_distance_pairs()

        if freeze_arm_joints:
            if x_home is None:
                raise ValueError("x_home must be provided when freeze_arm_joints=True")
            if arm_joint_indices is None:
                raise ValueError("arm_joint_indices must be provided when freeze_arm_joints=True")
            # Note: arm_joint_indices should match the total number of robots (flattened)
            total_robots = len(self._robot_target_pairs)
            if len(arm_joint_indices) != total_robots:
                raise ValueError(f"Number of arm_joint_indices ({len(arm_joint_indices)}) must match total number of robots ({total_robots})")

    def _build_robot_target_pairs(self) -> List[Tuple[str, str]]:
        """
        Build a flattened list of (robot_name, target_name) pairs from potentially nested lists.

        Returns:
            List of (robot_name, target_name) tuples
        """
        pairs = []
        for robots, targets in zip(self.robot_names, self.target_names):
            if isinstance(robots, list):
                # Nested list case: multiple robots for potentially same/different targets
                for robot, target in zip(robots, targets):
                    pairs.append((robot, target))
            else:
                # Simple case: single robot-target pair
                pairs.append((robots, targets))
        return pairs

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

    def check_gripper_constraints(self, q_state: np.ndarray) -> Tuple[bool, List, List]:
        """
        Manually check gripper constraints using eval.
        Constraints with weight=0 are skipped.

        Args:
            q_state: Joint state to check

        Returns:
            Tuple of (all_satisfied, eq_constraints, ineq_constraints)
        """
        self.config.setJointState(q_state)
        self.config.computeCollisions()

        eq_constraints = []
        ineq_constraints = []

        # Check constraints for each robot-target pair (skip if gripper_weight == 0)
        if self.gripper_weight != 0:
            for robot_name, target_name in self._robot_target_pairs:
                scalar_product_xy = self.config.eval(ry.FS.scalarProductXY, [target_name, robot_name])
                scalar_product_yy = self.config.eval(ry.FS.scalarProductYY, [target_name, robot_name])
                val_xy = float(scalar_product_xy[0][0])
                val_yy = float(scalar_product_yy[0][0])
                eq_constraints.append((f"scalarProductXY_{robot_name}_{target_name}", val_xy, 0))
                eq_constraints.append((f"scalarProductYY_{robot_name}_{target_name}", val_yy, 0))

                position_rel = self.config.eval(ry.FS.positionRel, [robot_name, target_name])
                pos_rel = position_rel[0]
                val_pos_z = float(pos_rel[2])
                upper, lower = self.position_rel_z_bounds
                ineq_constraints.append((f"positionRel_{robot_name}_{target_name}_z", val_pos_z, upper, lower))

        # Check group distance constraints (skip if distance_weight == 0)
        if self.distance_weight != 0:
            for robot_i, robot_j, target_dist in self._group_distance_pairs:
                dist_val = self.config.eval(ry.FS.distance, [robot_i, robot_j])
                actual_dist = float(dist_val[0][0])
                eq_constraints.append((f"distance_{robot_i}_{robot_j}", actual_dist, -target_dist))

        accumulated_collisions = self.config.eval(ry.FS.accumulatedCollisions, [])
        val_collisions = float(accumulated_collisions[0][0])
        eq_constraints.append(("accumulatedCollisions", val_collisions, 0))

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
            if num_attempt == 0 and initial_state is not None:
                komo.initWithConstant(initial_state)
            elif num_attempt > 0:
                x_init = generate_random_initial_state(self.config)
                komo.initWithConstant(x_init)

            solver = ry.NLP_Solver(komo.nlp(), verbose=0)

            if self.damping is not None:
                solver.setOptions(damping=self.damping)
            if self.wolfe is not None:
                solver.setOptions(wolfe=self.wolfe)
                
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
                if keyframes is not None and len(keyframes) > 0:
                    is_feasible, eq_vals, ineq_vals = self.check_gripper_constraints(keyframes[0])
                    if is_feasible:
                        return retval, keyframes

        return retval, None

    def solve(self, initial_state: np.ndarray, view: bool = False) -> Tuple[Any, Optional[ry.KOMO]]:
        """
        Solve the optimization problem for a given initial state.

        Args:
            initial_state: Initial joint state
            view: Whether to visualize the solution

        Returns:
            Tuple of (RetWrapper result, komo object)
        """
        # Apply joint freezing if enabled
        if self.freeze_arm_joints:
            q_init = initial_state.copy()
            for robot_idx, joint_indices in enumerate(self.arm_joint_indices):
                for joint_idx in joint_indices:
                    if joint_idx < len(self.x_home):
                        q_init[joint_idx] = self.x_home[joint_idx]
            initial_state = q_init

        komo = ry.KOMO(self.config, 1, 1, 0, True)

        # Add joint state regularization objective
        komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [self.joint_weight], initial_state)

        # Add gripper constraints for each robot-target pair (using flattened pairs)
        for robot_name, target_name in self._robot_target_pairs:
            # Alignment constraints (scalar products)
            komo.addObjective([], ry.FS.scalarProductXY, [target_name, robot_name], ry.OT.eq, [self.gripper_weight], [0])
            komo.addObjective([], ry.FS.scalarProductYY, [target_name, robot_name], ry.OT.eq, [self.gripper_weight], [0])

            # Position constraints (inequality)
            upper, lower = self.position_rel_z_bounds
            komo.addObjective([], ry.FS.positionRel, [robot_name, target_name], ry.OT.ineq, [self.gripper_weight], [0, 0, upper])
            komo.addObjective([], ry.FS.positionRel, [robot_name, target_name], ry.OT.ineq, [-self.gripper_weight], [0, 0, lower])

        # Add group distance constraints
        for robot_i, robot_j, target_dist in self._group_distance_pairs:
            # komo.addObjective([], ry.FS.distance, [robot_i, robot_j], ry.OT.eq, [self.distance_weight], [-target_dist])
            komo.addObjective([], ry.FS.distance, [robot_i, robot_j], ry.OT.sos, [self.distance_weight], [-target_dist]) # MARK
            # ry.FS.poseRel # MARK

        # Collision avoidance
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [self.collision_weight])

        ret_dict, keyframes = self.solve_komo_problem(komo, initial_state=initial_state, view=view)

        class RetWrapper:
            def __init__(self, ret_dict, keyframes):
                self.feasible = ret_dict.get("feasible", False) and keyframes is not None
                self.eq = ret_dict.get("eq", float("inf"))
                self.ineq = ret_dict.get("ineq", float("inf"))
                self.sos = ret_dict.get("sos", float("inf"))
                self.keyframes = keyframes

        ret = RetWrapper(ret_dict, keyframes)

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

    # Calculate robot initial positions for Phase 1
    # All robots use calculate_position_perpendicular
    # For Phase 1: r3 targets element_5, so position relative to element_5
    robot_1_pos_phase1 = RobotPositionCalculator.calculate_position_perpendicular(element_4.position, element_4.direction, ROBOT_DISTANCE)
    robot_2_pos_phase1 = RobotPositionCalculator.calculate_position_perpendicular(element_4.position, element_4.direction, ROBOT_DISTANCE + 0.5)
    robot_3_pos_phase1 = RobotPositionCalculator.calculate_position_perpendicular(element_5.position, element_5.direction, ROBOT_DISTANCE)

    # Calculate robot initial positions for Phase 2
    # Only r3 uses calculate_position_perpendicular (targets element_6)
    robot_3_pos_phase2 = RobotPositionCalculator.calculate_position_perpendicular(element_6.position, element_6.direction, ROBOT_DISTANCE)

    # Add robots to configuration (using Phase 1 positions initially)
    r1_base_frame = RobotPositionCalculator.add_robot_to_config(C, ry.raiPath("panda/panda.g"), "r1_", robot_1_pos_phase1)
    r2_base_frame = RobotPositionCalculator.add_robot_to_config(C, ry.raiPath("panda/panda.g"), "r2_", robot_2_pos_phase1)
    r3_base_frame = RobotPositionCalculator.add_robot_to_config(C, ry.raiPath("panda/panda.g"), "r3_", robot_3_pos_phase1)

    base1 = C.getFrame("r1_panda_link0")
    base2 = C.getFrame("r2_panda_link0")
    base3 = C.getFrame("r3_panda_link0")
    # base1.setJoint(ry.JT.transXY, [-0.25, -0.25, 0.25, 0.25])
    # base2.setJoint(ry.JT.transXY, [-0.25, -0.25, 0.25, 0.25])
    # base3.setJoint(ry.JT.transXY, [-0.25, -0.25, 0.25, 0.25])
    base1.setJoint(ry.JT.transXYPhi, [-0.25, -0.25, -np.pi, 0.25, 0.25, np.pi])
    base2.setJoint(ry.JT.transXYPhi, [-0.25, -0.25, -np.pi, 0.25, 0.25, np.pi])
    base3.setJoint(ry.JT.transXYPhi, [-0.25, -0.25, -np.pi, 0.25, 0.25, np.pi])
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

    x_home = C.getJointState()

    # Solver configuration parameters (shared)
    joint_weight = 0.1
    gripper_weight = 5.11
    position_rel_z_bounds = (0.4, -0.4)
    constraint_eps = 1e-3
    max_attempts = 1
    freeze_arm_joints = True

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
        np.array([[-1, 0.3], [-1, -1]]),  # r1-r2 base distance = 0.5
        None,  # Group 1 (r3): single robot, no distance constraint
    ]
    distance_weight = 0
    collision_weight = 10

    # =========================================================================
    # Phase 1 Setup: element_6 disabled, r3 targets element_5
    # =========================================================================
    robot_names_phase1 = [["r1_gripper", "r2_gripper"], "r3_gripper"]
    target_names_phase1 = [["element_4", "element_4"], "element_5"]

    solver_phase1 = SingleKeyFrameSolver(
        config=C,
        robot_names=robot_names_phase1,
        target_names=target_names_phase1,
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
    )

    # =========================================================================
    # Phase 2 Setup: full setup, r3 targets element_6
    # =========================================================================
    robot_names_phase2 = [["r1_gripper", "r2_gripper"], "r3_gripper"]
    target_names_phase2 = [["element_4", "element_4"], "element_6"]

    solver_phase2 = SingleKeyFrameSolver(
        config=C,
        robot_names=robot_names_phase2,
        target_names=target_names_phase2,
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
    )

    num_initial_states = 100
    print(f"Generating {num_initial_states} random initial states")
    print(f"Joint weight: {joint_weight}")
    print(f"Gripper weight: {gripper_weight}")
    print(f"Freeze arm joints: {freeze_arm_joints}")
    print(f"\nTwo-phase optimization:")
    print(f"  Phase 1: Robot group -> element_4, Single robot -> element_5 (element_6 disabled)")
    print(f"  Phase 2: Robot group -> element_4, Single robot -> element_6 (full setup)")

    np.random.seed(42)

    # Generate initial states for Phase 1
    initial_states_phase1 = []
    for i in range(num_initial_states):
        q_init = generate_random_initial_state(C)

        # Freeze arm joints to home position
        q_init_freeze = q_init.copy()
        q_init_freeze[0 * 10 + 3 : 0 * 10 + 3 + 7] = x_home[0 * 10 + 3 : 0 * 10 + 3 + 7]
        q_init_freeze[1 * 10 + 3 : 1 * 10 + 3 + 7] = x_home[1 * 10 + 3 : 1 * 10 + 3 + 7]
        q_init_freeze[2 * 10 + 3 : 2 * 10 + 3 + 7] = x_home[2 * 10 + 3 : 2 * 10 + 3 + 7]

        initial_states_phase1.append(q_init_freeze)
        print(f"Initial state {i+1}: generated")

    pp.wait_for_user()

    # Iterative optimization parameters
    max_iterations = 50  # Maximum iterations (1->2->1->2->...)
    convergence_threshold = 1e-3  # Threshold for robot group configuration convergence

    def check_robot_group_convergence(q_phase1: np.ndarray, q_phase2: np.ndarray, threshold: float) -> Tuple[bool, float]:
        """
        Check if robot group configuration has converged between Phase 1 and Phase 2.
        
        Convergence condition: L1 norm of robot group (r1 + r2) configuration difference
        between Phase 1 and Phase 2 is below the threshold.
        
        Args:
            q_phase1: Joint state from Phase 1
            q_phase2: Joint state from Phase 2
            threshold: Convergence threshold for L1 norm
            
        Returns:
            Tuple of (is_converged, l1_norm_diff)
        """
        # Extract robot group joints (r1 + r2)
        q_group_phase1 = np.concatenate([q_phase1[robot_1_joint_indices], q_phase1[robot_2_joint_indices]])
        q_group_phase2 = np.concatenate([q_phase2[robot_1_joint_indices], q_phase2[robot_2_joint_indices]])
        # Use L1 norm (1st order norm)
        diff = np.linalg.norm(q_group_phase2 - q_group_phase1, ord=1)
        return diff < threshold, diff

    all_results = []

    for state_idx, initial_state in enumerate(initial_states_phase1):
        print(f"\n{'='*60}")
        print(f"Initial State {state_idx + 1}/{num_initial_states}")
        print(f"{'='*60}")

        # History for tracking results at each iteration
        q_history_phase1 = []  # Results from Phase 1 iterations
        q_history_phase2 = []  # Results from Phase 2 iterations

        # Current state for robot group (will be updated each iteration)
        q_robot_group_current = initial_state.copy()

        # Previous results for single robot (r3)
        q_r3_prev_phase1 = None  # r3 result from previous Phase 1
        q_r3_prev_phase2 = None  # r3 result from previous Phase 2

        converged = False
        final_success = False
        iteration = 0

        while iteration < max_iterations and not converged:
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

            # =================================================================
            # Phase 1: r3 -> element_5 (element_6 contact stays enabled for collision avoidance)
            # =================================================================
            print(f"  Phase 1: r3 -> element_5")

            # Prepare initial state for Phase 1
            init_phase1 = q_robot_group_current.copy()

            # Set r3 initial value:
            # - First iteration: use calculated position with randomization
            # - Subsequent iterations: use result from previous Phase 1 (if exists)
            if q_r3_prev_phase1 is not None:
                # Use r3 result from previous Phase 1
                for idx in robot_3_joint_indices:
                    init_phase1[idx] = q_r3_prev_phase1[idx]
            else:
                # First Phase 1: randomize r3 around calculated position
                init_phase1[robot_3_joint_indices[0]] = robot_3_pos_phase1[0] + np.random.uniform(-0.25, 0.25)
                init_phase1[robot_3_joint_indices[1]] = robot_3_pos_phase1[1] + np.random.uniform(-0.25, 0.25)
                init_phase1[robot_3_joint_indices[2]] = np.random.uniform(-np.pi, np.pi)

            C.setJointState(init_phase1)
            ret_phase1, _ = solver_phase1.solve(init_phase1, view=False)

            if not ret_phase1.feasible:
                print(f"    ✗ Phase 1 failed at iteration {iteration + 1}, continuing...")
                # Phase 1 failed, but don't exit - continue to next iteration
                # Reset q_r3_prev_phase1 to force re-randomization in next iteration
                q_r3_prev_phase1 = None
                iteration += 1
                continue

            q_phase1 = ret_phase1.keyframes[0]
            q_history_phase1.append(q_phase1.copy())
            q_r3_prev_phase1 = q_phase1.copy()  # Save r3 result for next Phase 1
            print(f"    ✓ Phase 1 feasible (eq={ret_phase1.eq:.3e})")

            # =================================================================
            # Phase 2: r3 -> element_6
            # =================================================================
            print(f"  Phase 2: r3 -> element_6")

            # Prepare initial state for Phase 2
            # Robot group uses Phase 1 result
            init_phase2 = q_phase1.copy()

            # Set r3 initial value:
            # - First Phase 2: use calculated position with randomization
            # - Subsequent iterations: use result from previous Phase 2 (if exists)
            if q_r3_prev_phase2 is not None:
                # Use r3 result from previous Phase 2
                for idx in robot_3_joint_indices:
                    init_phase2[idx] = q_r3_prev_phase2[idx]
            else:
                # First Phase 2: randomize r3 around calculated position
                init_phase2[robot_3_joint_indices[0]] = robot_3_pos_phase2[0] + np.random.uniform(-0.25, 0.25)
                init_phase2[robot_3_joint_indices[1]] = robot_3_pos_phase2[1] + np.random.uniform(-0.25, 0.25)
                init_phase2[robot_3_joint_indices[2]] = np.random.uniform(-np.pi, np.pi)

            C.setJointState(init_phase2)
            ret_phase2, _ = solver_phase2.solve(init_phase2, view=False)

            if not ret_phase2.feasible:
                print(f"    ✗ Phase 2 failed at iteration {iteration + 1}, continuing...")
                # Phase 2 failed, but don't exit - continue to next iteration
                # Reset q_r3_prev_phase2 to force re-randomization in next iteration
                q_r3_prev_phase2 = None
                iteration += 1
                continue

            q_phase2 = ret_phase2.keyframes[0]
            q_history_phase2.append(q_phase2.copy())
            q_r3_prev_phase2 = q_phase2.copy()  # Save r3 result for next Phase 2
            print(f"    ✓ Phase 2 feasible (eq={ret_phase2.eq:.3e})")

            # Check convergence of robot group between Phase 1 and Phase 2
            # Convergence: both phases feasible AND L1 norm of robot group diff < threshold
            is_converged, l1_diff = check_robot_group_convergence(q_phase1, q_phase2, convergence_threshold)
            print(f"  Robot group L1 diff (Phase1 vs Phase2): {l1_diff:.6f}")
            
            if is_converged:
                converged = True
                final_success = True
                print(f"  ✓ Robot group converged at iteration {iteration + 1} (L1 diff={l1_diff:.6f} < {convergence_threshold})")
            else:
                # Update robot group current state for next iteration
                q_robot_group_current = q_phase2.copy()
                print(f"  ✗ Not converged yet (L1 diff={l1_diff:.6f} >= {convergence_threshold})")

            iteration += 1

        # Record results
        result = {
            "state_idx": state_idx,
            "initial_state": initial_state.tolist(),
            "converged": converged,
            "final_success": final_success,
            "num_iterations": iteration,
            "q_history_phase1": [q.tolist() for q in q_history_phase1],
            "q_history_phase2": [q.tolist() for q in q_history_phase2],
            "q_final_phase1": q_history_phase1[-1].tolist() if q_history_phase1 else None,
            "q_final_phase2": q_history_phase2[-1].tolist() if q_history_phase2 else None,
        }
        all_results.append(result)

        if final_success:
            print(f"\n✓ Converged after {iteration} iteration(s) for initial state {state_idx + 1}")
        else:
            print(f"\n✗ Failed after {iteration} iteration(s) for initial state {state_idx + 1}")

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    converged_count = sum(1 for r in all_results if r["converged"])
    avg_iterations = np.mean([r["num_iterations"] for r in all_results if r["converged"]]) if converged_count > 0 else 0
    print(f"Converged: {converged_count}/{num_initial_states}")
    print(f"Average iterations (converged): {avg_iterations:.2f}")
    print(f"Max iterations allowed: {max_iterations}")
    print(f"Convergence threshold: {convergence_threshold}")

    # Collect all converged configurations
    all_converged = [r for r in all_results if r["converged"]]

    output_dir = "komo_results"
    os.makedirs(output_dir, exist_ok=True)

    if len(all_converged) > 0:
        print(f"\nSaving {len(all_converged)} converged configurations...")

        results_file = os.path.join(output_dir, "converged_iterative_configs.json")
        with open(results_file, "w") as f:
            json.dump(all_converged, f, indent=2)
        print(f"Saved to {results_file}")

    if len(all_converged) > 0:
        print("\nConverged configurations summary:")
        for r in all_converged:
            print(f"  State {r['state_idx']+1}: Converged in {r['num_iterations']} iteration(s)")

        print(f"\n{'='*60}")
        print(f"Viewing all {len(all_converged)} converged configurations")
        print(f"{'='*60}")
        
        for idx, result in enumerate(all_converged):
            print(f"\n{'='*60}")
            print(f"Configuration {idx + 1}/{len(all_converged)} (State {result['state_idx'] + 1})")
            print(f"Converged in {result['num_iterations']} iteration(s)")
            print(f"{'='*60}")

            # Show final Phase 1 result
            print(f"\n--- Final Phase 1 Result (r3 -> element_5) ---")
            q_phase1 = np.array(result["q_final_phase1"])
            C.setJointState(q_phase1)
            C.view()

            print(f"  Distance constraints:")
            for robot_i, robot_j, target_dist in solver_phase1._group_distance_pairs:
                actual_dist = C.eval(ry.FS.distance, [robot_i, robot_j])[0][0]
                print(f"    {robot_i} - {robot_j}: target={target_dist:.3f}, actual={actual_dist:.3f}")

            pp.wait_for_user()

            # Show final Phase 2 result
            print(f"\n--- Final Phase 2 Result (r3 -> element_6) ---")
            q_phase2 = np.array(result["q_final_phase2"])
            C.setJointState(q_phase2)
            C.view()

            print(f"  Distance constraints:")
            for robot_i, robot_j, target_dist in solver_phase2._group_distance_pairs:
                actual_dist = C.eval(ry.FS.distance, [robot_i, robot_j])[0][0]
                print(f"    {robot_i} - {robot_j}: target={target_dist:.3f}, actual={actual_dist:.3f}")

            pp.wait_for_user()

            # Show iteration history if more than 1 iteration
            if result["num_iterations"] > 1:
                print(f"\n--- Iteration History ---")
                for iter_idx in range(result["num_iterations"]):
                    print(f"\nIteration {iter_idx + 1}:")

                    # Phase 1 of this iteration
                    if iter_idx < len(result["q_history_phase1"]):
                        q_p1 = np.array(result["q_history_phase1"][iter_idx])
                        C.setJointState(q_p1)
                        print(f"  Phase 1 (r3 -> element_5)")
                        C.view()
                        pp.wait_for_user()

                    # Phase 2 of this iteration
                    if iter_idx < len(result["q_history_phase2"]):
                        q_p2 = np.array(result["q_history_phase2"][iter_idx])
                        C.setJointState(q_p2)
                        print(f"  Phase 2 (r3 -> element_6)")
            C.view()
            pp.wait_for_user()
    else:
        print("\nNo converged configurations found!")
        C.view()
        pp.wait_for_user()
