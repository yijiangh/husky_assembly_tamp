import numpy as np
import pybullet_planning as pp
import robotic as ry
import json
import os
from typing import List, Tuple, Optional, Dict, Any, Union


def horizontal_cylinder_quaternion(direction):
    dir_norm = direction / np.linalg.norm(direction)
    z_axis = np.array([0, 0, 1])
    rot_axis = np.cross(z_axis, dir_norm)
    if np.linalg.norm(rot_axis) < 1e-6:
        rot_axis = np.array([1, 0, 0])
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    angle = np.pi / 2
    return [np.cos(angle / 2), rot_axis[0] * np.sin(angle / 2), rot_axis[1] * np.sin(angle / 2), rot_axis[2] * np.sin(angle / 2)]


def normalize_vector(vec, default=None):
    if default is None:
        default = np.array([1, 0, 0])
    return vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 1e-6 else default


def generate_random_initial_state(config: ry.Config) -> np.ndarray:
    """Generate a random initial joint state"""
    low, high = config.getJointLimits()
    q_random = np.random.uniform(low, high, size=len(low))
    return q_random


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
                stopTolerance=1e-4,
                stepMax=0.5,
                damping=0.1,
                stepInc=2.0,
                stepDec=0.3,
                wolfe=0.001,
                muInit=0.1,
                muInc=2.0,
                muMax=100000.0,
            )

            retval = solver.solve()
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
            komo.addObjective([], ry.FS.distance, [robot_i, robot_j], ry.OT.eq, [self.distance_weight], [-target_dist])

        # Collision avoidance
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)

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

CYLINDER_SIZE = [1.0, 0.01]
PROTRUSION_OFFSET = 0.15
VERTICAL_DISTANCE = 0.04
HALF_LENGTH = 0.5
VERTICAL_Z = 0.5
HORIZONTAL_Z = [0.75, 0.77, 0.79]
ROBOT_DISTANCE = -1

v1_pos = [0.5, 0.0, VERTICAL_Z]
v2_pos = [-0.25, 0.433, VERTICAL_Z]
v3_pos = [-0.25, -0.433, VERTICAL_Z]


def create_horizontal_element(config: ry.Config, name: str, v_start: np.ndarray, v_end: np.ndarray, protrusion_target: np.ndarray, z_pos: float, color: list[float]) -> tuple[np.ndarray, np.ndarray]:
    edge_dir = np.array([v_end[0] - v_start[0], v_end[1] - v_start[1], 0])
    edge_mid = np.array([(v_start[0] + v_end[0]) / 2, (v_start[1] + v_end[1]) / 2, z_pos])
    protrusion_dir = np.array([protrusion_target[0] - edge_mid[0], protrusion_target[1] - edge_mid[1], 0])
    protrusion_dir = normalize_vector(protrusion_dir) * PROTRUSION_OFFSET
    element_pos = edge_mid + protrusion_dir

    config.addFrame(name).setShape(ry.ST.cylinder, CYLINDER_SIZE).setPosition(element_pos.tolist()).setQuaternion(horizontal_cylinder_quaternion(edge_dir)).setColor(color).setContact(1)

    return element_pos, edge_dir


def calculate_element_end(element_pos, edge_dir, target_pos):
    edge_dir_norm = normalize_vector(edge_dir)
    dir_to_target = np.array([target_pos[0] - element_pos[0], target_pos[1] - element_pos[1], 0])
    dir_to_target_norm = normalize_vector(dir_to_target, edge_dir_norm)
    dot = np.dot(edge_dir_norm, dir_to_target_norm)
    return element_pos + edge_dir_norm * HALF_LENGTH * (1 if dot > 0 else -1)


def position_vertical_element(element_end, initial_v_pos):
    dir_to_v = np.array([initial_v_pos[0] - element_end[0], initial_v_pos[1] - element_end[1], 0])
    dir_to_v_norm = normalize_vector(dir_to_v)
    v_pos = (element_end + dir_to_v_norm * VERTICAL_DISTANCE).tolist()
    v_pos[2] = VERTICAL_Z
    return v_pos


def create_vertical_element(config: ry.Config, name: str, position: list[float], color: list[float]) -> None:
    config.addFrame(name).setShape(ry.ST.cylinder, CYLINDER_SIZE).setPosition(position).setColor(color).setContact(1)


def calculate_robot_position(element_pos, edge_dir, distance):
    edge_dir_norm = normalize_vector(edge_dir)
    perp_dir = np.array([-edge_dir_norm[1], edge_dir_norm[0], 0])
    element_xy = np.array([element_pos[0], element_pos[1], 0.0])
    robot_pos = element_xy + perp_dir * distance
    return robot_pos.tolist()


if __name__ == "__main__":
    element_4_pos, edge_23_dir = create_horizontal_element(C, "element_4", v2_pos, v3_pos, v1_pos, HORIZONTAL_Z[0], [1, 1, 0])
    element_5_pos, edge_31_dir = create_horizontal_element(C, "element_5", v3_pos, v1_pos, v2_pos, HORIZONTAL_Z[1], [1, 0, 1])
    element_6_pos, edge_12_dir = create_horizontal_element(C, "element_6", v1_pos, v2_pos, v3_pos, HORIZONTAL_Z[2], [0, 1, 1])

    element_4_end = calculate_element_end(element_4_pos, edge_23_dir, v1_pos)
    element_5_end = calculate_element_end(element_5_pos, edge_31_dir, v2_pos)
    element_6_end = calculate_element_end(element_6_pos, edge_12_dir, v3_pos)
    edge_12_dir_norm = normalize_vector(edge_12_dir)
    dir_to_v3 = np.array([v3_pos[0] - element_6_pos[0], v3_pos[1] - element_6_pos[1], 0])
    dot_6 = np.dot(edge_12_dir_norm, normalize_vector(dir_to_v3, edge_12_dir_norm))
    element_6_other_end = element_6_pos - edge_12_dir_norm * HALF_LENGTH * (1 if dot_6 > 0 else -1)

    v1_pos = position_vertical_element(element_4_end, v1_pos)
    v2_pos = position_vertical_element(element_5_end, v2_pos)
    v3_pos = position_vertical_element(element_6_other_end, v3_pos)

    create_vertical_element(C, "element_1", v1_pos, [1, 0, 0])
    create_vertical_element(C, "element_2", v2_pos, [0, 1, 0])
    create_vertical_element(C, "element_3", v3_pos, [0, 0, 1])

    robot_1_pos = calculate_robot_position(element_4_pos, edge_23_dir, ROBOT_DISTANCE)
    robot_2_pos = calculate_robot_position(element_4_pos, edge_23_dir, ROBOT_DISTANCE + 0.5)
    # robot_2_pos = calculate_robot_position(element_5_pos, edge_31_dir, ROBOT_DISTANCE)
    robot_3_pos = calculate_robot_position(element_6_pos, edge_12_dir, ROBOT_DISTANCE)

    r1_base_frame = C.addFile(ry.raiPath("panda/panda.g"), "r1_").setPosition(robot_1_pos).setQuaternion([1, 0, 0, 1])
    r2_base_frame = C.addFile(ry.raiPath("panda/panda.g"), "r2_").setPosition(robot_2_pos).setQuaternion([1, 0, 0, 1])
    r3_base_frame = C.addFile(ry.raiPath("panda/panda.g"), "r3_").setPosition(robot_3_pos).setQuaternion([1, 0, 0, 1])

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

    # Solver configuration parameters
    joint_weight = 0.1
    gripper_weight = 5.11
    position_rel_z_bounds = (0.4, -0.4)
    constraint_eps = 1e-3
    max_attempts = 1
    freeze_arm_joints = True

    # Robot and target names
    # robot_names = ["r1_gripper", "r2_gripper", "r3_gripper"]
    # target_names = ["element_4", "element_5", "element_6"]
    robot_names = [["r1_gripper", "r2_gripper"], "r3_gripper"]
    target_names = [["element_4", "element_4"], "element_6"]

    # Arm joint indices for freezing (assuming 10 DOF per robot: 3 base + 7 arm)
    arm_joint_indices = [
        list(range(0 * 10 + 3, 0 * 10 + 3 + 7)),  # Robot 1 arm joints
        list(range(1 * 10 + 3, 1 * 10 + 3 + 7)),  # Robot 2 arm joints
        list(range(2 * 10 + 3, 2 * 10 + 3 + 7)),  # Robot 3 arm joints
    ]

    # Distance frame names (grouped like robot_names, using base link frames)
    distance_frame_names = [["r1_panda_link0", "r2_panda_link0"], "r3_panda_link0"]

    # Group distance constraints (upper triangular matrix, -1 means no constraint)
    # For group 0 with [r1_base, r2_base]: distance between r1 and r2 base links
    group_distance_constraints = [
        np.array([[-1, 0.5], [-1, -1]]),  # r1-r2 base distance = 0.5
        None,  # Group 1 (r3): single robot, no distance constraint
    ]
    distance_weight = 10

    # Initialize solver
    solver = SingleKeyFrameSolver(
        config=C,
        robot_names=robot_names,
        target_names=target_names,
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
    )

    num_initial_states = 100
    print(f"Generating {num_initial_states} random initial states")
    print(f"Joint weight: {joint_weight}")
    print(f"Gripper weight: {gripper_weight}")
    print(f"Freeze arm joints: {freeze_arm_joints}")

    np.random.seed(42)

    initial_states = []
    for i in range(num_initial_states):
        q_init = generate_random_initial_state(C)

        q_init_freeze = q_init.copy()
        q_init_freeze[0 * 10 + 3 : 0 * 10 + 3 + 7] = x_home[0 * 10 + 3 : 0 * 10 + 3 + 7]
        q_init_freeze[1 * 10 + 3 : 1 * 10 + 3 + 7] = x_home[1 * 10 + 3 : 1 * 10 + 3 + 7]
        q_init_freeze[2 * 10 + 3 : 2 * 10 + 3 + 7] = x_home[2 * 10 + 3 : 2 * 10 + 3 + 7]

        initial_states.append(q_init_freeze)
        print(f"Initial state {i+1}: {q_init_freeze}...")

    pp.wait_for_user()

    all_results = []

    for state_idx, initial_state in enumerate(initial_states):
        print(f"\n{'='*60}")
        print(f"Initial State {state_idx + 1}/{num_initial_states}")
        print(f"{'='*60}")

        C.setJointState(initial_state)

        feasible_config = None
        feasible_ret = None
        feasible_komo = None

        feasible_configs_for_state = []

        ret, komo = solver.solve(initial_state, view=False)

        if ret.feasible:
            q = ret.keyframes
            _, eq_vals, ineq_vals = solver.check_gripper_constraints(q[0])
            feasible_configs_for_state.append(
                {
                    "joint_weight": joint_weight,
                    "gripper_weight": gripper_weight,
                    "distance_weight": distance_weight,
                    "q": q[0].tolist(),
                    "ret_eq": float(ret.eq),
                    "ret_ineq": float(ret.ineq),
                    "ret_sos": float(ret.sos),
                    "eq_constraints": {name: float(val) for name, val, _ in eq_vals},
                    "ineq_constraints": {name: float(val) for name, val, _, _ in ineq_vals},
                }
            )

        if len(feasible_configs_for_state) > 0:
            feasible_config = (feasible_configs_for_state[0]["joint_weight"], feasible_configs_for_state[0]["gripper_weight"])
            feasible_q = feasible_configs_for_state[0]["q"]
        else:
            feasible_config = None
            feasible_q = None

        result = {
            "state_idx": state_idx,
            "initial_state": initial_state.tolist(),
            "feasible": feasible_config is not None,
            "feasible_configs": feasible_configs_for_state,
            "joint_weight": feasible_config[0] if feasible_config else None,
            "gripper_weight": feasible_config[1] if feasible_config else None,
            "q": feasible_q,
        }
        all_results.append(result)

        if feasible_config is None:
            print(f"✗ No feasible configuration found for initial state {state_idx + 1}")
        else:
            print(f"✓ Feasible configuration found for initial state {state_idx + 1}")

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    feasible_count = sum(1 for r in all_results if r["feasible"])
    total_feasible_configs = sum(len(r["feasible_configs"]) for r in all_results)
    print(f"Feasible solutions found: {feasible_count}/{num_initial_states} initial states")
    print(f"Total feasible configurations: {total_feasible_configs}")

    all_feasible_configs = []
    for r in all_results:
        for cfg in r["feasible_configs"]:
            all_feasible_configs.append(
                {
                    "state_idx": r["state_idx"],
                    "joint_weight": cfg["joint_weight"],
                    "gripper_weight": cfg["gripper_weight"],
                    "distance_weight": cfg["distance_weight"],
                    "q": cfg["q"],
                    "eq": cfg["ret_eq"],
                    "ineq": cfg["ret_ineq"],
                    "sos": cfg["ret_sos"],
                    "eq_constraints": cfg["eq_constraints"],
                    "ineq_constraints": cfg["ineq_constraints"],
                }
            )

    output_dir = "komo_results"
    os.makedirs(output_dir, exist_ok=True)

    if len(all_feasible_configs) > 0:
        print(f"\nSaving {len(all_feasible_configs)} feasible configurations...")

        results_file = os.path.join(output_dir, "feasible_configs.json")
        with open(results_file, "w") as f:
            json.dump(all_feasible_configs, f, indent=2)
        print(f"Saved to {results_file}")

    if len(all_feasible_configs) > 0:
        print("\nFeasible configurations summary:")
        for r in all_results:
            if r["feasible"]:
                print(f"  State {r['state_idx']+1}: {len(r['feasible_configs'])} feasible config(s)")

        print(f"\n{'='*60}")
        print(f"Viewing all {len(all_feasible_configs)} feasible configurations")
        print(f"{'='*60}")

        for idx, cfg in enumerate(all_feasible_configs):
            print(f"\nConfig {idx + 1}/{len(all_feasible_configs)}:")
            print(f"  State ID: {cfg['state_idx'] + 1}")
            print(f"  Joint Weight: {cfg['joint_weight']:.3e}")
            print(f"  Gripper Weight: {cfg['gripper_weight']:.3e}")
            print(f"  Distance Weight: {cfg['distance_weight']:.3e}")
            print(f"  Optimization Errors:")
            print(f"    EQ: {cfg['eq']:.3e}")
            print(f"    INEQ: {cfg['ineq']:.3e}")
            print(f"    SOS: {cfg['sos']:.3e}")

            q = np.array(cfg["q"])
            C.setJointState(q)
            C.view()
            
            print(f"  Distance constraints:")
            for robot_i, robot_j, target_dist in solver._group_distance_pairs:
                print(f"    {robot_i} - {robot_j}: {target_dist:.3e}, actual: {-C.eval(ry.FS.distance, [robot_i, robot_j])[0][0]:.3e}")
            pp.wait_for_user()
    else:
        print("\nNo feasible configurations found for any initial state!")
        C.view()
        pp.wait_for_user()
