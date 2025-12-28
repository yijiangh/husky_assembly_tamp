import numpy as np
import pybullet_planning as pp
import robotic as ry
import json
import os
from typing import List, Tuple

# Import solver API
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.komo_multi_frame_solver import (
    MultiPhaseKomoSolver,
    CylinderElement,
    GeometryCalculator,
    RobotPositionCalculator,
    ConstraintManager,
    create_horizontal_element,
    create_vertical_element,
)


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
    # Phase 1: r1 face element_4, r2 face element_5
    robot_base_poses_phase1 = [
        RobotPositionCalculator.calculate_pose_toward_target(element_4.position, element_4.direction, ROBOT_DISTANCE, element_4.position),
        RobotPositionCalculator.calculate_pose_toward_target(element_5.position, element_5.direction, ROBOT_DISTANCE, element_5.position),
    ]

    # Phase 2: r1 face element_6, r2 face element_5
    robot_base_poses_phase2 = [
        RobotPositionCalculator.calculate_pose_toward_target(element_6.position, element_6.direction, ROBOT_DISTANCE, element_6.position),
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
    # r1_base_frame = RobotPositionCalculator.add_robot_to_config(C, ry.raiPath("panda/panda.g"), "rpp_", [0, 0, 0], [1, 0, 0, 0])
    # r2_base_frame = RobotPositionCalculator.add_robot_to_config(C, ry.raiPath("panda/panda.g"), "r2_", [0, 0, 0], [1, 0, 0, 0])
    # r3_base_frame = RobotPositionCalculator.add_robot_to_config(C, ry.raiPath("panda/panda.g"), "r3_", [0, 0, 0], [1, 0, 0, 0])

    husky_dual_urdf = "/home/jeong/summer_research/husky_assembly/ext/husky-assembly-teleop/data/husky_urdf/mt_husky_dual_ur5_e_moveit_config/urdf/husky_dual_ur5_e_no_base_joint.urdf"
    husky_dual_g_standalone = "/home/jeong/summer_research/husky_assembly/ext/husky-assembly-teleop/data/husky_urdf/mt_husky_dual_ur5_e_moveit_config/urdf/husky_dual_ur5_e_no_base_joint_standalone.g"
    husky_urdf = "/home/jeong/summer_research/husky_assembly/ext/husky-assembly-teleop/data/husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e_no_base_joint.urdf"
    husky_g_standalone = "/home/jeong/summer_research/husky_assembly/ext/husky-assembly-teleop/data/husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e_no_base_joint_standalone.g"
    r1_base_frame = RobotPositionCalculator.add_robot_to_config(C, husky_dual_g_standalone, "r1_", [0, 0, 0], [1, 0, 0, 0])
    r2_base_frame = RobotPositionCalculator.add_robot_to_config(C, husky_g_standalone, "r2_", [0, 0, 0], [1, 0, 0, 0])

    # # base2 = C.getFrame("r2_panda_link0")
    # # base3 = C.getFrame("r3_panda_link0")
    # # base1.setJoint(ry.JT.transXY, [-0.25, -0.25, 0.25, 0.25])
    # # base2.setJoint(ry.JT.transXY, [-0.25, -0.25, 0.25, 0.25])
    # # base3.setJoint(ry.JT.transXY, [-0.25, -0.25, 0.25, 0.25])
    # base1.setJoint(ry.JT.transXYPhi, [-5, -5, -np.pi, 5, 5, np.pi])
    # base2.setJoint(ry.JT.transXYPhi, [-5, -5, -np.pi, 5, 5, np.pi])
    # base3.setJoint(ry.JT.transXYPhi, [-5, -5, -np.pi, 5, 5, np.pi])
    # base_frame_names = ["r1_panda_link0", "r2_panda_link0", "r3_panda_link0"]
    # initial_base_positions = [base1.getPosition(), base2.getPosition(), base3.getPosition()]
    # initial_base_quaternions = [base1.getQuaternion(), base2.getQuaternion(), base3.getQuaternion()]

    base_r1 = C.getFrame("r1_base_footprint")
    base_r2 = C.getFrame("r2_base_footprint")
    base_frame_names = ["r1_base_footprint", "r2_base_footprint"]
    base_r1.setJoint(ry.JT.transXYPhi, [-5, -5, -np.pi, 5, 5, np.pi])
    base_r2.setJoint(ry.JT.transXYPhi, [-5, -5, -np.pi, 5, 5, np.pi])
    initial_base_positions = [base_r1.getPosition(), base_r2.getPosition()]
    initial_base_quaternions = [base_r1.getQuaternion(), base_r2.getQuaternion()]

    all_joint_names = C.getJointNames()
    base_joint_indices = []
    for i, name in enumerate(all_joint_names):
        if any(base_name in name for base_name in base_frame_names):
            base_joint_indices.append(i)

    # C.view()
    # print(C.getFrameNames())
    # # print(C.getJointDimension())
    # q = [0] * 24
    # q[0] = 1
    # q[15] = 1.5
    # C.setJointState(q)
    # C.view()
    # pp.wait_for_user("fuck you here")

    def draw_pose(config, frame_name, pose_name_prefix, length=0.1):
        """Draw pose using marker frames"""
        frame = config.getFrame(frame_name)
        pos = frame.getPosition()
        quat = frame.getQuaternion()

        config.addFrame(f"{pose_name_prefix}_marker").setShape(ry.ST.marker, [length]).setPosition(pos).setQuaternion(quat).setColor([1, 1, 0])

    # Ensure the config reflects the Phase 1 base poses before capturing home
    RobotPositionCalculator.apply_base_poses(C, base_frame_names, robot_base_poses_phase1)

    C.view()
    # pp.wait_for_user("fuck you here")

    x_home = C.getJointState()
    x_home[4] = -np.pi / 2 - np.pi / 4
    x_home[10] = -np.pi / 2 + np.pi / 4
    x_home[19] = -np.pi / 2

    def build_base_pose_path(poses_by_phase: List[List[Tuple[np.ndarray, List[float]]]], template_state: np.ndarray, robot_joint_indices: List[List[int]]) -> np.ndarray:
        """Build a per-phase joint state path that encodes phase-specific base poses."""
        path = np.tile(template_state, (len(poses_by_phase), 1))
        for phase_idx, phase_poses in enumerate(poses_by_phase):
            for robot_idx, (pos, quat) in enumerate(phase_poses):
                yaw = 2 * np.arctan2(quat[3], quat[0])  # Recover yaw from [w, x, y, z]
                path[phase_idx, robot_joint_indices[robot_idx][:3]] = [pos[0], pos[1], yaw]
        return path

    # Solver configuration parameters (shared)
    # joint_weight = 0.1  # bak
    # gripper_weight = 5.11  # bak
    # position_rel_z_bounds = (0.45, -0.45)
    # constraint_eps = 1e-3
    # freeze_arm_joints = True
    # collision_weight = 10
    # pose_rel_weight = 10
    # enable_constraint_verification = True

    joint_weight = 0.1
    gripper_weight = 5.11  # bak
    position_rel_z_bounds = (0.45, -0.45)
    constraint_eps = 1e-3
    freeze_arm_joints = True
    collision_weight = 1
    pose_rel_weight = 0
    baselink_distance_weight = 1.0
    baselink_distance_target = 1.5
    enable_constraint_verification = False

    # Arm joint indices for freezing (assuming 10 DOF per robot: 3 base + 7 arm)
    arm_joint_indices = [
        list(range(3, 9)),  # Robot 1 arm joints
        list(range(9, 15)),  # Robot 1 fake arm joints
        list(range(18, 24)),  # Robot 2 arm joints
        # list(range(2 * 10 + 3, 2 * 10 + 3 + 7)),  # Robot 3 arm joints
    ]

    # Joint indices for each robot (base + arm = 10 DOF each)
    robot_1_joint_indices = list(range(0, 9))
    robot_1_fake_joint_indices = list(range(9, 15))
    robot_2_joint_indices = list(range(15, 24))
    robot_joint_indices = [robot_1_joint_indices, robot_1_fake_joint_indices, robot_2_joint_indices]
    robot_joint_indices_real = [robot_1_joint_indices + robot_1_fake_joint_indices, robot_2_joint_indices]
    # robot_3_joint_indices = list(range(2 * 10, 2 * 10 + 10))

    # Directed poseRel constraints (frame_i in frame_j coordinates)
    # pose_rel_constraints = [
    #     ("r2_panda_link0", "r1_panda_link0", [0.0, 0.5, 0.0, 1, 0, 0, 0]),
    # ]

    # =========================================================================
    # Multi-Phase Setup: Two phases with different r3 targets
    # Phase 1: r1 -> element_4, r2 -> element_5
    # Phase 2: r1 -> element_6, r2 -> element_5
    # =========================================================================
    # robot_names_phase1 = [["r1_gripper", "r2_gripper"], "r3_gripper"]
    # target_names_phase1 = [["element_4", "element_4"], "element_5"]
    robot_names_phase1 = ["r1_right_ur_arm_tool0", "r1_left_ur_arm_tool0", "r2_ur_arm_tool0"]
    target_names_phase1 = ["element_4", "element_4", "element_5"]
    baselink_names_phase1 = ["r1_base_footprint", "r1_base_footprint", "r2_base_footprint"]

    robot_names_phase2 = ["r1_right_ur_arm_tool0", "r1_left_ur_arm_tool0", "r2_ur_arm_tool0"]
    target_names_phase2 = ["element_6", "element_6", "element_5"]
    baselink_names_phase2 = ["r1_base_footprint", "r1_base_footprint", "r2_base_footprint"]

    # Combine into multi-phase format
    robot_names_phases = [robot_names_phase1, robot_names_phase2]
    target_names_phases = [target_names_phase1, target_names_phase2]
    baselink_names_phases = [baselink_names_phase1, baselink_names_phase2]
    
    phase_switch_robots = ["r2"]
    phase_switch_weight = 1.0

    solver = MultiPhaseKomoSolver(
        config=C,
        robot_names_phases=robot_names_phases,
        target_names_phases=target_names_phases,
        joint_weight=joint_weight,
        gripper_weight=gripper_weight,
        position_rel_z_bounds=position_rel_z_bounds,
        constraint_eps=constraint_eps,
        freeze_arm_joints=freeze_arm_joints,
        x_home=x_home,
        arm_joint_indices=arm_joint_indices,
        collision_weight=collision_weight,
        # pose_rel_constraints=pose_rel_constraints,
        pose_rel_weight=pose_rel_weight,
        enable_constraint_verification=enable_constraint_verification,
        baselink_distance_weight=baselink_distance_weight,
        baselink_distance_target=baselink_distance_target,
        baselink_names_phases=baselink_names_phases,
        phase_switch_robots=phase_switch_robots,
        phase_switch_weight=phase_switch_weight,
    )

    num_initial_states = 10
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
    base_pose_path_template = build_base_pose_path(robot_base_poses_phases, x_home, robot_joint_indices)

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

            # print(f"  PoseRel constraints (positionRel + scalarProductXX):")
            # for frame_i, frame_j, target_pose in solver.pose_rel_constraints:
            #     actual_pos = C.eval(ry.FS.positionRel, [frame_i, frame_j])[0]
            #     actual_align = C.eval(ry.FS.scalarProductXX, [frame_i, frame_j])[0][0]
            #     print(f"    {frame_i} - {frame_j}: target_pos={target_pose[:3]}, actual_pos={actual_pos.tolist()}, alignXX={actual_align:.3f}")

            pp.wait_for_user()

            # Show Phase 2 result
            print(f"\n--- Phase 2 Result (r3 -> element_6) ---")
            q_phase2 = np.array(result["q_phase2"])
            C.setJointState(q_phase2)
            C.view()

            # print(f"  PoseRel constraints (positionRel + scalarProductXX):")
            # for frame_i, frame_j, target_pose in solver.pose_rel_constraints:
            #     actual_pos = C.eval(ry.FS.positionRel, [frame_i, frame_j])[0]
            #     actual_align = C.eval(ry.FS.scalarProductXX, [frame_i, frame_j])[0][0]
            #     print(f"    {frame_i} - {frame_j}: target_pos={target_pose[:3]}, actual_pos={actual_pos.tolist()}, alignXX={actual_align:.3f}")

            pp.wait_for_user()
    else:
        print("\nNo feasible configurations found!")
        C.view()
        pp.wait_for_user()
