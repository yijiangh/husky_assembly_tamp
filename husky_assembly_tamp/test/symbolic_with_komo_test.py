import numpy as np
import pybullet_planning as pp
import robotic as ry
import json
import os
from typing import List, Tuple
from scipy.spatial.transform import Rotation as R

# Import solver API
from husky_assembly_tamp.solver.komo_multi_frame_solver import (
    MultiPhaseKomoSolver,
    CylinderElement,
    GeometryCalculator,
    RobotPositionCalculator,
    create_horizontal_element,
    create_vertical_element
)
from husky_assembly_tamp.utils.collision import Element, init_pb
from husky_assembly_tamp.robot.robot import PathWithIndex, Robot
from husky_assembly_tamp.robot.robot_setup import (
    RobotSetup,
    HUSKY_URDF_PATH,
    HUSKY_INIT_ARM_JOINT_ANGLES,
    HUSKY_DUAL_URDF_PATH,
    HUSKY_DUAL_INIT_ARM_JOINT_ANGLES,
    HUSKY_ARM_JOINT_NAMES,
    HUSKY_DUAL_ARM_JOINT_NAMES_LEFT,
    HUSKY_DUAL_ARM_JOINT_NAMES_RIGHT,
)
from husky_assembly_tamp.symbolic_planner.planner import Planner
from husky_assembly_tamp.utils.util import CounterModule
from husky_assembly_tamp.utils.params import DATA_DIR

if __name__ == "__main__":
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
    
    C.view()
    
    # Initialize PyBullet for element bodies
    init_pb()
    
    # Collect all elements in order: vertical elements first (1, 2, 3), then horizontal elements (4, 5, 6)
    # Store element info with their vertex positions for axis_endpoints calculation
    elements_info = [
        (element_1, None, None),  # vertical, no vertex info needed
        (element_2, None, None),  # vertical, no vertex info needed
        (element_3, None, None),  # vertical, no vertex info needed
        (element_4, v2_pos, v3_pos),  # horizontal: v2_pos -> v3_pos
        (element_5, v3_pos, v1_pos),  # horizontal: v3_pos -> v1_pos
        (element_6, v1_pos, v2_pos),  # horizontal: v1_pos -> v2_pos
    ]
    
    # Create PyBullet bodies and element_from_index dictionary
    element_bodies = []
    goal_poses = []
    
    with pp.LockRenderer():
        for i, (element, v_start, v_end) in enumerate(elements_info):
            # Calculate axis endpoints for the element
            if element.direction is not None and v_start is not None and v_end is not None:
                # Horizontal element: use actual vertex positions, but at the z-height of the element
                # The element connects v_start and v_end at z = element.position[2]
                endpoint1 = np.array([v_start[0], v_start[1], element.position[2]])
                endpoint2 = np.array([v_end[0], v_end[1], element.position[2]])
                axis_endpoints = [endpoint1, endpoint2]
            else:
                # Vertical element: endpoints are top and bottom
                half_length = element.length / 2
                endpoint1 = element.position + np.array([0, 0, half_length])
                endpoint2 = element.position - np.array([0, 0, half_length])
                axis_endpoints = [endpoint1, endpoint2]
            
            # Create PyBullet cylinder body
            # Convert quaternion to euler for PyBullet
            quat = element.quaternion  # [w, x, y, z]
            # scipy expects [x, y, z, w] format
            rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            euler_angles = rot.as_euler('xyz', degrees=False)
            euler = pp.Euler(euler_angles[0], euler_angles[1], euler_angles[2])
            
            # Create cylinder body
            body = pp.create_cylinder(
                radius=element.radius,
                height=element.length,
                color=element.color
            )
            
            # Set initial pose (far away, similar to plan_generator.py)
            init_pose = pp.Pose(point=(5, 0, 0), euler=pp.Euler(0, 1.5708, 0))
            pp.set_pose(body, init_pose)
            
            # Set goal pose (current position and orientation)
            goal_pose = pp.Pose(point=element.position.tolist(), euler=euler)
            
            element_bodies.append(body)
            goal_poses.append(goal_pose)
    
    # Create element_from_index dictionary similar to plan_generator.py
    element_from_index = {}
    for i, ((element, v_start, v_end), body) in enumerate(zip(elements_info, element_bodies)):
        # Calculate axis endpoints for the element
        if element.direction is not None and v_start is not None and v_end is not None:
            # Horizontal element: use actual vertex positions, but at the z-height of the element
            endpoint1 = np.array([v_start[0], v_start[1], element.position[2]])
            endpoint2 = np.array([v_end[0], v_end[1], element.position[2]])
            axis_endpoints = [endpoint1, endpoint2]
        else:
            # Vertical element: endpoints are top and bottom
            half_length = element.length / 2
            endpoint1 = element.position + np.array([0, 0, half_length])
            endpoint2 = element.position - np.array([0, 0, half_length])
            axis_endpoints = [endpoint1, endpoint2]
        
        element_from_index[i] = Element(i, body, pp.get_pose(body), goal_poses[i], axis_endpoints)
    
    # Define contact pairs: which elements are in contact with each other
    # Based on the triangular structure:
    # - element_1 (0, vertical at v1) connects to element_4 (3) and element_6 (5)
    # - element_2 (1, vertical at v2) connects to element_4 (3) and element_5 (4)
    # - element_3 (2, vertical at v3) connects to element_5 (4) and element_6 (5)
    contact_id_pairs = [
        [0, 3],  # element_1 <-> element_4
        [1, 4],  # element_2 <-> element_5
        [2, 5],  # element_3 <-> element_6
        [3, 4],  # element_4 <-> element_5
        [3, 5],  # element_4 <-> element_6
        [4, 5],  # element_5 <-> element_6
    ]
    
    # Grounded elements: the three vertical columns (elements 0, 1, 2)
    grounded_elements_index = [0, 1, 2]
    
    # Set up robots for planning
    robot_types = ["husky_dual", "husky", "husky"]
    robot_names = ["r0", "r1", "fake"]
    robot_num = len(robot_types)
    path_storage = PathWithIndex()
    counter = CounterModule()
    
    with pp.HideOutput():
        robots = []
        for i, (robot_type, robot_name) in enumerate(zip(robot_types, robot_names)):
            # Load robot from URDF based on robot type
            if robot_type == "husky_dual":
                urdf_path = HUSKY_DUAL_URDF_PATH
                init_joint_angles = HUSKY_DUAL_INIT_ARM_JOINT_ANGLES
                joint_names_left = HUSKY_DUAL_ARM_JOINT_NAMES_LEFT
                joint_names_right = HUSKY_DUAL_ARM_JOINT_NAMES_RIGHT
            else:  # husky or other single arm types
                urdf_path = HUSKY_URDF_PATH
                init_joint_angles = HUSKY_INIT_ARM_JOINT_ANGLES
                joint_names = HUSKY_ARM_JOINT_NAMES
            
            robot_id = pp.load_model(urdf_path)
            
            # Set initial joint positions based on robot type
            if robot_type == "husky_dual":
                # Set left arm joints
                pp.set_joint_positions(
                    robot_id,
                    pp.joints_from_names(robot_id, joint_names_left),
                    init_joint_angles[:6]
                )
                # Set right arm joints
                pp.set_joint_positions(
                    robot_id,
                    pp.joints_from_names(robot_id, joint_names_right),
                    init_joint_angles[6:]
                )
            else:  # Single arm husky
                pp.set_joint_positions(
                    robot_id,
                    pp.joints_from_names(robot_id, joint_names),
                    init_joint_angles
                )
            
            # Collect obstacle IDs (all element bodies)
            obstacles = element_bodies.copy()
            
            # Create robot_data dictionary for direct loading
            robot_data = {
                "robot_id": robot_id,
                "obstacles": obstacles,
                "target_bar": None,  # No target bar initially
                "initial_attachments": [],
                "joint_values": init_joint_angles,
            }
            
            # Create RobotSetup with direct loading
            rb = RobotSetup(robot_name, robot_type=robot_type, robot_data=robot_data)
            robots.append(Robot(i, rb, element_from_index, counter, [], path_storage))
    
    # =========================================================================
    # Setup KOMO solver configuration (needed for planner validation)
    # =========================================================================
    print("\n" + "="*60)
    print("Setting up KOMO solver configuration for planner...")
    print("="*60)
    
    # Robot .g file paths for adding to ry.Config
    husky_dual_g_standalone = os.path.join(DATA_DIR, "husky_urdf/mt_husky_dual_ur5_e_moveit_config/urdf/husky_dual_ur5_e_no_base_joint_standalone.g")
    husky_g_standalone = os.path.join(DATA_DIR, "husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e_no_base_joint_standalone.g")
    
    # Add robots to ry.Config
    robot_base_frames = []
    robot_gripper_frames = []
    robot_base_frame_names = []
    
    for i, (robot_type, robot_name) in enumerate(zip(robot_types, robot_names)):
        if robot_name == "fake":
            continue  # Skip fake robots
        
        # Determine robot .g file path
        if robot_type == "husky_dual":
            robot_g_file = husky_dual_g_standalone
            prefix = f"{robot_name}_"
            # Dual arm has left and right grippers
            gripper_frames = [f"{robot_name}_right_ur_arm_tool0", f"{robot_name}_left_ur_arm_tool0"]
            base_frame_name = f"{robot_name}_base_footprint"
        else:  # husky single arm
            robot_g_file = husky_g_standalone
            prefix = f"{robot_name}_"
            gripper_frames = [f"{robot_name}_ur_arm_tool0"]
            base_frame_name = f"{robot_name}_base_footprint"
        
        # Add robot to config
        base_frame = RobotPositionCalculator.add_robot_to_config(
            C, robot_g_file, prefix, [0, 0, 0], [1, 0, 0, 0]
        )
        robot_base_frames.append(base_frame)
        robot_gripper_frames.append(gripper_frames)
        robot_base_frame_names.append(base_frame_name)
        
        # Set base joint for mobile base
        base_frame_obj = C.getFrame(base_frame_name)
        base_frame_obj.setJoint(ry.JT.transXYPhi, [-5, -5, -np.pi, 5, 5, np.pi])
    
    # Create planner and plan the symbolic sequence with solver validation
    planner = Planner(robot_num=robot_num, robots=robots)
    print("\n" + "="*60)
    print("Planning symbolic assembly sequence with solver validation...")
    print("="*60)
    path_index, keyframes_list = planner.Plan(
        element_from_index, 
        contact_id_pairs, 
        grounded_elements_index,
        config=C,
        robot_gripper_frames=robot_gripper_frames,
        robot_base_frame_names=robot_base_frame_names,
    )
    
    # Get element objects for visualization
    element_object_list = Planner.GetElementObjects(element_from_index, contact_id_pairs, grounded_elements_index)
    
    # Print the planned sequence
    print("\n" + "="*60)
    print("Planned Assembly Sequence:")
    print("="*60)
    for step_num, index_list in enumerate(path_index):
        element_names = [f"element_{i+1}" for i in index_list]
        keyframe_status = "✓ Solved" if keyframes_list[step_num] is not None else "✗ Not solved"
        print(f"Step {step_num + 1}: Assemble {index_list} ({', '.join(element_names)}) - {keyframe_status}")
    print("="*60 + "\n")
    
    # =========================================================================
    # Use saved keyframes from planner (no need to solve again)
    # =========================================================================
    print("\n" + "="*60)
    print("Using saved keyframes from planner...")
    print("="*60)
    
    # Get all joint names
    all_joint_names = C.getJointNames()
    arm_joint_indices = [
        list(range(3, 9)),  # Robot 1 arm joints
        list(range(9, 15)),  # Robot 1 fake arm joints
        list(range(18, 24)),  # Robot 2 arm joints
    ]
    robot_1_joint_indices = list(range(0, 9))
    robot_1_fake_joint_indices = list(range(9, 15))
    robot_2_joint_indices = list(range(15, 24))
    robot_joint_indices = [robot_1_joint_indices, robot_1_fake_joint_indices, robot_2_joint_indices]
    robot_joint_indices_real = [robot_1_joint_indices + robot_1_fake_joint_indices, robot_2_joint_indices]
    
    element_dict = {
        "element_1": element_1,
        "element_2": element_2,
        "element_3": element_3,
        "element_4": element_4,
        "element_5": element_5,
        "element_6": element_6,
    }
    
    def build_base_pose_path(poses_by_phase: List[List[Tuple[np.ndarray, List[float]]]], template_state: np.ndarray, robot_joint_indices: List[List[int]]) -> np.ndarray:
        """Build a per-phase joint state path that encodes phase-specific base poses."""
        path = np.tile(template_state, (len(poses_by_phase), 1))
        for phase_idx, phase_poses in enumerate(poses_by_phase):
            for robot_idx, (pos, quat) in enumerate(phase_poses):
                yaw = 2 * np.arctan2(quat[3], quat[0])  # Recover yaw from [w, x, y, z]
                path[phase_idx, robot_joint_indices[robot_idx][:3]] = [pos[0], pos[1], yaw]
        return path
    
    def arrange_frames_for_step(
        element_indices: List[int],
        robot_gripper_frames: List[List[str]],
        robot_base_frame_names: List[str]
    ) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
        """
        Arrange elements into frames based on the number of valid robots.
        
        Args:
            element_indices: List of element indices to arrange
            robot_gripper_frames: List of gripper frame names for each robot
            robot_base_frame_names: List of base frame names for each robot
        
        Returns:
            Tuple of (robot_names_phases, target_names_phases, baselink_names_phases)
        """
        num_elements = len(element_indices)
        
        if num_elements == 0:
            return [], [], []
        
        if num_elements == 1:
            # Single element: single frame (single robot grasps single element)
            print(f"  Single element step: using single frame solver")
            
            element_idx = element_indices[0]
            element_name = f"element_{element_idx + 1}"
            
            # Use first available robot
            if len(robot_gripper_frames) == 0:
                print(f"  Warning: No robots available, skipping...")
                return [], [], []
            
            robot_idx = 1
            grippers = robot_gripper_frames[robot_idx]
            
            # For dual-arm robots, use both left and right tools for the same element
            if len(grippers) > 1:
                # Dual-arm: use both arms
                robot_gripper_names = grippers  # [right_ur_arm_tool0, left_ur_arm_tool0]
                baselink_names = [robot_base_frame_names[robot_idx], robot_base_frame_names[robot_idx]]
            else:
                # Single-arm: use the only gripper
                robot_gripper_names = [grippers[0]]
                baselink_names = [robot_base_frame_names[robot_idx]]
            
            # Single phase: robot(s) grasp one element
            robot_names_phases = [robot_gripper_names]
            target_names_phases = [[element_name] * len(robot_gripper_names)]
            baselink_names_phases = [baselink_names]
            
            print(f"  Robot(s): {robot_gripper_names}")
            print(f"  Target: {element_name}")
            
            return robot_names_phases, target_names_phases, baselink_names_phases
        
        # Multiple elements: arrange into frames
        print(f"  Multiple elements step ({num_elements} elements): arranging into frames")
        
        # Check the number of valid robots
        robot_num = len(robot_gripper_frames)
        
        # Identify single-arm and dual-arm robots
        single_arm_robot_indices = []
        dual_arm_robot_indices = []
        for i, grippers in enumerate(robot_gripper_frames):
            if len(grippers) == 1:
                single_arm_robot_indices.append(i)
            else:
                dual_arm_robot_indices.append(i)
        
        dual_robot_num = len(dual_arm_robot_indices)
        
        # Check if robot_num >= len(element_indices): arrange as single frame
        if robot_num >= num_elements:
            # Enough robots: one robot per element, single frame
            print(f"  Enough robots ({robot_num}): using single frame with all robots")
            
            robot_names_phases = [[]]
            target_names_phases = [[]]
            baselink_names_phases = [[]]
            
            for i, element_idx in enumerate(element_indices):
                element_name = f"element_{element_idx + 1}"
                grippers = robot_gripper_frames[i]
                
                # For dual-arm robots, use both left and right tools for the same element
                if len(grippers) > 1:
                    # Dual-arm: use both arms
                    robot_gripper_names = grippers  # [right_ur_arm_tool0, left_ur_arm_tool0]
                    robot_names_phases[0].extend(robot_gripper_names)
                    target_names_phases[0].extend([element_name, element_name])
                    baselink_names_phases[0].extend([robot_base_frame_names[i], robot_base_frame_names[i]])
                    print(f"  Frame 1: Robot {robot_gripper_names[0]} & {robot_gripper_names[1]} -> {element_name}")
                else:
                    # Single-arm: use the only gripper
                    robot_gripper_name = grippers[0]
                    robot_names_phases[0].append(robot_gripper_name)
                    target_names_phases[0].append(element_name)
                    baselink_names_phases[0].append(robot_base_frame_names[i])
                    print(f"  Frame 1: Robot {robot_gripper_name} -> {element_name}")
            
            return robot_names_phases, target_names_phases, baselink_names_phases
        
        # Check if robot_num < len(element_indices) and dual_robot_num + robot_num >= len(element_indices)
        # Arrange as dual-frames task and keep the conf of all single-arm robots
        if robot_num < num_elements and dual_robot_num + robot_num >= num_elements:
            print(f"  More elements ({num_elements}) than robots ({robot_num}), but dual_robot_num ({dual_robot_num}) + robot_num ({robot_num}) >= num_elements: arranging as dual-frames task")
            
            # Calculate number of frames needed (at least 2 frames)
            num_frames = 2
            
            # Determine which elements will be held unchanged (by single-arm robots)
            # Use single-arm robots for elements that need to persist across frames
            num_unchanged_elements = min(len(single_arm_robot_indices), num_elements - robot_num)
            unchanged_element_indices = element_indices[:num_unchanged_elements] if num_unchanged_elements > 0 else []
            new_element_indices = element_indices[num_unchanged_elements:] if num_unchanged_elements > 0 else element_indices
            
            robot_names_phases = []
            target_names_phases = []
            baselink_names_phases = []
            
            # Distribute new elements across frames
            elements_per_frame = len(new_element_indices) // num_frames if num_frames > 0 else 0
            remaining_elements = len(new_element_indices) % num_frames if num_frames > 0 else 0
            
            new_element_idx = 0
            for frame_idx in range(num_frames):
                frame_robots = []
                frame_targets = []
                frame_baselinks = []
                
                # First, assign unchanged elements (held by single-arm robots in all frames)
                # Keep the conf of all single-arm robots
                for unchanged_idx, unchanged_element_idx in enumerate(unchanged_element_indices):
                    if unchanged_idx < len(single_arm_robot_indices):
                        robot_idx = single_arm_robot_indices[unchanged_idx]
                        element_name = f"element_{unchanged_element_idx + 1}"
                        grippers = robot_gripper_frames[robot_idx]
                        
                        # Single-arm robot: use the only gripper
                        robot_gripper_name = grippers[0]
                        
                        frame_robots.append(robot_gripper_name)
                        frame_targets.append(element_name)
                        frame_baselinks.append(robot_base_frame_names[robot_idx])
                
                # Then, assign new elements for this frame
                num_new_elements_this_frame = elements_per_frame + (1 if frame_idx < remaining_elements else 0)
                
                # Use remaining robots (dual-arm or unused single-arm) for new elements
                available_robot_indices = list(range(len(robot_gripper_frames)))
                # Remove robots already used for unchanged elements
                for unchanged_idx in range(len(unchanged_element_indices)):
                    if unchanged_idx < len(single_arm_robot_indices):
                        if single_arm_robot_indices[unchanged_idx] in available_robot_indices:
                            available_robot_indices.remove(single_arm_robot_indices[unchanged_idx])
                
                # Assign new elements to available robots
                for _ in range(num_new_elements_this_frame):
                    if new_element_idx >= len(new_element_indices) or len(available_robot_indices) == 0:
                        break
                    
                    element_idx = new_element_indices[new_element_idx]
                    element_name = f"element_{element_idx + 1}"
                    robot_idx = available_robot_indices.pop(0)
                    grippers = robot_gripper_frames[robot_idx]
                    
                    # For dual-arm robots, use both left and right tools for the same element
                    if len(grippers) > 1:
                        # Dual-arm: use both arms
                        robot_gripper_names = grippers  # [right_ur_arm_tool0, left_ur_arm_tool0]
                        frame_robots.extend(robot_gripper_names)
                        frame_targets.extend([element_name, element_name])
                        frame_baselinks.extend([robot_base_frame_names[robot_idx], robot_base_frame_names[robot_idx]])
                    else:
                        # Single-arm: use the only gripper
                        robot_gripper_name = grippers[0]
                        frame_robots.append(robot_gripper_name)
                        frame_targets.append(element_name)
                        frame_baselinks.append(robot_base_frame_names[robot_idx])
                    
                    new_element_idx += 1
                
                if len(frame_robots) > 0:
                    robot_names_phases.append(frame_robots)
                    target_names_phases.append(frame_targets)
                    baselink_names_phases.append(frame_baselinks)
                    
                    print(f"  Frame {frame_idx + 1}:")
                    for r, t in zip(frame_robots, frame_targets):
                        print(f"    {r} -> {t}")
            
            return robot_names_phases, target_names_phases, baselink_names_phases
        
        # Fallback: More elements than robots, arrange into multiple frames
        # Strategy: Use single-arm robots to hold unchanged elements across frames
        print(f"  More elements ({num_elements}) than robots ({robot_num}): arranging into multiple frames")
        
        # Calculate number of frames needed
        # Each frame can handle num_robots elements
        num_frames = (num_elements + robot_num - 1) // robot_num  # Ceiling division
        if num_frames < 2:
            num_frames = 2  # At least 2 frames
        
        # Determine which elements will be held unchanged (by single-arm robots)
        # Use single-arm robots for elements that need to persist across frames
        num_unchanged_elements = min(len(single_arm_robot_indices), num_elements - robot_num)
        unchanged_element_indices = element_indices[:num_unchanged_elements] if num_unchanged_elements > 0 else []
        new_element_indices = element_indices[num_unchanged_elements:] if num_unchanged_elements > 0 else element_indices
        
        robot_names_phases = []
        target_names_phases = []
        baselink_names_phases = []
        
        # Distribute new elements across frames
        elements_per_frame = len(new_element_indices) // num_frames if num_frames > 0 else 0
        remaining_elements = len(new_element_indices) % num_frames if num_frames > 0 else 0
        
        new_element_idx = 0
        for frame_idx in range(num_frames):
            frame_robots = []
            frame_targets = []
            frame_baselinks = []
            
            # First, assign unchanged elements (held by single-arm robots in all frames)
            for unchanged_idx, unchanged_element_idx in enumerate(unchanged_element_indices):
                if unchanged_idx < len(single_arm_robot_indices):
                    robot_idx = single_arm_robot_indices[unchanged_idx]
                    element_name = f"element_{unchanged_element_idx + 1}"
                    grippers = robot_gripper_frames[robot_idx]
                    
                    # Single-arm robot: use the only gripper
                    robot_gripper_name = grippers[0]
                    
                    frame_robots.append(robot_gripper_name)
                    frame_targets.append(element_name)
                    frame_baselinks.append(robot_base_frame_names[robot_idx])
            
            # Then, assign new elements for this frame
            num_new_elements_this_frame = elements_per_frame + (1 if frame_idx < remaining_elements else 0)
            
            # Use remaining robots (dual-arm or unused single-arm) for new elements
            available_robot_indices = list(range(len(robot_gripper_frames)))
            # Remove robots already used for unchanged elements
            for unchanged_idx in range(len(unchanged_element_indices)):
                if unchanged_idx < len(single_arm_robot_indices):
                    if single_arm_robot_indices[unchanged_idx] in available_robot_indices:
                        available_robot_indices.remove(single_arm_robot_indices[unchanged_idx])
            
            # Assign new elements to available robots
            for _ in range(num_new_elements_this_frame):
                if new_element_idx >= len(new_element_indices) or len(available_robot_indices) == 0:
                    break
                
                element_idx = new_element_indices[new_element_idx]
                element_name = f"element_{element_idx + 1}"
                robot_idx = available_robot_indices.pop(0)
                grippers = robot_gripper_frames[robot_idx]
                
                # For dual-arm robots, use both left and right tools for the same element
                if len(grippers) > 1:
                    # Dual-arm: use both arms
                    robot_gripper_names = grippers  # [right_ur_arm_tool0, left_ur_arm_tool0]
                    frame_robots.extend(robot_gripper_names)
                    frame_targets.extend([element_name, element_name])
                    frame_baselinks.extend([robot_base_frame_names[robot_idx], robot_base_frame_names[robot_idx]])
                else:
                    # Single-arm: use the only gripper
                    robot_gripper_name = grippers[0]
                    frame_robots.append(robot_gripper_name)
                    frame_targets.append(element_name)
                    frame_baselinks.append(robot_base_frame_names[robot_idx])
                
                new_element_idx += 1
            
            if len(frame_robots) > 0:
                robot_names_phases.append(frame_robots)
                target_names_phases.append(frame_targets)
                baselink_names_phases.append(frame_baselinks)
                
                print(f"  Frame {frame_idx + 1}:")
                for r, t in zip(frame_robots, frame_targets):
                    print(f"    {r} -> {t}")
        
        return robot_names_phases, target_names_phases, baselink_names_phases
    
    # Use saved keyframes from planner
    print(f"\nUsing saved keyframes for {len(path_index)} steps...")
    all_keyframes = []
    all_baselink_names_phases = []  # Track which baselinks were used for each step
    
    for step_num, element_indices in enumerate(path_index):
        print(f"\n--- Step {step_num + 1}: Assembling elements {element_indices} ---")
        
        num_elements = len(element_indices)
        
        if num_elements == 0:
            print(f"  Warning: No elements in step {step_num + 1}, skipping...")
            all_keyframes.append(None)
            all_baselink_names_phases.append(None)
            continue
        
        # Get saved keyframes for this step
        keyframes = keyframes_list[step_num]
        
        # Get baselink names for visualization (needed for later visualization)
        robot_names_phases, target_names_phases, baselink_names_phases = arrange_frames_for_step(
            element_indices,
            robot_gripper_frames,
            robot_base_frame_names
        )
        
        if keyframes is not None and len(keyframes) > 0:
            print(f"  ✓ Using saved keyframes ({len(keyframes)} phase(s))")
            all_keyframes.extend(keyframes)
            all_baselink_names_phases.extend([baselink_names_phases] * len(keyframes))
            
            # Visualize solution
            if num_elements == 1:
                # Single frame: show the one solution
                C.setJointState(keyframes[0])
                C.view()
                print(f"  Press Enter to continue to next step...")
                # pp.wait_for_user()
            else:
                # Multi-phase: show each phase
                for phase_idx, keyframe in enumerate(keyframes):
                    print(f"  Showing phase {phase_idx + 1}/{len(keyframes)}...")
                    C.setJointState(keyframe)
                    C.view()
                    # if phase_idx < len(keyframes) - 1:
                    #     pp.wait_for_user()
                print(f"  Press Enter to continue to next step...")
                # pp.wait_for_user()
        else:
            print(f"  ✗ No saved keyframes for this step")
            all_keyframes.append(None)
            all_baselink_names_phases.append(None)
    
    print(f"\n{'='*60}")
    print("KOMO Solver Summary:")
    print(f"{'='*60}")
    feasible_count = sum(1 for kf in all_keyframes if kf is not None)
    print(f"Successfully solved: {feasible_count}/{len(path_index)} steps")
    print(f"{'='*60}\n")
    
    # **************************************************************************
    # Visualize the solution
    # **************************************************************************
    dual_arm_baselink_name = "r0_base_footprint"  # Dual-arm robot baselink name
    far_away_position = np.array([10.0, 10.0, 0.0])  # Position far away from the scene
    
    # Get joint indices for dual-arm robot base (first robot: indices 0-2 for base)
    # Based on robot_joint_indices_real, r0 is the first robot (indices 0-14)
    # Base joints are the first 3: [x, y, yaw]
    dual_arm_base_joint_indices = [0, 1, 2]  # First robot's base joints
    
    for step_idx, (keyframe, baselink_names_phases) in enumerate(zip(all_keyframes, all_baselink_names_phases)):
        if keyframe is None:
            continue  # Skip failed solutions
        
        # Create a copy of the keyframe to modify
        modified_keyframe = keyframe.copy()
        
        # Check if only a single robot is used
        if baselink_names_phases is not None:
            # Flatten baselink names from all phases and get unique baselinks
            used_baselinks = set()
            for phase_baselinks in baselink_names_phases:
                if isinstance(phase_baselinks, list):
                    used_baselinks.update(phase_baselinks)
                else:
                    used_baselinks.add(phase_baselinks)
            
            # Check if only one unique baselink is used and dual-arm robot is not used
            if len(used_baselinks) == 1 and dual_arm_baselink_name not in used_baselinks:
                # Modify the keyframe to move dual-arm robot far away
                if len(modified_keyframe) > max(dual_arm_base_joint_indices):
                    # Set base position (x, y) and yaw to far away
                    modified_keyframe[dual_arm_base_joint_indices[0]] = far_away_position[0]
                    modified_keyframe[dual_arm_base_joint_indices[1]] = far_away_position[1]
                    modified_keyframe[dual_arm_base_joint_indices[2]] = 0.0  # yaw = 0
                    print(f"  Step {step_idx + 1}: Moved dual-arm robot ({dual_arm_baselink_name}) far away")
        
        print(f"  Step {step_idx + 1}: Modified keyframe: {modified_keyframe.shape}")
        # Set the modified keyframe state
        C.setJointState(modified_keyframe)
        C.view()
        pp.wait_for_user()