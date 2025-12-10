import numpy as np
import pybullet_planning as pp
import robotic as ry
import json
import os
from typing import List, Tuple, Optional
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


@dataclass
class CylinderElement:
    """Represents a cylindrical element with position, orientation, and geometric properties."""

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
        """Add this element to a ry.Config."""
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
        """Create a vertical cylinder element (aligned with Z-axis)."""
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
        """Create a horizontal cylinder element aligned with the given direction."""
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
        """Get the endpoint of the cylinder that is closer to the target position."""
        if self.direction is None:
            raise ValueError("Direction not set for this element")

        dir_norm = normalize_vector(self.direction)
        half_length = self.length / 2

        dir_to_target = np.array([target_pos[0] - self.position[0], target_pos[1] - self.position[1], 0])
        dir_to_target_norm = normalize_vector(dir_to_target, dir_norm)
        dot = np.dot(dir_norm, dir_to_target_norm)

        return self.position + dir_norm * half_length * (1 if dot > 0 else -1)

    def get_other_end_position(self, target_pos: np.ndarray) -> np.ndarray:
        """Get the endpoint of the cylinder that is farther from the target position."""
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
        """Calculate position and direction for a horizontal element."""
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
        """Calculate robot position perpendicular to an element's edge direction."""
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
        """Add a robot to the configuration."""
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
    """Create a horizontal cylinder element and add it to the config."""
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
    """Create a vertical cylinder element and add it to the config."""
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


def setup_environment(config: ry.Config):
    """Setup the environment with elements and robots, return element objects."""
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

    # Create horizontal elements (beams)
    element_4 = create_horizontal_element(config, "element_4", v2_pos, v3_pos, v1_pos, HORIZONTAL_Z[0], [1, 1, 0], length=CYLINDER_LENGTH, radius=CYLINDER_RADIUS, protrusion_offset=PROTRUSION_OFFSET, contact=True)
    element_5 = create_horizontal_element(config, "element_5", v3_pos, v1_pos, v2_pos, HORIZONTAL_Z[1], [1, 0, 1], length=CYLINDER_LENGTH, radius=CYLINDER_RADIUS, protrusion_offset=PROTRUSION_OFFSET, contact=True)
    element_6 = create_horizontal_element(config, "element_6", v1_pos, v2_pos, v3_pos, HORIZONTAL_Z[2], [0, 1, 1], length=CYLINDER_LENGTH, radius=CYLINDER_RADIUS, protrusion_offset=PROTRUSION_OFFSET, contact=True)

    # Calculate element endpoints for vertical element positioning
    element_4_end = element_4.get_end_position(v1_pos)
    element_5_end = element_5.get_end_position(v2_pos)
    element_6_other_end = element_6.get_other_end_position(v3_pos)

    # Calculate vertical element positions
    v1_final = GeometryCalculator.calculate_vertical_element_position(element_4_end, v1_pos, VERTICAL_DISTANCE, VERTICAL_Z)
    v2_final = GeometryCalculator.calculate_vertical_element_position(element_5_end, v2_pos, VERTICAL_DISTANCE, VERTICAL_Z)
    v3_final = GeometryCalculator.calculate_vertical_element_position(element_6_other_end, v3_pos, VERTICAL_DISTANCE, VERTICAL_Z)

    # Create vertical elements (columns)
    element_1 = create_vertical_element(config, "element_1", v1_final, [1, 0, 0], length=CYLINDER_LENGTH, radius=CYLINDER_RADIUS, contact=True)
    element_2 = create_vertical_element(config, "element_2", v2_final, [0, 1, 0], length=CYLINDER_LENGTH, radius=CYLINDER_RADIUS, contact=True)
    element_3 = create_vertical_element(config, "element_3", v3_final, [0, 0, 1], length=CYLINDER_LENGTH, radius=CYLINDER_RADIUS, contact=True)

    # Calculate robot initial positions
    robot_1_pos = RobotPositionCalculator.calculate_position_perpendicular(element_4.position, element_4.direction, ROBOT_DISTANCE)
    robot_2_pos = RobotPositionCalculator.calculate_position_perpendicular(element_4.position, element_4.direction, ROBOT_DISTANCE + 0.5)
    robot_3_pos = RobotPositionCalculator.calculate_position_perpendicular(element_5.position, element_5.direction, ROBOT_DISTANCE)

    # Add robots to configuration
    RobotPositionCalculator.add_robot_to_config(config, ry.raiPath("panda/panda.g"), "r1_", robot_1_pos)
    RobotPositionCalculator.add_robot_to_config(config, ry.raiPath("panda/panda.g"), "r2_", robot_2_pos)
    RobotPositionCalculator.add_robot_to_config(config, ry.raiPath("panda/panda.g"), "r3_", robot_3_pos)

    # Set base joints
    base1 = config.getFrame("r1_panda_link0")
    base2 = config.getFrame("r2_panda_link0")
    base3 = config.getFrame("r3_panda_link0")
    base1.setJoint(ry.JT.transXYPhi, [-0.25, -0.25, -np.pi, 0.25, 0.25, np.pi])
    base2.setJoint(ry.JT.transXYPhi, [-0.25, -0.25, -np.pi, 0.25, 0.25, np.pi])
    base3.setJoint(ry.JT.transXYPhi, [-0.25, -0.25, -np.pi, 0.25, 0.25, np.pi])

    return {
        "element_4": element_4,
        "element_5": element_5,
        "element_6": element_6,
    }


def evaluate_constraints(config: ry.Config, robot_names: List[str], target_names: List[str], distance_pairs: List[Tuple[str, str, float]]):
    """Evaluate and print constraint information."""
    config.computeCollisions()
    
    print("  Gripper Constraints:")
    for robot_name, target_name in zip(robot_names, target_names):
        scalar_product_xy = config.eval(ry.FS.scalarProductXY, [target_name, robot_name])
        scalar_product_yy = config.eval(ry.FS.scalarProductYY, [target_name, robot_name])
        position_rel = config.eval(ry.FS.positionRel, [robot_name, target_name])
        
        val_xy = float(scalar_product_xy[0][0])
        val_yy = float(scalar_product_yy[0][0])
        pos_rel_z = float(position_rel[0][2])
        
        print(f"    {robot_name} -> {target_name}:")
        print(f"      scalarProductXY: {val_xy:.6f} (target: 0)")
        print(f"      scalarProductYY: {val_yy:.6f} (target: 0)")
        print(f"      positionRel_z: {pos_rel_z:.6f} (bounds: [-0.4, 0.4])")
    
    print("  Distance Constraints:")
    for frame_i, frame_j, target_dist in distance_pairs:
        dist_val = config.eval(ry.FS.distance, [frame_i, frame_j])
        actual_dist = float(dist_val[0][0])
        print(f"    {frame_i} - {frame_j}: target={target_dist:.3f}, actual={actual_dist:.3f}, error={abs(actual_dist - (-target_dist)):.6f}")
    
    # Collision check
    accumulated_collisions = config.eval(ry.FS.accumulatedCollisions, [])
    val_collisions = float(accumulated_collisions[0][0])
    print(f"  Accumulated Collisions: {val_collisions:.6f}")


def main():
    print("=" * 70)
    print("Multi-Arm KOMO Visualization")
    print("=" * 70)
    
    # Load converged configurations
    results_file = "/home/jeong/summer_research/komo_results/converged_iterative_configs.json"
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found!")
        return
    
    with open(results_file, "r") as f:
        all_converged = json.load(f)
    
    print(f"Loaded {len(all_converged)} converged configurations from {results_file}")
    
    if len(all_converged) == 0:
        print("No converged configurations to visualize!")
        return
    
    # Create two separate environments
    print("\nSetting up Phase 1 environment...")
    C1 = ry.Config()
    elements1 = setup_environment(C1)
    
    print("Setting up Phase 2 environment...")
    C2 = ry.Config()
    elements2 = setup_environment(C2)
    
    # Robot and target names for each phase
    robot_names_phase1 = ["r1_gripper", "r2_gripper", "r3_gripper"]
    target_names_phase1 = ["element_4", "element_4", "element_5"]
    
    robot_names_phase2 = ["r1_gripper", "r2_gripper", "r3_gripper"]
    target_names_phase2 = ["element_4", "element_4", "element_6"]
    
    # Distance constraint pairs
    distance_pairs = [("r1_panda_link0", "r2_panda_link0", 0.3)]
    
    # Joint indices for each robot
    robot_1_joint_indices = list(range(0 * 10, 0 * 10 + 10))
    robot_2_joint_indices = list(range(1 * 10, 1 * 10 + 10))
    robot_3_joint_indices = list(range(2 * 10, 2 * 10 + 10))
    
    print("\n" + "=" * 70)
    print("Starting visualization...")
    print("=" * 70)
    print("\nPress Enter to advance through each configuration.")
    print("Phase 1 window shows: r1, r2 -> element_4, r3 -> element_5")
    print("Phase 2 window shows: r1, r2 -> element_4, r3 -> element_6")
    
    pp.wait_for_user()
    
    # Visualize each converged configuration
    for idx, result in enumerate(all_converged):
        print("\n" + "=" * 70)
        print(f"Configuration {idx + 1}/{len(all_converged)}")
        print(f"  Original State Index: {result['state_idx'] + 1}")
        print(f"  Converged in: {result['num_iterations']} iteration(s)")
        print("=" * 70)
        
        q_final_phase1 = result.get("q_final_phase1")
        q_final_phase2 = result.get("q_final_phase2")
        
        if q_final_phase1 is None or q_final_phase2 is None:
            print("  Warning: Missing final configuration, skipping...")
            continue
        
        q_phase1 = np.array(q_final_phase1)
        q_phase2 = np.array(q_final_phase2)
        
        # Set joint states for both environments
        C1.setJointState(q_phase1)
        C2.setJointState(q_phase2)
        
        # Print robot base positions
        print("\n--- Robot Base Positions ---")
        for robot_idx, (r_name, indices) in enumerate([("r1", robot_1_joint_indices), ("r2", robot_2_joint_indices), ("r3", robot_3_joint_indices)]):
            base_x_p1 = q_phase1[indices[0]]
            base_y_p1 = q_phase1[indices[1]]
            base_phi_p1 = q_phase1[indices[2]]
            base_x_p2 = q_phase2[indices[0]]
            base_y_p2 = q_phase2[indices[1]]
            base_phi_p2 = q_phase2[indices[2]]
            print(f"  {r_name}: Phase1=({base_x_p1:.3f}, {base_y_p1:.3f}, φ={np.degrees(base_phi_p1):.1f}°), Phase2=({base_x_p2:.3f}, {base_y_p2:.3f}, φ={np.degrees(base_phi_p2):.1f}°)")
        
        # Print Phase 1 constraints
        print("\n--- Phase 1 Constraints (r3 -> element_5) ---")
        evaluate_constraints(C1, robot_names_phase1, target_names_phase1, distance_pairs)
        
        # Print Phase 2 constraints
        print("\n--- Phase 2 Constraints (r3 -> element_6) ---")
        evaluate_constraints(C2, robot_names_phase2, target_names_phase2, distance_pairs)
        
        # Open both views
        C1.view(False, f"Phase 1: Config {idx+1}/{len(all_converged)} - r3->element_5")
        C2.view(False, f"Phase 2: Config {idx+1}/{len(all_converged)} - r3->element_6")
        
        print("\n[Press Enter to continue to next configuration...]")
        pp.wait_for_user()
        
        # Option to view iteration history
        num_iterations = result["num_iterations"]
        q_history_phase1 = result.get("q_history_phase1", [])
        q_history_phase2 = result.get("q_history_phase2", [])
        
        if num_iterations > 1 and len(q_history_phase1) > 0 and len(q_history_phase2) > 0:
            print(f"\nView iteration history? ({num_iterations} iterations)")
            print("[Press Enter to view history, or type 'skip' and Enter to skip]")
            
            # For simplicity, always show history
            for iter_idx in range(min(len(q_history_phase1), len(q_history_phase2))):
                print(f"\n--- Iteration {iter_idx + 1}/{num_iterations} ---")
                
                q_p1 = np.array(q_history_phase1[iter_idx])
                q_p2 = np.array(q_history_phase2[iter_idx])
                
                C1.setJointState(q_p1)
                C2.setJointState(q_p2)
                
                # Print brief constraint summary
                C1.computeCollisions()
                C2.computeCollisions()
                
                coll1 = float(C1.eval(ry.FS.accumulatedCollisions, [])[0][0])
                coll2 = float(C2.eval(ry.FS.accumulatedCollisions, [])[0][0])
                
                dist_p1 = float(C1.eval(ry.FS.distance, ["r1_panda_link0", "r2_panda_link0"])[0][0])
                dist_p2 = float(C2.eval(ry.FS.distance, ["r1_panda_link0", "r2_panda_link0"])[0][0])
                
                print(f"  Phase 1: collisions={coll1:.4f}, r1-r2 dist={dist_p1:.4f}")
                print(f"  Phase 2: collisions={coll2:.4f}, r1-r2 dist={dist_p2:.4f}")
                
                C1.view(False, f"Phase 1: Iter {iter_idx+1}/{num_iterations}")
                C2.view(False, f"Phase 2: Iter {iter_idx+1}/{num_iterations}")
                
                pp.wait_for_user()
    
    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    
    # Final summary
    print(f"\nSummary:")
    print(f"  Total converged configurations: {len(all_converged)}")
    avg_iterations = np.mean([r["num_iterations"] for r in all_converged])
    print(f"  Average iterations to converge: {avg_iterations:.2f}")
    
    C1.view()
    C2.view()
    pp.wait_for_user()


if __name__ == "__main__":
    main()

