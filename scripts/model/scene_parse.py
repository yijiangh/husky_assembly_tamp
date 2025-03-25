import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple

import pybullet_planning as pp
from scipy.spatial.transform import Rotation

import numpy as np
import yaml

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from multi_tangent.collision import create_collision_bodies
from utils.collision import init_pb
from utils.params import *


class SceneParser:
    """
    A class for parsing and visualizing 3D scene data from YAML files.

    This class handles loading scene data, approximating geometric shapes with spheres,
    and visualizing the scene using PyBullet.
    """

    def __init__(self, scene_file: str):
        """
        Initialize the SceneParser with a scene file path.

        Args:
            scene_file (str): Path to the YAML scene file
        """
        self.scene_file = scene_file
        self.scene_data = None
        self.channels_info = []  # Store channel information
        self.robot_info = None  # Store robot information

    def load_scene(self):
        """
        Load and parse the scene data from the YAML file.
        Converts the raw data into a SimpleNamespace object for easier access.
        """
        with open(self.scene_file, "r") as file:
            raw_data = yaml.safe_load(file)
            self.scene_data = self._convert_to_namespace(raw_data)
            # Extract robot information
            if hasattr(self.scene_data, "robot"):
                self.robot_info = self.scene_data.robot
            else:
                raise ValueError("Robot information not found in scene file")

    def _convert_to_namespace(self, data: Dict) -> SimpleNamespace:
        """
        Recursively convert a dictionary to a SimpleNamespace object.

        Args:
            data (Dict): Dictionary data to convert

        Returns:
            SimpleNamespace: Converted data structure
        """
        if isinstance(data, dict):
            return SimpleNamespace(**{k: self._convert_to_namespace(v) for k, v in data.items()})
        elif isinstance(data, list):
            return [self._convert_to_namespace(item) for item in data]
        else:
            return data

    def approximate_elements_with_spheres(self) -> List[Dict]:
        """
        Approximate geometric elements in the scene with spheres.

        Returns:
            List[Dict]: List of approximated elements with sphere properties
        """
        if not self.scene_data:
            raise ValueError("Scene data is not loaded. Call load_scene() first.")

        approximated_elements = []
        for element in self.scene_data.elements:
            shape_type = element.shape.type
            if shape_type == "cylinder":
                approximated_elements.extend(self._approximate_cylinder(element))
            elif shape_type == "cuboid":
                approximated_elements.extend(self._approximate_cuboid(element))
            elif shape_type == "sphere":
                approximated_elements.append(element)  # No approximation needed
        return approximated_elements

    def _approximate_cylinder(self, element: SimpleNamespace) -> List[Dict]:
        """
        Approximate a cylinder with a series of spheres along its axis.

        Args:
            element (SimpleNamespace): Cylinder element to approximate

        Returns:
            List[Dict]: List of spheres approximating the cylinder
        """
        # Get cylinder center point and parameters
        center = np.array(element.position)
        height = element.shape.parameters.height
        radius = element.shape.parameters.radius
        orientation = element.orientation  # Assuming quaternion
        sphere_radius = element.sphere_fit.radius
        count = element.sphere_fit.count

        # Calculate rotated direction vector
        base_direction = np.array([0, 0, 1])  # Default cylinder axis
        rotation_matrix = self._quaternion_to_rotation_matrix(orientation)
        direction = rotation_matrix @ base_direction
        direction = direction / np.linalg.norm(direction)  # Normalize direction vector

        # Calculate cylinder start and end points (relative to center)
        half_height = height / 2
        start = center - direction * half_height
        end = center + direction * half_height

        # Generate sphere positions through linear interpolation
        spheres = []
        for i in range(count):
            t = i / (count - 1) if count > 1 else 0.5
            position = (1 - t) * start + t * end
            spheres.append({"id": f"{element.id}_sphere_{i+1}", "position": position.tolist(), "radius": sphere_radius})
        return spheres

    def _approximate_cuboid(self, element: SimpleNamespace) -> List[Dict]:
        """
        Approximate a cuboid with spheres (placeholder method).

        Args:
            element (SimpleNamespace): Cuboid element to approximate

        Returns:
            List[Dict]: List of spheres approximating the cuboid
        """
        # Placeholder for cuboid approximation
        return []

    def _quaternion_to_rotation_matrix(self, q: List[float]) -> np.ndarray:
        """
        Convert a quaternion to a 3x3 rotation matrix.

        Args:
            q (List[float]): Quaternion [x, y, z, w]

        Returns:
            np.ndarray: 3x3 rotation matrix
        """
        x, y, z, w = q
        return np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
            ]
        )

    def visualize_scene(self):
        """
        Visualize the scene using PyBullet.
        Creates collision bodies for cylinders and spheres, and visualizes channels.
        """
        # Initialize PyBullet
        init_pb()

        # Prepare line data for create_collision_bodies
        line_pts_flattened = []
        radius_per_edge = []

        # Extract line data from scene elements
        for element in self.scene_data.elements:
            if element.shape.type == "cylinder":
                # Get cylinder center point and parameters
                center = np.array(element.position)
                height = element.shape.parameters.height
                radius = element.shape.parameters.radius
                orientation = element.orientation

                # Calculate rotated direction vector
                base_direction = np.array([0, 0, 1])  # Default cylinder axis
                rotation_matrix = self._quaternion_to_rotation_matrix(orientation)
                direction = rotation_matrix @ base_direction
                direction = direction / np.linalg.norm(direction)  # Normalize direction vector

                # Calculate cylinder start and end points (relative to center)
                half_height = height / 2
                start = center - direction * half_height
                end = center + direction * half_height

                # Add to line data
                line_pts_flattened.extend([start, end])
                radius_per_edge.append(radius)

        # Create elements using create_collision_bodies
        with pp.LockRenderer():
            element_bodies = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)

        # Create sphere approximations using pp.create_sphere
        sphere_bodies = []
        with pp.LockRenderer():
            for sphere in self.approximate_elements_with_spheres():
                sphere_body = pp.create_sphere(radius=sphere["radius"], color=(0, 1, 0, 0.5))  # Green semi-transparent
                pp.set_pose(sphere_body, pp.Pose(point=sphere["position"]))
                sphere_bodies.append(sphere_body)

        # Visualize channels
        if hasattr(self.scene_data, "channels_info"):
            channel_bodies = []
            with pp.LockRenderer():
                for channel in self.scene_data.channels_info:
                    # Get channel parameters
                    channel_center = np.array(channel.center)
                    channel_dir = np.array(channel.direction)
                    channel_type = channel.type
                    channel_size = channel.size
                    channel_thickness = channel.thickness

                    # Calculate channel dimensions
                    if channel_type == "ellipse":
                        a = channel_size[0]  # Major axis
                        b = channel_size[1]  # Minor axis
                        radius = min(a, b) / 2
                        height = channel_thickness
                    else:  # rectangle
                        width = channel_size[0]
                        height = channel_size[1]
                        radius = min(width, height) / 2
                        height = channel_thickness

                    # Create flat transparent cylinder
                    cylinder_body = pp.create_cylinder(radius=radius, height=height, color=(0, 1, 1, 0.3))

                    # Build coordinate system based on channel_dir
                    z_axis = channel_dir / np.linalg.norm(channel_dir)

                    # Choose any vector not parallel to z-axis as temporary x-axis
                    temp_x = np.array([1, 0, 0])
                    if np.abs(np.dot(temp_x, z_axis)) > 0.9:  # If too close to parallel
                        temp_x = np.array([0, 1, 0])

                    # Calculate y-axis
                    y_axis = np.cross(z_axis, temp_x)
                    y_axis = y_axis / np.linalg.norm(y_axis)

                    # Calculate x-axis
                    x_axis = np.cross(y_axis, z_axis)
                    x_axis = x_axis / np.linalg.norm(x_axis)

                    # Build rotation matrix
                    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

                    # Convert to Euler angles
                    rotation = Rotation.from_matrix(rotation_matrix)
                    rotation_euler = rotation.as_euler("xyz", degrees=False).tolist()

                    # Set cylinder position and orientation
                    pp.set_pose(
                        cylinder_body,
                        pp.Pose(point=channel_center, euler=rotation_euler),
                    )
                    channel_bodies.append(cylinder_body)

                    # Create channel direction indicator line
                    line_body = pp.add_line(
                        channel_center, channel_center + channel_dir * 0.25, color=(0, 1, 1, 1), width=4
                    )
                    channel_bodies.append(line_body)

        print(
            f"Scene visualization complete: {len(element_bodies)} elements, {len(sphere_bodies)} sphere approximations"
        )
        if hasattr(self.scene_data, "channels_info"):
            print(f"Number of channels: {len(self.scene_data.channels_info)}")

        # Keep GUI running
        pp.wait_for_user("Press Enter to exit...")

    def get_robot_start_pose(self) -> List[float]:
        """
        Get the robot's start joint configuration.

        Returns:
            List[float]: Robot's start joint angles [j1, j2, j3, j4, j5, j6]
        """
        if not self.robot_info:
            raise ValueError("Robot information not loaded")
        return self.robot_info.start

    def get_robot_target_pose(self) -> List[float]:
        """
        Get the robot's target joint configuration.

        Returns:
            List[float]: Robot's target joint angles [j1, j2, j3, j4, j5, j6]
        """
        if not self.robot_info:
            raise ValueError("Robot information not loaded")
        return self.robot_info.target

    def get_robot_grasp_offset(self) -> List[float]:
        """
        Get the robot's grasp offset in tool_link frame.

        Returns:
            List[float]: Grasp offset [x, y, z] in tool_link frame
        """
        if not self.robot_info:
            raise ValueError("Robot information not loaded")
        return self.robot_info.grasp_offset

    def get_robot_pose_2d(self) -> List[float]:
        """
        Get the robot's 2D pose.

        Returns:
            List[float]: Robot's 2D pose [x, y, yaw]
        """
        if not self.robot_info:
            raise ValueError("Robot information not loaded")
        return self.robot_info.pose_2d

    def get_element_info(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract line points and radii information from scene elements.

        Returns:
            Tuple[List[np.ndarray], List[float]]: A tuple containing:
                - List of line points (each point is a numpy array)
                - List of radii for each line segment
        """
        if not self.scene_data:
            raise ValueError("Scene data is not loaded. Call load_scene() first.")

        line_pts_flattened = []
        radius_per_edge = []

        # Extract line data from scene elements
        for element in self.scene_data.elements:
            if element.shape.type == "cylinder":
                # Get cylinder center point and parameters
                center = np.array(element.position)
                height = element.shape.parameters.height
                radius = element.shape.parameters.radius
                orientation = element.orientation

                # Calculate rotated direction vector
                base_direction = np.array([0, 0, 1])  # Default cylinder axis
                rotation_matrix = self._quaternion_to_rotation_matrix(orientation)
                direction = rotation_matrix @ base_direction
                direction = direction / np.linalg.norm(direction)  # Normalize direction vector

                # Calculate cylinder start and end points (relative to center)
                half_height = height / 2
                start = center - direction * half_height
                end = center + direction * half_height

                # Add to line data
                line_pts_flattened.extend([start, end])
                radius_per_edge.append(radius)

        return line_pts_flattened, radius_per_edge


if __name__ == "__main__":
    parser = SceneParser("/home/jeong/summer_research/eth_ws/src/husky_assembly/scripts/model/scenes/cuboid_1.yml")
    parser.load_scene()
    parser.visualize_scene()
