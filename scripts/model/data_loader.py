import argparse
import glob
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pybullet_planning as pp
import torch
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from model.scene_parse import SceneParser
from multi_tangent.collision import create_collision_bodies
from robot.robot_setup import RobotSetup
from utils.collision import init_pb


class PointCloudDataset(Dataset):
    """
    PyTorch Dataset for loading point clouds and trajectory data
    """

    def __init__(
        self,
        scene_names: Union[str, List[str]],
        task_names: Union[str, List[str]],
        algorithm_names: Union[str, List[str]],
        data_loader=None,
        num_points: int = 1024,
        num_grasp_points: int = 256,  # Number of points to sample for grasped element
        normal_channel: bool = True,
        trajectory_length: Optional[int] = None,
        transform: Optional[Callable] = None,
        add_noise: bool = False,
        noise_ratio: float = 0.05,
        noise_scale: float = 0.01
    ):
        """
        Initialize point cloud dataset

        Args:
            scene_names: Scene name or list
            task_names: Task name or list
            algorithm_names: Algorithm name or list
            data_loader: Existing DataLoader instance, if None create new instance
            num_points: Number of points to sample for each element
            num_grasp_points: Number of points to sample for grasped element
            normal_channel: Whether to include normals
            trajectory_length: Target trajectory length, if specified re-interpolate trajectory
            transform: Transformation function applied to point clouds
            add_noise: Whether to add random noise to points
            noise_ratio: Ratio of points to add noise
            noise_scale: Scale of the noise
        """
        self.data_loader = data_loader if data_loader else SceneDataLoader()
        self.num_points = num_points
        self.num_grasp_points = num_grasp_points
        self.normal_channel = normal_channel
        self.trajectory_length = trajectory_length
        self.transform = transform
        self.add_noise = add_noise
        self.noise_ratio = noise_ratio
        self.noise_scale = noise_scale

        # Convert single string to list
        if isinstance(scene_names, str):
            scene_names = [scene_names]
        if isinstance(task_names, str):
            task_names = [task_names]
        if isinstance(algorithm_names, str):
            algorithm_names = [algorithm_names]

        self.scene_names = scene_names
        self.task_names = task_names
        self.algorithm_names = algorithm_names

        # Store all data source information
        self.data_sources = []

        # Collect all matching data
        for scene in scene_names:
            tasks = task_names if task_names else self.data_loader.list_tasks_for_scene(scene)
            for task in tasks:
                algorithms = (
                    algorithm_names if algorithm_names else self.data_loader.list_algorithms_for_task(scene, task)
                )
                for algorithm in algorithms:
                    trajectories = self.data_loader.load_trajectories(scene, task, algorithm)
                    if trajectories:
                        self.data_sources.append(
                            {"scene": scene, "task": task, "algorithm": algorithm, "count": len(trajectories)}
                        )

        # Calculate total data count
        self.total_count = sum(source["count"] for source in self.data_sources)

        # Cache point clouds for each scene to avoid repeated calculations
        self.point_cloud_cache = {}
        # Cache robot info for each scene to avoid repeated loading
        self.robot_info_cache = {}
        # Cache grasped element point clouds
        self.grasped_element_cache = {}

    def __len__(self):
        """Return dataset size"""
        return self.total_count

    def __getitem__(self, idx):
        """Get a single data sample"""
        # Find the corresponding data source
        current_idx = idx
        source = None
        for s in self.data_sources:
            if current_idx < s["count"]:
                source = s
                break
            current_idx -= s["count"]

        if source is None:
            raise IndexError(f"Index {idx} out of dataset range")

        scene = source["scene"]
        task = source["task"]
        algorithm = source["algorithm"]

        # Load scene configuration only once
        parser = None
        scene_task_key = f"{scene}_{task}"

        # Get or load robot information
        if scene_task_key not in self.robot_info_cache:
            if parser is None:
                parser = self.data_loader.load_scene_config(scene, task)

            # Get robot joint angles and grasp offset
            robot_start_pose = parser.get_robot_start_pose()
            robot_target_pose = parser.get_robot_target_pose()
            grasp_offset = parser.get_robot_grasp_offset()

            # Get robot 2D pose for coordinate transformation
            robot_pose_2d = parser.get_robot_pose_2d(output_type="array")

            # Cache robot information
            self.robot_info_cache[scene_task_key] = {
                "start_pose": robot_start_pose,
                "target_pose": robot_target_pose,
                "grasp_offset": grasp_offset,
                "pose_2d": robot_pose_2d,
            }
        else:
            robot_info = self.robot_info_cache[scene_task_key]
            robot_start_pose = robot_info["start_pose"]
            robot_target_pose = robot_info["target_pose"]
            grasp_offset = robot_info["grasp_offset"]
            robot_pose_2d = robot_info["pose_2d"]

        # Load or generate point cloud from cache
        if scene_task_key not in self.point_cloud_cache:
            # Load from data source
            if parser is None:
                parser = self.data_loader.load_scene_config(scene, task)
            line_pts, radius_per_edge = parser.get_element_info()

            # Generate point cloud data and element labels
            point_cloud, element_labels = self.data_loader._sample_points_from_elements(
                line_pts, 
                radius_per_edge, 
                num_points=self.num_points, 
                normal_channel=self.normal_channel,
                add_noise=self.add_noise,
                noise_ratio=self.noise_ratio,
                noise_scale=self.noise_scale
            )
            self.point_cloud_cache[scene_task_key] = (point_cloud, element_labels)
        else:
            point_cloud, element_labels = self.point_cloud_cache[scene_task_key]

        # Generate default grasped element point cloud (cylinder at origin)
        # Use cache to avoid repetitive computation
        if "default_grasped_element" not in self.grasped_element_cache:
            # Create a default cylinder at origin with length 1 and radius 0.01
            start_point = np.array([0.0, 0.0, -0.5])  # Bottom of cylinder
            end_point = np.array([0.0, 0.0, 0.5])  # Top of cylinder
            radius = 0.01

            # Sample points from this default cylinder
            grasped_line_pts = [start_point, end_point]
            grasped_radius = [radius]

            grasped_point_cloud, _ = self.data_loader._sample_points_from_elements(
                grasped_line_pts, grasped_radius, num_points=self.num_grasp_points, normal_channel=self.normal_channel, add_noise=self.add_noise, noise_ratio=self.noise_ratio, noise_scale=self.noise_scale
            )
            self.grasped_element_cache["default_grasped_element"] = grasped_point_cloud
        else:
            grasped_point_cloud = self.grasped_element_cache["default_grasped_element"]

        # Transform scene point cloud to robot coordinate system
        # Note: grasped_point_cloud remains in original coordinates as requested
        point_cloud = self._transform_to_robot_frame(point_cloud, robot_pose_2d)

        # Load trajectory data with target_length parameter
        trajectories = self.data_loader.load_trajectories(scene, task, algorithm, target_length=self.trajectory_length)
        trajectory = trajectories[current_idx]

        # Apply transformation if specified
        if self.transform:
            point_cloud = self.transform(point_cloud)
            # Apply same transform to grasped element if needed
            grasped_point_cloud = self.transform(grasped_point_cloud)

        # Convert to tensors
        point_cloud_tensor = torch.FloatTensor(point_cloud)
        element_labels_tensor = torch.LongTensor(element_labels)
        trajectory_tensor = torch.FloatTensor(trajectory)
        grasped_point_cloud_tensor = torch.FloatTensor(grasped_point_cloud)
        robot_start_pose_tensor = torch.FloatTensor(robot_start_pose)
        robot_target_pose_tensor = torch.FloatTensor(robot_target_pose)
        grasp_offset_tensor = torch.FloatTensor(grasp_offset)

        return {
            "point_cloud": point_cloud_tensor,
            "element_labels": element_labels_tensor,
            "trajectory": trajectory_tensor,
            "grasped_point_cloud": grasped_point_cloud_tensor,
            "robot_start_pose": robot_start_pose_tensor,
            "robot_target_pose": robot_target_pose_tensor,
            "grasp_offset": grasp_offset_tensor,
            "scene": scene,
            "task": task,
            "algorithm": algorithm,
        }

    def _transform_to_robot_frame(self, point_cloud, robot_pose_2d):
        """
        Transform point cloud from scene coordinate frame to robot coordinate frame

        Args:
            point_cloud: Point cloud data with shape [N, D], where D is 3 (coordinates only) or 6 (coordinates + normals)
            robot_pose_2d: Robot pose in the scene [x, y, yaw]

        Returns:
            np.ndarray: Transformed point cloud data
        """
        # Copy point cloud to avoid modifying original data
        transformed_cloud = point_cloud.copy()

        # Parse robot pose
        robot_x, robot_y, robot_yaw = robot_pose_2d

        # Calculate rotation matrix
        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)
        rotation_matrix = np.array([[cos_yaw, sin_yaw, 0], [-sin_yaw, cos_yaw, 0], [0, 0, 1]])

        # Apply transformation to point coordinates
        points = transformed_cloud[:, :3]

        # 1. Translation: Move points from scene frame to robot-centered frame
        points[:, 0] -= robot_x
        points[:, 1] -= robot_y

        # 2. Rotation: Rotate points according to robot orientation
        points_rotated = np.dot(points, rotation_matrix)
        transformed_cloud[:, :3] = points_rotated

        # Also rotate normal vectors if present
        if transformed_cloud.shape[1] > 3:  # Has normal vectors
            normals = transformed_cloud[:, 3:6]
            normals_rotated = np.dot(normals, rotation_matrix)
            transformed_cloud[:, 3:6] = normals_rotated

        return transformed_cloud

    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=4, **kwargs):
        """
        Get PyTorch DataLoader

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker threads to load data
            **kwargs: Additional arguments to pass to torch.utils.data.DataLoader

        Returns:
            torch.utils.data.DataLoader: PyTorch DataLoader
        """
        return TorchDataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)


class SceneDataLoader:
    """
    Data loader for loading scene configuration and robot trajectory data

    Data sources:
    - Trajectory data: data/${scene_name}/${task_name}/${algorithm_name}/plan_{id}.npy
    - Scene configuration: scenes/${scene_name}/${task_name}.yml
    """

    def __init__(self):
        """
        Initialize data loader
        """
        self.data_dir = os.path.join(HERE, "model", "data")
        self.scenes_dir = os.path.join(HERE, "model", "scenes")

    def list_available_scenes(self) -> List[str]:
        """
        List all available scenes

        Returns:
            List[str]: Scene name list
        """
        return [os.path.basename(f) for f in glob.glob(os.path.join(self.scenes_dir, "*")) if os.path.isdir(f)]

    def list_tasks_for_scene(self, scene_name: str) -> List[str]:
        """
        List all tasks for a specified scene

        Args:
            scene_name: Scene name

        Returns:
            List[str]: Task name list
        """
        yaml_files = glob.glob(os.path.join(self.scenes_dir, scene_name, "*.yml"))
        return [os.path.splitext(os.path.basename(f))[0] for f in yaml_files]

    def list_algorithms_for_task(self, scene_name: str, task_name: str) -> List[str]:
        """
        List all algorithms for a specified scene and task

        Args:
            scene_name: Scene name
            task_name: Task name

        Returns:
            List[str]: Algorithm name list
        """
        task_dir = os.path.join(self.data_dir, scene_name, task_name)
        if not os.path.exists(task_dir):
            return []
        return [os.path.basename(f) for f in glob.glob(os.path.join(task_dir, "*")) if os.path.isdir(f)]

    def load_trajectories(
        self, scene_name: str, task_name: str, algorithm_name: str, target_length: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Load all trajectory data for a specified scene, task, and algorithm, and optionally re-interpolate to specified length

        Args:
            scene_name: Scene name
            task_name: Task name
            algorithm_name: Algorithm name
            target_length: Target trajectory length, if specified re-interpolate trajectory

        Returns:
            List[np.ndarray]: Trajectory data list
        """
        alg_dir = os.path.join(self.data_dir, scene_name, task_name, algorithm_name)
        if not os.path.exists(alg_dir):
            return []

        trajectory_files = sorted(glob.glob(os.path.join(alg_dir, "plan_*.npy")))
        raw_trajectories = []

        # Load all raw trajectories and find the longest trajectory length
        max_length = 0
        for traj_file in trajectory_files:
            try:
                trajectory = np.load(traj_file)
                raw_trajectories.append(trajectory)
                max_length = max(max_length, len(trajectory))
            except Exception as e:
                print(f"Cannot load trajectory file {traj_file}: {e}")

        # Determine actual used interpolation length
        actual_target_length = target_length
        if target_length is not None and max_length > target_length:
            print(
                f"Warning: Maximum trajectory length ({max_length}) is greater than target length ({target_length}), will use maximum length as interpolation target"
            )
            actual_target_length = max_length

        # Perform interpolation processing
        trajectories = []
        for trajectory in raw_trajectories:
            if actual_target_length is not None and len(trajectory) != actual_target_length:
                trajectory = self._interpolate_trajectory(trajectory, actual_target_length)
            trajectories.append(trajectory)

        return trajectories

    def _interpolate_trajectory(self, trajectory: np.ndarray, target_length: int) -> np.ndarray:
        """
        Re-interpolate trajectory to specified length, ensuring all points in original trajectory are preserved

        Args:
            trajectory: Original trajectory data with shape [N, D], where N is time step count, D is each step dimension
            target_length: Target trajectory length

        Returns:
            np.ndarray: Re-interpolated trajectory with shape [target_length, D]
        """
        # Original trajectory length and dimension
        orig_length, dims = trajectory.shape

        # If target length is less than original length, down-sampling is needed
        if target_length <= orig_length:
            # Select equal intervals
            indices = np.round(np.linspace(0, orig_length - 1, target_length)).astype(int)
            return trajectory[indices]

        # Create new trajectory array, initialized to zero
        new_trajectory = np.zeros((target_length, dims))

        # First ensure all points in original trajectory are preserved
        # Calculate indices of original points to be preserved in new trajectory
        orig_indices_in_new = np.round(np.linspace(0, target_length - 1, orig_length)).astype(int)

        # Place original points into new trajectory
        for i, idx in enumerate(orig_indices_in_new):
            new_trajectory[idx] = trajectory[i]

        # Create mask to mark which positions have been assigned values
        mask = np.zeros(target_length, dtype=bool)
        mask[orig_indices_in_new] = True

        # Create interpolation for positions without assigned values
        for i in range(target_length):
            if not mask[i]:
                # Find nearest known points on both sides
                left_idx = np.max(orig_indices_in_new[orig_indices_in_new < i]) if any(orig_indices_in_new < i) else 0
                right_idx = (
                    np.min(orig_indices_in_new[orig_indices_in_new > i])
                    if any(orig_indices_in_new > i)
                    else target_length - 1
                )

                # If left and right indices are the same, interpolation cannot be performed, use nearest point
                if left_idx == right_idx:
                    new_trajectory[i] = new_trajectory[left_idx]
                    continue

                # Calculate interpolation weights
                left_orig_idx = np.where(orig_indices_in_new == left_idx)[0][0]
                right_orig_idx = np.where(orig_indices_in_new == right_idx)[0][0]

                weight = (i - left_idx) / (right_idx - left_idx)

                # Linear interpolation
                new_trajectory[i] = (1 - weight) * trajectory[left_orig_idx] + weight * trajectory[right_orig_idx]

        return new_trajectory

    def load_scene_config(self, scene_name: str, task_name: str) -> SceneParser:
        """
        Load and parse scene configuration

        Args:
            scene_name: Scene name
            task_name: Task name

        Returns:
            SceneParser: Scene configuration parser
        """
        scene_file = os.path.join(self.scenes_dir, scene_name, f"{task_name}.yml")
        if not os.path.exists(scene_file):
            raise FileNotFoundError(f"Scene configuration file does not exist: {scene_file}")

        parser = SceneParser(scene_file)
        parser.load_scene()
        return parser

    def prepare_point_cloud_data(
        self,
        scene_name: str,
        task_name: str,
        algorithm_name: str,
        normal_channel: bool = True,
        num_points: int = 1024,
        trajectory_length: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Prepare point cloud data, element labels, and corresponding trajectory for PointNet model

        Args:
            scene_name: Scene name
            task_name: Task name
            algorithm_name: Algorithm name
            normal_channel: Whether to include normal channel
            num_points: Number of points to sample for each element
            trajectory_length: Target trajectory length, if specified re-interpolate trajectory

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]: Point cloud data, element labels, and corresponding trajectory
        """
        # Load trajectory
        trajectories = self.load_trajectories(scene_name, task_name, algorithm_name, trajectory_length)
        if not trajectories:
            return [], [], []

        # Load scene configuration
        parser = self.load_scene_config(scene_name, task_name)

        # Get scene element information
        line_pts, radius_per_edge = parser.get_element_info()

        # Sample points from elements (include normal vector calculation and element labels)
        point_cloud, element_labels = self._sample_points_from_elements(
            line_pts, radius_per_edge, num_points=num_points, normal_channel=normal_channel
        )

        # Create same point cloud and label for each trajectory
        point_clouds = [point_cloud] * len(trajectories)
        element_labels_list = [element_labels] * len(trajectories)

        return point_clouds, element_labels_list, trajectories

    def _sample_points_from_elements(
        self,
        line_pts: List[np.ndarray],
        radius_per_edge: List[float],
        num_points: int = 1024,
        normal_channel: bool = True,
        add_noise: bool = False,
        noise_ratio: float = 0.05,
        noise_scale: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points from surface of element cylinder and add element labels

        Args:
            line_pts: List of line points
            radius_per_edge: Radius of each line
            num_points: Number of points to sample for each element
            normal_channel: Whether to calculate normal vector
            add_noise: Whether to add random noise to some points
            noise_ratio: Ratio of points to add noise
            noise_scale: Scale of the noise

        Returns:
            Tuple[np.ndarray, np.ndarray]: Point cloud data and element labels
        """
        # Sample points from cylinder surface
        sampled_points = []
        normals = [] if normal_channel else None
        element_labels = []  # Store element index of each point

        # Sample points for each cylinder element
        num_elements = len(radius_per_edge)
        for element_idx in range(num_elements):
            i = element_idx * 2  # Each element defined by two points
            if i + 1 >= len(line_pts):
                break

            start, end = line_pts[i], line_pts[i + 1]
            radius = radius_per_edge[element_idx]

            # Calculate cylinder axis direction
            direction = end - start
            if np.linalg.norm(direction) < 1e-6:  # Avoid zero length case
                continue
            direction = direction / np.linalg.norm(direction)

            # Create cylinder coordinate system
            if abs(direction[0]) < 0.9:
                v = np.array([1.0, 0.0, 0.0])
            else:
                v = np.array([0.0, 1.0, 0.0])

            base1 = np.cross(direction, v)
            base1 = base1 / np.linalg.norm(base1)

            base2 = np.cross(direction, base1)
            base2 = base2 / np.linalg.norm(base2)

            # Sample num_points points for this element
            for _ in range(num_points):
                # Random position along axis
                t = np.random.uniform(0, 1)
                center = start * (1 - t) + end * t

                # Random angle on cylinder surface
                theta = np.random.uniform(0, 2 * np.pi)

                # Calculate point on cylinder surface
                radial_vec = np.cos(theta) * base1 + np.sin(theta) * base2
                surface_point = center + radius * radial_vec
                
                # Add random noise to some points if requested
                if add_noise and np.random.random() < noise_ratio:
                    # Generate random noise vector
                    noise = np.random.normal(0, noise_scale, 3)
                    # Apply noise to surface point
                    surface_point = surface_point + noise
                    
                    # If normal vectors are needed, slightly disturb them too
                    if normal_channel:
                        # Small disturbance to normal direction
                        normal_noise = np.random.normal(0, noise_scale/3, 3)
                        radial_vec = radial_vec + normal_noise
                        # Renormalize to unit vector
                        radial_vec = radial_vec / np.linalg.norm(radial_vec)
                
                # Always append point and label regardless of noise
                sampled_points.append(surface_point)
                element_labels.append(element_idx)
                normals.append(radial_vec)

        # Convert to numpy array
        points_array = np.array(sampled_points)
        labels_array = np.array(element_labels)

        # If normal vector is needed, concatenate points and normal vectors
        if normal_channel and points_array.size > 0:
            normals_array = np.array(normals)
            return np.hstack((points_array, normals_array)), labels_array

        return points_array, labels_array

    def visualize_point_cloud(self, scene_name: str, task_name: str, num_points: int = 1024, show_normals: bool = True):
        """
        Read scene file, generate point cloud data, and use matplotlib for visualization

        Args:
            scene_name: Scene name
            task_name: Task name
            num_points: Number of points to sample for each element
            show_normals: Whether to show normal vectors
        """
        # Load scene
        parser = self.load_scene_config(scene_name, task_name)

        # Get element information
        line_pts, radius_per_edge = parser.get_element_info()

        # Sample point cloud data
        point_cloud, _ = self._sample_points_from_elements(
            line_pts, radius_per_edge, num_points=num_points, normal_channel=True
        )

        # Create 3D figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Separate point coordinates and normal vectors
        points = point_cloud[:, :3]
        normals = point_cloud[:, 3:] if point_cloud.shape[1] > 3 else None

        # Calculate colors for drawing
        num_elements = len(radius_per_edge)
        colors = []

        # Assign unique color to each element
        for i in range(0, len(line_pts), 2):
            if i + 1 >= len(line_pts):
                break

            element_index = i // 2
            # Create color cycle to distinguish different elements
            color = plt.cm.tab20(element_index % 20)
            # Assign same color to all points of current element
            colors.extend([color] * num_points)

        # Draw point cloud
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, marker="o", s=5, alpha=0.8)

        # Optionally draw normal vectors
        if show_normals and normals is not None:
            # Only show part of normal vectors for clarity
            skip = 10  # Show one normal vector every 10 points
            for i in range(0, len(points), skip):
                # Draw short vector to represent normal vector direction
                ax.quiver(
                    points[i, 0],
                    points[i, 1],
                    points[i, 2],
                    normals[i, 0],
                    normals[i, 1],
                    normals[i, 2],
                    color="red",
                    length=0.02,
                    normalize=True,
                )

        # Draw element center lines
        for i in range(0, len(line_pts), 2):
            if i + 1 >= len(line_pts):
                break

            start, end = line_pts[i], line_pts[i + 1]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], "k-", linewidth=1, alpha=0.5)

        # Set figure properties
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Point Cloud Visualization: {scene_name}/{task_name}")

        # Set axis proportions equal to maintain correct shape
        max_range = (
            np.array(
                [
                    points[:, 0].max() - points[:, 0].min(),
                    points[:, 1].max() - points[:, 1].min(),
                    points[:, 2].max() - points[:, 2].min(),
                ]
            ).max()
            / 2.0
        )

        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()
        plt.show()

    def create_dataset(
        self,
        scene_names: Union[str, List[str]],
        task_names: Union[str, List[str]] = None,
        algorithm_names: Union[str, List[str]] = None,
        num_points: int = 1024,
        num_grasp_points: int = 256,
        normal_channel: bool = True,
        trajectory_length: Optional[int] = None,
        transform: Optional[Callable] = None,
        add_noise: bool = False,
        noise_ratio: float = 0.05,
        noise_scale: float = 0.01
    ) -> PointCloudDataset:
        """
        Create PyTorch point cloud dataset

        Args:
            scene_names: Scene name or list
            task_names: Task name or list, if None use all tasks
            algorithm_names: Algorithm name or list, if None use all algorithms
            num_points: Number of points to sample for each element
            num_grasp_points: Number of points to sample for grasped element
            normal_channel: Whether to include normal vector
            trajectory_length: Target trajectory length, if specified re-interpolate trajectory
            transform: Transformation function applied to point clouds
            add_noise: Whether to add random noise to points
            noise_ratio: Ratio of points to add noise
            noise_scale: Scale of the noise

        Returns:
            PointCloudDataset: PyTorch dataset
        """
        return PointCloudDataset(
            scene_names=scene_names,
            task_names=task_names,
            algorithm_names=algorithm_names,
            data_loader=self,
            num_points=num_points,
            num_grasp_points=num_grasp_points,
            normal_channel=normal_channel,
            trajectory_length=trajectory_length,
            transform=transform,
            add_noise=add_noise,
            noise_ratio=noise_ratio,
            noise_scale=noise_scale
        )

    def check_trajectory_collisions(
        self,
        scene_name: str,
        task_name: str,
        algorithm_name: str,
        interpolation_steps: int = 5000,
    ) -> List[str]:
        """
        检查指定场景/任务/算法下所有轨迹的碰撞情况

        Args:
            scene_name: 场景名称
            task_name: 任务名称
            algorithm_name: 算法名称
            interpolation_steps: 插值后的轨迹点数量

        Returns:
            List[str]: 发生碰撞的轨迹文件列表
        """

        # 加载场景配置
        parser = self.load_scene_config(scene_name, task_name)

        # 获取场景元素信息
        line_pts, radius_per_edge = parser.get_element_info()

        init_pb()

        # 创建机器人实例
        rb = RobotSetup("r0")

        # 从配置文件获取机器人2D位姿
        robot_pose_2d = parser.get_robot_pose_2d(output_type="array")
        robot_x, robot_y, robot_yaw = robot_pose_2d

        # 设置机器人位姿
        pp.set_pose(rb.robot, pp.Pose(point=(robot_x, robot_y, 0), euler=pp.Euler(0, 0, robot_yaw)))

        # 创建被抓取的元素
        line_pts_grasped = [np.array([0, 0, 0]), np.array([0, 0, 1])]
        grasped_element = create_collision_bodies(line_pts_grasped, [0.01], viewer=False)[0]

        # 从配置文件获取抓取偏移量
        grasp_offset = parser.get_robot_grasp_offset()

        # 设置被抓取元素的位姿并创建附着关系
        pp.set_pose(
            grasped_element,
            pp.multiply(
                pp.get_link_pose(rb.robot, rb.tool_link),
                pp.Pose(point=tuple(grasp_offset), euler=pp.Euler(1.5708, 0, 0)),
            ),
        )
        grasp_attachment = pp.create_attachment(rb.robot, rb.tool_link, grasped_element)

        # 创建碰撞检测函数
        element_bodies = create_collision_bodies(line_pts, radius_per_edge, viewer=False)
        extra_disabled_collisions = [
            (
                (rb.robot, pp.link_from_name(rb.robot, "ur_arm_wrist_3_link")),
                (rb.ee_attachment.child, pp.BASE_LINK),
            ),
            (
                (rb.ee_attachment.child, pp.BASE_LINK),
                (grasp_attachment.child, pp.BASE_LINK),
            ),
        ]

        collision_fn = pp.get_collision_fn(
            rb.robot,
            rb.arm_joints,
            obstacles=element_bodies,
            attachments=[grasp_attachment, rb.ee_attachment] + rb.attachments,
            self_collisions=True,
            disabled_collisions=rb.disabled_collisions,
            extra_disabled_collisions=extra_disabled_collisions,
            max_distance=0.0,
        )

        # 加载所有轨迹
        trajectories = self.load_trajectories(
            scene_name=scene_name, task_name=task_name, algorithm_name=algorithm_name, target_length=interpolation_steps
        )

        # 检查每条轨迹
        collision_files = []
        task_dir = os.path.join(self.data_dir, scene_name, task_name, algorithm_name)
        traj_files = sorted(glob.glob(os.path.join(task_dir, "plan_*.npy")))

        pp.wait_for_user("按回车键继续...")

        print(f"\n开始检查轨迹碰撞:")
        print(f"场景: {scene_name}")
        print(f"任务: {task_name}")
        print(f"算法: {algorithm_name}")
        print(f"轨迹数量: {len(traj_files)}")
        print("=" * 50)

        for i, (traj_file, trajectory) in enumerate(zip(traj_files, trajectories)):
            file_name = os.path.basename(traj_file)
            print(f"\r检查轨迹 {i+1}/{len(traj_files)}: {file_name}", end="")

            # 检查轨迹中的每个配置
            for conf in trajectory:
                if collision_fn(conf):
                    collision_files.append(file_name)
                    print(f"\n发现碰撞: {file_name}")
                    break

        print("\n\n碰撞检查完成!")
        if collision_files:
            print("\n以下轨迹存在碰撞:")
            for file in collision_files:
                print(f"- {file}")
        else:
            print("\n所有轨迹均无碰撞!")

        return collision_files

    def reorganize_all_trajectories(self) -> None:
        """
        重新排序所有场景/任务/算法下的轨迹文件
        """
        print("\n开始扫描数据集...")

        # 遍历所有场景
        for scene_name in os.listdir(self.data_dir):
            scene_dir = os.path.join(self.data_dir, scene_name)
            if not os.path.isdir(scene_dir):
                continue

            # 遍历所有任务
            for task_name in os.listdir(scene_dir):
                task_dir = os.path.join(scene_dir, task_name)
                if not os.path.isdir(task_dir):
                    continue

                # 遍历所有算法
                for algorithm_name in os.listdir(task_dir):
                    alg_dir = os.path.join(task_dir, algorithm_name)
                    if not os.path.isdir(alg_dir):
                        continue

                    # 获取所有轨迹文件
                    traj_files = glob.glob(os.path.join(alg_dir, "plan_*.npy"))
                    if not traj_files:
                        continue

                    # 自定义排序函数，提取plan_后的数字进行排序
                    def get_plan_number(filename):
                        basename = os.path.basename(filename)
                        try:
                            return int(basename.split("_")[1].split(".")[0])
                        except (IndexError, ValueError):
                            return float("inf")  # 对于非标准命名的文件排在最后

                    # 按数字顺序排序
                    traj_files.sort(key=get_plan_number)

                    # 修改这部分，将编号从0开始
                    file_mapping = {}  # {old_name: new_name}
                    for i, old_file in enumerate(traj_files):
                        old_name = os.path.basename(old_file)
                        new_name = f"plan_{i}.npy"  # 改为从0开始编号
                        if old_name != new_name:
                            file_mapping[old_name] = new_name

                    if not file_mapping:
                        continue

                    print(f"\n发现需要重排序的轨迹:")
                    print(f"场景: {scene_name}")
                    print(f"任务: {task_name}")
                    print(f"算法: {algorithm_name}")
                    print(f"轨迹数量: {len(traj_files)}")
                    print("=" * 50)

                    print("\n文件重命名映射:")
                    for old, new in file_mapping.items():
                        print(f"{old} -> {new}")

                    # 执行重命名
                    print("\n开始重命名...")
                    temp_files = {}  # 用于存储临时文件名，避免重命名冲突

                    # 第一步：将所有文件重命名为临时名称
                    for old_name, new_name in file_mapping.items():
                        old_path = os.path.join(alg_dir, old_name)
                        temp_name = f"temp_{old_name}"
                        temp_path = os.path.join(alg_dir, temp_name)
                        os.rename(old_path, temp_path)
                        temp_files[temp_name] = new_name
                        print(f"临时重命名: {old_name} -> {temp_name}")

                    # 第二步：将临时文件重命名为目标名称
                    for temp_name, new_name in temp_files.items():
                        temp_path = os.path.join(alg_dir, temp_name)
                        new_path = os.path.join(alg_dir, new_name)
                        os.rename(temp_path, new_path)
                        print(f"最终重命名: {temp_name} -> {new_name}")

                    print("\n当前目录重排序完成!")

            print(f"\n场景 {scene_name} 处理完成!")

        print("\n所有数据重排序完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据集工具")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["check", "reorganize"],
        help="工具模式: check(检查碰撞) 或 reorganize(重新排序)",
    )

    # 碰撞检查模式的参数
    check_group = parser.add_argument_group("碰撞检查参数")
    check_group.add_argument("--scene", type=str, help="场景名称")
    check_group.add_argument("--task", type=str, help="任务名称")
    check_group.add_argument("--algorithm", type=str, help="算法名称")
    check_group.add_argument("--interpolation-steps", type=int, default=5000, help="插值步数")

    args = parser.parse_args()

    data_loader = SceneDataLoader()

    if args.mode == "check":
        # 检查必要参数
        if not all([args.scene, args.task, args.algorithm]):
            parser.error("碰撞检查模式需要指定 --scene, --task 和 --algorithm")

        data_loader.check_trajectory_collisions(
            scene_name=args.scene,
            task_name=args.task,
            algorithm_name=args.algorithm,
            interpolation_steps=args.interpolation_steps,
        )
    else:  # reorganize mode
        data_loader.reorganize_all_trajectories()
