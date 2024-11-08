import os
import random
import sys
from typing import Callable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from utils.params import *

sys.path.append(PROJECT_DIR)

from robot.robot_setup import INIT_ARM_JOINT_ANGLES


def is_point_in_polygon(point, polygon):
    """
    Check if the point (x, y) is inside the 2D polygon (projected on the XY plane).
    """
    if polygon is None:
        return False
    x, y = point[0], point[1]
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def distance_point_to_line(point, line_start, line_end):
    """
    Calculate the perpendicular distance from a point to a line segment in 2D.
    """
    line_vec = np.array(line_end) - np.array(line_start)
    point_vec = np.array(point) - np.array(line_start)
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_unitvec, point_vec_scaled)
    t = np.clip(t, 0, 1)
    nearest = line_start + t * line_vec
    dist = np.linalg.norm(nearest - point)
    return dist


def distance_point_to_line_3d(point, line_start, line_end):
    """
    Calculate the perpendicular distance from a point to a line segment in 3D.
    """
    line_vec = np.array(line_end) - np.array(line_start)
    point_vec = np.array(point) - np.array(line_start)
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    t = np.dot(line_unitvec, point_vec) / line_len
    t = np.clip(t, 0, 1)
    nearest = np.array(line_start) + t * line_vec
    dist = np.linalg.norm(nearest - np.array(point))
    return dist


def point_to_line_segment_vector(point, line_start, line_end):
    """
    @brief: calculate the vector from the point to the nearest point on the line segment in 3D space\n
    ---
    @param:\n
        point: [x, y, z], the coordinates of the point in 3D\n
        line_start: [x1, y1, z1], the start of the line segment in 3D\n
        line_end: [x2, y2, z2], the end of the line segment in 3D\n
    ---
    @return:\n
        direction_vector: np.ndarray, The direction vector from the point to the nearest point on the line segment\n
    """
    # Convert input to numpy arrays
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)

    # Compute line segment vector and point vector
    line_vec = line_end - line_start
    point_vec = point - line_start

    # Compute projection of point_vec onto line_vec
    line_len_sq = np.dot(line_vec, line_vec)  # Squared length of the line segment
    t = np.dot(point_vec, line_vec) / line_len_sq  # Projection factor

    # Clamp t to the range [0, 1] to ensure the projection falls on the segment
    t = np.clip(t, 0, 1)

    # Calculate the nearest point on the line segment
    nearest_point = line_start + t * line_vec

    # Calculate the direction vector from the point to the nearest point
    direction_vector = nearest_point - point

    return direction_vector


def calculate_yaw(direction_vector):
    """
    @brief: calculate the yaw angle of the direction vector's projection onto the XY plane\n
    ---
    @param:\n
        direction_vector: np.ndarray, the direction vector in 3D space\n
    ---
    @return:\n
        yaw_angle: float, the yaw angle in radians\n
    """
    # Project the direction vector onto the XY plane
    xy_projection = direction_vector[:2]  # Ignore the z component

    # Calculate the yaw angle
    yaw_angle = np.arctan2(xy_projection[1], xy_projection[0])  # atan2(y, x)

    return yaw_angle


def sample_points_near_projection(line_start, line_end, d, num_samples):
    """
    @brief: Projects a 3D line segment onto the XY plane and samples points near the projection.\n
    ---
    @param:\n
        line_start: [x1, y1, z1] - The start of the line segment in 3D.\n
        line_end: [x2, y2, z2] - The end of the line segment in 3D.\n
        d: float - The maximum distance from the projection to sample points.\n
        num_samples: int - The number of points to sample.\n
    ---
    @return:\n
        samples: np.ndarray - An array of sampled points near the projection on the XY plane.\n
    """
    line_start_xy = np.array(line_start[:2])
    line_end_xy = np.array(line_end[:2])

    t_values = np.linspace(0, 1, num_samples)
    line_points = np.outer(1 - t_values, line_start_xy) + np.outer(t_values, line_end_xy)

    samples = []
    for point in line_points:
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, d)
        offset = distance * np.array([np.cos(angle), np.sin(angle)])
        sample_point = point + offset
        point_item = np.hstack((sample_point, np.array([0])))
        direction_vector = point_to_line_segment_vector(point_item, line_start, line_end)
        yaw_angle = calculate_yaw(direction_vector)
        samples.append((point_item, yaw_angle))

    return samples


def is_valid_pose(
    candidate_point,
    candidate_orientation,
    target_edge,
    edges,
    safety_distance,
    reach_distance,
    collision_fn=None,
    attach_point=None,
):
    """
    Check if the candidate pose maintains a safe distance from all edges.
    """
    for edge in edges:
        if distance_point_to_line_3d(candidate_point, edge[0], edge[1]) < safety_distance:
            return False
        # if distance_point_to_line(candidate_point[:2], edge[0][:2], edge[1][:2]) < safety_distance:
        #     return False

    if distance_point_to_line_3d(candidate_point, target_edge[0], target_edge[1]) > reach_distance:
        return False

    # if attach_point is not None:
    #     if np.linalg.norm(candidate_point - attach_point) > reach_distance:
    #         return False

    if collision_fn is not None:
        conf = np.hstack((candidate_point[:2], np.array([candidate_orientation]), INIT_ARM_JOINT_ANGLES))
        collision = collision_fn(conf, diagnosis=False)
        if collision:
            return False

    return True


def robot_pose_sampler(
    vertices: List[List[float]],
    edges: List[List[List[float]]],
    target_edge: List[List[float]],
    sample_max_distance: float,
    safety_distance: float,
    reach_distance: float,
    sampling_number: int,
    attach_point=None,
    collision_fn: Union[Callable[[np.ndarray, bool], bool], None] = None,
    max_attempt: int = 10,
) -> Tuple[np.ndarray, float]:
    """
    Sample a base pose 2d.

    Params:
        vertices ([[x, y, z]]): vertices of assembled structure
        edges ([[edge_start, edge_end]]): edges of assembled structure
        target_edge ([edge_start, edge_end]): element to assemble
        sample_max_distance (float): the max 2D distance between target edge and sampled points
        safety_distance (float): the 3D distance between all elements and sampled points > safety_distance
        reach_distance (float): the 3D distance between target edge and sampled points < reach_distance
        sampling_number (int): the max sample numbers
        attach_point (None, not used): grasp point on the elements
        collision_fn (None | Callable[[np.ndarray, bool], bool], None): collision function to judge config with manipulator init conf, (robot_conf, diagnosis) -> False if no collision found, True otherwise
        max_attempt (int): max attempts to generate sample pose

    Returns:
        position (np.ndarray): [x, y, z]
        yaw (float): yaw (deg)
    """
    vertices = np.array(vertices)
    target_edge = np.array(target_edge)

    if len(edges) > 1 and len(np.unique(vertices[:, :2], axis=0)) >= 3:
        hull = ConvexHull(vertices[:, :2])
        projected_polygon = vertices[hull.vertices][:, :2]
    else:
        hull = None
        projected_polygon = None

    sample_idx = 0

    while True:
        sample_idx += 1
        if sample_idx > max_attempt:
            return None
        candidates = sample_points_near_projection(target_edge[0], target_edge[1], sample_max_distance, sampling_number)
        valid_candidates = []
        for candidate_point, candidate_orientation in candidates:
            projected_point = candidate_point[:2]
            if not is_point_in_polygon(projected_point, projected_polygon) and is_valid_pose(
                candidate_point,
                candidate_orientation,
                target_edge,
                edges,
                safety_distance,
                reach_distance,
                collision_fn,
                attach_point,
            ):
                valid_candidates.append((candidate_point, candidate_orientation))

        if valid_candidates == []:
            continue

        return random.choice(valid_candidates)
        # return valid_candidates


def plot_structure_and_poses(vertices, edges, target_edge, poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the structure using the edges
    for edge in edges:
        ax.plot(
            [edge[0][0], edge[1][0]],
            [edge[0][1], edge[1][1]],
            [edge[0][2], edge[1][2]],
            color="blue",
        )

    for point in vertices:
        ax.scatter(point[0], point[1], point[2], color="red")

    # Highlight the target edge
    ax.plot(
        [target_edge[0][0], target_edge[1][0]],
        [target_edge[0][1], target_edge[1][1]],
        [target_edge[0][2], target_edge[1][2]],
        color="red",
        linewidth=2,
    )

    # Plot the sampled poses
    for pose in poses:
        point = pose[0]
        ax.scatter(point[0], point[1], point[2], color="green")

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Structure and Sampled Poses Near Target Edge")

    plt.show()


if __name__ == "__main__":
    vertices = [[0, 0, 0], [1, 0, 0]]
    edges = [[[0, 0, 0], [1, 0, 0]]]
    target_edge = [[0, 0, 0], [1, 0, 0]]
    poses = robot_pose_sampler(vertices, edges, target_edge, 1.0, 0.75, 1.0, 100)
    print(poses)
    position, yaw = robot_pose_sampler(vertices, edges, target_edge, 1.0, 0.75, 1.0, 100)
    print(position, yaw)
    plot_structure_and_poses(vertices, edges, target_edge, poses)
