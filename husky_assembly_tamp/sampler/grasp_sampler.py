import os
import random
import sys
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pybullet_planning as pp
from husky_assembly_tamp.utils.collision import Element
from scipy.spatial.transform import Rotation as R


def closest_point_on_segment_to_ray(
    x: float, y: float, yaw: float, start: List[float], end: List[float]
) -> Tuple[List[float], float]:
    """
    Calculate closest point on the segment reference to ray start from [x, y] and angle yaw.

    Params:
        x (float): ray start point x
        y (float): ray start point y
        yaw (float): ray direction yaw
        start (List[float]): [x, y]
        end (List[float]): [x, y]

    Returns:
        point (List[float]): [x, y]
        alpha (float): (1 - alpha) * start + alpha * end
    """
    ray_origin = np.array([x, y])
    ray_dir = np.array([np.cos(yaw), np.sin(yaw)])

    start = np.array(start)
    end = np.array(end)

    seg_dir = end - start

    ray_normal = np.array([-ray_dir[1], ray_dir[0]])

    start_proj = np.dot(ray_normal, start - ray_origin)
    end_proj = np.dot(ray_normal, end - ray_origin)

    alpha_min = -start_proj / (end_proj - start_proj) if end_proj != start_proj else 0.0

    alpha_min = np.clip(alpha_min, 0.0, 1.0)

    closest_point = (1 - alpha_min) * start + alpha_min * end
    return closest_point, alpha_min


def normalize(v) -> np.ndarray:
    """
    Normalize an array.
    """
    return v / np.linalg.norm(v)


def project_point_onto_plane(point, plane_point, normal):
    """
    Project a point onto a plane.

    Params:
        point (np.ndarray): target point
        plane_point (np.ndarray): a point on the plane
        normal (np.ndarray): plane normal

    Returns:
        point (np.ndarray): projection point
    """
    normal = normal / np.linalg.norm(normal)
    d = -plane_point.dot(normal)
    t = -(point.dot(normal) + d) / np.linalg.norm(normal) ** 2
    projection = point + t * normal
    return projection


def random_point_on_plane(p1, p2, distance=0.5):
    # 计算向量 l
    l = np.array(p2) - np.array(p1)
    l_unit = l / np.linalg.norm(l)  # 归一化

    # 找到与 l 不平行的任意向量
    if l_unit[0] != 0 or l_unit[1] != 0:
        arbitrary_vector = np.array([0, 0, 1])
    else:
        arbitrary_vector = np.array([1, 0, 0])

    # 计算与 l 垂直的向量 u
    u = np.cross(l_unit, arbitrary_vector)
    u /= np.linalg.norm(u)  # 归一化

    # 计算与 l 和 u 垂直的向量 v
    v = np.cross(l_unit, u)

    # 随机角度 theta
    theta = np.random.uniform(0, 2 * np.pi)

    # 计算随机点
    random_point = np.array(p1) + distance * (np.cos(theta) * u + np.sin(theta) * v)
    return random_point


def preview_point_calculation(frame: List[int], element_from_index) -> List[float]:
    """
    Calculate preview points based on assembled structure.

    Params:
        frame (List[int]): indices of assembled structure including current element
        element_from_index ({index: Element}): element dict

    Returns:
        point (List[float]): [x, y, z], preview point
    """
    if len(frame) != 1:
        point = None
        for index in frame:
            element: Element = element_from_index[index]
            if point is None:
                point = element.axis_endpoints[0]
            else:
                point = np.vstack((point, element.axis_endpoints[0]))
            point = np.vstack((point, element.axis_endpoints[1]))
        point = point.mean(axis=0)
    else:
        element: Element = element_from_index[frame[0]]
        point = random_point_on_plane(element.axis_endpoints[0], element.axis_endpoints[1])
    return point.tolist()


def grasp_redirector_preview(
    edge_start: List[float],
    edge_end: List[float],
    attach_pose: Tuple[Tuple[float], Tuple[float]],
    preview_point: List[float],
    redirect_radius: float = np.pi / 4,
) -> Tuple[Tuple[float], Tuple[float]]:
    """
    Calculate the gripper's pose based on the preview point.

    Params:
        edge_start (List[float]): start point of the edge
        edge_end (List[float]): end point of the edge
        attach_pose (Tuple[Tuple[float], Tuple[float]]): ((x, y, z), (x, y, z, w)), attach pose of the target element, world_from_gripper
        preview_point (List[float]): preview point in the positive direction of the z axis
        redirect_radius (float, np.pi / 4): swing range in the positive direction of z axis

    Returns:
        pose (Tuple[Tuple[float], Tuple[float]]): ((x, y, z), (x, y, z, w)), attach pose of the target element, world_from_gripper
    """
    edge_start = np.array(edge_start).reshape((3,))
    edge_end = np.array(edge_end).reshape((3,))
    attach_point = np.array(attach_pose[0]).reshape((3,))
    preview_point = np.array(preview_point).reshape((3,))

    AB = edge_end - edge_start
    normal = AB / np.linalg.norm(AB)
    projection_Q = project_point_onto_plane(preview_point, attach_point, normal)
    z_direction = normalize(projection_Q - attach_point)
    y_direction = normalize(edge_end - edge_start)
    x_direction = np.cross(y_direction, z_direction)

    # pp.draw_point(projection_Q, size=0.25)

    x_direction = x_direction.reshape((3, 1))
    y_direction = y_direction.reshape((3, 1))
    z_direction = z_direction.reshape((3, 1))

    new_rotation_matrix = np.hstack((x_direction, y_direction, z_direction))
    new_rotation = R.from_matrix(new_rotation_matrix)

    pose = (attach_pose[0], tuple(new_rotation.as_quat().tolist()))
    new_pose_delta = pp.Pose(
        point=[0, 0, 0], euler=pp.Euler(0, np.random.uniform(-redirect_radius, redirect_radius), 0)
    )
    new_pose = pp.multiply(pose, new_pose_delta)

    return new_pose


def grasp_redirector_robot(
    edge_start: List[float],
    edge_end: List[float],
    attach_pose: Tuple[Tuple[float], Tuple[float]],
    base_pose_2d: List[float],
    reference_height: float = 0.5,
    redirect_radius: float = np.pi / 4,
) -> Tuple[Tuple[float], Tuple[float]]:
    """
    Generate a gripper pose based on the line between the robot and the attach point.

    Params:
        edge_start (List[float]): start point of the edge
        edge_end (List[float]): end point of the edge
        attach_pose (Tuple[Tuple[float], Tuple[float]]): ((x, y, z), (x, y, z, w)), attach pose of the target element, world_from_gripper
        base_pose_2d (List[float]): [x, y, yaw], pose 2d of the robot
        reference_height (float, 0.5): in order to generate the grasping direction, a height offset needs to be set
        redirect_radius (float, np.pi / 4): swing range in the positive direction of z axis

    Returns:
        pose (Tuple[Tuple[float], Tuple[float]]): ((x, y, z), (x, y, z, w)), attach pose of the target element, world_from_gripper
    """
    edge_start = np.array(edge_start).reshape((3,))
    edge_end = np.array(edge_end).reshape((3,))

    edge_vec = normalize(edge_end - edge_start)

    direction = np.array(attach_pose[0]) - np.array(base_pose_2d[:2] + [reference_height])
    direction = normalize(direction)

    proj = np.dot(direction, edge_vec) * edge_vec

    perp = normalize(direction - proj)

    z_direction = perp.reshape((3,))
    y_direction = edge_vec.reshape((3,))
    x_direction = np.cross(y_direction, z_direction)

    new_rotation_matrix = np.vstack((x_direction, y_direction, z_direction)).transpose()
    new_rotation = R.from_matrix(new_rotation_matrix)

    pose = (attach_pose[0], tuple(new_rotation.as_quat().tolist()))
    new_pose_delta = pp.Pose(
        point=[0, 0, 0], euler=pp.Euler(0, np.random.uniform(-redirect_radius, redirect_radius), 0)
    )
    new_pose = pp.multiply(pose, new_pose_delta)

    return new_pose


def grasp_sampler(
    base_position: np.ndarray,
    base_yaw: float,
    index: int,
    assembled: List[int],
    element_from_index: Dict,
    sample_range: float = 0.0,
    reachable_margin: float = 0.3,
    grasp_method: str = "robot",
    redirect_method: str = "robot",
) -> Tuple[Tuple[float], Tuple[float]]:
    """
    Sample a grasp pose given base position and base yaw.

    Params:
        base_position (np.ndarray): [x, y, z] position of robot base
        base_yaw (float): yaw (deg) of robot base
        index (int): index of current element
        assembled (List[int]): indices of assembled structure
        element_from_index ({index: Element}): dict of elements
        sample_range (float, 0.0): the distance to sample around the reference point
        reachable_margin (float, 0.3): the radius of the circle centered at the center of the bar
        grasp_method (str, "robot"): grasp generation method robot/cylinder
        redirect_method (str, "robot"): redirect method robot/preview/none(only for cylinder)

    Returns:
        pose (Tuple[Tuple[float], Tuple[float]]): ((x, y, z), (x, y, z, w)), gripper_from_body
    """

    cur_element: Element = element_from_index[index]
    target_edge = cur_element.axis_endpoints
    target_pose = cur_element.goal_pose

    # -------------------- grasp --------------------#
    if grasp_method == "robot":
        closest_point, alpha = closest_point_on_segment_to_ray(
            base_position[0], base_position[1], base_yaw, target_edge[0][:2], target_edge[1][:2]
        )
        start = np.array(target_edge[0])
        end = np.array(target_edge[1])
        edge_length = np.linalg.norm(end - start)

        sample_alpha = sample_range / edge_length
        reachable_alpha = reachable_margin / edge_length

        sample_alpha_min = max(0.0, alpha - sample_alpha)
        sample_alpha_max = min(1.0, alpha + sample_alpha)
        reachable_alpha_min = max(0.0, 0.5 - reachable_alpha)
        reachable_alpha_max = min(1.0, 0.5 + reachable_alpha)

        alpha_min = max(sample_alpha_min, reachable_alpha_min)
        alpha_max = min(sample_alpha_max, reachable_alpha_max)

        if alpha_min > alpha_max:
            # alpha_sample = random.uniform(reachable_alpha_min, reachable_alpha_max)
            if alpha >= reachable_alpha_max:
                alpha_sample = reachable_alpha_max
            if alpha <= reachable_alpha_min:
                alpha_sample = reachable_alpha_min
        else:
            alpha_sample = random.uniform(alpha_min, alpha_max)

        grasp_point_world_arr = (1 - alpha_sample) * start + alpha_sample * end
        attach_temp = (tuple(grasp_point_world_arr), (0, 0, 0, 1))

    else:
        start = np.array(target_edge[0])
        end = np.array(target_edge[1])
        edge_length = np.linalg.norm(end - start)

        safety_margin_length = edge_length / 2.0 - sample_range
        grasp_gen = pp.get_side_cylinder_grasps(cur_element.body, safety_margin_length=safety_margin_length)
        gripper_from_body = next(grasp_gen)
        return gripper_from_body

    # -------------------- redirect --------------------#
    if redirect_method == "robot":
        world_from_gripper = grasp_redirector_robot(
            target_edge[0], target_edge[1], attach_temp, base_position.tolist()[:2] + [base_yaw]
        )
        world_from_body = target_pose
        gripper_from_body = pp.multiply(pp.invert(world_from_gripper), world_from_body)

    elif redirect_method == "preview":
        preview_point = preview_point_calculation(assembled + [index], element_from_index)
        world_from_gripper = grasp_redirector_preview(target_edge[0], target_edge[1], attach_temp, preview_point)

        world_from_body = target_pose
        gripper_from_body = pp.multiply(pp.invert(world_from_gripper), world_from_body)

    else:
        raise RuntimeError("Redirect method error!")

    return gripper_from_body


if __name__ == "__main__":

    # Example values
    x, y, yaw = 0, 0, np.pi / 4  # Ray origin at (0, 0) and direction of 45 degrees
    start = [1, 0]
    end = [0, 6]

    closest_point_analytical, alpha_min_analytical = closest_point_on_segment_to_ray(x, y, yaw, start, end)
    print(closest_point_analytical, alpha_min_analytical)
