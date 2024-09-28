import numpy as np
import pybullet_planning as pp
from scipy.spatial.transform import Rotation as R
from collision import Element


def normalize(v):
    return v / np.linalg.norm(v)


def project_point_onto_plane(point, plane_point, normal):
    normal = normal / np.linalg.norm(normal)
    d = -plane_point.dot(normal)
    t = -(point.dot(normal) + d) / np.linalg.norm(normal) ** 2
    projection = point + t * normal
    return projection


def preview_point_calculation(frame, element_from_index):
    point = None
    for index in frame:
        element: Element = element_from_index[index]
        if point is None:
            point = element.axis_endpoints[0]
        else:
            point = np.vstack((point, element.axis_endpoints[0]))
        point = np.vstack((point, element.axis_endpoints[1]))
    return point.mean(axis=0).tolist()


def redirector(edge_start, edge_end, attach, target_point):
    edge_start = np.array(edge_start).reshape((3,))
    edge_end = np.array(edge_end).reshape((3,))
    attach_point = np.array(attach[0]).reshape((3,))
    target_point = np.array(target_point).reshape((3,))

    AB = edge_end - edge_start
    normal = AB / np.linalg.norm(AB)
    projection_Q = project_point_onto_plane(target_point, attach_point, normal)
    z_direction = normalize(projection_Q - attach_point)
    y_direction = normalize(edge_end - edge_start)
    x_direction = np.cross(y_direction, z_direction)

    pp.draw_point(projection_Q, size=0.25)

    x_direction = x_direction.reshape((3, 1))
    y_direction = y_direction.reshape((3, 1))
    z_direction = z_direction.reshape((3, 1))

    new_rotation_matrix = np.hstack((x_direction, y_direction, z_direction))
    new_rotation = R.from_matrix(new_rotation_matrix)

    pose = (attach[0], tuple(new_rotation.as_quat().tolist()))
    new_pose_delta = pp.Pose(point=[0, 0, 0], euler=pp.Euler(0, np.random.uniform(-np.pi/6, np.pi/6), 0))
    new_pose = pp.multiply(pose, new_pose_delta)

    return new_pose
