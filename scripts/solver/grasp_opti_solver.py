import itertools
import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Union

import pybullet_planning as pp

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import casadi as ca
import numpy as np
from robot.robot_setup import RobotSetup
from utils.util import HideOutput
from utils.collision import collision_info


def Point2Segment(p: ca.MX, start: np.ndarray, end: np.ndarray) -> ca.MX:
    A = start
    B = end
    P = p

    AB = B - A
    AP = P - A

    t = ca.dot(AP, AB) / ca.dot(AB, AB)
    t_clamped = ca.fmax(0, ca.fmin(1, t))
    Q = A + t_clamped * AB
    distance = ca.norm_2(P - Q)

    return distance


class GraspOptiSolver(object):

    MANIPULATOR_CONTROL_JOINT_NAMES = [
        "ur_arm_shoulder_pan_joint",
        "ur_arm_shoulder_lift_joint",
        "ur_arm_elbow_joint",
        "ur_arm_wrist_1_joint",
        "ur_arm_wrist_2_joint",
        "ur_arm_wrist_3_joint",
    ]

    MANIPULATOR_REDUCED_MODEL_JOINT_NAMES = [
        "ur_arm_base_link-base_fixed_joint",
        "ur_arm_shoulder_pan_joint",
        "ur_arm_shoulder_lift_joint",
        "ur_arm_elbow_joint",
        "ur_arm_wrist_1_joint",
        "ur_arm_wrist_2_joint",
        "ur_arm_wrist_3_joint",
        "ur_arm_wrist_3-flange",
        "ur_arm_flange-tool0",
        "tool0-bar_tcp_fixed_joint",
    ]

    BASE_CONTROL_JOINT_NAMES = []

    BASE_REDUCED_MODEL_JOINT_NAMES = [
        "base_footprint_joint",
        "top_plate_joint",
        "top_plate_front_joint",
        "arm_mount_joint",
        # "ur_arm_base_link-base_fixed_joint",
    ]

    def __init__(self, urdf_path: str, robots: List[RobotSetup]) -> None:
        self.urdf_path = urdf_path
        self.robots = robots
        self.num_robots = len(robots)
        self.CollisionInit()
        self.SolverInit()

    def eval(
        self, name: str, obj: ca.MX, sym: List[ca.MX], data: List[np.ndarray], verbose: bool = False
    ) -> np.ndarray:
        fn = ca.Function("fn", sym, [obj])
        if sym != []:
            fn_result = fn(*data).toarray()
        else:
            fn_result = fn()["o0"].toarray()
        if verbose:
            print(name, "\n", fn_result)
        return fn_result

    def GetInvertT(self, T: ca.MX) -> ca.MX:
        T_rot: ca.MX = T[:3, :3]
        T_trans: ca.MX = T[:3, 3]
        T_inv = ca.vertcat(
            ca.horzcat(
                T_rot.T,
                -T_rot.T @ T_trans,
            ),
            ca.MX([0, 0, 0, 1]).T,
        )
        return T_inv

    def GetJointPose(
        self,
        pose_2d: Union[List[float], np.ndarray, None],
        q: Union[List[float], np.ndarray],
        joint_name: str,
        frame: str = "world",
    ) -> np.ndarray:
        if joint_name == "base_joint":
            world_from_base: np.ndarray = np.eye(4)
            world_from_base[0, 3] = pose_2d[0]
            world_from_base[1, 3] = pose_2d[1]
            yaw = pose_2d[2]
            R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            world_from_base[:3, :3] = R_z
            world_from_joint: np.ndarray = world_from_base
        else:
            connect_from_joint: np.ndarray = self.connect_from_joint_dict[joint_name][0][0](
                q[: self.connect_from_joint_dict[joint_name][2]]
            ).toarray()
            base_from_joint: np.ndarray = self.base_from_connect @ connect_from_joint
            if frame == "local":
                return base_from_joint

            world_from_base: np.ndarray = np.eye(4)
            world_from_base[0, 3] = pose_2d[0]
            world_from_base[1, 3] = pose_2d[1]
            yaw = pose_2d[2]
            R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            world_from_base[:3, :3] = R_z
            world_from_joint: np.ndarray = world_from_base @ base_from_joint

        return world_from_joint

    def IKObjectiveFunction(self, T: ca.MX, T_tar: ca.MX) -> ca.MX:
        """
        Generate objective function seperately.

        Params:
            T (ca.MX): 4x4 matrix
            T_tar (ca.MX): 4x4 matrix

        Returns:
            ca.MX: objective value
        """
        err = 0
        for i in range(4):
            for j in range(4):
                err += (T[i, j] - T_tar[i, j]) ** 2
        return err

    def GraspIKObjectiveFunction(self, robot_idx: int) -> ca.MX:

        q = self.grasp_var_q_list[robot_idx]
        p = self.grasp_var_p_list[robot_idx]
        b = self.grasp_var_b_list[robot_idx]
        world_from_element = self.grasp_param_T_element_list[robot_idx]

        # **************************************************************************
        # compute pose of gripper
        # **************************************************************************

        # world_from_base
        world_from_base: ca.MX = ca.MX.eye(4)
        world_from_base[0, 3] = b[0]
        world_from_base[1, 3] = b[1]
        yaw = b[2]
        R_z = ca.vertcat(
            ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0), ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0), ca.horzcat(0, 0, 1)
        )
        world_from_base[:3, :3] = R_z

        # connect_from_gripper
        connect_from_gripper = RobotSetup.symbolic_forward(
            self.urdf_path,
            self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES,
            self.MANIPULATOR_CONTROL_JOINT_NAMES,
            q=q,
            output_type="matrix",
        )

        T_gripper = world_from_base @ self.base_from_connect @ connect_from_gripper

        # **************************************************************************
        # compute pose of gripper, target
        # **************************************************************************

        # element_from_gripper
        element_from_gripper = ca.MX.eye(4)
        element_from_gripper[0, 3] = p[0]
        element_from_gripper[1, 3] = p[1]

        pitch = p[2]
        R_y = ca.vertcat(
            ca.horzcat(ca.cos(pitch), 0, ca.sin(pitch)),
            ca.horzcat(0, 1, 0),
            ca.horzcat(-ca.sin(pitch), 0, ca.cos(pitch)),
        )
        element_from_gripper[:3, :3] = R_y

        # world_from_gripper, T_tar
        T_gripper_tar = world_from_element @ element_from_gripper

        obj = self.IKObjectiveFunction(T_gripper, T_gripper_tar)
        return obj

    def GraspObjectiveFunction(self, robot_idx: int, c: ca.MX) -> ca.MX:
        p = self.grasp_var_p_list[robot_idx]
        x = p[0]
        t = x - c  # TODO
        obj = ca.if_else(t > 0, t**2, 0)
        return obj

    def CollisionVisualize(self, enable: bool = True):
        q_val = pp.get_joint_positions(self.robot.robot, self.robot.arm_joints)
        b_val = pp.get_joint_positions(self.robot.robot, self.robot.base_joints)

        # **************************************************************************
        # compute collisions between robot and obstacles
        # **************************************************************************

        # world_from_base
        world_from_base: ca.MX = ca.MX.eye(4)
        world_from_base[0, 3] = self.grasp_var_base[0]
        world_from_base[1, 3] = self.grasp_var_base[1]
        yaw = self.grasp_var_base[2]
        R_z = ca.vertcat(
            ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0), ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0), ca.horzcat(0, 0, 1)
        )
        world_from_base[:3, :3] = R_z

        with pp.LockRenderer():

            # -------------------- loop: 遍历碰撞体 --------------------#
            for link_name in self.collision_info:
                infos = self.collision_info[link_name][0]
                mnt_joint = self.collision_info[link_name][1]
                radius = self.collision_info[link_name][2]

                if enable:
                    # -------------------- 获取joint位置 --------------------#
                    if mnt_joint == "base_joint":
                        world_from_joint = world_from_base
                    else:
                        base_from_joint = self.base_from_connect @ self.connect_from_joint_dict[mnt_joint][1]
                        world_from_joint = world_from_base @ base_from_joint
                    for info_id, info in enumerate(infos):
                        offset = info[0]
                        radius = info[1]
                        visual_id = info[2]
                        joint_from_sphere = np.eye(4)
                        joint_from_sphere[0, 3] = offset[0]
                        joint_from_sphere[1, 3] = offset[1]
                        joint_from_sphere[2, 3] = offset[2]
                        world_from_sphere = world_from_joint @ joint_from_sphere
                        p = ca.vertcat(world_from_sphere[0, 3], world_from_sphere[1, 3], world_from_sphere[2, 3])
                        center = self.eval("", p, [self.grasp_var_q, self.grasp_var_base], [q_val, b_val])
                        if visual_id == -1:
                            new_id = pp.create_sphere(radius, color=(1, 0, 0, 0.5))
                            pp.set_point(new_id, center)
                            self.collision_info[link_name][0][info_id][2] = new_id
                        else:
                            pp.set_point(visual_id, center)
                else:
                    for info_id, info in enumerate(infos):
                        visual_id = info[2]
                        if visual_id == -1:
                            pass
                        else:
                            pp.remove_body(visual_id)
                            self.collision_info[link_name][0][info_id][2] = -1

    def CollisionObjectiveFunction(
        self,
        index: int,
        robot_idx: int,
        assembled: List[int],
        element_from_index: dict,
    ) -> ca.MX:

        b = self.grasp_var_b_list[robot_idx]

        # **************************************************************************
        # compute collisions between robot and obstacles
        # **************************************************************************

        # world_from_base
        world_from_base: ca.MX = ca.MX.eye(4)
        world_from_base[0, 3] = b[0]
        world_from_base[1, 3] = b[1]
        yaw = b[2]
        R_z = ca.vertcat(
            ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0), ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0), ca.horzcat(0, 0, 1)
        )
        world_from_base[:3, :3] = R_z

        # -------------------- loop: 遍历碰撞体 --------------------#
        indices = [index] + assembled
        total_obj = 0
        for link_name in self.collision_info:
            infos = self.collision_info[link_name][0]
            mnt_joint = self.collision_info[link_name][1]
            weight = self.collision_info[link_name][2]
            # -------------------- 获取joint位置 --------------------#
            if mnt_joint == "base_joint":
                world_from_joint = world_from_base
            else:
                base_from_joint = self.base_from_connect @ self.connect_from_joint_dict[mnt_joint][1][robot_idx]
                world_from_joint = world_from_base @ base_from_joint
            for info in infos:
                offset = info[0]
                radius = info[1]
                eps = 0.01
                joint_from_sphere = np.eye(4)
                joint_from_sphere[0, 3] = offset[0]
                joint_from_sphere[1, 3] = offset[1]
                joint_from_sphere[2, 3] = offset[2]
                world_from_sphere = world_from_joint @ joint_from_sphere
                p = ca.vertcat(world_from_sphere[0, 3], world_from_sphere[1, 3], world_from_sphere[2, 3])
                elements = []
                if link_name == "gripper_link":
                    for element_index in assembled:
                        element: Element = element_from_index[element_index]
                        dist = Point2Segment(
                            p, np.array(element.axis_endpoints[0]), np.array(element.axis_endpoints[1])
                        )
                        obj = ca.if_else(dist - radius > eps, 1.0 / (dist - radius), 1.0 / eps)
                        total_obj += weight * obj
                        elements.append(element_index)
                else:
                    for element_index in indices:
                        element: Element = element_from_index[element_index]
                        dist = Point2Segment(
                            p, np.array(element.axis_endpoints[0]), np.array(element.axis_endpoints[1])
                        )
                        obj = ca.if_else(dist - radius > eps, 1.0 / (dist - radius), 1.0 / eps)
                        total_obj += weight * obj
                        elements.append(element_index)
        return total_obj

    def CollisionConstrain(
        self,
        index: int,
        robot_idx: int,
        assembled: List[int],
        element_from_index: dict,
    ) -> ca.MX:

        b = self.grasp_var_b_list[robot_idx]

        # **************************************************************************
        # compute collisions between robot and obstacles
        # **************************************************************************

        # world_from_base
        world_from_base: ca.MX = ca.MX.eye(4)
        world_from_base[0, 3] = b[0]
        world_from_base[1, 3] = b[1]
        yaw = b[2]
        R_z = ca.vertcat(
            ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0), ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0), ca.horzcat(0, 0, 1)
        )
        world_from_base[:3, :3] = R_z

        # -------------------- loop: 遍历碰撞体 --------------------#
        indices = [index] + assembled
        total_constrain = 0
        for link_name in self.collision_info:
            infos = self.collision_info[link_name][0]
            mnt_joint = self.collision_info[link_name][1]
            # -------------------- 获取joint位置 --------------------#
            if mnt_joint == "base_joint":
                world_from_joint = world_from_base
            else:
                base_from_joint = self.base_from_connect @ self.connect_from_joint_dict[mnt_joint][1][robot_idx]
                world_from_joint = world_from_base @ base_from_joint
            for info in infos:
                offset = info[0]
                radius = info[1]
                eps = 0.01
                joint_from_sphere = np.eye(4)
                joint_from_sphere[0, 3] = offset[0]
                joint_from_sphere[1, 3] = offset[1]
                joint_from_sphere[2, 3] = offset[2]
                world_from_sphere = world_from_joint @ joint_from_sphere
                p = ca.vertcat(world_from_sphere[0, 3], world_from_sphere[1, 3], world_from_sphere[2, 3])
                elements = []
                if link_name == "gripper_link":
                    for element_index in assembled:
                        element: Element = element_from_index[element_index]
                        dist = Point2Segment(
                            p, np.array(element.axis_endpoints[0]), np.array(element.axis_endpoints[1])
                        )
                        constrain = ca.if_else(dist - radius > eps, 0, -1)
                        total_constrain += constrain
                        elements.append(element_index)
                else:
                    for element_index in indices:
                        element: Element = element_from_index[element_index]
                        dist = Point2Segment(
                            p, np.array(element.axis_endpoints[0]), np.array(element.axis_endpoints[1])
                        )
                        constrain = ca.if_else(dist - radius > eps, 0, -1)
                        total_constrain += constrain
                        elements.append(element_index)
        return total_constrain

    def SelfCollisionObjectiveFunction(self, robot_idx: int) -> ca.MX:

        b = self.grasp_var_b_list[robot_idx]

        # **************************************************************************
        # compute collisions between robot links
        # **************************************************************************

        # world_from_base
        world_from_base: ca.MX = ca.MX.eye(4)
        world_from_base[0, 3] = b[0]
        world_from_base[1, 3] = b[1]
        yaw = b[2]
        R_z = ca.vertcat(
            ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0), ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0), ca.horzcat(0, 0, 1)
        )
        world_from_base[:3, :3] = R_z

        # -------------------- loop: 遍历碰撞对 --------------------#
        total_obj = 0
        eps = 0.01
        for collision_pair in self.self_collision_pairs:
            link_1_name, link_2_name = collision_pair

            # -------------------- 获取 link 1 碰撞体信息 --------------------#
            link_1_infos = self.collision_info[link_1_name][0]
            link_1_mnt_joint = self.collision_info[link_1_name][1]
            link_1_weight = self.collision_info[link_1_name][2]

            # -------------------- 获取 link 2 碰撞体信息 --------------------#
            link_2_infos = self.collision_info[link_2_name][0]
            link_2_mnt_joint = self.collision_info[link_2_name][1]
            link_2_weight = self.collision_info[link_2_name][2]

            # -------------------- 获取 joint 1 位置 --------------------#
            if link_1_mnt_joint == "base_joint":
                world_from_joint_1 = world_from_base
            else:
                base_from_joint_1 = (
                    self.base_from_connect @ self.connect_from_joint_dict[link_1_mnt_joint][1][robot_idx]
                )
                world_from_joint_1 = world_from_base @ base_from_joint_1

            # -------------------- 获取 joint 2 位置 --------------------#
            if link_2_mnt_joint == "base_joint":
                world_from_joint_2 = world_from_base
            else:
                base_from_joint_2 = (
                    self.base_from_connect @ self.connect_from_joint_dict[link_2_mnt_joint][1][robot_idx]
                )
                world_from_joint_2 = world_from_base @ base_from_joint_2

            # -------------------- 遍历碰撞体 --------------------#
            for info_pair in list(itertools.product(link_1_infos, link_2_infos)):
                info_1, info_2 = info_pair

                # -------------------- sphere 1 --------------------#
                offset_1 = info_1[0]
                radius_1 = info_1[1]
                joint_from_sphere_1 = np.eye(4)
                joint_from_sphere_1[0, 3] = offset_1[0]
                joint_from_sphere_1[1, 3] = offset_1[1]
                joint_from_sphere_1[2, 3] = offset_1[2]
                world_from_sphere_1 = world_from_joint_1 @ joint_from_sphere_1
                p_1 = ca.vertcat(world_from_sphere_1[0, 3], world_from_sphere_1[1, 3], world_from_sphere_1[2, 3])

                # -------------------- sphere 2 --------------------#
                offset_2 = info_2[0]
                radius_2 = info_2[1]
                joint_from_sphere_2 = np.eye(4)
                joint_from_sphere_2[0, 3] = offset_2[0]
                joint_from_sphere_2[1, 3] = offset_2[1]
                joint_from_sphere_2[2, 3] = offset_2[2]
                world_from_sphere_2 = world_from_joint_2 @ joint_from_sphere_2
                p_2 = ca.vertcat(world_from_sphere_2[0, 3], world_from_sphere_2[1, 3], world_from_sphere_2[2, 3])

                # -------------------- 计算 objective function --------------------#
                dist = ca.norm_2(p_1 - p_2)
                sum_radius = radius_1 + radius_2
                obj = ca.if_else(dist - sum_radius > eps, 1.0 / (dist - sum_radius), 1.0 / eps)
                total_obj += min(link_1_weight, link_2_weight) * obj

        return total_obj

    def SelfCollisionConstrain(self, robot_idx: int) -> ca.MX:

        b = self.grasp_var_b_list[robot_idx]

        # **************************************************************************
        # compute collisions between robot links
        # **************************************************************************

        # world_from_base
        world_from_base: ca.MX = ca.MX.eye(4)
        world_from_base[0, 3] = b[0]
        world_from_base[1, 3] = b[1]
        yaw = b[2]
        R_z = ca.vertcat(
            ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0), ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0), ca.horzcat(0, 0, 1)
        )
        world_from_base[:3, :3] = R_z

        # -------------------- loop: 遍历碰撞对 --------------------#
        total_constrain = 0
        eps = 0.01
        for collision_pair in self.self_collision_pairs:
            link_1_name, link_2_name = collision_pair

            # -------------------- 获取 link 1 碰撞体信息 --------------------#
            link_1_infos = self.collision_info[link_1_name][0]
            link_1_mnt_joint = self.collision_info[link_1_name][1]
            link_1_weight = self.collision_info[link_1_name][2]

            # -------------------- 获取 link 2 碰撞体信息 --------------------#
            link_2_infos = self.collision_info[link_2_name][0]
            link_2_mnt_joint = self.collision_info[link_2_name][1]
            link_2_weight = self.collision_info[link_2_name][2]

            # -------------------- 获取 joint 1 位置 --------------------#
            if link_1_mnt_joint == "base_joint":
                world_from_joint_1 = world_from_base
            else:
                base_from_joint_1 = (
                    self.base_from_connect @ self.connect_from_joint_dict[link_1_mnt_joint][1][robot_idx]
                )
                world_from_joint_1 = world_from_base @ base_from_joint_1

            # -------------------- 获取 joint 2 位置 --------------------#
            if link_2_mnt_joint == "base_joint":
                world_from_joint_2 = world_from_base
            else:
                base_from_joint_2 = (
                    self.base_from_connect @ self.connect_from_joint_dict[link_2_mnt_joint][1][robot_idx]
                )
                world_from_joint_2 = world_from_base @ base_from_joint_2

            # -------------------- 遍历碰撞体 --------------------#
            for info_pair in list(itertools.product(link_1_infos, link_2_infos)):
                info_1, info_2 = info_pair

                # -------------------- sphere 1 --------------------#
                offset_1 = info_1[0]
                radius_1 = info_1[1]
                joint_from_sphere_1 = np.eye(4)
                joint_from_sphere_1[0, 3] = offset_1[0]
                joint_from_sphere_1[1, 3] = offset_1[1]
                joint_from_sphere_1[2, 3] = offset_1[2]
                world_from_sphere_1 = world_from_joint_1 @ joint_from_sphere_1
                p_1 = ca.vertcat(world_from_sphere_1[0, 3], world_from_sphere_1[1, 3], world_from_sphere_1[2, 3])

                # -------------------- sphere 2 --------------------#
                offset_2 = info_2[0]
                radius_2 = info_2[1]
                joint_from_sphere_2 = np.eye(4)
                joint_from_sphere_2[0, 3] = offset_2[0]
                joint_from_sphere_2[1, 3] = offset_2[1]
                joint_from_sphere_2[2, 3] = offset_2[2]
                world_from_sphere_2 = world_from_joint_2 @ joint_from_sphere_2
                p_2 = ca.vertcat(world_from_sphere_2[0, 3], world_from_sphere_2[1, 3], world_from_sphere_2[2, 3])

                # -------------------- 计算 objective function --------------------#
                dist = ca.norm_2(p_1 - p_2)
                sum_radius = radius_1 + radius_2
                constrain = ca.if_else(dist - sum_radius > eps, 0, -1)
                total_constrain += min(link_1_weight, link_2_weight) * constrain

        return total_constrain

    def RobotCollisionObjectiveFunction(self, robot_idx_1: int, robot_idx_2: int) -> ca.MX:

        b_1 = self.grasp_var_b_list[robot_idx_1]
        b_2 = self.grasp_var_b_list[robot_idx_2]

        # **************************************************************************
        # compute collisions between robot links
        # **************************************************************************

        # world_from_base
        world_from_base_1: ca.MX = ca.MX.eye(4)
        world_from_base_1[0, 3] = b_1[0]
        world_from_base_1[1, 3] = b_1[1]
        yaw_1 = b_1[2]
        R_z_1 = ca.vertcat(
            ca.horzcat(ca.cos(yaw_1), -ca.sin(yaw_1), 0),
            ca.horzcat(ca.sin(yaw_1), ca.cos(yaw_1), 0),
            ca.horzcat(0, 0, 1),
        )
        world_from_base_1[:3, :3] = R_z_1

        world_from_base_2: ca.MX = ca.MX.eye(4)
        world_from_base_2[0, 3] = b_2[0]
        world_from_base_2[1, 3] = b_2[1]
        yaw_2 = b_2[2]
        R_z_2 = ca.vertcat(
            ca.horzcat(ca.cos(yaw_2), -ca.sin(yaw_2), 0),
            ca.horzcat(ca.sin(yaw_2), ca.cos(yaw_2), 0),
            ca.horzcat(0, 0, 1),
        )
        world_from_base_2[:3, :3] = R_z_2

        # -------------------- loop: 遍历碰撞对 --------------------#
        total_obj = 0
        eps = 0.01
        for collision_pair in self.robot_collision_pairs:
            link_1_name, link_2_name = collision_pair

            # -------------------- 获取 link 1 碰撞体信息 --------------------#
            link_1_infos = self.collision_info[link_1_name][0]
            link_1_mnt_joint = self.collision_info[link_1_name][1]
            link_1_weight = self.collision_info[link_1_name][2]

            # -------------------- 获取 link 2 碰撞体信息 --------------------#
            link_2_infos = self.collision_info[link_2_name][0]
            link_2_mnt_joint = self.collision_info[link_2_name][1]
            link_2_weight = self.collision_info[link_2_name][2]

            # -------------------- 获取 joint 1 位置 --------------------#
            if link_1_mnt_joint == "base_joint":
                world_from_joint_1 = world_from_base_1
            else:
                base_from_joint_1 = (
                    self.base_from_connect @ self.connect_from_joint_dict[link_1_mnt_joint][1][robot_idx_1]
                )
                world_from_joint_1 = world_from_base_1 @ base_from_joint_1

            # -------------------- 获取 joint 2 位置 --------------------#
            if link_2_mnt_joint == "base_joint":
                world_from_joint_2 = world_from_base_2
            else:
                base_from_joint_2 = (
                    self.base_from_connect @ self.connect_from_joint_dict[link_2_mnt_joint][1][robot_idx_2]
                )
                world_from_joint_2 = world_from_base_2 @ base_from_joint_2

            # -------------------- 遍历碰撞体 --------------------#
            for info_pair in list(itertools.product(link_1_infos, link_2_infos)):
                info_1, info_2 = info_pair

                # -------------------- sphere 1 --------------------#
                offset_1 = info_1[0]
                radius_1 = info_1[1]
                joint_from_sphere_1 = np.eye(4)
                joint_from_sphere_1[0, 3] = offset_1[0]
                joint_from_sphere_1[1, 3] = offset_1[1]
                joint_from_sphere_1[2, 3] = offset_1[2]
                world_from_sphere_1 = world_from_joint_1 @ joint_from_sphere_1
                p_1 = ca.vertcat(world_from_sphere_1[0, 3], world_from_sphere_1[1, 3], world_from_sphere_1[2, 3])

                # -------------------- sphere 2 --------------------#
                offset_2 = info_2[0]
                radius_2 = info_2[1]
                joint_from_sphere_2 = np.eye(4)
                joint_from_sphere_2[0, 3] = offset_2[0]
                joint_from_sphere_2[1, 3] = offset_2[1]
                joint_from_sphere_2[2, 3] = offset_2[2]
                world_from_sphere_2 = world_from_joint_2 @ joint_from_sphere_2
                p_2 = ca.vertcat(world_from_sphere_2[0, 3], world_from_sphere_2[1, 3], world_from_sphere_2[2, 3])

                # -------------------- 计算 objective function --------------------#
                dist = ca.norm_2(p_1 - p_2)
                sum_radius = radius_1 + radius_2
                obj = ca.if_else(dist - sum_radius > eps, 1.0 / (dist - sum_radius), 1.0 / eps)
                total_obj += min(link_1_weight, link_2_weight) * obj

        return total_obj

    def RobotCollisionConstrain(self, robot_idx_1: int, robot_idx_2: int) -> ca.MX:

        b_1 = self.grasp_var_b_list[robot_idx_1]
        b_2 = self.grasp_var_b_list[robot_idx_2]

        # **************************************************************************
        # compute collisions between robot links
        # **************************************************************************

        # world_from_base
        world_from_base_1: ca.MX = ca.MX.eye(4)
        world_from_base_1[0, 3] = b_1[0]
        world_from_base_1[1, 3] = b_1[1]
        yaw_1 = b_1[2]
        R_z_1 = ca.vertcat(
            ca.horzcat(ca.cos(yaw_1), -ca.sin(yaw_1), 0),
            ca.horzcat(ca.sin(yaw_1), ca.cos(yaw_1), 0),
            ca.horzcat(0, 0, 1),
        )
        world_from_base_1[:3, :3] = R_z_1

        world_from_base_2: ca.MX = ca.MX.eye(4)
        world_from_base_2[0, 3] = b_2[0]
        world_from_base_2[1, 3] = b_2[1]
        yaw_2 = b_2[2]
        R_z_2 = ca.vertcat(
            ca.horzcat(ca.cos(yaw_2), -ca.sin(yaw_2), 0),
            ca.horzcat(ca.sin(yaw_2), ca.cos(yaw_2), 0),
            ca.horzcat(0, 0, 1),
        )
        world_from_base_2[:3, :3] = R_z_2

        # -------------------- loop: 遍历碰撞对 --------------------#
        total_constrain = 0
        eps = 0.01
        for collision_pair in self.robot_collision_pairs:
            link_1_name, link_2_name = collision_pair

            # -------------------- 获取 link 1 碰撞体信息 --------------------#
            link_1_infos = self.collision_info[link_1_name][0]
            link_1_mnt_joint = self.collision_info[link_1_name][1]
            link_1_weight = self.collision_info[link_1_name][2]

            # -------------------- 获取 link 2 碰撞体信息 --------------------#
            link_2_infos = self.collision_info[link_2_name][0]
            link_2_mnt_joint = self.collision_info[link_2_name][1]
            link_2_weight = self.collision_info[link_2_name][2]

            # -------------------- 获取 joint 1 位置 --------------------#
            if link_1_mnt_joint == "base_joint":
                world_from_joint_1 = world_from_base_1
            else:
                base_from_joint_1 = (
                    self.base_from_connect @ self.connect_from_joint_dict[link_1_mnt_joint][1][robot_idx_1]
                )
                world_from_joint_1 = world_from_base_1 @ base_from_joint_1

            # -------------------- 获取 joint 2 位置 --------------------#
            if link_2_mnt_joint == "base_joint":
                world_from_joint_2 = world_from_base_2
            else:
                base_from_joint_2 = (
                    self.base_from_connect @ self.connect_from_joint_dict[link_2_mnt_joint][1][robot_idx_2]
                )
                world_from_joint_2 = world_from_base_2 @ base_from_joint_2

            # -------------------- 遍历碰撞体 --------------------#
            for info_pair in list(itertools.product(link_1_infos, link_2_infos)):
                info_1, info_2 = info_pair

                # -------------------- sphere 1 --------------------#
                offset_1 = info_1[0]
                radius_1 = info_1[1]
                joint_from_sphere_1 = np.eye(4)
                joint_from_sphere_1[0, 3] = offset_1[0]
                joint_from_sphere_1[1, 3] = offset_1[1]
                joint_from_sphere_1[2, 3] = offset_1[2]
                world_from_sphere_1 = world_from_joint_1 @ joint_from_sphere_1
                p_1 = ca.vertcat(world_from_sphere_1[0, 3], world_from_sphere_1[1, 3], world_from_sphere_1[2, 3])

                # -------------------- sphere 2 --------------------#
                offset_2 = info_2[0]
                radius_2 = info_2[1]
                joint_from_sphere_2 = np.eye(4)
                joint_from_sphere_2[0, 3] = offset_2[0]
                joint_from_sphere_2[1, 3] = offset_2[1]
                joint_from_sphere_2[2, 3] = offset_2[2]
                world_from_sphere_2 = world_from_joint_2 @ joint_from_sphere_2
                p_2 = ca.vertcat(world_from_sphere_2[0, 3], world_from_sphere_2[1, 3], world_from_sphere_2[2, 3])

                # -------------------- 计算 objective function --------------------#
                dist = ca.norm_2(p_1 - p_2)
                sum_radius = radius_1 + radius_2
                constrain = ca.if_else(dist - sum_radius > eps, 0, -1)
                total_constrain += min(link_1_weight, link_2_weight) * constrain

        return total_constrain

    def CollisionInit(self):
        # {link_name: [infos: [offset, radius, visual_id], joint_name, weight]}
        self.collision_info = collision_info
        keys = list(self.collision_info.keys())
        keys_combinations = list(itertools.combinations(keys, 2))
        for collision_pair in self.robots[0].disabled_collisions:
            link_1_name = pp.get_link_name(self.robots[0].robot, collision_pair[0])
            link_2_name = pp.get_link_name(self.robots[0].robot, collision_pair[1])
            pair = (link_1_name, link_2_name)
            idx = self.FindCollisionPair(keys_combinations, pair)
            if idx != -1:
                keys_combinations.remove(keys_combinations[idx])
        pair = ("ur_arm_wrist_3_link", "gripper_link")
        idx = self.FindCollisionPair(keys_combinations, pair)
        if idx != -1:
            keys_combinations.remove(keys_combinations[idx])

        self.self_collision_pairs = keys_combinations

        self.robot_collision_pairs = list(itertools.product(keys, keys))
        self.robot_collision_pairs = [("base_link", "base_link")]

    def FindCollisionPair(self, collision_list: List[Tuple[str]], collision_pair: Tuple[str]) -> int:
        sorted_pair = tuple(sorted(collision_pair))
        sorted_pair_list = [tuple(sorted(pair)) for pair in collision_list]
        if sorted_pair in sorted_pair_list:
            idx = sorted_pair_list.index(sorted_pair)
        else:
            idx = -1
        return idx

    def SolverInit(self):

        # **************************************************************************
        # params init
        # **************************************************************************

        self.Nq = len(self.MANIPULATOR_CONTROL_JOINT_NAMES)
        self.Np = 3
        self.Nb = 3

        # **************************************************************************
        # solver for grasp
        # **************************************************************************

        self.grasp_solver = ca.Opti()

        # -------------------- set variables --------------------#
        self.grasp_var_q_list = [self.grasp_solver.variable(self.Nq) for _ in range(self.num_robots)]
        self.grasp_var_p_list = [self.grasp_solver.variable(self.Np) for _ in range(self.num_robots)]
        self.grasp_var_b_list = [self.grasp_solver.variable(self.Nb) for _ in range(self.num_robots)]

        # -------------------- set parameters --------------------#
        self.grasp_param_T_element_list = [
            self.grasp_solver.parameter(4, 4) for _ in range(self.num_robots)
        ]  # world_from_element
        self.grasp_param_c = self.grasp_solver.parameter(1)

        # -------------------- get transformation matices --------------------#
        self.connect_from_j6_fn = RobotSetup.symbolic_forward(
            self.urdf_path, self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES, self.MANIPULATOR_CONTROL_JOINT_NAMES
        )
        self.connect_from_joint_dict = {}
        for joint_idx, joint_name in enumerate(self.MANIPULATOR_CONTROL_JOINT_NAMES):
            end_idx = self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES.index(joint_name) + 1
            fk_fn_list = [
                RobotSetup.symbolic_forward(
                    self.urdf_path,
                    self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES[:end_idx],
                    self.MANIPULATOR_CONTROL_JOINT_NAMES[: joint_idx + 1],
                    q=q,
                )
                for q in self.grasp_var_q_list
            ]
            fk_mat_list = [
                RobotSetup.symbolic_forward(
                    self.urdf_path,
                    self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES[:end_idx],
                    self.MANIPULATOR_CONTROL_JOINT_NAMES[: joint_idx + 1],
                    q=q,
                    output_type="matrix",
                )
                for q in self.grasp_var_q_list
            ]
            self.connect_from_joint_dict[joint_name] = (fk_fn_list, fk_mat_list, joint_idx + 1)

        base_from_connect_sym = RobotSetup.symbolic_forward(
            self.urdf_path, self.BASE_REDUCED_MODEL_JOINT_NAMES, self.BASE_CONTROL_JOINT_NAMES, output_type="matrix"
        )
        self.base_from_connect = self.eval("base_from_connect", base_from_connect_sym, [], [])

        # -------------------- get matrix --------------------#
        self.grasp_connect_from_gripper_list = [
            RobotSetup.symbolic_forward(
                self.urdf_path,
                self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES,
                self.MANIPULATOR_CONTROL_JOINT_NAMES,
                q=q,
                output_type="matrix",
            )
            for q in self.grasp_var_q_list
        ]
        self.grasp_gripper_from_connect_list = [self.GetInvertT(mat) for mat in self.grasp_connect_from_gripper_list]

        # -------------------- ik obj: grasp_var_q, grasp_var_p, grasp_var_base, grasp_param_T_element --------------------#
        self.ik_obj_list = [self.GraspIKObjectiveFunction(robot_idx) for robot_idx in range(self.num_robots)]

        # -------------------- grasp obj: grasp_var_p, grasp_param_c --------------------#
        self.grasp_obj_list = [
            self.GraspObjectiveFunction(robot_idx, self.grasp_param_c) for robot_idx in range(self.num_robots)
        ]

        # -------------------- self collision obj: grasp_var_q, grasp_var_base --------------------#
        self.self_collision_obj_list = [
            self.SelfCollisionObjectiveFunction(robot_idx) for robot_idx in range(self.num_robots)
        ]

        # -------------------- self collision constrain >= 0 --------------------#
        self.self_collision_constrain_list = [
            self.SelfCollisionConstrain(robot_idx) for robot_idx in range(self.num_robots)
        ]

    def forward(self, q: Union[np.ndarray, List[float]]) -> np.ndarray:

        if isinstance(q, np.ndarray):
            q = q.tolist()

        T = np.array(self.connect_from_j6_fn(q))

        return T

    def solve(
        self,
        indices: List[int],
        element_from_index: dict,
        assembled: Union[List[int], None] = None,
        verbose: bool = False,
    ) -> Union[Union[np.ndarray, List[float], None], Tuple[Union[np.ndarray, List[float], None], bool]]:
        """
        Calculate grasp solution.

        Params:
            indices ([int]): indices of elements
            element_from_index ({index: Element}): dict of elements
            assembled ([index], None): indices of assembled elements
            verbose (bool, False): whether to print the solve status

        Returns:
            q (np.ndarray | [float] | None): joint conf to the target pose
            success (bool, [optional]): solve status
        """
        elements: List[Element] = [element_from_index[index] for index in indices]
        world_form_element_list = [pp.pose_transformation.tform_from_pose(element.goal_pose) for element in elements]

        # the y axis needs to be along the direction of the element
        rotate_pose_list = [
            world_form_element
            @ pp.pose_transformation.tform_from_pose(pp.Pose(point=[0, 0, 0], euler=pp.Euler(np.pi / 2, 0, 0)))
            for world_form_element in world_form_element_list
        ]

        temp_grasp_solver = self.grasp_solver

        # **************************************************************************
        # 设置目标函数
        # **************************************************************************

        obj = 0
        for robot_idx, index in enumerate(indices):
            # 优化目标函数中必须有collision的部分，否则优化时间和优化方向难以确定，并且优化结果不一定满足约束
            # 设置了gripper的collision，gripper只需要考虑与其他element的碰撞
            # obj = self.CollisionObjectiveFunction(index, assembled, element_from_index)
            temp_indices = indices.copy()
            temp_indices.remove(index)
            temp_assembled = assembled + temp_indices
            obj += (
                0.1 * self.ik_obj_list[robot_idx]
                + 0.1 * self.grasp_obj_list[robot_idx]
                + 0.1 * self.self_collision_obj_list[robot_idx]
                + 5 * self.CollisionObjectiveFunction(index, robot_idx, temp_assembled, element_from_index)
            )

        for robot_pair in itertools.combinations(range(self.num_robots), 2):
            obj += 10 * self.RobotCollisionObjectiveFunction(robot_pair[0], robot_pair[1])

        temp_grasp_solver.minimize(obj)

        # **************************************************************************
        # 设置约束
        # **************************************************************************

        for robot_idx, index in enumerate(indices):
            temp_indices = indices.copy()
            temp_indices.remove(index)
            temp_assembled = assembled + temp_indices

            q = self.grasp_var_q_list[robot_idx]
            p = self.grasp_var_p_list[robot_idx]
            b = self.grasp_var_b_list[robot_idx]
            ik_obj = self.ik_obj_list[robot_idx]
            self_collision_constrain = self.self_collision_constrain_list[robot_idx]

            # 关节角约束
            var_q_lb = np.pi + ca.vec(q)
            var_q_ub = np.pi - ca.vec(q)

            # p := (x, y, theta) 的约束
            var_p_x_lb = 0.001 + p[0]
            var_p_x_ub = 0.001 - p[0]
            var_p_y_lb = self.grasp_param_c + p[1]
            var_p_y_ub = self.grasp_param_c - p[1]
            var_p_theta_lb = np.pi + p[2]
            var_p_theta_ub = np.pi - p[2]

            # base := (x, y, yaw) 的约束
            var_base_yaw_lb = np.pi + b[2]
            var_base_yaw_ub = np.pi - b[2]

            # grasp 约束
            ik_lb = 0.000001 + ik_obj
            ik_ub = 0.000001 - ik_obj

            temp_grasp_solver.subject_to(var_q_lb >= 0)
            temp_grasp_solver.subject_to(var_q_ub >= 0)

            temp_grasp_solver.subject_to(var_p_x_lb >= 0)
            temp_grasp_solver.subject_to(var_p_x_ub >= 0)
            temp_grasp_solver.subject_to(var_p_y_lb >= 0)
            temp_grasp_solver.subject_to(var_p_y_ub >= 0)
            temp_grasp_solver.subject_to(var_p_theta_lb >= 0)
            temp_grasp_solver.subject_to(var_p_theta_ub >= 0)

            temp_grasp_solver.subject_to(var_base_yaw_lb >= 0)
            temp_grasp_solver.subject_to(var_base_yaw_ub >= 0)

            temp_grasp_solver.subject_to(ik_lb >= 0)
            temp_grasp_solver.subject_to(ik_ub >= 0)

            temp_grasp_solver.subject_to(self_collision_constrain >= 0)

            # -------------------- collision约束 --------------------#
            # 优化约束中必须有collision的部分，否则可能会发生碰撞
            collision_constrain = self.CollisionConstrain(index, robot_idx, temp_assembled, element_from_index)
            temp_grasp_solver.subject_to(collision_constrain >= 0)

        for robot_pair in itertools.combinations(range(self.num_robots), 2):
            constrain = self.RobotCollisionConstrain(robot_pair[0], robot_pair[1])
            temp_grasp_solver.subject_to(constrain >= 0)

        # **************************************************************************
        # 设置参数值
        # **************************************************************************

        temp_grasp_solver.set_value(self.grasp_param_c, 0.35)
        for robot_idx, index in enumerate(indices):
            temp_grasp_solver.set_value(self.grasp_param_T_element_list[robot_idx], rotate_pose_list[robot_idx])

        # **************************************************************************
        # 设置求解器参数
        # **************************************************************************
        p_opts = {"expand": True, "print_time": 0}
        if verbose:
            s_opts = {"max_iter": 10000}
        else:
            s_opts = {"max_iter": 10000, "print_level": 0}
        self.grasp_solver.solver("ipopt", p_opts, s_opts)

        attempts = 0
        while True:
            try:
                # **************************************************************************
                # 设置初始值
                # **************************************************************************

                for robot_idx, index in enumerate(indices):
                    q_init = np.random.uniform(-np.pi, np.pi, self.Nq)
                    p_init = np.concatenate([np.random.uniform(-0.35, 0.35, 2), np.random.uniform(-np.pi, np.pi, 1)])
                    base_init = np.concatenate([np.random.uniform(10, 10, 2), np.random.uniform(-np.pi, np.pi, 1)])
                    temp_grasp_solver.set_initial(self.grasp_var_q_list[robot_idx], q_init)
                    temp_grasp_solver.set_initial(self.grasp_var_p_list[robot_idx], p_init)
                    temp_grasp_solver.set_initial(self.grasp_var_b_list[robot_idx], base_init)

                if verbose:
                    grasp_solution = temp_grasp_solver.solve()
                else:
                    with HideOutput():
                        grasp_solution = temp_grasp_solver.solve()

                # for robot_idx, index in enumerate(indices):
                #     temp_indices = indices.copy()
                #     temp_indices.remove(index)
                #     temp_assembled = assembled + temp_indices
                #     collision_constrain = self.CollisionConstrain(index, robot_idx, temp_assembled, element_from_index)
                #     # self.eval(
                #     #     "collision_constrain",
                #     #     collision_constrain,
                #     #     [self.grasp_var_b_list[robot_idx], self.grasp_var_q_list[robot_idx]],
                #     #     [
                #     #         grasp_solution.value(self.grasp_var_b_list[robot_idx]),
                #     #         grasp_solution.value(self.grasp_var_q_list[robot_idx]),
                #     #     ],
                #     #     verbose=True,
                #     # )

                # for robot_pair in itertools.combinations(range(self.num_robots), 2):
                #     constrain = self.RobotCollisionConstrain(robot_pair[0], robot_pair[1])
                #     # self.eval(
                #     #     "robot_collision_constrain",
                #     #     constrain,
                #     #     [
                #     #         self.grasp_var_b_list[robot_pair[0]],
                #     #         self.grasp_var_b_list[robot_pair[1]],
                #     #         self.grasp_var_q_list[robot_pair[0]],
                #     #         self.grasp_var_q_list[robot_pair[1]],
                #     #     ],
                #     #     [
                #     #         grasp_solution.value(self.grasp_var_b_list[robot_pair[0]]),
                #     #         grasp_solution.value(self.grasp_var_b_list[robot_pair[1]]),
                #     #         grasp_solution.value(self.grasp_var_q_list[robot_pair[0]]),
                #     #         grasp_solution.value(self.grasp_var_q_list[robot_pair[1]]),
                #     #     ],
                #     #     verbose=True,
                #     # )

                # TODO
                return [
                    grasp_solution.value(self.grasp_var_q_list[robot_idx]) for robot_idx in range(self.num_robots)
                ], [grasp_solution.value(self.grasp_var_b_list[robot_idx]) for robot_idx in range(self.num_robots)]

            except Exception as e:
                if verbose:
                    print(e)
                attempts += 1
                if attempts >= 20:
                    if verbose:
                        print("Max attempts reached. Exiting.")
                    break  # 达到最大重试次数时退出

        return None, None


if __name__ == "__main__":
    import os
    import sys
    import time
    from functools import partial

    import casadi as ca
    import numpy as np
    import pybullet as p
    import pybullet_planning as pp

    HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(HERE)

    import utils.load_multi_tangent as load_multi_tangent
    from multi_tangent.collision import create_collision_bodies
    from multi_tangent.convert import flatten_list
    from solver.grasp_opti_solver import GraspOptiSolver
    from solver.ik_pinocchio_solver import PinocchioSolver
    from utils.collision import Element, create_couplers, init_pb
    from utils.params import *
    from utils.parse import parse_mt_geometric

    np.set_printoptions(precision=3, suppress=True)

    urdf_path = "/home/jeong/summer_research/eth_ws/src/husky_assembly/data/husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf"
    # solver = PinocchioSolver(urdf_path)
    # opti_ik_solver = GraspOptiSolver(urdf_path)

    # **************************************************************************
    # link forward
    # **************************************************************************

    # print("\n==================== link forward ====================\n")

    # pin_ik_solver_relative = partial(solver.ik, tip_name="ur_arm_tool0", link=True)

    # target_pose = np.array([[0, 0, 1, 0.7], [0, 1, 0, 0.25], [-1, 0, 0, 0.5], [0, 0, 0, 1]])

    # q_pin = pin_ik_solver_relative(
    #     target_pose, qinit=np.array([-1.29891436, 0.3618943, -1.79107962, -1.71236292, -1.84276683, 1.57065429])
    # )

    # print("> ik solver: q_pin")
    # print(q_pin)
    # print()

    # print("> forward: pose real")
    # print(target_pose)
    # print()

    # pose = solver.forward(q_pin, "ur_arm_tool0", link=True, relative_output=True)
    # print("> forward: pose pinocchio")
    # print(pose)
    # print()

    # pose = opti_ik_solver.forward(q_pin)
    # print("> forward: pose opti")
    # print(pose)
    # print()

    # **************************************************************************
    # joint forward
    # **************************************************************************

    # print("\n==================== joint forward ====================\n")

    # pin_ik_solver_relative = partial(solver.ik, tip_name="ur_arm_wrist_3_joint", link=False)

    # base_from_tip = np.array([[0, 0, 1, 0.5], [0, 1, 0, 0.25], [-1, 0, 0, 0.5], [0, 0, 0, 1]])

    # q_pin = pin_ik_solver_relative(base_from_tip, qinit=np.array([0, 0, 0, 0, 0, 0]))

    # print("> ik solver: q_pin")
    # print(q_pin)
    # print()

    # print("> forward: pose real")
    # print(base_from_tip)
    # print()

    # pose = solver.forward(q_pin, "ur_arm_wrist_3_joint", link=False, relative_output=True)
    # print("> forward: pose pinocchio")
    # print(pose)
    # print()

    # pose = opti_ik_solver.forward(q_pin)
    # print("> forward: pose opti")
    # print(pose)
    # print()

    # **************************************************************************
    # link ik
    # **************************************************************************

    # print("\n==================== link ik ====================\n")

    # q_zero = np.array([0, 0, 0, 0, 0, 0])

    # pin_ik_solver_relative = partial(solver.ik, tip_name="ur_arm_tool0", link=True)

    # target_pose = np.array([[0, 0, 1, 0.7], [0, 1, 0, 0.25], [-1, 0, 0, 0.5], [0, 0, 0, 1]])

    # q_pin = pin_ik_solver_relative(target_pose, qinit=q_zero)
    # print("> ik solver: q_pin")
    # print(q_pin)
    # print()

    # q_opti = opti_ik_solver.ik(target_pose, qinit=q_zero)
    # print("> ik solver: q_opti")
    # print(q_opti)
    # print()

    # pose = solver.forward(q_opti, "ur_arm_tool0", link=True, relative_output=True)
    # print("> forward: q_opti")
    # print(pose)
    # print()

    # **************************************************************************
    # slider visualization
    # **************************************************************************

    # # -------------------- init --------------------#
    # init_pb()
    # rb = RobotSetup("r0")
    # line_pts_flattened = [np.array([-0.25, 0.5, 1]), np.array([0.75, 0.5, 1])]
    # radius_per_edge = [0.01]
    # element_body = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)[0]
    # q_zero = np.array([0, 0, 0, 0, 0, 0])

    # # -------------------- init slider --------------------#
    # x_slider = p.addUserDebugParameter(f"x", 0.0, 1.0, 0.25)
    # y_slider = p.addUserDebugParameter(f"y", -1.0, 1.0, 0.5)
    # z_slider = p.addUserDebugParameter(f"z", 0.0, 2.0, 1.0)

    # # -------------------- loop --------------------#
    # while True:
    #     x_value = p.readUserDebugParameter(x_slider)
    #     y_value = p.readUserDebugParameter(y_slider)
    #     z_value = p.readUserDebugParameter(z_slider)

    #     pp.set_point(element_body, [x_value, y_value, z_value])

    #     element_pose = pp.multiply(pp.multiply(
    #         pp.get_pose(element_body), pp.Pose(point=[0, 0, 0], euler=pp.Euler(1.5708, 1.5708, 1.5708))
    #     ), pp.Pose(point=[0, 0, 0], euler=pp.Euler(0, 0, 1.5708)))

    #     element_plot_handle = pp.draw_pose(element_pose, length=0.2, lifetime=1.0)
    #     ee_plot_handle = pp.draw_pose(pp.get_link_pose(rb.robot, pp.link_from_name(rb.robot, "bar_tcp")), length=0.2, lifetime=1.0)

    #     element_pose_relative = rb.get_relative_pose(element_pose)
    #     connect_from_element = pp.pose_transformation.tform_from_pose(element_pose_relative)

    #     world_from_element = pp.pose_transformation.tform_from_pose(element_pose)

    #     q_opti, base_opti = opti_ik_solver.ik(world_from_element, q_init=q_zero)
    #     rb.set_joint_positions(rb.arm_joints, q_opti)
    #     rb.set_joint_positions(rb.base_joints, base_opti)

    #     time.sleep(0.05)

    # **************************************************************************
    # sequence visualization
    # **************************************************************************

    # -------------------- Load process file --------------------#
    mt_file_name = MT_FILE_NAME + ".json"
    line_pt_pairs, contact_id_pairs, bar_radius = parse_mt_geometric(mt_file_name)
    line_pt_pairs: List[List[List[float]]]  # bar list
    contact_id_pairs: List[List[float]]  # contact pairs
    bar_radius: float
    line_pts_flattened: List[np.ndarray] = flatten_list(np.array(line_pt_pairs))  # numpy points list
    vertices: List[List] = flatten_list(line_pt_pairs)  # points list

    # -------------------- Eliminate Z-axis deviation --------------------#
    min_z = np.min(line_pts_flattened, axis=0)[2]
    line_pts_flattened = [np.array([0, 0, -min_z]) + point for point in line_pts_flattened]

    radius_per_edge = [bar_radius] * int(len(line_pts_flattened) / 2)

    # -------------------- Elements Init --------------------#
    init_pb()
    goal_poses = []
    with pp.LockRenderer():
        element_bodies = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)
        half_coupler_from_contact_pair = create_couplers(line_pts_flattened, contact_id_pairs)
        for i, e in enumerate(element_bodies):
            pp.add_text(str(i), pp.get_point(e))
            goal_poses.append(pp.get_pose(e))
            pp.set_pose(e, pp.Pose(point=(5, 0, 0), euler=pp.Euler(0, 1.5708, 0)))
    element_from_index = {
        i: Element(i, e, pp.get_pose(e), goal_poses[i], [line_pts_flattened[2 * i], line_pts_flattened[2 * i + 1]])
        for i, e in enumerate(element_bodies)
    }

    # -------------------- robot init --------------------#
    rb_0 = RobotSetup("r0")
    rb_1 = RobotSetup("r1")
    robot_list = [rb_0, rb_1]
    q_zero = np.array([0, 0, 0, 0, 0, 0])
    opti_ik_solver = GraspOptiSolver(urdf_path, robot_list)

    # -------------------- debugger --------------------#
    continue_button = p.addUserDebugParameter("continue", 1, 0, 0)
    collision_show_button = p.addUserDebugParameter("collision show", 1, 0, 0)

    # -------------------- visualization --------------------#
    sequence = [[0, 1], [2, 3], [4, 5]]
    assembled = []
    collision_show = True
    for step in sequence:
        for i in step:
            pp.set_pose(element_from_index[i].body, element_from_index[i].goal_pose)
            pp.set_color(element_from_index[i].body, pp.RED)
            pp.draw_pose(element_from_index[i].goal_pose, length=0.25)

        # world_from_element = pp.pose_transformation.tform_from_pose(element_from_index[i].goal_pose)
        q_opti_list, base_opti_list = opti_ik_solver.solve(step, element_from_index, assembled=assembled)

        if q_opti_list is None or base_opti_list is None:
            pp.wait_for_user("Solve failed!")
            continue

        for temp_rb, q_opti, b_opti in zip(robot_list, q_opti_list, base_opti_list):
            temp_rb: RobotSetup
            temp_rb.set_joint_positions(temp_rb.arm_joints, q_opti)
            temp_rb.set_joint_positions(temp_rb.base_joints, b_opti)

        prev_continue_button_value = p.readUserDebugParameter(continue_button)
        prev_collision_show_button_value = p.readUserDebugParameter(collision_show_button)

        while True:
            current_continue_button_value = p.readUserDebugParameter(continue_button)
            if current_continue_button_value > prev_continue_button_value:
                prev_continue_button_value = current_continue_button_value
                for i in step:
                    assembled.append(i)
                break

            current_collision_show_button_value = p.readUserDebugParameter(collision_show_button)
            if current_collision_show_button_value > prev_collision_show_button_value:
                prev_collision_show_button_value = current_collision_show_button_value
                opti_ik_solver.CollisionVisualize(collision_show)
                collision_show = not collision_show

            time.sleep(0.02)

    # **************************************************************************
    # 碰撞体测试
    # **************************************************************************

    # init_pb()

    # # -------------------- robot init --------------------#
    # rb = RobotSetup("r0")
    # q = np.array(pp.get_joint_positions(rb.robot, rb.arm_joints))

    # # -------------------- 绑定碰撞信息 --------------------#
    # link_names = [
    #     "base_link",
    #     "front_left_wheel_link",
    #     "front_right_wheel_link",
    #     "rear_left_wheel_link",
    #     "rear_right_wheel_link",
    #     "top_chassis_link",
    #     "front_bumper_link",
    #     "rear_bumper_link",
    #     "top_plate_link",
    #     "ur_arm_base_link_inertia",
    #     "ur_arm_shoulder_link",
    #     "ur_arm_upper_arm_link",
    #     "ur_arm_forearm_link",
    #     "ur_arm_wrist_1_link",
    #     "ur_arm_wrist_2_link",
    #     "ur_arm_wrist_3_link",
    #     "ipad_rack_link",
    # ]

    # # with pp.LockRenderer():
    # #     for link_name in link_names:
    # #         pp.set_color(rb.robot, (0, 0, 1, 0.2), pp.link_from_name(rb.robot, link_name))
    # #         # pp.wait_for_user(link_name)

    # collision_infos = {
    #     "base_link": [
    #         [
    #             ((0.316, 0.253, 0.2), 0.2),
    #             ((0.316, 0, 0.2), 0.2),
    #             ((0.316, -0.253, 0.2), 0.2),
    #             ((-0.316, 0.253, 0.2), 0.2),
    #             ((-0.316, 0, 0.2), 0.2),
    #             ((-0.316, -0.253, 0.2), 0.2),
    #             ((0, 0.253, 0.2), 0.2),
    #             ((0, 0, 0.2), 0.2),
    #             ((0, -0.253, 0.2), 0.2),
    #         ],
    #         "base_joint",
    #     ],
    #     "ur_arm_base_link_inertia": [
    #         [
    #             ((0.389, 0, 0.41), 0.075),
    #         ],
    #         "base_joint",
    #     ],
    #     "ur_arm_shoulder_link": [
    #         [
    #             ((0.389, 0, 0.516), 0.075),
    #         ],
    #         "ur_arm_shoulder_pan_joint",
    #     ],
    #     "ur_arm_upper_arm_link": [
    #         [
    #             ((0.526, 0, 0.516), 0.075),
    #             ((0.526, 0, 0.616), 0.05),
    #             ((0.526, 0, 0.716), 0.05),
    #             ((0.526, 0, 0.816), 0.05),
    #             ((0.526, 0, 0.936), 0.075),
    #         ],
    #         "ur_arm_shoulder_lift_joint",
    #     ],
    #     "ur_arm_forearm_link": [
    #         [
    #             ((0.4, 0, 0.936), 0.05),
    #             ((0.4, 0, 1.036), 0.05),
    #             ((0.4, 0, 1.136), 0.05),
    #             ((0.4, 0, 1.236), 0.05),
    #             ((0.4, 0, 1.336), 0.05),
    #             ((0.35, 0, 1.336), 0.05),
    #         ],
    #         "ur_arm_elbow_joint",
    #     ],
    #     "ur_arm_wrist_1_link": [
    #         [
    #             ((0.525, 0, 1.336), 0.05),
    #         ],
    #         "ur_arm_wrist_1_joint",
    #     ],
    #     "ur_arm_wrist_2_link": [
    #         [
    #             ((0.525, -0.1, 1.336), 0.05),
    #         ],
    #         "ur_arm_wrist_2_joint",
    #     ],
    #     "ur_arm_wrist_3_link": [
    #         [
    #             ((0.625, -0.1, 1.336), 0.05),
    #         ],
    #         "ur_arm_wrist_3_joint",
    #     ],
    # }

    # # -------------------- init collision --------------------#
    # sphere_attachments = {}

    # with pp.LockRenderer():
    #     for link_name in collision_infos.keys():
    #         attachments = []
    #         for sphere_data in collision_infos[link_name][0]:
    #             sphere = pp.create_sphere(sphere_data[1], color=(1, 0, 0, 0.5))
    #             pp.set_point(sphere, sphere_data[0])
    #             attachment = pp.create_attachment(rb.robot, pp.link_from_name(rb.robot, link_name), sphere)
    #             attachments.append(attachment)
    #         sphere_attachments[link_name] = attachments

    # # -------------------- init collision offsets --------------------#
    # collision_offsets = {}
    # for link_name in collision_infos.keys():
    #     sphere_datas = collision_infos[link_name][0]
    #     sphere_mnt_joint = collision_infos[link_name][1]
    #     sphere_offsets = []
    #     for sphere_data in sphere_datas:
    #         # world_from_joint
    #         joint_pose = opti_ik_solver.GetJointPose(np.array([0, 0, 0]), q, sphere_mnt_joint)
    #         world_from_joint = pp.pose_transformation.pose_from_tform(joint_pose)
    #         # world_from_sphere
    #         sphere_pose = np.eye(4)
    #         sphere_pose[0, 3] = sphere_data[0][0]
    #         sphere_pose[1, 3] = sphere_data[0][1]
    #         sphere_pose[2, 3] = sphere_data[0][2]
    #         world_from_sphere = pp.pose_transformation.pose_from_tform(sphere_pose)
    #         # joint_from_sphere
    #         joint_from_sphere = pp.multiply(pp.invert(world_from_joint), world_from_sphere)
    #         offset_mat = pp.pose_transformation.tform_from_pose(joint_from_sphere)
    #         offset_data = ((offset_mat[0, 3], offset_mat[1, 3], offset_mat[2, 3]), sphere_data[1])
    #         sphere_offsets.append(offset_data)
    #     collision_offsets[link_name] = [sphere_offsets, sphere_mnt_joint]

    # sphere = pp.create_sphere(0.075)

    # # -------------------- init slider --------------------#
    # x_slider = p.addUserDebugParameter(f"x", -1.0, 1.0, 0)
    # y_slider = p.addUserDebugParameter(f"y", -1.0, 1.0, 0)
    # yaw_slider = p.addUserDebugParameter(f"yaw", -np.pi, np.pi, 0)

    # x_sphere_slider = p.addUserDebugParameter(f"x_sphere", -1.0, 1.0, 0)
    # y_sphere_slider = p.addUserDebugParameter(f"y_sphere", -1.0, 1.0, 0)
    # z_sphere_slider = p.addUserDebugParameter(f"z_sphere", -1.0, 1.0, 0)

    # j1_slider = p.addUserDebugParameter(f"j1", -np.pi, np.pi, 0)
    # j2_slider = p.addUserDebugParameter(f"j2", -np.pi, np.pi, 0)
    # j3_slider = p.addUserDebugParameter(f"j3", -np.pi, np.pi, 0)
    # j4_slider = p.addUserDebugParameter(f"j4", -np.pi, np.pi, 0)
    # j5_slider = p.addUserDebugParameter(f"j5", -np.pi, np.pi, 0)
    # j6_slider = p.addUserDebugParameter(f"j6", -np.pi, np.pi, 0)

    # # -------------------- loop --------------------#
    # while True:
    #     x_value = p.readUserDebugParameter(x_slider)
    #     y_value = p.readUserDebugParameter(y_slider)
    #     yaw_value = p.readUserDebugParameter(yaw_slider)

    #     x_sphere_value = p.readUserDebugParameter(x_sphere_slider)
    #     y_sphere_value = p.readUserDebugParameter(y_sphere_slider)
    #     z_sphere_value = p.readUserDebugParameter(z_sphere_slider)

    #     rb.set_joint_positions(rb.base_joints, np.array([x_value, y_value, yaw_value]))
    #     for link_name in sphere_attachments.keys():
    #         for attachment in sphere_attachments[link_name]:
    #             attachment: pp.Attachment
    #             attachment.assign()

    #     pp.set_point(sphere, np.array([x_sphere_value, y_sphere_value, z_sphere_value]))
    #     # print(np.array([x_sphere_value, y_sphere_value, z_sphere_value]))

    #     j1_value = p.readUserDebugParameter(j1_slider)
    #     j2_value = p.readUserDebugParameter(j2_slider)
    #     j3_value = p.readUserDebugParameter(j3_slider)
    #     j4_value = p.readUserDebugParameter(j4_slider)
    #     j5_value = p.readUserDebugParameter(j5_slider)
    #     j6_value = p.readUserDebugParameter(j6_slider)
    #     rb.set_joint_positions(rb.arm_joints, np.array([j1_value, j2_value, j3_value, j4_value, j5_value, j6_value]))

    # **************************************************************************
    # 关节位置测试
    # **************************************************************************

    # init_pb()

    # # -------------------- robot init --------------------#
    # rb = RobotSetup("r0")
    # q = np.array(pp.get_joint_positions(rb.robot, rb.arm_joints))

    # # -------------------- init slider --------------------#
    # x_slider = p.addUserDebugParameter(f"x", -1.0, 1.0, 0)
    # y_slider = p.addUserDebugParameter(f"y", -1.0, 1.0, 0)
    # yaw_slider = p.addUserDebugParameter(f"yaw", -np.pi, np.pi, 0)

    # j1_slider = p.addUserDebugParameter(f"j1", -np.pi, np.pi, 0)
    # j2_slider = p.addUserDebugParameter(f"j2", -np.pi, np.pi, 0)
    # j3_slider = p.addUserDebugParameter(f"j3", -np.pi, np.pi, 0)
    # j4_slider = p.addUserDebugParameter(f"j4", -np.pi, np.pi, 0)
    # j5_slider = p.addUserDebugParameter(f"j5", -np.pi, np.pi, 0)
    # j6_slider = p.addUserDebugParameter(f"j6", -np.pi, np.pi, 0)

    # # -------------------- loop --------------------#
    # while True:
    #     x_value = p.readUserDebugParameter(x_slider)
    #     y_value = p.readUserDebugParameter(y_slider)
    #     yaw_value = p.readUserDebugParameter(yaw_slider)
    #     rb.set_joint_positions(rb.base_joints, np.array([x_value, y_value, yaw_value]))

    #     j1_value = p.readUserDebugParameter(j1_slider)
    #     j2_value = p.readUserDebugParameter(j2_slider)
    #     j3_value = p.readUserDebugParameter(j3_slider)
    #     j4_value = p.readUserDebugParameter(j4_slider)
    #     j5_value = p.readUserDebugParameter(j5_slider)
    #     j6_value = p.readUserDebugParameter(j6_slider)
    #     q = np.array([j1_value, j2_value, j3_value, j4_value, j5_value, j6_value])

    #     rb.set_joint_positions(rb.arm_joints, q)

    #     for joint_name in opti_ik_solver.MANIPULATOR_CONTROL_JOINT_NAMES:
    #         pose_mat = opti_ik_solver.GetJointPose(np.array([x_value, y_value, yaw_value]), q, joint_name)
    #         pp.draw_pose(pp.pose_transformation.pose_from_tform(pose_mat), length=0.15, lifetime=1.0)

    # **************************************************************************
    # disabled collisions visualization
    # **************************************************************************

    # # -------------------- init --------------------#
    # init_pb()
    # rb = RobotSetup("r0")
    # q_zero = np.array([0, 0, 0, 0, 0, 0])

    # pp.wait_for_user()

    # keys = list(opti_ik_solver.collision_info.keys())
    # keys_combinations = list(itertools.combinations(keys, 2))

    # print("======================================== opti ik combinations")
    # print(keys_combinations)

    # # -------------------- loop --------------------#
    # for collision_pair in rb.disabled_collisions:
    #     link_1_name = pp.get_link_name(rb.robot, collision_pair[0])
    #     link_2_name = pp.get_link_name(rb.robot, collision_pair[1])
    #     pair = (link_1_name, link_2_name)
    #     idx = opti_ik_solver.FindCollisionPair(keys_combinations, pair)
    #     # print(
    #     #     f"collision_pair: {collision_pair}, link 1: {link_1_name}, link 2: {link_2_name}, idx: {idx}"
    #     # )
    #     if idx != -1:
    #         keys_combinations.remove(keys_combinations[idx])

    # print("======================================== new opti ik combinations")
    # print(keys_combinations)

    # for collision_pair in keys_combinations:
    #     link_1_idx = pp.link_from_name(rb.robot, collision_pair[0])
    #     link_2_idx = pp.link_from_name(rb.robot, collision_pair[1])

    #     pp.set_color(rb.robot, (0, 0, 1, 0.4), link_1_idx)
    #     pp.set_color(rb.robot, (0, 0, 1, 0.4), link_2_idx)

    #     pp.wait_for_user(f"{collision_pair[0], collision_pair[1]}")

    #     pp.set_color(rb.robot, pp.GREY, link_1_idx)
    #     pp.set_color(rb.robot, pp.GREY, link_2_idx)
