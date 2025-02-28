import os
import sys
from typing import Dict, List, Tuple, Union

import casadi as ca
import numpy as np

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from robot.robot_setup import RobotSetup
from utils.collision import collision_info
from utils.utils import HideOutput


class SDF(object):

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

    debug_sphere_visual_id = -1
    debug_line_visual_id = -1

    def __init__(self, urdf_path: str, robot: RobotSetup) -> None:
        self.urdf_path = urdf_path
        self.robot = robot

        self.Nq = len(self.MANIPULATOR_CONTROL_JOINT_NAMES)
        self.q = ca.MX.sym("q", self.Nq, 1)

        self.collision_info = collision_info

        self.ForwardInit()

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

    def ForwardInit(self) -> None:
        self.connect_from_j6_fn = RobotSetup.symbolic_forward(
            self.urdf_path, self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES, self.MANIPULATOR_CONTROL_JOINT_NAMES
        )
        self.connect_from_joint_dict = {}
        for joint_idx, joint_name in enumerate(self.MANIPULATOR_CONTROL_JOINT_NAMES):
            end_idx = self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES.index(joint_name) + 1
            fk_fn = RobotSetup.symbolic_forward(
                self.urdf_path,
                self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES[:end_idx],
                self.MANIPULATOR_CONTROL_JOINT_NAMES[: joint_idx + 1],
                q=self.q,
            )
            fk_mat = RobotSetup.symbolic_forward(
                self.urdf_path,
                self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES[:end_idx],
                self.MANIPULATOR_CONTROL_JOINT_NAMES[: joint_idx + 1],
                q=self.q,
                output_type="matrix",
            )

            self.connect_from_joint_dict[joint_name] = (fk_fn, fk_mat, joint_idx + 1)

        base_from_connect_sym = RobotSetup.symbolic_forward(
            self.urdf_path, self.BASE_REDUCED_MODEL_JOINT_NAMES, self.BASE_CONTROL_JOINT_NAMES, output_type="matrix"
        )
        self.base_from_connect = self.eval("base_from_connect", base_from_connect_sym, [], [])

    def SphereApproximation(
        self,
        p: Union[List[float], np.ndarray, ca.MX],
        q: Union[List[float], np.ndarray, ca.MX],
        x: Union[List[float], np.ndarray, ca.MX],
    ) -> Tuple[ca.MX, Dict]:

        # -------------------- 计算base的转移矩阵 --------------------#
        world_from_base: ca.MX = ca.MX.eye(4)
        world_from_base[0, 3] = p[0]
        world_from_base[1, 3] = p[1]
        yaw = p[2]
        R_z = ca.vertcat(
            ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0), ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0), ca.horzcat(0, 0, 1)
        )
        world_from_base[:3, :3] = R_z

        metadata_list = []
        sdf_list = []

        # -------------------- loop 1: 遍历robot links --------------------#
        for link_name in self.collision_info:
            infos = self.collision_info[link_name][0]
            mnt_joint = self.collision_info[link_name][1]
            weight = self.collision_info[link_name][2]

            # -------------------- 获取joint位置 --------------------#
            if mnt_joint == "base_joint":
                world_from_joint = world_from_base
            else:
                base_from_joint = self.base_from_connect @ self.connect_from_joint_dict[mnt_joint][1]
                world_from_joint = world_from_base @ base_from_joint

            # -------------------- loop 2: 遍历link上的碰撞小球 --------------------#
            for info_idx, info in enumerate(infos):
                offset = info[0]
                radius = info[1]

                # -------------------- 计算小球位置 --------------------#
                joint_from_sphere = np.eye(4)
                joint_from_sphere[0, 3] = offset[0]
                joint_from_sphere[1, 3] = offset[1]
                joint_from_sphere[2, 3] = offset[2]
                world_from_sphere = world_from_joint @ joint_from_sphere
                c = ca.vertcat(world_from_sphere[0, 3], world_from_sphere[1, 3], world_from_sphere[2, 3])

                # -------------------- 计算距离和SDF --------------------#
                dist = ca.norm_2(x - c)
                sdf_i = dist - radius
                sdf_list.append(sdf_i)
                metadata_list.append({"link_name": link_name, "mnt_joint": mnt_joint, "info_idx": info_idx})

        # -------------------- 判断是否数值输出 --------------------#
        is_numeric = all(isinstance(arg, (list, np.ndarray)) for arg in [p, q, x])
        if is_numeric:
            sdf_vec = ca.vertcat(*sdf_list)
            sdf_robot = ca.mmin(sdf_vec)
            sdf = float(self.eval("SphereApproximation", sdf_robot, [self.q], [q]))

            sdf_values = self.eval("SphereApproximationList", sdf_vec, [self.q], [q]).flatten()
            min_index = np.argmin(sdf_values)
            min_metadata = metadata_list[min_index]
        else:
            sdf = ca.vertcat(*sdf_list)
            min_metadata = {"link_name": "unknown", "mnt_joint": "unknown", "info_idx": -1}

        return sdf, min_metadata

    def SphereSDFVisualize(
        self,
        meta_data: Dict,
        p: Union[List[float], np.ndarray],
        q: Union[List[float], np.ndarray],
        x: Union[List[float], np.ndarray],
    ):
        link_name = meta_data["link_name"]
        mnt_joint = meta_data["mnt_joint"]
        info_idx = meta_data["info_idx"]

        world_from_base: ca.MX = ca.MX.eye(4)
        world_from_base[0, 3] = p[0]
        world_from_base[1, 3] = p[1]
        yaw = p[2]
        R_z = ca.vertcat(
            ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0), ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0), ca.horzcat(0, 0, 1)
        )
        world_from_base[:3, :3] = R_z

        if mnt_joint == "base_joint":
            world_from_joint = world_from_base
        else:
            base_from_joint = self.base_from_connect @ self.connect_from_joint_dict[mnt_joint][1]
            world_from_joint = world_from_base @ base_from_joint

        if link_name != "unknown":
            info = self.collision_info[link_name][0][info_idx]
            offset = info[0]
            radius = info[1]
            visual_id = info[2]

            print(f"link_name: {link_name}, mnt_joint: {mnt_joint} ", info)

            joint_from_sphere = np.eye(4)
            joint_from_sphere[0, 3] = offset[0]
            joint_from_sphere[1, 3] = offset[1]
            joint_from_sphere[2, 3] = offset[2]
            world_from_sphere = world_from_joint @ joint_from_sphere
            sphere_center = ca.vertcat(world_from_sphere[0, 3], world_from_sphere[1, 3], world_from_sphere[2, 3])
            c = self.eval("", sphere_center, [self.q], [q])

            if self.debug_sphere_visual_id == -1:
                self.debug_sphere_visual_id = pp.create_sphere(radius, color=(1, 0, 0, 0.5))
            else:
                pp.remove_body(self.debug_sphere_visual_id)
                self.debug_sphere_visual_id = pp.create_sphere(radius, color=(1, 0, 0, 0.5))
            pp.set_point(self.debug_sphere_visual_id, c)

            if self.debug_line_visual_id == -1:
                self.debug_line_visual_id = pp.add_line(x, c)
            else:
                pp.remove_debug(self.debug_line_visual_id)
                self.debug_line_visual_id = pp.add_line(x, c)

    def __call__(
        self,
        p: Union[List[float], np.ndarray, ca.MX],
        q: Union[List[float], np.ndarray, ca.MX],
        x: Union[List[float], np.ndarray, ca.MX],
        method: str = "sphere",
        visualize: bool = False,
    ) -> Union[float, np.ndarray, ca.MX]:
        """
        Calculate SDF(p, q, x).

        Params:
            p (List[float] | np.ndarray | ca.MX): 2D pose of robot base [x, y, yaw]
            q (List[float] | np.ndarray | ca.MX): joint positions of robot
            x (List[float] | np.ndarray | ca.MX): 3D position of target point [x, y, z]
            method (str, "sphere"): SDF calculation method

        Returns:
            (List[float] | np.ndarray | ca.MX): SDF value
        """
        if method == "sphere":
            sdf, meta_info = self.SphereApproximation(p, q, x)
        else:
            raise NotImplementedError(f"Method {method} is not implemented.")

        if visualize:
            self.SphereSDFVisualize(meta_info, p, q, x)

        return sdf


if __name__ == "__main__":
    import pybullet_planning as pp
    from utils.collision import Element, create_couplers, init_pb
    import pybullet as p
    import time

    urdf_path = "/home/jeong/summer_research/eth_ws/src/husky_assembly/data/husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf"

    init_pb()

    rb = RobotSetup()

    sdf_calculator = SDF(urdf_path, rb)

    # print("")
    # print("")
    # print("")
    # print("")
    # print("")
    # print(sdf_calculator([0, 0, 0], rb.arm_init_angles, [1, 1, 1]))

    x = p.addUserDebugParameter("x", -2, 2, 0)
    y = p.addUserDebugParameter("y", -2, 2, 0)
    z = p.addUserDebugParameter("z", -2, 2, 0)

    x_sym = ca.MX.sym("x", 3)
    p_sym = ca.MX.sym("p", 3)
    q_sym = ca.MX.sym("q", 6)

    point_id = pp.create_sphere(0.05, color=pp.BLACK)

    while True:
        x_value = p.readUserDebugParameter(x)
        y_value = p.readUserDebugParameter(y)
        z_value = p.readUserDebugParameter(z)
        pp.set_point(point_id, [x_value, y_value, z_value])

        #-------------------- sphere：数值计算 --------------------#
        print("sphere numerical: ", sdf_calculator([0, 0, 0], rb.arm_init_angles, [x_value, y_value, z_value]))

        #-------------------- sphere：解析计算 --------------------#
        sdf_vec = sdf_calculator(x_sym, q_sym, p_sym)
        sdf_values = sdf_calculator.eval("", sdf_vec, [x_sym, p_sym, sdf_calculator.q], [[x_value, y_value, z_value], [0, 0, 0], rb.arm_init_angles])
        print("sphere analytical: ", sdf_values.min())

        time.sleep(1.0 / 60)
