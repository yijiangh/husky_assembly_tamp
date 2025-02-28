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


def eval(name: str, obj: ca.MX, sym: List[ca.MX], data: List[np.ndarray], verbose: bool = False) -> np.ndarray:
    fn = ca.Function("fn", sym, [obj])
    if sym != []:
        fn_result = fn(*data).toarray()
    else:
        fn_result = fn()["o0"].toarray()
    if verbose:
        print(name, "\n", fn_result)
    return fn_result


def generate_trajectory(
    start_pos: np.ndarray, end_pos: np.ndarray, v_max: float = np.pi / 6, n_segments: int = 5
) -> List[List[Tuple[float, float, List[float]]]]:
    # 计算每个关节的理论时间（避免除以零）
    delta_q = end_pos - start_pos
    joint_times = np.where(
        np.abs(delta_q) > 1e-6, np.abs(delta_q) / v_max, 1.0  # 判断是否有位移  # 无位移关节默认分配 1.0 秒
    )
    total_time = np.max(joint_times)  # 取所有关节时间的最大值
    segment_times = np.linspace(0, total_time, n_segments + 1)

    # 生成中间点
    waypoints = np.linspace(start_pos, end_pos, n_segments + 1)

    # 为每个关节生成轨迹
    joint_trajs = []
    for joint_idx in range(6):
        joint_traj = []
        for i in range(n_segments):
            # 时间段
            t_start = segment_times[i]
            t_end = segment_times[i + 1]
            duration = t_end - t_start

            # 边界条件
            q_start = waypoints[i, joint_idx]
            q_end = waypoints[i + 1, joint_idx]
            v = (q_end - q_start) / duration  # 恒定速度

            # 构建方程组 AX = B
            A = np.array(
                [
                    [1, t_start, t_start**2, t_start**3, t_start**4, t_start**5],
                    [0, 1, 2 * t_start, 3 * t_start**2, 4 * t_start**3, 5 * t_start**4],
                    [0, 0, 2, 6 * t_start, 12 * t_start**2, 20 * t_start**3],
                    [1, t_end, t_end**2, t_end**3, t_end**4, t_end**5],
                    [0, 1, 2 * t_end, 3 * t_end**2, 4 * t_end**3, 5 * t_end**4],
                    [0, 0, 2, 6 * t_end, 12 * t_end**2, 20 * t_end**3],
                ]
            )

            B = np.array(
                [
                    q_start,  # 初始位置
                    v,  # 初始速度
                    0,  # 初始加速度 (启发式)
                    q_end,  # 终止位置
                    v,  # 终止速度
                    0,  # 终止加速度 (启发式)
                ]
            )

            # 求解多项式系数
            coeffs = np.linalg.solve(A, B).tolist()
            joint_traj.append((t_start, t_end, coeffs))

        joint_trajs.append(joint_traj)

    return joint_trajs


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

    def __init__(self, urdf_path: str, robot: RobotSetup, q: Union[ca.MX, None] = None) -> None:
        self.urdf_path = urdf_path
        self.robot = robot

        self.Nq = len(self.MANIPULATOR_CONTROL_JOINT_NAMES)
        if q is None:
            self.q = ca.MX.sym("q", self.Nq, 1)
        else:
            self.q = q

        self.collision_info = collision_info

        self._BuildSymbolicFK()

    def _BuildSymbolicFK(self) -> None:
        self.connect_from_j6_fn = RobotSetup.symbolic_forward(
            self.urdf_path, self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES, self.MANIPULATOR_CONTROL_JOINT_NAMES
        )
        self.connect_from_joint_dict = {}
        for joint_idx, joint_name in enumerate(self.MANIPULATOR_CONTROL_JOINT_NAMES):
            end_idx = self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES.index(joint_name) + 1
            fk_mat = RobotSetup.symbolic_forward(
                self.urdf_path,
                self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES[:end_idx],
                self.MANIPULATOR_CONTROL_JOINT_NAMES[: joint_idx + 1],
                q=self.q,
                output_type="matrix",
            )

            self.connect_from_joint_dict[joint_name] = (fk_mat, joint_idx + 1)

        base_from_connect_sym = RobotSetup.symbolic_forward(
            self.urdf_path, self.BASE_REDUCED_MODEL_JOINT_NAMES, self.BASE_CONTROL_JOINT_NAMES, output_type="matrix"
        )
        self.base_from_connect = eval("base_from_connect", base_from_connect_sym, [], [])

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
                base_from_joint = self.base_from_connect @ self.connect_from_joint_dict[mnt_joint][0]
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

        sdf_vec = ca.vertcat(*sdf_list)
        sdf_robot = ca.mmin(sdf_vec)

        # -------------------- 判断是否数值输出 --------------------#
        is_numeric = all(isinstance(arg, (list, np.ndarray)) for arg in [p, q, x])

        if is_numeric:
            sdf = float(eval("SphereApproximation", sdf_robot, [self.q], [q]))
            sdf_values = eval("SphereApproximationList", sdf_vec, [self.q], [q]).flatten()
            min_index = np.argmin(sdf_values)
            min_metadata = metadata_list[min_index]
        else:
            sdf = sdf_robot
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
            base_from_joint = self.base_from_connect @ self.connect_from_joint_dict[mnt_joint][0]
            world_from_joint = world_from_base @ base_from_joint

        if link_name != "unknown":
            info = self.collision_info[link_name][0][info_idx]
            offset = info[0]
            radius = info[1]
            visual_id = info[2]

            # print(f"link_name: {link_name}, mnt_joint: {mnt_joint} ", info)

            joint_from_sphere = np.eye(4)
            joint_from_sphere[0, 3] = offset[0]
            joint_from_sphere[1, 3] = offset[1]
            joint_from_sphere[2, 3] = offset[2]
            world_from_sphere = world_from_joint @ joint_from_sphere
            sphere_center = ca.vertcat(world_from_sphere[0, 3], world_from_sphere[1, 3], world_from_sphere[2, 3])
            c = eval("", sphere_center, [self.q], [q])

            with pp.LockRenderer():
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


class SVSDF(object):
    """
    用于计算空间中的点到机器人关节轨迹的距离的类。
    """

    def __init__(
        self,
        robot_setup: RobotSetup,
        joint_trajectory: List[List[Tuple[float, float, List[float]]]],
        max_iter: int = 100,
        lr: float = 0.1,
        tol: float = 1e-4,
    ):
        """
        初始化 SVSDF 计算类。

        Params:
            joint_trajectory (List[trajectory]):
                - trajectory (List[Tuple[float, float, List[float]]]): 机器人关节轨迹，由一系列关节轨迹片段组成，每个片段是一个元组 (start_time, end_time, coefficients)
                - start_time: 片段的起始时间 (float)
                - end_time: 片段的终止时间 (float)
                - coefficients: 5次多项式的系数，列表形式，长度为6，按照 t^0, t^1, t^2, t^3, t^4, t^5 的顺序排列 (List[float])
        """
        self.joint_trajectory = joint_trajectory
        self.num_joints = len(joint_trajectory)
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol

        # -------------------- 获取轨迹总时间范围 --------------------#
        self.t_min = min(seg[0] for traj in joint_trajectory for seg in traj)
        self.t_max = max(seg[1] for traj in joint_trajectory for seg in traj)

        # -------------------- 构建符号化计算系统 --------------------#
        self.t_sym = ca.MX.sym("t", 1)
        self.q_sym = self._BuildSymbolicQ(self.t_sym)
        self.x_sym = ca.MX.sym("x", 3)

        urdf_path = "/home/jeong/summer_research/eth_ws/src/husky_assembly/data/husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf"
        self.sdf = SDF(urdf_path, robot_setup, self.q_sym)

        self.sdf_sym = self.sdf(ca.MX([0, 0, 0]), self.q_sym, self.x_sym)
        self.jac_sym = ca.gradient(self.sdf_sym, self.t_sym)

    def _BuildSymbolicQ(self, t: ca.MX) -> ca.MX:
        """
        构建符号化的关节位置 q(t)

        Params:
            t (ca.MX): t variable

        Returns:
            ca.MX: q(t)
        """
        q = ca.MX.zeros(6)
        for joint_idx in range(len(self.joint_trajectory)):
            traj = self.joint_trajectory[joint_idx]
            for start, end, coeffs in traj:
                cond = ca.logic_and(t >= start, t <= end)
                t_rel = t
                poly = (
                    coeffs[0]
                    + coeffs[1] * t_rel
                    + coeffs[2] * (t_rel**2)
                    + coeffs[3] * (t_rel**3)
                    + coeffs[4] * (t_rel**4)
                    + coeffs[5] * (t_rel**5)
                )
                q[joint_idx] = ca.if_else(cond, poly, q[joint_idx])
        return q

    def _GradientDescent(
        self, x: np.ndarray, t_init: float, lr: float = 0.1, max_iter: int = 100
    ) -> Tuple[float, float]:
        """
        带自适应学习率的梯度下降

        Params:
            x (np.ndarray): point in 3D space
            t_init (float): initial time
            lr (float, 0.1): learning rate
            max_iter (int, 100): maximum number of iterations

        Returns:
            (float, float): optimal time, sdf value at optimal time
        """
        t_curr = t_init
        prev_grad = None
        momentum = 0.9
        velocity = 0

        for _ in range(max_iter):
            grad = eval("", self.jac_sym, [self.t_sym, self.x_sym], [t_curr, x]).item()

            # -------------------- 动量加速 --------------------#
            velocity = momentum * velocity + (1 - momentum) * grad
            delta_t = -lr * velocity

            # -------------------- 自适应步长 --------------------#
            if prev_grad is not None and np.sign(grad) != np.sign(prev_grad):
                lr *= 0.5

            t_new = t_curr + delta_t
            t_new = np.clip(t_new, self.t_min, self.t_max)

            # -------------------- 收敛判断 --------------------#
            if abs(t_new - t_curr) < 1e-6:
                break
            t_curr = t_new
            prev_grad = grad

        return t_curr, eval("", self.sdf_sym, [self.t_sym, self.x_sym], [t_curr, x]).item()

    def EvaluateJointPosition(self, time: float) -> np.ndarray:
        """
        计算在给定时间和关节轨迹片段索引下的关节位置。

        Params:
            time (float): 目标时间点 (float)

        Returns:
            np.ndarray: 关节角度
        """
        joint_angles = []
        for joint_index in range(self.num_joints):
            joint_trajectory = self.joint_trajectory[joint_index]
            joint_angle = None

            out_range = True
            for start_time, end_time, coefficients in joint_trajectory:
                if start_time <= time <= end_time:
                    t = time
                    joint_angle = (
                        coefficients[0]
                        + coefficients[1] * t
                        + coefficients[2] * (t**2)
                        + coefficients[3] * (t**3)
                        + coefficients[4] * (t**4)
                        + coefficients[5] * (t**5)
                    )
                    out_range = False
                    break
            if out_range:
                start_time, end_time, coefficients = joint_trajectory[-1]
                t = end_time
                joint_angle = (
                    coefficients[0]
                    + coefficients[1] * t
                    + coefficients[2] * (t**2)
                    + coefficients[3] * (t**3)
                    + coefficients[4] * (t**4)
                    + coefficients[5] * (t**5)
                )

            if joint_angle is None:
                raise ValueError(f"Time {time} is outside the valid range for joint {joint_index}")
            joint_angles.append(joint_angle)

        return np.array(joint_angles)

    def __call__(self, x: np.ndarray) -> Tuple[float, float]:
        """
        计算点 x 到 SV 的最短距离。

        Params:
            x (float): point in 3D space

        Returns:
            (float, float): optimal time, svsdf value
        """
        min_sdf = float("inf")
        best_t = 0.0

        init_points = np.linspace(self.t_min, self.t_max, 20)

        for t_init in init_points:
            t_curr, sdf_val = self._GradientDescent(x, t_init)
            if sdf_val < min_sdf:
                min_sdf = sdf_val
                best_t = t_curr

        refined_t, refined_sdf = self._GradientDescent(x, best_t, lr=0.01, max_iter=50)
        return refined_t, refined_sdf


if __name__ == "__main__":

    # **************************************************************************
    # SDF test
    # **************************************************************************

    # import pybullet_planning as pp
    # from utils.collision import Element, create_couplers, init_pb
    # import pybullet as p
    # import time

    # urdf_path = "/home/jeong/summer_research/eth_ws/src/husky_assembly/data/husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf"

    # init_pb()

    # rb = RobotSetup()

    # sdf_calculator = SDF(urdf_path, rb)

    # x = p.addUserDebugParameter("x", -2, 2, 0)
    # y = p.addUserDebugParameter("y", -2, 2, 0)
    # z = p.addUserDebugParameter("z", -2, 2, 0)

    # x_sym = ca.MX.sym("x", 3)
    # p_sym = ca.MX.sym("p", 3)
    # q_sym = ca.MX.sym("q", 6)

    # point_id = pp.create_sphere(0.05, color=pp.BLACK)

    # while True:
    #     x_value = p.readUserDebugParameter(x)
    #     y_value = p.readUserDebugParameter(y)
    #     z_value = p.readUserDebugParameter(z)
    #     pp.set_point(point_id, [x_value, y_value, z_value])

    #     # -------------------- sphere：数值计算 --------------------#
    #     print("sphere numerical: ", sdf_calculator([0, 0, 0], rb.arm_init_angles, [x_value, y_value, z_value]))

    #     # -------------------- sphere：解析计算 --------------------#
    #     sdf_vec = sdf_calculator(x_sym, q_sym, p_sym)
    #     sdf_values = sdf_calculator.eval(
    #         "", sdf_vec, [x_sym, p_sym, sdf_calculator.q], [[x_value, y_value, z_value], [0, 0, 0], rb.arm_init_angles]
    #     )
    #     print("sphere analytical: ", sdf_values.min())

    #     time.sleep(1.0 / 60)

    # **************************************************************************
    # SVSDF test
    # **************************************************************************

    import pybullet_planning as pp
    from utils.collision import Element, create_couplers, init_pb
    import pybullet as p
    import time

    np.set_printoptions(precision=3)

    init_pb()

    rb = RobotSetup()

    # 定义轨迹参数
    start_pos = np.array([0, 0, 0, 0, 0, 0])
    end_pos = np.array([0, -np.pi / 2, -np.pi / 2, 0, 0, 0])
    v_max = np.pi / 6

    # 生成轨迹
    trajectory = generate_trajectory(start_pos, end_pos, v_max)

    # 创建 SVSDF 实例
    svsdf = SVSDF(rb, trajectory)

    # 计算点到轨迹的最小距离
    target_point = np.array([0.5, 0.5, 0.75])
    svsdf_tup = svsdf(target_point)

    print("")
    print("")
    print("")
    print("")
    print(f"SVSDF Value at point {target_point.tolist()} and time {svsdf_tup[0]}: {svsdf_tup[1]}")

    # continue_button = p.addUserDebugParameter("continue", 1, 0, 0)
    # prev_button_value = p.readUserDebugParameter(continue_button)

    # slider = p.addUserDebugParameter("replay", 0, 1, 0)

    # test_times = np.linspace(0, 3.0, 30)

    # # 创建可视化
    # test_point = [0.5, 0.5, 0.75]
    # sphere_id = pp.create_sphere(0.05, color=pp.BLACK)
    # pp.set_point(sphere_id, test_point)

    # while True:
    #     # button_value = p.readUserDebugParameter(continue_button)
    #     # if button_value > prev_button_value:
    #     #     prev_button_value = button_value
    #     #     for t in test_times:
    #     #         pos = svsdf.evaluate_joint_position(t)
    #     #         rb.set_joint_positions(rb.arm_joints, pos)
    #     #         sdf([0, 0, 0], pos, [0.5, 0.5, 0.75])
    #     #         time.sleep(1.0 / 240)

    #     slider_value = p.readUserDebugParameter(slider)
    #     time_idx = int(slider_value * (test_times.shape[0] - 1))
    #     t = test_times[time_idx]
    #     pos = svsdf.EvaluateJointPosition(t)
    #     rb.set_joint_positions(rb.arm_joints, pos)
    #     sdf([0, 0, 0], pos, test_point, visualize=True)
    #     time.sleep(1.0 / 60)
