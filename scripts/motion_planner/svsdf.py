import os
import sys
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import casadi as ca
import numpy as np

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from robot.robot_setup import RobotSetup
from utils.collision import collision_info
from utils.utils import HideOutput
from utils.utils_casadi import eval

Trajectory = List[  # for different joints
    List[  # trajectory
        Tuple[  # episode coeffs
            Union[float, ca.MX],  # start_time
            Union[float, ca.MX],  # end_time
            Union[List[float], ca.MX],  # coeff
        ]
    ]
]

NodeTrajectory = List[  # trajectories of different joints
    List[  # trajectory
        Tuple[
            Union[np.ndarray, ca.MX],  # q
            Union[np.ndarray, ca.MX],  # dq
            Union[np.ndarray, ca.MX],  # ddq
            Union[np.ndarray, ca.MX],  # t
        ]
    ]
]


def Traj2Arr(traj: Trajectory, num_joints: int, symbolic: bool = False) -> Union[np.ndarray, ca.MX]:
    opt_vars = []

    # 遍历每个关节的每个时间段
    for j in range(num_joints):
        for seg_idx, seg in enumerate(traj[j]):
            delta_t = seg[1] - seg[0]  # 时间间隔变量
            opt_vars.append(delta_t)

            # 多项式系数变量
            for k, c in enumerate(seg[2]):
                opt_vars.append(c)
    if symbolic:
        opt_var = ca.vertcat(*opt_vars)
    else:
        opt_var = np.array(opt_vars)
    return opt_var


def Arr2Traj(opt_var: Union[np.ndarray, ca.MX], num_joints: int, num_segments: int) -> Trajectory:
    traj = []
    idx = 0  # 当前处理的数组索引
    for j in range(num_joints):
        # 获取当前关节的段数和多项式系数数量
        n_coeff = 6
        segments = []
        current_time = 0.0  # 初始化当前时间为0
        for _ in range(num_segments):
            delta_t = opt_var[idx]
            idx += 1
            coeffs = [opt_var[idx + k] for k in range(n_coeff)]
            idx += n_coeff
            seg_start = current_time
            seg_end = current_time + delta_t
            segments.append((seg_start, seg_end, coeffs))
            current_time = seg_end  # 更新时间
        traj.append(segments)
    return traj


def NodeTraj2Arr(traj: NodeTrajectory, num_joints: int, symbolic: bool = False) -> Union[np.ndarray, ca.MX]:
    opt_vars = []

    for j in range(num_joints):
        for node_idx, node in enumerate(traj[j]):
            opt_vars.append(node[0])
            opt_vars.append(node[1])
            opt_vars.append(node[2])
            opt_vars.append(node[3])
    if symbolic:
        opt_var = ca.vertcat(*opt_vars)
    else:
        opt_var = np.array(opt_vars)
    return opt_var


def Arr2NodeTraj(opt_var: Union[np.ndarray, ca.MX], num_joints: int, num_segments: int) -> NodeTrajectory:
    traj = []
    idx = 0  # 当前处理的数组索引
    for j in range(num_joints):
        joint_traj = []
        for _ in range(num_segments - 1):
            q = opt_var[idx]
            idx += 1
            dq = opt_var[idx]
            idx += 1
            ddq = opt_var[idx]
            idx += 1
            t = opt_var[idx]
            idx += 1
            if isinstance(q, np.ndarray):
                joint_traj.append((q.item(), dq.item(), ddq.item(), t.item()))
            else:
                joint_traj.append((q, dq, ddq, t))
        traj.append(joint_traj)
    return traj


def NodeTraj2Traj(
    node_traj: NodeTrajectory,
    num_joints: int,
    q_init: np.ndarray,
    dq_init: np.ndarray,
    ddq_init: np.ndarray,
    q_target: np.ndarray,
    dq_target: np.ndarray,
    ddq_target: np.ndarray,
    max_total_time: float,
    is_symbolic: bool,
) -> Trajectory:
    traj = []

    for joint_idx in range(num_joints):
        node_traj_cur = node_traj[joint_idx].copy()
        node_traj_cur.append((q_target[joint_idx], dq_target[joint_idx], ddq_target[joint_idx], max_total_time))

        start_time = 0
        start_q = q_init[joint_idx]
        start_dq = dq_init[joint_idx]
        start_ddq = ddq_init[joint_idx]

        traj_cur = []
        for node_idx in range(len(node_traj_cur)):
            node_cur = node_traj_cur[node_idx]
            q = node_cur[0]
            dq = node_cur[1]
            ddq = node_cur[2]
            t = node_cur[3]

            duration = t - start_time

            # A_temp = np.array(
            #     [
            #         [1, 0, 0, 0, 0, 0],
            #         [0, 1, 0, 0, 0, 0],
            #         [0, 0, 2, 0, 0, 0],
            #         [1, duration, duration**2, duration**3, duration**4, duration**5],
            #         [0, 1, 2 * duration, 3 * duration**2, 4 * duration**3, 5 * duration**4],
            #         [0, 0, 2, 6 * duration, 12 * duration**2, 20 * duration**3],
            #     ]
            # )
            A_inv_temp = np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0.5, 0, 0, 0],
                    [
                        -10 / duration**3,
                        -6 / duration**2,
                        -3 / (2 * duration),
                        10 / duration**3,
                        -4 / duration**2,
                        1 / (2 * duration),
                    ],
                    [
                        15 / duration**4,
                        8 / duration**3,
                        3 / (2 * duration**2),
                        -15 / duration**4,
                        7 / duration**3,
                        -1 / duration**2,
                    ],
                    [
                        -6 / duration**5,
                        -3 / duration**4,
                        -1 / (2 * duration**3),
                        6 / duration**5,
                        -3 / duration**4,
                        1 / (2 * duration**3),
                    ],
                ]
            )
            Q_temp = np.array([start_q, start_dq, start_ddq, q, dq, ddq])

            if is_symbolic:
                # A_inv = ca.MX.zeros(6, 6)
                # for i in range(6):
                #     for j in range(6):
                #         A_inv[i, j] = A_inv_temp[i, j]
                # Q = ca.MX.zeros(6)
                # for i in range(6):
                #     Q[i] = Q_temp[i]
                coeffs = []
                for i in range(6):
                    vec = A_inv_temp[i, :]
                    val = 0
                    for j in range(6):
                        val += vec[j] * Q_temp[j]
                    coeffs.append(val)
                coeffs = ca.vertcat(*coeffs)
            else:
                A_inv = A_inv_temp
                Q = Q_temp
                coeffs = A_inv @ Q

            traj_cur.append((start_time, t, coeffs))

            start_q = q
            start_dq = dq
            start_ddq = ddq
            start_time = t

        traj.append(traj_cur)

    return traj


def Traj2NodeTraj(
    traj: Trajectory,
    num_joints: int,
) -> Trajectory:
    node_traj = []

    for joint_idx in range(num_joints):
        traj_cur = traj[joint_idx]

        node_traj_cur = []
        for seg_idx in range(len(traj_cur) - 1):
            start_time, end_time, coeffs = traj_cur[seg_idx]
            duration = end_time - start_time
            t = end_time
            q = (
                coeffs[0]
                + duration * coeffs[1]
                + duration**2 * coeffs[2]
                + duration**3 * coeffs[3]
                + duration**4 * coeffs[4]
                + duration**5 * coeffs[5]
            )
            dq = (
                coeffs[1]
                + 2 * duration * coeffs[2]
                + 3 * duration**2 * coeffs[3]
                + 4 * duration**3 * coeffs[4]
                + 5 * duration**4 * coeffs[5]
            )
            ddq = 2 * coeffs[2] + 6 * duration * coeffs[3] + 12 * duration**2 * coeffs[4] + 20 * duration**3 * coeffs[5]

            node_traj_cur.append((q, dq, ddq, t))

        node_traj.append(node_traj_cur)

    return node_traj


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
                    [1, 0, 0**2, 0**3, 0**4, 0**5],
                    [0, 1, 2 * 0, 3 * 0**2, 4 * 0**3, 5 * 0**4],
                    [0, 0, 2, 6 * 0, 12 * 0**2, 20 * 0**3],
                    [1, duration, duration**2, duration**3, duration**4, duration**5],
                    [0, 1, 2 * duration, 3 * duration**2, 4 * duration**3, 5 * duration**4],
                    [0, 0, 2, 6 * duration, 12 * duration**2, 20 * duration**3],
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

    # def SphereSDFVisualize(
    #     self,
    #     meta_data: Dict,
    #     p: Union[List[float], np.ndarray],
    #     q: Union[List[float], np.ndarray],
    #     x: Union[List[float], np.ndarray],
    # ):
    #     link_name = meta_data["link_name"]
    #     mnt_joint = meta_data["mnt_joint"]
    #     info_idx = meta_data["info_idx"]

    #     world_from_base: ca.MX = ca.MX.eye(4)
    #     world_from_base[0, 3] = p[0]
    #     world_from_base[1, 3] = p[1]
    #     yaw = p[2]
    #     R_z = ca.vertcat(
    #         ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0), ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0), ca.horzcat(0, 0, 1)
    #     )
    #     world_from_base[:3, :3] = R_z

    #     if mnt_joint == "base_joint":
    #         world_from_joint = world_from_base
    #     else:
    #         base_from_joint = self.base_from_connect @ self.connect_from_joint_dict[mnt_joint][0]
    #         world_from_joint = world_from_base @ base_from_joint

    #     if link_name != "unknown":
    #         info = self.collision_info[link_name][0][info_idx]
    #         offset = info[0]
    #         radius = info[1]
    #         visual_id = info[2]

    #         # print(f"link_name: {link_name}, mnt_joint: {mnt_joint} ", info)

    #         joint_from_sphere = np.eye(4)
    #         joint_from_sphere[0, 3] = offset[0]
    #         joint_from_sphere[1, 3] = offset[1]
    #         joint_from_sphere[2, 3] = offset[2]
    #         world_from_sphere = world_from_joint @ joint_from_sphere
    #         sphere_center = ca.vertcat(world_from_sphere[0, 3], world_from_sphere[1, 3], world_from_sphere[2, 3])
    #         c = eval("", sphere_center, [self.q], [q])

    #         with pp.LockRenderer():
    #             if self.debug_sphere_visual_id == -1:
    #                 self.debug_sphere_visual_id = pp.create_sphere(radius, color=(1, 0, 0, 0.5))
    #             else:
    #                 pp.remove_body(self.debug_sphere_visual_id)
    #                 self.debug_sphere_visual_id = pp.create_sphere(radius, color=(1, 0, 0, 0.5))
    #             pp.set_point(self.debug_sphere_visual_id, c)

    #             if self.debug_line_visual_id == -1:
    #                 self.debug_line_visual_id = pp.add_line(x, c)
    #             else:
    #                 pp.remove_debug(self.debug_line_visual_id)
    #                 self.debug_line_visual_id = pp.add_line(x, c)

    def __call__(
        self,
        p: Union[List[float], np.ndarray, ca.MX],
        q: Union[List[float], np.ndarray, ca.MX],
        x: Union[List[float], np.ndarray, ca.MX],
        method: str = "sphere",
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

        return sdf


class SVSDF(object):
    """
    用于计算空间中的点到机器人关节轨迹的距离的类。
    """

    def __init__(
        self,
        urdf_path: str,
        robot_setup: RobotSetup,
        joint_trajectory: Trajectory,
        traj_var: Union[ca.MX, None] = None,
        symbolic_traj: bool = False,
        node_traj: bool = False,
    ):
        """
        初始化 SVSDF 计算类。

        Params:
            urdf_path (str): urdf路径
            robot_setup (RobotSetup): 机器人设置
            joint_trajectory (List[trajectory]):
                - trajectory: 机器人关节轨迹，由一系列关节轨迹片段组成，每个片段是一个元组 (start_time, end_time, coefficients)
                - start_time: 片段的起始时间 (ca.MX)
                - end_time: 片段的终止时间 (ca.MX)
                - coefficients: 5次多项式的系数，列表形式，长度为6，按照 t^0, t^1, t^2, t^3, t^4, t^5 的顺序排列 (ca.MX)
            traj_var (Union[ca.MX, None], (default) None): 如果传入符号化的轨迹，这里需要传入一个完整的变量,
            symbolic_traj (bool, (default) False): 是否使用符号化的关节轨迹
            node_traj (bool, (default) False): 是否使用节点化的关节轨迹
        """
        self.traj_sym = joint_trajectory
        self.num_joints = len(joint_trajectory)
        self.num_segments = len(joint_trajectory[0])
        self.symbolic_traj = symbolic_traj
        self.traj_sym_var = traj_var
        self.node_traj = node_traj

        # -------------------- 构建符号化计算系统 --------------------#
        self.t_sym = ca.MX.sym("t", 1)
        self.p_sym = ca.MX.sym("p", 3)  # [x, y, yaw]
        self.q_sym = self._BuildSymbolicQ(self.t_sym)  # joint positions
        self.x_sym = ca.MX.sym("x", 3)  # [x, y, z]

        # urdf_path = "/home/jeong/summer_research/eth_ws/src/husky_assembly/data/husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf"
        self.sdf_solver = SDF(urdf_path, robot_setup, self.q_sym)

        # SDF(p, q(t, c, T), x)
        self.sdf_sym = self.sdf_solver(self.p_sym, self.q_sym, self.x_sym)
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
        for joint_idx in range(len(self.traj_sym)):
            traj = self.traj_sym[joint_idx]
            for start, end, coeffs in traj:
                cond = ca.logic_and(t >= start, t <= end)
                delta_t = t - start
                poly = (
                    coeffs[0]
                    + coeffs[1] * delta_t
                    + coeffs[2] * (delta_t**2)
                    + coeffs[3] * (delta_t**3)
                    + coeffs[4] * (delta_t**4)
                    + coeffs[5] * (delta_t**5)
                )
                q[joint_idx] = ca.if_else(cond, poly, q[joint_idx])
        return q

    def _GradientDescent(
        self,
        p: np.ndarray,
        x: np.ndarray,
        traj: Union[NodeTrajectory, Trajectory, None],
        t_init: float,
        t_max: float,
        lr: float = 0.1,
        max_iter: int = 100,
    ) -> Tuple[float, float]:
        """
        带自适应学习率的梯度下降

        Params:
            p (np.ndarray): robot 2D pose
            x (np.ndarray): point in 3D space
            traj (NodeTrajectory | Trajectory | None): trajectory
            t_init (float): initial time
            t_max (float): max time
            lr (float, 0.1): learning rate
            max_iter (int, 100): maximum number of iterations

        Returns:
            (float, float): optimal time, sdf value at optimal time
        """
        t_curr = t_init
        prev_grad = None
        momentum = 0.9
        velocity = 0

        if self.symbolic_traj:
            if self.node_traj:
                jac_sym_sim = eval(
                    "",
                    self.jac_sym,
                    [self.traj_sym_var, self.x_sym, self.p_sym],
                    [NodeTraj2Arr(traj, self.num_joints), x, p],
                    full=False,
                )
            else:
                jac_sym_sim = eval(
                    "",
                    self.jac_sym,
                    [self.traj_sym_var, self.x_sym, self.p_sym],
                    [Traj2Arr(traj, self.num_joints), x, p],
                    full=False,
                )
        else:
            jac_sym_sim = eval("", self.jac_sym, [self.x_sym, self.p_sym], [x, p], full=False)

        for _ in range(max_iter):
            grad = eval("", jac_sym_sim, [self.t_sym], [t_curr]).item()

            # -------------------- 动量加速 --------------------#
            velocity = momentum * velocity + (1 - momentum) * grad
            delta_t = -lr * velocity

            # -------------------- 自适应步长 --------------------#
            if prev_grad is not None and np.sign(grad) != np.sign(prev_grad):
                lr *= 0.5

            t_new = t_curr + delta_t
            t_new = np.clip(t_new, 0, t_max)

            # -------------------- 收敛判断 --------------------#
            if abs(t_new - t_curr) < 1e-4:
                break
            t_curr = t_new
            prev_grad = grad

        if self.symbolic_traj:
            if self.node_traj:
                svsdf_curr = eval(
                    "",
                    self.sdf_sym,
                    [self.t_sym, self.x_sym, self.p_sym, self.traj_sym_var],
                    [t_curr, x, p, NodeTraj2Arr(traj, self.num_joints)],
                ).item()
            else:
                svsdf_curr = eval(
                    "",
                    self.sdf_sym,
                    [self.t_sym, self.x_sym, self.p_sym, self.traj_sym_var],
                    [t_curr, x, p, Traj2Arr(traj, self.num_joints)],
                ).item()
        else:
            svsdf_curr = eval("", self.sdf_sym, [self.t_sym, self.x_sym, self.p_sym], [t_curr, x, p]).item()

        return t_curr, svsdf_curr

    def EvaluateJointPosition(self, time: float, traj: Trajectory) -> np.ndarray:
        """
        计算在给定时间和关节轨迹片段索引下的关节位置。

        Params:
            time (float): 目标时间点 (float)
            traj (Trajectory): 关节轨迹

        Returns:
            np.ndarray: 关节角度
        """
        joint_angles = []
        for joint_index in range(self.num_joints):
            joint_trajectory = traj[joint_index]
            joint_angle = None

            out_range = True
            for start_time, end_time, coefficients in joint_trajectory:
                if start_time <= time <= end_time:
                    t = time - start_time
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
                t = end_time - start_time
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

    def __call__(
        self,
        p: np.ndarray,
        x: np.ndarray,
        t_max: float,
        t_seed: Union[float, None] = None,
        traj: Union[None, Trajectory, NodeTrajectory] = None,
        symbolic_output: bool = False,
    ) -> Tuple[float, Union[float, ca.MX]]:
        """
        计算点 x 到 SV 的最短距离。

        Params:
            p (np.ndarray): robot 2D pose
            x (float): point in 3D space
            t_max (float): max time
            traj (Trajectory | NodeTrajectory | None, (default) None): trajectory
            symbolic_output (bool, (default) False): whether return symbolic output

        Returns:
            ((float, float) | ca.MX): optimal time, svsdf SDF(p, q(t, c, T), x)
        """
        if symbolic_output:
            return self.sdf_sym

        min_sdf = float("inf")
        best_t = 0.0

        # init_points = np.linspace(0, t_max, 5)
        if t_seed is None:
            init_points = t_max * np.random.random(20)
            init_points.sort()
        else:
            init_points = [t_seed]

        for t_init in init_points:
            print(f"Running at time {t_init}")
            t_curr, sdf_val = self._GradientDescent(p, x, traj, t_init, t_max)
            print(f"Optimal time {t_curr} and value {sdf_val}")
            if sdf_val < min_sdf:
                min_sdf = sdf_val
                best_t = t_curr

        refined_t, refined_sdf = self._GradientDescent(p, x, traj, best_t, t_max, lr=0.01, max_iter=50)

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

    import time

    import pybullet as p
    import pybullet_planning as pp
    from utils.collision import Element, create_couplers, init_pb

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
    urdf_path = "/home/jeong/summer_research/eth_ws/src/husky_assembly/data/husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf"
    svsdf = SVSDF(urdf_path, rb, trajectory)

    # 计算点到轨迹的最小距离
    robot_pose = np.array([0, 0, 0])
    target_point = np.array([0.5, -0.35, 0.75])
    # target_point = np.array([0, 0, 0.2])
    svsdf_tup = svsdf(robot_pose, target_point, 3.0)

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
    # test_point = [0.5, -0.35, 0.75]
    # sphere_id = pp.create_sphere(0.05, color=pp.BLACK)
    # pp.set_point(sphere_id, test_point)

    # while True:
    #     # button_value = p.readUserDebugParameter(continue_button)
    #     # if button_value > prev_button_value:
    #     #     prev_button_value = button_value
    #     #     for t in test_times:
    #     #         pos = svsdf.EvaluateJointPosition(t)
    #     #         rb.set_joint_positions(rb.arm_joints, pos)
    #     #         time.sleep(3.0 / 30)

    #     slider_value = p.readUserDebugParameter(slider)
    #     time_idx = int(slider_value * (test_times.shape[0] - 1))
    #     t = test_times[time_idx]
    #     pos = svsdf.EvaluateJointPosition(t)
    #     rb.set_joint_positions(rb.arm_joints, pos)
    #     # svsdf.sdf_solver([0, 0, 0], pos, test_point, visualize=True)
    #     print(f"SVSDF Value at point {test_point} and time {t}: {svsdf.EvaluateSDF(t, [0, 0, 0], test_point)}")
    #     time.sleep(1.0 / 60)
