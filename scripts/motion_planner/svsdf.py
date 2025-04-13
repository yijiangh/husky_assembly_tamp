import os
import sys
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import casadi as ca
import numpy as np
import pybullet_planning as pp

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from robot.robot_setup import RobotSetup
from utils.collision import collision_info
from utils.util import HideOutput
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
            q = coeffs[0] + duration * coeffs[1] + duration**2 * coeffs[2] + duration**3 * coeffs[3] + duration**4 * coeffs[4] + duration**5 * coeffs[5]
            dq = coeffs[1] + 2 * duration * coeffs[2] + 3 * duration**2 * coeffs[3] + 4 * duration**3 * coeffs[4] + 5 * duration**4 * coeffs[5]
            ddq = 2 * coeffs[2] + 6 * duration * coeffs[3] + 12 * duration**2 * coeffs[4] + 20 * duration**3 * coeffs[5]

            node_traj_cur.append((q, dq, ddq, t))

        node_traj.append(node_traj_cur)

    return node_traj


def generate_trajectory(start_pos: np.ndarray, end_pos: np.ndarray, v_max: float = np.pi / 6, n_segments: int = 5) -> List[List[Tuple[float, float, List[float]]]]:
    # 计算每个关节的理论时间（避免除以零）
    delta_q = end_pos - start_pos
    joint_times = np.where(np.abs(delta_q) > 1e-6, np.abs(delta_q) / v_max, 1.0)  # 判断是否有位移  # 无位移关节默认分配 1.0 秒
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

    def __init__(
        self,
        urdf_path: str,
        robot: RobotSetup,
        q: Union[ca.MX, None] = None,
        p_sym: Union[ca.MX, None] = None,
        x_sym: Union[ca.MX, None] = None,
        grasp_offset: Union[List[float], np.ndarray, ca.MX, None] = None,
        grasp_spheres: Union[List[Dict], None] = None,
    ) -> None:
        """
        Initialize SDF calculator.

        Args:
            urdf_path (str): Path to the URDF file.
            robot (RobotSetup): Robot setup information.
            q (Union[ca.MX, None], optional): Symbolic joint variables. Defaults to None.
            p_sym (Union[ca.MX, None], optional): Symbolic base pose variables. Defaults to None.
            x_sym (Union[ca.MX, None], optional): Symbolic target point variables. Defaults to None.
            grasp_offset (Union[List[float], np.ndarray, ca.MX, None], optional):
                Translation offset [x, y, z] from 'ur_arm_tool0' frame to the grasp object's center frame.
                Defaults to None.
            grasp_spheres (Union[List[Dict], None], optional):
                List of sphere approximations for the grasped object, relative to its center frame.
                Each dict should be {'offset': [x,y,z], 'radius': r}. Defaults to None.
        """
        self.urdf_path = urdf_path
        self.robot = robot

        self.Nq = len(self.MANIPULATOR_CONTROL_JOINT_NAMES)
        if q is None:
            self.q = ca.MX.sym("q", self.Nq, 1)
        else:
            self.q = q

        # 添加p_sym和x_sym成员变量，用于符号计算
        if p_sym is None:
            self.p_sym = ca.MX.sym("p", 3)  # [x, y, yaw]
        else:
            self.p_sym = p_sym

        if x_sym is None:
            self.x_sym = ca.MX.sym("x", 3)  # [x, y, z]
        else:
            self.x_sym = x_sym

        self.collision_info = collision_info
        self.grasp_offset = grasp_offset
        self.grasp_spheres = grasp_spheres  # List of {'offset': [x,y,z], 'radius': r}

        # Convert grasp_offset to ca.MX if it's numeric for consistency
        if self.grasp_offset is not None and isinstance(self.grasp_offset, (list, np.ndarray)):
            self.grasp_offset_mx = ca.MX(self.grasp_offset)
        elif isinstance(self.grasp_offset, ca.MX):
            self.grasp_offset_mx = self.grasp_offset
        else:
            self.grasp_offset_mx = None

        self._BuildSymbolicFK()

    def _BuildSymbolicFK(self) -> None:
        # FK for robot links mounted on specific joints
        self.connect_from_j6_fn = RobotSetup.symbolic_forward(self.urdf_path, self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES, self.MANIPULATOR_CONTROL_JOINT_NAMES)
        self.connect_from_joint_dict = {}
        for joint_idx, joint_name in enumerate(self.MANIPULATOR_CONTROL_JOINT_NAMES):
            try:
                end_idx = self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES.index(joint_name) + 1
                fk_mat = RobotSetup.symbolic_forward(
                    self.urdf_path,
                    self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES[:end_idx],
                    self.MANIPULATOR_CONTROL_JOINT_NAMES[: joint_idx + 1],
                    q=self.q,
                    output_type="matrix",
                )
                self.connect_from_joint_dict[joint_name] = (fk_mat, joint_idx + 1)
            except ValueError:
                print(f"Warning: Joint '{joint_name}' not found in MANIPULATOR_REDUCED_MODEL_JOINT_NAMES. Skipping FK calculation for this joint's links.")

        # FK from base to robot connection point
        base_from_connect_sym = RobotSetup.symbolic_forward(self.urdf_path, self.BASE_REDUCED_MODEL_JOINT_NAMES, self.BASE_CONTROL_JOINT_NAMES, output_type="matrix")
        self.base_from_connect = eval("base_from_connect", base_from_connect_sym, [], [])

        # FK from robot connection point to tool0 frame
        try:
            tool0_joint_name = "ur_arm_flange-tool0"  # or potentially "tool0-bar_tcp_fixed_joint" if that's the end
            tool0_idx = self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES.index(tool0_joint_name) + 1
            fk_tool0_mat_sym = RobotSetup.symbolic_forward(
                self.urdf_path,
                self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES[:tool0_idx],
                self.MANIPULATOR_CONTROL_JOINT_NAMES,
                q=self.q,
                output_type="matrix",
            )
            self.fk_tool0_fn = ca.Function("fk_tool0", [self.q], [fk_tool0_mat_sym])
        except ValueError:
            print(f"Warning: Joint '{tool0_joint_name}' not found in MANIPULATOR_REDUCED_MODEL_JOINT_NAMES. Cannot compute FK to tool0.")
            self.fk_tool0_fn = None

    def SphereApproximation(
        self,
        p: Union[List[float], np.ndarray, ca.MX],
        q: Union[List[float], np.ndarray, ca.MX],
        x: Union[List[float], np.ndarray, ca.MX],
    ) -> Tuple[ca.MX, Dict]:

        # -------------------- 计算base的转移矩阵 --------------------#
        world_from_base: ca.MX = ca.MX.eye(4)
        if isinstance(p, (list, np.ndarray)):
            p_mx = ca.MX(p)
        else:
            p_mx = p  # Assume it's already MX
        world_from_base[0, 3] = p_mx[0]
        world_from_base[1, 3] = p_mx[1]
        yaw = p_mx[2]
        R_z = ca.vertcat(ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0), ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0), ca.horzcat(0, 0, 1))
        world_from_base[:3, :3] = R_z

        metadata_list = []
        sdf_list = []

        # -------------------- loop 1: 遍历robot links --------------------#
        for link_name in self.collision_info:
            infos = self.collision_info[link_name][0]
            mnt_joint = self.collision_info[link_name][1]
            weight = self.collision_info[link_name][2]  # Weight is currently unused in SDF calc

            # -------------------- 获取joint位置 --------------------#
            if mnt_joint == "base_joint":
                world_from_joint = world_from_base
            elif mnt_joint in self.connect_from_joint_dict:
                connect_from_joint, _ = self.connect_from_joint_dict[mnt_joint]
                base_from_joint = self.base_from_connect @ connect_from_joint
                world_from_joint = world_from_base @ base_from_joint
            else:
                # Skip links whose mounting joint FK couldn't be calculated
                continue

            # -------------------- loop 2: 遍历link上的碰撞小球 --------------------#
            for info_idx, info in enumerate(infos):
                offset = info[0]
                radius = info[1]

                # -------------------- 计算小球位置 --------------------#
                joint_from_sphere = ca.MX.eye(4)  # Use MX for consistency
                joint_from_sphere[:3, 3] = ca.MX(offset)

                world_from_sphere = world_from_joint @ joint_from_sphere
                c = world_from_sphere[:3, 3]  # Extract translation vector

                # -------------------- 计算距离和SDF --------------------#
                dist = ca.norm_2(x - c)
                sdf_i = dist - radius
                sdf_list.append(sdf_i)
                metadata_list.append({"link_name": link_name, "mnt_joint": mnt_joint, "info_idx": info_idx})

        # -------------------- loop 3: 遍历 grasped object spheres --------------------#
        if self.grasp_spheres is not None and self.fk_tool0_fn is not None and self.grasp_offset_mx is not None:
            # FK to tool0
            connect_from_tool0 = self.fk_tool0_fn(q)  # FK from connection point to tool0
            base_from_tool0 = self.base_from_connect @ connect_from_tool0
            world_from_tool0 = world_from_base @ base_from_tool0

            # Transformation from tool0 to grasp center
            tool0_from_grasp_center = ca.MX.eye(4)
            # Rotation (Roll=1.5708, Pitch=0, Yaw=0) - Fixed Euler ZYX or Extrinsic XYZ
            roll = 1.5708
            Rx = ca.vertcat(ca.horzcat(1, 0, 0), ca.horzcat(0, ca.cos(roll), -ca.sin(roll)), ca.horzcat(0, ca.sin(roll), ca.cos(roll)))
            # Assuming Extrinsic XYZ fixed angles rotation order. R = Rz(0)Ry(0)Rx(roll) = Rx
            tool0_from_grasp_center[:3, :3] = Rx

            # Translation
            tool0_from_grasp_center[:3, 3] = self.grasp_offset_mx[:3]  # Use stored MX version

            # World to grasp center
            world_from_grasp_center = world_from_tool0 @ tool0_from_grasp_center

            # Loop through grasp spheres (relative to grasp center)
            for sphere_idx, grasp_sphere in enumerate(self.grasp_spheres):
                # grasp_sphere is expected to be like {'offset': [x,y,z], 'radius': r}
                sphere_offset = ca.MX(grasp_sphere["offset"])
                sphere_radius = grasp_sphere["radius"]

                grasp_center_from_sphere = ca.MX.eye(4)
                grasp_center_from_sphere[:3, 3] = sphere_offset

                world_from_sphere_grasp = world_from_grasp_center @ grasp_center_from_sphere
                c_grasp = world_from_sphere_grasp[:3, 3]

                dist_grasp = ca.norm_2(x - c_grasp)
                sdf_grasp_i = dist_grasp - sphere_radius
                sdf_list.append(sdf_grasp_i)
                metadata_list.append({"link_name": "grasped_object", "mnt_joint": "grasp", "info_idx": sphere_idx})

        # -------------------- 计算最终SDF --------------------#
        if not sdf_list:  # Handle case where no spheres are defined or FK failed
            sdf_robot = ca.MX(float("inf"))  # Return infinity if no spheres
            min_metadata = {"link_name": "none", "mnt_joint": "none", "info_idx": -1}
        else:
            sdf_vec = ca.vertcat(*sdf_list)
            sdf_robot = ca.mmin(sdf_vec)

            # -------------------- 判断是否数值输出 --------------------#
            is_numeric = all(isinstance(arg, (list, np.ndarray)) for arg in [p, q, x])

            if is_numeric:
                # Evaluate the symbolic expression numerically
                q_numeric = q if isinstance(q, np.ndarray) else np.array(q).flatten()
                p_numeric = p if isinstance(p, np.ndarray) else np.array(p).flatten()
                x_numeric = x if isinstance(x, np.ndarray) else np.array(x).flatten()

                # 使用正确的符号变量进行计算
                sdf_val = float(eval("SphereApproximation", sdf_robot, [self.q, self.p_sym, self.x_sym], [q_numeric, p_numeric, x_numeric]).item())
                sdf_values = eval("SphereApproximationList", sdf_vec, [self.q, self.p_sym, self.x_sym], [q_numeric, p_numeric, x_numeric]).toarray().flatten()

                if len(sdf_values) > 0:
                    min_index = np.argmin(sdf_values)
                    if min_index < len(metadata_list):
                        min_metadata = metadata_list[min_index]
                    else:
                        # Should not happen if lists are built correctly
                        min_metadata = {"link_name": "error", "mnt_joint": "error", "info_idx": -1}
                else:
                    min_metadata = {"link_name": "none", "mnt_joint": "none", "info_idx": -1}
                sdf = sdf_val  # Use the numerically evaluated single value
            else:
                sdf = sdf_robot  # Return the symbolic expression
                # Cannot determine min_metadata symbolically
                min_metadata = {"link_name": "symbolic", "mnt_joint": "symbolic", "info_idx": -1}

        return sdf, min_metadata

    def SphereSDFVisualize(
        self,
        meta_data: Dict,
        p: Union[List[float], np.ndarray],
        q: Union[List[float], np.ndarray],
        x: Union[List[float], np.ndarray],
    ):
        """Visualize the sphere corresponding to the minimum SDF."""
        link_name = meta_data["link_name"]
        mnt_joint = meta_data["mnt_joint"]
        info_idx = meta_data["info_idx"]

        # Ensure inputs are numpy arrays
        p_np = np.array(p).flatten()
        q_np = np.array(q).flatten()
        x_np = np.array(x).flatten()

        if link_name == "none" or link_name == "error" or link_name == "symbolic":
            print(f"Cannot visualize SDF sphere for metadata: {meta_data}")
            # Optionally remove existing debug items if they exist
            if self.debug_sphere_visual_id != -1:
                try:
                    pp.remove_body(self.debug_sphere_visual_id)
                except Exception:
                    pass  # Ignore if body doesn't exist
            if self.debug_line_visual_id != -1:
                try:
                    pp.remove_debug(self.debug_line_visual_id)
                except Exception:
                    pass  # Ignore if debug item doesn't exist
            return

        world_from_base_np = pp.tform_from_pose(pp.Pose(point=[p_np[0], p_np[1], 0], euler=[0, 0, p_np[2]]))
        base_from_connect_np = self.base_from_connect  # This is already evaluated

        c = None  # Sphere center
        radius = None  # Sphere radius

        if link_name == "grasped_object":
            if self.grasp_spheres is not None and self.fk_tool0_fn is not None and self.grasp_offset_mx is not None and info_idx < len(self.grasp_spheres):
                # Evaluate FK to tool0 numerically
                fk_tool0_sym = self.fk_tool0_fn(self.q)
                connect_from_tool0_np = eval("fk_tool0_numeric", fk_tool0_sym, [self.q], [q_np])

                base_from_tool0_np = base_from_connect_np @ connect_from_tool0_np
                world_from_tool0_np = world_from_base_np @ base_from_tool0_np

                # Tool0 to grasp center transform (numeric)
                tool0_from_grasp_center_np = np.eye(4)
                # Rotation (Roll=1.5708, Pitch=0, Yaw=0)
                rot_quat = pp.quat_from_euler([1.5708, 0, 0])
                tool0_from_grasp_center_np[:3, :3] = pp.matrix_from_quat(rot_quat)
                # Translation
                grasp_offset_np = eval("grasp_offset_numeric", self.grasp_offset_mx, [], [])  # Evaluate if symbolic
                tool0_from_grasp_center_np[:3, 3] = grasp_offset_np[:3].flatten()

                world_from_grasp_center_np = world_from_tool0_np @ tool0_from_grasp_center_np

                # Get specific sphere info
                grasp_sphere = self.grasp_spheres[info_idx]
                sphere_offset_np = np.array(grasp_sphere["offset"]).flatten()
                radius = grasp_sphere["radius"]

                grasp_center_from_sphere_np = np.eye(4)
                grasp_center_from_sphere_np[:3, 3] = sphere_offset_np

                world_from_sphere_grasp_np = world_from_grasp_center_np @ grasp_center_from_sphere_np
                c = world_from_sphere_grasp_np[:3, 3]
            else:
                print("Cannot visualize grasped object: Missing data or FK function.")
                return
        elif mnt_joint == "base_joint":
            world_from_joint_np = world_from_base_np
            info = self.collision_info[link_name][0][info_idx]
            offset_np = np.array(info[0]).flatten()
            radius = info[1]
            # visual_id = info[2] # Unused here

            joint_from_sphere_np = np.eye(4)
            joint_from_sphere_np[:3, 3] = offset_np
            world_from_sphere_np = world_from_joint_np @ joint_from_sphere_np
            c = world_from_sphere_np[:3, 3]
        elif mnt_joint in self.connect_from_joint_dict:
            connect_from_joint_sym, _ = self.connect_from_joint_dict[mnt_joint]
            connect_from_joint_np = eval("fk_joint_numeric", connect_from_joint_sym, [self.q], [q_np])

            base_from_joint_np = base_from_connect_np @ connect_from_joint_np
            world_from_joint_np = world_from_base_np @ base_from_joint_np

            info = self.collision_info[link_name][0][info_idx]
            offset_np = np.array(info[0]).flatten()
            radius = info[1]
            # visual_id = info[2] # Unused here

            joint_from_sphere_np = np.eye(4)
            joint_from_sphere_np[:3, 3] = offset_np
            world_from_sphere_np = world_from_joint_np @ joint_from_sphere_np
            c = world_from_sphere_np[:3, 3]
        else:
            print(f"Cannot visualize sphere: Mount joint '{mnt_joint}' FK not available.")
            return

        # Perform visualization if center and radius are valid
        if c is not None and radius is not None:
            with pp.LockRenderer():
                if self.debug_sphere_visual_id != -1:
                    try:
                        pp.remove_body(self.debug_sphere_visual_id)
                    except Exception:
                        pass  # Ignore error if body removed elsewhere
                    self.debug_sphere_visual_id = pp.create_sphere(radius, color=(1, 0, 0, 0.5))
                pp.set_point(self.debug_sphere_visual_id, c)

                if self.debug_line_visual_id != -1:
                    try:
                        pp.remove_debug(self.debug_line_visual_id)
                    except Exception:
                        pass  # Ignore error if debug item removed elsewhere
                self.debug_line_visual_id = pp.add_line(x_np, c, color=pp.RED)
        else:
            # Clean up debug items if calculation failed
            if self.debug_sphere_visual_id != -1:
                try:
                    pp.remove_body(self.debug_sphere_visual_id)
                except Exception:
                    pass
                self.debug_sphere_visual_id = -1
            if self.debug_line_visual_id != -1:
                try:
                    pp.remove_debug(self.debug_line_visual_id)
                except Exception:
                    pass
                self.debug_line_visual_id = -1

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
            (float | ca.MX): SDF value. Returns symbolic MX if any input is symbolic, otherwise float.
        """
        if method == "sphere":
            sdf, meta_info = self.SphereApproximation(p, q, x)
        else:
            raise NotImplementedError(f"Method {method} is not implemented.")

        # Check if inputs were numeric to decide whether to visualize
        is_numeric = all(isinstance(arg, (list, np.ndarray)) for arg in [p, q, x])
        if visualize and is_numeric:
            self.SphereSDFVisualize(meta_info, p, q, x)
        elif visualize and not is_numeric:
            print("Warning: Cannot visualize SDF with symbolic inputs.")

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
        obstacle_spheres: List[np.ndarray],  # External obstacles
        robot_pose: np.ndarray,  # Base pose [x, y, yaw]
        grasp_offset: Union[List[float], np.ndarray, None] = None,  # Grasped object offset rel to tool0
        grasp_spheres: Union[List[Dict], None] = None,  # Grasped object spheres rel to its center
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
            obstacle_spheres (List[np.ndarray]): List of external obstacle sphere centers [x, y, z].
            robot_pose (np.ndarray): Robot base pose [x, y, yaw].
            grasp_offset (Union[List[float], np.ndarray, None], optional): Grasped object offset. Defaults to None.
            grasp_spheres (Union[List[Dict], None], optional): Grasped object spheres. Defaults to None.
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
        self.robot_pose = robot_pose  # Store robot pose

        # -------------------- 构建符号化计算系统 --------------------#
        self.t_sym = ca.MX.sym("t", 1)
        self.p_sym = ca.MX.sym("p", 3)  # Base pose [x, y, yaw] - Used internally by SDF
        self.x_sym = ca.MX.sym("x", 3)  # Obstacle center [x, y, z] - Used internally by SDF

        if symbolic_traj:
            self.q_sym = self._BuildSymbolicQ(self.t_sym)  # joint positions q(t)
        else:
            # If traj is not symbolic, q_sym is needed as input to SDF's internal symbolic graph
            self.q_sym = ca.MX.sym("q", self.num_joints)

        # 初始化SDF计算器，传递符号变量self.q_sym, self.p_sym和self.x_sym，确保SDF内部能够使用它们
        self.sdf_solver = SDF(urdf_path, robot_setup, self.q_sym, self.p_sym, self.x_sym, grasp_offset=grasp_offset, grasp_spheres=grasp_spheres)

        # Build the SVSDF expression: min_t { min_x { SDF(p, q(t), x) } }
        # Here we calculate min_t { SDF(p, q(t), x_obs) } for a *specific* obstacle x_obs
        # The outer loop (in the planner) will handle minimizing over different obstacles x_obs.

        # sdf_sym_at_t represents SDF(robot_pose, q(t), obstacle_center)
        self.sdf_sym_at_t = self.sdf_solver(p=self.p_sym, q=self.q_sym, x=self.x_sym)

        # Specialize the symbolic SDF function for the fixed robot pose and a symbolic obstacle center
        self.sdf_func_sym_t_x = ca.Function("sdf_t_x", [self.t_sym, self.x_sym], [self.sdf_sym_at_t], ["t", "x_obs"], ["sdf"])

        # Gradient of SDF w.r.t. time 't', for a given obstacle 'x'
        self.jac_sym_t_x = ca.gradient(self.sdf_sym_at_t, self.t_sym)
        self.jac_func_sym_t_x = ca.Function("jac_t_x", [self.t_sym, self.x_sym], [self.jac_sym_t_x], ["t", "x_obs"], ["dsdf_dt"])

        # If trajectory is symbolic, we need functions that also take traj_var
        if self.symbolic_traj:
            if self.node_traj:
                self.sdf_func_sym_t_x_traj = ca.Function("sdf_t_x_traj", [self.t_sym, self.x_sym, self.traj_sym_var], [self.sdf_sym_at_t], ["t", "x_obs", "traj_var"], ["sdf"])
                self.jac_func_sym_t_x_traj = ca.Function("jac_t_x_traj", [self.t_sym, self.x_sym, self.traj_sym_var], [self.jac_sym_t_x], ["t", "x_obs", "traj_var"], ["dsdf_dt"])
            else:  # Polynomial trajectory
                self.sdf_func_sym_t_x_traj = ca.Function("sdf_t_x_traj", [self.t_sym, self.x_sym, self.traj_sym_var], [self.sdf_sym_at_t], ["t", "x_obs", "traj_var"], ["sdf"])
                self.jac_func_sym_t_x_traj = ca.Function("jac_t_x_traj", [self.t_sym, self.x_sym, self.traj_sym_var], [self.jac_sym_t_x], ["t", "x_obs", "traj_var"], ["dsdf_dt"])

    def _BuildSymbolicQ(self, t: ca.MX) -> ca.MX:
        """
        构建符号化的关节位置 q(t)

        Params:
            t (ca.MX): time variable

        Returns:
            ca.MX: q(t)
        """
        q = ca.MX.zeros(self.num_joints)

        # Reconstruct trajectory from traj_var if needed
        if self.node_traj:
            # This assumes traj_var directly represents the node trajectory array
            # We need to convert Arr2NodeTraj to be symbolic if needed, or assume traj_sym is pre-built symbolically
            # For now, assume self.traj_sym *is* the symbolic node trajectory structure
            # Or, more likely, q should be built from traj_var, not self.traj_sym directly
            raise NotImplementedError("Symbolic Q from Node Trajectory Variable not fully implemented")
            # Placeholder:
            # traj_nodes = Arr2NodeTraj(self.traj_sym_var, self.num_joints, self.num_segments) # Needs symbolic Arr2NodeTraj
            # traj_poly = NodeTraj2Traj(...) # Needs symbolic NodeTraj2Traj
            # Then use the logic below with traj_poly
        else:  # Polynomial trajectory from traj_var
            traj_poly = Arr2Traj(self.traj_sym_var, self.num_joints, self.num_segments)  # Assume Arr2Traj works symbolically

        # Calculate q(t) from the polynomial trajectory representation
        for joint_idx in range(self.num_joints):
            joint_poly_traj = traj_poly[joint_idx]
            q_joint = ca.MX(0)  # Default value if t is outside all segments (shouldn't happen ideally)
            # Iterate backwards to prioritize later segments if intervals overlap (they shouldn't)
            for seg_idx in range(len(joint_poly_traj) - 1, -1, -1):
                start_time, end_time, coefficients = joint_poly_traj[seg_idx]
                # Ensure coefficients are MX for symbolic calculation
                coeffs_mx = ca.MX(coefficients)

                cond = ca.logic_and(t >= start_time, t <= end_time)
                delta_t = t - start_time
                poly = coeffs_mx[0] + coeffs_mx[1] * delta_t + coeffs_mx[2] * (delta_t**2) + coeffs_mx[3] * (delta_t**3) + coeffs_mx[4] * (delta_t**4) + coeffs_mx[5] * (delta_t**5)
                q_joint = ca.if_else(cond, poly, q_joint)
            q[joint_idx] = q_joint
        return q

    def _GradientDescent(
        self,
        obstacle_center: np.ndarray,  # Center of the obstacle to check against
        traj: Union[NodeTrajectory, Trajectory, None],  # Numeric trajectory if not symbolic_traj
        t_init: float,
        t_max: float,
        lr: float = 0.1,
        max_iter: int = 100,
    ) -> Tuple[float, float]:
        """
        Gradient descent to find the minimum SDF value w.r.t. time for a specific obstacle.

        Params:
            obstacle_center (np.ndarray): Center [x,y,z] of the specific obstacle.
            traj (NodeTrajectory | Trajectory | None): Numeric trajectory if self.symbolic_traj is False.
            t_init (float): Initial time guess.
            t_max (float): Maximum trajectory time.
            lr (float, 0.1): Learning rate.
            max_iter (int, 100): Maximum number of iterations.

        Returns:
            (float, float): Optimal time t*, minimum SDF value SDF(t*) for the given obstacle.
        """
        t_curr = t_init
        prev_grad_sign = None
        momentum = 0.9
        velocity = 0

        # Select appropriate gradient and SDF functions based on whether trajectory is symbolic
        if self.symbolic_traj:
            if self.node_traj:
                jac_func = self.jac_func_sym_t_x_traj
                sdf_func = self.sdf_func_sym_t_x_traj
                traj_arr = NodeTraj2Arr(traj, self.num_joints)  # Assumes traj is provided for symbolic case too? Confusing.
                # Let's assume traj_var is used directly if symbolic_traj is True
                traj_input = self.traj_sym_var
            else:  # Symbolic polynomial trajectory
                jac_func = self.jac_func_sym_t_x_traj
                sdf_func = self.sdf_func_sym_t_x_traj
                # traj_arr = Traj2Arr(traj, self.num_joints) # Assume traj_var is the input
                traj_input = self.traj_sym_var
        else:  # Numeric trajectory
            jac_func = self.jac_func_sym_t_x  # Takes only t and x_obs
            sdf_func = self.sdf_func_sym_t_x  # Takes only t and x_obs

            # We need to evaluate q(t) numerically inside the loop if SDF depends on non-symbolic q
            q_numeric_t = self.EvaluateJointPosition(self.t_sym, traj)  # Get numeric q at symbolic t

            # SDF(p, q_numeric(t), x) - 使用SDF的x_sym符号变量而不是SVSDF的
            sdf_numeric_t_x = self.sdf_solver(p=self.robot_pose, q=q_numeric_t, x=self.x_sym)
            jac_numeric_t_x = ca.gradient(sdf_numeric_t_x, self.t_sym)

            # 更新函数定义，使用x_sym
            sdf_func = ca.Function("sdf_numeric_t_x", [self.t_sym, self.x_sym], [sdf_numeric_t_x])
            jac_func = ca.Function("jac_numeric_t_x", [self.t_sym, self.x_sym], [jac_numeric_t_x])

        for iter_count in range(max_iter):
            # Evaluate gradient
            if self.symbolic_traj:
                grad = jac_func(t=t_curr, x_obs=obstacle_center, traj_var=traj_input)["dsdf_dt"].toarray().item()
            else:  # Numeric trajectory
                grad = jac_func(t_curr, obstacle_center).toarray().item()

            # Stop if gradient is negligible
            if abs(grad) < 1e-5:
                # print(f"Gradient descent converged at iter {iter_count} due to small gradient.")
                break

            # -------------------- Momentum --------------------#
            # velocity = momentum * velocity + (1 - momentum) * grad # Simple momentum, maybe less stable
            velocity = momentum * velocity + grad  # Nesterov-style can be velocity = momentum * velocity - lr * grad; update = t_curr + momentum * velocity - lr * grad
            delta_t = -lr * velocity

            # -------------------- Adaptive Learning Rate (Bold Driver) --------------------#
            # Check sign change vs previous gradient
            current_grad_sign = np.sign(grad)
            if prev_grad_sign is not None and current_grad_sign != prev_grad_sign:
                lr *= 0.5  # Decrease LR on sign change
                velocity = 0  # Reset velocity on direction change
                # print(f"Iter {iter_count}: Grad sign changed, LR -> {lr:.4f}")
            else:
                lr *= 1.05  # Gently increase LR otherwise
                # print(f"Iter {iter_count}: Grad sign same, LR -> {lr:.4f}")
            lr = max(lr, 1e-5)  # Prevent LR from becoming too small

            t_new = t_curr + delta_t
            t_new = np.clip(t_new, 0, t_max)  # Project back into valid time range

            # -------------------- Convergence Check --------------------#
            if abs(t_new - t_curr) < 1e-4:
                # print(f"Gradient descent converged at iter {iter_count} due to small step size.")
                break

            t_curr = t_new
            prev_grad_sign = current_grad_sign

        # Evaluate final SDF value
        if self.symbolic_traj:
            svsdf_curr = sdf_func(t=t_curr, x_obs=obstacle_center, traj_var=traj_input)["sdf"].toarray().item()
        else:  # Numeric trajectory
            svsdf_curr = sdf_func(t_curr, obstacle_center).toarray().item()

        return t_curr, svsdf_curr

    def EvaluateJointPosition(self, time: Union[float, ca.MX], traj: Trajectory) -> Union[np.ndarray, ca.MX]:
        """
        计算在给定时间和关节轨迹片段索引下的关节位置。
        Handles both numeric time and symbolic time (ca.MX).

        Params:
            time (Union[float, ca.MX]): 目标时间点
            traj (Trajectory): 关节轨迹 (assumed numeric coefficients here for numeric q(t) evaluation)

        Returns:
            Union[np.ndarray, ca.MX]: 关节角度
        """
        is_symbolic_time = isinstance(time, ca.MX)

        if is_symbolic_time:
            q_out = ca.MX.zeros(self.num_joints)
        else:
            joint_angles = []

            for joint_index in range(self.num_joints):
                joint_trajectory = traj[joint_index]

                if is_symbolic_time:
                    q_joint = ca.MX(0)  # Default if time is outside all segments
                    # Iterate backwards for correct if_else logic
                    for seg_idx in range(len(joint_trajectory) - 1, -1, -1):
                        start_time, end_time, coefficients = joint_trajectory[seg_idx]
                        # Ensure coefficients are MX for symbolic calculation
                        coeffs_mx = ca.MX(coefficients)

                        cond = ca.logic_and(time >= start_time, time <= end_time)
                        t_rel = time - start_time
                        poly = coeffs_mx[0] + coeffs_mx[1] * t_rel + coeffs_mx[2] * (t_rel**2) + coeffs_mx[3] * (t_rel**3) + coeffs_mx[4] * (t_rel**4) + coeffs_mx[5] * (t_rel**5)
                        q_joint = ca.if_else(cond, poly, q_joint)
                    q_out[joint_index] = q_joint
                else:  # Numeric time
                    joint_angle = None
                    out_range = True
                    for start_time, end_time, coefficients in joint_trajectory:
                        if start_time <= time <= end_time:
                            # Ensure start_time is float for subtraction
                            t_rel = time - float(start_time)
                            # Ensure coefficients are floats/numeric
                            coeffs_num = np.array(coefficients).astype(float)

                            joint_angle = coeffs_num[0] + coeffs_num[1] * t_rel + coeffs_num[2] * (t_rel**2) + coeffs_num[3] * (t_rel**3) + coeffs_num[4] * (t_rel**4) + coeffs_num[5] * (t_rel**5)
                            out_range = False
                            break

                    # If time is beyond the last segment, evaluate at the end of the last segment
                    if out_range and joint_trajectory:
                        start_time, end_time, coefficients = joint_trajectory[-1]
                        t_rel = float(end_time) - float(start_time)
                        coeffs_num = np.array(coefficients).astype(float)
                        joint_angle = coeffs_num[0] + coeffs_num[1] * t_rel + coeffs_num[2] * (t_rel**2) + coeffs_num[3] * (t_rel**3) + coeffs_num[4] * (t_rel**4) + coeffs_num[5] * (t_rel**5)

                if joint_angle is None:
                    # This case should ideally not be reached if trajectory is well-defined
                    print(f"Warning: Time {time} seems invalid for joint {joint_index}. Defaulting to 0.")
                    joint_angle = 0.0

                joint_angles.append(joint_angle)

            return q_out if is_symbolic_time else np.array(joint_angles)

    def __call__(
        self,
        obstacle_center: np.ndarray,  # Center of the obstacle to compute SVSDF against
        t_max: float,
        t_seed: Union[float, None] = None,
        traj: Union[None, Trajectory, NodeTrajectory] = None,  # Numeric trajectory if not symbolic
        sdf_threshold: float = 0.0,
        num_init_points: int = 5,  # Number of points for multi-start gradient descent
    ) -> Tuple[float, float, List[Tuple[float, float]]]:
        """
        计算SVSDF：给定障碍物中心点，找出轨迹上SDF最小的点。

        Params:
            obstacle_center (np.ndarray): The center [x,y,z] of the obstacle to check against.
            t_max (float): Maximum time duration of the trajectory.
            t_seed (Union[float, None], optional): A specific initial time guess. If None, multiple points are used. Defaults to None.
            traj (Union[None, Trajectory, NodeTrajectory], optional):
                The numeric trajectory representation. Required if self.symbolic_traj is False. Defaults to None.
            sdf_threshold (float, optional): Threshold below which points are considered 'in collision'. Defaults to 0.0.
            num_init_points (int, optional): Number of initial time guesses for multi-start optimization. Defaults to 5.


        Returns:
            Tuple[float, float, List[Tuple[float, float]]]:
                - best_t (float): Time t* at which the minimum SDF occurs for this obstacle.
                - min_sdf (float): The minimum SDF value SDF(t*) for this obstacle.
                - collision_times (List[Tuple[float, float]]): List of (time, sdf_value) for points found below the threshold.
        """

        if not self.symbolic_traj and traj is None:
            raise ValueError("Numeric trajectory 'traj' must be provided when 'symbolic_traj' is False.")

        min_sdf = float("inf")
        best_t = 0.0
        collision_times = []  # Store (t, sdf) pairs below threshold

        # Determine initial points for gradient descent
        if t_seed is not None:
            init_points = [np.clip(t_seed, 0, t_max)]
        else:
            init_points = np.linspace(0, t_max, num_init_points)

        # Multi-start gradient descent
        results = []
        for t_init in init_points:
            # print(f"Running GD for obstacle {obstacle_center} starting at t={t_init:.3f}")
            t_opt, sdf_val = self._GradientDescent(obstacle_center, traj, t_init, t_max)
            results.append((t_opt, sdf_val))
            # print(f"  Result: t={t_opt:.4f}, sdf={sdf_val:.4f}")

        # Find the overall minimum from all starting points
        if results:
            best_t, min_sdf = min(results, key=lambda item: item[1])

        # Optional refinement step around the best minimum found
        # print(f"Refining around best t={best_t:.4f} (sdf={min_sdf:.4f})")
        refined_t, refined_sdf = self._GradientDescent(obstacle_center, traj, best_t, t_max, lr=0.01, max_iter=50)
        # print(f"  Refined: t={refined_t:.4f}, sdf={refined_sdf:.4f}")

        if refined_sdf < min_sdf:
            best_t = refined_t
            min_sdf = refined_sdf

        # Check all found local minima (including refined) for collisions
        all_results_final = results + [(refined_t, refined_sdf)]
        collision_candidates = {}  # Use dict to store unique collision times
        for t, sdf in all_results_final:
            if sdf <= sdf_threshold:
                # Store the one with the lowest sdf for a given time (or close times)
                found_close = False
                for existing_t in list(collision_candidates.keys()):
                    if abs(t - existing_t) < 1e-3:  # Group close times
                        if sdf < collision_candidates[existing_t]:
                            del collision_candidates[existing_t]  # Remove worse one
                            collision_candidates[t] = sdf  # Add better one
                        found_close = True
                        break
                if not found_close:
                    collision_candidates[t] = sdf

        collision_times = list(collision_candidates.items())

        return best_t, min_sdf, collision_times


if __name__ == "__main__":

    # **************************************************************************
    # SDF test
    # **************************************************************************

    import time

    import pybullet as p
    import pybullet_planning as pp
    from multi_tangent.collision import create_collision_bodies
    from utils.collision import Element, create_couplers, init_pb, element_collision_info

    urdf_path = "/home/jeong/summer_research/eth_ws/src/husky_assembly/data/husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf"

    init_pb()

    rb = RobotSetup()

    # --- Test with grasped object ---
    grasp_offset = [0.0, 0.1, 0.15]  # Example offset from tool0
    # Example grasped object: a sphere at its center
    grasp_spheres = element_collision_info

    line_pts_grasped = [np.array([0, 0, 0]), np.array([0, 0, 1])]
    grasped_element = create_collision_bodies(line_pts_grasped, [0.01], viewer=True)[0]
    pp.set_pose(
        grasped_element,
        pp.multiply(
            pp.get_link_pose(rb.robot, rb.tool_link),
            pp.Pose(point=grasp_offset, euler=pp.Euler(1.5708, 0, 0)),
        ),
    )

    # Symbolic variables for testing analytical evaluation (optional)
    x_sym = ca.MX.sym("x", 3)
    p_sym = ca.MX.sym("p", 3)
    q_sym = ca.MX.sym("q", 6)  # Use 6 for UR5

    # 更新SDF初始化
    sdf_calculator = SDF(urdf_path, rb, q=q_sym, p_sym=p_sym, x_sym=x_sym, grasp_offset=grasp_offset, grasp_spheres=grasp_spheres)
    # --- End grasped object test setup ---

    # 如果没有抓取物体
    # sdf_calculator = SDF(urdf_path, rb, q=q_sym, p_sym=p_sym, x_sym=x_sym)  # 传递符号变量

    x_slider = p.addUserDebugParameter("x", -2, 2, 0)
    y_slider = p.addUserDebugParameter("y", -2, 2, 0)
    z_slider = p.addUserDebugParameter("z", -2, 2, 0.5)  # Start point for testing

    point_id = pp.create_sphere(0.02, color=pp.BLACK)

    print("Starting SDF visualization loop...")
    while True:
        if not pp.is_connected():
            break  # Exit if simulator closed

        x_value = p.readUserDebugParameter(x_slider)
        y_value = p.readUserDebugParameter(y_slider)
        z_value = p.readUserDebugParameter(z_slider)
        target_point = np.array([x_value, y_value, z_value])

        pp.set_point(point_id, target_point)

        # Base pose and joint angles for testing
        test_p = np.array([0.0, 0.0, 0.0])
        test_q = rb.arm_init_angles

        # -------------------- sphere：数值计算 (with visualization) --------------------#
        # This call will now include the grasped object if initialized
        sdf_val_numeric = sdf_calculator(test_p, test_q, target_point, visualize=True)
        print(f"SDF numeric (target=[{x_value:.2f},{y_value:.2f},{z_value:.2f}]): {sdf_val_numeric:.4f}", end="\r")

        # -------------------- sphere：解析计算 (optional test) --------------------#
        # sdf_symbolic = sdf_calculator(p_sym, q_sym, x_sym) # Get symbolic expression
        # sdf_val_analytical = eval(
        #     "sdf_analytical",
        #     sdf_symbolic,
        #     [p_sym, q_sym, x_sym],
        #     [test_p, test_q, target_point]
        # ).toarray().item()
        # print(f"SDF analytical: {sdf_val_analytical:.4f}")

        time.sleep(1.0 / 240.0)  # Limit loop speed slightly

    print("SDF visualization loop finished.")
    pp.disconnect()

    # # **************************************************************************
    # # SVSDF test
    # # **************************************************************************

    # import time

    # import pybullet as p
    # import pybullet_planning as pp
    # from utils.collision import Element, create_couplers, init_pb

    # np.set_printoptions(precision=3)

    # init_pb()

    # rb = RobotSetup()

    # # 定义轨迹参数
    # start_pos = np.array([0, 0, 0, 0, 0, 0])
    # end_pos = np.array([0, -np.pi / 2, -np.pi / 2, 0, 0, 0])
    # v_max = np.pi / 6
    # n_segments = 5 # Match generate_trajectory default if used

    # # 生成轨迹 (numeric)
    # traj = generate_trajectory(start_pos, end_pos, v_max, n_segments=n_segments)
    # max_time = max([temp[-1][1] for temp in traj])

    # # 定义相关参数
    # robot_pose = np.array([0, 0, 0])
    # # Single external obstacle sphere for testing SVSDF
    # obstacle_center = np.array([0.5, -0.35, 0.75])

    # # --- Test SVSDF with grasped object ---
    # grasp_offset = [0.0, 0.0, 0.1]
    # grasp_spheres = [{'offset': [0.0, 0.0, 0.0], 'radius': 0.05}]
    # urdf_path = "/home/jeong/summer_research/eth_ws/src/husky_assembly/data/husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf"

    # # Create SVSDF instance (using numeric trajectory)
    # svsdf_calculator = SVSDF(
    #     urdf_path, rb, traj, [obstacle_center], robot_pose,
    #     grasp_offset=grasp_offset, grasp_spheres=grasp_spheres,
    #     symbolic_traj=False # Important: Specify traj is numeric
    # )
    # # --- End SVSDF grasped object setup ---

    # # # Original SVSDF without grasp
    # # svsdf_calculator = SVSDF(urdf_path, rb, traj, [obstacle_center], robot_pose, symbolic_traj=False)

    # # --- Calculate SVSDF for the obstacle ---
    # print(f"Calculating SVSDF for obstacle at {obstacle_center}...")
    # best_t, min_svsdf, collision_times = svsdf_calculator(obstacle_center, max_time)
    # print(f"SVSDF Result: Min SDF={min_svsdf:.4f} occurs at t={best_t:.4f}")
    # if collision_times:
    #     print(f"  Collision points found (t, sdf): {[(f'{t:.3f}', f'{s:.3f}') for t, s in collision_times]}")
    # else:
    #     print("  No collision points found below threshold.")

    # # --- Visualization Setup ---
    # time_slider = p.addUserDebugParameter("replay_time", 0, max_time, 0)
    # obs_x_slider = p.addUserDebugParameter("obs_x", -2, 2, obstacle_center[0])
    # obs_y_slider = p.addUserDebugParameter("obs_y", -2, 2, obstacle_center[1])
    # obs_z_slider = p.addUserDebugParameter("obs_z", -2, 2, obstacle_center[2])

    # # Visualize the single obstacle
    # obstacle_vis_id = pp.create_sphere(0.03, color=pp.BLACK)
    # pp.set_point(obstacle_vis_id, obstacle_center)

    # # Sphere to show the point on the robot closest to the obstacle at min SDF time
    # closest_robot_pt_vis_id = pp.create_sphere(0.03, color=pp.RED)
    # # Sphere to show the point on the robot at the current slider time
    # current_robot_pt_vis_id = pp.create_sphere(0.03, color=pp.BLUE)

    # print("Starting SVSDF visualization loop...")
    # while True:
    #     if not pp.is_connected(): break

    #     # --- Update obstacle position from sliders ---
    #     obs_x = p.readUserDebugParameter(obs_x_slider)
    #     obs_y = p.readUserDebugParameter(obs_y_slider)
    #     obs_z = p.readUserDebugParameter(obs_z_slider)
    #     current_obstacle_center = np.array([obs_x, obs_y, obs_z])
    #     pp.set_point(obstacle_vis_id, current_obstacle_center)

    #     # --- Recompute SVSDF if obstacle moved significantly ---
    #     # (Add logic here if needed, e.g., check distance moved > threshold)
    #     # For simplicity, we are not recomputing SVSDF dynamically in this loop.
    #     # We visualize based on the initial SVSDF calculation.

    #     # --- Update robot pose based on time slider ---
    #     current_time = p.readUserDebugParameter(time_slider)
    #     current_q = svsdf_calculator.EvaluateJointPosition(current_time, traj)
    #     rb.set_joint_positions(rb.arm_joints, current_q)

    #     # --- Visualize SDF at current time ---
    #     # Use the SDF calculator directly to get current SDF and visualize closest point
    #     # This reuses the SDF instance within SVSDF
    #     current_sdf_val = svsdf_calculator.sdf_solver(robot_pose, current_q, current_obstacle_center, visualize=True)

    #     # # --- Optionally, visualize the closest point found by SVSDF ---
    #     # q_at_best_t = svsdf_calculator.EvaluateJointPosition(best_t, traj)
    #     # # We need a way to get the *robot point* corresponding to min SDF, not just the value.
    #     # # This requires modification to SDF or SVSDF to return the closest point coordinates.
    #     # # Placeholder visualization: move robot to best_t pose
    #     # # rb.set_joint_positions(rb.arm_joints, q_at_best_t) # Uncomment to see pose at min SDF time

    #     print(f"Current t={current_time:.2f}, Current SDF={current_sdf_val:.4f} | Min SVSDF={min_svsdf:.4f} at t={best_t:.4f}", end='\r')

    #     pp.step_simulation()
    #     time.sleep(1.0 / 240.0)

    # print("\nSVSDF visualization loop finished.")
    # pp.disconnect()
