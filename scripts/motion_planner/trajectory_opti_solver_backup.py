import os
import sys
from typing import Dict, List, Tuple, Union

import casadi as ca
import numpy as np

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from motion_planner.svsdf import SVSDF, Arr2Traj, Traj2Arr, Trajectory, eval, generate_trajectory
from robot.robot_setup import RobotSetup
from utils.utils import HideOutput


class TrajectoryOptimizer:
    """
    符号化轨迹参数优化类

    功能:
        - 建立轨迹参数与SVSDF的数学关系
        - 构建包含安全距离约束的优化问题
        - 实现符号化参数优化流程
    """

    def __init__(
        self,
        urdf_path: str,
        robot: RobotSetup,
        num_segments: int,
        num_joints: int,
        max_total_time: float,
        x_target: np.ndarray,
        q_init: np.ndarray,
        q_final: np.ndarray,
    ):
        """
        符号化轨迹参数初始化

        Params:
            urdf_path (str): urdf路径
            robot (RobotSetup): 机器人设置
            num_segments (int): 每个关节的轨迹片段数
            num_joints (int): 机器人关节数目
            max_total_time (int): 轨迹允许的最大总时间
            x_target (np.ndaaray): 3D空间中目标点的坐标
        """
        self.urdf_path = urdf_path
        self.robot = robot
        self.num_segments = num_segments
        self.num_joints = num_joints
        self.max_total_time = max_total_time
        self.num_vars = self.num_joints * self.num_segments * (1 + 6)

        self.max_vel = np.pi / 2  # 最大关节速度 (rad/s)
        self.max_accel = np.pi  # 最大关节加速度 (rad/s²)
        self.x_target = x_target  # 目标点坐标
        self.q_init = q_init
        self.q_final = q_final

        # -------------------- 优化器 --------------------#
        self.opti = ca.Opti()
        self.X = self.opti.variable(self.num_vars)

        # -------------------- 轨迹提取 --------------------#
        self.symbolic_traj = Arr2Traj(self.X, self.num_joints, self.num_segments)

        # -------------------- 创建SVSDF实例 --------------------#
        self.svsdf = SVSDF(urdf_path, robot, self.symbolic_traj, self.X, symbolic_traj=True)
        self._AddConstraints()

    def _BuildObjectiveFunction(self) -> ca.MX:
        """
        构建安全距离目标函数。
        """
        # 获取目标点处的符号化SDF
        # _, sdf_sym = self.svsdf(p=np.array([0, 0, 0]), x=self.x_target, t_max=self.max_total_time, symbolic_output=True)
        sdf_sym = self.svsdf.sdf_sym

        return sdf_sym

        # # 添加时间惩罚项（最小化总时间）
        # total_time = self.symbolic_traj[0][-1][1]  # 取第一个关节的总时间
        # time_penalty = 0.01 * total_time

        # return sdf_sym - time_penalty

    def _AddConstraints(self) -> None:
        """
        添加动力学约束、首位约束以及连续性约束。
        """
        self.constraints = []

        t = self.svsdf.t_sym

        # -------------------- 遍历所有关节 -------------------- #
        for joint_idx in range(self.num_joints):
            traj = self.symbolic_traj[joint_idx]

            # -------------------- 1. 初始条件约束 -------------------- #
            first_seg = traj[0]
            start_t, end_t, coeffs = first_seg
            delta_t = end_t - start_t

            # 位置约束：q(0) = q_init
            self.constraints.append((f"init pos of j{joint_idx}", coeffs[0] == self.q_init[joint_idx]))
            # 速度约束：dq/dt(0) = 0 → coeffs[1] = 0
            self.constraints.append((f"init vel of j{joint_idx}", coeffs[1] == 0))
            # 加速度约束：d²q/dt²(0) = 0 → 2*coeffs[2] = 0
            self.constraints.append((f"init acc of j{joint_idx}", coeffs[2] == 0))

            # -------------------- 2. 终止条件约束 -------------------- #
            last_seg = traj[-1]
            start_t_last, end_t_last, coeffs_last = last_seg
            delta_t_last = end_t_last - start_t_last

            # q(T) = q_final 的计算
            q_T = (
                coeffs_last[0]
                + coeffs_last[1] * delta_t_last
                + coeffs_last[2] * (delta_t_last**2)
                + coeffs_last[3] * (delta_t_last**3)
                + coeffs_last[4] * (delta_t_last**4)
                + coeffs_last[5] * (delta_t_last**5)
            )
            self.constraints.append((f"end pos of j{joint_idx}", q_T == self.q_final[joint_idx]))

            # dq/dt(T) = 0 的计算
            dq_T = (
                coeffs_last[1]
                + 2 * coeffs_last[2] * delta_t_last
                + 3 * coeffs_last[3] * (delta_t_last**2)
                + 4 * coeffs_last[4] * (delta_t_last**3)
                + 5 * coeffs_last[5] * (delta_t_last**4)
            )
            self.constraints.append((f"end vel of j{joint_idx}", dq_T == 0))

            # d²q/dt²(T) = 0 的计算
            ddq_T = (
                2 * coeffs_last[2]
                + 6 * coeffs_last[3] * delta_t_last
                + 12 * coeffs_last[4] * (delta_t_last**2)
                + 20 * coeffs_last[5] * (delta_t_last**3)
            )
            self.constraints.append((f"end acc of j{joint_idx}", ddq_T == 0))
            self.constraints.append((f"end time of j{joint_idx}", end_t_last <= self.max_total_time))

            # -------------------- 3. 时间片约束 --------------------#
            for seg_idx in range(len(traj)):
                delta_t = traj[seg_idx][1] - traj[seg_idx][0]
                self.constraints.append((f"delta time of j{joint_idx}_seg{seg_idx}", delta_t >= 0))

            # -------------------- 4. 连续性约束 -------------------- #
            for seg_idx in range(len(traj) - 1):
                curr_seg = traj[seg_idx]
                next_seg = traj[seg_idx + 1]

                # 当前片段的结束时刻参数
                c_end = curr_seg[2]
                delta_t_curr = curr_seg[1] - curr_seg[0]

                # 下一个片段的开始时刻参数
                c_start = next_seg[2]

                # 位置连续：q_i(end) = q_{i+1}(start)
                q_end_curr = (
                    c_end[0]
                    + c_end[1] * delta_t_curr
                    + c_end[2] * (delta_t_curr**2)
                    + c_end[3] * (delta_t_curr**3)
                    + c_end[4] * (delta_t_curr**4)
                    + c_end[5] * (delta_t_curr**5)
                )
                self.constraints.append(
                    (f"pos continue of j{joint_idx}_seg{seg_idx}_{seg_idx+1}", q_end_curr == c_start[0])
                )

                # 速度连续：dq_i/dt(end) = dq_{i+1}/dt(start)
                dq_end_curr = (
                    c_end[1]
                    + 2 * c_end[2] * delta_t_curr
                    + 3 * c_end[3] * (delta_t_curr**2)
                    + 4 * c_end[4] * (delta_t_curr**3)
                    + 5 * c_end[5] * (delta_t_curr**4)
                )
                self.constraints.append(
                    (f"vel continue of j{joint_idx}_seg{seg_idx}_{seg_idx+1}", dq_end_curr == c_start[1])
                )

                # 加速度连续：d²q_i/dt²(end) = d²q_{i+1}/dt²(start)
                ddq_end_curr = (
                    2 * c_end[2]
                    + 6 * c_end[3] * delta_t_curr
                    + 12 * c_end[4] * (delta_t_curr**2)
                    + 20 * c_end[5] * (delta_t_curr**3)
                )
                self.constraints.append(
                    (f"acc continue of j{joint_idx}_seg{seg_idx}_{seg_idx+1}", ddq_end_curr == 2 * c_start[2])
                )

            # -------------------- 5. 速度与加速度约束 -------------------- #
            # for seg_idx in range(self.num_segments):
            #     seg = self.symbolic_traj[joint_idx][seg_idx]
            #     t_start = seg[0]
            #     delta_t = t - t_start
            #     coeffs = seg[2]

            #     # -------------------- 速度约束 -------------------- #
            #     # 构建速度函数: dq/dt = c1 + 2c2*t + 3c3*t² + 4c4*t³ + 5c5*t⁴
            #     vel_expr = (
            #         coeffs[1]
            #         + 2 * coeffs[2] * delta_t
            #         + 3 * coeffs[3] * delta_t**2
            #         + 4 * coeffs[4] * delta_t**3
            #         + 5 * coeffs[5] * delta_t**4
            #     )

            #     self.constraints.append((f"vel UB of j{joint_idx}_seg{seg_idx}", vel_expr <= self.max_vel))
            #     self.constraints.append((f"vel LB of j{joint_idx}_seg{seg_idx}", vel_expr >= -self.max_vel))

            #     # -------------------- 加速度约束 -------------------- #
            #     # 构建加速度函数: d²q/dt² = 2c2 + 6c3*t + 12c4*t² + 20c5*t³
            #     accel_expr = (
            #         2 * coeffs[2] + 6 * coeffs[3] * delta_t + 12 * coeffs[4] * delta_t**2 + 20 * coeffs[5] * delta_t**3
            #     )

            #     self.constraints.append((f"acc UB of j{joint_idx}_seg{seg_idx}", accel_expr <= self.max_accel))
            #     self.constraints.append((f"acc LB of j{joint_idx}_seg{seg_idx}", accel_expr >= -self.max_accel))

        # -------------------- 6. 其他约束 --------------------#
        # self.constraints.append((f"p bound", self.svsdf.p_sym == np.array([0, 0, 0])))
        # self.constraints.append((f"x bound", self.svsdf.x_sym == self.x_target))

    def optimize(self, traj_init: Trajectory) -> Dict:
        """梯度下降法实现带约束的轨迹优化"""

        # 初始化变量和参数
        var_cur = self.svsdf._Traj2Arr(traj_init)
        prev_obj = np.inf
        converged = False
        iter_count = 0
        max_iter = 100
        tol = 1e-4

        # -------------------- 构建优化问题 --------------------
        base_obj = self._BuildObjectiveFunction()

        # 预处理约束条件
        processed_constraints = []
        for name, constr in self.constraints:
            # 分解约束结构
            if constr.is_op(ca.OP_LE):
                expr = constr.dep(0) - constr.dep(1)
                processed_constraints.append(("ineq", name, expr))
            elif constr.is_op(ca.OP_EQ):
                expr = constr.dep(0) - constr.dep(1)
                processed_constraints.append(("eq", name, expr))

        # **************************************************************************
        # 测试代码
        # **************************************************************************

        # 步骤1: 计算t*
        traj_cur = self.svsdf._Arr2Traj(var_cur)
        t_max = max([temp[-1][1] for temp in traj_cur])
        t_star, svsdf_star = self.svsdf(p=np.zeros(3), x=self.x_target, t_max=t_max, traj=traj_cur)

        if svsdf_star > 0:
            print("SVSDF >= 0, exit!!!")
            return {"obj": svsdf_star, "traj": traj_cur}

        # 步骤2: 构建当前时刻目标函数
        svsdf_cur = eval(
            "",
            base_obj,
            [self.svsdf.t_sym, self.svsdf.x_sym, self.svsdf.p_sym],
            [t_star, self.x_target, np.array([0, 0, 0])],
            full=False,
        )
        Jo = ca.if_else(svsdf_cur > 0.05, 0, 0.05 - svsdf_cur)
        mu = 0.05
        obj = ca.if_else(Jo <= 0, 0, ca.if_else(Jo > mu, Jo - mu / 2, (mu - Jo / 2) * (Jo / mu) ** 3))

        # 步骤3: 构建约束条件
        self.opti.subject_to()  # 清空旧约束
        for constr_type, name, expr in processed_constraints:

            constr_expr = eval("", expr, [self.svsdf.t_sym], [t_star], full=False)

            if constr_type == "ineq":
                self.opti.subject_to(constr_expr <= 0)
            elif constr_type == "eq":
                self.opti.subject_to(constr_expr == 0)

        # 配置并求解优化问题
        self.opti.minimize(obj)

        p_opts = {"print_time": 0}
        s_opts = {"max_iter": 1000, "print_level": 0}
        s_opts = {"max_iter": 10000}

        self.opti.solver("ipopt", p_opts, s_opts)
        self.opti.set_initial(self.X, var_cur)

        try:
            sol = self.opti.solve()
            var_new = sol.value(self.X)
            current_obj_value = sol.value(obj)
        except RuntimeError as e:
            # return {"success": False, "message": f"求解失败: {str(e)}"}
            var_new = self.opti.debug.value(self.X)
            current_obj_value = self.opti.debug.value(obj)

        return {"obj": current_obj_value, "traj": self.svsdf._Arr2Traj(var_new)}

        # # -------------------- 主优化循环 --------------------#
        # while not converged and iter_count < max_iter:

        #     print("=======================================")
        #     print(f"iter: {iter_count}")
        #     print("=======================================")

        #     # 步骤1: 计算当前最优时间
        #     traj_cur = self.svsdf._Arr2Traj(var_cur)
        #     t_max = max([temp[-1][1] for temp in traj_cur])
        #     opti_t, svsdf_val_cur = self.svsdf(p=np.zeros(3), x=self.x_target, t_max=t_max, traj=traj_cur)

        #     if svsdf_val_cur > 0:
        #         print("SVSDF >= 0, exit!!!")
        #         return {"obj": svsdf_val_cur, "traj": traj_cur}

        #     # 步骤2: 构建当前时刻目标函数
        #     current_svsdf = eval(
        #         "",
        #         base_obj,
        #         [self.svsdf.t_sym, self.svsdf.x_sym, self.svsdf.p_sym],
        #         [opti_t, self.x_target, np.array([0, 0, 0])],
        #         full=False,
        #     )
        #     current_obj = ca.if_else(current_svsdf > 0.1, 0, 0.1 - current_svsdf)

        #     # 步骤3: 构建约束条件
        #     self.opti.subject_to()  # 清空旧约束
        #     for constr_type, name, expr in processed_constraints:

        #         constr_expr = eval("", expr, [self.svsdf.t_sym], [opti_t], full=False)

        #         if constr_type == "ineq":
        #             self.opti.subject_to(constr_expr <= 0)
        #         elif constr_type == "eq":
        #             self.opti.subject_to(constr_expr == 0)

        #     # 配置并求解优化问题
        #     self.opti.minimize(current_obj)

        #     p_opts = {"expand": True, "print_time": 0}
        #     s_opts = {"max_iter": 1000, "print_level": 0}
        #     s_opts = {"max_iter": 1000}

        #     self.opti.solver("ipopt", p_opts, s_opts)
        #     self.opti.set_initial(self.X, var_cur)

        #     try:
        #         sol = self.opti.solve()
        #         var_new = sol.value(self.X)
        #         current_obj_value = sol.value(current_obj)
        #     except RuntimeError as e:
        #         # return {"success": False, "message": f"求解失败: {str(e)}"}
        #         var_new = self.opti.debug.value(self.X)
        #         current_obj_value = self.opti.debug.value(current_obj)

        #     # 步骤4: 收敛判断
        #     var_diff = np.linalg.norm(var_new - var_cur)
        #     obj_diff = abs(current_obj_value - prev_obj)

        #     if var_diff < tol and obj_diff < tol:
        #         converged = True
        #     else:
        #         var_cur = var_new
        #         prev_obj = current_obj_value
        #         iter_count += 1


if __name__ == "__main__":
    import time

    import pybullet as p
    import pybullet_planning as pp
    from utils.collision import Element, create_couplers, init_pb

    np.set_printoptions(precision=8, suppress=True)

    urdf_path = "/home/jeong/summer_research/eth_ws/src/husky_assembly/data/husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf"

    init_pb()

    rb = RobotSetup("rb")
    rb_shadow = RobotSetup("rb_shadow")
    pp.set_color(rb_shadow.robot, (0, 0, 1, 0.5))

    x_target = np.array([0.5, -0.35, 0.75])

    optimizer = TrajectoryOptimizer(
        urdf_path,
        rb,
        num_segments=5,
        num_joints=6,
        max_total_time=10.0,
        x_target=np.array([0.5, -0.35, 0.75]),
        q_init=np.array([0, 0, 0, 0, 0, 0]),
        q_final=np.array([0, -np.pi / 2, -np.pi / 2, 0, 0, 0]),
    )

    pp.wait_for_duration(1)

    # 定义轨迹参数
    start_pos = np.array([0, 0, 0, 0, 0, 0])
    end_pos = np.array([0, -np.pi / 2, -np.pi / 2, 0, 0, 0])
    v_max = np.pi / 6

    # 生成轨迹
    trajectory = generate_trajectory(start_pos, end_pos, v_max)

    # 求解
    solution = optimizer.optimize(trajectory)

    max_time = max([temp[-1][1] for temp in trajectory])
    max_shadow_time = max([temp[-1][1] for temp in solution["traj"]])
    times = np.linspace(0, max_time, 1000)
    shadow_times = np.linspace(0, max_shadow_time, 1000)

    slider = p.addUserDebugParameter("replay", 0, 1, 0)
    slider_shadow = p.addUserDebugParameter("replay_shadow", 0, 1, 0)
    sphere_id = pp.create_sphere(0.05, color=pp.BLACK)
    pp.set_point(sphere_id, x_target.tolist())

    while True:
        slider_value = p.readUserDebugParameter(slider)
        slider_shadow_value = p.readUserDebugParameter(slider_shadow)
        time_idx = int(slider_value * (times.shape[0] - 1))
        time_shadow_idx = int(slider_shadow_value * (shadow_times.shape[0] - 1))
        t = times[time_idx]
        t_shadow = shadow_times[time_shadow_idx]
        pos = optimizer.svsdf.EvaluateJointPosition(t, trajectory)
        rb.set_joint_positions(rb.arm_joints, pos)
        pos_shadow = optimizer.svsdf.EvaluateJointPosition(t_shadow, solution["traj"])
        rb_shadow.set_joint_positions(rb_shadow.arm_joints, pos_shadow)
        time.sleep(1.0 / 60)
