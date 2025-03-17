import os
import sys
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import casadi as ca
import numpy as np

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from motion_planner.svsdf import (
    SVSDF,
    Arr2NodeTraj,
    NodeTraj2Arr,
    NodeTraj2Traj,
    NodeTrajectory,
    Traj2NodeTraj,
    Trajectory,
    generate_trajectory,
)
from robot.robot_setup import RobotSetup
from utils.utils import HideOutput, TermPrint
from utils.utils_casadi import eval


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
        sdf_threshold: float = 0.05,
    ) -> None:
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
        self.num_vars = self.num_joints * (self.num_segments - 1) * (3 + 1)

        self.symbolic_traj = True

        self.max_vel = np.pi / 2  # 最大关节速度 (rad/s)
        self.max_accel = np.pi  # 最大关节加速度 (rad/s²)
        self.x_target = x_target  # 目标点坐标
        self.q_init = q_init
        self.dq_init = np.zeros(self.num_joints)
        self.ddq_init = np.zeros(self.num_joints)

        self.q_target = q_final
        self.dq_target = np.zeros(self.num_joints)
        self.ddq_target = np.zeros(self.num_joints)

        self.p = np.zeros(3)

        self.sdf_threshold = sdf_threshold

        # -------------------- 优化变量定义 --------------------#
        self.opti = ca.Opti()
        self.X = self.opti.variable(self.num_vars)

        # -------------------- 轨迹提取 --------------------#
        self.node_traj = Arr2NodeTraj(self.X, self.num_joints, self.num_segments)
        self.traj = NodeTraj2Traj(
            self.node_traj,
            self.num_joints,
            self.q_init,
            self.dq_init,
            self.ddq_init,
            self.q_target,
            self.dq_target,
            self.ddq_target,
            self.max_total_time,
            self.symbolic_traj,
        )

        # -------------------- 创建SVSDF实例 --------------------#
        self.svsdf = SVSDF(urdf_path, robot, self.traj, self.X, symbolic_traj=self.symbolic_traj, node_traj=True)

        # -------------------- 添加系统约束 --------------------#
        self._AddConstraints()

        # -------------------- 预编译目标函数 --------------------#
        self.obj = self._BuildObjectiveFunction()

        # -------------------- 构建sensitivity --------------------#
        self.sensitivity = self._BuildSensitivityFunction()

    def _BuildObjectiveFunction(self) -> ca.MX:
        """
        构建安全距离目标函数。
        """

        # -------------------- SDF --------------------#
        sdf_sym = self.svsdf.sdf_sym
        G_s = ca.if_else(sdf_sym > self.sdf_threshold, 0, self.sdf_threshold - sdf_sym)
        cost_sdf = ca.if_else(G_s > 0, G_s, 0)
        # cost_sdf = -sdf_sym

        cost = cost_sdf

        return cost

    def _BuildSensitivityFunction(self) -> ca.MX:
        """
        构建敏感函数。
        """
        sdf_sym = self.svsdf.sdf_sym

        Ou = -sdf_sym
        var_p = self.X

        Ol = sdf_sym
        var_q = self.svsdf.t_sym

        g = ca.gradient(Ol, var_q) @ Ol
        sensitivity = -ca.inv(ca.gradient(g, var_q)) @ ca.gradient(g, var_p)

        return sensitivity

    def _AddConstraints(self) -> None:
        """
        添加动力学约束、首尾约束。
        """
        self.constraints = []

        # -------------------- 遍历所有关节 -------------------- #
        for joint_idx in range(self.num_joints):
            joint_traj = self.node_traj[joint_idx]

            # -------------------- 1. 时间片约束 --------------------#
            start_t = 0
            for node_idx in range(len(joint_traj)):
                delta_t = joint_traj[node_idx][3] - start_t
                self.constraints.append((f"delta time of j{joint_idx}_seg{node_idx}", delta_t >= 0))
                start_t += delta_t

            # -------------------- 2. 末尾时间约束 --------------------#
            self.constraints.append((f"time UB of j{joint_idx}", start_t <= self.max_total_time))

            # -------------------- 5. 速度与加速度约束 -------------------- #
            for node_idx in range(len(joint_traj)):
                q, dq, ddq, t = joint_traj[node_idx]

                # -------------------- 速度约束 -------------------- #
                # # 构建速度函数: dq/dt = c1 + 2c2*t + 3c3*t² + 4c4*t³ + 5c5*t⁴
                # vel_expr = (
                #     coeffs[1]
                #     + 2 * coeffs[2] * delta_t
                #     + 3 * coeffs[3] * delta_t**2
                #     + 4 * coeffs[4] * delta_t**3
                #     + 5 * coeffs[5] * delta_t**4
                # )
                # self.constraints.append((f"vel UB of j{joint_idx}_seg{seg_idx}", vel_expr <= self.max_vel))
                # self.constraints.append((f"vel LB of j{joint_idx}_seg{seg_idx}", vel_expr >= -self.max_vel))

                self.constraints.append((f"vel LB of j{joint_idx}_seg{node_idx}", dq >= -self.max_vel))
                self.constraints.append((f"vel UB of j{joint_idx}_seg{node_idx}", dq <= self.max_vel))

                # -------------------- 加速度约束 -------------------- #
                # # 构建加速度函数: d²q/dt² = 2c2 + 6c3*t + 12c4*t² + 20c5*t³
                # accel_expr = (
                #     2 * coeffs[2] + 6 * coeffs[3] * delta_t + 12 * coeffs[4] * delta_t**2 + 20 * coeffs[5] * delta_t**3
                # )

                # self.constraints.append((f"acc UB of j{joint_idx}_seg{seg_idx}", accel_expr <= self.max_accel))
                # self.constraints.append((f"acc LB of j{joint_idx}_seg{seg_idx}", accel_expr >= -self.max_accel))

                self.constraints.append((f"acc LB of j{joint_idx}_seg{node_idx}", ddq >= -self.max_accel))
                self.constraints.append((f"acc UB of j{joint_idx}_seg{node_idx}", ddq <= self.max_accel))

    def EvaluateSDF(self, t: float, node_traj: NodeTrajectory) -> float:
        return eval(
            "",
            self.svsdf.sdf_sym,
            [self.svsdf.t_sym, self.svsdf.p_sym, self.svsdf.x_sym, self.X],
            [t, self.p, self.x_target, NodeTraj2Arr(node_traj, self.num_joints)],
        ).item()

    def optimize(self, traj_init: Trajectory) -> Dict:
        """高斯牛顿法实现带约束的轨迹优化"""

        # 初始化变量和参数
        node_traj_init = Traj2NodeTraj(traj_init, self.num_joints)

        # 预处理约束条件
        processed_constraints = []
        for name, constr in self.constraints:
            if constr.is_op(ca.OP_LE):
                expr = constr.dep(0) - constr.dep(1)  # g(X) <= 0
                processed_constraints.append(("ineq", name, expr))
            elif constr.is_op(ca.OP_EQ):
                expr = constr.dep(0) - constr.dep(1)  # h(X) = 0
                processed_constraints.append(("eq", name, expr))

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

        # 预处理目标函数: (t, X = (q, dq, ddq, T)) -> obj
        obj = eval("", self.obj, [self.svsdf.x_sym, self.svsdf.p_sym], [self.x_target, self.p], full=False)
        fn_obj = ca.Function("fn_obj", [self.svsdf.t_sym, self.X], [obj])

        # 预处理敏感函数: (t, X = (q, dq, ddq, T)) -> sensitivity
        sensitivity = eval(
            "", self.sensitivity, [self.svsdf.x_sym, self.svsdf.p_sym], [self.x_target, self.p], full=False
        )
        fn_sensitivity = ca.Function("fn_sensitivity", [self.svsdf.t_sym, self.X], [sensitivity])

        # 预处理梯度信息: (t, X = (q, dq, ddq, T)) -> grad
        grad_obj_X = ca.gradient(obj, self.X) + ca.gradient(obj, self.svsdf.t_sym) @ sensitivity
        # grad_obj_X = ca.gradient(obj, self.X)
        fn_grad_obj_X = ca.Function("fn_grad_obj_X", [self.svsdf.t_sym, self.X], [grad_obj_X])

        # 预处理惩罚项和梯度
        w_ineq = 1e3  # 不等式约束惩罚权重
        v_eq = 1e3  # 等式约束惩罚权重
        total_penalty = 0
        grad_total_penalty = ca.MX.zeros(self.X.size1(), self.X.size2())

        for constr_type, name, expr in processed_constraints:
            if constr_type == "ineq":
                penalty = w_ineq * ca.fmax(0, expr) ** 2  # max(0, g(X))^2
                total_penalty += penalty
                grad_total_penalty += ca.gradient(penalty, self.X)
            elif constr_type == "eq":
                penalty = v_eq * expr**2  # h(X)^2
                total_penalty += penalty
                grad_total_penalty += ca.gradient(penalty, self.X)

        fn_total_penalty = ca.Function("fn_total_penalty", [self.X], [total_penalty])
        fn_grad_total_penalty = ca.Function("fn_grad_total_penalty", [self.X], [grad_total_penalty])

        # 预先计算t*
        t_star, svsdf_star, collision_times = self.svsdf(
            p=self.p, x=self.x_target, t_max=self.max_total_time, traj=node_traj_init
        )

        if svsdf_star > self.sdf_threshold:
            TermPrint.print(f"SVSDF > {self.sdf_threshold}, exit!!!")
            return {
                "obj": svsdf_star,
                "traj": NodeTraj2Traj(
                    node_traj_init,
                    self.num_joints,
                    self.q_init,
                    self.dq_init,
                    self.ddq_init,
                    self.q_target,
                    self.dq_target,
                    self.ddq_target,
                    self.max_total_time,
                    is_symbolic=False,
                ),
            }

        # -------------------- 外层循环 --------------------#

        # 参数初始化
        iter_count = 0
        max_iter = 5000
        lr = 0.75
        momentum = 0.9
        velocity = 0

        t_star_idx = np.argmin(np.array([pair[0] for pair in collision_times]))
        t_star_curr, svsdf_star_curr = collision_times[t_star_idx]
        collision_times.remove(collision_times[t_star_idx])

        node_traj_var_curr = NodeTraj2Arr(node_traj_init, self.num_joints).reshape((-1, 1))
        grad_obj_X_prev = None

        while iter_count < max_iter:
            iter_count += 1

            print("\n=======================================")
            print(f"iter: {iter_count}")
            print("=======================================\n")

            # 如果当前的svsdf大于阈值，需要进行判断
            if svsdf_star_curr > self.sdf_threshold:

                if len(collision_times) != 0:  # 如果还有潜在的碰撞，优先解决这些
                    TermPrint.print(f"There are still collisions: {collision_times}", "red")
                    t_star_idx = np.argmin(np.array([pair[0] for pair in collision_times]))
                    t_star_curr, svsdf_star_curr = collision_times[t_star_idx]
                    collision_times.remove(collision_times[t_star_idx])
                else:  # double check
                    TermPrint.print(f"No collisions, need double check!!!", "green")
                    t_star, svsdf_star, collision_times = self.svsdf(
                        p=self.p,
                        x=self.x_target,
                        t_max=self.max_total_time,
                        traj=Arr2NodeTraj(node_traj_var_curr, self.num_joints, self.num_segments),
                    )
                    if svsdf_star > self.sdf_threshold:
                        TermPrint.print(f"SVSDF > {self.sdf_threshold}, optimization finished!!!", "green")
                        return {
                            "obj": svsdf_star_curr,
                            "traj": NodeTraj2Traj(
                                Arr2NodeTraj(node_traj_var_curr, self.num_joints, self.num_segments),
                                self.num_joints,
                                self.q_init,
                                self.dq_init,
                                self.ddq_init,
                                self.q_target,
                                self.dq_target,
                                self.ddq_target,
                                self.max_total_time,
                                is_symbolic=False,
                            ),
                            "node_traj": Arr2NodeTraj(node_traj_var_curr, self.num_joints, self.num_segments),
                        }
                    else:
                        TermPrint.print(f"There are still collisions: {collision_times}", "red")
                        t_star_idx = np.argmin(np.array([pair[0] for pair in collision_times]))
                        t_star_curr, svsdf_star_curr = collision_times[t_star_idx]
                        collision_times.remove(collision_times[t_star_idx])
                        grad_obj_X_prev = None
                        velocity = 0
                        lr = 0.1

            # 构建当前时刻梯度 (t = t*, X = X_curr) -> grad
            grad_obj_X_curr = fn_grad_obj_X(t_star_curr, node_traj_var_curr).toarray()
            # 计算带约束的总梯度
            grad_total_penalty_curr = fn_grad_total_penalty(node_traj_var_curr).toarray()
            grad_obj_total = grad_obj_X_curr + grad_total_penalty_curr  # 总梯度

            # 动量加速
            # velocity = momentum * velocity + (1 - momentum) * grad_obj_X_curr
            # delta_X = -lr * velocity
            delta_X = -lr * grad_obj_total

            # 自适应步长
            if grad_obj_X_prev is not None and (np.sign(grad_obj_X_curr) != np.sign(grad_obj_X_prev)).any():
                lr *= 0.75

            node_traj_var_next = node_traj_var_curr + delta_X

            # print(
            #     "delta_X:\n",
            #     "\n".join(
            #         f"    joint: {joint_idx}: \n    {item}"
            #         for joint_idx, item in enumerate(Arr2NodeTraj(delta_X, self.num_joints, self.num_segments))
            #     ),
            # )
            # print(
            #     "new traj:\n",
            #     "\n".join(
            #         f"    joint: {joint_idx}: \n    {item}"
            #         for joint_idx, item in enumerate(
            #             Arr2NodeTraj(node_traj_var_new, self.num_joints, self.num_segments)
            #         )
            #     ),
            # )

            # 使用sensitivity更新t*
            sensitivity_curr = fn_sensitivity(t_star_curr, node_traj_var_curr).toarray()
            sensitivity_next = fn_sensitivity(t_star_curr, node_traj_var_next).toarray()
            t_star_next_sen = (
                t_star_curr + (delta_X.reshape((1, -1)) @ (0.5 * (sensitivity_curr + sensitivity_next))).item()
            )
            svsdf_star_next_sen = eval(
                "",
                self.svsdf.sdf_sym,
                [self.svsdf.x_sym, self.svsdf.p_sym, self.svsdf.t_sym, self.X],
                [self.x_target, self.p, t_star_next_sen, node_traj_var_next],
            ).item()

            if iter_count % 1 == 0:
                # 使用梯度下降更新准确的t*
                t_star_next, svsdf_star_next, _ = self.svsdf(
                    p=self.p,
                    x=self.x_target,
                    t_max=self.max_total_time,
                    t_seed=t_star_curr,
                    traj=Arr2NodeTraj(node_traj_var_next, self.num_joints, self.num_segments),
                )
                print(f"new t* sensitivity: {t_star_next_sen}, new t*: {t_star_next}")
                print(f"new svsdf sensitivity: {svsdf_star_next_sen}, new svsdf: {svsdf_star_next}")
                node_traj_var_curr = node_traj_var_next
                grad_obj_X_prev = grad_obj_X_curr
                t_star_curr = t_star_next
                svsdf_star_curr = svsdf_star_next
            else:
                print(f"new t* sensitivity: {t_star_next_sen}")
                print(f"new svsdf sensitivity: {svsdf_star_next_sen}")
                node_traj_var_curr = node_traj_var_next
                grad_obj_X_prev = grad_obj_X_curr
                t_star_curr = t_star_next_sen
                svsdf_star_curr = svsdf_star_next_sen

        return {
            "obj": svsdf_star_curr,
            "traj": NodeTraj2Traj(
                Arr2NodeTraj(node_traj_var_curr, self.num_joints, self.num_segments),
                self.num_joints,
                self.q_init,
                self.dq_init,
                self.ddq_init,
                self.q_target,
                self.dq_target,
                self.ddq_target,
                self.max_total_time,
                is_symbolic=False,
            ),
            "node_traj": Arr2NodeTraj(node_traj_var_curr, self.num_joints, self.num_segments),
        }


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
        max_total_time=3.0,
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
    trajectory = generate_trajectory(start_pos, end_pos, v_max, n_segments=5)

    # 求解
    solution = optimizer.optimize(trajectory)

    pp.wait_for_user("Optimization finished!")

    max_time = max([temp[-1][1] for temp in trajectory])
    times = np.linspace(0, max_time, 1000)

    slider = p.addUserDebugParameter("replay", 0, 1, 0)

    max_shadow_time = max([temp[-1][1] for temp in solution["traj"]])
    shadow_times = np.linspace(0, max_shadow_time, 1000)
    slider_shadow = p.addUserDebugParameter("replay_shadow", 0, 1, 0)

    sphere_id = pp.create_sphere(0.05, color=pp.RED)
    pp.set_point(sphere_id, x_target.tolist())

    while True:
        slider_value = p.readUserDebugParameter(slider)
        time_idx = int(slider_value * (times.shape[0] - 1))
        t = times[time_idx]
        pos = optimizer.svsdf.EvaluateJointPosition(t, trajectory)
        rb.set_joint_positions(rb.arm_joints, pos)

        slider_shadow_value = p.readUserDebugParameter(slider_shadow)
        time_shadow_idx = int(slider_shadow_value * (shadow_times.shape[0] - 1))
        t_shadow = shadow_times[time_shadow_idx]
        pos_shadow = optimizer.svsdf.EvaluateJointPosition(t_shadow, solution["traj"])
        rb_shadow.set_joint_positions(rb_shadow.arm_joints, pos_shadow)

        # print(
        #     f"raw time: {t}, new time: {t_shadow}, new SDF value: {optimizer.EvaluateSDF(t_shadow, solution['node_traj'])}"
        # )

        time.sleep(1.0 / 60)
