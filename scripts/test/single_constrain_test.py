#!/usr/bin/env python3

import os
import sys
import numpy as np
import pybullet as p
import pybullet_planning as pp
from compas_fab.robots import RobotSemantics
from compas_fab.robots.robot import RobotModel
from tracikpy import TracIKSolver
import time  # 导入时间模块用于生成文件名和可视化控制
import argparse
import math
from functools import partial

# Import OMPL libraries
try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys

    sys.path.insert(0, join(dirname(dirname(dirname(abspath(__file__)))), "py-bindings"))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from multi_tangent.collision import create_collision_bodies
from utils.collision import init_pb
from utils.params import *
from robot.robot_setup import RobotSetup, HUSKY_URDF_PATH, HUSKY_JOINT_NAMES, HUSKY_CONTROL_JOINT_NAMES, HUSKY_TOOL0_NAME
from ConstrainedPlanningCommon import *
from utils.util import interpolate


def normalize(x):
    """Normalize a vector"""
    norm = np.linalg.norm(x)
    if norm > 0 and np.isfinite(norm):
        return x / norm
    else:
        return x


class LinearEndEffectorConstraint(ob.Constraint):
    """
    约束机械臂末端效应器在一条直线上移动的约束类
    """

    def __init__(self, num_joints, line_point, line_direction, robot_setup):
        """
        初始化线性约束

        Parameters:
        -----------
        num_joints : int
            机械臂关节数量
        line_point : np.array
            直线上的一个点 [x, y, z]
        line_direction : np.array
            直线的方向向量 [dx, dy, dz]
        robot_setup : RobotSetup
            机器人设置对象，用于访问PyBullet中的机器人模型
        """
        # 约束维度：机械臂有num_joints个关节，末端效应器约束在直线上需要2个约束
        # (因为直线是1维的，而末端效应器在3D空间中，所以需要2个约束来限制其他2个自由度)
        super(LinearEndEffectorConstraint, self).__init__(num_joints, 2)

        self.num_joints = num_joints
        self.line_point = np.array(line_point)
        self.line_direction = normalize(np.array(line_direction))
        self.robot_setup = robot_setup
        self.robot_id = robot_setup.robot

        # 获取末端执行器链接ID
        self.end_effector_link = pp.link_from_name(self.robot_id, HUSKY_TOOL0_NAME)

        # 获取可控制的关节索引
        self.control_joints = [pp.joint_from_name(self.robot_id, joint_name) for joint_name in HUSKY_JOINT_NAMES]
        
        # 默认移动距离，可以通过外部设置修改
        self.move_distance = 0.2  # 默认20cm
        
        # 计算与直线方向垂直的两个向量，用于构造约束
        # 使用Gram-Schmidt过程构造正交向量
        if abs(self.line_direction[0]) < 0.9:
            temp_vec = np.array([1, 0, 0])
        else:
            temp_vec = np.array([0, 1, 0])

        # 计算第一个垂直向量
        self.perp_vec1 = normalize(temp_vec - np.dot(temp_vec, self.line_direction) * self.line_direction)
        # 计算第二个垂直向量
        self.perp_vec2 = normalize(np.cross(self.line_direction, self.perp_vec1))

    def function(self, x, out):
        """
        约束函数：定义末端效应器必须满足的约束条件

        Parameters:
        -----------
        x : array
            关节角度配置
        out : array
            约束函数的输出值
        """
        # 使用PyBullet计算真实的前向运动学
        end_effector_pos = self.pybullet_forward_kinematics(x)

        # 计算从直线上一点到末端效应器的向量
        point_to_ee = end_effector_pos - self.line_point

        # 约束条件：末端效应器到直线的距离在两个垂直方向上都为0
        out[0] = np.dot(point_to_ee, self.perp_vec1)
        out[1] = np.dot(point_to_ee, self.perp_vec2)

    def jacobian(self, x, out):
        """
        计算约束的雅可比矩阵

        Parameters:
        -----------
        x : array
            关节角度配置
        out : array
            雅可比矩阵
        """
        out[:, :] = np.zeros((self.getCoDimension(), self.getAmbientDimension()))

        # 使用数值微分计算雅可比矩阵
        epsilon = 1e-6
        base_pos = self.pybullet_forward_kinematics(x)

        for i in range(self.num_joints):
            # 创建扰动后的关节配置
            x_plus = np.array(x)
            x_plus[i] += epsilon

            # 计算扰动后的末端效应器位置
            pos_plus = self.pybullet_forward_kinematics(x_plus)

            # 计算数值雅可比
            pos_derivative = (pos_plus - base_pos) / epsilon

            # 更新约束雅可比矩阵
            out[0, i] = np.dot(pos_derivative, self.perp_vec1)
            out[1, i] = np.dot(pos_derivative, self.perp_vec2)

    def pybullet_forward_kinematics(self, joint_angles):
        """
        使用PyBullet计算前向运动学

        Parameters:
        -----------
        joint_angles : array
            关节角度

        Returns:
        --------
        np.array
            末端效应器位置 [x, y, z]
        """
        # 保存当前关节状态
        current_joints = pp.get_joint_positions(self.robot_id, self.control_joints)

        try:
            # 设置新的关节角度
            pp.set_joint_positions(self.robot_id, self.control_joints, joint_angles)

            # 获取末端效应器位姿
            pose = pp.get_link_pose(self.robot_id, self.end_effector_link)
            position = pose[0]  # 位置部分 (x, y, z)

            return np.array(position)

        finally:
            # 恢复原来的关节状态
            pp.set_joint_positions(self.robot_id, self.control_joints, current_joints)

    def isValid(self, state):
        """
        检查状态是否有效（无碰撞等）

        Parameters:
        -----------
        state : ob.State
            要检查的状态

        Returns:
        --------
        bool
            状态是否有效
        """
        # 提取关节角度
        joint_angles = np.array([state[i] for i in range(self.getAmbientDimension())])

        # 检查关节限制
        for angle in joint_angles:
            if angle < -2 * np.pi or angle > 2 * np.pi:
                return False

        # 保存当前关节状态
        current_joints = pp.get_joint_positions(self.robot_id, self.control_joints)

        try:
            # 设置新的关节角度
            pp.set_joint_positions(self.robot_id, self.control_joints, joint_angles)

            # 检查碰撞
            # 这里可以添加具体的碰撞检测逻辑
            # 例如：检查机器人是否与环境发生碰撞
            # is_collision = pp.pairwise_collision(self.robot_id, other_object_id)
            # if is_collision:
            #     return False

            return True

        finally:
            # 恢复原来的关节状态
            pp.set_joint_positions(self.robot_id, self.control_joints, current_joints)

    def createSpace(self):
        """
        创建状态空间

        Returns:
        --------
        ob.RealVectorStateSpace
            配置空间
        """
        space = ob.RealVectorStateSpace(self.num_joints)
        bounds = ob.RealVectorBounds(self.num_joints)

        # 获取真实的关节限制
        for i in range(self.num_joints):
            joint_info = pp.get_joint_info(self.robot_id, self.control_joints[i])
            lower_limit = joint_info.jointLowerLimit
            upper_limit = joint_info.jointUpperLimit

            # 如果关节是连续的（revolute joint），使用默认限制
            if lower_limit == 0 and upper_limit == -1:
                bounds.setLow(i, -2 * np.pi)
                bounds.setHigh(i, 2 * np.pi)
            else:
                bounds.setLow(i, lower_limit)
                bounds.setHigh(i, upper_limit)

        space.setBounds(bounds)
        return space

    def getStartAndGoalStates(self):
        """
        获取起始和目标状态

        Returns:
        --------
        tuple
            (start_state, goal_state)
        """
        # 使用当前机器人的关节配置作为起始状态
        start = np.array(pp.get_joint_positions(self.robot_id, self.control_joints))

        # 获取当前末端执行器位姿
        current_pose = pp.get_link_pose(self.robot_id, self.end_effector_link)
        current_position = np.array(current_pose[0])
        current_orientation = current_pose[1]  # 保持当前姿态

        # 沿直线方向移动一定距离来创建目标位置
        # 可以调整这个距离来控制目标点的远近
        move_distance = self.move_distance  # 使用可配置的移动距离
        target_position = current_position + move_distance * self.line_direction

        # 构造目标位姿（保持相同的姿态，只改变位置）
        target_pose = (target_position.tolist(), current_orientation)

        # 使用IK求解器计算目标关节配置
        print(f"计算IK解...")
        print(f"当前位置: {current_position}")
        print(f"目标位置: {target_position}")
        print(f"移动方向: {self.line_direction}")
        print(f"移动距离: {move_distance}")

        try:
            # 使用当前关节配置作为IK求解的初始猜测
            target_joints = self.robot_setup.get_relative_ik_solution(target_pose, q_init=start.tolist())

            if target_joints is not None:
                goal = np.array(target_joints)
                print(f"IK求解成功！目标关节配置: {goal}")

                # 验证IK解的有效性
                # 设置目标关节配置并检查末端执行器位置
                current_joints = pp.get_joint_positions(self.robot_id, self.control_joints)
                pp.set_joint_positions(self.robot_id, self.control_joints, goal)
                verification_pose = pp.get_link_pose(self.robot_id, self.end_effector_link)
                verification_position = np.array(verification_pose[0])

                # 恢复原始关节配置
                pp.set_joint_positions(self.robot_id, self.control_joints, current_joints)

                position_error = np.linalg.norm(verification_position - target_position)
                print(f"IK解验证 - 位置误差: {position_error:.6f} m")

                if position_error > 0.01:  # 1cm tolerance
                    print(f"警告：IK解精度较低，位置误差 {position_error:.6f} m")

            else:
                print("IK求解失败，使用备用目标配置")
                # 如果IK求解失败，使用一个备用配置
                goal = np.array([angle + 0.1 for angle in start])

        except Exception as e:
            print(f"IK求解过程中发生错误: {e}")
            print("使用备用目标配置")
            goal = np.array([angle + 0.1 for angle in start])

        return start, goal

    def getProjection(self, space):
        """
        创建投影评估器用于约束规划
        """
        class LinearConstraintProjection(ob.ProjectionEvaluator):
            def __init__(self, space, constraint):
                super(LinearConstraintProjection, self).__init__(space)
                self.constraint = constraint
                self.defaultCellSizes()

            def getDimension(self):
                return 2

            def defaultCellSizes(self):
                self.cellSizes_ = list2vec([.1, .1])

            def project(self, state, projection):
                # 获取关节角度
                state_dim = self.constraint.num_joints
                joint_angles = [state[i] for i in range(state_dim)]
                
                # 计算末端执行器位置
                ee_pos = self.constraint.pybullet_forward_kinematics(joint_angles)
                
                # 投影到约束流形的2D空间
                # 这里我们使用末端执行器在垂直于直线方向上的投影
                point_to_ee = ee_pos - self.constraint.line_point
                projection[0] = np.dot(point_to_ee, self.constraint.perp_vec1)
                projection[1] = np.dot(point_to_ee, self.constraint.perp_vec2)

        return LinearConstraintProjection(space, self)

    def dump(self, outfile):
        """输出约束参数信息"""
        print(f"Linear End-Effector Constraint", file=outfile)
        print(f"Joints: {self.num_joints}", file=outfile)
        print(f"Line Point: {self.line_point}", file=outfile)
        print(f"Line Direction: {self.line_direction}", file=outfile)

    def addBenchmarkParameters(self, bench):
        """添加基准测试参数"""
        bench.addExperimentParameter("constraint_type", "STRING", "LinearEndEffector")
        bench.addExperimentParameter("num_joints", "INTEGER", str(self.num_joints))
        bench.addExperimentParameter("line_direction_x", "REAL", str(self.line_direction[0]))
        bench.addExperimentParameter("line_direction_y", "REAL", str(self.line_direction[1]))
        bench.addExperimentParameter("line_direction_z", "REAL", str(self.line_direction[2]))


def linearConstraintPlanningOnce(cp, planner, output, interpolate_points=50):
    """执行一次约束规划"""
    cp.setPlanner(planner, "linear")
    
    # 解决规划问题
    stat = cp.solveOnce(output, "linear")
    
    if stat:
        # 获取解路径
        path = cp.ss.getSolutionPath()
        
        if path:
            print(f"\n=== 轨迹信息 ===")
            print(f"原始轨迹点数: {path.getStateCount()}")
            print(f"轨迹长度: {path.length():.6f}")
            
            # 将OMPL路径转换为numpy数组
            trajectory = []
            state_dim = cp.css.getDimension()  # 获取状态空间维度
            for i in range(path.getStateCount()):
                state = path.getState(i)
                joint_config = [state[j] for j in range(state_dim)]
                trajectory.append(joint_config)
            
            original_trajectory = np.array(trajectory)
            print(f"轨迹维度: {original_trajectory.shape}")
            
            # 对轨迹进行插值
            if len(original_trajectory) > 1:
                interpolated_trajectory = interpolate(original_trajectory, interpolate_points)
                
                print(f"\n=== 插值后轨迹 ===")
                print(f"插值后轨迹点数: {interpolated_trajectory.shape[0]}")
                print(f"轨迹维度: {interpolated_trajectory.shape}")
                
                # 打印插值轨迹的详细信息
                print(f"\n=== 插值轨迹详细信息 ===")
                print("时间步 | 关节角度 (弧度)")
                print("-" * 80)
                
                for i, config in enumerate(interpolated_trajectory):
                    joint_str = " ".join([f"{angle:8.4f}" for angle in config])
                    print(f"{i:6d} | {joint_str}")
                
                # 验证插值轨迹的约束满足情况
                print(f"\n=== 约束验证 ===")
                constraint = cp.constraint
                constraint_violations = []
                
                for i, config in enumerate(interpolated_trajectory):
                    # 计算约束函数值
                    constraint_values = np.zeros(constraint.getCoDimension())
                    constraint.function(config, constraint_values)
                    
                    # 计算约束违反程度
                    violation = np.linalg.norm(constraint_values)
                    constraint_violations.append(violation)
                    
                    if i % 10 == 0:  # 每10个点打印一次
                        print(f"点 {i:2d}: 约束违反度 = {violation:.6f}, 约束值 = [{constraint_values[0]:8.4f}, {constraint_values[1]:8.4f}]")
                
                max_violation = np.max(constraint_violations)
                avg_violation = np.mean(constraint_violations)
                print(f"最大约束违反度: {max_violation:.6f}")
                print(f"平均约束违反度: {avg_violation:.6f}")
                
                # 验证末端执行器位置是否在直线上
                print(f"\n=== 末端执行器轨迹验证 ===")
                ee_positions = []
                distances_to_line = []
                
                for i in range(0, len(interpolated_trajectory), 5):  # 每5个点验证一次
                    config = interpolated_trajectory[i]
                    ee_pos = constraint.pybullet_forward_kinematics(config)
                    ee_positions.append(ee_pos)
                    
                    # 计算到直线的距离
                    point_to_ee = ee_pos - constraint.line_point
                    distance_to_line = np.linalg.norm(np.cross(point_to_ee, constraint.line_direction))
                    distances_to_line.append(distance_to_line)
                    
                    print(f"点 {i:2d}: 末端位置 = [{ee_pos[0]:7.4f}, {ee_pos[1]:7.4f}, {ee_pos[2]:7.4f}], 到直线距离 = {distance_to_line:.6f}")
                
                max_distance = np.max(distances_to_line)
                avg_distance = np.mean(distances_to_line)
                print(f"最大到直线距离: {max_distance:.6f} m")
                print(f"平均到直线距离: {avg_distance:.6f} m")
                
                # 保存轨迹到文件
                if output:
                    trajectory_filename = "linear_constraint_trajectory.txt"
                    with open(trajectory_filename, 'w') as f:
                        f.write("# Linear Constraint Trajectory\n")
                        f.write(f"# Original points: {len(original_trajectory)}\n")
                        f.write(f"# Interpolated points: {len(interpolated_trajectory)}\n")
                        f.write(f"# Joints: {interpolated_trajectory.shape[1]}\n")
                        f.write("# Format: joint1 joint2 joint3 joint4 joint5 joint6\n")
                        f.write("#" + "-" * 70 + "\n")
                        
                        for config in interpolated_trajectory:
                            joint_str = " ".join([f"{angle:12.6f}" for angle in config])
                            f.write(f"{joint_str}\n")
                    
                    print(f"\n轨迹已保存到: {trajectory_filename}")
                
                return interpolated_trajectory
            else:
                print("轨迹点数不足，无法进行插值")
                return original_trajectory
    
    if output:
        ou.OMPL_INFORM("Dumping problem information to `linear_constraint_info.txt`.")
        with open("linear_constraint_info.txt", "w") as infofile:
            print(cp.spaceType, file=infofile)
            cp.constraint.dump(infofile)
    
    cp.atlasStats()
    return stat


def linearConstraintPlanningBench(cp, planners):
    """执行基准测试"""
    cp.setupBenchmark(planners, "linear")
    cp.constraint.addBenchmarkParameters(cp.bench)
    cp.runBenchmark()


def linearConstraintPlanning(robot_setup, options):
    """主要的线性约束规划函数"""
    # 创建线性约束
    num_joints = len(HUSKY_JOINT_NAMES)  # 使用实际的关节数量
    
    # 获取当前末端执行器位置作为直线上的一点
    current_pose = pp.get_link_pose(robot_setup.robot, pp.link_from_name(robot_setup.robot, HUSKY_TOOL0_NAME))
    line_point = np.array(current_pose[0])  # 当前位置
    
    # 使用命令行参数中的直线方向，如果没有则默认沿x轴
    if hasattr(options, 'line_direction'):
        line_direction = np.array(options.line_direction)
    else:
        line_direction = np.array([1.0, 0.0, 0.0])  # 默认沿x轴方向
    
    constraint = LinearEndEffectorConstraint(num_joints, line_point, line_direction, robot_setup)
    
    # 如果约束类需要移动距离参数，也传递给它
    if hasattr(options, 'move_distance'):
        constraint.move_distance = options.move_distance
    
    # 获取起始和目标状态（这里会计算IK解）
    start, goal = constraint.getStartAndGoalStates()
    
    # 可视化约束直线
    line_length = 0.5  # 直线长度
    line_start = line_point - line_length * line_direction
    line_end = line_point + line_length * line_direction
    
    # 绘制约束直线
    pp.add_line(line_start, line_end, color=[1, 0, 0], width=3)  # 红色直线
    print(f"绘制约束直线：从 {line_start} 到 {line_end}")
    
    # 可视化起始和目标位置
    # 设置起始配置并可视化
    pp.set_joint_positions(robot_setup.robot, constraint.control_joints, start)
    start_ee_pose = pp.get_link_pose(robot_setup.robot, constraint.end_effector_link)
    pp.draw_pose(start_ee_pose, length=0.05)
    pp.draw_point(start_ee_pose[0], size=0.02, color=[0, 1, 0])  # 绿色起始点
    
    # 设置目标配置并可视化
    pp.set_joint_positions(robot_setup.robot, constraint.control_joints, goal)
    goal_ee_pose = pp.get_link_pose(robot_setup.robot, constraint.end_effector_link)
    pp.draw_pose(goal_ee_pose, length=0.05)
    pp.draw_point(goal_ee_pose[0], size=0.02, color=[0, 0, 1])  # 蓝色目标点
    
    # 恢复起始配置
    pp.set_joint_positions(robot_setup.robot, constraint.control_joints, start)
    
    print(f"起始末端执行器位置: {start_ee_pose[0]}")
    print(f"目标末端执行器位置: {goal_ee_pose[0]}")
    
    # 验证起始和目标点是否在直线上
    start_distance_to_line = np.linalg.norm(np.cross(np.array(start_ee_pose[0]) - line_point, line_direction))
    goal_distance_to_line = np.linalg.norm(np.cross(np.array(goal_ee_pose[0]) - line_point, line_direction))
    
    print(f"起始点到直线距离: {start_distance_to_line:.6f} m")
    print(f"目标点到直线距离: {goal_distance_to_line:.6f} m")
    
    # 创建约束规划问题
    cp = ConstrainedProblem(options.space, constraint.createSpace(), constraint, options)
    
    # 注册投影评估器
    cp.css.registerProjection("linear", constraint.getProjection(cp.css))
    
    # 设置起始和目标状态
    sstart = ob.State(cp.css)
    sgoal = ob.State(cp.css)
    for i in range(cp.css.getDimension()):
        sstart[i] = start[i]
        sgoal[i] = goal[i]
    cp.setStartAndGoalStates(sstart, sgoal)
    
    # 设置状态有效性检查器
    cp.ss.setStateValidityChecker(ob.StateValidityCheckerFn(
        partial(LinearEndEffectorConstraint.isValid, constraint)))
    
    # 执行规划
    planners = options.planner.split(",")
    if not options.bench:
        interpolate_points = getattr(options, 'interpolate_points', 50)
        result = linearConstraintPlanningOnce(cp, planners[0], options.output, interpolate_points)
        return result
    else:
        linearConstraintPlanningBench(cp, planners)
        return None


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", action="store_true",
                        help="Dump found solution path (if one exists) in plain text and planning "
                        "graph in GraphML to `linear_path.txt` and `linear_graph.graphml` "
                        "respectively.")
    parser.add_argument("--bench", action="store_true",
                        help="Do benchmarking on provided planner list.")
    parser.add_argument("--line-direction", nargs=3, type=float, default=[1.0, 0.0, 0.0],
                        help="Direction of the constraint line as three floats (default: 1.0 0.0 0.0)")
    parser.add_argument("--move-distance", type=float, default=0.2,
                        help="Distance to move along the line for target pose (default: 0.2)")
    parser.add_argument("--interpolate-points", type=int, default=50,
                        help="Number of points to interpolate the trajectory to (default: 50)")
    
    addSpaceOption(parser)
    addPlannerOption(parser)
    addConstrainedOptions(parser)
    addAtlasOptions(parser)

    args = parser.parse_args()

    # 初始化pybullet环境
    init_pb()
    robot = RobotSetup("r0")

    print("=== 线性约束规划测试 ===")
    print("约束类型：末端执行器沿指定方向直线运动")
    print(f"直线方向: {args.line_direction}")
    print(f"移动距离: {args.move_distance}")

    # 设置初始机器人姿态
    print("\n1. 设置初始机器人姿态...")
    robot.set_joint_positions(robot.arm_joints, [0, -np.pi / 2 + 0.5, np.pi / 2 - 0.5, 0, 0, 0])

    # 显示当前末端执行器位姿
    current_pose = pp.get_link_pose(robot.robot, pp.link_from_name(robot.robot, HUSKY_TOOL0_NAME))
    print(f"当前末端执行器位置: {current_pose[0]}")
    print(f"当前末端执行器姿态: {current_pose[1]}")

    # 绘制当前位姿
    pp.draw_pose(current_pose, length=0.1)

    print(f"\n2. 规划参数设置:")
    print(f"   - 空间类型: {args.space}")
    print(f"   - 规划器: {args.planner}")
    print(f"   - 规划时间: {args.time} 秒")
    print(f"   - 约束容差: {args.tolerance}")

    print("\n3. 按任意键继续进行约束规划...")
    pp.wait_for_user()

    # 执行线性约束规划
    print("\n4. 开始执行线性约束规划...")
    print("   - 使用PyBullet计算真实的前向运动学")
    print("   - 使用RobotSetup的IK求解器计算目标配置")
    print("   - 约束末端执行器在直线上移动")
    
    result = linearConstraintPlanning(robot, args)

    print("\n5. 规划结果:")
    if result is not None and hasattr(result, '__len__') and len(result) > 0:
        # 如果result是轨迹数据（numpy数组）
        if isinstance(result, np.ndarray):
            print("   ✓ 规划成功！获得插值轨迹")
            print("   - 绿色点：起始位置")
            print("   - 蓝色点：目标位置") 
            print("   - 红色线：约束直线")
            print(f"   - 轨迹点数：{len(result)}")
            
            # 可选：可视化轨迹中的几个关键点
            print("\n   轨迹可视化...")
            key_indices = [0, len(result)//4, len(result)//2, 3*len(result)//4, len(result)-1]
            colors = [[1,0,0], [1,0.5,0], [1,1,0], [0,1,0], [0,0,1]]  # 红橙黄绿蓝
            
            for i, idx in enumerate(key_indices):
                if idx < len(result):
                    config = result[idx]
                    # 设置关节配置
                    pp.set_joint_positions(robot.robot, robot.arm_joints, config)
                    # 获取末端执行器位置
                    ee_pose = pp.get_link_pose(robot.robot, pp.link_from_name(robot.robot, HUSKY_TOOL0_NAME))
                    # 绘制点
                    pp.draw_point(ee_pose[0], size=0.015, color=colors[i])
                    print(f"   轨迹点 {idx}: 末端位置 = [{ee_pose[0][0]:7.4f}, {ee_pose[0][1]:7.4f}, {ee_pose[0][2]:7.4f}]")
                    
            # 交互式轨迹可视化
            print("\n6. 交互式轨迹可视化")
            print("   使用滑动条控制机器人沿轨迹移动")
            
            # 创建滑动条
            trajectory_slider = p.addUserDebugParameter("position", 0, len(result)-1, 0)
            speed_slider = p.addUserDebugParameter("speed", 0.1, 5.0, 1.0)
            auto_play_button = p.addUserDebugParameter("auto", 0, 1, 0)
            
            # 存储上一个轨迹点，用于检测变化
            prev_trajectory_point = -1
            prev_auto_play = 0
            auto_play_index = 0
            auto_play_direction = 1
            last_time = time.time()
            
            # 创建轨迹线可视化
            trajectory_line_ids = []
            print("   绘制轨迹线...")
            for i in range(len(result)-1):
                # 设置当前关节配置
                robot.set_joint_positions(robot.arm_joints, result[i])
                current_ee_pose = pp.get_link_pose(robot.robot, pp.link_from_name(robot.robot, HUSKY_TOOL0_NAME))
                
                # 设置下一个关节配置
                robot.set_joint_positions(robot.arm_joints, result[i+1])
                next_ee_pose = pp.get_link_pose(robot.robot, pp.link_from_name(robot.robot, HUSKY_TOOL0_NAME))
                
                # 绘制连接线
                line_id = pp.add_line(current_ee_pose[0], next_ee_pose[0], color=[0, 1, 1], width=2)  # 青色轨迹线
                trajectory_line_ids.append(line_id)
            
            # 创建当前位置指示器
            current_pos_sphere = None
            
            print("   开始交互式可视化...")
            print("   - 拖动'轨迹位置'滑动条手动控制")
            print("   - 设置'自动播放'为1开启自动播放")
            print("   - 调整'播放速度'控制自动播放速度")
            print("   - 按 ESC 键或 Q 键退出，或使用 Ctrl+C")
            
            try:
                while True:
                    # 读取滑动条值
                    try:
                        current_trajectory_point = int(p.readUserDebugParameter(trajectory_slider))
                        current_speed = p.readUserDebugParameter(speed_slider)
                        current_auto_play = p.readUserDebugParameter(auto_play_button)
                    except Exception:
                        # 如果无法读取滑动条，可能是窗口被关闭了
                        print("\n   检测到窗口关闭，退出可视化")
                        break
                    
                    # 处理自动播放
                    current_time = time.time()
                    if current_auto_play > 0.5:  # 自动播放开启
                        if current_time - last_time > (1.0 / current_speed):
                            auto_play_index += auto_play_direction
                            
                            # 到达边界时反向
                            if auto_play_index >= len(result) - 1:
                                auto_play_index = len(result) - 1
                                auto_play_direction = -1
                            elif auto_play_index <= 0:
                                auto_play_index = 0
                                auto_play_direction = 1
                            
                            # 更新滑动条位置
                            try:
                                p.setUserDebugParameter(trajectory_slider, auto_play_index)
                                current_trajectory_point = auto_play_index
                            except Exception:
                                print("\n   无法更新滑动条，退出可视化")
                                break
                            last_time = current_time
                    
                    # 如果轨迹点发生变化，更新机器人配置
                    if current_trajectory_point != prev_trajectory_point:
                        # 确保索引在有效范围内
                        trajectory_index = max(0, min(current_trajectory_point, len(result)-1))
                        
                        # 设置机器人关节配置
                        joint_config = result[trajectory_index]
                        robot.set_joint_positions(robot.arm_joints, joint_config)
                        
                        # 获取当前末端执行器位置
                        ee_pose = pp.get_link_pose(robot.robot, pp.link_from_name(robot.robot, HUSKY_TOOL0_NAME))
                        
                        # 更新当前位置指示器
                        if current_pos_sphere is not None:
                            try:
                                p.removeUserDebugItem(current_pos_sphere)
                            except:
                                pass  # 忽略删除错误
                        current_pos_sphere = pp.draw_point(ee_pose[0], size=0.03, color=[1, 0, 1])  # 紫色指示当前位置
                        
                        # 打印当前状态信息
                        print(f"\r   轨迹点 {trajectory_index:3d}/{len(result)-1}: "
                              f"末端位置 = [{ee_pose[0][0]:7.4f}, {ee_pose[0][1]:7.4f}, {ee_pose[0][2]:7.4f}] "
                              f"{'(自动播放)' if current_auto_play > 0.5 else '(手动控制)'}", end='', flush=True)
                        
                        prev_trajectory_point = current_trajectory_point
                    
                    # 短暂延迟以避免过度消耗CPU
                    time.sleep(0.01)
                    
                    # 检查用户是否按下了ESC键或其他退出条件
                    keys = p.getKeyboardEvents()
                    # ESC键的ASCII码是27
                    if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
                        break
                    # 也可以检查'q'键退出 (ASCII码113)
                    if 113 in keys and keys[113] & p.KEY_WAS_TRIGGERED:
                        break
                        
            except KeyboardInterrupt:
                print("\n   用户中断可视化")
            except Exception as e:
                print(f"\n   可视化过程中发生错误: {e}")
            finally:
                # 清理资源
                print("\n   清理可视化资源...")
                try:
                    if current_pos_sphere is not None:
                        p.removeUserDebugItem(current_pos_sphere)
                except:
                    pass
                
                try:
                    for line_id in trajectory_line_ids:
                        if line_id is not None:
                            p.removeUserDebugItem(line_id)
                except:
                    pass
                
                try:
                    p.removeUserDebugParameter(trajectory_slider)
                    p.removeUserDebugParameter(speed_slider) 
                    p.removeUserDebugParameter(auto_play_button)
                except:
                    pass
                
        else:
            print("   ✓ 规划成功！")
            print("   - 绿色点：起始位置")
            print("   - 蓝色点：目标位置") 
            print("   - 红色线：约束直线")
    else:
        print("   ✗ 规划失败")
    
    print("\n7. 按任意键退出...")
    pp.wait_for_user()
