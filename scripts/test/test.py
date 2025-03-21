import os
import sys
import time
from copy import deepcopy

import numpy as np
import pybullet as p
import pybullet_planning as pp

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from multi_tangent.collision import create_collision_bodies
from ompl_manipulator_test import TrajectoryOMPLSolver
from robot.robot_setup import RobotSetup
from utils.collision import init_pb

if __name__ == "__main__":

    init_pb()
    rb = RobotSetup("rb")

    rb.set_joint_positions(rb.arm_joints, np.array([0, 0, 0, 0, 0, 0]))

    line_pts_flattened = [np.array([0, 0, 0]), np.array([0, 0, 1])]
    radius_per_edge = [0.01] * int(len(line_pts_flattened) / 2)

    # load obstacles
    # pose_point_1 = [0.5, -0.35, 0.75]
    # pose_quat_1 = [0.7071, 0.7071, 0.0, 0.0]
    # body_1 = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)[0]
    # pp.set_pose(body_1, (tuple(pose_point_1), tuple(pose_quat_1[1:] + [pose_quat_1[0]])))

    pose_point_2 = [0.5, -0.35, 0.75]
    pose_quat_2 = [0.7071, 0.0, 0.7071, 0.0]
    body_2 = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)[0]
    pp.set_pose(body_2, (tuple(pose_point_2), tuple(pose_quat_2[1:] + [pose_quat_2[0]])))

    pose_point_3 = [1.3, 0.5, 0.5]
    pose_quat_3 = [1, 0.0, 0.0, 0.0]
    body_3 = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)[0]
    pp.set_pose(body_3, (tuple(pose_point_3), tuple(pose_quat_3[1:] + [pose_quat_3[0]])))

    # attach_body = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)[0]
    # pp.set_pose(attach_body, (tuple(grasp_pose_point), tuple(grasp_pose_quat[1:] + [grasp_pose_quat[0]])))
    # attachemnt: pp.Attachment = pp.create_attachment(rb.robot, rb.tool_link, attach_body)

    extra_disabled_collisions = [
        ((rb.robot, pp.link_from_name(rb.robot, "ur_arm_wrist_3_link")), (rb.ee_attachment.child, pp.BASE_LINK))
    ]

    collision_fn = pp.get_collision_fn(
        rb.robot,
        rb.arm_joints,
        obstacles=[body_2, body_3],
        attachments=[rb.ee_attachment] + rb.attachments,
        self_collisions=True,
        disabled_collisions=rb.disabled_collisions,
        extra_disabled_collisions=extra_disabled_collisions,
        max_distance=0.0,
    )

    # 创建规划器实例
    solver = TrajectoryOMPLSolver(collision_fn)

    # 设置起始和目标关节角度
    init_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    target_q = np.array([0, -np.pi / 2, -np.pi / 2, 0, 0, 0])

    # 执行规划并获取路径
    traj = solver.plan(init_q, target_q, time=10.0)
    if traj is not None:
        print("path: \n", traj)

    # -------------------- 下面是使用pybullet进行可视化的代码 --------------------#

    slider = p.addUserDebugParameter("replay", 0, 1, 0)

    while True:
        slider_value = p.readUserDebugParameter(slider)
        time_idx = int(slider_value * (traj.shape[0] - 1))
        joint_val = traj[time_idx]
        rb.set_joint_positions(rb.arm_joints, joint_val)
        time.sleep(1.0 / 60)
