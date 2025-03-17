# Third Party
import torch

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import Capsule, Cuboid, Cylinder, Mesh, Sphere, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

tensor_args = TensorDeviceType()

# load robot
config_file = load_yaml(join_path(get_robot_path(), "husky_ur5_e.yml"))["robot_cfg"]
robot_cfg = RobotConfig.from_dict(config_file, tensor_args)
kin_model = CudaRobotModel(robot_cfg.kinematics)

# load obstacles
pose_point_1 = [0.5, -0.35, 0.75]
# pose_point_1 = [5, 5, 5]
pose_quat_1 = [0.7071, 0.7071, 0.0, 0.0]
obstacle_1 = Cylinder(
    name="element_1",
    radius=0.01,
    height=1.0,
    pose=pose_point_1 + pose_quat_1,
    color=[1.0, 0, 0, 1.0],
)

pose_point_2 = [0.5, -0.35, 0.75]
# pose_point_2 = [5, 5, 5]
pose_quat_2 = [0.7071, 0.0, 0.7071, 0.0]
obstacle_2 = Cylinder(
    name="element_2",
    radius=0.01,
    height=1.0,
    pose=pose_point_2 + pose_quat_2,
    color=[1.0, 0, 0, 1.0],
)

pose_point_3 = [1.3, 0.5, 0.5]
# pose_point_3 = [5, 5, 5]
pose_quat_3 = [1, 0.0, 0.0, 0.0]
obstacle_3 = Cylinder(
    name="element_3",
    radius=0.01,
    height=1.0,
    pose=pose_point_3 + pose_quat_3,
    color=[1.0, 0, 0, 1.0],
)

# load grasp object
q_init = torch.zeros(1, 6).cuda()
state_init = kin_model.get_state(q_init)
x_init, y_init, z_init = state_init.ee_pose.position.cpu().numpy().flatten().tolist()
grasp_pose_point = [x_init + 0.15, y_init, z_init + 0.1]
grasp_pose_quat = [1, 0.0, 0.0, 0.0]
grasp_object = Cylinder(
    name="grasp_element",
    radius=0.01,
    height=1.0,
    pose=grasp_pose_point + grasp_pose_quat,
    color=[1.0, 0, 0, 1.0],
)

# create world config
world_config = WorldConfig(cylinder=[obstacle_1, obstacle_2, obstacle_3, grasp_object])
world_config_obb = WorldConfig.create_obb_world(world_config)

# create motion gen
motion_gen_config = MotionGenConfig.load_from_robot_config("husky_ur5_e.yml", world_config_obb, interpolation_dt=0.01)
motion_gen = MotionGen(motion_gen_config)
motion_gen.warmup()

# attach element
init_js = JointState(
    position=q_init, velocity=q_init, acceleration=q_init, jerk=q_init, joint_names=robot_cfg.cspace.joint_names
)
motion_gen.attach_objects_to_robot(
    init_js, ["grasp_element"], sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE
)

goal_pose = Pose.from_list(
    [0.6215999722480774, 0.3922000229358673, 1.051200032234192, -0.5, 0.5, -0.5, 0.5]
)  # x, y, z, qw, qx, qy, qz
start_state = JointState.from_position(
    torch.zeros(1, 6).cuda(), joint_names=config_file["kinematics"]["cspace"]["joint_names"]
)

traj = (
    motion_gen.plan_single(start_state, goal_pose, MotionGenPlanConfig(max_attempts=5)).get_interpolated_plan().position
)
# print("Success Paths: \n", traj)
traj = traj.cpu().numpy()
print("Trajectory Generated: \n", traj)

# -------------------- 下面是使用pybullet进行可视化的代码 --------------------#

import os
import sys
import time

import numpy as np
import pybullet as p
import pybullet_planning as pp

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from multi_tangent.collision import create_collision_bodies
from robot.robot_setup import RobotSetup
from utils.collision import init_pb

init_pb()
rb = RobotSetup("rb")
rb.set_joint_positions(rb.arm_joints, np.array([0, 0, 0, 0, 0, 0]))

line_pts_flattened = [np.array([0, 0, 0]), np.array([0, 0, 1])]
radius_per_edge = [0.01] * int(len(line_pts_flattened) / 2)

body_1 = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)[0]
pp.set_pose(body_1, (tuple(pose_point_1), tuple(pose_quat_1[1:] + [pose_quat_1[0]])))

body_2 = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)[0]
pp.set_pose(body_2, (tuple(pose_point_2), tuple(pose_quat_2[1:] + [pose_quat_2[0]])))

body_3 = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)[0]
pp.set_pose(body_3, (tuple(pose_point_3), tuple(pose_quat_3[1:] + [pose_quat_3[0]])))

attach_body = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)[0]
pp.set_pose(attach_body, (tuple(grasp_pose_point), tuple(grasp_pose_quat[1:] + [grasp_pose_quat[0]])))
attachemnt: pp.Attachment = pp.create_attachment(rb.robot, rb.tool_link, attach_body)

slider = p.addUserDebugParameter("replay", 0, 1, 0)

while True:
    slider_value = p.readUserDebugParameter(slider)
    time_idx = int(slider_value * (traj.shape[0] - 1))
    joint_val = traj[time_idx]
    rb.set_joint_positions(rb.arm_joints, joint_val)
    attachemnt.assign()
    time.sleep(1.0 / 60)

# **************************************************************************
# 官方代码
# **************************************************************************

# # Third Party
# import torch

# # cuRobo
# from curobo.types.math import Pose
# from curobo.types.robot import JointState
# from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

# world_config = {
#     "mesh": {
#         "base_scene": {
#             "pose": [10.5, 0.080, 1.6, 0.043, -0.471, 0.284, 0.834],
#             "file_path": "scene/nvblox/srl_ur10_bins.obj",
#         },
#     },
#     "cuboid": {
#         "table": {
#             "dims": [5.0, 5.0, 0.2],  # x, y, z
#             "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
#         },
#     },
# }

# # world_config = {
# #     "cuboid": {
# #         "test": {
# #             "dims": [5.0, 5.0, 0.2],  # x, y, z
# #             "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
# #         },
# #     },
# # }

# motion_gen_config = MotionGenConfig.load_from_robot_config(
#     "ur5e.yml",
#     world_config,
#     interpolation_dt=0.01,
# )
# motion_gen = MotionGen(motion_gen_config)
# motion_gen.warmup()

# goal_pose = Pose.from_list([-0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0])  # x, y, z, qw, qx, qy, qz
# start_state = JointState.from_position(
#     torch.zeros(1, 6).cuda(),
#     joint_names=[
#         "shoulder_pan_joint",
#         "shoulder_lift_joint",
#         "elbow_joint",
#         "wrist_1_joint",
#         "wrist_2_joint",
#         "wrist_3_joint",
#     ],
# )

# result = motion_gen.plan_single(start_state, goal_pose, MotionGenPlanConfig(max_attempts=1))
# traj = result.get_interpolated_plan()  # result.interpolation_dt has the dt between timesteps
# print("Trajectory Generated: ", result.success)

# **************************************************************************
# self collision check
# **************************************************************************

# # Third Party
# import torch

# # cuRobo
# from curobo.types.base import TensorDeviceType
# from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

# tensor_args = TensorDeviceType()

# world_config = {"cuboid": {"table": {"dims": [0.01, 0.01, 0.01], "pose": [2, 2, 2, 1, 0, 0, 0]}}}

# # load robot
# robot_file = "husky_ur5_e.yml"
# config = RobotWorldConfig.load_from_config(robot_file, world_config, collision_activation_distance=0.0)
# curobo_fn = RobotWorld(config)

# q = torch.zeros(1, 6).cuda()
# d_world, d_self = curobo_fn.get_world_self_collision_distance_from_joints(q)
# print(f"self collision: {d_self}")

# **************************************************************************
# mesh world test
# **************************************************************************

# from curobo.geom.types import WorldConfig, Cuboid, Mesh, Capsule, Cylinder, Sphere
# from curobo.util_file import get_assets_path, join_path

# obstacle_1 = Cuboid(
#     name="cube_1",
#     pose=[0.0, 0.0, 0.0, 0.043, -0.471, 0.284, 0.834],
#     dims=[0.2, 1.0, 0.2],
#     color=[0.8, 0.0, 0.0, 1.0],
# )

# # describe a mesh obstacle
# # import a mesh file:

# mesh_file = join_path(get_assets_path(), "scene/nvblox/srl_ur10_bins.obj")

# obstacle_2 = Mesh(
#     name="mesh_1",
#     pose=[0.0, 2, 0.5, 0.043, -0.471, 0.284, 0.834],
#     file_path=mesh_file,
#     scale=[0.5, 0.5, 0.5],
# )

# obstacle_3 = Capsule(
#     name="capsule",
#     radius=0.2,
#     base=[0, 0, 0],
#     tip=[0, 0, 0.5],
#     pose=[0.0, 5, 0.0, 0.043, -0.471, 0.284, 0.834],
#     color=[0, 1.0, 0, 1.0],
# )

# obstacle_4 = Cylinder(
#     name="cylinder_1",
#     radius=0.01,
#     height=1,
#     pose=[0.0, 0, 0.0, 1, 0, 0, 0],
#     color=[0, 1.0, 0, 1.0],
# )

# obstacle_5 = Sphere(
#     name="sphere_1",
#     radius=0.2,
#     pose=[0.0, 7, 0.0, 0.043, -0.471, 0.284, 0.834],
#     color=[0, 1.0, 0, 1.0],
# )

# world_model = WorldConfig(
#     # mesh=[obstacle_2],
#     # cuboid=[obstacle_1],
#     # capsule=[obstacle_3],
#     cylinder=[obstacle_4],
#     # sphere=[obstacle_5],
# )

# # assign random color to each obstacle for visualization
# world_model.randomize_color(r=[0.2, 0.7], g=[0.8, 1.0])

# file_path = "debug_mesh.obj"
# world_model.save_world_as_mesh(file_path)
