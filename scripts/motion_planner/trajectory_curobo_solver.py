import os
import sys
import time
from copy import deepcopy
from typing import Callable, Dict, List, Set, Tuple, Union

import numpy as np
import pybullet_planning as pp
import torch

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import Capsule, Cuboid, Cylinder, Mesh, Sphere, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_logger
from curobo.util_file import get_robot_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from robot.robot_setup import RobotSetup


class TrajectoryCuroboSolver:
    def __init__(self, robot_setup: RobotSetup, tensor_args: TensorDeviceType) -> None:
        self.robot_setup = robot_setup
        self.tensor_args = tensor_args
        self.robot_cfg = RobotConfig.from_dict(load_yaml(join_path(get_robot_path(), "husky_ur5_e.yml"))["robot_cfg"], tensor_args)
        self.kin_model = CudaRobotModel(self.robot_cfg.kinematics)
        self.logger = setup_logger("error", "trajectory_curobo_solver")

    def plan(
        self,
        q_init: np.ndarray,
        q_target: np.ndarray,
        max_time: int,
        max_attempts: int,
        element_bodies: List[int],
        element_infos: Dict,
        grasped_approximate_info: Union[None, Dict] = None,
        grasped_approximate_body: Union[None, int] = None,
        grasped_approximate_attachment: Union[None, pp.Attachment] = None,
        collision_fn: Callable = None,
    ) -> Dict:

        start_time = time.time()

        # -------------------- load obstacles --------------------#
        obstacles = []
        for element_body in element_bodies:
            info = element_infos[element_body]
            pose = pp.multiply(pp.invert(pp.get_pose(self.robot_setup.robot)), pp.get_pose(element_body))
            point = list(pose[0])
            quat = [pose[1][3], pose[1][0], pose[1][1], pose[1][2]]
            obstacle = Cylinder(name=f"element_{element_body}", radius=info["shape_parameters"]["radius"], height=info["shape_parameters"]["height"], pose=point + quat, color=[1.0, 0, 0, 1.0])
            obstacles.append(obstacle)

        # -------------------- init joint states --------------------#
        q_init_tensor = torch.tensor(q_init, dtype=torch.float32).reshape(1, 6).cuda()
        dq_init_tensor = torch.zeros(1, 6).cuda()
        ddq_init_tensor = torch.zeros(1, 6).cuda()
        dddq_init_tensor = torch.zeros(1, 6).cuda()
        q_target_tensor = torch.tensor(q_target, dtype=torch.float32).reshape(1, 6).cuda()

        # -------------------- load grasp object --------------------#
        if grasped_approximate_info is not None:
            self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, q_init)
            grasped_approximate_attachment.assign()
            approximate_delta_pose = pp.Pose(point=grasped_approximate_info["offset"], euler=grasped_approximate_info["rotation"])
            approximate_pose = pp.multiply(pp.get_link_pose(self.robot_setup.robot, self.robot_setup.tool_link), approximate_delta_pose)
            pp.set_pose(grasped_approximate_body, approximate_pose)
            grasp_pose_point = list(approximate_pose[0])
            grasp_pose_quat = [approximate_pose[1][3], approximate_pose[1][0], approximate_pose[1][1], approximate_pose[1][2]]
            grasp_object = Cylinder(name="grasp_element", radius=grasped_approximate_info["radius"], height=grasped_approximate_info["height"], pose=grasp_pose_point + grasp_pose_quat, color=[1.0, 0, 0, 1.0])
            obstacles.append(grasp_object)

        # -------------------- create world config --------------------#
        world_config = WorldConfig(cylinder=obstacles)
        world_config_obb = WorldConfig.create_obb_world(world_config)
        world_config_mesh = WorldConfig.create_mesh_world(world_config)

        # -------------------- create motion gen --------------------#
        motion_gen_config = MotionGenConfig.load_from_robot_config("husky_ur5_e.yml", world_config_mesh, interpolation_dt=0.001, interpolation_steps=50000)
        motion_gen = MotionGen(motion_gen_config)
        motion_gen.warmup()

        # -------------------- attach element --------------------#
        init_js = JointState(position=q_init_tensor, velocity=dq_init_tensor, acceleration=ddq_init_tensor, jerk=dddq_init_tensor, joint_names=self.robot_cfg.cspace.joint_names)
        if grasped_approximate_info is not None:
            motion_gen.attach_objects_to_robot(init_js, ["grasp_element"], sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE)

        # -------------------- set target and start --------------------#
        out = self.kin_model.get_state(q_target_tensor)
        goal_pose = Pose.from_list(out.ee_pose.position.cpu().numpy().flatten().tolist() + out.ee_pose.quaternion.cpu().numpy().flatten().tolist())
        start_state = JointState.from_position(q_init_tensor, joint_names=self.robot_cfg.cspace.joint_names)

        # -------------------- plan --------------------#
        while time.time() - start_time < max_time:
            current_time = time.time()
            result = motion_gen.plan_single(start_state, goal_pose, MotionGenPlanConfig(max_attempts=max_attempts, timeout=max_time - (current_time - start_time)))
            if result.success:
                path = result.get_interpolated_plan().position.cpu().numpy()
                # Check start and end points
                if len(path) > 0:
                    start_diff = np.linalg.norm(path[0] - q_init_tensor.cpu().numpy())
                    end_diff = np.linalg.norm(path[-1] - q_target_tensor.cpu().numpy())

                    print(f"Start point difference: {start_diff:.6f}, End point difference: {end_diff:.6f}")

                    # If the start or end point difference is too large, consider planning failed
                    if start_diff > 1e-4:
                        print("Planning failed due to large start difference")
                        continue
                    if end_diff > 1e-6:
                        print("Planning failed due to large end difference")
                        continue

                collision_free = True
                if collision_fn is not None:
                    for q in path:
                        if collision_fn(q):
                            collision_free = False
                            break
                if collision_free:
                    return {"success": True, "path": path}
            else:
                return {"success": False, "path": None}

        return {"success": False, "path": None}
