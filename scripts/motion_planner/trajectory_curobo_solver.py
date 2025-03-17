import os
import sys
from copy import deepcopy
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import pybullet_planning as pp
import torch

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import Capsule, Cuboid, Cylinder, Mesh, Sphere, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from robot.robot_setup import RobotSetup


class TrajectoryCuroboSolver:
    def __init__(self, urdf_path: str, robot_setup: RobotSetup, tensor_args: TensorDeviceType) -> None:
        self.urdf_path = urdf_path
        self.robot_setup = robot_setup
        self.tensor_args = tensor_args
        self.robot_cfg = RobotConfig.from_dict(
            load_yaml(join_path(get_robot_path(), "husky_ur5_e.yml"))["robot_cfg"], tensor_args
        )
        self.kin_model = CudaRobotModel(self.robot_cfg.kinematics)

    def plan(
        self,
        q_init: np.ndarray,
        q_target: np.ndarray,
        max_time: int,
        max_attempts: int,
        element_bodies: List[int],
        grasped_element: Union[None, int] = None,
        grasped_attachment: Union[None, pp.Attachment] = None,
    ) -> Dict:
        # -------------------- load obstacles --------------------#
        obstacles = []
        for element_body in element_bodies:
            pose = pp.multiply(pp.invert(pp.get_pose(self.robot_setup.robot)), pp.get_pose(element_body))
            point = list(pose[0])
            quat = [pose[1][3], pose[1][0], pose[1][1], pose[1][2]]
            obstacle = Cylinder(
                name=f"element_{element_body}",
                radius=0.01,
                height=1.0,
                pose=point + quat,
                color=[1.0, 0, 0, 1.0],
            )
            obstacles.append(obstacle)

        # -------------------- init joint states --------------------#
        q_init_tensor = torch.tensor(q_init, dtype=torch.float32).reshape(1, 6).cuda()
        dq_init_tensor = torch.zeros(1, 6).cuda()
        ddq_init_tensor = torch.zeros(1, 6).cuda()
        dddq_init_tensor = torch.zeros(1, 6).cuda()
        q_target_tensor = torch.tensor(q_target, dtype=torch.float32).reshape(1, 6).cuda()

        # -------------------- load grasp object --------------------#
        if grasped_element is not None and grasped_attachment is not None:
            self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, q_init)
            grasped_attachment.assign()
            grasp_element_pose = pp.multiply(
                pp.invert(pp.get_pose(self.robot_setup.robot)), pp.get_pose(grasped_element)
            )
            grasp_pose_point = list(grasp_element_pose[0])
            grasp_pose_quat = [
                grasp_element_pose[1][3],
                grasp_element_pose[1][0],
                grasp_element_pose[1][1],
                grasp_pose_quat[1][2],
            ]
            grasp_object = Cylinder(
                name="grasp_element",
                radius=0.01,
                height=1.0,
                pose=grasp_pose_point + grasp_pose_quat,
                color=[1.0, 0, 0, 1.0],
            )
            obstacles.append(grasp_object)

        # -------------------- create world config --------------------#
        world_config = WorldConfig(cylinder=obstacles)
        world_config_obb = WorldConfig.create_obb_world(world_config)

        # -------------------- create motion gen --------------------#
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            "husky_ur5_e.yml", world_config_obb, interpolation_dt=0.01
        )
        motion_gen = MotionGen(motion_gen_config)
        motion_gen.warmup()

        # -------------------- attach element --------------------#
        init_js = JointState(
            position=q_init_tensor,
            velocity=dq_init_tensor,
            acceleration=ddq_init_tensor,
            jerk=dddq_init_tensor,
            joint_names=self.robot_cfg.cspace.joint_names,
        )
        if grasped_element is not None and grasped_attachment is not None:
            motion_gen.attach_objects_to_robot(
                init_js, ["grasp_element"], sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE
            )

        # -------------------- set target and start --------------------#
        out = self.kin_model.get_state(q_target_tensor)
        goal_pose = Pose.from_list(
            out.ee_pose.position.cpu().numpy().flatten().tolist()
            + out.ee_pose.quaternion.cpu().numpy().flatten().tolist()
        )
        start_state = JointState.from_position(q_init_tensor, joint_names=self.robot_cfg.cspace.joint_names)

        # -------------------- plan --------------------#
        result = motion_gen.plan_single(
            start_state, goal_pose, MotionGenPlanConfig(max_attempts=max_attempts, timeout=max_time)
        )

        if result.success:
            path = result.get_interpolated_plan().position.cpu().numpy()
            return {"success": True, "path": path}
        else:
            return {"success": False, "path": None}
