from collections import namedtuple
from typing import Tuple, Union

import numpy as np
import pinocchio
from pinocchio.robot_wrapper import RobotWrapper

Joint = namedtuple("Joint", ["id", "type", "idx_qs", "idx_qn"])


class PinocchioSolver(object):

    CONTROL_JOINT_NAMES = [
        "x",
        "y",
        "theta",
        "ur_arm_shoulder_pan_joint",
        "ur_arm_shoulder_lift_joint",
        "ur_arm_elbow_joint",
        "ur_arm_wrist_1_joint",
        "ur_arm_wrist_2_joint",
        "ur_arm_wrist_3_joint",
    ]

    MANIPULATOR_JOINT_NAMES = [
        "ur_arm_shoulder_pan_joint",
        "ur_arm_shoulder_lift_joint",
        "ur_arm_elbow_joint",
        "ur_arm_wrist_1_joint",
        "ur_arm_wrist_2_joint",
        "ur_arm_wrist_3_joint",
    ]

    def __init__(self, urdf_path: str) -> None:
        self.whole_model = pinocchio.buildModelFromUrdf(urdf_path)
        self.whole_data = self.whole_model.createData()

        manipulator_base_joint_id = self.whole_model.getJointId(self.MANIPULATOR_JOINT_NAMES[0])
        manipulator_tip_joint_id = self.whole_model.getJointId(self.MANIPULATOR_JOINT_NAMES[-1])
        manipulator_joint_ids = list(range(manipulator_base_joint_id, manipulator_tip_joint_id + 1))

        all_joint_ids = list(range(1, self.whole_model.njoints))
        joints_to_lock = list(set(all_joint_ids) - set(manipulator_joint_ids))

        self.model = pinocchio.buildReducedModel(self.whole_model, joints_to_lock, pinocchio.neutral(self.whole_model))
        self.data = self.model.createData()

        self.InitJointInfo()

    def PrintTest(self):
        joint_info = {}
        for i in range(self.model.njoints):
            joint = self.model.joints[i]

            # 获取关节类型、idx_qs 和 nq
            joint_type = joint.shortname()  # 关节类型
            idx_qs = joint.idx_q  # 关节在配置向量中的起始索引
            nq = joint.nq  # 关节的自由度
            name = self.model.names[i]

            # 存储信息
            joint_info[name] = {"joint_id": i, "joint_type": joint_type, "idx_qs": idx_qs, "nq": nq}
        print(joint_info)

        print(f"{'Frame ID':<10} {'Link Name':<20} {'Parent Joint ID':<15} {'Placement (translation)'}")
        print("=" * 70)

        # 遍历所有帧
        for frame_id, frame in enumerate(self.model.frames):
            # 获取帧的名称、父关节ID和在父关节中的位姿
            frame_name = frame.name
            parent_joint_id = frame.parent
            placement_translation = frame.placement.translation

            # 打印信息
            print(f"{frame_id:<10} {frame_name:<20} {parent_joint_id:<15} {placement_translation}")

    def ik(
        self,
        pose: np.ndarray,
        base_name: str,
        tip_name: str,
        qinit: Union[np.ndarray, None] = None,
        link: bool = True,
        relative: bool = True,
        verbose: bool = False,
        output: str = "array",
        output_flag: bool = False,
        eps=1e-4,
        IT_MAX=1000,
        DT=1e-1,
        damp=1e-12,
    ) -> Union[Union[np.ndarray, None], Tuple[Union[np.ndarray, None], bool]]:
        """
        Calculate IK solution.

        Params:
            pose (np.ndarray, 4x4): SE(3) matrix
            base_name (str): name of link or joint
            tip_name (str): name of link or joint
            qinit (np.ndarray, None): initial guess of arm joint
            link (bool, True): whether use link name (d) or joint name
            relative (bool, True): whether use relative pose in base frame
            verbose (bool, False): whether print debug info
            output (str, "array"): output result type
            output_flag (bool, False): whether output includes solve status

        Returns:
            q (np.ndarray | None): joint conf to the target pose
            success (bool, [optional]): solve status
        """
        if qinit is not None:
            q = self.Joints2Pinocchio(qinit.copy())
        else:
            q = pinocchio.neutral(self.model)

        if link:
            base_link_id = self.model.getFrameId(base_name)
            tip_link_id = self.model.getFrameId(tip_name)

            tip_joint_id = self.model.frames[tip_link_id].parentJoint

            pinocchio.forwardKinematics(self.model, self.data, q)
            pinocchio.updateFramePlacements(self.model, self.data)

            world_from_base = self.data.oMf[base_link_id].homogeneous

        else:
            base_joint_id = self.model.getJointId(base_name)
            tip_joint_id = self.model.getJointId(tip_name)

            pinocchio.forwardKinematics(self.model, self.data, q)
            pinocchio.updateFramePlacements(self.model, self.data)

            world_from_base = self.data.oMi[base_joint_id].homogeneous

        if relative:
            target_pose_in_world = world_from_base @ pose
        else:
            target_pose_in_world = pose

        i = 0
        success = False
        while i < IT_MAX:
            pinocchio.forwardKinematics(self.model, self.data, q)
            pinocchio.updateFramePlacements(self.model, self.data)

            current_pose = self.data.oMi[tip_joint_id]
            iMd = current_pose.actInv(pinocchio.SE3(target_pose_in_world[:3, :3], target_pose_in_world[:3, 3]))

            err = pinocchio.log(iMd).vector

            if np.linalg.norm(err) < eps:
                success = True
                break

            J = pinocchio.computeJointJacobian(self.model, self.data, q, tip_joint_id)
            J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)

            v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pinocchio.integrate(self.model, q, v * DT)

            if verbose:
                if not i % 10:
                    print(f"Iteration {i}: error = {err.T}")
            i += 1

        if verbose:
            if success:
                print("Convergence achieved!")
            else:
                print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")

        if not success:
            rtn1 = None
        else:
            if output == "list":
                rtn1 = self.Pinocchio2Joints(q).tolist()
            else:
                rtn1 = self.Pinocchio2Joints(q)

        if output_flag:
            return rtn1, success
        else:
            return rtn1

    def InitJointInfo(self):
        self.joint_info = {}
        for i in range(self.model.njoints):
            joint = self.model.joints[i]

            name = self.model.names[i]
            type = joint.shortname()
            idx_qs = joint.idx_q
            nq = joint.nq

            self.joint_info[name] = Joint(i, type, idx_qs, nq)

    def Pinocchio2Joints(self, q_pin) -> np.ndarray:
        q_joint = np.zeros(len(self.MANIPULATOR_JOINT_NAMES))

        for key, value in self.joint_info.items():
            value: Joint
            if value.idx_qs == -1:
                continue

            if key in self.MANIPULATOR_JOINT_NAMES:
                index = self.MANIPULATOR_JOINT_NAMES.index(key)
                if value.idx_qn == 2:
                    q_joint[index] = np.arctan2(q_pin[value.idx_qs + 1], q_pin[value.idx_qs])
                else:
                    q_joint[index] = q_pin[value.idx_qs]

        return q_joint

    def Joints2Pinocchio(self, q_joint) -> np.ndarray:
        q_pin = np.zeros(self.model.nq)

        for key, value in self.joint_info.items():
            value: Joint
            if value.idx_qs == -1:
                continue

            if key in self.MANIPULATOR_JOINT_NAMES:
                joint_value = q_joint[self.MANIPULATOR_JOINT_NAMES.index(key)]
            else:
                joint_value = 0

            if value.idx_qn == 2:
                q_pin[value.idx_qs] = np.cos(joint_value)
                q_pin[value.idx_qs + 1] = np.sin(joint_value)
            else:
                q_pin[value.idx_qs] = joint_value

        return q_pin


if __name__ == "__main__":

    import os
    import sys
    from functools import partial

    import pybullet_planning as pp
    from tracikpy import TracIKSolver

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from utils.collision import init_pb
    from robot.robot_setup import RobotSetup

    robot_urdf = (
        "/home/jeong/summer_research/eth/husky_assembly/data/husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf"
    )

    pinocchio_solver = PinocchioSolver(robot_urdf)
    pin_ik_solver = partial(pinocchio_solver.ik, base_name="world_link", tip_name="ur_arm_tool0", relative=False)
    pin_ik_solver_relative = partial(
        pinocchio_solver.ik, base_name="ur_arm_base_link", tip_name="ur_arm_tool0", relative=True
    )

    trac_ik_solver = TracIKSolver(robot_urdf, "world_link", "ur_arm_tool0")
    trac_ik_solver_relative = TracIKSolver(robot_urdf, "ur_arm_base_link", "ur_arm_tool0")

    target_pose = np.array([[0, 0, 1, 0.5], [0, 1, 0, 0.25], [-1, 0, 0, 0.5], [0, 0, 0, 1]])

    q_trac = trac_ik_solver_relative.ik(target_pose, qinit=np.array([0, 0, 0, 0, 0, 0]))
    q_pin = pin_ik_solver_relative(target_pose, qinit=q_trac)

    print(q_pin)
    print(q_trac)

    init_pb()
    rb = RobotSetup("r0")

    pp.draw_point([0.5, 0.25, 0.5], size=0.5)

    rb.set_joint_positions(rb.arm_joints, q_pin)
    pp.wait_for_user()

    rb.set_joint_positions(rb.arm_joints, q_trac)
    pp.wait_for_user()
