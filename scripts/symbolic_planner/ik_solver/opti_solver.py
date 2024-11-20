import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Union

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from utils.utils import HideOutput


def ParseURDF(urdf_path: str) -> Dict:
    """
    Parse URDF file and extract joint info.

    Params:
        urdf_path (str): path of urdf file

    Returns:
        Dict: joint info
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        joint_type = joint.get("type")

        origin = joint.find("origin")
        if origin is not None:
            xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
            rpy = [float(r) for r in origin.get("rpy", "0 0 0").split()]
        else:
            xyz, rpy = [0, 0, 0], [0, 0, 0]

        axis = joint.find("axis")
        if axis is not None:
            axis = [float(a) for a in axis.get("xyz", "1 0 0").split()]
        else:
            axis = [1, 0, 0]

        joints[name] = {"type": joint_type, "origin": {"xyz": xyz, "rpy": rpy}, "axis": axis}

    return joints


def RPY2Matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Compute matrix given by rpy.

    Params:
        roll (float): roll
        pitch (float): pitch
        yaw (float): yaw

    Returns:
        np.ndarray: 3x3 matrix
    """
    Rx = np.zeros((3, 3))

    Rx[0, 0] = 1
    Rx[0, 1] = 0
    Rx[0, 2] = 0

    Rx[1, 0] = 0
    Rx[1, 1] = np.cos(roll)
    Rx[1, 2] = -np.sin(roll)

    Rx[2, 0] = 0
    Rx[2, 1] = np.sin(roll)
    Rx[2, 2] = np.cos(roll)

    Ry = np.zeros((3, 3))

    Ry[0, 0] = np.cos(pitch)
    Ry[0, 1] = 0
    Ry[0, 2] = np.sin(pitch)

    Ry[1, 0] = 0
    Ry[1, 1] = 1
    Ry[1, 2] = 0

    Ry[2, 0] = -np.sin(pitch)
    Ry[2, 1] = 0
    Ry[2, 2] = np.cos(pitch)

    Rz = np.zeros((3, 3))

    Rz[0, 0] = np.cos(yaw)
    Rz[0, 1] = -np.sin(yaw)
    Rz[0, 2] = 0

    Rz[1, 0] = np.sin(yaw)
    Rz[1, 1] = np.cos(yaw)
    Rz[1, 2] = 0

    Rz[2, 0] = 0
    Rz[2, 1] = 0
    Rz[2, 2] = 1

    return Rz @ Ry @ Rx


def Skew(v: ca.MX) -> ca.MX:
    """
    Generate skew-symmetric matrix given by axis.

    Params:
        v (ca.MX): vector of axis

    Returns:
        ca.MX: skew-symmetric matrix
    """
    assert v.size1() == 3, "输入向量必须是三维的"

    x = v[0]
    y = v[1]
    z = v[2]

    skew_matrix = ca.MX(3, 3)

    skew_matrix[0, 0] = 0
    skew_matrix[0, 1] = -z
    skew_matrix[0, 2] = y

    skew_matrix[1, 0] = z
    skew_matrix[1, 1] = 0
    skew_matrix[1, 2] = -x

    skew_matrix[2, 0] = -y
    skew_matrix[2, 1] = x
    skew_matrix[2, 2] = 0

    return skew_matrix


def Expm(A: ca.MX, n_terms: int = 20) -> ca.MX:
    """
    Compute the exponential exp(A) of the matrix A using Taylor series expansion.

    Params:
        A (ca.MX): matrix
        n_terms (int, 20): number of expanded items

    Returns:
        ca.MX: matrix exp(A)
    """
    if A.size1() != A.size2():
        raise ValueError("矩阵必须是方阵")

    exp_A = ca.MX.eye(A.size1())
    A_power = ca.MX.eye(A.size1())
    factorial = 1

    for n in range(1, n_terms + 1):
        A_power = A_power @ A
        factorial *= n
        exp_A += A_power / factorial

    return exp_A


def TransformMatrix(xyz: List[float], rpy: List[float], axis: List[float], q_val: ca.MX, joint_type: str) -> ca.MX:
    """
    Construct transformation matrix according to joint type.

    Params:
        xyz (List[float]): xyz
        rpy (List[float]): rpy
        axis (List[float]): axis
        q_val (ca.MX): symbolic joint angle
        joint_type (str): revolute/prismatic

    Returns:
        ca.MX: 4x4 matrix
    """
    T = ca.MX.eye(4)
    T[:3, :3] = RPY2Matrix(*rpy)
    T[:3, 3] = xyz

    if joint_type == "revolute":
        R_joint = ca.MX.eye(4)
        R_joint[:3, :3] = Expm(q_val * Skew(ca.MX(axis)))
        return T @ R_joint
    elif joint_type == "prismatic":
        P_joint = ca.MX.eye(4)
        P_joint[:3, 3] = ca.MX(axis) * q_val
        return T @ P_joint
    else:
        return T


def SymbolicForward(
    urdf_path: str,
    joint_name_list: List[str],
    manipulator_joint_list: List[str],
    q: Union[ca.MX, None] = None,
    output_type: str = "function",
) -> Union[ca.Function, ca.MX]:
    """
    Creates symbolic forward kinematics equations given a URDF file path and a list of joint names.

    Params:
        urdf_path (str): urdf path of robot
        joint_name_list (List[str]): name list of manipulator including all redundant joints
        manipulator_joint_list (List[str]): name list of manipulator
        q (ca.MX | None, None): joint variables
        output_type (str, "function"): "function"/"matrix"

    Returns:
        ca.Function: [q] --> np.ndarray (4x4)
    """
    joints = ParseURDF(urdf_path)
    if q is None:
        q = ca.MX.sym("q", len(manipulator_joint_list))
    T = ca.MX.eye(4)
    for i, joint_name in enumerate(joint_name_list):
        joint = joints[joint_name]
        if joint_name in manipulator_joint_list:
            i_q = manipulator_joint_list.index(joint_name)
            joint_T = TransformMatrix(
                joint["origin"]["xyz"], joint["origin"]["rpy"], joint["axis"], q[i_q], joint["type"]
            )
        else:
            joint_T = TransformMatrix(joint["origin"]["xyz"], joint["origin"]["rpy"], joint["axis"], 0, joint["type"])
        T = T @ joint_T

    if output_type == "function":
        fk_function = ca.Function("forward_kinematics", [q], [T])
        return fk_function
    elif output_type == "matrix":
        return T
    else:
        fk_function = ca.Function("forward_kinematics", [q], [T])
        return fk_function


class OptiSolver(object):

    MANIPULATOR_JOINT_NAMES = [
        "ur_arm_shoulder_pan_joint",
        "ur_arm_shoulder_lift_joint",
        "ur_arm_elbow_joint",
        "ur_arm_wrist_1_joint",
        "ur_arm_wrist_2_joint",
        "ur_arm_wrist_3_joint",
    ]

    REDUCED_MODEL_JOINT_NAMES = [
        "ur_arm_base_link-base_fixed_joint",
        "ur_arm_shoulder_pan_joint",
        "ur_arm_shoulder_lift_joint",
        "ur_arm_elbow_joint",
        "ur_arm_wrist_1_joint",
        "ur_arm_wrist_2_joint",
        "ur_arm_wrist_3_joint",
    ]

    def __init__(self, urdf_path: str) -> None:
        self.urdf_path = urdf_path

        self.SolverInit()

    def ObjectiveFunction(self, T: ca.MX, T_tar: ca.MX) -> ca.MX:
        """
        Generate objective function seperately.

        Params:
            T (ca.MX): 4x4 matrix
            T_tar (ca.MX): 4x4 matrix

        Returns:
            ca.MX: objective value
        """
        err = 0
        for i in range(4):
            for j in range(4):
                err += (T[i, j] - T_tar[i, j]) ** 2
        return err

    def SolverInit(self):
        self.solver = ca.Opti()

        # -------------------- set parameters --------------------#
        self.N = len(self.MANIPULATOR_JOINT_NAMES)
        self.q = self.solver.variable(self.N)
        self.T_tar = self.solver.parameter(4, 4)

        # -------------------- get matrix and objective function --------------------#
        self.fk_function = SymbolicForward(self.urdf_path, self.REDUCED_MODEL_JOINT_NAMES, self.MANIPULATOR_JOINT_NAMES)
        self.fk_matrix = SymbolicForward(
            self.urdf_path, self.REDUCED_MODEL_JOINT_NAMES, self.MANIPULATOR_JOINT_NAMES, q=self.q, output_type="matrix"
        )
        self.obj = self.ObjectiveFunction(self.fk_matrix, self.T_tar)

        # -------------------- set solver objective function and constrains --------------------#
        self.solver.minimize(self.obj)
        self.solver.subject_to(ca.vec(self.q) <= np.pi)
        self.solver.subject_to(ca.vec(self.q) >= -np.pi)

        # -------------------- set solver options --------------------#
        p_opts = {"expand": True, "print_time": 0}
        s_opts = {"max_iter": 100, "print_level": 0}
        self.solver.solver("ipopt", p_opts, s_opts)

    def forward(self, q: Union[np.ndarray, List[float]]) -> np.ndarray:

        if isinstance(q, np.ndarray):
            q = q.tolist()

        T = np.array(self.fk_function(q))

        return T

    def ik(
        self,
        pose: np.ndarray,
        qinit: Union[np.ndarray, None] = None,
        output: str = "array",
        output_flag: bool = False,
    ) -> Union[Union[np.ndarray, List[float], None], Tuple[Union[np.ndarray, List[float], None], bool]]:
        """
        Calculate IK solution.

        Params:
            pose (np.ndarray, 4x4): SE(3) matrix
            qinit (np.ndarray, None): initial guess of arm joint
            output (str, "array"): array/list, output result type
            output_flag (bool, False, [not used]): whether output includes solve status

        Returns:
            q (np.ndarray | [float] | None): joint conf to the target pose
            success (bool, [optional]): solve status
        """

        self.solver.set_value(self.T_tar, pose)
        self.solver.set_initial(self.q, qinit)
        with HideOutput():
            solution = self.solver.solve()
        return np.array(solution.value(self.q))


if __name__ == "__main__":
    import os
    import sys
    from functools import partial

    HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(HERE)

    import casadi as ca
    import numpy as np
    from ik_solver.opti_solver import OptiSolver
    from ik_solver.pinocchio_solver import PinocchioSolver

    np.set_printoptions(precision=3, suppress=True)

    urdf_path = (
        "/home/jeong/summer_research/eth/husky_assembly/data/husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf"
    )
    solver = PinocchioSolver(urdf_path)
    opti_ik_solver = OptiSolver(urdf_path)

    # **************************************************************************
    # link forward
    # **************************************************************************

    print("\n==================== link forward ====================\n")

    pin_ik_solver_relative = partial(solver.ik, tip_name="ur_arm_tool0", link=True)

    target_pose = np.array([[0, 0, 1, 0.7], [0, 1, 0, 0.25], [-1, 0, 0, 0.5], [0, 0, 0, 1]])

    q_pin = pin_ik_solver_relative(
        target_pose, qinit=np.array([-1.29891436, 0.3618943, -1.79107962, -1.71236292, -1.84276683, 1.57065429])
    )

    print("> ik solver: q_pin")
    print(q_pin)
    print()

    print("> forward: pose real")
    print(target_pose)
    print()

    pose = solver.forward(q_pin, "ur_arm_tool0", link=True, relative_output=True)
    print("> forward: pose pinocchio")
    print(pose)
    print()

    pose = opti_ik_solver.forward(q_pin)
    print("> forward: pose opti")
    print(pose)
    print()

    # **************************************************************************
    # joint forward
    # **************************************************************************

    print("\n==================== joint forward ====================\n")

    pin_ik_solver_relative = partial(solver.ik, tip_name="ur_arm_wrist_3_joint", link=False)

    base_from_tip = np.array([[0, 0, 1, 0.5], [0, 1, 0, 0.25], [-1, 0, 0, 0.5], [0, 0, 0, 1]])

    q_pin = pin_ik_solver_relative(
        base_from_tip, qinit=np.array([-1.29891436, 0.3618943, -1.79107962, -1.71236292, -1.84276683, 1.57065429])
    )

    print("> ik solver: q_pin")
    print(q_pin)
    print()

    print("> forward: pose real")
    print(base_from_tip)
    print()

    pose = solver.forward(q_pin, "ur_arm_wrist_3_joint", link=False, relative_output=True)
    print("> forward: pose pinocchio")
    print(pose)
    print()

    pose = opti_ik_solver.forward(q_pin)
    print("> forward: pose opti")
    print(pose)
    print()

    # **************************************************************************
    # link ik
    # **************************************************************************

    print("\n==================== link ik ====================\n")

    pin_ik_solver_relative = partial(solver.ik, tip_name="ur_arm_tool0", link=True)

    target_pose = np.array([[0, 0, 1, 0.7], [0, 1, 0, 0.25], [-1, 0, 0, 0.5], [0, 0, 0, 1]])

    q_pin = pin_ik_solver_relative(
        target_pose, qinit=np.array([-1.29891436, 0.3618943, -1.79107962, -1.71236292, -1.84276683, 1.57065429])
    )
    print("> ik solver: q_pin")
    print(q_pin)
    print()

    q_opti = opti_ik_solver.ik(target_pose, qinit=np.array([0, 0, 0, 0, 0, 0]))
    print("> ik solver: q_opti")
    print(q_opti)
    print()

    pose = solver.forward(q_opti, "ur_arm_tool0", link=True, relative_output=True)
    print("> forward: q_opti")
    print(pose)
    print()
