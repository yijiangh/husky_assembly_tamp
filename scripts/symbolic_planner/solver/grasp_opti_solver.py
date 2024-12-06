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
    control_joint_name_list: List[str],
    q: Union[ca.MX, None] = None,
    output_type: str = "function",
) -> Union[ca.Function, ca.MX]:
    """
    Creates symbolic forward kinematics equations given a URDF file path and a list of joint names.

    Params:
        urdf_path (str): urdf path of robot
        joint_name_list (List[str]): name list of manipulator including all redundant joints
        control_joint_name_list (List[str]): name list of controlled joints
        q (ca.MX | None, None): joint variables
        output_type (str, "function"): "function"/"matrix"

    Returns:
        ca.Function: [q] --> np.ndarray (4x4)
    """
    joints = ParseURDF(urdf_path)
    if q is None or len(control_joint_name_list) == 0:
        q = ca.MX.sym("q", len(control_joint_name_list))
    T = ca.MX.eye(4)
    for i, joint_name in enumerate(joint_name_list):
        joint = joints[joint_name]
        if joint_name in control_joint_name_list:
            i_q = control_joint_name_list.index(joint_name)
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


class BilevelOptimization(object):
    def __init__(
        self,
        upper_obj: ca.MX,
        lower_obj: ca.MX,
        upper_var: ca.MX,
        lower_var: ca.MX,
        lower_constrains,
        lower_solver: ca.Opti,
        upper_var_lower_param: Union[ca.MX, None] = None,
        lower_var_upper_param: Union[ca.MX, None] = None,
        upper_params: Union[List[ca.MX], None] = None,
        lower_params: Union[List[ca.MX], None] = None,
    ) -> None:
        self.upper_obj = upper_obj
        self.lower_obj = lower_obj

        self.upper_var = upper_var
        self.lower_var = lower_var

        self.lower_constrains = lower_constrains
        self.lower_solver = lower_solver

        self.upper_var_lower_param = upper_var_lower_param
        self.lower_var_upper_param = lower_var_upper_param

        self.upper_params = upper_params
        self.lower_params = lower_params

    def eval(self, name: str, obj: ca.MX, sym: List[ca.MX], data: List[np.ndarray]) -> np.ndarray:
        fn = ca.Function("fn", sym, [obj])
        fn_result = fn(*data).toarray()
        print(name, "\n", fn_result)
        return fn_result

    def solve(
        self,
        upper_var_init: np.ndarray,
        lower_var_init: np.ndarray,
        criteria: float,
        upper_params_init: Union[List[np.ndarray], None] = None,
        lower_params_init: Union[List[np.ndarray], None] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        upper_val = upper_var_init
        lower_val = lower_var_init
        last_upper_val = np.ones(upper_var_init.shape) * np.inf
        last_lower_val = np.ones(lower_var_init.shape) * np.inf

        # **************************************************************************
        # Loop
        # **************************************************************************

        while (
            np.linalg.norm(upper_val - last_upper_val) > criteria
            and np.linalg.norm(lower_val - last_lower_val) > criteria
        ):
            # **************************************************************************
            # step 1: solve lower problem
            # **************************************************************************

            # -------------------- set param --------------------#
            if self.lower_params is not None and lower_params_init is not None:
                for lower_param, lower_param_init in zip(self.lower_params, lower_params_init):
                    self.lower_solver.set_value(lower_param, lower_param_init)

            # -------------------- set cross param --------------------#
            if self.upper_var_lower_param is not None:
                self.lower_solver.set_value(self.upper_var_lower_param, upper_var_init)

            # -------------------- set init --------------------#
            self.lower_solver.set_initial(self.lower_var, lower_var_init)
            with HideOutput():
                lower_solution = self.lower_solver.solve()
                lower_var_sol_k = lower_solution.value(self.lower_var)

            # **************************************************************************
            # step 2: compute derivatives of upper obj
            # upper_var --> upper_val
            # lower_var --> lower_var_sol_k
            # upper_var_lower_param --> upper_val
            # lower_var_upper_param --> lower_var_sol_k
            # **************************************************************************

            # **************************************************************************
            # step 2.1: compute first derivative
            # **************************************************************************

            # # TODO: 这里需要去掉，但目前去掉会出现奇异矩阵
            # upper_params_init[1] = np.array([[0, 0, 1, 0.5], [0, 1, 0, 0.25], [-1, 0, 0, 0.5], [0, 0, 0, 1]])
            # lower_params_init[0] = np.array([[0, 0, 1, 0.5], [0, 1, 0, 0.25], [-1, 0, 0, 0.5], [0, 0, 0, 1]])

            pou_pu = ca.jacobian(self.upper_obj, self.upper_var) + ca.jacobian(
                self.upper_obj, self.upper_var_lower_param
            )
            pou_pl = ca.jacobian(self.upper_obj, self.lower_var_upper_param)  # TODO: 这里好像有问题

            pou_pu_res = self.eval(
                "pou_pu",
                pou_pu,
                [self.upper_var, self.lower_var_upper_param] + self.upper_params,
                [upper_val, lower_var_sol_k] + upper_params_init,
            )
            pou_pl_res = self.eval(
                "pou_pl",
                pou_pl,
                [self.upper_var, self.lower_var_upper_param] + self.upper_params,
                [upper_val, lower_var_sol_k] + upper_params_init,
            )

            g = ca.jacobian(self.lower_obj, self.lower_var)
            pg_pl = ca.jacobian(g, self.lower_var)
            pg_pu = ca.jacobian(g, self.upper_var_lower_param)
            S: ca.MX = -ca.inv(pg_pl) @ pg_pu  # dl_du

            S_res = self.eval(
                "S",
                S,
                [self.upper_var, self.lower_var_upper_param, self.lower_var, self.upper_var_lower_param]
                + self.upper_params
                + self.lower_params,
                [upper_val, lower_var_sol_k, lower_var_sol_k, upper_val] + upper_params_init + lower_params_init,
            )

            dou_du: ca.MX = pou_pu + pou_pl @ S

            dou_du_res = self.eval(
                "dou_du",
                dou_du,
                [self.upper_var, self.lower_var_upper_param, self.lower_var, self.upper_var_lower_param]
                + self.upper_params
                + self.lower_params,
                [upper_val, lower_var_sol_k, lower_var_sol_k, upper_val] + upper_params_init + lower_params_init,
            )

            # **************************************************************************
            # step 2.2: compute second derivative
            # **************************************************************************

            # -------------------- part 1 --------------------#
            ppou_pupu = ca.jacobian(pou_pu, self.upper_var)
            ppou_pupl = ca.jacobian(pou_pu, self.lower_var_upper_param)
            part_1 = ppou_pupu + ppou_pupl @ S  # part 1

            ppou_pupu_res = self.eval(
                "ppou_pupu",
                ppou_pupu,
                [self.upper_var, self.lower_var_upper_param, self.lower_var, self.upper_var_lower_param]
                + self.upper_params
                + self.lower_params,
                [upper_val, lower_var_sol_k, lower_var_sol_k, upper_val] + upper_params_init + lower_params_init,
            )

            ppou_pupl_res = self.eval(
                "ppou_pupl",
                ppou_pupl,
                [self.upper_var, self.lower_var_upper_param, self.lower_var, self.upper_var_lower_param]
                + self.upper_params
                + self.lower_params,
                [upper_val, lower_var_sol_k, lower_var_sol_k, upper_val] + upper_params_init + lower_params_init,
            )

            part_1_res = self.eval(
                "part_1",
                part_1,
                [self.upper_var, self.lower_var_upper_param, self.lower_var, self.upper_var_lower_param]
                + self.upper_params
                + self.lower_params,
                [upper_val, lower_var_sol_k, lower_var_sol_k, upper_val] + upper_params_init + lower_params_init,
            )

            # -------------------- part 2 --------------------#

            # -------------------- part 2.1 --------------------#
            ppou_plpl = ca.jacobian(pou_pl, self.lower_var_upper_param)
            ppou_plpu = ca.jacobian(pou_pl, self.upper_var)
            part_2_1 = S.T @ (ppou_plpl @ S + ppou_plpu)

            ppou_plpl_res = self.eval(
                "ppou_plpl",
                ppou_plpl,
                [self.upper_var, self.lower_var_upper_param, self.lower_var, self.upper_var_lower_param]
                + self.upper_params
                + self.lower_params,
                [upper_val, lower_var_sol_k, lower_var_sol_k, upper_val] + upper_params_init + lower_params_init,
            )

            ppou_plpu_res = self.eval(
                "ppou_plpu",
                ppou_plpu,
                [self.upper_var, self.lower_var_upper_param, self.lower_var, self.upper_var_lower_param]
                + self.upper_params
                + self.lower_params,
                [upper_val, lower_var_sol_k, lower_var_sol_k, upper_val] + upper_params_init + lower_params_init,
            )

            part_2_1_res = self.eval(
                "part_2_1",
                part_2_1,
                [self.upper_var, self.lower_var_upper_param, self.lower_var, self.upper_var_lower_param]
                + self.upper_params
                + self.lower_params,
                [upper_val, lower_var_sol_k, lower_var_sol_k, upper_val] + upper_params_init + lower_params_init,
            )

            # -------------------- part 2.2 --------------------#
            n = S.shape[0]
            m = S.shape[1]
            G = None
            for i in range(n):
                block = ca.MX.eye(m) * pou_pl[i]
                if G is None:
                    G = block
                else:
                    G = ca.horzcat(G, block)

            pS_pu = ca.jacobian(S, self.upper_var)
            pS_pl = ca.jacobian(S, self.lower_var)
            dS_du = pS_pu + pS_pl @ S
            part_2_2 = G @ dS_du

            part_2_2_res = self.eval(
                "part_2_2",
                part_2_2,
                [self.upper_var, self.lower_var_upper_param, self.lower_var, self.upper_var_lower_param]
                + self.upper_params
                + self.lower_params,
                [upper_val, lower_var_sol_k, lower_var_sol_k, upper_val] + upper_params_init + lower_params_init,
            )

            ddou_dudu = part_1 + part_2_1 + part_2_2

            ddou_dudu_res = self.eval(
                "ddou_dudu",
                ddou_dudu,
                [self.upper_var, self.lower_var_upper_param, self.lower_var, self.upper_var_lower_param]
                + self.upper_params
                + self.lower_params,
                [upper_val, lower_var_sol_k, lower_var_sol_k, upper_val] + upper_params_init + lower_params_init,
            )

            # **************************************************************************
            # step 3: compute direction of Newton's method
            # **************************************************************************

            d = -ca.inv(ddou_dudu) @ dou_du.T
            d_res = self.eval(
                "d",
                d,
                [self.upper_var, self.lower_var_upper_param, self.lower_var, self.upper_var_lower_param]
                + self.upper_params
                + self.lower_params,
                [upper_val, lower_var_sol_k, lower_var_sol_k, upper_val] + upper_params_init + lower_params_init,
            )

            # **************************************************************************
            # step 4: backtracking search
            # **************************************************************************

            # **************************************************************************
            # step 5: iteration
            # **************************************************************************
            break


class GraspOptiSolver(object):

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
        "tool0-bar_tcp_fixed_joint"
    ]

    BASE_CONTROL_JOINT_NAMES = []

    BASE_REDUCED_MODEL_JOINT_NAMES = [
        "base_footprint_joint",
        "top_plate_joint",
        "top_plate_front_joint",
        "arm_mount_joint",
        # "ur_arm_base_link-base_fixed_joint",
    ]

    def __init__(self, urdf_path: str) -> None:
        self.urdf_path = urdf_path

        self.SolverInit()

    def eval(
        self, name: str, obj: ca.MX, sym: List[ca.MX], data: List[np.ndarray], verbose: bool = False
    ) -> np.ndarray:
        fn = ca.Function("fn", sym, [obj])
        if sym != []:
            fn_result = fn(*data).toarray()
        else:
            fn_result = fn()["o0"].toarray()
        if verbose:
            print(name, "\n", fn_result)
        return fn_result

    def GetInvertT(self, T: ca.MX) -> ca.MX:
        T_rot: ca.MX = T[:3, :3]
        T_trans: ca.MX = T[:3, 3]
        T_inv = ca.vertcat(
            ca.horzcat(
                T_rot.T,
                -T_rot.T @ T_trans,
            ),
            ca.MX([0, 0, 0, 1]).T,
        )
        return T_inv

    def IKObjectiveFunction(self, T: ca.MX, T_tar: ca.MX) -> ca.MX:
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

    def GraspIKObjectiveFunction(self, q: ca.MX, p: ca.MX, base: ca.MX, world_from_element: ca.MX) -> ca.MX:

        # **************************************************************************
        # compute pose of gripper
        # **************************************************************************

        # world_from_base
        world_from_base: ca.MX = ca.MX.eye(4)
        world_from_base[0, 3] = base[0]
        world_from_base[1, 3] = base[1]
        yaw = base[2]
        R_z = ca.vertcat(
            ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0),
            ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0),
            ca.horzcat(0, 0, 1)
        )
        world_from_base[:3, :3] = R_z

        # connect_from_gripper
        connect_from_gripper = SymbolicForward(
            self.urdf_path,
            self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES,
            self.MANIPULATOR_CONTROL_JOINT_NAMES,
            q=q,
            output_type="matrix",
        )

        T_gripper = world_from_base @ self.base_from_connect @ connect_from_gripper

        # **************************************************************************
        # compute pose of gripper, target
        # **************************************************************************

        # element_from_gripper
        element_from_gripper = ca.MX.eye(4)
        element_from_gripper[0, 3] = p[0]
        element_from_gripper[1, 3] = p[1]

        pitch = p[2]
        R_y = ca.vertcat(
            ca.horzcat(ca.cos(pitch), 0, ca.sin(pitch)),
            ca.horzcat(0, 1, 0),
            ca.horzcat(-ca.sin(pitch), 0, ca.cos(pitch))
        )
        element_from_gripper[:3, :3] = R_y

        # world_from_gripper, T_tar
        T_gripper_tar = world_from_element @ element_from_gripper

        obj = self.IKObjectiveFunction(T_gripper, T_gripper_tar)
        return obj

    def GraspObjectiveFunction(self, p: ca.MX, c: ca.MX) -> ca.MX:
        x = p[0]
        t = x - c
        obj = ca.if_else(t > 0, t**2, 0)
        return obj

    def GroundParallelObjectiveFunction(self, world_from_base) -> ca.MX:
        err = 0
        T_tar = ca.MX.eye(4)
        for i in range(3):
            for j in range(3):
                err += (world_from_base[i, j] - T_tar[i, j]) ** 2
        return err

    def SolverInit(self):

        # **************************************************************************
        # params init
        # **************************************************************************

        self.Nq = len(self.MANIPULATOR_CONTROL_JOINT_NAMES)
        self.Np = 3
        self.Nb = 3

        self.connect_from_gripper_fn = SymbolicForward(
            self.urdf_path, self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES, self.MANIPULATOR_CONTROL_JOINT_NAMES
        )

        base_from_connect_sym = SymbolicForward(
            self.urdf_path, self.BASE_REDUCED_MODEL_JOINT_NAMES, self.BASE_CONTROL_JOINT_NAMES, output_type="matrix"
        )
        self.base_from_connect = self.eval("base_from_connect", base_from_connect_sym, [], [])

        # **************************************************************************
        # solver for IK
        # **************************************************************************

        # self.ik_solver = ca.Opti()

        # # -------------------- set parameters --------------------#
        # self.ik_var_q = self.ik_solver.variable(self.Nq)
        # self.ik_param_T_tar = self.ik_solver.parameter(4, 4)  # world(arm base)_from_target

        # # -------------------- get matrix --------------------#
        # self.ik_connect_from_gripper = SymbolicForward(
        #     self.urdf_path,
        #     self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES,
        #     self.MANIPULATOR_CONTROL_JOINT_NAMES,
        #     q=self.ik_var_q,
        #     output_type="matrix",
        # )  # world(arm base)_from_gripper

        # # -------------------- objective function --------------------#
        # self.ik_obj = self.IKObjectiveFunction(self.ik_connect_from_gripper, self.ik_param_T_tar)

        # # -------------------- constrains, geq 0 --------------------#
        # var_q_lb = np.pi + ca.vec(self.ik_var_q)
        # var_q_ub = np.pi - ca.vec(self.ik_var_q)

        # # -------------------- set solver objective function and constrains --------------------#
        # self.ik_solver.minimize(self.ik_obj)
        # self.ik_solver.subject_to(var_q_lb >= 0)
        # self.ik_solver.subject_to(var_q_ub >= 0)

        # # -------------------- set solver options --------------------#
        # p_opts = {"expand": True, "print_time": 0}
        # s_opts = {"max_iter": 100, "print_level": 0}
        # self.ik_solver.solver("ipopt", p_opts, s_opts)

        # **************************************************************************
        # solver for grasp
        # **************************************************************************

        self.grasp_solver = ca.Opti()

        # -------------------- set variables --------------------#
        self.grasp_var_q = self.grasp_solver.variable(self.Nq)
        self.grasp_var_p = self.grasp_solver.variable(self.Np)
        self.grasp_var_base = self.grasp_solver.variable(self.Nb)

        # -------------------- set parameters --------------------#
        self.grasp_param_T_element = self.grasp_solver.parameter(4, 4)  # world_from_element
        self.grasp_param_c = self.grasp_solver.parameter(1)

        # -------------------- get matrix --------------------#
        self.grasp_connect_from_gripper = SymbolicForward(
            self.urdf_path,
            self.MANIPULATOR_REDUCED_MODEL_JOINT_NAMES,
            self.MANIPULATOR_CONTROL_JOINT_NAMES,
            q=self.grasp_var_q,
            output_type="matrix",
        )
        self.grasp_gripper_from_connect = self.GetInvertT(self.grasp_connect_from_gripper)

        # -------------------- ik obj: grasp_var_q, grasp_var_p, grasp_var_base, grasp_param_T_element --------------------#
        ik_obj = self.GraspIKObjectiveFunction(self.grasp_var_q, self.grasp_var_p, self.grasp_var_base, self.grasp_param_T_element)

        # -------------------- grasp obj: grasp_var_p, grasp_param_c --------------------#
        grasp_obj = self.GraspObjectiveFunction(self.grasp_var_p, self.grasp_param_c)

        # -------------------- parallel obj: grasp_var_p, grasp_var_q, grasp_param_c, grasp_param_T_element --------------------#
        # element_from_gripper = ca.MX.eye(4)
        # element_from_gripper[0, 3] = self.grasp_var_p[0]
        # element_from_gripper[1, 3] = self.grasp_var_p[1]
        # world_from_gripper = self.grasp_param_T_element @ element_from_gripper
        # world_from_connect = world_from_gripper @ self.grasp_gripper_from_connect
        # world_from_base = world_from_connect @ np.linalg.inv(self.base_from_connect)
        # parallel_obj = self.GroundParallelObjectiveFunction(world_from_base)

        # -------------------- set obj --------------------#
        self.grasp_obj = ik_obj + grasp_obj

        # -------------------- constrains ≥0 --------------------#

        # 关节角约束
        var_q_lb = np.pi + ca.vec(self.grasp_var_q)
        var_q_ub = np.pi - ca.vec(self.grasp_var_q)

        # p := (x, y, theta) 的约束
        var_p_x_lb = 0.001 + self.grasp_var_p[0]
        var_p_x_ub = 0.001 - self.grasp_var_p[0]
        var_p_y_lb = self.grasp_param_c + self.grasp_var_p[1]
        var_p_y_ub = self.grasp_param_c - self.grasp_var_p[1]
        var_p_theta_lb = np.pi + self.grasp_var_p[2]
        var_p_theta_ub = np.pi - self.grasp_var_p[2]

        # base := (x, y, yaw) 的约束
        var_base_yaw_lb = np.pi + self.grasp_var_base[2]
        var_base_yaw_ub = np.pi - self.grasp_var_base[2]

        # TODO: 底盘平行constrain
        # element_from_gripper = ca.MX.eye(4)
        # element_from_gripper[0, 3] = self.grasp_var_p[0]
        # element_from_gripper[1, 3] = self.grasp_var_p[1]
        # world_from_gripper = self.grasp_param_T_element @ element_from_gripper
        # world_from_connect = world_from_gripper @ self.grasp_gripper_from_connect
        # world_from_base = world_from_connect @ np.linalg.inv(self.base_from_connect)

        # world_from_base_z_axis = world_from_base[:3, 2]
        # world_from_base_z = world_from_base[2, 3]

        # base_rot_lb = -ca.MX([0, 0, 1]) * 0.9999 + world_from_base_z_axis
        # base_rot_ub = ca.MX([0, 0, 1]) * 1.0001 - world_from_base_z_axis
        # base_z_lb = 0.001 + world_from_base_z
        # base_z_ub = 0.001 - world_from_base_z

        # TODO: 底盘在外面的constrain
        # TODO: collision constrain

        # -------------------- set solver objective function and constrains --------------------#
        self.grasp_solver.minimize(self.grasp_obj)

        self.grasp_solver.subject_to(var_q_lb >= 0)
        self.grasp_solver.subject_to(var_q_ub >= 0)

        self.grasp_solver.subject_to(var_p_x_lb >= 0)
        self.grasp_solver.subject_to(var_p_x_ub >= 0)
        self.grasp_solver.subject_to(var_p_y_lb >= 0)
        self.grasp_solver.subject_to(var_p_y_ub >= 0)
        self.grasp_solver.subject_to(var_p_theta_lb >= 0)
        self.grasp_solver.subject_to(var_p_theta_ub >= 0)

        self.grasp_solver.subject_to(var_base_yaw_lb >= 0)
        self.grasp_solver.subject_to(var_base_yaw_ub >= 0)

        # self.grasp_solver.subject_to(base_rot_lb >= 0)
        # self.grasp_solver.subject_to(base_rot_ub >= 0)
        # self.grasp_solver.subject_to(base_z_lb >= 0)
        # self.grasp_solver.subject_to(base_z_ub >= 0)

        # -------------------- set solver options --------------------#
        p_opts = {"expand": True, "print_time": 0}
        s_opts = {"max_iter": 10000, "print_level": 0}
        # s_opts = {"max_iter": 10000}
        self.grasp_solver.solver("ipopt", p_opts, s_opts)

    def forward(self, q: Union[np.ndarray, List[float]]) -> np.ndarray:

        if isinstance(q, np.ndarray):
            q = q.tolist()

        T = np.array(self.connect_from_gripper_fn(q))

        return T

    def ik(
        self,
        pose: np.ndarray,
        q_init: Union[np.ndarray, None] = None,
        output: str = "array",
        output_flag: bool = False,
    ) -> Union[Union[np.ndarray, List[float], None], Tuple[Union[np.ndarray, List[float], None], bool]]:
        """
        Calculate IK solution.

        Params:
            pose (np.ndarray, 4x4): SE(3) matrix
            q_init (np.ndarray, None): initial guess of arm joint
            output (str, "array"): array/list, output result type
            output_flag (bool, False, [not used]): whether output includes solve status

        Returns:
            q (np.ndarray | [float] | None): joint conf to the target pose
            success (bool, [optional]): solve status
        """

        # self.ik_solver.set_value(self.ik_param_T_tar, pose)

        # self.ik_solver.set_initial(self.ik_var_q, qinit)
        # with HideOutput():
        #     solution = np.array(self.ik_solver.solve().value(self.ik_var_q))

        # **************************************************************************
        # grasp
        # **************************************************************************

        p_init = np.array([0, 0, 0])
        base_init = np.array([0, 0, 0])

        self.grasp_solver.set_value(self.grasp_param_c, 0.5)
        self.grasp_solver.set_value(self.grasp_param_T_element, pose)

        self.grasp_solver.set_initial(self.grasp_var_q, q_init)
        self.grasp_solver.set_initial(self.grasp_var_p, p_init)
        self.grasp_solver.set_initial(self.grasp_var_base, base_init)

        grasp_solution = self.grasp_solver.solve()

        # **************************************************************************
        # verbose
        # **************************************************************************

        # print("q solution\n", grasp_solution.value(self.grasp_var_q))
        # print("p solution\n", grasp_solution.value(self.grasp_var_p))

        # element_from_gripper = ca.MX.eye(4)
        # element_from_gripper[0, 3] = self.grasp_var_p[0]
        # element_from_gripper[1, 3] = self.grasp_var_p[1]
        # world_from_gripper = self.grasp_param_T_element @ element_from_gripper
        # world_from_connect = world_from_gripper @ self.grasp_gripper_from_connect
        # world_from_base = world_from_connect @ np.linalg.inv(self.base_from_connect)

        # self.eval(
        #     "element_from_gripper",
        #     element_from_gripper,
        #     [self.grasp_var_p, self.grasp_var_q, self.grasp_param_T_element],
        #     [grasp_solution.value(self.grasp_var_p), grasp_solution.value(self.grasp_var_q), pose],
        #     verbose=True,
        # )

        return grasp_solution.value(self.grasp_var_q), grasp_solution.value(self.grasp_var_base)


if __name__ == "__main__":
    import os
    import sys
    from functools import partial

    import casadi as ca
    import numpy as np
    import pybullet as p
    import pybullet_planning as pp
    import time

    HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(HERE)

    import utils.load_multi_tangent as load_multi_tangent
    from eth.husky_assembly.scripts.symbolic_planner.solver.grasp_opti_solver import GraspOptiSolver
    from eth.husky_assembly.scripts.symbolic_planner.solver.ik_pinocchio_solver import PinocchioSolver
    from multi_tangent.collision import create_collision_bodies
    from robot.robot_setup import RobotSetup
    from utils.collision import init_pb

    np.set_printoptions(precision=3, suppress=True)

    urdf_path = (
        "/home/jeong/summer_research/eth/husky_assembly/data/husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf"
    )
    solver = PinocchioSolver(urdf_path)
    opti_ik_solver = GraspOptiSolver(urdf_path)

    # **************************************************************************
    # link forward
    # **************************************************************************

    # print("\n==================== link forward ====================\n")

    # pin_ik_solver_relative = partial(solver.ik, tip_name="ur_arm_tool0", link=True)

    # target_pose = np.array([[0, 0, 1, 0.7], [0, 1, 0, 0.25], [-1, 0, 0, 0.5], [0, 0, 0, 1]])

    # q_pin = pin_ik_solver_relative(
    #     target_pose, qinit=np.array([-1.29891436, 0.3618943, -1.79107962, -1.71236292, -1.84276683, 1.57065429])
    # )

    # print("> ik solver: q_pin")
    # print(q_pin)
    # print()

    # print("> forward: pose real")
    # print(target_pose)
    # print()

    # pose = solver.forward(q_pin, "ur_arm_tool0", link=True, relative_output=True)
    # print("> forward: pose pinocchio")
    # print(pose)
    # print()

    # pose = opti_ik_solver.forward(q_pin)
    # print("> forward: pose opti")
    # print(pose)
    # print()

    # **************************************************************************
    # joint forward
    # **************************************************************************

    # print("\n==================== joint forward ====================\n")

    # pin_ik_solver_relative = partial(solver.ik, tip_name="ur_arm_wrist_3_joint", link=False)

    # base_from_tip = np.array([[0, 0, 1, 0.5], [0, 1, 0, 0.25], [-1, 0, 0, 0.5], [0, 0, 0, 1]])

    # q_pin = pin_ik_solver_relative(base_from_tip, qinit=np.array([0, 0, 0, 0, 0, 0]))

    # print("> ik solver: q_pin")
    # print(q_pin)
    # print()

    # print("> forward: pose real")
    # print(base_from_tip)
    # print()

    # pose = solver.forward(q_pin, "ur_arm_wrist_3_joint", link=False, relative_output=True)
    # print("> forward: pose pinocchio")
    # print(pose)
    # print()

    # pose = opti_ik_solver.forward(q_pin)
    # print("> forward: pose opti")
    # print(pose)
    # print()

    # **************************************************************************
    # link ik
    # **************************************************************************

    # print("\n==================== link ik ====================\n")

    # q_zero = np.array([0, 0, 0, 0, 0, 0])

    # pin_ik_solver_relative = partial(solver.ik, tip_name="ur_arm_tool0", link=True)

    # target_pose = np.array([[0, 0, 1, 0.7], [0, 1, 0, 0.25], [-1, 0, 0, 0.5], [0, 0, 0, 1]])

    # q_pin = pin_ik_solver_relative(target_pose, qinit=q_zero)
    # print("> ik solver: q_pin")
    # print(q_pin)
    # print()

    # q_opti = opti_ik_solver.ik(target_pose, qinit=q_zero)
    # print("> ik solver: q_opti")
    # print(q_opti)
    # print()

    # pose = solver.forward(q_opti, "ur_arm_tool0", link=True, relative_output=True)
    # print("> forward: q_opti")
    # print(pose)
    # print()

    # **************************************************************************
    # visualization
    # **************************************************************************

    # -------------------- init --------------------#
    init_pb()
    rb = RobotSetup("r0")
    line_pts_flattened = [np.array([-0.25, 0.5, 1]), np.array([0.75, 0.5, 1])]
    radius_per_edge = [0.01]
    element_body = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)[0]
    q_zero = np.array([0, 0, 0, 0, 0, 0])

    # -------------------- init slider --------------------#
    x_slider = p.addUserDebugParameter(f"x", 0.0, 1.0, 0.25)
    y_slider = p.addUserDebugParameter(f"y", -1.0, 1.0, 0.5)
    z_slider = p.addUserDebugParameter(f"z", 0.0, 2.0, 1.0)

    # -------------------- loop --------------------#
    while True:
        x_value = p.readUserDebugParameter(x_slider)
        y_value = p.readUserDebugParameter(y_slider)
        z_value = p.readUserDebugParameter(z_slider)

        pp.set_point(element_body, [x_value, y_value, z_value])

        element_pose = pp.multiply(pp.multiply(
            pp.get_pose(element_body), pp.Pose(point=[0, 0, 0], euler=pp.Euler(1.5708, 1.5708, 1.5708))
        ), pp.Pose(point=[0, 0, 0], euler=pp.Euler(0, 0, 1.5708)))

        element_plot_handle = pp.draw_pose(element_pose, length=0.2, lifetime=1.0)
        ee_plot_handle = pp.draw_pose(pp.get_link_pose(rb.robot, pp.link_from_name(rb.robot, "bar_tcp")), length=0.2, lifetime=1.0)

        element_pose_relative = rb.get_relative_pose(element_pose)
        connect_from_element = pp.pose_transformation.tform_from_pose(element_pose_relative)

        world_from_element = pp.pose_transformation.tform_from_pose(element_pose)

        q_opti, base_opti = opti_ik_solver.ik(world_from_element, q_init=q_zero)
        rb.set_joint_positions(rb.arm_joints, q_opti)
        rb.set_joint_positions(rb.base_joints, base_opti)

        time.sleep(0.05)
