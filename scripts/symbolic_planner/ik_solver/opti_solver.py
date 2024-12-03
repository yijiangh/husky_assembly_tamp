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

    def LowerObjectiveFunction(self, q: ca.MX, p: ca.MX, T_element: ca.MX) -> ca.MX:
        # base_from_gripper, T
        T_gripper = SymbolicForward(
            self.urdf_path, self.REDUCED_MODEL_JOINT_NAMES, self.MANIPULATOR_JOINT_NAMES, q=q, output_type="matrix"
        )

        # element_from_gripper
        T_delta = ca.MX.eye(4)
        T_delta[0, 3] = p[0]
        T_delta[1, 3] = p[1]

        # base_from_gripper, T_tar
        T_gripper_tar = T_element @ T_delta

        obj = self.IKObjectiveFunction(T_gripper, T_gripper_tar)
        return obj

    def UpperObjectiveFunction(self, p: ca.MX, c: ca.MX) -> ca.MX:
        x = p[0]
        t = x - c
        obj = ca.if_else(t > 0, t**2, 0)
        return obj

    def SolverInit(self):

        # **************************************************************************
        # solver for IK, lower solver
        # **************************************************************************

        self.ik_solver = ca.Opti()

        # -------------------- set parameters --------------------#
        self.Nq = len(self.MANIPULATOR_JOINT_NAMES)
        self.q = self.ik_solver.variable(self.Nq)
        self.T_tar = self.ik_solver.parameter(4, 4)  # world(arm base)_from_element

        # -------------------- get matrix and objective function --------------------#
        self.fk_function = SymbolicForward(self.urdf_path, self.REDUCED_MODEL_JOINT_NAMES, self.MANIPULATOR_JOINT_NAMES)
        self.fk_matrix = SymbolicForward(
            self.urdf_path, self.REDUCED_MODEL_JOINT_NAMES, self.MANIPULATOR_JOINT_NAMES, q=self.q, output_type="matrix"
        )  # world(arm base)_from_gripper
        self.ik_obj = self.IKObjectiveFunction(self.fk_matrix, self.T_tar)

        # -------------------- set solver objective function and constrains --------------------#
        self.ik_solver.minimize(self.ik_obj)
        self.ik_solver.subject_to(ca.vec(self.q) <= np.pi)
        self.ik_solver.subject_to(ca.vec(self.q) >= -np.pi)

        # -------------------- set solver options --------------------#
        p_opts = {"expand": True, "print_time": 0}
        s_opts = {"max_iter": 100, "print_level": 0}
        self.ik_solver.solver("ipopt", p_opts, s_opts)

        # **************************************************************************
        # solver for Bi-level optimization:
        #   O_{U} --> O_{grasp}
        #   O_{L} --> IK objective function
        # **************************************************************************

        self.lower_solver = ca.Opti()
        self.upper_solver = ca.Opti()

        # -------------------- set variables --------------------#
        self.lower_var_q = self.lower_solver.variable(self.Nq)  # q
        self.upper_var_p = self.upper_solver.variable(2)  # p: (x, y, theta)

        # -------------------- set cross parameters --------------------#
        self.lower_var_upper_param_q = self.upper_solver.parameter(self.Nq)
        self.upper_var_lower_param_p = self.lower_solver.parameter(2)  # p: (x, y, theta)

        # -------------------- set parameters --------------------#
        self.lower_param_T_element = self.lower_solver.parameter(4, 4)

        self.upper_param_c = self.upper_solver.parameter(1)
        self.upper_param_T_element = self.upper_solver.parameter(4, 4)

        # -------------------- set objective functions --------------------#
        self.lower_obj = self.LowerObjectiveFunction(
            self.lower_var_q, self.upper_var_lower_param_p, self.lower_param_T_element
        )
        # self.upper_obj = self.UpperObjectiveFunction(
        #     self.upper_var_p, self.upper_param_c
        # ) + self.LowerObjectiveFunction(self.lower_var_upper_param_q, self.upper_var_p, self.upper_param_T_element)
        # self.upper_obj = self.UpperObjectiveFunction(self.upper_var_p, self.upper_param_c)
        self.upper_obj = self.LowerObjectiveFunction(
            self.lower_var_upper_param_q, self.upper_var_p, self.upper_param_T_element
        )

        self.lower_solver.minimize(self.lower_obj)
        self.upper_solver.minimize(self.upper_obj)

        self.lower_solver.subject_to(ca.vec(self.lower_var_q) <= np.pi)
        self.lower_solver.subject_to(ca.vec(self.lower_var_q) >= -np.pi)

        # -------------------- set solver options --------------------#
        p_opts = {"expand": True, "print_time": 0}
        s_opts = {"max_iter": 100, "print_level": 0}
        self.lower_solver.solver("ipopt", p_opts, s_opts)

        self.bilevel_solver = BilevelOptimization(
            self.upper_obj,
            self.lower_obj,
            self.upper_var_p,
            self.lower_var_q,
            None,
            self.lower_solver,
            upper_var_lower_param=self.upper_var_lower_param_p,
            lower_var_upper_param=self.lower_var_upper_param_q,
            upper_params=[self.upper_param_c, self.upper_param_T_element],
            lower_params=[self.lower_param_T_element],
        )

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

        self.ik_solver.set_value(self.T_tar, pose)
        self.ik_solver.set_initial(self.q, qinit)
        with HideOutput():
            solution = self.ik_solver.solve()

        pinit = np.array([1.0, 0])

        self.bilevel_solver.solve(pinit, qinit, 0.1, upper_params_init=[0.5, pose], lower_params_init=[pose])
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
        base_from_tip, qinit=np.array([0, 0, 0, 0, 0, 0])
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

    q_opti = opti_ik_solver.ik(target_pose, qinit=q_pin)
    print("> ik solver: q_opti")
    print(q_opti)
    print()

    pose = solver.forward(q_opti, "ur_arm_tool0", link=True, relative_output=True)
    print("> forward: q_opti")
    print(pose)
    print()
