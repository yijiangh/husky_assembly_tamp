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
import matplotlib.pyplot as plt

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
from robot.robot_setup import RobotSetup, HUSKY_URDF_PATH, HUSKY_ARM_JOINT_NAMES, HUSKY_CONTROL_JOINT_NAMES, HUSKY_TOOL0_NAME, HUSKY_DUAL_TOOL0_LEFT, HUSKY_DUAL_TOOL0_RIGHT, HUSKY_DUAL_ARM_JOINT_NAMES
from ConstrainedPlanningCommon import *
from utils.util import interpolate


class RelativeEndEffectorConstraint(ob.Constraint):
    """Constraint that keeps the relative translation between the two tool0 frames of the dual-arm Husky fixed.

    The constraint enforces that the vector connecting the left and right end-effectors stays
    identical to the one measured when the class is instantiated.
    """

    def __init__(self, num_joints: int, robot_setup: RobotSetup):
        # 3 positional constraints (dx, dy, dz)
        super(RelativeEndEffectorConstraint, self).__init__(num_joints, 3)

        self.robot_setup = robot_setup
        self.robot_id = robot_setup.robot
        self.num_joints = num_joints

        # Identify link indices for both arms
        self.left_tool_link = pp.link_from_name(self.robot_id, HUSKY_DUAL_TOOL0_LEFT)
        self.right_tool_link = pp.link_from_name(self.robot_id, HUSKY_DUAL_TOOL0_RIGHT)

        # Joint indices that the constraint acts on (all manipulator joints)
        self.control_joints = [pp.joint_from_name(self.robot_id, j) for j in HUSKY_DUAL_ARM_JOINT_NAMES]

        self.collision_fn = robot_setup.create_collision_fn()

        # Record the desired relative translation (left - right) at instantiation
        left_pose = pp.get_link_pose(self.robot_id, self.left_tool_link)[0]
        right_pose = pp.get_link_pose(self.robot_id, self.right_tool_link)[0]
        self.desired_delta = np.array(left_pose) - np.array(right_pose)

    # ------------------------------------------------------------------
    #   Core OMPL callbacks
    # ------------------------------------------------------------------
    def function(self, x, out):
        """Compute constraint residuals given joint configuration *x*.

        out[i] = f_i(x) where f(x) = (current_delta - desired_delta).
        """
        current_delta = self._relative_translation(x)
        diff = current_delta - self.desired_delta
        out[0], out[1], out[2] = diff  # dx, dy, dz

    def jacobian(self, x, out):
        """Finite-difference Jacobian of the constraint."""
        epsilon = 1e-6
        base_delta = self._relative_translation(x)

        # Initialise jacobian with zeros (codim x ambient)
        out[:, :] = np.zeros((self.getCoDimension(), self.getAmbientDimension()))

        for i in range(self.num_joints):
            x_plus = np.array(x)
            x_plus[i] += epsilon
            delta_plus = self._relative_translation(x_plus)
            deriv = (delta_plus - base_delta) / epsilon  # 3-vector
            out[0, i], out[1, i], out[2, i] = deriv

    # ------------------------------------------------------------------
    #   Helpers
    # ------------------------------------------------------------------
    def _relative_translation(self, joint_angles: np.ndarray) -> np.ndarray:
        """Return translation vector (left - right) for given joint angles."""
        # Backup current joint state
        current_conf = pp.get_joint_positions(self.robot_id, self.control_joints)
        try:
            self.robot_setup.set_joint_positions(self.control_joints, joint_angles)
            left_pos = pp.get_link_pose(self.robot_id, self.left_tool_link)[0]
            right_pos = pp.get_link_pose(self.robot_id, self.right_tool_link)[0]
            return np.array(left_pos) - np.array(right_pos)
        finally:
            self.robot_setup.set_joint_positions(self.control_joints, current_conf)

    def compute_violation(self, joint_angles: np.ndarray) -> float:
        """Compute constraint violation magnitude for given joint configuration."""
        current_delta = self._relative_translation(joint_angles)
        diff = current_delta - self.desired_delta
        return np.linalg.norm(diff)

    # ------------------------------------------------------------------
    #   Optional helpers for planning convenience
    # ------------------------------------------------------------------
    def isValid(self, state):
        """Basic state validity that checks joint limits."""
        j = np.array([state[i] for i in range(self.getAmbientDimension())])
        # Simple bound check: assume revolute joints within ±2π
        if np.any(j < -2 * np.pi) or np.any(j > 2 * np.pi):
            return False
        if self.collision_fn(j):
            return False
        return True

    def createSpace(self):
        """Create OMPL RealVector state space using joint limits from PyBullet."""
        space = ob.RealVectorStateSpace(self.num_joints)
        bounds = ob.RealVectorBounds(self.num_joints)
        for i, joint_index in enumerate(self.control_joints):
            info = pp.get_joint_info(self.robot_id, joint_index)
            lo, hi = info.jointLowerLimit, info.jointUpperLimit
            if lo == 0 and hi == -1:  # continuous joint
                lo, hi = -2 * np.pi, 2 * np.pi
            bounds.setLow(i, lo)
            bounds.setHigh(i, hi)
        space.setBounds(bounds)
        return space

    def dump(self, outfile):
        print("RelativeEndEffectorConstraint", file=outfile)
        print(f"Desired delta: {self.desired_delta}", file=outfile)

    def getProjection(self, space):
        """Return a ProjectionEvaluator mapping state -> relative translation (3-D)."""

        class RelProjection(ob.ProjectionEvaluator):
            def __init__(self, space, constraint):
                super(RelProjection, self).__init__(space)
                self.constraint = constraint
                self.defaultCellSizes()

            def getDimension(self):
                return 3

            def defaultCellSizes(self):
                # Equal cell sizes for each coordinate
                self.cellSizes_ = list2vec([0.05, 0.05, 0.05])

            def project(self, state, projection):
                joint_angles = [state[i] for i in range(self.constraint.num_joints)]
                delta = self.constraint._relative_translation(joint_angles)
                projection[0], projection[1], projection[2] = delta

        return RelProjection(space, self)


# ------------------------------------------------------------------------------------
#   Planning helpers (analogous to single arm script)
# ------------------------------------------------------------------------------------


def compute_and_plot_constraint_violations(trajectory: np.ndarray, constraint: RelativeEndEffectorConstraint, output_dir: str = "./") -> np.ndarray:
    """
    计算轨迹中每个点的约束违反度，绘制曲线并保存图片。

    Args:
        trajectory: 轨迹数组，形状为 (n_points, n_joints)
        constraint: 约束对象
        output_dir: 输出目录

    Returns:
        violations: 每个点的违反度数组
    """
    print("Computing constraint violations for trajectory...")

    n_points = trajectory.shape[0]
    violations = np.zeros(n_points)

    # 计算每个点的违反度
    for i in range(n_points):
        violations[i] = constraint.compute_violation(trajectory[i])
        if (i + 1) % 50 == 0 or i == n_points - 1:
            print(f"  Progress: {i + 1}/{n_points} points processed")

    # 统计信息
    max_violation = np.max(violations)
    mean_violation = np.mean(violations)
    final_violation = violations[-1]

    print(f"\nConstraint Violation Statistics:")
    print(f"  Max violation: {max_violation:.6f} m")
    print(f"  Mean violation: {mean_violation:.6f} m")
    print(f"  Final violation: {final_violation:.6f} m")

    # 绘制违反度曲线
    plt.figure(figsize=(12, 8))

    # 主图：违反度随时间变化
    plt.subplot(2, 1, 1)
    time_steps = np.arange(n_points)
    plt.plot(time_steps, violations, "b-", linewidth=2, label="Constraint Violation")
    plt.axhline(y=mean_violation, color="r", linestyle="--", alpha=0.7, label=f"Mean: {mean_violation:.6f} m")
    plt.xlabel("Trajectory Point Index")
    plt.ylabel("Violation Magnitude (m)")
    plt.title("Dual-Arm Relative Position Constraint Violation")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 子图：违反度的对数刻度（如果有很小的值）
    plt.subplot(2, 1, 2)
    plt.semilogy(time_steps, violations + 1e-10, "g-", linewidth=2, label="Constraint Violation (log scale)")
    plt.xlabel("Trajectory Point Index")
    plt.ylabel("Violation Magnitude (m, log scale)")
    plt.title("Constraint Violation (Logarithmic Scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # 保存图片
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"dual_constraint_violations_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"✓ Constraint violation plot saved to: {filepath}")

    # 同时保存违反度数据
    data_filename = f"dual_constraint_violations_data_{timestamp}.txt"
    data_filepath = os.path.join(output_dir, data_filename)

    # 保存数据：第一列是时间步，第二列是违反度
    violation_data = np.column_stack([time_steps, violations])
    np.savetxt(data_filepath, violation_data, fmt="%.8f", header="TimeStep ConstraintViolation(m)")
    print(f"✓ Constraint violation data saved to: {data_filepath}")

    plt.show()

    return violations


def relativeConstraintPlanningOnce(cp, planner, output=False, interpolate_points=50):
    """Solve once and return interpolated trajectory (numpy array) or None."""
    cp.setPlanner(planner, "relative")

    stat = cp.solveOnce(output, "relative")
    if not stat:
        print("✗ Planning failed.")
        return None

    path = cp.ss.getSolutionPath()
    if not path:
        print("✗ No path object returned.")
        return None

    print(f"✓ Found path with {path.getStateCount()} states, length {path.length():.4f}")

    # Convert OMPL path to numpy trajectory (states x dof)
    trajectory = []
    state_dim = cp.css.getDimension()
    for i in range(path.getStateCount()):
        st = path.getState(i)
        trajectory.append([st[j] for j in range(state_dim)])

    arr = np.array(trajectory)
    if interpolate_points and arr.shape[0] > 1:
        arr_interp = interpolate(arr, interpolate_points)
        print(f"✓ Interpolated to {arr_interp.shape[0]} points.")
        return arr_interp
    return arr


def relativeConstraintPlanning(robot_setup: RobotSetup, start_conf: np.ndarray, goal_conf: np.ndarray, options):
    num_joints = len(robot_setup.arm_joints)
    constraint = RelativeEndEffectorConstraint(num_joints, robot_setup)

    cp = ConstrainedProblem(options.space, constraint.createSpace(), constraint, options)

    # Register projection evaluator for PJ/TB/AT spaces
    cp.css.registerProjection("relative", constraint.getProjection(cp.css))

    # Build OMPL state wrappers
    sstart = ob.State(cp.css)
    sgoal = ob.State(cp.css)
    for i in range(num_joints):
        sstart[i] = start_conf[i]
        sgoal[i] = goal_conf[i]
    cp.setStartAndGoalStates(sstart, sgoal)

    # Basic validity checker from constraint
    cp.ss.setStateValidityChecker(ob.StateValidityCheckerFn(partial(RelativeEndEffectorConstraint.isValid, constraint)))

    planners = options.planner.split(",")
    if not options.bench:
        interp_pts = getattr(options, "interpolate_points", 50)
        result = relativeConstraintPlanningOnce(cp, planners[0], options.output, interp_pts)
        return result, constraint
    else:
        cp.setupBenchmark(planners, "relative")
        cp.constraint.addBenchmarkParameters(cp.bench)
        cp.runBenchmark()
        return None, constraint


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # CLI Parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", action="store_true", help="Dump found solution path and planning graph.")
    parser.add_argument("--bench", action="store_true", help="Run benchmark instead of single planning run.")
    parser.add_argument("--interpolate-points", type=int, default=300, help="Number of points to interpolate the trajectory to (default 300)")
    parser.add_argument("--plot-violations", action="store_true", help="Compute and plot constraint violations along the trajectory")

    addSpaceOption(parser)
    addPlannerOption(parser)
    addConstrainedOptions(parser)
    addAtlasOptions(parser)

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Environment & Robot Setup
    # ------------------------------------------------------------------
    print("Initializing PyBullet environment and robot setup...")
    init_pb()
    robot = RobotSetup("r0", robot_type="husky_dual")
    print("✓ Robot setup complete.")

    # ------------------------------------------------------------------
    # Define start & goal configurations (12-DoF – two 6-axis arms)
    # ------------------------------------------------------------------
    start_conf = np.array(
        [
            -1.5021426049088822,
            2.875807965487516,
            1.8672209938146547,
            -3.1037391209227634,
            2.353848891975927,
            -1.47385655348834,
            1.5021347688211746,
            0.265786601011112,
            -1.8672242339039837,
            -0.0378579625998521,
            -2.353833622287848,
            -1.6677457011931158,
        ]
    )

    target_conf = np.array(
        [
            -1.8926153301031614,
            3.487309932900661,
            1.0134375755152472,
            -3.236291831601351,
            2.3060968246509237,
            -2.0113978398481405,
            1.89260890974217,
            -0.34571159289373343,
            -1.0134519530540729,
            0.09469383179844693,
            -2.3060853906423824,
            -1.1302087699453331,
        ]
    )

    # ------------------------------------------------------------------
    # Visualize start & goal configurations
    # ------------------------------------------------------------------
    print("\nVisualizing configurations...")
    robot.set_joint_positions(robot.arm_joints, start_conf)
    print("✓ Start configuration set.")
    pp.wait_for_user("Start configuration visualized - press a key to continue…")

    robot.set_joint_positions(robot.arm_joints, target_conf)
    print("✓ Goal configuration set.")
    pp.wait_for_user("Goal configuration visualized - press a key to plan…")

    # ------------------------------------------------------------------
    # Motion Planning
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("=== Dual-arm relative constraint planning ===")
    print("=" * 60)
    print(f"Space type: {args.space}")
    print(f"Planner: {args.planner}")
    print(f"Time limit: {args.time}s")
    print(f"Tolerance: {args.tolerance}")
    print(f"Interpolation points: {args.interpolate_points}")
    print("-" * 60)

    print("\nPlanning...")
    tic = time.time()
    result_traj, constraint = relativeConstraintPlanning(robot, start_conf, target_conf, args)
    toc = time.time()

    if result_traj is not None:
        print(f"\n✓ Planning succeeded in {toc - tic:.3f} s – {result_traj.shape[0]} waypoints.")

        # ------------------------------------------------------------------
        # Save trajectory if requested
        # ------------------------------------------------------------------
        if args.output:
            output_file = "dual_constraint_trajectory.txt"
            np.savetxt(output_file, result_traj, fmt="%.8f")
            print(f"✓ Trajectory saved to '{output_file}'.")

        # ------------------------------------------------------------------
        # Compute and plot constraint violations (default enabled)
        # ------------------------------------------------------------------
        if args.plot_violations or True:  # 默认启用
            print("\n" + "-" * 60)
            print("Constraint violation analysis")
            print("-" * 60)

            # 计算并绘制约束违反度
            violations = compute_and_plot_constraint_violations(result_traj, constraint, output_dir="./")

            print("\nConstraint analysis complete.")

        # ------------------------------------------------------------------
        # Interactive Visualization
        # ------------------------------------------------------------------
        print("\n" + "-" * 60)
        print("Interactive trajectory playback")
        print("-" * 60)
        print("Use slider to scrub through trajectory, ESC/q to exit...")

        # Draw polyline between left EE positions for context
        left_link = pp.link_from_name(robot.robot, HUSKY_DUAL_TOOL0_LEFT)
        right_link = pp.link_from_name(robot.robot, HUSKY_DUAL_TOOL0_RIGHT)

        print("Drawing trajectory visualization...")
        line_ids = []
        for i in range(result_traj.shape[0] - 1):
            # set pose i
            robot.set_joint_positions(robot.arm_joints, result_traj[i])
            left_pose_i = pp.get_link_pose(robot.robot, left_link)[0]
            right_pose_i = pp.get_link_pose(robot.robot, right_link)[0]

            robot.set_joint_positions(robot.arm_joints, result_traj[i + 1])
            left_pose_j = pp.get_link_pose(robot.robot, left_link)[0]
            right_pose_j = pp.get_link_pose(robot.robot, right_link)[0]

            # draw left polyline cyan, right yellow
            # line_ids.append(pp.add_line(left_pose_i, left_pose_j, color=[0, 1, 1], width=2))
            # line_ids.append(pp.add_line(right_pose_i, right_pose_j, color=[1, 1, 0], width=2))

        print("✓ Trajectory visualization ready.")

        # Slider interface
        slider = p.addUserDebugParameter("traj_idx", 0, result_traj.shape[0] - 1, 0)
        current_index = -1

        try:
            while True:
                idx = int(p.readUserDebugParameter(slider))
                if idx != current_index:
                    current_index = idx
                    robot.set_joint_positions(robot.arm_joints, result_traj[current_index])
                time.sleep(0.01)

                keys = p.getKeyboardEvents()
                if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
                    break
                if 113 in keys and keys[113] & p.KEY_WAS_TRIGGERED:
                    break
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            # Cleanup visualization elements
            print("Cleaning up visualization...")
            for lid in line_ids:
                try:
                    p.removeUserDebugItem(lid)
                except Exception:
                    pass
            try:
                p.removeUserDebugParameter(slider)
            except Exception:
                pass
            print("✓ Cleanup complete.")

    else:
        print(f"\n✗ Planning failed after {toc - tic:.3f} s.")
        print("Consider adjusting planning parameters or initial configurations.")
