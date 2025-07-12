#!/usr/bin/env python3

import argparse
import os
import sys
import time
from functools import partial
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_planning as pp

# Import OMPL libraries
try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    import sys
    from os.path import abspath, dirname, join

    sys.path.insert(0, join(dirname(dirname(dirname(abspath(__file__)))), "py-bindings"))
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from ConstrainedPlanningCommon import *
from robot.robot_setup import HUSKY_DUAL_ARM_JOINT_NAMES, HUSKY_DUAL_TOOL0_LEFT, HUSKY_DUAL_TOOL0_RIGHT, RobotSetup
from utils.collision import init_pb
from utils.params import *
from utils.util import interpolate


class RelativeEndEffectorConstraint(ob.Constraint):
    """Constraint that keeps the relative position of the left end-effector in the right end-effector's coordinate frame fixed.

    The constraint enforces that the left end-effector maintains its position relative to the right
    end-effector's coordinate frame, which is appropriate for grasping a rigid object like a rod.
    """

    def __init__(self, num_joints: int, robot_setup: RobotSetup):
        # 3 positional constraints (dx, dy, dz)
        super(RelativeEndEffectorConstraint, self).__init__(num_joints, 6)

        self.robot_setup = robot_setup
        self.robot_id = robot_setup.robot
        self.num_joints = num_joints

        # Identify link indices for both arms
        self.left_tool_link = pp.link_from_name(self.robot_id, HUSKY_DUAL_TOOL0_LEFT)
        self.right_tool_link = pp.link_from_name(self.robot_id, HUSKY_DUAL_TOOL0_RIGHT)

        # Joint indices that the constraint acts on (all manipulator joints)
        self.control_joints = [pp.joint_from_name(self.robot_id, j) for j in HUSKY_DUAL_ARM_JOINT_NAMES]

        self.collision_fn = robot_setup.create_collision_fn()

        # Record the desired relative position of left EE in right EE's coordinate frame
        left_pose = pp.get_link_pose(self.robot_id, self.left_tool_link)  # world_from_left
        right_pose = pp.get_link_pose(self.robot_id, self.right_tool_link)  # world_from_right

        # Get left position in right's coordinate frame
        right_from_left = pp.multiply(pp.invert(right_pose), left_pose)
        left_from_right = pp.multiply(pp.invert(left_pose), right_pose)

        self.desired_right_from_left = np.array(right_from_left[0])
        self.desired_left_from_right = np.array(left_from_right[0])

    # ------------------------------------------------------------------
    #   Core OMPL callbacks
    # ------------------------------------------------------------------
    def function(self, x, out):
        """Compute constraint residuals given joint configuration *x*.

        out[i] = f_i(x) where f(x) = (current_relative_pos - desired_relative_pos).
        """
        current_right_from_left, current_left_from_right = self._relative_position(x)
        diff_right = current_right_from_left - self.desired_right_from_left
        out[0], out[1], out[2] = diff_right  # dx, dy, dz

        diff_left = current_left_from_right - self.desired_left_from_right
        out[3], out[4], out[5] = diff_left
        # out[3], out[4], out[5] = 0, 0, 0

    def jacobian(self, x, out):
        """Finite-difference Jacobian of the constraint."""
        epsilon = 1e-6
        base_relative_right, base_relative_left = self._relative_position(x)

        # Initialise jacobian with zeros (codim x ambient)
        out[:, :] = np.zeros((self.getCoDimension(), self.getAmbientDimension()))

        for i in range(self.num_joints):
            x_plus = np.array(x)
            x_plus[i] += epsilon
            relative_pos_plus_right, relative_pos_plus_left = self._relative_position(x_plus)
            deriv = (relative_pos_plus_right - base_relative_right) / epsilon  # 3-vector
            out[0, i], out[1, i], out[2, i] = deriv

            deriv = (relative_pos_plus_left - base_relative_left) / epsilon  # 3-vector
            out[3, i], out[4, i], out[5, i] = deriv
            # out[3, i], out[4, i], out[5, i] = 0, 0, 0

    # ------------------------------------------------------------------
    #   Helpers
    # ------------------------------------------------------------------
    def _relative_position(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return position of left end-effector in right end-effector's coordinate frame for given joint angles."""
        # Backup current joint state
        current_conf = pp.get_joint_positions(self.robot_id, self.control_joints)
        try:
            self.robot_setup.set_joint_positions(self.control_joints, joint_angles)
            left_pose = pp.get_link_pose(self.robot_id, self.left_tool_link)
            right_pose = pp.get_link_pose(self.robot_id, self.right_tool_link)
            right_from_left = pp.multiply(pp.invert(right_pose), left_pose)
            left_from_right = pp.multiply(pp.invert(left_pose), right_pose)

            # Get left position in right's coordinate frame
            left_in_right_frame = np.array(right_from_left[0])
            right_in_left_frame = np.array(left_from_right[0])

            return left_in_right_frame, right_in_left_frame
        finally:
            self.robot_setup.set_joint_positions(self.control_joints, current_conf)

    def compute_violation(self, joint_angles: np.ndarray) -> float:
        """Compute constraint violation magnitude for given joint configuration."""
        current_right_from_left, current_left_from_right = self._relative_position(joint_angles)
        diff_right = current_right_from_left - self.desired_right_from_left
        diff_left = current_left_from_right - self.desired_left_from_right
        return (np.linalg.norm(diff_right) + np.linalg.norm(diff_left)) / 2

    # ------------------------------------------------------------------
    #   Optional helpers for planning convenience
    # ------------------------------------------------------------------
    def isValid(self, state):
        """Basic state validity that checks joint limits."""
        j = np.array([state[i] for i in range(self.getAmbientDimension())])
        # Simple bound check: assume revolute joints within ±2π
        if np.any(j < -2 * np.pi) or np.any(j > 2 * np.pi):
            return False
        if self.collision_fn(j, diagnosis=False):
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

    def getProjection(self, space):
        """Return a ProjectionEvaluator mapping state -> relative translation (3-D)."""

        class RelProjection(ob.ProjectionEvaluator):
            def __init__(self, space, constraint):
                super(RelProjection, self).__init__(space)
                self.constraint = constraint
                self.defaultCellSizes()

            def getDimension(self):
                return 6

            def defaultCellSizes(self):
                # Equal cell sizes for each coordinate
                self.cellSizes_ = list2vec([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

            def project(self, state, projection):
                joint_angles = [state[i] for i in range(self.constraint.num_joints)]
                relative_right, relative_left = self.constraint._relative_position(joint_angles)
                projection[0], projection[1], projection[2] = relative_right
                projection[3], projection[4], projection[5] = relative_left

        return RelProjection(space, self)


# ------------------------------------------------------------------------------------
#   Planning helpers (analogous to single arm script)
# ------------------------------------------------------------------------------------


def compute_and_plot_constraint_violations(trajectory: np.ndarray, constraint: RelativeEndEffectorConstraint, output_dir: str = "./") -> np.ndarray:
    """
    Compute constraint violation for each point in trajectory, plot curves and save images.

    Args:
        trajectory: Trajectory array with shape (n_points, n_joints)
        constraint: Constraint object
        output_dir: Output directory

    Returns:
        violations: Array of violation magnitudes for each point
    """
    print("Computing constraint violations for trajectory...")

    n_points = trajectory.shape[0]
    violations = np.zeros(n_points)

    # Compute violation for each point
    for i in range(n_points):
        violations[i] = constraint.compute_violation(trajectory[i])
        if (i + 1) % 50 == 0 or i == n_points - 1:
            print(f"  Progress: {i + 1}/{n_points} points processed")

    # Statistical information
    max_violation = np.max(violations)
    mean_violation = np.mean(violations)
    final_violation = violations[-1]

    print(f"\nConstraint Violation Statistics:")
    print(f"  Max violation: {max_violation:.6f} m")
    print(f"  Mean violation: {mean_violation:.6f} m")
    print(f"  Final violation: {final_violation:.6f} m")

    # Plot violation curves
    plt.figure(figsize=(12, 8))

    # Main plot: violation vs time
    plt.subplot(2, 1, 1)
    time_steps = np.arange(n_points)
    plt.plot(time_steps, violations, "b-", linewidth=2, label="Constraint Violation")
    plt.axhline(y=mean_violation, color="r", linestyle="--", alpha=0.7, label=f"Mean: {mean_violation:.6f} m")
    plt.xlabel("Trajectory Point Index")
    plt.ylabel("Violation Magnitude (m)")
    plt.title("Dual-Arm Relative Position Constraint Violation")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Subplot: violation in logarithmic scale (for small values)
    plt.subplot(2, 1, 2)
    plt.semilogy(time_steps, violations + 1e-10, "g-", linewidth=2, label="Constraint Violation (log scale)")
    plt.xlabel("Trajectory Point Index")
    plt.ylabel("Violation Magnitude (m, log scale)")
    plt.title("Constraint Violation (Logarithmic Scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Save plot
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"dual_constraint_violations_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"✓ Constraint violation plot saved to: {filepath}")

    # Also save violation data
    data_filename = f"dual_constraint_violations_data_{timestamp}.txt"
    data_filepath = os.path.join(output_dir, data_filename)

    # Save data: first column is time step, second column is violation
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


def relativeConstraintPlanning(robot_setup: RobotSetup, start_conf: np.ndarray, goal_conf: np.ndarray, options, constraint=None):
    num_joints = len(robot_setup.arm_joints)
    if constraint is None:
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
            -2.000957703613043,
            -2.8274063096076927,
            0.7599204916355927,
            -1.417949677637465,
            2.247283007586109,
            -0.9687684451721333,
            2.0009525404422375,
            -0.31418100640279406,
            -0.7599358045326311,
            -1.7236269235075836,
            -2.2472696640064473,
            -2.1728147476121156,
        ]
    )

    target_conf = np.array(
        [
            -1.8252972693363294,
            -3.0657951326796105,
            1.0227352141951294,
            -1.4044710260259439,
            2.0807387954828176,
            -0.9012167418971372,
            1.8252547582772554,
            -0.07602775941548445,
            -1.0227588851777303,
            -1.7356745249695096,
            -2.080403151523388,
            -2.239946767445506,
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
    # Pre-planning constraint validation
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Pre-planning constraint validation")
    print("-" * 60)

    # Create constraint object for validation
    num_joints = len(robot.arm_joints)
    constraint = RelativeEndEffectorConstraint(num_joints, robot)

    # Check start configuration constraint violation
    start_violation = constraint.compute_violation(start_conf)
    print(f"Start configuration constraint violation: {start_violation:.6f} m")

    # Check target configuration constraint violation
    target_violation = constraint.compute_violation(target_conf)
    print(f"Target configuration constraint violation: {target_violation:.6f} m")

    # Warn if violations are too large
    max_acceptable_violation = 0.01  # 1cm threshold
    if start_violation > max_acceptable_violation:
        print(f"⚠️  WARNING: Start configuration has large constraint violation ({start_violation:.6f} m > {max_acceptable_violation} m)")

    if target_violation > max_acceptable_violation:
        print(f"⚠️  WARNING: Target configuration has large constraint violation ({target_violation:.6f} m > {max_acceptable_violation} m)")
        print("   This may make planning more difficult or result in poor trajectory quality.")

    if start_violation <= max_acceptable_violation and target_violation <= max_acceptable_violation:
        print("✓ Both start and target configurations have acceptable constraint violations.")

    print("-" * 60)

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
    result_traj, constraint = relativeConstraintPlanning(robot, start_conf, target_conf, args, constraint)
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
        if args.plot_violations or True:  # Default enabled
            print("\n" + "-" * 60)
            print("Constraint violation analysis")
            print("-" * 60)

            # Compute and plot constraint violations
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
