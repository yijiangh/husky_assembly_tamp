#!/usr/bin/env python3
from __future__ import annotations

"""Simplified linear end-effector constrained planning demo (v2).

This script mimics *single_constrain_test.py* but leverages the generic
:class:`OMPLConstrainedPlanner` wrapper defined in *ompl_planner.py* to keep
boiler-plate to a minimum.
"""

# -----------------------------------------------------------------------------
# Standard / third-party imports
# -----------------------------------------------------------------------------
import argparse
import os
import sys
import time
from functools import partial
from typing import List, Tuple

import numpy as np
import pybullet as p
import pybullet_planning as pp

# -----------------------------------------------------------------------------
# Local project imports (robots, utils, OMPL helpers)
# -----------------------------------------------------------------------------
HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(HERE, os.pardir)))  # make *scripts* root importable

from ConstrainedPlanningCommon import addAtlasOptions, addConstrainedOptions, addPlannerOption, addSpaceOption  # noqa: E402
from ompl_planner import OMPLConstrainedPlanner  # noqa: E402
from robot.robot_setup import (
    HUSKY_ARM_JOINT_NAMES,
    HUSKY_TOOL0_NAME,
    RobotSetup,
)  # noqa: E402
from utils.collision import init_pb  # noqa: E402
from utils.util import interpolate  # noqa: E402

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    """Return *v* normalised (no-op if zero-norm)."""
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def make_perp_vectors(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors perpendicular to *direction*."""
    # Use Gram-Schmidt with an arbitrary helper vector
    if abs(direction[0]) < 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    else:
        helper = np.array([0.0, 1.0, 0.0])
    v1 = helper - np.dot(helper, direction) * direction
    v1 = normalize(v1)
    v2 = np.cross(direction, v1)
    v2 = normalize(v2)
    return v1, v2


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

def main(argv: List[str] | None = None):  # noqa: C901 – keep main cohesive
    # ------------------------------------------------------------------
    # Parse CLI args (reuse helpers from *ConstrainedPlanningCommon*)
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Linear EE constrained planning (simplified)")
    parser.add_argument(
        "--line-direction",
        nargs=3,
        type=float,
        default=[1.0, 0.0, 0.0],
        metavar=("dx", "dy", "dz"),
        help="Direction of the constraint line (default: 1 0 0)",
    )
    parser.add_argument(
        "--move-distance",
        type=float,
        default=0.2,
        help="Distance to move along the line for goal pose (default: 0.2 m)",
    )
    parser.add_argument(
        "--output",
        action="store_true",
        help="Dump trajectory to *linear_constraint_trajectory.txt*.",
    )
    parser.add_argument(
        "--interpolate-points",
        type=int,
        default=50,
        help="Number of samples in the interpolated trajectory (default: 50)",
    )

    # Generic OMPL options ------------------------------------------------
    addSpaceOption(parser)
    addPlannerOption(parser)
    addConstrainedOptions(parser)
    addAtlasOptions(parser)

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # PyBullet & robot setup
    # ------------------------------------------------------------------
    init_pb()
    robot_setup = RobotSetup("r0")
    robot_id = robot_setup.robot
    arm_joints = robot_setup.arm_joints  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Build constraint parameters
    # ------------------------------------------------------------------
    line_dir = normalize(np.asarray(args.line_direction, dtype=float))
    ee_pose = pp.get_link_pose(robot_id, pp.link_from_name(robot_id, HUSKY_TOOL0_NAME))
    line_point = np.asarray(ee_pose[0])
    perp_vec1, perp_vec2 = make_perp_vectors(line_dir)

    # ------------------------------------------------------------------
    # Helpers: FK via PyBullet (with context manager-style restore)
    # ------------------------------------------------------------------
    def fk_pybullet(q: np.ndarray) -> np.ndarray:
        """Return EE position for joint configuration *q* using PyBullet."""
        current_joints = pp.get_joint_positions(robot_id, arm_joints)
        try:
            pp.set_joint_positions(robot_id, arm_joints, q.tolist())
            pose = pp.get_link_pose(robot_id, pp.link_from_name(robot_id, HUSKY_TOOL0_NAME))
            return np.asarray(pose[0])
        finally:
            pp.set_joint_positions(robot_id, arm_joints, current_joints)

    # ------------------------------------------------------------------
    # Constraint *function* and *jacobian* as required by OMPL
    # ------------------------------------------------------------------
    ambient_dim = len(HUSKY_ARM_JOINT_NAMES)
    codim = 2  # EE constrained to line -> 2 scalar constraints

    def constraint_function(x: np.ndarray, out: np.ndarray) -> None:  # noqa: D401
        """f(q) = 0 defining EE distance to line in two orthogonal directions."""
        ee_pos = fk_pybullet(x)
        vec = ee_pos - line_point
        out[0] = float(np.dot(vec, perp_vec1))
        out[1] = float(np.dot(vec, perp_vec2))

    def constraint_jacobian(x: np.ndarray, out: np.ndarray) -> None:
        epsilon = 1e-6
        base_pos = fk_pybullet(x)
        for i in range(ambient_dim):
            x_eps = np.array(x, copy=True)
            x_eps[i] += epsilon
            pos_eps = fk_pybullet(x_eps)
            deriv = (pos_eps - base_pos) / epsilon
            out[0, i] = np.dot(deriv, perp_vec1)
            out[1, i] = np.dot(deriv, perp_vec2)

    # ------------------------------------------------------------------
    # Joint bounds list from URDF / PyBullet
    # ------------------------------------------------------------------
    joint_bounds: List[Tuple[float, float]] = []
    for j in arm_joints:
        info = pp.get_joint_info(robot_id, j)
        lo, hi = info.jointLowerLimit, info.jointUpperLimit
        if lo == 0 and hi == -1:  # continuous joint
            lo, hi = -2 * np.pi, 2 * np.pi
        joint_bounds.append((float(lo), float(hi)))

    # ------------------------------------------------------------------
    # Define *is_valid* collision checker (trivial, but placeholder)
    # ------------------------------------------------------------------
    def is_valid(state) -> bool:  # noqa: D401
        # Basic joint limit check (already enforced by bounds, but keep) -- can extend with collisions
        for i in range(ambient_dim):
            if not (joint_bounds[i][0] <= state[i] <= joint_bounds[i][1]):
                return False
        return True

    # ------------------------------------------------------------------
    # Determine start / goal joint configurations
    # ------------------------------------------------------------------
    start_conf = np.asarray(pp.get_joint_positions(robot_id, arm_joints))

    # Goal – move EE along *line_dir* by *move_distance*
    target_pos = line_point + args.move_distance * line_dir
    target_pose = (target_pos.tolist(), ee_pose[1])  # keep current orientation

    goal_conf = robot_setup.get_relative_ik_solution(target_pose, q_init=start_conf.tolist())
    if goal_conf is None:
        print("[WARN] IK failed – falling back to small joint offset goal.")
        goal_conf = start_conf + 0.1
    goal_conf = np.asarray(goal_conf, dtype=float)

    # ------------------------------------------------------------------
    # Instantiate *OMPLConstrainedPlanner* and solve
    # ------------------------------------------------------------------
    planner = OMPLConstrainedPlanner(
        function=constraint_function,
        jacobian=constraint_jacobian,
        ambient_dim=ambient_dim,
        codim=codim,
        bounds=joint_bounds,
        is_valid=is_valid,
        space_type=args.space,
        planner_name=args.planner.split(",")[0],
        interpolate_points=args.interpolate_points,
        max_planning_time=args.time,
    )

    print("\nPlanning …")
    tic = time.time()
    trajectory = planner.plan(start_conf, goal_conf)
    toc = time.time()

    if trajectory is None:
        print("✗ Planning failed.")
        return

    print(f"✓ Planning succeeded in {toc - tic:.3f} s – {trajectory.shape[0]} waypoints.")

    # ------------------------------------------------------------------
    # Save path if requested
    # ------------------------------------------------------------------
    if args.output:
        np.savetxt("linear_constraint_trajectory.txt", trajectory, fmt="%.8f")
        print("Trajectory saved to *linear_constraint_trajectory.txt*.")

    # ------------------------------------------------------------------
    # Quick interactive viewer (optional)
    # ------------------------------------------------------------------
    print("\nInteractive playback – use slider to scrub, ESC/q to quit …")
    slider = p.addUserDebugParameter("traj_idx", 0, trajectory.shape[0] - 1, 0)
    sphere_id = None
    try:
        while True:
            idx = int(p.readUserDebugParameter(slider))
            joint_conf = trajectory[min(idx, trajectory.shape[0] - 1)]
            pp.set_joint_positions(robot_id, arm_joints, joint_conf)
            ee_pose_now = pp.get_link_pose(robot_id, pp.link_from_name(robot_id, HUSKY_TOOL0_NAME))
            # Update marker
            if sphere_id is not None:
                try:
                    p.removeUserDebugItem(sphere_id)
                except Exception:
                    pass
            sphere_id = pp.draw_point(ee_pose_now[0], size=0.02, color=[1, 0, 1])
            time.sleep(0.01)
            # Exit on ESC/q
            keys = p.getKeyboardEvents()
            if (27 in keys and keys[27] & p.KEY_WAS_TRIGGERED) or (
                113 in keys and keys[113] & p.KEY_WAS_TRIGGERED
            ):
                break
    finally:
        if sphere_id is not None:
            p.removeUserDebugItem(sphere_id)
        p.removeUserDebugParameter(slider)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
