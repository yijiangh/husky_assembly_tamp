import argparse
import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pybullet_planning as pp

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from motion_planner.trajectory_dual_constrained_solver import TrajectoryDualConstrainedSolver
from utils.params import DATA_DIR, PROJECT_DIR


def _angle_between_unit_vectors(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    if u_norm > 0.0:
        u = u / u_norm
    if v_norm > 0.0:
        v = v / v_norm
    dot = float(np.dot(u, v))
    dot = max(-1.0, min(1.0, dot))
    return float(np.arccos(dot))


def _compute_rel_pose_deltas(robot_setup, trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if trajectory.ndim != 2:
        raise ValueError("Trajectory must be 2D [N, D]")
    if trajectory.shape[1] not in (12,):
        raise ValueError(f"Expected 12-DOF dual-arm trajectory; got D={trajectory.shape[1]}")

    N = trajectory.shape[0]
    pos_d = np.zeros((N, 3), dtype=float)
    rot_err_d = np.zeros((N, 3), dtype=float)

    robot_setup.set_joint_positions(robot_setup.arm_joints, trajectory[0])
    world_from_left0 = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
    world_from_right0 = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
    right_from_left0 = pp.multiply(pp.invert(world_from_right0), world_from_left0)
    R0 = pp.tform_from_pose(right_from_left0)[:3, :3]
    p0 = np.asarray(right_from_left0[0], dtype=float)

    for i in range(N):
        conf = trajectory[i]
        robot_setup.set_joint_positions(robot_setup.arm_joints, conf)

        world_from_left = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
        world_from_right = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
        right_from_left = pp.multiply(pp.invert(world_from_right), world_from_left)

        pos = np.asarray(right_from_left[0], dtype=float) - p0

        R_cur = pp.tform_from_pose(right_from_left)[:3, :3]
        x_err = _angle_between_unit_vectors(R_cur[:, 0], R0[:, 0])
        y_err = _angle_between_unit_vectors(R_cur[:, 1], R0[:, 1])
        z_err = _angle_between_unit_vectors(R_cur[:, 2], R0[:, 2])

        pos_d[i] = np.array(pos, dtype=float)
        rot_err_d[i] = np.array([x_err, y_err, z_err], dtype=float)

    return pos_d, rot_err_d


def _plot_and_save(pos_d: np.ndarray, rot_err_d: np.ndarray, out_path: str, title: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    t = np.arange(pos_d.shape[0])
    axes[0].plot(t, pos_d[:, 0], label="dx")
    axes[0].plot(t, pos_d[:, 1], label="dy")
    axes[0].plot(t, pos_d[:, 2], label="dz")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Translation (m)")
    axes[0].set_title(f"Left tool0 in Right tool0 frame (pos delta)")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend()

    axes[1].plot(t, rot_err_d[:, 0], label="x-axis err")
    axes[1].plot(t, rot_err_d[:, 1], label="y-axis err")
    axes[1].plot(t, rot_err_d[:, 2], label="z-axis err")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Rotation (rad)")
    axes[1].set_title(f"Left tool0 in Right tool0 frame (ori delta)")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend()

    fig.suptitle(title)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Dual-arm projection accuracy test: sample 1000 configs and plot left-in-right tool0 deltas vs start.")
    parser.add_argument("--design_case", type=str, required=True, help="Design case directory name under DATA_DIR/husky_assembly_design_study")
    parser.add_argument("--target_name", type=str, required=True, help="Target state base name without suffix, e.g., robotx_box_A6-S4_end")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI when reconstructing scene")
    parser.add_argument("--count", type=int, default=100, help="Number of valid projected configurations to collect")
    parser.add_argument("--max_attempts_per_right", type=int, default=5, help="Projection attempts per random right-arm sample")
    parser.add_argument("--out", type=str, default=None, help="Output plot path (.png). Default: PROJECT_DIR/data/plots/<target>_left_in_right_tool0_deltas.png")
    args = parser.parse_args()

    # Initialize scene similar to trajectory_dual_constrained_solver
    design_study_path = os.path.join(DATA_DIR, "husky_assembly_design_study")
    target_cell_state_path = os.path.join(design_study_path, args.design_case, "RobotCellStates", f"{args.target_name}_RobotCellState.json")
    if not os.path.isfile(target_cell_state_path):
        raise FileNotFoundError(f"RobotCellState not found: {target_cell_state_path}")

    robot_setup, target_conf, projector = TrajectoryDualConstrainedSolver.initialize_robot_setup_for_planning(
        robot_name="r0",
        robot_type="husky_dual",
        target_cell_state_path=target_cell_state_path,
        use_scene_parser_gui=bool(args.gui),
        scene_parser_verbose=False,
    )

    # Collision filter
    collision_fn = robot_setup.create_collision_fn(obstacle_bodies=robot_setup.obstacles)

    # Sample projected configs
    projected_confs = []
    rng = np.random.default_rng()
    print(f"Sampling valid projected configurations... target count = {args.count}")
    while len(projected_confs) < int(args.count):
        right_conf = rng.uniform(-np.pi, np.pi, 6)
        projected = projector.project_multiple(right_conf, max_attempts=int(max(1, args.max_attempts_per_right)), collision_fn=collision_fn)
        if projected is None or projected.size == 0:
            continue
        projected_confs.append(projected[0])
        # if len(projected_confs) % 50 == 0:
        #     print(f"Collected {len(projected_confs)} / {args.count}")
        print(f"Collected {len(projected_confs)} / {args.count}")

    # Build trajectory: start frame is the target configuration
    start_conf = np.array(target_conf, dtype=float)
    trajectory = np.vstack([start_conf.reshape(1, -1), np.asarray(projected_confs, dtype=float)])

    # Compute deltas with respect to the start frame
    with pp.WorldSaver():
        pos_d, rot_err_d = _compute_rel_pose_deltas(robot_setup, trajectory)

    # Prepare output path
    if args.out is None:
        out_dir = os.path.join(PROJECT_DIR, "data", "plots")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{args.target_name}_ik_random_test.png")
    else:
        out_path = args.out

    title = f"Scene: {args.design_case} | Target: {args.target_name}"
    _plot_and_save(pos_d, rot_err_d, out_path, title)
    print(f"Plot saved to: {out_path}")

    # Cleanup
    robot_setup.cleanup()


if __name__ == "__main__":
    main()


