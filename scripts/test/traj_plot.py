import argparse
import json
import os
import re
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pybullet_planning as pp

# Local imports
HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
import sys
sys.path.append(HERE)

from motion_planner.trajectory_dual_constrained_solver import TrajectoryDualConstrainedSolver
from utils.params import DATA_DIR, PROJECT_DIR


def _derive_scene_keys_from_target_name(target_name: str) -> List[str]:
    """Derive filename keys from a target name for matching trajectories.

    Examples:
        target_name = "robotx_box_A6-S4_end" -> ["robotx_box_A6-S4_end", "robotx_box"]
    """
    keys = [target_name]
    m = re.search(r"(.*)_A\d+", target_name)
    if m:
        keys.append(m.group(1))
    return list(dict.fromkeys([k for k in keys if k]))


def _find_candidate_trajectories(
    target_name: str,
    search_dirs: List[str],
    exts: Tuple[str, ...] = (".json", ".npy"),
) -> List[str]:
    """Search search_dirs recursively for trajectory files matching scene keys.

    Preference order: files containing target_name; then files containing scene prefix; filenames with
    'trajectory' or 'traj'.
    """
    keys = _derive_scene_keys_from_target_name(target_name)
    candidates: List[str] = []

    def is_traj_filename(name: str) -> bool:
        lower = name.lower()
        return ("traj" in lower) or ("trajectory" in lower)

    for root_dir in search_dirs:
        if not root_dir or not os.path.isdir(root_dir):
            continue
        for r, _, files in os.walk(root_dir):
            for f in files:
                if not f.endswith(exts):
                    continue
                if not is_traj_filename(f):
                    continue
                path = os.path.join(r, f)
                # rank matches by key position (0 is target_name), then filename length
                rank = None
                for i, k in enumerate(keys):
                    if k in f:
                        rank = (i, len(f))
                        break
                if rank is not None:
                    candidates.append((rank, path))

    # If nothing found, try exact solver output in PROJECT_DIR/data
    default_solver_out = os.path.join(PROJECT_DIR, "data", f"{target_name}_robot_trajectory.json")
    if os.path.isfile(default_solver_out):
        candidates.append(((0, len(os.path.basename(default_solver_out))), default_solver_out))

    # Sort by (key-rank, filename length) then path for determinism
    candidates.sort(key=lambda x: (x[0][0], x[0][1], x[1]))
    return [p for _, p in candidates]


def _load_trajectory(traj_path: str) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Load trajectory points as an array [N, D]. Supports:
    - compas_fab JointTrajectory JSON (dtype in root)
    - raw JSON list of lists
    - .npy arrays

    Returns:
        (trajectory_array, joint_names | None)
    """
    if traj_path.lower().endswith(".npy"):
        arr = np.load(traj_path)
        if arr.ndim != 2:
            raise ValueError(f"Unsupported npy trajectory shape: {arr.shape}")
        return np.asarray(arr, dtype=float), None

    with open(traj_path, "r") as f:
        raw = json.load(f)

    # compas_fab JointTrajectory JSON (manual parse, no compas dependency)
    if isinstance(raw, dict) and isinstance(raw.get("dtype"), str) and "JointTrajectory" in raw["dtype"]:
        data = raw.get("data", {})
        points = data.get("points", [])
        if not isinstance(points, list) or not points:
            raise ValueError("Invalid JointTrajectory JSON: data.points missing or empty")
        traj = np.array([np.array(p.get("joint_values", []), dtype=float) for p in points], dtype=float)
        joint_names = data.get("joint_names", None)
        return traj, list(joint_names) if joint_names else None

    # Raw JSON list of lists
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        arr = np.array(raw, dtype=float)
        return arr, None

    raise ValueError(f"Unrecognized trajectory format: {traj_path}")


def _compute_rel_pose_deltas(robot_setup, trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute deltas of left-in-right tool0 pose relative to the first frame.

    Args:
        robot_setup: initialized RobotSetup (dual-arm)
        trajectory: [N, D] joint angles; D must be 12 for dual arms

    Returns:
        (pos_deltas [N,3], euler_deltas [N,3]) where euler is (roll, pitch, yaw) in radians
    """
    if trajectory.ndim != 2:
        raise ValueError("Trajectory must be 2D [N, D]")
    if trajectory.shape[1] not in (12,):
        raise ValueError(f"Expected 12-DOF dual-arm trajectory; got D={trajectory.shape[1]}")

    N = trajectory.shape[0]
    pos_d = np.zeros((N, 3), dtype=float)
    eul_d = np.zeros((N, 3), dtype=float)

    # Set first frame and compute baseline relative pose
    robot_setup.set_joint_positions(robot_setup.arm_joints, trajectory[0])
    world_from_left0 = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
    world_from_right0 = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
    right_from_left0 = pp.multiply(pp.invert(world_from_right0), world_from_left0)

    for i in range(N):
        conf = trajectory[i]
        robot_setup.set_joint_positions(robot_setup.arm_joints, conf)

        world_from_left = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
        world_from_right = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
        right_from_left = pp.multiply(pp.invert(world_from_right), world_from_left)

        # delta = inv(first) * current
        delta = pp.multiply(pp.invert(right_from_left0), right_from_left)
        pos, quat = delta
        eul = pp.euler_from_quat(quat)

        pos_d[i] = np.array(pos, dtype=float)
        eul_d[i] = np.array(eul, dtype=float)

    return pos_d, eul_d


def _plot_and_save(pos_d: np.ndarray, eul_d: np.ndarray, out_path: str, title: str) -> None:
    """Plot position and Euler angle deltas and save to file.

    Euler deltas are shown in degrees.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    t = np.arange(pos_d.shape[0])
    # Position deltas
    axes[0].plot(t, pos_d[:, 0], label="dx")
    axes[0].plot(t, pos_d[:, 1], label="dy")
    axes[0].plot(t, pos_d[:, 2], label="dz")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Translation (m)")
    axes[0].set_title(f"Left tool0 in Right tool0 frame (pos delta)")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend()

    # Euler deltas
    eul_deg = np.rad2deg(eul_d)
    axes[1].plot(t, eul_deg[:, 0], label="droll")
    axes[1].plot(t, eul_deg[:, 1], label="dpitch")
    axes[1].plot(t, eul_deg[:, 2], label="dyaw")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Rotation (rad)")
    axes[1].set_title(f"Left tool0 in Right tool0 frame (ori delta)")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend()

    fig.suptitle(title)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot left-vs-right tool0 relative pose deltas for a trajectory.")
    parser.add_argument("--design_case", type=str, required=True, help="Design case directory name under DATA_DIR/husky_assembly_design_study")
    parser.add_argument("--target_name", type=str, required=True, help="Target state base name without suffix, e.g., robotx_box_A6-S4_end")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI when reconstructing scene")
    parser.add_argument("--traj", type=str, default=None, help="Explicit trajectory file path (.json or .npy)")
    parser.add_argument("--out", type=str, default=None, help="Output plot path (.png). Default: alongside trajectory")
    args = parser.parse_args()

    # Build RobotCellState path and initialize robot+scene like constrained solver
    design_study_path = os.path.join(DATA_DIR, "husky_assembly_design_study")
    target_cell_state_path = os.path.join(
        design_study_path, args.design_case, "RobotCellStates", f"{args.target_name}_RobotCellState.json"
    )
    if not os.path.isfile(target_cell_state_path):
        raise FileNotFoundError(f"RobotCellState not found: {target_cell_state_path}")

    robot_setup, _, _ = TrajectoryDualConstrainedSolver.initialize_robot_setup_for_planning(
        robot_name="r0",
        robot_type="husky_dual",
        target_cell_state_path=target_cell_state_path,
        use_scene_parser_gui=bool(args.gui),
        scene_parser_verbose=False,
    )

    traj_path = args.traj
    if traj_path is None:
        search_dirs = [os.path.join(PROJECT_DIR, "data")]
        candidates = _find_candidate_trajectories(args.target_name, search_dirs)
        if not candidates:
            raise FileNotFoundError(
                f"No matching trajectory found under PROJECT_DIR/data for '{args.target_name}'"
            )
        traj_path = candidates[0]
        print(f"Using trajectory: {traj_path}")
    else:
        if not os.path.isfile(traj_path):
            raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    trajectory, _ = _load_trajectory(traj_path)
    if trajectory.shape[1] != len(robot_setup.arm_joints):
        # Best-effort: if 6-DOF single-arm provided, fail with clear message
        raise ValueError(
            f"Trajectory DOF ({trajectory.shape[1]}) does not match dual-arm DOF ({len(robot_setup.arm_joints)})."
        )

    with pp.WorldSaver():
        pos_d, eul_d = _compute_rel_pose_deltas(robot_setup, trajectory)

    # Prepare output
    if args.out is None:
        base_dir = os.path.join(os.path.dirname(traj_path), "plots")
        os.makedirs(base_dir, exist_ok=True)
        out_path = os.path.join(base_dir, f"{args.target_name}_left_in_right_tool0_deltas.png")
    else:
        out_path = args.out

    title = f"Scene: {args.design_case} | Target: {args.target_name}"
    _plot_and_save(pos_d, eul_d, out_path, title)
    print(f"Plot saved to: {out_path}")

    # Cleanup
    robot_setup.cleanup()


if __name__ == "__main__":
    main()


