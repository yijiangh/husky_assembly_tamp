from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pybullet
import pybullet_planning as pp
from compas_fab.robots import RobotSemantics
from compas_robots import RobotModel

from husky_assembly_tamp.utils.util import normalize_angles, setup_logger


logger = setup_logger("stage1_path_validation")

PoseLike = Tuple[np.ndarray, np.ndarray]
FullConf = np.ndarray
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
AXIS_NAMES = ("x", "y", "z")
DEFAULT_DENSE_JOINT_VALIDATION_STEP_RAD = float(np.radians(0.1))


def get_disabled_collisions_from_link_names(
    robot: int,
    link_name_pairs: Sequence[Tuple[str, str]],
) -> list[Tuple[int, int]]:
    disabled_pairs: list[Tuple[int, int]] = []
    for link1_name, link2_name in link_name_pairs:
        if not (pp.has_link(robot, link1_name) and pp.has_link(robot, link2_name)):
            continue
        disabled_pairs.append((pp.link_from_name(robot, link1_name), pp.link_from_name(robot, link2_name)))
    return disabled_pairs


def status_label(value: Optional[bool]) -> str:
    if value is None:
        return "N/A"
    return "PASS" if value else "FAIL"


def import_matplotlib_pyplot(interactive: bool = False):
    try:
        import matplotlib

        if interactive and "agg" in matplotlib.get_backend().lower():
            # Monitor-triggered validation should pop up a plot instead of
            # writing an image file. Try common GUI backends when matplotlib
            # defaulted to Agg.
            for backend in ("QtAgg", "TkAgg"):
                try:
                    matplotlib.use(backend, force=True)
                    break
                except Exception:
                    continue
        if not interactive:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        logger.warning(f"Skipping trajectory validation plot (matplotlib unavailable): {exc}")
        return None


def quat_to_rotation_matrix(quat: Sequence[float]) -> np.ndarray:
    return np.asarray(pybullet.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)


def axis_angle_deg(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm_product = float(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if norm_product <= 1e-12:
        return 0.0
    cosine = float(np.clip(np.dot(vec1, vec2) / norm_product, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def rotation_axis_differences_deg(reference_quat: Sequence[float], current_quat: Sequence[float]) -> Dict[str, float]:
    ref_rot = quat_to_rotation_matrix(reference_quat)
    cur_rot = quat_to_rotation_matrix(current_quat)
    return {
        "x": axis_angle_deg(ref_rot[:, 0], cur_rot[:, 0]),
        "y": axis_angle_deg(ref_rot[:, 1], cur_rot[:, 1]),
        "z": axis_angle_deg(ref_rot[:, 2], cur_rot[:, 2]),
    }


def get_joint_labels(num_joints: int) -> list[str]:
    if num_joints == 12:
        return [f"L{i + 1}" for i in range(6)] + [f"R{i + 1}" for i in range(6)]
    return [f"q{i + 1}" for i in range(num_joints)]


def joint_path_to_raw_deg(joint_path_rad: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert a list of joint configurations to a 2D array of degrees, with
    NO ±π unwrapping — values are exactly what the UR controller consumes."""
    arr = np.asarray(joint_path_rad, dtype=float)
    if arr.size == 0:
        return np.empty((0, 0), dtype=float)
    return np.degrees(arr)


def maybe_normalize_angles(values: Sequence[float] | np.ndarray, use_angle_normalization: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if use_angle_normalization:
        return np.asarray(normalize_angles(arr), dtype=float)
    return arr


def refine_joint_path_for_validation(
    joint_path: Sequence[FullConf],
    max_joint_step_rad: float,
    use_angle_normalization: bool,
) -> Tuple[list[FullConf], list[Dict[str, Any]]]:
    if len(joint_path) == 0:
        return [], []

    normalized_joint_path = [maybe_normalize_angles(conf, use_angle_normalization) for conf in joint_path]
    if len(normalized_joint_path) == 1 or max_joint_step_rad <= 0.0:
        metadata = [
            {
                "dense_waypoint_index": idx,
                "waypoint_index": idx,
                "segment_index": max(0, idx - 1),
                "segment_alpha": 1.0,
                "path_fraction": 0.0 if len(normalized_joint_path) <= 1 else float(idx) / float(len(normalized_joint_path) - 1),
            }
            for idx in range(len(normalized_joint_path))
        ]
        return normalized_joint_path, metadata

    dense_path: list[FullConf] = [normalized_joint_path[0]]
    dense_metadata: list[Dict[str, Any]] = [
        {
            "dense_waypoint_index": 0,
            "waypoint_index": 0,
            "segment_index": 0,
            "segment_alpha": 0.0,
            "path_fraction": 0.0,
        }
    ]
    num_segments = len(normalized_joint_path) - 1
    for segment_idx, (prev_conf, next_conf) in enumerate(zip(normalized_joint_path[:-1], normalized_joint_path[1:])):
        delta = maybe_normalize_angles(
            np.asarray(next_conf, dtype=float) - np.asarray(prev_conf, dtype=float),
            use_angle_normalization,
        )
        num_steps = max(1, int(np.ceil(float(np.max(np.abs(delta))) / float(max_joint_step_rad))))
        for step_idx in range(1, num_steps + 1):
            alpha = float(step_idx) / float(num_steps)
            conf = maybe_normalize_angles(np.asarray(prev_conf, dtype=float) + alpha * delta, use_angle_normalization)
            dense_path.append(conf)
            dense_metadata.append(
                {
                    "dense_waypoint_index": len(dense_path) - 1,
                    "waypoint_index": int(round(segment_idx + alpha)),
                    "segment_index": segment_idx,
                    "segment_alpha": alpha,
                    "path_fraction": float(segment_idx + alpha) / float(num_segments),
                }
            )

    return dense_path, dense_metadata


def compute_relative_transform_drift(
    robot: int,
    arm_joints: Sequence[int],
    tool_link_left: int,
    tool_link_right: int,
    joint_path: Sequence[FullConf],
) -> Tuple[list[float], Dict[str, list[float]]]:
    translation_errors_m: list[float] = []
    rotation_axis_errors_deg = {axis_name: [] for axis_name in AXIS_NAMES}
    base_relative_pose = None
    for conf in joint_path:
        pp.set_joint_positions(robot, arm_joints, conf)
        world_from_left = pp.get_link_pose(robot, tool_link_left)
        world_from_right = pp.get_link_pose(robot, tool_link_right)
        relative_pose = pp.multiply(pp.invert(world_from_left), world_from_right)
        if base_relative_pose is None:
            base_relative_pose = relative_pose
        translation_errors_m.append(float(np.linalg.norm(np.asarray(relative_pose[0]) - np.asarray(base_relative_pose[0]))))
        axis_diffs_deg = rotation_axis_differences_deg(base_relative_pose[1], relative_pose[1])
        for axis_name in AXIS_NAMES:
            rotation_axis_errors_deg[axis_name].append(axis_diffs_deg[axis_name])
    return translation_errors_m, rotation_axis_errors_deg


def update_joint_continuity_result(
    result: Dict[str, Any],
    command_joint_path: Sequence[FullConf],
) -> None:
    # Joint flips must be checked on the exact command waypoints, not on
    # inserted validation samples.
    if len(command_joint_path) >= 2:
        step_max_deltas = []
        for prev_conf, next_conf in zip(command_joint_path[:-1], command_joint_path[1:]):
            step_delta = np.abs(np.asarray(next_conf, dtype=float) - np.asarray(prev_conf, dtype=float))
            step_max_deltas.append(float(np.max(step_delta)))
        if step_max_deltas:
            max_delta = max(step_max_deltas)
            threshold = float(result["joint_continuity_threshold_rad"])
            first_bad_step = next(
                (idx + 1 for idx, delta in enumerate(step_max_deltas) if delta > threshold),
                None,
            )
            result["joint_continuity_max_delta_rad"] = max_delta
            result["joint_continuity_first_bad_step"] = first_bad_step
            result["joint_continuity_ok"] = first_bad_step is None
    else:
        result["joint_continuity_ok"] = True
        result["joint_continuity_max_delta_rad"] = 0.0


def update_relative_transform_result(
    result: Dict[str, Any],
    relative_translation_errors_m: Sequence[float],
    relative_rotation_axis_errors_deg: Dict[str, Sequence[float]],
    relative_translation_threshold_m: float,
    relative_rotation_axis_threshold_deg: float,
) -> None:
    # EE constraint is the left->right tool relative transform drift.
    if relative_translation_errors_m and relative_rotation_axis_errors_deg["x"]:
        result["relative_transform_max_translation_m"] = float(max(relative_translation_errors_m))
        result["relative_transform_max_axis_angle_deg"] = {
            axis_name: float(max(relative_rotation_axis_errors_deg[axis_name]))
            for axis_name in AXIS_NAMES
        }
        result["relative_transform_ok"] = bool(
            result["relative_transform_max_translation_m"] <= relative_translation_threshold_m
            and all(
                float(result["relative_transform_max_axis_angle_deg"][axis_name]) <= relative_rotation_axis_threshold_deg
                for axis_name in AXIS_NAMES
            )
        )


def save_validation_plot(
    *,
    out_path: Optional[str],
    stage: int,
    target_label: Optional[str],
    position_res: Optional[float],
    rotation_res: Optional[float],
    dense_joint_validation_step_rad: float,
    collision_free: Optional[bool],
    joint_continuity_ok: Optional[bool],
    joint_continuity_max_delta_rad: Optional[float],
    joint_continuity_threshold_rad: float,
    joint_path_source: Optional[str],
    joint_path_reason: Optional[str],
    relative_translation_errors_m: Sequence[float],
    relative_rotation_axis_errors_deg: Dict[str, Sequence[float]],
    coarse_relative_translation_errors_m: Optional[Sequence[float]] = None,
    coarse_relative_rotation_axis_errors_deg: Optional[Dict[str, Sequence[float]]] = None,
    relative_translation_threshold_m: float,
    relative_rotation_axis_threshold_deg: float,
    collision_breakdown: Dict[str, Any],
    joint_path_deg: Optional[np.ndarray],
    original_joint_path_deg: Optional[np.ndarray] = None,
    wrap_segments: Optional[Sequence[Tuple[int, int, float]]] = None,
    save_plot: bool = True,
    show_plot: bool = False,
) -> Optional[str]:
    plt = import_matplotlib_pyplot(interactive=show_plot and not save_plot)
    if plt is None:
        return None

    fig, axes = plt.subplots(8, 1, figsize=(11, 22), sharex=False)
    title = (
        f"Stage {stage} validation | collisions: {status_label(collision_free)}"
        f" | joint continuity: {status_label(joint_continuity_ok)}"
    )
    if joint_continuity_max_delta_rad is not None:
        title += f" (max dq={joint_continuity_max_delta_rad:.3f} rad, thresh={joint_continuity_threshold_rad:.3f})"
    fig.suptitle(title)
    subtitle_parts = []
    if target_label:
        subtitle_parts.append(f"target: {target_label}")
    if position_res is not None:
        subtitle_parts.append(f"position res: {position_res:g}")
    if rotation_res is not None:
        subtitle_parts.append(f"rotation res: {rotation_res:g}")
    subtitle_parts.append(
        f"dense joint validation step: {dense_joint_validation_step_rad:.6f} rad "
        f"({np.degrees(dense_joint_validation_step_rad):.3f} deg)"
    )
    fig.text(0.5, 0.955, " | ".join(subtitle_parts), ha="center", va="top", fontsize=9)

    def path_fraction_xs(num_samples: int) -> np.ndarray:
        if num_samples <= 1:
            return np.zeros(max(0, num_samples), dtype=float)
        return np.linspace(0.0, 1.0, num=num_samples)

    def set_shared_nonnegative_ylim(axis_pair, value_sequences: Sequence[Sequence[float]]) -> None:
        values = [np.asarray(seq, dtype=float).reshape(-1) for seq in value_sequences if len(seq) > 0]
        if not values:
            return
        max_value = float(max(np.max(values_arr) for values_arr in values))
        upper = max(max_value * 1.05, 1e-9)
        for axis in axis_pair:
            axis.set_ylim(0.0, upper)

    refined_translation_mm = 1000.0 * np.asarray(relative_translation_errors_m, dtype=float)
    coarse_translation_mm = (
        1000.0 * np.asarray(coarse_relative_translation_errors_m, dtype=float)
        if coarse_relative_translation_errors_m
        else np.asarray([], dtype=float)
    )

    if relative_translation_errors_m:
        xs = path_fraction_xs(len(relative_translation_errors_m))
        axes[0].plot(
            xs,
            refined_translation_mm,
            color="#1f77b4",
            linewidth=1.8,
            label="refined",
        )
        axes[0].set_ylabel("Translation drift (mm)")
        axes[0].set_title("Left-right end-effector relative translation drift (refined smoothed path)")
        axes[0].legend(loc="best")
    else:
        axes[0].axis("off")
        reason = joint_path_reason or "joint path unavailable"
        axes[0].text(0.5, 0.5, f"No refined translation drift plot available.\nReason: {reason}", ha="center", va="center", fontsize=11)

    if coarse_relative_translation_errors_m:
        coarse_translation_xs = path_fraction_xs(len(coarse_relative_translation_errors_m))
        axes[1].plot(
            coarse_translation_xs,
            coarse_translation_mm,
            color="#ff7f0e",
            linewidth=1.6,
            marker="o",
            markersize=4.0,
            markeredgewidth=0.0,
            label="coarse before refinement",
        )
        axes[1].set_ylabel("Translation drift (mm)")
        axes[1].set_title("Left-right end-effector relative translation drift (smoothed path before refinement)")
        axes[1].legend(loc="best")
    else:
        axes[1].axis("off")
        reason = joint_path_reason or "joint path unavailable"
        axes[1].text(0.5, 0.5, f"No coarse translation drift plot available.\nReason: {reason}", ha="center", va="center", fontsize=11)

    set_shared_nonnegative_ylim((axes[0], axes[1]), (refined_translation_mm, coarse_translation_mm))

    color_map = {"x": "#d9534f", "y": "#5cb85c", "z": "#337ab7"}
    refined_rotation_values = [
        np.asarray(relative_rotation_axis_errors_deg[axis_name], dtype=float)
        for axis_name in AXIS_NAMES
        if relative_rotation_axis_errors_deg.get(axis_name)
    ]
    coarse_rotation_values = [
        np.asarray(coarse_relative_rotation_axis_errors_deg[axis_name], dtype=float)
        for axis_name in AXIS_NAMES
        if coarse_relative_rotation_axis_errors_deg and coarse_relative_rotation_axis_errors_deg.get(axis_name)
    ]
    if relative_rotation_axis_errors_deg.get("x"):
        xs = path_fraction_xs(len(relative_rotation_axis_errors_deg["x"]))
        for axis_name in AXIS_NAMES:
            axes[2].plot(
                xs,
                relative_rotation_axis_errors_deg[axis_name],
                color=color_map[axis_name],
                linewidth=1.8,
                label=f"{axis_name}-axis",
            )
        axes[2].set_ylabel("Axis drift (deg)")
        axes[2].set_title("Left-right end-effector relative rotation drift by axis (refined smoothed path)")
        axes[2].legend(loc="best")
    else:
        axes[2].axis("off")
        reason = joint_path_reason or "joint path unavailable"
        axes[2].text(0.5, 0.5, f"No refined rotation drift plot available.\nReason: {reason}", ha="center", va="center", fontsize=11)

    if coarse_relative_rotation_axis_errors_deg and coarse_relative_rotation_axis_errors_deg.get("x"):
        coarse_rotation_xs = path_fraction_xs(len(coarse_relative_rotation_axis_errors_deg["x"]))
        for axis_name in AXIS_NAMES:
            axes[3].plot(
                coarse_rotation_xs,
                coarse_relative_rotation_axis_errors_deg[axis_name],
                color=color_map[axis_name],
                linewidth=1.4,
                marker="o",
                markersize=4.0,
                markeredgewidth=0.0,
                label=f"{axis_name}-axis",
            )
        axes[3].set_ylabel("Axis drift (deg)")
        axes[3].set_title("Left-right end-effector relative rotation drift by axis (smoothed path before refinement)")
        axes[3].legend(loc="best")
    else:
        axes[3].axis("off")
        reason = joint_path_reason or "joint path unavailable"
        axes[3].text(0.5, 0.5, f"No coarse rotation drift plot available.\nReason: {reason}", ha="center", va="center", fontsize=11)

    set_shared_nonnegative_ylim((axes[2], axes[3]), [*refined_rotation_values, *coarse_rotation_values])

    for axis in axes[:4]:
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Path fraction")

    # Wrap segments = ±2π discontinuities in the raw joint path. Plotted as
    # red dashed verticals so the operator can see exactly which command
    # transitions the UR driver will see as huge implied velocities (e.g.
    # 21.9 rad/s in 0.002s = 0.044 rad/cycle = ~2π spread over the segment).
    wrap_segments = list(wrap_segments or [])
    wrap_labels_full = get_joint_labels(12) if (joint_path_deg is not None and joint_path_deg.shape[1] == 12) else None

    def _draw_wrap_markers(axis, x_indices_iter, *, annotate_top: bool, joint_filter: Optional[range] = None) -> None:
        if not wrap_segments:
            return
        drawn = False
        y_top = axis.get_ylim()[1]
        for seg_idx, joint_idx, dq_rad in wrap_segments:
            if joint_filter is not None and joint_idx not in joint_filter:
                continue
            for xi in x_indices_iter(seg_idx):
                axis.axvline(xi, color="#d9534f", linewidth=0.9, linestyle="--",
                             alpha=0.7, label=("wrap segment (|dq|>π)" if not drawn else None))
                drawn = True
            if annotate_top:
                jl = wrap_labels_full[joint_idx] if wrap_labels_full else f"q{joint_idx+1}"
                axis.annotate(f"{jl}\nseg {seg_idx}->{seg_idx+1}\nΔ={np.degrees(dq_rad):+.0f}°",
                              xy=(seg_idx, y_top), xytext=(seg_idx, y_top),
                              fontsize=7, color="#d9534f", ha="center", va="top")

    # Raw command-space values (no ±π unwrap), so wraps appear as visible
    # vertical jumps — exactly what the UR driver consumes.
    if joint_path_deg is not None and joint_path_deg.size > 0:
        xs = np.arange(joint_path_deg.shape[0], dtype=int)
        labels = get_joint_labels(joint_path_deg.shape[1])
        for joint_idx in range(joint_path_deg.shape[1]):
            axes[4].plot(
                xs,
                joint_path_deg[:, joint_idx],
                linewidth=1.2,
                marker="o",
                markersize=2.5,
                markeredgewidth=0.0,
                label=labels[joint_idx],
            )
        axes[4].set_ylabel("Joint angle (deg)")
        axes[4].set_xlabel("Waypoint index")
        axes[4].set_title("Joint value evolution on smoothed path (raw)")
        _draw_wrap_markers(axes[4], lambda s: (s, s + 1), annotate_top=True)
        axes[4].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, ncol=1)
    else:
        axes[4].axis("off")
        reason = joint_path_reason or "joint path unavailable"
        axes[4].text(0.5, 0.5, f"No joint evolution plot available.\nReason: {reason}", ha="center", va="center", fontsize=11)

    if original_joint_path_deg is not None and original_joint_path_deg.size > 0:
        xs = np.arange(original_joint_path_deg.shape[0], dtype=int)
        labels = get_joint_labels(original_joint_path_deg.shape[1])
        for joint_idx in range(original_joint_path_deg.shape[1]):
            axes[5].plot(
                xs,
                original_joint_path_deg[:, joint_idx],
                linewidth=1.2,
                marker="o",
                markersize=2.5,
                markeredgewidth=0.0,
                label=labels[joint_idx],
            )
        axes[5].set_ylabel("Joint angle (deg)")
        axes[5].set_xlabel("Waypoint index")
        axes[5].set_title("Joint value evolution on original unsmoothed path (raw)")
        _draw_wrap_markers(axes[5], lambda s: (s, s + 1), annotate_top=True)
        axes[5].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, ncol=1)
    else:
        axes[5].axis("off")
        axes[5].text(0.5, 0.5, "No original unsmoothed joint path available.", ha="center", va="center", fontsize=11)

    # axes[6], axes[7]: per-segment joint deltas (waypoint i -> i+1) for left
    # and right arm. Uses the raw command-space joint path, so ±2π wrap
    # segments show up as ±360° spikes — exactly what the UR driver sees.
    delta_source = joint_path_deg
    if (delta_source is not None and delta_source.size > 0
            and delta_source.shape[0] >= 2 and delta_source.shape[1] >= 12):
        delta_deg = np.diff(delta_source, axis=0)
        seg_xs = np.arange(delta_deg.shape[0], dtype=int)
        labels = get_joint_labels(delta_source.shape[1])
        for arm_idx, (slot, title, joint_slice) in enumerate((
            (6, "Left arm joint deltas between consecutive waypoints (raw)", slice(0, 6)),
            (7, "Right arm joint deltas between consecutive waypoints (raw)", slice(6, 12)),
        )):
            for joint_idx in range(joint_slice.start, joint_slice.stop):
                axes[slot].plot(
                    seg_xs,
                    delta_deg[:, joint_idx],
                    linewidth=1.2,
                    marker="o",
                    markersize=2.5,
                    markeredgewidth=0.0,
                    label=labels[joint_idx],
                )
            axes[slot].axhline(0.0, color="#888888", linewidth=0.6, linestyle="--")
            axes[slot].set_ylabel("Δ joint (deg)")
            axes[slot].set_xlabel("Segment index (i -> i+1)")
            axes[slot].set_title(title)
            _draw_wrap_markers(
                axes[slot], lambda s: (s,), annotate_top=True,
                joint_filter=range(joint_slice.start, joint_slice.stop),
            )
            axes[slot].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, ncol=1)
    else:
        for slot in (6, 7):
            axes[slot].axis("off")
        reason = joint_path_reason or "joint path too short or wrong width"
        axes[6].text(0.5, 0.5, f"No joint delta plot available.\nReason: {reason}", ha="center", va="center", fontsize=11)

    details = [
        f"joint path source: {joint_path_source or 'n/a'}",
        f"bar-robot collisions: {collision_breakdown['bar_robot']['count']}",
        f"robot self-collisions: {collision_breakdown['robot_self']['count']}",
        f"robot-static collisions: {collision_breakdown['robot_static']['count']}",
        f"bar-static collisions: {collision_breakdown['bar_static']['count']}",
    ]
    fig.text(0.02, 0.02, " | ".join(details), fontsize=9)

    fig.tight_layout(rect=(0.0, 0.04, 0.86, 0.93))
    saved_path = None
    if save_plot:
        if out_path is None:
            raise ValueError("out_path is required when save_plot=True")
        fig.savefig(out_path, dpi=180)
        saved_path = out_path
    if show_plot:
        try:
            fig.canvas.manager.set_window_title(f"Stage {stage} validation")
            plt.show(block=False)
            plt.pause(0.001)
        except Exception as exc:
            logger.warning(f"Could not display validation plot: {exc}")
            plt.close(fig)
    else:
        plt.close(fig)
    return saved_path


def validate_stage_trajectory(
    *,
    stage: int,
    scene: Dict[str, Any],
    path: Optional[Sequence[PoseLike]],
    joint_path: Optional[Sequence[FullConf]],
    joint_path_source: Optional[str],
    joint_path_reason: Optional[str],
    urdf_path: str,
    srdf_path: str,
    grasp_mask_links: Sequence[str],
    joint_continuity_threshold_rad: float = 10.0 * np.pi / 180.0,
    relative_translation_threshold_m: float = 1e-3,
    relative_rotation_axis_threshold_deg: float = float(np.degrees(1e-2)),
    dense_joint_validation_step_rad: float = DEFAULT_DENSE_JOINT_VALIDATION_STEP_RAD,
    target_label: Optional[str] = None,
    position_res: Optional[float] = None,
    rotation_res: Optional[float] = None,
    original_joint_path: Optional[Sequence[FullConf]] = None,
    use_angle_normalization: bool = False,
    reports_dir: str = REPORTS_DIR,
    skip_relative_transform: bool = False,
    skip_dense_collision_checks: bool = False,
    save_plot: bool = True,
    show_plot: bool = False,
    bar_pose_source: str = "left_grasp",
) -> Dict[str, Any]:
    if bar_pose_source not in {"left_grasp", "path"}:
        raise ValueError(f"Unsupported bar_pose_source: {bar_pose_source!r}")

    out_path = None
    if save_plot:
        os.makedirs(reports_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(reports_dir, f"trajectory_validation_stage{stage}_{timestamp}.png")

    result: Dict[str, Any] = {
        "plot_path": None,
        "path_waypoints": 0 if path is None else len(path),
        "joint_path_waypoints": 0 if joint_path is None else len(joint_path),
        "dense_joint_validation_waypoints": 0,
        "dense_joint_validation_step_rad": float(dense_joint_validation_step_rad),
        "joint_path_source": joint_path_source,
        "joint_path_reason": joint_path_reason,
        "collision_free": None,
        "collision_breakdown": {
            "bar_robot": {"count": 0, "first_index": None, "first_dense_index": None},
            "robot_self": {"count": 0, "first_index": None, "first_dense_index": None},
            "robot_static": {"count": 0, "first_index": None, "first_dense_index": None},
            "bar_static": {"count": 0, "first_index": None, "first_dense_index": None},
        },
        "collision_failures": [],
        "dense_collision_checks_skipped": bool(skip_dense_collision_checks),
        "joint_continuity_ok": None,
        "joint_continuity_threshold_rad": float(joint_continuity_threshold_rad),
        "joint_continuity_max_delta_rad": None,
        "joint_continuity_first_bad_step": None,
        "raw_wrap_segments": [],
        "raw_wrap_segment_count": 0,
        "relative_transform_ok": None,
        "relative_translation_threshold_m": float(relative_translation_threshold_m),
        "relative_rotation_axis_threshold_deg": float(relative_rotation_axis_threshold_deg),
        "relative_transform_max_translation_m": None,
        "relative_transform_max_axis_angle_deg": {axis_name: None for axis_name in AXIS_NAMES},
        "bar_pose_source": bar_pose_source,
    }

    if not path:
        result["joint_path_reason"] = result["joint_path_reason"] or "no_path"
        result["plot_path"] = save_validation_plot(
            out_path=out_path,
            stage=stage,
            target_label=target_label,
            position_res=position_res,
            rotation_res=rotation_res,
            dense_joint_validation_step_rad=dense_joint_validation_step_rad,
            collision_free=None,
            joint_continuity_ok=None,
            joint_continuity_max_delta_rad=None,
            joint_continuity_threshold_rad=joint_continuity_threshold_rad,
            joint_path_source=joint_path_source,
            joint_path_reason=result["joint_path_reason"],
            relative_translation_errors_m=[],
            relative_rotation_axis_errors_deg={axis_name: [] for axis_name in AXIS_NAMES},
            relative_translation_threshold_m=relative_translation_threshold_m,
            relative_rotation_axis_threshold_deg=relative_rotation_axis_threshold_deg,
            collision_breakdown=result["collision_breakdown"],
            joint_path_deg=None,
            original_joint_path_deg=None,
            save_plot=save_plot,
            show_plot=show_plot,
        )
        return result

    if joint_path is None or len(joint_path) != len(path):
        result["joint_path_reason"] = result["joint_path_reason"] or "joint_path_missing_or_length_mismatch"
        result["plot_path"] = save_validation_plot(
            out_path=out_path,
            stage=stage,
            target_label=target_label,
            position_res=position_res,
            rotation_res=rotation_res,
            dense_joint_validation_step_rad=dense_joint_validation_step_rad,
            collision_free=None,
            joint_continuity_ok=None,
            joint_continuity_max_delta_rad=None,
            joint_continuity_threshold_rad=joint_continuity_threshold_rad,
            joint_path_source=joint_path_source,
            joint_path_reason=result["joint_path_reason"],
            relative_translation_errors_m=[],
            relative_rotation_axis_errors_deg={axis_name: [] for axis_name in AXIS_NAMES},
            relative_translation_threshold_m=relative_translation_threshold_m,
            relative_rotation_axis_threshold_deg=relative_rotation_axis_threshold_deg,
            collision_breakdown=result["collision_breakdown"],
            joint_path_deg=None,
            original_joint_path_deg=None,
            save_plot=save_plot,
            show_plot=show_plot,
        )
        return result

    robot = scene["robot"]
    arm_joints = scene["arm_joints"]
    tool_link_left = scene["tool_link_left"]
    tool_link_right = scene["tool_link_right"]

    relative_translation_errors_m: list[float] = []
    relative_rotation_axis_errors_deg = {axis_name: [] for axis_name in AXIS_NAMES}
    base_relative_pose = None
    collision_keys = ("bar_robot", "robot_self", "robot_static", "bar_static")
    collision_flags = {key: False for key in collision_keys}

    # Validate the exact command-space values. No ±π unwrap — what we send is
    # what we display, matching what the UR controller consumes.
    command_joint_path = [np.asarray(conf, dtype=float) for conf in joint_path]
    joint_path_deg = joint_path_to_raw_deg(command_joint_path)

    # Detect ±2π wrap segments in the raw command path. These are
    # continuous-joint discontinuities that the UR driver sees as huge
    # implied velocities (External control speed limit violations).
    wrap_segments: list[Tuple[int, int, float]] = []
    if len(command_joint_path) >= 2:
        raw_arr = np.asarray(command_joint_path, dtype=float)
        raw_dq = np.diff(raw_arr, axis=0)
        wrap_mask = np.abs(raw_dq) > np.pi
        if wrap_mask.any():
            seg_idxs, joint_idxs = np.where(wrap_mask)
            wrap_segments = [(int(s), int(j), float(raw_dq[s, j]))
                             for s, j in zip(seg_idxs, joint_idxs)]
            result["raw_wrap_segments"] = [
                {"segment_index": s, "joint_index": j, "delta_rad": dq}
                for s, j, dq in wrap_segments
            ]
            result["raw_wrap_segment_count"] = len(wrap_segments)
            logger.warning(
                f"path_validation: {len(wrap_segments)} ±2π wrap segment(s) "
                f"detected in raw joint path (will trigger UR speed-limit violations): "
                f"{[(s, j, f'{float(np.degrees(dq)):+.1f}deg') for s, j, dq in wrap_segments]}"
            )
    original_joint_path_deg = None
    if original_joint_path is not None:
        command_original_joint_path = [np.asarray(conf, dtype=float) for conf in original_joint_path]
        original_joint_path_deg = joint_path_to_raw_deg(command_original_joint_path)
    if skip_relative_transform:
        coarse_relative_translation_errors_m = []
        coarse_relative_rotation_axis_errors_deg = {axis_name: [] for axis_name in AXIS_NAMES}
    else:
        coarse_relative_translation_errors_m, coarse_relative_rotation_axis_errors_deg = compute_relative_transform_drift(
            robot,
            arm_joints,
            tool_link_left,
            tool_link_right,
            command_joint_path,
        )
    update_joint_continuity_result(result, command_joint_path)
    if skip_dense_collision_checks:
        relative_translation_errors_m = list(coarse_relative_translation_errors_m)
        relative_rotation_axis_errors_deg = {
            axis_name: list(coarse_relative_rotation_axis_errors_deg[axis_name])
            for axis_name in AXIS_NAMES
        }
        result["dense_joint_validation_waypoints"] = 0
        result["dense_joint_validation_step_rad"] = 0.0
        update_relative_transform_result(
            result,
            relative_translation_errors_m,
            relative_rotation_axis_errors_deg,
            relative_translation_threshold_m,
            relative_rotation_axis_threshold_deg,
        )
        result["plot_path"] = save_validation_plot(
            out_path=out_path,
            stage=stage,
            target_label=target_label,
            position_res=position_res,
            rotation_res=rotation_res,
            dense_joint_validation_step_rad=0.0,
            collision_free=result["collision_free"],
            joint_continuity_ok=result["joint_continuity_ok"],
            joint_continuity_max_delta_rad=result["joint_continuity_max_delta_rad"],
            joint_continuity_threshold_rad=joint_continuity_threshold_rad,
            joint_path_source=joint_path_source,
            joint_path_reason=result["joint_path_reason"],
            relative_translation_errors_m=relative_translation_errors_m,
            relative_rotation_axis_errors_deg=relative_rotation_axis_errors_deg,
            coarse_relative_translation_errors_m=coarse_relative_translation_errors_m,
            coarse_relative_rotation_axis_errors_deg=coarse_relative_rotation_axis_errors_deg,
            relative_translation_threshold_m=relative_translation_threshold_m,
            relative_rotation_axis_threshold_deg=relative_rotation_axis_threshold_deg,
            collision_breakdown=result["collision_breakdown"],
            joint_path_deg=joint_path_deg,
            original_joint_path_deg=original_joint_path_deg,
            wrap_segments=wrap_segments,
            save_plot=save_plot,
            show_plot=show_plot,
        )
        return result

    bar_body = scene["bar_body"]
    bar_from_left = pp.invert(scene["grasp_bar_from_left"])
    static_obstacles = [body for body in scene["collision_obstacles"] if body != robot]

    robot_model = RobotModel.from_urdf_file(urdf_path)
    semantics = RobotSemantics.from_srdf_file(srdf_path, robot_model)
    disabled_collisions = get_disabled_collisions_from_link_names(robot, semantics.disabled_collisions)
    extra_disabled_collisions = []
    for link_name in grasp_mask_links:
        if not pp.has_link(robot, link_name):
            continue
        extra_disabled_collisions.append(((robot, pp.link_from_name(robot, link_name)), (bar_body, pp.BASE_LINK)))

    robot_self_collision_fn = pp.get_collision_fn(
        robot,
        arm_joints,
        obstacles=[],
        attachments=[],
        self_collisions=True,
        disabled_collisions=disabled_collisions,
        extra_disabled_collisions=[],
        max_distance=0.0,
    )
    robot_static_collision_fn = pp.get_collision_fn(
        robot,
        arm_joints,
        obstacles=static_obstacles,
        attachments=[],
        self_collisions=False,
        disabled_collisions=disabled_collisions,
        extra_disabled_collisions=[],
        max_distance=0.0,
    )
    bar_robot_collision_fn = pp.get_floating_body_collision_fn(
        bar_body,
        obstacles=[robot],
        disabled_collisions=extra_disabled_collisions,
    )
    bar_static_collision_fn = pp.get_floating_body_collision_fn(
        bar_body,
        obstacles=static_obstacles,
        disabled_collisions=[],
    )
    dense_joint_path, dense_metadata = refine_joint_path_for_validation(
        command_joint_path,
        dense_joint_validation_step_rad,
        False,
    )
    result["dense_joint_validation_waypoints"] = len(dense_joint_path)

    # Replay a dense joint-space trajectory for collision and end-effector drift
    # checks. Constrained planning derives the bar pose from left-tool FK and the
    # grasp transform; free-space planning supplies a static bar pose path.
    for idx, (conf, sample_meta) in enumerate(zip(dense_joint_path, dense_metadata)):
        pp.set_joint_positions(robot, arm_joints, conf)
        world_from_left = pp.get_link_pose(robot, tool_link_left)
        if bar_pose_source == "path":
            bar_pose = path[sample_meta["waypoint_index"]]
        else:
            bar_pose = pp.multiply(world_from_left, bar_from_left)
        pp.set_pose(bar_body, bar_pose)

        collision_fn_map = {
            "bar_robot": (bar_robot_collision_fn, bar_pose),
            "robot_self": (robot_self_collision_fn, conf),
            "robot_static": (robot_static_collision_fn, conf),
            "bar_static": (bar_static_collision_fn, bar_pose),
        }
        hit_map = {key: bool(fn(arg)) for key, (fn, arg) in collision_fn_map.items()}
        failure_record = {
            "waypoint_index": sample_meta["waypoint_index"],
            "dense_waypoint_index": idx,
            "segment_index": sample_meta["segment_index"],
            "segment_alpha": sample_meta["segment_alpha"],
            "path_fraction": sample_meta["path_fraction"],
            "collision_keys": [],
            "pose_position": np.asarray(bar_pose[0], dtype=float).tolist(),
            "pose_quaternion": np.asarray(bar_pose[1], dtype=float).tolist(),
            "bar_pose": {
                "position": np.asarray(bar_pose[0], dtype=float).tolist(),
                "quaternion": np.asarray(bar_pose[1], dtype=float).tolist(),
            },
            "joint_values": np.asarray(conf, dtype=float).tolist(),
        }
        for key, hit in hit_map.items():
            if not hit:
                continue
            collision_flags[key] = True
            result["collision_breakdown"][key]["count"] += 1
            if result["collision_breakdown"][key]["first_index"] is None:
                result["collision_breakdown"][key]["first_index"] = sample_meta["waypoint_index"]
                result["collision_breakdown"][key]["first_dense_index"] = idx
            failure_record["collision_keys"].append(key)
        if failure_record["collision_keys"]:
            result["collision_failures"].append(failure_record)

        if not skip_relative_transform:
            world_from_right = pp.get_link_pose(robot, tool_link_right)
            relative_pose = pp.multiply(pp.invert(world_from_left), world_from_right)
            if base_relative_pose is None:
                base_relative_pose = relative_pose
            relative_translation_errors_m.append(float(np.linalg.norm(np.asarray(relative_pose[0]) - np.asarray(base_relative_pose[0]))))
            axis_diffs_deg = rotation_axis_differences_deg(base_relative_pose[1], relative_pose[1])
            for axis_name in AXIS_NAMES:
                relative_rotation_axis_errors_deg[axis_name].append(axis_diffs_deg[axis_name])

    pp.set_joint_positions(robot, arm_joints, command_joint_path[-1])
    if bar_pose_source == "path":
        pp.set_pose(bar_body, path[-1])
    else:
        pp.set_pose(bar_body, pp.multiply(pp.get_link_pose(robot, tool_link_left), bar_from_left))

    result["collision_free"] = not any(collision_flags.values())
    update_relative_transform_result(
        result,
        relative_translation_errors_m,
        relative_rotation_axis_errors_deg,
        relative_translation_threshold_m,
        relative_rotation_axis_threshold_deg,
    )

    result["plot_path"] = save_validation_plot(
        out_path=out_path,
        stage=stage,
        target_label=target_label,
        position_res=position_res,
        rotation_res=rotation_res,
        dense_joint_validation_step_rad=dense_joint_validation_step_rad,
        collision_free=result["collision_free"],
        joint_continuity_ok=result["joint_continuity_ok"],
        joint_continuity_max_delta_rad=result["joint_continuity_max_delta_rad"],
        joint_continuity_threshold_rad=joint_continuity_threshold_rad,
        joint_path_source=joint_path_source,
        joint_path_reason=result["joint_path_reason"],
        relative_translation_errors_m=relative_translation_errors_m,
        relative_rotation_axis_errors_deg=relative_rotation_axis_errors_deg,
        coarse_relative_translation_errors_m=coarse_relative_translation_errors_m,
        coarse_relative_rotation_axis_errors_deg=coarse_relative_rotation_axis_errors_deg,
        relative_translation_threshold_m=relative_translation_threshold_m,
        relative_rotation_axis_threshold_deg=relative_rotation_axis_threshold_deg,
        collision_breakdown=result["collision_breakdown"],
        joint_path_deg=joint_path_deg,
        original_joint_path_deg=original_joint_path_deg,
        wrap_segments=wrap_segments,
        save_plot=save_plot,
        show_plot=show_plot,
    )
    return result
