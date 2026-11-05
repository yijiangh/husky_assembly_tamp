"""The figure of a planned motion: every joint along the path, and the holding error where a bar is carried.

Three stacked panels, all against the concatenated waypoint index:

1. every joint's value along the whole path, with a dashed line where each
   movement hands over to the next;
2. the end-effector position error (mm) of the movements that carry the bar;
3. the same for the rotation error (deg).

Panels 2 and 3 use a log scale, because a healthy plan sits several orders of
magnitude under the tolerance, which a linear axis would flatten to zero. The
tolerance is drawn as a red dotted line, so "is this plan good" is just "does
every curve stay below the red line". The error is measured with the tamp path
validator's own drift maths, so the numbers mean the same thing as the ones
``validate_stage_trajectory`` reports.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pybullet_planning as pp
from matplotlib.transforms import blended_transform_factory

from husky_assembly_tamp.motion_planner.api import TOOL_LINK_LEFT, TOOL_LINK_RIGHT
from husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.path_validation import (
    compute_relative_transform_drift,
)

# * How far the two flanges may drift from their rigid relation along a carry.
EE_POSITION_TOLERANCE_MM = 0.5
EE_ROTATION_TOLERANCE_DEG = float(np.degrees(1e-2))

# * The movement roles that carry the bar in both grippers (their grasp must stay rigid).
CARRYING_ROLES = ("M1", "M2")


def short_joint_name(name: str) -> str:
    """Shorten a full joint name to its last two words, for plot labels.

    ``'left_ur_arm_shoulder_pan_joint'`` becomes ``'shoulder_pan'``: long
    enough to tell the six arm joints apart, short enough for a legend.

    Args:
        name (str): the full joint name.

    Returns:
        str: the shortened label.
    """
    base = re.sub(r"_joint$", "", str(name))
    return "_".join(base.split("_")[-2:])


def draw_movement_boundaries(axis, segments, *, at_top: bool, show_counts: bool,
                             max_index: Optional[int] = None) -> None:
    """Mark where each movement starts along a concatenated-waypoint x axis.

    A dashed vertical line at every hand-over plus a label naming the
    movement. Short movements next to long ones start almost on top of each
    other, so labels that would overprint are stepped down a row.

    Args:
        axis: the matplotlib axes to draw on.
        segments: ``[(role, start_state, path), ...]`` in playback order.
        at_top (bool): put the labels along the top edge (else the bottom).
        show_counts (bool): append each movement's waypoint count to its label.
        max_index (Optional[int]): stop before this waypoint index, for panels
            zoomed to part of the path. None labels every movement.
    """
    total = sum(len(path) for _role, _state, path in segments)
    span = max_index if max_index else total
    # x from the data (waypoint index), y from the axes box, so the rows sit at
    # a fixed height whatever the data range or log scale does.
    transform = blended_transform_factory(axis.transData, axis.transAxes)
    rows = (0.985, 0.945, 0.905, 0.865) if at_top else (0.015, 0.055, 0.095, 0.135)

    row = 0
    previous_boundary = None
    boundary = 0
    for role, _state, path in segments:
        if max_index is None or boundary < max_index:
            crowded = previous_boundary is not None and (boundary - previous_boundary) < 0.08 * span
            row = (row + 1) % len(rows) if crowded else 0
            axis.axvline(boundary, color="gray", ls="--", lw=1)
            axis.text(boundary + 0.2, rows[row], f"{role} ({len(path)})" if show_counts else role,
                      transform=transform, va="top" if at_top else "bottom", fontsize=8, color="#333")
            previous_boundary = boundary
        boundary += len(path)


def measure_grasp_rigidity(planner, path: Sequence[Sequence[float]],
                           joint_names_12: Sequence[str]) -> Tuple[List[float], List[float]]:
    """How much the two flanges drift apart along a bar-carrying movement.

    While the bar is held by both arms the transform from the left flange to
    the right one is physically fixed, so any change along the path means the
    planned motion would fight itself (or drop the bar). Measured against the
    FIRST waypoint of the movement.

    ! Moves the PyBullet robot as it takes forward kinematics; the caller
    ! restores the scene afterwards (``planner.set_robot_cell_state``).

    Args:
        planner: the PyBulletPlanner whose scene holds the robot.
        path (Sequence[Sequence[float]]): the movement's waypoints, twelve values each.
        joint_names_12 (Sequence[str]): the twelve arm joints, in order.

    Returns:
        Tuple[List[float], List[float]]: per waypoint, the position drift in
        millimetres and the worst per-axis rotation drift in degrees.
    """
    robot_puid = planner.client.robot_puid
    arm_joints = pp.joints_from_names(robot_puid, joint_names_12)
    translations_m, rotations_deg = compute_relative_transform_drift(
        robot_puid, arm_joints,
        pp.link_from_name(robot_puid, TOOL_LINK_LEFT),
        pp.link_from_name(robot_puid, TOOL_LINK_RIGHT),
        [np.asarray(q, dtype=float) for q in path],
    )
    position_mm = [float(v) * 1e3 for v in translations_m]
    # The three per-axis angles answer different questions (which way did the
    # grasp twist); for a pass/fail curve the worst one is what matters.
    rotation_deg = [max(rotations_deg["x"][i], rotations_deg["y"][i], rotations_deg["z"][i])
                    for i in range(len(position_mm))]
    return position_mm, rotation_deg


def measure_ee_constraints(planner, segments, joint_names_12: Sequence[str]) -> List[dict]:
    """Run the holding check on every movement that carries the bar (``CARRYING_ROLES``).

    Args:
        planner: the PyBulletPlanner whose scene holds the robot.
        segments: ``[(role, start_state, path), ...]`` in playback order.
        joint_names_12 (Sequence[str]): the twelve arm joints, in order.

    Returns:
        List[dict]: one entry per check, each with ``role``, ``label``,
        ``offset`` (its first waypoint's index in the concatenated path),
        ``position_mm`` and ``rotation_deg`` (per-waypoint error lists).
    """
    checks = []
    offset = 0
    for role, _state, path in segments:
        if role in CARRYING_ROLES:
            position_mm, rotation_deg = measure_grasp_rigidity(planner, path, joint_names_12)
            checks.append({"role": role, "label": f"{role} grasp rigidity", "offset": offset,
                           "position_mm": position_mm, "rotation_deg": rotation_deg})
        offset += len(path)
    return checks


def print_ee_constraint_summary(checks: Sequence[dict]) -> bool:
    """Print the worst error of each holding check, with a pass/fail mark.

    Args:
        checks (Sequence[dict]): the entries from :func:`measure_ee_constraints`.

    Returns:
        bool: True when every check stayed inside both tolerances.
    """
    if not checks:
        print("[ee-check] no bar-carrying movements in this plan.")
        return True
    print("\n=============== END-EFFECTOR CONSTRAINT CHECK ===============")
    print(f"  tolerances: {EE_POSITION_TOLERANCE_MM:.3f} mm / {EE_ROTATION_TOLERANCE_DEG:.3f} deg")
    all_ok = True
    for check in checks:
        worst_mm = max(check["position_mm"]) if check["position_mm"] else 0.0
        worst_deg = max(check["rotation_deg"]) if check["rotation_deg"] else 0.0
        ok = worst_mm <= EE_POSITION_TOLERANCE_MM and worst_deg <= EE_ROTATION_TOLERANCE_DEG
        all_ok = all_ok and ok
        print(f"  {check['label']:22s}: worst {worst_mm:9.4f} mm / {worst_deg:8.4f} deg  within tolerance = {ok}")
    print("=============================================================")
    return all_ok


def plot_planned_motion(planner, segments, joint_names_12: Sequence[str], out_path: str, *,
                        title: str = "") -> Optional[List[dict]]:
    """Save the figure of a planned motion (see the module docstring), and print its holding check.

    Args:
        planner: the PyBulletPlanner whose scene holds the robot (used for FK).
        segments: ``[(role, start_state, path), ...]`` in playback order.
        joint_names_12 (Sequence[str]): the twelve arm joints, in order.
        out_path (str): the png to write.
        title (str): extra text for the figure title, e.g. the action id.

    Returns:
        Optional[List[dict]]: the holding checks (see :func:`measure_ee_constraints`),
        or None when there was nothing to plot.
    """
    waypoints = np.asarray([wp for _role, _state, path in segments for wp in path], dtype=float)
    if waypoints.size == 0:
        print("[plot] no waypoints to plot.")
        return None

    checks = measure_ee_constraints(planner, segments, joint_names_12)
    # Forward kinematics left the robot at the last waypoint it evaluated; put the scene back.
    planner.set_robot_cell_state(segments[-1][1])

    fig, (ax_joints, ax_position, ax_rotation) = plt.subplots(
        3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [2.0, 1.0, 1.0]})

    # * Panel 1: joint values along the concatenated path. A dot per waypoint
    # * shows how finely the planner sampled, which the line alone hides.
    colors = plt.get_cmap("tab20")
    x_all = np.arange(waypoints.shape[0])
    for j in range(waypoints.shape[1]):
        ax_joints.plot(x_all, waypoints[:, j], lw=1.2, color=colors(j % 20), marker="o", markersize=1.6,
                       markeredgewidth=0.0,
                       label=f"[{'L' if j < 6 else 'R'}] {short_joint_name(joint_names_12[j])}")
    draw_movement_boundaries(ax_joints, segments, at_top=True, show_counts=True)
    ax_joints.set_ylabel("joint value (rad)")
    ax_joints.set_title(f"Planned trajectory{' -- ' + title if title else ''}")
    ax_joints.legend(loc="upper right", fontsize=7, ncol=2)

    # * Panels 2 and 3: the holding errors. A log axis cannot show an exact
    # * zero, so the curves are floored just under the smallest interesting error.
    floor = 1e-6
    last_constrained = max((c["offset"] + len(c["position_mm"]) for c in checks), default=0)
    for axis, key, tolerance, unit in ((ax_position, "position_mm", EE_POSITION_TOLERANCE_MM, "mm"),
                                       (ax_rotation, "rotation_deg", EE_ROTATION_TOLERANCE_DEG, "deg")):
        for check in checks:
            values = np.maximum(np.asarray(check[key], dtype=float), floor)
            axis.plot(np.arange(len(values)) + check["offset"], values, lw=1.4, label=check["label"])
        axis.axhline(tolerance, color="red", ls=":", lw=1.2, label=f"tolerance {tolerance:.3g} {unit}")
        axis.set_yscale("log")
        draw_movement_boundaries(axis, segments, at_top=False, show_counts=False, max_index=last_constrained)
        axis.set_ylabel(f"EE {'position' if unit == 'mm' else 'rotation'} error ({unit})")
        axis.grid(True, which="both", alpha=0.25)
        if checks:
            axis.legend(loc="upper right", fontsize=7, ncol=2)
            # Zoom to the movements that carry the bar: an unconstrained tail
            # would squeeze the curves that matter into a sliver.
            margin = max(1.0, 0.02 * last_constrained)
            axis.set_xlim(-margin, last_constrained - 1 + margin)
        else:
            axis.text(0.5, 0.5, "no bar-carrying movements in this plan", transform=axis.transAxes,
                      ha="center", va="center", color="#777")

    constrained_roles = sorted({c["role"] for c in checks})
    ax_joints.set_xlabel("waypoint index (concatenated " + " -> ".join(role for role, _s, _p in segments) + ")")
    ax_rotation.set_xlabel("waypoint index -- same axis as above, zoomed to the carrying movement(s) "
                           f"{', '.join(constrained_roles) if constrained_roles else '(none)'}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[plot] planned-motion figure saved -> {out_path}")
    print_ee_constraint_summary(checks)
    return checks
