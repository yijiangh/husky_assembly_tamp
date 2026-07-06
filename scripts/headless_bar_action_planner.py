"""Headless planner + slider replay for a single BarAssemblyAction movement.

Mirrors the planning + replay workflow of
``husky-assembly-teleop/scripts/headless_live_monitor_test.py`` but stays
platform-independent and has no dependency on ``husky_assembly_teleop`` or ros2.

Goal: debug how Rhino-generated ``BarAssemblyAction`` JSONs load into
compas_fab + pybullet, run planning, and confirm collision + ACM setup
are correct end-to-end.

CLI:
    python scripts/headless_bar_action_planner.py [<data_root>]
        --bar-action B6.json
        --movement {M1,M2,M3,M4}
        [--gui]
        [--max-time 60]
        [--no-replay]
        [--probe-endpoints]   # M1: report start/goal feasibility, skip the RRT
        [--diagnosis]         # M4: draw birrt trees live (needs --gui); no
                              # LockRenderer, timeout-only stop
        [--load {clean,solved}]  # clean = Rhino export; solved = reuse the
                                 # half-solved sidecar and plan only the rest
        [--cell <RobotCell.json>]

Default ``<data_root>`` matches ``tests/debug_load_bar_action.py``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


# This script lives in the husky_assembly_tamp repo at ``scripts/``. All the
# packages it needs are sibling checkouts under the parent ``external/`` folder,
# so locate them relative to this file:
#   .../external/husky_assembly_tamp/scripts/  <- SCRIPT_DIR (this file's folder)
#   .../external/husky_assembly_tamp/          <- TAMP_SRC (the tamp package root)
#   .../external/                              <- EXTERNAL_DIR (holds the siblings)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ``TESTS_DIR`` is kept as an alias of the script folder because helpers below
# still write their output figures next to the script under this name.
TESTS_DIR = SCRIPT_DIR
TAMP_SRC = os.path.dirname(SCRIPT_DIR)
EXTERNAL_DIR = os.path.dirname(TAMP_SRC)
CFAB_SRC = os.path.join(EXTERNAL_DIR, "compas_fab", "src")
RSDS_SRC = os.path.join(EXTERNAL_DIR, "rs_data_structure")

for _p in (SCRIPT_DIR, TAMP_SRC, CFAB_SRC, RSDS_SRC):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Home-configuration constants come from the ``config`` module that sits next
# to this script, so the planner stays standalone (no dependency on
# husky_assembly_teleop or ros2). SCRIPT_DIR is already on sys.path above, so
# ``config`` resolves.
from types import SimpleNamespace  # noqa: E402
from config import HUSKY_DUAL_ARM_HOME_CONF_12  # noqa: E402

_config = SimpleNamespace(
    HUSKY_DUAL_ARM_HOME_CONF_12=list(HUSKY_DUAL_ARM_HOME_CONF_12),
    HOME_CONF_LEFT_6=list(HUSKY_DUAL_ARM_HOME_CONF_12[:6]),
    HOME_CONF_RIGHT_6=list(HUSKY_DUAL_ARM_HOME_CONF_12[6:]),
)

# Movement dataclasses. Imported here (after ``core`` is cached above) rather
# than inside functions: they are cheap and carry no import-order risk of
# their own now that ``core`` has already loaded.
from rs_data_structure.bar_action import (  # noqa: E402
    IndependentDualArmFreeMovement,
    EndEffectorConstrainedDualArmFreeMovement,
    EndEffectorConstrainedDualArmLinearMovement,
    IndependentDualArmLinearMovement,
)

# PyBullet + compas_fab backends and the planner API. Heavier than the imports
# above, but kept at module top for clarity. The CLI arg/path validation in
# main() still runs first, so a bad invocation just pays the import cost.
import pybullet  # noqa: E402
import pybullet_planning as pp  # noqa: E402
from compas.data import json_dump, json_load  # noqa: E402
from compas_fab.backends import (  # noqa: E402
    CollisionCheckError,
    PyBulletClient,
    PyBulletPlanner,
)
from husky_assembly_tamp.motion_planner.api import (  # noqa: E402
    _ARM_SUFFIXES,
    TOOL_LINK_LEFT,
    TOOL_LINK_RIGHT,
    _bar_body_id,
    _build_cfab_collision_fn,
    _collect_obstacle_puids,
    _conf12_from_state,
    _conf12_from_target,
    _derive_constrained_start_for_plan,
    plan_constrained_dual_arm,
    plan_constrained_dual_arm_linear,
    plan_dual_arm_linear_independent,
    plan_free_dual_arm,
)
from husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.core import (  # noqa: E402
    validate_dual_arm_bar_pose,
)


# Insync gdrive location of the design-study data, per machine. Pick whichever
# exists so the same script runs on both the Windows/Rhino box and the Linux box.
_DATA_ROOT_CANDIDATES = (
    # Linux (robot / ros2 machine)
    "/home/su/Insync/yijiang94817@gmail.com"
    "/Google Drive - Shared with me/2025-03 Husky Assembly/data_design_study",
    # Windows (Rhino design machine)
    r"C:\Users\yijiangh\Insync\yijiang94817@gmail.com"
    r"\Google Drive - Shared with me\2025-03 Husky Assembly\data_design_study",
)
DEFAULT_DATA_ROOT = next(
    (p for p in _DATA_ROOT_CANDIDATES if os.path.isdir(p)),
    _DATA_ROOT_CANDIDATES[0],
)
# DEFAULT_PROBLEM = "2026-05-14_foc_demo_reduced"
# DEFAULT_BAR_ACTION = "B226.json"
DEFAULT_PROBLEM = "2026-05-16_double_kissing_jig_demo"
DEFAULT_BAR_ACTION = "B6.json"

LEFT_GROUP = "base_left_arm_manipulator"
RIGHT_GROUP = "base_right_arm_manipulator"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_ROLE_RE = re.compile(r"_M([0-9])_")


def match_role(mv) -> Optional[str]:
    """Figure out which of the four movement roles a Movement plays.

    The reliable signal is the ``_M<n>_`` tag inside ``movement_id`` (for
    example ``'B6_M1_CDFM_home_to_approach'`` is role ``'M1'``). When the id has
    no such tag, fall back to the Python type of the movement, which is only
    good enough for the roles that are unambiguous by type.

    Args:
        mv: A Movement object from the BarAction (one of the
            ``Robotic*Movement`` dataclasses).

    Returns:
        Optional[str]: The role string ``'M0'``..``'M4'``, or ``None`` when the
        role cannot be decided (for example an untagged
        ``IndependentDualArmFreeMovement``, which is shared by M0 and M4).
    """
    mid = getattr(mv, "movement_id", None) or ""
    m = _ROLE_RE.search(mid)
    if m:
        return f"M{m.group(1)}"
    if isinstance(mv, EndEffectorConstrainedDualArmFreeMovement):
        return "M1"
    if isinstance(mv, EndEffectorConstrainedDualArmLinearMovement):
        return "M2"
    if isinstance(mv, IndependentDualArmLinearMovement):
        return "M3"
    if isinstance(mv, IndependentDualArmFreeMovement):
        # Shared by M0 and M4 -- needs the '_M<n>_' id tag to disambiguate.
        return None
    return None


def fill_missing_config(state, rcell, home_left: Sequence[float],
                        home_right: Sequence[float]) -> None:
    """Fill a state's robot configuration with the home pose when it has none.

    Test-only convenience. If ``state`` already carries a robot configuration,
    nothing happens. Otherwise a fresh zero configuration is built and the two
    arms are set to the given home joint values.

    Args:
        state: The RobotCellState to fill in place. May be ``None`` (ignored).
        rcell: The RobotCell, used to build a zero configuration and to look up
            the left/right arm joint names.
        home_left (Sequence[float]): Home joint values for the left arm.
        home_right (Sequence[float]): Home joint values for the right arm.

    Returns:
        None. ``state.robot_configuration`` is modified in place.
    """
    if state is None or state.robot_configuration is not None:
        return
    cfg = rcell.zero_full_configuration()
    left_names = list(rcell.get_configurable_joint_names(LEFT_GROUP))
    right_names = list(rcell.get_configurable_joint_names(RIGHT_GROUP))
    for n, v in zip(left_names, home_left):
        cfg[n] = float(v)
    for n, v in zip(right_names, home_right):
        cfg[n] = float(v)
    state.robot_configuration = cfg


def start_planner(rcell, *, use_gui: bool) -> Tuple[PyBulletClient, PyBulletPlanner]:
    """Boot a PyBullet client and planner with ``rcell`` already loaded.

    Args:
        rcell: The RobotCell to load into the planning scene.
        use_gui (bool): Open the interactive PyBullet window when True; run
            headless (DIRECT) when False. GUI mode also turns on the right-side
            debug parameter panel that the replay sliders live in.

    Returns:
        Tuple[PyBulletClient, PyBulletPlanner]: The connected client and the
        planner bound to it.
    """
    # enable_debug_gui=True exposes the right-side parameter panel that
    # addUserDebugParameter sliders live in (cfab defaults this to False).
    client = PyBulletClient(
        connection_type="gui" if use_gui else "direct",
        verbose=True,
        enable_debug_gui=use_gui,
    )
    client.__enter__()
    pp.set_client(client.client_id)
    pp.CLIENTS[client.client_id] = True if use_gui else None
    planner = PyBulletPlanner(client)
    t0 = time.time()
    with pp.LockRenderer(False):
        planner.set_robot_cell(rcell)
    print(f"[pb] set_robot_cell: {time.time() - t0:.2f}s")
    return client, planner


def color_rigid_body(planner, rb_name: str, rgba: Sequence[float]) -> None:
    """Re-tint every PyBullet sub-body of a rigid body to one color.

    Visual only — this does not change collision behaviour. The color sticks
    across ``set_robot_cell_state`` calls (compas_fab only refreshes pose, not
    visual properties), so one call after the cell loads keeps the active bar
    the same color through the whole replay.

    Args:
        planner: The PyBulletPlanner whose scene holds the rigid body.
        rb_name (str): Rigid-body name to recolor.
        rgba (Sequence[float]): Red, green, blue, alpha in the 0..1 range.

    Returns:
        None.
    """
    cid = planner.client.client_id
    for body_id in planner.client.rigid_bodies_puids[rb_name]:
        pybullet.changeVisualShape(
            body_id, -1, rgbaColor=list(rgba),
            physicsClientId=cid,
        )


def check_collision(planner, state, *, label: str = "") -> bool:
    """Run a full-report collision check on one cell state.

    Args:
        planner: The PyBulletPlanner to check against.
        state: The RobotCellState to test.
        label (str): Optional tag printed with the result to identify which
            movement or step this check belongs to.

    Returns:
        bool: True when no collisions are reported, False when at least one
        colliding pair is found (each pair is printed).
    """
    tag = f" [{label}]" if label else ""
    try:
        planner.check_collision(state, options={"full_report": True, "verbose": False})
        print(f"[OK]{tag} no collisions reported.")
        return True
    except CollisionCheckError as exc:
        pairs = list(getattr(exc, "collision_pairs", []) or [])
        print(f"[!!]{tag} {len(pairs)} colliding pair(s):")
        for line in str(exc).splitlines():
            print(f"     {line}")
        return False


def replay_with_slider(planner, template_state, path: Sequence[Sequence[float]],
                       joint_names_12: Sequence[str]) -> None:
    """Scrub a single planned path with a blocking PyBullet slider.

    Adds a debug-parameter slider to the GUI and, as it moves, drives the robot
    to the matching waypoint. Blocks until the GUI closes or Ctrl-C.

    Args:
        planner: The PyBulletPlanner driving the scene.
        template_state: The movement's start_state, copied per waypoint so the
            held bar and allowed-contact set stay correct during replay.
        path (Sequence[Sequence[float]]): Waypoints, each a 12-vec ordered by
            ``joint_names_12``.
        joint_names_12 (Sequence[str]): The twelve arm-joint names, left then
            right, that each waypoint maps onto.

    Returns:
        None.
    """
    cid = planner.client.client_id
    n = len(path)
    if n == 0:
        print("[replay] empty path, skipping slider.")
        return
    print(f"[replay] {n} waypoint(s); Ctrl-C to exit slider.")

    slider = pybullet.addUserDebugParameter(
        f"Replay t (0..{n - 1})", 0.0, float(n - 1), 0.0,
        physicsClientId=cid,
    )

    states = []
    for q in path:
        s = template_state.copy()
        for name, val in zip(joint_names_12, q):
            s.robot_configuration[name] = float(val)
        states.append(s)

    try:
        while pp.has_gui():
            t = pybullet.readUserDebugParameter(slider, physicsClientId=cid)
            idx = int(round(t))
            idx = max(0, min(n - 1, idx))
            with pp.LockRenderer(False):
                planner.set_robot_cell_state(states[idx])
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("[replay] interrupted.")


def replay_segments(planner, segments, joint_names_12: Sequence[str]) -> None:
    """Scrub a chained sequence of movements end-to-end with one slider.

    Each segment carries its own ``template_state`` (that movement's
    start_state, with the phase's attachments / allowed-contact set), so the
    held bar renders correctly: attached during M1/M2, released during M3/M4.
    The slider walks all segments as one continuous path and prints a line each
    time it crosses into the next movement.

    Args:
        planner: The PyBulletPlanner driving the scene.
        segments: List of ``(role, template_state, path)`` tuples, one per
            solved movement, in playback order.
        joint_names_12 (Sequence[str]): The twelve arm-joint names, left then
            right, that each waypoint maps onto.

    Returns:
        None.
    """
    cid = planner.client.client_id

    # Flatten to (state_for_this_waypoint, q), remembering role boundaries.
    built = []
    labels = []
    for role, template_state, path in segments:
        for q in path:
            s = template_state.copy()
            for name, val in zip(joint_names_12, q):
                s.robot_configuration[name] = float(val)
            built.append(s)
            labels.append(role)

    n = len(built)
    if n == 0:
        print("[replay] empty path, skipping slider.")
        return
    print(f"[replay] {n} waypoint(s) across {len(segments)} movement(s); "
          "Ctrl-C to exit slider.")

    slider = pybullet.addUserDebugParameter(
        f"Replay t (0..{n - 1})", 0.0, float(n - 1), 0.0,
        physicsClientId=cid,
    )

    last_role = None
    try:
        while pp.has_gui():
            t = pybullet.readUserDebugParameter(slider, physicsClientId=cid)
            idx = int(round(t))
            idx = max(0, min(n - 1, idx))
            if labels[idx] != last_role:
                print(f"[replay] -> {labels[idx]}")
                last_role = labels[idx]
            with pp.LockRenderer(False):
                planner.set_robot_cell_state(built[idx])
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("[replay] interrupted.")


def select_movement(action, role: str):
    """Pick the Movement that plays a given role from an action.

    The primary signal is ``match_role`` (the ``_M<n>_`` id tag). When ids are
    untagged, fall back to the movement class: the two constrained classes and
    the independent-linear class each map to exactly one role. M0 and M4 share
    ``IndependentDualArmFreeMovement``, so untagged they are told apart by order
    (first = M0, second = M4).

    Args:
        action: The BarAssemblyAction holding the list of movements.
        role (str): The role to find, ``'M0'``..``'M4'``.

    Returns:
        The matching Movement, or ``None`` when no movement fits the role.
    """
    for mv in action.movements:
        if match_role(mv) == role:
            return mv
    _role_by_type = {
        EndEffectorConstrainedDualArmFreeMovement: "M1",
        EndEffectorConstrainedDualArmLinearMovement: "M2",
        IndependentDualArmLinearMovement: "M3",
    }
    for mv in action.movements:
        if _role_by_type.get(type(mv)) == role:
            return mv
    free = [mv for mv in action.movements
            if isinstance(mv, IndependentDualArmFreeMovement)]
    if role == "M0" and free:
        return free[0]
    if role == "M4" and len(free) > 1:
        return free[1]
    return None


def _color_bool(value: bool) -> str:
    """Render a boolean as green ``True`` / red ``False`` (ANSI)."""
    color = "\033[32m" if value else "\033[31m"  # green / red
    return f"{color}{value}\033[0m"


def print_roster(movements: Sequence, tag: str = "roster") -> None:
    """Print which movements have a start conf and a trajectory.

    Used both as a pre-flight view and after each accepted trajectory. For each
    movement it prints the id and two color-coded booleans: whether its
    ``start_state`` carries a robot configuration, and whether it has a planned
    trajectory. Headless adaptation of ``husky_monitor._print_movement_roster``.

    Args:
        movements (Sequence): Movements in planning order; an entry may be
            ``None`` when a role did not resolve.
        tag (str): Short label printed in the header line.

    Returns:
        None.
    """
    print(f"[{tag}] movement roster:")
    for i, m in enumerate(movements):
        if m is None:
            print(f"  [{i}] <missing>")
            continue
        has_conf = (m.start_state is not None
                    and getattr(m.start_state, "robot_configuration", None) is not None)
        has_traj = getattr(m, "trajectory", None) is not None
        print(f"  [{i}] {m.movement_id!r}")
        print(f"     - start state: has robot_conf = {_color_bool(has_conf)}")
        print(f"     - has trajectory = {_color_bool(has_traj)}")


def apply_conf12(state, rcell, joint_names_12: Sequence[str],
                 q: Sequence[float]) -> None:
    """Write a 12-vec of joint values into a state's robot configuration.

    Builds a zero full configuration first when the state has none, so chained
    movements start exactly where the previous one ended.

    Args:
        state: The RobotCellState to update in place.
        rcell: The RobotCell, used to build a zero configuration when needed.
        joint_names_12 (Sequence[str]): The twelve arm-joint names the values
            map onto, left then right.
        q (Sequence[float]): The twelve joint values to write.

    Returns:
        None.
    """
    if state.robot_configuration is None:
        state.robot_configuration = rcell.zero_full_configuration()
    for name, val in zip(joint_names_12, q):
        state.robot_configuration[name] = float(val)


def vec12_from_conf(conf, joint_names_12: Sequence[str]) -> np.ndarray:
    """Pull a 12-vec (left then right arm joints) out of a configuration.

    Args:
        conf: A compas Configuration to read joint values from.
        joint_names_12 (Sequence[str]): The twelve arm-joint names to extract,
            in order.

    Returns:
        np.ndarray: The twelve joint values as a float array.
    """
    return np.asarray([float(conf[n]) for n in joint_names_12], dtype=float)


def accept_trajectory(mv, path: Sequence[Sequence[float]], *, role: str,
                      index: int, movements, rcell,
                      joint_names_12: Sequence[str], source: str = "Plan") -> bool:
    """Chain the joint configs forward after a movement is planned.

    Headless adaptation of ``husky_monitor._accept_trajectory`` (trimmed of ROS
    logging, visualizer wiring, disk save, CDFM validation and the synthetic-M0
    backfill).

    What it does, by role:
      - M1/M4 own their start: mirror ``path[0]`` into ``mv.start_state``.
      - M2/M3 must already carry a propagated start config; the trajectory is
        rejected if that config is missing or disagrees with ``path[0]``.
      - M1/M2/M3 forward-propagate ``path[-1]`` into the *next* movement's
        ``start_state.robot_configuration`` (M0/M4 end the chain).
      - A backward continuity check compares against the previous movement's
        ``path[-1]``.

    Args:
        mv: The Movement that was just planned; its ``trajectory`` is set here.
        path (Sequence[Sequence[float]]): The planned waypoints, each a 12-vec
            ordered by ``joint_names_12``.
        role (str): This movement's role, ``'M1'``..``'M4'``.
        index (int): Position of this movement inside ``movements``.
        movements: All movements in planning order (used to reach the next and
            previous ones).
        rcell: The RobotCell, used to build configurations when propagating.
        joint_names_12 (Sequence[str]): The twelve arm-joint names, in order.
        source (str): Short label printed with each log line.

    Returns:
        bool: True when the trajectory is accepted, False on a chain break (the
        caller should stop chaining).
    """
    mv.trajectory = path
    if path:
        start_vec = np.asarray(path[0], dtype=float)

        if role in ("M2", "M3") and mv.start_state is not None:
            existing = mv.start_state.robot_configuration
            if existing is None:
                print(f"[{source}] {mv.movement_id!r} has no propagated start_conf; "
                      "rejecting trajectory.")
                mv.trajectory = None
                return False
            diff = float(np.abs(start_vec - vec12_from_conf(existing, joint_names_12)).max())
            if diff > 1e-3:
                print(f"[{source}] start of {mv.movement_id!r} differs from propagated "
                      f"start_conf by max {diff:.4f} rad/m; rejecting trajectory.")
                mv.trajectory = None
                return False
        elif mv.start_state is not None:
            # M1/M4 own their generated start_conf.
            apply_conf12(mv.start_state, rcell, joint_names_12, start_vec)

        # Forward-chain propagation.
        if role in ("M0", "M4"):
            pass
        elif index + 1 < len(movements):
            next_mv = movements[index + 1]
            if next_mv is not None and next_mv.start_state is not None:
                existing = next_mv.start_state.robot_configuration
                new_end = np.asarray(path[-1], dtype=float)
                if existing is None:
                    apply_conf12(next_mv.start_state, rcell, joint_names_12, new_end)
                    print(f"[{source}] propagated {mv.movement_id!r}.traj[-1] -> "
                          f"{next_mv.movement_id!r}.start_state.robot_configuration "
                          "(was None).")
                else:
                    diff = float(np.abs(
                        new_end - vec12_from_conf(existing, joint_names_12)
                    ).max())
                    if diff > 1e-3:
                        print(f"[{source}] end of {mv.movement_id!r} differs from existing "
                              f"{next_mv.movement_id!r}.start by max {diff:.4f} rad/m; "
                              "overwriting (chain rule).")
                    apply_conf12(next_mv.start_state, rcell, joint_names_12, new_end)

        # Backward continuity check.
        if index > 0:
            prev_mv = movements[index - 1]
            prev_path = getattr(prev_mv, "trajectory", None)
            if prev_path:
                diff = float(np.abs(
                    np.asarray(prev_path[-1], dtype=float) - start_vec
                ).max())
                if diff > 1e-3:
                    print(f"[{source}] start of {mv.movement_id!r} differs from "
                          f"{prev_mv.movement_id!r}.trajectory[-1] by max {diff:.4f} rad/m.")
                else:
                    print(f"[{source}] start agrees with {prev_mv.movement_id!r}."
                          f"trajectory[-1] (max diff {diff:.6f}).")

    print_roster(movements, source)
    return True


def solved_action_path(clean_action_path: str) -> str:
    """Sidecar path for the half-solved BarAction, next to the clean file.

    For ``.../BarActions/B6.json`` this returns ``.../BarActions/B6.solved.json``.
    The clean file (the Rhino export) is never overwritten; only this sidecar is.
    """
    stem, ext = os.path.splitext(clean_action_path)
    return f"{stem}.solved{ext}"


def save_solved_action(action, path: str) -> None:
    """Write the current (partly planned) action to its half-solved sidecar.

    Called after each movement is planned so a failed run still leaves every
    already-solved movement on disk. Trajectories are normalized to plain float
    lists so the JSON is portable and round-trips through ``json_load``.
    Overwrites any previous sidecar.

    Args:
        action: The BarAssemblyAction being planned (mutated in place as
            movements are solved).
        path (str): Destination sidecar path (see :func:`solved_action_path`).

    Returns:
        None.
    """
    for mv in action.movements:
        traj = getattr(mv, "trajectory", None)
        if traj is not None:
            mv.trajectory = [[float(v) for v in wp] for wp in traj]
    json_dump(action, path)
    print(f"[save] half-solved BarAction -> {path}")


def _path_from_jt(jt, joint_names_12: Sequence[str]) -> Optional[List[List[float]]]:
    """Convert a JointTrajectory into a list of 12-vecs ordered by names.

    Args:
        jt: A compas_fab JointTrajectory, or ``None``.
        joint_names_12 (Sequence[str]): The twelve arm-joint names to read from
            each trajectory point, in order.

    Returns:
        Optional[List[List[float]]]: One 12-vec per trajectory point, or
        ``None`` when ``jt`` is ``None``.
    """
    if jt is None:
        return None
    return [
        [float(p.joint_values[p.joint_names.index(n)]) for n in joint_names_12]
        for p in jt.points
    ]


def _make_tree_draw_fn(planner, robot_puid, arm_joints, tool_link_left, tool_link_right,
                       start_conf, goal_conf):
    """Build a ``draw_fn`` for ``pp.solve_motion_plan`` that renders the birrt trees.

    pybullet_planning's rrt_connect calls ``draw_fn(config, segment, *valid)``
    where ``segment`` is ``[]`` for a raw sample / tree root and
    ``[child, parent]`` for a tree edge. It does NOT tag which of the two trees a
    node belongs to, so we recover that: every edge records ``child -> parent``,
    and we retrace a node to its root — root ~= start => forward tree (red),
    root ~= goal => backward tree (blue), an unrooted raw sample => gray.

    For each node we draw a point at BOTH arms' FK tool0 positions; for each tree
    edge we draw a parent->child line for both arms, in the tree's color. Nodes /
    edges are de-duplicated (rrt_connect re-draws the whole tree every iteration);
    raw samples are drawn each time to show the sampling spread. No-op without a
    GUI.

    Args:
        planner: unused directly (kept for symmetry / future use).
        robot_puid (int): pybullet body id of the robot.
        arm_joints (Sequence[int]): the 12 arm joint ids (left then right).
        tool_link_left, tool_link_right (int): tool0 link ids to FK.
        start_conf, goal_conf (Sequence[float]): the birrt endpoints, used to
            classify which root a node traces back to.

    Returns:
        Callable: a ``draw_fn(config, segment, *valid)`` closure.
    """
    RED = (1.0, 0.1, 0.1)     # forward tree (rooted at start)
    BLUE = (0.1, 0.3, 1.0)    # backward tree (rooted at goal)
    GRAY = (0.55, 0.55, 0.55)  # raw sample, not (yet) attached to either tree

    def _key(q):
        return tuple(round(float(v), 5) for v in q)

    start_key, goal_key = _key(start_conf), _key(goal_conf)
    parent_of = {}
    drawn_nodes = set()
    drawn_edges = set()

    def _fk(q):
        pp.set_joint_positions(robot_puid, arm_joints, q)
        return (pp.get_link_pose(robot_puid, tool_link_left)[0],
                pp.get_link_pose(robot_puid, tool_link_right)[0])

    def _root_color(key):
        cur, seen = key, set()
        while cur in parent_of and cur not in seen:
            seen.add(cur)
            cur = parent_of[cur]
        if cur == start_key:
            return RED
        if cur == goal_key:
            return BLUE
        return GRAY

    def draw_fn(config, segment, *_valid):
        if not pp.has_gui():
            return
        if segment:
            # Tree edge: segment == [child_config, parent_config].
            child, parent = segment[0], segment[1]
            ckey, pkey = _key(child), _key(parent)
            # First-seen parent wins. As the two trees' frontiers approach, their
            # nodes can share a rounded config key; overwriting would let a key
            # point at conflicting parents and form a CYCLE, so retrace would
            # never reach a root and the edge would be mis-colored gray. A real
            # tree is acyclic, so keeping the first parent avoids that.
            parent_of.setdefault(ckey, pkey)
            need_node = ckey not in drawn_nodes
            need_edge = (ckey, pkey) not in drawn_edges
            if not (need_node or need_edge):
                return
            color = _root_color(ckey)
            if need_node:
                cl, cr = _fk(child)
                pp.draw_point(cl, size=0.012, color=color)
                pp.draw_point(cr, size=0.012, color=color)
                drawn_nodes.add(ckey)
            if need_edge:
                cl, cr = _fk(child)
                pl, pr = _fk(parent)
                pp.add_line(cl, pl, color=color, width=1)
                pp.add_line(cr, pr, color=color, width=1)
                drawn_edges.add((ckey, pkey))
        else:
            # Raw sample or a tree root (both arrive with an empty segment).
            color = _root_color(_key(config))
            if color == GRAY:
                # Candidate sample: draw every one (tiny) to show sampling spread.
                l, r = _fk(config)
                pp.draw_point(l, size=0.006, color=GRAY)
                pp.draw_point(r, size=0.006, color=GRAY)
            elif _key(config) not in drawn_nodes:
                # Start / goal root: draw once, bigger.
                l, r = _fk(config)
                pp.draw_point(l, size=0.02, color=color)
                pp.draw_point(r, size=0.02, color=color)
                drawn_nodes.add(_key(config))

    return draw_fn


def plan_movement(planner, state, role: str, selected, *, active_bar_id: str,
                  active_bar_rb_name: Optional[str],
                  joint_names_12: Sequence[str], max_time: float,
                  derive_start: bool = True, draw: bool = False):
    """Send one movement to the right planner API for its role.

    For M1, ``derive_start`` (the default) asks the planner to compute a
    feasible, grasp-consistent start instead of trusting the (placeholder)
    start config in the cell state — see
    ``api.plan_constrained_dual_arm(derive_start=...)``.

    Args:
        planner: The PyBulletPlanner to plan with.
        state: The movement's start RobotCellState.
        role (str): The movement role, ``'M1'``..``'M4'``.
        selected: The Movement object for this role.
        active_bar_id (str): The bar id from the action (needed by M1/M2).
        active_bar_rb_name (Optional[str]): The bar's rigid-body name in the
            cell (``bar_<id>``), passed to the planner.
        joint_names_12 (Sequence[str]): The twelve arm-joint names, in order.
        max_time (float): Planning time budget in seconds.
        derive_start (bool): M1 only — derive a fresh feasible start instead of
            trusting the cell state's start config.
        draw (bool): M4 only — pass a live search-tree ``draw_fn`` (built by
            :func:`_make_tree_draw_fn`) into ``plan_free_dual_arm`` and let
            timeout be the only stop control. See ``--diagnosis``.

    Returns:
        Tuple[Optional[list], dict]: ``(path, info)`` where ``path`` is a list
        of 12-vec waypoints (or ``None`` on failure) and ``info`` carries a
        ``failure_reason`` when planning fails.
    """
    goal_conf = selected.target_configuration
    goal_ee_frames = selected.target_ee_frames or None

    if draw and role != "M4":
        # Tree drawing is wired only through M4's free birrt (solve_motion_plan's
        # draw_fn). M1 uses the task-space pose RRT and M2/M3 are IK loops (no
        # birrt tree), so there's nothing to draw for them here.
        print(f"[diagnosis] tree drawing is M4-only; {role} planned normally "
              "(renderer still unlocked).")

    if role == "M0" and isinstance(selected, IndependentDualArmFreeMovement):
        # M0 (current pose -> M1 start) is normally left unplanned offline (the
        # live monitor plans it). This branch supports on-demand testing: plan a
        # free dual-arm motion to M0's goal, which is M1's start config filled in
        # by the M1 backfill step in main() after M1 has been planned.
        if goal_conf is None:
            return None, {"failure_reason": (
                "M0 has no target_configuration yet; plan M1 first so its start "
                "config is backfilled into M0.target_configuration."
            )}
        print("[plan] plan_free_dual_arm (M0 -> M1 start config)")
        return plan_free_dual_arm(planner, state, goal_conf, max_time=max_time)
    if role == "M1" and isinstance(selected, EndEffectorConstrainedDualArmFreeMovement):
        if not active_bar_id:
            return None, {"failure_reason": "M1 needs active_bar_id on the action."}
        if not goal_ee_frames:
            return None, {"failure_reason": "M1 needs target_ee_frames on the movement."}
        print(f"[plan] plan_constrained_dual_arm "
              f"(active_bar_id={active_bar_id}, derive_start={derive_start})")
        return plan_constrained_dual_arm(
            planner, state,
            active_bar_id=active_bar_rb_name,
            goal_ee_frames=goal_ee_frames,
            max_time=max_time,
            derive_start=derive_start,
        )
    if role == "M2" and isinstance(selected, EndEffectorConstrainedDualArmLinearMovement):
        if not active_bar_id:
            return None, {"failure_reason": "M2 needs active_bar_id on the action."}
        print(f"[plan] plan_constrained_dual_arm_linear (active_bar_id={active_bar_id})")
        jt = plan_constrained_dual_arm_linear(
            planner, state,
            active_bar_id=active_bar_rb_name,
            goal_conf=goal_conf,
            goal_ee_frames=goal_ee_frames,
        )
        path = _path_from_jt(jt, joint_names_12)
        return path, {"failure_reason": None if jt is not None else "linear-ik failed"}
    if role == "M3" and isinstance(selected, IndependentDualArmLinearMovement):
        print("[plan] plan_dual_arm_linear_independent")
        jt = plan_dual_arm_linear_independent(
            planner, state,
            goal_conf=goal_conf,
            goal_ee_frames=goal_ee_frames,
        )
        path = _path_from_jt(jt, joint_names_12)
        return path, {"failure_reason": None if jt is not None else "linear-ik failed"}
    if role == "M4" and isinstance(selected, IndependentDualArmFreeMovement):
        # M4 returns to a fixed dual-arm home. Override the (placeholder) target
        # from the action with the known-good home config (left 6 then right 6,
        # matching joint_names_12).
        goal_conf = list(_config.HUSKY_DUAL_ARM_HOME_CONF_12)
        extra = {}
        if draw:
            # Diagnosis: build a live tree draw_fn over both arms' tool0 FK and
            # pass it into the normal call, plus a large max_iterations so
            # timeout is the only stop control (default 20 would give up first).
            robot_puid = planner.client.robot_puid
            arm_joints = pp.joints_from_names(robot_puid, joint_names_12)
            draw_fn = _make_tree_draw_fn(
                planner, robot_puid, arm_joints,
                pp.link_from_name(robot_puid, TOOL_LINK_LEFT),
                pp.link_from_name(robot_puid, TOOL_LINK_RIGHT),
                _conf12_from_state(state, joint_names_12),
                _conf12_from_target(goal_conf, joint_names_12),
            )
            extra = {"draw_fn": draw_fn, "max_iterations": 10_000_000}
            print("[plan] plan_free_dual_arm + tree drawing "
                  "(diagnosis; goal = HUSKY_DUAL_ARM_HOME_CONF_12)")
        else:
            print("[plan] plan_free_dual_arm (goal = HUSKY_DUAL_ARM_HOME_CONF_12)")
        return plan_free_dual_arm(planner, state, goal_conf, max_time=max_time, **extra)

    return None, {
        "failure_reason": (
            f"role {role!r} does not match movement type {type(selected).__name__}"
        )
    }


def plot_conf_comparison(joint_names_12, start_conf, goal_conf, out_path, *, show=False):
    """Save (and optionally show) a per-joint START-vs-GOAL bar chart.

    One figure, 12 grouped bars (left arm 0-5, right arm 6-11). Each joint gets
    a START bar and a GOAL bar side by side, with the |delta| annotated on top,
    so a big swing on an otherwise-small bar move is obvious at a glance.
    """
    start = np.asarray(start_conf, dtype=float)
    goal = np.asarray(goal_conf, dtype=float)
    delta = goal - start
    n = len(joint_names_12)
    x = np.arange(n)
    w = 0.4

    def _short(name):
        base = re.sub(r"_joint$", "", str(name))
        return "_".join(base.split("_")[-2:])

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w / 2, start, w, label="START", color="#1f6fb2")
    ax.bar(x + w / 2, goal, w, label="GOAL", color="#d98218")
    ax.axhline(0.0, color="k", lw=0.6)

    top = float(max(start.max(), goal.max()))
    bot = float(min(start.min(), goal.min()))
    span = (top - bot) or 1.0
    ax.set_ylim(bot - 0.12 * span, top + 0.28 * span)
    for xi in x:
        yv = max(start[xi], goal[xi])
        ax.text(xi, yv + 0.03 * span, f"Δ{abs(delta[xi]):.2f}",
                ha="center", va="bottom", fontsize=7, color="#444")

    # Divider + arm labels.
    ax.axvline(5.5, color="gray", ls="--", lw=1)
    ax.text(2.5, top + 0.20 * span, "LEFT arm", ha="center", fontsize=10, color="#555")
    ax.text(8.5, top + 0.20 * span, "RIGHT arm", ha="center", fontsize=10, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels([_short(j) for j in joint_names_12], rotation=90, fontsize=8)
    ax.set_ylabel("joint value (rad)")
    ax.set_title(f"M1 start vs goal per-joint conf  (max |Δ| = {np.abs(delta).max():.3f} rad)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[probe] per-joint figure saved -> {out_path}")
    if show:
        try:
            plt.show()
        except Exception as exc:  # non-interactive backend, etc.
            print(f"[probe] plt.show() skipped ({exc}).")
    plt.close(fig)


def probe_endpoints(planner, rcell, action, active_bar_rb_name: Optional[str],
                    joint_names_12: Sequence[str], *, use_gui: bool = False) -> int:
    """Report M1 start/goal endpoint feasibility WITHOUT running the RRT.

    Runs only the goal-IK plus start-derivation stage of
    ``plan_constrained_dual_arm(derive_start=True)`` and prints, for both
    endpoints, whether the derived dual-arm conf is collision-free and holds the
    bar grasp-consistently, plus the largest per-joint start-to-goal delta. A
    large delta with a tiny bar move means start and goal landed on different IK
    branches (the RRT then struggles to connect them). Fast (~5 s) compared to
    the full stochastic RRT. M1 only (constrained dual-arm movement).

    Args:
        planner: The PyBulletPlanner to probe with.
        rcell: The RobotCell (used to fill a missing start config with home).
        action: The BarAssemblyAction holding the movements.
        active_bar_rb_name (Optional[str]): The active bar's rigid-body name.
        joint_names_12 (Sequence[str]): The twelve arm-joint names, in order.

    Returns:
        int: 0 when both endpoints are feasible, otherwise 2.
    """
    selected = select_movement(action, "M1")
    if selected is None or not isinstance(selected, EndEffectorConstrainedDualArmFreeMovement):
        print("[probe] no M1 constrained dual-arm movement found; endpoint probe is M1-only.")
        return 2
    state = selected.start_state
    if state is None:
        print("[probe] M1 start_state is None.")
        return 2
    goal_ee_frames = selected.target_ee_frames or None
    if not goal_ee_frames:
        print("[probe] M1 has no target_ee_frames.")
        return 2
    if state.robot_configuration is None:
        fill_missing_config(state, rcell, _config.HOME_CONF_LEFT_6, _config.HOME_CONF_RIGHT_6)
    planner.set_robot_cell_state(state)

    robot_puid = planner.client.robot_puid
    arm_joints = pp.joints_from_names(robot_puid, joint_names_12)
    tool_link_left = pp.link_from_name(robot_puid, TOOL_LINK_LEFT)
    tool_link_right = pp.link_from_name(robot_puid, TOOL_LINK_RIGHT)
    bar_body = _bar_body_id(planner, active_bar_rb_name)
    obstacles = _collect_obstacle_puids(planner, exclude={active_bar_rb_name})

    # Only the goal-IK + start-derivation stage (no RRT).
    (
        start_conf,
        world_from_bar_start,
        world_from_bar_goal,
        goal_conf_arr,
        grasp_l,
        grasp_r,
        info,
    ) = _derive_constrained_start_for_plan(
        planner, state,
        active_bar_id=active_bar_rb_name,
        bar_body=bar_body,
        obstacles=obstacles,
        robot_puid=robot_puid,
        arm_joints=arm_joints,
        tool_link_left=tool_link_left,
        tool_link_right=tool_link_right,
        joint_names_12=joint_names_12,
        goal_conf=None,
        goal_ee_frames=goal_ee_frames,
        random_seed=None,
        max_ik_attempts=20,
        bar_sweep_box=None,
    )
    if start_conf is None:
        print(f"\n[probe] derivation FAILED: {info.get('failure_reason')}")
        return 2

    # Independent collision check of both endpoints via cfab (reset the cached
    # state before each, since collide()/derive touch the pybullet world).
    planner.set_robot_cell_state(state)
    collide = _build_cfab_collision_fn(planner, state, joint_names_12)
    goal_hit = collide(goal_conf_arr)
    planner.set_robot_cell_state(state)
    start_hit = collide(start_conf)

    def _grasp_ok(conf, bar_pose):
        return validate_dual_arm_bar_pose(
            robot=robot_puid, arm_joints=arm_joints,
            tool_link_left=tool_link_left, tool_link_right=tool_link_right,
            full_conf=conf, bar_pose=bar_pose,
            grasp_bar_from_left=grasp_l, grasp_bar_from_right=grasp_r,
            pos_tolerance=1e-3, ori_tolerance=1e-2,
        )

    goal_ok = _grasp_ok(goal_conf_arr, world_from_bar_goal)
    start_ok = _grasp_ok(start_conf, world_from_bar_start)

    def _fmt(vec):
        return "[" + ", ".join(f"{float(v):+.4f}" for v in vec) + "]"

    print("\n=================== M1 ENDPOINT FEASIBILITY ===================")
    print(f"  GOAL  conf : {_fmt(goal_conf_arr)}")
    print(f"        bar pos (xyz): {np.round(world_from_bar_goal[0], 4)}")
    print(f"        collision-free : {not goal_hit}    grasp-consistent : {goal_ok}")
    print(f"  START conf : {_fmt(start_conf)}")
    print(f"        bar pos (xyz): {np.round(world_from_bar_start[0], 4)}")
    print(f"        collision-free : {not start_hit}    grasp-consistent : {start_ok}")
    d = float(np.abs(np.asarray(start_conf) - np.asarray(goal_conf_arr)).max())
    print(f"  max |start-goal| joint delta: {d:.4f} rad")
    feasible = (not goal_hit) and (not start_hit) and goal_ok and start_ok
    print(f"\n[probe] both endpoints feasible = {feasible}")
    print("===============================================================")

    # Per-joint comparison figure (always saved; shown when --gui).
    fig_path = os.path.join(TESTS_DIR, "m1_endpoint_confs.png")
    plot_conf_comparison(joint_names_12, start_conf, goal_conf_arr, fig_path, show=use_gui)

    # PyBullet slider to toggle between the two confs (GUI only). idx 0 = START,
    # idx 1 = GOAL; the attached bar re-poses with each conf.
    if use_gui:
        print("[probe] pb slider: t=0 -> START, t=1 -> GOAL  (Ctrl-C to exit).")
        planner.set_robot_cell_state(state)
        replay_with_slider(
            planner, state,
            [np.asarray(start_conf, dtype=float), np.asarray(goal_conf_arr, dtype=float)],
            joint_names_12,
        )

    return 0 if feasible else 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Parse CLI args, load the cell and action, then plan and optionally replay.

    Loads the RobotCell and BarAction JSONs, resolves the requested movement
    role(s), runs planning (chaining M1->M2->M3->M4 when ``--movement all``),
    and, in GUI mode, replays the result on a slider. See the module docstring
    for the full CLI.

    Returns:
        int: Process exit code — 0 on success, 1 for bad inputs / setup errors,
        2 when planning or feasibility checks fail.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "data_root", nargs="?", default=DEFAULT_DATA_ROOT,
        help="Parent folder containing per-problem subfolders.",
    )
    parser.add_argument(
        "--problem", default=DEFAULT_PROBLEM,
        help="Subfolder under <data_root> with BarActions/ + RobotCell.json.",
    )
    parser.add_argument(
        "--bar-action", default=DEFAULT_BAR_ACTION,
        help="BarAction filename inside <data_root>/<problem>/BarActions/.",
    )
    parser.add_argument(
        "--movement", required=True, choices=("M0", "M1", "M2", "M3", "M4", "all"),
        help="Which movement role to plan. 'all' chains M1->M2->M3->M4 (M0 is "
             "left unplanned offline -- the live monitor plans it). 'M0' only "
             "works after M1 has been planned so its goal config is backfilled.",
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--max-time", type=float, default=60.0)
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument(
        "--no-derive-start", dest="derive_start", action="store_false",
        help="Trust M1's start config from the cell state instead of deriving "
             "a feasible grasp-consistent start (default: derive).",
    )
    parser.set_defaults(derive_start=True)
    parser.add_argument(
        "--load", choices=("clean", "solved"), default="clean",
        help="Which BarAction to load. 'clean' (default) = the Rhino export, "
             "plan the requested movement(s) from scratch. 'solved' = the "
             "half-solved sidecar (<bar-action>.solved.json): reuse movements "
             "that already have a trajectory and only plan the ones still "
             "missing. The clean file is never overwritten; the sidecar is "
             "rewritten after each movement is planned.",
    )
    parser.add_argument(
        "--probe-endpoints", action="store_true",
        help="M1 only: report whether a feasible, collision-free start and goal "
             "dual-arm conf can be derived (goal-IK + start-derivation stage), "
             "then exit WITHOUT running the RRT. Fast feasibility check.",
    )
    parser.add_argument(
        "--diagnosis", action="store_true",
        help="M4 only: draw the birrt search trees live in the GUI (red = "
             "forward/start tree, blue = backward/goal tree, gray = raw samples; "
             "points at both arms' tool0, lines = tree edges). Turns OFF the "
             "LockRenderer around planning so the trees render as they grow, and "
             "makes timeout the only stop control. Requires --gui.",
    )
    parser.add_argument(
        "--cell", default=None,
        help="Path to RobotCell.json (default: <data_root>/RobotCell.json).",
    )
    args = parser.parse_args()

    if args.diagnosis and not args.gui:
        print("[diagnosis] --diagnosis draws in the GUI; without --gui the trees "
              "won't render (planning still runs with the renderer 'unlocked').")

    data_root = os.path.abspath(args.data_root)
    if not os.path.isdir(data_root):
        print(f"[X] missing data_root: {data_root}")
        return 1

    problem_dir = os.path.join(data_root, args.problem)
    if not os.path.isdir(problem_dir):
        print(f"[X] missing problem dir: {problem_dir}")
        return 1

    clean_action_path = os.path.join(problem_dir, "BarActions", args.bar_action)
    if not os.path.isfile(clean_action_path):
        print(f"[X] missing BarAction file: {clean_action_path}")
        return 1

    # The clean file (Rhino export) is never overwritten; the half-solved
    # snapshot is always written to this sidecar, regardless of which we load.
    save_path = solved_action_path(clean_action_path)
    if args.load == "solved":
        if not os.path.isfile(save_path):
            print(f"[X] --load solved but no half-solved file yet: {save_path}")
            print("    Run once with --load clean to produce it.")
            return 1
        action_path = save_path
    else:
        action_path = clean_action_path

    cell_path = args.cell or os.path.join(problem_dir, "RobotCell.json")
    if not os.path.isfile(cell_path):
        print(f"[X] missing RobotCell.json: {cell_path}")
        return 1

    print(f"[load] RobotCell    <- {cell_path}")
    rcell = json_load(cell_path)
    print(f"  robot model   : {getattr(rcell.robot_model, 'name', '<?>')}")
    print(f"  tool models   : {sorted(rcell.tool_models.keys())}")
    print(f"  rigid bodies  : {len(rcell.rigid_body_models)}")

    print(f"[load] BarAction ({args.load}) <- {action_path}")
    action = json_load(action_path)
    active_bar_id = getattr(action, "active_bar_id", "") or ""
    print(f"  action_id     : {action.action_id}")
    print(f"  active_bar_id : {active_bar_id}")
    print(f"  movements     : {len(action.movements)}")

    # The cell stores bars as rigid bodies named ``bar_<bar_id>``. The
    # planner API needs that rigid-body name, not the bare bar id.
    def _resolve_bar_rb_name(cell, bar_id: str) -> Optional[str]:
        for candidate in (bar_id, f"bar_{bar_id}"):
            if candidate in cell.rigid_body_models:
                return candidate
        return None

    active_bar_rb_name = _resolve_bar_rb_name(rcell, active_bar_id) if active_bar_id else None
    if active_bar_id and active_bar_rb_name is None:
        print(f"[X] active_bar_id {active_bar_id!r} not found in cell.rigid_body_models")
        return 1
    if active_bar_rb_name and active_bar_rb_name != active_bar_id:
        print(f"  bar rigid-body name: {active_bar_rb_name}")

    # Planning sequence: the order movements are planned in (not just a set of
    # roles). Order matters — each solved movement's start/end config propagates
    # into its neighbours (see accept_trajectory), so M1 must plan before M2, etc.
    planning_sequence = (
        ["M1", "M2", "M3", "M4"] if args.movement == "all" else [args.movement]
    )

    print(f"\n[pb] starting PyBullet ({'GUI' if args.gui else 'DIRECT'})")
    _client, planner = start_planner(rcell, use_gui=args.gui)

    try:
        # 12-vec joint names — read from the cell, shared across movements.
        left_names = [n for n in rcell.get_configurable_joint_names(LEFT_GROUP)
                      if any(n.endswith(s) for s in _ARM_SUFFIXES)]
        right_names = [n for n in rcell.get_configurable_joint_names(RIGHT_GROUP)
                       if any(n.endswith(s) for s in _ARM_SUFFIXES)]
        joint_names_12 = left_names + right_names

        # Resolve the movements once, in planning order, and mutate their
        # start_states in place — accept_trajectory propagates each planned
        # endpoint into the next movement's start_state (the key chaining step).
        movements = [select_movement(action, r) for r in planning_sequence]

        segments = []          # (role, state, path) for each solved movement
        results = []           # (role, ok, detail) for the final roll-up

        # Pre-flight: show what will be planned, in order, before we start.
        print_roster(movements, tag="pre-flight")

        # Endpoint feasibility probe (M1): derive + report start/goal, no RRT.
        if args.probe_endpoints:
            return probe_endpoints(
                planner, rcell, action, active_bar_rb_name, joint_names_12,
                use_gui=args.gui,
            )

        for index, role in enumerate(planning_sequence):
            selected = movements[index]
            if selected is None:
                msg = f"no movement matches role {role!r} in {args.bar_action}"
                print(f"[X] {msg}")
                results.append((role, False, msg))
                break  # chain can't continue past a missing movement

            print(f"\n[pick] {role} -> {type(selected).__name__} {selected.movement_id}")

            state = selected.start_state
            if state is None:
                msg = f"{selected.movement_id}: start_state is None."
                print(f"[X] {msg}")
                results.append((role, False, msg))
                break

            # Reuse an already-planned trajectory from the half-solved file:
            # skip planning, keep it for replay + chaining. Its start_state (and
            # the next movement's propagated start) were saved with it, so the
            # chain stays consistent and we only plan the still-missing roles.
            existing_traj = getattr(selected, "trajectory", None)
            if args.load == "solved" and existing_traj:
                path = [[float(v) for v in wp] for wp in existing_traj]
                print(f"[reuse] {role}: {len(path)} waypoint(s) from half-solved file.")
                segments.append((role, state, path))
                results.append((role, True, f"reused {len(path)} waypoint(s)"))
                continue

            # M1 derives its own start; M2/M3/M4 should already carry a start
            # config propagated from the previous movement's accept_trajectory.
            # Home is only a fallback when nothing has set one (e.g. a single
            # non-M1 movement run in isolation, or M1 before it derives).
            if state.robot_configuration is None:
                fill_missing_config(
                    state, rcell, _config.HOME_CONF_LEFT_6, _config.HOME_CONF_RIGHT_6,
                )

            # Apply start_state to the live scene.
            t0 = time.time()
            with pp.LockRenderer(False):
                planner.set_robot_cell_state(state)
            print(f"[pb] set_robot_cell_state: {time.time() - t0:.2f}s")

            # Tint the active bar vivid blue so it's easy to track during replay.
            if args.gui and active_bar_rb_name is not None:
                color_rigid_body(planner, active_bar_rb_name, rgba=(0.1, 0.4, 1.0, 1.0))

            # Primary ACM check. Skip it only for an M1 that derives its own
            # start: the config in `state` here is just the HOME placeholder the
            # planner is about to replace, so the check would flag that
            # placeholder's (expected) collisions and mislead. M2/M3/M4 (real
            # propagated starts) and a --no-derive-start M1 (trusted cell start)
            # still get checked.
            if role == "M1" and args.derive_start:
                print(f"[skip-check] {selected.movement_id}: start conf is a HOME "
                      "placeholder (planner derives its own start); skipping "
                      "pre-plan collision check.")
            else:
                check_collision(planner, state, label=selected.movement_id)

            plan_kwargs = dict(
                active_bar_id=active_bar_id,
                active_bar_rb_name=active_bar_rb_name,
                joint_names_12=joint_names_12,
                max_time=args.max_time,
                derive_start=args.derive_start,
                draw=args.diagnosis,
            )
            if args.diagnosis:
                # Diagnosis: keep the renderer UNLOCKED so the search trees draw
                # live as they grow (the whole point of --diagnosis).
                path, info = plan_movement(planner, state, role, selected, **plan_kwargs)
            else:
                # Lock the renderer during planning: the RRT/IK sets many
                # intermediate configs and redrawing each one dominates wall-clock
                # in GUI mode. (No-op in DIRECT.)
                with pp.LockRenderer():
                    path, info = plan_movement(planner, state, role, selected, **plan_kwargs)

            if path is None:
                reason = (info or {}).get("failure_reason", "<unknown>")
                print(f"[plan] {role} FAILED: {reason}")
                results.append((role, False, reason))
                break  # can't chain the next movement without this one's end config

            print(f"[plan] {role} OK: {len(path)} waypoint(s)")
            for k, v in (info or {}).items():
                if k in ("profile", "smooth_profile", "path_poses", "derived_start_conf"):
                    continue
                print(f"  {k}: {v}")

            # Chain bookkeeping: write traj[0] into this movement's start_state
            # (M1/M4), validate it (M2/M3), and propagate traj[-1] into the next
            # movement's start_state. Mirrors husky_monitor._accept_trajectory.
            accepted = accept_trajectory(
                selected, path,
                role=role, index=index, movements=movements,
                rcell=rcell, joint_names_12=joint_names_12, source="Plan",
            )
            if not accepted:
                results.append((role, False, "chain rejected (start mismatch)"))
                break

            # M0 (live-deployment lead-in) is left unplanned offline; its goal is
            # M1's start config. Now that M1 is planned and owns its start config,
            # copy it into M0.target_configuration (backward fill). Mirrors the
            # live monitor's synthetic-M0 backfill.
            if role == "M1":
                m0 = select_movement(action, "M0")
                m1_start = getattr(selected.start_state, "robot_configuration", None)
                if m0 is not None and m1_start is not None:
                    m0.target_configuration = m1_start
                    print("[Plan] backfilled M0.target_configuration <- M1 start config.")

            segments.append((role, selected.start_state, path))
            results.append((role, True, f"{len(path)} waypoint(s)"))

            # Snapshot progress after every solved movement, so a later failure
            # still leaves this one on disk to reload with --load solved.
            save_solved_action(action, save_path)

        # Roll-up summary (most useful in batch mode).
        if len(planning_sequence) > 1:
            print("\n[summary]")
            for role, ok, detail in results:
                mark = "OK  " if ok else "FAIL"
                print(f"  [{mark}] {role}: {detail}")

        all_ok = bool(segments) and all(ok for _, ok, _ in results)

        # Replay.
        if not segments:
            return 2
        if args.no_replay or not args.gui:
            print("[replay] skipped (use --gui without --no-replay to enable).")
            return 0 if all_ok else 2
        if len(segments) == 1:
            _role, state, path = segments[0]
            replay_with_slider(planner, state, path, joint_names_12)
        else:
            replay_segments(planner, segments, joint_names_12)
        return 0 if all_ok else 2
    finally:
        try:
            pp.disconnect()
        except Exception as exc:
            print(f"[pb] disconnect raised ({exc}); continuing.")


if __name__ == "__main__":
    raise SystemExit(main())
