"""Headless planner + slider replay for a single BarAssemblyAction movement.

Mirrors the planning + replay workflow of
``husky-assembly-teleop/scripts/headless_live_monitor_test.py`` but stays
platform-independent and has no dependency on ``husky_assembly_teleop`` or ros2.

Goal: debug how Rhino-generated ``BarAssemblyAction`` JSONs load into
compas_fab + pybullet, run planning, and confirm collision + ACM setup
are correct end-to-end.

CLI:
    python external/husky_assembly_tamp/scripts/headless_bar_action_planner.py [<data_root>]
        --bar-action B6.json     # or --all for every clean bar in the problem
        --movement {M1,M2,M3,M4}
        [--all]               # solve for EVERY clean BarActions/<bar>.json (skip
                              # sidecars like B6.solved.json). Works in BOTH modes:
                              # motion planning, or --solve-keyframes base+IK.
        [--solve-keyframes]   # base+IK keyframe solve instead of motion planning
        [--gui]
        [--max-time 60]
        [--no-replay]
        [--probe-endpoints]   # M1: report start/goal feasibility, skip the RRT
        [--diagnosis]         # M4: draw birrt trees live (needs --gui); no
                              # LockRenderer, timeout-only stop
        [--load {clean,solved_motion,solved_keyframe}]
                              # clean = Rhino export; solved_motion = reuse this
                              # bar's motion sidecar (continue the motion plan);
                              # solved_keyframe = load the keyframe-solve sidecar
                              # (its solved base + keyframe configs) and plan the
                              # trajectories from there
        [--cell <RobotCell.json>]

``<data_root>`` resolves as: the positional argument if given, else the
``HUSKY_ASSEMBLY_DATA_ROOT`` environment variable, else a clear error.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import List, Optional, Sequence, Tuple


# This script lives in the husky_assembly_tamp repo at ``scripts/``. Its
# dependencies are either pip-installed (`pip install -e` this repo into a venv,
# e.g. on Ubuntu) or sibling checkouts under the parent ``external/`` folder when
# this repo is vendored inside the design repo -- locate the siblings relative to
# this file (each is only added if the folder actually exists):
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

# On a Windows machine with Rhino, wire Rhino's pip site-envs (numpy, compas,
# pybullet, ...) onto sys.path so a plain py39 interpreter finds them. No-op
# inside Rhino and on machines without Rhino (e.g. an Ubuntu venv).
from _rhino_env_bootstrap import bootstrap_rhino_site_envs  # noqa: E402

bootstrap_rhino_site_envs(verbose=False)

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Cheap canary for the whole compas / compas_fab / pybullet stack: fail with
# setup instructions instead of a bare ModuleNotFoundError deep in the imports.
try:
    import compas  # noqa: F401, E402
except ImportError as exc:
    raise SystemExit(
        "Missing planning dependencies. Either run with Rhino's py39 interpreter "
        "(its site-envs are wired automatically), or `pip install -e "
        "<husky_assembly_tamp>` into your venv. Underlying error: " + str(exc)
    )

# Movement dataclasses. Imported at the top rather than inside functions: they
# are cheap (pure compas Data classes) and carry no import-order risk.
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

# Keyframe-solve mode (base sampling + IK, the headless twin of RSIKKeyframe).
# Everything comes from the keyframe package in THIS repo -- the design repo's
# Rhino scripts are never imported here; all data comes from the exported JSONs.
from husky_assembly_tamp.keyframe.dual_arm_ik import resolve_arm_groups  # noqa: E402
from husky_assembly_tamp.keyframe.ik_keyframe import (  # noqa: E402
    frame_to_mm4,
    mm4_to_frame,
    solve_keyframe_chain,
)
from husky_assembly_tamp.keyframe.walkable_ground import (  # noqa: E402
    load_walkable_grounds,
    solve_chain_with_base_search,
)


# DEFAULT_PROBLEM = "2026-05-14_foc_demo_reduced"
# DEFAULT_BAR_ACTION = "B226.json"
DEFAULT_PROBLEM = "2026-05-16_double_kissing_jig_demo"
DEFAULT_BAR_ACTION = "B6.json"


def resolve_data_root(cli_value) -> str:
    """Resolve the data root folder: CLI value -> env var -> clear error.

    The data root is machine-specific (a synced folder of exported problems), so
    it is never baked into the code: pass it as the positional argument, or set
    the ``HUSKY_ASSEMBLY_DATA_ROOT`` environment variable once per machine.

    Args:
        cli_value (str | None): the positional ``data_root`` argument, if given.

    Returns:
        str: the absolute data-root path.

    Raises:
        SystemExit: when neither source provides a path -- the message says
            exactly what to pass or set (no silent guessed default).
    """
    if cli_value:
        return os.path.abspath(cli_value)
    env_value = os.environ.get("HUSKY_ASSEMBLY_DATA_ROOT")
    if env_value:
        return os.path.abspath(env_value)
    raise SystemExit(
        "[X] no data root: pass it as the positional argument "
        "(e.g. `headless_bar_action_planner.py <data_root> ...`) or set the "
        "HUSKY_ASSEMBLY_DATA_ROOT environment variable to the folder that holds "
        "the per-problem subfolders."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def arm_joint_names_12(rcell, groups) -> List[str]:
    """The twelve arm-joint names (left 6 then right 6) read from the cell.

    Filters each planning group's configurable joints down to the six UR arm
    joints (dropping e.g. a mobile-base joint the group may also carry), using
    the arm-joint suffix list shared with the motion planners.

    Args:
        rcell: the loaded RobotCell (its semantics name the joints).
        groups (tuple[str, str]): the ``(left_group, right_group)`` names, e.g.
            from ``resolve_arm_groups(rcell)``.

    Returns:
        List[str]: the twelve joint names, left arm first.
    """
    left_group, right_group = groups
    left = [n for n in rcell.get_configurable_joint_names(left_group)
            if any(n.endswith(s) for s in _ARM_SUFFIXES)]
    right = [n for n in rcell.get_configurable_joint_names(right_group)
             if any(n.endswith(s) for s in _ARM_SUFFIXES)]
    return left + right


def home_conf12_from_action(action, joint_names_12: Sequence[str]) -> np.ndarray:
    """The 12-vec home configuration authored into M4's target_configuration.

    The exported BarAction JSON is the source of truth for the home pose: the
    Rhino export writes it as M4's goal (``core.bar_action._build_m4`` on the
    design side). The planner reads it from there instead of keeping its own
    copy of the constant -- so a re-exported action always wins.

    Args:
        action: the loaded BarAssemblyAction.
        joint_names_12 (Sequence[str]): the twelve arm-joint names, in order.

    Returns:
        np.ndarray: the twelve home joint values (left arm first).

    Raises:
        RuntimeError: when M4 (or its target_configuration) is missing -- the
            planner does not invent a home; re-export the bar with the current
            exporter.
    """
    m4 = select_movement(action, "M4")
    target = getattr(m4, "target_configuration", None) if m4 is not None else None
    if target is None:
        raise RuntimeError(
            f"{getattr(action, 'action_id', '<action>')}: M4 has no "
            "target_configuration, so no home pose is available. The Rhino export "
            "writes the home config there -- re-export this bar with the current "
            "exporter (the planner does not invent a home)."
        )
    return vec12_from_conf(target, joint_names_12)


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


def fill_missing_config(state, rcell, groups, home_left: Sequence[float],
                        home_right: Sequence[float]) -> None:
    """Fill a state's robot configuration with the home pose when it has none.

    Planner-side convenience for movements whose start config is deliberately
    absent in the export. If ``state`` already carries a robot configuration,
    nothing happens. Otherwise a fresh zero configuration is built and the two
    arms are set to the given home joint values (read from M4's authored target
    via :func:`home_conf12_from_action` -- the JSON, not a Python constant).

    Args:
        state: The RobotCellState to fill in place. May be ``None`` (ignored).
        rcell: The RobotCell, used to build a zero configuration and to look up
            the left/right arm joint names.
        groups (tuple[str, str]): the ``(left_group, right_group)`` names.
        home_left (Sequence[float]): Home joint values for the left arm.
        home_right (Sequence[float]): Home joint values for the right arm.

    Returns:
        None. ``state.robot_configuration`` is modified in place.
    """
    if state is None or state.robot_configuration is not None:
        return
    left_group, right_group = groups
    cfg = rcell.zero_full_configuration()
    left_names = list(rcell.get_configurable_joint_names(left_group))
    right_names = list(rcell.get_configurable_joint_names(right_group))
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


def conf12_to_configuration(rcell, joint_names_12: Sequence[str], q: Sequence[float]):
    """Build a compas ``Configuration`` from a 12-vec (for ``target_configuration``).

    Starts from the cell's zero full configuration and writes the twelve arm-joint
    values onto it, so the result is a complete configuration the movement schema
    can store as its goal.

    Args:
        rcell: The RobotCell, used to build the zero full configuration.
        joint_names_12 (Sequence[str]): the twelve arm-joint names, left then right.
        q (Sequence[float]): the twelve joint values to write.

    Returns:
        Configuration: the full configuration with the arm joints set to ``q``.
    """
    cfg = rcell.zero_full_configuration()
    for name, val in zip(joint_names_12, q):
        cfg[name] = float(val)
    return cfg


def accept_solved_movement(mv, *, role: str, index: int, movements, rcell,
                           joint_names_12: Sequence[str], path=None,
                           end_conf12=None, source: str = "solve") -> bool:
    """Record a solved movement and chain its END config to the next movement.

    A movement is "solved" either by MOTION planning (a full ``path`` of 12-vec
    waypoints) or by KEYFRAME IK (only its end configuration, ``end_conf12``, with
    no trajectory). This runs the state propagation common to BOTH -- so, for
    example, M3's retreat config becomes M4's start whether or not a trajectory was
    planned. Renamed + generalized from the old ``accept_trajectory`` (headless
    adaptation of ``husky_monitor._accept_trajectory``).

    What it does:
      - motion only: store ``path`` on ``mv.trajectory`` and set/validate the
        movement's start config from ``path[0]`` (M1/M4 own their start; M2/M3
        must already carry the propagated start and are rejected on a mismatch);
      - both: record the movement's END config as ``mv.target_configuration`` when
        it has none yet (M4's authored home / M0's backfilled goal are preserved);
      - both: forward-propagate the END config into the NEXT movement's
        ``start_state.robot_configuration`` (M0/M4 terminate the chain);
      - motion only: a backward continuity check vs the previous movement's
        ``trajectory[-1]``.

    Args:
        mv: the just-solved Movement.
        role (str): this movement's role, ``'M0'``..``'M4'``.
        index (int): position of ``mv`` inside ``movements``.
        movements: the movement list to reach the next/previous ones (planning
            order for motion; ``action.movements`` for keyframe).
        rcell: the RobotCell, used to build configurations.
        joint_names_12 (Sequence[str]): the twelve arm-joint names, in order.
        path: motion waypoints (each a 12-vec), or ``None`` for a keyframe solve.
        end_conf12: the movement's END config as a 12-vec; defaults to ``path[-1]``.
            Required when ``path`` is ``None``.
        source (str): short label printed with each log line.

    Returns:
        bool: True when accepted, False on a chain break (motion start mismatch or
        a missing endpoint), so the caller can stop chaining.
    """
    if end_conf12 is None:
        if not path:
            print(f"[{source}] {getattr(mv, 'movement_id', '?')!r}: no path and no "
                  "end_conf12 to accept.")
            return False
        end_conf12 = path[-1]
    end_vec = np.asarray(end_conf12, dtype=float)

    # --- motion: store the trajectory + set/validate this movement's start. ---
    if path:
        mv.trajectory = path
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

    # --- both: record the movement's goal config (fill-if-empty so authored
    # targets like M4's home survive). ---
    if getattr(mv, "target_configuration", None) is None:
        mv.target_configuration = conf12_to_configuration(rcell, joint_names_12, end_vec)

    # --- both: forward-propagate the END config into the next movement's start. ---
    if role not in ("M0", "M4") and index + 1 < len(movements):
        next_mv = movements[index + 1]
        if next_mv is not None and next_mv.start_state is not None:
            existing = next_mv.start_state.robot_configuration
            if existing is None:
                apply_conf12(next_mv.start_state, rcell, joint_names_12, end_vec)
                print(f"[{source}] propagated {mv.movement_id!r} end -> "
                      f"{next_mv.movement_id!r}.start_state.robot_configuration "
                      "(was None).")
            else:
                diff = float(np.abs(end_vec - vec12_from_conf(existing, joint_names_12)).max())
                if diff > 1e-3:
                    print(f"[{source}] end of {mv.movement_id!r} differs from existing "
                          f"{next_mv.movement_id!r}.start by max {diff:.4f} rad/m; "
                          "overwriting (chain rule).")
                apply_conf12(next_mv.start_state, rcell, joint_names_12, end_vec)

    # --- motion only: backward continuity check vs the previous trajectory. ---
    if path and index > 0:
        prev_mv = movements[index - 1]
        prev_path = getattr(prev_mv, "trajectory", None)
        if prev_path:
            diff = float(np.abs(
                np.asarray(prev_path[-1], dtype=float) - np.asarray(path[0], dtype=float)
            ).max())
            if diff > 1e-3:
                print(f"[{source}] start of {mv.movement_id!r} differs from "
                      f"{prev_mv.movement_id!r}.trajectory[-1] by max {diff:.4f} rad/m.")
            else:
                print(f"[{source}] start agrees with {prev_mv.movement_id!r}."
                      f"trajectory[-1] (max diff {diff:.6f}).")

    print_roster(movements, source)
    return True


def solved_action_path(clean_action_path: str, kind: str) -> str:
    """Sidecar path for the half-solved BarAction, next to the clean file.

    The two solve modes write SEPARATE sidecars so a keyframe solve and a motion
    plan for the same bar don't overwrite each other:

    - ``kind="keyframe"`` -> ``.../BarActions/B6.solved_keyframe.json``
      (base + IK keyframe configs from ``--solve-keyframes``)
    - ``kind="motion"``   -> ``.../BarActions/B6.solved_motion.json``
      (full motion-planned trajectories)

    The clean file (the Rhino export) is never overwritten; only these sidecars
    are. Their dotted stems mean the ``--all`` clean-file scan skips them.

    Args:
        clean_action_path (str): the clean ``<bar>.json`` export path.
        kind (str): ``"keyframe"`` or ``"motion"``.

    Returns:
        str: the sidecar path.
    """
    tag = {"keyframe": "solved_keyframe", "motion": "solved_motion"}[kind]
    stem, ext = os.path.splitext(clean_action_path)
    return f"{stem}.{tag}{ext}"


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
        # M4 returns to the fixed dual-arm home AUTHORED INTO THE EXPORT (the
        # Rhino side writes it as M4's target_configuration). The JSON is the
        # source of truth -- there is no Python-side home constant to fall back on.
        if goal_conf is None:
            return None, {"failure_reason": (
                "M4 has no target_configuration (the authored home pose). "
                "Re-export this bar with the current exporter -- the planner "
                "does not invent a home."
            )}
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
                  "(diagnosis; goal = M4 target_configuration, the authored home)")
        else:
            print("[plan] plan_free_dual_arm (goal = M4 target_configuration, "
                  "the authored home)")
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
                    joint_names_12: Sequence[str], *, groups, home12,
                    use_gui: bool = False) -> int:
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
        groups (tuple[str, str]): the ``(left_group, right_group)`` names.
        home12 (np.ndarray | None): the authored home 12-vec from the JSON, used
            only when M1's start config is absent; ``None`` when the export has
            no M4 target (then a missing start config is a hard error).

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
        if home12 is None:
            print("[probe] M1 has no start config and the action has no authored "
                  "home (M4 target_configuration) to fill it from; re-export the bar.")
            return 2
        fill_missing_config(state, rcell, groups, home12[:6], home12[6:])
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


# ---------------------------------------------------------------------------
# Keyframe-solve mode (base sampling + IK -- the headless twin of RSIKKeyframe)
# ---------------------------------------------------------------------------


def _bar_midpoint_mm(m2) -> Optional[np.ndarray]:
    """Bar assembled midpoint (mm) = midpoint of M2's two EE target points.

    The mate movement (M2) targets the two tool0 flanges at the assembled pose;
    their midpoint is a good stand-in for the bar center, and it needs no bar
    geometry -- just the frames already in the action. Used as the sampling seed
    reference + heading in ``core.walkable_ground``.

    Args:
        m2: the M2 (mate) Movement.

    Returns:
        Optional[np.ndarray]: the midpoint in mm, or None if M2 lacks targets.
    """
    targets = getattr(m2, "target_ee_frames", None) or {}
    left = targets.get("left")
    right = targets.get("right")
    if left is None or right is None:
        return None
    pl = np.asarray(left.point, dtype=float) * 1000.0
    pr = np.asarray(right.point, dtype=float) * 1000.0
    return 0.5 * (pl + pr)


def _avg_insertion_dir_mm(m2) -> Optional[np.ndarray]:
    """Average male-joint insertion direction = avg of M2's two tool0 +Z axes.

    The tools push the male joints into their mates along +tool0_z (tool block
    local +Z points out of the flange toward the joint), so the average of the
    two assembled tool0 Z-axes is the assembly/insertion axis. This is the
    direction the seed mobile base faces (standing behind the bar), removing the
    left/right side ambiguity of the old bar-axis heuristic.

    Args:
        m2: the M2 (mate) Movement; its ``target_ee_frames`` are the assembled
            tool0 frames.

    Returns:
        Optional[np.ndarray]: the (unnormalized) insertion direction vector, or
        None if M2 lacks left/right targets.
    """
    targets = getattr(m2, "target_ee_frames", None) or {}
    left = targets.get("left")
    right = targets.get("right")
    if left is None or right is None:
        return None
    avg = np.asarray(left.zaxis, dtype=float) + np.asarray(right.zaxis, dtype=float)
    if float(np.linalg.norm(avg)) < 1e-9:
        return None
    return avg


def plot_config_vs_limits(state, rcell, joint_names: Sequence[str], width: int = 34) -> str:
    """Return an ASCII plot of each arm joint value within its ``[lower, upper]`` limit.

    One line per joint, e.g.::

        L J1 shoulder_pan     -6.28 [----------|(-1.10)---------------] +6.28
        R J5 wrist_2          -6.28 [-----------------------|(+2.71)--] +6.28  !OUT-OF-LIMIT

    The ``|`` marks where the value sits inside the joint's range and ``(v)`` is
    the value in radians; a value outside its limits is flagged. Continuous joints
    (no limit) are printed as such. Reads limits from
    ``rcell.robot_model.get_joint_by_name(name).limit`` (lower/upper).

    Args:
        state: the solved RobotCellState (its ``robot_configuration`` is read).
        rcell: the RobotCell (its ``robot_model`` supplies the joint limits).
        joint_names (Sequence[str]): the arm joint names in order (left 6 then
            right 6), used both to read values and to label ``L/R J1..J6``.
        width (int): inner width of the ``[...]`` bar in characters.

    Returns:
        str: the multi-line plot (one line per joint).
    """
    cfg = state.robot_configuration
    model = rcell.robot_model
    n_left = len(joint_names) // 2
    lines = []
    for idx, name in enumerate(joint_names):
        val = float(cfg[name])
        joint = model.get_joint_by_name(name)
        lim = getattr(joint, "limit", None)
        side = "L" if idx < n_left else "R"
        jnum = (idx % n_left) + 1 if n_left else idx + 1
        short = "_".join(re.sub(r"_joint$", "", str(name)).split("_")[-2:])
        label = f"{side} J{jnum} {short:<16}"
        if lim is None:
            lines.append(f"{label} [continuous / no limit]   value={val:+.3f}")
            continue
        lo, hi = float(lim.lower), float(lim.upper)
        frac = (val - lo) / (hi - lo) if hi > lo else 0.5
        pos = int(round(max(0.0, min(1.0, frac)) * (width - 1)))
        left = "-" * pos
        right = "-" * (width - 1 - pos)
        flag = "" if lo <= val <= hi else "  !OUT-OF-LIMIT"
        lines.append(f"{label} {lo:+6.2f} [{left}|({val:+.2f}){right}] {hi:+6.2f}{flag}")
    return "\n".join(lines)


def make_keyframe_pause_hook(planner, rcell, joint_names: Sequence[str]):
    """Build an ``after_solve(role, state)`` hook for ``solve_keyframe_chain``.

    On each solved keyframe the hook (1) pushes the pose into PyBullet so it is
    visible in the GUI, (2) prints the joint-vs-limit plot, and (3) calls
    ``pp.wait_if_gui`` to pause. ``wait_if_gui`` only blocks when a GUI is
    connected, so this is a no-op pause in DIRECT mode. It is the place to
    inspect *why* M2 fails from M1's just-solved pose.

    Args:
        planner: the active PyBulletPlanner (its cell is shown/paused).
        rcell: the RobotCell (for joint limits in the plot).
        joint_names (Sequence[str]): arm joint names (left 6 then right 6).

    Returns:
        callable: ``after_solve(role, state)`` for ``solve_keyframe_chain``.
    """
    def _hook(role, state):
        try:
            planner.set_robot_cell_state(state)
        except Exception as exc:  # visualization must never break the solve
            print(f"[viz] {role}: could not push state to PyBullet ({exc}).")
        print(f"\n[viz] {role} solved -- arm joints vs limits:")
        print(plot_config_vs_limits(state, rcell, joint_names))
        pp.wait_if_gui(f"{role} solved; press Enter in the PyBullet window to continue")
    return _hook


def _write_back_solve(action, base_frame_mm: np.ndarray, solved: dict,
                      rcell, joint_names_12: Sequence[str]) -> None:
    """Stamp the found base + keyframe configs back onto the action's movements.

    The M1->M2->M3 chain solves IK for each movement's GOAL ee-frames, so
    ``solved["M1"]`` is the APPROACH config, ``solved["M2"]`` the ASSEMBLED config,
    and ``solved["M3"]`` the RETREAT config. The base frame goes on every
    movement's ``start_state``; the per-keyframe configs are chained onto the
    movements with the SAME propagation the motion path uses
    (:func:`accept_solved_movement`, called with ``path=None`` since a keyframe has
    no trajectory). So each solved movement's end config becomes its
    ``target_configuration`` and the next movement's start config:

      - approach  -> M1.target + M2.start,
      - assembled -> M2.target + M3.start,
      - retreat   -> M3.target + M4.start (M4 then runs retreat -> home).

    The retreat config used to be discarded, leaving M3's goal config and M4's
    start config empty; the shared propagation now fills them.

    Args:
        action: the BarAssemblyAction being solved (mutated in place).
        base_frame_mm (np.ndarray): the accepted 4x4 mm base frame.
        solved (dict): ``{"M1": state, "M2": state, "M3": state}`` from the chain.
        rcell: the RobotCell, used to build configurations.
        joint_names_12 (Sequence[str]): the twelve arm-joint names, in order.

    Returns:
        None.
    """
    base_frame = mm4_to_frame(base_frame_mm)
    for mv in action.movements:
        if mv.start_state is not None:
            mv.start_state.robot_base_frame = base_frame

    # Chain the solved keyframe configs with the shared propagation. Process in
    # order so each movement's start is already set by the previous propagation.
    for role in ("M1", "M2", "M3"):
        mv = select_movement(action, role)
        if mv is None or role not in solved:
            continue
        end_vec = vec12_from_conf(solved[role].robot_configuration, joint_names_12)
        accept_solved_movement(
            mv, role=role, index=action.movements.index(mv),
            movements=action.movements, rcell=rcell,
            joint_names_12=joint_names_12, end_conf12=end_vec, source="keyframe",
        )


def solve_keyframes_for_action(planner, action, clean_action_path: str, *,
                               base_mode: str, grounds: dict, groups,
                               after_solve=None) -> Tuple[bool, str]:
    """Solve base + IK keyframes for one BarAction; write back on success.

    Args:
        planner: the active PyBulletPlanner (cell already loaded).
        action: the loaded BarAssemblyAction.
        clean_action_path (str): the clean file path (the .solved.json sidecar is
            derived from it; the clean file is never overwritten).
        base_mode (str): ``"saved"`` (use the stored base, no sampling) or
            ``"sample"`` (auto-seed + expanding-radius search on the ground).
        grounds (dict): ``{ground_id: TriangleSoup}`` from
            ``load_walkable_grounds`` (only used for ``"sample"``).
        groups (tuple[str, str]): the ``(left_group, right_group)`` names derived
            from the cell semantics (``resolve_arm_groups``).
        after_solve (callable | None): optional ``after_solve(role, state)`` hook
            forwarded to ``solve_keyframe_chain`` in the ``saved`` path (used by
            --gui to pause + plot each keyframe); not used in ``sample`` mode.

    Returns:
        Tuple[bool, str]: ``(ok, detail)`` -- detail is a short human message.
    """
    m1 = select_movement(action, "M1")
    m2 = select_movement(action, "M2")
    m3 = select_movement(action, "M3")
    if m1 is None or m2 is None or m3 is None:
        return False, "missing one of M1/M2/M3"
    if m1.start_state is None:
        return False, "M1 has no start_state"
    movements = {"M1": m1, "M2": m2, "M3": m3}
    ordered = [("M1", m1), ("M2", m2), ("M3", m3)]

    rcell = planner.client.robot_cell
    joint_names_12 = arm_joint_names_12(rcell, groups)
    # The home pose orders ssik's cold candidate pairs; it comes from the JSON
    # (M4's authored target), never from a Python constant. A missing home is a
    # per-bar failure (batch runs keep going), with the fix in the message.
    try:
        home12 = home_conf12_from_action(action, joint_names_12)
    except RuntimeError as exc:
        return False, str(exc)

    if base_mode == "saved":
        # Use the base frame stored in the export as-is (no sampling), then solve
        # the keyframe IK chain at it. We do NOT require any saved start
        # configuration: keyframe IK solves for each movement's GOAL ee-frames,
        # so a movement with no start config just gets a cold solve (any valid
        # IK), and the chain warm-starts each later movement from the previous
        # one's solution to stay on the same IK branch. M1 in particular is
        # *designed* to carry no start config (its start is planner-filled later),
        # so a missing start config must never abort the solve.
        base_frame = getattr(m1.start_state, "robot_base_frame", None)
        if base_frame is None:
            return False, "no saved base frame on M1 start_state"
        used_base = frame_to_mm4(base_frame)
        solved = solve_keyframe_chain(
            planner, ordered, used_base, check_collision=True, after_solve=after_solve,
            groups=groups, home_conf_12=home12,
        )
    else:  # sample
        gids = getattr(action, "walkable_ground_ids", None) or []
        soups = [grounds[g] for g in gids if g in grounds]
        if not soups:
            return False, f"no associated WalkableGround (ids={gids})"
        midpoint = _bar_midpoint_mm(m2)
        if midpoint is None:
            return False, "M2 has no left/right target_ee_frames"
        solved, used_base = solve_chain_with_base_search(
            planner, movements, soups, midpoint,
            heading_dir_mm=_avg_insertion_dir_mm(m2), check_collision=True,
            groups=groups, home_conf_12=home12,
        )

    if solved is None:
        return False, "IK chain failed for all base attempts"

    _write_back_solve(action, used_base, solved, rcell, joint_names_12)
    save_path = solved_action_path(clean_action_path, "keyframe")
    save_solved_action(action, save_path)
    o = used_base[:3, 3]
    return True, f"base ({o[0]:.0f},{o[1]:.0f},{o[2]:.0f})mm -> {os.path.basename(save_path)}"


def run_solve_keyframes(args, problem_dir: str) -> int:
    """Keyframe-solve entry point: single or batch base+IK over BarAction file(s).

    Args:
        args: parsed CLI args.
        problem_dir (str): the resolved ``<data_root>/<problem>`` directory.

    Returns:
        int: 0 when every action solved, 2 when any failed, 1 on setup errors.
    """
    cell_path = args.cell or os.path.join(problem_dir, "RobotCell.json")
    if not os.path.isfile(cell_path):
        print(f"[X] missing RobotCell.json: {cell_path}")
        return 1

    actions_dir = os.path.join(problem_dir, "BarActions")
    if args.all_bars:
        if not os.path.isdir(actions_dir):
            print(f"[X] missing BarActions dir: {actions_dir}")
            return 1
        action_files = sorted(
            os.path.join(actions_dir, f) for f in os.listdir(actions_dir)
            # Only CLEAN export files "<bar_id>.json". Skip any sidecar that has an
            # extra dotted tag in the middle -- e.g. "B6.solved.json" (half-solved
            # write-back) or "B6.20260430_002744.json" (timestamped capture) -- by
            # requiring the stem (name minus ".json") to contain no dot.
            if f.endswith(".json") and "." not in os.path.splitext(f)[0]
        )
        if not action_files:
            print(f"[X] no clean BarAction files in {actions_dir}")
            return 1
    else:
        one = os.path.join(actions_dir, args.bar_action)
        if not os.path.isfile(one):
            print(f"[X] missing BarAction file: {one}")
            return 1
        action_files = [one]

    grounds = {}
    if args.base == "sample":
        wg_path = args.walkable_ground or os.path.join(problem_dir, "WalkableGround.json")
        if not os.path.isfile(wg_path):
            print(f"[X] --base sample needs WalkableGround.json: {wg_path}")
            return 1
        grounds = load_walkable_grounds(wg_path)
        print(f"[load] WalkableGround <- {wg_path} ({len(grounds)} ground(s))")

    print(f"[load] RobotCell    <- {cell_path}")
    rcell = json_load(cell_path)
    # The planning-group names come from the cell's own SRDF semantics (embedded
    # in RobotCell.json) -- nothing is hardcoded here. The solvers read the cell
    # straight from the planner (`start_planner` pushes it), so no extra
    # registration step is needed.
    groups = resolve_arm_groups(rcell)
    print(f"  planning groups: left={groups[0]!r} right={groups[1]!r}")

    print(f"\n[pb] starting PyBullet ({'GUI' if args.gui else 'DIRECT'})")
    _client, planner = start_planner(rcell, use_gui=args.gui)

    # In --gui mode, pause + plot after each solved keyframe so the pose can be
    # inspected (e.g. "M1 solved, why does M2 fail from here?"). Only the `saved`
    # base mode threads this hook through -- `sample` mode would otherwise pause on
    # every one of many base attempts. Off (None) in headless batch runs.
    after_solve = None
    if args.gui and args.base == "saved":
        after_solve = make_keyframe_pause_hook(
            planner, rcell, arm_joint_names_12(rcell, groups)
        )

    results = []
    try:
        for path in action_files:  # `path` is always the CLEAN <bar>.json
            bar_file = os.path.basename(path)
            # --load names which file to solve from: 'clean' = the Rhino export;
            # 'solved_keyframe' = this bar's prior keyframe solve (e.g. so --base
            # saved can pick up the base it already found); 'solved_motion' = its
            # motion sidecar. Either way the write-back goes to the keyframe sidecar
            # derived from the clean path, so the motion sidecar is never touched.
            if args.load == "clean":
                load_path = path
            else:
                kind = "motion" if args.load == "solved_motion" else "keyframe"
                sidecar = solved_action_path(path, kind)
                if not os.path.isfile(sidecar):
                    print(f"\n[solve] {bar_file}: --load {args.load} but no "
                          f"{os.path.basename(sidecar)}; skipping.")
                    results.append((bar_file, False, f"no {os.path.basename(sidecar)}"))
                    continue
                load_path = sidecar
            action = json_load(load_path)
            bar_id = getattr(action, "active_bar_id", "") or bar_file
            print(f"\n[solve] {bar_id} ({os.path.basename(load_path)}) base={args.base}")
            ok, detail = solve_keyframes_for_action(
                planner, action, path, base_mode=args.base, grounds=grounds,
                groups=groups, after_solve=after_solve,
            )
            mark = "OK  " if ok else "FAIL"
            print(f"  [{mark}] {bar_id}: {detail}")
            results.append((bar_id, ok, detail))

        # Print the summary BEFORE disconnecting. In --gui mode pp.disconnect()
        # tears down the PyBullet window, which on Windows invalidates the console
        # stdout handle -- any print() after it then raises "OSError [WinError 6]
        # The handle is invalid". So all output happens here first, and disconnect
        # (in the finally) is the very last thing we do.
        n_ok = sum(1 for _, ok, _ in results if ok)
        print(f"\n[summary] keyframe-solve: {n_ok}/{len(results)} solved")
        for bar_id, ok, detail in results:
            mark = "OK  " if ok else "FAIL"
            print(f"  [{mark}] {bar_id}: {detail}")
        try:
            sys.stdout.flush()
        except Exception:
            pass
    finally:
        # Last operation: no stdout writes after this (see note above). A raising
        # disconnect must not itself try to print (that would hit the same
        # invalidated handle), so swallow it silently.
        try:
            pp.disconnect()
        except Exception:
            pass
    return 0 if results and all(ok for _, ok, _ in results) else 2


def plan_one_action(planner, rcell, clean_action_path: str, args, groups):
    """Run full motion planning for ONE BarAction file (no PyBullet lifecycle / replay).

    Loads the action named by ``--load``: the clean export (``clean``), this bar's
    motion sidecar to continue an in-progress plan (``solved_motion``), or the
    keyframe sidecar from ``--solve-keyframes`` to plan on top of its solved base +
    keyframe configs (``solved_keyframe``). Runs the planning sequence (chaining
    M1->M2->M3->M4 for
    ``--movement all``), and writes the motion sidecar after each solved movement.
    The
    PyBullet client/planner is created once by the caller and shared across a batch;
    replay stays in the caller so ``--all`` runs skip it.

    Args:
        planner: the active PyBulletPlanner (dual-arm cell already loaded).
        rcell: the loaded RobotCell.
        clean_action_path (str): path to the clean ``BarActions/<bar>.json``.
        args: parsed CLI args.
        groups (tuple[str, str]): the ``(left_group, right_group)`` names derived
            from the cell semantics (``resolve_arm_groups``).

    Returns:
        tuple: ``(all_ok, segments, joint_names_12, active_bar_id)`` where
        ``segments`` is ``[(role, state, path), ...]`` for optional single-bar
        replay (empty on a setup error or in ``--probe-endpoints`` mode).
    """
    bar_file = os.path.basename(clean_action_path)

    # The clean file (Rhino export) is never overwritten; the half-solved motion
    # snapshot is always written to this sidecar, regardless of which we load.
    save_path = solved_action_path(clean_action_path, "motion")
    if args.load == "clean":
        action_path = clean_action_path
    else:
        # --load names exactly which partly-solved sidecar to start from:
        #   solved_motion   -- this bar's motion sidecar: continue an in-progress
        #                      motion plan, keeping its already-planned trajectories.
        #   solved_keyframe -- the keyframe-solve sidecar from --solve-keyframes: it
        #                      carries the solved robot base + per-keyframe
        #                      start/goal configs but NO trajectories, so motion
        #                      planning starts from that base and fills in the
        #                      trajectories between the keyframes.
        # Either way the write-back still goes to save_path (the motion sidecar), so
        # loading solved_keyframe never overwrites the keyframe solve.
        kind = "motion" if args.load == "solved_motion" else "keyframe"
        action_path = solved_action_path(clean_action_path, kind)
        if not os.path.isfile(action_path):
            print(f"[X] {bar_file}: --load {args.load} but no such sidecar yet: "
                  f"{action_path}")
            return False, [], [], ""

    print(f"[load] BarAction ({args.load}) <- {action_path}")
    action = json_load(action_path)
    active_bar_id = getattr(action, "active_bar_id", "") or ""
    print(f"  action_id     : {action.action_id}")
    print(f"  active_bar_id : {active_bar_id}")
    print(f"  movements     : {len(action.movements)}")

    # The cell stores bars as rigid bodies named ``bar_<bar_id>``. The planner API
    # needs that rigid-body name, not the bare bar id.
    def _resolve_bar_rb_name(cell, bar_id: str) -> Optional[str]:
        for candidate in (bar_id, f"bar_{bar_id}"):
            if candidate in cell.rigid_body_models:
                return candidate
        return None

    active_bar_rb_name = _resolve_bar_rb_name(rcell, active_bar_id) if active_bar_id else None
    if active_bar_id and active_bar_rb_name is None:
        print(f"[X] {bar_file}: active_bar_id {active_bar_id!r} not in cell.rigid_body_models")
        return False, [], [], active_bar_id
    if active_bar_rb_name and active_bar_rb_name != active_bar_id:
        print(f"  bar rigid-body name: {active_bar_rb_name}")

    # Planning sequence: the order movements are planned in (not just a set of
    # roles). Order matters — each solved movement's start/end config propagates
    # into its neighbours (see accept_trajectory), so M1 must plan before M2, etc.
    planning_sequence = (
        ["M1", "M2", "M3", "M4"] if args.movement == "all" else [args.movement]
    )

    # 12-vec joint names — read from the cell, shared across movements.
    joint_names_12 = arm_joint_names_12(rcell, groups)

    # The authored home pose, read from the JSON (M4's target_configuration). It
    # is only needed as the fill-in for movements exported without a start config;
    # a missing home only fails a movement that actually needs the fill.
    try:
        home12 = home_conf12_from_action(action, joint_names_12)
    except RuntimeError as exc:
        home12 = None
        print(f"[!] {exc}")

    # Resolve the movements once, in planning order, and mutate their start_states
    # in place — accept_trajectory propagates each planned endpoint into the next
    # movement's start_state (the key chaining step).
    movements = [select_movement(action, r) for r in planning_sequence]

    segments = []          # (role, state, path) for each solved movement
    results = []           # (role, ok, detail) for the roll-up

    # Pre-flight: show what will be planned, in order, before we start.
    print_roster(movements, tag="pre-flight")

    # Endpoint feasibility probe (M1): derive + report start/goal, no RRT.
    if args.probe_endpoints:
        code = probe_endpoints(
            planner, rcell, action, active_bar_rb_name, joint_names_12,
            groups=groups, home12=home12, use_gui=args.gui,
        )
        return code == 0, [], joint_names_12, active_bar_id

    for index, role in enumerate(planning_sequence):
        selected = movements[index]
        if selected is None:
            msg = f"no movement matches role {role!r} in {bar_file}"
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

        # Reuse an already-planned trajectory from the loaded sidecar: skip
        # planning, keep it for replay + chaining. Only the motion sidecar carries
        # trajectories; a keyframe sidecar has none, so this simply never fires for
        # --load solved_keyframe (every movement is planned fresh from its base).
        existing_traj = getattr(selected, "trajectory", None)
        if args.load != "clean" and existing_traj:
            path = [[float(v) for v in wp] for wp in existing_traj]
            print(f"[reuse] {role}: {len(path)} waypoint(s) from half-solved file.")
            segments.append((role, state, path))
            results.append((role, True, f"reused {len(path)} waypoint(s)"))
            continue

        # M1 derives its own start; M2/M3/M4 should already carry a start config
        # propagated from the previous movement's accept_trajectory. The authored
        # home (from the JSON) is only a fallback when nothing has set one.
        if state.robot_configuration is None:
            if home12 is None:
                msg = (f"{selected.movement_id}: no start config and no authored "
                       "home (M4 target_configuration) to fill it from; "
                       "re-export the bar.")
                print(f"[X] {msg}")
                results.append((role, False, msg))
                break
            fill_missing_config(state, rcell, groups, home12[:6], home12[6:])

        # Apply start_state to the live scene.
        t0 = time.time()
        with pp.LockRenderer(False):
            planner.set_robot_cell_state(state)
        print(f"[pb] set_robot_cell_state: {time.time() - t0:.2f}s")

        # Tint the active bar vivid blue so it's easy to track during replay.
        if args.gui and active_bar_rb_name is not None:
            color_rigid_body(planner, active_bar_rb_name, rgba=(0.1, 0.4, 1.0, 1.0))

        # Primary ACM check. Skip it only for an M1 that derives its own start (the
        # config here is just the HOME placeholder the planner is about to replace).
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
            # Diagnosis: keep the renderer UNLOCKED so the search trees draw live.
            path, info = plan_movement(planner, state, role, selected, **plan_kwargs)
        else:
            # Lock the renderer during planning (redraw of intermediate configs
            # dominates wall-clock in GUI mode; no-op in DIRECT).
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

        # Chain bookkeeping: propagate traj[0]/[-1] into neighbouring start_states.
        accepted = accept_solved_movement(
            selected, role=role, index=index, movements=movements,
            rcell=rcell, joint_names_12=joint_names_12, path=path, source="Plan",
        )
        if not accepted:
            results.append((role, False, "chain rejected (start mismatch)"))
            break

        # M0 is left unplanned offline; backfill its goal from M1's start config.
        if role == "M1":
            m0 = select_movement(action, "M0")
            m1_start = getattr(selected.start_state, "robot_configuration", None)
            if m0 is not None and m1_start is not None:
                m0.target_configuration = m1_start
                print("[Plan] backfilled M0.target_configuration <- M1 start config.")

        segments.append((role, selected.start_state, path))
        results.append((role, True, f"{len(path)} waypoint(s)"))

        # Snapshot progress after every solved movement.
        save_solved_action(action, save_path)

    # Per-bar roll-up (most useful in --movement all).
    if len(planning_sequence) > 1:
        print("\n[summary]")
        for role, ok, detail in results:
            mark = "OK  " if ok else "FAIL"
            print(f"  [{mark}] {role}: {detail}")

    all_ok = bool(segments) and all(ok for _, ok, _ in results)
    return all_ok, segments, joint_names_12, active_bar_id


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
        "data_root", nargs="?", default=None,
        help="Parent folder containing per-problem subfolders. Falls back to the "
             "HUSKY_ASSEMBLY_DATA_ROOT environment variable; errors if neither "
             "is given.",
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
        "--movement", default=None, choices=("M0", "M1", "M2", "M3", "M4", "all"),
        help="Which movement role to plan (required unless --solve-keyframes). "
             "'all' chains M1->M2->M3->M4 (M0 is left unplanned offline -- the "
             "live monitor plans it). 'M0' only works after M1 has been planned "
             "so its goal config is backfilled.",
    )
    parser.add_argument(
        "--solve-keyframes", action="store_true",
        help="Keyframe-solve mode (the headless twin of RSIKKeyframe): solve the "
             "M1->M2->M3 IK chain to find a robot base + per-keyframe configs, "
             "instead of running motion planning. Writes the found base into every "
             "movement's start_state + a .solved.json sidecar. Ignores --movement.",
    )
    parser.add_argument(
        "--all", dest="all_bars", action="store_true",
        help="Solve for EVERY clean BarActions/<bar>.json in the problem dir "
             "(files with an extra dotted tag like B6.solved.json are skipped), "
             "instead of just --bar-action. Applies to BOTH modes: with "
             "--solve-keyframes it base+IK-solves every bar; without it, it runs "
             "full motion planning for every bar. Single-bar replay / "
             "--probe-endpoints / --diagnosis are not available with --all.",
    )
    parser.add_argument(
        "--base", choices=("sample", "saved"), default="sample",
        help="Keyframe-solve mode: 'sample' (default) auto-seeds on the bar's "
             "associated WalkableGround and grows the sample radius until the "
             "chain solves; 'saved' strictly uses the base already stored in the "
             "BarAction (no sampling).",
    )
    parser.add_argument(
        "--walkable-ground", default=None,
        help="Keyframe-solve mode: path to WalkableGround.json (default: "
             "<problem>/WalkableGround.json). Only needed for --base sample.",
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--max-time", type=float, default=60.0,
        help="Motion-planning time budget in seconds, per movement -- how long "
             "the RRT motion planner is allowed to search for a collision-free "
             "path before giving up (default: 60). Not the IK-solve budget.",
    )
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument(
        "--no-derive-start", dest="derive_start", action="store_false",
        help="Trust M1's start config from the cell state instead of deriving "
             "a feasible grasp-consistent start (default: derive).",
    )
    parser.set_defaults(derive_start=True)
    parser.add_argument(
        "--load", choices=("clean", "solved_motion", "solved_keyframe"),
        default="clean",
        help="Which BarAction file to load. 'clean' (default) = the Rhino export, "
             "solve from scratch. 'solved_motion' = this bar's motion sidecar "
             "<bar>.solved_motion.json (keep its already-planned trajectories, plan "
             "only the missing ones). 'solved_keyframe' = the keyframe-solve "
             "sidecar <bar>.solved_keyframe.json from --solve-keyframes (its solved "
             "base + per-keyframe configs, with trajectories still to plan) -- use "
             "this to run motion planning on top of a keyframe solve, or (in "
             "--solve-keyframes mode) so --base saved picks up the base it found. "
             "The clean file is never overwritten; motion planning always writes "
             "<bar>.solved_motion.json and --solve-keyframes always writes "
             "<bar>.solved_keyframe.json, so loading one sidecar never overwrites "
             "the other.",
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

    data_root = resolve_data_root(args.data_root)
    if not os.path.isdir(data_root):
        print(f"[X] missing data_root: {data_root}")
        return 1

    problem_dir = os.path.join(data_root, args.problem)
    if not os.path.isdir(problem_dir):
        print(f"[X] missing problem dir: {problem_dir}")
        return 1

    # Keyframe-solve mode has its own single/batch lifecycle (no --movement, its
    # own cell + planner + write-back). Branch off before the planning setup.
    if args.solve_keyframes:
        return run_solve_keyframes(args, problem_dir)

    if args.movement is None:
        print("[X] --movement is required unless --solve-keyframes is set.")
        return 1

    # Single bar (--bar-action) or every clean bar (--all). "Clean" = a plain
    # "<bar_id>.json" with no extra dotted tag in the stem, so sidecars like
    # "B6.solved.json" / timestamped captures are skipped.
    actions_dir = os.path.join(problem_dir, "BarActions")
    if args.all_bars:
        if args.probe_endpoints or args.diagnosis:
            print("[X] --all is a batch run; --probe-endpoints / --diagnosis are "
                  "single-bar debug modes. Drop --all or the debug flag.")
            return 1
        if not os.path.isdir(actions_dir):
            print(f"[X] missing BarActions dir: {actions_dir}")
            return 1
        action_files = sorted(
            os.path.join(actions_dir, f) for f in os.listdir(actions_dir)
            if f.endswith(".json") and "." not in os.path.splitext(f)[0]
        )
        if not action_files:
            print(f"[X] no clean BarAction files in {actions_dir}")
            return 1
    else:
        one = os.path.join(actions_dir, args.bar_action)
        if not os.path.isfile(one):
            print(f"[X] missing BarAction file: {one}")
            return 1
        action_files = [one]

    cell_path = args.cell or os.path.join(problem_dir, "RobotCell.json")
    if not os.path.isfile(cell_path):
        print(f"[X] missing RobotCell.json: {cell_path}")
        return 1

    print(f"[load] RobotCell    <- {cell_path}")
    rcell = json_load(cell_path)
    print(f"  robot model   : {getattr(rcell.robot_model, 'name', '<?>')}")
    print(f"  tool models   : {sorted(rcell.tool_models.keys())}")
    print(f"  rigid bodies  : {len(rcell.rigid_body_models)}")
    # Planning-group names come from the cell's own SRDF semantics, not constants.
    groups = resolve_arm_groups(rcell)
    print(f"  planning groups: left={groups[0]!r} right={groups[1]!r}")

    print(f"\n[pb] starting PyBullet ({'GUI' if args.gui else 'DIRECT'})")
    _client, planner = start_planner(rcell, use_gui=args.gui)

    try:
        single = (len(action_files) == 1)
        results = []          # (bar_file, all_ok)
        replayable = None     # (segments, joint_names_12) for a single-bar replay
        for clean_action_path in action_files:
            bar_file = os.path.basename(clean_action_path)
            print(f"\n[plan] ===== {bar_file} =====")
            all_ok, segments, joint_names_12, _bar_id = plan_one_action(
                planner, rcell, clean_action_path, args, groups,
            )
            results.append((bar_file, all_ok))
            if single:
                replayable = (segments, joint_names_12)

        # Batch roll-up (print BEFORE any disconnect so --gui teardown can't
        # invalidate the console stdout handle mid-print).
        if not single:
            n_ok = sum(1 for _, ok in results if ok)
            print(f"\n[summary] planning: {n_ok}/{len(results)} bar(s) fully planned")
            for bar_file, ok in results:
                print(f"  [{'OK  ' if ok else 'FAIL'}] {bar_file}")

        # Replay is single-bar + GUI only.
        if single and replayable is not None:
            segments, joint_names_12 = replayable
            if not segments:
                return 0 if results and results[0][1] else 2
            if args.no_replay or not args.gui:
                print("[replay] skipped (use --gui without --no-replay to enable).")
            elif len(segments) == 1:
                _role, state, path = segments[0]
                replay_with_slider(planner, state, path, joint_names_12)
            else:
                replay_segments(planner, segments, joint_names_12)

        return 0 if results and all(ok for _, ok in results) else 2
    finally:
        try:
            pp.disconnect()
        except Exception as exc:
            print(f"[pb] disconnect raised ({exc}); continuing.")


if __name__ == "__main__":
    _exit_code = main()
    # PyBullet's teardown on Windows (especially the --gui window) invalidates the
    # console stdout handle, so any flush here -- and CPython's own exit-time
    # finalizer flush -- would raise a benign "OSError [WinError 6] The handle is
    # invalid" (shown as 'Exception ignored in: <stdout>') AFTER the run has
    # finished, without affecting the result. Guard the manual flush (the handle
    # may already be dead), then hard-exit with os._exit to skip the finalizer
    # flush entirely. Safe here: main() has returned and PyBullet is disconnected.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.flush()
        except Exception:
            pass
    os._exit(_exit_code if isinstance(_exit_code, int) else 0)
