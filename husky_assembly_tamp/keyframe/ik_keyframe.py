"""Shared IK-keyframe solve logic (Rhino-free).

This is the one place that implements "load a movement's ``start_state``, solve
IK for its ``target_ee_frames``", chained across M1 -> M2 -> M3. Both the Rhino
front-end (``rs_ik_keyframe``) and the headless keyframe solver
(``scripts/headless_bar_action_planner.py --solve-keyframes``) call
``solve_keyframe_chain`` so the solve is defined exactly once.

The movements come from ``core.bar_action.build_assembly_movements`` (Rhino) or
from a loaded ``BarAssemblyAction`` JSON (headless). Either way each movement
already carries the per-keyframe collision context (attachments + allowed-touch
policy) on its ``start_state`` and the EE goal on its ``target_ee_frames`` -- the
only thing IK supplies is ``robot_configuration``.
"""

from __future__ import annotations

import numpy as np

from husky_assembly_tamp.keyframe import dual_arm_ik


def frame_to_mm4(frame) -> np.ndarray:
    """Convert a compas ``Frame`` (meters) to a 4x4 numpy transform in mm.

    The inverse of :func:`mm4_to_frame`. ``solve_dual_arm_ik`` wants its tool0
    targets in mm (project convention), but ``Movement.target_ee_frames`` stores
    compas ``Frame``s in meters, so every target is run through here.

    Args:
        frame (compas.geometry.Frame): pose in meters.

    Returns:
        np.ndarray: 4x4 homogeneous transform with mm translation; columns 0-2
        are the frame's x/y/z axes.
    """
    matrix = np.eye(4, dtype=float)
    matrix[:3, 0] = np.asarray(frame.xaxis, dtype=float)
    matrix[:3, 1] = np.asarray(frame.yaxis, dtype=float)
    matrix[:3, 2] = np.asarray(frame.zaxis, dtype=float)
    matrix[:3, 3] = np.asarray(frame.point, dtype=float) * 1000.0  # m -> mm
    return matrix


def mm4_to_frame(matrix_mm):
    """Convert a 4x4 matrix with mm translation to a compas ``Frame`` (meters).

    The inverse of :func:`frame_to_mm4`. Used e.g. to stamp a solved base frame
    (mm, the project convention) back onto a movement's ``start_state`` (compas,
    meters).

    Args:
        matrix_mm: 4x4 array-like with mm translation.

    Returns:
        compas.geometry.Frame: the same pose in meters.
    """
    # Imported here so importing this module stays free of the compas stack.
    from compas.geometry import Frame

    m = np.asarray(matrix_mm, dtype=float)
    origin = m[:3, 3] / 1000.0
    return Frame(
        list(map(float, origin)),
        list(map(float, m[:3, 0])),
        list(map(float, m[:3, 1])),
    )


def solve_keyframe_chain(
    planner,
    ordered_movements,
    base_frame_mm,
    *,
    check_collision: bool = True,
    verbose_pairs: bool = False,
    after_solve=None,
    groups=None,
    home_conf_12=None,
):
    """Solve IK for a sequence of movements, chaining configs forward.

    For each ``(role, movement)`` in order:

    1. Copy the movement's ``start_state`` (which carries the per-keyframe
       attachments + allowed-touch policy).
    2. Seed its ``robot_configuration`` with the *previous* movement's solved
       config -- this realizes the chain (M2 starts where M1 ended, M3 where M2
       ended). The first movement (M1) has no warm-start -- its start config is
       ``None`` (planner-filled) -- so ``solve_dual_arm_ik`` falls back to a
       default seed and finds the approach basin via cold random restarts.
    3. Solve dual-arm IK for the movement's ``target_ee_frames`` at
       ``base_frame_mm``.

    All movements share ``base_frame_mm`` (one robot base for the whole bar);
    ``solve_dual_arm_ik`` overrides each state's stored base frame with it.

    Args:
        planner (PyBulletPlanner): active planner with the dual-arm cell loaded.
        ordered_movements (list): ``[(role, movement), ...]`` in solve order,
            e.g. ``[("M1", m1), ("M2", m2), ("M3", m3)]``. Each ``movement`` needs
            a ``start_state`` and a ``target_ee_frames`` dict with ``"left"`` /
            ``"right"`` compas ``Frame``s.
        base_frame_mm (np.ndarray): 4x4 mm robot base frame shared by all solves.
        check_collision (bool): pass-through to ``solve_dual_arm_ik``.
        verbose_pairs (bool): pass-through; prints the collision-pair summary.
        after_solve (callable | None): optional debug hook called as
            ``after_solve(role, solved_state)`` right after each movement solves
            (before the next one starts). Used by the headless planner to show the
            just-solved keyframe in the GUI, print its joint-vs-limit plot, and
            pause. It runs between M1 and M2, so it is the place to inspect *why*
            a later movement fails from an earlier one's pose.
        groups (tuple[str, str] | None): the ``(left_group, right_group)``
            planning-group names; ``None`` -> derived from the cell semantics.
        home_conf_12 (Sequence[float] | None): the 12 home joint values (left 6
            then right 6), needed by the ssik backend on the COLD first solve.
            Rhino passes its config constant; the headless planner passes M4's
            ``target_configuration`` from the BarAction JSON.

    Returns:
        dict | None: ``{role: solved_state}`` once every movement solves, or
        ``None`` at the first failure (so a base-sampling caller can try another
        base).
    """
    results = {}
    prev_config = None
    for role, movement in ordered_movements:
        state = movement.start_state.copy()
        # Chain: each movement begins at the previous movement's solved config.
        # That warm-start is a good seed, so those solves descend deterministically
        # (max_restart_iter=1). The FIRST movement has no warm-start, so it uses the
        # cold default (random restarts) to find the collision-free basin.
        if prev_config is not None:
            state.robot_configuration = prev_config.copy()
            max_restart_iter = 1
        else:
            max_restart_iter = None  # -> config.IK_MAX_RESTART_ITER (cold restarts)

        targets = movement.target_ee_frames or {}
        left_frame = targets.get("left")
        right_frame = targets.get("right")
        if left_frame is None or right_frame is None:
            print(
                f"keyframe.ik_keyframe.solve_keyframe_chain: {role} has no "
                "left/right target_ee_frames; aborting chain."
            )
            return None

        print(f"keyframe.ik_keyframe.solve_keyframe_chain: solving {role} ...")
        solved = dual_arm_ik.solve_dual_arm_ik(
            planner,
            state,
            base_frame_mm,
            frame_to_mm4(left_frame),
            frame_to_mm4(right_frame),
            check_collision=check_collision,
            max_restart_iter=max_restart_iter,
            verbose_pairs=verbose_pairs,
            groups=groups,
            home_conf_12=home_conf_12,
        )
        if solved is None:
            print(f"keyframe.ik_keyframe.solve_keyframe_chain: {role} IK failed.")
            return None

        results[role] = solved
        prev_config = solved.robot_configuration
        if after_solve is not None:
            after_solve(role, solved)
    return results


def find_first_unsolvable_movement(
    planner,
    ordered_movements,
    base_frame_mm,
    *,
    check_collision: bool = True,
    groups=None,
    home_conf_12=None,
):
    """Walk the keyframe chain and return the first movement that fails to solve.

    Mirrors :func:`solve_keyframe_chain`'s warm-chaining (each movement seeded from
    the previous movement's solved config) but, instead of returning the whole
    solution, stops at the FIRST movement whose IK yields no solution and returns
    it. A caller can then enumerate that movement's candidate IK pairs to diagnose
    why the chain failed at this base.

    Args:
        planner: the dual-arm planner.
        ordered_movements (list): ``[(role, movement), ...]`` in solve order, e.g.
            ``[("M1", m1), ("M2", m2), ("M3", m3)]``.
        base_frame_mm (np.ndarray): 4x4 mm robot base frame shared by all solves.
        check_collision (bool): pass-through to ``dual_arm_ik.solve_dual_arm_ik``.
        groups (tuple[str, str] | None): pass-through planning-group names.
        home_conf_12 (Sequence[float] | None): pass-through home configuration
            (needed by the ssik backend on the cold first solve).

    Returns:
        tuple | None: ``(role, movement)`` of the first movement that fails to
        solve at ``base_frame_mm``, or ``None`` if every movement solves (nothing
        to diagnose).
    """
    prev_config = None
    for role, movement in ordered_movements:
        state = movement.start_state.copy()
        # Same warm/cold signal solve_keyframe_chain uses (1 == warm-started).
        if prev_config is not None:
            state.robot_configuration = prev_config.copy()
            max_restart_iter = 1
        else:
            max_restart_iter = None

        targets = movement.target_ee_frames or {}
        left_frame = targets.get("left")
        right_frame = targets.get("right")
        if left_frame is None or right_frame is None:
            # No EE goal -> this movement is where the chain can't even be posed.
            return role, movement

        solved = dual_arm_ik.solve_dual_arm_ik(
            planner,
            state,
            base_frame_mm,
            frame_to_mm4(left_frame),
            frame_to_mm4(right_frame),
            check_collision=check_collision,
            max_restart_iter=max_restart_iter,
            groups=groups,
            home_conf_12=home_conf_12,
        )
        if solved is None:
            return role, movement
        prev_config = solved.robot_configuration
    return None
