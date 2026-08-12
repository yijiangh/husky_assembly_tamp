"""Replay + check a bar action plan (IK keyframes OR planned motion) from JSON.

``--load`` selects which file and which view:

- ``clean`` (default) / ``solved_keyframe``: plot + scrub the discrete KEYFRAME
  configs (M1.start -> M2.start -> M3.start -> M4.target), each rendered with that
  movement's own attachments (bar held in M1/M2, released in M3/M4). Since M3.start
  == M2.end, the authored M2 start->end pair proves M2 is reachable.
- ``solved_motion``: plot + scrub the full PLANNED TRAJECTORIES of every movement
  (M1..M4 waypoints), and collision-check every waypoint, so you can verify the
  planned motion end to end.

Run:
    C:/Users/yijiangh/.rhinocode/py39-rh8/python.exe \
        external/husky_assembly_tamp/scripts/replay_bar_action_plan.py --gui
        [--load clean|solved_keyframe|solved_motion]
        [--bar-action B6.json] [--problem <name>] [--data-root <folder>]

The data root resolves like the planner's: ``--data-root`` if given, else the
``HUSKY_ASSEMBLY_DATA_ROOT`` environment variable, else a clear error.
"""
import argparse
import os
import re
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

from _rhino_env_bootstrap import bootstrap_rhino_site_envs

bootstrap_rhino_site_envs(verbose=False)

import numpy as np
import matplotlib.pyplot as plt
import pybullet_planning as pp
from compas.data import json_load

import headless_bar_action_planner as H


def to_vec12(c, joint_names_12):
    """Best-effort convert a Configuration / list into a 12-vec, or None."""
    if c is None:
        return None
    if hasattr(c, "joint_values") or hasattr(c, "keys"):
        try:
            return H.vec12_from_conf(c, joint_names_12)
        except Exception:
            return None
    arr = np.asarray(c, dtype=float).ravel()
    return arr if arr.shape[0] == len(joint_names_12) else None


def make_display_state(mv, rcell, joint_names_12, vec12, base_state):
    """A RobotCellState showing ``vec12``, using ``mv``'s start_state as template.

    Keeping the movement's own start_state preserves that phase's attachments /
    allowed-contact set, so the held bar renders in the phases that carry it.
    ``base_state`` is a known-good fallback template when ``mv`` has none.
    """
    template = getattr(mv, "start_state", None) or base_state
    state = template.copy()
    if state.robot_configuration is None:
        state.robot_configuration = rcell.zero_full_configuration()
    for name, val in zip(joint_names_12, vec12):
        state.robot_configuration[name] = float(val)
    return state


def _state_with_conf(template_state, rcell, joint_names_12, wp):
    """A copy of ``template_state`` with the 12-vec ``wp`` written onto its config."""
    state = template_state.copy()
    if state.robot_configuration is None:
        state.robot_configuration = rcell.zero_full_configuration()
    for name, val in zip(joint_names_12, wp):
        state.robot_configuration[name] = float(val)
    return state


def _short(name):
    base = re.sub(r"_joint$", "", str(name))
    return "_".join(base.split("_")[-2:])


def _quiet_collision_free(planner, state) -> bool:
    """True if ``state`` is collision-free (quiet -- no per-pair spew)."""
    try:
        planner.check_collision(state, options={"full_report": False, "verbose": False})
        return True
    except Exception:
        return False


def _joint_names_12(rcell):
    """The 12 arm-joint names, with the groups derived from the cell semantics."""
    return H.arm_joint_names_12(rcell, H.resolve_arm_groups(rcell))


# ---------------------------------------------------------------------------
# Keyframe view (clean / solved_keyframe)
# ---------------------------------------------------------------------------


def _run_keyframes(planner, rcell, action, joint_names_12, args) -> int:
    """Plot + (with --gui) scrub the discrete authored keyframe configs."""
    keyframes = []  # (label, vec12, mv)
    for role in ("M1", "M2", "M3", "M4"):
        mv = H.select_movement(action, role)
        if mv is None:
            print(f"  {role}: <missing>")
            continue
        s = to_vec12(getattr(getattr(mv, "start_state", None), "robot_configuration", None), joint_names_12)
        t = to_vec12(getattr(mv, "target_configuration", None), joint_names_12)
        print(f"  {role}: start_conf={'yes' if s is not None else 'no'}  "
              f"target_conf={'yes' if t is not None else 'no'}")
        if s is not None:
            keyframes.append((f"{role}.start", s, mv))
        if t is not None:
            keyframes.append((f"{role}.target", t, mv))

    if not keyframes:
        print("[X] no authored configs found.")
        return 2

    base_state = next(mv.start_state for _, _, mv in keyframes if mv.start_state is not None)
    states = [
        (label, make_display_state(mv, rcell, joint_names_12, vec, base_state), vec)
        for label, vec, mv in keyframes
    ]

    print("\n[collision] per authored keyframe (full report):")
    collision_ok = {}
    for label, state, _vec in states:
        collision_ok[label] = H.check_collision(planner, state, label=label)

    # Plot the authored configs (one line per keyframe, x = joint index).
    n = len(joint_names_12)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(14, 6.5))
    cmap = plt.get_cmap("tab10")
    for i, (label, vec, _) in enumerate(keyframes):
        ax.plot(x, vec, marker="o", lw=1.6, ms=5, color=cmap(i % 10), label=label)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.axvline(5.5, color="gray", ls="--", lw=1)
    ymax = max(v.max() for _, v, _ in keyframes)
    ax.text(2.5, ymax, "LEFT arm", ha="center", fontsize=10, color="#555")
    ax.text(8.5, ymax, "RIGHT arm", ha="center", fontsize=10, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels([_short(j) for j in joint_names_12], rotation=90, fontsize=8)
    ax.set_ylabel("joint value (rad)")
    ax.set_title(f"Authored keyframe configs ({args.load} {args.bar_action})")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    # 'clean' load shows the Rhino-authored configs; 'solved_keyframe' shows the
    # re-solved keyframes -- name the file after which one so they don't collide.
    suffix = "clean_authored_confs" if args.load == "clean" else "keyframe_confs"
    out = os.path.join(args.plot_dir, f"{args.bar_name}_{suffix}.png")
    fig.savefig(out, dpi=150)
    print(f"\n[replay] keyframe figure saved -> {out}")
    plt.close(fig)

    print("\n=================== AUTHORED CONF SUMMARY ===================")
    prev = None
    for label, _state, vec in states:
        delta = "" if prev is None else f"   max|d| from prev = {np.abs(vec - prev).max():.3f} rad"
        mark = "collision-free" if collision_ok.get(label) else "COLLIDING"
        print(f"  {label:12s} : {mark}{delta}")
        prev = vec
    print("============================================================")

    if not args.gui:
        print("[replay] add --gui to scrub the configs in PyBullet.")
        return 0

    # One segment per keyframe (single-waypoint path), so the slider steps between
    # authored configs. Reuses the headless replay_segments.
    segments = [(label, state, [vec]) for label, state, vec in states]
    print("[replay] slider steps: " + " -> ".join(f"{i}:{lbl}" for i, (lbl, _, _) in enumerate(states)))
    print("[replay] Ctrl-C in console to exit.")
    H.replay_segments(planner, segments, joint_names_12)
    return 0


# ---------------------------------------------------------------------------
# Motion view (solved_motion)
# ---------------------------------------------------------------------------


def _run_motion(planner, rcell, action, joint_names_12, args) -> int:
    """Plot + collision-check + (with --gui) scrub each movement's planned trajectory."""
    # Collect one segment per movement that carries a trajectory.
    segments = []  # (role, start_state, path)
    for role in ("M1", "M2", "M3", "M4"):
        mv = H.select_movement(action, role)
        traj = getattr(mv, "trajectory", None) if mv is not None else None
        # Sidecars store a JointTrajectory; older ones store bare 12-vec lists.
        # path_from_trajectory takes either and hands back plain waypoints.
        path = H.path_from_trajectory(traj, joint_names_12) if traj is not None else None
        print(f"  {role}: {len(path) if path else 0} trajectory waypoint(s)")
        if path and mv.start_state is not None:
            segments.append((role, mv.start_state, path))

    if not segments:
        print("[!] no planned trajectories in this file; falling back to keyframe view.\n")
        return _run_keyframes(planner, rcell, action, joint_names_12, args)

    total = sum(len(p) for _, _, p in segments)
    print(f"\n[motion] {len(segments)} movement(s), {total} total waypoint(s).")

    # Collision-check every waypoint (quiet); report per-movement.
    print("\n[collision] per-waypoint check (quiet):")
    for role, start_state, path in segments:
        bad = [
            i for i, wp in enumerate(path)
            if not _quiet_collision_free(
                planner, _state_with_conf(start_state, rcell, joint_names_12, wp)
            )
        ]
        if bad:
            print(f"  {role}: {len(bad)}/{len(path)} COLLIDING waypoint(s) at {bad}")
        else:
            print(f"  {role}: {len(path)}/{len(path)} collision-free")

    # Plot the trajectories: joint value vs global waypoint index, movement bounds.
    all_wp = np.array([wp for _, _, path in segments for wp in path], dtype=float)
    fig, ax = plt.subplots(figsize=(14, 6.5))
    cmap = plt.get_cmap("tab20")
    xg = np.arange(all_wp.shape[0])
    for j in range(all_wp.shape[1]):
        ax.plot(xg, all_wp[:, j], lw=1.2, color=cmap(j % 20),
                label=f"[{'L' if j < 6 else 'R'}] {_short(joint_names_12[j])}")
    boundary = 0
    ytop = ax.get_ylim()[1]
    for role, _s, path in segments:
        ax.axvline(boundary, color="gray", ls="--", lw=1)
        ax.text(boundary + 0.2, ytop, role, va="top", fontsize=9, color="#333")
        boundary += len(path)
    ax.set_xlabel("waypoint index (concatenated M1..M4)")
    ax.set_ylabel("joint value (rad)")
    ax.set_title(f"Planned trajectory ({args.load} {args.bar_action})")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    out = os.path.join(args.plot_dir, f"{args.bar_name}_motion_confs.png")
    fig.savefig(out, dpi=150)
    print(f"\n[replay] motion figure saved -> {out}")
    plt.close(fig)

    if not args.gui:
        print("[replay] add --gui to scrub the trajectory in PyBullet.")
        return 0

    print("[replay] slider walks all waypoints M1..M4; Ctrl-C in console to exit.")
    H.replay_segments(planner, segments, joint_names_12)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--load", choices=("clean", "solved_keyframe", "solved_motion"), default="clean",
        help="Which BarAction file + view: 'clean' (Rhino export, keyframes), "
             "'solved_keyframe' (keyframe-solve sidecar, keyframes), or "
             "'solved_motion' (motion sidecar, planned trajectories).",
    )
    parser.add_argument("--gui", action="store_true",
                        help="Open a PyBullet slider to scrub through the plan.")
    parser.add_argument("--bar-action", default=H.DEFAULT_BAR_ACTION)
    parser.add_argument("--problem", default=H.DEFAULT_PROBLEM)
    parser.add_argument(
        "--data-root", default=None,
        help="Parent folder containing per-problem subfolders. Falls back to the "
             "HUSKY_ASSEMBLY_DATA_ROOT environment variable; errors if neither "
             "is given.",
    )
    args = parser.parse_args()

    data_root = H.resolve_data_root(args.data_root)
    problem_dir = os.path.join(data_root, args.problem)
    cell_path = os.path.join(problem_dir, "RobotCell.json")
    clean_path = os.path.join(problem_dir, "BarActions", args.bar_action)
    if args.load == "clean":
        action_path = clean_path
    else:
        kind = "motion" if args.load == "solved_motion" else "keyframe"
        action_path = H.solved_action_path(clean_path, kind)
    if not os.path.isfile(action_path):
        print(f"[X] missing action file ({args.load}): {action_path}")
        return 1

    # Where the plots go: right next to the BarAction JSON we are reading (the
    # problem's BarActions folder), prefixed with the bar action name so plots
    # for different bars don't overwrite each other. e.g. reading B6.json writes
    # B6_keyframe_confs.png / B6_motion_confs.png / B6_clean_authored_confs.png.
    args.plot_dir = os.path.dirname(action_path)
    args.bar_name = os.path.splitext(os.path.basename(args.bar_action))[0]

    print(f"[load] cell   <- {cell_path}")
    print(f"[load] action <- {action_path}  ({args.load})")
    rcell = json_load(cell_path)
    action = json_load(action_path)
    joint_names_12 = _joint_names_12(rcell)

    _client, planner = H.start_planner(rcell, use_gui=args.gui)
    try:
        if args.load == "solved_motion":
            return _run_motion(planner, rcell, action, joint_names_12, args)
        return _run_keyframes(planner, rcell, action, joint_names_12, args)
    finally:
        try:
            pp.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
