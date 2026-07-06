"""Plot + PyBullet-toggle the authored per-movement arm configs from a BarAction.

The Rhino export already carries keyframe configs (each movement's
start_state.robot_configuration, and sometimes target_configuration). Since M3's
start == M2's end, the authored M2 start->end pair proves M2 is reachable. This
plots those authored 12-vec configs and, with --gui, opens a PyBullet slider so
you can scrub through them (M1.start -> M2.start -> M3.start -> M4.target), each
rendered with that movement's own attachments (bar held in M1/M2, etc.).

Run:
    C:/Users/yijiangh/.rhinocode/py39-rh8/python.exe tests/_probe_clean_confs.py --gui
    ... --load solved      # view the half-solved file's configs instead
"""
import argparse
import os
import re
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--load", choices=("clean", "solved"), default="clean",
                        help="Which BarAction to read the authored configs from.")
    parser.add_argument("--gui", action="store_true",
                        help="Open a PyBullet slider to scrub through the configs.")
    parser.add_argument("--bar-action", default=H.DEFAULT_BAR_ACTION)
    parser.add_argument("--problem", default=H.DEFAULT_PROBLEM)
    args = parser.parse_args()

    data_root = os.path.abspath(H.DEFAULT_DATA_ROOT)
    problem_dir = os.path.join(data_root, args.problem)
    cell_path = os.path.join(problem_dir, "RobotCell.json")
    clean_path = os.path.join(problem_dir, "BarActions", args.bar_action)
    action_path = H.solved_action_path(clean_path) if args.load == "solved" else clean_path
    if not os.path.isfile(action_path):
        print(f"[X] missing action file ({args.load}): {action_path}")
        return 1

    print(f"[load] cell   <- {cell_path}")
    print(f"[load] action <- {action_path}  ({args.load})")
    rcell = json_load(cell_path)
    action = json_load(action_path)

    left_names = [n for n in rcell.get_configurable_joint_names(H.LEFT_GROUP)
                  if any(n.endswith(s) for s in H._ARM_SUFFIXES)]
    right_names = [n for n in rcell.get_configurable_joint_names(H.RIGHT_GROUP)
                   if any(n.endswith(s) for s in H._ARM_SUFFIXES)]
    joint_names_12 = left_names + right_names

    # Collect authored configs in planning order. Keep the owning movement so
    # the GUI slider can render each with its own attachments.
    keyframes = []  # (label, vec12, mv)
    for role in ("M1", "M2", "M3", "M4"):
        mv = H.select_movement(action, role)
        if mv is None:
            print(f"  {role}: <missing>")
            continue
        s = to_vec12(getattr(getattr(mv, "start_state", None), "robot_configuration", None), joint_names_12)
        t = to_vec12(getattr(mv, "target_configuration", None), joint_names_12)
        print(f"  {role}: start_conf={'yes' if s is not None else 'no'}  target_conf={'yes' if t is not None else 'no'}")
        if s is not None:
            keyframes.append((f"{role}.start", s, mv))
        if t is not None:
            keyframes.append((f"{role}.target", t, mv))

    if not keyframes:
        print("[X] no authored configs found.")
        return 2

    def _short(name):
        base = re.sub(r"_joint$", "", str(name))
        return "_".join(base.split("_")[-2:])

    # Boot the planner (DIRECT unless --gui) to collision-check each authored
    # keyframe, then plot, then optionally open the slider.
    _client, planner = H.start_planner(rcell, use_gui=args.gui)
    try:
        base_state = next(mv.start_state for _, _, mv in keyframes if mv.start_state is not None)
        states = [
            (label, make_display_state(mv, rcell, joint_names_12, vec, base_state), vec)
            for label, vec, mv in keyframes
        ]

        # Collision check each authored keyframe. Each display state carries that
        # movement's attachments + touch_bodies ACM, so this reflects the exported
        # allowed-contact policy. (Arm<->arm self-collision is CC1 / SRDF.)
        print("\n[collision] per authored keyframe (full report):")
        collision_ok = {}
        for label, state, _vec in states:
            collision_ok[label] = H.check_collision(planner, state, label=label)

        # Plot the authored configs.
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
        ax.set_title(f"Authored arm configs across movements ({args.load} {args.bar_action})")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        fig.tight_layout()
        out = os.path.join(_TESTS, "clean_authored_confs.png")
        fig.savefig(out, dpi=150)
        print(f"\n[probe] figure saved -> {out}")
        plt.close(fig)

        # Combined summary: collision status + max per-joint delta from the
        # previous authored keyframe (how big each authored move is).
        print("\n=================== AUTHORED CONF SUMMARY ===================")
        prev = None
        for label, _state, vec in states:
            delta = "" if prev is None else f"   max|d| from prev = {np.abs(vec - prev).max():.3f} rad"
            mark = "collision-free" if collision_ok.get(label) else "COLLIDING"
            print(f"  {label:12s} : {mark}{delta}")
            prev = vec
        print("============================================================")

        if not args.gui:
            print("[probe] add --gui to scrub the configs in PyBullet.")
            return 0

        # PyBullet slider: one segment per keyframe (single-waypoint path), so the
        # slider steps between authored configs. Reuses the headless replay_segments.
        segments = [(label, state, [vec]) for label, state, vec in states]
        print("[probe] slider steps: " + " -> ".join(f"{i}:{lbl}" for i, (lbl, _, _) in enumerate(states)))
        print("[probe] Ctrl-C in console to exit.")
        H.replay_segments(planner, segments, joint_names_12)
    finally:
        try:
            pp.disconnect()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
