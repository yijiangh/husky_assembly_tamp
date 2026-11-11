"""Round the corners of a saved bar carry in bar pose space, without planning it again.

A carry from ``plan_process_carry.py`` (the roadmap or the BiRRT, then the
shortcut smoother) is a chain of straight pieces in bar pose space. The arms
can only follow the joint corners where two pieces meet by stopping there, so
the trajectory runs stop-and-go. This script rounds each corner with a curve
in pose space (``smooth.round_dual_arm_pose_path``), solves the joints again
along it with the production repair (calibrated IK, the 10 degree joint-step
gate, the full collision check in the carry's scene), and saves the result in
the same solution shape. Start and end joints stay exactly as saved.

Run from the repository root::

    .venv/bin/python -u scripts/round_process_carry.py <process folder> --bar B6 \\
        --solution <process folder>/solutions/B6_J.json --save-dir <dir> --save-tag rounded

``--solution`` takes the process's own ``solutions/<bar>_J.json`` or a tagged
``<bar>_J.solved_motion.<tag>.json``; its carry movement is rounded. The result
goes to ``<save-dir>/<bar>_J.solved_motion.<tag>.json`` (check it with
``prm_check_solved.py``); ``--write-solution`` also writes it as the process's
``solutions/<bar>_J.json``. The scene is rebuilt from the process folder at the
saved base and start posture, as ``prm_check_solved.py`` does.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from husky_assembly_tamp.benchmark.sibling_packages import ensure_sibling_packages  # noqa: E402

ensure_sibling_packages()

import numpy as np  # noqa: E402
import pybullet_planning as pp  # noqa: E402

from husky_assembly_tamp.benchmark.process_export import (  # noqa: E402
    carry_movement,
    is_process_folder,
    jointing_action,
    read_process,
    require_core,
    short_id,
    carry_scene,
)
from husky_assembly_tamp.motion_planner.api import (  # noqa: E402
    TOOL_LINK_LEFT,
    TOOL_LINK_RIGHT,
    _arm_joint_names,
    _build_cfab_collision_fn,
    _pp_pose_from_frame,
)
from husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.core import (  # noqa: E402
    DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
)
from husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.smooth import (  # noqa: E402
    DEFAULT_ROUND_MIN_RADIUS,
    DEFAULT_ROUND_RADIUS,
    DEFAULT_ROUND_TOLERANCE,
    round_dual_arm_pose_path,
)
from husky_assembly_tamp.motion_planner.holding_prm.holding import continuous_poses  # noqa: E402
from husky_assembly_tamp.motion_planner.holding_prm.store import write_json_atomic  # noqa: E402

from plan_process_carry import solution_record, solved_path, write_process_solution  # noqa: E402

#: The resolutions the saved carries are planned at (the hall manual's plan command).
DEFAULT_POSITION_RES = 0.004
DEFAULT_ROTATION_RES = 0.01


def read_carry(path: str, movement_id: str):
    """The saved carry of a solution file: its planner block, base, start joints and trajectory.

    Args:
        path (str): a solution file (one or several movements).
        movement_id (str): the carry movement.

    Returns:
        Tuple: (raw planner settings, base pose or None, start joints or None, trajectory).

    Raises:
        ValueError: when the file is not a solution file or holds no solved path for the movement.
    """
    from scaffolding_assembly_core.process.read import check_keys
    from scaffolding_assembly_core.process.solutions import _solution

    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    problems: List[str] = []
    check_keys(raw, "solution", os.path.basename(path), problems)
    if problems:
        raise ValueError("\n".join(problems))
    result = _solution(raw).movements.get(movement_id)
    if result is None or result.status != "solved" or result.trajectory is None:
        raise ValueError(f"{path} holds no solved path for {movement_id}")
    robot = result.trajectory.robot
    settings = dict((raw.get("planner") or {}).get("settings") or {})
    return settings, result.bases.get(robot), result.start.get(robot), result.trajectory


def round_carry(args) -> int:
    """Round the saved carry and save it; 0 when rounded, 2 when it stayed as it was."""
    from scaffolding_assembly_core.mirrors.compas_fab import CompasFabMirror

    process = read_process(args.process_dir)
    action = jointing_action(process, args.bar)
    movement = carry_movement(action)
    settings, base, start, trajectory = read_carry(args.solution, movement.id)
    print(f"[load] {movement.id}: {len(trajectory.positions)} waypoints <- {args.solution}")

    scene, _action, _movement = carry_scene(process, short_id(action.bar), joints=start, base=base)
    mirror = CompasFabMirror(action.robot, log=print)
    try:
        mirror.sync(scene)
        left_names, right_names = _arm_joint_names(mirror.cell)
        names = list(left_names) + list(right_names)
        column = {name: index for index, name in enumerate(trajectory.joint_names)}
        confs = [np.array([row[column[name]] for name in names], dtype=float) for row in trajectory.positions]
        with mirror.lend() as planner:
            state = mirror.state
            planner.set_robot_cell_state(state)
            robot = planner.client.robot_puid
            arm_joints = pp.joints_from_names(robot, names)
            link_left = pp.link_from_name(robot, TOOL_LINK_LEFT)
            link_right = pp.link_from_name(robot, TOOL_LINK_RIGHT)
            collides = _build_cfab_collision_fn(planner, state, names)

            # * The bar's poses along the saved joints, and the grasps: the left as authored, the right as the
            # * first waypoint holds it (as prm_check_solved.py measures them).
            grasp_left = pp.invert(_pp_pose_from_frame(state.rigid_body_states[action.bar].attachment_frame))
            bar_poses = []
            with pp.WorldSaver():
                for conf in confs:
                    pp.set_joint_positions(robot, arm_joints, conf)
                    bar_poses.append(pp.multiply(pp.get_link_pose(robot, link_left), pp.invert(grasp_left)))
                pp.set_joint_positions(robot, arm_joints, confs[0])
                grasp_right = pp.multiply(pp.invert(bar_poses[0]), pp.get_link_pose(robot, link_right))
            bar_poses = [(tuple(p), tuple(q)) for p, q in continuous_poses(bar_poses)]
            context = {"robot": robot, "arm_joints": arm_joints, "tool_link_left": link_left,
                       "tool_link_right": link_right, "grasp_bar_from_left": grasp_left,
                       "grasp_bar_from_right": grasp_right, "ik_options": None}

            profile: Dict[str, Any] = {}
            started = time.time()
            with pp.WorldSaver():
                poses, rounded = round_dual_arm_pose_path(
                    bar_poses, confs, scene=context, joint_collision_fn=collides,
                    position_res=args.position_res, rotation_res=args.rotation_res, radius=args.radius,
                    tolerance=args.tolerance, min_radius=args.min_radius,
                    joint_continuity_threshold_rad=DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD, profile_out=profile)
            print(f"[round] {profile['n_corners']} corner(s) at {1000.0 * profile['tolerance']:g} mm tolerance, "
                  f"{profile['rounds']} re-solve(s) in {time.time() - started:.1f} s; radii (mm): "
                  + (", ".join(f"{1000.0 * r:.0f}" for r in profile.get("radii", [])) or "none"))
            for failure in profile.get("failures", []):
                print(f"  {1000.0 * failure['tolerance']:g} mm, shrunk corner(s) {failure['shrunk']}: "
                      f"{failure['forward'] or failure['reason']}")
            if not profile["rounded"]:
                print(f"[round] left as it was: {profile['reason']}")
                return 2
            print(f"[round] {len(poses)} waypoints (was {len(confs)})")

        tag_settings = dict(settings)
        tag_settings["round"] = {"radius": args.radius, "tolerance": args.tolerance, "min_radius": args.min_radius,
                                 "position_res": args.position_res, "rotation_res": args.rotation_res,
                                 "radii": [round(r, 4) for r in profile["radii"]], "source": os.path.basename(
                                     args.solution)}
        info = {"path_poses": [[list(p), list(q)] for p, q in poses]}
        record = solution_record(process, action, movement, action.robot, names, rounded, info,
                                 base if base is not None else scene.robots[action.robot].base, tag_settings)
        os.makedirs(args.save_dir, exist_ok=True)
        out = solved_path(args.save_dir, action.id, args.save_tag)
        write_json_atomic(out, record)
        print(f"[save] rounded carry -> {out}")
        if args.write_solution:
            print(f"[save] process solution -> {write_process_solution(process, action, record)}")
    finally:
        mirror.close()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Parse the command line and round the carry; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("process_dir", help="the process folder (holds process.json and actions/)")
    parser.add_argument("--bar", required=True, help="bar id, e.g. B6")
    parser.add_argument("--solution", required=True,
                        help="the saved carry: <process>/solutions/<bar>_J.json or a tagged solved_motion file")
    parser.add_argument("--radius", type=float, default=DEFAULT_ROUND_RADIUS,
                        help="corner radius tried first (m, largest bar feature-point move from the corner)")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_ROUND_TOLERANCE,
                        help="samples within this of a straight piece belong to it (m)")
    parser.add_argument("--min-radius", type=float, default=DEFAULT_ROUND_MIN_RADIUS,
                        help="a corner whose radius shrinks below this stays sharp (m)")
    parser.add_argument("--position-res", type=float, default=DEFAULT_POSITION_RES,
                        help="largest bar move between waypoints of the result (m)")
    parser.add_argument("--rotation-res", type=float, default=DEFAULT_ROTATION_RES,
                        help="largest bar turn between waypoints of the result (rad)")
    parser.add_argument("--save-dir", required=True, help="where the rounded carry goes")
    parser.add_argument("--save-tag", default="rounded", help="tag in the file name")
    parser.add_argument("--write-solution", action="store_true",
                        help="also write <process>/solutions/<bar>_J.json")
    args = parser.parse_args(argv)
    if not is_process_folder(args.process_dir):
        print(f"[X] {args.process_dir} is not a process folder (no process.json).")
        return 1
    try:
        require_core()
    except ImportError as exc:
        print(f"[X] {exc}")
        return 1
    return round_carry(args)


if __name__ == "__main__":
    sys.exit(main())
