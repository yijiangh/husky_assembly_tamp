"""Check a planned bar carry on its own, in the real scene, whatever planner made it.

Reads a carry saved by ``plan_process_carry.py`` (``<bar>_J.solved_motion.<tag>.json``,
in the core's solution shape) and tests its path without trusting the planner
that produced it. The scene is rebuilt from the process folder by the shared
core: the start of the carry, the robot at the base pose and the start posture
saved in the file. Then:

1. the bar path is cut into fine steps (2 mm / 0.002 rad by default) and walked
   again by the production repair (``core.repair_joint_path``): calibrated IK at
   every fine bar pose, the 10 degree joint-step gate, and the full production
   collision check in that scene (built bars, joints, support robots);
2. every saved waypoint is collision-checked as it is, and the grasp rigidity
   (how far the bar slips in the right hand) is measured on the saved joints;
3. the lowest point the carried bar, the carried joint halves and the arm
   tools reach is reported (no planner checks the floor; see the report);
4. the length of the path: how far the bar travels and turns, and the joint
   travel (the largest joint move of each step, summed);
5. the holding error (how far the two flanges drift from the rigid relation the
   grasps prescribe) at every saved waypoint, and along a dense joint-space
   interpolation between the waypoints (0.1 degree per joint, the step of
   ``path_validation.py``), which is what the robot shows while its controller
   tracks the path; plus how often the joints reverse direction;
6. the saved waypoints joined by straight joint moves (0.1 degree per joint) and
   every point collision-checked: what a coarse path could hide between waypoints.

Every collision check here is strict: nothing the arms carry or wear may touch a
standing body, also not the receiver the bar is about to be joined to
(``motion_planner/contacts.py``).

Run from the repository root::

    .venv/bin/python -u scripts/prm_check_solved.py <process folder> --bar B21 --tags prm birrt1

``--solved-dir`` names the folder of the saved files (the planner's
``--save-dir``); without it the files are looked for where
``plan_process_carry.py`` puts them by default. ``--holding-only`` skips the
slow parts (repair, collisions, lowest point) and reports only items 4 and 5.
One line of JSON per file is appended to ``--out`` when given.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from json import dumps
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence

# * The package sits one level up from scripts/; the sibling checkouts
# * (compas_fab, the core) are put on the path by the benchmark package.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from husky_assembly_tamp.benchmark.sibling_packages import ensure_sibling_packages  # noqa: E402

ensure_sibling_packages()

import numpy as np  # noqa: E402
import pybullet  # noqa: E402
import pybullet_planning as pp  # noqa: E402
from pybullet_planning import interpolate_poses  # noqa: E402

from husky_assembly_tamp.benchmark.process_export import (  # noqa: E402
    carry_scene,
    default_plans_dir,
    is_process_folder,
    jointing_action,
    read_process,
    require_core,
    short_id,
)
from husky_assembly_tamp.motion_planner.api import (  # noqa: E402
    TOOL_LINK_LEFT,
    TOOL_LINK_RIGHT,
    _arm_joint_names,
    _build_cfab_collision_fn,
    _pp_pose_from_frame,
)
from husky_assembly_tamp.motion_planner.contacts import strict_carry_state  # noqa: E402
from husky_assembly_tamp.motion_planner.dense_check import dense_check  # noqa: E402
from husky_assembly_tamp.motion_planner.holding_prm.holding import (  # noqa: E402
    HoldingModel,
    arm_limits,
    continuous_poses,
    matrix_from_pose,
    invert_transform,
    violation_from_tools,
)
from husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.core import (  # noqa: E402
    DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
    repair_joint_path,
)
from husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.path_validation import (  # noqa: E402
    DEFAULT_DENSE_JOINT_VALIDATION_STEP_RAD,
    refine_joint_path_for_validation,
)

# * Joint moves smaller than these do not count as moves when direction reversals
# * are counted. 0.2 degree is the band the report quotes: it hides the wiggles
# * of a few thousandths of a degree that the IK leaves on a finely cut path and
# * keeps the real back-and-forth of a joint. 0.006 degree is kept for reference.
REVERSAL_DEAD_BAND_RAD = float(np.radians(0.2))
REVERSAL_FINE_BAND_RAD = 1e-4


# -----------------------------------------------------------------------------
# * Small helpers
# -----------------------------------------------------------------------------


def solved_file(solved_dir: str, bar_id: str, tag: str) -> str:
    """The tagged file ``plan_process_carry.py`` wrote for a bar's jointing action, in the given folder."""
    return os.path.join(solved_dir, f"{bar_id}_J.solved_motion.{tag}.json")


def fine_poses(poses: Sequence[Any], position_res: float, rotation_res: float) -> List[Any]:
    """Cut a bar path between its saved poses into fine steps.

    Args:
        poses (Sequence[Any]): pybullet poses ``(point, quaternion)``, start first.
        position_res (float): longest move between fine poses (m).
        rotation_res (float): largest turn between fine poses (rad).

    Returns:
        List[Any]: the fine poses, both ends included.
    """
    out = [poses[0]]
    for a, b in zip(poses[:-1], poses[1:]):
        steps = list(interpolate_poses(a, b, pos_step_size=position_res, ori_step_size=rotation_res))
        out.extend(steps[1:])
    return out


def turn_between(quaternion_a: Sequence[float], quaternion_b: Sequence[float]) -> float:
    """The angle (rad) of the rotation that takes one orientation to the other."""
    dot = abs(float(np.dot(quaternion_a, quaternion_b)))
    return 2.0 * float(np.arccos(min(1.0, dot)))


def moving_bodies(planner, state) -> Dict[str, List[int]]:
    """The simulator bodies that ride on the arms: carried rigid bodies and arm tools.

    Args:
        planner: the planner holding the scene.
        state: the movement's start state.

    Returns:
        Dict[str, List[int]]: name -> simulator body ids.
    """
    client = planner.client
    out: Dict[str, List[int]] = {}
    for name, body in state.rigid_body_states.items():
        if (body.attached_to_link or body.attached_to_tool) and not body.is_hidden:
            out[name] = list(client.rigid_bodies_puids.get(name, []))
    for name, tool in state.tool_states.items():
        if tool.attached_to_group:
            ids = client.tools_puids.get(name)
            out[name] = list(ids) if isinstance(ids, (list, tuple)) else [ids]
    return out


def lowest_point(body_ids: Sequence[int]) -> float:
    """The lowest world height of some simulator bodies, from their bounding boxes (all links)."""
    low = np.inf
    for body in body_ids:
        for link in range(-1, pybullet.getNumJoints(body)):
            low = min(low, pybullet.getAABB(body, link)[0][2])
    return float(low)


# -----------------------------------------------------------------------------
# * The measurements of one path
# -----------------------------------------------------------------------------


def holding_along_path(robot: int, arm_joints: Sequence[int], link_left: int, link_right: int,
                       grasp_left: Any, grasp_right: Any, confs: Sequence[np.ndarray],
                       step_rad: float = DEFAULT_DENSE_JOINT_VALIDATION_STEP_RAD) -> Dict[str, Any]:
    """The holding error at the saved waypoints and between them, and the joints' reversals.

    Between two waypoints the robot controller moves every joint in a straight
    line, so the path is interpolated the same way (``refine_joint_path_for_validation``,
    at most ``step_rad`` per joint per piece) and the holding error measured on every
    piece. The error is the planner's own measure (``violation_from_tools``): the
    larger of the two "where the other flange should be" gaps, as a distance and an
    angle.

    Args:
        robot (int): PyBullet body id of the robot.
        arm_joints (Sequence[int]): the twelve arm joint ids.
        link_left (int): the left tool link.
        link_right (int): the right tool link.
        grasp_left: the left tool flange in the bar frame, as a PyBullet pose.
        grasp_right: the right tool flange in the bar frame, as a PyBullet pose.
        confs (Sequence[np.ndarray]): the saved waypoints.
        step_rad (float): largest joint move per interpolation piece.

    Returns:
        Dict[str, Any]: worst errors at the waypoints and between them (mm, deg),
        where the worst between-waypoint error sits (fraction along the path),
        the number of dense points, and the joint direction reversals.
    """
    grasp = {"left": matrix_from_pose(grasp_left), "right": matrix_from_pose(grasp_right)}
    model = HoldingModel(world_from_base={}, grasp=grasp, grasp_inv={arm: invert_transform(grasp[arm]) for arm in grasp},
                         limits=arm_limits(), home_conf12=np.asarray(confs[0], dtype=float), joint_names_12=[])

    def error_at(conf: np.ndarray):
        pp.set_joint_positions(robot, arm_joints, conf)
        left = pp.tform_from_pose(pp.get_link_pose(robot, link_left))
        right = pp.tform_from_pose(pp.get_link_pose(robot, link_right))
        return violation_from_tools(model, left, right)

    dense, meta = refine_joint_path_for_validation(confs, step_rad, use_angle_normalization=False)
    with pp.WorldSaver():
        at_points = [error_at(conf) for conf in confs]
        between = [error_at(conf) for conf in dense]
    worst_index = int(np.argmax([pos for pos, _rot in between]))

    # * Direction reversals: a joint that was moving one way and then moves the
    # * other way, counted on the saved waypoints, ignoring moves in the dead band.
    def reversals(dead_band_rad: float) -> int:
        count = 0
        for joint in range(len(confs[0])):
            steps = [b[joint] - a[joint] for a, b in zip(confs[:-1], confs[1:])]
            signs = [np.sign(step) for step in steps if abs(step) > dead_band_rad]
            count += sum(1 for a, b in zip(signs[:-1], signs[1:]) if a != b)
        return int(count)

    return {
        "holding_at_waypoints_pos_mm": round(1000.0 * max(pos for pos, _rot in at_points), 4),
        "holding_at_waypoints_rot_deg": round(float(np.degrees(max(rot for _pos, rot in at_points))), 4),
        "holding_between_pos_mm": round(1000.0 * max(pos for pos, _rot in between), 4),
        "holding_between_rot_deg": round(float(np.degrees(max(rot for _pos, rot in between))), 4),
        "holding_between_worst_at": round(float(meta[worst_index]["path_fraction"]), 3),
        "dense_step_deg": round(float(np.degrees(step_rad)), 3),
        "n_dense_points": len(dense),
        "joint_reversals": reversals(REVERSAL_DEAD_BAND_RAD),
        "joint_reversals_dead_band_deg": round(float(np.degrees(REVERSAL_DEAD_BAND_RAD)), 3),
        "joint_reversals_fine": reversals(REVERSAL_FINE_BAND_RAD),
    }


def path_lengths(confs: Sequence[np.ndarray], bar_poses: Sequence[Any]) -> Dict[str, Any]:
    """How long a path is: the bar's travel and turning, the joints' travel and largest step.

    Args:
        confs (Sequence[np.ndarray]): the saved waypoints.
        bar_poses (Sequence[Any]): the bar's pose at each waypoint (PyBullet poses).

    Returns:
        Dict[str, Any]: the lengths, rounded for the record.
    """
    bar_travel = sum(float(np.linalg.norm(np.subtract(b[0], a[0]))) for a, b in zip(bar_poses[:-1], bar_poses[1:]))
    bar_turn = sum(turn_between(a[1], b[1]) for a, b in zip(bar_poses[:-1], bar_poses[1:]))
    joint_travel = sum(float(np.max(np.abs(b - a))) for a, b in zip(confs[:-1], confs[1:]))
    largest_step = max(float(np.max(np.abs(b - a))) for a, b in zip(confs[:-1], confs[1:])) if len(confs) > 1 else 0.0
    return {
        "n_waypoints": len(confs),
        "saved_max_step_deg": round(float(np.degrees(largest_step)), 2),
        "bar_travel_m": round(bar_travel, 3), "bar_turn_deg": round(float(np.degrees(bar_turn)), 1),
        "joint_travel_deg": round(float(np.degrees(joint_travel)), 1),
    }


def check_path(planner, state, confs: Sequence[Sequence[float]], bar_name: str, record: Dict[str, Any],
               joint_names_12: Sequence[str], position_res: float, rotation_res: float,
               holding_only: bool = False, dense_step_deg: float = 0.1) -> Dict[str, Any]:
    """Measure a saved carry path in the scene of its start state: items 1 to 6 of the module docstring.

    Args:
        planner: the planner with the scene loaded.
        state: the carry's start state (the base pose and the attachments).
        confs (Sequence[Sequence[float]]): the saved waypoints, twelve values each.
        bar_name (str): the carried bar's rigid-body name in the cell.
        record (Dict[str, Any]): the record so far (file name, movement); filled in and returned.
        joint_names_12 (Sequence[str]): the twelve arm joints.
        position_res (float): fine step of the repair walk (m).
        rotation_res (float): fine step of the repair walk (rad).
        holding_only (bool): skip the repair, the collision checks and the lowest point.
        dense_step_deg (float): largest joint move between the points of the dense check; 0 skips it.

    Returns:
        Dict[str, Any]: the verdict (``ok``) and what was measured.
    """
    confs = [np.asarray(conf, dtype=float) for conf in confs]
    planner.set_robot_cell_state(state)
    robot = planner.client.robot_puid
    arm_joints = pp.joints_from_names(robot, joint_names_12)
    link_left = pp.link_from_name(robot, TOOL_LINK_LEFT)
    link_right = pp.link_from_name(robot, TOOL_LINK_RIGHT)
    collides = _build_cfab_collision_fn(planner, state, joint_names_12)
    carried = moving_bodies(planner, state)
    base_z = float(state.robot_base_frame.point.z)

    # * The grasps: the left one as authored, the right one as the path's first waypoint has it.
    grasp_left = pp.invert(_pp_pose_from_frame(state.rigid_body_states[bar_name].attachment_frame))
    bar_poses, right_in_bar = [], []
    with pp.WorldSaver():
        for conf in confs:
            pp.set_joint_positions(robot, arm_joints, conf)
            world_from_bar = pp.multiply(pp.get_link_pose(robot, link_left), pp.invert(grasp_left))
            bar_poses.append(world_from_bar)
            right_in_bar.append(pp.multiply(pp.invert(world_from_bar), pp.get_link_pose(robot, link_right)))
    # Keep the quaternion signs continuous, or the fine cut below takes a sign flip for a full turn.
    bar_poses = [(tuple(pose[0]), tuple(pose[1])) for pose in continuous_poses(bar_poses)]
    grasp_right = right_in_bar[0]
    slip_mm = max(1000.0 * float(np.linalg.norm(np.subtract(pose[0], grasp_right[0]))) for pose in right_in_bar)

    # * The cheap measures: path lengths, and the holding error at and between the waypoints.
    record.update(path_lengths(confs, bar_poses))
    record.update(holding_along_path(robot, arm_joints, link_left, link_right, grasp_left, grasp_right, confs))
    record["right_grasp_slip_mm"] = round(slip_mm, 4)
    if holding_only:
        record.update({"ok": True, "holding_only": True})
        return record

    # * 1. The production repair on the finely cut bar path.
    poses = fine_poses(bar_poses, position_res, rotation_res)
    context = {"robot": robot, "arm_joints": arm_joints, "tool_link_left": link_left, "tool_link_right": link_right,
               "grasp_bar_from_left": grasp_left, "grasp_bar_from_right": grasp_right,
               "ik_options": {"artifact_set": "calibrated"}}
    start = perf_counter()
    with pp.WorldSaver():
        _repaired, repair = repair_joint_path(context, poses, confs[0], confs[-1], joint_collision_fn=collides,
                                              joint_continuity_threshold_rad=DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD)
    repair_s = perf_counter() - start

    # * 2. The saved waypoints as they are: collision.
    hits = [index for index, conf in enumerate(confs) if collides(conf)]

    # * 3. The lowest point of what the arms carry, over the saved waypoints.
    # The goal itself may sit low by design (a ground joint lands on the floor), so
    # the lowest point of the last waypoint is kept apart for comparison.
    low, low_name, low_index, low_at_goal = np.inf, "", -1, np.inf
    for index, conf in enumerate(confs):
        waypoint_state = state.copy()
        for name, value in zip(joint_names_12, conf):
            waypoint_state.robot_configuration[name] = float(value)
        planner.set_robot_cell_state(waypoint_state)
        for name, ids in carried.items():
            height = lowest_point(ids)
            if height < low:
                low, low_name, low_index = height, name, index
            if index == len(confs) - 1:
                low_at_goal = min(low_at_goal, height)
    planner.set_robot_cell_state(state)

    # * 6. Between the saved waypoints: straight joint moves, every point collision-checked.
    dense = (dense_check(confs, lambda points: [collides(point) for point in points], float(np.radians(dense_step_deg)))
             if dense_step_deg > 0 else None)
    dense_hit = (dense or {}).get("first_hit")
    planner.set_robot_cell_state(state)

    record.update({
        "ok": bool(repair["success"]) and not hits and (dense is None or dense["ok"]),
        "saved_first_hit": {"waypoint": hits[0], "pairs": collides.explain(confs[hits[0]])} if hits else None,
        "dense_ok": None if dense is None else dense["ok"],
        "dense_points": None if dense is None else dense["n_points"],
        "dense_s": None if dense is None else dense["seconds"],
        "dense_first_hit": None if not dense_hit else {"segment": dense_hit["segment"], "alpha": dense_hit["alpha"],
                                                       "pairs": collides.explain(dense_hit["conf"])},
        "n_fine_poses": len(poses),
        "repair_success": bool(repair["success"]), "repair_reason": repair.get("reason"),
        "repair_direction": repair.get("direction"), "repair_s": round(repair_s, 1),
        "repair_max_step_deg": None if repair.get("max_step_rad") is None else round(float(np.degrees(repair["max_step_rad"])), 2),
        "saved_waypoints_colliding": len(hits),
        "lowest_z_m": round(low, 4), "lowest_z_above_base_frame_m": round(low - base_z, 4),
        "lowest_body": low_name, "lowest_at_waypoint": low_index, "lowest_z_at_goal_m": round(low_at_goal, 4),
    })
    return record


# -----------------------------------------------------------------------------
# * The saved file and its scene
# -----------------------------------------------------------------------------


def load_solved_run(path: str):
    """Read a carry saved by ``plan_process_carry.py``.

    Args:
        path (str): the file (the core's solution shape, one movement).

    Returns:
        Tuple: the movement id, the base pose the plan was made at (or None),
        the start joints (name -> value, or None) and the trajectory (or None).

    Raises:
        ValueError: when the file is not a solution file, or holds not exactly one movement.
    """
    # ? The core reads solutions only from <process>/solutions/; the experiment's
    # ? tagged files live elsewhere, so its private parser is used on the raw JSON.
    from scaffolding_assembly_core.process.read import check_keys
    from scaffolding_assembly_core.process.solutions import _solution

    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    problems: List[str] = []
    check_keys(raw, "solution", os.path.basename(path), problems)
    if problems:
        raise ValueError("\n".join(problems))
    solution = _solution(raw)
    if len(solution.movements) != 1:
        raise ValueError(f"{path}: expected one movement, found {sorted(solution.movements)}")
    (movement_id, result), = solution.movements.items()
    robot = result.trajectory.robot if result.trajectory is not None else ""
    return movement_id, result.bases.get(robot), result.start.get(robot), result.trajectory


def check_file(mirror, process, action, path: str, position_res: float, rotation_res: float,
               holding_only: bool = False, dense_step_deg: float = 0.1) -> Dict[str, Any]:
    """Check one saved carry in the scene the core rebuilds for it.

    Args:
        mirror: the acting robot's ``CompasFabMirror``.
        process: the process.
        action: the bar's jointing action.
        path (str): the saved file.
        position_res (float): fine step of the repair walk (m).
        rotation_res (float): fine step of the repair walk (rad).
        holding_only (bool): skip the slow parts.
        dense_step_deg (float): step of the dense check; 0 skips it.

    Returns:
        Dict[str, Any]: the verdict (``ok``) and what was measured.
    """
    record: Dict[str, Any] = {"file": os.path.basename(path)}
    movement_id, base, start, trajectory = load_solved_run(path)
    record["movement"] = movement_id
    if trajectory is None:
        record.update({"ok": False, "reason": "no saved path"})
        return record
    # The scene of the carry's start, the robot at the saved base and start posture.
    scene, _action, _movement = carry_scene(process, short_id(action.bar), joints=start, base=base)
    mirror.sync(scene)
    left_names, right_names = _arm_joint_names(mirror.cell)
    joint_names_12 = list(left_names) + list(right_names)
    index = {name: position for position, name in enumerate(trajectory.joint_names)}
    confs = [[row[index[name]] for name in joint_names_12] for row in trajectory.positions]
    # ! Strict: the scene's allowances for the pending mate do not hold along a carry.
    state, _removed = strict_carry_state(mirror.state)
    with mirror.lend() as planner:
        return check_path(planner, state, confs, action.bar, record, joint_names_12, position_res,
                          rotation_res, holding_only=holding_only, dense_step_deg=dense_step_deg)


# -----------------------------------------------------------------------------
# * Command line
# -----------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """Check the tagged files of one bar; exit code 0 when all pass."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("problem_dir", help="the process folder (holds process.json and actions/)")
    parser.add_argument("--bar", required=True, help="bar id, e.g. B21")
    parser.add_argument("--tags", nargs="+", required=True, help="save tags of the files to check")
    parser.add_argument("--solved-dir", default="",
                        help="folder holding the saved files (default: where plan_process_carry.py saves by default)")
    parser.add_argument("--position-res", type=float, default=0.002, help="fine step of the repair walk (m)")
    parser.add_argument("--rotation-res", type=float, default=0.002, help="fine step of the repair walk (rad)")
    parser.add_argument("--out", default="", help="append one line of JSON per file to this file")
    parser.add_argument("--holding-only", action="store_true",
                        help="only the path lengths and the holding error at and between waypoints (fast)")
    parser.add_argument("--dense-step-deg", type=float, default=0.1,
                        help="largest joint move between the points of the dense collision check; 0 skips it")
    args = parser.parse_args(argv)

    if not is_process_folder(args.problem_dir):
        print(f"[X] {args.problem_dir} is not a process folder (no process.json).")
        return 1
    try:
        require_core()
    except ImportError as exc:
        print(f"[X] {exc}")
        return 1
    from scaffolding_assembly_core.mirrors.compas_fab import CompasFabMirror

    process = read_process(args.problem_dir)
    action = jointing_action(process, args.bar)
    solved_dir = args.solved_dir or default_plans_dir(args.problem_dir, os.environ.get("HUSKY_BENCH_ROOT"))
    mirror = CompasFabMirror(action.robot, log=print)
    all_ok = True
    try:
        for tag in args.tags:
            path = solved_file(solved_dir, args.bar, tag)
            if not os.path.isfile(path):
                # A missing file is reported, not skipped quietly.
                record = {"file": os.path.basename(path), "ok": False, "reason": "file not found"}
            else:
                record = check_file(mirror, process, action, path, args.position_res, args.rotation_res,
                                    holding_only=args.holding_only, dense_step_deg=args.dense_step_deg)
            record.update({"bar": args.bar, "tag": tag})
            all_ok = all_ok and record["ok"]
            print("[check] " + dumps(record))
            if args.out:
                with open(args.out, "a", encoding="utf-8") as handle:
                    handle.write(dumps(record) + "\n")
    finally:
        mirror.close()
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
