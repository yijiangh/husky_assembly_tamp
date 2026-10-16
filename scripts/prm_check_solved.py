"""Check a planned bar carry on its own, in the real scene, whatever planner made it.

Reads the bar-carry movement of a solved-motion file written by
``headless_bar_action_planner.py`` (``<bar>__J.solved_motion.<tag>.json``) and
tests its saved path without trusting the planner that produced it:

1. the bar path is cut into fine steps (2 mm / 0.002 rad by default) and walked
   again by the production repair (``core.repair_joint_path``): calibrated IK at
   every fine bar pose, the 10 degree joint-step gate, and the full production
   collision check in the scene of the movement's own start state (built bars,
   joints, support robots; the base pose saved in the file);
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
   tracks the path; plus how often the joints reverse direction.

Run from the repository root::

    .venv/Scripts/python.exe -u scripts/prm_check_solved.py <problem folder> --bar B4 --tags prm birrt

``--solved-dir`` names the folder of the solved files when they were saved
elsewhere (the planner's ``--save-dir``). ``--holding-only`` skips the slow
parts (repair, collisions, lowest point) and reports only items 4 and 5.
One line of JSON per file is appended to ``--out`` when given.
"""

from __future__ import annotations

import argparse
import os
import sys
from json import dumps
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence

# ! The planner script next to this file sets up the import paths (sibling
# ! checkouts, Rhino site folders) when it is imported, so it comes first.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from headless_bar_action_planner import (  # noqa: E402
    arm_joint_names_12,
    find_carry_movement,
    path_from_trajectory,
    start_planner,
)

import numpy as np  # noqa: E402
import pybullet  # noqa: E402
import pybullet_planning as pp  # noqa: E402
from compas.data import json_load  # noqa: E402
from pybullet_planning import interpolate_poses  # noqa: E402

from husky_assembly_tamp.keyframe.dual_arm_ik import resolve_arm_groups  # noqa: E402
from husky_assembly_tamp.motion_planner.api import (  # noqa: E402
    TOOL_LINK_LEFT,
    TOOL_LINK_RIGHT,
    _build_cfab_collision_fn,
    _pp_pose_from_frame,
)
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
    """The tagged solved-motion file of a bar's jointing action, in the given folder."""
    return os.path.join(solved_dir, f"{bar_id}__J.solved_motion.{tag}.json")


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
# * The check of one file
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


def complete_state(state, rcell, problem_dir: str, bar_id: str) -> List[str]:
    """Add to a saved start state the bodies and tools a newer export of the cell has.

    A solved-motion file keeps the start state of the day it was planned. When the
    design is exported again with more bodies (the RobArch export of 2026-10-05
    added a ground slab, ``ground_WG0``), the planner refuses the old state. The
    missing states are copied from the current export's own action file of the
    same bar, so they are the export's values, not guesses; a body the export does
    not know either is an error.

    Args:
        state: the saved start state, changed in place.
        rcell: the current cell.
        problem_dir (str): the exported problem folder.
        bar_id (str): the bar, e.g. "B4".

    Returns:
        List[str]: the names that were added (empty when nothing was missing).
    """
    missing_bodies = [name for name in rcell.rigid_body_models if name not in state.rigid_body_states]
    missing_tools = [name for name in rcell.tool_models if name not in state.tool_states]
    if not missing_bodies and not missing_tools:
        return []
    current = json_load(os.path.join(problem_dir, "BarActions", f"{bar_id}__J.json"))
    current_state = find_carry_movement(current).start_state
    for name in missing_bodies:
        if name not in current_state.rigid_body_states:
            raise ValueError(f"the cell has body {name!r} but neither the saved state nor the current "
                             f"export's {bar_id}__J.json knows it; look into the export")
        state.rigid_body_states[name] = current_state.rigid_body_states[name]
    for name in missing_tools:
        if name not in current_state.tool_states:
            raise ValueError(f"the cell has tool {name!r} but neither the saved state nor the current "
                             f"export's {bar_id}__J.json knows it; look into the export")
        state.tool_states[name] = current_state.tool_states[name]
    return missing_bodies + missing_tools


def check_file(planner, rcell, path: str, joint_names_12: Sequence[str], position_res: float,
               rotation_res: float, holding_only: bool = False, problem_dir: str = "") -> Dict[str, Any]:
    """Check the bar carry saved in one solved-motion file.

    Args:
        planner: the planner with the whole cell loaded.
        rcell: the cell.
        path (str): the solved-motion file.
        joint_names_12 (Sequence[str]): the twelve arm joints.
        position_res (float): fine step of the repair walk (m).
        rotation_res (float): fine step of the repair walk (rad).
        holding_only (bool): skip the repair, the collision checks and the lowest
            point; report only the path lengths and the holding error.
        problem_dir (str): the exported problem folder, used to complete a saved
            state that predates a re-export of the cell (see ``complete_state``).

    Returns:
        Dict[str, Any]: the verdict (``ok``) and what was measured.
    """
    action = json_load(path)
    movement = find_carry_movement(action)
    confs = path_from_trajectory(getattr(movement, "trajectory", None), joint_names_12)
    record: Dict[str, Any] = {"file": os.path.basename(path), "movement": movement.movement_id}
    if not confs:
        record.update({"ok": False, "reason": "no saved path"})
        return record
    confs = [np.asarray(conf, dtype=float) for conf in confs]
    state = movement.start_state
    added = complete_state(state, rcell, problem_dir, action.active_bar_id) if problem_dir else []
    if added:
        print(f"[check] {os.path.basename(path)}: saved state completed with {added} from the current export")
        record["added_from_current_export"] = added
    bar_name = next(name for name in (action.active_bar_id, f"bar_{action.active_bar_id}")
                    if name in rcell.rigid_body_models)
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

    record.update({
        "ok": bool(repair["success"]) and not hits,
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
# * Command line
# -----------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """Check the tagged solved-motion files of one bar; exit code 0 when all pass."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("problem_dir", help="the exported problem folder (holds RobotCell.json and BarActions/)")
    parser.add_argument("--bar", required=True, help="bar id, e.g. B4")
    parser.add_argument("--tags", nargs="+", required=True, help="save tags of the solved files to check")
    parser.add_argument("--solved-dir", default="",
                        help="folder holding the solved files (default: <problem folder>/BarActions)")
    parser.add_argument("--position-res", type=float, default=0.002, help="fine step of the repair walk (m)")
    parser.add_argument("--rotation-res", type=float, default=0.002, help="fine step of the repair walk (rad)")
    parser.add_argument("--out", default="", help="append one line of JSON per file to this file")
    parser.add_argument("--holding-only", action="store_true",
                        help="only the path lengths and the holding error at and between waypoints (fast)")
    args = parser.parse_args(argv)

    rcell = json_load(os.path.join(args.problem_dir, "RobotCell.json"))
    groups = resolve_arm_groups(rcell)
    joint_names_12 = arm_joint_names_12(rcell, groups)
    _client, planner = start_planner(rcell, use_gui=False)
    all_ok = True
    try:
        for tag in args.tags:
            path = solved_file(args.solved_dir or os.path.join(args.problem_dir, "BarActions"), args.bar, tag)
            if not os.path.isfile(path):
                # A missing file is reported, not skipped quietly.
                record = {"file": os.path.basename(path), "ok": False, "reason": "file not found"}
            else:
                record = check_file(planner, rcell, path, joint_names_12, args.position_res, args.rotation_res,
                                    holding_only=args.holding_only, problem_dir=args.problem_dir)
            record.update({"bar": args.bar, "tag": tag})
            all_ok = all_ok and record["ok"]
            print("[check] " + dumps(record))
            if args.out:
                with open(args.out, "a", encoding="utf-8") as handle:
                    handle.write(dumps(record) + "\n")
    finally:
        pp.disconnect()
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
