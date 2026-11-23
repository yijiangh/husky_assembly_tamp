"""Plan one bar's carry of a schema 2.1 process folder, with the roadmap planner or the BiRRT.

It reads a process folder (``process.json`` and ``actions/``, see
``scaffolding_assembly_core/doc/process_format.md``) and runs the experiment
of the holding-roadmap report in a real scene: the robot at its planned base,
the bars built so far and the halves on them, a support robot where the
process places one; the bar carry is planned from a derived start to the
authored goal with ``plan_constrained_dual_arm``, on a roadmap precomputed
for the bar (``--cdfm-backend prm``) or with the task-space BiRRT
(``--cdfm-backend birrt``).

Run from the repository root::

    .venv/bin/python -u scripts/plan_process_carry.py <process folder> --bar B21 \\
        --cdfm-backend prm --prm-workers 8 --save-dir <root>/prm/process_plans --save-tag prm
    .venv/bin/python -u scripts/plan_process_carry.py <process folder> --bar B21 \\
        --cdfm-backend birrt --max-time 300 --save-dir <root>/prm/process_plans --save-tag birrt1

Each run writes, into ``--save-dir`` (nothing is written into the process folder
unless ``--write-solution`` is given):

    <bar>_J.solved_motion.<tag>.json             the path, in the core's solution shape
    <bar>_J.solved_motion.<tag>.plan_info.json   how the plan went (also when it failed)
    <bar>_J_planned_motion.<tag>.png             the joint curves and the holding error

The found path's corners are rounded in bar pose space (``--round-radius``,
``--round-tolerance``; ``--round-radius 0`` keeps them), so the arms run
through them instead of stopping at each. ``round_process_carry.py`` rounds a
carry saved before.

``--save-carry`` also keeps the plan next to the roadmap as ``carry.json``
(``benchmark/saved_carry.py``): the carry the monitor loads the bar for.
``--start-from`` plans like the monitor then does: from such a carry's first
row as a fixed start, with its path as a seed (``--start-from auto`` reads the
roadmap's own), e.g. with ``--base-shift`` for a base parked off.

``scripts/prm_check_solved.py`` checks the saved path on its own, in the same
scene. The roadmap backend needs the bar's roadmap under
``<--prm-root>/prm/<--prm-run>`` (default run name ``<process folder>__<bar>``),
built by ``prm_pilot.py build --problem-dir <process folder> --bar <bar>`` after
``benchmark_cdfm.py build-cells <process folder>``.

How the scene is built: the core reads the process, ``process.scene_at`` gives
the world at the start of the carry, the robot's open joints are filled with
the posture the process implies there (``carry.assumed_start_all``; the planner
derives its own start anyway), and the core's ``CompasFabMirror`` turns the
scene into the compas_fab cell and state the planner API takes. That is the
same path Rhino's commands and ``plan_process.py`` of the design workflow use.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
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

from husky_assembly_tamp.benchmark.drive_in import DriveIn  # noqa: E402
from husky_assembly_tamp.benchmark.follow_on import FollowOn  # noqa: E402
from husky_assembly_tamp.benchmark.plot_motion import plot_planned_motion  # noqa: E402
from husky_assembly_tamp.benchmark.saved_carry import carry_from_plan, carry_path, read_carry, write_carry  # noqa: E402
from husky_assembly_tamp.benchmark.process_export import (  # noqa: E402
    carry_movement,
    default_plans_dir,
    goal_joints,
    goal_link_poses,
    is_process_folder,
    jointing_action,
    pose_lists,
    read_process,
    require_core,
    shifted_pose,
    carry_scene,
)
from husky_assembly_tamp.motion_planner import api  # noqa: E402
from husky_assembly_tamp.motion_planner import ssik_ik  # noqa: E402
from husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.smooth import (  # noqa: E402
    DEFAULT_ROUND_RADIUS,
    DEFAULT_ROUND_TOLERANCE,
)
from husky_assembly_tamp.motion_planner.holding_prm.store import write_json_atomic  # noqa: E402

# * The defaults the report's runs were made with.
DEFAULT_CDFM_POSITION_RES = 0.01
DEFAULT_CDFM_ROTATION_RES = 0.025
DEFAULT_MAX_TIME_S = 60.0
DEFAULT_MAX_ITERATIONS = 2000
DEFAULT_MAX_ATTEMPTS = 5
# The role name the planner API and the plots use for the bar carry.
CARRY_ROLE = "M1"


# -----------------------------------------------------------------------------
# * Files
# -----------------------------------------------------------------------------


def solved_path(save_dir: str, action_id: str, tag: str) -> str:
    """The solved-motion file of a run: ``<save_dir>/<action>.solved_motion[.<tag>].json``."""
    suffix = f".{tag}" if tag else ""
    return os.path.join(save_dir, f"{action_id}.solved_motion{suffix}.json")


def plan_info_path(solved: str) -> str:
    """The plan summary next to a solved file: ``<stem>.plan_info.json``."""
    return os.path.splitext(solved)[0] + ".plan_info.json"


def plot_path(save_dir: str, action_id: str, tag: str) -> str:
    """The figure of a run: ``<save_dir>/<action>_planned_motion[.<tag>].png``."""
    suffix = f".{tag}" if tag else ""
    return os.path.join(save_dir, f"{action_id}_planned_motion{suffix}.png")


def _plain(report: Optional[dict]) -> Optional[Dict[str, Any]]:
    """A planner report cut down to its plain numbers, texts and short lists."""
    if not report:
        return None
    return {key: value for key, value in report.items()
            if isinstance(value, (int, float, str, bool)) or (isinstance(value, (list, tuple)) and len(value) <= 40
                                                              and all(isinstance(v, (int, float)) for v in value))}


def follow_on_summary(follow: Optional[FollowOn], names: Sequence[str], confs, info: Optional[dict],
                      authored: Sequence[float]) -> Optional[Dict[str, Any]]:
    """What the follow-on check found for a plan: the end posture, the joints after it, the postures refused.

    Args:
        follow (Optional[FollowOn]): the check, or None when it did not run.
        names (Sequence[str]): the twelve arm joints.
        confs: the planned path, or None.
        info (Optional[dict]): the planner's info.
        authored (Sequence[float]): the authored end posture.

    Returns:
        Optional[Dict[str, Any]]: None when the check did not run.
    """
    if follow is None:
        return None
    summary: Dict[str, Any] = {"insert": follow.insert.id, "retreat": follow.retreat.id,
                               "rejected": ((info or {}).get("prm") or {}).get("goals_rejected", [])}
    if confs is not None:
        end = np.asarray(confs[-1], dtype=float)
        off = end - np.asarray(authored, dtype=float)
        turns = np.round(off / (2.0 * np.pi))
        # The returned path's own last row is asked too, when the later steps moved it off the accepted posture.
        summary["end_refused"] = "" if follow.keyframes_at(end) is not None else follow(end)
        keyframes = follow.keyframes_at(end) or {}
        summary.update({
            "end_joints": dict(zip(names, end.tolist())),
            "end_off_authored_deg": float(np.degrees(np.abs(off).max())),
            # Whole turns apart and nothing else: the same arm posture as authored.
            "end_is_authored_posture": bool(np.abs(off - turns * 2.0 * np.pi).max() < 1e-3),
            "insert_joints": dict(zip(names, keyframes["insert"])) if keyframes else None,
            "retreat_joints": dict(zip(names, keyframes["retreat"])) if keyframes else None,
        })
    summary["seconds"] = follow.seconds
    return summary


def save_plan_info(path: str, movement_id: str, backend: str, seconds: float, confs, info: Optional[dict],
                   base_shift: str, goal: str, follow_on: Optional[Dict[str, Any]] = None,
                   drive_in: Optional[Dict[str, Any]] = None, start_from: str = "") -> Dict[str, Any]:
    """Write a small summary of one bar-carry plan, next to the solved file.

    Args:
        path (str): where to write.
        movement_id (str): the planned movement.
        backend (str): "birrt" or "prm".
        seconds (float): wall time of the planning call.
        confs: the planned path, or None when planning failed.
        info (Optional[dict]): the planner's info.
        base_shift (str): the ``--base-shift`` in force ("" for none).
        goal (str): "frames" (any posture at the goal pose) or "joints" (the authored posture).
        follow_on (Optional[Dict[str, Any]]): the follow-on check's result (``follow_on_summary``), when it ran.
        drive_in (Optional[Dict[str, Any]]): the drive-in check's result (``DriveIn.summary``), when it ran.
        start_from (str): the carry file the plan started from (``--start-from``), "" for a derived start.

    Returns:
        Dict[str, Any]: the summary written.
    """
    info = info or {}
    profile = info.get("profile") or {}
    summary = {
        "movement": movement_id, "backend": backend, "ok": confs is not None, "plan_s": float(seconds),
        "n_waypoints": None if confs is None else len(confs), "failure_reason": info.get("failure_reason"),
        "base_shift": base_shift, "goal": goal, "follow_on": follow_on, "drive_in": drive_in,
        "start_from": start_from, "strict_contacts": info.get("strict_contacts"),
        # The time after the roadmap's answer: repair, smoothing, corner rounding.
        "post": {"smooth": _plain(info.get("smooth_profile")), "round": _plain(info.get("round_profile")),
                 "repair": _plain(info.get("repair"))},
        # The search backend: how hard it worked. Only plain numbers and the stop reasons are kept.
        "search": {key: value for key, value in profile.items()
                   if isinstance(value, (int, float, str, bool)) or key == "extend_stop_reasons"},
        # The roadmap backend: route, tolerance, counters and time per phase.
        "prm": info.get("prm"),
    }
    write_json_atomic(path, summary)
    print(f"[save] plan summary -> {path}")
    return summary


def solution_record(process, action, movement, robot: str, names: Sequence[str], confs, info: dict,
                    base, planner_settings: Dict[str, Any]) -> Dict[str, Any]:
    """The planned carry as a file in the core's solution shape (format section 10.1), for one movement.

    The file is read back by ``prm_check_solved.py``: its ``bases`` entry is
    the base pose the plan was made at (shifted or not), ``start`` the derived
    start posture, ``trajectory`` the path and ``path_poses`` the bar's poses
    along it.

    Args:
        process: the process.
        action: the jointing action.
        movement: the carry.
        robot (str): the acting robot.
        names (Sequence[str]): the twelve arm joints, in the path's order.
        confs: the planned path (one twelve-vector per waypoint).
        info (dict): the planner's info (``path_poses`` is taken from it).
        base (Pose): the base pose the plan was made at.
        planner_settings (Dict[str, Any]): recorded under ``planner.settings``.

    Returns:
        Dict[str, Any]: the file's content.
    """
    # ? The core offers no public "solution to JSON" apart from write_solution,
    # ? which writes only into <process>/solutions/; the experiment keeps its
    # ? tagged files elsewhere, so the private helpers are used here.
    from scaffolding_assembly_core.geometry import Pose
    from scaffolding_assembly_core.process.solutions import (
        SOLUTION_FORMAT, MovementResult, Planner, Trajectory, _planner, _result, solved_against)
    from scaffolding_assembly_core.process.version import writer_info
    from scaffolding_assembly_core.process.write import _writer

    rows = tuple(tuple(float(v) for v in q) for q in confs)
    start_given = movement.start.robots[robot].joints is not None
    poses = info.get("path_poses") or []
    result = MovementResult(
        status="solved",
        bases={robot: base},
        start={robot: dict(zip(names, rows[0]))},
        start_overridden=bool(start_given),
        end={robot: dict(zip(names, rows[-1]))},
        trajectory=Trajectory(robot, tuple(names), rows),
        path_poses={action.bar: tuple(Pose.from_arrays(p[0], p[1]) for p in poses)} if poses else {},
    )
    planner_block = Planner(name="husky_assembly_tamp", repo="husky_assembly_tamp",
                            ik_backend=ssik_ik.ik_backend(), settings=dict(planner_settings))
    against = solved_against(process, action.id)
    return {"format": SOLUTION_FORMAT, "writer": _writer(writer_info()), "planner": _planner(planner_block),
            "action": action.id, "solved_against": {"process": against.process, "action": against.action},
            "movements": {movement.id: _result(result)}}


def write_process_solution(process, action, record: Dict[str, Any]) -> str:
    """Also write the plan as ``<process>/solutions/<action>.json``, the file Rhino's RSLoadSolvedBarAction reads.

    Every other movement of the action is written ``not_planned``.

    Args:
        process: the process.
        action: the jointing action.
        record (Dict[str, Any]): the file from :func:`solution_record`.

    Returns:
        str: the file written.
    """
    from scaffolding_assembly_core.process.solutions import MovementResult, Solution, _solution, write_solution

    solution = _solution(record)
    results = {m.id: solution.movements.get(m.id)
               or MovementResult("not_planned", reason="only the carry was planned")
               for m in action.movements}
    return str(write_solution(process, Solution(action.id, solution.solved_against, solution.planner, results)))


# -----------------------------------------------------------------------------
# * Planning
# -----------------------------------------------------------------------------


def parse_base_shift(value: str) -> Optional[List[float]]:
    """``"dx,dy,deg"`` -> ``[dx, dy, deg]``, or None for an empty value.

    Raises:
        ValueError: when the value does not have three numbers.
    """
    if not value:
        return None
    parts = [float(v) for v in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--base-shift needs 'dx,dy,dyaw_deg'; got {value!r}")
    return parts


def prm_options_for(args, bar_id: str, problem_name: str) -> Dict[str, Any]:
    """The roadmap backend's options from the command line (see ``api.plan_constrained_dual_arm``).

    Args:
        args: the parsed arguments.
        bar_id (str): the bar, e.g. "B21".
        problem_name (str): the process folder's name (the default run name is ``<name>__<bar>``).

    Returns:
        Dict[str, Any]: ``roadmap_dir``, ``workers``, ``tolerances_mm``, ``tube_rounds``,
        ``use_pretracked``, the fallbacks' settings and ``query_tag``.

    Raises:
        SystemExit: when no data root is known.
    """
    root = args.prm_root or os.environ.get("HUSKY_BENCH_ROOT")
    if not root:
        raise SystemExit("--cdfm-backend prm needs --prm-root (or HUSKY_BENCH_ROOT): the folder holding prm/<run>.")
    run = args.prm_run or f"{problem_name}__{bar_id}"
    return {"roadmap_dir": os.path.join(root, "prm", run), "workers": args.prm_workers,
            "tolerances_mm": [float(t) for t in args.prm_tolerances_mm.split(",") if t.strip()],
            "tube_rounds": args.prm_tube_rounds, "use_pretracked": not args.prm_no_pretracked,
            # When no carry pose connects: loading poses walked to from the goal, then waypoints out of the structure.
            "walk_in": not args.prm_no_walk_in, "via_rounds": args.prm_via_rounds,
            "walk_grid_step_m": args.prm_walk_grid_step, "save_derived": not args.prm_no_walk_in,
            # The class of loading pose the operator asked for; any class when not.
            "start_carry": "" if args.cdfm_home_anchor == "all" else args.cdfm_home_anchor,
            # The plan is saved as a query of the roadmap, for its dashboard page.
            "query_tag": f"{bar_id}_{args.save_tag or 'online'}"}


def print_roadmap_summary(prm: Optional[dict]) -> None:
    """Print what the roadmap backend did: the route it used and where the time went.

    Args:
        prm (Optional[dict]): ``info["prm"]`` from the planner (None for the BiRRT: nothing is printed).
    """
    if not prm:
        return
    timings = prm.get("timings") or {}
    print("  roadmap: " + ", ".join(f"{key[:-2]} {value:.1f} s" for key, value in timings.items()))
    if prm.get("starts"):
        print(f"  roadmap: carry poses free in this scene: {prm['starts']['free_in_scene']} of "
              f"{prm['starts']['carry_nodes']} roadmap start nodes")
    if prm.get("route"):
        start = prm.get("start") or {}
        print(f"  roadmap: route of {len(prm['route'])} nodes within {prm['tolerance_mm']:g} mm "
              f"(worst edge {max(prm['route_scores_mm']):.1f} mm), from carry pose {start.get('label')} "
              f"({start.get('carry')}); {prm['n_waypoints_1cm']} waypoints at 1 cm, {prm['n_waypoints']} returned")
    elif prm.get("best_gap_left_mm") is not None:
        print(f"  roadmap: no route within the tolerances; the best one left needs {prm['best_gap_left_mm']:.1f} mm")
    print(f"  roadmap: edges looked at {prm.get('edges_looked_at', 0)} {prm.get('edge_verdicts', {})}; "
          f"blocked by the scene {prm.get('scene_blocked_edges', 0)}; dropped in the final check "
          f"{len(prm.get('dropped_in_final_check', []))}; scene bodies {prm.get('scene', {})}")


def round_options(args) -> Optional[Dict[str, Any]]:
    """``plan_constrained_dual_arm``'s ``round_corners`` from the command line; None with ``--round-radius 0``."""
    if args.round_radius <= 0.0:
        return None
    return {"radius": args.round_radius, "tolerance": args.round_tolerance, "max_rounds": args.round_max_rounds}


def print_round_summary(profile: Optional[dict]) -> None:
    """Print how the corners were rounded (nothing when rounding was off).

    Args:
        profile (Optional[dict]): ``info["round_profile"]`` from the planner.
    """
    if not profile:
        return
    if profile.get("rounded"):
        radii = ", ".join(f"{1000.0 * r:.0f}" for r in profile.get("radii", [])) or "none"
        print(f"  rounded: {profile['n_corners']} corner(s) at {1000.0 * profile['tolerance']:g} mm tolerance, "
              f"radii (mm) {radii}, {profile['rounds']} re-solve(s), {profile['wall_s']:.1f} s")
    else:
        print(f"  [!] corners not rounded, the path is as planned: {profile.get('reason')}")


def plan_carry(args) -> int:
    """Plan the carry of one bar and save what came out.

    Args:
        args: the parsed arguments.

    Returns:
        int: 0 when a path was found, 2 when planning failed.
    """
    from scaffolding_assembly_core.mirrors.compas import frame_from_pose
    from scaffolding_assembly_core.mirrors.compas_fab import CompasFabMirror
    from scaffolding_assembly_core.process.carry import assumed_start_all

    process = read_process(args.process_dir)
    problem_name = os.path.basename(os.path.normpath(args.process_dir))
    action = jointing_action(process, args.bar)
    movement = carry_movement(action)
    robot = action.robot
    print(f"[load] {action.id} ({action.label}) <- {process.folder}")
    print(f"  carry         : {movement.id} ({movement.label})")
    print(f"  robot         : {robot}, bar {action.bar}")
    print(f"  built so far  : {len(movement.start.built)} bar(s), {len(movement.start.present)} bodies present")

    # * Where the robot starts: the joints the process implies here (its own
    # * target, or where the previous movement ended). The planner derives the
    # * real start; this only poses the robot somewhere sensible.
    assumed, source = assumed_start_all(process)[(action.id, movement.id)]
    if movement.start.robots[robot].joints is not None:
        print("  start joints  : authored by the process")
    elif assumed:
        print(f"  start joints  : open; posed as at {source} until the planner derives its start")
    else:
        print("  start joints  : open; posed at the authored home until the planner derives its start")

    # * The base: as planned, or parked off its planned pose (a test of the roadmap's tolerance).
    shift = parse_base_shift(args.base_shift)
    base = None
    if shift is not None:
        planned = movement.start.robots[robot].base
        if planned is None:
            raise SystemExit(f"{movement.id}: the process does not place {robot}'s base, so it cannot be shifted.")
        base = shifted_pose(planned, shift[0], shift[1], float(np.radians(shift[2])))
        print(f"[plan] base shifted by {shift[0] * 100:.1f} cm, {shift[1] * 100:.1f} cm, {shift[2]:.1f} deg "
              "from its planned pose (--base-shift)")
    # * --start-from: the start is fixed, the first row of a carry planned before, and its path seeds the plan.
    seed, seed_file = None, ""
    if args.start_from:
        if args.cdfm_backend != "prm":
            raise SystemExit("--start-from needs --cdfm-backend prm.")
        seed_file = args.start_from
        if seed_file == "auto":
            seed_file = carry_path(prm_options_for(args, args.bar, problem_name)["roadmap_dir"])
        seed = read_carry(seed_file)
        if seed["action"] != action.id:
            raise SystemExit(f"{seed_file} is the carry of {seed['action']}, not of {action.id}.")
        assumed = dict(zip(seed["joint_names"], seed["path"][0]))
        print(f"[plan] start fixed at the first row of {seed_file} ({len(seed['path'])} rows, planned at base "
              f"({seed['base'][0]:.3f}, {seed['base'][1]:.3f})); its path seeds the plan")
    scene, _action, _movement = carry_scene(process, args.bar, joints=assumed, base=base)
    base_used = scene.robots[robot].base

    # * The scene as a compas_fab world, built by the core's mirror.
    mirror = CompasFabMirror(robot, log=print)
    if args.gui:
        mirror.set_gui(True)
    t0 = time.time()
    mirror.sync(scene)
    print(f"[pb] scene synced in {time.time() - t0:.1f} s")
    if mirror.parked:
        print(f"[pb] parked (base or joints not given by the process): {', '.join(mirror.parked)}")
    left_names, right_names = api._arm_joint_names(mirror.cell)
    names = list(left_names) + list(right_names)
    if seed is not None and list(seed["joint_names"]) != names:
        raise SystemExit(f"{seed_file}: its joints are not this robot's twelve arm joints in the planner's order.")

    # * The goal: the authored flange poses (any posture there), or the authored posture itself.
    goal_poses = goal_link_poses(movement, robot)
    goal_conf = goal_joints(movement, robot, names) if args.goal == "joints" else None
    goal_frames = None if goal_conf is not None else {side: frame_from_pose(pose) for side, pose in goal_poses.items()}

    prm_options = prm_options_for(args, args.bar, problem_name) if args.cdfm_backend == "prm" else None
    if prm_options:
        print(f"[plan] roadmap backend: {prm_options['roadmap_dir']} ({args.prm_workers} worker(s))")
    print(f"[plan] resolutions: M1 pos {args.cdfm_position_res} m / rot {args.cdfm_rotation_res} rad | "
          f"home anchor {args.cdfm_home_anchor} | goal {args.goal}")

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    solved = solved_path(save_dir, action.id, args.save_tag)

    print(f"[plan] plan_constrained_dual_arm (active_bar_id={action.bar}, derive_start={seed is None}, "
          f"backend={args.cdfm_backend})")
    # * With the goal as flange poses the roadmap picks the end posture: keep only one the insert and the retreat
    # * can follow from.
    # * The search cannot be asked while it plans: its end posture is checked once, afterwards.
    follow = None
    if goal_frames is not None and not args.no_follow_on:
        follow = FollowOn(process, args.bar, names, base=base, log=print)
        print(f"[plan] end posture resolved by IK; it must let {follow.insert.id} and {follow.retreat.id} follow")
    # * The start is a loading posture: keep only one the robot can drive in with from the station.
    drive = None
    if seed is not None:
        print("[plan] the start is fixed: no drive-in check (it was made when the carry was planned)")
    elif args.cdfm_backend == "prm" and args.station is not None and not args.no_drive_in:
        station = [args.station[0], args.station[1], float(np.radians(args.station[2]))]
        drive = DriveIn(process, args.bar, names, station, base=base, time_limit=args.drive_time_limit,
                        log=lambda line: print(line[:200]))
        print(f"[plan] the start must be a loading posture the base can drive in with from the station at "
              f"({args.station[0]:.2f}, {args.station[1]:.2f}) m, {args.station[2]:.0f} deg")
    elif args.cdfm_backend == "prm":
        print("[plan] ! loading postures are NOT checked for the drive in (no --station, or --no-drive-in)")
    authored = goal_joints(movement, robot, names)
    if prm_options and goal_frames is not None:
        # * The same arm posture a whole turn apart may or may not let the insert stay inside the joint limits: the
        # * end postures are offered on the turns nearest home and nearest the authored joints.
        prm_options["goal_turn_reference"] = [float(v) for v in authored]
    if seed is not None:
        # * End on the seed's arm posture (the one its insert was checked from), on its whole turns.
        prm_options["seed_path"] = seed["path"]
        # * Where the robot stood when the carry was planned, and where it stands now: the seed is solved again.
        prm_options["seed_base_planned"] = [float(v) for v in seed["base"]]
        prm_options["seed_base_live"] = [float(v) for v in (*base_used.position, *base_used.orientation)]
        prm_options["goal_turn_reference"] = [float(v) for v in seed["path"][-1]]
    plan_t0 = time.time()
    with mirror.lend() as planner:
        state = mirror.state
        confs, info = api.plan_constrained_dual_arm(
            planner, state,
            active_bar_id=action.bar,
            goal_conf=goal_conf,
            goal_ee_frames=goal_frames,
            position_res=args.cdfm_position_res,
            rotation_res=args.cdfm_rotation_res,
            max_time=args.max_time,
            max_iterations=args.max_iterations,
            max_attempts=args.max_attempts,
            use_birrt=True,
            derive_start=seed is None,
            start_home_anchor=None if args.cdfm_home_anchor == "all" else args.cdfm_home_anchor,
            backend=args.cdfm_backend,
            prm_options=prm_options,
            round_corners=round_options(args),
            goal_accept=follow if args.cdfm_backend == "prm" else None,
            start_accept=drive,
        )
        plan_s = time.time() - plan_t0
        summary = save_plan_info(
            plan_info_path(solved), movement.id, args.cdfm_backend, plan_s, confs, info, args.base_shift, args.goal,
            follow_on_summary(follow, names, confs, info, authored),
            drive.summary(confs[0] if confs is not None else None) if drive is not None else None, seed_file)
        if follow is not None:
            follow.close()
        if drive is not None:
            drive.close()

        if confs is None:
            reason = (info or {}).get("failure_reason", "<unknown>")
            print(f"[plan] {CARRY_ROLE} FAILED: {reason} (after {plan_s:.1f} s)")
            print_roadmap_summary((info or {}).get("prm"))
            profile = (info or {}).get("profile") or {}
            if profile.get("extend_stop_reasons"):
                print(f"  extend_stop_reasons: {profile['extend_stop_reasons']}")
            mirror.close()
            return 2

        print(f"[plan] {CARRY_ROLE} OK: {len(confs)} waypoint(s) in {plan_s:.1f} s")
        profile = (info or {}).get("profile") or {}
        if profile.get("extend_stop_reasons"):
            print(f"  extend_stop_reasons: {profile['extend_stop_reasons']}")
        for key, value in (info or {}).items():
            if key in ("profile", "smooth_profile", "round_profile", "path_poses", "derived_start_conf", "prm"):
                continue
            print(f"  {key}: {value}")
        print_roadmap_summary((info or {}).get("prm"))
        print_round_summary((info or {}).get("round_profile"))

        # How far the path ends from the authored posture (the insert starts there).
        off = float(np.abs(np.asarray(confs[-1], dtype=float) - np.asarray(authored)).max())
        if off > 1e-3 and follow is not None:
            print(f"[plan] {movement.id}: the path ends {np.degrees(off):.1f} deg (largest joint) from the authored "
                  f"target joints; {follow.insert.id} and {follow.retreat.id} were planned from where it ends "
                  "(their joints are in the plan summary).")
        elif off > 1e-3:
            print(f"[!] {movement.id}: the path ends {np.degrees(off):.1f} deg (largest joint) from the authored "
                  "target joints, on another arm posture at the same goal pose. The movements after it start "
                  "from the authored posture; plan with --goal joints to end exactly there.")
        else:
            print(f"[plan] {movement.id}: the path ends on the authored target joints.")

        settings = {"backend": args.cdfm_backend, "tag": args.save_tag, "goal": args.goal,
                    "base_shift": args.base_shift, "max_time": args.max_time,
                    "position_res": args.cdfm_position_res, "rotation_res": args.cdfm_rotation_res,
                    "home_anchor": args.cdfm_home_anchor, "prm": prm_options or {},
                    "round": round_options(args) or {}}
        record = solution_record(process, action, movement, robot, names, confs, info or {}, base_used, settings)
        write_json_atomic(solved, record)
        print(f"[save] solved carry -> {solved}")
        if args.write_solution:
            print(f"[save] process solution -> {write_process_solution(process, action, record)}")
        if args.save_carry:
            kept = write_carry(prm_options["roadmap_dir"], carry_from_plan(record, summary, problem_name))
            print(f"[save] the roadmap's carry -> {kept}")

        if not args.no_plot:
            save_plot(planner, mirror, names, confs, action, plot_path(save_dir, action.id, args.save_tag))
    print(f"\n[timing] {CARRY_ROLE}: {plan_s:.1f} s")
    mirror.close()
    return 0


def save_plot(planner, mirror, names: Sequence[str], confs, action, out_path: str) -> None:
    """Draw the joint curves and the holding error of the planned carry (``benchmark.plot_motion``).

    Args:
        planner: the lent planner.
        mirror: the mirror (for a state at the path's first waypoint).
        names (Sequence[str]): the twelve arm joints.
        confs: the planned path.
        action: the jointing action.
        out_path (str): the png to write.
    """
    start_state = mirror.state_at(dict(zip(names, confs[0])))
    plot_planned_motion(planner, [(CARRY_ROLE, start_state, confs)], list(names), out_path, title=action.id)


# -----------------------------------------------------------------------------
# * Command line
# -----------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """Parse the command line, plan the carry and save it; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("process_dir", help="the process folder (holds process.json and actions/)")
    parser.add_argument("--bar", required=True, help="bar id, e.g. B21")
    parser.add_argument("--cdfm-backend", choices=("birrt", "prm"), default="birrt",
                        help="the planner behind plan_constrained_dual_arm: the task-space BiRRT or the "
                             "roadmap precomputed for the bar")
    parser.add_argument("--goal", choices=("frames", "joints"), default="frames",
                        help="end at the authored flange poses in any posture (frames, as the report's runs) "
                             "or exactly at the authored posture (joints)")
    parser.add_argument("--no-follow-on", action="store_true",
                        help="with --goal frames on the roadmap: take any end posture, without checking that the "
                             "insert and the retreat can be planned from it")
    parser.add_argument("--max-time", type=float, default=DEFAULT_MAX_TIME_S, help="BiRRT time budget (s)")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS, help="BiRRT iteration cap")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS, help="BiRRT restarts")
    parser.add_argument("--cdfm-position-res", type=float, default=DEFAULT_CDFM_POSITION_RES,
                        help="bar translation step of the path (m)")
    parser.add_argument("--cdfm-rotation-res", type=float, default=DEFAULT_CDFM_ROTATION_RES,
                        help="bar rotation step of the path (rad)")
    parser.add_argument("--cdfm-home-anchor", choices=("all", "horizontal", "vertical", "back"), default="all",
                        help="the class of loading pose the carry may start from: the bar across the front "
                             "(horizontal), upright in front (vertical) or along the robot over its back (back)")
    parser.add_argument("--prm-root", default="",
                        help="the benchmark data root holding prm/<run> (default: HUSKY_BENCH_ROOT)")
    parser.add_argument("--prm-run", default="", help="the roadmap's run name (default: <process folder>__<bar>)")
    parser.add_argument("--prm-workers", type=int, default=1, help="worker processes of the roadmap backend")
    parser.add_argument("--prm-tolerances-mm", default="10,20,30",
                        help="route tolerances the roadmap backend tries, in order")
    parser.add_argument("--prm-tube-rounds", type=int, default=1, help="tube rounds of the roadmap backend")
    parser.add_argument("--prm-no-pretracked", action="store_true",
                        help="ignore the roadmap's tracking file (track every route edge now)")
    parser.add_argument("--prm-no-walk-in", action="store_true",
                        help="when no carry pose of the roadmap connects, give up instead of walking the goal to "
                             "other loading poses of the three classes")
    parser.add_argument("--prm-via-rounds", type=int, default=8,
                        help="rounds of waypoints grown from the goal out of the structure after that (0: none)")
    parser.add_argument("--prm-walk-grid-step", type=float, default=0.3,
                        help="spacing of the grid of loading poses tried in each class (m)")
    parser.add_argument("--round-radius", type=float, default=DEFAULT_ROUND_RADIUS,
                        help="round the path's corners in bar pose space with this radius (m, largest bar "
                             "feature-point move from the corner; 0: keep the corners)")
    parser.add_argument("--round-tolerance", type=float, default=DEFAULT_ROUND_TOLERANCE,
                        help="bar path samples within this of a straight piece belong to it when the corners are "
                             "looked for (m)")
    parser.add_argument("--round-max-rounds", type=int, default=40,
                        help="most re-solves of the whole path the corner rounding may try before it gives the "
                             "path back as planned")
    parser.add_argument("--station", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"), default=None,
                        help="the operator's station, the base pose in the process's world frame: the roadmap "
                             "backend then starts only from loading postures the base can drive in with from there")
    parser.add_argument("--no-drive-in", action="store_true", help="do not check the drive in from the station")
    parser.add_argument("--drive-time-limit", type=float, default=8.0,
                        help="time limit of one base search of the drive-in check, seconds (a refusal for lack of "
                             "time takes up to about twice it)")
    parser.add_argument("--save-carry", action="store_true",
                        help="with the roadmap backend: also keep the plan next to the roadmap as carry.json, the "
                             "carry the monitor loads the bar for (not with --start-from or --base-shift)")
    parser.add_argument("--start-from", default="",
                        help="a carry.json ('auto': the roadmap's own): plan from its first row as a fixed start, "
                             "with its path as a seed, as the monitor does at the live base")
    parser.add_argument("--base-shift", default="",
                        help="park the base off its planned pose: 'dx,dy,dyaw_deg' in the base's own frame")
    parser.add_argument("--save-dir", default="",
                        help="where the results go (default: <data root>/prm/plans/<process folder>, "
                             "or <process folder>/solutions/plans without a data root)")
    parser.add_argument("--save-tag", default="", help="a tag in the file names, so runs do not overwrite each other")
    parser.add_argument("--write-solution", action="store_true",
                        help="also write <process>/solutions/<bar>_J.json, the file Rhino loads")
    parser.add_argument("--no-plot", action="store_true", help="skip the figure")
    parser.add_argument("--gui", action="store_true", help="show the scene in PyBullet's window")
    args = parser.parse_args(argv)

    if not is_process_folder(args.process_dir):
        print(f"[X] {args.process_dir} is not a process folder (no process.json).")
        return 1
    try:
        require_core()
    except ImportError as exc:
        print(f"[X] {exc}")
        return 1
    if not args.save_dir:
        args.save_dir = default_plans_dir(args.process_dir, args.prm_root or os.environ.get("HUSKY_BENCH_ROOT"))
    if args.save_carry and (args.cdfm_backend != "prm" or args.start_from or args.base_shift):
        print("[X] --save-carry keeps a roadmap plan made at the planned base with a derived start.")
        return 1
    try:
        parse_base_shift(args.base_shift)
    except ValueError as exc:
        print(f"[X] {exc}")
        return 1
    return plan_carry(args)


if __name__ == "__main__":
    sys.exit(main())
