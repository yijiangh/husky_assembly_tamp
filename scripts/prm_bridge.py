"""Bridge from the holding roadmap to the production planner (this script imports the planner core).

    validate  re-solve every exact path a refinement stored with the production
              repair (``core.repair_joint_path``) at the benchmark's fine step
              (2 mm / 0.002 rad), the production continuity gate and the
              production collision check: "exact" then means "a path the
              production planner would accept"
    birrt     (way 1 of the refinement plan) for starts whose roadmap route
              does not track exactly, grow a task-space BiRRT whose trees start
              from the route's exact pieces and whose samples stay in a tube
              around the route

Examples (PowerShell, from the repository root, with HUSKY_BENCH_ROOT set):

    .venv\\Scripts\\python.exe scripts\\prm_bridge.py validate --name B18_g2_n4000 --refine all_k2.5_p10mm_mid
    .venv\\Scripts\\python.exe scripts\\prm_bridge.py birrt --name B18_g2_n4000 --refine all_k2.5_p10mm_mid_tube3

! Importing the planner core opens its log file in "w" mode, which would empty
! the log a running benchmark batch is writing. This script points the log
! folder somewhere else BEFORE the core is imported (the three lines right
! below), and records the planner log's size and time before and after.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from time import perf_counter, time
from typing import Any, Dict, List, Optional, Sequence

# The package sits one level up from scripts/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# * Redirect the planner's log folder first; everything below may import the core.
from husky_assembly_tamp.utils import util as planner_util  # noqa: E402

PLANNER_LOG_DIR = planner_util.LOG_DIR
BRIDGE_LOG_DIR = os.path.join(PLANNER_LOG_DIR, "prm_bridge")
planner_util.LOG_DIR = BRIDGE_LOG_DIR

import numpy as np  # noqa: E402
from pybullet_planning import interpolate_poses  # noqa: E402

from husky_assembly_tamp.benchmark.scene import load_trimmed_problem, open_scene, quiet_planner_logging  # noqa: E402
from husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.core import (  # noqa: E402
    DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
    repair_joint_path,
)
from husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.seeded import as_pose, plan_seeded_birrt  # noqa: E402
from husky_assembly_tamp.motion_planner.holding_prm.holding import (  # noqa: E402
    HoldingModel,
    bar_from_arm,
    interpolate_confs,
    linf,
    pose_from_matrix,
    target_bar_pose,
)
from husky_assembly_tamp.motion_planner.holding_prm.refine import refine_path  # noqa: E402
from husky_assembly_tamp.motion_planner.holding_prm.store import (  # noqa: E402
    load_roadmap,
    provenance_record,
    read_json,
    run_dir,
    write_json_atomic,
)

# * The benchmark's fine pass (benchmark/cdfm.py): the resolution production paths are checked at.
FINE_POSITION_RES_M = 0.002
FINE_ROTATION_RES_RAD = 0.002

# * The core's own log file, which must not be touched by this script.
CORE_LOG = os.path.join(PLANNER_LOG_DIR, "dual_arm_task_space_rrt_core.log")


# -----------------------------------------------------------------------------
# * One scene per worker process
# -----------------------------------------------------------------------------

_WORKER: Dict[str, Any] = {}


def init_worker(root: str, source_key: str, bar_id: str) -> None:
    """Open this process's scene and the production solver context. Runs once per process.

    Args:
        root (str): the benchmark data root.
        source_key (str): trimmed-scene cache key.
        bar_id (str): the bar.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    quiet_planner_logging()
    problem, cell, template = load_trimmed_problem(root, source_key, bar_id)
    scene = open_scene(cell, template, problem)
    _WORKER.clear()
    _WORKER.update({
        "scene": scene,
        # The solver context repair_joint_path expects; ik_options None = the calibrated production solver.
        "context": {
            "robot": scene.robot_puid, "arm_joints": scene.arm_joints,
            "tool_link_left": scene.tool_link_left, "tool_link_right": scene.tool_link_right,
            "grasp_bar_from_left": scene.grasp_bar_from_left, "grasp_bar_from_right": scene.grasp_bar_from_right,
            "ik_options": None,
        },
        "collides": scene.collision_fn(),
    })


def densify(bar_poses: Sequence[Sequence[Sequence[float]]], position_res: float,
            rotation_res: float) -> List[Any]:
    """Cut the bar path between stored waypoints into the production's fine steps.

    Args:
        bar_poses (Sequence): ``[[x, y, z], [qx, qy, qz, qw]]`` poses, start first.
        position_res (float): longest move between fine poses (m).
        rotation_res (float): largest turn between fine poses (rad).

    Returns:
        List[Any]: the fine poses (pybullet tuples), start and end included.
    """
    poses = [tuple(map(tuple, bar_poses[0]))]
    for a, b in zip(bar_poses[:-1], bar_poses[1:]):
        steps = list(interpolate_poses(tuple(map(tuple, a)), tuple(map(tuple, b)),
                                       pos_step_size=position_res, ori_step_size=rotation_res))
        poses.extend(steps[1:])
    return poses


def production_repair(bar_poses: Sequence[Any], confs: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Re-solve a path with the production repair at the fine step; returns the repair's info.

    Args:
        bar_poses (Sequence[Any]): the path's bar poses (left-flange convention), start first.
        confs (Sequence[Sequence[float]]): its configurations (only the two ends are used:
            the repair must start at the first and land on the last).

    Returns:
        Dict[str, Any]: the repair's info (success, reason, direction, max step,
        ...) plus ``n_fine_poses``.
    """
    poses = densify(bar_poses, FINE_POSITION_RES_M, FINE_ROTATION_RES_RAD)
    _repaired, info = repair_joint_path(
        _WORKER["context"], poses, np.asarray(confs[0]), np.asarray(confs[-1]),
        joint_collision_fn=_WORKER["collides"], joint_continuity_threshold_rad=DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
    )
    return {**info, "n_fine_poses": len(poses)}


def task_validate(task: Dict[str, Any]) -> Dict[str, Any]:
    """Re-solve one stored exact path with the production repair.

    Args:
        task (Dict[str, Any]): ``label``, ``confs`` (the exact waypoints) and
            ``bar_poses`` (their bar poses, as the left flange places the bar).

    Returns:
        Dict[str, Any]: ``label``, the repair's ``info`` (success, reason,
        direction, max step, ...), ``n_fine_poses`` and ``seconds``.
    """
    started = perf_counter()
    info = production_repair(task["bar_poses"], task["confs"])
    return {"label": task["label"], "info": info, "n_fine_poses": info["n_fine_poses"],
            "seconds": perf_counter() - started}


def task_birrt(task: Dict[str, Any]) -> Dict[str, Any]:
    """Run the seeded BiRRT for one start, then the production repair on what it found.

    Args:
        task (Dict[str, Any]): ``start_index``, the two chains (``start_poses``,
            ``start_confs``, ``goal_poses``, ``goal_confs``), the tube
            (``guide_poses``, ``guide_weights``) and ``options`` (keyword
            arguments of ``plan_seeded_birrt``).

    Returns:
        Dict[str, Any]: ``start_index``, the planner's ``info``, the repair's
        ``repair`` info (None when no path was found), and the path
        (``bar_poses``, ``confs``) when found.
    """
    poses, confs, info = plan_seeded_birrt(
        _WORKER["context"], _WORKER["collides"],
        [as_pose(p) for p in task["start_poses"]], task["start_confs"],
        [as_pose(p) for p in task["goal_poses"]], task["goal_confs"],
        [as_pose(p) for p in task["guide_poses"]], task["guide_weights"], **task["options"],
    )
    out = {"start_index": task["start_index"], "info": info, "repair": None,
           "chain_lengths": [len(task["start_confs"]), len(task["goal_confs"])]}
    if poses is not None:
        pose_lists = [[list(p[0]), list(p[1])] for p in poses]
        out["repair"] = production_repair(pose_lists, confs)
        out["bar_poses"] = pose_lists
        out["confs"] = np.round(np.asarray(confs), 7).tolist()
    return out


# -----------------------------------------------------------------------------
# * validate
# -----------------------------------------------------------------------------


def stored_paths(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every exact path a refinement stored, with a label saying where it came from."""
    out = []
    headline = record["headline"]
    if headline.get("confs"):
        out.append({"label": f"headline_s{headline['best_start_index']}", "confs": headline["confs"],
                    "bar_poses": headline["bar_poses"]})
    for answer in record["tolerances"]:
        shortest = answer.get("shortest")
        if shortest and shortest.get("confs"):
            out.append({"label": f"shortest_{answer['tolerance_mm']:g}mm_s{shortest['start_index']}",
                        "confs": shortest["confs"], "bar_poses": shortest["bar_poses"]})
    for key, path in record["paths"].items():
        out.append({"label": f"start_{key}", "confs": path["confs"], "bar_poses": path["bar_poses"]})
    return out


def log_stamp() -> Optional[List[float]]:
    """Size and modification time of the core's log file (None when it does not exist)."""
    if not os.path.isfile(CORE_LOG):
        return None
    status = os.stat(CORE_LOG)
    return [float(status.st_size), float(status.st_mtime)]


def cmd_validate(root: str, args: argparse.Namespace) -> None:
    """Run the production repair on every stored exact path of one refinement."""
    stamp_before = log_stamp()
    record = read_json(refine_path(root, args.name, args.refine))
    config = read_json(os.path.join(run_dir(root, args.name), "config.json"))["settings"]
    tasks = stored_paths(record)
    if args.max_paths:
        tasks = tasks[: args.max_paths]
    print(f"[prm bridge] validate {args.name} / {args.refine}: {len(tasks)} stored paths, "
          f"fine step {FINE_POSITION_RES_M * 1000:g} mm / {FINE_ROTATION_RES_RAD} rad, {args.workers} workers")
    started = perf_counter()
    init_args = (root, config["source_key"], config["bar_id"])
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"),
                             initializer=init_worker, initargs=init_args) as executor:
        for result in executor.map(task_validate, tasks):
            info = result["info"]
            print(f"  {result['label']:<28s} {'ok' if info['success'] else 'FAILED':<7s} "
                  f"{result['n_fine_poses']:>6d} fine poses, direction {info['direction']}, "
                  f"reason {info['reason']}, largest step "
                  f"{np.degrees(info['max_step_rad']) if info['max_step_rad'] is not None else float('nan'):.2f} deg, "
                  f"{result['seconds']:.0f} s", flush=True)
            results.append(result)
    stamp_after = log_stamp()
    out = {
        "run_name": args.name, "refine_tag": args.refine, "created": time(),
        "fine_position_res_m": FINE_POSITION_RES_M, "fine_rotation_res_rad": FINE_ROTATION_RES_RAD,
        "results": results, "n_ok": sum(1 for r in results if r["info"]["success"]),
        "wall_s": perf_counter() - started,
        "core_log_untouched": stamp_before == stamp_after, "core_log_stamp": [stamp_before, stamp_after],
        "provenance": provenance_record(),
    }
    path = os.path.join(run_dir(root, args.name), "refine", f"{args.refine}_validate.json")
    write_json_atomic(path, out)
    print(f"[prm bridge] {out['n_ok']} of {len(results)} paths pass the production repair "
          f"({out['wall_s']:.0f} s); core log untouched: {out['core_log_untouched']}; wrote {path}")


# -----------------------------------------------------------------------------
# * birrt (way 1)
# -----------------------------------------------------------------------------

# * Spacing of the guide poses laid along the route's joint lines, and how much
# * more often a guide pose on a failed edge is picked.
GUIDE_STEP_DEG = 2.5
GUIDE_FAILED_EDGE_WEIGHT = 4.0


def left_bar_pose(model: HoldingModel, conf: Sequence[float]) -> List[List[float]]:
    """The bar pose a configuration holds, as the left flange places it (the planner's convention)."""
    return pose_from_matrix(bar_from_arm(model, "left", np.asarray(conf, dtype=float)[:6]))


def exact_chains(attempt: Dict[str, Any], gate: float, use_chains: bool) -> Dict[str, List[Any]]:
    """The exact waypoints from the start up to the first failed edge, and from the goal back to the last one.

    Args:
        attempt (Dict[str, Any]): an ``unresolved`` entry of a refinement record.
        gate (float): the continuity gate; a chain also stops where two
            pieces do not join within it.
        use_chains (bool): False keeps only the two end configurations (the
            control: a tube-guided BiRRT without the roadmap's exact pieces).

    Returns:
        Dict[str, List[Any]]: ``start_confs`` (start first) and ``goal_confs`` (goal first).
    """
    guide = attempt["guide_confs"]
    chains = {"start_confs": [guide[0]], "goal_confs": [guide[-1]]}
    if not use_chains:
        return chains
    for name, pieces in (("start_confs", attempt["pieces"]),
                         ("goal_confs", [None if p is None else p[::-1] for p in reversed(attempt["pieces"])])):
        chain = chains[name]
        for piece in pieces:
            if piece is None or linf(piece[0], chain[-1]) > gate:
                break
            chain.extend(piece[1:] if linf(piece[0], chain[-1]) < 1e-9 else piece)
    return chains


def guide_tube(model: HoldingModel, attempt: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Bar poses along the route's joint lines (the tube's centre line), weighted toward failed edges."""
    step = float(np.radians(GUIDE_STEP_DEG))
    poses, weights = [], []
    confs = attempt["guide_confs"]
    for k, edge in enumerate(attempt["edges"]):
        line = interpolate_confs(confs[k], confs[k + 1], step, include_ends=True)
        poses.extend(pose_from_matrix(target_bar_pose(model, conf, "mid")) for conf in line)
        weights.extend([1.0 if edge["free"] else GUIDE_FAILED_EDGE_WEIGHT] * len(line))
    return {"guide_poses": poses, "guide_weights": weights}


def cmd_birrt(root: str, args: argparse.Namespace) -> None:
    """Run the seeded BiRRT on every unresolved start of one refinement."""
    stamp_before = log_stamp()
    record = read_json(refine_path(root, args.name, args.refine))
    config = read_json(os.path.join(run_dir(root, args.name), "config.json"))["settings"]
    model = load_roadmap(root, args.name).model
    unresolved = record.get("unresolved", {})
    keys = sorted(unresolved, key=int)
    if args.starts:
        wanted = {part.strip() for part in args.starts.split(",")}
        keys = [k for k in keys if k in wanted]
    options = {"max_time": args.max_time, "tube_pos_sigma_m": args.tube_pos_mm / 1000.0,
               "tube_rot_sigma_rad": float(np.radians(args.tube_rot_deg)), "tip_bias": args.tip_bias,
               "random_seed": args.seed}
    tasks = []
    for key in keys:
        attempt = unresolved[key]
        chains = exact_chains(attempt, DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD, not args.no_chains)
        tasks.append({
            "start_index": int(key), **chains,
            "start_poses": [left_bar_pose(model, c) for c in chains["start_confs"]],
            "goal_poses": [left_bar_pose(model, c) for c in chains["goal_confs"]],
            **guide_tube(model, attempt), "options": options,
        })
    label = "nochains" if args.no_chains else "chains"
    print(f"[prm bridge] birrt {args.name} / {args.refine} ({label}): {len(tasks)} unresolved starts, "
          f"{args.max_time:g} s each, {args.workers} workers")
    started = perf_counter()
    init_args = (root, config["source_key"], config["bar_id"])
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"),
                             initializer=init_worker, initargs=init_args) as executor:
        for result in executor.map(task_birrt, tasks):
            info, repair = result["info"], result["repair"]
            verdict = "no path" if repair is None else ("ok" if repair["success"] else f"repair {repair['reason']}")
            print(f"  start {result['start_index']:>4d}: {verdict:<24s} chains {result['chain_lengths']}, "
                  f"{info['iterations']} iterations, {info['n_ik_solves']} IK solves, {info['wall_s']:.0f} s",
                  flush=True)
            results.append(result)
    out = {
        "run_name": args.name, "refine_tag": args.refine, "created": time(), "use_chains": not args.no_chains,
        "options": options, "results": results,
        "n_found": sum(1 for r in results if r["repair"] is not None),
        "n_ok": sum(1 for r in results if r["repair"] is not None and r["repair"]["success"]),
        "wall_s": perf_counter() - started,
        "core_log_untouched": stamp_before == log_stamp(), "provenance": provenance_record(),
    }
    path = os.path.join(run_dir(root, args.name), "refine", f"{args.refine}_birrt_{label}.json")
    write_json_atomic(path, out)
    print(f"[prm bridge] {out['n_found']} of {len(results)} starts found a path, {out['n_ok']} pass the "
          f"production repair ({out['wall_s']:.0f} s); core log untouched: {out['core_log_untouched']}; wrote {path}")


# -----------------------------------------------------------------------------
# * Command line
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The argument parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="re-solve stored exact paths with the production repair")
    validate.add_argument("--root", default=None, help="benchmark data root (or HUSKY_BENCH_ROOT)")
    validate.add_argument("--name", required=True, help="roadmap run")
    validate.add_argument("--refine", required=True, help="refinement tag")
    validate.add_argument("--max-paths", type=int, default=0, help="only the first N stored paths (0 = all)")
    validate.add_argument("--workers", type=int, default=6)

    birrt = commands.add_parser("birrt", help="seeded BiRRT on the starts a refinement left unresolved")
    birrt.add_argument("--root", default=None, help="benchmark data root (or HUSKY_BENCH_ROOT)")
    birrt.add_argument("--name", required=True, help="roadmap run")
    birrt.add_argument("--refine", required=True, help="refinement tag")
    birrt.add_argument("--starts", default="", help="only these start indices (default: all unresolved)")
    birrt.add_argument("--max-time", type=float, default=120.0, help="seconds per start")
    birrt.add_argument("--tube-pos-mm", type=float, default=50.0, help="spread of a sample's shift")
    birrt.add_argument("--tube-rot-deg", type=float, default=8.6, help="spread of a sample's turn")
    birrt.add_argument("--tip-bias", type=float, default=0.1, help="chance of aiming at the other tree's tip")
    birrt.add_argument("--no-chains", action="store_true",
                       help="control: start the trees from the two ends only, not the route's exact pieces")
    birrt.add_argument("--seed", type=int, default=0)
    birrt.add_argument("--workers", type=int, default=6)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run one subcommand."""
    args = build_parser().parse_args(argv)
    root = args.root or os.environ.get("HUSKY_BENCH_ROOT")
    if not root:
        raise SystemExit("No benchmark data root. Pass --root, or set HUSKY_BENCH_ROOT.")
    if args.command == "validate":
        cmd_validate(os.path.abspath(root), args)
    elif args.command == "birrt":
        cmd_birrt(os.path.abspath(root), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
