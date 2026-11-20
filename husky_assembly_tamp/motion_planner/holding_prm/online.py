"""Planning a bar carry on a precomputed roadmap, in a scene the roadmap has never seen.

The roadmap of a bar is computed ahead of time with only the robot, its tools
and the carried bar (``prm_pilot.py build --problem-dir`` and ``track``). Its
nodes are joint configurations, so nothing in it depends on where the robot
stands or on what has been built around it. At plan time this module adds
what only the real scene can tell:

    1. the robot's real base pose: the arm bases are read from the live
       scene, so "the goal" becomes the goal as the robot now sees it;
    2. the goal itself: solved fresh at the real pose, as new nodes joined to
       the nearest roadmap nodes;
    3. the scene: built bars, joints, a support robot. Only pairs that involve
       something stationary are checked for what the roadmap already knows
       (``sampling.make_collision_fns`` -> ``scene_only``); anything solved
       fresh gets the full check.

The search is lazy: find the shortest route within a tolerance, check and
track only that route's edges, drop what fails, search again. The winning
route is then walked at the caller's resolution and collision-checked at
every fine step, in parallel; a failure there drops the edge and the search
goes on. Nothing here falls back to another planner: when no route survives,
the caller is told why.

! ``parallel`` imports ``benchmark.scene``, which imports ``motion_planner.api``,
! so ``api`` must import this module inside the function that uses it.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import asdict, dataclass, field
from time import perf_counter, time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pybullet_planning import get_joint_positions, get_link_pose, link_from_name, tform_from_pose

from husky_assembly_tamp.motion_planner.holding_prm.holding import (
    ARMS,
    HoldingModel,
    bar_from_arm,
    continuous_poses,
    geometry_to_lists,
    node_geometry,
    pose_from_matrix,
    pose_gap,
    tool0_world,
)
from husky_assembly_tamp.motion_planner.holding_prm.parallel import (
    WorkerPool,
    task_check,
    task_edges,
    task_expand,
    task_fine,
)
from husky_assembly_tamp.motion_planner.holding_prm.pilot import chunked, edge_tasks
from husky_assembly_tamp.motion_planner.holding_prm.refine import QueryGraph, RefineSettings, Tracker, tube_round
from husky_assembly_tamp.motion_planner.holding_prm.roadmap import EDGE_COLUMNS, NODE_COLUMNS, Roadmap, candidate_pairs
from husky_assembly_tamp.motion_planner.holding_prm.sampling import (
    WorkerTools,
    add_counters,
    make_collision_fns,
    robot_side_signature,
    with_tool_names_of,
    without_prefix,
    signature_differences,
)
from husky_assembly_tamp.motion_planner.holding_prm.scenes import build_scene_cell, save_scene
from husky_assembly_tamp.motion_planner.holding_prm.search import EdgeTable, minimax_to_goals, route_from
from husky_assembly_tamp.motion_planner.holding_prm.walk_in import (
    WalkSettings,
    add_chains,
    class_poses,
    grid_candidates,
    midpoint_of,
    read_derived,
    roadmap_carry_poses,
    save_derived,
    via_round,
    walk_in_starts,
)
from husky_assembly_tamp.motion_planner.holding_prm.store import (
    EDGE_DECIMALS,
    NODE_DECIMALS,
    SCHEMA_VERSION,
    load_roadmap,
    query_path,
    rounded_columns,
    update_query_index,
    write_json_atomic,
)
from husky_assembly_tamp.motion_planner.ssik_ik import ARM_BASE_LINK, ARM_SLICE


# * Waypoint pairs per final-check task.
FINE_CHUNK = 12

# * Configurations per node-check task.
CHECK_CHUNK = 24

# * How closely the numpy forward kinematics on the live arm bases must agree
# * with PyBullet's own flange poses (metres). A larger gap means the base
# * pose was applied differently from what this module assumes.
FK_AGREEMENT_M = 1e-5


@dataclass
class OnlineSettings:
    """What an online plan may spend and how fine its answer is.

    Attributes:
        roadmap_dir (str): the roadmap's folder, ``<root>/prm/<run>``.
        workers (int): 1 = everything in the caller's process, on the
            caller's planner; more = that many worker processes, each
            opening the scene from files.
        tolerances_mm (List[float]): roadmap tolerances to try, smallest
            first: a route within the first tolerance that has one is used.
        tube_rounds (int): when no route survives, rounds of new nodes drawn
            around the best remaining route before trying again.
        tube_draws (int): driver draws per tube round.
        position_res_m (float): longest bar move between waypoints of the
            returned path.
        rotation_res_rad (float): largest bar turn between them.
        k_endpoint (int): neighbours each new goal (or start) node tries to join.
        radius_endpoint_deg (float): largest joint move of such an edge.
        scene_cache_dir (str): where scene files for the workers go; default
            ``<roadmap_dir>/scenes``.
        use_pretracked (bool): load the roadmap's overnight tracking file when
            it has one. False tracks every edge a route needs now (for
            comparison runs: it shows what the overnight tracking saves).
        save_query (bool): save the plan as a query of the roadmap
            (``<roadmap_dir>/queries/<query_tag>.json.gz``, with the scene
            files it needs), so the roadmap page can show it in 3D.
        query_tag (str): the query's name; an existing one is replaced.
        walk_in (bool): when no carry pose of the roadmap connects, walk the goal postures to loading poses of the
            same three classes and use those they reach as starts (``walk_in.walk_in_starts``). Poses saved by an
            earlier plan (``save_derived``) are then used from the start.
        via_rounds (int): after that, rounds of waypoints grown from the goal out of the structure
            (``walk_in.via_round``), each followed by a search; needs ``walk_in``.
        walk_grid_step_m (float): spacing of the grid of loading poses ``walk_in`` tries in each class.
        last_tolerances_mm (List[float]): wider tolerances tried only after the fallbacks found nothing.
        save_derived (bool): save a loading pose that gave a path next to the roadmap, for the next plan.
        verbose (bool): print progress.
    """

    roadmap_dir: str
    workers: int = 1
    tolerances_mm: List[float] = field(default_factory=lambda: [10.0, 20.0, 30.0])
    tube_rounds: int = 1
    tube_draws: int = 400
    position_res_m: float = 0.002
    rotation_res_rad: float = 0.002
    k_endpoint: int = 10
    radius_endpoint_deg: float = 90.0
    scene_cache_dir: str = ""
    use_pretracked: bool = True
    save_query: bool = True
    query_tag: str = "online"
    walk_in: bool = False
    via_rounds: int = 0
    walk_grid_step_m: float = 0.3
    last_tolerances_mm: List[float] = field(default_factory=lambda: [45.0, 60.0])
    save_derived: bool = False
    verbose: bool = True


class RoadmapMismatch(ValueError):
    """The roadmap was built for something else than what the scene carries."""


# -----------------------------------------------------------------------------
# * Loading, and making sure the roadmap fits the scene
# -----------------------------------------------------------------------------


def load_roadmap_dir(roadmap_dir: str) -> Roadmap:
    """Read a roadmap from its folder.

    Args:
        roadmap_dir (str): ``<root>/prm/<run>``.

    Returns:
        Roadmap: the roadmap.

    Raises:
        FileNotFoundError: when there is no roadmap there, saying how to build one.
    """
    path = os.path.join(roadmap_dir, "roadmap.json.gz")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No precomputed roadmap at {roadmap_dir}. Build one for this bar first:\n"
            "  scripts/benchmark_cdfm.py build-cells <problem folder> --bars <bar>\n"
            "  scripts/prm_pilot.py build --name <problem>__<bar> --problem-dir <problem folder> --bar <bar> "
            "--starts-file carry/carry_v1/starts.json\n"
            "  scripts/prm_pilot.py track --name <problem>__<bar> --max-score-mm 15"
        )
    prm_dir, run_name = os.path.split(os.path.normpath(roadmap_dir))
    return load_roadmap(os.path.dirname(prm_dir), run_name)


def pretracked_file(roadmap_dir: str) -> Optional[str]:
    """The overnight tracking file of a roadmap with the largest score, or None when there is none."""
    best, best_score = None, -1.0
    for path in glob.glob(os.path.join(roadmap_dir, "tracked_*mm.json.gz")):
        match = re.search(r"tracked_([0-9.]+)mm", os.path.basename(path))
        if match and float(match.group(1)) > best_score:
            best, best_score = path, float(match.group(1))
    return best


def check_roadmap_fits(roadmap: Roadmap, start_state, bar_rb_name: str,
                       grasp_bar_from_right: Optional[np.ndarray]) -> None:
    """Stop when the roadmap was built for another bar, grasp or load than the scene has.

    Args:
        roadmap (Roadmap): the precomputed roadmap.
        start_state: the live start state.
        bar_rb_name (str): the carried bar's rigid-body name.
        grasp_bar_from_right (Optional[np.ndarray]): the right grasp the
            caller derived from its goal frames (4x4), when it has one.

    Raises:
        RoadmapMismatch: listing every difference.
    """
    expected = (roadmap.meta.get("export") or {}).get("signature")
    if not expected:
        raise RoadmapMismatch(
            "This roadmap carries no record of what it was built for (it was built from a benchmark "
            "library). Build one from the exported problem with 'prm_pilot.py build --problem-dir'."
        )
    found = robot_side_signature(start_state, bar_rb_name)
    # * The monitor names the process's bodies under a prefix (process/bars/B20); compare them as the export names them.
    expected_bar = expected.get("bar") or ""
    if bar_rb_name != expected_bar and bar_rb_name.endswith("/" + expected_bar):
        found = without_prefix(found, bar_rb_name[:-len(expected_bar)])
    differences = signature_differences(with_tool_names_of(expected, found), found)
    if grasp_bar_from_right is not None:
        stored = roadmap.model.grasp["right"]
        distance, angle = pose_gap(stored, grasp_bar_from_right)
        if distance > 1e-4 or angle > 1e-4:
            differences.append(f"right grasp differs by {distance * 1000:.2f} mm / {np.degrees(angle):.3f} deg")
    if differences:
        raise RoadmapMismatch(
            "The roadmap does not match what the robot carries in this scene, so its precomputed "
            "collision verdicts do not apply:\n  " + "\n  ".join(differences)
            + "\nRebuild the roadmap for this bar and grasp (build-cells, then prm_pilot.py build and track)."
        )


def live_model(roadmap: Roadmap, robot_puid: int, arm_joints: Sequence[int]) -> HoldingModel:
    """The roadmap's holding model, moved to where the robot really stands.

    Only the arm bases change: they are read from the live scene. Grasps,
    limits and home stay the roadmap's. The numpy forward kinematics on the
    new bases is then compared with PyBullet's own flange poses.

    Args:
        roadmap (Roadmap): the precomputed roadmap.
        robot_puid (int): the robot's PyBullet body.
        arm_joints (Sequence[int]): its twelve arm joints.

    Returns:
        HoldingModel: the model in the world of the live scene.

    Raises:
        RuntimeError: when the two forward kinematics disagree.
    """
    base = roadmap.model
    world_from_base = {
        arm: np.asarray(tform_from_pose(get_link_pose(robot_puid, link_from_name(robot_puid, ARM_BASE_LINK[arm]))),
                        dtype=float) for arm in ARMS
    }
    model = HoldingModel(world_from_base=world_from_base, grasp=base.grasp, grasp_inv=base.grasp_inv,
                         limits=base.limits, home_conf12=base.home_conf12, joint_names_12=base.joint_names_12)
    # The live joints are whatever the start state set; compare the flanges there.
    current = np.asarray(get_joint_positions(robot_puid, list(arm_joints)), dtype=float)
    for arm, link_name in (("left", "left_ur_arm_tool0"), ("right", "right_ur_arm_tool0")):
        expected = np.asarray(tform_from_pose(get_link_pose(robot_puid, link_from_name(robot_puid, link_name))))
        gap = float(np.abs(tool0_world(model, arm, current[ARM_SLICE[arm]]) - expected).max())
        if gap > FK_AGREEMENT_M:
            raise RuntimeError(
                f"The {arm} flange is {gap * 1000:.3f} mm away from where the roadmap's kinematics puts it "
                "at the live base pose. The base pose or the robot description differs from the one the "
                "solvers were generated from; do not plan on this roadmap until that is understood."
            )
    return model


# -----------------------------------------------------------------------------
# * Workers
# -----------------------------------------------------------------------------


def open_pool(planner, start_state, joint_names_12: Sequence[str], model: HoldingModel, roadmap: Roadmap,
              settings: OnlineSettings) -> Tuple[WorkerPool, Dict[str, Any]]:
    """Workers that hold the real scene: this process alone, or several processes opening it from files.

    Args:
        planner: the caller's planner (cell loaded, start state set).
        start_state: the live start state.
        joint_names_12 (Sequence[str]): the twelve arm joints.
        model (HoldingModel): the live holding model.
        roadmap (Roadmap): the roadmap (edge settings, problem).
        settings (OnlineSettings): how many workers, where scene files go.

    Returns:
        Tuple[WorkerPool, Dict[str, Any]]: the pool, and what the scene holds
        (counts, and the scene key when files were written).
    """
    edge_settings = roadmap.config["edge_settings"]
    cell, small = build_scene_cell(planner.client.robot_cell, start_state)
    facts = {"bodies": len(small.rigid_body_states), "tools": len(small.tool_states)}
    # The scene files: what worker processes open, and what the roadmap page
    # draws later (so they are also written when the plan is to be saved).
    if settings.workers > 1 or settings.save_query:
        cache_dir = settings.scene_cache_dir or os.path.join(settings.roadmap_dir, "scenes")
        cell_path, state_path, facts["scene_key"] = save_scene(cache_dir, cell, small,
                                                               roadmap.meta["problem"]["bar_rb_name"])
        facts["scene_files"] = {"cell": cell_path, "state": state_path}
    if settings.workers <= 1:
        checks = make_collision_fns(planner, start_state, joint_names_12)
        tools = WorkerTools(model, checks["full"], checks["scene_only"])
        return WorkerPool.in_process(tools, edge_settings, settings.verbose), facts
    pool = WorkerPool.from_scene_files(facts["scene_files"]["cell"], facts["scene_files"]["state"],
                                       roadmap.meta["problem"], settings.workers, edge_settings, settings.verbose)
    return pool, facts


# -----------------------------------------------------------------------------
# * Saving a plan as a query of the roadmap, for the roadmap page
# -----------------------------------------------------------------------------


def save_online_query(settings: OnlineSettings, roadmap: Roadmap, model: HoldingModel, tracker: Tracker,
                      goal_node_of: Dict[int, Dict[str, Any]], start_ids: Sequence[int],
                      start_anchor: Dict[int, Dict[str, Any]], n_new_endpoint_nodes: int,
                      answer: Optional[Tuple[Sequence[int], Sequence[Sequence[float]], float]],
                      info: Dict[str, Any], world_from_bar_goal: np.ndarray) -> str:
    """Write the plan as a query file in the roadmap's folder, in the shape the roadmap page reads.

    The page joins the roadmap's own tables with the query's new nodes (goal
    postures, tube nodes), its new edges, the final verdict of every edge
    (scene-blocked edges show as blocked), each node's gap to the goal set,
    and the plan's exact path, which it walks with its stepper.

    Args:
        settings (OnlineSettings): the plan's settings (the tag, the roadmap folder).
        roadmap (Roadmap): the roadmap.
        model (HoldingModel): the live holding model (unused for geometry; kept for the signature).
        tracker (Tracker): the search's graph, with every verdict.
        goal_node_of (Dict[int, Dict[str, Any]]): node id -> the goal posture's record.
        start_ids (Sequence[int]): the start nodes.
        start_anchor (Dict[int, Dict[str, Any]]): node id -> the carry-pose anchor it is.
        n_new_endpoint_nodes (int): how many nodes after the roadmap's are goal
            or start postures (the rest are tube nodes).
        answer: ``(route, exact path, tolerance)`` when a path was found, else None.
        info (Dict[str, Any]): the plan's info so far (timings, counters, scene).
        world_from_bar_goal (np.ndarray): the goal pose, 4x4.

    Returns:
        str: the query file written.
    """
    prm_dir, run_name = os.path.split(os.path.normpath(settings.roadmap_dir))
    root = os.path.dirname(prm_dir)
    tag = settings.query_tag
    ratio = float(roadmap.config["rot_deg_per_mm"])
    confs = tracker.confs
    table = tracker.table

    # * The nodes the plan added: goal and start postures, then tube nodes. Their
    # * geometry is taken in the roadmap's own frame (base at the origin), like
    # * the roadmap's nodes and the scene file, not in the world.
    new_nodes: Dict[str, list] = {name: [] for name in NODE_COLUMNS}
    for node in range(roadmap.n_nodes, len(confs)):
        record = {"conf": confs[node].tolist(), **geometry_to_lists(node_geometry(roadmap.model, confs[node])),
                  "kind": "endpoint" if node < roadmap.n_nodes + n_new_endpoint_nodes else "tube",
                  "mode": "", "driver": "", "anchor": -1, "draw": -1}
        for name in NODE_COLUMNS:
            new_nodes[name].append(record[name])

    # * Which nodes stand for the goal and for the starts.
    endpoint: Dict[str, list] = {name: [] for name in (
        "id", "role", "start_index", "carry", "difficulty", "pair", "turns", "library_conf", "reused")}

    def endpoint_row(node: int, role: str, start_index: int, carry: str, difficulty: str, pair: int,
                     turns: str) -> None:
        """One row of the endpoint table."""
        for name, value in zip(endpoint, (node, role, start_index, carry, difficulty, pair, turns, False,
                                          node < roadmap.n_nodes)):
            endpoint[name].append(value)

    for node, found in sorted(goal_node_of.items()):
        endpoint_row(node, "goal", -1, "", "", int(found.get("pair", -1)), "+".join(found.get("turns", [])))
    for node in start_ids:
        anchor = start_anchor.get(node, {})
        endpoint_row(node, "start", int(anchor.get("start_index", -1)), str(anchor.get("carry", "")),
                     str(anchor.get("label", "")), int(anchor.get("pair_index", -1)), str(anchor.get("turn", "")))

    # * The edges the plan added, in the roadmap's column layout.
    new_edges: Dict[str, list] = {name: [] for name in EDGE_COLUMNS}
    for row in range(roadmap.n_edges, len(table.i)):
        a, b = int(table.i[row]), int(table.j[row])
        record = {
            "i": a, "j": b, "len_l2": float(table.len_l2[row]),
            "len_linf": float(np.max(np.abs(confs[a] - confs[b]))), "n_checks_planned": 0,
            "checked": bool(table.checked[row]), "free": bool(table.free[row]), "t_collide": None,
            "viol_pos_max": float(table.viol_pos_max[row]), "viol_rot_max": float(table.viol_rot_max[row]),
            "t_viol_max": None, "n_viol_points": 0,
        }
        for name in EDGE_COLUMNS:
            new_edges[name].append(record[name])

    goal_ids = sorted(goal_node_of)
    gap, parent = minimax_to_goals(len(confs), table, table.free, goal_ids)
    best_start = min(start_ids, key=lambda node: gap[node]) if start_ids else None
    best_gap = float(gap[best_start]) if best_start is not None and np.isfinite(gap[best_start]) else None

    def route_report(nodes: Sequence[int]) -> Dict[str, Any]:
        """What the page shows about a route: its nodes and its worst edge."""
        scores = [float(table.score[tracker.row_of[(min(a, b), max(a, b))]]) for a, b in zip(nodes[:-1], nodes[1:])]
        return {"nodes": [int(n) for n in nodes], "max_score_mm": max(scores) if scores else 0.0}

    plans = []
    minimax_route = None
    if answer is not None:
        route, path, tolerance = answer
        plans.append({"name": f"exact path (route within {tolerance:g} mm)", "tolerance_mm": float(tolerance),
                      "nodes": [int(n) for n in route], "n_points": len(path),
                      "confs": [[round(float(v), 6) for v in conf] for conf in path]})
        minimax_route = route_report(route)
    elif best_gap is not None:
        minimax_route = route_report(route_from(best_start, parent))

    scene = dict(info.get("scene") or {})
    files = scene.pop("scene_files", None)
    if files:
        folder = os.path.basename(os.path.dirname(files["state"]))
        state_name = os.path.splitext(os.path.basename(files["state"]))[0]
        # Paths as the dashboard folder sees them; the glb is written by "prm_pilot.py export".
        scene["files"] = {"cell": f"{folder}/cell.json", "state": f"{folder}/{state_name}.json"}
        scene["asset"] = f"../scenes/{folder}/{state_name}.glb"

    record = {
        "schema_version": SCHEMA_VERSION, "tag": tag, "run_name": run_name, "bar_id": roadmap.meta["bar_id"],
        "library": roadmap.meta.get("library", ""), "goal_index": None, "kind": "online",
        "settings": asdict(settings), "rot_deg_per_mm": ratio, "created": time(),
        "goal": {"world_from_bar": pose_from_matrix(world_from_bar_goal), "n_library_pairs": 0,
                 "n_library_pairs_found": 0, "n_goal_nodes": len(goal_ids)},
        "n_roadmap_nodes": roadmap.n_nodes, "n_start_nodes": len(start_ids),
        "endpoint": endpoint,
        "endpoint_nodes": rounded_columns(new_nodes, NODE_DECIMALS),
        "endpoint_edges": rounded_columns(new_edges, EDGE_DECIMALS),
        "edge_checked": table.checked.tolist(), "edge_free": table.free.tolist(),
        "node_gap_mm": [None if not np.isfinite(g) else float(g) for g in gap], "node_parent": parent.tolist(),
        "headline": {"best_gap_mm": best_gap, "best_start_index": None, "minimax_route": minimax_route,
                     "shortest_at_best_gap": None},
        "tolerances": [], "starts": [], "by_difficulty": {},
        "plans": plans, "scene": scene,
        "failure_reason": info.get("failure_reason"), "timings": info.get("timings"),
        "edges_looked_at": info.get("edges_looked_at"), "edge_verdicts": info.get("edge_verdicts"),
        "scene_blocked_edges": info.get("scene_blocked_edges"),
    }
    path = query_path(root, run_name, tag)
    write_json_atomic(path, record)
    update_query_index(root, run_name, {
        "tag": tag, "kind": "online", "goal_index": None, "starts": "carry poses", "n_starts": len(start_ids),
        "file": f"{tag}.json.gz", "best_gap_mm": best_gap, "best_start_index": None,
        "starts_connected": {}, "found": answer is not None,
    })
    return path


# -----------------------------------------------------------------------------
# * The plan
# -----------------------------------------------------------------------------


def check_confs(pool: WorkerPool, confs: Sequence[Sequence[float]], which: str,
                counters: Dict[str, float]) -> List[bool]:
    """Collision verdicts of some configurations ("scene" or "full" check), in the workers."""
    tasks = [{"chunk": n, "confs": [list(map(float, c)) for c in piece], "which": which}
             for n, piece in enumerate(chunked(list(confs), CHECK_CHUNK))]
    verdicts: List[bool] = []
    for result in pool.run(task_check, tasks, f"{which} check"):
        add_counters(counters, result["counters"])
        verdicts.extend(result["collides"])
    return verdicts


def final_check(pool: WorkerPool, confs: Sequence[Sequence[float]], spans: Sequence[Tuple[int, int, int]],
                settings: OnlineSettings, counters: Dict[str, float]) -> Tuple[Optional[List[List[float]]], Optional[int], str]:
    """Walk a route at the caller's step and collision-check every fine configuration, in parallel.

    Args:
        pool (WorkerPool): workers holding the real scene.
        confs (Sequence[Sequence[float]]): the route's exact waypoints.
        spans (Sequence[Tuple[int, int, int]]): per edge of the route, its
            table row and its first and last waypoint index.
        settings (OnlineSettings): the fine step.
        counters (Dict[str, float]): tally.

    Returns:
        Tuple: the fine path (None on failure), the row of the edge to drop
        (None on success) and the reason.
    """
    tasks = []
    for number, first in enumerate(range(0, len(confs) - 1, FINE_CHUNK)):
        tasks.append({"chunk": number, "first_index": first, "confs": [list(c) for c in confs[first:first + FINE_CHUNK + 1]],
                      "position_res_m": settings.position_res_m, "rotation_res_rad": settings.rotation_res_rad})
    path: List[List[float]] = [list(map(float, confs[0]))]
    for result in pool.run(task_fine, tasks, "final check"):
        add_counters(counters, result["counters"])
        for offset, step in enumerate(result["steps"]):
            if not step["free"]:
                index = result["first_index"] + offset
                # The edge whose waypoints this step lies between.
                row = next((r for r, first, last in spans if first <= index < last), spans[-1][0])
                return None, int(row), step["reason"]
            path.extend(step["confs"])
    return path, None, "ok"


def plan_on_roadmap(planner, start_state, joint_names_12: Sequence[str], active_bar_id: str,
                    world_from_bar_goal: np.ndarray, settings: OnlineSettings, *,
                    robot_puid: int, arm_joints: Sequence[int],
                    start_conf: Optional[Sequence[float]] = None, goal_conf: Optional[Sequence[float]] = None,
                    grasp_bar_from_right: Optional[np.ndarray] = None,
                    goal_accept: Optional[Callable[[np.ndarray], str]] = None,
                    goal_turn_reference: Optional[Sequence[float]] = None,
                    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
    """Plan the bar carry on a precomputed roadmap, in the live scene.

    Args:
        planner: the caller's planner, with the cell loaded and ``start_state`` set.
        start_state: the live start state (base pose, attachments, the scene).
        joint_names_12 (Sequence[str]): the twelve arm joints.
        active_bar_id (str): the carried bar's rigid-body name.
        world_from_bar_goal (np.ndarray): where the bar must end up, 4x4, world.
        settings (OnlineSettings): roadmap folder, workers, tolerances, step.
        robot_puid (int): the robot's PyBullet body.
        arm_joints (Sequence[int]): its twelve arm joints.
        start_conf (Optional[Sequence[float]]): start from exactly this
            configuration; None = let the search pick among the roadmap's
            carry poses.
        goal_conf (Optional[Sequence[float]]): end on exactly this
            configuration; None = any collision-free posture at the goal pose.
        grasp_bar_from_right (Optional[np.ndarray]): the caller's right grasp, to compare.
        goal_accept (Optional[Callable[[np.ndarray], str]]): asked about the
            goal posture of every route found, before its fine check: it
            returns "" to take the posture, or why not (the posture is then
            dropped and the search goes on; ``info["goals_rejected"]`` lists
            them). It may use the caller's planner; the state is put back after.
        goal_turn_reference (Optional[Sequence[float]]): twelve joints (e.g. the authored end joints). When the
            first search finds nothing and ``settings.walk_in`` is on, every goal posture is offered a second time
            where they differ: on the whole turns nearest these instead of nearest home. The same arm posture a
            whole turn apart can decide whether the next movement stays inside the joint limits.

    Returns:
        Tuple[Optional[List[np.ndarray]], Dict[str, Any]]: the path (twelve
        joint values per waypoint, start first; None when none was found) and
        what happened: ``failure_reason`` when it failed, the route, the
        tolerance used, counters and timings per phase.

    Raises:
        FileNotFoundError: no roadmap at ``settings.roadmap_dir``.
        RoadmapMismatch: the roadmap was built for another bar, grasp or load.
    """
    wall = perf_counter()
    say = print if settings.verbose else (lambda *args, **kwargs: None)
    timings: Dict[str, float] = {}
    counters: Dict[str, Dict[str, float]] = {"endpoints": {}, "endpoint_edges": {}, "nodes": {}, "final": {},
                                             "tube_sample": {}, "tube_edges": {}}
    info: Dict[str, Any] = {"roadmap": settings.roadmap_dir, "workers": settings.workers, "timings": timings,
                            "counters": counters}

    # * 1. The roadmap, checked against what the scene carries, at the real base pose.
    phase = perf_counter()
    roadmap = load_roadmap_dir(settings.roadmap_dir)
    check_roadmap_fits(roadmap, start_state, active_bar_id, grasp_bar_from_right)
    model = live_model(roadmap, robot_puid, arm_joints)
    home = model.home_conf12.tolist()
    ratio = float(roadmap.config["rot_deg_per_mm"])
    timings["load_s"] = perf_counter() - phase
    say(f"[prm online] roadmap {os.path.basename(os.path.normpath(settings.roadmap_dir))}: {roadmap.n_nodes} nodes, "
        f"{roadmap.n_edges} edges; {settings.workers} worker(s)")

    phase = perf_counter()
    pool, scene_facts = open_pool(planner, start_state, joint_names_12, model, roadmap, settings)
    info["scene"] = scene_facts
    try:
        # Start every worker now, so the phases below are timed without it.
        timings["worker_start_s"] = pool.startup_s + pool.warm_up()
        timings["scene_setup_s"] = perf_counter() - phase - timings["worker_start_s"]

        # * 2. Goal nodes: every collision-free posture at the goal pose (full check, in the scene).
        phase = perf_counter()
        if goal_conf is not None:
            hit = check_confs(pool, [goal_conf], "full", counters["endpoints"])[0]
            goal_found = [] if hit else [{"conf": [float(v) for v in goal_conf], "pair": -1, "turns": ["given"]}]
            goal_twins = []
        else:
            turn_references = [["home", home]]
            if goal_turn_reference is not None:
                turn_references.append(["given", [float(v) for v in goal_turn_reference]])
            result = pool.run(task_expand, [{"chunk": 0, "world_from_bar": pose_from_matrix(world_from_bar_goal),
                                             "turn_references": turn_references}], "goal")[0]
            add_counters(counters["endpoints"], result["counters"])
            # * The postures on the given turns only wait for the fallbacks: the first search is the one a caller
            # * without a reference gets.
            goal_found = [found for found in result["confs"] if "home" in found["turns"]]
            goal_twins = [found for found in result["confs"] if "home" not in found["turns"]]
        goal_confs = [np.asarray(found["conf"], dtype=float) for found in goal_found]
        timings["goal_s"] = perf_counter() - phase
        if not goal_confs and not goal_twins:
            info["failure_reason"] = "prm_goal_unholdable"
            return None, info

        # * 3. Start nodes: the roadmap's carry poses that are free in this scene, or the given start.
        phase = perf_counter()
        roadmap_confs = roadmap.confs()
        new_confs: List[np.ndarray] = []
        new_roles: List[str] = []

        def node_for(conf: np.ndarray, role: str) -> int:
            """The roadmap node at this configuration, or a new node for it."""
            distances = np.max(np.abs(roadmap_confs - conf[None, :]), axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] < 1e-6:
                return nearest
            new_confs.append(conf)
            new_roles.append(role)
            return roadmap.n_nodes + len(new_confs) - 1

        goal_node_of: Dict[int, Dict[str, Any]] = {}
        for found, conf in zip(goal_found, goal_confs):
            goal_node_of.setdefault(node_for(conf, "goal"), found)
        goal_ids = sorted(goal_node_of)
        start_anchor = {}
        if start_conf is not None:
            start_ids = [node_for(np.asarray(start_conf, dtype=float), "start")]
        else:
            candidates = {int(a["node"]): a for a in roadmap.anchors if a.get("role") == "start" and a.get("node", -1) >= 0}
            ids = sorted(candidates)
            blocked = check_confs(pool, roadmap_confs[ids], "scene", counters["nodes"]) if ids else []
            start_ids = [node for node, hit in zip(ids, blocked) if not hit]
            start_anchor = candidates
            info["starts"] = {"carry_nodes": len(ids), "free_in_scene": len(start_ids)}
        timings["starts_s"] = perf_counter() - phase
        if not start_ids:
            info["failure_reason"] = "prm_no_start_free"
            return None, info

        # * Join the new nodes to their nearest roadmap nodes (holding error measured now, tracking later).
        phase = perf_counter()
        all_confs = np.vstack([roadmap_confs] + [np.asarray(new_confs).reshape(-1, 12)])
        new_ids = list(range(roadmap.n_nodes, len(all_confs)))
        endpoint_edges: Dict[str, list] = {name: [] for name in EDGE_COLUMNS}
        if new_ids:
            pairs = candidate_pairs(all_confs, settings.k_endpoint, float(np.radians(settings.radius_endpoint_deg)),
                                    query_ids=new_ids)
            for result in pool.run(task_edges, edge_tasks(all_confs, pairs, 64, False), "goal edges"):
                add_counters(counters["endpoint_edges"], result["counters"])
                for record in result["edges"]:
                    for name in EDGE_COLUMNS:
                        endpoint_edges[name].append(record[name])
        table = EdgeTable.from_columns([roadmap.edges, endpoint_edges], ratio)
        graph = QueryGraph(model=model, build=roadmap.config, query={"rot_deg_per_mm": ratio}, confs=all_confs,
                           table=table, n_roadmap_edges=roadmap.n_edges, goal_ids=goal_ids)
        refine_settings = RefineSettings(run_name=os.path.basename(os.path.normpath(settings.roadmap_dir)),
                                         workers=settings.workers, tube_target_mm=min(settings.tolerances_mm))
        tracker = Tracker(graph, refine_settings, pool, scene_check=True)
        pretracked = pretracked_file(settings.roadmap_dir) if settings.use_pretracked else None
        info["pretracked_edges"] = tracker.load_pretracked(pretracked) if pretracked else 0
        timings["connect_s"] = perf_counter() - phase
        say(f"  {len(goal_ids)} goal nodes ({len(new_ids)} new), {len(start_ids)} start nodes free in the scene, "
            f"{len(endpoint_edges['i'])} new edges, {info['pretracked_edges']} edges tracked overnight")

        # * 4-6. Lazy search, tolerance by tolerance; the winner gets the fine check.
        phase = perf_counter()
        dropped_final: List[Dict[str, Any]] = []
        goals_accepted: set = set()
        goals_rejected: List[Dict[str, Any]] = []
        info["goals_rejected"] = goals_rejected
        tube_log: List[Dict[str, Any]] = []

        tolerances = sorted(settings.tolerances_mm)

        def search() -> Optional[tuple]:
            """The first route, smallest tolerance first, whose end posture is taken and that passes the fine check."""
            for tolerance in tolerances:
                while True:
                    found = tracker.shortest_exact(start_ids, goal_ids, tolerance)
                    if not found:
                        break
                    route, waypoints, assembly = found
                    goal = int(route[-1])
                    if goal_accept is not None and goal not in goals_accepted:
                        # * The caller's say on this end posture (can the insert and the retreat follow from it?).
                        accept_phase = perf_counter()
                        why_not = goal_accept(np.asarray(tracker.confs[goal], dtype=float))
                        planner.set_robot_cell_state(start_state)
                        timings["goal_accept_s"] = timings.get("goal_accept_s", 0.0) + perf_counter() - accept_phase
                        if why_not:
                            goals_rejected.append({"node": goal, "reason": str(why_not)})
                            goal_ids.remove(goal)
                            goal_node_of.pop(goal, None)
                            say(f"  goal posture {goal} refused: {why_not}; {len(goal_ids)} left")
                            continue
                        goals_accepted.add(goal)
                    fine_phase = perf_counter()
                    path, bad_row, reason = final_check(pool, waypoints, assembly["spans"], settings, counters["final"])
                    timings["final_check_s"] = timings.get("final_check_s", 0.0) + perf_counter() - fine_phase
                    if path is None:
                        # A problem between the 1 cm waypoints: drop that edge and look again.
                        tracker.table.free[bad_row] = False
                        dropped_final.append({"row": bad_row, "reason": reason})
                        continue
                    return route, waypoints, assembly, path, tolerance
            return None

        def search_with_tubes() -> Optional[tuple]:
            """Search; while nothing survives, draw new nodes around the best route that is left and search again."""
            found = search()
            for _round in range(int(settings.tube_rounds)):
                if found is not None:
                    break
                gap, parent = minimax_to_goals(len(tracker.confs), tracker.table, tracker.table.free, goal_ids)
                reachable = [node for node in start_ids if np.isfinite(gap[node])]
                if not reachable:
                    break
                best = min(reachable, key=lambda node: gap[node])
                made = tube_round(tracker, [route_from(best, parent)], len(tube_log) + 1, settings.tube_draws,
                                  counters)
                made.pop("confs", None)
                made["gap_before_mm"] = float(gap[best])
                tube_log.append(made)
                say(f"  no route yet; tube round {len(tube_log)} around the best one ({gap[best]:.1f} mm): "
                    f"{made['nodes']} nodes, {made['edges']} edges")
                found = search()
            return found

        # * The fallbacks' bookkeeping: loading poses walked to from the goal, and waypoints on the way out.
        walk_settings = WalkSettings(grid_step_m=settings.walk_grid_step_m, k=settings.k_endpoint,
                                     radius_deg=settings.radius_endpoint_deg)
        counters["walk"] = {}
        walk_log: Dict[str, Any] = {"stages": []}
        info["walk_in"] = walk_log
        walk_targets: Dict[int, List[np.ndarray]] = {}
        walk_ends: Dict[int, List[int]] = {}

        def walk_to(candidates: List[Dict[str, Any]], name: str) -> None:
            """Walk the goal postures to these loading poses and make starts of the ones they reach."""
            walk_phase = perf_counter()
            made = walk_in_starts(tracker, roadmap, model, goal_ids, candidates, walk_settings, counters["walk"])
            for node, note in made["starts"].items():
                if node not in start_ids:
                    start_ids.append(node)
                start_anchor.setdefault(node, note)
            for goal in made["targets"]:
                walk_targets.setdefault(goal, []).extend(made["targets"][goal])
                walk_ends.setdefault(goal, []).extend(made["chain_ends"][goal])
            timings["walk_in_s"] = timings.get("walk_in_s", 0.0) + perf_counter() - walk_phase
            walk_log["stages"].append({"stage": name, "poses": len(candidates), **made["tally"]})
            say(f"  walk-in ({name}): {len(candidates)} poses x {len(goal_ids)} goal postures: "
                f"{made['tally']['arrived_free']} arrive free, {made['tally']['free_all_the_way']} free all the way; "
                f"{made['tally']['starts']} starts added")

        def add_goal_twins() -> None:
            """Offer the goal postures on the caller's whole turns too, as goal nodes joined to the roadmap (once)."""
            lone = [(None, [np.asarray(found["conf"], dtype=float)]) for found in goal_twins]
            for found, nodes in zip(goal_twins, add_chains(tracker, lone, walk_settings, counters["endpoint_edges"],
                                                           "goal turns")):
                if nodes[0] not in goal_node_of:
                    goal_node_of[nodes[0]] = found
                    goal_ids.append(nodes[0])
            if goal_twins:
                say(f"  {len(goal_twins)} goal postures also offered on the given whole turns; {len(goal_ids)} in all")
            goal_twins.clear()

        def grow_waypoints() -> Optional[tuple]:
            """Rounds of waypoints from the goal postures out of the structure, a search after each."""
            if not goal_ids:
                return None
            goal_middle = midpoint_of(model, tracker.confs[goal_ids[0]])
            frontier = [(goal, goal) for goal in goal_ids]
            frontier += [(node, goal) for goal in goal_ids for node in walk_ends.get(goal, [])]
            for round_number in range(1, int(settings.via_rounds) + 1):
                via_phase = perf_counter()
                made = via_round(tracker, model, frontier, walk_targets, goal_middle, walk_settings, counters["walk"])
                for node, note in made["starts"].items():
                    if node not in start_ids:
                        start_ids.append(node)
                    start_anchor.setdefault(node, note)
                frontier = made["frontier"]
                timings["via_s"] = timings.get("via_s", 0.0) + perf_counter() - via_phase
                walk_log["stages"].append({"stage": f"waypoints {round_number}", **made["tally"]})
                say(f"  waypoint round {round_number}: {made['tally']['waypoints']} waypoints, "
                    f"{made['tally']['starts_reached']} loading poses reached")
                found = search()
                if found is not None or not frontier:
                    return found
            return None

        # A caller that fixed the start gets no other one.
        walk_in = settings.walk_in and start_conf is None
        derived = class_poses(roadmap, read_derived(settings.roadmap_dir), "derived") if walk_in else []
        if derived:
            # * This bar needed other loading poses before: they are part of its carry poses now.
            add_goal_twins()
            walk_to(derived, "saved poses")
            answer = search() or grow_waypoints()
        else:
            answer = search_with_tubes()
        if answer is None and walk_in and (goal_ids or goal_twins):
            add_goal_twins()
            walk_to(roadmap_carry_poses(roadmap), "roadmap poses")
            answer = search()
            if answer is None:
                walk_to(grid_candidates(roadmap, walk_settings), "class grid")
                answer = search() or grow_waypoints()
            if answer is None:
                # * Last resort: routes whose roadmap edges stray further from holding (the tracker still makes
                # * every waypoint exact), and new nodes around the best of them.
                tolerances.extend(sorted(t for t in settings.last_tolerances_mm if t > tolerances[-1]))
                answer = search_with_tubes()
        timings["search_s"] = (perf_counter() - phase - timings.get("final_check_s", 0.0)
                               - timings.get("goal_accept_s", 0.0) - timings.get("walk_in_s", 0.0)
                               - timings.get("via_s", 0.0))

        reasons: Dict[str, int] = {}
        # Only edges a route asked about in this plan; the night's file holds thousands more.
        for row in sorted(tracker.asked_rows):
            reason = tracker.records[row]["reason"]
            reasons[reason] = reasons.get(reason, 0) + 1
        counters["track"] = tracker.counters
        info.update({"edges_looked_at": len(tracker.asked_rows),
                     "edge_verdicts": reasons, "scene_blocked_edges": len(tracker.scene_blocked),
                     "dropped_in_final_check": dropped_final, "tube": tube_log,
                     "retracks": tracker.retracks})
        if answer is None:
            gap, _parent = minimax_to_goals(len(tracker.confs), tracker.table, tracker.table.free, goal_ids)
            finite = [float(gap[node]) for node in start_ids if np.isfinite(gap[node])]
            info["failure_reason"] = "prm_no_route" if goal_ids else "prm_goals_rejected"
            info["best_gap_left_mm"] = min(finite) if finite else None
            timings["total_s"] = perf_counter() - wall
            if settings.save_query:
                info["query_file"] = save_online_query(settings, roadmap, model, tracker, goal_node_of, start_ids,
                                                       start_anchor, len(new_ids), None, info, world_from_bar_goal)
            return None, info

        route, waypoints, assembly, path, tolerance = answer
        anchor = start_anchor.get(route[0], {})
        info.update({
            "tolerance_mm": tolerance, "route": [int(n) for n in route],
            "route_scores_mm": [float(tracker.table.score[tracker.row_of[(min(a, b), max(a, b))]])
                                for a, b in zip(route[:-1], route[1:])],
            "start": {"node": int(route[0]), "carry": anchor.get("carry"), "label": anchor.get("label"),
                      "source": anchor.get("source", "roadmap"), "midpoint": anchor.get("midpoint"),
                      "roll_deg": anchor.get("roll_deg"), "free_walk": anchor.get("free_walk")},
            "n_waypoints_1cm": len(waypoints), "n_waypoints": len(path), "joins": assembly["joins"],
            "path_poses": continuous_poses(
                [pose_from_matrix(bar_from_arm(model, "left", np.asarray(conf)[:6])) for conf in path]),
        })
        info["start"]["found_after"] = walk_log["stages"][-1]["stage"] if walk_log["stages"] else "carry poses"
        if settings.save_derived and anchor.get("source") == "grid":
            walk_log["saved"] = save_derived(settings.roadmap_dir, [anchor])
        timings["total_s"] = perf_counter() - wall
        if settings.save_query:
            info["query_file"] = save_online_query(settings, roadmap, model, tracker, goal_node_of, start_ids,
                                                   start_anchor, len(new_ids), (route, path, tolerance), info,
                                                   world_from_bar_goal)
        return [np.asarray(conf, dtype=float) for conf in path], info
    finally:
        pool.close()
