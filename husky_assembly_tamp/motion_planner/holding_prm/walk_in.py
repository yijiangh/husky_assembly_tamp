"""What the online plan tries when no precomputed carry pose connects to the goal.

Both fallbacks walk the bar on straight lines from a goal posture, both arms following step by step, so whatever
they reach lies on that posture's own arm posture and joint turns:

    1. Loading poses walked to (:func:`walk_in_starts`). Each goal posture is walked to the roadmap's carry poses
       and to a grid of other poses of the same three classes (``carry.CARRY_CLASSES``). A pose it arrives on
       collision-free is a start; the walk's configurations join the graph as a chain of nodes from the goal to it.
    2. Waypoints on the way out (:func:`via_round`). Where every such walk is blocked, the free part of the best
       walks (toward the loading poses, straight slides, turns on the spot) is kept as a chain, and the next round
       walks on from where it ends.

Chain nodes are joined to their nearest roadmap nodes, so the roadmap can carry a route on from any of them.
Loading poses found here are saved next to the roadmap (``derived_starts.json``) and used from the start next time.
"""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from time import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from husky_assembly_tamp.motion_planner.holding_prm.carry import (
    CARRY_CLASSES,
    axis_rotation,
    candidate_grid,
    carry_bar_pose,
    pair_margins,
)
from husky_assembly_tamp.motion_planner.holding_prm.holding import (
    HoldingModel,
    bar_from_arm,
    grasp_midpoint_offset,
    invert_transform,
    linf,
    pose_from_matrix,
    pose_gap,
)
from husky_assembly_tamp.motion_planner.holding_prm.parallel import WorkerPool, task_edges, task_walk
from husky_assembly_tamp.motion_planner.holding_prm.pilot import chunked, edge_tasks
from husky_assembly_tamp.motion_planner.holding_prm.refine import Tracker
from husky_assembly_tamp.motion_planner.holding_prm.roadmap import Roadmap
from husky_assembly_tamp.motion_planner.holding_prm.sampling import add_counters
from husky_assembly_tamp.motion_planner.holding_prm.store import read_json, write_json_atomic

# * The file of loading poses found for one roadmap, in the starts-file layout (class, midpoint, roll).
DERIVED_FILE = "derived_starts.json"

# * Poses per walking task: small, so a goal posture's walks spread over all workers.
WALK_CHUNK = 8

# * Sets of loading poses larger than this get the coarse first pass (``WalkSettings.screen_factor``).
SCREEN_ABOVE = 64

# * Two configurations closer than this (largest joint difference, radians) are one node.
SAME_NODE_RAD = 1e-6


@dataclass
class WalkSettings:
    """How the fallbacks walk and what they keep.

    Attributes:
        grid_step_m (float): spacing of the grid of loading poses tried in each class's box.
        position_res_m (float): longest move of the grasp midpoint per step of a walk.
        rotation_res_rad (float): largest bar turn per step.
        screen_factor (float): the first pass over many loading poses (does the walk arrive?) steps this much
            coarser; what it finds is walked again at the steps above.
        per_class (int): loading poses made starts, per class and goal posture.
        targets_per_class (int): loading poses the waypoint rounds walk toward, per class and goal posture.
        drive_tries (int): loading poses per class a ``start_accept`` may refuse before the class is given up.
        node_step_deg (float): a walk's configurations become chain nodes about this far apart (largest joint).
        k (int): roadmap nodes each chain node tries to join.
        radius_deg (float): largest joint move of such an edge.
        via_reach_m (float): how far a waypoint round slides the bar in each direction.
        via_turn_deg (float): how far it turns the bar on the spot, about each world axis.
        via_min_steps (int): a blocked walk gives a waypoint only when this many steps of it are free.
        via_keep (int): waypoints kept per configuration walked from.
        via_frontier (int): configurations a round walks from (those furthest from the goal).
    """

    grid_step_m: float = 0.3
    position_res_m: float = 0.01
    rotation_res_rad: float = 0.025
    screen_factor: float = 2.0
    per_class: int = 2
    targets_per_class: int = 8
    drive_tries: int = 6
    node_step_deg: float = 8.0
    k: int = 10
    radius_deg: float = 90.0
    via_reach_m: float = 0.3
    via_turn_deg: float = 45.0
    via_min_steps: int = 5
    via_keep: int = 4
    via_frontier: int = 8


# -----------------------------------------------------------------------------
# * Loading poses to walk to
# -----------------------------------------------------------------------------


def read_derived(roadmap_dir: str) -> List[Dict[str, Any]]:
    """The loading poses saved for a roadmap by earlier plans (empty when there are none)."""
    path = os.path.join(roadmap_dir, DERIVED_FILE)
    return list(read_json(path)["starts"]) if os.path.isfile(path) else []


def save_derived(roadmap_dir: str, found: Sequence[Dict[str, Any]]) -> int:
    """Add loading poses to a roadmap's saved ones, skipping those already there.

    Args:
        roadmap_dir (str): the roadmap's folder.
        found (Sequence[Dict[str, Any]]): poses with ``carry``, ``midpoint`` and ``roll_deg``.

    Returns:
        int: how many were new.
    """
    saved = read_derived(roadmap_dir)

    def key(pose: Dict[str, Any]) -> Tuple:
        return (pose["carry"], *[round(float(v), 4) for v in pose["midpoint"]], round(float(pose["roll_deg"]), 3))

    known = {key(pose) for pose in saved}
    new = 0
    for pose in found:
        if key(pose) in known:
            continue
        known.add(key(pose))
        new += 1
        saved.append({"label": f"derived_{pose['carry']}_{sum(p['carry'] == pose['carry'] for p in saved)}",
                      "carry": pose["carry"], "midpoint": [float(v) for v in pose["midpoint"]],
                      "roll_deg": float(pose["roll_deg"]), "found": time(), "free_walk": bool(pose.get("free_walk")),
                      "drive_in": bool(pose.get("drive_in"))})
    if new:
        write_json_atomic(os.path.join(roadmap_dir, DERIVED_FILE), {"kind": "derived", "starts": saved})
    return new


def robot_from_roadmap(roadmap: Roadmap, model: HoldingModel) -> np.ndarray:
    """The transform from the roadmap's frame (robot at the origin) to the world of the live scene."""
    return model.world_from_base["left"] @ invert_transform(roadmap.model.world_from_base["left"])


def roadmap_carry_poses(roadmap: Roadmap) -> List[Dict[str, Any]]:
    """The carry poses a roadmap was built on, one per pose of its starts file, in the roadmap's frame."""
    poses: Dict[int, Dict[str, Any]] = {}
    for anchor in roadmap.anchors:
        if anchor.get("role") == "start" and anchor.get("start_index") is not None:
            poses.setdefault(int(anchor["start_index"]), {
                "carry": anchor.get("carry", ""), "label": anchor.get("label", ""), "source": "file",
                "pose": bar_from_arm(roadmap.model, "left", np.asarray(anchor["conf"], dtype=float)[:6])})
    return [poses[index] for index in sorted(poses)]


def class_poses(roadmap: Roadmap, entries: Sequence[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """Loading poses given by class, midpoint and roll, as bar poses in the roadmap's frame."""
    return [{"carry": entry["carry"], "label": entry.get("label", ""), "source": source,
             "midpoint": list(entry["midpoint"]), "roll_deg": float(entry["roll_deg"]),
             "pose": carry_bar_pose(roadmap.model, entry["carry"], entry["midpoint"],
                                    float(np.radians(entry["roll_deg"])))} for entry in entries]


# -----------------------------------------------------------------------------
# * Walking, and turning walks into chains of nodes
# -----------------------------------------------------------------------------


def run_walks(pool: WorkerPool, jobs: Sequence[Tuple[Sequence[float], np.ndarray]], check: str, keep_confs: bool,
              settings: WalkSettings, counters: Dict[str, float], label: str,
              coarser: float = 1.0) -> List[Dict[str, Any]]:
    """Walk the bar from a configuration to a pose, for many such pairs at once; the records come back in order."""
    tasks = [{"chunk": number, "jobs": [(list(map(float, conf)), pose_from_matrix(pose)) for conf, pose in piece],
              "position_res_m": settings.position_res_m * coarser,
              "rotation_res_rad": settings.rotation_res_rad * coarser,
              "check": check, "keep_confs": keep_confs} for number, piece in enumerate(chunked(list(jobs), WALK_CHUNK))]
    walks: List[Dict[str, Any]] = []
    for result in pool.run(task_walk, tasks, label):
        add_counters(counters, result["counters"])
        walks.extend(result["walks"])
    return walks


def chain_of(confs: Sequence[Sequence[float]], step_rad: float) -> List[np.ndarray]:
    """A walk's configurations thinned to chain nodes about ``step_rad`` apart; the last one is always kept."""
    kept: List[np.ndarray] = []
    last: Optional[np.ndarray] = None
    for index, conf in enumerate(confs):
        conf = np.asarray(conf, dtype=float)
        if last is None or index == len(confs) - 1 or linf(conf, last) >= step_rad:
            kept.append(conf)
            last = conf
    return kept


def add_chains(tracker: Tracker, chains: Sequence[Tuple[Optional[int], Sequence[np.ndarray]]],
               settings: WalkSettings, counters: Dict[str, float], label: str) -> List[List[int]]:
    """Add chains of configurations to the graph: each joined in a row from its first node, and to the roadmap.

    Args:
        tracker (Tracker): the graph being searched.
        chains (Sequence[Tuple[Optional[int], Sequence[np.ndarray]]]): per chain, the node it starts from (None =
            it hangs on its roadmap neighbours alone) and the configurations after it, in walking order.
        settings (WalkSettings): how many roadmap nodes each new node tries to join, and how far.
        counters (Dict[str, float]): tally of the edge measuring.
        label (str): shown in the progress lines.

    Returns:
        List[List[int]]: per chain, the node of each of its configurations.
    """
    existing = tracker.confs
    tree = cKDTree(existing)
    new_confs: List[np.ndarray] = []
    pairs = set()
    ids_per_chain: List[List[int]] = []

    def node_of(conf: np.ndarray) -> int:
        """The node at this configuration: one the graph has, one added just now, or a new one."""
        distance, nearest = tree.query(conf, p=np.inf)
        if distance < SAME_NODE_RAD:
            return int(nearest)
        for offset, other in enumerate(new_confs):
            if linf(conf, other) < SAME_NODE_RAD:
                return len(existing) + offset
        new_confs.append(conf)
        return len(existing) + len(new_confs) - 1

    for first, confs in chains:
        previous, ids = first, []
        for conf in confs:
            node = node_of(np.asarray(conf, dtype=float))
            if previous is not None and node != previous:
                pairs.add((min(int(previous), node), max(int(previous), node)))
            ids.append(node)
            previous = node
        ids_per_chain.append(ids)
    if not new_confs:
        return ids_per_chain
    # * Every new node also tries its nearest nodes of the graph as it was: the way onto the roadmap.
    radius = float(np.radians(settings.radius_deg))
    distances, neighbours = tree.query(np.asarray(new_confs), k=min(settings.k, len(existing)), p=np.inf,
                                       distance_upper_bound=radius)
    for offset, (row_d, row_n) in enumerate(zip(np.atleast_2d(distances), np.atleast_2d(neighbours))):
        for distance, other in zip(row_d, row_n):
            if np.isfinite(distance) and other < len(existing):
                pairs.add((int(other), len(existing) + offset))
    pairs = {pair for pair in pairs if pair not in tracker.row_of}
    all_confs = np.vstack([existing, np.asarray(new_confs)])
    records = []
    if pairs:
        tasks = edge_tasks(all_confs, np.array(sorted(pairs), dtype=int), 64, False)
        for result in tracker.pool.run(task_edges, tasks, label):
            add_counters(counters, result["counters"])
            records.extend(result["edges"])
    tracker.add_nodes_and_edges(np.asarray(new_confs), records)
    return ids_per_chain


def midpoint_of(model: HoldingModel, conf: Sequence[float]) -> np.ndarray:
    """Where the point halfway between the grasps is at a configuration (world)."""
    return (bar_from_arm(model, "left", np.asarray(conf, dtype=float)[:6]) @ grasp_midpoint_offset(model))[:3, 3]


# -----------------------------------------------------------------------------
# * 1. Loading poses walked to from the goal
# -----------------------------------------------------------------------------


def walk_in_starts(tracker: Tracker, roadmap: Roadmap, model: HoldingModel, goal_ids: Sequence[int],
                   candidates: Sequence[Dict[str, Any]], settings: WalkSettings,
                   counters: Dict[str, float],
                   start_accept: Optional[Callable[[np.ndarray], str]] = None) -> Dict[str, Any]:
    """Walk every goal posture to loading poses; add the best of each class to the graph as starts.

    A pose counts when the walk arrives on it and the arrival is collision-free. Per goal posture and class the
    best ``per_class`` are kept: walks free all the way first, then the roadmap's own carry poses, then the
    longest free part, then the most comfortable arm posture. A walk free all the way joins the graph whole (a
    finished path); of a blocked one, the free part from the goal and the arrival on its own.

    Args:
        tracker (Tracker): the graph being searched.
        roadmap (Roadmap): the roadmap (its frame places the poses).
        model (HoldingModel): the live holding model.
        goal_ids (Sequence[int]): the goal nodes to walk from.
        candidates (Sequence[Dict[str, Any]]): loading poses (``carry``, ``label``, ``source``, ``pose`` in the
            roadmap's frame, and ``midpoint`` and ``roll_deg`` unless they are the roadmap's own).
        settings (WalkSettings): steps and how many to keep.
        counters (Dict[str, float]): tally.
        start_accept (Optional[Callable[[np.ndarray], str]]): the caller's say on a loading posture, best ranked
            first: "" to take it, or why not. A refused pose is dropped for every goal posture, and a class is
            given up after ``settings.drive_tries`` refusals.

    Returns:
        Dict[str, Any]: ``starts`` (node -> what start it is: class, label, source, the goal it was walked from,
        whether the walk was free), ``targets`` (goal node -> the kept poses: ``pose`` in the world and the
        start's ``note``; what later rounds walk toward), ``chain_ends`` (goal node -> the nodes where blocked
        walks' free parts end) and ``tally``.
    """
    to_world = robot_from_roadmap(roadmap, model)
    world_poses = [to_world @ candidate["pose"] for candidate in candidates]
    goal_ids = [int(goal) for goal in goal_ids]
    jobs = [(tracker.confs[goal], pose) for goal in goal_ids for pose in world_poses]
    # Many poses are screened at a coarser step; a few are walked at the real one right away.
    coarser = settings.screen_factor if len(candidates) > SCREEN_ABOVE else 1.0
    arrivals = run_walks(tracker.pool, jobs, "arrival", False, settings, counters, "walk in", coarser=coarser)
    free = [index for index, walk in enumerate(arrivals) if walk["free"]]
    # * The walks that arrive free are walked again with every step checked, to know how much of each is free.
    checked = run_walks(tracker.pool, [jobs[index] for index in free], "all", True, settings, counters, "walk check")
    tally = {"walks": len(jobs), "reached": int(sum(walk["reached"] for walk in arrivals)),
             "arrived_free": len(free), "free_all_the_way": int(sum(walk["reached"] for walk in checked))}
    source_rank = {"file": 0, "derived": 1, "grid": 2}
    step = float(np.radians(settings.node_step_deg))
    chains: List[Tuple[Optional[int], List[np.ndarray]]] = []
    notes: List[Dict[str, Any]] = []
    targets: Dict[int, List[Dict[str, Any]]] = {goal: [] for goal in goal_ids}
    refused_poses: set = set()
    refusals: Dict[str, int] = {}
    wanted = max(settings.per_class, settings.targets_per_class)
    for goal_number, goal in enumerate(goal_ids):
        per_class: Dict[str, List[Tuple[Tuple, int, Dict[str, Any], Dict[str, Any]]]] = {}
        for index, full in zip(free, checked):
            if index // len(world_poses) != goal_number:
                continue
            candidate = candidates[index % len(world_poses)]
            comfort = pair_margins(model, arrivals[index]["conf"])
            rank = (not full["reached"], source_rank.get(candidate["source"], 3), -full["steps"] / full["of"],
                    -comfort["comfort"], -comfort["limit_margin_deg"])
            per_class.setdefault(candidate["carry"], []).append((rank, index, candidate, full))
        for carry, rows in per_class.items():
            ranked = []
            for row in sorted(rows, key=lambda row: row[0]):
                pose_number = row[1] % len(world_poses)
                if len(ranked) >= wanted or (start_accept and refusals.get(carry, 0) >= settings.drive_tries):
                    break
                if pose_number in refused_poses:
                    continue
                if start_accept is not None and start_accept(np.asarray(arrivals[row[1]]["conf"], dtype=float)):
                    refused_poses.add(pose_number)
                    refusals[carry] = refusals.get(carry, 0) + 1
                    continue
                ranked.append(row)
            for place, (_rank, index, candidate, full) in enumerate(ranked):
                note = {"goal": goal, "carry": carry, "label": candidate["label"], "source": candidate["source"],
                        "midpoint": candidate.get("midpoint"), "roll_deg": candidate.get("roll_deg"),
                        "free_walk": bool(full["reached"]), "free_fraction": full["steps"] / full["of"],
                        "length_m": arrivals[index]["distance_m"]}
                if place < settings.targets_per_class:
                    targets[goal].append({"pose": world_poses[index % len(world_poses)], "note": note})
                if place >= settings.per_class:
                    continue
                if full["reached"]:
                    chains.append((goal, chain_of(full["confs"], step)))
                    notes.append({**note, "role": "start"})
                    continue
                if full["steps"] >= settings.via_min_steps:
                    chains.append((goal, chain_of(full["confs"], step)))
                    notes.append({**note, "role": "via"})
                # * A blocked walk's arrival is still a start: the roadmap may reach it from another side.
                chains.append((None, [np.asarray(arrivals[index]["conf"], dtype=float)]))
                notes.append({**note, "role": "start"})
    ids = add_chains(tracker, chains, settings, counters, "walk-in edges")
    starts: Dict[int, Dict[str, Any]] = {}
    chain_ends: Dict[int, List[int]] = {goal: [] for goal in goal_ids}
    for note, nodes in zip(notes, ids):
        if note["role"] == "start":
            starts.setdefault(nodes[-1], note)
        else:
            chain_ends[note["goal"]].append(nodes[-1])
    tally["starts"] = len(starts)
    tally["starts_free_walk"] = int(sum(note["free_walk"] for note in starts.values()))
    if start_accept is not None:
        tally["refused_drive"] = len(refused_poses)
    return {"starts": starts, "targets": targets, "chain_ends": chain_ends, "tally": tally}


# -----------------------------------------------------------------------------
# * 2. Waypoints on the way out
# -----------------------------------------------------------------------------


def via_targets(model: HoldingModel, conf: Sequence[float], settings: WalkSettings) -> List[np.ndarray]:
    """Where a waypoint round tries to move the bar from one configuration: 26 slides and 6 turns on the spot."""
    pose = bar_from_arm(model, "left", np.asarray(conf, dtype=float)[:6])
    middle = (pose @ grasp_midpoint_offset(model))[:3, 3]
    out = []
    for direction in itertools.product((-1.0, 0.0, 1.0), repeat=3):
        if any(direction):
            slid = pose.copy()
            slid[:3, 3] += settings.via_reach_m * np.asarray(direction) / np.linalg.norm(direction)
            out.append(slid)
    for axis in np.eye(3):
        for sign in (1.0, -1.0):
            turn = axis_rotation(axis, sign * float(np.radians(settings.via_turn_deg)))
            turned = np.eye(4)
            turned[:3, :3] = turn @ pose[:3, :3]
            turned[:3, 3] = middle + turn @ (pose[:3, 3] - middle)
            out.append(turned)
    return out


def via_round(tracker: Tracker, model: HoldingModel, frontier: Sequence[Tuple[int, int]],
              targets: Dict[int, List[np.ndarray]], goal_middle: np.ndarray, settings: WalkSettings,
              counters: Dict[str, float]) -> Dict[str, Any]:
    """Walk on from each frontier node; keep the free part of the best walks as chains, and any start reached.

    Args:
        tracker (Tracker): the graph being searched.
        model (HoldingModel): the live holding model.
        frontier (Sequence[Tuple[int, int]]): (node to walk from, the goal node it descends from).
        targets (Dict[int, List[Dict[str, Any]]]): goal node -> the loading poses its walks aim for (``pose`` in
            the world, ``note``: see :func:`walk_in_starts`).
        goal_middle (np.ndarray): where the grasp midpoint is at the goal (world).
        settings (WalkSettings): reach, steps, how many to keep.
        counters (Dict[str, float]): tally.

    Returns:
        Dict[str, Any]: ``starts`` (node -> the note of the loading pose a walk arrived on, free all the way),
        ``frontier`` (the next round's, furthest from the goal first) and ``tally``.
    """
    jobs, owner = [], []
    for node, goal in frontier:
        conf = tracker.confs[node]
        for target in targets.get(goal, []):
            jobs.append((conf, target["pose"]))
            owner.append((node, goal, target["note"]))
        for pose in via_targets(model, conf, settings):
            jobs.append((conf, pose))
            owner.append((node, goal, None))
    walks = run_walks(tracker.pool, jobs, "all", True, settings, counters, "via walks")
    step = float(np.radians(settings.node_step_deg))
    chains: List[Tuple[int, List[np.ndarray]]] = []
    kinds: List[Tuple[str, Any]] = []
    by_node: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    for (node, goal, note), walk in zip(owner, walks):
        if note is not None and walk["reached"]:
            chains.append((node, chain_of(walk["confs"], step)))
            kinds.append(("start", {**note, "free_walk": False, "via": True}))
        elif walk["steps"] >= settings.via_min_steps:
            by_node.setdefault(node, []).append((goal, walk))
    for node, rows in by_node.items():
        here = float(np.linalg.norm(midpoint_of(model, tracker.confs[node]) - goal_middle))
        # A waypoint is worth keeping when it is no nearer the goal than where the walk began; the longest first.
        rows = [(goal, walk) for goal, walk in rows
                if np.linalg.norm(midpoint_of(model, walk["conf"]) - goal_middle) >= here - 1e-6
                or walk["angle_rad"] > 1e-6]
        for goal, walk in sorted(rows, key=lambda row: -row[1]["steps"])[: settings.via_keep]:
            chains.append((node, chain_of(walk["confs"], step)))
            kinds.append(("via", goal))
    ids = add_chains(tracker, chains, settings, counters, "via edges")
    starts = {nodes[-1]: note for (kind, note), nodes in zip(kinds, ids) if kind == "start"}
    ends = {nodes[-1]: goal for (kind, goal), nodes in zip(kinds, ids) if kind == "via"}
    old = {node for node, _goal in frontier}
    chosen = spread_out(model, tracker.confs, [node for node in ends if node not in old], sorted(old), goal_middle,
                        settings.via_frontier)
    return {"starts": starts, "frontier": [(node, ends[node]) for node in chosen],
            "tally": {"walked_from": len(frontier), "walks": len(jobs), "starts_reached": len(starts),
                      "waypoints": len(ends)}}


def spread_out(model: HoldingModel, confs: np.ndarray, candidates: Sequence[int], walked_from: Sequence[int],
               goal_middle: np.ndarray, count: int) -> List[int]:
    """Pick waypoints to walk on from: far from the goal, and far from each other and from where walks began.

    The gap between two is the distance of their grasp midpoints plus 0.3 m per radian the bar is turned between
    them. The first pick is the one furthest from the goal; each next one is the one furthest from everything
    picked and walked from so far.

    Args:
        model (HoldingModel): the live holding model.
        confs (np.ndarray): node configurations.
        candidates (Sequence[int]): the new waypoints.
        walked_from (Sequence[int]): the nodes this round walked from.
        goal_middle (np.ndarray): where the grasp midpoint is at the goal (world).
        count (int): how many to pick.

    Returns:
        List[int]: the picked nodes, in picking order.
    """
    offset = grasp_midpoint_offset(model)
    frames = {node: bar_from_arm(model, "left", confs[node][:6]) @ offset
              for node in set(candidates) | set(walked_from)}

    def gap(a: int, b: int) -> float:
        distance, angle = pose_gap(frames[a], frames[b])
        return distance + 0.3 * angle

    picked: List[int] = []
    left = list(candidates)
    nearest = {node: min([gap(node, other) for other in walked_from] or [np.inf]) for node in left}
    while left and len(picked) < count:
        if picked:
            best = max(left, key=lambda node: nearest[node])
        else:
            best = max(left, key=lambda node: float(np.linalg.norm(frames[node][:3, 3] - goal_middle)))
        picked.append(best)
        left.remove(best)
        for node in left:
            nearest[node] = min(nearest[node], gap(node, best))
    return picked


def grid_candidates(roadmap: Roadmap, settings: WalkSettings) -> List[Dict[str, Any]]:
    """The grid of loading poses of the three classes (``carry.candidate_grid``), in the roadmap's frame."""
    entries = [{**entry, "label": f"grid_{entry['carry']}"} for entry in candidate_grid(settings.grid_step_m)
               if entry["carry"] in CARRY_CLASSES]
    return class_poses(roadmap, entries, "grid")
