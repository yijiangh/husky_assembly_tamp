"""The build, query and export steps of the roadmap pilot, as plain functions.

``scripts/prm_pilot.py`` parses the command line and calls these; tests and
notebooks can call them directly.

Build (once per bar and scene)
    1. Anchors: the home configuration, the goal's branch pairs and a spread of
       library starts, each on its own 2*pi turn and on the home turn.
    2. Sampling, in parallel, until the node budget is met.
    3. Nearest-neighbour pairs; every pair measured, in parallel (violation
       always; collision unless the build is lazy).
    4. Saved to ``<root>/prm/<run>/``.

Query (once per goal and choice of starts)
    1. Endpoints: every collision-free branch pair at the goal pose and at
       every chosen start pose, in parallel.
    2. Endpoint edges to their nearest roadmap nodes (violation measured for
       all; collision checked lazily by default -- see below).
    3. For every node, the smallest tolerance at which it reaches the goal set
       (its "gap"), made exact for the best node of every start by collision
       checking the edges of its route and repeating.
    4. The shortest path under each tolerance, filtered and penalised.
    5. Saved to ``<root>/prm/<run>/queries/<tag>.json.gz``.

Why endpoint edges are checked lazily by default: with every start pose of a
goal expanded to all its branch pairs there can be ten thousand endpoint
nodes and a hundred thousand candidate edges, and only the few on the routes
that decide each start's answer matter. Checking those, and repeating until
every deciding route is fully checked, gives the same answer as checking
everything.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from math import ceil
from time import perf_counter, time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from husky_assembly_tamp.motion_planner.holding_prm.carry import carry_bar_pose
from husky_assembly_tamp.motion_planner.holding_prm.holding import (
    HoldingModel,
    arm_limits,
    pose_from_matrix,
    rebranch12,
)
from husky_assembly_tamp.motion_planner.holding_prm.parallel import (
    WorkerPool,
    task_edges,
    task_expand,
    task_model,
    task_nodes,
    task_sample,
)
from husky_assembly_tamp.motion_planner.holding_prm.roadmap import (
    EDGE_COLUMNS,
    NODE_COLUMNS,
    Roadmap,
    candidate_pairs,
    edge_cost,
)
from husky_assembly_tamp.motion_planner.holding_prm.sampling import SAMPLING_MODES, SamplerConfig, add_counters
from husky_assembly_tamp.motion_planner.holding_prm.search import (
    EdgeTable,
    build_graph,
    lazy_search,
    minimax_to_goals,
    path_report,
    route_from,
    shortest_between_sets,
)
from husky_assembly_tamp.motion_planner.holding_prm.store import (
    EDGE_DECIMALS,
    NODE_DECIMALS,
    SCHEMA_VERSION,
    copy_dashboard,
    list_queries,
    load_roadmap,
    optional_float,
    prm_root,
    provenance_record,
    query_path,
    read_json,
    rounded_columns,
    run_dir,
    save_roadmap,
    update_query_index,
    update_run_index,
    write_json_atomic,
)


# * The start classes the library labels each start with (from its straight-line probe).
DIFFICULTY_CLASSES = ("free", "blocked", "broken")

# * The most chunks the first sampling round makes. Not tied to the worker
# * count, so the node set does not depend on how many workers ran the build.
FIRST_ROUND_CHUNKS = 16


# -----------------------------------------------------------------------------
# * Settings
# -----------------------------------------------------------------------------


@dataclass
class BuildSettings:
    """Everything that decides what roadmap a build makes.

    Attributes:
        run_name (str): the run folder name under ``<root>/prm``.
        library (str): the problem library (``cdfm_v1``, ``smoke``).
        bar_id (str): the bar.
        source_key (str): the trimmed-scene cache the library was built from.
        goals (Optional[List[int]]): goal indices whose branch pairs become
            anchors; None = every goal of the bar.
        anchor_starts (int): library starts per goal used as anchors.
        anchor_turns (str): "both" (each anchor as given and moved onto the
            home turn), "given" or "home".
        nodes (int): sampled nodes to keep (anchors come on top).
        k (int): neighbours each node tries to join.
        radius_deg (float): the largest joint move an edge may have.
        collision_step_deg (float): largest joint move between collision checks.
        viol_step_deg (float): largest joint move between violation points.
        rot_deg_per_mm (float): how rotation error folds into the mm score.
        box_margin_deg (float): half-width of the ``anchor_box`` draw.
        gauss_sigma_deg (float): spread of the ``anchor_gauss`` draw.
        line_sigma_deg (float): noise on the ``anchor_line`` draw.
        mode_mix (Dict[str, float]): weight of each sampling mode.
        max_branches (int): other-arm branches kept per draw (0 = all).
        seed (int): the random seed.
        workers (int): worker processes.
        lazy (bool): skip collision checks of roadmap edges at build time.
        sample_chunk (int): draws per sampling task.
        edge_chunk (int): edges per edge task.
        starts_file (str): a starts file (see :func:`file_starts`) whose
            poses also become anchors, every collision-free branch pair of
            each on the home turn; "" = none.
    """

    run_name: str
    library: str
    bar_id: str
    source_key: str
    goals: Optional[List[int]] = None
    anchor_starts: int = 40
    anchor_turns: str = "both"
    nodes: int = 2000
    k: int = 10
    radius_deg: float = 90.0
    collision_step_deg: float = 10.0
    viol_step_deg: float = 2.5
    rot_deg_per_mm: float = 0.4
    box_margin_deg: float = 90.0
    gauss_sigma_deg: float = 30.0
    line_sigma_deg: float = 10.0
    mode_mix: Dict[str, float] = field(default_factory=lambda: {
        "anchor_box": 0.2, "anchor_gauss": 0.3, "anchor_line": 0.5, "global_box": 0.0})
    # * One branch per draw: the other arm's branch nearest the draw centre.
    # * Measured on the smoke roadmap (2026-09-30): with every branch kept, the
    # * other arm's median distance to the anchor was 182 degrees -- flipped
    # * elbows and wrists that sit far from every other node and join nothing.
    max_branches: int = 1
    seed: int = 0
    workers: int = 6
    lazy: bool = False
    sample_chunk: int = 32
    edge_chunk: int = 64
    starts_file: str = ""

    def edge_settings(self) -> Dict[str, float]:
        """The edge-measuring settings the workers need, in radians."""
        return {"collision_step_rad": float(np.radians(self.collision_step_deg)),
                "viol_step_rad": float(np.radians(self.viol_step_deg)),
                "rot_deg_per_mm": float(self.rot_deg_per_mm)}


# -----------------------------------------------------------------------------
# * The library
# -----------------------------------------------------------------------------


def load_library_entry(root: str, library: str, bar_id: str) -> Dict[str, Any]:
    """Read one bar of a problem library.

    ! Read directly, not through ``benchmark.library.load_bar_library``: that
    ! module imports ``benchmark.sampling``, which imports the planner core,
    ! and importing the core empties the planner's log file.

    Args:
        root (str): the benchmark data root.
        library (str): the library name.
        bar_id (str): the bar.

    Returns:
        Dict[str, Any]: the bar's entry (``problem``, ``goals``, ...).
    """
    path = os.path.join(root, "library", library, "bars", f"{bar_id}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No library entry for {bar_id} in library {library!r} ({path}). "
            "Check --library and --bar, or build the library first."
        )
    return read_json(path)


def find_goal(entry: Dict[str, Any], goal_index: int) -> Dict[str, Any]:
    """Return the goal with this index (the library's own numbering)."""
    for goal in entry["goals"]:
        if int(goal["goal_index"]) == int(goal_index):
            return goal
    raise KeyError(f"Bar {entry['bar_id']} has no goal {goal_index}; it has "
                   f"{[g['goal_index'] for g in entry['goals']]}.")


def spread_starts(starts: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Pick ``limit`` starts spread across difficulty classes and carry anchors.

    The same rule as ``benchmark.library.select_starts`` (copied, because that
    module cannot be imported here; see :func:`load_library_entry`):
    production's own pick (lowest rank) first, then round-robin over
    (difficulty, anchor) buckets, easiest class first.

    Args:
        starts (Sequence[Dict[str, Any]]): candidate starts.
        limit (int): how many to keep.

    Returns:
        List[Dict[str, Any]]: the chosen starts, by rank.
    """
    if limit <= 0:
        return []
    if len(starts) <= limit:
        return list(starts)
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for start in starts:
        buckets.setdefault((start.get("difficulty", "broken"), start["anchor"]), []).append(start)
    for bucket in buckets.values():
        bucket.sort(key=lambda start: start["rank"])
    chosen = [min(starts, key=lambda start: start["rank"])]
    chosen_ids = {chosen[0]["start_index"]}
    order = sorted(buckets, key=lambda key: ({"free": 0, "blocked": 1, "broken": 2}.get(key[0], 3), key[1]))
    position = 0
    while len(chosen) < limit:
        added = False
        for key in order:
            bucket = buckets[key]
            if position >= len(bucket) or len(chosen) >= limit:
                continue
            added = True
            if bucket[position]["start_index"] not in chosen_ids:
                chosen.append(bucket[position])
                chosen_ids.add(bucket[position]["start_index"])
        if not added:
            break
        position += 1
    return sorted(chosen, key=lambda start: start["rank"])


def choose_starts(goal: Dict[str, Any], spec: str, max_starts: Optional[int]) -> List[Dict[str, Any]]:
    """The library starts a query plans from.

    Args:
        goal (Dict[str, Any]): the goal entry.
        spec (str): "all", one or more classes ("free", "free,blocked"), or
            start indices ("0,12").
        max_starts (Optional[int]): keep at most this many (spread as in
            :func:`spread_starts`); None = no limit.

    Returns:
        List[Dict[str, Any]]: the starts.
    """
    starts = goal["starts"]
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if spec == "all":
        chosen = list(starts)
    elif parts and all(part.isdigit() for part in parts):
        by_index = {int(start["start_index"]): start for start in starts}
        missing = [int(p) for p in parts if int(p) not in by_index]
        if missing:
            raise KeyError(f"Goal {goal['goal_index']} has no starts {missing}.")
        chosen = [by_index[int(p)] for p in parts]
    else:
        unknown = [p for p in parts if p not in DIFFICULTY_CLASSES]
        if unknown:
            raise ValueError(f"Unknown start selection {unknown}; use all, "
                             f"{', '.join(DIFFICULTY_CLASSES)}, or start indices.")
        chosen = [start for start in starts if start.get("difficulty") in parts]
    if max_starts is not None and len(chosen) > max_starts:
        chosen = spread_starts(chosen, max_starts)
    return chosen


def resolve_starts_file(root: str, path: str) -> str:
    """A starts file given as a path, or relative to ``<root>/prm`` (e.g. ``carry/carry_v1/starts.json``).

    Args:
        root (str): the benchmark data root.
        path (str): what was given.

    Returns:
        str: the file.

    Raises:
        FileNotFoundError: when neither exists.
    """
    if os.path.isfile(path):
        return path
    under_prm = os.path.join(prm_root(root), path)
    if os.path.isfile(under_prm):
        return under_prm
    raise FileNotFoundError(f"No starts file {path!r} (also looked for {under_prm}).")


def file_starts(path: str, model: HoldingModel) -> List[Dict[str, Any]]:
    """Start poses from a starts file instead of the library (e.g. carry candidates).

    A starts file (written by ``scripts/prm_carry_search.py starts``) lists
    carry poses by class, grasp-midpoint position and roll, so the same file
    gives a start pose for any bar; or it lists bar poses directly
    (``world_from_bar``). Each entry becomes a start shaped like a library
    start, so the rest of the query does not care where it came from.

    Args:
        path (str): the starts file.
        model (HoldingModel): the bar's holding model (grasps, home).

    Returns:
        List[Dict[str, Any]]: starts with ``start_index`` (position in the
        file), ``anchor`` (the carry class), ``label``, ``difficulty``
        ("file"), ``rank``, ``world_from_bar`` and ``conf`` (home: there is no
        library configuration, so both turn references are the home turn).
    """
    entries = read_json(path)["starts"]
    starts = []
    for index, entry in enumerate(entries):
        if "world_from_bar" in entry:
            world_from_bar = entry["world_from_bar"]
        else:
            world_from_bar = pose_from_matrix(carry_bar_pose(model, entry["carry"], entry["midpoint"],
                                                             float(np.radians(entry["roll_deg"]))))
        starts.append({"start_index": index, "anchor": entry.get("carry", ""), "label": entry.get("label", str(index)),
                       "difficulty": "file", "rank": index, "world_from_bar": world_from_bar,
                       "conf": model.home_conf12.tolist(), "direct_path": None})
    return starts


def make_anchors(entry: Dict[str, Any], goals: Optional[List[int]], anchor_starts: int,
                 anchor_turns: str, limits: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """The configurations the sampler centres its draws on (and that become nodes).

    Args:
        entry (Dict[str, Any]): the library entry of the bar.
        goals (Optional[List[int]]): goal indices to take; None = all.
        anchor_starts (int): starts per goal (spread across classes).
        anchor_turns (str): "both", "given" or "home".
        limits (Dict[str, np.ndarray]): joint limits, for the turn change.

    Returns:
        List[Dict[str, Any]]: one dict per distinct anchor configuration, with
        ``role`` (home, goal, start), ``turn``, ``conf`` and where it came from.
    """
    home = np.asarray(entry["problem"]["home_conf12"], dtype=float)
    sources: List[Dict[str, Any]] = [{"role": "home", "conf": home}]
    chosen_goals = entry["goals"] if goals is None else [find_goal(entry, g) for g in goals]
    for goal in chosen_goals:
        for pair in goal["free_branch_pairs"]:
            sources.append({"role": "goal", "goal_index": goal["goal_index"],
                            "pair_index": pair["pair_index"], "conf": np.asarray(pair["conf"], dtype=float)})
        for start in spread_starts(goal["starts"], anchor_starts):
            sources.append({"role": "start", "goal_index": goal["goal_index"],
                            "start_index": start["start_index"], "carry": start["anchor"],
                            "difficulty": start.get("difficulty"), "conf": np.asarray(start["conf"], dtype=float)})

    anchors: List[Dict[str, Any]] = []
    for source in sources:
        versions = []
        if anchor_turns in ("both", "given"):
            versions.append(("given", source["conf"]))
        if anchor_turns in ("both", "home"):
            versions.append(("home", rebranch12(source["conf"], home, limits)))
        for turn, conf in versions:
            # Skip a version identical to one already listed (the common case:
            # most anchors are already on the home turn).
            if any(np.max(np.abs(conf - a["conf"])) < 1e-9 for a in anchors):
                continue
            anchors.append({**source, "turn": turn, "conf": conf})
    for anchor in anchors:
        anchor["conf"] = [float(v) for v in anchor["conf"]]
    return anchors


# -----------------------------------------------------------------------------
# * Small helpers
# -----------------------------------------------------------------------------


def chunked(items: Sequence[Any], size: int) -> List[Sequence[Any]]:
    """Split a sequence into consecutive pieces of at most ``size``."""
    return [items[start:start + size] for start in range(0, len(items), size)]


def edge_tasks(confs: np.ndarray, pairs: np.ndarray, chunk_size: int, check_collision: bool,
               first_chunk: int = 0) -> List[Dict[str, Any]]:
    """Package node pairs into edge tasks, each carrying the configurations it needs."""
    tasks = []
    for number, piece in enumerate(chunked(np.asarray(pairs), chunk_size)):
        piece = np.asarray(piece, dtype=int).reshape(-1, 2)
        tasks.append({"chunk": first_chunk + number, "pairs": piece.tolist(),
                      "confs_a": confs[piece[:, 0]].tolist(), "confs_b": confs[piece[:, 1]].tolist(),
                      "check_collision": bool(check_collision)})
    return tasks


def graph_summary(n_nodes: int, edges: Dict[str, list], anchor_nodes: Dict[str, List[int]],
                  rot_deg_per_mm: float) -> Dict[str, Any]:
    """Size, connectivity and violation spread of a roadmap's free edges.

    Args:
        n_nodes (int): node count.
        edges (Dict[str, list]): the edge columns.
        anchor_nodes (Dict[str, List[int]]): node ids of each anchor role.
        rot_deg_per_mm (float): score folding.

    Returns:
        Dict[str, Any]: component count and sizes, which anchor roles share
        the largest component, degree and edge-score percentiles.
    """
    columns = {name: np.asarray(values) for name, values in edges.items()}
    free = columns["free"].astype(bool) if len(columns["free"]) else np.zeros(0, dtype=bool)
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(zip(columns["i"][free].tolist(), columns["j"][free].tolist()))
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    largest = components[0] if components else set()
    score = np.maximum(columns["viol_pos_max"].astype(float) * 1000.0,
                       np.degrees(columns["viol_rot_max"].astype(float)) / rot_deg_per_mm) if len(free) else np.zeros(0)
    degrees = np.array([degree for _node, degree in graph.degree()]) if n_nodes else np.zeros(0)
    percentiles = [5, 25, 50, 75, 95]
    return {
        "n_components": len(components),
        "largest_component": len(largest),
        "component_sizes_top10": [len(c) for c in components[:10]],
        "isolated_nodes": int(sum(1 for c in components if len(c) == 1)),
        "anchors_in_largest": {role: int(sum(1 for n in ids if n in largest)) for role, ids in anchor_nodes.items()},
        "anchors_total": {role: len(ids) for role, ids in anchor_nodes.items()},
        "degree_mean": float(degrees.mean()) if len(degrees) else 0.0,
        "degree_percentiles": {str(p): float(np.percentile(degrees, p)) for p in percentiles} if len(degrees) else {},
        "free_edge_score_mm_percentiles": (
            {str(p): float(np.percentile(score[free], p)) for p in percentiles} if free.any() else {}),
        "free_edge_len_linf_deg_percentiles": (
            {str(p): float(np.degrees(np.percentile(columns["len_linf"][free].astype(float), p))) for p in percentiles}
            if free.any() else {}),
    }


def parallel_summary(counters: Dict[str, float], wall_s: float) -> Dict[str, float]:
    """Worker CPU seconds against wall-clock seconds for one phase."""
    busy = float(counters.get("seconds_worker", 0.0))
    return {"wall_s": wall_s, "worker_s": busy, "speedup": busy / wall_s if wall_s > 0 else 0.0}


# -----------------------------------------------------------------------------
# * Build
# -----------------------------------------------------------------------------


def file_anchors(root: str, settings: BuildSettings, existing: Sequence[Dict[str, Any]],
                 verbose: bool) -> List[Dict[str, Any]]:
    """Anchors from a starts file: every collision-free branch pair of each pose, on the home turn.

    The poses must be solved before the sampler is set up (the sampler's
    anchors go to the workers when they start), so a short-lived pool does it.

    Args:
        root (str): the benchmark data root.
        settings (BuildSettings): the build (bar, scene, starts file, workers).
        existing (Sequence[Dict[str, Any]]): anchors already listed (no duplicates).
        verbose (bool): print progress.

    Returns:
        List[Dict[str, Any]]: the new anchors (role "start", with ``carry``,
        ``label`` and ``start_index`` = position in the file).
    """
    with WorkerPool(root, settings.source_key, settings.bar_id, min(settings.workers, 4), None,
                    settings.edge_settings(), verbose) as pool:
        model = HoldingModel.from_dict(pool.run(task_model, [{"chunk": 0}], "model")[0]["model"])
        starts = file_starts(resolve_starts_file(root, settings.starts_file), model)
        home = model.home_conf12.tolist()
        results = pool.run(task_expand, [{"chunk": n, "world_from_bar": s["world_from_bar"],
                                          "turn_references": [["home", home]]} for n, s in enumerate(starts)],
                           "file starts")
    new: List[Dict[str, Any]] = []
    for start, result in zip(starts, results):
        for found in result["confs"]:
            conf = np.asarray(found["conf"], dtype=float)
            if any(np.max(np.abs(conf - np.asarray(a["conf"]))) < 1e-9 for a in list(existing) + new):
                continue
            new.append({"role": "start", "goal_index": None, "start_index": start["start_index"],
                        "carry": start["anchor"], "label": start["label"], "difficulty": "file",
                        "turn": "home", "conf": [float(v) for v in conf]})
    if verbose:
        print(f"  starts file: {len(starts)} poses, {len(new)} branch pairs added as anchors")
    return new


def build_roadmap(root: str, settings: BuildSettings, verbose: bool = True) -> Roadmap:
    """Build a roadmap and save it under ``<root>/prm/<run_name>``.

    Args:
        root (str): the benchmark data root.
        settings (BuildSettings): what to build.
        verbose (bool): print progress.

    Returns:
        Roadmap: the roadmap (also written to disk).
    """
    wall_start = perf_counter()
    say = print if verbose else (lambda *args, **kwargs: None)
    entry = load_library_entry(root, settings.library, settings.bar_id)
    anchors = make_anchors(entry, settings.goals, settings.anchor_starts, settings.anchor_turns, arm_limits())
    if settings.starts_file:
        anchors.extend(file_anchors(root, settings, anchors, verbose))
    sampler = SamplerConfig(
        anchors=np.asarray([a["conf"] for a in anchors], dtype=float),
        anchor_roles=[a["role"] for a in anchors],
        mode_mix=dict(settings.mode_mix),
        box_margin_rad=float(np.radians(settings.box_margin_deg)),
        gauss_sigma_rad=float(np.radians(settings.gauss_sigma_deg)),
        line_sigma_rad=float(np.radians(settings.line_sigma_deg)),
        max_branches=settings.max_branches,
    )
    unknown_modes = [mode for mode in settings.mode_mix if mode not in SAMPLING_MODES]
    if unknown_modes or sum(settings.mode_mix.values()) <= 0:
        raise ValueError(f"Bad sampling mix {settings.mode_mix}; modes are {SAMPLING_MODES}.")
    say(f"[prm build] {settings.run_name}: bar {settings.bar_id}, library {settings.library}, "
        f"{len(anchors)} anchors, target {settings.nodes} nodes, {settings.workers} workers")

    timings: Dict[str, float] = {}
    counters: Dict[str, Dict[str, float]] = {"anchors": {}, "sample": {}, "edges": {}}
    phase = perf_counter()
    with WorkerPool(root, settings.source_key, settings.bar_id, settings.workers,
                    sampler.to_dict(), settings.edge_settings(), verbose) as pool:
        timings["pool_startup_s"] = pool.startup_s
        model = HoldingModel.from_dict(pool.run(task_model, [{"chunk": 0}], "model")[0]["model"])
        roadmap = Roadmap(model=model)

        # * 1. Anchors become nodes (unless they collide).
        phase = perf_counter()
        anchor_confs = [a["conf"] for a in anchors]
        results = pool.run(task_nodes, [{"chunk": n, "confs": list(piece)}
                                        for n, piece in enumerate(chunked(anchor_confs, 16))], "anchors")
        checked = [node for result in results for node in result["nodes"]]
        for result in results:
            add_counters(counters["anchors"], result["counters"])
        for anchor_index, (anchor, node) in enumerate(zip(anchors, checked)):
            if node.pop("collides"):
                anchor["node"] = -1
                continue
            anchor["node"] = roadmap.add_node({**node, "anchor": anchor_index}, kind="anchor")
        timings["anchors_s"] = perf_counter() - phase
        say(f"  anchors: {sum(a['node'] >= 0 for a in anchors)} of {len(anchors)} collision-free")

        # * 2. Sample in rounds until the node budget is met.
        phase = perf_counter()
        sampled: List[Dict[str, Any]] = []
        next_chunk, round_number, kept_so_far = 0, 0, [0]

        def sample_progress(result: Dict[str, Any]) -> str:
            kept_so_far[0] += len(result["nodes"])
            return f"{kept_so_far[0] + len(sampled)} nodes"

        while len(sampled) < settings.nodes:
            round_number += 1
            if round_number == 1:
                # Assume about one kept node per two draws until measured.
                n_chunks = min(FIRST_ROUND_CHUNKS, max(1, ceil(settings.nodes / (0.5 * settings.sample_chunk))))
            else:
                per_chunk = len(sampled) / max(1, next_chunk)
                if per_chunk <= 0 and next_chunk >= 20 * FIRST_ROUND_CHUNKS:
                    raise RuntimeError(
                        f"The sampler kept no node in {next_chunk * settings.sample_chunk} draws. "
                        "Every candidate collided or was unreachable; check the scene and the anchors."
                    )
                n_chunks = int(ceil((settings.nodes - len(sampled)) / max(per_chunk, 0.5) * 1.1))
                n_chunks = max(4, n_chunks)
            kept_so_far[0] = 0
            tasks = [{"chunk": c, "seed": settings.seed, "n_draws": settings.sample_chunk}
                     for c in range(next_chunk, next_chunk + n_chunks)]
            for result in pool.run(task_sample, tasks, f"sample round {round_number}", sample_progress):
                sampled.extend(result["nodes"])
                add_counters(counters["sample"], result["counters"])
            next_chunk += n_chunks
        # Chunks come back sorted, so this cut is the same whatever the worker count.
        dropped = len(sampled) - settings.nodes
        for node in sampled[: settings.nodes]:
            roadmap.add_node(node, kind="sampled")
        counters["sample"]["kept_beyond_budget"] = dropped
        timings["sample_s"] = perf_counter() - phase

        # * 3. Pair nearest neighbours and measure every pair.
        phase = perf_counter()
        confs = roadmap.confs()
        pairs = candidate_pairs(confs, settings.k, float(np.radians(settings.radius_deg)))
        timings["knn_s"] = perf_counter() - phase
        say(f"  {roadmap.n_nodes} nodes, {len(pairs)} candidate edges "
            f"({'violation only, lazy' if settings.lazy else 'violation and collision'})")
        phase = perf_counter()
        blocked = [0]

        def edge_progress(result: Dict[str, Any]) -> str:
            blocked[0] += sum(1 for e in result["edges"] if not e["free"])
            return f"{blocked[0]} blocked so far"

        for result in pool.run(task_edges, edge_tasks(confs, pairs, settings.edge_chunk, not settings.lazy),
                               "edges", edge_progress):
            roadmap.add_edges(result["edges"])
            add_counters(counters["edges"], result["counters"])
        timings["edges_s"] = perf_counter() - phase

    timings["total_s"] = perf_counter() - wall_start
    anchor_nodes: Dict[str, List[int]] = {}
    for anchor in anchors:
        if anchor["node"] >= 0:
            anchor_nodes.setdefault(anchor["role"], []).append(anchor["node"])
    roadmap.anchors = anchors
    roadmap.counters = counters
    roadmap.timings = timings
    roadmap.config = {**asdict(settings), "edge_settings": settings.edge_settings()}
    roadmap.meta = {
        "schema_version": SCHEMA_VERSION, "run_name": settings.run_name, "bar_id": settings.bar_id,
        "library": settings.library, "source_key": settings.source_key, "created": time(),
        "problem": entry["problem"], "provenance": provenance_record(),
        "summary": graph_summary(roadmap.n_nodes, roadmap.edges, anchor_nodes, settings.rot_deg_per_mm),
        "parallel": {name: parallel_summary(counters[name], timings[f"{name}_s"])
                     for name in ("anchors", "sample", "edges")},
    }
    save_roadmap(root, settings.run_name, roadmap)
    write_json_atomic(os.path.join(run_dir(root, settings.run_name), "config.json"), {
        "settings": roadmap.config, "meta": {k: v for k, v in roadmap.meta.items() if k != "problem"},
        "counters": counters, "timings": timings,
    })
    summary = roadmap.meta["summary"]
    update_run_index(root, settings.run_name, {
        "bar_id": settings.bar_id, "library": settings.library, "goals": settings.goals,
        "n_nodes": roadmap.n_nodes, "n_edges": roadmap.n_edges,
        "n_free_edges": int(np.sum(np.asarray(roadmap.edges["free"], dtype=bool))),
        "largest_component": summary["largest_component"], "total_s": timings["total_s"],
    })
    print_build_summary(roadmap)
    return roadmap


def print_build_summary(roadmap: Roadmap) -> None:
    """Print what a build made and what it cost."""
    counters, timings, summary = roadmap.counters, roadmap.timings, roadmap.meta["summary"]
    free = int(np.sum(np.asarray(roadmap.edges["free"], dtype=bool)))
    sample = counters["sample"]
    edges = counters["edges"]
    print(f"[prm build] done in {timings['total_s']:.0f} s: {roadmap.n_nodes} nodes, "
          f"{roadmap.n_edges} edges ({free} free)")
    print(f"  sampling: {int(sample.get('draws', 0))} draws, {int(sample.get('ik_calls', 0))} IK calls "
          f"({int(sample.get('ik_reachable', 0))} reachable), {int(sample.get('collision_checks', 0))} "
          f"collision checks, {int(sample.get('kept', 0))} kept, {timings['sample_s']:.0f} s")
    print(f"  edges: {int(edges.get('collision_checks', 0))} collision checks, "
          f"{int(edges.get('fk_pairs', 0))} violation points, {timings['edges_s']:.0f} s")
    print(f"  graph: {summary['n_components']} components, largest {summary['largest_component']}, "
          f"anchors in largest {summary['anchors_in_largest']} of {summary['anchors_total']}")
    for name, stats in roadmap.meta["parallel"].items():
        print(f"  parallel {name}: {stats['worker_s']:.0f} worker-s in {stats['wall_s']:.0f} s "
              f"(speed-up {stats['speedup']:.1f}x)")


# -----------------------------------------------------------------------------
# * Query
# -----------------------------------------------------------------------------


@dataclass
class QuerySettings:
    """What a query asks of a roadmap.

    Attributes:
        run_name (str): the roadmap run.
        goal_index (int): the goal (library numbering).
        starts (str): "all", classes ("free,blocked") or indices ("0,12").
        max_starts (Optional[int]): at most this many starts (spread).
        k_endpoint (int): neighbours each endpoint node tries to join.
        radius_endpoint_deg (float): largest joint move of an endpoint edge.
        tolerances_mm (List[float]): tolerances to search under.
        w_hold (float): penalty weight of the penalised search.
        endpoint_check (str): "lazy" (default) or "eager" collision checks
            of endpoint edges.
        workers (int): worker processes.
        tag (str): the query's name (file name).
        starts_file (str): take the start poses from this file instead of
            the library (``starts`` and ``max_starts`` are then ignored).
    """

    run_name: str
    goal_index: int
    starts: str = "all"
    max_starts: Optional[int] = None
    k_endpoint: int = 10
    radius_endpoint_deg: float = 90.0
    tolerances_mm: List[float] = field(default_factory=lambda: [2.0, 5.0, 10.0, 30.0])
    w_hold: float = 10.0
    endpoint_check: str = "lazy"
    workers: int = 6
    tag: str = ""
    starts_file: str = ""


def wrapped_match(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> bool:
    """Whether two configurations are the same physical pose (equal up to whole turns)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bool(np.max(np.abs(np.angle(np.exp(1j * diff)))) < tol)


def query_roadmap(root: str, settings: QuerySettings, verbose: bool = True) -> Dict[str, Any]:
    """Ask a saved roadmap how well the chosen starts connect to a goal.

    Args:
        root (str): the benchmark data root.
        settings (QuerySettings): the query.
        verbose (bool): print progress and the summary.

    Returns:
        Dict[str, Any]: the saved query record.
    """
    wall_start = perf_counter()
    say = print if verbose else (lambda *args, **kwargs: None)
    roadmap = load_roadmap(root, settings.run_name)
    build = roadmap.config
    model = roadmap.model
    ratio = float(build["rot_deg_per_mm"])
    entry = load_library_entry(root, build["library"], build["bar_id"])
    goal = find_goal(entry, settings.goal_index)
    if settings.starts_file:
        starts = file_starts(resolve_starts_file(root, settings.starts_file), model)
        tag = settings.tag or f"g{settings.goal_index:03d}_file"
    else:
        starts = choose_starts(goal, settings.starts, settings.max_starts)
        tag = settings.tag or f"g{settings.goal_index:03d}_{settings.starts.replace(',', '-')}"
    home = model.home_conf12
    say(f"[prm query] {settings.run_name} / {tag}: goal {settings.goal_index}, {len(starts)} starts, "
        f"{roadmap.n_nodes} roadmap nodes")

    timings: Dict[str, float] = {}
    counters: Dict[str, Dict[str, float]] = {"endpoints": {}, "endpoint_edges": {}, "lazy_checks": {}}
    with WorkerPool(root, build["source_key"], build["bar_id"], settings.workers, None,
                    build["edge_settings"], verbose) as pool:
        # Start every worker first, so the phase times below are the query's own.
        timings["pool_startup_s"] = pool.startup_s + pool.warm_up()

        # * 1. Every collision-free branch pair at the goal pose and at each start pose.
        phase = perf_counter()
        poses = [("goal", goal, goal["primary_conf"])] + [("start", s, s["conf"]) for s in starts]
        tasks = [{"chunk": number, "world_from_bar": item["world_from_bar"],
                  "turn_references": [["given", list(given)], ["home", home.tolist()]]}
                 for number, (_role, item, given) in enumerate(poses)]
        results = pool.run(task_expand, tasks, "endpoints")
        for result in results:
            add_counters(counters["endpoints"], result["counters"])
        timings["endpoints_s"] = perf_counter() - phase

        # * 2. Endpoint nodes: reuse a roadmap node when one sits at the same configuration.
        roadmap_confs = roadmap.confs()
        tree = cKDTree(roadmap_confs) if len(roadmap_confs) else None
        endpoint: Dict[str, list] = {name: [] for name in (
            "id", "role", "start_index", "carry", "difficulty", "pair", "turns", "library_conf", "reused")}
        new_nodes: Dict[str, list] = {name: [] for name in NODE_COLUMNS}
        next_id = roadmap.n_nodes
        library_goal_pairs = [np.asarray(p["conf"]) for p in goal["free_branch_pairs"]]
        matched_library_pairs = set()
        for (role, item, _given), result in zip(poses, results):
            for found in result["confs"]:
                conf = np.asarray(found["conf"], dtype=float)
                reused = False
                if tree is not None:
                    distance, nearest = tree.query(conf, p=np.inf)
                    if distance < 1e-6:
                        node_id, reused = int(nearest), True
                if not reused:
                    node_id = next_id
                    next_id += 1
                    for name in NODE_COLUMNS:
                        new_nodes[name].append(found.get(name, {"kind": "endpoint", "mode": "", "driver": "",
                                                                 "anchor": -1, "draw": -1}.get(name)))
                if role == "goal":
                    for index, pair_conf in enumerate(library_goal_pairs):
                        if wrapped_match(conf, pair_conf):
                            matched_library_pairs.add(index)
                endpoint["id"].append(node_id)
                endpoint["role"].append(role)
                endpoint["start_index"].append(int(item["start_index"]) if role == "start" else -1)
                endpoint["carry"].append(item.get("anchor", "") if role == "start" else "")
                endpoint["difficulty"].append(item.get("difficulty", "") if role == "start" else "")
                endpoint["pair"].append(int(found["pair"]))
                endpoint["turns"].append("+".join(found["turns"]))
                endpoint["library_conf"].append(bool(found["dist_given"] < 1e-6))
                endpoint["reused"].append(reused)
        new_ids = list(range(roadmap.n_nodes, next_id))
        all_confs = np.vstack([roadmap_confs, np.asarray(new_nodes["conf"], dtype=float).reshape(-1, 12)])
        start_ids = sorted({i for i, r in zip(endpoint["id"], endpoint["role"]) if r == "start"})
        goal_ids = sorted({i for i, r in zip(endpoint["id"], endpoint["role"]) if r == "goal"})
        say(f"  endpoints: {len(goal_ids)} goal nodes (library lists {len(library_goal_pairs)} pairs, "
            f"{len(matched_library_pairs)} found again), {len(start_ids)} start nodes, "
            f"{sum(endpoint['reused'])} shared with the roadmap")
        if not goal_ids:
            raise RuntimeError(f"No collision-free way to hold the bar at goal {settings.goal_index}.")

        # * 3. Endpoint edges: violation for all, collision now or lazily.
        phase = perf_counter()
        pairs = candidate_pairs(all_confs, settings.k_endpoint, float(np.radians(settings.radius_endpoint_deg)),
                                query_ids=new_ids)
        eager = settings.endpoint_check == "eager"
        endpoint_edges: Dict[str, list] = {name: [] for name in EDGE_COLUMNS}
        for result in pool.run(task_edges, edge_tasks(all_confs, pairs, 256, eager), "endpoint edges"):
            add_counters(counters["endpoint_edges"], result["counters"])
            for record in result["edges"]:
                for name in EDGE_COLUMNS:
                    endpoint_edges[name].append(record[name])
        timings["endpoint_edges_s"] = perf_counter() - phase
        table = EdgeTable.from_columns([roadmap.edges, endpoint_edges], ratio)
        row_of = table.index_of()
        say(f"  {len(pairs)} endpoint edges; {int(np.sum(~table.checked))} of {len(table.i)} edges not yet "
            "collision-checked")

        def check_rows(rows: List[int]) -> List[Dict[str, Any]]:
            """Collision-check the given table rows in the workers; returns their records."""
            row_pairs = np.stack([table.i[rows], table.j[rows]], axis=1)
            records = []
            for result in pool.run(task_edges, edge_tasks(all_confs, row_pairs, 16, True), "lazy checks"):
                add_counters(counters["lazy_checks"], result["counters"])
                records.extend(result["edges"])
            return records

        # * 4. Every node's gap to the goal set, made exact for the best node of each start.
        phase = perf_counter()
        start_of = {}
        for node_id, role, start_index in zip(endpoint["id"], endpoint["role"], endpoint["start_index"]):
            if role == "start":
                start_of.setdefault(node_id, start_index)
        verify_rounds = 0
        while True:
            gap, parent = minimax_to_goals(len(all_confs), table, table.usable(optimistic=True), goal_ids)
            best_node: Dict[int, int] = {}
            for node_id, start_index in start_of.items():
                if start_index not in best_node or gap[node_id] < gap[best_node[start_index]]:
                    best_node[start_index] = node_id
            unchecked = set()
            for node_id in best_node.values():
                if not np.isfinite(gap[node_id]):
                    continue
                route = route_from(node_id, parent)
                for a, b in zip(route[:-1], route[1:]):
                    row = row_of[(min(a, b), max(a, b))]
                    if not table.checked[row]:
                        unchecked.add(row)
            if not unchecked:
                break
            verify_rounds += 1
            rows = sorted(unchecked)
            for row, record in zip(rows, check_rows(rows)):
                table.checked[row] = True
                table.free[row] = bool(record["free"])
        timings["gaps_s"] = perf_counter() - phase
        say(f"  gaps verified in {verify_rounds} rounds")

        # * 5. Paths: the best start's minimax route, then each tolerance.
        phase = perf_counter()
        viol_step = float(build["edge_settings"]["viol_step_rad"])
        lazy_tally = {"rounds": 0, "edges_checked": 0, "edges_blocked": 0}

        def search(tolerance_mm: Optional[float], penalised: bool) -> Optional[List[int]]:
            """One lazy search: filtered by, or penalised above, a tolerance."""
            def find_path(mask: np.ndarray) -> Optional[List[int]]:
                if penalised:
                    weights = edge_cost(table.len_l2, table.score, tolerance_mm, settings.w_hold)
                    found = shortest_between_sets(build_graph(table, mask, weights, start_ids, goal_ids))
                else:
                    within = mask & (table.score <= tolerance_mm)
                    found = shortest_between_sets(build_graph(table, within, table.len_l2, start_ids, goal_ids))
                return found[0] if found else None
            path, tally = lazy_search(find_path, table, check_rows)
            add_counters(lazy_tally, tally)
            return path

        def describe(path: Optional[List[int]]) -> Optional[Dict[str, Any]]:
            """Path report plus which start and goal it used."""
            if not path:
                return None
            report = path_report(model, all_confs, path, table, viol_step, ratio)
            report["start"] = endpoint_info(endpoint, path[0], "start")
            report["goal"] = endpoint_info(endpoint, path[-1], "goal")
            return report

        start_gaps = {s: float(gap[n]) for s, n in best_node.items()}
        finite = {s: g for s, g in start_gaps.items() if np.isfinite(g)}
        headline: Dict[str, Any] = {"best_gap_mm": None}
        if finite:
            best_start = min(finite, key=finite.get)
            best_gap = finite[best_start]
            headline = {
                "best_gap_mm": best_gap, "best_start_index": best_start,
                "minimax_route": describe(route_from(best_node[best_start], parent)),
                "shortest_at_best_gap": describe(search(best_gap * (1 + 1e-9) + 1e-9, penalised=False)),
            }
        answers = []
        for tolerance in settings.tolerances_mm:
            answers.append({
                "tolerance_mm": tolerance,
                "starts_connected": int(sum(1 for g in finite.values() if g <= tolerance)),
                "filtered": describe(search(tolerance, penalised=False)),
                "penalised": describe(search(tolerance, penalised=True)),
            })
        timings["paths_s"] = perf_counter() - phase

    # The gaps and routes may have changed while the paths were checked; the
    # table only ever loses edges, so recompute them once on the final table.
    gap, parent = minimax_to_goals(len(all_confs), table, table.usable(optimistic=True), goal_ids)
    timings["total_s"] = perf_counter() - wall_start
    start_rows = start_table(starts, endpoint, gap)
    record = {
        "schema_version": SCHEMA_VERSION, "tag": tag, "run_name": settings.run_name,
        "bar_id": build["bar_id"], "library": build["library"], "goal_index": settings.goal_index,
        "settings": asdict(settings), "rot_deg_per_mm": ratio, "created": time(),
        "goal": {"world_from_bar": goal["world_from_bar"], "n_library_pairs": len(library_goal_pairs),
                 "n_library_pairs_found": len(matched_library_pairs), "n_goal_nodes": len(goal_ids)},
        "n_roadmap_nodes": roadmap.n_nodes, "n_start_nodes": len(start_ids),
        "endpoint": endpoint,
        "endpoint_nodes": rounded_columns(new_nodes, NODE_DECIMALS),
        "endpoint_edges": rounded_columns(endpoint_edges, EDGE_DECIMALS),
        # The lazy checks changed some edges; store the final verdicts of every edge.
        "edge_checked": table.checked.tolist(), "edge_free": table.free.tolist(),
        "node_gap_mm": [optional_float(g) for g in gap], "node_parent": parent.tolist(),
        "headline": headline, "tolerances": answers, "starts": start_rows,
        "by_difficulty": difficulty_summary(start_rows, settings.tolerances_mm),
        "verify_rounds": verify_rounds, "lazy": lazy_tally, "counters": counters, "timings": timings,
        "parallel": {name: parallel_summary(counters[key], timings[t]) for name, key, t in (
            ("endpoints", "endpoints", "endpoints_s"), ("endpoint_edges", "endpoint_edges", "endpoint_edges_s"))},
        "provenance": provenance_record(),
    }
    path = query_path(root, settings.run_name, tag)
    write_json_atomic(path, record)
    update_query_index(root, settings.run_name, {
        "tag": tag, "goal_index": settings.goal_index, "starts": settings.starts, "n_starts": len(starts),
        "file": f"{tag}.json.gz", "best_gap_mm": headline.get("best_gap_mm"),
        "best_start_index": headline.get("best_start_index"),
        "starts_connected": {str(a["tolerance_mm"]): a["starts_connected"] for a in answers},
    })
    if verbose:
        print_query_summary(record)
    return record


def endpoint_info(endpoint: Dict[str, list], node_id: int, role: str) -> Optional[Dict[str, Any]]:
    """What the endpoint table says about one node in one role (None if it has no such role)."""
    for row, (candidate, candidate_role) in enumerate(zip(endpoint["id"], endpoint["role"])):
        if candidate == node_id and candidate_role == role:
            return {name: values[row] for name, values in endpoint.items()}
    return None


def start_table(starts: Sequence[Dict[str, Any]], endpoint: Dict[str, list],
                gap: np.ndarray) -> List[Dict[str, Any]]:
    """One row per start: its library labels and its best gap to the goal set.

    Args:
        starts (Sequence[Dict[str, Any]]): the library starts of the query.
        endpoint (Dict[str, list]): the endpoint table.
        gap (np.ndarray): every node's gap (mm).

    Returns:
        List[Dict[str, Any]]: rows with ``start_index``, ``carry`` anchor,
        ``difficulty``, ``direct_path_free``, ``n_nodes``, ``best_gap_mm``,
        ``best_node``, ``best_pair``, ``best_turns`` and ``library_conf_gap_mm``
        (the gap of the library's own start configuration).
    """
    rows_of_start: Dict[int, List[int]] = {}
    for row, (role, start_index) in enumerate(zip(endpoint["role"], endpoint["start_index"])):
        if role == "start":
            rows_of_start.setdefault(int(start_index), []).append(row)
    rows = []
    for start in starts:
        index = int(start["start_index"])
        members = rows_of_start.get(index, [])
        gaps = [float(gap[endpoint["id"][r]]) for r in members]
        best = int(np.argmin(gaps)) if gaps else -1
        library = [float(gap[endpoint["id"][r]]) for r in members if endpoint["library_conf"][r]]
        rows.append({
            "start_index": index, "label": start.get("label", str(index)),
            "carry": start.get("anchor"), "difficulty": start.get("difficulty"),
            "rank": start.get("rank"), "direct_path_free": bool((start.get("direct_path") or {}).get("free")),
            "n_nodes": len(members),
            "best_gap_mm": optional_float(gaps[best]) if gaps else None,
            "best_node": endpoint["id"][members[best]] if gaps else None,
            "best_pair": endpoint["pair"][members[best]] if gaps else None,
            "best_turns": endpoint["turns"][members[best]] if gaps else None,
            "library_conf_gap_mm": optional_float(min(library)) if library else None,
        })
    return rows


def difficulty_summary(rows: Sequence[Dict[str, Any]], tolerances_mm: Sequence[float]) -> Dict[str, Any]:
    """Per library difficulty class: how many starts, their median gap, and how many connect per tolerance."""
    summary = {}
    for label in DIFFICULTY_CLASSES + ("all",):
        members = [r for r in rows if label == "all" or r["difficulty"] == label]
        gaps = np.array([r["best_gap_mm"] if r["best_gap_mm"] is not None else np.inf for r in members])
        finite = gaps[np.isfinite(gaps)]
        summary[label] = {
            "n": len(members), "n_reachable": int(len(finite)),
            "median_gap_mm": float(np.median(finite)) if len(finite) else None,
            "connected": {str(t): int(np.sum(gaps <= t)) for t in tolerances_mm},
        }
    return summary


def print_query_summary(record: Dict[str, Any]) -> None:
    """Print the query's answer: the best gap, and each tolerance."""
    headline = record["headline"]
    print(f"[prm query] done in {record['timings']['total_s']:.0f} s")
    if headline.get("best_gap_mm") is None:
        print("  no start reaches the goal set at any tolerance (the free roadmap does not connect them)")
    else:
        route = headline["minimax_route"]
        print(f"  best gap {headline['best_gap_mm']:.2f} mm from start {headline['best_start_index']} "
              f"(pair {route['start']['pair']}, turn {route['start']['turns']}) to goal pair "
              f"{route['goal']['pair']} ({route['goal']['turns']}); route of {len(route['nodes'])} nodes")
    print("  tolerance  starts-connected  filtered path (nodes, max mm)   penalised path (max mm)")
    for answer in record["tolerances"]:
        filtered, penalised = answer["filtered"], answer["penalised"]
        f_text = f"{len(filtered['nodes'])} nodes, {filtered['max_score_mm']:.2f}" if filtered else "none"
        p_text = f"{penalised['max_score_mm']:.2f}" if penalised else "none"
        print(f"  {answer['tolerance_mm']:>6.1f} mm  {answer['starts_connected']:>6d} / {len(record['starts']):<6d}"
              f"     {f_text:<28s}  {p_text}")
    for label, stats in record["by_difficulty"].items():
        median = f"{stats['median_gap_mm']:.1f}" if stats["median_gap_mm"] is not None else "-"
        print(f"  {label:>8s}: {stats['n']} starts, median gap {median} mm, connected {stats['connected']}")


# -----------------------------------------------------------------------------
# * Export and info
# -----------------------------------------------------------------------------


def export_dashboard(root: str, run_name: str) -> str:
    """Copy the roadmap page into the run and write the small index it starts from.

    Args:
        root (str): the benchmark data root.
        run_name (str): the run.

    Returns:
        str: the dashboard folder.
    """
    config = read_json(os.path.join(run_dir(root, run_name), "config.json"))
    roadmap_meta = load_roadmap(root, run_name).meta
    out_dir = copy_dashboard(root, run_name)
    problem = roadmap_meta["problem"]
    write_json_atomic(os.path.join(out_dir, "prm.json"), {
        "schema_version": SCHEMA_VERSION, "run_name": run_name, "bar_id": roadmap_meta["bar_id"],
        "library": roadmap_meta["library"], "source_key": roadmap_meta["source_key"],
        "assets": f"../../../assets/{roadmap_meta['source_key']}",
        "problem": {key: problem[key] for key in ("bar_length_m", "bar_radius_m", "bar_from_midpoint",
                                                   "attached_link", "grasp_bar_from_left",
                                                   "grasp_bar_from_right", "joint_names_12")},
        "rot_deg_per_mm": config["settings"]["rot_deg_per_mm"],
        "roadmap": "../roadmap.json.gz",
        "queries": [{**q, "file": f"../queries/{q['file']}"} for q in list_queries(root, run_name)],
    })
    return out_dir


def print_info(root: str, run_name: str) -> None:
    """Print a saved run's build summary and its queries."""
    roadmap = load_roadmap(root, run_name)
    print_build_summary(roadmap)
    for query in list_queries(root, run_name):
        best = query.get("best_gap_mm")
        print(f"  query {query['tag']}: goal {query['goal_index']}, {query['n_starts']} starts, best gap "
              f"{'-' if best is None else f'{best:.2f} mm'}, connected {query['starts_connected']}")
