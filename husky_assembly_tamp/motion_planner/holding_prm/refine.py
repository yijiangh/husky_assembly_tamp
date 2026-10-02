"""Refinement: from roadmap routes within a tolerance to paths that hold the bar exactly.

A saved query gives every start its best roadmap route and that route's gap
(the worst holding error of its straight joint-space edges). This module turns
such routes into exact paths, in the build order of the plan:

    1. Exact gaps. The edges on every start's best route are tracked exactly
       (``tracking.track_edge``: walk the bar, solve both arms, continuity
       gate, collision). An edge that fails is removed and the routes are
       searched again, until every start's best route is made only of tracked
       edges. Its gap is then the start's EXACT gap: the smallest roadmap
       tolerance at which this roadmap yields a path that holds the bar
       exactly. (The same loop as the query's lazy collision check, with
       "track it" in place of "collision-check it".)
    2. Assembly. The tracked edges of a route are joined into one chain of
       waypoints. Where the last waypoint of one edge is not the same IK
       solution as the first of the next (two solutions at one bar pose: a
       branch apart), the next edge is tracked again, seeded from that last
       waypoint, so the whole route is one chain.
    3. Per tolerance, the shortest exact route from the whole start set.
    4. Optional tube rounds (way 2 of the plan): for starts still above the
       target tolerance, draw new nodes in a narrow tube around their routes
       (``sampling`` mode ``path_tube``), join them with short edges, and
       repeat 1-3 on the enlarged graph.

Results go to ``<root>/prm/<run>/refine/<tag>.json.gz``; the saved query is
only read, never changed.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from time import perf_counter, time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from husky_assembly_tamp.motion_planner.holding_prm.holding import HoldingModel, interpolate_confs, linf
from husky_assembly_tamp.motion_planner.holding_prm.parallel import (
    WorkerPool,
    task_edges,
    task_sample_with,
    task_track,
)
from husky_assembly_tamp.motion_planner.holding_prm.pilot import chunked, edge_tasks
from husky_assembly_tamp.motion_planner.holding_prm.roadmap import candidate_pairs
from husky_assembly_tamp.motion_planner.holding_prm.sampling import SamplerConfig, add_counters
from husky_assembly_tamp.motion_planner.holding_prm.search import (
    EdgeTable,
    build_graph,
    lazy_search,
    minimax_to_goals,
    route_from,
    shortest_between_sets,
)
from husky_assembly_tamp.motion_planner.holding_prm.store import (
    SCHEMA_VERSION,
    load_roadmap,
    optional_float,
    provenance_record,
    query_path,
    read_json,
    run_dir,
    update_index,
    write_json_atomic,
)
from husky_assembly_tamp.motion_planner.holding_prm.tracking import SAME_SOLUTION_RAD, TrackSettings, exact_path_report


# * Edges per tracking task. Tracking one edge costs a fraction of a second to
# * a few seconds (mostly collision checks), so small chunks keep workers even.
TRACK_CHUNK = 4

# * Tube draws per sampling task.
TUBE_CHUNK = 32

# * Spacing of the tube points laid along a route's joint lines.
TUBE_POINT_STEP_DEG = 2.5

# * How much more often a tube point on a "bad" edge (over the target
# * tolerance, or untrackable) is picked than one on a good edge.
TUBE_BAD_EDGE_WEIGHT = 4.0


# -----------------------------------------------------------------------------
# * Settings
# -----------------------------------------------------------------------------


@dataclass
class RefineSettings:
    """What a refinement run does.

    Attributes:
        run_name (str): the roadmap run.
        query_tag (str): the saved query whose starts, goal and endpoint edges are used.
        tag (str): this refinement's name (file name); default from the settings.
        tolerances_mm (List[float]): roadmap tolerances to report exact routes for.
        max_gap_mm (float): starts whose roadmap gap is above this are left out
            (tracking can only remove edges, so their exact gap would be larger).
        store_paths_mm (float): store the full exact path of every start whose
            exact gap is at most this (the headline and per-tolerance paths are
            always stored).
        knot_step_deg (float): see ``TrackSettings`` (0 = nodes only).
        position_res_m (float): see ``TrackSettings``.
        rotation_res_rad (float): see ``TrackSettings``.
        lead (str): see ``TrackSettings``.
        edge_sample (int): also track this many random usable edges (score at
            most ``max_gap_mm``), for unbiased trackability statistics.
        tube_rounds (int): rounds of tube draws (0 = none).
        tube_target_mm (float): starts whose exact gap is above this get tube draws.
        tube_max_starts (int): at most this many starts per round (smallest gap first).
        tube_draws_per_start (int): driver draws per start per round.
        tube_sigma_deg (float): spread of a tube draw, per joint.
        tube_k (int): neighbours each tube node tries to join.
        tube_radius_deg (float): largest joint move of a tube edge.
        tube_patience (int): stop after this many rounds in a row with no
            target improving.
        seed (int): random seed (edge sample and tube draws).
        workers (int): worker processes.
        pretracked (str): a file written by :func:`pretrack_roadmap` (the
            overnight work): its roadmap-edge verdicts and waypoints are used
            instead of tracking those edges again; "" = none.
        unresolved_mm (float): starts the roadmap brings within this but
            that have no exact path within it are written out for the seeded
            BiRRT (way 1), with their original route edge by edge.
        unresolved_max (int): at most this many of them, smallest roadmap gap first.
    """

    run_name: str
    query_tag: str = "all"
    tag: str = ""
    tolerances_mm: List[float] = field(default_factory=lambda: [2.0, 5.0, 10.0, 30.0])
    max_gap_mm: float = 30.0
    store_paths_mm: float = 10.0
    knot_step_deg: float = 2.5
    position_res_m: float = 0.01
    rotation_res_rad: float = 0.025
    lead: str = "mid"
    edge_sample: int = 0
    tube_rounds: int = 0
    tube_target_mm: float = 10.0
    tube_max_starts: int = 20
    tube_draws_per_start: int = 200
    tube_sigma_deg: float = 5.0
    tube_k: int = 10
    tube_radius_deg: float = 30.0
    tube_patience: int = 2
    seed: int = 0
    workers: int = 6
    pretracked: str = ""
    unresolved_mm: float = 30.0
    unresolved_max: int = 24

    def track_settings(self) -> TrackSettings:
        """The tracker's settings."""
        return TrackSettings(knot_step_deg=self.knot_step_deg, position_res_m=self.position_res_m,
                             rotation_res_rad=self.rotation_res_rad, lead=self.lead)

    def default_tag(self) -> str:
        """A name built from the settings that change the answer."""
        knots = f"k{self.knot_step_deg:g}" if self.knot_step_deg > 0 else "nodes"
        tube = f"_tube{self.tube_rounds}" if self.tube_rounds else ""
        return f"{self.query_tag}_{knots}_p{self.position_res_m * 1000:g}mm_{self.lead}{tube}"


# -----------------------------------------------------------------------------
# * The graph of a saved query
# -----------------------------------------------------------------------------


@dataclass
class QueryGraph:
    """A saved query's whole graph, rebuilt in memory.

    Attributes:
        model (HoldingModel): the holding model.
        build (Dict[str, Any]): the roadmap's build settings.
        query (Dict[str, Any]): the saved query record.
        confs (np.ndarray): (N, 12) every node: roadmap nodes, then the query's
            endpoint nodes (in the order the query numbered them).
        table (EdgeTable): roadmap edges then endpoint edges, with the final
            collision verdicts the query left.
        n_roadmap_edges (int): how many of the table's rows are roadmap edges.
        goal_ids (List[int]): the goal nodes.
    """

    model: HoldingModel
    build: Dict[str, Any]
    query: Dict[str, Any]
    confs: np.ndarray
    table: EdgeTable
    n_roadmap_edges: int
    goal_ids: List[int]


def load_query_graph(root: str, run_name: str, query_tag: str) -> QueryGraph:
    """Rebuild a saved query's graph (nodes, edges and collision verdicts).

    Args:
        root (str): the benchmark data root.
        run_name (str): the roadmap run.
        query_tag (str): the query.

    Returns:
        QueryGraph: the graph.

    Raises:
        RuntimeError: when the query's edge verdicts do not line up with its
            edges (the query was written by a different roadmap).
    """
    roadmap = load_roadmap(root, run_name)
    record = read_json(query_path(root, run_name, query_tag))
    ratio = float(record["rot_deg_per_mm"])
    extra = np.asarray(record["endpoint_nodes"]["conf"], dtype=float).reshape(-1, 12)
    confs = np.vstack([roadmap.confs(), extra])
    table = EdgeTable.from_columns([roadmap.edges, record["endpoint_edges"]], ratio)
    checked = np.asarray(record["edge_checked"], dtype=bool)
    free = np.asarray(record["edge_free"], dtype=bool)
    if len(checked) != len(table.i) or record["n_roadmap_nodes"] != roadmap.n_nodes:
        raise RuntimeError(
            f"Query {query_tag!r} does not match roadmap {run_name!r} ({len(checked)} edge verdicts for "
            f"{len(table.i)} edges, {record['n_roadmap_nodes']} roadmap nodes recorded against "
            f"{roadmap.n_nodes}). The roadmap was probably rebuilt after the query; run the query again."
        )
    table.checked, table.free = checked, free
    endpoint = record["endpoint"]
    goal_ids = sorted({i for i, role in zip(endpoint["id"], endpoint["role"]) if role == "goal"})
    return QueryGraph(model=roadmap.model, build=roadmap.config, query=record, confs=confs, table=table,
                      n_roadmap_edges=len(roadmap.edges["i"]), goal_ids=goal_ids)


def start_nodes(query: Dict[str, Any], start_indices: Sequence[int]) -> Dict[int, int]:
    """Node id -> start index, for the chosen starts (the query's own rule: first start wins).

    Args:
        query (Dict[str, Any]): the saved query record.
        start_indices (Sequence[int]): which starts to include.

    Returns:
        Dict[int, int]: the map.
    """
    wanted = set(int(s) for s in start_indices)
    endpoint = query["endpoint"]
    owner: Dict[int, int] = {}
    for node_id, role, start_index in zip(endpoint["id"], endpoint["role"], endpoint["start_index"]):
        if role == "start":
            owner.setdefault(int(node_id), int(start_index))
    return {node: start for node, start in owner.items() if start in wanted}


def oriented_confs(record: Dict[str, Any], forward: bool) -> List[List[float]]:
    """An edge's waypoints for the direction a route walks it.

    Args:
        record (Dict[str, Any]): a tracking record (``confs`` walked from node
            i to node j; ``confs_back`` walked from j to i, when tracked).
        forward (bool): the route walks the edge from i to j.

    Returns:
        List[List[float]]: the waypoints, in the route's order; empty when
        the edge did not track in that direction.
    """
    return record["confs"] if forward else record.get("confs_back", [])


def best_node_per_start(start_of: Dict[int, int], gap: np.ndarray) -> Dict[int, int]:
    """Each start's node with the smallest gap."""
    best: Dict[int, int] = {}
    for node_id, start_index in start_of.items():
        if start_index not in best or gap[node_id] < gap[best[start_index]]:
            best[start_index] = node_id
    return best


# -----------------------------------------------------------------------------
# * The tracker state: which edges are tracked, and what they gave
# -----------------------------------------------------------------------------


class Tracker:
    """The graph under refinement, with every edge's tracking verdict.

    The edge table here is a copy whose ``free``/``checked`` mean "trackable"
    and "tracked", so the search code (minimax, lazy search) works on it
    unchanged. It starts with every edge that is usable for collision (free,
    or not yet collision-checked) marked free and untracked: the tracker's
    own collision check, on the solved waypoints, is the one that counts.

    Attributes:
        graph (QueryGraph): the saved query's graph.
        settings (RefineSettings): the run's settings.
        pool (WorkerPool): the worker processes.
        confs (np.ndarray): node configurations (grows with tube nodes).
        table (EdgeTable): edges with their tracking state (grows with tube edges).
        records (Dict[int, Dict[str, Any]]): tracking record of each tracked row.
        kinds (List[str]): "roadmap", "endpoint" or "tube" for each row.
        counters (Dict[str, float]): work done in the workers.
        join_blocked (List[int]): rows removed because a route could not be
            continued across them (see :meth:`assemble`).
    """

    def __init__(self, graph: QueryGraph, settings: RefineSettings, pool: WorkerPool):
        self.graph = graph
        self.settings = settings
        self.pool = pool
        self.track_settings = settings.track_settings()
        base = graph.table
        self.confs = graph.confs.copy()
        self.table = EdgeTable(i=base.i.copy(), j=base.j.copy(), len_l2=base.len_l2.copy(),
                               free=base.usable(optimistic=True).copy(), checked=np.zeros(len(base.i), dtype=bool),
                               viol_pos_max=base.viol_pos_max.copy(), viol_rot_max=base.viol_rot_max.copy(),
                               score=base.score.copy())
        self.row_of = self.table.index_of()
        self.kinds = ["roadmap"] * graph.n_roadmap_edges + ["endpoint"] * (len(base.i) - graph.n_roadmap_edges)
        self.records: Dict[int, Dict[str, Any]] = {}
        # Re-tracks seeded from a chain's last waypoint: (row, from, to, seed) -> record.
        self.seeded: Dict[Tuple, Dict[str, Any]] = {}
        self.counters: Dict[str, float] = {}
        self.join_blocked: List[int] = []
        self.retracks = 0

    # * ---- Tracking edges in the workers ------------------------------------

    def _run(self, items: Sequence[Tuple[int, int, int, Optional[np.ndarray]]], label: str) -> List[Dict[str, Any]]:
        """Track edges given as (row, from node, to node, seed or None); results in order."""
        tasks = []
        for number, piece in enumerate(chunked(list(items), TRACK_CHUNK)):
            tasks.append({
                "chunk": number, "rows": [row for row, _a, _b, _s in piece],
                "confs_a": [self.confs[a].tolist() for _r, a, _b, _s in piece],
                "confs_b": [self.confs[b].tolist() for _r, _a, b, _s in piece],
                "seeds": [None if s is None else np.asarray(s).tolist() for _r, _a, _b, s in piece],
                "settings": self.track_settings.to_dict(),
            })
        records = []
        for result in self.pool.run(task_track, tasks, label):
            add_counters(self.counters, result["counters"])
            records.extend(result["records"])
        return records

    def track_rows(self, rows: Sequence[int]) -> List[Dict[str, Any]]:
        """Track table rows in their stored direction (i -> j), remembering the verdicts.

        Already tracked rows are not tracked again. Returns one record per
        given row, in order, each with ``free`` = trackable (the shape the lazy
        search expects).
        """
        todo = sorted({int(r) for r in rows if int(r) not in self.records})
        if todo:
            items = [(row, int(self.table.i[row]), int(self.table.j[row]), None) for row in todo]
            for record in self._run(items, "track"):
                row = record["row"]
                self.records[row] = record
                self.table.checked[row] = True
                self.table.free[row] = bool(record["free"])
        return [self.records[int(r)] for r in rows]

    def load_pretracked(self, path: str) -> int:
        """Take roadmap-edge verdicts from a pretracking file instead of tracking them again.

        Args:
            path (str): a file written by :func:`pretrack_roadmap` for this roadmap.

        Returns:
            int: how many edges were loaded.

        Raises:
            RuntimeError: when the file was made with other tracking settings
                or for another roadmap (its verdicts would not apply).
        """
        data = read_json(path)
        if data["track_settings"] != self.track_settings.to_dict():
            raise RuntimeError(f"Pretracked file {path} was made with tracker settings {data['track_settings']}, "
                               f"not {self.track_settings.to_dict()}; track again or match the settings.")
        if data["n_roadmap_edges"] != self.graph.n_roadmap_edges:
            raise RuntimeError(f"Pretracked file {path} belongs to a roadmap with {data['n_roadmap_edges']} edges, "
                               f"this one has {self.graph.n_roadmap_edges}.")
        for record in data["records"]:
            row = int(record["row"])
            self.records[row] = record
            self.table.checked[row] = True
            self.table.free[row] = bool(record["free"])
        return len(data["records"])

    # * ---- Extending the graph (tube rounds) --------------------------------

    def add_nodes_and_edges(self, new_confs: np.ndarray, edge_records: Sequence[Dict[str, Any]]) -> None:
        """Append tube nodes and their (untracked) edges.

        Args:
            new_confs (np.ndarray): (M, 12) new node configurations.
            edge_records (Sequence[Dict[str, Any]]): edge records from
                ``roadmap.evaluate_edge`` (violation measured, collision not).
        """
        self.confs = np.vstack([self.confs, np.asarray(new_confs, dtype=float).reshape(-1, 12)])
        if not edge_records:
            return
        ratio = float(self.graph.query["rot_deg_per_mm"])
        columns = {name: [record[name] for record in edge_records]
                   for name in ("i", "j", "len_l2", "viol_pos_max", "viol_rot_max")}
        columns["free"] = [True] * len(edge_records)
        columns["checked"] = [False] * len(edge_records)
        extra = EdgeTable.from_columns([columns], ratio)
        self.table = EdgeTable(**{name: np.concatenate([getattr(self.table, name), getattr(extra, name)])
                                  for name in ("i", "j", "len_l2", "free", "checked", "viol_pos_max",
                                               "viol_rot_max", "score")})
        self.row_of = self.table.index_of()
        self.kinds.extend(["tube"] * len(edge_records))

    # * ---- Joining tracked edges into one chain -----------------------------

    def assemble_many(self, routes: Dict[Any, Sequence[int]]) -> Dict[Any, Tuple[Optional[List[List[float]]], Dict[str, Any]]]:
        """Join the tracked edges of many routes into chains of exact waypoints.

        Each route is walked edge by edge. Normally an edge's waypoints start
        where the previous edge's ended (both are the shared node). When they
        do not -- the previous edge's chain ended on another branch, near a
        straight elbow or wrist -- the edge is tracked again, seeded from
        where the chain actually is, so the route stays one continuous chain.
        All routes advance together: the re-tracks every route needs at its
        current edge go to the workers as one batch, and every seeded
        re-track is remembered, so assembling the same routes again costs
        nothing.

        Args:
            routes (Dict[Any, Sequence[int]]): key -> node ids, start first;
                every edge tracked and trackable.

        Returns:
            Dict[Any, Tuple]: key -> (the waypoints or None, info with
            ``joins`` (edges re-tracked from the previous waypoint),
            ``failed_row`` (the row the chain could not cross, or None) and
            ``blame_row`` (the row to drop when it failed: the last edge whose
            chain ended away from its end node -- the one that left the
            roadmap's branch -- or else the failed row itself)).
        """
        # Two waypoints at one node are the same solution within this (a branch apart otherwise).
        same = SAME_SOLUTION_RAD
        state = {key: {"route": list(route), "edge": 0, "pieces": [], "joins": 0, "failed_row": None,
                       "drift_row": None, "done": len(route) < 2} for key, route in routes.items()}
        while True:
            needed: Dict[Tuple, Tuple[int, int, int, np.ndarray]] = {}
            for walk in state.values():
                route, pieces = walk["route"], walk["pieces"]
                while not walk["done"]:
                    if walk["edge"] >= len(route) - 1:
                        walk["done"] = True
                        break
                    a, b = route[walk["edge"]], route[walk["edge"] + 1]
                    row = self.row_of[(min(a, b), max(a, b))]
                    record = self.records.get(row)
                    if record is None or not record["free"]:
                        walk["failed_row"], walk["done"] = row, True
                        break
                    # Each direction has its own waypoints (walked in that direction).
                    confs = oriented_confs(record, int(self.table.i[row]) == a)
                    if not confs:
                        # The edge tracks only the other way: this route cannot use it.
                        walk["failed_row"], walk["done"] = row, True
                        walk["drift_row"] = None  # the edge itself is to blame
                        break
                    if pieces and linf(confs[0], pieces[-1]) > same:
                        # * Carry the chain on from where it is: a re-track seeded from the last waypoint.
                        key = (row, a, b, tuple(np.round(pieces[-1], 6)))
                        if key not in self.seeded:
                            needed[key] = (row, a, b, np.asarray(pieces[-1]))
                            break  # wait for this batch of re-tracks
                        again = self.seeded[key]
                        walk["joins"] += 1
                        if not again["free"]:
                            walk["failed_row"], walk["done"] = row, True
                            break
                        confs = again["confs"]
                    # Skip a first waypoint identical to the previous last one (the shared node).
                    skip = 1 if pieces and linf(confs[0], pieces[-1]) <= same else 0
                    pieces.extend(confs[skip:])
                    # Remember the last edge that left the roadmap's branch (ended away from its node).
                    if linf(pieces[-1], self.confs[b]) > same:
                        walk["drift_row"] = row
                    else:
                        walk["drift_row"] = None
                    walk["edge"] += 1
            if not needed:
                break
            keys = list(needed)
            self.retracks += len(keys)
            for key, record in zip(keys, self._run([needed[key] for key in keys], "re-track")):
                self.seeded[key] = record
        out = {}
        for key, walk in state.items():
            failed = walk["failed_row"]
            blame = None if failed is None else (walk["drift_row"] if walk["drift_row"] is not None else failed)
            out[key] = (walk["pieces"] if failed is None else None,
                        {"joins": walk["joins"], "failed_row": failed, "blame_row": blame})
        return out

    def assemble(self, route: Sequence[int]) -> Tuple[Optional[List[List[float]]], Dict[str, Any]]:
        """Join the tracked edges of one route into a chain (see :meth:`assemble_many`)."""
        return self.assemble_many({0: route})[0]

    # * ---- The exact-gap loop -----------------------------------------------

    def exact_gaps(self, start_of: Dict[int, int], goal_ids: Sequence[int],
                   max_gap_mm: float) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[int, Any], int]:
        """Track routes until every start's best route is fully tracked, then assemble them.

        Args:
            start_of (Dict[int, int]): node id -> start index.
            goal_ids (Sequence[int]): goal nodes.
            max_gap_mm (float): starts whose current gap is above this are not
                tracked further.

        Returns:
            Tuple: ``gap`` and ``parent`` (minimax over trackable edges), the
            best node of each start, the assembled route of each start within
            ``max_gap_mm`` (``{start: (route, confs, info)}``), and the number
            of tracking rounds.
        """
        rounds = 0
        while True:
            # * Track every untracked edge on the current best routes; repeat.
            while True:
                gap, parent = minimax_to_goals(len(self.confs), self.table, self.table.free, goal_ids)
                best = best_node_per_start(start_of, gap)
                rows = set()
                for node_id in best.values():
                    if not np.isfinite(gap[node_id]) or gap[node_id] > max_gap_mm:
                        continue
                    route = route_from(node_id, parent)
                    for a, b in zip(route[:-1], route[1:]):
                        row = self.row_of[(min(a, b), max(a, b))]
                        if not self.table.checked[row]:
                            rows.add(row)
                if not rows:
                    break
                rounds += 1
                print(f"  [exact gaps] round {rounds}: tracking {len(rows)} edges "
                      f"({len(self.records)} tracked so far)", flush=True)
                self.track_rows(sorted(rows))

            # * Join each fully tracked route into one chain.
            routes = {start_index: route_from(node_id, parent) for start_index, node_id in best.items()
                      if np.isfinite(gap[node_id]) and gap[node_id] <= max_gap_mm}
            assembled: Dict[int, Any] = {}
            failed = set()
            for start_index, (confs, info) in self.assemble_many(routes).items():
                if confs is None:
                    failed.add(info["blame_row"])
                else:
                    assembled[start_index] = (routes[start_index], confs, info)
            if not failed:
                return gap, parent, best, assembled, rounds
            # * A route could not be continued: drop the edge that left the
            # * roadmap's branch on the way (or the one it could not cross) and search again.
            for row in failed:
                self.table.free[row] = False
                self.join_blocked.append(int(row))

    def shortest_exact(self, start_ids: Sequence[int], goal_ids: Sequence[int],
                       tolerance_mm: float) -> Optional[Tuple[List[int], List[List[float]], Dict[str, Any]]]:
        """The shortest route within a tolerance whose edges all track, assembled.

        Args:
            start_ids (Sequence[int]): start nodes.
            goal_ids (Sequence[int]): goal nodes.
            tolerance_mm (float): edges with a larger score are not used.

        Returns:
            Optional[Tuple]: (route, waypoints, assembly info), or None.
        """
        while True:
            def find_path(mask: np.ndarray) -> Optional[List[int]]:
                within = mask & (self.table.score <= tolerance_mm)
                found = shortest_between_sets(build_graph(self.table, within, self.table.len_l2, start_ids, goal_ids))
                return found[0] if found else None

            path, _tally = lazy_search(find_path, self.table, self.track_rows)
            if path is None:
                return None
            confs, info = self.assemble(path)
            if confs is not None:
                return path, confs, info
            self.table.free[info["blame_row"]] = False
            self.join_blocked.append(int(info["blame_row"]))


# -----------------------------------------------------------------------------
# * Tube draws around routes (way 2)
# -----------------------------------------------------------------------------


def tube_points(tracker: Tracker, routes: Sequence[Sequence[int]],
                target_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    """Points laid along the joint lines of some routes, weighted toward their bad edges.

    Args:
        tracker (Tracker): the graph (configurations, edges, verdicts).
        routes (Sequence[Sequence[int]]): node-id routes.
        target_mm (float): an edge scoring above this counts as bad.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (P, 12) points and their weights.
    """
    points, weights = [], []
    step = float(np.radians(TUBE_POINT_STEP_DEG))
    for route in routes:
        for a, b in zip(route[:-1], route[1:]):
            row = tracker.row_of[(min(a, b), max(a, b))]
            bad = tracker.table.score[row] > target_mm or (tracker.table.checked[row] and not tracker.table.free[row])
            line = interpolate_confs(tracker.confs[a], tracker.confs[b], step, include_ends=True)
            points.extend(line)
            weights.extend([TUBE_BAD_EDGE_WEIGHT if bad else 1.0] * len(line))
    return np.asarray(points, dtype=float).reshape(-1, 12), np.asarray(weights, dtype=float)


def tube_round(tracker: Tracker, routes: Sequence[Sequence[int]], round_number: int,
               n_draws: int, counters: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Draw nodes around the given routes, join them with short edges, add both to the graph.

    Args:
        tracker (Tracker): the graph under refinement.
        routes (Sequence[Sequence[int]]): the routes to draw around.
        round_number (int): which round (seeds the draws).
        n_draws (int): driver draws in all.
        counters (Dict[str, Dict[str, float]]): per-phase tallies to add to.

    Returns:
        Dict[str, Any]: what the round made (draws, nodes, edges, the new node
        configurations).
    """
    settings = tracker.settings
    points, weights = tube_points(tracker, routes, settings.tube_target_mm)
    sampler = SamplerConfig(
        anchors=points, anchor_roles=["tube"] * len(points), mode_mix={"path_tube": 1.0},
        box_margin_rad=0.0, gauss_sigma_rad=0.0, line_sigma_rad=0.0, max_branches=1,
        tube_points=points, tube_weights=weights, tube_sigma_rad=float(np.radians(settings.tube_sigma_deg)),
    ).to_dict()
    n_chunks = max(1, int(np.ceil(n_draws / TUBE_CHUNK)))
    # Chunk indices far from the build's own, so the random streams never repeat.
    tasks = [{"chunk": c, "chunk_index": 1_000_000 * round_number + c, "seed": settings.seed,
              "n_draws": TUBE_CHUNK, "sampler": sampler} for c in range(n_chunks)]
    nodes = []
    for result in tracker.pool.run(task_sample_with, tasks, f"tube draws {round_number}"):
        nodes.extend(result["nodes"])
        add_counters(counters["tube_sample"], result["counters"])
    if not nodes:
        return {"round": round_number, "draws": n_chunks * TUBE_CHUNK, "nodes": 0, "edges": 0, "confs": []}

    first_new = len(tracker.confs)
    new_confs = np.asarray([node["conf"] for node in nodes], dtype=float)
    all_confs = np.vstack([tracker.confs, new_confs])
    new_ids = list(range(first_new, len(all_confs)))
    pairs = candidate_pairs(all_confs, settings.tube_k, float(np.radians(settings.tube_radius_deg)), query_ids=new_ids)
    records = []
    # Violation only: the tracker does the collision checks that count, on the solved waypoints.
    for result in tracker.pool.run(task_edges, edge_tasks(all_confs, pairs, 256, False), f"tube edges {round_number}"):
        add_counters(counters["tube_edges"], result["counters"])
        records.extend(result["edges"])
    tracker.add_nodes_and_edges(new_confs, records)
    return {"round": round_number, "draws": n_chunks * TUBE_CHUNK, "nodes": len(nodes), "edges": len(records),
            "confs": np.round(new_confs, 6).tolist()}


# -----------------------------------------------------------------------------
# * The refinement run
# -----------------------------------------------------------------------------


def refine_path(root: str, run_name: str, tag: str) -> str:
    """The file of one refinement: ``<root>/prm/<run>/refine/<tag>.json.gz``."""
    return os.path.join(run_dir(root, run_name), "refine", f"{tag}.json.gz")


def refine_query(root: str, settings: RefineSettings, verbose: bool = True) -> Dict[str, Any]:
    """Turn a saved query's roadmap routes into exact paths, and save the result.

    Args:
        root (str): the benchmark data root.
        settings (RefineSettings): what to do.
        verbose (bool): print progress and the summary.

    Returns:
        Dict[str, Any]: the saved record.
    """
    wall_start = perf_counter()
    say = print if verbose else (lambda *args, **kwargs: None)
    tag = settings.tag or settings.default_tag()
    graph = load_query_graph(root, settings.run_name, settings.query_tag)
    query = graph.query
    model = graph.model
    timings: Dict[str, float] = {"load_s": perf_counter() - wall_start}

    # * The starts worth refining: those the roadmap already brings within max_gap_mm.
    roadmap_gap = {int(row["start_index"]): row["best_gap_mm"] for row in query["starts"]}
    considered = sorted(s for s, g in roadmap_gap.items() if g is not None and g <= settings.max_gap_mm)
    start_of = start_nodes(query, considered)
    start_ids = sorted(start_of)
    labels = {int(row["start_index"]): row for row in query["starts"]}
    say(f"[prm refine] {settings.run_name} / {settings.query_tag} -> {tag}: {len(considered)} of "
        f"{len(roadmap_gap)} starts within {settings.max_gap_mm:g} mm on the roadmap, "
        f"{len(start_ids)} start nodes, {len(graph.goal_ids)} goal nodes, {len(graph.confs)} nodes, "
        f"{len(graph.table.i)} edges")

    counters: Dict[str, Dict[str, float]] = {"tube_sample": {}, "tube_edges": {}}
    tube_log: List[Dict[str, Any]] = []
    build = graph.build
    # * Each start's best route on the roadmap as the query left it (before any
    # * tracking). The first tracking round tracks exactly these routes, so every
    # * edge on them gets a verdict; the seeded BiRRT (way 1) starts from them.
    gap0, parent0 = minimax_to_goals(len(graph.confs), graph.table, graph.table.usable(optimistic=True), graph.goal_ids)
    best0 = best_node_per_start(start_of, gap0)

    with WorkerPool(root, build["source_key"], build["bar_id"], settings.workers, None,
                    build["edge_settings"], verbose) as pool:
        # Start every worker first, so the phase times below are the refinement's own.
        timings["pool_startup_s"] = pool.startup_s + pool.warm_up()
        tracker = Tracker(graph, settings, pool)
        if settings.pretracked:
            # A path, or a file name inside the roadmap's own folder (e.g. tracked_10mm.json.gz).
            pretracked = settings.pretracked
            if not os.path.isfile(pretracked):
                pretracked = os.path.join(run_dir(root, settings.run_name), settings.pretracked)
            n_loaded = tracker.load_pretracked(pretracked)
            say(f"  pretracked: {n_loaded} roadmap edges' verdicts loaded from {pretracked}")

        # * 0. Optional unbiased sample of edges, for trackability statistics.
        phase = perf_counter()
        if settings.edge_sample > 0:
            rng = np.random.default_rng(settings.seed)
            pool_rows = np.flatnonzero(tracker.table.free & (tracker.table.score <= settings.max_gap_mm))
            chosen = rng.choice(pool_rows, size=min(settings.edge_sample, len(pool_rows)), replace=False)
            say(f"  tracking a random sample of {len(chosen)} edges")
            tracker.track_rows(sorted(int(r) for r in chosen))
        timings["edge_sample_s"] = perf_counter() - phase

        # * 1-2. Exact gaps and assembled routes.
        phase = perf_counter()
        gap, parent, best, assembled, rounds = tracker.exact_gaps(start_of, graph.goal_ids, settings.max_gap_mm)
        timings["exact_gaps_s"] = perf_counter() - phase
        say(f"  exact gaps: {rounds} rounds, {len(tracker.records)} edges tracked, "
            f"{sum(1 for r in tracker.records.values() if r['free'])} trackable, "
            f"{len(assembled)} starts with an exact route within {settings.max_gap_mm:g} mm")

        # * 4. Tube rounds around the routes of starts still above the target.
        phase = perf_counter()
        idle = 0
        for round_number in range(1, settings.tube_rounds + 1):
            targets = sorted(
                (s for s in considered
                 if s not in assembled or gap[best[s]] > settings.tube_target_mm),
                key=lambda s: gap[best[s]] if s in best else np.inf)[: settings.tube_max_starts]
            routes = [route_from(best[s], parent) for s in targets if s in best and np.isfinite(gap[best[s]])]
            if not routes:
                say("  tube: no start left above the target with any route")
                break
            before = {s: float(gap[best[s]]) for s in targets if s in best}
            made = tube_round(tracker, routes, round_number, settings.tube_draws_per_start * len(routes), counters)
            gap, parent, best, assembled, more_rounds = tracker.exact_gaps(start_of, graph.goal_ids,
                                                                            settings.max_gap_mm)
            after = {s: float(gap[best[s]]) for s in before}
            improved = [s for s in before if after[s] < before[s] - 1e-9]
            made.update({"targets": targets, "gap_before_mm": before, "gap_after_mm": after,
                         "improved": improved, "track_rounds": more_rounds})
            tube_log.append(made)
            say(f"  tube round {round_number}: {made['nodes']} nodes, {made['edges']} edges; "
                f"{len(improved)} of {len(before)} targets improved, "
                f"{sum(1 for s in before if after[s] <= settings.tube_target_mm)} now within "
                f"{settings.tube_target_mm:g} mm")
            idle = 0 if improved else idle + 1
            if idle >= settings.tube_patience:
                break
        timings["tube_s"] = perf_counter() - phase

        # * 3. Per tolerance: how many starts have an exact route, and the shortest one.
        phase = perf_counter()
        answers = []
        for tolerance in settings.tolerances_mm:
            found = tracker.shortest_exact(start_ids, graph.goal_ids, tolerance) if tolerance <= settings.max_gap_mm else None
            answer: Dict[str, Any] = {
                "tolerance_mm": tolerance,
                "starts_roadmap": int(sum(1 for s in considered if roadmap_gap[s] <= tolerance)),
                "starts_exact": int(sum(1 for s in assembled if gap[best[s]] <= tolerance)),
                "shortest": None,
            }
            if found:
                route, confs, info = found
                answer["shortest"] = {"start_index": start_of[route[0]], "route": route,
                                      "route_scores_mm": route_scores(tracker, route), **info,
                                      **exact_path_report(model, confs)}
            answers.append(answer)
        timings["tolerances_s"] = perf_counter() - phase

        # * Starts the roadmap brings within unresolved_mm but with no exact route
        # * there: their original route, edge by edge, for the seeded BiRRT.
        unresolved = {}
        waiting = sorted((s for s in considered if roadmap_gap[s] <= settings.unresolved_mm
                          and not (s in assembled and gap[best[s]] <= settings.unresolved_mm)),
                         key=lambda s: roadmap_gap[s])[: settings.unresolved_max]
        for s in waiting:
            route = route_from(best0[s], parent0)
            tracker.track_rows([tracker.row_of[(min(a, b), max(a, b))] for a, b in zip(route[:-1], route[1:])])
            unresolved[str(s)] = route_attempt(tracker, route)

    # The route each unresolved start's BiRRT is guided along.
    for attempt in unresolved.values():
        attempt["guide_confs"] = np.round(tracker.confs[attempt["route"]], 7).tolist()

    # * The per-start table, the stored paths and the headline.
    start_rows, paths = [], {}
    for s in considered:
        node = best.get(s)
        exact = float(gap[node]) if node is not None and s in assembled else None
        row = {"start_index": s, "carry": labels[s].get("carry"), "difficulty": labels[s].get("difficulty"),
               "roadmap_gap_mm": roadmap_gap[s], "exact_gap_mm": optional_float(exact)}
        if s in assembled:
            route, confs, info = assembled[s]
            report = exact_path_report(model, confs, store_confs=exact <= settings.store_paths_mm)
            row.update({"route_nodes": len(route), "n_waypoints": report["n_waypoints"], "joins": info["joins"],
                        "max_step_deg": report["max_step_deg"], "path_stored": "confs" in report})
            if "confs" in report:
                paths[str(s)] = {"start_index": s, "route": route, "route_scores_mm": route_scores(tracker, route),
                                 **info, **report}
        start_rows.append(row)
    exact_rows = [r for r in start_rows if r["exact_gap_mm"] is not None]
    headline: Dict[str, Any] = {"best_exact_gap_mm": None}
    if exact_rows:
        best_row = min(exact_rows, key=lambda r: r["exact_gap_mm"])
        s = best_row["start_index"]
        route, confs, info = assembled[s]
        headline = {"best_exact_gap_mm": best_row["exact_gap_mm"], "best_start_index": s,
                    "roadmap_gap_mm": roadmap_gap[s], "route": route, "route_scores_mm": route_scores(tracker, route),
                    **info, **exact_path_report(model, confs)}

    timings["total_s"] = perf_counter() - wall_start
    record = {
        "schema_version": SCHEMA_VERSION, "tag": tag, "run_name": settings.run_name,
        "query_tag": settings.query_tag, "bar_id": build["bar_id"], "library": build["library"],
        "goal_index": query["goal_index"], "created": time(),
        "settings": asdict(settings), "track_settings": tracker.track_settings.to_dict(),
        "rot_deg_per_mm": query["rot_deg_per_mm"],
        "n_starts_considered": len(considered), "n_starts_query": len(roadmap_gap),
        "headline": headline, "tolerances": answers, "starts": start_rows, "paths": paths,
        "unresolved": unresolved,
        "by_carry": carry_summary(start_rows, settings.tolerances_mm),
        "edges": edge_table(tracker),
        "join_blocked": tracker.join_blocked, "retracks": tracker.retracks,
        "tube": tube_log, "exact_gap_rounds": rounds,
        "counters": {"track": tracker.counters, **counters}, "timings": timings,
        "provenance": provenance_record(),
    }
    write_json_atomic(refine_path(root, settings.run_name, tag), record)
    update_index(os.path.join(run_dir(root, settings.run_name), "refine", "index.json"), "tag", {
        "updated": time(), "tag": tag, "query_tag": settings.query_tag, "file": f"{tag}.json.gz",
        "best_exact_gap_mm": headline.get("best_exact_gap_mm"), "best_start_index": headline.get("best_start_index"),
        "starts_exact": {str(a["tolerance_mm"]): a["starts_exact"] for a in answers},
    }, "refines")
    if verbose:
        print_refine_summary(record)
    return record


def pretrack_roadmap(root: str, run_name: str, max_score_mm: float, track_settings: TrackSettings,
                     workers: int, verbose: bool = True) -> str:
    """Track every collision-free roadmap edge within a score, once, and save the verdicts (overnight work).

    Roadmap edges come first in every query's edge table, in the same order,
    so these verdicts serve every later query of the roadmap: a refinement
    then only tracks the query's own endpoint edges.

    Args:
        root (str): the benchmark data root.
        run_name (str): the roadmap run.
        max_score_mm (float): track edges whose score is at most this.
        track_settings (TrackSettings): the tracker settings (a refinement
            can only use the file with the same settings).
        workers (int): worker processes.
        verbose (bool): print progress.

    Returns:
        str: the file written: ``<root>/prm/<run>/tracked_<score>mm.json.gz``.
    """
    started = perf_counter()
    roadmap = load_roadmap(root, run_name)
    build = roadmap.config
    table = EdgeTable.from_columns([roadmap.edges], float(build["rot_deg_per_mm"]))
    rows = np.flatnonzero(table.free & table.checked & (table.score <= max_score_mm))
    confs = roadmap.confs()
    if verbose:
        print(f"[prm track] {run_name}: tracking {len(rows)} of {len(table.i)} roadmap edges "
              f"(collision-free, score <= {max_score_mm:g} mm), {workers} workers")
    records, counters = [], {}
    tasks = []
    for number, piece in enumerate(chunked(rows.tolist(), TRACK_CHUNK)):
        tasks.append({"chunk": number, "rows": piece,
                      "confs_a": [confs[table.i[r]].tolist() for r in piece],
                      "confs_b": [confs[table.j[r]].tolist() for r in piece],
                      "seeds": None, "settings": track_settings.to_dict()})
    with WorkerPool(root, build["source_key"], build["bar_id"], workers, None, build["edge_settings"],
                    verbose) as pool:
        for result in pool.run(task_track, tasks, "pretrack"):
            add_counters(counters, result["counters"])
            for record in result["records"]:
                for key in ("confs", "confs_back"):
                    record[key] = np.round(np.asarray(record.get(key, [])), 7).tolist()
                records.append(record)
    path = os.path.join(run_dir(root, run_name), f"tracked_{max_score_mm:g}mm.json.gz")
    write_json_atomic(path, {
        "run_name": run_name, "created": time(), "max_score_mm": max_score_mm,
        "track_settings": track_settings.to_dict(), "n_roadmap_edges": len(table.i),
        "records": records, "counters": counters, "wall_s": perf_counter() - started,
        "provenance": provenance_record(),
    })
    if verbose:
        n_free = sum(1 for r in records if r["free"])
        print(f"[prm track] done in {perf_counter() - started:.0f} s: {n_free} of {len(records)} edges track "
              f"exactly; wrote {path}")
    return path


def route_scores(tracker: Tracker, route: Sequence[int]) -> List[float]:
    """The roadmap score (mm) of each edge of a route."""
    return [float(tracker.table.score[tracker.row_of[(min(a, b), max(a, b))]]) for a, b in zip(route[:-1], route[1:])]


def route_attempt(tracker: Tracker, route: Sequence[int]) -> Dict[str, Any]:
    """A route edge by edge: score, tracking verdict, and the exact waypoints where it tracked.

    Args:
        tracker (Tracker): the graph with its tracking verdicts (every edge of
            the route must be tracked).
        route (Sequence[int]): node ids, start first.

    Returns:
        Dict[str, Any]: ``route``; ``edges`` (per edge: ``row``, ``score_mm``,
        ``free`` = tracked exactly, ``reason``); ``pieces`` (per edge, its
        exact waypoints in route direction, or None where it failed).
    """
    edges, pieces = [], []
    for a, b in zip(route[:-1], route[1:]):
        row = tracker.row_of[(min(a, b), max(a, b))]
        record = tracker.records[row]
        edges.append({"row": int(row), "score_mm": float(tracker.table.score[row]),
                      "free": bool(record["free"]), "reason": record["reason"],
                      "fail_fraction": record["fail_fraction"]})
        confs = oriented_confs(record, int(tracker.table.i[row]) == a) if record["free"] else []
        if confs:
            pieces.append(np.round(np.asarray(confs), 7).tolist())
        else:
            pieces.append(None)
    return {"route": [int(n) for n in route], "edges": edges, "pieces": pieces}


def edge_table(tracker: Tracker) -> Dict[str, list]:
    """Every tracked edge's verdict, as columns (no waypoints)."""
    rows = sorted(tracker.records)
    table = tracker.table
    out: Dict[str, list] = {name: [] for name in (
        "row", "i", "j", "kind", "score_mm", "len_linf_deg", "free", "reason", "fail_fraction",
        "n_waypoints", "n_knots", "max_dev_deg", "max_step_deg", "seconds")}
    for row in rows:
        record = tracker.records[row]
        out["row"].append(row)
        out["i"].append(int(table.i[row]))
        out["j"].append(int(table.j[row]))
        out["kind"].append(tracker.kinds[row])
        out["score_mm"].append(round(float(table.score[row]), 4))
        out["len_linf_deg"].append(round(float(np.degrees(linf(tracker.confs[table.i[row]], tracker.confs[table.j[row]]))), 3))
        out["free"].append(bool(record["free"]))
        out["reason"].append(record["reason"])
        out["fail_fraction"].append(round(record["fail_fraction"], 3))
        out["n_waypoints"].append(record["n_waypoints"])
        out["n_knots"].append(record["n_knots"])
        out["max_dev_deg"].append(round(float(np.degrees(record["max_dev_rad"])), 4))
        out["max_step_deg"].append(round(float(np.degrees(record["max_step_rad"])), 4))
        out["seconds"].append(round(record["seconds"], 4))
    return out


def carry_summary(rows: Sequence[Dict[str, Any]], tolerances_mm: Sequence[float]) -> Dict[str, Any]:
    """Per carry anchor: starts considered, and how many have an exact route within each tolerance."""
    summary = {}
    for carry in sorted({r["carry"] for r in rows}) + ["all"]:
        members = [r for r in rows if carry == "all" or r["carry"] == carry]
        exact = [r["exact_gap_mm"] for r in members if r["exact_gap_mm"] is not None]
        summary[carry] = {
            "n": len(members), "n_exact": len(exact),
            "median_exact_gap_mm": float(np.median(exact)) if exact else None,
            "exact_within": {str(t): int(sum(1 for g in exact if g <= t)) for t in tolerances_mm},
            "roadmap_within": {str(t): int(sum(1 for r in members if r["roadmap_gap_mm"] <= t)) for t in tolerances_mm},
        }
    return summary


def print_refine_summary(record: Dict[str, Any]) -> None:
    """Print what a refinement found."""
    timings = record["timings"]
    edges = record["edges"]
    n_edges = len(edges["row"])
    n_free = sum(edges["free"])
    reasons: Dict[str, int] = {}
    for reason in edges["reason"]:
        reasons[reason] = reasons.get(reason, 0) + 1
    print(f"[prm refine] done in {timings['total_s']:.0f} s: {n_edges} edges tracked, {n_free} trackable "
          f"({reasons}); {record['retracks']} re-tracks at joins, {len(record['join_blocked'])} edges dropped there")
    headline = record["headline"]
    if headline.get("best_exact_gap_mm") is None:
        print("  no start has an exact route within the considered gap")
    else:
        print(f"  best exact route: start {headline['best_start_index']}, roadmap tolerance "
              f"{headline['best_exact_gap_mm']:.2f} mm (its roadmap gap {headline['roadmap_gap_mm']:.2f} mm), "
              f"{headline['n_waypoints']} waypoints, largest joint step {headline['max_step_deg']:.2f} deg, "
              f"holding error {headline['max_viol_pos_mm']:.5f} mm")
    print("  tolerance  roadmap-starts  exact-starts  shortest exact route")
    for answer in record["tolerances"]:
        shortest = answer["shortest"]
        text = (f"start {shortest['start_index']}, {len(shortest['route'])} nodes, {shortest['n_waypoints']} waypoints"
                if shortest else "none")
        print(f"  {answer['tolerance_mm']:>6.1f} mm  {answer['starts_roadmap']:>10d}  {answer['starts_exact']:>12d}"
              f"   {text}")
    for carry, stats in record["by_carry"].items():
        print(f"  {carry:>10s}: {stats['n']} starts, exact within {stats['exact_within']} "
              f"(roadmap within {stats['roadmap_within']})")
