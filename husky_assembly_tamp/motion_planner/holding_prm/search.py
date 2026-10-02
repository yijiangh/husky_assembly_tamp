"""Searching the roadmap from a SET of starts to a SET of goals.

The query does not fix one start or one goal configuration. A virtual
"source" node is joined, at zero cost, to every start node, and every goal
node is joined, at zero cost, to a virtual "sink"; one search from source to
sink then chooses the start pose, the start branch pair and the goal branch
pair together.

Three questions are asked of the same graph:

1. **Minimax (bottleneck).** What is the smallest tolerance at which ANY start
   connects to ANY goal? This is the minimum spanning tree trick: in a
   minimum spanning tree, the tree path between two nodes has the smallest
   possible largest edge. With the sink joined to the goals, one tree gives,
   for EVERY node at once, the smallest largest violation of any route from
   that node to the goal set (its "gap") and the next node on that route.
2. **Filtered.** Under a tolerance t, keep only edges whose worst violation is
   at most t, and find the shortest (joint-space length) path.
3. **Penalised.** Keep every free edge, add a large penalty for violation over
   t, and find the cheapest path -- always returns a path when the free graph
   connects the sets, and says how far over the tolerance it had to go.

Edge weights come from the raw numbers stored on the edges, so every
tolerance is a re-search, never a re-measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree

from husky_assembly_tamp.motion_planner.holding_prm.holding import (
    HoldingModel,
    holding_violation,
    interpolate_confs,
    l2,
    linf,
    violation_score_mm,
)
from husky_assembly_tamp.motion_planner.holding_prm.roadmap import edge_cost


# The two virtual nodes. Strings, so they can never clash with a node id.
SOURCE = "SOURCE"
SINK = "SINK"


@dataclass
class EdgeTable:
    """All measured edges of one query (roadmap edges plus endpoint edges), as arrays.

    Attributes:
        i (np.ndarray): one end of each edge.
        j (np.ndarray): the other end.
        len_l2 (np.ndarray): joint-space length (radians, Euclidean).
        free (np.ndarray): collision-free (meaningful where ``checked``).
        checked (np.ndarray): whether the collision check has been run.
        viol_pos_max (np.ndarray): worst position error along the edge (m).
        viol_rot_max (np.ndarray): worst rotation error along the edge (rad).
        score (np.ndarray): worst violation folded into millimetres.
    """

    i: np.ndarray
    j: np.ndarray
    len_l2: np.ndarray
    free: np.ndarray
    checked: np.ndarray
    viol_pos_max: np.ndarray
    viol_rot_max: np.ndarray
    score: np.ndarray

    @classmethod
    def from_columns(cls, column_sets: Sequence[Dict[str, list]], rot_deg_per_mm: float) -> "EdgeTable":
        """Join several edge column dicts (roadmap, endpoint edges) into one table.

        Args:
            column_sets (Sequence[Dict[str, list]]): dicts of edge columns.
            rot_deg_per_mm (float): see :func:`holding.violation_score_mm`.

        Returns:
            EdgeTable: the joined table.
        """
        def join(name: str, dtype) -> np.ndarray:
            parts = [np.asarray(columns[name], dtype=dtype) for columns in column_sets if len(columns[name])]
            return np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)

        viol_pos = join("viol_pos_max", float)
        viol_rot = join("viol_rot_max", float)
        return cls(
            i=join("i", int), j=join("j", int), len_l2=join("len_l2", float),
            free=join("free", bool), checked=join("checked", bool),
            viol_pos_max=viol_pos, viol_rot_max=viol_rot,
            score=np.maximum(viol_pos * 1000.0, np.degrees(viol_rot) / rot_deg_per_mm),
        )

    def usable(self, optimistic: bool = False) -> np.ndarray:
        """Which edges the search may use.

        Args:
            optimistic (bool): also use edges not yet collision-checked,
                assuming they are free (the lazy search does this).

        Returns:
            np.ndarray: a boolean mask.
        """
        return self.free & (self.checked | optimistic)

    def index_of(self) -> Dict[Tuple[int, int], int]:
        """Map each unordered node pair to its edge's row."""
        return {(min(a, b), max(a, b)): row for row, (a, b) in enumerate(zip(self.i.tolist(), self.j.tolist()))}


# -----------------------------------------------------------------------------
# * Building the graph and searching it
# -----------------------------------------------------------------------------


def build_graph(table: EdgeTable, mask: np.ndarray, weights: np.ndarray,
                start_ids: Sequence[int], goal_ids: Sequence[int]) -> nx.Graph:
    """A weighted graph of the usable edges, with the source and sink attached.

    Args:
        table (EdgeTable): the edges.
        mask (np.ndarray): which edges to include.
        weights (np.ndarray): one weight per edge.
        start_ids (Sequence[int]): nodes joined to the source at zero cost.
        goal_ids (Sequence[int]): nodes joined to the sink at zero cost.

    Returns:
        nx.Graph: the graph; each edge carries ``weight`` and ``row`` (its row
        in the table, -1 for the virtual edges).
    """
    graph = nx.Graph()
    for row in np.flatnonzero(mask):
        a, b, w = int(table.i[row]), int(table.j[row]), float(weights[row])
        # If a pair appears twice, keep the cheaper edge.
        if graph.has_edge(a, b) and graph[a][b]["weight"] <= w:
            continue
        graph.add_edge(a, b, weight=w, row=int(row))
    for node in start_ids:
        graph.add_edge(SOURCE, int(node), weight=0.0, row=-1)
    for node in goal_ids:
        graph.add_edge(int(node), SINK, weight=0.0, row=-1)
    return graph


def shortest_between_sets(graph: nx.Graph) -> Optional[Tuple[List[int], float]]:
    """The cheapest route from the source to the sink.

    Args:
        graph (nx.Graph): a graph from :func:`build_graph`.

    Returns:
        Optional[Tuple[List[int], float]]: the node ids from a start node to a
        goal node (virtual nodes removed) and the total weight, or None when
        no start is connected to any goal.
    """
    if SOURCE not in graph or SINK not in graph:
        return None
    try:
        path = nx.dijkstra_path(graph, SOURCE, SINK, weight="weight")
    except nx.NetworkXNoPath:
        return None
    return [int(node) for node in path[1:-1]], float(nx.path_weight(graph, path, weight="weight"))


def minimax_to_goals(n_nodes: int, table: EdgeTable, mask: np.ndarray,
                     goal_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """For every node, the smallest tolerance at which it reaches the goal set.

    Builds the minimum spanning tree of the usable edges weighted by their
    violation score, with the sink joined to every goal node at zero, and
    walks it outward from the sink. Uses scipy's sparse-graph routines: on
    the B18 all-starts graph (126k edges) about thirty times faster than the
    networkx version (:func:`minimax_to_goals_nx`, kept as the reference the
    tests compare against), and this runs once per verification round.

    Args:
        n_nodes (int): how many nodes (roadmap plus endpoints).
        table (EdgeTable): the edges.
        mask (np.ndarray): which edges are usable.
        goal_ids (Sequence[int]): the goal nodes.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``gap`` (mm, infinite when the node
        cannot reach any goal at all) and ``parent`` (the next node on the
        route toward the goals; -1 for goal nodes and unreachable nodes).
    """
    gap = np.full(n_nodes, np.inf)
    parent = np.full(n_nodes, -1, dtype=int)
    goals = np.unique(np.asarray(goal_ids, dtype=int))
    if len(goals) == 0:
        return gap, parent
    sink = n_nodes
    rows = np.flatnonzero(mask)
    a = np.concatenate([table.i[rows], goals])
    b = np.concatenate([table.j[rows], np.full(len(goals), sink)])
    # ! scipy reads a stored zero as "no edge", so every weight is lifted by a
    # ! tiny amount (the sink's zero-cost edges included); 1e-9 mm changes no gap.
    weight = np.concatenate([table.score[rows], np.zeros(len(goals))]) + 1e-9
    # One entry per unordered pair, keeping the cheapest (scipy would add duplicates up).
    low, high = np.minimum(a, b), np.maximum(a, b)
    order = np.lexsort((weight, high, low))
    low, high, weight = low[order], high[order], weight[order]
    first = np.ones(len(low), dtype=bool)
    first[1:] = (low[1:] != low[:-1]) | (high[1:] != high[:-1])
    low, high, weight = low[first], high[first], weight[first]
    size = n_nodes + 1
    graph = coo_matrix((weight, (low, high)), shape=(size, size)).tocsr()
    tree = minimum_spanning_tree(graph).tocoo()
    # The weight of each tree edge, looked up by its unordered pair.
    tree_keys = np.minimum(tree.row, tree.col).astype(np.int64) * size + np.maximum(tree.row, tree.col)
    key_order = np.argsort(tree_keys)
    tree_keys, tree_weights = tree_keys[key_order], tree.data[key_order] - 1e-9
    symmetric = coo_matrix((np.ones(2 * len(tree.row)), (np.concatenate([tree.row, tree.col]),
                                                        np.concatenate([tree.col, tree.row]))), shape=(size, size))
    # Breadth-first from the sink: each node's gap is the larger of its
    # parent's gap and the edge that joins them (parents come first in this order).
    order, predecessors = breadth_first_order(symmetric.tocsr(), sink, directed=False, return_predecessors=True)
    children = order[1:]
    ups = predecessors[children]
    edge_keys = np.minimum(children, ups).astype(np.int64) * size + np.maximum(children, ups)
    edge_weights = tree_weights[np.searchsorted(tree_keys, edge_keys)].tolist()
    best = [np.inf] * size
    best[sink] = 0.0
    for node, up, w in zip(children.tolist(), ups.tolist(), edge_weights):
        best[node] = max(best[up], w)
    best = np.asarray(best)
    reached = children[children < n_nodes]
    gap[reached] = best[reached]
    parent[reached] = np.where(predecessors[reached] < n_nodes, predecessors[reached], -1)
    return gap, parent


def minimax_to_goals_nx(n_nodes: int, table: EdgeTable, mask: np.ndarray,
                        goal_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """The networkx version of :func:`minimax_to_goals` (the reference the tests compare against).

    Builds the minimum spanning tree of the usable edges weighted by their
    violation score, with the sink joined to every goal node at zero, and
    walks it outward from the sink.

    Args:
        n_nodes (int): how many nodes (roadmap plus endpoints).
        table (EdgeTable): the edges.
        mask (np.ndarray): which edges are usable.
        goal_ids (Sequence[int]): the goal nodes.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``gap`` (mm, infinite when the node
        cannot reach any goal at all) and ``parent`` (the next node on the
        route toward the goals; -1 for goal nodes and unreachable nodes).
    """
    graph = build_graph(table, mask, table.score, start_ids=[], goal_ids=goal_ids)
    gap = np.full(n_nodes, np.inf)
    parent = np.full(n_nodes, -1, dtype=int)
    if SINK not in graph:
        return gap, parent
    tree = nx.minimum_spanning_tree(graph, weight="weight", algorithm="kruskal")
    # Breadth-first from the sink: each node's gap is the larger of its
    # parent's gap and the edge that joins them.
    best = {SINK: 0.0}
    for up, down in nx.bfs_edges(tree, SINK):
        best[down] = max(best[up], float(tree[up][down]["weight"]))
        if isinstance(down, int):
            gap[down] = best[down]
            parent[down] = up if isinstance(up, int) else -1
    return gap, parent


def route_from(node: int, parent: np.ndarray, limit: int = 100000) -> List[int]:
    """Follow the minimax tree from a node to the goal set.

    Args:
        node (int): where to start.
        parent (np.ndarray): from :func:`minimax_to_goals`.
        limit (int): a guard against a malformed tree.

    Returns:
        List[int]: the node ids, ending at a goal node.
    """
    route = [int(node)]
    while parent[route[-1]] >= 0 and len(route) < limit:
        route.append(int(parent[route[-1]]))
    return route


def lazy_search(find_path: Callable[[np.ndarray], Optional[List[int]]], table: EdgeTable,
                check_rows: Callable[[List[int]], List[Dict[str, Any]]],
                max_rounds: int = 200) -> Tuple[Optional[List[int]], Dict[str, int]]:
    """Search first, collision-check only the edges the answer uses, repeat.

    This is the lazy roadmap idea: assume every unchecked edge is free, find
    the best path, check just that path's unchecked edges, drop the ones that
    turn out to collide, and search again. When a path comes back with every
    edge checked and free, it is the best path of the true graph.

    Args:
        find_path (Callable[[np.ndarray], Optional[List[int]]]): runs one
            search given the usable-edge mask; returns node ids or None.
        table (EdgeTable): the edges; ``free`` and ``checked`` are updated.
        check_rows (Callable[[List[int]], List[Dict[str, Any]]]): collision
            checks the given rows, returning one edge record per row.
        max_rounds (int): give up after this many searches.

    Returns:
        Tuple[Optional[List[int]], Dict[str, int]]: the path (or None) and a
        tally of rounds and edges checked.
    """
    tally = {"rounds": 0, "edges_checked": 0, "edges_blocked": 0}
    index = table.index_of()
    while tally["rounds"] < max_rounds:
        tally["rounds"] += 1
        path = find_path(table.usable(optimistic=True))
        if path is None:
            return None, tally
        rows = [index[(min(a, b), max(a, b))] for a, b in zip(path[:-1], path[1:])]
        unchecked = [row for row in rows if not table.checked[row]]
        if not unchecked:
            return path, tally
        for row, record in zip(unchecked, check_rows(unchecked)):
            table.checked[row] = True
            table.free[row] = bool(record["free"])
            tally["edges_checked"] += 1
            tally["edges_blocked"] += int(not record["free"])
    return None, tally


# -----------------------------------------------------------------------------
# * Describing a path
# -----------------------------------------------------------------------------


def path_report(model: HoldingModel, confs: np.ndarray, path: Sequence[int], table: EdgeTable,
                viol_step_rad: float, rot_deg_per_mm: float) -> Dict[str, Any]:
    """Everything worth knowing about one path, including its dense violation profile.

    Args:
        model (HoldingModel): the holding model.
        confs (np.ndarray): (N, 12) configurations of every node id in use.
        path (Sequence[int]): node ids, start first.
        table (EdgeTable): the edges (to read each step's stored numbers).
        viol_step_rad (float): spacing of the dense points.
        rot_deg_per_mm (float): see :func:`holding.violation_score_mm`.

    Returns:
        Dict[str, Any]: ``nodes``; per edge ``edge_scores`` (mm) and
        ``edge_lengths``; the dense path ``dense_confs`` with ``dense_s``
        (joint-space distance travelled, radians) and ``profile_pos_mm`` /
        ``profile_rot_deg`` / ``profile_score_mm``; ``max_score_mm``,
        ``max_pos_mm``, ``max_rot_deg`` and ``argmax_s`` (where the worst point
        is); ``length_rad``; ``max_step_rad`` (largest joint move between dense
        points, a check that continuity holds by construction).
    """
    index = table.index_of()
    rows = [index[(min(a, b), max(a, b))] for a, b in zip(path[:-1], path[1:])]
    dense = [np.asarray(confs[path[0]], dtype=float)]
    for a, b in zip(path[:-1], path[1:]):
        segment = interpolate_confs(confs[a], confs[b], viol_step_rad, include_ends=True)
        dense.extend(segment[1:])
    dense = np.asarray(dense)

    distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    pos_mm, rot_deg = [], []
    for conf in dense:
        pos, rot = holding_violation(model, conf)
        pos_mm.append(pos * 1000.0)
        rot_deg.append(float(np.degrees(rot)))
    score = [violation_score_mm(p / 1000.0, np.radians(r), rot_deg_per_mm) for p, r in zip(pos_mm, rot_deg)]
    worst = int(np.argmax(score)) if score else 0
    steps = np.max(np.abs(np.diff(dense, axis=0)), axis=1) if len(dense) > 1 else np.zeros(1)
    return {
        "nodes": [int(n) for n in path],
        "edge_rows": rows,
        "edge_scores": [float(table.score[row]) for row in rows],
        "edge_lengths": [float(table.len_l2[row]) for row in rows],
        "length_rad": float(sum(l2(confs[a], confs[b]) for a, b in zip(path[:-1], path[1:]))),
        "max_edge_linf_rad": float(max([linf(confs[a], confs[b]) for a, b in zip(path[:-1], path[1:])] or [0.0])),
        "max_step_rad": float(steps.max()),
        "dense_confs": np.round(dense, 6).tolist(),
        "dense_s": np.round(distance, 5).tolist(),
        "profile_pos_mm": np.round(pos_mm, 4).tolist(),
        "profile_rot_deg": np.round(rot_deg, 4).tolist(),
        "profile_score_mm": np.round(score, 4).tolist(),
        "max_score_mm": float(score[worst]) if score else 0.0,
        "max_pos_mm": float(max(pos_mm)) if pos_mm else 0.0,
        "max_rot_deg": float(max(rot_deg)) if rot_deg else 0.0,
        "argmax_s": float(distance[worst]) if len(distance) else 0.0,
    }


def tolerance_answers(table: EdgeTable, mask: np.ndarray, start_ids: Sequence[int],
                      goal_ids: Sequence[int], tolerance_mm: float,
                      w_hold: float) -> Dict[str, Any]:
    """The filtered and the penalised search under one tolerance.

    Args:
        table (EdgeTable): the edges.
        mask (np.ndarray): usable edges.
        start_ids (Sequence[int]): start nodes.
        goal_ids (Sequence[int]): goal nodes.
        tolerance_mm (float): the tolerance.
        w_hold (float): the penalty weight.

    Returns:
        Dict[str, Any]: ``filtered`` and ``penalised``, each with ``found``,
        ``path`` (node ids) and ``cost``.
    """
    answers: Dict[str, Any] = {"tolerance_mm": float(tolerance_mm)}
    within = mask & (table.score <= tolerance_mm)
    found = shortest_between_sets(build_graph(table, within, table.len_l2, start_ids, goal_ids))
    answers["filtered"] = {"found": found is not None, "path": found[0] if found else None,
                           "cost": found[1] if found else None}
    weights = edge_cost(table.len_l2, table.score, tolerance_mm, w_hold)
    found = shortest_between_sets(build_graph(table, mask, weights, start_ids, goal_ids))
    answers["penalised"] = {"found": found is not None, "path": found[0] if found else None,
                            "cost": found[1] if found else None}
    return answers
