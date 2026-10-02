"""The roadmap: its nodes, which of them are joined, and what each edge costs.

An edge is the straight line in joint space between two nodes. Because it is
straight, no joint ever jumps along it: cut into pieces of at most 10 degrees
per joint, every piece would pass the production continuity gate. What an
edge can do wrong is the other two things:

* touch something (checked at every piece boundary, with the full-cell
  collision check; one touch rejects the edge), and
* let the two flanges drift apart from the bar -- the holding violation --
  because both ends hold the bar exactly but the points in between need not.

The violation is measured more finely than collision (default every 2.5
degrees) because it is nearly free: two forward kinematics calls per point.
Both raw numbers, position and rotation, are kept on the edge, so the same
roadmap can be searched again under any tolerance without recomputing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from husky_assembly_tamp.motion_planner.holding_prm.holding import (
    HoldingModel,
    holding_violation,
    interpolate_confs,
    l2,
    linf,
    violation_score_mm,
)
from husky_assembly_tamp.motion_planner.holding_prm.sampling import WorkerTools, count


# The columns every edge record carries, in the order they are stored.
EDGE_COLUMNS = (
    "i", "j", "len_l2", "len_linf", "n_checks_planned", "checked", "free", "t_collide",
    "viol_pos_max", "viol_rot_max", "t_viol_max", "n_viol_points",
)

# The columns every node carries.
NODE_COLUMNS = (
    "conf", "tool_left", "tool_right", "ee_mid", "bar_pos", "bar_quat",
    "viol_pos", "viol_rot", "kind", "mode", "driver", "anchor", "draw",
)


# -----------------------------------------------------------------------------
# * Which nodes to try joining
# -----------------------------------------------------------------------------


def candidate_pairs(confs: np.ndarray, k: int, radius_rad: float,
                    query_ids: Optional[Sequence[int]] = None) -> np.ndarray:
    """The pairs of nodes worth testing as edges.

    Each node (or each of ``query_ids``) is paired with its ``k`` nearest
    neighbours, where "near" is the largest single joint move between them
    (raw values, no wrapping). Neighbours further than ``radius_rad`` are
    never paired: a long straight line in joint space drifts far from holding
    the bar and costs many collision checks.

    Args:
        confs (np.ndarray): (N, 12) node configurations.
        k (int): neighbours per node.
        radius_rad (float): the largest joint move an edge may have.
        query_ids (Optional[Sequence[int]]): only these nodes look for
            neighbours (their neighbours may be any node). ``None`` = all.

    Returns:
        np.ndarray: (M, 2) unique pairs ``i < j``, sorted, so the same
        roadmap always splits into the same chunks of work.
    """
    n = len(confs)
    if n < 2:
        return np.zeros((0, 2), dtype=int)
    tree = cKDTree(confs)
    ids = np.arange(n) if query_ids is None else np.asarray(query_ids, dtype=int)
    if len(ids) == 0:
        return np.zeros((0, 2), dtype=int)
    distances, neighbours = tree.query(confs[ids], k=min(k + 1, n), p=np.inf,
                                       distance_upper_bound=radius_rad)
    distances = np.atleast_2d(distances)
    neighbours = np.atleast_2d(neighbours)
    pairs = set()
    for row, i in enumerate(ids):
        for distance, j in zip(distances[row], neighbours[row]):
            # cKDTree marks "no neighbour inside the radius" with an infinite
            # distance and an index equal to n.
            if not np.isfinite(distance) or j >= n or j == i:
                continue
            pairs.add((min(int(i), int(j)), max(int(i), int(j))))
    return np.array(sorted(pairs), dtype=int).reshape(-1, 2)


# -----------------------------------------------------------------------------
# * Checking one edge
# -----------------------------------------------------------------------------


def bisection_order(n: int) -> List[int]:
    """The indices 0..n-1, middle first, then the middles of each half, and so on.

    A collision along an edge is usually a stretch rather than a single
    point, so testing the middle first finds it sooner than walking from one
    end (the same reason ``pybullet_planning`` checks paths this way).

    Args:
        n (int): how many indices.

    Returns:
        List[int]: the visiting order.
    """
    order: List[int] = []
    intervals = [(0, n - 1)]
    while intervals:
        next_intervals = []
        for low, high in intervals:
            if low > high:
                continue
            middle = (low + high) // 2
            order.append(middle)
            next_intervals.append((low, middle - 1))
            next_intervals.append((middle + 1, high))
        intervals = next_intervals
    return order


def edge_violation(model: HoldingModel, qa: np.ndarray, qb: np.ndarray, step_rad: float,
                   rot_deg_per_mm: float) -> Dict[str, float]:
    """The largest holding violation along the straight line qa -> qb.

    Args:
        model (HoldingModel): the holding model.
        qa (np.ndarray): one end.
        qb (np.ndarray): the other end.
        step_rad (float): spacing of the points measured (largest joint move).
        rot_deg_per_mm (float): how rotation and position error are folded
            into one score; only used to say WHERE the worst point is.

    Returns:
        Dict[str, float]: ``viol_pos_max`` (m) and ``viol_rot_max`` (rad), each
        the largest along the line, ``t_viol_max`` (0..1, where the worst
        point by combined score sits) and ``n_viol_points``.
    """
    points = interpolate_confs(qa, qb, step_rad)
    worst_pos, worst_rot, worst_score, worst_t = 0.0, 0.0, -1.0, 0.5
    n = len(points)
    for index, conf in enumerate(points):
        pos, rot = holding_violation(model, conf)
        worst_pos = max(worst_pos, pos)
        worst_rot = max(worst_rot, rot)
        score = violation_score_mm(pos, rot, rot_deg_per_mm)
        if score > worst_score:
            worst_score = score
            worst_t = (index + 1) / (n + 1)
    return {"viol_pos_max": worst_pos, "viol_rot_max": worst_rot,
            "t_viol_max": worst_t, "n_viol_points": n}


def evaluate_edge(tools: WorkerTools, i: int, j: int, qa: np.ndarray, qb: np.ndarray, *,
                  collision_step_rad: float, viol_step_rad: float, rot_deg_per_mm: float,
                  check_collision: bool, counters: Dict[str, float]) -> Dict[str, Any]:
    """Measure one edge: its length, its worst holding violation, and whether it is free.

    The two end nodes are already known to be collision-free, so only the
    interior piece boundaries are checked, middle first, stopping at the
    first contact.

    Args:
        tools (WorkerTools): model and collision check.
        i (int): id of one end node.
        j (int): id of the other end node.
        qa (np.ndarray): configuration of node ``i``.
        qb (np.ndarray): configuration of node ``j``.
        collision_step_rad (float): largest joint move between collision checks.
        viol_step_rad (float): largest joint move between violation points.
        rot_deg_per_mm (float): see :func:`holding.violation_score_mm`.
        check_collision (bool): False leaves the edge unchecked (lazy mode).
        counters (Dict[str, float]): tally.

    Returns:
        Dict[str, Any]: one value per name in ``EDGE_COLUMNS``.
    """
    qa = np.asarray(qa, dtype=float)
    qb = np.asarray(qb, dtype=float)
    record: Dict[str, Any] = {"i": int(i), "j": int(j), "len_l2": l2(qa, qb), "len_linf": linf(qa, qb)}
    record.update(edge_violation(tools.model, qa, qb, viol_step_rad, rot_deg_per_mm))
    count(counters, "fk_pairs", record["n_viol_points"])

    checks = interpolate_confs(qa, qb, collision_step_rad)
    record["n_checks_planned"] = len(checks)
    record["checked"] = bool(check_collision)
    record["free"] = True
    record["t_collide"] = -1.0
    if check_collision:
        count(counters, "edges_checked")
        for index in bisection_order(len(checks)):
            if tools.timed_collides(checks[index], counters):
                record["free"] = False
                record["t_collide"] = (index + 1) / (len(checks) + 1)
                count(counters, "edges_blocked")
                break
    return record


def edge_cost(len_l2: np.ndarray, score_mm: np.ndarray, tolerance_mm: float,
              w_hold: float) -> np.ndarray:
    """The search weight of edges under one tolerance.

    Joint-space length, plus a large penalty for every multiple of the
    tolerance the edge's worst violation goes over it (the meeting's "big
    coefficient"). An edge inside the tolerance costs just its length.

    Args:
        len_l2 (np.ndarray): edge lengths (radians, Euclidean over 12 joints).
        score_mm (np.ndarray): each edge's worst violation score (mm).
        tolerance_mm (float): the tolerance.
        w_hold (float): the penalty per tolerance-multiple of excess.

    Returns:
        np.ndarray: the weights.
    """
    excess = np.maximum(0.0, np.asarray(score_mm) / tolerance_mm - 1.0)
    return np.asarray(len_l2) + w_hold * excess


# -----------------------------------------------------------------------------
# * The roadmap itself
# -----------------------------------------------------------------------------


@dataclass
class Roadmap:
    """A set of nodes that hold the bar, and the edges measured between them.

    Stored column by column (one list per field), which is how it goes to disk
    and how the dashboard reads it.

    Attributes:
        model (HoldingModel): the holding model the roadmap was built with.
        nodes (Dict[str, list]): one list per name in ``NODE_COLUMNS``.
        edges (Dict[str, list]): one list per name in ``EDGE_COLUMNS``.
        anchors (List[Dict[str, Any]]): what each anchor is (role, goal,
            start, pair, turn); nodes of kind "anchor" point into this list.
        config (Dict[str, Any]): the build settings.
        counters (Dict[str, float]): the work done, by phase.
        timings (Dict[str, float]): wall-clock seconds, by phase.
        meta (Dict[str, Any]): run name, bar, library, provenance.
    """

    model: HoldingModel
    nodes: Dict[str, list] = field(default_factory=lambda: {name: [] for name in NODE_COLUMNS})
    edges: Dict[str, list] = field(default_factory=lambda: {name: [] for name in EDGE_COLUMNS})
    anchors: List[Dict[str, Any]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    counters: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        """How many nodes."""
        return len(self.nodes["conf"])

    @property
    def n_edges(self) -> int:
        """How many measured edges (free or not)."""
        return len(self.edges["i"])

    def confs(self) -> np.ndarray:
        """All node configurations as an (N, 12) array."""
        return np.asarray(self.nodes["conf"], dtype=float).reshape(-1, 12)

    def add_node(self, node: Dict[str, Any], kind: str) -> int:
        """Append one node and return its id.

        Args:
            node (Dict[str, Any]): ``conf`` plus the geometry fields, and
                optionally ``mode``, ``driver``, ``anchor``, ``draw``.
            kind (str): "sampled" or "anchor".

        Returns:
            int: the new node's id.
        """
        defaults = {"kind": kind, "mode": "", "driver": "", "anchor": -1, "draw": -1}
        for name in NODE_COLUMNS:
            self.nodes[name].append(node.get(name, defaults.get(name)))
        return self.n_nodes - 1

    def add_edges(self, records: Sequence[Dict[str, Any]]) -> None:
        """Append measured edges (dicts with every name in ``EDGE_COLUMNS``)."""
        for record in records:
            for name in EDGE_COLUMNS:
                self.edges[name].append(record[name])

    def edge_arrays(self) -> Dict[str, np.ndarray]:
        """The edge columns as numpy arrays."""
        return {name: np.asarray(values) for name, values in self.edges.items()}
