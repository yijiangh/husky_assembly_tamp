import argparse
from ast import Pass
import cProfile
import io
import math
import os
import pathlib
import pstats
import sys
import time
from itertools import takewhile
from typing import Callable, List, Optional, Tuple, Union
import copy

import numpy as np
import pybullet
import pybullet_planning as pp
from pybullet_planning.interfaces.planner_interface.joint_motion_planning import get_difference_fn, get_refine_fn
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

from husky_assembly_tamp.model.target_parse import TargetParser
from husky_assembly_tamp.robot.dual_arm_projection import DualArmProjection
from husky_assembly_tamp.robot.robot_setup import RobotSetup
from husky_assembly_tamp.utils.params import DATA_DIR, PROJECT_DIR
from husky_assembly_tamp.utils.util import angles_distance, normalize_angles

# DEFAULT_RESOLUTION = math.radians(1.0)
DEFAULT_RESOLUTION = np.deg2rad(5.0)
DEFAULT_NORM = 2

bar_from_right = None
bar_from_left = None

np.set_printoptions(precision=4, suppress=False)


class PlanProfiler:
    """Lightweight accumulator for targeted timing of planning sub-operations.

    Usage:
        profiler = PlanProfiler()
        with profiler.measure("expand_cart_path_ladder_graph"):
            ...
        profiler.report()
    """

    def __init__(self):
        self._totals: dict = {}   # label -> total seconds
        self._counts: dict = {}   # label -> call count

    class _Ctx:
        def __init__(self, profiler: "PlanProfiler", label: str):
            self._p = profiler
            self._label = label
            self._t0 = None

        def __enter__(self):
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, *_):
            elapsed = time.perf_counter() - self._t0
            self._p._totals[self._label] = self._p._totals.get(self._label, 0.0) + elapsed
            self._p._counts[self._label] = self._p._counts.get(self._label, 0) + 1

    def measure(self, label: str) -> "_Ctx":
        return self._Ctx(self, label)

    def report(self, title: str = "Planning Profiler Report") -> None:
        if not self._totals:
            print(f"[{title}] No data recorded.")
            return
        total_all = sum(self._totals.values())
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
        print(f"  {'Operation':<35} {'Total(s)':>8}  {'Calls':>6}  {'Avg(ms)':>9}  {'%':>6}")
        print(f"  {'-' * 35}  {'-' * 8}  {'-' * 6}  {'-' * 9}  {'-' * 6}")
        for label in sorted(self._totals, key=lambda k: -self._totals[k]):
            tot = self._totals[label]
            cnt = self._counts[label]
            avg_ms = (tot / cnt * 1000) if cnt > 0 else 0.0
            pct = (tot / total_all * 100) if total_all > 0 else 0.0
            print(f"  {label:<35} {tot:>8.3f}  {cnt:>6}  {avg_ms:>9.2f}  {pct:>5.1f}%")
        print(f"  {'TOTAL':<35} {total_all:>8.3f}")
        print(f"{'=' * 60}\n")

    def get_cumulative(self, label: str) -> float:
        """Return total accumulated seconds for *label*, or 0.0 if never recorded."""
        return self._totals.get(label, 0.0)


def cspace_linear_extend(q1: np.ndarray, q2: np.ndarray, robot_setup: RobotSetup, projector: DualArmProjection):
    """Create extension function for path planning."""
    resolutions = np.array([DEFAULT_RESOLUTION for _ in robot_setup.arm_joints])

    q1_right = np.array(q1[6:])
    q2_right = np.array(q2[6:])

    right_diff = angles_distance(q1_right, q2_right)
    right_steps = int(np.ceil(np.linalg.norm(right_diff / resolutions[6:], ord=DEFAULT_NORM)))

    q_left_init = np.array(q1[:6])
    q_left_target = np.array(q2[:6])

    for i in range(right_steps + 1):
        if right_steps == 0:
            t = 0.0
        else:
            t = i / right_steps

        q_right_interp = q1_right + t * right_diff
        q_right_interp = normalize_angles(q_right_interp)

        with pp.LockRenderer():
            projected_conf = projector.project(q_right_interp, q_left_init)

        if i < right_steps and projected_conf is not None and np.linalg.norm(angles_distance(projected_conf[:6], q_left_init)) < 0.5:
            q_left_init = np.array(projected_conf[:6])
            continue
        elif i >= right_steps and projected_conf is not None and np.linalg.norm(angles_distance(projected_conf[:6], q_left_target)) < 0.1:
            q_left_init = np.array(projected_conf[:6])
            continue
        else:
            return False
    return True


class Capsule(object):

    def __init__(self, pose, config=None, parent=None, robot_setup=None, projector=None):
        self.pose = pose
        if config is not None:
            self.config = config
        else:
            self.config = []
        self.parent = parent
        self.connection = []
        self.feature_vec = None

    def retrace(self):
        sequence = []
        node = self
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]

    def draw(self, draw_fn, **kwargs):
        segment = [] if self.parent is None else [self, self.parent]
        draw_fn(self, segment, **kwargs)

    def set_parent(self, parent: "Capsule", dist_thresh: float = 2.0):
        self.parent = parent
        for _ in range(len(self.config)):
            self.connection.append([])
        for i in range(len(self.config)):
            for j in range(len(parent.config)):
                dist = np.linalg.norm(angles_distance(self.config[i], parent.config[j]))
                # print(f"i: {i}, j: {j}, dist: {dist}")
                if dist < dist_thresh:
                # if dist < 100.0:
                    self.connection[i].append(j)

    def __str__(self):
        return "Capsule(" + str(self.pose) + ", " + str(len(self.config)) + ")"

    __repr__ = __str__


def asymmetric_extend(c1: Capsule, c2: Capsule, extend_fn, backward=False):
    if backward:
        return reversed(list(extend_fn(c2, c1)))
    return extend_fn(c1, c2)


def cart_extend_towards_capsule(tree: List[Capsule], target: Capsule, distance_fn, extend_fn, collision_fn, robot_setup, projector, swap=False, tree_frequency=1, **kwargs):
    target = copy.deepcopy(target)
    target.parent = None
    # Fast nearest-neighbor search using KDTree when feature vectors are available
    try:
        if hasattr(target, "feature_vec") and target.feature_vec is not None:
            vecs = []
            for n in tree:
                v = getattr(n, "feature_vec", None)
                if v is None:
                    vecs = []
                    break
                vecs.append(v)
            if vecs:
                kdt = KDTree(np.vstack(vecs))
                _, idx = kdt.query(target.feature_vec)
                last = tree[int(idx)]
            else:
                last = None
        else:
            last = None
    except Exception:
        last = None

    def _dist(n):
        d = distance_fn(n, target)
        if np.isscalar(d):
            return float(d)
        arr = np.asarray(d, dtype=float)
        if arr.shape == ():
            return float(arr)
        return float(np.linalg.norm(arr, ord=2))

    if last is None:
        last = pp.utils.argmin(_dist, tree)
    # extend = list(asymmetric_extend(last, target, extend_fn, backward=swap))
    extend = list(extend_fn(last, target))
    safe = list(takewhile(pp.utils.negate(collision_fn), extend))
    for i, c in enumerate(safe[1:]):
        c: Capsule
        c.parent = last
        tree.append(c)
        last = c

        # if (i % tree_frequency == 0) or (i == len(safe) - 1):
        #     c.parent = last
        #     tree.append(c)
        #     last = c
    success = len(extend) == len(safe)
    return last, success, tree


def ladder_graph_search(nodes: List[Capsule], distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None, instant: bool = False):
    """Enumerate all feasible configuration paths across capsule rungs.

    Args:
        nodes: Ladder of `Capsule` rungs with `config` lists and inter-rung `connection` info.
        distance_fn: Function mapping (q_prev, q_curr) -> scalar step distance. If None,
                     uses L2 norm of angular difference via `angles_distance`.
        instant: If True, return immediately with the first feasible path found during DFS
                 enumeration (in exploration order) and stop searching further.

    Returns:
        When `instant` is False: List of tuples (path, chosen_indices, total_distance), sorted by
        total_distance ascending.
        When `instant` is True: Either a single-element list with the first feasible result found,
        or an empty list if no feasible path exists.
        - path: List[np.ndarray] of configurations, one per rung.
        - chosen_indices: List[int] indices into each rung's `config`.
        - total_distance: float cumulative distance along adjacent rung transitions.
        Empty list if no feasible path exists.
    """
    if nodes is None or len(nodes) == 0:
        return []

    # Default distance function in C-space
    def _default_dist(q1: np.ndarray, q2: np.ndarray) -> float:
        return float(np.linalg.norm(angles_distance(np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)), ord=2))

    dist_fn = distance_fn if distance_fn is not None else _default_dist

    if len(nodes) == 1:
        first_node = nodes[0]
        if first_node is None or first_node.config is None or len(first_node.config) == 0:
            return []
        # All single-node choices have zero path length
        if instant:
            # Return only the first option when instant is requested
            return [([first_node.config[0]], [0], 0.0)]
        results = []
        for idx, q in enumerate(first_node.config):
            results.append(([q], [idx], 0.0))
        return results

    # If any rung has zero configs, no path exists
    for n in nodes:
        if n is None or n.config is None or len(n.config) == 0:
            return []

    num_nodes = len(nodes)

    # Build adjacency maps for each consecutive rung pair: prev_idx -> List[curr_idx]
    edges_per_level: List[dict] = []
    for level in range(num_nodes - 1):
        prev_node = nodes[level]
        curr_node = nodes[level + 1]
        prev_size = len(prev_node.config)
        curr_size = len(curr_node.config)
        edges = {i: [] for i in range(prev_size)}

        if getattr(curr_node, "parent", None) is prev_node:
            # Use curr_node.connection[curr_idx] -> list of prev indices
            for curr_idx in range(curr_size):
                conns = curr_node.connection[curr_idx] if curr_idx < len(curr_node.connection) else []
                for prev_idx in conns:
                    if 0 <= prev_idx < prev_size:
                        edges[prev_idx].append(curr_idx)
        elif getattr(prev_node, "parent", None) is curr_node:
            # Use prev_node.connection[prev_idx] -> list of curr indices
            for prev_idx in range(prev_size):
                conns = prev_node.connection[prev_idx] if prev_idx < len(prev_node.connection) else []
                for curr_idx in conns:
                    if 0 <= curr_idx < curr_size:
                        edges[prev_idx].append(curr_idx)
        else:
            # Unknown relation between rungs; treat as no edges
            return []

        edges_per_level.append(edges)

    # DFS to enumerate all feasible index sequences and their paths/costs
    results: List[Tuple[List[np.ndarray], List[int], float]] = []
    early_result: Optional[Tuple[List[np.ndarray], List[int], float]] = None

    def dfs(rung_idx: int, curr_idx: int, chosen_indices: List[int], path: List[np.ndarray], total_cost: float):
        nonlocal early_result
        if instant and early_result is not None:
            return
        # At rung_idx refers to current rung index (0-based). If we reached last rung, record
        if rung_idx == num_nodes - 1:
            if instant:
                early_result = (path, chosen_indices, float(total_cost))
            else:
                results.append((path, chosen_indices, float(total_cost)))
            return
        # Explore all successors on next rung
        successors = edges_per_level[rung_idx].get(curr_idx, [])
        for nxt_idx in successors:
            q_prev = nodes[rung_idx].config[curr_idx]
            q_curr = nodes[rung_idx + 1].config[nxt_idx]
            step_cost = dist_fn(q_prev, q_curr)
            dfs(rung_idx + 1, nxt_idx, chosen_indices + [nxt_idx], path + [q_curr], total_cost + step_cost)
            if instant and early_result is not None:
                return

    # Start from every feasible index on the first rung that has at least one outgoing edge
    first_edges = edges_per_level[0]
    for start_idx in range(len(nodes[0].config)):
        if len(first_edges.get(start_idx, [])) == 0:
            continue
        q0 = nodes[0].config[start_idx]
        dfs(0, start_idx, [start_idx], [q0], 0.0)
        if instant and early_result is not None:
            return [early_result]

    if instant:
        return [early_result] if early_result is not None else []

    # Sort by total cost ascending
    results.sort(key=lambda item: item[2])
    return results


def ladder_graph_shortest_path(nodes: List[Capsule], distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None):
    """Compute the minimum-cost path across ladder rungs via dynamic programming.

    Returns:
        Tuple (path, chosen_indices, total_cost) or None if no feasible path exists.
    """
    if nodes is None or len(nodes) == 0:
        return None
    for n in nodes:
        if n is None or n.config is None or len(n.config) == 0:
            return None

    def _default_dist(q1: np.ndarray, q2: np.ndarray) -> float:
        return float(np.linalg.norm(angles_distance(np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)), ord=2))

    dist_fn = distance_fn if distance_fn is not None else _default_dist

    num_nodes = len(nodes)
    if num_nodes == 1:
        return ([nodes[0].config[0]], [0], 0.0)

    # Build adjacency maps for each consecutive rung pair: prev_idx -> List[curr_idx]
    edges_per_level: List[dict] = []
    for level in range(num_nodes - 1):
        prev_node = nodes[level]
        curr_node = nodes[level + 1]
        prev_size = len(prev_node.config)
        curr_size = len(curr_node.config)
        edges = {i: [] for i in range(prev_size)}

        if getattr(curr_node, "parent", None) is prev_node:
            for curr_idx in range(curr_size):
                conns = curr_node.connection[curr_idx] if curr_idx < len(curr_node.connection) else []
                for prev_idx in conns:
                    if 0 <= prev_idx < prev_size:
                        edges[prev_idx].append(curr_idx)
        elif getattr(prev_node, "parent", None) is curr_node:
            for prev_idx in range(prev_size):
                conns = prev_node.connection[prev_idx] if prev_idx < len(prev_node.connection) else []
                for curr_idx in conns:
                    if 0 <= curr_idx < curr_size:
                        edges[prev_idx].append(curr_idx)
        else:
            return None

        edges_per_level.append(edges)

    # DP over rungs
    prev_costs = [0.0 for _ in range(len(nodes[0].config))]
    backpointers: List[List[Optional[int]]] = []

    for level in range(num_nodes - 1):
        prev_node = nodes[level]
        curr_node = nodes[level + 1]
        curr_costs = [math.inf for _ in range(len(curr_node.config))]
        curr_prev = [None for _ in range(len(curr_node.config))]

        edges = edges_per_level[level]
        for prev_idx, curr_indices in edges.items():
            if prev_idx < 0 or prev_idx >= len(prev_node.config):
                continue
            prev_cost = prev_costs[prev_idx]
            if not math.isfinite(prev_cost):
                continue
            q_prev = prev_node.config[prev_idx]
            for curr_idx in curr_indices:
                q_curr = curr_node.config[curr_idx]
                cost = prev_cost + dist_fn(q_prev, q_curr)
                if cost < curr_costs[curr_idx]:
                    curr_costs[curr_idx] = cost
                    curr_prev[curr_idx] = prev_idx

        backpointers.append(curr_prev)
        prev_costs = curr_costs

    # Pick best end node
    if all(not math.isfinite(c) for c in prev_costs):
        return None
    end_idx = int(np.argmin(prev_costs))
    total_cost = float(prev_costs[end_idx])

    # Reconstruct indices backward
    indices = [end_idx]
    for level in reversed(range(num_nodes - 1)):
        prev_idx = backpointers[level][indices[-1]]
        if prev_idx is None:
            return None
        indices.append(prev_idx)
    indices = list(reversed(indices))

    path = [nodes[r].config[idx] for r, idx in enumerate(indices)]
    return path, indices, total_cost


def plot_ladder_graph(capsule_path: List[Capsule], highlight_feasible: bool = False) -> Optional[str]:
    """
    Draw ladder graph nodes (per-rung IK solutions) and inter-rung connections, then save as SVG.

    When highlight_feasible is True, computes a feasible joint sequence via ladder_graph_search
    and highlights its nodes and connecting edges.
    """
    if capsule_path is None or len(capsule_path) == 0:
        return None

    rung_sizes = [len(node.config) if (node is not None and hasattr(node, "config") and node.config is not None) else 0 for node in capsule_path]
    if sum(rung_sizes) == 0:
        return None

    num_rungs = len(capsule_path)
    max_rung_size = max(rung_sizes) if len(rung_sizes) > 0 else 0

    x_spacing = 1.6
    y_spacing = 1.0
    node_radius = 0.06

    fig_w = max(6.0, 0.8 + x_spacing * max(1, num_rungs - 1))
    fig_h = max(4.0, 1.0 + y_spacing * max(3, max_rung_size))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    rung_positions: List[List[Tuple[float, float]]] = []
    for r, size in enumerate(rung_sizes):
        x = r * x_spacing
        if size <= 0:
            rung_positions.append([])
            continue
        total_height = (size - 1) * y_spacing
        y0 = -0.5 * total_height
        positions = [(x, y0 + i * y_spacing) for i in range(size)]
        rung_positions.append(positions)

    # Draw edges between adjacent rungs using stored connection info
    for r in range(num_rungs - 1):
        left = capsule_path[r]
        right = capsule_path[r + 1]
        if left is None or right is None:
            continue
        left_size = len(left.config) if left.config is not None else 0
        right_size = len(right.config) if right.config is not None else 0
        if left_size == 0 or right_size == 0:
            continue

        if getattr(right, "parent", None) is left:
            # right.connection[curr_idx] -> list of prev indices in left
            for curr_idx in range(right_size):
                conns = right.connection[curr_idx] if curr_idx < len(right.connection) else []
                for prev_idx in conns:
                    if 0 <= prev_idx < left_size:
                        x0, y0 = rung_positions[r][prev_idx]
                        x1, y1 = rung_positions[r + 1][curr_idx]
                        ax.plot([x0, x1], [y0, y1], color="0.6", linewidth=1.2, alpha=0.8)
        elif getattr(left, "parent", None) is right:
            # left.connection[prev_idx] -> list of curr indices in right
            for prev_idx in range(left_size):
                conns = left.connection[prev_idx] if prev_idx < len(left.connection) else []
                for curr_idx in conns:
                    if 0 <= curr_idx < right_size:
                        x0, y0 = rung_positions[r][prev_idx]
                        x1, y1 = rung_positions[r + 1][curr_idx]
                        ax.plot([x0, x1], [y0, y1], color="0.6", linewidth=1.2, alpha=0.8)
        else:
            continue

    # Draw nodes
    for r, positions in enumerate(rung_positions):
        if len(positions) == 0:
            continue
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        if r == 0:
            facecolor = "#2ca02c"  # green
        elif r == num_rungs - 1:
            facecolor = "#d62728"  # red
        else:
            facecolor = "#1f77b4"  # blue
        ax.scatter(xs, ys, s=(node_radius * 650) ** 2 / (fig.dpi**2), c=facecolor, edgecolors="k", linewidths=0.6, zorder=3)
        for i, (x, y) in enumerate(positions):
            ax.text(x, y + 0.08, f"{i}", ha="center", va="bottom", fontsize=8, color="k")

    # Optionally highlight all feasible paths across rungs, colored by total distance (shortest=green, longest=red)
    if highlight_feasible:

        def _js_dist(q1: np.ndarray, q2: np.ndarray) -> float:
            return float(np.linalg.norm(angles_distance(np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)), ord=2))

        results = ladder_graph_search(capsule_path, distance_fn=_js_dist)
        if results is not None and len(results) > 0:
            distances = [d for (_, _, d) in results]
            dmin = min(distances)
            dmax = max(distances)
            denom = (dmax - dmin) if (dmax - dmin) > 1e-12 else 1.0
            cmap = plt.get_cmap("RdYlGn_r")  # 0 -> green, 1 -> red

            for feasible_path, chosen_indices, total_d in results:
                t = float((total_d - dmin) / denom)
                color = cmap(t)

                # Draw highlighted edges between successive chosen nodes
                for r in range(num_rungs - 1):
                    i0 = chosen_indices[r]
                    i1 = chosen_indices[r + 1]
                    if i0 is None or i1 is None:
                        continue
                    if i0 < 0 or i1 < 0:
                        continue
                    if i0 >= len(rung_positions[r]) or i1 >= len(rung_positions[r + 1]):
                        continue
                    x0, y0 = rung_positions[r][i0]
                    x1, y1 = rung_positions[r + 1][i1]
                    ax.plot([x0, x1], [y0, y1], color=color, linewidth=2.8, alpha=0.95, zorder=2)

                # Overlay highlighted nodes for this path
                for r, idx in enumerate(chosen_indices):
                    if idx is None or idx < 0:
                        continue
                    if idx >= len(rung_positions[r]):
                        continue
                    xh, yh = rung_positions[r][idx]
                    ax.scatter([xh], [yh], s=(node_radius * 780) ** 2 / (fig.dpi**2), c=[color], edgecolors="k", linewidths=0.7, zorder=4)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Rung Index")
    ax.set_ylabel("Node Index (layout)")
    ax.set_title("Ladder Graph (Capsule Path)")
    ax.set_xticks([i * x_spacing for i in range(num_rungs)])
    ax.set_xticklabels([str(i) for i in range(num_rungs)])
    ax.margins(x=0.15, y=0.15)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    out_dir = os.path.join(PROJECT_DIR, "plots")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"ladder_graph_{timestamp}.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Ladder graph saved to: {out_path}")
    return out_path


def plot_capsule_path(capsule_path: List[Capsule]):
    for capsule in capsule_path:
        pp.draw_pose(capsule.pose, length=0.025)


def check_capsule_path(capsule_path: List[Capsule]):
    """Remove capsules with duplicate poses, keeping only the first occurrence.

    Deduplication uses rounded pose values to be robust to small floating errors.
    Parents of the kept capsules are rewired to maintain a valid chain.

    Args:
        capsule_path: Sequence of `Capsule` nodes to deduplicate.

    Returns:
        A new list containing capsules with unique poses (by rounded values), in order.
    """
    if capsule_path is None or len(capsule_path) == 0:
        return capsule_path

    def pose_to_tuple(pose, decimals: int = 3):
        # Handle (pos, quat) tuples/lists directly; otherwise convert from transform
        if isinstance(pose, (list, tuple)) and len(pose) == 2:
            pos, orn = pose
            pos = np.asarray(pos, dtype=float).reshape(3)
            orn = np.asarray(orn, dtype=float).reshape(4)
        else:
            T = pp.tform_from_pose(pose)
            pos = np.asarray(T[:3, 3], dtype=float)
            Rm = np.asarray(T[:3, :3], dtype=float)
            orn = R.from_matrix(Rm).as_quat()
        return tuple(np.round(pos, decimals=decimals)) + tuple(np.round(orn, decimals=decimals))

    seen = set()
    deduped: List[Capsule] = []
    last_kept_copy: Optional[Capsule] = None

    for cap in capsule_path:
        try:
            key = pose_to_tuple(cap.pose)
        except Exception:
            # If pose cannot be parsed, treat it as unique by id fallback
            key = (id(cap.pose),)

        if key in seen:
            continue

        seen.add(key)

        # IMPORTANT: Do not mutate original tree nodes here.
        # Create an independent copy to build a clean chain for downstream processing.
        cap_copy = copy.deepcopy(cap)
        # Track the origin node in its source tree so we can cache expansions back later
        try:
            setattr(cap_copy, "_origin", cap)
        except Exception:
            pass
        cap_copy.parent = last_kept_copy
        deduped.append(cap_copy)
        last_kept_copy = cap_copy

    return deduped


def rrt_connect_capsule(
    solver: "TrajectoryDualCartConstrainedSolver",
    start: Capsule,
    goal: Capsule,
    distance_fn,
    sample_fn,
    extend_fn,
    collision_fn,
    robot_setup,
    projector,
    max_iterations=10000,
    max_time=pp.INF,
    verbose=False,
    draw_fn=None,
    enforce_alternate=False,
    profiler: Optional["PlanProfiler"] = None,
    enable_ik: bool = True,
    ladder_search: str = "shortest",
    return_task_path: bool = False,
    enable_collision: bool = True,
    debug_tree_out: Optional[dict] = None,
    **kwargs,
):
    def _export_tree(nodes: List[Capsule]):
        if nodes is None:
            return {"points": [], "edges": []}
        id_to_idx = {}
        points = []
        for n in nodes:
            idx = len(points)
            id_to_idx[id(n)] = idx
            try:
                p = np.asarray(n.pose[0], dtype=float).reshape(3)
                points.append([float(p[0]), float(p[1]), float(p[2])])
            except Exception:
                points.append([0.0, 0.0, 0.0])
        edges = []
        for n in nodes:
            if n.parent is None:
                continue
            pid = id(n.parent)
            cid = id(n)
            if pid in id_to_idx and cid in id_to_idx:
                edges.append([id_to_idx[pid], id_to_idx[cid]])
        return {"points": points, "edges": edges}

    def _fill_tree_out(success_flag: bool, iterations_used: int):
        if debug_tree_out is None:
            return
        start_pose_xyz = debug_tree_out.get("start_pose")
        goal_pose_xyz = debug_tree_out.get("goal_pose")
        debug_tree_out.clear()
        debug_tree_out["success"] = bool(success_flag)
        debug_tree_out["iterations"] = int(iterations_used)
        if start_pose_xyz is not None:
            debug_tree_out["start_pose"] = start_pose_xyz
        if goal_pose_xyz is not None:
            debug_tree_out["goal_pose"] = goal_pose_xyz
        debug_tree_out["tree1"] = _export_tree(nodes1)
        debug_tree_out["tree2"] = _export_tree(nodes2)

    start_time = time.time()
    if collision_fn(start):
        print(f"Start configuration in collision.")
        return None

    if collision_fn(goal):
        print(f"Goal configuration in collision.")
        return None

    nodes1, nodes2 = [start], [goal]
    for iteration in range(max_iterations):
        if max_time <= pp.elapsed_time(start_time):
            break
        if enforce_alternate:
            swap = iteration % 2
        else:
            swap = len(nodes1) > len(nodes2)
        # swap = False
        tree1, tree2 = nodes1, nodes2
        if swap:
            tree1, tree2 = nodes2, nodes1

        target = sample_fn()
        bodies = []
        if draw_fn:
            bodies = pp.draw_pose(target.pose, length=0.1)
            # draw_fn(target, [])

        last1, _, tree1 = cart_extend_towards_capsule(tree1, target, distance_fn, extend_fn, collision_fn, robot_setup, projector, swap, **kwargs)
        last1_independent = copy.deepcopy(last1)
        last1_independent.parent = None
        last2, success, tree2 = cart_extend_towards_capsule(tree2, last1_independent, distance_fn, extend_fn, collision_fn, robot_setup, projector, not swap, **kwargs)

        # if draw_fn:
        #     for sp in tree1 + tree2:
        #         sp.draw(draw_fn)

        if draw_fn:
            for sp in nodes1:
                sp.draw(draw_fn, color="red")

            for sp in nodes2:
                sp.draw(draw_fn, color="blue")

        if success:
            if not enable_ik:
                # Stage 1: task-space connection found; no joint configs to extract.
                # Keep iterating so draw_fn can fill the workspace view.
                if verbose:
                    print(f"[Stage 1] Task-space connection at iter {iteration} "
                          f"(nodes: {len(nodes1)+len(nodes2)}); skipping ladder graph.")
                if return_task_path:
                    path1, path2 = last1.retrace(), last2.retrace()
                    if swap:
                        path1, path2 = path2, path1
                    capsule_nodes = path1 + path2[::-1]
                    _fill_tree_out(True, iteration + 1)
                    return check_capsule_path(capsule_nodes)
                if draw_fn:
                    for debug_body in bodies:
                        pp.remove_debug(debug_body)
                continue

            path1, path2 = last1.retrace(), last2.retrace()
            if swap:
                path1, path2 = path2, path1
            if verbose:
                print(f"RRT connect capsule: {iteration} iterations, {len(nodes1) + len(nodes2)} nodes")
            capsule_nodes = path1 + path2[::-1]
            capsule_path = check_capsule_path(capsule_nodes)
            capsule_path = solver._decimate_capsule_path(capsule_path)

            # Fast attempt: project continuously along the capsule pose chain.
            _prof = profiler if profiler is not None else PlanProfiler()
            with _prof.measure("project_chain"):
                proj_path = solver.try_project_chain(capsule_path, enable_collision=enable_collision)
            if proj_path is not None:
                _fill_tree_out(True, iteration + 1)
                return proj_path

            # Fast skip: if every rung in this path is already expanded (config length > 1),
            # we assume this path has been checked before and skip heavy processing.
            all_expanded = True
            for n in capsule_path:
                cfg = getattr(n, "config", None)
                if (cfg is None) or (len(cfg) <= 1):
                    all_expanded = False
                    break
            if all_expanded:
                if draw_fn:
                    for debug_body in bodies:
                        pp.remove_debug(debug_body)
                continue

            def _js_dist(q1: np.ndarray, q2: np.ndarray) -> float:
                return float(np.linalg.norm(angles_distance(np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)), ord=2))

            with pp.LockRenderer():
                with _prof.measure("expand_cart_path_ladder_graph (total)"):
                    expended_path, _ = solver.expand_cart_path_ladder_graph(capsule_path, profiler=_prof)
            with _prof.measure("ladder_graph_search"):
                if ladder_search == "enumerate":
                    results = ladder_graph_search(expended_path, distance_fn=_js_dist, instant=False)
                else:
                    shortest = ladder_graph_shortest_path(expended_path, distance_fn=_js_dist)
                    results = [shortest] if shortest is not None else []
            if results is None or len(results) == 0 or results[0] is None:
                for debug_body in bodies:
                    pp.remove_debug(debug_body)
                continue  # way 1: restart in the loop
                # break  # way 2: break out of the loop and restart out of the loop

            with pp.LockRenderer():
                plot_capsule_path(capsule_path)
            # plot_ladder_graph(expended_path, highlight_feasible=True)

            best_path, _, _ = results[0]
            _fill_tree_out(True, iteration + 1)
            return best_path

        for debug_body in bodies:
            pp.remove_debug(debug_body)
    _fill_tree_out(False, max_iterations)
    return None


def joint_space_smooth(path, extend_fn, max_iterations: int = 50, robot_setup: RobotSetup = None):
    if path is None or len(path) < 3:
        return path

    rng = np.random.default_rng()
    current_path = list(path)

    for _ in range(max_iterations):
        n = len(current_path)
        if n < 3:
            break

        i, j = sorted(rng.integers(low=0, high=n, size=2))
        if j - i <= 1:
            continue
        
        # i, j = 9, 15

        start = current_path[i]
        end = current_path[j]

        segment = list(extend_fn(start, end))
        if not segment:
            continue
        if any(s is None for s in segment):
            continue

        current_path = current_path[:i] + segment + current_path[j:]

    return current_path


def interpolate(path, extend_fn, robot_setup: RobotSetup = None):
    if path is None or len(path) <= 1:
        return path

    def _same_conf(q1, q2, tol: float = 1e-6) -> bool:
        try:
            a = np.asarray(q1, dtype=float)
            b = np.asarray(q2, dtype=float)
            diff = angles_distance(a, b)
            return float(np.linalg.norm(diff, ord=2)) <= tol
        except Exception:
            return False

    result = [path[0]]
    for i in range(len(path) - 1):
        q1 = path[i]
        q2 = path[i + 1]

        seg = list(extend_fn(q1, q2))
        if not seg or any(s is None for s in seg):
            # Fallback: keep the original neighbor
            if not _same_conf(result[-1], q2):
                result.append(q2)
            continue

        # Splice segment with de-duplication
        for s in seg:
            if not _same_conf(result[-1], s):
                result.append(s)
        # Ensure the end neighbor is present
        if not _same_conf(result[-1], q2):
            result.append(q2)

    return result


def random_restarts_capsule(
    solver: "TrajectoryDualCartConstrainedSolver",
    start: Capsule,
    goal: Capsule,
    distance_fn,
    sample_fn,
    extend_fn,
    collision_fn,
    robot_setup: RobotSetup,
    projector: DualArmProjection,
    restarts=pp.RRT_RESTARTS,
    smooth=pp.RRT_SMOOTHING,
    max_solutions=1,
    max_time=pp.INF,
    draw_fn=None,
    verbose=False,
    **kwargs,
):
    start_time = time.time()
    solutions = []
    # path = check_direct(start, goal, extend_fn, collision_fn, **kwargs) # TODO: check direct path
    path = None

    for attempt in range(restarts + 1):
        if (len(solutions) >= max_solutions) or (pp.elapsed_time(start_time) >= max_time):
            break
        pp.remove_all_debug()
        attempt_time = min(max_time - pp.elapsed_time(start_time), max_time / restarts)
        path = rrt_connect_capsule(solver, start, goal, distance_fn, sample_fn, extend_fn, collision_fn, robot_setup, projector, max_time=attempt_time, draw_fn=draw_fn, verbose=verbose, **kwargs)
        if path is None:
            continue
        # path = pp.smooth_path(path, extend_fn, collision_fn, max_smooth_iterations=smooth, max_time=max_time-pp.elapsed_time(start_time), **kwargs) # TODO: smooth path
        solutions.append(path)
        # if pp.compute_path_cost(path, distance_fn) < success_cost:
        #     break
    solutions = sorted(solutions, key=lambda path: pp.compute_path_cost(path, distance_fn))
    if verbose:
        print("Solutions ({}): {} | Time: {:.3f}".format(len(solutions), [(len(path), round(pp.compute_path_cost(path, distance_fn), 3)) for path in solutions], pp.elapsed_time(start_time)))
    if len(solutions) > 0:
        return solutions[0]
    return None


class TrajectoryDualCartConstrainedSolver(object):

    @staticmethod
    def initialize_robot_setup_for_planning(robot_name: str, robot_type: str, target_cell_state_path: str, use_scene_parser_gui: bool = True, scene_parser_verbose: bool = True) -> Tuple[RobotSetup, np.ndarray, DualArmProjection]:
        """
        Initialize robot setup for dual-arm constrained motion planning.

        This method encapsulates the complete initialization process for robot setup, including:
        1. Creating and configuring the RobotSetup instance
        2. Computing and normalizing the target joint configuration
        3. Calculating the relative transformation between left and right tool poses
        4. Creating the dual-arm constraint projector

        Args:
            robot_name (str): Unique identifier for the robot instance (e.g., "r0")
            robot_type (str): Type of robot to initialize (e.g., "husky_dual")
            target_cell_state_path (str): File path to the robot cell state JSON file containing
                                        target configuration and scene information
            use_scene_parser_gui (bool, optional): Whether to enable GUI for scene parsing.
                                                 Defaults to True.
            scene_parser_verbose (bool, optional): Whether to enable verbose output during
                                                 scene parsing. Defaults to True.

        Returns:
            Tuple[RobotSetup, np.ndarray, DualArmProjection]: A tuple containing:
                - robot_setup (RobotSetup): Fully configured robot setup instance with loaded
                                          scene and target configuration
                - target_conf (np.ndarray): Normalized target joint configuration (12 DOF)
                                           with angles in [-π, π] range
                - projector (DualArmProjection): Dual-arm constraint projector configured
                                                with the relative transformation between
                                                left and right tool poses

        Raises:
            FileNotFoundError: If the target_cell_state_path does not exist
            ValueError: If the robot setup fails to initialize properly
            RuntimeError: If unable to compute tool poses or create projector

        Example:
            ```python
            # Initialize robot setup for planning
            robot_setup, target_conf, projector = TrajectoryDualConstrainedSolver.initialize_robot_setup_for_planning(
                robot_name="r0",
                robot_type="husky_dual",
                target_cell_state_path="/path/to/target_state.json"
            )

            # Create solver with initialized components
            target_parser = TargetParser(design_path, targets_file)
            solver = TrajectoryDualConstrainedSolver(robot_setup, target_parser)
            ```

        Note:
            The target configuration is automatically normalized to the [-π, π] range to ensure
            consistent angle representation for motion planning algorithms. The dual-arm projector
            maintains the relative pose constraint between the left and right tool links as
            computed from the target configuration.
        """
        print("Initializing robot setup for planning...")

        # Create RobotSetup instance with specified parameters
        robot_setup = RobotSetup(robot_name, robot_type=robot_type, robot_cell_state_path=target_cell_state_path, use_scene_parser_gui=use_scene_parser_gui, scene_parser_verbose=scene_parser_verbose)

        # Extract and normalize target configuration
        target_conf = np.array(robot_setup.arm_target_angles)
        target_conf = (target_conf + np.pi) % (2 * np.pi) - np.pi
        print(f"Target configuration: {list(target_conf)}")

        # Compute relative transformation between left and right tool poses
        world_from_left = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
        world_from_right = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
        desired_right_from_left = pp.multiply(pp.invert(world_from_right), world_from_left)

        # Create dual-arm constraint projector
        projector = DualArmProjection(robot_setup, desired_right_from_left)

        print("✓ Robot setup initialization completed")
        return robot_setup, target_conf, projector

    def __init__(
        self,
        robot_setup: RobotSetup,
        target_parser: TargetParser,
        projector: DualArmProjection,
        ladder_expand_delta: float = np.pi / 4.0,
        start_goal_delta: float = np.pi / 4.0,
        dist_metric: str = "feature",
        ladder_search: str = "shortest",
        goal_sample_prob: float = 0.1,
        ladder_edge_thresh: float = 2.5,
        ladder_expand_attempts_small: int = 5,
        ladder_expand_attempts_full: int = 20,
        ladder_decimate_pos: float = 0.05,
        ladder_decimate_ang: float = 0.1,
        guide_sample_prob: float = 0.2,
        random_seed: Optional[int] = None,
    ):
        self.robot_setup = robot_setup
        self.target_parser = target_parser
        self.projector = projector
        self.ladder_expand_delta = float(ladder_expand_delta)
        self.start_goal_delta = float(start_goal_delta)
        self.dist_metric = dist_metric
        self.ladder_search = ladder_search
        self.goal_sample_prob = float(goal_sample_prob)
        self.ladder_edge_thresh = float(ladder_edge_thresh)
        self.ladder_expand_attempts_small = int(ladder_expand_attempts_small)
        self.ladder_expand_attempts_full = int(ladder_expand_attempts_full)
        self._goal_pose = None
        self._cached_collision_fn = self.robot_setup.create_collision_fn(
            obstacle_bodies=self.robot_setup.obstacles
        )
        self._bar_local_feature_points = self._compute_bar_feature_points()
        self._ik_cache = {}
        self._collision_cache = {}
        self.ladder_decimate_pos = float(ladder_decimate_pos)
        self.ladder_decimate_ang = float(ladder_decimate_ang)
        self.guide_sample_prob = float(guide_sample_prob)
        self._guide_poses = None
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def _compute_bar_feature_points(self) -> Optional[List[np.ndarray]]:
        """Compute a small set of feature points in the bar's local frame.

        Uses the current bar pose and its AABB in world coordinates, then
        transforms AABB corners into the bar frame. This yields a consistent
        set of points in the bar frame for the feature-point distance metric.
        """
        # TODO I think here is using a world-coordinate AABB. Ideally, we want a pose-specific AABB. Maybe you could just use sender. We can just use 长方体 as an approximation for a cylinder so it's easier to extract feature points 
        try:
            bar_id = self.robot_setup.target_bar
            world_from_bar = pp.get_pose(bar_id)
            aabb_min, aabb_max = pybullet.getAABB(bar_id)
            if aabb_min is None or aabb_max is None:
                return None

            aabb_min = np.asarray(aabb_min, dtype=float)
            aabb_max = np.asarray(aabb_max, dtype=float)
            corners_world = [
                np.array([aabb_min[0], aabb_min[1], aabb_min[2]]),
                np.array([aabb_min[0], aabb_max[1], aabb_max[2]]),
                np.array([aabb_max[0], aabb_min[1], aabb_max[2]]),
                np.array([aabb_max[0], aabb_max[1], aabb_min[2]]),
            ]
            bar_from_world = pp.invert(world_from_bar)
            corners_local = []
            for p_w in corners_world:
                p_local, _ = pp.multiply(bar_from_world, (p_w, [0, 0, 0, 1]))
                corners_local.append(np.asarray(p_local, dtype=float))
            return corners_local
        except Exception:
            return None

    def cart_linear_interp_z(self, q1: Capsule, q2: Capsule, position_res: float = 0.1, rotation_res: float = 0.1):
        def _quat_angle_between(q0, q1):
            q0 = np.asarray(q0, dtype=float)
            q1 = np.asarray(q1, dtype=float)
            n0 = max(1e-12, np.linalg.norm(q0))
            n1 = max(1e-12, np.linalg.norm(q1))
            q0 = q0 / n0
            q1 = q1 / n1
            d = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
            return float(math.acos(d))

        conf1 = np.asarray(q1.config[0], dtype=float)
        conf2 = np.asarray(q2.config[0], dtype=float)

        self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, conf1)
        # pose1 = pp.get_link_pose(self.robot_setup.robot, self.robot_setup.tool_link_right)
        pose1 = pp.get_pose(self.robot_setup.target_bar)
        pos1, quat1 = np.asarray(pose1[0], dtype=float), np.asarray(pose1[1], dtype=float)

        self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, conf2)
        # pose2 = pp.get_link_pose(self.robot_setup.robot, self.robot_setup.tool_link_right)
        pose2 = pp.get_pose(self.robot_setup.target_bar)
        pos2, quat2 = np.asarray(pose2[0], dtype=float), np.asarray(pose2[1], dtype=float)

        pos_dist = float(np.linalg.norm(pos2 - pos1))
        ang_dist = _quat_angle_between(quat1, quat2)
        num_steps = int(math.ceil(max(pos_dist / max(1e-12, position_res), ang_dist / max(1e-12, rotation_res))))

        waypoints = []
        if num_steps <= 0:
            waypoints.append((pos2, quat2))
            return waypoints

        R1 = R.from_quat(quat1).as_matrix()
        R2 = R.from_quat(quat2).as_matrix()
        z1 = R1[:, 2] / max(1e-12, np.linalg.norm(R1[:, 2]))
        z2 = R2[:, 2] / max(1e-12, np.linalg.norm(R2[:, 2]))
        cross_z = np.cross(z1, z2)
        axis_norm = np.linalg.norm(cross_z)
        if axis_norm < 1e-12:
            if float(np.dot(z1, z2)) < 0.0:
                trial = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(z1, trial)) > 0.9:
                    trial = np.array([0.0, 1.0, 0.0])
                axis = np.cross(z1, trial)
                axis = axis / max(1e-12, np.linalg.norm(axis))
                total_angle = math.pi
            else:
                axis = np.array([1.0, 0.0, 0.0])
                total_angle = 0.0
        else:
            axis = cross_z / axis_norm
            total_angle = math.acos(float(np.clip(np.dot(z1, z2), -1.0, 1.0)))

        for i in range(num_steps):
            fraction = float(i) / num_steps
            pos = (1.0 - fraction) * pos1 + fraction * pos2
            if total_angle == 0.0:
                Ri = R1
            else:
                tilt = R.from_rotvec(axis * (fraction * total_angle)).as_matrix()
                Ri = tilt @ R1
            quat = R.from_matrix(Ri).as_quat()
            waypoints.append((pos, np.asarray(quat, dtype=float)))

        # **************************************************************************
        # Optional
        # **************************************************************************
        # waypoints.append((pos2, quat2))

        return waypoints

    def cart_linear_interp(self, c1: Capsule, c2: Capsule, position_res: float = 0.005, rotation_res: float = 0.01):
        def _quat_angle_between(q0, q1):
            q0 = np.asarray(q0, dtype=float)
            q1 = np.asarray(q1, dtype=float)
            n0 = max(1e-12, np.linalg.norm(q0))
            n1 = max(1e-12, np.linalg.norm(q1))
            q0 = q0 / n0
            q1 = q1 / n1
            d = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
            return float(math.acos(d))

        pos1, quat1 = np.asarray(c1.pose[0], dtype=float), np.asarray(c1.pose[1], dtype=float)

        pos2, quat2 = np.asarray(c2.pose[0], dtype=float), np.asarray(c2.pose[1], dtype=float)

        pos_dist = float(np.linalg.norm(pos2 - pos1))
        ang_dist = _quat_angle_between(quat1, quat2)
        num_steps = int(math.ceil(max(pos_dist / max(1e-12, position_res), ang_dist / max(1e-12, rotation_res))))

        waypoints = []
        if num_steps <= 0:
            waypoints.append((pos2, quat2))
            return waypoints

        # Ensure shortest-path interpolation by flipping the second quaternion if needed
        if float(np.dot(quat1, quat2)) < 0.0:
            quat2 = -quat2

        key_times = np.array([0.0, 1.0], dtype=float)
        key_rots = R.from_quat(np.vstack([quat1, quat2]))
        slerp = Slerp(key_times, key_rots)

        times = np.linspace(0.0, 1.0, num_steps, endpoint=False, dtype=float)
        interp_rots = slerp(times)
        interp_quats = interp_rots.as_quat()

        for i, t in enumerate(times):
            pos = (1.0 - t) * pos1 + t * pos2
            quat = interp_quats[i]
            waypoints.append((np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)))

        waypoints.append((pos2, quat2))

        return waypoints

    def _get_sample_fn(self, enable_ik: bool = True):
        l, w, h = 2.2, 2.2, 0.6

        roll_range = (-np.pi, np.pi)
        pitch_range = (-np.pi, np.pi)
        yaw_range = (-np.pi, np.pi)

        def fn():

            global bar_from_right

            base_pos, _ = pp.get_pose(self.robot_setup.robot)
            cx, cy, cz = base_pos

            confs = None
            while confs is None:
                r = self.rng.random()
                if (self._goal_pose is not None) and (r < self.goal_sample_prob):
                    world_from_bar = self._goal_pose
                elif (self._guide_poses is not None) and (r < self.goal_sample_prob + self.guide_sample_prob):
                    world_from_bar = self._guide_poses[int(self.rng.integers(low=0, high=len(self._guide_poses)))]
                else:
                    x = cx + self.rng.uniform(-l / 2, l / 2)
                    y = cy + self.rng.uniform(-w / 2, w / 2)
                    z = cz + h / 2

                    roll = self.rng.uniform(*roll_range)
                    pitch = self.rng.uniform(*pitch_range)
                    yaw = self.rng.uniform(*yaw_range)

                    world_from_bar = pp.Pose(point=[x, y, z], euler=pp.Euler(roll, pitch, yaw))
                if enable_ik:
                    confs = self.projector.create_valid_confs(
                        self.robot_setup.ik_solver_right, world_from_bar, bar_from_right, delta=0.0, max_attempts=20, collision_fn=self._cached_collision_fn
                    )
                else:
                    confs = None

                if enable_ik and confs is not None:
                    cap = Capsule(world_from_bar, confs, parent=None, robot_setup=self.robot_setup, projector=self.projector)
                    self._attach_feature_vec(cap)
                    return cap
                elif not enable_ik:
                    cap = Capsule(world_from_bar, confs, parent=None, robot_setup=self.robot_setup, projector=self.projector)
                    self._attach_feature_vec(cap)
                    return cap

        return fn

    def _get_extend_fn(self, enable_ik: bool = True):
        def fn(c1: Capsule, c2: Capsule):
            way_points = self.cart_linear_interp(c1, c2, position_res=0.05, rotation_res=0.1)
            global bar_from_right

            last_capsule = copy.deepcopy(c1)
            last_capsule.parent = None
            # yield last_capsule

            for world_from_bar in way_points[1:]:
                world_from_right = pp.multiply(world_from_bar, bar_from_right)

                conf = None
                if enable_ik and hasattr(last_capsule, "config") and last_capsule.config is not None and len(last_capsule.config) > 0:
                    seed_idx = int(self.rng.integers(low=0, high=len(last_capsule.config)))
                    seed_right = last_capsule.config[seed_idx][6:]
                    seed_left = last_capsule.config[seed_idx][:6]

                    right_conf = self.robot_setup.ik_solver_right(world_from_right, seed_right)
                    if right_conf is not None:
                        conf = self.projector.project(right_conf, seed_left)

                if conf is not None:
                    capsule = Capsule(world_from_bar, config=[conf], parent=None, robot_setup=self.robot_setup, projector=self.projector)
                    self._attach_feature_vec(capsule)
                    yield capsule
                    last_capsule = capsule
                else:
                    capsule = Capsule(world_from_bar, parent=None, robot_setup=self.robot_setup, projector=self.projector)
                    self._attach_feature_vec(capsule)
                    yield capsule
                    if enable_ik:
                        break
                    else:
                        last_capsule = capsule

        return fn
    
    def _get_cspace_extend_with_projection_fn(self, enable_ik: bool = True):
        def fn(q1, q2):
            """Extend in configuration space by interpolating the RIGHT arm joints.

            For each right-arm interpolation step, attempt to recover a full dual-arm
            configuration using the projector. Yield a `Capsule` at every step with the
            corresponding bar pose and (when available) the full 12-DOF configuration.

            If projection fails at a step and `enable_ik` is True, yield the pose-only
            capsule and stop extending (mirrors the behavior of `_get_extend_fn`).
            """
            global bar_from_right

            # Guard: require configs to interpolate
            q1_full = np.asarray(q1, dtype=float)
            q2_full = np.asarray(q2, dtype=float)

            q1_right = np.asarray(q1_full[6:], dtype=float)
            q2_right = np.asarray(q2_full[6:], dtype=float)

            # Seed for the left arm comes from the last successful configuration
            last_q = copy.deepcopy(q1)

            seed_left = np.asarray(q1_full[:6], dtype=float)

            # Determine number of interpolation steps based on right-arm delta
            right_diff = angles_distance(q1_right, q2_right)
            resolutions = np.array([DEFAULT_RESOLUTION] * 6, dtype=float)
            steps = int(np.ceil(np.linalg.norm(right_diff / resolutions, ord=DEFAULT_NORM)))
            steps = max(1, steps)

            for i in range(steps + 1):
                t = float(i) / float(steps)
                q_right_interp = q1_right + t * right_diff
                q_right_interp = normalize_angles(q_right_interp)

                seed_left = last_q[:6]
                conf = self.projector.project(q_right_interp, seed_left)
                if conf is not None:
                    # WARNING/TODO: continuity is currently implicit (small right-arm interpolation
                    # + seed chaining). We do not explicitly reject projected states with a large
                    # joint-space jump vs. last_q. Add an explicit continuity threshold check later.
                    yield conf
                    last_q = conf
                else:
                    yield None
        return fn

    def _get_cart_dist_fn(self, metric: Optional[str] = None):
        metric = metric or self.dist_metric

        def _pose_to_world_points(pose, local_pts: List[np.ndarray]) -> List[np.ndarray]:
            pts = []
            for p_local in local_pts:
                p_world, _ = pp.multiply(pose, (p_local, [0, 0, 0, 1]))
                pts.append(np.asarray(p_world, dtype=float))
            return pts

        def _pose_to_Rp(pose):
            T = pp.tform_from_pose(pose)
            R = T[:3, :3]
            p = T[:3, 3]
            return R, p

        def _angle_between_unit_vectors(u: np.ndarray, v: np.ndarray) -> float:
            u = np.asarray(u, dtype=float)
            v = np.asarray(v, dtype=float)
            nu = np.linalg.norm(u)
            nv = np.linalg.norm(v)
            if nu > 0.0:
                u = u / nu
            if nv > 0.0:
                v = v / nv
            dot = float(np.dot(u, v))
            dot = max(-1.0, min(1.0, dot))
            return float(np.arccos(dot))

        def fn_pose6d(c1: Capsule, c2: Capsule):
            R1, p1 = _pose_to_Rp(c1.pose)
            R2, p2 = _pose_to_Rp(c2.pose)

            dp = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)

            ax = _angle_between_unit_vectors(R1[:, 0], R2[:, 0])
            ay = _angle_between_unit_vectors(R1[:, 1], R2[:, 1])
            az = _angle_between_unit_vectors(R1[:, 2], R2[:, 2])

            return np.array([dp[0], dp[1], dp[2], ax, ay, az], dtype=float)

        def fn_feature(c1: Capsule, c2: Capsule):
            if not self._bar_local_feature_points:
                return fn_pose6d(c1, c2)
            v1 = self._get_or_compute_feature_vec(c1)
            v2 = self._get_or_compute_feature_vec(c2)
            if v1 is None or v2 is None:
                return fn_pose6d(c1, c2)
            diff = v2 - v1
            return np.array([float(np.linalg.norm(diff))], dtype=float)

        if metric == "pose6d":
            return fn_pose6d
        if metric == "feature":
            return fn_feature
        # Fallback
        return fn_pose6d

        return fn

    def _get_or_compute_feature_vec(self, cap: Capsule) -> Optional[np.ndarray]:
        if cap is None:
            return None
        if getattr(cap, "feature_vec", None) is not None:
            return cap.feature_vec
        if not self._bar_local_feature_points:
            return None
        pts = []
        for p_local in self._bar_local_feature_points:
            p_world, _ = pp.multiply(cap.pose, (p_local, [0, 0, 0, 1]))
            pts.append(np.asarray(p_world, dtype=float))
        vec = np.concatenate(pts, axis=0)
        cap.feature_vec = vec
        return vec

    def _attach_feature_vec(self, cap: Capsule) -> None:
        if cap is None:
            return
        if self._bar_local_feature_points:
            _ = self._get_or_compute_feature_vec(cap)

    def _pose_key(self, pose, decimals: int = 3) -> Tuple[float, ...]:
        try:
            pos, orn = pose
            pos = np.asarray(pos, dtype=float).reshape(3)
            orn = np.asarray(orn, dtype=float).reshape(4)
        except Exception:
            T = pp.tform_from_pose(pose)
            pos = np.asarray(T[:3, 3], dtype=float)
            Rm = np.asarray(T[:3, :3], dtype=float)
            orn = R.from_matrix(Rm).as_quat()
        vals = tuple(np.round(pos, decimals=decimals)) + tuple(np.round(orn, decimals=decimals))
        return vals

    def _config_key(self, conf: np.ndarray, decimals: int = 3) -> Tuple[float, ...]:
        try:
            arr = np.asarray(conf, dtype=float).reshape(-1)
            return tuple(np.round(arr, decimals=decimals))
        except Exception:
            return (id(conf),)

    def _is_config_collision(self, conf: np.ndarray) -> bool:
        key = self._config_key(conf)
        if key in self._collision_cache:
            return self._collision_cache[key]
        coll = self._cached_collision_fn(conf)
        self._collision_cache[key] = coll
        return coll

    def _pose_distance(self, pose_a, pose_b) -> Tuple[float, float]:
        try:
            pa, qa = pose_a
            pb, qb = pose_b
            pa = np.asarray(pa, dtype=float)
            pb = np.asarray(pb, dtype=float)
            qa = np.asarray(qa, dtype=float)
            qb = np.asarray(qb, dtype=float)
        except Exception:
            Ta = pp.tform_from_pose(pose_a)
            Tb = pp.tform_from_pose(pose_b)
            pa = Ta[:3, 3]
            pb = Tb[:3, 3]
            qa = R.from_matrix(Ta[:3, :3]).as_quat()
            qb = R.from_matrix(Tb[:3, :3]).as_quat()
        dp = float(np.linalg.norm(pb - pa))
        # shortest-arc angle
        if float(np.dot(qa, qb)) < 0.0:
            qb = -qb
        ang = float(2.0 * math.acos(max(-1.0, min(1.0, float(np.dot(qa, qb)))))) / 2.0
        return dp, ang

    def _decimate_capsule_path(self, capsule_path: List[Capsule]) -> List[Capsule]:
        if capsule_path is None or len(capsule_path) <= 2:
            return capsule_path
        kept = [capsule_path[0]]
        last_pose = capsule_path[0].pose
        for cap in capsule_path[1:-1]:
            dp, dang = self._pose_distance(last_pose, cap.pose)
            if dp >= self.ladder_decimate_pos or dang >= self.ladder_decimate_ang:
                kept.append(cap)
                last_pose = cap.pose
        kept.append(capsule_path[-1])
        return kept

    def _bar_pose_from_conf(self, conf: np.ndarray):
        global bar_from_right
        self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, conf)
        right_pose = pp.get_link_pose(self.robot_setup.robot, self.robot_setup.tool_link_right)
        return pp.multiply(right_pose, pp.invert(bar_from_right))

    def _smooth_with_collision(self, path: List[np.ndarray], extend_fn, collision_fn, max_iterations: int = 50):
        if path is None or len(path) < 3:
            return path
        current = list(path)
        for _ in range(max_iterations):
            n = len(current)
            if n < 3:
                break
            i, j = sorted(self.rng.integers(low=0, high=n, size=2))
            if j - i <= 1:
                continue
            seg = list(extend_fn(current[i], current[j]))
            if not seg or any(s is None for s in seg):
                continue
            if any(collision_fn(Capsule(self._bar_pose_from_conf(s), config=[s])) for s in seg):
                continue
            current = current[:i] + seg + current[j:]
        return current

    def _get_collision_fn(self, enable_ik: bool = True, enable_collision: bool = True):
        """Build the collision predicate for a given planning stage.

        Stage 1  (enable_ik=False, enable_collision=False):
            No IK, no collision. Collision predicate always returns False.
        Stage 1b (enable_ik=False, enable_collision=True):
            No joint configs are available; use the floating-body collision
            check (bar pose vs. static obstacles at the neutral robot config).
        Stage 2  (enable_ik=True, enable_collision=False):
            Capsules carry joint configs.  Reject IK-failure sentinels (empty
            config) but skip the full robot-collision check.
        Stage 3  (enable_ik=True, enable_collision=True):
            Full check: reject empty configs + run joint-space collision fn.
        """
        floating_collision_fn = self.robot_setup.create_floating_body_collision_fn(
            obstacle_bodies=self.robot_setup.obstacles
        )
        joint_collision_fn = self._cached_collision_fn

        def fn(c: Capsule):
            if not enable_ik:
                if not enable_collision:
                    return False
                # Stage 1b: no joint config expected; check bar pose against obstacles
                return floating_collision_fn(c.pose)
            # Stages 2 & 3: IK is active
            if len(c.config) == 0:
                return True   # IK-failure sentinel → reject
            if not enable_collision:
                return False  # Stage 2: valid config but skip robot collision
            return self._is_config_collision(c.config[0])  # Stage 3: cached check

        return fn

    def _get_draw_fn(self, start: Capsule, target: Capsule):

        def pose_to_tuple(pose: Tuple[np.ndarray, np.ndarray], decimals: int = 6):
            pos, orn = pose
            pos_tuple = tuple(np.round(pos, decimals=decimals))
            orn_tuple = tuple(np.round(orn, decimals=decimals))
            return pos_tuple + orn_tuple

        def segment_to_tuple(pose1: Tuple[np.ndarray, np.ndarray], pose2: Tuple[np.ndarray, np.ndarray], decimals: int = 6):
            t1 = pose_to_tuple(pose1, decimals)
            t2 = pose_to_tuple(pose2, decimals)
            return tuple(sorted([t1, t2]))

        start_tree_set = set()
        target_tree_set = set()

        pose_cache = set()
        segment_cache = set()

        start_tree_set.add(pose_to_tuple(start.pose))
        target_tree_set.add(pose_to_tuple(target.pose))

        def fn(conf: Capsule, segment: List[Capsule], valid=None, valid_right=None, **kwargs):
            if len(segment) > 0:
                color = pp.BROWN
                pose_1_tup = pose_to_tuple(segment[0].pose)
                pose_2_tup = pose_to_tuple(segment[1].pose)

                if "color" in kwargs:
                    color = kwargs["color"]
                    if color == "red":
                        color = (1, 0, 0, 0.5)
                        width = 4.0
                    if color == "blue":
                        color = (0, 0, 1, 1)
                        width = 2.0
                else:
                    if pose_1_tup in start_tree_set:
                        color = pp.BLUE
                        start_tree_set.add(pose_2_tup)
                    elif pose_2_tup in start_tree_set:
                        color = pp.BLUE
                        start_tree_set.add(pose_1_tup)
                    elif pose_1_tup in target_tree_set:
                        color = pp.RED
                        target_tree_set.add(pose_2_tup)
                    elif pose_2_tup in target_tree_set:
                        color = pp.RED
                        target_tree_set.add(pose_1_tup)

                seg_tuple = segment_to_tuple(segment[0].pose, segment[1].pose)
                if seg_tuple not in segment_cache:
                    pp.add_line(segment[0].pose[0], segment[1].pose[0], width=width, color=color)
                    segment_cache.add(seg_tuple)

        return fn

    def generate_start_configuration(
        self, projector: DualArmProjection, delta_pose_point: List[float] = [0.4, 0.0, 0.75], delta_pose_euler: List[float] = [np.pi, np.pi / 2, np.pi / 2], max_attempts: int = 100, delta_angle: float = np.pi
    ) -> Tuple[np.ndarray, Tuple]:
        global bar_from_right, bar_from_left

        print("Initializing start configuration...")

        # Compute target bar pose from robot base and relative delta
        delta_pose = pp.Pose(point=delta_pose_point, euler=delta_pose_euler)
        world_from_base = pp.get_pose(self.robot_setup.robot)
        world_from_bar = pp.multiply(world_from_base, delta_pose)

        # Get IK solution handles for both arms
        left_start_ik_handle = self.robot_setup.ik_solver_left
        right_start_ik_handle = self.robot_setup.ik_solver_right

        # Generate valid configurations using dual-arm constraint projection
        start_confs = projector.create_valid_confs(
            right_start_ik_handle, world_from_bar, bar_from_right, delta=delta_angle, max_attempts=max_attempts, collision_fn=self._cached_collision_fn
        )

        # Select first valid configuration or exit if none found
        if start_confs is None:
            print(f"✗ Failed to generate start configuration after {max_attempts} attempts")
            print("Consider adjusting delta_pose_point, delta_pose_euler, or increasing max_attempts")
            exit()

        # Normalize configuration to [-π, π] range
        start_confs = normalize_angles(start_confs)

        # Set robot to computed configuration
        # self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, start_confs[0])

        print("✓ Start configuration generated successfully")
        return start_confs, world_from_bar

    def try_direct_path(self, extend_fn, start_capsule: Capsule, target_capsule: Capsule):
        collision_fn = self._get_collision_fn()
        with pp.LockRenderer():
            capsule_path = pp.direct_path(start_capsule, target_capsule, extend_fn, collision_fn)
        return capsule_path

    def try_project_chain(self, capsule_path: List[Capsule], enable_collision: bool) -> Optional[List[np.ndarray]]:
        """Attempt to recover a continuous joint path by chaining IK + projection along poses."""
        if capsule_path is None or len(capsule_path) == 0:
            return None
        if capsule_path[0].config is None or len(capsule_path[0].config) == 0:
            return None

        global bar_from_right
        q_prev = np.asarray(capsule_path[0].config[0], dtype=float)
        path = [q_prev]

        seed_right = q_prev[6:]
        seed_left = q_prev[:6]

        for cap in capsule_path[1:]:
            world_from_bar = cap.pose
            world_from_right = pp.multiply(world_from_bar, bar_from_right)
            right_conf = self.robot_setup.ik_solver_right(world_from_right, seed_right)
            if right_conf is None:
                return None
            conf = self.projector.project(right_conf, seed_left)
            if conf is None:
                return None
            if enable_collision and self._is_config_collision(conf):
                return None
            q_prev = np.asarray(conf, dtype=float)
            seed_right = q_prev[6:]
            seed_left = q_prev[:6]
            path.append(q_prev)

        return path

    def expand(self, capsule: Capsule, max_attempts: Optional[int] = None, delta: Optional[float] = None) -> Capsule:
        global bar_from_right
        bar_pose = capsule.pose
        right_ik_handle = self.robot_setup.ik_solver_right
        if max_attempts is None:
            max_attempts = self.ladder_expand_attempts_full
        if delta is None:
            delta = self.ladder_expand_delta
        cache_key = (self._pose_key(bar_pose), float(delta), int(max_attempts))
        if cache_key in self._ik_cache:
            confs = self._ik_cache[cache_key]
        else:
            confs = self.projector.create_valid_confs(
                right_ik_handle,
                bar_pose,
                bar_from_right,
                delta=delta,
                max_attempts=max_attempts,
                collision_fn=self._cached_collision_fn,
            )
            self._ik_cache[cache_key] = confs
        return Capsule(bar_pose, config=confs, parent=None, robot_setup=self.robot_setup, projector=self.projector)

    def expand_cart_path_ladder_graph(self, path: List[Capsule], profiler: Optional["PlanProfiler"] = None):
        # Returns: (expended_path, updates) where updates is a list of (origin_node, new_config)
        def _copy_and_update(origin_capsule, new_capsule):
            if origin_capsule is None:
                return
            try:
                new_conf = copy.deepcopy(new_capsule.config)
            except Exception:
                new_conf = new_capsule.config
            try:
                origin_capsule.config = new_conf
            except Exception:
                pass
            updates.append((origin_capsule, new_conf))

        def _build_connections(rungs: List[Capsule]):
            for i in range(1, len(rungs)):
                rungs[i].set_parent(rungs[i - 1], dist_thresh=self.ladder_edge_thresh)

        def _has_any_edge(rungs: List[Capsule]) -> bool:
            for i in range(1, len(rungs)):
                prev = rungs[i - 1]
                curr = rungs[i]
                if prev is None or curr is None:
                    return False
                if curr.connection is None or len(curr.connection) == 0:
                    return False
                if all(len(c) == 0 for c in curr.connection):
                    return False
            return True

        expended_path: List[Capsule] = []
        updates: List[Tuple[Capsule, Optional[List[np.ndarray]]]] = []

        # Pass 1: small expansion for missing rungs
        for capsule in path:
            origin = getattr(capsule, "_origin", None)
            if len(capsule.config) > 1 or capsule.parent is None:
                capsule.parent = None
                expended_path.append(capsule)
                _copy_and_update(origin, capsule)
            else:
                _prof = profiler if profiler is not None else PlanProfiler()
                with _prof.measure("expand_single (create_valid_confs)"):
                    expanded = self.expand(
                        capsule,
                        max_attempts=self.ladder_expand_attempts_small,
                        delta=min(self.ladder_expand_delta, np.pi / 6.0),
                    )
                expended_path.append(expanded)
                _copy_and_update(origin, expanded)

        _build_connections(expended_path)
        if _has_any_edge(expended_path):
            return expended_path, updates

        # Pass 2: full expansion only where needed
        for i, capsule in enumerate(expended_path):
            if capsule.config is None or len(capsule.config) <= 1:
                _prof = profiler if profiler is not None else PlanProfiler()
                with _prof.measure("expand_single (create_valid_confs)"):
                    expanded = self.expand(
                        capsule,
                        max_attempts=self.ladder_expand_attempts_full,
                        delta=self.ladder_expand_delta,
                    )
                expended_path[i] = expanded
                _copy_and_update(getattr(capsule, "_origin", None), expanded)

        _build_connections(expended_path)
        return expended_path, updates


    def plan(
        self,
        start_conf: np.ndarray,
        target_conf: np.ndarray,
        max_time: float = pp.INF,
        max_iterations: int = 200,
        max_attempts: int = 40,
        smooth_iterations: int = 10,
        use_draw: bool = False,
        verbose: bool = False,
        enable_ik: bool = True,
        enable_collision: bool = True,
        dist_metric: Optional[str] = None,
        ladder_search: Optional[str] = None,
        ladder_expand_delta: Optional[float] = None,
        start_goal_delta: Optional[float] = None,
        goal_sample_prob: Optional[float] = None,
        return_task_path: bool = False,
        guide_poses: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        warm_start_path: Optional[List[np.ndarray]] = None,
        warm_start_first: bool = True,
        start_bar_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        target_bar_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        random_seed: Optional[int] = None,
        diagnostics_out: Optional[dict] = None,
        debug_tree_out: Optional[dict] = None,
    ) -> Optional[List[np.ndarray]]:
        """Plan a dual-arm path from start_conf to target_conf.

        Three debug stages are selectable via enable_ik / enable_collision:

          Stage 1 — task-space only  (enable_ik=False, enable_collision=True)
            Bar poses are interpolated with no IK.  Collision uses the
            floating-body check (bar pose vs. static obstacles).  The planner
            always returns None; use use_draw=True + verbose=True to inspect
            the RRT workspace coverage.

          Stage 2 — IK on, collision off  (enable_ik=True, enable_collision=False)
            Each waypoint solves dual-arm IK; IK failures stop extension.
            Robot self/environment collision is not checked.

          Stage 3 — full  (enable_ik=True, enable_collision=True)  [default]
            Full IK + joint-space collision checking.

        Args:
            start_conf: 12-DOF joint configuration (left+right).
            target_conf: 12-DOF joint configuration (left+right).
            max_time: Maximum planning time per attempt.
            max_iterations: Maximum RRT-Connect iterations per attempt.
            max_attempts: Number of random restarts.
            smooth_iterations: Shortcut iterations (kept for API symmetry).
            use_draw: If True, enable debug drawing during planning.
            verbose: If True, print planner progress.
            enable_ik: Enable IK solving in extend/cspace functions (Stage 2+).
            enable_collision: Enable full joint-space collision checking (Stage 3).

        Returns:
            List of joint configurations (path) or None if planning fails.
        """
        global bar_from_right, bar_from_left

        # Preserve the caller-provided grasp transform when available.
        # Re-deriving it from the current bar body pose is only a fallback, because the
        # debug/GUI bar body may not match the start/goal configuration being planned.
        if bar_from_right is None or bar_from_left is None:
            with pp.WorldSaver():
                self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, target_conf)
                world_from_right = pp.get_link_pose(self.robot_setup.robot, self.robot_setup.tool_link_right)
                world_from_left = pp.get_link_pose(self.robot_setup.robot, self.robot_setup.tool_link_left)
                world_from_bar = pp.get_pose(self.robot_setup.target_bar)

                bar_from_right = pp.multiply(pp.invert(world_from_bar), world_from_right)
                bar_from_left = pp.multiply(pp.invert(world_from_bar), world_from_left)

        # Compute bar poses for start and target.
        # In Stage 1 callers may provide explicit task-space poses and skip IK entirely.
        def _bar_pose_from_conf(conf: np.ndarray):
            self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, conf)
            right_pose = pp.get_link_pose(self.robot_setup.robot, self.robot_setup.tool_link_right)
            return pp.multiply(right_pose, pp.invert(bar_from_right))

        world_from_bar_start = start_bar_pose if start_bar_pose is not None else _bar_pose_from_conf(start_conf)
        world_from_bar_target = target_bar_pose if target_bar_pose is not None else _bar_pose_from_conf(target_conf)
        self._goal_pose = world_from_bar_target
        if goal_sample_prob is not None:
            self.goal_sample_prob = float(goal_sample_prob)
        self._guide_poses = guide_poses
        if random_seed is not None:
            self.random_seed = int(random_seed)
            self.rng = np.random.default_rng(self.random_seed)
        if debug_tree_out is not None:
            debug_tree_out["start_pose"] = [
                float(world_from_bar_start[0][0]),
                float(world_from_bar_start[0][1]),
                float(world_from_bar_start[0][2]),
            ]
            debug_tree_out["goal_pose"] = [
                float(world_from_bar_target[0][0]),
                float(world_from_bar_target[0][1]),
                float(world_from_bar_target[0][2]),
            ]

        # Prepare confs (plural) for capsules using projector; fallback to provided confs.
        # In Stage 1 (enable_ik=False) the start/goal configs are already known; skip IK.
        robot_collision_fn = self._cached_collision_fn

        if enable_ik:
            delta_start_goal = self.start_goal_delta if start_goal_delta is None else float(start_goal_delta)
            start_confs_candidates = self.projector.create_valid_confs(
                self.robot_setup.ik_solver_right,
                world_from_bar_start,
                bar_from_right,
                delta=delta_start_goal,
                max_attempts=40,
                collision_fn=robot_collision_fn,
            )
            target_confs_candidates = self.projector.create_valid_confs(
                self.robot_setup.ik_solver_right,
                world_from_bar_target,
                bar_from_right,
                delta=delta_start_goal,
                max_attempts=40,
                collision_fn=robot_collision_fn,
            )
        else:
            start_confs_candidates = None   # _prepare_confs falls back to start_conf
            target_confs_candidates = None  # _prepare_confs falls back to target_conf

        def _prepare_confs(default_conf, confs_list):
            if confs_list is None or len(confs_list) == 0:
                return [normalize_angles(np.asarray(default_conf, dtype=float))]
            return [normalize_angles(np.asarray(q, dtype=float)) for q in confs_list]

        start_confs_prepared = _prepare_confs(start_conf, start_confs_candidates)
        target_confs_prepared = _prepare_confs(target_conf, target_confs_candidates)

        start_capsule = Capsule(world_from_bar_start, config=start_confs_prepared, parent=None, robot_setup=self.robot_setup, projector=self.projector)
        target_capsule = Capsule(world_from_bar_target, config=target_confs_prepared, parent=None, robot_setup=self.robot_setup, projector=self.projector)
        self._attach_feature_vec(start_capsule)
        self._attach_feature_vec(target_capsule)

        stage_label = (
            "Stage 1 (task-space only, no collision)" if (not enable_ik and not enable_collision) else
            "Stage 1b (task-space + floating collision)" if (not enable_ik and enable_collision) else
            "Stage 2 (IK, no collision)" if not enable_collision else
            "Stage 3 (full)"
        )
        if verbose:
            print(f"[plan] Planning stage: {stage_label}")

        if ladder_expand_delta is not None:
            self.ladder_expand_delta = float(ladder_expand_delta)
        if dist_metric is not None:
            self.dist_metric = dist_metric
        if ladder_search is not None:
            self.ladder_search = ladder_search

        extend_fn = self._get_extend_fn(enable_ik=enable_ik)
        collision_fn = self._get_collision_fn(enable_ik=enable_ik, enable_collision=enable_collision)
        distance_fn = self._get_cart_dist_fn(metric=self.dist_metric)
        sample_fn = self._get_sample_fn(enable_ik=False)  # sampling always pose-only
        draw_fn = self._get_draw_fn(start_capsule, target_capsule) if use_draw else None

        cspace_extend_fn = self._get_cspace_extend_with_projection_fn(enable_ik=enable_ik)

        profiler = PlanProfiler()

        # Failure attribution counters
        n_rrt_failed = 0      # RRT did not connect (task-space failure)
        n_ladder_failed = 0   # RRT connected, but ladder graph found no joint path

        if enable_collision and warm_start_path is not None and warm_start_first:
            with profiler.measure("warm_start_smooth"):
                smooth_path = self._smooth_with_collision(warm_start_path, cspace_extend_fn, collision_fn, max_iterations=50)
            with profiler.measure("warm_start_interp"):
                smooth_path = interpolate(smooth_path, cspace_extend_fn, robot_setup=self.robot_setup)
            if smooth_path is not None and len(smooth_path) > 1:
                profiler.report("plan() sub-operation breakdown (warm-start)")
                if diagnostics_out is not None:
                    diagnostics_out.clear()
                    diagnostics_out.update(
                        {
                            "success": True,
                            "stage_label": stage_label,
                            "attempts": 0,
                            "rrt_failed": n_rrt_failed,
                            "ladder_failed": n_ladder_failed,
                            "returned_from": "warm_start_pre_rrt",
                        }
                    )
                return smooth_path

        for attempt in range(max_attempts):

            pp.remove_all_debug()

            with profiler.measure("rrt_connect_capsule"):
                path = rrt_connect_capsule(
                    self,
                    start_capsule,
                    target_capsule,
                    distance_fn,
                    sample_fn,
                    extend_fn,
                    collision_fn,
                    self.robot_setup,
                    self.projector,
                    max_iterations=max_iterations,
                    max_time=max_time,
                    verbose=verbose,
                    draw_fn=draw_fn,
                    enforce_alternate=False,
                    profiler=profiler,
                    enable_ik=enable_ik,
                    ladder_search=self.ladder_search,
                    return_task_path=return_task_path,
                    enable_collision=enable_collision,
                    debug_tree_out=debug_tree_out,
                )

            if path is None:
                # Distinguish: did RRT not connect, or did it connect but ladder fail?
                # rrt_connect_capsule returns None in both cases; the profiler timing
                # for "ladder_graph_search" being nonzero indicates at least one connection.
                ladder_time = profiler.get_cumulative("ladder_graph_search")
                if ladder_time > 0:
                    n_ladder_failed += 1
                    if verbose:
                        print(f"[plan] Attempt {attempt+1}/{max_attempts}: "
                              f"RRT connected but ladder search failed.")
                else:
                    n_rrt_failed += 1
                    if verbose:
                        print(f"[plan] Attempt {attempt+1}/{max_attempts}: "
                              f"RRT did not connect (task-space failure).")
                continue

            if enable_collision and warm_start_path is not None:
                with profiler.measure("warm_start_smooth"):
                    smooth_path = self._smooth_with_collision(warm_start_path, cspace_extend_fn, collision_fn, max_iterations=30)
                with profiler.measure("warm_start_interp"):
                    smooth_path = interpolate(smooth_path, cspace_extend_fn, robot_setup=self.robot_setup)
                if smooth_path is not None and len(smooth_path) > 1:
                    if diagnostics_out is not None:
                        diagnostics_out.clear()
                        diagnostics_out.update(
                            {
                                "success": True,
                                "stage_label": stage_label,
                                "attempts": attempt + 1,
                                "rrt_failed": n_rrt_failed,
                                "ladder_failed": n_ladder_failed,
                                "returned_from": "warm_start_post_rrt",
                            }
                        )
                    return smooth_path

            if not enable_ik and return_task_path:
                # Return the raw task-space path (poses) for diagnosis
                if diagnostics_out is not None:
                    diagnostics_out.clear()
                    diagnostics_out.update(
                        {
                            "success": True,
                            "stage_label": stage_label,
                            "attempts": attempt + 1,
                            "rrt_failed": n_rrt_failed,
                            "ladder_failed": n_ladder_failed,
                            "returned_from": "task_path",
                        }
                    )
                return [c.pose if hasattr(c, "pose") else c for c in path]

            # Post-process in joint space using provided helpers
            if smooth_iterations is not None and int(smooth_iterations) > 0:
                with profiler.measure("smooth"):
                    path_smooth = joint_space_smooth(
                        path,
                        cspace_extend_fn,
                        max_iterations=int(smooth_iterations),
                        robot_setup=self.robot_setup,
                    )
            else:
                path_smooth = path
            with profiler.measure("interpolate"):
                path_res = interpolate(path_smooth, cspace_extend_fn, robot_setup=self.robot_setup)

            profiler.report("plan() sub-operation breakdown")

            for q in path_res:
                self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, q)
                pp.draw_pose(pp.get_link_pose(self.robot_setup.robot, self.robot_setup.tool_link_right), length=0.01)

            if diagnostics_out is not None:
                diagnostics_out.clear()
                diagnostics_out.update(
                    {
                        "success": True,
                        "stage_label": stage_label,
                        "attempts": attempt + 1,
                        "rrt_failed": n_rrt_failed,
                        "ladder_failed": n_ladder_failed,
                    }
                )
            return path_res

        print(f"[plan] {stage_label} — failed after {max_attempts} attempts: "
              f"RRT-failed={n_rrt_failed}, ladder-failed={n_ladder_failed}")
        profiler.report("plan() sub-operation breakdown (no path found)")
        if diagnostics_out is not None:
            diagnostics_out.clear()
            diagnostics_out.update(
                {
                    "success": False,
                    "stage_label": stage_label,
                    "attempts": max_attempts,
                    "rrt_failed": n_rrt_failed,
                    "ladder_failed": n_ladder_failed,
                }
            )
        return None

def main():
    """
    Example usage of TrajectoryDualConstrainedSolver.

    Usage:
        python trajectory_dual_cart_constrained_solver.py          # GUI on (default)
        python trajectory_dual_cart_constrained_solver.py --no-gui # GUI off (faster)
    """
    import argparse as _argparse
    _parser = _argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--no-gui", action="store_true", help="Disable PyBullet GUI")
    _args, _ = _parser.parse_known_args()
    gui = not _args.no_gui

    # np.random.seed(1281712)

    # Configuration paths
    design_study_path = os.path.join(DATA_DIR, "husky_assembly_design_study")

    # ------------------------------
    design_case = "250904_transfer_path_test"
    target_name = "IK_test__20250909_235058"
    # target_name = "IK_test__20250905_101010"
    state_name = "IK_test__GraspTargets"
    # ------------------------------ failed
    # design_case = "250707_RobotX_box_demo"
    # target_name = "robotx_box_A13-S_end"
    # state_name = "robotx_box_A13-S_end_GraspTargets"
    # ------------------------------
    # design_case = "250929_New_Antenna_with_GH_RH_Packed"
    # target_name = "D1"
    # state_name = "D1_GraspTargets"

    target_cell_state_path = os.path.join(design_study_path, design_case, "RobotCellStates", f"{target_name}_RobotCellState.json")

    # ------------------------------------------------------------------
    # Initialize Robot Setup for Planning
    # ------------------------------------------------------------------
    robot_setup, target_conf, projector = TrajectoryDualCartConstrainedSolver.initialize_robot_setup_for_planning(
        robot_name="r0", robot_type="husky_dual", target_cell_state_path=target_cell_state_path, use_scene_parser_gui=gui, scene_parser_verbose=True
    )

    world_from_right = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
    world_from_left = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
    world_from_bar = pp.get_pose(robot_setup.target_bar)
    world_from_bar_target = world_from_bar

    global bar_from_right, bar_from_left

    bar_from_right = pp.multiply(pp.invert(world_from_bar), world_from_right)
    bar_from_left = pp.multiply(pp.invert(world_from_bar), world_from_left)

    # ------------------------------------------------------------------
    # Initialize Target Parser
    # ------------------------------------------------------------------
    target_parser = TargetParser(os.path.join(design_study_path, design_case), f"{state_name}.json")

    # ------------------------------------------------------------------
    # Initialize Trajectory Solver
    # ------------------------------------------------------------------
    print("Initializing TrajectoryDualCartConstrainedSolver...")
    solver = TrajectoryDualCartConstrainedSolver(robot_setup, target_parser, projector)

    start_confs, _ = solver.generate_start_configuration(projector, max_attempts=20, delta_angle=np.pi * 2)
    robot_setup.set_joint_positions(robot_setup.arm_joints, start_confs[0])

    time_start = time.time()
    path = solver.plan(start_conf=start_confs[0], target_conf=target_conf, max_time=120, max_iterations=10000, use_draw=gui, verbose=True)
    print(f"Planning took {time.time() - time_start:.2f} seconds")

    if path is not None:
        print(f"Path found: {len(path)} waypoints")
        if gui:
            slider = pybullet.addUserDebugParameter("path_idx", 0, len(path) - 1, 0)
            current_index = -1
            while True:
                idx = int(pybullet.readUserDebugParameter(slider))
                if idx != current_index:
                    current_index = idx
                    conf = path[current_index]
                    robot_setup.set_joint_positions(robot_setup.arm_joints, conf)
                time.sleep(0.01)
    else:
        print("No path found")
        if gui:
            pp.wait_for_user("No path found")


if __name__ == "__main__":
    main()
