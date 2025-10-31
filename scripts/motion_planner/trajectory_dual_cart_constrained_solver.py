import argparse
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
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from model.target_parse import TargetParser
from robot.dual_arm_projection import DualArmProjection
from robot.robot_setup import RobotSetup
from utils.params import DATA_DIR, PROJECT_DIR
from utils.util import angles_distance, normalize_angles

# DEFAULT_RESOLUTION = math.radians(1.0)
DEFAULT_RESOLUTION = np.deg2rad(5.0)
DEFAULT_NORM = 2

bar_from_right = None
bar_from_left = None

np.set_printoptions(precision=4, suppress=False)


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

    def set_parent(self, parent: "Capsule"):
        self.parent = parent
        for _ in range(len(self.config)):
            self.connection.append([])
        for i in range(len(self.config)):
            for j in range(len(parent.config)):
                dist = np.linalg.norm(angles_distance(self.config[i], parent.config[j]))
                # print(f"i: {i}, j: {j}, dist: {dist}")
                if dist < 0.75:
                    self.connection[i].append(j)

    def __str__(self):
        return "Capsule(" + str(self.pose) + ", " + str(len(self.config)) + ")"

    __repr__ = __str__


def asymmetric_extend(c1: Capsule, c2: Capsule, extend_fn, backward=False):
    if backward:
        return reversed(list(extend_fn(c2, c1)))
    return extend_fn(c1, c2)


def extend_towards_capsule(tree: List[Capsule], target: Capsule, distance_fn, extend_fn, collision_fn, robot_setup, projector, swap=False, tree_frequency=1, **kwargs):
    target = copy.deepcopy(target)
    target.parent = None
    last = pp.utils.argmin(lambda n: float(np.linalg.norm(np.asarray(distance_fn(n, target), dtype=float), ord=2)), tree)
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


def configs_capsule(nodes: List[Capsule], distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None):
    """Enumerate all feasible configuration paths across capsule rungs.

    Args:
        nodes: Ladder of `Capsule` rungs with `config` lists and inter-rung `connection` info.
        distance_fn: Function mapping (q_prev, q_curr) -> scalar step distance. If None,
                     uses L2 norm of angular difference via `angles_distance`.

    Returns:
        List of tuples (path, chosen_indices, total_distance), sorted by total_distance ascending.
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
        # All single-node choices have zero path length; return each as a feasible path
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

    def dfs(rung_idx: int, curr_idx: int, chosen_indices: List[int], path: List[np.ndarray], total_cost: float):
        # At rung_idx refers to current rung index (0-based). If we reached last rung, record
        if rung_idx == num_nodes - 1:
            results.append((path, chosen_indices, float(total_cost)))
            return
        # Explore all successors on next rung
        successors = edges_per_level[rung_idx].get(curr_idx, [])
        for nxt_idx in successors:
            q_prev = nodes[rung_idx].config[curr_idx]
            q_curr = nodes[rung_idx + 1].config[nxt_idx]
            step_cost = dist_fn(q_prev, q_curr)
            dfs(rung_idx + 1, nxt_idx, chosen_indices + [nxt_idx], path + [q_curr], total_cost + step_cost)

    # Start from every feasible index on the first rung that has at least one outgoing edge
    first_edges = edges_per_level[0]
    for start_idx in range(len(nodes[0].config)):
        if len(first_edges.get(start_idx, [])) == 0:
            continue
        q0 = nodes[0].config[start_idx]
        dfs(0, start_idx, [start_idx], [q0], 0.0)

    # Sort by total cost ascending
    results.sort(key=lambda item: item[2])
    return results


def plot_ladder_graph(capsule_path: List[Capsule], highlight_feasible: bool = False) -> Optional[str]:
    """
    Draw ladder graph nodes (per-rung IK solutions) and inter-rung connections, then save as SVG.

    When highlight_feasible is True, computes a feasible joint sequence via configs_capsule
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

        results = configs_capsule(capsule_path, distance_fn=_js_dist)
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
    **kwargs,
):
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
        if draw_fn:
            bodies = pp.draw_pose(target.pose, length=0.1)
            # draw_fn(target, [])

        last1, _, tree1 = extend_towards_capsule(tree1, target, distance_fn, extend_fn, collision_fn, robot_setup, projector, swap, **kwargs)
        last1_independent = copy.deepcopy(last1)
        last1_independent.parent = None
        last2, success, tree2 = extend_towards_capsule(tree2, last1_independent, distance_fn, extend_fn, collision_fn, robot_setup, projector, not swap, **kwargs)

        # if draw_fn:
        #     for sp in tree1 + tree2:
        #         sp.draw(draw_fn)

        if draw_fn:
            for sp in nodes1:
                sp.draw(draw_fn, color="red")

            for sp in nodes2:
                sp.draw(draw_fn, color="blue")

        if success:
            path1, path2 = last1.retrace(), last2.retrace()
            if swap:
                path1, path2 = path2, path1
            if verbose:
                print(f"RRT connect capsule: {iteration} iterations, {len(nodes1) + len(nodes2)} nodes")
            capsule_nodes = path1 + path2[::-1]
            capsule_path = check_capsule_path(capsule_nodes)

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

            # with pp.LockRenderer():
            #     expended_path, updates = solver.expand_path(capsule_path)
            expended_path, updates = solver.expand_path(capsule_path)
            results = configs_capsule(expended_path, distance_fn=_js_dist)
            if results is None or len(results) == 0:
                for debug_body in bodies:
                    pp.remove_debug(debug_body)
                continue  # way 1: restart in the loop
                # break  # way 2: break out of the loop and restart out of the loop

            with pp.LockRenderer():
                plot_capsule_path(capsule_path)
            plot_ladder_graph(expended_path, highlight_feasible=True)

            best_path, _, _ = results[0]
            return best_path

        for debug_body in bodies:
            pp.remove_debug(debug_body)
    return None


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

    def __init__(self, robot_setup: RobotSetup, target_parser: TargetParser, projector: DualArmProjection):
        self.robot_setup = robot_setup
        self.target_parser = target_parser
        self.projector = projector

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
                x = cx + np.random.uniform(-l / 2, l / 2)
                y = cy + np.random.uniform(-w / 2, w / 2)
                z = cz + h / 2

                roll = np.random.uniform(*roll_range)
                pitch = np.random.uniform(*pitch_range)
                yaw = np.random.uniform(*yaw_range)

                world_from_bar = pp.Pose(point=[x, y, z], euler=pp.Euler(roll, pitch, yaw))
                if enable_ik:
                    confs = self.projector.create_valid_confs(
                        self.robot_setup.ik_solver_right, world_from_bar, bar_from_right, delta=0.0, max_attempts=20, collision_fn=self.robot_setup.create_collision_fn(obstacle_bodies=self.robot_setup.obstacles)
                    )
                else:
                    confs = None

                if enable_ik and confs is not None:
                    return Capsule(world_from_bar, confs, parent=None, robot_setup=self.robot_setup, projector=self.projector)
                elif not enable_ik:
                    return Capsule(world_from_bar, confs, parent=None, robot_setup=self.robot_setup, projector=self.projector)

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
                    seed_right = last_capsule.config[0][6:]
                    seed_left = last_capsule.config[0][:6]

                    right_conf = self.robot_setup.ik_solver_right(world_from_right, seed_right)
                    if right_conf is not None:
                        conf = self.projector.project(right_conf, seed_left)

                if conf is not None:
                    capsule = Capsule(world_from_bar, config=[conf], parent=None, robot_setup=self.robot_setup, projector=self.projector)
                    yield capsule
                    last_capsule = capsule
                else:
                    capsule = Capsule(world_from_bar, parent=None, robot_setup=self.robot_setup, projector=self.projector)
                    yield capsule
                    if enable_ik:
                        break
                    else:
                        last_capsule = capsule

        return fn

    def _get_distance_fn(self):
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

        def fn(c1: Capsule, c2: Capsule):
            R1, p1 = _pose_to_Rp(c1.pose)
            R2, p2 = _pose_to_Rp(c2.pose)

            dp = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)

            ax = _angle_between_unit_vectors(R1[:, 0], R2[:, 0])
            ay = _angle_between_unit_vectors(R1[:, 1], R2[:, 1])
            az = _angle_between_unit_vectors(R1[:, 2], R2[:, 2])

            return np.array([dp[0], dp[1], dp[2], ax, ay, az], dtype=float)

        return fn

    def _get_collision_fn(self):
        floating_collision_fn = self.robot_setup.create_floating_body_collision_fn(obstacle_bodies=self.robot_setup.obstacles)
        collision_fn = self.robot_setup.create_collision_fn(obstacle_bodies=self.robot_setup.obstacles)

        def fn(c: Capsule):
            # self.robot_setup.set_joint_positions(self.robot_setup.arm_joints, np.array([0] * 12))
            # return floating_collision_fn(c.pose)
            if len(c.config) == 0:
                return True
            return collision_fn(c.config[0])

        return fn

    def _get_draw_fn(self, start: Capsule, target: Capsule):

        def pose_to_tuple(pose: Tuple[np.ndarray, np.ndarray], decimals: int = 3):
            pos, orn = pose
            pos_tuple = tuple(np.round(pos, decimals=decimals))
            orn_tuple = tuple(np.round(orn, decimals=decimals))
            return pos_tuple + orn_tuple

        def segment_to_tuple(pose1: Tuple[np.ndarray, np.ndarray], pose2: Tuple[np.ndarray, np.ndarray], decimals: int = 3):
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
                        color = pp.RED
                    if color == "blue":
                        color = pp.BLUE
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
                    pp.add_line(segment[0].pose[0], segment[1].pose[0], width=2.0, color=color)
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
            right_start_ik_handle, world_from_bar, bar_from_right, delta=delta_angle, max_attempts=max_attempts, collision_fn=self.robot_setup.create_collision_fn(obstacle_bodies=self.robot_setup.obstacles)
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

    def expand(self, capsule: Capsule) -> Capsule:
        global bar_from_right
        bar_pose = capsule.pose
        right_ik_handle = self.robot_setup.ik_solver_right
        confs = self.projector.create_valid_confs(right_ik_handle, bar_pose, bar_from_right, delta=0.0, max_attempts=20, collision_fn=self.robot_setup.create_collision_fn(obstacle_bodies=self.robot_setup.obstacles))
        return Capsule(bar_pose, config=confs, parent=None, robot_setup=self.robot_setup, projector=self.projector)

    def expand_path(self, path: List[Capsule]):
        # Returns: (expended_path, updates) where updates is a list of (origin_node, new_config)
        expended_path: List[Capsule] = []
        updates: List[Tuple[Capsule, Optional[List[np.ndarray]]]] = []

        for capsule in path:
            origin = getattr(capsule, "_origin", None)

            if len(capsule.config) > 1 or capsule.parent is None:
                capsule.parent = None
                expended_path.append(capsule)
                if origin is not None:
                    try:
                        new_conf = copy.deepcopy(capsule.config)
                    except Exception:
                        new_conf = capsule.config
                    try:
                        origin.config = new_conf
                    except Exception:
                        pass
                    updates.append((origin, new_conf))
            else:
                expanded = self.expand(capsule)
                expended_path.append(expanded)
                if origin is not None:
                    try:
                        new_conf = copy.deepcopy(expanded.config)
                    except Exception:
                        new_conf = expanded.config
                    try:
                        origin.config = new_conf
                    except Exception:
                        pass
                    updates.append((origin, new_conf))

        for i, capsule in enumerate(expended_path):
            if i == 0:
                continue
            expended_path[i].set_parent(expended_path[i - 1])

        return expended_path, updates


def main():
    """
    Example usage of TrajectoryDualConstrainedSolver.
    """

    # np.random.seed(1281712)

    # Configuration paths
    design_study_path = os.path.join(DATA_DIR, "husky_assembly_design_study")

    # ------------------------------
    # design_case = "250904_transfer_path_test"
    # target_name = "IK_test__20250909_235058"
    # # target_name = "IK_test__20250905_101010"
    # state_name = "IK_test__GraspTargets"
    # ------------------------------ failed
    # design_case = "250707_RobotX_box_demo"
    # target_name = "robotx_box_A13-S_end"
    # state_name = "robotx_box_A13-S_end_GraspTargets"
    # ------------------------------
    design_case = "250707_RobotX_box_demo"
    target_name = "robotx_box_A6-S4_end"
    state_name = "robotx_box_A6-S4_end_GraspTargets"

    target_cell_state_path = os.path.join(design_study_path, design_case, "RobotCellStates", f"{target_name}_RobotCellState.json")

    # ------------------------------------------------------------------
    # Initialize Robot Setup for Planning
    # ------------------------------------------------------------------
    robot_setup, target_conf, projector = TrajectoryDualCartConstrainedSolver.initialize_robot_setup_for_planning(
        robot_name="r0", robot_type="husky_dual", target_cell_state_path=target_cell_state_path, use_scene_parser_gui=True, scene_parser_verbose=True
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

    start_confs, world_from_bar_start = solver.generate_start_configuration(projector, max_attempts=20, delta_angle=np.pi * 2)
    robot_setup.set_joint_positions(robot_setup.arm_joints, start_confs[0])

    robot_setup.set_joint_positions(robot_setup.arm_joints, target_conf)
    world_from_bar = pp.get_pose(robot_setup.target_bar)
    target_confs = projector.create_valid_confs(solver.robot_setup.ik_solver_right, world_from_bar, bar_from_right, delta=0.0, max_attempts=20, collision_fn=robot_setup.create_collision_fn(obstacle_bodies=robot_setup.obstacles))

    # pp.wait_for_user()

    start_capsule = Capsule(world_from_bar_start, config=start_confs, parent=None, robot_setup=robot_setup, projector=projector)
    target_capsule = Capsule(world_from_bar_target, config=target_confs, parent=None, robot_setup=robot_setup, projector=projector)

    way_points = solver.cart_linear_interp(start_capsule, target_capsule, position_res=0.1, rotation_res=0.05)

    # with pp.LockRenderer():
    #     for way_point in way_points:
    #         pp.draw_pose(way_point, length=0.25)

    # pp.wait_for_user()

    extend_fn = solver._get_extend_fn(enable_ik=True)
    collision_fn = solver._get_collision_fn()
    distance_fn = solver._get_distance_fn()
    sample_fn = solver._get_sample_fn(enable_ik=False)
    draw_fn = solver._get_draw_fn(start_capsule, target_capsule)

    path = None
    # capsule_path = solver.try_direct_path(extend_fn, start_capsule, target_capsule)
    # plot_capsule_path(capsule_path, highlight_feasible=True)
    # if capsule_path is not None:
    #     result = configs_capsule(capsule_path)
    #     path = None if result is None else result[0]

    time_start = time.time()

    if path is None:
        path = rrt_connect_capsule(solver, start_capsule, target_capsule, distance_fn, sample_fn, extend_fn, collision_fn, robot_setup, projector, draw_fn=draw_fn, verbose=True, enforce_alternate=False)
        # path = random_restarts_capsule(solver, start_capsule, target_capsule, distance_fn, sample_fn, extend_fn, collision_fn, robot_setup, projector, draw_fn=draw_fn, verbose=False, restarts=10, max_time=120)

    print(f"RRT connect capsule took {time.time() - time_start:.2f} seconds")

    if path is not None:
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
        pp.wait_for_user("No path found")


if __name__ == "__main__":
    main()
