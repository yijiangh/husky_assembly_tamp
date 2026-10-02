"""A task-space BiRRT that starts from an approximate path (way 1 of the holding-roadmap refinement).

The holding roadmap gives a start a route of straight joint-space edges. The
exact tracker (``holding_prm.tracking``) turns most of those edges into
waypoints that hold the bar exactly, but some edges fail (no IK, a branch
jump, or a collision on the corrected path). This planner bridges those
failures with the production planner's own machinery, keeping what already
worked:

* Tree A starts as the chain of exact waypoints from the start up to the
  first failed edge; tree B as the chain from the goal back to the last
  failed edge. Each waypoint is a tree node with its configuration, so the
  trees begin deep inside the route instead of at its two ends.
* Random targets are bar poses in a tube around the route's bar poses (a
  Gaussian shift and turn around a randomly picked guide pose, guide poses on
  failed edges picked more often), with a small chance of aiming straight at
  the other tree's tip.
* Growth and connection are ``core.extend_toward`` (warm IK from the parent
  waypoint, the continuity gate, the full collision check) and
  ``core.nearest_node``. Where the trees meet, the goal side is re-solved
  from the seam as ``core.plan_pose_birrt`` does.

! This module imports the planner core. It must never be imported from
! ``holding_prm`` (importing the core empties its log file); only
! ``scripts/prm_bridge.py`` uses it, after moving the log folder.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pybullet_planning.motion_planners.rrt import TreeNode

from husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.core import (
    DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
    _noop_pose_collision_fn,
    extend_toward,
    get_bar_feature_points,
    joint_step_exceeds_threshold,
    nearest_node,
    pose_to_feature_vec,
    solve_dual_arm_pose_ik,
)
from husky_assembly_tamp.motion_planner.holding_prm.holding import matrix_from_pose, pose_from_matrix

# A bar pose as the core passes it around: ((x, y, z), (qx, qy, qz, qw)).
Pose = Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]


def as_pose(pose: Sequence[Sequence[float]]) -> Pose:
    """Turn a ``[[x, y, z], [qx, qy, qz, qw]]`` pose into the core's tuple form."""
    return tuple(float(v) for v in pose[0]), tuple(float(v) for v in pose[1])


def chain_tree(poses: Sequence[Pose], confs: Sequence[Sequence[float]],
               feature_points: Sequence[np.ndarray]) -> Tuple[List[TreeNode], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """A tree that is a single chain of known, exact waypoints (root first).

    Args:
        poses (Sequence[Pose]): the bar poses, root first.
        confs (Sequence[Sequence[float]]): the matching 12-joint configurations.
        feature_points (Sequence[np.ndarray]): the bar feature points of the
            nearest-neighbour metric.

    Returns:
        Tuple: ``(nodes, node_confs, feature_vecs)`` in the form
        ``core.extend_toward`` expects (side caches keyed by ``id(node)``).
    """
    nodes: List[TreeNode] = []
    node_confs: Dict[int, np.ndarray] = {}
    feature_vecs: Dict[int, np.ndarray] = {}
    parent = None
    for pose, conf in zip(poses, confs):
        node = TreeNode(pose, parent=parent)
        nodes.append(node)
        node_confs[id(node)] = np.asarray(conf, dtype=float)
        vector = pose_to_feature_vec(pose, feature_points)
        if vector is not None:
            feature_vecs[id(node)] = vector
        parent = node
    return nodes, node_confs, feature_vecs


def sample_in_tube(rng: np.random.Generator, guide_poses: Sequence[Pose], guide_weights: np.ndarray,
                   pos_sigma_m: float, rot_sigma_rad: float) -> Pose:
    """A random bar pose near the route: a guide pose, shifted and turned a little.

    Args:
        rng (np.random.Generator): the random source.
        guide_poses (Sequence[Pose]): bar poses along the route.
        guide_weights (np.ndarray): how often each guide pose is picked.
        pos_sigma_m (float): spread of the shift, per axis (m).
        rot_sigma_rad (float): spread of the turn angle (rad), about a random axis.

    Returns:
        Pose: the sample.
    """
    index = int(rng.choice(len(guide_poses), p=guide_weights / guide_weights.sum()))
    frame = matrix_from_pose(guide_poses[index])
    frame[:3, 3] += rng.normal(0.0, pos_sigma_m, 3)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.normal(0.0, rot_sigma_rad)
    # Rodrigues' formula: the rotation by `angle` about `axis`.
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    turn = np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross
    frame[:3, :3] = frame[:3, :3] @ turn
    return as_pose(pose_from_matrix(frame))


def plan_seeded_birrt(
    ik_context: Dict[str, Any],
    joint_collision_fn: Callable[[Sequence[float]], bool],
    start_poses: Sequence[Pose],
    start_confs: Sequence[Sequence[float]],
    goal_poses: Sequence[Pose],
    goal_confs: Sequence[Sequence[float]],
    guide_poses: Sequence[Pose],
    guide_weights: Sequence[float],
    *,
    position_res: float = 0.01,
    rotation_res: float = 0.025,
    joint_continuity_threshold_rad: float = DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
    tube_pos_sigma_m: float = 0.05,
    tube_rot_sigma_rad: float = 0.15,
    tip_bias: float = 0.1,
    max_time: float = 60.0,
    max_iterations: int = 10 ** 6,
    random_seed: Optional[int] = None,
) -> Tuple[Optional[List[Pose]], Optional[List[np.ndarray]], Dict[str, Any]]:
    """Bridge the gaps of an approximate route with a task-space BiRRT that starts from its exact pieces.

    Args:
        ik_context (Dict[str, Any]): the core's solver context (robot,
            arm_joints, tool links, grasps, ik_options).
        joint_collision_fn (Callable): the production collision check (True = colliding).
        start_poses (Sequence[Pose]): tree A's chain: bar poses from the start onward.
        start_confs (Sequence[Sequence[float]]): its exact configurations.
        goal_poses (Sequence[Pose]): tree B's chain: bar poses from the goal backward.
        goal_confs (Sequence[Sequence[float]]): its exact configurations.
        guide_poses (Sequence[Pose]): bar poses along the whole route (the tube's centre line).
        guide_weights (Sequence[float]): how often each guide pose is picked.
        position_res (float): the extension's position step (m).
        rotation_res (float): the extension's rotation step (rad).
        joint_continuity_threshold_rad (float): the production continuity gate.
        tube_pos_sigma_m (float): spread of a sample's shift around its guide pose.
        tube_rot_sigma_rad (float): spread of a sample's turn around its guide pose.
        tip_bias (float): chance of aiming at the other tree's tip instead of the tube.
        max_time (float): give up after this many seconds.
        max_iterations (int): give up after this many iterations.
        random_seed (Optional[int]): seed of the sampler.

    Returns:
        Tuple: the bar poses and configurations of the path, start to goal
        (both None when no path was found), and ``info`` with ``success``,
        ``iterations``, ``wall_s``, ``n_ik_solves``, ``tree_sizes`` (initial
        and final), and ``extend_stop_reasons``.
    """
    started = time.time()
    rng = np.random.default_rng(random_seed)
    feature_points = get_bar_feature_points()
    weights = np.asarray(guide_weights, dtype=float)
    nodes_a, confs_a, fvecs_a = chain_tree(start_poses, start_confs, feature_points)
    nodes_b, confs_b, fvecs_b = chain_tree(goal_poses, goal_confs, feature_points)
    initial_sizes = (len(nodes_a), len(nodes_b))
    # The chain tips: the deepest exact waypoint of each tree, facing the gap.
    tip_a, tip_b = nodes_a[-1], nodes_b[-1]
    reasons: Counter = Counter()
    n_ik_solves = 0
    info: Dict[str, Any] = {"success": False}

    def extend(tree, confs, fvecs, source, target):
        """One core extension, counting its IK solves."""
        nonlocal n_ik_solves
        before = len(tree)
        last, reached, stop = extend_toward(
            nodes=tree, source=source, target_pose=target, collision_fn=_noop_pose_collision_fn,
            joint_collision_fn=joint_collision_fn, draw_color=(0, 0, 0, 0), use_draw=False,
            position_res=position_res, rotation_res=rotation_res, dist_metric="feature",
            feature_points=feature_points, feature_vecs=fvecs, enable_ik=True, node_confs=confs,
            ik_context=ik_context, joint_continuity_threshold_rad=joint_continuity_threshold_rad,
        )
        n_ik_solves += (len(tree) - before) + (0 if reached else 1)
        return last, reached, stop

    def resolve_forward(seed_conf, poses):
        """Re-solve poses in order from a seed, with the gate and collision (the core's stitch)."""
        nonlocal n_ik_solves
        out, current = [], np.asarray(seed_conf, dtype=float)
        for pose in poses:
            n_ik_solves += 1
            nxt = solve_dual_arm_pose_ik(
                robot=ik_context["robot"], arm_joints=ik_context["arm_joints"],
                tool_link_left=ik_context["tool_link_left"], tool_link_right=ik_context["tool_link_right"],
                bar_pose=pose, grasp_bar_from_left=ik_context["grasp_bar_from_left"],
                grasp_bar_from_right=ik_context["grasp_bar_from_right"], seed_conf=current,
                ik_options=ik_context.get("ik_options"),
            )
            if nxt is None:
                reasons["stitch_ik_failure"] += 1
                return None
            if joint_step_exceeds_threshold(nxt, current, joint_continuity_threshold_rad):
                reasons["stitch_continuity_fail"] += 1
                return None
            if joint_collision_fn(nxt):
                reasons["stitch_collision"] += 1
                return None
            out.append(np.asarray(nxt, dtype=float))
            current = nxt
        return out

    iteration = 0
    for iteration in range(1, max_iterations + 1):
        if time.time() - started >= max_time:
            break
        # The smaller tree grows first; A is always the start side.
        grow_a = len(nodes_a) <= len(nodes_b)
        tree, confs, fvecs = (nodes_a, confs_a, fvecs_a) if grow_a else (nodes_b, confs_b, fvecs_b)
        other, other_confs, other_fvecs = (nodes_b, confs_b, fvecs_b) if grow_a else (nodes_a, confs_a, fvecs_a)
        if rng.random() < tip_bias:
            target = (tip_b if grow_a else tip_a).config
        else:
            target = sample_in_tube(rng, guide_poses, weights, tube_pos_sigma_m, tube_rot_sigma_rad)

        nearest = nearest_node(tree, target, "feature", feature_points, fvecs)
        new_last, _reached, stop = extend(tree, confs, fvecs, nearest, target)
        reasons[stop] += 1
        if new_last is nearest:
            continue
        # * Connect: grow the other tree all the way to the new frontier.
        connect_nearest = nearest_node(other, new_last.config, "feature", feature_points, other_fvecs)
        connect_last, connected, connect_stop = extend(other, other_confs, other_fvecs, connect_nearest,
                                                       new_last.config)
        reasons[f"connect_{connect_stop}"] += 1
        if not connected:
            continue

        # * The trees meet at one pose. Start side: retrace in tree A; goal side:
        # * retrace in tree B, re-solved forward from the start side's seam so the
        # * whole path is one chain (the goal side keeps the goal's end pose).
        start_side = (new_last if grow_a else connect_last).retrace()
        goal_side = list(reversed((connect_last if grow_a else new_last).retrace()))[1:]
        start_confs_path = [np.asarray(confs_a[id(node)], dtype=float) for node in start_side]
        goal_poses_path = [node.config for node in goal_side]
        stitched = resolve_forward(start_confs_path[-1], goal_poses_path)
        if stitched is None:
            continue
        path_poses = [node.config for node in start_side] + goal_poses_path
        info = {"success": True}
        info.update({"iterations": iteration, "wall_s": time.time() - started, "n_ik_solves": n_ik_solves,
                     "tree_sizes": {"initial": initial_sizes, "final": (len(nodes_a), len(nodes_b))},
                     "extend_stop_reasons": dict(reasons)})
        return path_poses, start_confs_path + stitched, info

    info.update({"iterations": iteration, "wall_s": time.time() - started, "n_ik_solves": n_ik_solves,
                 "tree_sizes": {"initial": initial_sizes, "final": (len(nodes_a), len(nodes_b))},
                 "extend_stop_reasons": dict(reasons)})
    return None, None, info
