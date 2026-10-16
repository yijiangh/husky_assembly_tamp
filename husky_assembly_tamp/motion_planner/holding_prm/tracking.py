"""The exact tracker: turn one roadmap edge into waypoints that hold the bar exactly.

A roadmap edge is a straight line in joint space. Its two end nodes hold the
bar, but the points in between drift off the holding surface by up to the
edge's score (a few millimetres). The tracker walks the bar along the poses the
edge implies and solves both arms' inverse kinematics at every step, which
"projects" the edge back onto the holding surface:

    1. Knots. The joint line is cut every ``knot_step_deg`` of largest joint
       move (the two end nodes are always knots). At each knot the two flanges
       place the bar at two slightly different poses; the knot's bar pose is
       halfway between them (``holding.target_bar_pose``, lead "mid").
    2. Waypoints. Between two neighbouring knots the bar pose is interpolated
       on a straight line, cut into pieces no longer than ``position_res_m``
       and ``rotation_res_rad`` -- the same rule and the same two numbers as
       the task-space BiRRT's extension step.
    3. A chain of solves. The first waypoint is seeded from the edge's start
       node (or from a given configuration, when the edge continues a chain
       that arrived from somewhere else). Every later waypoint is seeded from
       the waypoint solved just before it, as the BiRRT extension does: each
       arm's branches are moved onto that waypoint's 2*pi turn and only those
       within the production continuity gate (10 degrees on any joint) of it
       are kept. When more than one is kept -- near a straight elbow or a
       straight wrist two branches come close together -- the one nearest the
       roadmap's joint line at that point is taken, so the chain does not hop
       onto the other branch there. (Measured on B18, 2026-10-01: with the
       nearest-to-previous rule alone, 4 of 60 random edges ended on the other
       elbow branch, 25-33 degrees from their end node.)
    4. Collision, last (it is the expensive check), middle first.

! The roadmap configurations are seeds and a yardstick only, never the
! reference: the solved waypoints ARE the path, and nothing requires them to
! reproduce a stored node. What is required is that neighbouring waypoints
! are close in joint space (also across a node, where one edge's chain hands
! over to the next: see track_edge_landing) and that every waypoint holds the
! bar and is free.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from husky_assembly_tamp.motion_planner.holding_prm.holding import (
    ARMS,
    BAR_LEADS,
    STEP_MAX_RAD,
    HoldingModel,
    bar_from_arm,
    grasp_midpoint_offset,
    holding_violation,
    interpolate_confs,
    interpolate_pose,
    invert_transform,
    linf,
    pose_from_matrix,
    pose_gap,
    target_bar_pose,
    tool_target_from_bar,
)
from husky_assembly_tamp.motion_planner.holding_prm.roadmap import bisection_order
from husky_assembly_tamp.motion_planner.holding_prm.sampling import WorkerTools, arm_branches, check_holding, count
from husky_assembly_tamp.motion_planner.ssik_ik import ARM_SLICE


# -----------------------------------------------------------------------------
# * Settings
# -----------------------------------------------------------------------------


@dataclass
class TrackSettings:
    """How densely and how strictly an edge is tracked.

    Attributes:
        knot_step_deg (float): spacing of the knots along the joint line
            (largest joint move). 0 = the two end nodes only, i.e. a straight
            bar move between them (a BiRRT extension seeded by the roadmap).
        position_res_m (float): longest bar move between waypoints (metres,
            measured at the grasp midpoint).
        rotation_res_rad (float): largest bar turn between waypoints.
        lead (str): which bar pose a knot aims for (see ``holding.BAR_LEADS``).
        step_max_rad (float): the continuity gate between neighbouring waypoints.
        check_collision (bool): run the full-cell collision check on the waypoints.
        prefer_guess (bool): among the branches within the gate of the previous
            waypoint, take the one nearest the roadmap's joint line (True) or
            the one nearest the previous waypoint (False, the BiRRT's rule).
        substep_check_deg (float): a step that moves any joint more than this
            is walked again at the fine resolution below (0 = never). Near a
            straight elbow or wrist the two branches can be closer than the
            10-degree gate, and a coarse step can hop from one to the other
            without passing the gate; walked finely, the arm stays on its
            branch. Measured on B18 (2026-10-01): one 9.6-degree hop at a
            4.8-degree elbow made the production check land 9.4 degrees off;
            with a 2-degree threshold, hops of 4 degrees at 2-degree elbows
            remained, so nearly every step is walked finely now.
        fine_position_res_m (float): the fine walk's position step (the
            benchmark's fine pass, which the production check uses).
        fine_rotation_res_rad (float): the fine walk's rotation step.
    """

    knot_step_deg: float = 2.5
    position_res_m: float = 0.01
    rotation_res_rad: float = 0.025
    lead: str = "mid"
    step_max_rad: float = STEP_MAX_RAD
    check_collision: bool = True
    prefer_guess: bool = True
    substep_check_deg: float = 0.5
    fine_position_res_m: float = 0.002
    fine_rotation_res_rad: float = 0.002

    def __post_init__(self) -> None:
        if self.lead not in BAR_LEADS:
            raise ValueError(f"Unknown bar lead {self.lead!r}; expected one of {BAR_LEADS}.")
        if self.position_res_m <= 0 or self.rotation_res_rad <= 0:
            raise ValueError("The tracker's position and rotation steps must be positive.")

    def to_dict(self) -> Dict[str, Any]:
        """The settings as plain values (for worker processes and for disk)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackSettings":
        """Rebuild settings saved with :meth:`to_dict`."""
        return cls(**data)


# -----------------------------------------------------------------------------
# * Where the bar goes along an edge
# -----------------------------------------------------------------------------


def edge_waypoints(model: HoldingModel, qa: np.ndarray, qb: np.ndarray,
                   settings: TrackSettings) -> Tuple[List[np.ndarray], np.ndarray, int]:
    """The bar poses to walk along the edge qa -> qb, and the roadmap's guess at each.

    Args:
        model (HoldingModel): the holding model.
        qa (np.ndarray): start node configuration.
        qb (np.ndarray): end node configuration.
        settings (TrackSettings): knot spacing, waypoint steps, lead.

    Returns:
        Tuple[List[np.ndarray], np.ndarray, int]: the bar poses (4x4 each, the
        first at the start node and the last at the end node), the joint-line
        configuration at the same place for each (shape (W, 12); only a
        yardstick), and the number of knots.
    """
    if settings.knot_step_deg > 0:
        knots = interpolate_confs(qa, qb, float(np.radians(settings.knot_step_deg)), include_ends=True)
    else:
        knots = np.stack([qa, qb])
    # * Knot poses are handled about the grasp midpoint (see holding.grasp_midpoint_offset),
    # * so interpolation between them turns the bar about its middle, not about one end.
    offset = grasp_midpoint_offset(model)
    offset_inv = invert_transform(offset)
    centres = [target_bar_pose(model, knot, settings.lead) @ offset for knot in knots]

    poses = [centres[0] @ offset_inv]
    guesses = [knots[0]]
    for k in range(len(knots) - 1):
        distance, angle = pose_gap(centres[k], centres[k + 1])
        # The BiRRT's rule: enough pieces that neither step is exceeded.
        pieces = max(1, int(ceil(max(distance / settings.position_res_m,
                                     angle / settings.rotation_res_rad) - 1e-12)))
        for piece in range(1, pieces + 1):
            fraction = piece / pieces
            poses.append(interpolate_pose(centres[k], centres[k + 1], fraction) @ offset_inv)
            guesses.append(knots[k] + fraction * (knots[k + 1] - knots[k]))
    return poses, np.asarray(guesses), len(knots)


# -----------------------------------------------------------------------------
# * Walking one step finely
# -----------------------------------------------------------------------------


def fine_walk(model: HoldingModel, from_pose: np.ndarray, to_pose: np.ndarray, from_conf: np.ndarray,
              settings: TrackSettings, counters: Dict[str, float]) -> Tuple[Optional[np.ndarray], str]:
    """Walk the bar from one pose to the next in fine steps, each arm on the branch nearest its last solve.

    This is what the production repair does between two waypoints (bar frame
    interpolated at its fine step, warm IK nearest the previous solution, the
    continuity gate at every step), so a step that passes here is a step the
    production check will follow the same way.

    Args:
        model (HoldingModel): the holding model.
        from_pose (np.ndarray): the bar's 4x4 pose at the previous waypoint.
        to_pose (np.ndarray): the bar's 4x4 pose at the next waypoint.
        from_conf (np.ndarray): the configuration at the previous waypoint.
        settings (TrackSettings): fine steps and the gate.
        counters (Dict[str, float]): tally.

    Returns:
        Tuple[Optional[np.ndarray], str]: the configuration reached at
        ``to_pose`` (None on failure) and "ok", "ik_failure" or "continuity".
    """
    distance, angle = pose_gap(from_pose, to_pose)
    pieces = max(1, int(ceil(max(distance / settings.fine_position_res_m,
                                 angle / settings.fine_rotation_res_rad) - 1e-12)))
    current = np.asarray(from_conf, dtype=float)
    count(counters, "fine_walks")
    for piece in range(1, pieces + 1):
        pose = interpolate_pose(from_pose, to_pose, piece / pieces)
        conf = np.empty(12)
        for arm in ARMS:
            part = ARM_SLICE[arm]
            branches = arm_branches(model, arm, tool_target_from_bar(model, arm, pose), current[part],
                                    allow_rescue=False, counters=counters)
            if not branches:
                return None, "ik_failure"
            conf[part] = branches[0]
        if linf(conf, current) > settings.step_max_rad:
            return None, "continuity"
        current = conf
    return current, "ok"


def fine_segment(tools: WorkerTools, conf_a: np.ndarray, conf_b: np.ndarray, position_res_m: float,
                 rotation_res_rad: float, counters: Dict[str, float]) -> Dict[str, Any]:
    """Fill in the path between two neighbouring exact waypoints at a fine step, and check it.

    The planner's waypoints are about 1 cm apart; a path handed to the robot
    is wanted at the production's step (2 mm by default). This walks the bar
    from one waypoint to the next at that step -- the bar frame interpolated,
    both arms solved nearest the previous solution, the continuity gate at
    every step, exactly what the production repair does -- and runs the full
    collision check on every configuration in between.

    Args:
        tools (WorkerTools): model and (full) collision check of the real scene.
        conf_a (np.ndarray): the waypoint to start from (holds the bar exactly).
        conf_b (np.ndarray): the next waypoint.
        position_res_m (float): longest bar move per fine step.
        rotation_res_rad (float): largest bar turn per fine step.
        counters (Dict[str, float]): tally.

    Returns:
        Dict[str, Any]: ``free`` (the walk arrives on ``conf_b`` and nothing
        in between collides), ``reason`` ("ok", "ik_failure", "continuity",
        "branch_change" when it arrives on another solution, "collision"),
        and ``confs`` (the configurations after ``conf_a`` up to and including
        ``conf_b``; empty when not free).
    """
    model = tools.model
    from_pose = bar_from_arm(model, "left", conf_a[:6])
    to_pose = bar_from_arm(model, "left", conf_b[:6])
    distance, angle = pose_gap(from_pose, to_pose)
    pieces = max(1, int(ceil(max(distance / position_res_m, angle / rotation_res_rad) - 1e-12)))
    current = np.asarray(conf_a, dtype=float)
    confs: List[np.ndarray] = []
    for piece in range(1, pieces + 1):
        pose = interpolate_pose(from_pose, to_pose, piece / pieces)
        conf = np.empty(12)
        for arm in ARMS:
            part = ARM_SLICE[arm]
            branches = arm_branches(model, arm, tool_target_from_bar(model, arm, pose), current[part],
                                    allow_rescue=False, counters=counters)
            if not branches:
                return {"free": False, "reason": "ik_failure", "confs": []}
            conf[part] = branches[0]
        if linf(conf, current) > STEP_MAX_RAD:
            return {"free": False, "reason": "continuity", "confs": []}
        confs.append(conf)
        current = conf
    if linf(current, conf_b) > SAME_SOLUTION_RAD:
        return {"free": False, "reason": "branch_change", "confs": []}
    # The last one is the waypoint itself (already checked); the ones before it are new.
    for conf in confs[:-1]:
        if tools.timed_collides(conf, counters):
            return {"free": False, "reason": "collision", "confs": []}
    confs[-1] = np.asarray(conf_b, dtype=float)
    return {"free": True, "reason": "ok", "confs": [conf.tolist() for conf in confs]}


# -----------------------------------------------------------------------------
# * Tracking one edge
# -----------------------------------------------------------------------------


def track_edge(tools: WorkerTools, qa: np.ndarray, qb: np.ndarray, settings: TrackSettings,
               seed: Optional[np.ndarray] = None,
               counters: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Walk the bar along one edge and solve both arms exactly at every waypoint.

    Args:
        tools (WorkerTools): model and collision check.
        qa (np.ndarray): start node configuration (seeds the first solve).
        qb (np.ndarray): end node configuration.
        settings (TrackSettings): how densely and strictly to track.
        seed (Optional[np.ndarray]): seed the first solve from this instead of
            ``qa`` -- used when the edge continues a chain whose last solved
            waypoint is not exactly ``qa``. The continuity gate then also
            applies between this seed and the first waypoint.
        counters (Optional[Dict[str, float]]): tally to add the work to.

    Returns:
        Dict[str, Any]: ``free`` (True when every waypoint was solved, passed
        the gate and is collision-free; named like an edge record so the lazy
        search can read it), ``reason`` ("ok", "ik_failure", "continuity" or
        "collision"), ``fail_index`` / ``fail_fraction`` (where it stopped),
        ``n_waypoints``, ``n_knots``, ``max_step_rad`` (largest joint move
        between neighbouring solved waypoints, including from the seed),
        ``max_dev_rad`` (largest difference between a solved waypoint and the
        joint line at the same place: how far the correction moved the
        joints), ``max_viol_pos_m`` (holding error of the solved waypoints, a
        sanity number at the 1e-7 level), ``seconds``, and ``confs`` (the
        solved waypoints, start to end; empty when the edge failed).
    """
    started = perf_counter()
    counters = {} if counters is None else counters
    model = tools.model
    qa = np.asarray(qa, dtype=float)
    qb = np.asarray(qb, dtype=float)
    poses, guesses, n_knots = edge_waypoints(model, qa, qb, settings)
    count(counters, "edges_tracked")
    count(counters, "waypoints_planned", len(poses))

    previous = np.asarray(seed if seed is not None else qa, dtype=float)
    # The bar pose the previous configuration holds (the seed holds the bar exactly).
    previous_pose = bar_from_arm(model, "left", previous[:6])
    substep_check_rad = float(np.radians(settings.substep_check_deg))
    solved: List[np.ndarray] = []
    reason, fail_index = "ok", -1
    max_step, max_dev, max_viol = 0.0, 0.0, 0.0

    # * The chain: each waypoint is seeded from the one solved just before it.
    for index, (pose, guess) in enumerate(zip(poses, guesses)):
        conf = np.empty(12)
        for arm in ARMS:
            part = ARM_SLICE[arm]
            # Every branch, moved onto the previous waypoint's 2*pi turn, nearest first.
            branches = arm_branches(model, arm, tool_target_from_bar(model, arm, pose), previous[part],
                                    allow_rescue=False, counters=counters)
            if not branches:
                reason = "ik_failure"
                break
            within = [q for q in branches if np.max(np.abs(q - previous[part])) <= settings.step_max_rad]
            if not within:
                # Even the nearest branch jumps more than the gate allows.
                reason = "continuity"
                break
            if settings.prefer_guess and len(within) > 1:
                # Two branches close together (near a straight elbow or wrist):
                # stay on the roadmap's side.
                count(counters, "branch_ties")
                conf[part] = min(within, key=lambda q: float(np.max(np.abs(q - guess[part]))))
            else:
                conf[part] = within[0]
        if reason != "ok":
            fail_index = index
            break
        step = linf(conf, previous)
        if substep_check_rad > 0 and step > substep_check_rad:
            # * A big step may hide a hop to the other branch near a straight
            # * elbow or wrist: walk it finely and keep where the arm really arrives.
            walked, why = fine_walk(model, previous_pose, pose, previous, settings, counters)
            if walked is None:
                reason, fail_index = why, index
                break
            if linf(walked, conf) > 1e-6:
                count(counters, "hops_corrected")
            conf = walked
            step = linf(conf, previous)
            if step > settings.step_max_rad:
                reason, fail_index = "continuity", index
                break
        check_holding(model, conf, "the exact tracker")
        max_step = max(max_step, step)
        max_dev = max(max_dev, linf(conf, guess))
        max_viol = max(max_viol, holding_violation(model, conf)[0])
        solved.append(conf)
        previous = conf
        previous_pose = pose

    # * Collision last, middle first. A waypoint equal to an end node is known free.
    if reason == "ok" and settings.check_collision:
        to_check = [w for w, conf in enumerate(solved)
                    if not ((w == 0 and linf(conf, qa) < 1e-6) or (w == len(solved) - 1 and linf(conf, qb) < 1e-6))]
        for position in bisection_order(len(to_check)):
            if tools.timed_collides(solved[to_check[position]], counters):
                reason, fail_index = "collision", to_check[position]
                break

    free = reason == "ok"
    count(counters, f"track_{reason}")
    return {
        "free": free, "reason": reason, "fail_index": int(fail_index),
        "fail_fraction": float(fail_index / max(1, len(poses) - 1)) if fail_index >= 0 else -1.0,
        "n_waypoints": len(poses), "n_knots": int(n_knots),
        "max_step_rad": float(max_step), "max_dev_rad": float(max_dev), "max_viol_pos_m": float(max_viol),
        "seconds": perf_counter() - started,
        "confs": [conf.tolist() for conf in solved] if free else [],
    }


# * Two configurations at the same bar pose are "the same solution" within this
# * (radians, largest joint difference). Solves on one branch agree to about
# * 1e-9; a different branch is degrees away -- but near a straight elbow it can
# * be closer than the 10-degree gate, which is why the gate is not the test here.
SAME_SOLUTION_RAD = 1e-3


def track_edge_landing(tools: WorkerTools, qa: np.ndarray, qb: np.ndarray, settings: TrackSettings,
                       seed: Optional[np.ndarray] = None,
                       counters: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Track an edge in both directions, so that a route can use it either way as one chain.

    A route is one chain of waypoints, so where two edges meet, the next
    edge's chain must start where the arriving one ended. The next edge was
    tracked from the shared node, so the arriving chain must end on the
    node's own IK solution there (the same solution, ``SAME_SOLUTION_RAD``;
    both are solves at the same bar pose). Arriving "within the gate" is not
    enough: near a straight elbow the mirror branch can be closer than 10
    degrees, and jumping between two solutions at one pose is a branch flip
    no walk makes (measured on B18: the production check then left the path
    at such a node and collided further on). Two things make this depend on
    the edge:

    * When the edge's joint line passes through a fully stretched elbow (or a
      straight wrist), the corrected path, halfway between what the two arms
      imply, stays short of the point where the two elbow branches meet, so
      the chain cannot cross over and arrives on the mirror branch, 25-35
      degrees from the node on the elbow joint (measured on B18, 2026-10-01;
      finer steps do not change it). The edge then "changes branch".
    * Near such a point, which branch a walk keeps depends on the direction
      it walks in. Waypoints walked from one end, read backwards, are not
      what a walk from the other end gives -- and the production check walks
      forward. So each direction keeps its own waypoints.

    Both directions are tracked. The edge is usable when at least one of them
    arrives at its far node; a route may then use it only in a direction that
    did (``refine.Tracker.assemble_many`` drops the edge when a route needs
    the other one).

    Args:
        tools (WorkerTools): model and collision check.
        qa (np.ndarray): one node's configuration.
        qb (np.ndarray): the other node's configuration.
        settings (TrackSettings): tracker settings.
        seed (Optional[np.ndarray]): see :func:`track_edge`. A seeded chain
            continues a route from where it is, so only that one direction
            (``qa`` to ``qb``) is tracked.
        counters (Optional[Dict[str, float]]): tally.

    Returns:
        Dict[str, Any]: the record of :func:`track_edge` for the walk from
        ``qa`` to ``qb``, with ``confs`` (that walk's waypoints, or empty
        when it failed or arrived elsewhere), ``confs_back`` (the walk from
        ``qb`` to ``qa`` in its own order, or empty), ``directions`` ("both",
        "forward", "backward" or "none"), ``free`` (usable in at least one
        direction) and ``reason`` ("ok", or why the forward walk failed:
        "branch_change" when it arrived elsewhere).
    """
    counters = {} if counters is None else counters

    def walk(start: np.ndarray, end: np.ndarray, seed_conf: Optional[np.ndarray]) -> Dict[str, Any]:
        """One direction; a walk that ends on another solution than its far node counts as a branch change."""
        result = track_edge(tools, start, end, settings, seed=seed_conf, counters=counters)
        if result["free"] and linf(result["confs"][-1], end) > SAME_SOLUTION_RAD:
            count(counters, "track_branch_change")
            result.update({"free": False, "reason": "branch_change", "confs": []})
        return result

    if seed is not None:
        # * A seeded chain continues a route from where it is: one direction
        # * only, and it may end on another solution than the far node (then
        # * the route's next edge is continued from there in the same way).
        record = track_edge(tools, qa, qb, settings, seed=seed, counters=counters)
        record["confs_back"] = []
        record["directions"] = "forward" if record["free"] else "none"
        return record
    record = walk(qa, qb, None)
    record["confs_back"] = []
    # * The other direction, walked on its own (not the forward waypoints read backwards).
    back = walk(qb, qa, None)
    record["seconds"] += back["seconds"]
    record["confs_back"] = back["confs"] if back["free"] else []
    record["directions"] = {(True, True): "both", (True, False): "forward",
                            (False, True): "backward", (False, False): "none"}[(record["free"], back["free"])]
    if not record["free"] and back["free"]:
        record.update({"free": True, "reason": "ok"})
    return record


# -----------------------------------------------------------------------------
# * Describing an exact path
# -----------------------------------------------------------------------------


def exact_path_report(model: HoldingModel, confs: Sequence[Sequence[float]],
                      store_confs: bool = True) -> Dict[str, Any]:
    """Check and describe a path of exactly-holding waypoints.

    Args:
        model (HoldingModel): the holding model.
        confs (Sequence[Sequence[float]]): the waypoints, start first.
        store_confs (bool): include the waypoints and their bar poses.

    Returns:
        Dict[str, Any]: ``n_waypoints``, ``length_rad`` (joint-space length),
        ``max_step_rad`` / ``max_step_deg`` (largest joint move between
        neighbours), ``max_viol_pos_mm`` / ``max_viol_rot_deg`` (holding error
        over all waypoints), and with ``store_confs`` the ``confs`` and
        ``bar_poses`` (the bar as the LEFT flange places it, the convention of
        ``SceneCtx.bar_pose_at``, as ``[[x, y, z], [qx, qy, qz, qw]]``).
    """
    path = np.asarray(confs, dtype=float).reshape(-1, 12)
    steps = np.max(np.abs(np.diff(path, axis=0)), axis=1) if len(path) > 1 else np.zeros(1)
    violations = [holding_violation(model, conf) for conf in path]
    report: Dict[str, Any] = {
        "n_waypoints": int(len(path)),
        "length_rad": float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0,
        "max_step_rad": float(steps.max()),
        "max_step_deg": float(np.degrees(steps.max())),
        "max_viol_pos_mm": float(max(v[0] for v in violations) * 1000.0) if violations else 0.0,
        "max_viol_rot_deg": float(np.degrees(max(v[1] for v in violations))) if violations else 0.0,
    }
    if store_confs:
        report["confs"] = np.round(path, 7).tolist()
        report["bar_poses"] = [pose_from_matrix(bar_from_arm(model, "left", conf[:6])) for conf in path]
    return report
