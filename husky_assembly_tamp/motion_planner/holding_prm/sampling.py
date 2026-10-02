"""Roadmap nodes: configurations that hold the bar exactly.

A random 12-joint vector essentially never holds the bar, so nodes are made
the way the meeting suggested: draw six joint values for ONE arm (the
"driver"), use forward kinematics to find where that puts the bar, then solve
the OTHER arm's inverse kinematics at the flange pose the bar asks for. Every
analytical branch of that second arm gives one candidate node, and each
candidate is kept when the whole robot is collision-free.

Where the driver arm's joints are drawn from matters a lot. Drawing over the
full UR limits (shoulder pan is plus or minus two turns) mostly produces
wound-up arms nobody would use, so draws are centred on "anchors": the home
configuration, the goal's branch pairs and a spread of start configurations.
Four ways of drawing are mixed, and each node remembers which one made it:

    anchor_box    uniform in a box of +-margin around one anchor
    anchor_gauss  a Gaussian around one anchor
    anchor_line   a point on the straight line between two anchors (a start
                  and a goal when both exist), plus a little Gaussian noise
    global_box    uniform in the box spanned by all anchors (off by default:
                  for real problems that box is nearly the whole joint range)
    path_tube     a small Gaussian around a point of a given path (the
                  refinement step's draws; never part of a normal build)

This module also solves the endpoints of a query: every collision-free branch
pair at a given bar pose (:func:`expand_bar_pose`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from compas_fab.backends import CollisionCheckError

from husky_assembly_tamp.benchmark.scene import state_with_conf12
from husky_assembly_tamp.keyframe.ssik_inprocess import solve
from husky_assembly_tamp.motion_planner.holding_prm.holding import (
    ARMS,
    OTHER_ARM,
    HoldingModel,
    bar_from_arm,
    geometry_to_lists,
    holding_violation,
    invert_transform,
    node_geometry,
    rebranch12,
    tool_target_from_bar,
)
from husky_assembly_tamp.motion_planner.ssik_ik import ARM_SLICE, rebranch_toward_seed


# * A node that came out of inverse kinematics must hold the bar to this
# * accuracy, the same tolerance as the production FK gate
# * (core.validate_dual_arm_bar_pose uses 1e-4 m and 1e-4 rad).
HOLDING_CHECK_POS_M = 1e-4
HOLDING_CHECK_ROT_RAD = 1e-4

# The ways of drawing the driver arm's joints.
# ! New modes go at the END: sample_chunk lists the active modes in this order,
# ! and reordering would change which nodes an old seed produces.
SAMPLING_MODES = ("anchor_box", "anchor_gauss", "anchor_line", "global_box", "path_tube")


# -----------------------------------------------------------------------------
# * Collision checking and counting
# -----------------------------------------------------------------------------


def make_collision_fn(scene) -> Callable[[Sequence[float]], bool]:
    """The production full-cell collision check, as ``conf12 -> True if colliding``.

    This is the same check as ``api._build_cfab_collision_fn`` (the one the
    planner and the benchmark use): copy the template state, write the twelve
    joints into it, and let compas_fab check self-collision, tools, the
    attached bar and the environment with the scene's allowed-collision lists.

    ! It is written out here instead of calling ``SceneCtx.collision_fn()``
    ! because that function imports ``dual_arm_task_space_rrt.core`` (for its
    ! explain helper), and importing core empties the planner's log file.

    Args:
        scene: a ``SceneCtx`` from ``benchmark.scene.open_scene``.

    Returns:
        Callable[[Sequence[float]], bool]: the check.
    """
    planner = scene.planner
    template = scene.template_state
    joint_names_12 = scene.problem.joint_names_12
    options = {"verbose": False}

    def collides(conf12: Sequence[float]) -> bool:
        """Return True when this configuration touches something it must not."""
        state = state_with_conf12(template, conf12, joint_names_12)
        try:
            planner.check_collision(state, options=options)
        except CollisionCheckError:
            return True
        return False

    return collides


def new_counters() -> Dict[str, float]:
    """An empty tally of the work done: calls, outcomes and seconds spent."""
    return {}


def count(counters: Dict[str, float], key: str, amount: float = 1) -> None:
    """Add to one entry of a tally, creating it when missing."""
    counters[key] = counters.get(key, 0) + amount


def add_counters(total: Dict[str, float], part: Dict[str, float]) -> Dict[str, float]:
    """Add one tally into another, in place, and return the total."""
    for key, value in part.items():
        total[key] = total.get(key, 0) + value
    return total


@dataclass
class WorkerTools:
    """What a worker needs to make and check nodes and edges.

    Attributes:
        model (HoldingModel): the holding model.
        collides (Callable): the full-cell collision check.
    """

    model: HoldingModel
    collides: Callable[[Sequence[float]], bool]

    def timed_collides(self, conf12: Sequence[float], counters: Dict[str, float]) -> bool:
        """Run the collision check, counting the call and its time."""
        start = perf_counter()
        hit = self.collides(conf12)
        count(counters, "collision_checks")
        count(counters, "seconds_collision", perf_counter() - start)
        if hit:
            count(counters, "collisions")
        return hit


# -----------------------------------------------------------------------------
# * Inverse kinematics for one arm, on the cached arm base
# -----------------------------------------------------------------------------


def arm_branches(model: HoldingModel, arm: str, world_from_tool: np.ndarray,
                 seed6: Optional[Sequence[float]], allow_rescue: bool,
                 counters: Dict[str, float]) -> List[np.ndarray]:
    """Every analytical inverse-kinematics branch of one arm at a flange pose.

    The same answer as ``ssik_ik.ssik_arm_branches``, but the arm base pose
    comes from the holding model instead of being read from PyBullet on every
    call.

    Args:
        model (HoldingModel): the holding model (arm bases, limits).
        arm (str): "left" or "right".
        world_from_tool (np.ndarray): the 4x4 world pose ``tool0`` must reach.
        seed6 (Optional[Sequence[float]]): when given, every branch is moved by
            whole turns toward it and the list is sorted nearest-first (raw
            largest-joint distance), as the production adapter does.
        allow_rescue (bool): let ssik try its slow numeric rescue when the
            exact solver finds nothing. Off for sampling (fail fast), on for
            the one-off endpoint solves.
        counters (Dict[str, float]): tally to add the call and its time to.

    Returns:
        List[np.ndarray]: the branches (six joint values each); empty when the
        pose is out of reach or every branch breaks a joint limit.
    """
    start = perf_counter()
    target_in_base = invert_transform(model.world_from_base[arm]) @ world_from_tool
    branches = [q for q, _residual in solve(arm, target_in_base, allow_rescue=allow_rescue)]
    if seed6 is not None:
        seed = np.asarray(seed6, dtype=float)
        branches = [rebranch_toward_seed(q, seed, model.limits[arm]) for q in branches]
        branches.sort(key=lambda q: float(np.max(np.abs(q - seed))))
    count(counters, "ik_calls")
    count(counters, "seconds_ik", perf_counter() - start)
    if branches:
        count(counters, "ik_reachable")
        count(counters, "ik_branches", len(branches))
    return branches


def check_holding(model: HoldingModel, conf12: np.ndarray, where: str) -> None:
    """Stop loudly if a configuration that should hold the bar does not.

    Every node is built so it holds the bar exactly; a node that does not
    means the arm bases, the grasps or the solver disagree with each other,
    and every number the roadmap reports afterwards would be wrong.

    Args:
        model (HoldingModel): the holding model.
        conf12 (np.ndarray): the configuration.
        where (str): what produced it, for the error message.

    Raises:
        RuntimeError: when the violation is above the FK-gate tolerance.
    """
    viol_pos, viol_rot = holding_violation(model, conf12)
    if viol_pos > HOLDING_CHECK_POS_M or viol_rot > HOLDING_CHECK_ROT_RAD:
        raise RuntimeError(
            f"A configuration made by {where} does not hold the bar: position error "
            f"{viol_pos * 1000:.3f} mm, rotation error {np.degrees(viol_rot):.4f} deg "
            f"(allowed {HOLDING_CHECK_POS_M * 1000} mm / {HOLDING_CHECK_ROT_RAD} rad). "
            "The arm base poses, the grasps or the ssik artifacts disagree; check "
            "that the scene, the library and the solver come from the same export."
        )


# -----------------------------------------------------------------------------
# * The node sampler
# -----------------------------------------------------------------------------


@dataclass
class SamplerConfig:
    """How the driver arm's joints are drawn.

    Attributes:
        anchors (np.ndarray): (A, 12) configurations draws are centred on.
        anchor_roles (List[str]): "home", "goal" or "start" for each anchor.
        mode_mix (Dict[str, float]): weight of each sampling mode.
        box_margin_rad (float): half-width of the ``anchor_box`` box, per joint.
        gauss_sigma_rad (float): spread of the ``anchor_gauss`` draw, per joint.
        line_sigma_rad (float): noise added to an ``anchor_line`` point.
        max_branches (int): keep at most this many of the other arm's
            branches per draw, nearest the anchor first (0 = all of them).
        tube_points (Optional[np.ndarray]): (P, 12) points of the path the
            ``path_tube`` mode draws around; None when that mode is not used.
        tube_weights (Optional[np.ndarray]): how often each tube point is
            picked (any positive numbers; normalised when drawing).
        tube_sigma_rad (float): spread of the ``path_tube`` draw, per joint.
    """

    anchors: np.ndarray
    anchor_roles: List[str]
    mode_mix: Dict[str, float]
    box_margin_rad: float
    gauss_sigma_rad: float
    line_sigma_rad: float
    max_branches: int = 0
    tube_points: Optional[np.ndarray] = None
    tube_weights: Optional[np.ndarray] = None
    tube_sigma_rad: float = 0.0
    _global_box: Optional[Tuple[np.ndarray, np.ndarray]] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return the settings as plain lists, for the worker processes and for disk."""
        data = {
            "anchors": np.asarray(self.anchors).tolist(),
            "anchor_roles": list(self.anchor_roles),
            "mode_mix": dict(self.mode_mix),
            "box_margin_rad": float(self.box_margin_rad),
            "gauss_sigma_rad": float(self.gauss_sigma_rad),
            "line_sigma_rad": float(self.line_sigma_rad),
            "max_branches": int(self.max_branches),
        }
        if self.tube_points is not None:
            data["tube_points"] = np.asarray(self.tube_points).tolist()
            data["tube_weights"] = np.asarray(self.tube_weights).tolist()
            data["tube_sigma_rad"] = float(self.tube_sigma_rad)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SamplerConfig":
        """Rebuild settings saved with :meth:`to_dict`."""
        has_tube = data.get("tube_points") is not None
        return cls(
            anchors=np.asarray(data["anchors"], dtype=float),
            anchor_roles=list(data["anchor_roles"]),
            mode_mix=dict(data["mode_mix"]),
            box_margin_rad=float(data["box_margin_rad"]),
            gauss_sigma_rad=float(data["gauss_sigma_rad"]),
            line_sigma_rad=float(data["line_sigma_rad"]),
            max_branches=int(data.get("max_branches", 0)),
            tube_points=np.asarray(data["tube_points"], dtype=float) if has_tube else None,
            tube_weights=np.asarray(data["tube_weights"], dtype=float) if has_tube else None,
            tube_sigma_rad=float(data.get("tube_sigma_rad", 0.0)),
        )

    def global_box(self, model: HoldingModel) -> Tuple[np.ndarray, np.ndarray]:
        """The box spanned by all anchors plus the margin, clipped to the limits."""
        if self._global_box is None:
            limits = np.vstack([model.limits["left"], model.limits["right"]])
            low = np.maximum(self.anchors.min(axis=0) - self.box_margin_rad, limits[:, 0])
            high = np.minimum(self.anchors.max(axis=0) + self.box_margin_rad, limits[:, 1])
            self._global_box = (low, high)
        return self._global_box


def draw_driver(rng: np.random.Generator, config: SamplerConfig, model: HoldingModel,
                mode: str, driver: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Draw the driver arm's six joints, and the configuration to seed the other arm from.

    Args:
        rng (np.random.Generator): the random source.
        config (SamplerConfig): the sampling settings.
        model (HoldingModel): supplies the joint limits.
        mode (str): one of ``SAMPLING_MODES``.
        driver (str): the arm whose joints are drawn.

    Returns:
        Tuple[np.ndarray, np.ndarray, int]: the six driver joints, a full
        12-joint "centre" whose other-arm half seeds the inverse kinematics
        (so the other arm's branches land on the same 2*pi turn as the
        anchors), and the index of the anchor the draw was centred on.
    """
    part = ARM_SLICE[driver]
    low, high = model.limits[driver][:, 0], model.limits[driver][:, 1]

    # * The tube draw is handled first and on its own, so the anchor modes below
    # * consume the random source exactly as before (old seeds give old nodes).
    if mode == "path_tube":
        if config.tube_points is None or len(config.tube_points) == 0:
            raise ValueError("The path_tube mode needs tube_points (the path to draw around).")
        weights = np.asarray(config.tube_weights, dtype=float)
        point = int(rng.choice(len(config.tube_points), p=weights / weights.sum()))
        centre = np.asarray(config.tube_points[point], dtype=float).copy()
        q = centre[part] + rng.normal(0.0, config.tube_sigma_rad, 6)
        # -1: this draw was centred on a path point, not on an anchor.
        return np.clip(q, low, high), centre, -1

    anchor_index = int(rng.integers(len(config.anchors)))
    centre = config.anchors[anchor_index].copy()

    if mode == "anchor_box":
        q = rng.uniform(centre[part] - config.box_margin_rad, centre[part] + config.box_margin_rad)
    elif mode == "anchor_gauss":
        q = centre[part] + rng.normal(0.0, config.gauss_sigma_rad, 6)
    elif mode == "anchor_line":
        # Pair the anchor with one of a different role when there is one, so
        # most lines run between a start and a goal: the straight joint-space
        # line between two endpoints is the most natural guess for a path.
        roles = config.anchor_roles
        others = [i for i, role in enumerate(roles) if role != roles[anchor_index]]
        partner = int(rng.choice(others)) if others else int(rng.integers(len(config.anchors)))
        fraction = rng.uniform(0.0, 1.0)
        centre = centre + fraction * (config.anchors[partner] - centre)
        q = centre[part] + rng.normal(0.0, config.line_sigma_rad, 6)
    elif mode == "global_box":
        box_low, box_high = config.global_box(model)
        q = rng.uniform(box_low[part], box_high[part])
        centre[part] = q
    else:
        raise ValueError(f"Unknown sampling mode {mode!r}; expected one of {SAMPLING_MODES}.")
    return np.clip(q, low, high), centre, anchor_index


def lift_draw(tools: WorkerTools, config: SamplerConfig, driver: str, q_driver: np.ndarray,
              centre: np.ndarray, counters: Dict[str, float]) -> List[np.ndarray]:
    """Turn one driver-arm draw into the collision-free nodes it yields.

    Args:
        tools (WorkerTools): model and collision check.
        config (SamplerConfig): the sampling settings.
        driver (str): the arm whose joints were drawn.
        q_driver (np.ndarray): its six joints.
        centre (np.ndarray): the 12-joint centre of the draw; its other-arm
            half is the seed the other arm's branches are turned toward.
        counters (Dict[str, float]): tally.

    Returns:
        List[np.ndarray]: 12-joint configurations, each holding the bar
        exactly and collision-free.
    """
    model = tools.model
    other = OTHER_ARM[driver]
    # * Where the driver arm puts the bar, and where that asks the other flange to be.
    world_from_bar = bar_from_arm(model, driver, q_driver)
    target = tool_target_from_bar(model, other, world_from_bar)
    branches = arm_branches(model, other, target, centre[ARM_SLICE[other]],
                            allow_rescue=False, counters=counters)
    if config.max_branches > 0:
        branches = branches[: config.max_branches]

    kept = []
    for q_other in branches:
        conf12 = np.empty(12)
        conf12[ARM_SLICE[driver]] = q_driver
        conf12[ARM_SLICE[other]] = q_other
        check_holding(model, conf12, "the node sampler")
        count(counters, "candidates")
        if not tools.timed_collides(conf12, counters):
            kept.append(conf12)
    return kept


def sample_chunk(tools: WorkerTools, config: SamplerConfig, seed: int, chunk_index: int,
                 n_draws: int) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Make the nodes of one chunk of draws.

    A chunk's random source is seeded from ``(seed, chunk_index)`` alone, so
    the same chunk gives the same nodes whichever worker runs it and in
    whatever order the chunks finish.

    Args:
        tools (WorkerTools): model and collision check.
        config (SamplerConfig): the sampling settings.
        seed (int): the build's random seed.
        chunk_index (int): which chunk this is.
        n_draws (int): how many driver draws to make.

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, float]]: the kept nodes (each
        with ``conf``, ``mode``, ``driver``, ``anchor``, ``draw`` and the
        geometry of :func:`holding.node_geometry`) and the tally.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(chunk_index)]))
    modes = [m for m in SAMPLING_MODES if config.mode_mix.get(m, 0.0) > 0.0]
    weights = np.array([config.mode_mix[m] for m in modes], dtype=float)
    weights /= weights.sum()
    counters = new_counters()
    nodes = []
    for draw in range(n_draws):
        # The driver alternates, so both arms drive equally often.
        driver = ARMS[(chunk_index * n_draws + draw) % 2]
        mode = modes[int(rng.choice(len(modes), p=weights))]
        q_driver, centre, anchor_index = draw_driver(rng, config, tools.model, mode, driver)
        count(counters, "draws")
        count(counters, f"draws_{mode}")
        for conf12 in lift_draw(tools, config, driver, q_driver, centre, counters):
            count(counters, "kept")
            count(counters, f"kept_{mode}")
            count(counters, f"kept_driver_{driver}")
            nodes.append({
                "conf": conf12.tolist(), "mode": mode, "driver": driver,
                "anchor": anchor_index, "draw": chunk_index * n_draws + draw,
                **geometry_to_lists(node_geometry(tools.model, conf12)),
            })
    return nodes, counters



# -----------------------------------------------------------------------------
# * Endpoints: every collision-free branch pair at one bar pose
# -----------------------------------------------------------------------------


def expand_bar_pose(tools: WorkerTools, world_from_bar: np.ndarray,
                    turn_references: Sequence[Tuple[str, Sequence[float]]],
                    counters: Dict[str, float]) -> List[Dict[str, Any]]:
    """All collision-free ways of holding the bar at one pose.

    Both arms are solved for every analytical branch at their grasp, and every
    left-right combination is tried (up to 8 x 8 = 64). This is the pilot's
    own version of ``core.solve_endpoint_dual_arm_ik(exhaustive=True)``.

    Each branch pair is one physical pose of the robot, but each joint with a
    range wider than a turn can reach that pose in two ways, a full turn apart.
    The roadmap measures raw joint differences, so which way matters. For
    every name and reference in ``turn_references`` the pair is moved onto
    the turn nearest that reference; different results become different
    nodes, and each node lists the references that produced it (for example
    ``["given", "home"]`` when both agree). The collision check is done once
    per pair, since the twins are the same physical pose.

    Args:
        tools (WorkerTools): model and collision check.
        world_from_bar (np.ndarray): the bar's 4x4 world pose.
        turn_references (Sequence[Tuple[str, Sequence[float]]]): named
            12-joint references, e.g. ``[("given", library_conf), ("home", home)]``.
        counters (Dict[str, float]): tally.

    Returns:
        List[Dict[str, Any]]: one entry per distinct configuration, with
        ``conf``, ``pair`` (index of the branch pair), ``turns`` (reference
        names) and ``dist_given`` (largest joint difference to the first
        reference, 0 for the reference itself).
    """
    model = tools.model
    branches = {
        arm: arm_branches(model, arm, tool_target_from_bar(model, arm, world_from_bar),
                          seed6=None, allow_rescue=True, counters=counters)
        for arm in ARMS
    }
    first_reference = np.asarray(turn_references[0][1], dtype=float)
    results = []
    for pair_index, (q_left, q_right) in enumerate(product(branches["left"], branches["right"])):
        raw = np.concatenate([q_left, q_right])
        count(counters, "endpoint_pairs")
        check_holding(model, raw, "the endpoint solver")
        if tools.timed_collides(raw, counters):
            continue
        count(counters, "endpoint_pairs_free")
        # * One node per distinct turn; remember every reference that led to it.
        versions: List[Tuple[np.ndarray, List[str]]] = []
        for name, reference in turn_references:
            shifted = rebranch12(raw, reference, model.limits)
            for conf, names in versions:
                if np.max(np.abs(conf - shifted)) < 1e-9:
                    names.append(name)
                    break
            else:
                versions.append((shifted, [name]))
        for conf, names in versions:
            results.append({
                "conf": conf.tolist(), "pair": pair_index, "turns": names,
                "dist_given": float(np.max(np.abs(conf - first_reference))),
            })
    return results
