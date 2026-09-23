"""Stage 1: sample goal bar poses the robot can actually reach.

The benchmark deliberately ignores where the design wanted each bar to go. It
asks instead: given this bar held in this grasp, how well does the planner move
it to an arbitrary place around the robot? So goals are drawn at random in a
box around the robot and kept only when the robot can hold the bar there --
both arms reaching, no collision.

"Can hold it there" is decided by the same function the planner uses to solve
its own endpoints, with the same compas_fab collision check. A goal this stage
accepts is therefore a goal the planner agrees is a valid endpoint; anything it
later fails at is a search failure, not a bad problem.

Rejected samples are counted by reason, because "the robot could not reach
two thirds of the box" is itself a result worth seeing.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet_planning as pp

from husky_assembly_tamp.benchmark.scene import SceneCtx, pose_to_lists
from husky_assembly_tamp.motion_planner.api import _ssik_pair_goal_branch_with_home
from husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt import core


# * The box goals are drawn from, in metres, relative to the robot base at the
# * world origin. Wide enough to cover everything both arms can reach and a
# * good deal that they cannot, so the accept rate itself is informative.
GOAL_BOX_XY_M = 1.2
GOAL_BOX_Z_MIN_M = 0.1
GOAL_BOX_Z_MAX_M = 1.8

# Give up on a bar rather than spin forever if almost nothing is reachable.
MAX_SAMPLES_PER_GOAL = 400


def bar_seed(library_seed: int, bar_id: str) -> int:
    """Return a stable random seed for one bar.

    Derived from the bar's name rather than its position in a list, so adding
    or removing a bar never shifts another bar's goals.

    Args:
        library_seed (int): the seed for the whole library.
        bar_id (str): the bar's name.

    Returns:
        int: a 64-bit seed.
    """
    digest = hashlib.sha256(f"{library_seed}:{bar_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def uniform_quaternion(rng: np.random.Generator) -> Tuple[float, float, float, float]:
    """Draw a rotation uniformly at random from all possible orientations.

    Uses Shoemake's method. Sampling three Euler angles uniformly instead would
    bunch the results near the poles and quietly bias the benchmark toward
    orientations that are easier or harder than average.

    Args:
        rng (np.random.Generator): the random source.

    Returns:
        Tuple[float, float, float, float]: the rotation as ``(x, y, z, w)``.
    """
    u1, u2, u3 = rng.random(3)
    root1 = np.sqrt(1.0 - u1)
    root2 = np.sqrt(u1)
    return (
        float(root1 * np.sin(2.0 * np.pi * u2)),
        float(root1 * np.cos(2.0 * np.pi * u2)),
        float(root2 * np.sin(2.0 * np.pi * u3)),
        float(root2 * np.cos(2.0 * np.pi * u3)),
    )


def sample_bar_pose(rng: np.random.Generator, bar_from_midpoint: Sequence[float]):
    """Draw one random bar pose in the goal box.

    The box constrains where the bar's MIDDLE lands, not its frame origin: the
    exported bar frame sits at one end, so placing frames uniformly would push
    long bars systematically further out than short ones and make the two
    incomparable.

    Args:
        rng (np.random.Generator): the random source.
        bar_from_midpoint (Sequence[float]): the bar's midpoint in bar-local
            coordinates.

    Returns:
        tuple: the bar's ``(position, quaternion)`` world pose.
    """
    midpoint = np.array([
        rng.uniform(-GOAL_BOX_XY_M, GOAL_BOX_XY_M),
        rng.uniform(-GOAL_BOX_XY_M, GOAL_BOX_XY_M),
        rng.uniform(GOAL_BOX_Z_MIN_M, GOAL_BOX_Z_MAX_M),
    ])
    quaternion = uniform_quaternion(rng)
    # Shift the frame origin back so the MIDPOINT lands on the sampled point.
    offset = pp.multiply(
        ((0.0, 0.0, 0.0), quaternion),
        (tuple(float(v) for v in bar_from_midpoint), (0.0, 0.0, 0.0, 1.0)),
    )[0]
    origin = midpoint - np.asarray(offset, dtype=float)
    return (tuple(float(v) for v in origin), quaternion)


def evaluate_goal_pose(scene: SceneCtx, world_from_bar, seed_conf: Sequence[float],
                       collision_fn, *, all_branch_pairs: bool = True):
    """Decide whether the robot can hold the bar at one pose, and why not.

    This is the production endpoint solver, so its verdict is the planner's
    verdict. It is asked for a report and, by default, for every branch pair,
    because for a GOAL the alternatives matter: the dashboard lets you cycle
    through the other ways the robot could have grasped the same pose.

    Args:
        scene (SceneCtx): the live scene.
        world_from_bar: the candidate bar pose.
        seed_conf (Sequence[float]): configuration to stay close to, normally
            the authored home.
        collision_fn: the collision predicate to reject with.
        all_branch_pairs (bool): enumerate every branch pair, not just the
            first that works.

    Returns:
        Tuple[Optional[np.ndarray], Dict[str, Any]]: the accepted configuration
        (or None) and the endpoint-IK report.
    """
    report: Dict[str, Any] = {}
    conf = core.solve_endpoint_dual_arm_ik(
        robot=scene.robot_puid,
        arm_joints=scene.arm_joints,
        tool_link_left=scene.tool_link_left,
        tool_link_right=scene.tool_link_right,
        bar_pose=world_from_bar,
        grasp_bar_from_left=scene.grasp_bar_from_left,
        grasp_bar_from_right=scene.grasp_bar_from_right,
        seed_conf=np.asarray(seed_conf, dtype=float),
        rng=np.random.default_rng(0),  # ignored by the ssik backend
        max_attempts=1,
        collision_fn=collision_fn,
        exhaustive=all_branch_pairs,
        report=report,
    )
    return conf, report


def pick_primary_goal_conf(scene: SceneCtx, world_from_bar, fallback_conf,
                           collision_fn):
    """Choose the goal configuration the planner itself would choose.

    A reachable bar pose usually has several valid arm configurations, and
    which one is picked matters: the planner pairs the goal branch with a
    branch it can also reach the carry pose on, or start and goal end up on
    joint solutions no continuous motion connects. Delegating to the
    production chooser keeps the benchmark's goals on the same branches
    production would plan to.

    Args:
        scene (SceneCtx): the live scene.
        world_from_bar: the goal bar pose.
        fallback_conf: the configuration to keep if pairing cannot improve it.
        collision_fn: the collision predicate.

    Returns:
        Tuple[np.ndarray, str]: the chosen configuration and where it came
        from -- ``"paired"`` when the chooser changed it, ``"fallback"`` otherwise.
    """
    paired = _ssik_pair_goal_branch_with_home(
        planner=scene.planner,
        robot_puid=scene.robot_puid,
        arm_joints=scene.arm_joints,
        tool_link_left=scene.tool_link_left,
        tool_link_right=scene.tool_link_right,
        world_from_bar_goal=world_from_bar,
        world_from_mobile_base=None,
        grasp_bar_from_left=scene.grasp_bar_from_left,
        grasp_bar_from_right=scene.grasp_bar_from_right,
        goal_conf_fallback=np.asarray(fallback_conf, dtype=float),
        cfab_collision_fn=collision_fn,
        home_anchors=None,
    )
    paired = np.asarray(paired, dtype=float)
    source = "fallback" if np.allclose(paired, fallback_conf, atol=1e-12) else "paired"
    return paired, source


def sample_goals(scene: SceneCtx, *, n_goals: int, library_seed: int,
                 all_branch_pairs: bool = True,
                 verbose: bool = True) -> Dict[str, Any]:
    """Sample ``n_goals`` reachable, collision-free goal poses for one bar.

    Args:
        scene (SceneCtx): the live scene for this bar.
        n_goals (int): how many accepted goals to collect.
        library_seed (int): the library's master seed.
        all_branch_pairs (bool): record every valid grasp of each goal, not
            just the first.
        verbose (bool): print a line per accepted goal.

    Returns:
        Dict[str, Any]: ``goals`` (one record each), ``rejections`` (counts by
        reason), ``n_sampled`` and ``wall_s``.

    Raises:
        RuntimeError: if the box yields too few reachable poses. That means the
            bar or the box is wrong, and silently returning a short list would
            hide it.
    """
    problem = scene.problem
    rng = np.random.default_rng(bar_seed(library_seed, problem.bar_id))
    collision_fn = scene.reporting_collision_fn()
    home_conf = np.asarray(problem.home_conf12, dtype=float)

    goals: List[Dict[str, Any]] = []
    rejections: Dict[str, int] = {}
    n_sampled = 0
    t0 = time.time()

    while len(goals) < n_goals:
        if n_sampled >= MAX_SAMPLES_PER_GOAL * n_goals:
            raise RuntimeError(
                f"{problem.bar_id}: only {len(goals)} of {n_goals} goals were "
                f"reachable after {n_sampled} samples. Either this bar cannot "
                "be carried around the robot at all, or the goal box is wrong."
            )
        n_sampled += 1
        world_from_bar = sample_bar_pose(rng, problem.bar_from_midpoint)

        scene.set_state(scene.template_state)
        conf, report = evaluate_goal_pose(
            scene, world_from_bar, home_conf, collision_fn,
            all_branch_pairs=all_branch_pairs,
        )
        reason = report.get("reason", "ik_unreachable_both")
        if conf is None:
            rejections[reason] = rejections.get(reason, 0) + 1
            continue

        scene.set_state(scene.template_state)
        primary_conf, primary_source = pick_primary_goal_conf(
            scene, world_from_bar, conf, collision_fn,
        )

        free_pairs = [
            {
                "pair_index": entry["pair_index"],
                "seed_distance": entry["seed_distance"],
                "conf": entry["conf"],
            }
            for entry in report.get("pairs", [])
            if entry["verdict"] == "feasible"
        ]
        goals.append({
            "goal_index": len(goals),
            "world_from_bar": pose_to_lists(world_from_bar),
            "primary_conf": [float(v) for v in primary_conf],
            "primary_source": primary_source,
            "n_branches_left": report.get("n_branches_left", 0),
            "n_branches_right": report.get("n_branches_right", 0),
            "free_branch_pairs": free_pairs,
            "n_free_branch_pairs": len(free_pairs),
        })
        if verbose:
            midpoint = np.asarray(world_from_bar[0]) + np.asarray(problem.bar_from_midpoint)
            print(f"  goal {len(goals) - 1:3d}: {len(free_pairs):2d} free grasp(s), "
                  f"reach {np.linalg.norm(midpoint[:2]):.2f} m, "
                  f"primary from {primary_source}")

    return {
        "goals": goals,
        "rejections": rejections,
        "n_sampled": n_sampled,
        "wall_s": time.time() - t0,
    }
