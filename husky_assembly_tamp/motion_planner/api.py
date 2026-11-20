"""Motion planning entry points for husky dual-arm system.

All public functions take a compas_fab :class:`PyBulletPlanner` whose
``robot_cell`` is already loaded, plus a :class:`RobotCellState` for the
start and a 12-vector / :class:`Configuration` for the goal. Collision
checks go through ``planner.check_collision`` so obstacles, attached
tools, attached rigid bodies, and the ACM all come from the cell state.

Public functions
----------------
- plan_free_dual_arm(planner, start_state, goal_conf, ...)
- plan_constrained_dual_arm(planner, start_state, goal_conf, *, active_bar_id, ...)
- plan_constrained_dual_arm_linear(planner, start_state, goal_conf, *, active_bar_id, ...)
- plan_dual_arm_linear_independent(planner, start_state, goal_conf, ...)

The previous ros2- / teleop-coupled API is preserved as ``api_archive.py``
for legacy callers (live teleop monitor, etc.). New callers should use
this module.

Caveat on the constrained planner's pose-space pre-reject (CC.4 only):
joints in the assembly cell are attached to robot tool0 links, NOT to the
bar. When the pose-RRT samples a new bar world pose, the joints stay
anchored to the robot — their cfab collision checks reflect the joints'
current tool0-anchored positions, not their hypothetical positions had
they followed the bar. That's fine because this layer is a fast pre-reject
on the bar mesh; the full joint-space check (post-IK, via
``joint_collision_fn``) catches any joint-collision the bar-pose layer
missed.
"""
from __future__ import annotations

import logging
import time
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet_planning as pp

from compas.geometry import Frame, Transformation
from compas_fab.backends import CollisionCheckError, InverseKinematicsError
from compas_fab.backends.pybullet.exceptions import PlanningGroupNotSupported
from compas_fab.robots import (
    FrameTarget,
    JointTrajectory,
    JointTrajectoryPoint,
    RobotCellState,
    TargetMode,
)
from compas_robots.model import Joint

# * This package — shared ssik helpers (backend switch, branch enumeration).
from husky_assembly_tamp.motion_planner import ssik_ik
from husky_assembly_tamp.motion_planner.contacts import strict_carry_state
from husky_assembly_tamp.motion_planner.floor import floor_added


logger = logging.getLogger(__name__)

LEFT_GROUP = "base_left_arm_manipulator"
RIGHT_GROUP = "base_right_arm_manipulator"

_ARM_SUFFIXES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

TOOL_LINK_LEFT = "left_ur_arm_tool0"
TOOL_LINK_RIGHT = "right_ur_arm_tool0"

# * The planners plan_constrained_dual_arm can run: the task-space search, or a
# * roadmap precomputed per bar (motion_planner/holding_prm).
PLANNING_BACKENDS = ("birrt", "prm")


# ---------------------------------------------------------------------------
# Small helpers (private to this module)
# ---------------------------------------------------------------------------


def _arm_joint_names(robot_cell) -> Tuple[List[str], List[str]]:
    """Return ([6 left arm joints], [6 right arm joints]) in canonical order."""
    def _filter(group):
        return [
            n for n in robot_cell.get_configurable_joint_names(group)
            if any(n.endswith(s) for s in _ARM_SUFFIXES)
        ]
    left = _filter(LEFT_GROUP)
    right = _filter(RIGHT_GROUP)
    assert len(left) == 6 and len(right) == 6, (
        f"expected 6 arm joints per side; got L={left}, R={right}"
    )
    return left, right


def _conf12_from_state(state: RobotCellState, joint_names_12: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [float(state.robot_configuration[n]) for n in joint_names_12],
        dtype=float,
    )


def _conf12_from_target(target_configuration, joint_names_12: Sequence[str]) -> np.ndarray:
    """Extract a 12-vec from a compas_robots Configuration OR a length-12 sequence."""
    try:
        return np.asarray(
            [float(target_configuration[n]) for n in joint_names_12], dtype=float,
        )
    except (TypeError, KeyError):
        arr = np.asarray(list(target_configuration), dtype=float)
        if arr.shape != (12,):
            raise ValueError(
                f"goal_conf must be Configuration with the 12 arm joints or length-12 "
                f"sequence; got shape {arr.shape}"
            )
        return arr


def _state_with_conf12(
    template_state: RobotCellState,
    conf_12: Sequence[float],
    joint_names_12: Sequence[str],
) -> RobotCellState:
    """Clone template_state, overwrite the 12 arm joint values."""
    s = template_state.copy()
    for n, v in zip(joint_names_12, conf_12):
        s.robot_configuration[n] = float(v)
    return s


def _build_cfab_collision_fn(planner, template_state, joint_names_12):
    """Return a closure ``conf_12 -> bool`` (True == colliding).

    Uses ``planner.check_collision`` on a clone of ``template_state`` per call,
    so ACM, attached tools, and attached rigid bodies stay consistent.

    The closure also carries ``explain(conf_12)``, which reruns the check with
    the full report on and returns the colliding parts by name. It is meant to
    be called only after the plain check said "collision", so the fast path
    (no report, stops at the first contact) is unchanged.
    """
    from .dual_arm_task_space_rrt.core import collision_pair_names

    cc_opts = {"verbose": False}
    explain_opts = {"verbose": False, "full_report": True}

    def _fn(conf_12, *_args, **_kw):
        s = _state_with_conf12(template_state, conf_12, joint_names_12)
        try:
            planner.check_collision(s, options=cc_opts)
        except CollisionCheckError:
            return True
        return False

    def _explain(conf_12):
        """Name every pair of parts that collide at ``conf_12``."""
        s = _state_with_conf12(template_state, conf_12, joint_names_12)
        try:
            planner.check_collision(s, options=explain_opts)
        except CollisionCheckError as exc:
            return collision_pair_names(getattr(exc, "collision_pairs", None),
                                        planner.client.robot_cell)
        return []

    _fn.explain = _explain
    return _fn


def _collect_obstacle_puids(planner, *, exclude=()) -> List[int]:
    """Return a flat list of pybullet body ids for all rigid bodies in the
    cell except those listed in ``exclude``.

    Tools are not included — attached tools move with the robot and are
    handled by compas_fab's collision check via the robot's URDF + ACM.
    """
    obstacles: List[int] = []
    rb_puids = getattr(planner.client, "rigid_bodies_puids", {})
    for name, puids in rb_puids.items():
        if name in exclude:
            continue
        obstacles.extend(puids)
    return obstacles


def _bar_body_id(planner, active_bar_id: str) -> int:
    """Return the primary pybullet body id of the active bar."""
    puids = planner.client.rigid_bodies_puids[active_bar_id]
    if not puids:
        raise ValueError(f"rigid body {active_bar_id!r} has no pybullet bodies")
    return puids[0]


def _fk_link_pose_pp(planner, conf_12, robot_puid, arm_joints, tool_link):
    """Return pybullet (pos, quat) of tool_link with arms at conf_12.

    Wrapped in ``pp.WorldSaver`` so the live world is untouched.
    """
    with pp.WorldSaver():
        pp.set_joint_positions(robot_puid, arm_joints, conf_12)
        return pp.get_link_pose(robot_puid, tool_link)


def _fk_link_frame(planner, state, link_name) -> Frame:
    """Apply ``state`` and return the world frame of ``link_name`` as a compas Frame."""
    planner.set_robot_cell_state(state)
    client = planner.client
    link_id = client.robot_link_puids[link_name]
    pos, quat = pp.get_link_pose(client.robot_puid, link_id)
    # pybullet quat is xyzw; compas Frame.from_quaternion takes wxyz.
    return Frame.from_quaternion([quat[3], quat[0], quat[1], quat[2]], point=list(pos))


def _pp_pose_from_frame(frame: Frame):
    """Convert a compas Frame to a pybullet ``(pos, quat_xyzw)`` pair."""
    return (list(frame.point), list(frame.quaternion.xyzw))


def _ik_dual_arm_at_frames(planner, seed_state, left_frame, right_frame, joint_names_12):
    """Solve IK on both arms at ``left_frame`` / ``right_frame``.

    Uses ``seed_state`` as the IK seed. Returns a 12-vec numpy array of arm
    joint values (left then right) or raises CollisionCheckError /
    InverseKinematicsError on failure.

    With the ssik backend (the default) each arm takes its analytical branch
    nearest to the seed; no collision check either way (mirrors the gradient
    path's ``check_collision: False`` -- callers re-check / re-solve for
    collisions themselves).
    """
    if ssik_ik.ik_backend() == "ssik":
        # Apply the seed state first so pybullet holds the right base placement
        # (ssik targets are expressed in each arm's base-link frame).
        planner.set_robot_cell_state(seed_state)
        robot_puid = planner.client.robot_puid
        seed_12 = _conf12_from_state(seed_state, joint_names_12)
        conf_12 = seed_12.copy()
        frames = {"left": left_frame, "right": right_frame}
        for arm in ("left", "right"):
            branches = ssik_ik.ssik_arm_branches(
                robot_puid, arm, _pp_pose_from_frame(frames[arm]),
                seed_12[ssik_ik.ARM_SLICE[arm]], max_solutions=8,
            )
            if not branches:
                # Same exception type the compas_fab path raises, so the
                # callers' goal_ik_failed reporting works unchanged.
                raise InverseKinematicsError(
                    f"ssik: no IK branches for the {arm} arm at its goal frame"
                )
            conf_12[ssik_ik.ARM_SLICE[arm]] = branches[0]
        return conf_12

    left_target = FrameTarget(
        left_frame, target_mode=TargetMode.ROBOT,
        tolerance_position=0.001, tolerance_orientation=0.01,
    )
    right_target = FrameTarget(
        right_frame, target_mode=TargetMode.ROBOT,
        tolerance_position=0.001, tolerance_orientation=0.01,
    )
    ik_opts = {
        "max_results": 20,
        "max_descend_iterations": 200,
        "return_full_configuration": True,
        "check_collision": False,
        "verbose": False,
    }
    state = seed_state.copy()
    planner.set_robot_cell_state(state)
    conf_L = planner.inverse_kinematics(left_target, state, LEFT_GROUP, ik_opts)
    state.robot_configuration = conf_L
    conf_LR = planner.inverse_kinematics(right_target, state, RIGHT_GROUP, ik_opts)
    return np.asarray(
        [float(conf_LR[n]) for n in joint_names_12], dtype=float,
    )


def _ssik_pair_goal_branch_with_home(
    planner,
    robot_puid,
    arm_joints,
    tool_link_left,
    tool_link_right,
    world_from_bar_goal,
    world_from_mobile_base,
    grasp_bar_from_left,
    grasp_bar_from_right,
    goal_conf_fallback,
    cfab_collision_fn,
    home_anchors=None,
    report=None,
):
    """Re-pick the M1 goal conf on the IK branch pair most compatible with the home pose.

    Why this exists: with real joint limits enforced (the ssik backend), the IK
    branch that holds the bar at the APPROACH (goal) pose and the branch that
    holds it at the canonical HOME (loading) pose are not automatically the
    same -- picking them independently can land them hundreds of degrees apart
    in joint space, and then no continuous in-limit arm motion connects start
    to goal (measured on B6: independent picks were 104-230 deg apart while the
    best joint pairing is 38/67 deg). So: enumerate the full legal branch set
    at BOTH poses per arm, rank goal branches by their distance to the nearest
    home branch, and return the first FK-valid + collision-free combination.
    The start derivation then seeds from this goal and naturally lands on the
    matching home branch.

    (The old gradient backend never faced this: PyBullet's IK ignores joint
    limits, so it happily tracked through out-of-limit space.)

    Args:
        planner: compas_fab PyBulletPlanner (used indirectly via collision fn).
        robot_puid (int): PyBullet body id.
        arm_joints: PyBullet joint indices of the 12 arm joints.
        tool_link_left (int): left tool0 link index.
        tool_link_right (int): right tool0 link index.
        world_from_bar_goal: goal bar pose, PyBullet ``(pos, quat)``.
        world_from_mobile_base: robot base pose, PyBullet ``(pos, quat)`` or None.
        grasp_bar_from_left: left grasp transform (bar-from-tool).
        grasp_bar_from_right: right grasp transform (bar-from-tool).
        goal_conf_fallback (np.ndarray): the already-solved goal 12-vec to keep
            when pairing cannot improve on it (e.g. anchor pose unreachable).
        cfab_collision_fn: ``conf12 -> bool`` collision predicate (True = hit).
        home_anchors: ``HOME_BAR_ANCHORS`` carry-mode selection (None/"all" =
            every anchor) -- must match what the start derivation will sample,
            so the goal branch is ranked against homes it can actually reach.

    Returns:
        np.ndarray: the (possibly re-picked) goal 12-vec.
    """
    from .dual_arm_task_space_rrt.core import (
        home_bar_anchor_variants,
        validate_dual_arm_bar_pose,
    )
    from husky_assembly_tamp.keyframe import ssik_inprocess

    if world_from_mobile_base is None:
        world_from_mobile_base = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    mb_from_bar_goal = pp.multiply(pp.invert(world_from_mobile_base), world_from_bar_goal)
    # All candidate home anchors the start derivation may use: the selected
    # carry anchors' canonical orientations plus their rotated variants. A goal
    # branch counts as home-compatible if it is near a legal branch at ANY of
    # them.
    home_poses_world = [
        pp.multiply(world_from_mobile_base, anchor_mb)
        for _label, anchor_mb in home_bar_anchor_variants(
            mb_from_bar_goal, grasp_bar_from_left, grasp_bar_from_right,
            anchors=home_anchors,
        )
    ]

    # * Per arm: rank every legal goal branch by its distance to the NEAREST
    # * legal branch across all home anchor variants (after 2*pi re-branching
    # * toward the goal branch).
    per_arm = {}
    for arm, grasp in (("left", grasp_bar_from_left), ("right", grasp_bar_from_right)):
        goal_branches = ssik_ik.ssik_arm_branches(
            robot_puid, arm, pp.multiply(world_from_bar_goal, grasp), None
        )
        home_branches = []
        for home_pose in home_poses_world:
            home_branches.extend(
                ssik_ik.ssik_arm_branches(robot_puid, arm, pp.multiply(home_pose, grasp), None)
            )
        if not goal_branches or not home_branches:
            # No anchor variant (or the goal) is reachable for this arm -- keep
            # the seeded goal; the sweep downstream may still find something.
            return goal_conf_fallback
        limits = ssik_inprocess.joint_limits(arm)
        ranked = []
        for goal_q in goal_branches:
            nearest_home = min(
                float(np.max(np.abs(
                    ssik_ik.rebranch_toward_seed(home_q, goal_q, limits) - goal_q
                )))
                for home_q in home_branches
            )
            ranked.append((nearest_home, goal_q))
        ranked.sort(key=lambda pair: pair[0])
        per_arm[arm] = ranked

    # * Combine arms: try goal-branch pairs in order of the WORSE arm's cross
    # * distance; the first FK-valid + collision-free combination wins.
    top_k = 4  # 4x4 combos is plenty -- the best pairing is almost always early
    combos = [
        (max(d_left, d_right), np.concatenate([q_left, q_right]))
        for d_left, q_left in per_arm["left"][:top_k]
        for d_right, q_right in per_arm["right"][:top_k]
    ]
    combos.sort(key=lambda pair: pair[0])
    for cross_dist, conf in combos:
        if not validate_dual_arm_bar_pose(
            robot=robot_puid, arm_joints=arm_joints,
            tool_link_left=tool_link_left, tool_link_right=tool_link_right,
            full_conf=conf, bar_pose=world_from_bar_goal,
            grasp_bar_from_left=grasp_bar_from_left,
            grasp_bar_from_right=grasp_bar_from_right,
        ):
            continue
        if cfab_collision_fn(conf):
            continue
        print(f"[ssik] goal branch paired with home anchor "
              f"(worst-arm cross distance {np.rad2deg(cross_dist):.1f} deg)")
        if report is not None:
            report["cross_distance_deg"] = float(np.rad2deg(cross_dist))
        return np.asarray(conf, dtype=float)
    # No valid combination -- keep the seeded goal rather than failing here.
    if report is not None:
        report["cross_distance_deg"] = None
    return goal_conf_fallback


# How long the goal-branch probe may spend proving that an authored goal conf
# can walk to some home pose. Nearly always answered within a second; the cap
# only matters when the answer is "no".
GOAL_BRANCH_PROBE_S = 8.0


def _planning_context(start_state, active_bar_id, joint_names_12, goal_conf_arr,
                      world_from_bar_goal, grasp_bar_from_left, grasp_bar_from_right,
                      goal_conf_authored=None, pairing_report=None):
    """Everything a diagnostics viewer needs to rebuild one M1 planning scene.

    The frames the planner works in, the goal it aims at (as authored and as
    actually used), and the rigid grasp it preserves -- gathered once so the
    derive-only and the full-plan paths describe themselves the same way.

    Args:
        start_state: the RobotCellState the plan runs on.
        active_bar_id (str): the held bar's rigid-body name.
        joint_names_12 (Sequence[str]): the twelve arm joint names, in order.
        goal_conf_arr: the goal 12-vec the planner uses.
        world_from_bar_goal: the goal bar pose, pybullet ``(pos, quat)``.
        grasp_bar_from_left: left grasp transform (bar-from-tool0).
        grasp_bar_from_right: right grasp transform (bar-from-tool0).
        goal_conf_authored: the goal 12-vec the caller asked for, when the
            branch pairing may have moved it; None means "the same".
        pairing_report (dict | None): what the branch pairing / probe recorded.

    Returns:
        dict: JSON-able context (poses as pybullet pairs).
    """
    identity = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    bar_state = (start_state.rigid_body_states or {}).get(active_bar_id)
    attachment_frame = getattr(bar_state, "attachment_frame", None)
    base_frame = getattr(start_state, "robot_base_frame", None)
    used = np.asarray(goal_conf_arr, dtype=float)
    authored = used if goal_conf_authored is None else np.asarray(goal_conf_authored, dtype=float)
    report = pairing_report or {}
    return {
        "joint_names_12": list(joint_names_12),
        "world_from_mobile_base": (_pp_pose_from_frame(base_frame)
                                   if base_frame is not None else identity),
        "goal_conf_authored": [float(v) for v in authored],
        "goal_conf": [float(v) for v in used],
        "goal_rebranch_max_rad": float(np.max(np.abs(used - authored))),
        "world_from_bar_goal": world_from_bar_goal,
        "pairing_cross_distance_deg": report.get("cross_distance_deg"),
        "goal_branch_probe": report.get("probe"),
        "grasp_bar_from_left": grasp_bar_from_left,
        "grasp_bar_from_right": grasp_bar_from_right,
        "attach_link": getattr(bar_state, "attached_to_link", None),
        "tool0_from_bar": (_pp_pose_from_frame(attachment_frame)
                           if attachment_frame is not None else identity),
    }


def _derive_constrained_start_for_plan(
    planner,
    start_state,
    *,
    active_bar_id,
    bar_body,
    obstacles,
    robot_puid,
    arm_joints,
    tool_link_left,
    tool_link_right,
    joint_names_12,
    goal_conf,
    goal_ee_frames,
    random_seed,
    max_ik_attempts,
    bar_sweep_box,
    position_res: Optional[float] = None,
    rotation_res: Optional[float] = None,
    home_anchor: Optional[str] = None,
):
    """Compute a feasible constrained start (bar pose + 12-vec joint conf).

    Goal-first procedure matching ``dual_arm_task_space_rrt.run`` Stage 3:
      1. solve the goal conf (from ``goal_ee_frames`` IK or ``goal_conf``),
      2. read the rigid grasp from the authored bar attachment in
         ``start_state`` (robust to a bad/placeholder start conf),
      3. FK at the goal to get the goal bar pose + both grasps,
      4. sample a bar "home" pose and solve dual-arm IK for that grasp via
         :func:`derive_constrained_start`.

    ``position_res`` / ``rotation_res`` set how finely the ssik backward-tracked
    direct path between the home pose and the goal is sampled (metres / radians).
    They matter beyond the start pose itself: when that direct path comes back
    collision-free, the caller returns it AS the M1 path and skips the RRT, so
    these two values become the spacing of the delivered trajectory. Candidate
    SCREENING inside the derivation runs at its own coarse step regardless
    (only a winning direct path is re-tracked at these values), so requesting a
    fine resolution no longer starves the home-pose sweep. ``None`` (the
    default) leaves them to
    :func:`~.dual_arm_task_space_rrt.core.derive_constrained_start_tracked`, the
    single place those defaults live.

    ``home_anchor`` restricts the derivation to ONE carry mode from
    ``HOME_BAR_ANCHORS`` ("horizontal", "vertical" or "back"); ``None`` (the
    default) samples all of them hierarchically -- again keeping core's own
    default as the single source of truth.

    Returns ``(start_conf, world_from_bar_start, world_from_bar_goal,
    goal_conf_arr, grasp_bar_from_left, grasp_bar_from_right, info)``. On
    failure the first six entries are ``None`` and ``info`` carries a
    ``failure_reason``.
    """
    from .dual_arm_task_space_rrt.core import (
        derive_constrained_start,
        derive_constrained_start_tracked,
        resolve_home_anchors,
        solve_endpoint_dual_arm_ik,
    )

    _fail = (None, None, None, None, None, None)

    # Validate the anchor label up front (cheap) so a typo fails before the
    # expensive goal IK below, not minutes into the sweep.
    home_anchor_selection = [home_anchor] if home_anchor is not None else None
    resolve_home_anchors(home_anchor_selection)

    # Wall time per stage, returned in info["stage_times"] so a slow
    # derivation can be attributed (goal branch pairing vs the sweep etc.).
    stage_t: Dict[str, float] = {}
    t0 = time.perf_counter()
    # 1. goal conf.
    if goal_ee_frames is not None:
        left_goal_frame = goal_ee_frames.get("left")
        right_goal_frame = goal_ee_frames.get("right")
        if left_goal_frame is None or right_goal_frame is None:
            raise ValueError("goal_ee_frames must contain 'left' and 'right' Frame entries")
        try:
            goal_conf_arr = _ik_dual_arm_at_frames(
                planner, start_state, left_goal_frame, right_goal_frame, joint_names_12,
            )
        except (InverseKinematicsError, CollisionCheckError) as exc:
            return (*_fail, {"failure_reason": f"goal_ik_failed: {getattr(exc, 'message', exc)}"})
        planner.set_robot_cell_state(start_state)
    else:
        goal_conf_arr = _conf12_from_target(goal_conf, joint_names_12)
    stage_t["goal_conf"] = time.perf_counter() - t0
    # Kept so the diagnostics can say whether (and how far) the ssik branch
    # pairing below moved the goal off the conf the caller asked for.
    goal_conf_authored = np.asarray(goal_conf_arr, dtype=float).copy()
    pairing_report = {}

    # 2. rigid grasp from the authored attachment (tool0_from_bar).
    bar_state = (start_state.rigid_body_states or {}).get(active_bar_id)
    attach_link_name = getattr(bar_state, "attached_to_link", None)
    attachment_frame = getattr(bar_state, "attachment_frame", None)
    if attach_link_name is None or attachment_frame is None:
        return (*_fail, {
            "failure_reason": (
                f"derive_start needs {active_bar_id!r} attached to a tool link with an "
                "attachment_frame in start_state"
            )
        })
    tool0_from_bar = _pp_pose_from_frame(attachment_frame)
    attach_link = pp.link_from_name(robot_puid, attach_link_name)

    # 3. FK at goal -> goal bar pose + both grasps.
    world_from_attach_goal = _fk_link_pose_pp(
        planner, goal_conf_arr, robot_puid, arm_joints, attach_link,
    )
    world_from_bar_goal = pp.multiply(world_from_attach_goal, tool0_from_bar)
    world_from_tool0_L_goal = _fk_link_pose_pp(
        planner, goal_conf_arr, robot_puid, arm_joints, tool_link_left,
    )
    world_from_tool0_R_goal = _fk_link_pose_pp(
        planner, goal_conf_arr, robot_puid, arm_joints, tool_link_right,
    )
    inv_bar_goal = pp.invert(world_from_bar_goal)
    grasp_bar_from_left = pp.multiply(inv_bar_goal, world_from_tool0_L_goal)
    grasp_bar_from_right = pp.multiply(inv_bar_goal, world_from_tool0_R_goal)

    # 4. derive the start. Keep the husky's (possibly non-identity) base so the
    # home anchor is interpreted in the robot's mobile-base frame.
    world_from_mobile_base = None
    base_frame = getattr(start_state, "robot_base_frame", None)
    if base_frame is not None:
        world_from_mobile_base = _pp_pose_from_frame(base_frame)

    # Collision-check candidate starts through cfab (loaded RobotCell + ACM +
    # attached bar), so no husky URDF/SRDF file on disk is required.
    cfab_collision_fn = _build_cfab_collision_fn(planner, start_state, joint_names_12)

    # Sweep settings shared by the probe below and the real derivation. Only
    # values the caller actually chose are forwarded, so an unset one keeps
    # derive_constrained_start_tracked's own default (one source of truth).
    walk_kwargs: Dict[str, Any] = {}
    if bar_sweep_box is not None:
        walk_kwargs["bar_sweep_box"] = bar_sweep_box
    if position_res is not None:
        walk_kwargs["position_res"] = position_res
    if rotation_res is not None:
        walk_kwargs["rotation_res"] = rotation_res
    if home_anchor_selection is not None:
        walk_kwargs["anchors"] = home_anchor_selection

    # * An AUTHORED goal conf (M2's own start) is kept whenever it can already
    # * walk to a home pose. The branch pairing below exists for goals stuck on
    # * a branch no home can reach -- but it also moves M1's end off the conf
    # * M2 starts from, and on easy scenes that alone was what made every home
    # * pose fail (B3: 722 of 728 walks broke on a branch flip after the goal
    # * was moved 215 deg; with the authored conf kept, a clear corridor turned
    # * up in 36 s). So the pairing is now the exception, decided by the real
    # * question -- a short walk with collisions off, which returns at the
    # * first home the goal reaches continuously -- not by a branch-distance
    # * metric. A goal solved from frames has no authored conf to protect and
    # * is paired as before.
    pairing_needed = ssik_ik.ik_backend() == "ssik"
    if pairing_needed and goal_ee_frames is None:
        t0 = time.perf_counter()
        probe_pose, probe_conf, probe_info = derive_constrained_start_tracked(
            robot_puid, arm_joints, tool_link_left, tool_link_right,
            grasp_bar_from_left, grasp_bar_from_right, world_from_bar_goal, goal_conf_arr,
            world_from_mobile_base=world_from_mobile_base,
            joint_collision_fn=None,          # reachability only, not clearance
            max_time=GOAL_BRANCH_PROBE_S,
            **walk_kwargs,
        )
        stage_t["goal_branch_probe"] = time.perf_counter() - t0
        pairing_needed = probe_conf is None
        pairing_report["probe"] = {
            "authored_reaches_home": probe_conf is not None,
            "variant": probe_info.get("variant"),
            "candidates_walked": len(probe_info.get("candidates") or []),
            "t_s": stage_t["goal_branch_probe"],
        }
        if probe_conf is not None:
            print(f"[ssik] authored goal conf kept: it walks to home "
                  f"{probe_info.get('variant')!r} continuously "
                  f"({stage_t['goal_branch_probe']:.1f} s).")
        else:
            print(f"[ssik] authored goal conf reaches no home pose within "
                  f"{GOAL_BRANCH_PROBE_S:.0f} s -- re-pairing its branch.")

    # * Re-pick the goal conf on the branch pair that is compatible with the
    # * home anchor, so a continuous in-limit motion start->goal can exist at
    # * all (see _ssik_pair_goal_branch_with_home for the geometry).
    t0 = time.perf_counter()
    if pairing_needed:
        goal_conf_arr = _ssik_pair_goal_branch_with_home(
            planner, robot_puid, arm_joints, tool_link_left, tool_link_right,
            world_from_bar_goal, world_from_mobile_base,
            grasp_bar_from_left, grasp_bar_from_right,
            goal_conf_arr, cfab_collision_fn,
            home_anchors=home_anchor_selection,
            report=pairing_report,
        )
        # cfab probes above left the cache at the last tested conf; reset.
        planner.set_robot_cell_state(start_state)

    stage_t["goal_branch_pairing"] = time.perf_counter() - t0

    def _with_context(payload):
        """Stamp the planning context onto a result dict.

        Built at call time because the collision gate below may still re-solve
        the goal conf after this point.

        Args:
            payload (dict): the info dict about to be returned.

        Returns:
            dict: the same dict, with ``context`` filled in.
        """
        payload["context"] = _planning_context(
            start_state, active_bar_id, joint_names_12, goal_conf_arr,
            world_from_bar_goal, grasp_bar_from_left, grasp_bar_from_right,
            goal_conf_authored=goal_conf_authored, pairing_report=pairing_report)
        return payload

    # Goal must be collision-free too (mirrors dual_arm_task_space_rrt.run): the
    # frame IK above runs with collision off and may return a colliding branch.
    # If so, re-solve at the same bar pose + grasps with random restarts.
    t0 = time.perf_counter()
    if cfab_collision_fn(goal_conf_arr):
        rng = np.random.default_rng(random_seed)
        resolved_goal = solve_endpoint_dual_arm_ik(
            robot=robot_puid,
            arm_joints=arm_joints,
            tool_link_left=tool_link_left,
            tool_link_right=tool_link_right,
            bar_pose=world_from_bar_goal,
            grasp_bar_from_left=grasp_bar_from_left,
            grasp_bar_from_right=grasp_bar_from_right,
            seed_conf=goal_conf_arr,
            rng=rng,
            max_attempts=max_ik_attempts,
            collision_fn=cfab_collision_fn,
        )
        if resolved_goal is None:
            stage_t["goal_cc_resolve"] = time.perf_counter() - t0
            return (*_fail, _with_context({
                "failure_reason": "goal_in_collision", "stage_times": stage_t}))
        goal_conf_arr = np.asarray(resolved_goal, dtype=float)
        # cfab probes left the cache at the last sampled conf; reset to start.
        planner.set_robot_cell_state(start_state)

    stage_t["goal_cc_resolve"] = time.perf_counter() - t0

    derive_kwargs = dict(
        bar_body=bar_body,
        obstacles=obstacles,
        world_from_mobile_base=world_from_mobile_base,
        random_seed=random_seed,
        max_ik_attempts=max_ik_attempts,
        joint_collision_fn=cfab_collision_fn,
    )
    if bar_sweep_box is not None:
        derive_kwargs["bar_sweep_box"] = bar_sweep_box
    # Only forward a chosen anchor; unset keeps core's all-anchors default.
    if home_anchor_selection is not None:
        derive_kwargs["anchors"] = home_anchor_selection

    world_from_bar_start = start_conf = None
    tracked_info = None  # set by the ssik tracked derivation below
    if ssik_ik.ik_backend() == "ssik":
        # * ssik first choice: derive the start by TRACKING the goal conf
        # * backward to a home pose -- the arrival config is on the goal's own
        # * IK branch by construction, so a continuous in-limit joint path
        # * start->goal exists (see derive_constrained_start_tracked).
        tracked_kwargs = dict(
            walk_kwargs,
            world_from_mobile_base=world_from_mobile_base,
            joint_collision_fn=cfab_collision_fn,
        )
        t0 = time.perf_counter()
        world_from_bar_start, start_conf, tracked_info = derive_constrained_start_tracked(
            robot_puid,
            arm_joints,
            tool_link_left,
            tool_link_right,
            grasp_bar_from_left,
            grasp_bar_from_right,
            world_from_bar_goal,
            goal_conf_arr,
            **tracked_kwargs,
        )
        stage_t["tracked_sweep"] = time.perf_counter() - t0
        if start_conf is not None:
            partial_note = ""
            if tracked_info.get("partial"):
                partial_note = (f", PARTIAL start {tracked_info.get('partial_distance_m', 0.0):.3f} m "
                                f"from goal (M0 covers the rest)")
            direct_path_note = ", direct path COLLISION-FREE (RRT will be skipped)" \
                if "direct_path" in tracked_info else ""
            print(f"[ssik] start derived by backward tracking "
                  f"(home variant {tracked_info.get('variant')}, delta {tracked_info.get('delta')}, "
                  f"max |start-goal| {np.max(np.abs(start_conf - goal_conf_arr)):.3f} rad"
                  f"{partial_note}{direct_path_note})")
            # Where blocked direct paths collided, as fractions along goal->home
            # (0 = at the goal exit, 1 = at home). Clustered-low means straight
            # direct paths are hopeless and the RRT must find the detour.
            fractions = tracked_info.get("blocked_fractions")
            if fractions:
                print(f"[ssik]   {len(fractions)} direct paths blocked; collision at "
                      f"fraction min {min(fractions):.2f} / median "
                      f"{sorted(fractions)[len(fractions) // 2]:.2f} / max {max(fractions):.2f} "
                      f"of the way goal->home")
        else:
            print(f"[ssik] backward-tracked start derivation found nothing "
                  f"({tracked_info.get('track_breaks')} track breaks over "
                  f"{tracked_info.get('tracked_deltas')} home candidates, "
                  f"{tracked_info.get('arrival_collisions')} arrival collisions); "
                  f"falling back to the cold endpoint sweep.")

    # Everything the sweep recorded except the (large) direct path itself: the
    # counters, the per-candidate trace and the timing profile.
    tracked_report = (
        {k: v for k, v in tracked_info.items() if k != "direct_path"}
        if tracked_info is not None else None
    )
    if start_conf is None or world_from_bar_start is None:
        t0 = time.perf_counter()
        world_from_bar_start, start_conf = derive_constrained_start(
            robot_puid,
            arm_joints,
            tool_link_left,
            tool_link_right,
            grasp_bar_from_left,
            grasp_bar_from_right,
            world_from_bar_goal,
            seed_conf=goal_conf_arr,
            **derive_kwargs,
        )
        stage_t["cold_sweep"] = time.perf_counter() - t0
    if start_conf is None or world_from_bar_start is None:
        return (*_fail, _with_context({
            "failure_reason": "start_derivation_failed",
            "stage_times": stage_t,
            "tracked": tracked_report,
        }))

    info = _with_context({
        "derived_start_conf": [float(x) for x in start_conf],
        "stage_times": stage_t,
        "tracked": tracked_report,
    })
    # Forward a fully collision-free tracked direct path (goal->home order) when
    # the ssik derivation found one -- the caller can use it as the finished
    # M1 path and skip the RRT (direct-connect-first).
    if ssik_ik.ik_backend() == "ssik" and tracked_info is not None and "direct_path" in tracked_info:
        info["direct_path"] = tracked_info["direct_path"]
    return (
        np.asarray(start_conf, dtype=float),
        world_from_bar_start,
        world_from_bar_goal,
        goal_conf_arr,
        grasp_bar_from_left,
        grasp_bar_from_right,
        info,
    )


def _joint_trajectory_from_path_12(path_12, joint_names_12) -> JointTrajectory:
    """Wrap a list of 12-vecs into a compas_fab JointTrajectory."""
    types = [Joint.REVOLUTE] * len(joint_names_12)
    points = []
    for i, q in enumerate(path_12):
        q = [float(x) for x in q]
        if len(q) != 12:
            raise ValueError(f"path_12[{i}] must be length 12, got {len(q)}")
        points.append(
            JointTrajectoryPoint(
                joint_values=q,
                joint_types=types,
                joint_names=list(joint_names_12),
            )
        )
    return JointTrajectory(
        trajectory_points=points,
        joint_names=list(joint_names_12),
    )


# ---------------------------------------------------------------------------
# Public planners
# ---------------------------------------------------------------------------


def plan_free_dual_arm(
    planner,
    start_state: RobotCellState,
    goal_conf,
    *,
    max_time: float = 10.0,
    max_iterations: int = 20,
    joint_resolution: float = 0.05,
    smooth_iterations: int = 20,
    debug: bool = False,
    draw_fn=None,
) -> Tuple[Optional[List[np.ndarray]], dict]:
    """12-DOF joint-space BiRRT between ``start_state`` and ``goal_conf``.

    Uses ``planner.check_collision`` as the collision predicate — obstacles,
    attached tools, attached rigid bodies, and the ACM all come from the
    cell state. ``pybullet_planning.solve_motion_plan`` runs the BiRRT.

    ``draw_fn`` (optional) is forwarded to ``solve_motion_plan`` for live
    search-tree visualization; see pybullet_planning's ``rrt_connect`` for its
    ``draw_fn(config, segment, *valid)`` contract. ``None`` (default) draws
    nothing.

    Returns ``(path, info)`` where ``path`` is a list of 12-vec numpy
    arrays, or ``None`` on failure (``info['failure_reason']`` populated).
    """
    robot_cell = planner.client.robot_cell
    left_names, right_names = _arm_joint_names(robot_cell)
    joint_names_12 = left_names + right_names

    start_conf = _conf12_from_state(start_state, joint_names_12)
    goal_conf_arr = _conf12_from_target(goal_conf, joint_names_12)

    robot_puid = planner.client.robot_puid
    arm_joints = pp.joints_from_names(robot_puid, joint_names_12)

    planner.set_robot_cell_state(start_state)
    collision_fn = _build_cfab_collision_fn(planner, start_state, joint_names_12)
    resolutions = np.ones(12) * float(joint_resolution)
    sample_fn = pp.get_sample_fn(robot_puid, arm_joints)
    distance_fn = pp.get_distance_fn(robot_puid, arm_joints)
    extend_fn = pp.get_extend_fn(robot_puid, arm_joints, resolutions=resolutions)

    info: Dict[str, Any] = {
        "max_time": float(max_time),
        "max_iterations": int(max_iterations),
        "joint_resolution": float(joint_resolution),
    }

    raw_path = None
    with pp.WorldSaver():
        pp.set_joint_positions(robot_puid, arm_joints, start_conf)
        if not pp.check_initial_end(start_conf, goal_conf_arr, collision_fn, diagnosis=debug):
            info["failure_reason"] = "start_or_goal_in_collision"
            return None, info
        raw_path = pp.solve_motion_plan(
            start_conf,
            goal_conf_arr,
            distance_fn,
            sample_fn,
            extend_fn,
            collision_fn,
            algorithm="birrt",
            max_time=max_time,
            max_iterations=int(max_iterations),
            smooth=int(smooth_iterations),
            diagnosis=debug,
            coarse_waypoints=False,
            draw_fn=draw_fn,
        )

    if raw_path is None:
        info["failure_reason"] = "birrt_failed"
        return None, info
    return [np.asarray(q, dtype=float) for q in raw_path], info


def _round_corners(info: Dict[str, Any], options: Dict[str, Any], path_poses, path_confs, post_scene,
                   joint_collision_fn, feature_points, position_res: float, rotation_res: float,
                   threshold) -> Tuple[Any, Any]:
    """Round a found path's corners in bar pose space (``plan_constrained_dual_arm``'s ``round_corners``).

    Records the report as ``info["round_profile"]`` and the new poses as ``info["path_poses"]``.

    Returns:
        Tuple: the poses and joints, rounded, or as they were when rounding failed.
    """
    from .dual_arm_task_space_rrt.smooth import round_dual_arm_pose_path

    profile: Dict[str, Any] = {}
    # The planner's own resolution, gate and feature points unless the options name their own.
    kwargs: Dict[str, Any] = {"feature_points": feature_points, "position_res": position_res,
                              "rotation_res": rotation_res, "joint_continuity_threshold_rad": threshold}
    kwargs.update(options)
    kwargs.update({"scene": post_scene, "joint_collision_fn": joint_collision_fn, "profile_out": profile})
    path_poses, path_confs = round_dual_arm_pose_path(path_poses, path_confs, **kwargs)
    info["round_profile"] = profile
    info["path_poses"] = path_poses
    return path_poses, path_confs


def plan_constrained_dual_arm(
    planner,
    start_state: RobotCellState,
    *,
    active_bar_id: str,
    goal_conf=None,
    goal_ee_frames: Optional[dict] = None,
    feature_points=None,
    stage: int = 3,
    position_res: float = 0.01,
    rotation_res: float = 0.025,
    max_time: float = 30.0,
    max_iterations: int = 2000,
    max_attempts: int = 5,
    use_birrt: bool = True,
    enable_smoothing: bool = True,
    smooth_max_iterations: int = 100,
    smooth_max_time: float = 10.0,
    joint_continuity_threshold_rad: Optional[float] = None,
    random_seed: Optional[int] = None,
    use_draw: bool = False,
    derive_start: bool = False,
    start_random_seed: Optional[int] = None,
    start_max_ik_attempts: int = 20,
    start_bar_sweep_box: Optional[
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    ] = None,
    start_home_anchor: Optional[str] = None,
    record_search: bool = False,
    ik_options: Optional[Dict[str, Any]] = None,
    repair: Optional[Dict[str, Any]] = None,
    backend: str = "birrt",
    prm_options: Optional[Dict[str, Any]] = None,
    round_corners: Optional[Dict[str, Any]] = None,
    goal_accept: Optional[Callable[[np.ndarray], str]] = None,
    floor: bool = True,
    strict_contacts: bool = True,
    start_accept: Optional[Callable[[np.ndarray], str]] = None,
) -> Tuple[Optional[List[np.ndarray]], dict]:
    """Constrained dual-arm SE(3) RRT with a rigid bar grasp.

    ``strict_contacts`` (default True) plans with nothing the arms carry or
    wear allowed to touch a standing body: the contacts the scene allows
    for the mate this transfer ends before (the tool and the receiver, the
    two halves) would let the path cut through the receiver
    (``contacts.py``). A goal that then collides only with such a body fails
    as ``goal_touches_pending_mate``: the insert's offset is too small.

    ``floor`` (default True) plans with a slab under
    the robot, its top level with the robot's base frame: the arms, their
    tools and everything they carry must stay above it (``floor.py``).

    ``goal_accept`` (roadmap backend, with ``goal_ee_frames``) is asked about
    the end posture the roadmap resolved at the goal poses, as twelve joint
    values: it returns "" to take it, or why not, and the search then goes on
    to another posture (``info["prm"]["goals_rejected"]``). A caller uses it
    to keep only postures the next movements can follow from.

    ``start_accept`` (roadmap backend, with ``derive_start``) is asked the
    same way about the loading posture a route starts from
    (``info["prm"]["starts_rejected"]``): a caller uses it to keep only
    postures the robot can drive in with (``benchmark/drive_in.py``).

    ``backend`` picks the planner: ``"birrt"`` (default) is the task-space
    search described below; ``"prm"`` plans on a roadmap precomputed for this
    bar with only the robot, its tools and the carried bar, and checks the
    real scene at plan time (``holding_prm.online.plan_on_roadmap``).
    ``prm_options`` then needs ``roadmap_dir`` (the roadmap's folder) and may
    give ``workers``, ``tolerances_mm``, ``tube_rounds``, ``tube_draws``,
    ``scene_cache_dir``, ``verbose`` and the fallbacks for a goal no carry pose
    connects to (``walk_in``, ``via_rounds``, ``walk_grid_step_m``,
    ``save_derived``: see ``OnlineSettings``) and ``goal_turn_reference``
    (twelve joints, e.g. the authored end joints: end postures are also
    offered on the whole turns nearest them). With the roadmap, ``derive_start=True``
    lets the search pick the start among the roadmap's carry poses (returned as
    ``info['derived_start_conf']``), ``goal_ee_frames`` accepts any
    collision-free posture at the goal pose and ``goal_conf`` asks for exactly
    that one; ``repair`` and ``enable_smoothing`` apply as for the search. The
    roadmap backend never falls back to the search: with no roadmap for the
    bar it raises, and with no route it returns ``failure_reason``.

    ``ik_options`` and ``repair`` exist for the benchmark's planner variants
    and are ``None`` in production. ``ik_options`` is handed to every
    per-waypoint IK of the search (which ssik solver set, how loose the FK
    gate is, whether to diagnose continuity stops -- see
    ``core._solve_dual_arm_pose_ik_ssik``). ``repair`` asks for the found
    pose path to be re-solved afterwards on the calibrated solver with the
    production gates (``core.repair_joint_path``); its one key is the
    ``joint_continuity_threshold_rad`` to enforce there. With repair on, the
    search result is reported as ``info["search_success"]``, the repair's
    verdict as ``info["repair"]``, and a path is returned only when the repair
    succeeded (``failure_reason`` = ``repair_<cause>`` otherwise). Smoothing
    then runs with the same strict settings, so it keeps the path valid.

    ``round_corners`` (None = off) rounds the corners of the found bar path
    in pose space after the smoothing (``smooth.round_dual_arm_pose_path``,
    its keyword arguments: ``radius``, ``tolerance``, ...; ``{}`` for the
    defaults), so the arms can run through them instead of stopping at each;
    its report is ``info["round_profile"]``. A path it cannot round comes
    back as it was.

    ``record_search=True`` keeps every failed tree extension with its cause
    (see ``SearchRecorder`` in the planner core) and returns it as
    ``info["profile"]["attempts"]``, next to both trees with their joint
    configurations under ``info["profile"]["tree_a"]`` / ``["tree_b"]``. It is
    off by default: it costs an extra IK or collision probe per failure and
    a few MB of memory, which the benchmark accepts and production does not.

    By default this runs the bidirectional RRT-Connect search
    (``use_birrt=True`` -> ``plan_pose_birrt``), which benchmarking showed to
    be the clear winner. Set ``use_birrt=False`` to fall back to the archival
    single start-rooted tree (``plan_pose_rrt``), kept only for comparison.

    Exactly one of ``goal_conf`` (12-vec / Configuration) or
    ``goal_ee_frames`` (dict ``{'left': Frame, 'right': Frame}``) must be
    given. The latter matches the Rhino-exported Movement schema where M1
    carries left/right tool0 goal frames instead of a full target joint
    configuration.

    Internally derives:
      - ``world_from_bar_start`` from ``start_state.rigid_body_states[active_bar_id].frame``
      - ``grasp_bar_from_left/right`` by FK at the start configuration
      - ``world_from_bar_goal``: from ``goal_ee_frames['left']`` + the start
        grasp, or by FK at ``goal_conf`` + propagating the grasp.

    ``derive_start`` (M1 / home->approach): instead of trusting
    ``start_state.robot_configuration`` as the start, compute a feasible one.
    Solve the goal first, read the rigid grasp from the authored bar
    attachment, then sample a bar "home" pose and solve dual-arm IK for that
    grasp (``derive_constrained_start``) to get a collision-free, grasp-
    consistent start (bar pose + joint conf). The derived 12-vec is returned
    in ``info['derived_start_conf']``. This mirrors the standalone
    ``dual_arm_task_space_rrt.run`` Stage-3 procedure and is the right choice
    when the caller has no trustworthy start configuration (e.g. the Rhino
    export leaves M1's start joints as a placeholder).

    ``start_home_anchor`` picks the carry mode the derived home pose may use:
    None (default) samples all of ``HOME_BAR_ANCHORS`` hierarchically --
    "horizontal" (bar across the front, the original behavior), "vertical"
    (bar upright in front) and "back" (bar fore-aft over the robot) -- while a
    specific label restricts the derivation to that single anchor.

    ``position_res`` / ``rotation_res`` (metres / radians) are the SE(3) step
    the search advances the bar by, and they double as the goal-reached
    tolerance. They reach three places: the RRT's extend step, the shortcut
    smoother, and -- with ``derive_start=True`` on the ssik backend -- the
    backward-tracked direct path, which is returned as the path outright whenever
    it comes back collision-free. Lowering them therefore makes the delivered
    trajectory denser on every one of those routes.

    Stages:
      1 -> pose-only RRT, no IK, no joint-space collision
      2 -> pose RRT + IK in extend
      3 -> pose RRT + IK + joint-space collision (full)

    Returns ``(path, info)``.
    """
    if (goal_conf is None) == (goal_ee_frames is None):
        raise ValueError(
            "exactly one of goal_conf / goal_ee_frames must be provided"
        )
    # * Argument checks come before the planner core is imported, so a wrong
    # * call fails without loading it.
    if stage not in (1, 2, 3):
        raise ValueError(f"stage must be 1, 2, or 3; got {stage}")
    if repair is not None and stage < 3:
        raise ValueError("repair needs stage 3 (the repair re-solves with full collision)")
    if backend not in PLANNING_BACKENDS:
        raise ValueError(f"backend must be one of {PLANNING_BACKENDS}; got {backend!r}")
    if backend == "prm" and stage != 3:
        raise ValueError("the roadmap backend always plans with IK and full collision (stage 3)")
    if (goal_accept is not None or start_accept is not None) and backend != "prm":
        raise ValueError("goal_accept and start_accept are only asked by the roadmap backend")
    if strict_contacts:
        # * Plan again with the strict state. Nothing but the arguments is local yet.
        arguments = {name: value for name, value in locals().items()
                     if name not in ("planner", "start_state", "strict_contacts")}
        strict_state, removed = strict_carry_state(start_state)
        path, info = plan_constrained_dual_arm(planner, strict_state, strict_contacts=False, **arguments)
        info["strict_contacts"] = {"allowances_removed": len(removed)}
        known_goal = goal_conf if goal_conf is not None else (prm_options or {}).get("goal_turn_reference")
        if path is None and known_goal is not None and removed:
            # Why the goal failed, when it did: is it only the mate's parts, which the state no longer lets touch?
            left_names, right_names = _arm_joint_names(planner.client.robot_cell)
            hits = _build_cfab_collision_fn(planner, strict_state, left_names + right_names).explain(known_goal)
            planner.set_robot_cell_state(start_state)
            lost = {frozenset(pair) for pair in removed}
            if hits and all(frozenset(hit) in lost for hit in hits):
                info["goal_failure_before_strict"] = info.get("failure_reason")
                info["failure_reason"] = "goal_touches_pending_mate"
                info["strict_contacts"].update(
                    goal_hits=[list(hit) for hit in hits],
                    note="at the goal these parts of the pending mate overlap: the insert's offset is too small")
        return path, info
    if floor:
        # * Plan again with the slab in the cell and in the state. Nothing but the arguments is local yet.
        arguments = {name: value for name, value in locals().items()
                     if name not in ("planner", "start_state", "floor")}
        with floor_added(planner, start_state) as with_floor:
            return plan_constrained_dual_arm(planner, with_floor, floor=False, **arguments)

    from .dual_arm_task_space_rrt.core import (
        DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
        SearchRecorder,
        get_bar_feature_points,
        plan_pose_birrt,
        plan_pose_rrt,
        repair_joint_path,
    )
    from .dual_arm_task_space_rrt.smooth import smooth_dual_arm_pose_path

    robot_cell = planner.client.robot_cell
    left_names, right_names = _arm_joint_names(robot_cell)
    joint_names_12 = left_names + right_names

    robot_puid = planner.client.robot_puid
    arm_joints = pp.joints_from_names(robot_puid, joint_names_12)
    tool_link_left = pp.link_from_name(robot_puid, TOOL_LINK_LEFT)
    tool_link_right = pp.link_from_name(robot_puid, TOOL_LINK_RIGHT)

    # Apply start_state once so pybullet world matches what we will read.
    planner.set_robot_cell_state(start_state)
    bar_body = _bar_body_id(planner, active_bar_id)
    obstacles = _collect_obstacle_puids(planner, exclude={active_bar_id})

    if backend == "prm":
        # * The precomputed roadmap takes over from here; see the function below. Its workers stay open for the
        # * dense check of the finished path, and are closed here.
        pools: List[Any] = []
        try:
            return _plan_constrained_dual_arm_on_roadmap(
                planner, start_state,
                active_bar_id=active_bar_id,
                goal_conf=goal_conf,
                goal_ee_frames=goal_ee_frames,
                joint_names_12=joint_names_12,
                robot_puid=robot_puid,
                arm_joints=arm_joints,
                tool_link_left=tool_link_left,
                tool_link_right=tool_link_right,
                derive_start=derive_start,
                position_res=position_res,
                rotation_res=rotation_res,
                max_time=max_time,
                enable_smoothing=enable_smoothing,
                smooth_max_iterations=smooth_max_iterations,
                smooth_max_time=smooth_max_time,
                joint_continuity_threshold_rad=joint_continuity_threshold_rad,
                random_seed=random_seed,
                feature_points=feature_points,
                repair=repair,
                prm_options=prm_options,
                round_corners=round_corners,
                goal_accept=goal_accept,
                start_accept=start_accept,
                pools=pools,
            )
        finally:
            for pool in pools:
                pool.close()

    if derive_start:
        # Goal-first procedure (mirrors dual_arm_task_space_rrt.run Stage 3):
        # the start joint conf is computed, not taken from start_state.
        (
            start_conf,
            world_from_bar_start,
            world_from_bar_goal,
            goal_conf_arr,
            grasp_bar_from_left,
            grasp_bar_from_right,
            derive_info,
        ) = _derive_constrained_start_for_plan(
            planner, start_state,
            active_bar_id=active_bar_id,
            bar_body=bar_body,
            obstacles=obstacles,
            robot_puid=robot_puid,
            arm_joints=arm_joints,
            tool_link_left=tool_link_left,
            tool_link_right=tool_link_right,
            joint_names_12=joint_names_12,
            goal_conf=goal_conf,
            goal_ee_frames=goal_ee_frames,
            random_seed=(start_random_seed if start_random_seed is not None else random_seed),
            max_ik_attempts=start_max_ik_attempts,
            bar_sweep_box=start_bar_sweep_box,
            home_anchor=start_home_anchor,
            # The ssik tracked direct path may BE the returned path (see the
            # direct-connect shortcut below), so it has to be sampled at the
            # resolution asked of this plan, not at its own default.
            position_res=position_res,
            rotation_res=rotation_res,
        )
        if start_conf is None:
            return None, derive_info
        # cfab cache was touched by the goal IK / FK probes above.
        planner.set_robot_cell_state(start_state)

        # * Direct-connect shortcut (ssik): the start derivation may have found
        # * a fully collision-free tracked direct path goal->home. Reversed, that
        # * IS the M1 path -- every waypoint already passed IK continuity + the
        # * stage-3 collision check -- so the RRT search is skipped entirely.
        direct_path = derive_info.get("direct_path")
        if direct_path is not None:
            direct_path_poses, direct_path_confs = direct_path
            path_poses = list(reversed(direct_path_poses))
            path_confs = [np.asarray(q, dtype=float) for q in reversed(direct_path_confs)]
            info: Dict[str, Any] = {
                "stage": stage,
                "max_time": float(max_time),
                "ik_backend": ssik_ik.ik_backend(),
                "planner": "direct_path",
                "derived_start_conf": derive_info.get("derived_start_conf"),
                "path_poses": path_poses,
                "start_conf": [float(v) for v in start_conf],
                "world_from_bar_start": world_from_bar_start,
            }
            # The derivation's own record rides along, so a run file can show
            # the sweep that produced this corridor.
            for key in ("tracked", "context", "stage_times"):
                if key in derive_info:
                    info[key] = derive_info[key]
            return path_confs, info
    else:
        start_conf = _conf12_from_state(start_state, joint_names_12)

        # FK at start: world_from_tool0_{L,R} at start configuration.
        world_from_tool0_L_start = _fk_link_pose_pp(
            planner, start_conf, robot_puid, arm_joints, tool_link_left,
        )
        world_from_tool0_R_start = _fk_link_pose_pp(
            planner, start_conf, robot_puid, arm_joints, tool_link_right,
        )

        # world_from_bar_start: read directly from pybullet after set_robot_cell_state.
        # The bar may be attached to a link (frame=None in rigid_body_states); reading
        # the live pybullet pose covers both attached and free-standing cases.
        world_from_bar_start = pp.get_pose(bar_body)

        # Rigid grasps: grasp_bar_from_{L,R} = inv(world_from_bar_start) * world_from_tool0_{L,R}_start
        inv_bar_start = pp.invert(world_from_bar_start)
        grasp_bar_from_left = pp.multiply(inv_bar_start, world_from_tool0_L_start)
        grasp_bar_from_right = pp.multiply(inv_bar_start, world_from_tool0_R_start)

        # world_from_bar_goal: derived either from goal_ee_frames['left'] or via FK at goal_conf.
        if goal_ee_frames is not None:
            left_goal_frame = goal_ee_frames.get("left")
            if left_goal_frame is None:
                raise ValueError("goal_ee_frames must contain a 'left' Frame entry")
            world_from_tool0_L_goal = _pp_pose_from_frame(left_goal_frame)
            # Use planner IK at the goal frames to seed goal_conf for plan_pose_rrt.
            right_goal_frame = goal_ee_frames.get("right")
            if right_goal_frame is None:
                raise ValueError("goal_ee_frames must contain a 'right' Frame entry")
            try:
                goal_conf_arr = _ik_dual_arm_at_frames(
                    planner, start_state, left_goal_frame, right_goal_frame, joint_names_12,
                )
            except (InverseKinematicsError, CollisionCheckError) as exc:
                return None, {
                    "failure_reason": f"goal_ik_failed: {getattr(exc, 'message', exc)}",
                }
            # Restore cfab cache to start_state after IK.
            planner.set_robot_cell_state(start_state)
        else:
            goal_conf_arr = _conf12_from_target(goal_conf, joint_names_12)
            world_from_tool0_L_goal = _fk_link_pose_pp(
                planner, goal_conf_arr, robot_puid, arm_joints, tool_link_left,
            )
        world_from_bar_goal = pp.multiply(
            world_from_tool0_L_goal, pp.invert(grasp_bar_from_left),
        )
    if feature_points is None:
        feature_points = get_bar_feature_points()

    enable_ik = stage >= 2
    enforce_collision = stage >= 3
    if joint_continuity_threshold_rad is None and enable_ik:
        joint_continuity_threshold_rad = DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD

    info: Dict[str, Any] = {
        "stage": stage,
        "max_time": float(max_time),
        "joint_continuity_threshold_rad": joint_continuity_threshold_rad,
        # Which IK engine solved the waypoints/endpoints (observability only).
        "ik_backend": ssik_ik.ik_backend(),
        # Which search algorithm ran (single-tree rrt vs bidirectional connect).
        "planner": "birrt" if use_birrt else "rrt",
        # The experiment knobs in force (empty in production).
        "ik_options": dict(ik_options or {}),
        "repair_settings": dict(repair or {}),
    }
    info["start_conf"] = [float(v) for v in start_conf]
    info["world_from_bar_start"] = world_from_bar_start
    if derive_start:
        info["derived_start_conf"] = derive_info.get("derived_start_conf")
        for key in ("tracked", "context", "stage_times"):
            if key in derive_info:
                info[key] = derive_info[key]
    else:
        # Planned from a stored start: no sweep to report, but the viewer still
        # needs the frames and the goal to draw the search.
        info["context"] = _planning_context(
            start_state, active_bar_id, joint_names_12, goal_conf_arr,
            world_from_bar_goal, grasp_bar_from_left, grasp_bar_from_right)

    # * The settings every re-solve AFTER the search uses (repair, smoothing).
    # * With repair on these are the production ones -- calibrated solver,
    # * exact FK gate, the strict continuity gate -- whatever the search used.
    if repair is not None:
        post_ik_options: Dict[str, Any] = {"artifact_set": "calibrated"}
        post_threshold = repair.get(
            "joint_continuity_threshold_rad", DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
        )
    else:
        post_ik_options = dict(ik_options or {})
        post_threshold = joint_continuity_threshold_rad

    with pp.WorldSaver():
        joint_collision_fn = None
        if enforce_collision:
            joint_collision_fn = _build_cfab_collision_fn(
                planner, start_state, joint_names_12,
            )
            info["joint_collision_backend"] = "cfab"

        ik_context = None
        if enable_ik:
            ik_context = {
                "robot": robot_puid,
                "arm_joints": arm_joints,
                "tool_link_left": tool_link_left,
                "tool_link_right": tool_link_right,
                "grasp_bar_from_left": grasp_bar_from_left,
                "grasp_bar_from_right": grasp_bar_from_right,
                # Experiment knobs for every IK the search makes (None = production).
                "ik_options": ik_options,
            }
        planner_profile: Dict[str, Any] = {}
        # * Bidirectional connect is now the default: it handles narrow-passage
        # * goals (a goal pose inside a cluttered pocket that a single start-
        # * rooted tree cannot thread into) by also growing a goal-rooted tree
        # * out of the pocket. Both trees stay on one IK branch sheet thanks to
        # * the ssik pairing + tracked start, so tree-connects pass the
        # * continuity gate. The single-tree plan_pose_rrt is kept for archival
        # * comparison only (use_birrt=False).
        rrt_fn = plan_pose_birrt if use_birrt else plan_pose_rrt
        path_poses, path_confs = rrt_fn(
            robot=robot_puid,
            bar_body=bar_body,
            obstacle_bodies=obstacles,
            start_pose=world_from_bar_start,
            goal_pose=world_from_bar_goal,
            planner=planner,
            start_state=start_state,
            active_bar_id=active_bar_id,
            start_conf=start_conf,
            goal_conf=goal_conf_arr,
            enable_collision=enforce_collision,
            enable_ik=enable_ik,
            ik_context=ik_context,
            joint_collision_fn=joint_collision_fn,
            feature_points=feature_points,
            position_res=position_res,
            rotation_res=rotation_res,
            max_time=max_time,
            max_iterations=max_iterations,
            max_attempts=max_attempts,
            random_seed=random_seed,
            use_draw=use_draw,
            joint_continuity_threshold_rad=joint_continuity_threshold_rad,
            # ! plan_pose_rrt's diagnostics parameter is named debug_tree_out
            # ! (previously passed as profile_out, which silently landed in
            # ! **_unused_kwargs and left info["profile"] empty).
            debug_tree_out=planner_profile,
            recorder=SearchRecorder() if record_search else None,
        )
        info["profile"] = planner_profile
        info["path_poses"] = path_poses
        if path_poses is None:
            info["failure_reason"] = planner_profile.get("outcome", "rrt_failed")
            planner.set_robot_cell_state(start_state)
            return None, info

        # The scene every re-solve after the search works in: the same robot
        # and grasps, the post-search solver settings (see above).
        post_scene = {
            "robot": robot_puid,
            "arm_joints": arm_joints,
            "tool_link_left": tool_link_left,
            "tool_link_right": tool_link_right,
            "grasp_bar_from_left": grasp_bar_from_left,
            "grasp_bar_from_right": grasp_bar_from_right,
            "ik_options": post_ik_options,
        }

        # * Repair (planner variants): the search found a bar path; re-solve
        # * its joints strictly on the calibrated solver before anything else
        # * touches it. The verdict is the experiment's measurement, so it is
        # * recorded whether or not it passes.
        if repair is not None and path_confs is not None:
            info["search_success"] = True
            repaired, repair_info = repair_joint_path(
                post_scene, path_poses, start_conf, goal_conf_arr,
                joint_collision_fn=joint_collision_fn,
                joint_continuity_threshold_rad=post_threshold,
            )
            info["repair"] = repair_info
            if repaired is None:
                info["failure_reason"] = f"repair_{repair_info.get('reason') or 'failed'}"
                planner.set_robot_cell_state(start_state)
                return None, info
            path_confs = repaired

        if enable_smoothing and path_confs is not None:
            smooth_profile: Dict[str, Any] = {}
            path_poses, path_confs = smooth_dual_arm_pose_path(
                path_poses=path_poses,
                path_confs=path_confs,
                scene=post_scene,
                pose_collision_fn=None,
                joint_collision_fn=joint_collision_fn,
                feature_points=feature_points,
                position_res=position_res,
                rotation_res=rotation_res,
                joint_continuity_threshold_rad=post_threshold,
                max_smooth_iterations=smooth_max_iterations,
                max_time=smooth_max_time,
                random_seed=random_seed,
                profile_out=smooth_profile,
            )
            info["smooth_profile"] = smooth_profile
            info["path_poses"] = path_poses

        if round_corners is not None and path_confs is not None:
            path_poses, path_confs = _round_corners(
                info, round_corners, path_poses, path_confs, post_scene, joint_collision_fn, feature_points,
                position_res, rotation_res, post_threshold)

    # Restore: WorldSaver already rewound the pybullet world; this re-syncs
    # cfab's Python-side cache to start_state (the RRT's last collision check
    # left the cache at the final sample).
    planner.set_robot_cell_state(start_state)

    if path_confs is None:
        if stage == 1:
            info["pose_only_success"] = True
            return None, info
        info["failure_reason"] = "no_joint_path"
        return None, info
    return [np.asarray(q, dtype=float) for q in path_confs], info


def _plan_constrained_dual_arm_on_roadmap(
    planner,
    start_state: RobotCellState,
    *,
    active_bar_id: str,
    goal_conf,
    goal_ee_frames: Optional[dict],
    joint_names_12: List[str],
    robot_puid: int,
    arm_joints: List[int],
    tool_link_left: int,
    tool_link_right: int,
    derive_start: bool,
    position_res: float,
    rotation_res: float,
    max_time: float,
    enable_smoothing: bool,
    smooth_max_iterations: int,
    smooth_max_time: float,
    joint_continuity_threshold_rad: Optional[float],
    random_seed: Optional[int],
    feature_points,
    repair: Optional[Dict[str, Any]],
    prm_options: Optional[Dict[str, Any]],
    round_corners: Optional[Dict[str, Any]] = None,
    goal_accept: Optional[Callable[[np.ndarray], str]] = None,
    start_accept: Optional[Callable[[np.ndarray], str]] = None,
    pools: Optional[List[Any]] = None,
) -> Tuple[Optional[List[np.ndarray]], dict]:
    """The roadmap backend of :func:`plan_constrained_dual_arm`.

    Reads the grasp from the authored bar attachment, works out where the bar
    must end up, hands the plan to ``holding_prm.online.plan_on_roadmap`` and
    then applies the same optional repair and smoothing as the search backend.

    Args:
        planner: the planner, with ``start_state`` already set.
        start_state (RobotCellState): the movement's start state.
        active_bar_id (str): the carried bar's rigid-body name.
        goal_conf: the goal configuration, or None.
        goal_ee_frames (Optional[dict]): the two goal tool frames, or None.
        joint_names_12 (List[str]): the twelve arm joints.
        robot_puid (int): the robot's PyBullet body.
        arm_joints (List[int]): its twelve arm joints.
        tool_link_left (int): left tool0 link index.
        tool_link_right (int): right tool0 link index.
        derive_start (bool): let the roadmap pick the start among its carry poses.
        position_res (float): bar step of the returned path (m).
        rotation_res (float): bar turn step of the returned path (rad).
        max_time (float): only recorded; the roadmap plan is not time-limited.
        enable_smoothing (bool): shortcut-smooth the path afterwards.
        smooth_max_iterations (int): smoothing budget.
        smooth_max_time (float): smoothing budget (s).
        joint_continuity_threshold_rad (Optional[float]): gate for the smoothing re-solves.
        random_seed (Optional[int]): seed of the smoothing.
        feature_points: bar feature points for the smoothing metric.
        repair (Optional[Dict[str, Any]]): re-solve the path with the production repair.
        prm_options (Optional[Dict[str, Any]]): ``roadmap_dir`` and the optional settings.
        round_corners (Optional[Dict[str, Any]]): round the path's corners afterwards (see the caller).
        goal_accept (Optional[Callable[[np.ndarray], str]]): the caller's say on an end posture (see the caller).
        start_accept (Optional[Callable[[np.ndarray], str]]): the caller's say on a loading posture (see the caller).
        pools (Optional[List[Any]]): gets the plan's worker pool, which the caller closes; with it the finished
            path is checked between its waypoints too (``prm_options["dense_check_step_deg"]``, 0.1; 0 skips it).

    Returns:
        Tuple[Optional[List[np.ndarray]], dict]: the path and ``info``, shaped
        as the search backend's.

    Raises:
        ValueError: when ``prm_options`` has no ``roadmap_dir``, or the bar is
            not attached to the left tool link.
    """
    # ! Imported here, not at the top: holding_prm.parallel imports
    # ! benchmark.scene, which imports this module.
    from .dual_arm_task_space_rrt.core import (
        DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
        get_bar_feature_points,
        repair_joint_path,
    )
    from .dual_arm_task_space_rrt.smooth import smooth_dual_arm_pose_path
    from .holding_prm.holding import matrix_from_pose
    from .dense_check import dense_check
    from .holding_prm.online import OnlineSettings, check_confs, plan_on_roadmap

    options = dict(prm_options or {})
    if not options.get("roadmap_dir"):
        raise ValueError("backend='prm' needs prm_options['roadmap_dir'] (the precomputed roadmap's folder).")

    # * The grasp, from the authored attachment (the export hangs the bar off one tool0 link).
    bar_state = start_state.rigid_body_states[active_bar_id]
    if bar_state.attached_to_link != TOOL_LINK_LEFT or bar_state.attachment_frame is None:
        raise ValueError(
            f"{active_bar_id} must be attached to {TOOL_LINK_LEFT} with an attachment frame for the roadmap "
            f"backend (it is attached to {bar_state.attached_to_link!r})."
        )
    grasp_bar_from_left = pp.invert(_pp_pose_from_frame(bar_state.attachment_frame))

    # * Where the bar must end up, and the right-hand grasp, from the goal.
    if goal_ee_frames is not None:
        if goal_ee_frames.get("left") is None or goal_ee_frames.get("right") is None:
            raise ValueError("goal_ee_frames must contain 'left' and 'right' Frame entries")
        world_from_tool0_L_goal = _pp_pose_from_frame(goal_ee_frames["left"])
        world_from_tool0_R_goal = _pp_pose_from_frame(goal_ee_frames["right"])
        goal_conf_arr = None
    else:
        goal_conf_arr = _conf12_from_target(goal_conf, joint_names_12)
        world_from_tool0_L_goal = _fk_link_pose_pp(planner, goal_conf_arr, robot_puid, arm_joints, tool_link_left)
        world_from_tool0_R_goal = _fk_link_pose_pp(planner, goal_conf_arr, robot_puid, arm_joints, tool_link_right)
    world_from_bar_goal = pp.multiply(world_from_tool0_L_goal, pp.invert(grasp_bar_from_left))
    grasp_bar_from_right = pp.multiply(pp.invert(world_from_bar_goal), world_from_tool0_R_goal)

    def as_matrix(pose) -> np.ndarray:
        """A pybullet pose as a 4x4 transform."""
        return matrix_from_pose([list(pose[0]), list(pose[1])])

    settings = OnlineSettings(
        roadmap_dir=str(options["roadmap_dir"]),
        workers=int(options.get("workers", 1)),
        tolerances_mm=[float(t) for t in options.get("tolerances_mm", (10.0, 20.0, 30.0))],
        tube_rounds=int(options.get("tube_rounds", 1)),
        tube_draws=int(options.get("tube_draws", 400)),
        position_res_m=float(position_res),
        rotation_res_rad=float(rotation_res),
        scene_cache_dir=str(options.get("scene_cache_dir", "")),
        use_pretracked=bool(options.get("use_pretracked", True)),
        save_query=bool(options.get("save_query", True)),
        query_tag=str(options.get("query_tag", "online")),
        walk_in=bool(options.get("walk_in", False)),
        via_rounds=int(options.get("via_rounds", 0)),
        walk_grid_step_m=float(options.get("walk_grid_step_m", 0.3)),
        last_tolerances_mm=[float(t) for t in options.get("last_tolerances_mm", (45.0, 60.0))],
        save_derived=bool(options.get("save_derived", False)),
        verbose=bool(options.get("verbose", True)),
    )
    start_conf = None if derive_start else _conf12_from_state(start_state, joint_names_12)
    path_confs, prm_info = plan_on_roadmap(
        planner, start_state, joint_names_12, active_bar_id, as_matrix(world_from_bar_goal), settings,
        robot_puid=robot_puid, arm_joints=arm_joints, start_conf=start_conf, goal_conf=goal_conf_arr,
        grasp_bar_from_right=as_matrix(grasp_bar_from_right), goal_accept=goal_accept,
        goal_turn_reference=options.get("goal_turn_reference"), keep_pool=pools, start_accept=start_accept,
    )
    path_poses = prm_info.pop("path_poses", None)
    if path_poses is not None:
        path_poses = [(tuple(p[0]), tuple(p[1])) for p in path_poses]

    info: Dict[str, Any] = {
        "stage": 3,
        "max_time": float(max_time),
        "joint_continuity_threshold_rad": joint_continuity_threshold_rad or DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
        "ik_backend": ssik_ik.ik_backend(),
        "planner": "prm",
        "repair_settings": dict(repair or {}),
        # The search backend's callers read these; the roadmap has no trees.
        "profile": {},
        "prm": prm_info,
        "path_poses": path_poses,
    }
    # The roadmap's checks moved the planner's state around; put it back.
    planner.set_robot_cell_state(start_state)
    if path_confs is None:
        info["failure_reason"] = prm_info.get("failure_reason", "prm_failed")
        return None, info
    if derive_start:
        info["derived_start_conf"] = [float(v) for v in path_confs[0]]

    post_threshold = (repair or {}).get("joint_continuity_threshold_rad") or info["joint_continuity_threshold_rad"]
    # * The path as each step leaves it: the dense check returns the latest one that passes.
    stages = [("route", path_poses, path_confs)]
    if repair is not None or enable_smoothing or round_corners is not None:
        post_start = perf_counter()
        with pp.WorldSaver():
            joint_collision_fn = _build_cfab_collision_fn(planner, start_state, joint_names_12)
            post_scene = {
                "robot": robot_puid,
                "arm_joints": arm_joints,
                "tool_link_left": tool_link_left,
                "tool_link_right": tool_link_right,
                "grasp_bar_from_left": grasp_bar_from_left,
                "grasp_bar_from_right": grasp_bar_from_right,
                "ik_options": {"artifact_set": "calibrated"} if repair is not None else None,
            }
            if repair is not None:
                info["search_success"] = True
                repaired, repair_info = repair_joint_path(
                    post_scene, path_poses, path_confs[0], path_confs[-1],
                    joint_collision_fn=joint_collision_fn,
                    joint_continuity_threshold_rad=post_threshold,
                )
                info["repair"] = repair_info
                if repaired is None:
                    info["failure_reason"] = f"repair_{repair_info.get('reason') or 'failed'}"
                    planner.set_robot_cell_state(start_state)
                    return None, info
                path_confs = repaired
                stages = [("route", path_poses, path_confs)]
            if enable_smoothing:
                smooth_profile: Dict[str, Any] = {}
                path_poses, path_confs = smooth_dual_arm_pose_path(
                    path_poses=path_poses,
                    path_confs=path_confs,
                    scene=post_scene,
                    pose_collision_fn=None,
                    joint_collision_fn=joint_collision_fn,
                    feature_points=feature_points if feature_points is not None else get_bar_feature_points(),
                    position_res=position_res,
                    rotation_res=rotation_res,
                    joint_continuity_threshold_rad=post_threshold,
                    max_smooth_iterations=smooth_max_iterations,
                    max_time=smooth_max_time,
                    random_seed=random_seed,
                    profile_out=smooth_profile,
                )
                info["smooth_profile"] = smooth_profile
                info["path_poses"] = path_poses
                stages.append(("smoothed", path_poses, path_confs))
            if round_corners is not None and path_poses is not None:
                path_poses, path_confs = _round_corners(
                    info, round_corners, path_poses, path_confs, post_scene, joint_collision_fn,
                    feature_points if feature_points is not None else get_bar_feature_points(),
                    position_res, rotation_res, post_threshold)
                stages.append(("rounded", path_poses, path_confs))
        prm_info.setdefault("timings", {})["post_process_s"] = perf_counter() - post_start
        planner.set_robot_cell_state(start_state)

    # * The dense check: straight joint moves between the waypoints, every point collision-checked in the workers.
    step_deg = float(options.get("dense_check_step_deg", 0.1))
    if pools and step_deg > 0.0:
        check_start = perf_counter()
        counters = prm_info.setdefault("counters", {}).setdefault("dense", {})
        explain = _build_cfab_collision_fn(planner, start_state, joint_names_12).explain
        report: Dict[str, Any] = {"step_deg": step_deg, "used": None, "failed": []}
        for name, poses, confs in reversed(stages):
            found = dense_check(confs, lambda dense: check_confs(pools[0], dense, "full", counters),
                                float(np.radians(step_deg)))
            report["n_points"] = found["n_points"]
            if found["ok"]:
                report["used"] = name
                path_poses, path_confs = poses, confs
                break
            hit = found["first_hit"]
            report["failed"].append({"path": name, "segment": hit["segment"], "alpha": hit["alpha"],
                                     "pairs": explain(hit["conf"])})
        planner.set_robot_cell_state(start_state)
        report["ok"] = report["used"] is not None
        prm_info["dense_check"] = report
        prm_info.setdefault("timings", {})["dense_check_s"] = perf_counter() - check_start
        if not report["ok"]:
            info["failure_reason"] = "dense_check_collision"
            return None, info
        info["path_poses"] = path_poses
    return [np.asarray(q, dtype=float) for q in path_confs], info


# === DEBUG (temporary; remove after M2 waypoint-1 debugging) ==================
def _dbg_probe_ik_failure(planner, state, target, ik_options, group, waypoint,
                          side, joint_names_12, in_group_joints, other_joints):
    """Re-probe a failed group IK to split unreachable / other-arm-recruited /
    collision. Runs on a copy of ``state`` so the real loop is undisturbed."""
    seed = {n: float(state.robot_configuration[n]) for n in joint_names_12}
    print(f"\n[DBG wp{waypoint}] {side} IK failed under the real options; probing why...")
    print(f"[DBG]   seed in-group = {[round(seed[n], 3) for n in in_group_joints]}")
    print(f"[DBG]   seed other    = {[round(seed[n], 3) for n in other_joints]}")

    # (1) Same target + seed, but collision OFF: does the group even converge?
    opts = dict(ik_options)
    opts["check_collision"] = False
    try:
        conf = planner.inverse_kinematics(target, state.copy(), group, opts)
    except PlanningGroupNotSupported:
        print("[DBG]   collision-OFF: CONVERGED but pybullet moved an OUT-OF-GROUP "
              "joint (whole-body IK recruited the OTHER arm) -> cfab rejects it.")
        print("[DBG]   => this arm alone cannot reach the target from the seed.")
        return
    except (InverseKinematicsError, CollisionCheckError) as e2:
        print(f"[DBG]   collision-OFF: STILL no solution ({type(e2).__name__}).")
        print("[DBG]   => target is unreachable / non-convergent, not a collision issue.")
        return

    sol = {n: float(conf[n]) for n in joint_names_12}
    d_in = max(abs(sol[n] - seed[n]) for n in in_group_joints)
    d_ot = max(abs(sol[n] - seed[n]) for n in other_joints)
    print(f"[DBG]   collision-OFF: CONVERGED in-group. |d| in-group={d_in:.3f} "
          f"other-arm={d_ot:.3f} rad")

    # (2) Does that converged solution collide? Mirror the loop's skip flags
    # (CC1 self + CC2 tool active; CC3/4/5 env skipped when skip_env_collisions).
    s = state.copy()
    for n in joint_names_12:
        s.robot_configuration[n] = sol[n]
    cc_opts = {"full_report": True, "verbose": False}
    for k in ("_skip_cc3", "_skip_cc4", "_skip_cc5"):
        if ik_options.get(k):
            cc_opts[k] = True
    try:
        planner.check_collision(s, options=cc_opts)
        print("[DBG]   that solution is COLLISION-FREE under the loop's CC flags.")
    except CollisionCheckError as cc:
        print("[DBG]   that solution COLLIDES under the loop's CC flags:")
        for line in str(cc).splitlines():
            print(f"[DBG]       {line}")
# === END DEBUG ================================================================


def _run_dual_arm_cartesian_ik_loop(
    planner,
    robot_cell,
    start_state,
    left_frames,
    right_frames,
    *,
    max_results: int = 20,
    max_descend_iterations: int = 200,
    skip_env_collisions: bool = True,
    joint_continuity_threshold_rad: Optional[float] = None,
    report: Optional[Dict[str, Any]] = None,
) -> Optional[JointTrajectory]:
    """Per-waypoint synchronized dual-arm IK with continuity check.

    Same flow as the previous version of this helper, but the wrapper into
    a JointTrajectory uses the local ``_joint_trajectory_from_path_12``
    instead of pulling the function from the teleop package.

    ``report`` (optional dict) is filled when the loop fails, so the caller can
    show why: see :func:`_run_dual_arm_cartesian_ik_loop_ssik`. The gradient
    backend only fills ``waypoint``, ``n_waypoints``, ``reason`` and the two
    target frames.
    """
    from .dual_arm_task_space_rrt.core import (
        DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
        joint_step_exceeds_threshold,
    )

    assert len(left_frames) == len(right_frames), (
        f"left/right frame lists must be equal length; got {len(left_frames)} vs {len(right_frames)}"
    )

    # * Backend dispatch: analytical ssik (the default) replaces the compas_fab
    # * descent IK below. Set HUSKY_IK_BACKEND=gradient to get the old path.
    if ssik_ik.ik_backend() == "ssik":
        return _run_dual_arm_cartesian_ik_loop_ssik(
            planner,
            robot_cell,
            start_state,
            left_frames,
            right_frames,
            skip_env_collisions=skip_env_collisions,
            joint_continuity_threshold_rad=joint_continuity_threshold_rad,
            report=report,
        )

    left_arm_joints, right_arm_joints = _arm_joint_names(robot_cell)
    joint_names_12 = left_arm_joints + right_arm_joints

    def _failed(i: int, reason: str) -> None:
        """Note which waypoint failed and why in ``report`` (if given); return None."""
        if report is not None:
            report.update(waypoint=i, n_waypoints=len(left_frames), reason=reason,
                          left_frame=left_frames[i], right_frame=right_frames[i],
                          skip_env_collisions=skip_env_collisions)
        return None

    ik_options = {
        "max_results": max_results,
        "max_descend_iterations": max_descend_iterations,
        "return_full_configuration": True,
        "check_collision": True,
        "verbose": False,
    }
    if skip_env_collisions:
        ik_options["_skip_cc3"] = True
        ik_options["_skip_cc4"] = True
        ik_options["_skip_cc5"] = True
    if joint_continuity_threshold_rad is None:
        joint_continuity_threshold_rad = DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD

    state = start_state.copy()
    planner.set_robot_cell_state(state)

    path_12: List[List[float]] = []
    N = len(left_frames)
    for i, (lf, rf) in enumerate(zip(left_frames, right_frames)):
        if i == 0:
            path_12.append([float(state.robot_configuration[n]) for n in joint_names_12])
            continue
        left_target = FrameTarget(
            lf, target_mode=TargetMode.ROBOT,
            tolerance_position=0.001, tolerance_orientation=0.01,
        )
        right_target = FrameTarget(
            rf, target_mode=TargetMode.ROBOT,
            tolerance_position=0.001, tolerance_orientation=0.01,
        )
        try:
            conf_L = planner.inverse_kinematics(left_target, state, LEFT_GROUP, ik_options)
        except (InverseKinematicsError, CollisionCheckError) as e:
            logger.warning(f"[cartesian IK loop] waypoint {i}: LEFT FAIL: {getattr(e, 'message', e)}")
            # === DEBUG (temporary; remove after M2 debugging) ===
            _dbg_probe_ik_failure(planner, state, left_target, ik_options, LEFT_GROUP,
                                  i, "LEFT", joint_names_12, left_arm_joints, right_arm_joints)
            # === END DEBUG ===
            return _failed(i, "left_unreachable")
        except PlanningGroupNotSupported:
            planner.set_robot_cell_state(state)
            try:
                conf_L = planner.inverse_kinematics(left_target, state, LEFT_GROUP, ik_options)
            except Exception as e2:
                logger.warning(f"[cartesian IK loop] waypoint {i}: LEFT FAIL after retry: {e2}")
                # === DEBUG (temporary; remove after M2 debugging) ===
                _dbg_probe_ik_failure(planner, state, left_target, ik_options, LEFT_GROUP,
                                      i, "LEFT", joint_names_12, left_arm_joints, right_arm_joints)
                # === END DEBUG ===
                return _failed(i, "left_unreachable")
        for n in right_arm_joints:
            conf_L[n] = float(state.robot_configuration[n])
        state.robot_configuration = conf_L
        try:
            conf_LR = planner.inverse_kinematics(right_target, state, RIGHT_GROUP, ik_options)
        except (InverseKinematicsError, CollisionCheckError) as e:
            logger.warning(f"[cartesian IK loop] waypoint {i}: RIGHT FAIL: {getattr(e, 'message', e)}")
            return _failed(i, "right_unreachable")
        except PlanningGroupNotSupported:
            planner.set_robot_cell_state(state)
            try:
                conf_LR = planner.inverse_kinematics(right_target, state, RIGHT_GROUP, ik_options)
            except Exception as e2:
                logger.warning(f"[cartesian IK loop] waypoint {i}: RIGHT FAIL after retry: {e2}")
                return _failed(i, "right_unreachable")
        for n in left_arm_joints:
            conf_LR[n] = float(state.robot_configuration[n])
        state.robot_configuration = conf_LR
        next_vec = [float(conf_LR[n]) for n in joint_names_12]
        if joint_step_exceeds_threshold(next_vec, path_12[-1], joint_continuity_threshold_rad):
            diff = float(
                np.abs(np.asarray(next_vec, dtype=float)
                       - np.asarray(path_12[-1], dtype=float)).max()
            )
            logger.warning(
                f"[cartesian IK loop] waypoint {i}: joint step {diff:.4f} rad exceeds "
                f"threshold {float(joint_continuity_threshold_rad):.4f}; rejecting IK branch."
            )
            return _failed(i, "no_candidate")
        path_12.append(next_vec)

    return _joint_trajectory_from_path_12(path_12, joint_names_12)


def _run_dual_arm_cartesian_ik_loop_ssik(
    planner,
    robot_cell,
    start_state,
    left_frames,
    right_frames,
    *,
    skip_env_collisions: bool = True,
    joint_continuity_threshold_rad: Optional[float] = None,
    report: Optional[Dict[str, Any]] = None,
) -> Optional[JointTrajectory]:
    """ssik variant of :func:`_run_dual_arm_cartesian_ik_loop` (M2/M3 linear).

    Per waypoint, each arm's analytical branches are enumerated with the
    previous waypoint as the seed (nearest-first after 2*pi re-branching);
    candidate left/right combinations are then accepted through the SAME two
    gates the gradient loop used -- the raw joint-step continuity threshold and
    a collision check honoring the ``skip_env_collisions`` CC-skip flags.
    First surviving combination wins; if none survives the loop fails just
    like the gradient version (return ``None``).

    Args:
        planner: compas_fab PyBulletPlanner with the cell loaded.
        robot_cell: the loaded RobotCell (for joint-name resolution).
        start_state: RobotCellState at the first waypoint (seeds the chain).
        left_frames: per-waypoint left tool0 world Frames (meters).
        right_frames: per-waypoint right tool0 world Frames (meters).
        skip_env_collisions (bool): skip robot-vs-environment checks (CC3/4/5),
            keeping self + tool checks -- contact with the workpiece is
            expected during mate/insertion.
        joint_continuity_threshold_rad (Optional[float]): max raw per-joint
            step between consecutive waypoints (default: the shared RRT value).
        report (Optional[dict]): the loop adds these only when it fails, so the
            caller can show why (the linear planners add ``waypoint_frames``
            before the loop, success or not):
            ``waypoint`` / ``n_waypoints`` (index of the failing waypoint and
            the count), ``left_frame`` / ``right_frame`` (its tool0 targets),
            ``reason`` ('left_unreachable' | 'right_unreachable' |
            'no_candidate'), ``previous_conf`` (the 12 joints of the waypoint
            before), ``candidates`` (12-vecs, nearest first), ``rejected_by``
            (per candidate: 'continuity' | 'collision'), ``max_step`` (per
            candidate, rad), ``threshold`` (rad), ``first_colliding_state``
            (RobotCellState of the nearest candidate the collision check
            rejected, or None), ``skip_env_collisions``.
            ! Waypoint 0 (the start) is never collision-checked here.

    Returns:
        Optional[JointTrajectory]: the synchronized 12-DOF trajectory, or
        ``None`` when a waypoint is unreachable / discontinuous / colliding.
    """
    from .dual_arm_task_space_rrt.core import (
        DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
        joint_step_exceeds_threshold,
    )

    left_arm_joints, right_arm_joints = _arm_joint_names(robot_cell)
    joint_names_12 = left_arm_joints + right_arm_joints
    if joint_continuity_threshold_rad is None:
        joint_continuity_threshold_rad = DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD

    # Same collision flags the gradient loop passed to its IK descent.
    cc_opts: Dict[str, Any] = {"verbose": False}
    if skip_env_collisions:
        cc_opts["_skip_cc3"] = True
        cc_opts["_skip_cc4"] = True
        cc_opts["_skip_cc5"] = True

    # Apply the start state so pybullet holds the right base placement (ssik
    # targets are expressed in each arm's base-link frame).
    state = start_state.copy()
    planner.set_robot_cell_state(state)
    robot_puid = planner.client.robot_puid

    path_12: List[List[float]] = []
    for i, (lf, rf) in enumerate(zip(left_frames, right_frames)):
        if i == 0:
            # Waypoint 0 is the current configuration, exactly as the gradient loop.
            path_12.append([float(state.robot_configuration[n]) for n in joint_names_12])
            continue
        previous = np.asarray(path_12[-1], dtype=float)
        branches_left = ssik_ik.ssik_arm_branches(
            robot_puid, "left", _pp_pose_from_frame(lf), previous[0:6], max_solutions=8,
        )
        branches_right = ssik_ik.ssik_arm_branches(
            robot_puid, "right", _pp_pose_from_frame(rf), previous[6:12], max_solutions=8,
        )
        if not branches_left or not branches_right:
            side = "LEFT" if not branches_left else "RIGHT"
            logger.warning(f"[cartesian IK loop/ssik] waypoint {i}: {side} unreachable")
            if report is not None:
                report.update(waypoint=i, n_waypoints=len(left_frames),
                              left_frame=lf, right_frame=rf,
                              reason=f"{side.lower()}_unreachable",
                              previous_conf=previous.tolist(), candidates=[], rejected_by=[],
                              max_step=[], threshold=float(joint_continuity_threshold_rad),
                              first_colliding_state=None,
                              skip_env_collisions=skip_env_collisions)
            return None
        # * Candidate combinations, smallest max-joint-step first: for linear
        # * tracking the nearest pair almost always wins; the rest are fallbacks
        # * (e.g. when the nearest branch collides).
        candidates = sorted(
            (np.concatenate([ql, qr]) for ql in branches_left for qr in branches_right),
            key=lambda conf: float(np.max(np.abs(conf - previous))),
        )
        accepted = None
        # Why each candidate was turned down, kept for ``report`` (cheap: one
        # string per candidate; the first colliding state is a reference).
        rejected_by: List[str] = []
        first_colliding_state = None
        for conf in candidates:
            next_vec = [float(v) for v in conf]
            # Cheap gate first: raw joint-step continuity (same threshold + check
            # as the gradient loop).
            if joint_step_exceeds_threshold(next_vec, path_12[-1], joint_continuity_threshold_rad):
                rejected_by.append("continuity")
                continue
            trial_state = _state_with_conf12(state, next_vec, joint_names_12)
            try:
                planner.check_collision(trial_state, options=cc_opts)
            except CollisionCheckError:
                rejected_by.append("collision")
                if first_colliding_state is None:
                    first_colliding_state = trial_state
                continue
            accepted = next_vec
            state = trial_state
            break
        if accepted is None:
            nearest_step = float(np.max(np.abs(candidates[0] - previous)))
            logger.warning(
                f"[cartesian IK loop/ssik] waypoint {i}: no branch combination passes "
                f"continuity+collision (nearest max joint step {nearest_step:.4f} rad, "
                f"threshold {float(joint_continuity_threshold_rad):.4f})."
            )
            if report is not None:
                report.update(waypoint=i, n_waypoints=len(left_frames),
                              left_frame=lf, right_frame=rf, reason="no_candidate",
                              previous_conf=previous.tolist(),
                              candidates=[c.tolist() for c in candidates],
                              rejected_by=rejected_by,
                              max_step=[float(np.max(np.abs(c - previous))) for c in candidates],
                              threshold=float(joint_continuity_threshold_rad),
                              first_colliding_state=first_colliding_state,
                              skip_env_collisions=skip_env_collisions)
            return None
        path_12.append(accepted)

    return _joint_trajectory_from_path_12(path_12, joint_names_12)


def plan_constrained_dual_arm_linear(
    planner,
    start_state: RobotCellState,
    *,
    active_bar_id: str,
    goal_conf=None,
    goal_ee_frames: Optional[dict] = None,
    max_step_distance: float = 0.005,
    max_step_angle: float = 0.05,
    max_results: int = 20,
    max_descend_iterations: int = 200,
    skip_env_collisions: bool = True,
    joint_continuity_threshold_rad: Optional[float] = None,
    report: Optional[Dict[str, Any]] = None,
) -> Optional[JointTrajectory]:
    """Linear dual-arm motion with bar held rigidly.

    Exactly one of ``goal_conf`` or ``goal_ee_frames`` must be provided.

    Bar starts at ``start_state.rigid_body_states[active_bar_id].frame``; the
    goal bar pose is derived from ``goal_ee_frames['left']`` + start grasp
    (preferred) or from FK at ``goal_conf`` + start grasp.

    ``report`` (optional dict) is filled when no path is found: see
    :func:`_run_dual_arm_cartesian_ik_loop_ssik`. Its ``waypoint_frames``
    (``{"left": [Frame], "right": [Frame]}``, the tool0 waypoints IK follows,
    waypoint 0 = the start) is filled whenever the waypoint loop runs.
    """
    if (goal_conf is None) == (goal_ee_frames is None):
        raise ValueError(
            "exactly one of goal_conf / goal_ee_frames must be provided"
        )

    from compas_fab.backends.pybullet.backend_features.pybullet_plan_cartesian_motion import (
        FrameInterpolator,
    )

    robot_cell = planner.client.robot_cell
    left_arm_joints, right_arm_joints = _arm_joint_names(robot_cell)
    joint_names_12 = left_arm_joints + right_arm_joints
    start_conf = _conf12_from_state(start_state, joint_names_12)

    state = start_state.copy()
    for n, v in zip(joint_names_12, start_conf):
        state.robot_configuration[n] = float(v)
    planner.set_robot_cell_state(state)

    start_left_frame = _fk_link_frame(planner, state, TOOL_LINK_LEFT)
    start_right_frame = _fk_link_frame(planner, state, TOOL_LINK_RIGHT)

    # Read the bar's world pose from pybullet (state is applied above). When the
    # bar is attached to a tool link its ``rigid_body_states[...].frame`` is None,
    # so reading the live pose covers both attached and free-standing cases
    # (same approach as plan_constrained_dual_arm).
    planner.set_robot_cell_state(state)
    bar_body = _bar_body_id(planner, active_bar_id)
    bar_pos, bar_quat = pp.get_pose(bar_body)
    start_world_from_bar = Frame.from_quaternion(
        [bar_quat[3], bar_quat[0], bar_quat[1], bar_quat[2]], point=list(bar_pos),
    )
    inv_bar_start = Transformation.from_frame(start_world_from_bar).inverse()
    bar_from_left_tool0 = Frame.from_transformation(
        inv_bar_start * Transformation.from_frame(start_left_frame)
    )
    bar_from_right_tool0 = Frame.from_transformation(
        inv_bar_start * Transformation.from_frame(start_right_frame)
    )

    # Goal bar frame.
    if goal_ee_frames is not None:
        left_goal_frame = goal_ee_frames.get("left")
        if left_goal_frame is None:
            raise ValueError("goal_ee_frames must contain a 'left' Frame entry")
        goal_world_from_bar = Frame.from_transformation(
            Transformation.from_frame(left_goal_frame)
            * Transformation.from_frame(bar_from_left_tool0).inverse()
        )
    else:
        goal_conf_arr = _conf12_from_target(goal_conf, joint_names_12)
        goal_state = _state_with_conf12(state, goal_conf_arr, joint_names_12)
        goal_left_frame = _fk_link_frame(planner, goal_state, TOOL_LINK_LEFT)
        goal_world_from_bar = Frame.from_transformation(
            Transformation.from_frame(goal_left_frame)
            * Transformation.from_frame(bar_from_left_tool0).inverse()
        )

    # Restore planner cache to start_state before IK loop.
    planner.set_robot_cell_state(state)

    options = {"max_step_distance": max_step_distance, "max_step_angle": max_step_angle}
    bar_interp = FrameInterpolator(start_world_from_bar, goal_world_from_bar, options)
    N = max(2, bar_interp.regular_interpolation_steps + 1)

    left_frames: List[Frame] = []
    right_frames: List[Frame] = []
    for i in range(N):
        t = i / (N - 1) if N > 1 else 0.0
        bar_t = bar_interp.get_interpolated_frame(t)
        bar_t_tf = Transformation.from_frame(bar_t)
        left_frames.append(
            Frame.from_transformation(bar_t_tf * Transformation.from_frame(bar_from_left_tool0))
        )
        right_frames.append(
            Frame.from_transformation(bar_t_tf * Transformation.from_frame(bar_from_right_tool0))
        )

    # The tool0 path IK is asked to follow, kept for the caller's drawing
    # (filled on success too, unlike the failure fields).
    if report is not None:
        report["waypoint_frames"] = {"left": left_frames, "right": right_frames}

    return _run_dual_arm_cartesian_ik_loop(
        planner, robot_cell, state,
        left_frames, right_frames,
        max_results=max_results,
        max_descend_iterations=max_descend_iterations,
        skip_env_collisions=skip_env_collisions,
        joint_continuity_threshold_rad=joint_continuity_threshold_rad,
        report=report,
    )


def plan_dual_arm_linear_independent(
    planner,
    start_state: RobotCellState,
    *,
    goal_conf=None,
    goal_ee_frames: Optional[dict] = None,
    target_left_frame: Optional[Frame] = None,
    target_right_frame: Optional[Frame] = None,
    max_step_distance: float = 0.005,
    max_step_angle: float = 0.05,
    max_results: int = 20,
    max_descend_iterations: int = 200,
    skip_env_collisions: bool = True,
    joint_continuity_threshold_rad: Optional[float] = None,
    report: Optional[Dict[str, Any]] = None,
) -> Optional[JointTrajectory]:
    """Linear dual-arm motion with each arm interpolating independently.

    The goal frames for left/right tool0 can come from (in order of preference):
      1. ``target_left_frame`` + ``target_right_frame`` explicitly,
      2. ``goal_ee_frames`` dict with ``'left'`` and ``'right'`` keys,
      3. forward kinematics at ``goal_conf``.

    ``report`` (optional dict) is filled when no path is found: see
    :func:`_run_dual_arm_cartesian_ik_loop_ssik`. Its ``waypoint_frames``
    (``{"left": [Frame], "right": [Frame]}``, the tool0 waypoints IK follows,
    waypoint 0 = the start) is filled whenever the waypoint loop runs.
    """
    from compas_fab.backends.pybullet.backend_features.pybullet_plan_cartesian_motion import (
        FrameInterpolator,
    )

    robot_cell = planner.client.robot_cell
    left_arm_joints, right_arm_joints = _arm_joint_names(robot_cell)
    joint_names_12 = left_arm_joints + right_arm_joints
    start_conf = _conf12_from_state(start_state, joint_names_12)

    state = start_state.copy()
    for n, v in zip(joint_names_12, start_conf):
        state.robot_configuration[n] = float(v)
    planner.set_robot_cell_state(state)

    start_left_frame = _fk_link_frame(planner, state, TOOL_LINK_LEFT)
    start_right_frame = _fk_link_frame(planner, state, TOOL_LINK_RIGHT)

    if goal_ee_frames is not None:
        if target_left_frame is None:
            target_left_frame = goal_ee_frames.get("left")
        if target_right_frame is None:
            target_right_frame = goal_ee_frames.get("right")

    if target_left_frame is None or target_right_frame is None:
        if goal_conf is None:
            raise ValueError(
                "must provide target_left_frame + target_right_frame, "
                "goal_ee_frames, or goal_conf"
            )
        goal_conf_arr = _conf12_from_target(goal_conf, joint_names_12)
        goal_state = _state_with_conf12(state, goal_conf_arr, joint_names_12)
        if target_left_frame is None:
            target_left_frame = _fk_link_frame(planner, goal_state, TOOL_LINK_LEFT)
        if target_right_frame is None:
            target_right_frame = _fk_link_frame(planner, goal_state, TOOL_LINK_RIGHT)
        planner.set_robot_cell_state(state)

    options = {"max_step_distance": max_step_distance, "max_step_angle": max_step_angle}
    left_interp = FrameInterpolator(start_left_frame, target_left_frame, options)
    right_interp = FrameInterpolator(start_right_frame, target_right_frame, options)
    N = max(
        2,
        max(left_interp.regular_interpolation_steps, right_interp.regular_interpolation_steps) + 1,
    )

    left_frames: List[Frame] = []
    right_frames: List[Frame] = []
    for i in range(N):
        t = i / (N - 1) if N > 1 else 0.0
        left_frames.append(left_interp.get_interpolated_frame(t))
        right_frames.append(right_interp.get_interpolated_frame(t))

    # The tool0 path IK is asked to follow, kept for the caller's drawing
    # (filled on success too, unlike the failure fields).
    if report is not None:
        report["waypoint_frames"] = {"left": left_frames, "right": right_frames}

    return _run_dual_arm_cartesian_ik_loop(
        planner, robot_cell, state,
        left_frames, right_frames,
        max_results=max_results,
        max_descend_iterations=max_descend_iterations,
        skip_env_collisions=skip_env_collisions,
        joint_continuity_threshold_rad=joint_continuity_threshold_rad,
        report=report,
    )
