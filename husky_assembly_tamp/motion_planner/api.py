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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    """
    cc_opts = {"verbose": False}

    def _fn(conf_12, *_args, **_kw):
        s = _state_with_conf12(template_state, conf_12, joint_names_12)
        try:
            planner.check_collision(s, options=cc_opts)
        except CollisionCheckError:
            return True
        return False

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
    """
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
):
    """Compute a feasible constrained start (bar pose + 12-vec joint conf).

    Goal-first procedure matching ``dual_arm_task_space_rrt.run`` Stage 3:
      1. solve the goal conf (from ``goal_ee_frames`` IK or ``goal_conf``),
      2. read the rigid grasp from the authored bar attachment in
         ``start_state`` (robust to a bad/placeholder start conf),
      3. FK at the goal to get the goal bar pose + both grasps,
      4. sample a bar "home" pose and solve dual-arm IK for that grasp via
         :func:`derive_constrained_start`.

    Returns ``(start_conf, world_from_bar_start, world_from_bar_goal,
    goal_conf_arr, grasp_bar_from_left, grasp_bar_from_right, info)``. On
    failure the first six entries are ``None`` and ``info`` carries a
    ``failure_reason``.
    """
    from .dual_arm_task_space_rrt.core import (
        derive_constrained_start,
        solve_endpoint_dual_arm_ik,
    )

    _fail = (None, None, None, None, None, None)

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

    # Goal must be collision-free too (mirrors dual_arm_task_space_rrt.run): the
    # frame IK above runs with collision off and may return a colliding branch.
    # If so, re-solve at the same bar pose + grasps with random restarts.
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
            return (*_fail, {"failure_reason": "goal_in_collision"})
        goal_conf_arr = np.asarray(resolved_goal, dtype=float)
        # cfab probes left the cache at the last sampled conf; reset to start.
        planner.set_robot_cell_state(start_state)

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
    if start_conf is None or world_from_bar_start is None:
        return (*_fail, {"failure_reason": "start_derivation_failed"})

    info = {"derived_start_conf": [float(x) for x in start_conf]}
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
) -> Tuple[Optional[List[np.ndarray]], dict]:
    """Constrained dual-arm SE(3) RRT with a rigid bar grasp.

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
    from .dual_arm_task_space_rrt.core import (
        DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
        get_bar_feature_points,
        plan_pose_rrt,
    )
    from .dual_arm_task_space_rrt.smooth import smooth_dual_arm_pose_path

    if stage not in (1, 2, 3):
        raise ValueError(f"stage must be 1, 2, or 3; got {stage}")

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
        )
        if start_conf is None:
            return None, derive_info
        # cfab cache was touched by the goal IK / FK probes above.
        planner.set_robot_cell_state(start_state)
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
    }
    if derive_start:
        info["derived_start_conf"] = derive_info.get("derived_start_conf")

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
            }
        planner_profile: Dict[str, Any] = {}
        path_poses, path_confs = plan_pose_rrt(
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
            profile_out=planner_profile,
        )
        info["profile"] = planner_profile
        info["path_poses"] = path_poses
        if path_poses is None:
            info["failure_reason"] = planner_profile.get("outcome", "rrt_failed")
            planner.set_robot_cell_state(start_state)
            return None, info

        if enable_smoothing and path_confs is not None:
            smooth_scene = {
                "robot": robot_puid,
                "arm_joints": arm_joints,
                "tool_link_left": tool_link_left,
                "tool_link_right": tool_link_right,
                "grasp_bar_from_left": grasp_bar_from_left,
                "grasp_bar_from_right": grasp_bar_from_right,
            }
            smooth_profile: Dict[str, Any] = {}
            path_poses, path_confs = smooth_dual_arm_pose_path(
                path_poses=path_poses,
                path_confs=path_confs,
                scene=smooth_scene,
                pose_collision_fn=None,
                joint_collision_fn=joint_collision_fn,
                feature_points=feature_points,
                position_res=position_res,
                rotation_res=rotation_res,
                joint_continuity_threshold_rad=joint_continuity_threshold_rad,
                max_smooth_iterations=smooth_max_iterations,
                max_time=smooth_max_time,
                random_seed=random_seed,
                profile_out=smooth_profile,
            )
            info["smooth_profile"] = smooth_profile
            info["path_poses"] = path_poses

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
) -> Optional[JointTrajectory]:
    """Per-waypoint synchronized dual-arm IK with continuity check.

    Same flow as the previous version of this helper, but the wrapper into
    a JointTrajectory uses the local ``_joint_trajectory_from_path_12``
    instead of pulling the function from the teleop package.
    """
    from .dual_arm_task_space_rrt.core import (
        DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
        joint_step_exceeds_threshold,
    )

    assert len(left_frames) == len(right_frames), (
        f"left/right frame lists must be equal length; got {len(left_frames)} vs {len(right_frames)}"
    )

    left_arm_joints, right_arm_joints = _arm_joint_names(robot_cell)
    joint_names_12 = left_arm_joints + right_arm_joints

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
            return None
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
                return None
        for n in right_arm_joints:
            conf_L[n] = float(state.robot_configuration[n])
        state.robot_configuration = conf_L
        try:
            conf_LR = planner.inverse_kinematics(right_target, state, RIGHT_GROUP, ik_options)
        except (InverseKinematicsError, CollisionCheckError) as e:
            logger.warning(f"[cartesian IK loop] waypoint {i}: RIGHT FAIL: {getattr(e, 'message', e)}")
            return None
        except PlanningGroupNotSupported:
            planner.set_robot_cell_state(state)
            try:
                conf_LR = planner.inverse_kinematics(right_target, state, RIGHT_GROUP, ik_options)
            except Exception as e2:
                logger.warning(f"[cartesian IK loop] waypoint {i}: RIGHT FAIL after retry: {e2}")
                return None
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
            return None
        path_12.append(next_vec)

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
) -> Optional[JointTrajectory]:
    """Linear dual-arm motion with bar held rigidly.

    Exactly one of ``goal_conf`` or ``goal_ee_frames`` must be provided.

    Bar starts at ``start_state.rigid_body_states[active_bar_id].frame``; the
    goal bar pose is derived from ``goal_ee_frames['left']`` + start grasp
    (preferred) or from FK at ``goal_conf`` + start grasp.
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

    return _run_dual_arm_cartesian_ik_loop(
        planner, robot_cell, state,
        left_frames, right_frames,
        max_results=max_results,
        max_descend_iterations=max_descend_iterations,
        skip_env_collisions=skip_env_collisions,
        joint_continuity_threshold_rad=joint_continuity_threshold_rad,
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
) -> Optional[JointTrajectory]:
    """Linear dual-arm motion with each arm interpolating independently.

    The goal frames for left/right tool0 can come from (in order of preference):
      1. ``target_left_frame`` + ``target_right_frame`` explicitly,
      2. ``goal_ee_frames`` dict with ``'left'`` and ``'right'`` keys,
      3. forward kinematics at ``goal_conf``.
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

    return _run_dual_arm_cartesian_ik_loop(
        planner, robot_cell, state,
        left_frames, right_frames,
        max_results=max_results,
        max_descend_iterations=max_descend_iterations,
        skip_env_collisions=skip_env_collisions,
        joint_continuity_threshold_rad=joint_continuity_threshold_rad,
    )
