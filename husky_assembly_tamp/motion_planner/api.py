"""Atom planning API for husky dual-arm motion planning.

Provides a uniform call surface used by both:
- the live PyBullet teleop monitor (single online plan request), and
- the offline TAMP chained planner (sequence of plan requests sharing scene).

Two parallel planners share input/output shape:
- plan_free_dual_arm: 12-DOF joint-space BiRRT, no end-effector constraint.
- plan_constrained_dual_arm: SE(3) bar-pose RRT with rigid dual-arm grasp.

Helpers:
- derive_grasps_from_state: extract rigid grasp transforms from a goal cell
  state (FK at goal_conf + bar pose at goal).
- derive_constrained_start: pick a "home" world_from_bar_start and solve
  endpoint IK to get start_conf for the constrained planner.

All planners take a SceneContext dict (live PyBullet body ids - never spin up
a new client) and operate inside pp.WorldSaver() to leave the live scene
untouched.

SceneContext keys (dict):
  robot: int                          # PyBullet body id of the dual-arm robot
  arm_joints: Sequence[int]           # 12 joint ids in left-then-right order
  joint_names: Sequence[str]          # 12 names; for assertion/order check
  tool_link_left: int                 # PyBullet link id of left tool0
  tool_link_right: int                # PyBullet link id of right tool0
  obstacles: Sequence[int]            # static obstacle body ids
  attachments: Optional[Sequence[pp.Attachment]]   # used by free planner
                                      # (dual_arm_index='both' requires len==2)
  disabled_collisions: Optional[set]  # currently informational; constrained
                                      # planner uses submodule SRDF (limitation)
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pybullet_planning as pp


logger = logging.getLogger(__name__)

_REQUIRED_SCENE_KEYS = (
    "robot",
    "arm_joints",
    "tool_link_left",
    "tool_link_right",
    "obstacles",
)


def _validate_scene(scene: Dict[str, Any]) -> None:
    if not isinstance(scene, dict):
        raise TypeError("scene must be a dict (SceneContext)")
    for key in _REQUIRED_SCENE_KEYS:
        if key not in scene:
            raise KeyError(f"SceneContext missing required key: {key!r}")
    if len(scene["arm_joints"]) != 12:
        raise ValueError(
            f"SceneContext arm_joints must have length 12, got {len(scene['arm_joints'])}"
        )


def plan_free_dual_arm(
    scene: Dict[str, Any],
    start_conf: Sequence[float],
    goal_conf: Sequence[float],
    *,
    max_time: float = 10.0,
    max_iterations: int = 20,
    debug: bool = False,
    joint_resolution: float = 0.05,
    cfab_collision_fn=None,
) -> Tuple[Optional[List[np.ndarray]], dict]:
    """Free-space dual-arm BiRRT.

    Wraps husky_assembly_teleop.utils.plan_transit_motion with
    dual_arm_index="both". The robot is set to start_conf inside a
    pp.WorldSaver() before calling the underlying planner; the saver
    restores robot+world state afterwards regardless of success.

    Returns (path_confs | None, info dict). path_confs is a list of
    np.ndarray with shape (12,).
    """
    _validate_scene(scene)
    if len(start_conf) != 12 or len(goal_conf) != 12:
        raise ValueError("start_conf and goal_conf must have length 12")
    attachments = scene.get("attachments")
    if not isinstance(attachments, list) or len(attachments) != 2:
        raise ValueError(
            "plan_free_dual_arm requires scene['attachments'] to be a list of "
            "exactly 2 pp.Attachment (left, right) — plan_transit_motion's "
            "dual_arm_index='both' branch enforces this."
        )

    # TODO this is disgusting circular import, plan_transit_motion should live in this repo
    from husky_assembly_teleop.utils import plan_transit_motion

    info: Dict[str, Any] = {
        "max_time": max_time,
        "max_iterations": int(max_iterations),
        "joint_resolution": float(joint_resolution),
    }
    with pp.WorldSaver():
        pp.set_joint_positions(scene["robot"], scene["arm_joints"], start_conf)
        raw = plan_transit_motion(
            scene["robot"],
            np.asarray(goal_conf, dtype=float),
            attachments,
            list(scene.get("obstacles") or []),
            debug=debug,
            disabled_collisions=scene.get("disabled_collisions"),
            dual_arm_index="both",
            joint_resolution=joint_resolution,
            max_time=max_time,
            max_iterations=max_iterations,
            # Optional list of EE type strings (len 2 for dual-arm), e.g.
            # ["assembly_tool_v3_left", "assembly_tool_v3_right"]. Lets
            # plan_transit_motion add a wrist_2_link disable per arm when
            # the mounted tool extends past wrist_3.
            ee_types=scene.get("ee_types"),
            cfab_collision_fn=cfab_collision_fn,
        )
    if raw is None:
        info["failure_reason"] = "free_planner_failed"
        return None, info
    path = [np.asarray(q, dtype=float) for q in raw]
    return path, info


def derive_grasps_from_state(
    robot: int,
    arm_joints: Sequence[int],
    tool_link_left: int,
    tool_link_right: int,
    goal_conf: Sequence[float],
    world_from_bar_goal,
):
    """Derive rigid dual-arm grasp transforms from a goal cell state.

    Inside pp.WorldSaver(): set robot to goal_conf, FK both tool0 links,
    compute grasp_bar_from_* = inv(world_from_bar_goal) * world_from_tool0_*.

    Returns (grasp_bar_from_left, grasp_bar_from_right) as PoseLike.
    """
    if len(goal_conf) != 12:
        raise ValueError("goal_conf must have length 12")
    with pp.WorldSaver():
        pp.set_joint_positions(robot, arm_joints, goal_conf)
        world_from_tool0_L = pp.get_link_pose(robot, tool_link_left)
        world_from_tool0_R = pp.get_link_pose(robot, tool_link_right)
    inv_bar = pp.invert(world_from_bar_goal)
    grasp_bar_from_left = pp.multiply(inv_bar, world_from_tool0_L)
    grasp_bar_from_right = pp.multiply(inv_bar, world_from_tool0_R)
    return grasp_bar_from_left, grasp_bar_from_right


def derive_constrained_start(
    *args,
    **kwargs,
):
    """Compatibility wrapper; implementation lives in dual_arm_task_space_rrt.core."""
    from husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.core import (
        derive_constrained_start as _derive_constrained_start,
    )

    return _derive_constrained_start(*args, **kwargs)


def plan_constrained_dual_arm(
    scene: Dict[str, Any],
    start_conf: Sequence[float],
    goal_conf: Sequence[float],
    *,
    bar_body: int,
    grasp_bar_from_left,
    grasp_bar_from_right,
    feature_points: Optional[Sequence[np.ndarray]],
    world_from_bar_start,
    world_from_bar_goal,
    stage: int = 3,
    position_res: float = 0.01,
    rotation_res: float = 0.025,
    max_time: float = 30.0,           # per-attempt budget; matches plan_pose_rrt default
    max_iterations: int = 2000,
    max_attempts: int = 5,             # matches plan_pose_rrt default
    enable_smoothing: bool = True,
    smooth_max_iterations: int = 100,
    smooth_max_time: float = 10.0,
    joint_continuity_threshold_rad: Optional[float] = None,
    random_seed: Optional[int] = None,
    use_draw: bool = False,
    cfab_session=None,
    cfab_template_state=None,
) -> Tuple[Optional[List[np.ndarray]], dict]:
    """Constrained dual-arm SE(3) RRT with rigid grasp constraint.

    Wraps plan_pose_rrt + smooth_dual_arm_pose_path. Both start_conf and
    goal_conf must already satisfy the rigid grasp constraint (the caller
    derives start_conf via derive_constrained_start).

    stage:
      1 -> pose-only RRT, no IK, no robot collision (path_confs = None)
      2 -> pose RRT + IK in extend, no robot collision
      3 -> pose RRT + IK + joint-space robot collision (full)

    cfab_session / cfab_template_state (optional, stage 3 only): when both
    provided, build the joint collision predicate via cfab's
    PyBulletCheckCollision (5-step CC with SRDF + per-state touch_links)
    instead of the pp.get_collision_fn path. Falls back to pp when either is
    None.
    """
    _validate_scene(scene)
    if len(start_conf) != 12 or len(goal_conf) != 12:
        raise ValueError("start_conf and goal_conf must have length 12")
    if grasp_bar_from_right is None:
        raise ValueError("grasp_bar_from_right is required for dual-arm constrained plan")
    if stage not in (1, 2, 3):
        raise ValueError(f"stage must be 1, 2, or 3; got {stage}")

    from .dual_arm_task_space_rrt.core import (
        DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
        plan_pose_rrt,
        get_joint_collision_fn,
        get_joint_collision_fn_cfab,
    )
    from .dual_arm_task_space_rrt.smooth import smooth_dual_arm_pose_path

    enable_ik = stage >= 2
    enforce_collision = stage >= 3
    if joint_continuity_threshold_rad is None and enable_ik:
        joint_continuity_threshold_rad = DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD
    info: Dict[str, Any] = {
        "stage": stage,
        "max_time": max_time,
        "joint_continuity_threshold_rad": joint_continuity_threshold_rad,
    }

    saved_bar_pose = pp.get_pose(bar_body)
    with pp.WorldSaver():
        joint_collision_fn = None
        if enforce_collision:
            if cfab_session is not None and cfab_template_state is not None:
                joint_collision_fn = get_joint_collision_fn_cfab(
                    cfab_session, cfab_template_state,
                )
                info["joint_collision_backend"] = "cfab"
            else:
                joint_collision_fn = get_joint_collision_fn(
                    robot=scene["robot"],
                    arm_joints=scene["arm_joints"],
                    obstacle_bodies=list(scene["obstacles"]),
                    tool_link_left=scene["tool_link_left"],
                    bar_body=bar_body,
                    grasp_bar_from_left=grasp_bar_from_left,
                )
                info["joint_collision_backend"] = "pp"
        ik_context = None
        if enable_ik:
            ik_context = {
                "robot": scene["robot"],
                "arm_joints": scene["arm_joints"],
                "tool_link_left": scene["tool_link_left"],
                "tool_link_right": scene["tool_link_right"],
                "grasp_bar_from_left": grasp_bar_from_left,
                "grasp_bar_from_right": grasp_bar_from_right,
            }
        planner_profile: Dict[str, Any] = {}
        path_poses, path_confs = plan_pose_rrt(
            robot=scene["robot"],
            bar_body=bar_body,
            obstacle_bodies=list(scene["obstacles"]),
            start_pose=world_from_bar_start,
            goal_pose=world_from_bar_goal,
            start_conf=np.asarray(start_conf, dtype=float),
            goal_conf=np.asarray(goal_conf, dtype=float),
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
            pp.set_pose(bar_body, saved_bar_pose)
            return None, info

        if enable_smoothing and path_confs is not None:
            # smooth_dual_arm_pose_path consumes a `scene` dict (different
            # shape from api SceneContext): it uses keys robot/arm_joints/
            # tool_link_left/tool_link_right/grasp_bar_from_left/
            # grasp_bar_from_right. Build it here from api scene + grasps.
            smooth_scene = {
                "robot": scene["robot"],
                "arm_joints": scene["arm_joints"],
                "tool_link_left": scene["tool_link_left"],
                "tool_link_right": scene["tool_link_right"],
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

    pp.set_pose(bar_body, saved_bar_pose)
    # If we routed CC through cfab, the adapter's last set_robot_cell_state
    # call left cfab.client._robot_cell_state at the final RRT sample. pp's
    # WorldSaver restored body/joint positions in the PyBullet world, but
    # cfab's Python-side cache is independent. Subsequent IK callers
    # (e.g. _plan_M2_dispatch's plan_constrained_dual_arm_linear) read both
    # the pybullet world AND this cache, so leave it pointed at the template
    # we received to avoid stale-state leakage between movements.
    if cfab_session is not None and cfab_template_state is not None:
        try:
            cfab_session.planner.set_robot_cell_state(cfab_template_state)
        except Exception:
            pass
    if path_confs is None:
        # stage 1 intentionally produces no joint path; this is success, not failure.
        # caller should consult info["path_poses"] when stage == 1.
        if stage == 1:
            info["pose_only_success"] = True
            return None, info
        info["failure_reason"] = "no_joint_path"
        return None, info
    return [np.asarray(q, dtype=float) for q in path_confs], info


def _fk_link_frame(planner, state, link_name):
    """Forward kinematics: world frame of `link_name` at the given state.

    Pushes state via planner.set_robot_cell_state(state) then reads the link
    pose from the pybullet client. Returns compas.geometry.Frame.
    """
    import pybullet_planning as pp
    from compas.geometry import Frame
    from compas_fab.robots import RobotCellState  # noqa
    planner.set_robot_cell_state(state)
    client = planner.client
    link_id = client.robot_link_puids[link_name]
    pose = pp.get_link_pose(client.robot_puid, link_id)
    pos, quat = pose
    return Frame.from_quaternion([quat[3], quat[0], quat[1], quat[2]], point=list(pos))


def _run_dual_arm_cartesian_ik_loop(
    planner,
    robot_cell,
    start_state,
    left_frames,
    right_frames,
    *,
    max_results=20,
    max_descend_iterations=200,
    skip_env_collisions=True,
    joint_continuity_threshold_rad=None,
):
    """Solve per-waypoint IK for synchronized dual-arm cartesian motion.

    Uses the previous waypoint's full configuration as the IK seed for the
    next. Returns a single compas_fab JointTrajectory with joint_names =
    LEFT + RIGHT (12 joints) on success, or None on the first IK failure.

    Skips env collisions via _skip_cc3/4/5 when skip_env_collisions=True.
    """
    from copy import deepcopy
    from compas_fab.backends import CollisionCheckError, InverseKinematicsError
    from compas_fab.robots import FrameTarget, JointTrajectory, JointTrajectoryPoint, TargetMode
    from .dual_arm_task_space_rrt.core import (
        DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
        joint_step_exceeds_threshold,
    )
    from compas_fab.backends.pybullet.exceptions import PlanningGroupNotSupported

    def _wrap_angles_in_state(s):
        # PyBullet IK numerically drifts non-target joints. When seed values
        # are far from zero (e.g. wrist_1=4.32 rad from an M1 trajectory
        # endpoint), small deltas can exceed cfab's TOL.is_close and trip
        # _check_configuration_match_group("...not in the group"). Wrap arm
        # joints into [-pi, pi] (same physical pose for revolute joints) so
        # PyBullet's solver has minimal room to drift.
        import math
        for n in s.robot_configuration.joint_names:
            v = float(s.robot_configuration[n])
            if abs(v) > math.pi:
                wrapped = (v + math.pi) % (2 * math.pi) - math.pi
                s.robot_configuration[n] = wrapped
        return s

    assert len(left_frames) == len(right_frames), \
        f"left/right frame lists must be equal length; got {len(left_frames)} vs {len(right_frames)}"

    LEFT_GROUP = "base_left_arm_manipulator"
    RIGHT_GROUP = "base_right_arm_manipulator"
    left_joint_names = list(robot_cell.get_configurable_joint_names(LEFT_GROUP))
    right_joint_names = list(robot_cell.get_configurable_joint_names(RIGHT_GROUP))

    def _is_arm_joint(name):
        return any(name.endswith(suf) for suf in (
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ))
    left_arm_joints = [n for n in left_joint_names if _is_arm_joint(n)]
    right_arm_joints = [n for n in right_joint_names if _is_arm_joint(n)]
    assert len(left_arm_joints) == 6 and len(right_arm_joints) == 6, \
        f"expected 6 arm joints per side, got L={left_arm_joints}, R={right_arm_joints}"
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
    _wrap_angles_in_state(state)
    planner.set_robot_cell_state(state)

    path_12 = []
    N = len(left_frames)
    for i, (lf, rf) in enumerate(zip(left_frames, right_frames)):
        if i == 0:
            # Both callers (plan_constrained_dual_arm_linear,
            # plan_dual_arm_linear_independent) build left_frames[0] /
            # right_frames[0] as FK(start_conf, *_ur_arm_tool0). So the
            # propagated input start_conf is a hard constraint for the
            # first waypoint. Skip IK here so an equivalent wrapped IK
            # branch cannot replace the chain handoff from the previous move.
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
            print(f"[cartesian IK loop] waypoint {i}: LEFT FAIL: {getattr(e, 'message', e)}")
            return None
        except PlanningGroupNotSupported as e:
            print(f"[cartesian IK loop] waypoint {i}: LEFT IK drift; retrying with wrapped seed.")
            _wrap_angles_in_state(state)
            planner.set_robot_cell_state(state)
            try:
                conf_L = planner.inverse_kinematics(left_target, state, LEFT_GROUP, ik_options)
            except Exception as e2:
                print(f"[cartesian IK loop] waypoint {i}: LEFT FAIL after retry: {e2}")
                return None
        # Right-arm joints in conf_L may have drifted from PyBullet IK's
        # numerical solver. Restore them from state (LEFT IK should not have
        # moved RIGHT). Then write the LEFT-arm joints from conf_L.
        for n in right_arm_joints:
            conf_L[n] = float(state.robot_configuration[n])
        state.robot_configuration = conf_L
        try:
            conf_LR = planner.inverse_kinematics(right_target, state, RIGHT_GROUP, ik_options)
        except (InverseKinematicsError, CollisionCheckError) as e:
            print(f"[cartesian IK loop] waypoint {i}: RIGHT FAIL: {getattr(e, 'message', e)}")
            return None
        except PlanningGroupNotSupported as e:
            print(f"[cartesian IK loop] waypoint {i}: RIGHT IK drift; retrying with wrapped seed.")
            _wrap_angles_in_state(state)
            planner.set_robot_cell_state(state)
            try:
                conf_LR = planner.inverse_kinematics(right_target, state, RIGHT_GROUP, ik_options)
            except Exception as e2:
                print(f"[cartesian IK loop] waypoint {i}: RIGHT FAIL after retry: {e2}")
                return None
        for n in left_arm_joints:
            conf_LR[n] = float(state.robot_configuration[n])
        state.robot_configuration = conf_LR
        next_vec = [float(conf_LR[n]) for n in joint_names_12]
        if joint_step_exceeds_threshold(next_vec, path_12[-1], joint_continuity_threshold_rad):
            diff = float(np.abs(np.asarray(next_vec, dtype=float) - np.asarray(path_12[-1], dtype=float)).max())
            print(
                f"[cartesian IK loop] waypoint {i}: joint step {diff:.4f} rad exceeds "
                f"threshold {float(joint_continuity_threshold_rad):.4f} rad; rejecting IK branch."
            )
            return None
        path_12.append(next_vec)

    from husky_assembly_teleop.utils import joint_trajectory_from_path
    return joint_trajectory_from_path(path_12)


def plan_constrained_dual_arm_linear(
    planner,
    robot_cell,
    start_state,
    start_conf,
    goal_world_from_bar,
    bar_from_left_tool0,
    bar_from_right_tool0,
    *,
    max_step_distance=0.005,
    max_step_angle=0.05,
    max_results=20,
    max_descend_iterations=200,
    skip_env_collisions=True,
    joint_continuity_threshold_rad=None,
):
    """Linear dual-arm motion with bar-held inter-EE constraint.

    Interpolate the bar's world frame from start (FK at start_conf) to
    goal_world_from_bar. Derive left/right tool0 targets at each waypoint
    by composing bar_t * bar_from_left_tool0 (and right). IK both arms
    per waypoint with previous conf as seed.

    Mirrors GH_dual_arm_approach_plan.py:43-200 algorithm.
    Returns JointTrajectory(12) or None on IK failure.
    """
    from compas.geometry import Frame, Transformation
    from compas_fab.backends.pybullet.backend_features.pybullet_plan_cartesian_motion import (
        FrameInterpolator,
    )

    if len(start_conf) != 12:
        raise ValueError("start_conf must be length 12")

    state = start_state.copy()
    LEFT_GROUP = "base_left_arm_manipulator"
    RIGHT_GROUP = "base_right_arm_manipulator"
    left_arm_joints = [n for n in robot_cell.get_configurable_joint_names(LEFT_GROUP)
                       if any(n.endswith(s) for s in (
                           "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                           "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"))]
    right_arm_joints = [n for n in robot_cell.get_configurable_joint_names(RIGHT_GROUP)
                        if any(n.endswith(s) for s in (
                            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"))]
    for n, v in zip(left_arm_joints + right_arm_joints, start_conf):
        state.robot_configuration[n] = float(v)
    planner.set_robot_cell_state(state)

    start_left_frame = _fk_link_frame(planner, state, "left_ur_arm_tool0")
    start_right_frame = _fk_link_frame(planner, state, "right_ur_arm_tool0")

    inv_left = bar_from_left_tool0.inverted()
    start_world_from_bar = Frame.from_transformation(
        Transformation.from_frame(start_left_frame) * inv_left
    )

    options = {
        "max_step_distance": max_step_distance,
        "max_step_angle": max_step_angle,
    }
    bar_interp = FrameInterpolator(start_world_from_bar, goal_world_from_bar, options)
    N = max(2, bar_interp.regular_interpolation_steps + 1)

    left_frames = []
    right_frames = []
    for i in range(N):
        t = i / (N - 1) if N > 1 else 0.0
        bar_t = bar_interp.get_interpolated_frame(t)
        bar_t_tf = Transformation.from_frame(bar_t)
        left_frames.append(Frame.from_transformation(bar_t_tf * bar_from_left_tool0))
        right_frames.append(Frame.from_transformation(bar_t_tf * bar_from_right_tool0))

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
    robot_cell,
    start_state,
    start_conf,
    target_left_frame,
    target_right_frame,
    *,
    max_step_distance=0.005,
    max_step_angle=0.05,
    max_results=20,
    max_descend_iterations=200,
    skip_env_collisions=True,
    joint_continuity_threshold_rad=None,
):
    """Linear dual-arm motion with INDEPENDENT EE interpolation (M3 retreat).

    Each arm interpolates from its FK-at-start frame to its target frame
    independently. Synchronized by padding to max waypoint count. IK both
    arms per waypoint with previous conf as seed.

    Mirrors GH_dual_arm_retreat_plan.py:54-200 algorithm.
    Returns JointTrajectory(12) or None on IK failure.
    """
    from compas_fab.backends.pybullet.backend_features.pybullet_plan_cartesian_motion import (
        FrameInterpolator,
    )

    if len(start_conf) != 12:
        raise ValueError("start_conf must be length 12")

    state = start_state.copy()
    LEFT_GROUP = "base_left_arm_manipulator"
    RIGHT_GROUP = "base_right_arm_manipulator"
    left_arm_joints = [n for n in robot_cell.get_configurable_joint_names(LEFT_GROUP)
                       if any(n.endswith(s) for s in (
                           "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                           "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"))]
    right_arm_joints = [n for n in robot_cell.get_configurable_joint_names(RIGHT_GROUP)
                        if any(n.endswith(s) for s in (
                            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"))]
    for n, v in zip(left_arm_joints + right_arm_joints, start_conf):
        state.robot_configuration[n] = float(v)
    planner.set_robot_cell_state(state)

    start_left_frame = _fk_link_frame(planner, state, "left_ur_arm_tool0")
    start_right_frame = _fk_link_frame(planner, state, "right_ur_arm_tool0")

    options = {"max_step_distance": max_step_distance, "max_step_angle": max_step_angle}
    left_interp = FrameInterpolator(start_left_frame, target_left_frame, options)
    right_interp = FrameInterpolator(start_right_frame, target_right_frame, options)
    N = max(2,
            max(left_interp.regular_interpolation_steps,
                right_interp.regular_interpolation_steps) + 1)

    left_frames = []
    right_frames = []
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
