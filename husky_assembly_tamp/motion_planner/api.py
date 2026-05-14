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
) -> Tuple[Optional[List[np.ndarray]], dict]:
    """Constrained dual-arm SE(3) RRT with rigid grasp constraint.

    Wraps plan_pose_rrt + smooth_dual_arm_pose_path. Both start_conf and
    goal_conf must already satisfy the rigid grasp constraint (the caller
    derives start_conf via derive_constrained_start).

    stage:
      1 -> pose-only RRT, no IK, no robot collision (path_confs = None)
      2 -> pose RRT + IK in extend, no robot collision
      3 -> pose RRT + IK + joint-space robot collision (full)
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
            joint_collision_fn = get_joint_collision_fn(
                robot=scene["robot"],
                arm_joints=scene["arm_joints"],
                obstacle_bodies=list(scene["obstacles"]),
                tool_link_left=scene["tool_link_left"],
                bar_body=bar_body,
                grasp_bar_from_left=grasp_bar_from_left,
            )
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
    if path_confs is None:
        # stage 1 intentionally produces no joint path; this is success, not failure.
        # caller should consult info["path_poses"] when stage == 1.
        if stage == 1:
            info["pose_only_success"] = True
            return None, info
        info["failure_reason"] = "no_joint_path"
        return None, info
    return [np.asarray(q, dtype=float) for q in path_confs], info
