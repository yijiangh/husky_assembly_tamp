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


def _grid_in_box(
    box: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    step: float,
) -> List[Tuple[float, float, float]]:
    (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = box
    xs = np.arange(x_lo, x_hi + 0.5 * step, step)
    ys = np.arange(y_lo, y_hi + 0.5 * step, step)
    zs = np.arange(z_lo, z_hi + 0.5 * step, step)
    return [(float(x), float(y), float(z)) for x in xs for y in ys for z in zs]


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
    robot: int,
    arm_joints: Sequence[int],
    tool_link_left: int,
    tool_link_right: int,
    grasp_bar_from_left,
    grasp_bar_from_right,
    world_from_bar_goal,
    seed_conf: Sequence[float],
    *,
    bar_body: Optional[int] = None,
    obstacles: Sequence[int] = (),
    world_from_mobile_base=None,
    bar_sweep_box: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (-0.3, 0.3),
        (-0.3, 0.3),
        (-0.3, 0.3),
    ),
    bar_sweep_step: float = 0.1,
    num_geometric_candidates: int = 20,
    bar_axis_step_rad: float = float(np.deg2rad(30.0)),
    max_ik_attempts: int = 20,
    random_seed: Optional[int] = None,
) -> Tuple[Optional[Tuple], Optional[np.ndarray]]:
    """Derive a constraint-satisfying start (bar_pose, joint_conf).

    Fixed-bar strategy: anchor the bar pose in the mobile-base frame at
    `MOBILE_BASE_FROM_BAR_HOME_POSITION` (shifted by the grasp midpoint so
    the bar's geometric center sits at the home), with orientation derived
    from the grasps via `bar_orientation_from_grasps`. Sweep
    `(dx, dy, dz)` closest-first; for each candidate position run
    `auto_compute_home_bar_pose` (which spins the bar around its long axis
    and validates IK + collisions). When `bar_body` is provided, the
    validator is collision-aware against `obstacles`; otherwise it's
    kinematic-only.

    Returns (world_from_bar_start, start_conf). Both None if no
    collision-free home pose was found.
    """
    from .stage1.minimal_rrt import (
        MOBILE_BASE_FROM_BAR_HOME_POSITION,
        auto_compute_home_bar_pose,
        bar_orientation_from_grasps,
        get_joint_collision_fn,
        solve_endpoint_dual_arm_ik,
    )

    if len(seed_conf) != 12:
        raise ValueError("seed_conf must have length 12")

    identity_pose = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    if world_from_mobile_base is None:
        world_from_mobile_base = identity_pose
    mobile_base_from_world = pp.invert(world_from_mobile_base)

    mb_from_bar_goal = pp.multiply(mobile_base_from_world, world_from_bar_goal)
    mb_from_tool0_L_goal = pp.multiply(mb_from_bar_goal, grasp_bar_from_left)
    mb_from_tool0_R_goal = pp.multiply(mb_from_bar_goal, grasp_bar_from_right)
    grasp_targets_mb = [
        (mb_from_bar_goal, mb_from_tool0_L_goal),
        (mb_from_bar_goal, mb_from_tool0_R_goal),
    ]

    home_bar_quat = bar_orientation_from_grasps(grasp_targets_mb)

    # Anchor the geometric MIDPOINT between the two grasps at the home position
    # rather than the bar-frame origin (which sits at one grasp end for typical
    # datasets); without this shift the bar dangles ~1m off one arm.
    bar_from_tool0_left_local = pp.multiply(pp.invert(mb_from_bar_goal), mb_from_tool0_L_goal)
    bar_from_tool0_right_local = pp.multiply(pp.invert(mb_from_bar_goal), mb_from_tool0_R_goal)
    grasp_midpoint_in_bar = 0.5 * (
        np.asarray(bar_from_tool0_left_local[0], dtype=float)
        + np.asarray(bar_from_tool0_right_local[0], dtype=float)
    )
    midpoint_in_mb = np.asarray(
        pp.multiply(
            ((0.0, 0.0, 0.0), home_bar_quat),
            (tuple(grasp_midpoint_in_bar.tolist()), (0.0, 0.0, 0.0, 1.0)),
        )[0],
        dtype=float,
    )
    base_pos_mb = np.asarray(MOBILE_BASE_FROM_BAR_HOME_POSITION, dtype=float) - midpoint_in_mb

    rng = np.random.default_rng(random_seed)

    joint_collision_fn = None
    if bar_body is not None:
        joint_collision_fn = get_joint_collision_fn(
            robot=robot,
            arm_joints=arm_joints,
            obstacle_bodies=list(obstacles),
            tool_link_left=tool_link_left,
            bar_body=bar_body,
            grasp_bar_from_left=grasp_bar_from_left,
        )

    deltas = sorted(
        _grid_in_box(bar_sweep_box, bar_sweep_step),
        key=lambda d: float(np.linalg.norm(d)),
    )

    saved_bar_pose = pp.get_pose(bar_body) if bar_body is not None else None
    found: Dict[str, Any] = {"world_from_bar": None, "conf": None}
    chosen_ctx: Optional[Dict[str, Any]] = None

    with pp.WorldSaver():
        for d in deltas:
            mb_from_bar_candidate = (
                tuple((base_pos_mb + np.asarray(d, dtype=float)).tolist()),
                home_bar_quat,
            )

            def ik_validator(bar_pose_mb, _found=found):
                world_from_bar = pp.multiply(world_from_mobile_base, bar_pose_mb)
                conf = solve_endpoint_dual_arm_ik(
                    robot=robot,
                    arm_joints=arm_joints,
                    tool_link_left=tool_link_left,
                    tool_link_right=tool_link_right,
                    bar_pose=world_from_bar,
                    grasp_bar_from_left=grasp_bar_from_left,
                    grasp_bar_from_right=grasp_bar_from_right,
                    seed_conf=np.asarray(seed_conf, dtype=float),
                    rng=rng,
                    max_attempts=max_ik_attempts,
                    collision_fn=joint_collision_fn,
                )
                if conf is None:
                    return False
                _found["world_from_bar"] = world_from_bar
                _found["conf"] = conf
                return True

            ctx = auto_compute_home_bar_pose(
                grasp_targets_mb,
                mobile_base_from_bar=mb_from_bar_candidate,
                ik_validator=ik_validator,
                num_geometric_candidates=num_geometric_candidates,
                bar_axis_step_rad=bar_axis_step_rad,
                allow_unvalidated_fallback=False,
            )
            if ctx.get("ik_validated", False):
                chosen_ctx = ctx
                break

    if saved_bar_pose is not None:
        pp.set_pose(bar_body, saved_bar_pose)

    if chosen_ctx is None or found["conf"] is None:
        logger.warning(
            "derive_constrained_start: no collision-free home pose across %d deltas (kinematic_only=%s)",
            len(deltas),
            bar_body is None,
        )
        return None, None

    world_from_bar_start = pp.multiply(world_from_mobile_base, chosen_ctx["mobile_base_from_bar_start"])
    return world_from_bar_start, found["conf"]


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

    from .stage1.minimal_rrt import (
        plan_pose_rrt,
        smooth_dual_arm_pose_path,
        get_joint_collision_fn,
    )

    enable_ik = stage >= 2
    enforce_collision = stage >= 3
    info: Dict[str, Any] = {"stage": stage, "max_time": max_time}

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
