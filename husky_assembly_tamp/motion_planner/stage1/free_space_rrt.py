"""Free-space RRT planners for single-arm and dual-arm joint-space planning.

Unlike the constrained pose-space RRT in minimal_rrt.py (which maintains a
rigid bar grasp between both arms), these planners operate directly in joint
space using pybullet_planning's standard motion planning primitives. No bar
is grasped; no relative EE transform is enforced.

Planners
--------
single-arm-free : 6-DOF BiRRT for one arm (left or right).
dual-arm-free   : 12-DOF BiRRT for both arms simultaneously.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet_planning as pp
from compas_fab.robots import RobotSemantics
from compas_robots import RobotModel
from pybullet_planning.motion_planners.rrt_connect import rrt_connect
from pybullet_planning.motion_planners.smoothing import smooth_path
from pybullet_planning.motion_planners.utils import remove_redundant

from husky_assembly_tamp.motion_planner.stage1.minimal_rrt import (
    DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
    DEFAULT_USE_ANGLE_NORMALIZATION,
    HUSKY_DUAL_ARM_JOINT_NAMES,
    HUSKY_DUAL_SRDF_PATH,
    HUSKY_DUAL_URDF_PATH,
    TOOL_LINK_LEFT,
    TOOL_LINK_RIGHT,
    get_disabled_collisions_from_link_names,
    log_validation_summary,
    maybe_normalize_angles,
    setup_planning_scene,
    summarize_joint_continuity,
    teardown_planning_scene,
)
from husky_assembly_tamp.motion_planner.stage1.path_validation import validate_stage_trajectory


logger = logging.getLogger(__name__)

LEFT_ARM_JOINT_NAMES: List[str] = list(HUSKY_DUAL_ARM_JOINT_NAMES[:6])
RIGHT_ARM_JOINT_NAMES: List[str] = list(HUSKY_DUAL_ARM_JOINT_NAMES[6:])

TREE_COLOR_LEFT = (0.2, 0.2, 0.85, 0.45)
TREE_COLOR_RIGHT = (0.2, 0.85, 0.2, 0.45)

DEFAULT_JOINT_RESOLUTION = 0.1


def get_arm_joint_names(active_arm: str) -> List[str]:
    """Return the 6 joint names for the requested arm."""
    if active_arm == "left":
        return list(LEFT_ARM_JOINT_NAMES)
    if active_arm == "right":
        return list(RIGHT_ARM_JOINT_NAMES)
    raise ValueError(f"Unknown active_arm: {active_arm!r}. Expected 'left' or 'right'.")


def get_tool_link_name(active_arm: str) -> str:
    """Return the tool0 link name for the requested arm."""
    if active_arm == "left":
        return TOOL_LINK_LEFT
    if active_arm == "right":
        return TOOL_LINK_RIGHT
    raise ValueError(f"Unknown active_arm: {active_arm!r}")


def make_free_space_draw_fn(
    robot: int,
    arm_joints: Sequence[int],
    tool_links: Sequence[int],
    colors: Sequence[Tuple[float, float, float, float]],
) -> Callable:
    """Create a draw_fn callback that visualizes BiRRT tree edges in task space."""
    assert len(tool_links) == len(colors)

    def _draw(config, segment, valid1=True, valid2=True):
        if len(segment) < 2:
            return
        child_conf, parent_conf = segment[0], segment[1]
        for tool_link, color in zip(tool_links, colors):
            pp.set_joint_positions(robot, arm_joints, child_conf)
            child_pos = pp.get_link_pose(robot, tool_link)[0]
            pp.set_joint_positions(robot, arm_joints, parent_conf)
            parent_pos = pp.get_link_pose(robot, tool_link)[0]
            pp.add_line(parent_pos, child_pos, width=1.5, color=color)

    return _draw


def build_free_space_collision_fn(
    robot: int,
    arm_joints: Sequence[int],
    obstacle_bodies: Sequence[int],
    urdf_path: str = HUSKY_DUAL_URDF_PATH,
    srdf_path: str = HUSKY_DUAL_SRDF_PATH,
) -> Callable:
    """Build a collision function for free-space planning with no attachments."""
    robot_model = RobotModel.from_urdf_file(urdf_path)
    semantics = RobotSemantics.from_srdf_file(srdf_path, robot_model)
    disabled_collisions = get_disabled_collisions_from_link_names(
        robot,
        semantics.disabled_collisions,
    )
    return pp.get_collision_fn(
        robot,
        arm_joints,
        obstacles=list(obstacle_bodies),
        attachments=[],
        self_collisions=True,
        disabled_collisions=disabled_collisions,
        extra_disabled_collisions=[],
        max_distance=0.0,
    )


def plan_free_space_motion(
    robot: int,
    arm_joints: Sequence[int],
    start_conf: Sequence[float],
    goal_conf: Sequence[float],
    collision_fn: Callable,
    *,
    max_time: float = 30.0,
    max_iterations: int = 2000,
    joint_resolution: float = DEFAULT_JOINT_RESOLUTION,
    smooth_iterations: Optional[int] = 100,
    draw_fn: Optional[Callable] = None,
) -> Tuple[Optional[List[tuple]], float]:
    """Plan a free-space joint-space path using BiRRT."""
    n_joints = len(arm_joints)
    resolutions = np.ones(n_joints) * joint_resolution

    sample_fn = pp.get_sample_fn(robot, arm_joints)
    distance_fn = pp.get_distance_fn(robot, arm_joints)
    extend_fn = pp.get_extend_fn(robot, arm_joints, resolutions=resolutions)

    start_tuple = tuple(float(v) for v in start_conf)
    goal_tuple = tuple(float(v) for v in goal_conf)

    if collision_fn(start_tuple):
        logger.warning("Start configuration is in collision.")
        return None, 0.0
    if collision_fn(goal_tuple):
        logger.warning("Goal configuration is in collision.")
        return None, 0.0

    t0 = time.perf_counter()
    if draw_fn is not None:
        # pp.solve_motion_plan checks the direct path before calling BiRRT. For
        # GUI runs, bypass that shortcut so the requested tree is actually drawn.
        path = rrt_connect(
            start_tuple,
            goal_tuple,
            distance_fn,
            sample_fn,
            extend_fn,
            collision_fn,
            max_time=max_time,
            max_iterations=max_iterations,
            draw_fn=draw_fn,
        )
        if path:
            path = remove_redundant(path)
        path = smooth_path(
            path,
            extend_fn,
            collision_fn,
            max_smooth_iterations=smooth_iterations,
            max_time=max(0.0, max_time - (time.perf_counter() - t0)),
        )
    else:
        path = pp.solve_motion_plan(
            start_tuple,
            goal_tuple,
            distance_fn,
            sample_fn,
            extend_fn,
            collision_fn,
            algorithm="birrt",
            max_time=max_time,
            max_iterations=max_iterations,
            smooth=smooth_iterations,
            draw_fn=draw_fn,
        )
    planning_time = time.perf_counter() - t0

    if path is not None:
        logger.info(
            "Free-space BiRRT found path with %d waypoints in %.3f s.",
            len(path),
            planning_time,
        )
    else:
        logger.warning("Free-space BiRRT failed after %.3f s.", planning_time)

    return path, planning_time


def run_free_space_trial(
    *,
    planner_mode: str,
    active_arm: str = "left",
    grasp_json: str,
    start_state_json: str,
    end_state_json: str,
    use_gui: bool = False,
    max_time: float = 30.0,
    max_iterations: int = 2000,
    max_attempts: int = 5,
    joint_resolution: float = DEFAULT_JOINT_RESOLUTION,
    enable_smoothing: bool = True,
    smooth_iterations: int = 100,
    random_seed: Optional[int] = None,
    lock_renderer_during_search: bool = True,
    scene_spec: Optional[Dict[str, Any]] = None,
    validation_reports_dir: Optional[str] = None,
    swap_grasps: bool = False,
    joint_continuity_threshold_rad: float = DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
    use_angle_normalization: bool = DEFAULT_USE_ANGLE_NORMALIZATION,
    enable_collision: bool = True,
    include_built_bars: bool = False,
) -> Dict[str, Any]:
    """Run a free-space joint-space planning trial."""
    del enable_collision, include_built_bars

    scene = setup_planning_scene(
        grasp_json=grasp_json,
        start_state_json=start_state_json,
        end_state_json=end_state_json,
        use_gui=use_gui,
        scene_spec=scene_spec,
        swap_grasps=swap_grasps,
    )
    scene["bar_pose_source"] = "path"

    try:
        robot = scene["robot"]
        all_arm_joints = scene["arm_joints"]
        tool_link_left = scene["tool_link_left"]
        tool_link_right = scene["tool_link_right"]
        start_joint_values = np.asarray(scene["start_joint_values"], dtype=float)
        end_joint_values = np.asarray(scene["end_joint_values"], dtype=float)

        if planner_mode == "single-arm-free":
            joint_names = get_arm_joint_names(active_arm)
            planning_joints = pp.joints_from_names(robot, joint_names)
            if active_arm == "left":
                start_conf = start_joint_values[:6]
                goal_conf = end_joint_values[:6]
            else:
                start_conf = start_joint_values[6:]
                goal_conf = end_joint_values[6:]
        elif planner_mode == "dual-arm-free":
            planning_joints = all_arm_joints
            start_conf = start_joint_values
            goal_conf = end_joint_values
        else:
            raise ValueError(f"Unknown planner_mode: {planner_mode!r}")

        obstacle_bodies = [body for body in scene["collision_obstacles"] if body != robot]
        collision_fn = build_free_space_collision_fn(robot, planning_joints, obstacle_bodies)

        draw_fn = None
        if use_gui:
            if planner_mode == "single-arm-free":
                tool_link = pp.link_from_name(robot, get_tool_link_name(active_arm))
                color = TREE_COLOR_LEFT if active_arm == "left" else TREE_COLOR_RIGHT
                draw_fn = make_free_space_draw_fn(robot, planning_joints, [tool_link], [color])
            elif planner_mode == "dual-arm-free":
                draw_fn = make_free_space_draw_fn(
                    robot,
                    planning_joints,
                    [tool_link_left, tool_link_right],
                    [TREE_COLOR_LEFT, TREE_COLOR_RIGHT],
                )

        path = None
        planning_time_s = 0.0
        t_total_start = time.perf_counter()

        for attempt in range(max_attempts):
            if random_seed is not None:
                np.random.seed(random_seed + attempt)

            effective_smooth = smooth_iterations if enable_smoothing else None
            planning_kwargs = dict(
                robot=robot,
                arm_joints=planning_joints,
                start_conf=start_conf,
                goal_conf=goal_conf,
                collision_fn=collision_fn,
                max_time=max_time,
                max_iterations=max_iterations,
                joint_resolution=joint_resolution,
                smooth_iterations=effective_smooth,
                draw_fn=draw_fn,
            )
            if use_gui and lock_renderer_during_search:
                with pp.LockRenderer():
                    attempt_path, attempt_time = plan_free_space_motion(**planning_kwargs)
            else:
                attempt_path, attempt_time = plan_free_space_motion(**planning_kwargs)

            planning_time_s += attempt_time
            if attempt_path is not None:
                path = attempt_path
                logger.info("Free-space planning succeeded on attempt %d/%d.", attempt + 1, max_attempts)
                break
            logger.info("Attempt %d/%d failed, retrying...", attempt + 1, max_attempts)

        path_confs = None
        if path is not None:
            if planner_mode == "single-arm-free":
                full_confs = []
                for conf in path:
                    full = np.array(start_joint_values, dtype=float)
                    if active_arm == "left":
                        full[:6] = conf
                    else:
                        full[6:] = conf
                    full_confs.append(full)
                path_confs = full_confs
            else:
                path_confs = [np.asarray(conf, dtype=float) for conf in path]

        pose_path = None
        if path_confs is not None:
            fixed_bar_pose = scene["world_from_bar_start"]
            pose_path = [fixed_bar_pose for _ in path_confs]
            pp.set_joint_positions(robot, all_arm_joints, path_confs[-1])
            pp.set_pose(scene["bar_body"], fixed_bar_pose)

        coarse_continuity = None
        if path_confs is not None:
            coarse_continuity = summarize_joint_continuity(
                path_confs,
                threshold_rad=joint_continuity_threshold_rad,
                use_angle_normalization=use_angle_normalization,
            )

        validation_joint_path = None
        validation_joint_path_source = None
        validation_joint_path_reason = "planner_joint_path_unavailable"
        if path_confs is not None:
            validation_joint_path = [
                maybe_normalize_angles(conf, use_angle_normalization)
                for conf in path_confs
            ]
            validation_joint_path_source = "planner"
            validation_joint_path_reason = None

        validation_kwargs = dict(
            stage=3,
            scene=scene,
            path=pose_path,
            joint_path=validation_joint_path,
            original_joint_path=None,
            joint_path_source=validation_joint_path_source,
            joint_path_reason=validation_joint_path_reason,
            urdf_path=HUSKY_DUAL_URDF_PATH,
            srdf_path=HUSKY_DUAL_SRDF_PATH,
            grasp_mask_links=[],
            target_label=f"free-space-{planner_mode}",
            use_angle_normalization=use_angle_normalization,
            skip_relative_transform=True,
            bar_pose_source="path",
        )
        if validation_reports_dir is not None:
            validation_kwargs["reports_dir"] = validation_reports_dir

        t_validation = time.perf_counter()
        if use_gui:
            with pp.LockRenderer():
                validation = validate_stage_trajectory(**validation_kwargs)
        else:
            validation = validate_stage_trajectory(**validation_kwargs)
        validation_time_s = time.perf_counter() - t_validation
        log_validation_summary(validation)

        path_found = path is not None
        validated_success = path_found
        if path_found:
            validated_success = (
                validated_success
                and bool(validation.get("joint_continuity_ok"))
                and bool(validation.get("collision_free"))
            )
        runtime_s = time.perf_counter() - t_total_start

        return {
            "stage": 0,
            "planner_mode": planner_mode,
            "active_arm": active_arm,
            "scene": scene,
            "path": pose_path,
            "path_confs": path_confs,
            "path_before_smoothing": None,
            "path_confs_before_smoothing": None,
            "joint_continuity": coarse_continuity,
            "validation_joint_path": validation_joint_path,
            "validation_joint_path_source": validation_joint_path_source,
            "validation": validation,
            "start_conf": np.asarray(start_joint_values, dtype=float),
            "goal_conf": np.asarray(end_joint_values, dtype=float),
            "planning_time_s": planning_time_s,
            "smoothing_time_s": 0.0,
            "validation_time_s": validation_time_s,
            "runtime_s": runtime_s,
            "path_found": path_found,
            "success": bool(validated_success),
            "smoothing": None,
        }
    except Exception:
        teardown_planning_scene()
        raise
