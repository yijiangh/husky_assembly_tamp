# Codex Task: Add Free-Space RRT Planners

## Objective

Add two new joint-space BiRRT planners to the existing dual-arm constrained motion planning codebase. These use `pybullet_planning`'s standard `pp.solve_motion_plan` instead of the existing custom pose-space RRT. The two new modes are:

- **`single-arm-free`**: 6-DOF joint-space BiRRT for one arm (left or right)
- **`dual-arm-free`**: 12-DOF joint-space BiRRT for both arms in composite space

The existing constrained planner must remain untouched and be the default.

---

## Repository Layout

Working directory: `/Users/huangyijiang/Code/husky-assembly-teleop/external/husky_assembly_tamp`

Key files (all paths relative to `husky_assembly_tamp/motion_planner/stage1/`):

| File | Role |
|------|------|
| `minimal_rrt.py` | Core constrained RRT planner + CLI entry point |
| `real_state_study.py` | Batch orchestrator with reporting, video, replay |
| `path_validation.py` | Trajectory validation (collision, continuity, EE drift) |
| `debug_runner.py` | Analysis runner with markdown reports + tree plots |

External reference (read-only, do NOT modify):
- `pybullet_planning` library at `/Users/huangyijiang/Code/husky-assembly-teleop/external/pybullet_planning/`

---

## Deliverables — 4 File Changes

### File 1: CREATE `free_space_rrt.py`

**Path**: `husky_assembly_tamp/motion_planner/stage1/free_space_rrt.py`

This is the bulk of the work (~300 lines). It contains all new planner logic.

#### Imports and Constants

```python
"""Free-space RRT planners for single-arm and dual-arm joint-space planning.

Unlike the constrained pose-space RRT in minimal_rrt.py (which maintains a
rigid bar grasp between both arms), these planners operate directly in joint
space using pybullet_planning's standard motion planning primitives.  No bar
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

from husky_assembly_tamp.motion_planner.stage1.minimal_rrt import (
    HUSKY_DUAL_ARM_JOINT_NAMES,
    HUSKY_DUAL_SRDF_PATH,
    HUSKY_DUAL_URDF_PATH,
    STAGE3_GRASP_MASK_LINKS,
    TOOL_LINK_LEFT,
    TOOL_LINK_RIGHT,
    DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
    DEFAULT_USE_ANGLE_NORMALIZATION,
    get_disabled_collisions_from_link_names,
    log_validation_summary,
    maybe_normalize_angles,
    run_visualization_loop,
    setup_planning_scene,
    summarize_joint_continuity,
    teardown_planning_scene,
)
from husky_assembly_tamp.motion_planner.stage1.path_validation import (
    validate_stage_trajectory,
)

logger = logging.getLogger(__name__)

# Arm joint name subsets (first 6 = left, last 6 = right)
LEFT_ARM_JOINT_NAMES: List[str] = list(HUSKY_DUAL_ARM_JOINT_NAMES[:6])
RIGHT_ARM_JOINT_NAMES: List[str] = list(HUSKY_DUAL_ARM_JOINT_NAMES[6:])

# Tree-drawing colours (RGBA) — distinct from the constrained RRT's red
TREE_COLOR_LEFT = (0.2, 0.2, 0.85, 0.45)   # blue
TREE_COLOR_RIGHT = (0.2, 0.85, 0.2, 0.45)  # green

DEFAULT_JOINT_RESOLUTION = 0.05  # radians, per-joint step for extend_fn
```

**IMPORTANT**: Verify that every name imported from `minimal_rrt` actually exists. In particular:
- `HUSKY_DUAL_ARM_JOINT_NAMES` is defined at line 49
- `HUSKY_DUAL_URDF_PATH` — search for this constant near the top of minimal_rrt.py
- `HUSKY_DUAL_SRDF_PATH` — same
- `TOOL_LINK_LEFT` at line 63, `TOOL_LINK_RIGHT` at line 64
- `STAGE3_GRASP_MASK_LINKS` at line 65
- `DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD` — search for it
- `DEFAULT_USE_ANGLE_NORMALIZATION` — search for it
- `get_disabled_collisions_from_link_names` at line 611
- `log_validation_summary` — search for `def log_validation_summary`; if it doesn't exist, write a small local helper that logs validation results
- `maybe_normalize_angles` at line 159
- `run_visualization_loop` at line 2013
- `setup_planning_scene` at line 1473
- `summarize_joint_continuity` at line 939
- `teardown_planning_scene` at line 1614

If any import fails, find the actual name/location in minimal_rrt.py and adjust.

#### Function: `get_arm_joint_names`

```python
def get_arm_joint_names(active_arm: str) -> List[str]:
    """Return the 6 joint names for the requested arm."""
    if active_arm == "left":
        return list(LEFT_ARM_JOINT_NAMES)
    elif active_arm == "right":
        return list(RIGHT_ARM_JOINT_NAMES)
    raise ValueError(f"Unknown active_arm: {active_arm!r}. Expected 'left' or 'right'.")
```

#### Function: `get_tool_link_name`

```python
def get_tool_link_name(active_arm: str) -> str:
    """Return the tool0 link name for the requested arm."""
    if active_arm == "left":
        return TOOL_LINK_LEFT
    elif active_arm == "right":
        return TOOL_LINK_RIGHT
    raise ValueError(f"Unknown active_arm: {active_arm!r}")
```

#### Function: `make_free_space_draw_fn`

This creates the callback that `pp.solve_motion_plan` calls via the BiRRT algorithm to visualize tree growth.

**How the callback is invoked**: Inside pybullet_planning's `rrt_connect.py`, `TreeNode.draw(draw_fn)` calls:
```python
draw_fn(self.config, [self.config, self.parent.config], True, True)
```
So `segment[0]` = child config, `segment[1]` = parent config. For the root node, `segment = []`.

```python
def make_free_space_draw_fn(
    robot: int,
    arm_joints: Sequence[int],
    tool_links: Sequence[int],
    colors: Sequence[Tuple[float, float, float, float]],
) -> Callable:
    """Create a draw_fn callback that visualises BiRRT tree edges in task space.

    For each edge, sets joint positions, does FK on each tool_link, and draws
    a line between the parent and child FK positions using pp.add_line.

    Parameters
    ----------
    robot : int
        PyBullet body id.
    arm_joints : Sequence[int]
        Joint indices being planned over (6 or 12).
    tool_links : Sequence[int]
        One or two PyBullet link indices to visualise.
    colors : Sequence[tuple]
        One RGBA colour per tool_link.
    """
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
```

**Safety note**: The planner works purely with configuration tuples and never reads PyBullet joint state. Setting joint positions for FK during drawing is safe. After planning, the caller restores joints to the final configuration.

#### Function: `build_free_space_collision_fn`

```python
def build_free_space_collision_fn(
    robot: int,
    arm_joints: Sequence[int],
    obstacle_bodies: Sequence[int],
    urdf_path: str = HUSKY_DUAL_URDF_PATH,
    srdf_path: str = HUSKY_DUAL_SRDF_PATH,
) -> Callable:
    """Build a collision function for free-space planning (no attachments).

    Uses SRDF disabled collisions for self-collision pruning. No bar
    attachment since the arms are not grasping anything.
    """
    robot_model = RobotModel.from_urdf_file(urdf_path)
    semantics = RobotSemantics.from_srdf_file(srdf_path, robot_model)
    disabled_collisions = get_disabled_collisions_from_link_names(
        robot, semantics.disabled_collisions
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
```

#### Function: `plan_free_space_motion`

```python
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
    """Plan a free-space joint-space path using BiRRT.

    Parameters
    ----------
    robot : int
        PyBullet body id.
    arm_joints : Sequence[int]
        Joint indices to plan over (6 for single-arm, 12 for dual-arm).
    start_conf, goal_conf : Sequence[float]
        Start and goal joint configurations.
    collision_fn : Callable
        collision_fn(config) -> bool (True if in collision).
    max_time : float
        Wall-clock time limit in seconds.
    max_iterations : int
        Maximum BiRRT iterations.
    joint_resolution : float
        Per-joint extension step size in radians.
    smooth_iterations : Optional[int]
        Shortcut smoothing iterations. None disables smoothing.
    draw_fn : Optional[Callable]
        Visualisation callback for tree drawing.

    Returns
    -------
    (path, planning_time_s)
        path is a list of config tuples, or None on failure.
    """
    n_joints = len(arm_joints)
    resolutions = np.ones(n_joints) * joint_resolution

    sample_fn = pp.get_sample_fn(robot, arm_joints)
    distance_fn = pp.get_distance_fn(robot, arm_joints)
    extend_fn = pp.get_extend_fn(robot, arm_joints, resolutions=resolutions)

    start_tuple = tuple(float(v) for v in start_conf)
    goal_tuple = tuple(float(v) for v in goal_conf)

    # Validate start and goal are collision-free
    if collision_fn(start_tuple):
        logger.warning("Start configuration is in collision.")
        return None, 0.0
    if collision_fn(goal_tuple):
        logger.warning("Goal configuration is in collision.")
        return None, 0.0

    t0 = time.perf_counter()
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
            len(path), planning_time,
        )
    else:
        logger.warning("Free-space BiRRT failed after %.3f s.", planning_time)

    return path, planning_time
```

**Note on `smooth` kwarg**: `pp.solve_motion_plan` passes `smooth=N` to its internal `smooth_path()`. Passing `smooth=None` disables smoothing entirely (smooth_path returns early when `max_smooth_iterations is None`). Passing `smooth=0` runs zero iterations. Use `smooth=None` to disable.

#### Function: `run_free_space_trial`

This is the main entry point. It MUST return a dict with the SAME keys as `run_stage_trial()` in minimal_rrt.py (lines 1929-1949) so all downstream code (reporting, video, replay, visualization) works unchanged.

```python
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
    """Run a free-space joint-space planning trial.

    Returns a dict with the same keys as run_stage_trial() so downstream
    reporting, video capture, and replay code works unchanged.
    """
```

**Step-by-step implementation logic:**

**Step 1 — Setup scene**:
```python
scene = setup_planning_scene(
    grasp_json=grasp_json,
    start_state_json=start_state_json,
    end_state_json=end_state_json,
    use_gui=use_gui,
    scene_spec=scene_spec,
    swap_grasps=swap_grasps,
)
robot = scene["robot"]
all_arm_joints = scene["arm_joints"]  # all 12 joint indices
tool_link_left = scene["tool_link_left"]
tool_link_right = scene["tool_link_right"]
start_joint_values = np.asarray(scene["start_joint_values"], dtype=float)  # 12-DOF
end_joint_values = np.asarray(scene["end_joint_values"], dtype=float)      # 12-DOF
```

The `setup_planning_scene` function (minimal_rrt.py:1473) returns a dict with keys documented at lines 1574-1611. It creates the PyBullet world, loads the robot URDF, bar mesh, ghost markers, etc.

**Step 2 — Determine planning joints and configs**:
```python
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
```

**Step 3 — Build collision function**:
```python
# scene["collision_obstacles"] (line 1572) includes the robot body but
# excludes bar_body, ghosts, and grasp markers. Filter out the robot
# because pp.get_collision_fn handles self-collision internally.
obstacle_bodies = [b for b in scene["collision_obstacles"] if b != robot]

collision_fn = build_free_space_collision_fn(
    robot, planning_joints, obstacle_bodies
)
```

**Step 4 — Build draw_fn (when GUI enabled)**:
```python
draw_fn = None
if use_gui:
    if planner_mode == "single-arm-free":
        tool_link = pp.link_from_name(robot, get_tool_link_name(active_arm))
        color = TREE_COLOR_LEFT if active_arm == "left" else TREE_COLOR_RIGHT
        draw_fn = make_free_space_draw_fn(robot, planning_joints, [tool_link], [color])
    elif planner_mode == "dual-arm-free":
        draw_fn = make_free_space_draw_fn(
            robot, planning_joints,
            [tool_link_left, tool_link_right],
            [TREE_COLOR_LEFT, TREE_COLOR_RIGHT],
        )
```

**Step 5 — Plan with restarts**:
```python
path = None
planning_time_s = 0.0
t_total_start = time.perf_counter()

for attempt in range(max_attempts):
    if random_seed is not None:
        np.random.seed(random_seed + attempt)

    effective_smooth = smooth_iterations if enable_smoothing else None

    if use_gui and lock_renderer_during_search:
        with pp.LockRenderer():
            attempt_path, attempt_time = plan_free_space_motion(
                robot, planning_joints, start_conf, goal_conf, collision_fn,
                max_time=max_time,
                max_iterations=max_iterations,
                joint_resolution=joint_resolution,
                smooth_iterations=effective_smooth,
                draw_fn=draw_fn,
            )
    else:
        attempt_path, attempt_time = plan_free_space_motion(
            robot, planning_joints, start_conf, goal_conf, collision_fn,
            max_time=max_time,
            max_iterations=max_iterations,
            joint_resolution=joint_resolution,
            smooth_iterations=effective_smooth,
            draw_fn=draw_fn,
        )

    planning_time_s += attempt_time
    if attempt_path is not None:
        path = attempt_path
        logger.info("Free-space planning succeeded on attempt %d/%d.", attempt + 1, max_attempts)
        break
    logger.info("Attempt %d/%d failed, retrying...", attempt + 1, max_attempts)
```

**Step 6 — Expand to full 12-DOF path**:
```python
path_confs = None
if path is not None:
    if planner_mode == "single-arm-free":
        # Pad each 6-DOF waypoint to 12-DOF with inactive arm held at start
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
        # dual-arm-free: already 12-DOF
        path_confs = [np.asarray(conf, dtype=float) for conf in path]
```

**Step 7 — Compute pose path (for visualization/replay)**:
```python
pose_path = None
if path_confs is not None:
    pose_path = []
    grasp_bar_from_left = scene["grasp_bar_from_left"]
    for conf in path_confs:
        pp.set_joint_positions(robot, all_arm_joints, conf)
        world_from_left = pp.get_link_pose(robot, tool_link_left)
        bar_pose = pp.multiply(world_from_left, pp.invert(grasp_bar_from_left))
        pose_path.append(bar_pose)
    # Restore final state
    pp.set_joint_positions(robot, all_arm_joints, path_confs[-1])
    pp.set_pose(scene["bar_body"], pose_path[-1])
```

**Step 8 — Coarse joint continuity**:
```python
coarse_continuity = None
if path_confs is not None:
    coarse_continuity = summarize_joint_continuity(
        path_confs, use_angle_normalization=use_angle_normalization
    )
```

**Step 9 — Validation**:
```python
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
    grasp_mask_links=[],  # no grasp mask in free-space mode
    target_label=f"free-space-{planner_mode}",
    use_angle_normalization=use_angle_normalization,
    skip_relative_transform=True,
)
if validation_reports_dir is not None:
    validation_kwargs["reports_dir"] = validation_reports_dir

t_validation = time.perf_counter()
validation = validate_stage_trajectory(**validation_kwargs)
validation_time_s = time.perf_counter() - t_validation
log_validation_summary(validation)
```

**Step 10 — Determine success**:
```python
path_found = path is not None
validated_success = path_found
if path_found:
    validated_success = (
        validated_success
        and bool(validation.get("joint_continuity_ok"))
        and bool(validation.get("collision_free"))
    )
runtime_s = time.perf_counter() - t_total_start
```

**Step 11 — Return dict** (MUST match `run_stage_trial` at minimal_rrt.py lines 1929-1949):
```python
return {
    "stage": 0,  # sentinel for free-space
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
```

---

### File 2: MODIFY `path_validation.py`

**Path**: `husky_assembly_tamp/motion_planner/stage1/path_validation.py`

Three surgical edits. Do NOT change any other code.

#### Edit 2a — Add parameter (line 421)

Find the `validate_stage_trajectory` function signature (line 401). Add `skip_relative_transform: bool = False` as the last keyword-only parameter, right before the closing `):` of the signature. Place it after the `reports_dir` parameter.

Before:
```python
    reports_dir: str = REPORTS_DIR,
) -> Dict[str, Any]:
```
After:
```python
    reports_dir: str = REPORTS_DIR,
    skip_relative_transform: bool = False,
) -> Dict[str, Any]:
```

#### Edit 2b — Guard coarse relative transform (line 568)

Find this exact code block:
```python
    coarse_relative_translation_errors_m, coarse_relative_rotation_axis_errors_deg = compute_relative_transform_drift(
        robot,
        arm_joints,
        tool_link_left,
        tool_link_right,
        normalized_joint_path,
    )
```

Replace with:
```python
    if skip_relative_transform:
        coarse_relative_translation_errors_m = []
        coarse_relative_rotation_axis_errors_deg = {axis_name: [] for axis_name in AXIS_NAMES}
    else:
        coarse_relative_translation_errors_m, coarse_relative_rotation_axis_errors_deg = compute_relative_transform_drift(
            robot,
            arm_joints,
            tool_link_left,
            tool_link_right,
            normalized_joint_path,
        )
```

#### Edit 2c — Guard dense-loop relative transform (lines 624-631)

Find these exact lines inside the `for idx, (conf, sample_meta) in enumerate(zip(dense_joint_path, dense_metadata)):` loop:
```python
        world_from_right = pp.get_link_pose(robot, tool_link_right)
        relative_pose = pp.multiply(pp.invert(world_from_left), world_from_right)
        if base_relative_pose is None:
            base_relative_pose = relative_pose
        relative_translation_errors_m.append(float(np.linalg.norm(np.asarray(relative_pose[0]) - np.asarray(base_relative_pose[0]))))
        axis_diffs_deg = rotation_axis_differences_deg(base_relative_pose[1], relative_pose[1])
        for axis_name in AXIS_NAMES:
            relative_rotation_axis_errors_deg[axis_name].append(axis_diffs_deg[axis_name])
```

Wrap in a conditional:
```python
        if not skip_relative_transform:
            world_from_right = pp.get_link_pose(robot, tool_link_right)
            relative_pose = pp.multiply(pp.invert(world_from_left), world_from_right)
            if base_relative_pose is None:
                base_relative_pose = relative_pose
            relative_translation_errors_m.append(float(np.linalg.norm(np.asarray(relative_pose[0]) - np.asarray(base_relative_pose[0]))))
            axis_diffs_deg = rotation_axis_differences_deg(base_relative_pose[1], relative_pose[1])
            for axis_name in AXIS_NAMES:
                relative_rotation_axis_errors_deg[axis_name].append(axis_diffs_deg[axis_name])
```

**Downstream safety**: Lines 661-673 already have `if relative_translation_errors_m and relative_rotation_axis_errors_deg["x"]:` guard — empty lists are handled gracefully. The validation plot will show empty subplots for EE drift, which is correct.

---

### File 3: MODIFY `minimal_rrt.py`

**Path**: `husky_assembly_tamp/motion_planner/stage1/minimal_rrt.py`

Two edits to the `main()` function (starts at line 2159).

#### Edit 3a — Add argparse flags (insert between lines 2212 and 2213)

Find these consecutive lines:
```python
    parser.add_argument(
        "--lock-renderer-during-search",
        action="store_true",
        help="Lock the PyBullet renderer while the tree is being expanded, then show the result afterward",
    )
    parser.set_defaults(floating_collision=False, lock_renderer_during_search=True)
```

Insert the two new arguments BETWEEN the `--lock-renderer-during-search` argument and the `parser.set_defaults` call:

```python
    parser.add_argument(
        "--planner",
        choices=["dual-arm-constrained", "single-arm-free", "dual-arm-free"],
        default="dual-arm-constrained",
        help=(
            "Planning mode: dual-arm-constrained (default, pose-space RRT "
            "maintaining bar grasp), single-arm-free (6-DOF joint-space BiRRT "
            "for one arm), dual-arm-free (12-DOF joint-space BiRRT for both arms)"
        ),
    )
    parser.add_argument(
        "--active-arm",
        choices=["left", "right"],
        default="left",
        help="Which arm to plan for in single-arm-free mode (default: left)",
    )
```

#### Edit 3b — Add dispatch (replace lines 2217-2243)

Find these lines:
```python
    use_gui = not args.no_gui
    debug_tree_out: Dict = {}
    result = run_stage_trial(
        stage=args.stage,
        grasp_json=args.grasp_json,
        start_state_json=args.start_state,
        end_state_json=args.end_state,
        use_gui=use_gui,
        dist_metric=args.dist_metric,
        goal_bias=args.goal_bias,
        position_res=args.position_res,
        rotation_res=args.rotation_res,
        max_time=args.max_time,
        max_iterations=args.max_iterations,
        max_attempts=args.max_attempts,
        endpoint_ik_attempts=args.endpoint_ik_attempts,
        random_seed=args.random_seed,
        enable_collision=args.floating_collision,
        enable_smoothing=args.smoothing,
        smooth_max_iterations=args.smooth_iterations,
        smooth_max_time=args.smooth_max_time,
        smooth_min_cost_improvement=args.smooth_min_improvement,
        joint_continuity_threshold_rad=args.joint_continuity_threshold,
        use_angle_normalization=args.use_angle_normalization,
        lock_renderer_during_search=args.lock_renderer_during_search,
        swap_grasps=args.swap_grasps,
        debug_tree_out=debug_tree_out,
    )
```

Replace with:
```python
    use_gui = not args.no_gui

    if args.planner == "dual-arm-constrained":
        debug_tree_out: Dict = {}
        result = run_stage_trial(
            stage=args.stage,
            grasp_json=args.grasp_json,
            start_state_json=args.start_state,
            end_state_json=args.end_state,
            use_gui=use_gui,
            dist_metric=args.dist_metric,
            goal_bias=args.goal_bias,
            position_res=args.position_res,
            rotation_res=args.rotation_res,
            max_time=args.max_time,
            max_iterations=args.max_iterations,
            max_attempts=args.max_attempts,
            endpoint_ik_attempts=args.endpoint_ik_attempts,
            random_seed=args.random_seed,
            enable_collision=args.floating_collision,
            enable_smoothing=args.smoothing,
            smooth_max_iterations=args.smooth_iterations,
            smooth_max_time=args.smooth_max_time,
            smooth_min_cost_improvement=args.smooth_min_improvement,
            joint_continuity_threshold_rad=args.joint_continuity_threshold,
            use_angle_normalization=args.use_angle_normalization,
            lock_renderer_during_search=args.lock_renderer_during_search,
            swap_grasps=args.swap_grasps,
            debug_tree_out=debug_tree_out,
        )
    else:
        from husky_assembly_tamp.motion_planner.stage1.free_space_rrt import run_free_space_trial
        result = run_free_space_trial(
            planner_mode=args.planner,
            active_arm=args.active_arm,
            grasp_json=args.grasp_json,
            start_state_json=args.start_state,
            end_state_json=args.end_state,
            use_gui=use_gui,
            max_time=args.max_time,
            max_iterations=args.max_iterations,
            max_attempts=args.max_attempts,
            joint_resolution=args.position_res,
            enable_smoothing=args.smoothing,
            smooth_iterations=args.smooth_iterations,
            random_seed=args.random_seed,
            lock_renderer_during_search=args.lock_renderer_during_search,
            swap_grasps=args.swap_grasps,
            joint_continuity_threshold_rad=args.joint_continuity_threshold,
            use_angle_normalization=args.use_angle_normalization,
            enable_collision=args.floating_collision,
        )
```

The code AFTER this block (visualization loop at lines 2245-2255 and teardown at line 2257) reads `result["scene"]`, `result["path"]`, `result["path_confs"]` — all present in both paths. Leave that code UNTOUCHED.

---

### File 4: MODIFY `real_state_study.py`

**Path**: `husky_assembly_tamp/motion_planner/stage1/real_state_study.py`

Two edits.

#### Edit 4a — Add argparse flags in `parse_args()` (insert before line 967)

Find these lines at the end of `parse_args()`:
```python
    parser.add_argument("--video-frame-sleep", type=float, default=0.02, help="Replay frame interval used to derive batch video FPS")
    args = parser.parse_args()
```

Insert between them:
```python
    parser.add_argument(
        "--planner",
        choices=["dual-arm-constrained", "single-arm-free", "dual-arm-free"],
        default="dual-arm-constrained",
        help="Planning mode",
    )
    parser.add_argument(
        "--active-arm",
        choices=["left", "right"],
        default="left",
        help="Which arm to plan for in single-arm-free mode",
    )
```

#### Edit 4b — Add dispatch in `main()` (replace lines 1029-1051)

Find these lines:
```python
            stage_runner = {
                1: run_stage1_trial,
                2: run_stage2_trial,
                3: run_stage3_trial,
            }[args.stage]
            result = stage_runner(
                grasp_json=spec["grasp_json"],
                start_state_json=args.start_state,
                end_state_json=spec["state_json"],
                use_gui=args.gui,
                position_res=args.position_res,
                rotation_res=args.rotation_res,
                endpoint_ik_attempts=args.endpoint_ik_attempts,
                joint_continuity_threshold_rad=args.joint_continuity_threshold,
                max_time=args.max_time,
                max_iterations=args.max_iterations,
                max_attempts=args.max_attempts,
                random_seed=args.random_seed,
                lock_renderer_during_search=args.lock_renderer_during_search,
                scene_spec=scene_spec,
                validation_reports_dir=support_dir(),
                swap_grasps=args.swap_grasps,
            )
```

Replace with:
```python
            if args.planner == "dual-arm-constrained":
                stage_runner = {
                    1: run_stage1_trial,
                    2: run_stage2_trial,
                    3: run_stage3_trial,
                }[args.stage]
                result = stage_runner(
                    grasp_json=spec["grasp_json"],
                    start_state_json=args.start_state,
                    end_state_json=spec["state_json"],
                    use_gui=args.gui,
                    position_res=args.position_res,
                    rotation_res=args.rotation_res,
                    endpoint_ik_attempts=args.endpoint_ik_attempts,
                    joint_continuity_threshold_rad=args.joint_continuity_threshold,
                    max_time=args.max_time,
                    max_iterations=args.max_iterations,
                    max_attempts=args.max_attempts,
                    random_seed=args.random_seed,
                    lock_renderer_during_search=args.lock_renderer_during_search,
                    scene_spec=scene_spec,
                    validation_reports_dir=support_dir(),
                    swap_grasps=args.swap_grasps,
                )
            else:
                from husky_assembly_tamp.motion_planner.stage1.free_space_rrt import run_free_space_trial
                result = run_free_space_trial(
                    planner_mode=args.planner,
                    active_arm=args.active_arm,
                    grasp_json=spec["grasp_json"],
                    start_state_json=args.start_state,
                    end_state_json=spec["state_json"],
                    use_gui=args.gui,
                    max_time=args.max_time,
                    max_iterations=args.max_iterations,
                    max_attempts=args.max_attempts,
                    joint_resolution=args.position_res,
                    enable_smoothing=True,
                    smooth_iterations=100,
                    random_seed=args.random_seed,
                    lock_renderer_during_search=args.lock_renderer_during_search,
                    scene_spec=scene_spec,
                    validation_reports_dir=support_dir(),
                    swap_grasps=args.swap_grasps,
                    joint_continuity_threshold_rad=args.joint_continuity_threshold,
                    use_angle_normalization=True,
                    enable_collision=True,
                    include_built_bars=args.include_built_bars,
                )
```

All code AFTER this block (lines 1052-1133: `summarize_result`, smoothing plot, trajectory save, video record, visualization loop) remains UNTOUCHED. It reads from `result` using the same keys.

---

## Critical Implementation Notes

### 1. Import verification
Before writing `free_space_rrt.py`, verify every import from `minimal_rrt.py` exists:
```bash
cd /Users/huangyijiang/Code/husky-assembly-teleop/external/husky_assembly_tamp
grep -n "def log_validation_summary\|def summarize_joint_continuity\|def maybe_normalize_angles\|def setup_planning_scene\|def teardown_planning_scene\|def run_visualization_loop\|def get_disabled_collisions_from_link_names\|HUSKY_DUAL_URDF_PATH\|HUSKY_DUAL_SRDF_PATH\|DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD\|DEFAULT_USE_ANGLE_NORMALIZATION\|STAGE3_GRASP_MASK_LINKS" husky_assembly_tamp/motion_planner/stage1/minimal_rrt.py
```
If `log_validation_summary` doesn't exist, write a small local function:
```python
def _log_validation_summary(validation: Dict[str, Any]) -> None:
    logger.info(
        "Validation: collision_free=%s joint_continuity_ok=%s",
        validation.get("collision_free"),
        validation.get("joint_continuity_ok"),
    )
```

### 2. `setup_planning_scene` call signature
Check the actual signature of `setup_planning_scene` (line 1473). It may have different parameter names than shown. Match exactly. Key parameters:
- `grasp_json`, `start_state_json`, `end_state_json`, `use_gui` — these are positional or keyword
- `scene_spec` — optional dict with pre-computed values
- `swap_grasps` — optional bool

### 3. No existing code changes
Do NOT modify `plan_pose_rrt`, `extend_toward`, `smooth_dual_arm_pose_path`, or any other existing planning function. The constrained planner path must be bit-for-bit identical when `--planner dual-arm-constrained` (the default).

### 4. `pp.solve_motion_plan` `draw_fn` flow
The `draw_fn` flows through: `solve_motion_plan(**kwargs)` -> `birrt(**kwargs)` -> `random_restarts(rrt_connect, **kwargs)` -> `rrt_connect(..., draw_fn=draw_fn)`. Inside `rrt_connect`, it calls `TreeNode.draw(draw_fn)` which calls `draw_fn(config, [config, parent_config], True, True)`. All extra kwargs are absorbed by `**kwargs` at each level, so no TypeError.

### 5. `scene["collision_obstacles"]` contents
Defined at line 1572: `collision_obstacles = [body for body in pp.get_bodies() if body not in non_obstacle_bodies]`. This INCLUDES the robot itself. You MUST filter out the robot: `[b for b in scene["collision_obstacles"] if b != robot]` because `pp.get_collision_fn` handles self-collision checking separately.

---

## Verification Tests

Run these after implementation to verify correctness.

### Static tests (no PyBullet needed)

```bash
# Test 1: Module imports without error
python -c "
from husky_assembly_tamp.motion_planner.stage1.free_space_rrt import (
    plan_free_space_motion, run_free_space_trial,
    make_free_space_draw_fn, build_free_space_collision_fn,
    get_arm_joint_names, get_tool_link_name,
    LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES,
    TREE_COLOR_LEFT, TREE_COLOR_RIGHT,
)
print('Import OK')
"

# Test 2: CLI flags in minimal_rrt.py
python -m husky_assembly_tamp.motion_planner.stage1.minimal_rrt --help 2>&1 | grep -q "planner" && echo "PASS: --planner found" || echo "FAIL"
python -m husky_assembly_tamp.motion_planner.stage1.minimal_rrt --help 2>&1 | grep -q "active-arm" && echo "PASS: --active-arm found" || echo "FAIL"

# Test 3: CLI flags in real_state_study.py
python -m husky_assembly_tamp.motion_planner.stage1.real_state_study --help 2>&1 | grep -q "planner" && echo "PASS: --planner found" || echo "FAIL"
python -m husky_assembly_tamp.motion_planner.stage1.real_state_study --help 2>&1 | grep -q "active-arm" && echo "PASS: --active-arm found" || echo "FAIL"

# Test 4: path_validation skip_relative_transform parameter
python -c "
import inspect
from husky_assembly_tamp.motion_planner.stage1.path_validation import validate_stage_trajectory
sig = inspect.signature(validate_stage_trajectory)
assert 'skip_relative_transform' in sig.parameters, 'Missing parameter'
p = sig.parameters['skip_relative_transform']
assert p.default is False, f'Wrong default: {p.default}'
print('PASS: skip_relative_transform param OK, default=False')
"

# Test 5: run_free_space_trial signature
python -c "
import inspect
from husky_assembly_tamp.motion_planner.stage1.free_space_rrt import run_free_space_trial
sig = inspect.signature(run_free_space_trial)
required = ['planner_mode', 'active_arm', 'grasp_json', 'start_state_json', 'end_state_json',
            'max_time', 'max_iterations', 'max_attempts', 'joint_resolution',
            'enable_smoothing', 'smooth_iterations', 'use_gui', 'scene_spec']
missing = [p for p in required if p not in sig.parameters]
assert not missing, f'Missing params: {missing}'
print('PASS: run_free_space_trial signature OK')
"

# Test 6: get_arm_joint_names correctness
python -c "
from husky_assembly_tamp.motion_planner.stage1.free_space_rrt import get_arm_joint_names
left = get_arm_joint_names('left')
right = get_arm_joint_names('right')
assert len(left) == 6, f'Left arm should have 6 joints, got {len(left)}'
assert len(right) == 6, f'Right arm should have 6 joints, got {len(right)}'
assert all('left' in j for j in left), 'Left joints should contain left'
assert all('right' in j for j in right), 'Right joints should contain right'
print('PASS: joint names correct')
"
```

### Integration tests (require PyBullet + data files)

```bash
# Test 7: Single-arm-free planning (headless)
python -m husky_assembly_tamp.motion_planner.stage1.minimal_rrt \
    --planner single-arm-free --active-arm left --no-gui \
    --max-time 15 --max-iterations 1000 --max-attempts 3
# Expected: "Free-space BiRRT found path with N waypoints" or failure after retries
# Expected: Validation summary logged (collision_free, joint_continuity_ok)
# Expected: NO relative_transform errors/warnings

# Test 8: Dual-arm-free planning (headless)
python -m husky_assembly_tamp.motion_planner.stage1.minimal_rrt \
    --planner dual-arm-free --no-gui \
    --max-time 15 --max-iterations 1000 --max-attempts 3

# Test 9: Default planner unchanged (regression test)
python -m husky_assembly_tamp.motion_planner.stage1.minimal_rrt \
    --no-gui --max-time 15 --max-iterations 500 --max-attempts 1
# Expected: Same behavior as before, uses plan_pose_rrt

# Test 10: real_state_study with free-space (single target)
python -m husky_assembly_tamp.motion_planner.stage1.real_state_study \
    --planner single-arm-free --active-arm left \
    --targets G1 --max-time 15 --max-iterations 1000 --max-attempts 3
# Expected: Report generated, trajectory saved, video recorded
```

### GUI visual tests (manual)

```bash
# Test 11: Single-arm tree drawing
python -m husky_assembly_tamp.motion_planner.stage1.minimal_rrt \
    --planner single-arm-free --active-arm left \
    --max-time 15 --max-iterations 500
# VERIFY: Blue tree lines from left arm tool0 positions in PyBullet window

# Test 12: Dual-arm tree drawing
python -m husky_assembly_tamp.motion_planner.stage1.minimal_rrt \
    --planner dual-arm-free \
    --max-time 15 --max-iterations 500
# VERIFY: Blue (left) + green (right) tree lines in PyBullet window
```
