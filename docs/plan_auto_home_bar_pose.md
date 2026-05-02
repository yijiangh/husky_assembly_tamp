# Plan: Auto-compute Home Bar Pose from Grasps

## Context

The start/home bar pose is derived from a fixed left EE position (`MOBILE_BASE_FROM_TOOL0_LEFT_HOME`) plus grasp transforms. For certain grasps, the right arm lands in an awkward or unreachable configuration. We need to auto-optimize the bar orientation to produce comfortable dual-arm start poses, governed by three rules:

1. **Fixed left EE position**: Always place the left arm EE at roughly `MOBILE_BASE_FROM_TOOL0_LEFT_HOME + [0, 0, 0.2]` (0.2 m up).
2. **Bar-axis rotation (Rule 2)**: Rotate the bar around its longitudinal axis (bar frame z-axis) by angle `theta` so the **average** of both EEs' local z-axes points forward (`+x` in mobile base frame).
3. **EE-axis flip (Rule 3)**: Try rotating the bar 180 deg around the left EE's local z-axis, and pick whichever flip gives better z-alignment score.

Additionally, candidates must be **IK-feasible**: each candidate bar pose is validated via `evaluate_endpoint_ik()` (dual-arm IK + optional collision check) before being accepted.

## Current transform chain

```
grasp_bar_from_left  = inv(world_from_bar) * world_from_tool0_left    (from grasp JSON)
grasp_bar_from_right = inv(world_from_bar) * world_from_tool0_right   (from grasp JSON)
tool0_left_from_bar  = inv(grasp_bar_from_left)

bar_pose       = left_ee_home * tool0_left_from_bar
right_tool_ee  = bar_pose * bar_from_tool0_right
```

## Key files

| File | Line | What |
|------|------|------|
| `husky_assembly_tamp/motion_planner/stage1/minimal_rrt.py` | 78 | `MOBILE_BASE_FROM_TOOL0_LEFT_HOME` constant |
| `husky_assembly_tamp/motion_planner/stage1/minimal_rrt.py` | 295–314 | `derive_home_start_poses_from_grasps()` — current logic, **do not modify** (keep for backward compat) |
| `husky_assembly_tamp/motion_planner/stage1/minimal_rrt.py` | 666–699 | `solve_endpoint_dual_arm_ik()` — IK solver used by the validator |
| `husky_assembly_tamp/motion_planner/stage1/real_state_study.py` | 71–121 | `derive_start_pose_from_home_left_tool()` — applies offset + yaw, calls the above |
| `husky_assembly_tamp/motion_planner/stage1/real_state_study.py` | 276–320 | `evaluate_endpoint_ik()` — IK + collision validator for a given bar pose |
| `husky_assembly_tamp/motion_planner/stage1/real_state_study.py` | 455–524 | `parse_args()` — CLI flags |
| `husky_assembly_tamp/motion_planner/stage1/real_state_study.py` | 14–29 | imports from `minimal_rrt` |

## Design: two-phase candidate selection

The algorithm is split into two phases to avoid running expensive IK on all 720 grid samples:

1. **Phase 1 — geometric ranking** (in `minimal_rrt.py`): grid-search over (flip, theta), score by average-z-alignment, return top-K candidates sorted by score.
2. **Phase 2 — IK validation** (in `real_state_study.py`): iterate through top-K candidates, call `evaluate_endpoint_ik()` for each, accept the first one that passes IK (and collision if Stage 3).

This works because `evaluate_endpoint_ik()` takes `bar_pose` as a parameter — it computes target EE poses analytically from `bar_pose * grasp`, so a single scene setup can test any number of candidate bar poses. The collision checker uses a pybullet `Attachment` that auto-positions the bar from the solved joint config, so it is also bar-pose-independent.

## Implementation steps

### Step 1: Add constant in `minimal_rrt.py` (near line 77)

```python
DEFAULT_HOME_LEFT_TOOL_Z_OFFSET = 0.2
```

### Step 2: Add `auto_compute_home_bar_pose()` in `minimal_rrt.py` (after line 314)

Insert a new function right after `derive_home_start_poses_from_grasps()`. Keep the existing function untouched.

```python
def auto_compute_home_bar_pose(
    grasp_targets: Sequence[GraspTarget],
    mobile_base_from_tool0_left: PoseLike = MOBILE_BASE_FROM_TOOL0_LEFT_HOME,
    forward_direction: np.ndarray = np.array([1.0, 0.0, 0.0]),
    ik_validator: Optional[Callable[[PoseLike], bool]] = None,
    num_geometric_candidates: int = 20,
) -> Dict[str, Any]:
    """Auto-compute the home bar pose by optimizing bar-axis rotation and EE-axis flip.

    Two-phase approach:
    1. Geometric ranking: grid-search over (flip_yaw, theta), score by average EE
       z-axis alignment with forward_direction. Collect top-K candidates.
    2. IK validation (if ik_validator is provided): iterate through top-K candidates
       in score order, accept the first one where ik_validator(bar_pose) returns True.
       If all top-K fail, fall back to the best geometric candidate.

    Parameters
    ----------
    grasp_targets : list of GraspTarget
        Two grasp targets (left, right).
    mobile_base_from_tool0_left : PoseLike
        Fixed left EE home pose (already with any position offset applied).
    forward_direction : np.ndarray
        Target direction for the average EE z-axis (default: +x in mobile base frame).
    ik_validator : callable, optional
        A function bar_pose -> bool that returns True if the bar pose is IK-feasible
        (and optionally collision-free). When None, pure geometric ranking is used.
    num_geometric_candidates : int
        Number of top geometric candidates to pass to the IK validation phase.
    """
    if len(grasp_targets) < 2:
        raise ValueError("Expected two grasp targets to auto-compute the home bar pose.")

    # Extract grasp transforms
    mobile_base_from_bar_left, mobile_base_from_tool0_left_goal = grasp_targets[0]
    mobile_base_from_bar_right, mobile_base_from_tool0_right_goal = grasp_targets[1]
    bar_from_tool0_left = pp.multiply(pp.invert(mobile_base_from_bar_left), mobile_base_from_tool0_left_goal)
    tool0_left_from_bar = pp.invert(bar_from_tool0_left)
    bar_from_tool0_right = pp.multiply(pp.invert(mobile_base_from_bar_right), mobile_base_from_tool0_right_goal)

    forward = np.asarray(forward_direction, dtype=float)
    forward = forward / np.linalg.norm(forward)

    # Phase 1: geometric ranking — collect all (score, flip, theta, bar_pose)
    all_candidates = []
    for flip_yaw in [0.0, np.pi]:
        adjusted_left = pp.multiply(
            mobile_base_from_tool0_left,
            pp.Pose(euler=pp.Euler(yaw=flip_yaw)),
        )
        bar_base = pp.multiply(adjusted_left, tool0_left_from_bar)

        for theta in np.linspace(-np.pi, np.pi, 360, endpoint=False):
            bar_rotated = pp.multiply(bar_base, pp.Pose(euler=pp.Euler(yaw=theta)))
            left_ee = pp.multiply(bar_rotated, bar_from_tool0_left)
            right_ee = pp.multiply(bar_rotated, bar_from_tool0_right)

            left_z = np.array(pp.tform_from_pose(left_ee))[:3, 2]
            right_z = np.array(pp.tform_from_pose(right_ee))[:3, 2]
            avg_z = left_z + right_z
            avg_z_norm = np.linalg.norm(avg_z)
            if avg_z_norm < 1e-9:
                continue
            avg_z = avg_z / avg_z_norm

            score = float(np.dot(avg_z, forward))
            all_candidates.append((score, flip_yaw, theta, bar_rotated))

    # Sort descending by geometric score
    all_candidates.sort(key=lambda x: -x[0])

    # Phase 2: IK validation on top-K candidates
    top_candidates = all_candidates[:num_geometric_candidates]
    chosen = top_candidates[0]  # fallback: best geometric candidate

    if ik_validator is not None:
        for candidate in top_candidates:
            score, flip_yaw, theta, bar_pose = candidate
            if ik_validator(bar_pose):
                chosen = candidate
                break
        else:
            logger.warning(
                "No IK-feasible candidate found among top %d geometric candidates; "
                "falling back to best geometric candidate.",
                num_geometric_candidates,
            )

    best_score, best_flip, best_theta, best_bar_pose = chosen

    # Recompute final poses
    adjusted_left_final = pp.multiply(
        mobile_base_from_tool0_left,
        pp.Pose(euler=pp.Euler(yaw=best_flip)),
    )
    bar_final = pp.multiply(
        pp.multiply(adjusted_left_final, tool0_left_from_bar),
        pp.Pose(euler=pp.Euler(yaw=best_theta)),
    )
    right_tool_final = pp.multiply(bar_final, bar_from_tool0_right)

    return {
        "mobile_base_from_tool0_left_start": adjusted_left_final,
        "mobile_base_from_bar_start": bar_final,
        "mobile_base_from_tool0_right_start": right_tool_final,
        "tool0_left_from_bar": tool0_left_from_bar,
        "bar_from_tool0_right": bar_from_tool0_right,
        "chosen_flip_yaw": best_flip,
        "chosen_bar_axis_theta": best_theta,
        "alignment_score": best_score,
    }
```

**Notes on the algorithm:**
- The bar's z-axis IS the longitudinal axis (from `BAR_BOX_DIMS = (0.03, 0.03, 1.0)`), so `pp.Euler(yaw=theta)` rotates around the correct axis.
- Bar radius is only 0.015 m, so bar-axis rotation barely moves EE positions (~0.015 m max displacement). Orientations are what change meaningfully.
- 360-sample grid at 1-deg resolution produces 720 total candidates (2 flips x 360 thetas). Geometric scoring is pure matrix math — fast.
- Top-K (default 20) candidates are passed to the IK phase. Each IK call does up to `endpoint_ik_attempts` (default 20) IK attempts, so worst case is 20 × 20 = 400 IK solves — acceptable for a one-shot startup computation.
- If no top-K candidate passes IK, the best geometric candidate is used as fallback (with a warning).

### Step 3: Update imports in `real_state_study.py` (line 14–29)

Add `auto_compute_home_bar_pose` and `DEFAULT_HOME_LEFT_TOOL_Z_OFFSET` to the import block:

```python
from husky_assembly_tamp.motion_planner.stage1.minimal_rrt import (
    ...
    auto_compute_home_bar_pose,
    DEFAULT_HOME_LEFT_TOOL_Z_OFFSET,
    ...
)
```

### Step 4: Update `parse_args()` in `real_state_study.py`

**a) Change default offset** — find line with `default=[0.0, 0.0, 0.0]` under `--home-left-tool-offset` and change to:
```python
default=[0.0, 0.0, 0.2],
```

**b) Add `--auto-home-pose` flag** — add after the `--home-left-tool-local-yaw` argument:
```python
parser.add_argument(
    "--auto-home-pose",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Auto-compute bar-axis rotation and flip for the home pose (ignores --home-left-tool-local-yaw when enabled)",
)
```

**c) Add `--auto-home-ik-candidates` flag** — optional tuning knob:
```python
parser.add_argument(
    "--auto-home-ik-candidates",
    type=int,
    default=20,
    help="Number of top geometric candidates to IK-validate during auto home pose computation",
)
```

### Step 5: Update `derive_start_pose_from_home_left_tool()` in `real_state_study.py`

Add `auto_home_pose: bool` and `ik_validator` parameters. When `auto_home_pose=True`, use the new auto function with IK validation; when False, preserve existing behavior.

```python
def derive_start_pose_from_home_left_tool(
    spec: Dict[str, Any],
    common_start: Dict[str, Any],
    home_left_tool_offset: List[float],
    home_left_tool_local_yaw: float,
    auto_home_pose: bool = True,
    ik_validator: Optional[Callable] = None,
    num_geometric_candidates: int = 20,
) -> Dict[str, Any]:
    if len(spec["grasp_targets"]) < 2:
        raise ValueError(f"Target {spec['target_name']} requires two grasp targets to derive the home start pose.")

    # (keep existing bar-position-delta warning code unchanged — lines 79–102)
    ...

    # Apply position offset to left tool home pose
    mobile_base_from_tool0_left_home = (
        np.asarray(common_start["mobile_base_from_tool0_left_home"][0], dtype=float) + np.asarray(home_left_tool_offset, dtype=float),
        common_start["mobile_base_from_tool0_left_home"][1],
    )

    if auto_home_pose:
        # Auto-compute: ignore home_left_tool_local_yaw
        start_pose_context = auto_compute_home_bar_pose(
            spec["grasp_targets"],
            mobile_base_from_tool0_left=mobile_base_from_tool0_left_home,
            ik_validator=ik_validator,
            num_geometric_candidates=num_geometric_candidates,
        )
        logger.info(
            "Auto home pose: flip_yaw=%.4f rad, bar_axis_theta=%.4f rad, alignment_score=%.4f",
            start_pose_context["chosen_flip_yaw"],
            start_pose_context["chosen_bar_axis_theta"],
            start_pose_context["alignment_score"],
        )
    else:
        # Manual mode: apply yaw and use existing function
        if abs(float(home_left_tool_local_yaw)) > 0.0:
            mobile_base_from_tool0_left_home = pp.multiply(
                mobile_base_from_tool0_left_home,
                pp.Pose(euler=pp.Euler(yaw=float(home_left_tool_local_yaw))),
            )
        start_pose_context = derive_home_start_poses_from_grasps(
            spec["grasp_targets"],
            mobile_base_from_tool0_left=mobile_base_from_tool0_left_home,
        )

    return {
        "mobile_base_from_tool0_left_home": start_pose_context["mobile_base_from_tool0_left_start"],
        "world_from_bar_start": start_pose_context["mobile_base_from_bar_start"],
        "derived_right_tool_pose": start_pose_context["mobile_base_from_tool0_right_start"],
    }
```

### Step 6: Build the `ik_validator` callback and thread through `main()`

The `ik_validator` callback wraps `evaluate_endpoint_ik()`. It needs a scene, so it must be constructed **after** scene setup.

**Problem**: `derive_start_pose_from_home_left_tool()` is called *before* `setup_planning_scene()`, creating a chicken-and-egg dependency.

**Solution**: split the auto-computation into two invocations:

1. **First call** (before scene setup): `auto_home_pose=True, ik_validator=None` — pure geometric ranking. Use this to get the initial bar pose for scene setup.
2. **Second call** (after scene setup, optional): if IK validation is desired, construct the validator from the scene and re-run with the validator. Or more efficiently, just validate the top-K candidates directly in the calling code.

**Recommended approach** — validate after scene setup in calling code:

```python
# In run_endpoint_ik_diagnosis() and the planning branch of main():

# 1. Geometric auto-computation (no IK)
start_context = derive_start_pose_from_home_left_tool(
    spec, common_start, args.home_left_tool_offset, args.home_left_tool_local_yaw,
    auto_home_pose=args.auto_home_pose,
    ik_validator=None,  # no scene yet
    num_geometric_candidates=args.auto_home_ik_candidates,
)

# 2. Set up scene with geometric-best bar pose
scene_spec = {
    "world_from_bar_start": start_context["world_from_bar_start"],
    ...
}
scene = setup_planning_scene(..., scene_spec=scene_spec)

# 3. If auto_home_pose, validate with IK using the scene
if args.auto_home_pose:
    rng = np.random.default_rng(args.random_seed)
    grasp_bar_from_right = scene["grasp_bar_from_right"]

    # Build collision checker for Stage 3
    joint_collision_checker = None
    if args.stage == 3:
        env_obstacles = [b for b in scene["collision_obstacles"] if b != scene["robot"]]
        joint_collision_checker = get_joint_collision_fn(
            robot=scene["robot"], arm_joints=scene["arm_joints"],
            obstacle_bodies=env_obstacles, tool_link_left=scene["tool_link_left"],
            bar_body=scene["bar_body"], grasp_bar_from_left=scene["grasp_bar_from_left"],
        )

    def ik_validator(bar_pose):
        result = evaluate_endpoint_ik(
            scene=scene,
            endpoint_name="start_auto",
            bar_pose=bar_pose,
            seed_conf=np.asarray(scene["start_joint_values"], dtype=float),
            grasp_bar_from_right=grasp_bar_from_right,
            rng=rng,
            endpoint_ik_attempts=args.endpoint_ik_attempts,
            joint_collision_checker=joint_collision_checker,
        )
        return result["ik_ok"] and (joint_collision_checker is None or result["collision_free"])

    # Re-run auto-computation with IK validation
    start_context_validated = derive_start_pose_from_home_left_tool(
        spec, common_start, args.home_left_tool_offset, args.home_left_tool_local_yaw,
        auto_home_pose=True,
        ik_validator=ik_validator,
        num_geometric_candidates=args.auto_home_ik_candidates,
    )

    # Update scene if the validated bar pose differs
    if start_context_validated["world_from_bar_start"] != start_context["world_from_bar_start"]:
        new_bar_pose = start_context_validated["world_from_bar_start"]
        pp.set_pose(scene["bar_body"], new_bar_pose)
        pp.set_pose(scene["ghost_start"], new_bar_pose)
        scene["world_from_bar_start"] = new_bar_pose
        scene["start_pose"] = new_bar_pose
        start_context = start_context_validated
```

**Key insight**: `evaluate_endpoint_ik()` takes `bar_pose` as a parameter and computes EE targets analytically via `bar_pose * grasp`. It does NOT depend on the bar body's pybullet world pose. The collision checker uses a pybullet `Attachment` that auto-positions the bar from the solved joint config. So a single scene can test any number of candidate bar poses without re-setup.

### Step 7: Update report scope section

In `write_report()`, add a line after the existing parameter lines:
```python
lines.append(f"- Auto home pose: `{args.auto_home_pose}`")
```

## Verification

1. `python -m husky_assembly_tamp.motion_planner.stage1.real_state_study --diagnose-endpoint-ik both --gui --targets G1 G2 V1`
   - Visually confirm left EE at red-box position (+0.2 m up), both z-axes roughly forward, right arm comfortable
2. `python -m husky_assembly_tamp.motion_planner.stage1.real_state_study --no-auto-home-pose --home-left-tool-offset 0 0 0 --targets G1` should reproduce previous behavior
3. Full Stage 3 planning on a few targets to confirm paths are still found
4. Test with `--auto-home-ik-candidates 0` to confirm pure-geometric fallback works (IK validation skipped when 0 candidates)
