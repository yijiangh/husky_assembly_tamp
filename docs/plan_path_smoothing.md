# Dual-arm-aware shortcut smoothing for `minimal_rrt`

## Context

`plan_pose_rrt` in `husky_assembly_tamp/motion_planner/stage1/minimal_rrt.py:996` returns a dense, collision-free SE(3) path of bar poses. For stages 2/3 it also returns a parallel list of seed-chained dual-arm joint configurations (`minimal_rrt.py:1207-1212`). The raw RRT paths are zig-zaggy in task space and can contain joint excursions that are wasteful on hardware. We need a post-processing step that shortens the path while still respecting three constraints that the generic `pybullet_planning.motion_planners.smoothing.smooth_path` does not know about:

1. **Joint continuity** (no >~10° deltas per step across the 12 joints).
2. **Dual-arm end-effector closure** on the bar grasps (via `solve_dual_arm_pose_ik` at `minimal_rrt.py:689`).
3. **Collision-freeness** — floating-bar (stage 1) or full robot + attached-bar (stage 3).

The critical subtlety is IK **seed chaining**: the joint conf at dense waypoint `k` depends on the conf at `k-1`. After a shortcut is spliced in at index `i..j`, the new joint conf at `j` is generally in a different IK branch than the original `path_confs[j]`, so the seed chain from `j+1` onwards drifts too. The smoother therefore has to re-IK **the entire tail from i to the very end of the path**, not just the shortcut segment — otherwise a discontinuity appears at the splice point.

Outcome: a new `smooth_dual_arm_pose_path` function added to `minimal_rrt.py`, invoked automatically after every successful `plan_pose_rrt` run (with sensible defaults). Stages 2/3 get the full IK-aware treatment; stage 1 uses only floating-bar pose collision checks.

Reference for the base algorithm: `external/pybullet_planning/src/pybullet_planning/motion_planners/smoothing.py:35` (`smooth_path`). This plan adapts it to dual-arm constraints.

## Design

### Data flow recap

Inputs (for stages 2/3):
- `path_poses: List[PoseLike]` — dense bar poses at `position_res`/`rotation_res` granularity.
- `path_confs: List[FullConf]` — parallel 12-DOF joint configs, one per pose.

For stage 1, `path_confs is None` and only the pose path exists.

Output: same shape, with waypoints removed / replaced to reduce task-space cost under `pose_distance` (`minimal_rrt.py:476`).

### Coarsening for sampling bias

Raw dense paths can have hundreds of consecutive near-collinear points. Picking two random dense indices overwhelmingly produces tiny no-op shortcuts. To match the `pybullet_planning.motion_planners.smoothing.smooth_path` style and make each iteration meaningful, we sample shortcut endpoints from **inflection indices** of the dense path.

Add a helper (private, near `pose_distance` at `minimal_rrt.py:476`):

```python
def _pose_path_inflection_indices(
    path_poses: Sequence[PoseLike],
    feature_points: Sequence[np.ndarray],
    tolerance: float = 1e-3,
) -> List[int]:
    """Return ascending indices into path_poses that mark inflection points in the
    dense path. Uses pose_to_feature_vec so the geometry matches pose_distance."""
```

Implementation mirrors `pybullet_planning.motion_planners.utils.waypoints_from_path` at `external/pybullet_planning/src/pybullet_planning/motion_planners/utils.py:133` but:
- Runs on the 24-D feature vectors produced by `pose_to_feature_vec(pose, feature_points)` (`minimal_rrt.py:466`).
- Returns **indices** into the input list (not values) so we can look up both poses and confs by index.
- Uses raw vector subtraction (`vec_j - vec_i`) instead of a `difference_fn`; unit-vector direction changes beyond `tolerance` mark an inflection.
- Always includes the first and last index.

Recomputed after each accepted shortcut (cheap — O(N) with one feature vector per pose).

### New function: `smooth_dual_arm_pose_path`

Add right after `reconstruct_joint_path_for_pose_path` (around `minimal_rrt.py:975`).

```python
def smooth_dual_arm_pose_path(
    path_poses: Sequence[PoseLike],
    path_confs: Optional[Sequence[FullConf]],
    *,
    scene: Dict[str, Any],
    pose_collision_fn: Optional[Callable[[PoseLike], bool]] = None,
    joint_collision_fn: Optional[Callable[[FullConf], bool]] = None,
    dist_metric: str = "feature",
    feature_points: Optional[Sequence[np.ndarray]] = None,
    position_res: float = 0.05,
    rotation_res: float = 0.1,
    joint_continuity_threshold_rad: Optional[float] = None,
    use_angle_normalization: bool = DEFAULT_USE_ANGLE_NORMALIZATION,
    max_smooth_iterations: int = 100,
    max_time: float = 10.0,
    min_cost_improvement: float = 0.0,
    inflection_tolerance: float = 1e-3,
    random_seed: Optional[int] = None,
    profile_out: Optional[Dict[str, Any]] = None,
) -> Tuple[List[PoseLike], Optional[List[FullConf]]]:
```

Main loop (pseudocode):

```python
if path_poses is None or len(path_poses) < 3 or max_smooth_iterations <= 0:
    return (list(path_poses) if path_poses is not None else None,
            list(path_confs) if path_confs is not None else None)

rng = np.random.default_rng(random_seed)
feature_points = list(feature_points) if feature_points is not None else get_bar_feature_points()
current_poses = list(path_poses)
current_confs = list(path_confs) if path_confs is not None else None
current_cost = _pose_path_cost(current_poses, dist_metric, feature_points)

# profile_out init: cost_before, waypoints_before, counters zeroed.

start_time = time.time()
inflection_idxs = _pose_path_inflection_indices(current_poses, feature_points, inflection_tolerance)

for iteration in range(max_smooth_iterations):
    if (time.time() - start_time) >= max_time:
        break
    if len(inflection_idxs) < 3:
        break  # nothing left to shortcut

    bump(profile_out, "shortcut_attempts")

    # Sample two inflection positions with at least one inflection strictly between.
    ii = int(rng.integers(0, len(inflection_idxs) - 2))
    jj = int(rng.integers(ii + 2, len(inflection_idxs)))
    i = inflection_idxs[ii]
    j = inflection_idxs[jj]

    # Build the dense shortcut between current_poses[i] and current_poses[j].
    shortcut = list(
        pp.interpolate_poses(
            current_poses[i],
            current_poses[j],
            pos_step_size=max(position_res, 1e-6),
            ori_step_size=max(rotation_res, 1e-6),
        )
    )
    # interpolate_poses includes both endpoints; splice the whole dense path as:
    candidate_poses = list(current_poses[:i]) + shortcut + list(current_poses[j + 1:])

    # Cost check first (cheapest).
    new_cost = _pose_path_cost(candidate_poses, dist_metric, feature_points)
    if (current_cost - new_cost) <= min_cost_improvement:
        bump(profile_out, "cost_rejections")
        continue

    # Stage-1 pose collision check on the *new* shortcut segment only.
    if pose_collision_fn is not None:
        if any(pose_collision_fn(p) for p in shortcut[1:-1]):
            bump(profile_out, "collision_rejections")
            continue

    # IK re-propagation (stages 2/3). Reuses reconstruct_joint_path_for_pose_path,
    # which already solves seed-chained dual-arm IK with continuity + joint
    # collision checks and returns (None, failure_reason) on failure.
    new_confs = None
    if current_confs is not None:
        joint_suffix, reason = reconstruct_joint_path_for_pose_path(
            scene=scene,
            pose_path=candidate_poses[i:],
            start_conf=current_confs[i],
            joint_collision_fn=joint_collision_fn,
            joint_continuity_threshold_rad=joint_continuity_threshold_rad,
            use_angle_normalization=use_angle_normalization,
            profile_out=profile_out,
        )
        if joint_suffix is None:
            if reason and reason.startswith("ik_failure"):
                bump(profile_out, "ik_failures")
            elif reason and reason.startswith("continuity"):
                bump(profile_out, "continuity_rejections")
            elif reason and reason.startswith("collision"):
                bump(profile_out, "collision_rejections")
            continue
        new_confs = list(current_confs[:i]) + list(joint_suffix)
        assert len(new_confs) == len(candidate_poses)

    # Accept
    current_poses = candidate_poses
    if current_confs is not None:
        current_confs = new_confs
    current_cost = new_cost
    inflection_idxs = _pose_path_inflection_indices(current_poses, feature_points, inflection_tolerance)
    bump(profile_out, "accepts")

# profile_out finalize: cost_after, waypoints_after, smooth_time_s.
return current_poses, current_confs
```

Key points:

- **Coarsening is only a sampling bias**, not a change of representation. The internal state is always the dense pose/conf list. After each accepted shortcut, we recompute the inflection indices so subsequent samples land on the new control points. This matches the user's verbal algorithm ("pick two indices in the path") while avoiding degenerate adjacent-index samples.
- **Re-IK "until the very end"** is delegated to `reconstruct_joint_path_for_pose_path` at `minimal_rrt.py:922`, which already does exactly that: seed-chained `solve_dual_arm_pose_ik` with joint continuity and optional joint collision checks, returning `(None, failure_reason)` on failure. We never need to touch `solve_dual_arm_pose_ik` directly — the helper is the right-sized primitive.
- **Stage 1** passes `path_confs=None` and `pose_collision_fn=<floating bar collision>`; the IK branch is skipped entirely. Stage 2 passes `joint_collision_fn=None`. Stage 3 passes `joint_collision_fn` and relies on the helper for collision checks (no `pose_collision_fn`).
- **Cost metric** uses `pose_distance(...)` from `minimal_rrt.py:476` so the smoother's notion of "shorter" matches the RRT's notion of nearest-neighbor distance. Sum over consecutive dense pairs is the feature-space arc length, which strictly decreases when a shortcut flattens a detour.
- **Pose collision sweep** for stage 1 only checks `shortcut[1:-1]` (the newly-inserted interior points). The endpoints were already validated as part of the previous iteration's state.

### Re-using existing machinery

| Need | Reuse |
|---|---|
| Task-space distance / cost | `pose_distance` (`minimal_rrt.py:476`) |
| Bar task-space interpolation | `pp.interpolate_poses` (same call used in `extend_toward` at `minimal_rrt.py:819`) |
| Seed-chained dual-arm IK along a pose path with continuity + collision checks | `reconstruct_joint_path_for_pose_path` (`minimal_rrt.py:922`) |
| Feature-point projection for inflection detection | `pose_to_feature_vec` (`minimal_rrt.py:466`) |
| Feature points for the bar | `get_bar_feature_points()` / `scene["feature_points"]` |
| Stage-1 pose collision | `get_pose_collision_fn` (`minimal_rrt.py:547`) |
| Stage-3 joint collision | `get_joint_collision_fn` (`minimal_rrt.py:570`) — already built in `run_stage_trial` at line 1565 |
| Angle normalization | `maybe_normalize_angles` (used inside `reconstruct_joint_path_for_pose_path`) |

No new external imports are required.

Helper `_pose_path_cost(path_poses, dist_metric, feature_points) -> float`: one-line wrapper that sums `pose_distance(path_poses[k], path_poses[k+1], dist_metric, feature_points)` across consecutive pairs. Place it next to `_pose_path_inflection_indices`.

### Integration in `run_stage_trial`

Smoothing runs **on by default**. New kwargs on `run_stage_trial` (`minimal_rrt.py:1450`):

```python
enable_smoothing: bool = True,
smooth_max_iterations: int = 100,
smooth_max_time: float = 10.0,
smooth_min_cost_improvement: float = 0.0,
```

Call site: insert between `plan_pose_rrt` (line 1628/1630) and the path-found logging block (line 1633):

```python
if path is not None and enable_smoothing:
    smooth_profile: Dict[str, Any] = {}
    smooth_pose_collision_fn = (
        get_pose_collision_fn(scene["bar_body"], scene["collision_obstacles"], True)
        if (stage == 1 and enable_collision)
        else None
    )
    path, path_confs = smooth_dual_arm_pose_path(
        path_poses=path,
        path_confs=path_confs,
        scene=scene,
        pose_collision_fn=smooth_pose_collision_fn,
        joint_collision_fn=joint_collision_fn,  # None for stages 1/2, set for stage 3
        dist_metric=dist_metric,
        feature_points=scene["feature_points"],
        position_res=position_res,
        rotation_res=rotation_res,
        joint_continuity_threshold_rad=(joint_continuity_threshold_rad if enable_ik else None),
        use_angle_normalization=use_angle_normalization,
        max_smooth_iterations=smooth_max_iterations,
        max_time=smooth_max_time,
        min_cost_improvement=smooth_min_cost_improvement,
        random_seed=random_seed,
        profile_out=smooth_profile,
    )
    if planner_profile_out is not None:
        planner_profile_out["smoothing"] = smooth_profile
    logger.info(
        "Smoothing: cost %.4f -> %.4f, waypoints %d -> %d, accepts %d/%d",
        smooth_profile.get("cost_before", 0.0),
        smooth_profile.get("cost_after", 0.0),
        smooth_profile.get("waypoints_before", 0),
        smooth_profile.get("waypoints_after", 0),
        smooth_profile.get("accepts", 0),
        smooth_profile.get("shortcut_attempts", 0),
    )
```

Downstream: the existing `pp.set_pose(... path[-1])` at line 1635, `set_joint_positions(... path_confs[-1])` at line 1637, coarse continuity summary at line 1662, and `validate_stage_trajectory` call at line 1693 all operate on `(path, path_confs)` and automatically re-validate the smoothed output. The returned result dict (lines 1701-1715) already contains `path` and `path_confs`, so GUI visualization in `run_visualization_loop` also gets the smoothed version for free.

### CLI flags (`minimal_rrt.py:1925-1966`)

Add to the argparse block:

```python
parser.add_argument(
    "--smoothing",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run dual-arm-aware shortcut smoothing on the planned path (use --no-smoothing to disable)",
)
parser.add_argument("--smooth-iterations", type=int, default=100, help="Max shortcut iterations for path smoothing")
parser.add_argument("--smooth-max-time", type=float, default=10.0, help="Max wall time (s) for path smoothing")
parser.add_argument("--smooth-min-improvement", type=float, default=0.0, help="Minimum cost improvement per accepted shortcut")
```

Thread them into the `run_stage_trial(...)` call at line 1970:

```python
enable_smoothing=args.smoothing,
smooth_max_iterations=args.smooth_iterations,
smooth_max_time=args.smooth_max_time,
smooth_min_cost_improvement=args.smooth_min_improvement,
```

### Profile schema

`smooth_profile` populated by the smoother (surfaced in `planner_profile_out["smoothing"]`):

```
cost_before, cost_after,
waypoints_before, waypoints_after,
shortcut_attempts, cost_rejections,
ik_failures, continuity_rejections, collision_rejections,
accepts, smooth_time_s
```

These let benchmarks show whether smoothing is doing real work and which rejection mode dominates (useful for tuning `joint_continuity_threshold_rad` and `position_res`).

## Critical files

- `husky_assembly_tamp/motion_planner/stage1/minimal_rrt.py`
  - Add `_pose_path_inflection_indices` helper near line 490 (next to `pose_distance`/`pose_to_feature_vec`).
  - Add `_pose_path_cost` helper next to it.
  - Add `smooth_dual_arm_pose_path` after `reconstruct_joint_path_for_pose_path` (line 975).
  - Add `enable_smoothing`, `smooth_max_iterations`, `smooth_max_time`, `smooth_min_cost_improvement` kwargs to `run_stage_trial` (line 1450) and insert the call site between lines 1630 and 1633.
  - Add `--smoothing` / `--no-smoothing`, `--smooth-iterations`, `--smooth-max-time`, `--smooth-min-improvement` flags in `main()` around line 1955 and thread them into `run_stage_trial` at line 1970.

No other files need edits. `path_validation.py` and `run_visualization_loop` already consume `(path, path_confs)` unchanged.

## Verification

1. **Smoothing off vs on, same seed**:
   ```
   python -m husky_assembly_tamp.motion_planner.stage1.minimal_rrt --stage 3 --no-gui --random-seed 0 --no-smoothing
   python -m husky_assembly_tamp.motion_planner.stage1.minimal_rrt --stage 3 --no-gui --random-seed 0
   ```
   Expect: both succeed, the second reports smaller `len(path)`, `cost_after < cost_before` in the `smoothing` profile, `joint_continuity_ok = pass`, `collision_free = pass`, and `validate_stage_trajectory` still passes.

2. **Stage 1, 2, 3 sanity with smoothing on**:
   - Stage 1: `shortcut_attempts > 0`, only `collision_rejections` and `cost_rejections` can be nonzero. `ik_failures`, `continuity_rejections` stay 0.
   - Stage 2: only `ik_failures`, `continuity_rejections`, `cost_rejections` can be nonzero. `collision_rejections` stays 0.
   - Stage 3: any rejection counter can fire; final path still passes `validate_stage_trajectory`.

3. **Determinism**: same `--random-seed` → identical `(path, path_confs)` and identical `smoothing` profile numbers across two runs. This proves the RNG is threaded correctly.

4. **GUI replay**: `--stage 3` with GUI: scrub the "Path t" slider and confirm the bar + dual-arm robot move continuously, no visible collision, and the shorter path is visible in the pybullet line overlay.

5. **Benchmark delta**: rerun an existing batched stage-3 report (e.g. the one that produced `husky_assembly_tamp/logs/stage1_real_state_study.log`) with the new default and confirm success rate does not regress (smoothing must never turn a passing plan into a failing one — failed shortcut candidates are silently skipped).

## Out of scope

- Probability-weighted segment selection like the upstream `smooth_path` (uniform-over-inflections is good enough for v1).
- Caching partial IK suffixes between iterations (every accepted shortcut re-IKs the full tail; if profile numbers show this dominates, we can incrementally only re-IK until the first joint that matches the cached conf within tolerance).
- Continuous/sweep collision checks between consecutive interpolated poses (the existing RRT doesn't do this either, so smoothing doesn't need to introduce a stricter criterion).
- Trajectory time-parameterization or velocity smoothing — this is purely geometric path shortening.
