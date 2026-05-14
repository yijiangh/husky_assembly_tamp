# Code Simplification Analysis: `minimal_rrt.py` & `real_state_study.py`

## 1. What each file does

### `minimal_rrt.py` (1986 lines) — Planner library + dev CLI

A single-tree RRT planner for dual-arm bar transfer in SE(3) pose space, with three progressive constraint stages:
- **Stage 1**: Pure task-space search (can the bar move through space?)
- **Stage 2**: + dual-arm IK at every tree node (can both arms realize that motion?)
- **Stage 3**: + robot collision checking (can they do it without collision?)

It also contains: data loaders, mesh processing, IK solvers, scene setup/teardown, GUI visualization, profiling infrastructure, and an argparse CLI — all in one file.

### `real_state_study.py` (815 lines) — Batch experiment runner

Runs the planner from `minimal_rrt.py` against real design-study targets in batch. For each target, it:
1. Loads per-target grasp + goal data from the design study folder
2. Auto-computes a comfortable start bar pose (flip + bar-axis rotation)
3. Optionally validates the start pose with IK before planning
4. Runs the chosen stage planner
5. Produces a markdown report + JSON summary

Also has an endpoint-IK diagnosis mode (`--diagnose-endpoint-ik`) that tests IK feasibility at start/goal without running the full planner.

---

## 2. Usage instructions

### Running a single test case (dev CLI in `minimal_rrt.py`)

```bash
# Stage 3 planning on the default test case with GUI
python -m husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.run --stage 3

# Stage 1 only, headless
python -m husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.run --stage 1 --no-gui

# Custom grasp/state files
python -m husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.run \
    --grasp-json path/to/GraspTargets.json \
    --start-state path/to/start_RobotCellState.json \
    --end-state path/to/end_RobotCellState.json \
    --stage 3
```

### Running batch experiments (real_state_study.py)

```bash
# Full Stage 3 study on all default targets
python -m husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.run --stage 3

# Specific targets with GUI
python -m husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.run \
    --targets G1 V1 H1 --stage 3 --gui

# Endpoint IK diagnosis only (no planning)
python -m husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.run \
    --diagnose-endpoint-ik both --targets G1 G2 --gui

# Disable auto home pose, use manual offset/yaw
python -m husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.run \
    --no-auto-home-pose \
    --home-left-tool-offset 0 0 0.2 \
    --home-left-tool-local-yaw 3.14159

# Include already-built bars in the scene for collision
python -m husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.run \
    --include-built-bars --enable-built-bar-collision --targets V2
```

Key CLI flags for `real_state_study.py`:
| Flag | Default | Purpose |
|------|---------|---------|
| `--stage` | 3 | Planning stage (1/2/3) |
| `--targets` | G1..V3 | Which bars to plan |
| `--gui` | off | Show PyBullet GUI |
| `--auto-home-pose / --no-auto-home-pose` | on | Auto-compute start bar orientation |
| `--diagnose-endpoint-ik start/goal/both` | off | IK check mode (skips planning) |
| `--include-built-bars` | off | Load prior bars for collision |
| `--max-time` | 30s | Per-attempt time limit |
| `--max-attempts` | 5 | RRT restart budget |

---

## 3. Complexity inventory

### 3.1 `minimal_rrt.py` — 12 distinct responsibilities in one file

| Responsibility | Lines | Functions |
|---------------|-------|-----------|
| Constants & types | 1–106 | — |
| Data loaders (JSON, mesh) | 108–280 | 12 functions |
| Home pose computation | 283–404 | 3 functions |
| Design study spec | 407–463 | 1 function |
| Distance/sampling/nearest | 466–545 | 5 functions |
| Collision setup | 547–599 | 3 functions |
| Profiling helpers | 602–622 | 3 functions |
| IK solvers | 625–789 | 5 functions |
| **RRT core** | 792–1222 | 5 functions |
| Scene management | 1225–1441 | 6 functions |
| Stage runners | 1444–1721 | 5 functions |
| Visualization | 1770–1900 | 1 function |
| Dev CLI | 1902–1986 | 1 function |

### 3.2 Over-parameterized functions

| Function | Params | Notes |
|----------|--------|-------|
| `plan_pose_rrt` | 24 | The core RRT; many params just set defaults |
| `run_stage_trial` | 18 | Mostly passed through to `plan_pose_rrt` |
| `extend_toward` | 17 | Inner loop; params threaded from above |
| `setup_planning_scene` | 6 | But `scene_spec` dict carries ~8 more |
| `real_state_study.parse_args` | 26 flags | Many rarely changed from defaults |

### 3.3 Specific issues

1. **`profile_out` everywhere** (~30 call sites): `add_profile_time()`, `bump_profile_count()` sprinkled through every function. Adds 2-3 lines per call site.

2. **`debug_tree_out`**: Only used in dev CLI. Adds params to `plan_pose_rrt` and `update_debug_tree()`.

3. **`use_angle_normalization` toggle**: Default=True, never changed in batch experiments. Every IK/continuity call wraps values in `maybe_normalize_angles()`.

4. **`dist_metric` option**: "feature" vs "pose6d". Only "feature" is used in practice. Adds branching in `nearest_node`, `extend_toward`, `pose_distance`.

5. **Three thin wrappers**: `run_stage1_trial`, `run_stage2_trial`, `run_stage3_trial` are one-liners that pass `stage=N`.

6. **Duplicate scene dict fields**: `start_pose` == `world_from_bar_start`, `end_pose` == `world_from_bar_goal`. Both stored, both used.

7. **`compute_common_start_context()`** in `real_state_study.py`: Creates a 2-key dict. Could be inlined.

8. **Dead debug logs**: `logger.info('start')` at lines 526, 541 of `real_state_study.py`.

9. **Two auto-home validation functions**: `validate_auto_home_start_context` (needs existing scene) and `validate_auto_home_start_context_with_temporary_scene` (creates a throwaway scene). The temporary-scene variant exists because the auto-computation happens before the planning scene is set up.

10. **Markdown report generators** (~120 lines): Verbose string-building for two report types (planning + IK diagnosis). Most information is already in the JSON output.

11. **`suppress_native_output`**: 25-line context manager used exactly once.

12. **`hold_gui_pose`** + complex if/elif dispatch in `run_endpoint_ik_diagnosis`: 20+ lines of GUI logic.

---

## 4. Simplification plan

### Principle: Remove what's never varied, inline what's trivial, split what's unrelated.

### 4.1 Hardcode always-on options

**Remove `use_angle_normalization` toggle.** Hardcode to True. Remove `maybe_normalize_angles()` — just call `normalize_angles()` directly. This removes the parameter from `plan_pose_rrt`, `extend_toward`, `solve_single_arm_ik`, `solve_dual_arm_pose_ik`, `solve_endpoint_dual_arm_ik`, `summarize_joint_continuity`, `reconstruct_joint_path_for_pose_path`, `run_stage_trial`. Estimated ~40 lines removed.

**Remove `dist_metric` parameter.** Hardcode to "feature". Remove the "pose6d" branch from `nearest_node`, `pose_distance`. Estimated ~15 lines removed.

### 4.2 Remove profiling & debug infrastructure from hot path

**Remove `profile_out` parameter** from `plan_pose_rrt`, `extend_toward`, `solve_dual_arm_pose_ik`, `solve_endpoint_dual_arm_ik`, `reconstruct_joint_path_for_pose_path`. Delete `add_profile_time()`, `bump_profile_count()`. If profiling is needed later, it can be re-added with a decorator or context manager pattern rather than per-line instrumentation. Estimated ~80 lines removed.

**Remove `debug_tree_out` parameter** from `plan_pose_rrt`. Delete `update_debug_tree()`, `export_tree()`. Estimated ~30 lines removed.

### 4.3 Consolidate stage runners

**Delete `run_stage1_trial`, `run_stage2_trial`, `run_stage3_trial`.** Callers use `run_stage_trial(stage=N)` directly. In `real_state_study.py`, replace the `{1: run_stage1_trial, ...}` dispatch dict with a direct call.

### 4.4 Deduplicate scene dict fields

**Remove `start_pose` and `end_pose`** from the scene dict. Only keep `world_from_bar_start` and `world_from_bar_goal`. Update the ~5 call sites that reference `start_pose`/`end_pose`.

### 4.5 Clean up `real_state_study.py`

**Inline `compute_common_start_context()`** — it's a 2-key dict construction. Move to `main()`.

**Remove dead debug logs** — `logger.info('start')` at lines 526, 541.

**Merge the two auto-home validation functions** into one: `validate_auto_home_start_context_with_temporary_scene` already wraps `validate_auto_home_start_context`. Keep only the temporary-scene version (rename to `validate_auto_home_with_ik`) since that's the only call site from `main()`. Estimated ~30 lines removed.

**Simplify the GUI hold-pose dispatch** in `run_endpoint_ik_diagnosis` — the 15-line if/elif chain at lines 570-582 can be collapsed to a simple priority list.

### 4.6 Optional: split `minimal_rrt.py` into focused modules

This is the highest-impact change but also the most invasive. The file has 12 responsibilities. A clean split:

| New file | What moves there | ~Lines |
|----------|-----------------|--------|
| `minimal_rrt.py` | Constants, types, RRT core (`plan_pose_rrt`, `extend_toward`, sampling, distance), stage runner | ~700 |
| `scene.py` | `setup_planning_scene`, `teardown_planning_scene`, visualization markers, `run_visualization_loop` | ~300 |
| `ik.py` | All IK functions, collision setup, joint continuity, joint path reconstruction | ~350 |
| `data.py` | JSON loaders, mesh processing, design study spec, home pose computation | ~350 |

**Trade-off**: Cleaner separation vs. churn. Recommend doing this only if the file continues growing. For now, changes 4.1–4.5 are lower-risk and achieve ~200 lines of reduction.

### 4.7 Summary of estimated impact

| Change | Lines removed | Risk |
|--------|-------------|------|
| Hardcode angle normalization | ~40 | Low |
| Hardcode dist_metric=feature | ~15 | Low |
| Remove profile_out | ~80 | Low (re-add if needed) |
| Remove debug_tree_out | ~30 | Low |
| Delete stage N wrappers | ~10 | Low |
| Deduplicate scene fields | ~10 | Low |
| Clean up real_state_study | ~50 | Low |
| **Total** | **~235** | |
| Optional: split into modules | 0 (reorg) | Medium |

This would bring `minimal_rrt.py` from ~1986 to ~1800 lines and `real_state_study.py` from ~815 to ~765 lines. The module split (4.6) doesn't reduce total lines but makes each file single-purpose.
