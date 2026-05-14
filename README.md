# husky_assembly_tamp

## Quick Start

From `external/husky_assembly_tamp`:

```bash
pip install -e .
```

## Three-Tier Debug Infrastructure

Use `--stage` to isolate failures by subsystem:

1. **Stage 1** (`--stage 1`): task-space only (no IK, no collision) — verify task-space exploration/connectivity
2. **Stage 2** (`--stage 2`): task-space + IK (no collision) — expose IK feasibility issues
3. **Stage 3** (`--stage 3`): full pipeline (IK + collision) — production-equivalent behavior

Interpretation:
- Stage 1 fails: sampling/metric/tree growth issue in task space.
- Stage 1 passes, Stage 2 fails: IK/projection bottleneck.
- Stage 2 passes, Stage 3 fails: collision feasibility bottleneck.

## Dual-Arm Constrained Planner (Stage 1/2/3)

The `dual_arm_task_space_rrt` package implements a **pose-space RRT** that plans motions for both arms while they grasp a bar, maintaining the rigid relative transform between the grippers throughout the trajectory. Module layout: `core.py` (RRT + IK), `smooth.py` (shortcut smoothing), `run.py` (scene helpers + batch CLI + post-plan output).

### `run.py` — Batch runner (single or multi target)

Runs the planner on one or more gdrive targets, then writes a Markdown report with tables, validation plots, trajectory JSON, and MP4 videos. Requires `--gdrive-state` or `--gdrive-bar-action`.

```bash
# Single BarAction target, headless
python -m husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.run \
  --gdrive-bar-action --targets B1.json --movement M1 --stage 3

# Multi-target batch, GUI off (default)
python -m husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.run \
  --gdrive-bar-action --targets B1.json B2.json B3.json --movement M1 --stage 3

# Gdrive RobotCellState targets
python -m husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.run \
  --gdrive-state --targets B3_approach.json --stage 3

# GUI with per-target slider replay
python -m husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.run \
  --gdrive-bar-action --targets B1.json --gui --visualize-path --stage 3
```

Key flags:

| Flag | Purpose |
|------|---------|
| `--stage {1,2,3}` | Planning tier: 1 = task-space only, 2 = + IK, 3 = + collision |
| `--gui` | Enable PyBullet GUI (default: headless) |
| `--visualize-path` / `--no-visualize-path` | Open slider viewer after each target (when GUI enabled) |
| `--gdrive-state` | Use gdrive RobotCellState convention; `--targets` are state filenames |
| `--gdrive-bar-action` | Use gdrive BarAction inputs; `--targets` are BarAction filenames |
| `--targets NAME [NAME ...]` | Targets to plan (default: `[B1.json]` for BarAction or `[B3_approach.json]` for state) |
| `--movement ID` | Movement selector for `--gdrive-bar-action` (default `M1`) |
| `--gdrive-problem DIR` | Dataset directory under `GDRIVE_DATA_DIRECTORY` |
| `--floating-collision` | Enable bar-obstacle collision in Stage 1 (Stage 3 always has collision) |
| `--lock-renderer-during-search` / `--no-lock-renderer-during-search` | Lock renderer during RRT (default: on) |
| `--smoothing` / `--no-smoothing` | Toggle shortcut smoothing on the planned path |
| `--include-built-bars` | Import already-built bars into the scene as obstacles |
| `--enable-built-bar-collision` | Enable collision on those imported bars |

Hyperparameters (`--max-time`, `--max-iterations`, `--max-attempts`, `--position-res`, `--rotation-res`, `--goal-bias`, `--dist-metric`, `--smooth-iterations`, `--smooth-max-time`, `--endpoint-ik-attempts`, `--joint-continuity-threshold`, etc.) have sensible defaults.

Reports are written to `husky_assembly_tamp/motion_planner/dual_arm_task_space_rrt/reports/`.

### Adapting to a different robot cell / design study

The planner consumes gdrive-convention datasets. To use a **different robot cell or set of targets**:

**1. Dataset directory**

Datasets live under `GDRIVE_DATA_DIRECTORY` (defined near the top of `dual_arm_task_space_rrt/run.py`). Each dataset directory contains `RobotCell.json` and `RobotCellStates/` (single-state inputs) or `BarActions/` (BarAction inputs). Select a dataset at runtime via `--gdrive-problem`.

**2. Target list defaults**

If `--targets` is not passed, the runner falls back to `[B3_approach.json]` for `--gdrive-state` or `[B1.json]` for `--gdrive-bar-action`.

**3. Per-input file naming convention**

`build_gdrive_scene_spec` (single state) expects bodies tagged `active_bar_*`, `active_<other>_*` (rigidly bound to the bar), and `env_*`. Grasps come from FK at the cell state's joint values.

`build_gdrive_bar_action_scene_spec` reads a BarAction JSON; target EE frames come from the selected movement.

**4. Home pose constant**

`MOBILE_BASE_FROM_TOOL0_LEFT_HOME` (`dual_arm_task_space_rrt/run.py`) is the reference left-tool pose used to derive start bar poses. The runner always auto-computes bar-axis rotation via `derive_constrained_start` (`dual_arm_task_space_rrt/core.py`) to find an IK-feasible home pose from the grasp geometry.

> **Tip**: When adapting with Claude Code, point it at the gdrive builders (`build_gdrive_scene_spec`, `build_gdrive_bar_action_scene_spec`) and the `GDRIVE_DATA_DIRECTORY` constant.

## GUI Notes

- Windows/Linux: full interactive sliders (path scrubber, collision replay).
- macOS: interactive sliders are bypassed; auto-runs planning and keeps visualization playback.
- Default headless: `run.py` runs without GUI unless `--gui` is passed.
