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

Two scripts implement a **pose-space RRT** that plans motions for both arms while they grasp a bar, maintaining the rigid relative transform between the grippers throughout the trajectory.

### `minimal_rrt.py` — Single-trial planner

Plans one start-to-goal motion and optionally opens a GUI for interactive replay. Requires a gdrive input.

```bash
python -m husky_assembly_tamp.motion_planner.stage1.minimal_rrt \
  --gdrive-bar-action B1.json --stage 3 --no-gui
```

Key flags:

| Flag | Purpose |
|------|---------|
| `--stage {1,2,3}` | Planning tier: 1 = task-space only, 2 = + IK, 3 = + collision |
| `--no-gui` | Headless mode (omit for PyBullet GUI with interactive path slider) |
| `--gdrive-state PATH` | Gdrive RobotCellState (e.g. `B3_approach.json`); grasps come from FK |
| `--gdrive-bar-action PATH` | Gdrive BarAction; target EE frames from the selected movement |
| `--movement ID` | Movement selector for `--gdrive-bar-action` (default `M1`) |
| `--gdrive-problem DIR` | Dataset directory under `GDRIVE_DATA_DIRECTORY` when the input is a bare filename |
| `--floating-collision` | Enable bar-obstacle collision in Stage 1 (Stage 3 always has collision) |
| `--lock-renderer-during-search` | Suppress live tree redraw; show result only (default: on) |
| `--smoothing` / `--no-smoothing` | Toggle shortcut smoothing on the planned path |

Hyperparameters (`--max-time`, `--max-iterations`, `--max-attempts`, `--position-res`, `--rotation-res`, `--goal-bias`, `--dist-metric`, `--smooth-iterations`, `--smooth-max-time`, `--endpoint-ik-attempts`, `--joint-continuity-threshold`, etc.) have sensible defaults and can be left untouched for most runs.

### `real_state_study.py` — Batch benchmark over design targets

Loops over multiple bar targets, runs the planner on each, and produces a Markdown report with tables, validation plots, trajectory JSON, and MP4 videos. Requires `--gdrive` or `--gdrive-bar-action`.

```bash
# Run gdrive RobotCellState targets
python -m husky_assembly_tamp.motion_planner.stage1.real_state_study \
  --gdrive --stage 3 --targets B3_approach.json

# Run gdrive BarAction targets
python -m husky_assembly_tamp.motion_planner.stage1.real_state_study \
  --gdrive-bar-action --stage 3 --targets B1.json
```

Key flags (on top of the shared ones):

| Flag | Purpose |
|------|---------|
| `--gdrive` | Use gdrive RobotCellState convention; `--targets` are state filenames |
| `--gdrive-bar-action` | Use gdrive BarAction inputs; `--targets` are BarAction filenames |
| `--gdrive-problem DIR` | Dataset directory under `GDRIVE_DATA_DIRECTORY` |
| `--movement ID` | Movement selector for `--gdrive-bar-action` (default `M1`) |
| `--targets NAME [NAME ...]` | Which targets to benchmark |
| `--gui` | Enable PyBullet GUI (headless by default) |
| `--visualize-path` / `--no-visualize-path` | Open slider viewer after each target (when GUI enabled) |
| `--include-built-bars` | Import already-built bars into the scene as obstacles |
| `--enable-built-bar-collision` | Enable collision on those imported bars |
| `--auto-home-pose` / `--no-auto-home-pose` | Auto-compute optimal start bar pose from grasp geometry |

Reports are written to `husky_assembly_tamp/motion_planner/stage1/reports/`.

### Adapting to a different robot cell / design study

The planner consumes gdrive-convention datasets. To use a **different robot cell or set of targets**:

**1. Dataset directory**

Datasets live under `GDRIVE_DATA_DIRECTORY` (defined near the top of `minimal_rrt.py`). Each dataset directory contains `RobotCell.json` and `RobotCellStates/` (single-state inputs) or `BarActions/` (BarAction inputs). Select a dataset at runtime via `--gdrive-problem`.

**2. Target list defaults**

`DEFAULT_TARGET_NAMES` in `real_state_study.py` is the fallback when `--targets` is not passed. `GDRIVE_DEFAULT_TARGETS` and `GDRIVE_DEFAULT_BAR_ACTION_TARGETS` are mode-specific overrides used in gdrive flows.

**3. Per-input file naming convention**

`build_gdrive_scene_spec` (single state) expects bodies tagged `active_bar_*`, `active_<other>_*` (rigidly bound to the bar), and `env_*`. Grasps come from FK at the cell state's joint values.

`build_gdrive_bar_action_scene_spec` reads a BarAction JSON; target EE frames come from the selected movement.

**4. Home pose constant**

`MOBILE_BASE_FROM_TOOL0_LEFT_HOME` (minimal_rrt.py) is the reference left-tool pose used to derive start bar poses. If your robot's home configuration is significantly different, update this or rely on `--auto-home-pose` (enabled by default in `real_state_study.py`) which computes it from the grasp geometry.

> **Tip**: When adapting with Claude Code, point it at the gdrive builders (`build_gdrive_scene_spec`, `build_gdrive_bar_action_scene_spec`) and the `GDRIVE_DATA_DIRECTORY` constant.

## GUI Notes

- Windows/Linux: full interactive sliders (path scrubber, collision replay).
- macOS: interactive sliders are bypassed; auto-runs planning and keeps visualization playback.
- `--no-gui` / headless: fully headless execution (default for `real_state_study.py`).
