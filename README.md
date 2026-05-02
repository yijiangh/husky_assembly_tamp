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

Plans one start-to-goal motion and optionally opens a GUI for interactive replay.

```bash
python -m husky_assembly_tamp.motion_planner.stage1.minimal_rrt \
  --stage 3 --no-gui
```

Key flags:

| Flag | Purpose |
|------|---------|
| `--stage {1,2,3}` | Planning tier: 1 = task-space only, 2 = + IK, 3 = + collision |
| `--no-gui` | Headless mode (omit for PyBullet GUI with interactive path slider) |
| `--grasp-json PATH` | Grasp targets JSON (default: built-in test file) |
| `--start-state PATH` | Start RobotCellState JSON |
| `--end-state PATH` | Goal RobotCellState JSON |
| `--swap-grasps` | Swap the first two grasps from the grasp JSON |
| `--floating-collision` | Enable bar-obstacle collision in Stage 1 (Stage 3 always has collision) |
| `--lock-renderer-during-search` | Suppress live tree redraw; show result only (default: on) |
| `--smoothing` / `--no-smoothing` | Toggle shortcut smoothing on the planned path |

Hyperparameters (`--max-time`, `--max-iterations`, `--max-attempts`, `--position-res`, `--rotation-res`, `--goal-bias`, `--dist-metric`, `--smooth-iterations`, `--smooth-max-time`, `--endpoint-ik-attempts`, `--joint-continuity-threshold`, etc.) have sensible defaults and can be left untouched for most runs.

### `real_state_study.py` — Batch benchmark over design targets

Loops over multiple bar targets from the design study, runs the planner on each, and produces a Markdown report with tables, validation plots, trajectory JSON, and MP4 videos.

```bash
# Run all default targets
python -m husky_assembly_tamp.motion_planner.stage1.real_state_study --stage 3

# Run a subset of targets
python -m husky_assembly_tamp.motion_planner.stage1.real_state_study \
  --stage 3 --targets G1 V1 H1

# Diagnose endpoint IK without planning
python -m husky_assembly_tamp.motion_planner.stage1.real_state_study \
  --stage 3 --diagnose-endpoint-ik both --targets G1
```

Key flags (on top of the shared ones):

| Flag | Purpose |
|------|---------|
| `--targets NAME [NAME ...]` | Which bar targets to benchmark (default: G1 G2 G3 G4 V1 V2 H1 D1 V3) |
| `--design-root PATH` | Root directory of the design study data |
| `--robot-cell-json PATH` | Path to RobotCell.json (for bar mesh loading) |
| `--gui` | Enable PyBullet GUI (headless by default) |
| `--visualize-path` / `--no-visualize-path` | Open slider viewer after each target (when GUI enabled) |
| `--include-built-bars` | Import already-built bars into the scene as obstacles |
| `--enable-built-bar-collision` | Enable collision on those imported bars |
| `--diagnose-endpoint-ik {start,goal,both}` | Skip planning; only check if start/goal IK is feasible |
| `--auto-home-pose` / `--no-auto-home-pose` | Auto-compute optimal start bar pose from grasp geometry |

Reports are written to `husky_assembly_tamp/motion_planner/stage1/reports/`.

### Adapting to a different robot cell / design study

The default data paths and target names are hardcoded for the current antenna assembly design study. To use a **different robot cell or set of targets**, you need to update a few constants:

**1. Default data paths in `minimal_rrt.py`**

`build_default_paths()` (line ~1405) returns the default grasp JSON, start state, and end state paths:

```python
# minimal_rrt.py, around line 1405
def build_default_paths() -> Tuple[str, str, str]:
    robot_cell_dir = os.path.join(DATA_DIR, "husky_assembly_design_study",
                                   "250904_transfer_path_test", "RobotCellStates")
    grasp_json = os.path.join(robot_cell_dir, "IK_test__GraspTargets.json")
    start_state = os.path.join(robot_cell_dir, "IK_test__20250905_101010_RobotCellState.json")
    end_state = os.path.join(robot_cell_dir, "IK_test__20250909_235058_RobotCellState.json")
    return grasp_json, start_state, end_state
```

Update these paths to point to your robot cell state directory, or override at runtime with `--grasp-json`, `--start-state`, `--end-state`.

**2. Design study root in `real_state_study.py`**

`default_design_root()` (line ~51) returns the root directory where per-target data lives:

```python
# real_state_study.py, around line 51
def default_design_root() -> str:
    return os.path.join(DATA_DIR, "husky_assembly_design_study",
                         "250929_New_Antenna_with_GH_RH_Packed")
```

Override at runtime with `--design-root PATH`, or update the function.

**3. Target names**

Two lists define the known bar targets:

- `DESIGN_STUDY_BAR_SEQUENCE` in `minimal_rrt.py` (line ~84) — ordered list of all bar names, used for built-bar import ordering
- `DEFAULT_TARGET_NAMES` in `real_state_study.py` (line ~46) — default `--targets` list

If your design study has different bars, update both lists. `DESIGN_STUDY_BAR_SEQUENCE` must include every bar name that can appear, in assembly order. `DEFAULT_TARGET_NAMES` should list the bars you want to benchmark by default.

**4. Per-target file naming convention**

`build_real_design_goal_spec()` (minimal_rrt.py line ~407) expects this layout under `<design-root>/RobotCellStates/`:

```
<design-root>/
  RobotCell.json                          # bar mesh definitions
  RobotCellStates/
    <TARGET>_RobotCellState.json          # joint state + base frame at goal
    <TARGET>_GraspTargets.json            # grasp transforms for this bar
```

Where `<TARGET>` matches the name in `DESIGN_STUDY_BAR_SEQUENCE` (e.g., `G1`, `V1`, `H1`).

**5. Home pose constant**

`MOBILE_BASE_FROM_TOOL0_LEFT_HOME` (minimal_rrt.py line ~79) is the reference left-tool pose used to derive start bar poses. If your robot's home configuration is significantly different, you may need to update this or rely on `--auto-home-pose` (enabled by default in `real_state_study.py`) which computes it from the grasp geometry.

> **Tip**: If you're adapting to a new design study with Claude Code, point it at the constants listed above and describe your new data layout. The changes are mechanical: update file paths, target name lists, and optionally the home pose.

## GUI Notes

- Windows/Linux: full interactive sliders (path scrubber, collision replay).
- macOS: interactive sliders are bypassed; auto-runs planning and keeps visualization playback.
- `--no-gui` / headless: fully headless execution (default for `real_state_study.py`).
