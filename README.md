# husky_assembly_tamp

This README is intentionally focused on running and debugging:

`husky_assembly_tamp/motion_planner/trajectory_testbench.py`

All previous public API descriptions were removed and will be rebuilt later.

## Quick Start

From `external/husky_assembly_tamp`:

```bash
pip install -e .
python -m husky_assembly_tamp.motion_planner.trajectory_testbench
```

The testbench defaults to:
- planner backend: `birrt`
- stage: `3` (full planning)
- dataset files under `data/husky_assembly_design_study/250904_transfer_path_test/RobotCellStates`

If your files are elsewhere, pass explicit paths:

```bash
python -m husky_assembly_tamp.motion_planner.trajectory_testbench \
  --grasp-json /path/to/IK_test__GraspTargets.json \
  --start-state /path/to/start_RobotCellState.json \
  --end-state /path/to/end_RobotCellState.json \
  --traj-dir /path/to/output_dir
```

## Three-Tier Debug Infrastructure

Use `--stage` to isolate failures by subsystem.

1. Stage 1 (`--stage 1`): task-space only
- IK: off
- collision: off
- purpose: verify task-space exploration/connectivity without projection/collision noise

2. Stage 2 (`--stage 2`): task-space + IK
- IK: on
- collision: off
- purpose: expose projection/IK feasibility issues

3. Stage 3 (`--stage 3`): full pipeline
- IK: on
- collision: on
- purpose: production-equivalent behavior with collision validation

Recommended debugging sequence:

```bash
# 1) Can the task-space planner connect at all?
python -m husky_assembly_tamp.motion_planner.trajectory_testbench --stage 1 --return-task-path

# 2) If stage 1 succeeds, does IK/projection break it?
python -m husky_assembly_tamp.motion_planner.trajectory_testbench --stage 2

# 3) If stage 2 succeeds, test full collision-aware planning
python -m husky_assembly_tamp.motion_planner.trajectory_testbench --stage 3
```

Interpretation:
- Stage 1 fails: sampling/metric/tree growth issue in task space.
- Stage 1 passes, Stage 2 fails: IK/projection bottleneck.
- Stage 2 passes, Stage 3 fails: collision feasibility bottleneck.

## Useful Runtime Knobs

```bash
python -m husky_assembly_tamp.motion_planner.trajectory_testbench \
  --stage 3 \
  --dist-metric feature \
  --ladder-search shortest \
  --goal-bias 0.1 \
  --guide-bias 0.2 \
  --max-time 30 \
  --max-iterations 2000 \
  --max-attempts 5
```

Common options:
- `--planner {birrt,constrained_bimanual}`
- `--dist-metric {feature,pose6d}`
- `--ladder-search {shortest,enumerate}`
- `--expand-delta <rad>` and `--start-goal-delta <rad>`
- `--warm-start-first` (Stage 3 warm-start behavior)
- `--no-gui` (headless)

## Outputs and Profiling

Each run provides:
- log file: `husky_assembly_tamp/logs/trajectory_testbench.log`
- cProfile dump: `husky_assembly_tamp/motion_planner/plan_profile.prof`
- Snakeviz launch attempt for profile browsing
- saved trajectory JSON in `--traj-dir` as `testbench_<timestamp>_JointTrajectory.json` (joint-space paths only)

## Stage 1 Benchmarking

For the standalone Stage 1 pose-space planner, use:

`husky_assembly_tamp/motion_planner/stage1/minimal_rrt.py`

For repeated trials, plots, profile dumps, and Markdown report generation, use:

`husky_assembly_tamp/motion_planner/stage1/debug_runner.py`

Run a single Stage 1 debug session:

```bash
python -m husky_assembly_tamp.motion_planner.stage1.minimal_rrt \
  --position-res 0.1 \
  --rotation-res 0.2 \
  --max-time 30 \
  --max-iterations 2000 \
  --max-attempts 5
```

Run the Stage 1 benchmarking batch headlessly:

```bash
python -m husky_assembly_tamp.motion_planner.stage1.debug_runner \
  --analysis-trials 10 \
  --analysis-seed-start 0 \
  --position-res 0.1 \
  --rotation-res 0.2 \
  --max-time 30 \
  --max-iterations 2000 \
  --max-attempts 5
```

Outputs are written under:

`husky_assembly_tamp/motion_planner/stage1/reports`

Expected benchmarking artifacts:
- `failure_analysis_<timestamp>.csv`
- `failure_analysis_<timestamp>.json`
- `failure_distribution_<timestamp>.png`
- `stage1_success_<timestamp>.png`
- `runtime_by_seed_<timestamp>.png`
- `tree_structure_stage1_seed<seed>_<timestamp>.png`
- `planner_breakdown_<timestamp>.png`
- `plan_profile_seed<seed>_<timestamp>.prof`
- `plan_profile_seed<seed>_<timestamp>.txt`
- `debug_report_<timestamp>.md`

Useful options:
- `--gui` to run the debug runner with PyBullet GUI; headless is the default
- `--lock-renderer-during-search` to suppress live redraw during tree expansion and only visualize the result afterward
- `--no-floating-collision` to disable floating-body collision checks
- `--profile-seed <seed>` to choose which analysis seed gets full `cProfile` capture

## GUI Notes

- Windows/Linux: full interactive sliders/buttons (`Plan Path`, start/end pose sliders, path scrubber, load trajectory).
- macOS: interactive sliders are bypassed; testbench auto-runs planning and keeps visualization playback.
- `--no-gui`: fully headless execution.
