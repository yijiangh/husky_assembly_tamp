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

## Stage 1/2/3 Benchmarking

For the standalone Stage 1 pose-space planner, use:

`husky_assembly_tamp/motion_planner/stage1/minimal_rrt.py`

For repeated trials, plots, profile dumps, and Markdown report generation, use:

`husky_assembly_tamp/motion_planner/stage1/debug_runner.py`

Run a single Stage 1 debug session:

```bash
python -m husky_assembly_tamp.motion_planner.stage1.minimal_rrt \
  --stage 1 \
  --position-res 0.1 \
  --rotation-res 0.2 \
  --max-time 30 \
  --max-iterations 2000 \
  --max-attempts 5
```

Run a single Stage 2 debug session:

```bash
python -m husky_assembly_tamp.motion_planner.stage1.minimal_rrt \
  --stage 2 \
  --position-res 0.1 \
  --rotation-res 0.2 \
  --endpoint-ik-attempts 20 \
  --max-time 30 \
  --max-iterations 2000 \
  --max-attempts 5
```

Run a single Stage 3 debug session:

```bash
python -m husky_assembly_tamp.motion_planner.stage1.minimal_rrt \
  --stage 3 \
  --position-res 0.1 \
  --rotation-res 0.2 \
  --endpoint-ik-attempts 20 \
  --max-time 30 \
  --max-iterations 2000 \
  --max-attempts 5
```

Run the Stage 3 benchmarking batch headlessly:

```bash
python -m husky_assembly_tamp.motion_planner.stage1.debug_runner \
  --stage 3 \
  --analysis-trials 10 \
  --analysis-seed-start 0 \
  --position-res 0.1 \
  --rotation-res 0.2 \
  --endpoint-ik-attempts 20 \
  --max-time 30 \
  --max-iterations 2000 \
  --max-attempts 5
```

Run a comparative Stage 1/2/3 benchmarking batch headlessly:

```bash
python -m husky_assembly_tamp.motion_planner.stage1.debug_runner \
  --compare-stages \
  --analysis-trials 10 \
  --analysis-seed-start 0 \
  --position-res 0.1 \
  --rotation-res 0.2 \
  --endpoint-ik-attempts 20 \
  --max-time 30 \
  --max-iterations 2000 \
  --max-attempts 5
```

Outputs are written under:

`husky_assembly_tamp/motion_planner/stage1/reports`

Top-level `reports/` is intended to contain the Markdown reports you actually open first.

All supporting CSV/JSON/PNG/profile artifacts are written under:

`husky_assembly_tamp/motion_planner/stage1/reports/_support`

Expected benchmarking artifacts:
- `_support/failure_analysis_stage<stage>_<timestamp>.csv`
- `_support/failure_analysis_stage<stage>_<timestamp>.json`
- `_support/failure_distribution_stage<stage>_<timestamp>.png`
- `_support/stage<stage>_success_<timestamp>.png`
- `_support/runtime_by_seed_stage<stage>_<timestamp>.png`
- `_support/tree_structure_stage<stage>_seed<seed>_<timestamp>.png`
- `_support/planner_breakdown_stage<stage>_<timestamp>.png`
- `_support/trajectory_validation_stage<stage>_<timestamp>.png`
- `_support/plan_profile_stage<stage>_seed<seed>_<timestamp>.prof`
- `_support/plan_profile_stage<stage>_seed<seed>_<timestamp>.txt`
- `debug_report_stage<stage>_<timestamp>.md`
- `_support/stage_comparison_<timestamp>.csv`
- `_support/stage_comparison_<timestamp>.json`
- `_support/failure_distribution_comparison_<timestamp>.png`
- `_support/success_rate_comparison_<timestamp>.png`
- `_support/runtime_comparison_<timestamp>.png`
- `_support/planner_breakdown_comparison_<timestamp>.png`
- `stage_comparison_report_<timestamp>.md`

Useful options:
- `--gui` to run the debug runner with PyBullet GUI; headless is the default
- `--stage 1`, `--stage 2`, or `--stage 3` to choose between task-space only, IK-enabled, and collision-aware planning
- `--lock-renderer-during-search` to suppress live redraw during tree expansion and only visualize the result afterward
- `--endpoint-ik-attempts <N>` to increase endpoint seed search for Stage 2/3
- `--floating-collision` to enable floating-body collision checks in Stage 1; Stage 3 always enables robot collision checks
- `--compare-stages` to run Stage 1, Stage 2, and Stage 3 in one batch and emit a cross-stage comparison report
- `--profile-seed <seed>` to choose which analysis seed gets full `cProfile` capture

## GUI Notes

- Windows/Linux: full interactive sliders/buttons (`Plan Path`, start/end pose sliders, path scrubber, load trajectory).
- macOS: interactive sliders are bypassed; testbench auto-runs planning and keeps visualization playback.
- `--no-gui`: fully headless execution.
