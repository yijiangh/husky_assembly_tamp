# Dual ARM Planner Change Log

## v0.1 - 2026-03-07

This tag marks the current improved baseline of the dual-arm constrained planner.

### Major Improvements

- Added staged planning modes for diagnosis and bottleneck isolation:
  - Stage 1: task-space only (no IK, no collision)
  - Stage 2: IK enabled, collision disabled
  - Stage 3: full planning (IK + collision)
- Added feature-point distance metric as the default task-space metric, while keeping `pose6d` as an option.
- Added guided sampling:
  - goal-bias sampling
  - optional guide-pose sampling for Stage 3
- Added Stage-3 warm-start pipeline seeded by Stage-2 paths.
- Added warm-start smoothing/interpolation hooks to shortcut expensive full planning when feasible.

### Performance and Robustness Changes

- Added KD-tree nearest-neighbor acceleration for pose-tree expansion when feature vectors are available.
- Added IK expansion cache keyed by pose and expansion parameters.
- Added collision check cache keyed by quantized joint configuration.
- Added capsule-path decimation before ladder expansion to reduce ladder graph size.
- Added fast projection-chain recovery to bypass full ladder graph solve when continuous projection succeeds.
- Added two-pass ladder expansion strategy (small pass first, then full pass only when necessary).
- Added dynamic-programming shortest-path ladder solver as a faster alternative to full enumeration.

### Debugging and Profiling Infrastructure

- Added internal planner operation profiler (`PlanProfiler`) with per-operation timing report.
- Added explicit failure attribution counters (RRT-connect failures vs ladder-search failures).
- Extended testbench profiling (`cProfile` output and Snakeviz integration).
- Expanded testbench CLI knobs for stage selection, metric selection, ladder strategy, warm-start, biases, and runtime limits.
- Added macOS/headless-friendly planning flow and GUI bypass options.

### Backend/API Integration

- Extended backend forwarding so planner options are fully passed through:
  - staging flags
  - metric selection
  - ladder strategy/parameters
  - warm-start and guide path options

### Documentation

- Added algorithm/design documentation in `docs/algorithm_description.tex` (+ generated artifacts).

