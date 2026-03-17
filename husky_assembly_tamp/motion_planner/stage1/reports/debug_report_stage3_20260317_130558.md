# Stage 3 Debugging Report (20260317_130558)

## Scope

This report summarizes results from:

- `_support/failure_analysis_stage3_20260317_130558.json`
- `_support/failure_analysis_stage3_20260317_130558.csv`
- `_support/failure_distribution_stage3_20260317_130558.png`
- `_support/stage3_success_20260317_130558.png`
- `_support/runtime_by_seed_stage3_20260317_130558.png`
- `_support/tree_structure_stage3_seed0_20260317_130558.png`
- `_support/trajectory_validation_stage3_20260317_130613.png`
- `_support/planner_breakdown_stage3_20260317_130558.png`
- `_support/plan_profile_stage3_seed0_20260317_130558.txt`

Run setup:

- Trials: `5` seeds (`0..4`)
- Per-attempt max time: `3.0s`
- Dist metric: `feature`
- Position resolution: `0.02 m`
- Rotation resolution: `0.05 rad`
- Endpoint IK attempts: `20`
- Joint continuity threshold: `0.5 rad`
- Post-plan refinement: `on`
- Initial refine position resolution: `0.01 m`
- Initial refine rotation resolution: `0.025 rad`
- Refine max passes: `2`
- Collision: `on`

---

## 1) Workspace Tree Visualization

### Stage 3 (seed 0)
![Stage 3 Tree](_support/tree_structure_stage3_seed0_20260317_130558.png)

Observation:

- The tree image shows the task-space exploration footprint used by the single-tree Stage 3 RRT.
- This is the quickest way to see whether the sampler is exploring broadly or repeatedly getting trapped near the start or obstacle boundary.

---

## 2) Trajectory Validation

![Trajectory Validation](_support/trajectory_validation_stage3_20260317_130613.png)

First-seed validation summary:

- Collision-free replay: **PASS**
- Joint continuity: **PASS**
- Relative transform consistency: **PASS**
- Joint-path source: `planner`
- Refinement status: `already_continuous`
- Refinement max dq: `0.2842 -> 0.2842 rad`
- Refinement waypoints: `127 -> 127`

---

## 3) Failure Distribution Analysis

### Distribution plot
![Failure Distribution](_support/failure_distribution_stage3_20260317_130558.png)

From `summary.counts`:

- `task_space_failure`: **0 / 5** (0%)
- `ik_failure`: **0 / 5** (0%)
- `continuity_failure`: **0 / 5** (0%)
- `collision_failure`: **0 / 5** (0%)
- `success`: **5 / 5** (100%)

### Bottleneck conclusion

Dominant failure mode in this run is **none**.

---

## 4) Runtime and Bottleneck Breakdown

### Validated-success plot
![Stage 3 Success Rate](_support/stage3_success_20260317_130558.png)

### Runtime-by-seed plot
![Runtime by Seed](_support/runtime_by_seed_stage3_20260317_130558.png)

### Planner breakdown plot
![Planner Breakdown](_support/planner_breakdown_stage3_20260317_130558.png)

From `summary`:

- Stage 3 validated success rate: **100%**
- Stage 3 task-space path-found rate: **100%**
- Stage 3 avg runtime: **1.135 s**
- Stage 3 avg iterations: **54.6**
- Stage 3 avg nodes created: **304.6**
- Stage 3 avg poses checked: **356.4**
- Stage 3 avg IK calls: **730.4**
- Stage 3 avg IK failures: **35.2**
- Stage 3 refinement used in **0 / 5** trials
- Stage 3 avg max dq: **0.2696 -> 0.2696 rad**
- Stage 3 avg collision hits: **34.8**

Detailed `cProfile` summary: `_support/plan_profile_stage3_seed0_20260317_130558.txt`

Interpretation:

- The runtime plot shows whether failures correlate with long searches or early exits.
- The planner breakdown plot shows which internal planner phases consume the most time on average.
- The saved `cProfile` text report is the lower-level function-call view for deeper bottleneck inspection.

---

## Final Answer to Debugging Goals

1. **Workspace tree visualization**: Achieved. A Stage 3 tree image is generated for the first seed in the batch.
2. **Failure distribution analysis**: Achieved. Successes and failures are categorized across seeds and visualized.
3. **Per-stage trajectory validation support**: Achieved. The report links the first-seed validation replay plot and validation summary.
