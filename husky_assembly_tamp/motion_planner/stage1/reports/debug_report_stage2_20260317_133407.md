# Stage 2 Debugging Report (20260317_133407)

## Scope

This report summarizes results from:

- `_support/failure_analysis_stage2_20260317_133407.json`
- `_support/failure_analysis_stage2_20260317_133407.csv`
- `_support/failure_distribution_stage2_20260317_133407.png`
- `_support/stage2_success_20260317_133407.png`
- `_support/runtime_by_seed_stage2_20260317_133407.png`
- `_support/tree_structure_stage2_seed0_20260317_133407.png`
- `_support/trajectory_validation_stage2_20260317_133413.png`
- `_support/planner_breakdown_stage2_20260317_133407.png`
- `_support/plan_profile_stage2_seed0_20260317_133407.txt`

Run setup:

- Trials: `5` seeds (`0..4`)
- Per-attempt max time: `3.0s`
- Dist metric: `feature`
- Position resolution: `0.01 m`
- Rotation resolution: `0.025 rad`
- Endpoint IK attempts: `20`
- Joint continuity threshold: `0.2 rad`
- Post-plan refinement: `on`
- Initial refine position resolution: `0.005 m`
- Initial refine rotation resolution: `0.0125 rad`
- Refine max passes: `2`
- Collision: `off`

---

## 1) Workspace Tree Visualization

### Stage 2 (seed 0)
![Stage 2 Tree](_support/tree_structure_stage2_seed0_20260317_133407.png)

Observation:

- The tree image shows the task-space exploration footprint used by the single-tree Stage 2 RRT.
- This is the quickest way to see whether the sampler is exploring broadly or repeatedly getting trapped near the start or obstacle boundary.

---

## 2) Trajectory Validation

![Trajectory Validation](_support/trajectory_validation_stage2_20260317_133413.png)

First-seed validation summary:

- Collision-free replay: **FAIL**
- Joint continuity: **PASS**
- Relative transform consistency: **PASS**
- Joint-path source: `planner`
- Refinement status: `already_continuous`
- Refinement max dq: `0.1792 -> 0.1792 rad`
- Refinement waypoints: `386 -> 386`

---

## 3) Failure Distribution Analysis

### Distribution plot
![Failure Distribution](_support/failure_distribution_stage2_20260317_133407.png)

From `summary.counts`:

- `task_space_failure`: **0 / 5** (0%)
- `ik_failure`: **0 / 5** (0%)
- `continuity_failure`: **2 / 5** (40%)
- `collision_failure`: **2 / 5** (40%)
- `success`: **1 / 5** (20%)

### Bottleneck conclusion

Dominant failure mode in this run is **continuity_failure**.

---

## 4) Runtime and Bottleneck Breakdown

### Validated-success plot
![Stage 2 Success Rate](_support/stage2_success_20260317_133407.png)

### Runtime-by-seed plot
![Runtime by Seed](_support/runtime_by_seed_stage2_20260317_133407.png)

### Planner breakdown plot
![Planner Breakdown](_support/planner_breakdown_stage2_20260317_133407.png)

From `summary`:

- Stage 2 validated success rate: **20%**
- Stage 2 task-space path-found rate: **60%**
- Stage 2 avg runtime: **2.546 s**
- Stage 2 avg iterations: **40.6**
- Stage 2 avg nodes created: **965.6**
- Stage 2 avg poses checked: **1002.2**
- Stage 2 avg IK calls: **2039.8**
- Stage 2 avg IK failures: **71.2**
- Stage 2 refinement used in **0 / 5** trials
- Stage 2 avg max dq: **0.1390 -> 0.1390 rad**

Detailed `cProfile` summary: `_support/plan_profile_stage2_seed0_20260317_133407.txt`

Interpretation:

- The runtime plot shows whether failures correlate with long searches or early exits.
- The planner breakdown plot shows which internal planner phases consume the most time on average.
- The saved `cProfile` text report is the lower-level function-call view for deeper bottleneck inspection.

---

## Final Answer to Debugging Goals

1. **Workspace tree visualization**: Achieved. A Stage 2 tree image is generated for the first seed in the batch.
2. **Failure distribution analysis**: Achieved. Successes and failures are categorized across seeds and visualized.
3. **Per-stage trajectory validation support**: Achieved. The report links the first-seed validation replay plot and validation summary.
