# Stage 1 Debugging Report (20260317_144137)

## Scope

This report summarizes results from:

- `_support/failure_analysis_stage1_20260317_144137.json`
- `_support/failure_analysis_stage1_20260317_144137.csv`
- `_support/failure_distribution_stage1_20260317_144137.png`
- `_support/stage1_success_20260317_144137.png`
- `_support/runtime_by_seed_stage1_20260317_144137.png`
- `_support/tree_structure_stage1_seed0_20260317_144137.png`
- `_support/trajectory_validation_stage1_20260317_144137.png`
- `_support/planner_breakdown_stage1_20260317_144137.png`
- `_support/plan_profile_stage1_seed0_20260317_144137.txt`

Run setup:

- Trials: `5` seeds (`0..4`)
- Per-attempt max time: `30.0s`
- Dist metric: `feature`
- Position resolution: `0.01 m`
- Rotation resolution: `0.025 rad`
- Collision: `off`

---

## 1) Workspace Tree Visualization

### Stage 1 (seed 0)
![Stage 1 Tree](_support/tree_structure_stage1_seed0_20260317_144137.png)

Observation:

- The tree image shows the task-space exploration footprint used by the single-tree Stage 1 RRT.
- This is the quickest way to see whether the sampler is exploring broadly or repeatedly getting trapped near the start or obstacle boundary.

---

## 2) Trajectory Validation

![Trajectory Validation](_support/trajectory_validation_stage1_20260317_144137.png)

First-seed validation summary:

- Collision-free replay: **N/A**
- Joint continuity: **N/A**
- Relative transform consistency: **N/A**
- Joint-path source: `reconstructed`

---

## 3) Failure Distribution Analysis

### Distribution plot
![Failure Distribution](_support/failure_distribution_stage1_20260317_144137.png)

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
![Stage 1 Success Rate](_support/stage1_success_20260317_144137.png)

### Runtime-by-seed plot
![Runtime by Seed](_support/runtime_by_seed_stage1_20260317_144137.png)

### Planner breakdown plot
![Planner Breakdown](_support/planner_breakdown_stage1_20260317_144137.png)

From `summary`:

- Stage 1 validated success rate: **100%**
- Stage 1 task-space path-found rate: **100%**
- Stage 1 avg runtime: **0.038 s**
- Stage 1 avg iterations: **16.0**
- Stage 1 avg nodes created: **1234.4**
- Stage 1 avg poses checked: **1233.4**

Detailed `cProfile` summary: `_support/plan_profile_stage1_seed0_20260317_144137.txt`

Interpretation:

- The runtime plot shows whether failures correlate with long searches or early exits.
- The planner breakdown plot shows which internal planner phases consume the most time on average.
- The saved `cProfile` text report is the lower-level function-call view for deeper bottleneck inspection.

---

## Final Answer to Debugging Goals

1. **Workspace tree visualization**: Achieved. A Stage 1 tree image is generated for the first seed in the batch.
2. **Failure distribution analysis**: Achieved. Successes and failures are categorized across seeds and visualized.
3. **Per-stage trajectory validation support**: Achieved. The report links the first-seed validation replay plot and validation summary.
