# Stage Comparison Debugging Report (20260317_144137)

## Scope

This report compares Stages 1, 2, and 3 across the same seed range.

Run setup:

- Trials per stage: `5` seeds (`0..4`)
- Per-attempt max time: `30.0s`
- Dist metric: `feature`
- Position resolution: `0.01 m`
- Rotation resolution: `0.025 rad`
- Endpoint IK attempts: `20`
- Joint continuity threshold: `0.2 rad`

---

## 1) Workspace Tree Visualization

The first seed is rendered for each stage so the exploration footprint can be compared directly.

### Stage 1
![Stage 1 Tree](_support/tree_structure_stage1_seed0_20260317_144137.png)

### Stage 2
![Stage 2 Tree](_support/tree_structure_stage2_seed0_20260317_144137.png)

### Stage 3
![Stage 3 Tree](_support/tree_structure_stage3_seed0_20260317_144137.png)

Observation:

- Stage 1 isolates task-space exploration.
- Stage 2 shows how dual-arm IK feasibility prunes the same task-space search.
- Stage 3 shows the additional pruning introduced by collision checking.

---

## 2) Trajectory Validation

The first-seed trajectory replay validation plot is included for each stage.

### Stage 1 Validation
![Stage 1 Validation](_support/trajectory_validation_stage1_20260317_144137.png)

- Collision-free: **N/A**, joint continuity: **N/A**, relative transform: **N/A**
- Joint-path source: `reconstructed`

### Stage 2 Validation
![Stage 2 Validation](_support/trajectory_validation_stage2_20260317_144142.png)

- Collision-free: **FAIL**, joint continuity: **PASS**, relative transform: **PASS**
- Joint-path source: `planner`
- Max dq: `0.1792 rad`

### Stage 3 Validation
![Stage 3 Validation](_support/trajectory_validation_stage3_20260317_144214.png)

- Collision-free: **PASS**, joint continuity: **PASS**, relative transform: **PASS**
- Joint-path source: `planner`
- Max dq: `0.1576 rad`

---

## 3) Failure Distribution Analysis

![Failure Distribution Comparison](_support/failure_distribution_comparison_20260317_144137.png)

| Stage | Task-space | IK | Continuity | Collision | Success | Dominant failure |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Stage 1 | 0 | 0 | 0 | 0 | 5 | none |
| Stage 2 | 0 | 0 | 0 | 4 | 1 | collision_failure |
| Stage 3 | 0 | 0 | 0 | 0 | 5 | none |

Interpretation:

- Stage 1 failures are pure task-space failures.
- New IK failures in Stage 2 quantify the cost of enforcing dual-arm feasibility.
- New continuity failures show where seed-chained IK can find a pose path but not a smooth joint realization.
- New collision failures quantify the extra cost of self/environment avoidance once IK already succeeds.

---

## 4) Per-Stage Comparison

![Success Rate Comparison](_support/success_rate_comparison_20260317_144137.png)

![Runtime Comparison](_support/runtime_comparison_20260317_144137.png)

![Planner Breakdown Comparison](_support/planner_breakdown_comparison_20260317_144137.png)

| Stage | Validated success | Path found | Avg runtime (s) | Avg iterations | Avg nodes | Avg poses checked | Avg IK calls | Avg collision hits | Avg max dq |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Stage 1 | 100% | 100% | 0.038 | 16.0 | 1234.4 | 1233.4 | 0.0 | 0.0 | n/a |
| Stage 2 | 20% | 100% | 5.754 | 107.8 | 1789.4 | 1889.4 | 3874.8 | 0.0 | 0.1467 |
| Stage 3 | 100% | 100% | 1.868 | 54.6 | 538.8 | 590.8 | 1198.8 | 33.6 | 0.1399 |

Detailed stage reports:

- `debug_report_stage1_20260317_144137.md`
- `debug_report_stage2_20260317_144137.md`
- `debug_report_stage3_20260317_144137.md`

---

## Final Answer to Debugging Goals

1. **Workspace tree visualization**: Achieved. The report includes one tree image per stage for the same seed.
2. **Failure distribution analysis**: Achieved. Failure categories are compared side by side across all three stages.
3. **Per-stage comparison**: Achieved. Success rate, runtime, bottleneck mix, and planner timing are summarized side by side across Stages 1, 2, and 3.
