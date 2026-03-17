# Resolution Sweep Report (20260317_125511)

## Scope

This report compares Stage 1, Stage 2, and Stage 3 across multiple task-space interpolation resolutions.

Run setup:

- Trials per stage/resolution pair: `5` seeds (`0..4`)
- Dist metric: `feature`
- Endpoint IK attempts: `20`
- Joint continuity threshold: `0.5 rad`
- Post-plan refinement: `on`

CSV: `_support/resolution_sweep_20260317_125511.csv`
JSON: `_support/resolution_sweep_20260317_125511.json`

---

## Results

| Resolution | Stage | Validated success | Path found | Avg runtime (s) | Dominant failure | Avg continuity rejects | Avg max dq (coarse -> final) |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| pos=0.050m rot=0.100rad | Stage 1 | 100% | 100% | 0.010 | none | 0.0 | 0.0000 -> 0.0000 |
| pos=0.050m rot=0.100rad | Stage 2 | 40% | 80% | 1.280 | collision_failure | 4.2 | 0.4083 -> 0.4083 |
| pos=0.050m rot=0.100rad | Stage 3 | 80% | 80% | 0.872 | collision_failure | 4.8 | 0.3998 -> 0.3998 |
| pos=0.030m rot=0.070rad | Stage 1 | 100% | 100% | 0.015 | none | 0.0 | 0.0000 -> 0.0000 |
| pos=0.030m rot=0.070rad | Stage 2 | 20% | 40% | 1.698 | continuity_failure | 1.0 | 0.3464 -> 0.3464 |
| pos=0.030m rot=0.070rad | Stage 3 | 40% | 40% | 1.108 | collision_failure | 1.2 | 0.3117 -> 0.3117 |
| pos=0.020m rot=0.050rad | Stage 1 | 100% | 100% | 0.025 | none | 0.0 | 0.0000 -> 0.0000 |
| pos=0.020m rot=0.050rad | Stage 2 | 20% | 80% | 1.872 | collision_failure | 0.4 | 0.3163 -> 0.3163 |
| pos=0.020m rot=0.050rad | Stage 3 | 100% | 100% | 1.219 | none | 0.4 | 0.2696 -> 0.2696 |

Interpretation:

- Stage 1 should remain largely insensitive to the continuity threshold because it does not plan in joint space.
- Stage 2 indicates whether finer Cartesian interpolation is enough to recover validated smooth IK paths.
- Stage 3 indicates whether continuity and collision can both be satisfied at the same resolution before a ladder-graph fallback is needed.
