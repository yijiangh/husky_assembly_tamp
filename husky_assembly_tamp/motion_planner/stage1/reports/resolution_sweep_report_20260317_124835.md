# Resolution Sweep Report (20260317_124835)

## Scope

This report compares Stage 1, Stage 2, and Stage 3 across multiple task-space interpolation resolutions.

Run setup:

- Trials per stage/resolution pair: `1` seeds (`0..0`)
- Dist metric: `feature`
- Endpoint IK attempts: `20`
- Joint continuity threshold: `0.5 rad`
- Post-plan refinement: `on`

CSV: `_support/resolution_sweep_20260317_124835.csv`
JSON: `_support/resolution_sweep_20260317_124835.json`

---

## Results

| Resolution | Stage | Validated success | Path found | Avg runtime (s) | Dominant failure | Avg continuity rejects | Avg max dq (coarse -> final) |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| pos=0.050m rot=0.100rad | Stage 1 | 100% | 100% | 0.015 | none | 0.0 | 0.0000 -> 0.0000 |
| pos=0.050m rot=0.100rad | Stage 2 | 0% | 100% | 1.014 | collision_failure | 6.0 | 0.4176 -> 0.4176 |
| pos=0.050m rot=0.100rad | Stage 3 | 100% | 100% | 1.177 | none | 6.0 | 0.4078 -> 0.4078 |
| pos=0.030m rot=0.070rad | Stage 1 | 100% | 100% | 0.021 | none | 0.0 | 0.0000 -> 0.0000 |
| pos=0.030m rot=0.070rad | Stage 2 | 0% | 100% | 0.625 | collision_failure | 1.0 | 0.4708 -> 0.4708 |
| pos=0.030m rot=0.070rad | Stage 3 | 0% | 0% | 1.214 | collision_failure | 1.0 | 0.0000 -> 0.0000 |
| pos=0.020m rot=0.050rad | Stage 1 | 100% | 100% | 0.032 | none | 0.0 | 0.0000 -> 0.0000 |
| pos=0.020m rot=0.050rad | Stage 2 | 0% | 100% | 0.708 | collision_failure | 0.0 | 0.3493 -> 0.3493 |
| pos=0.020m rot=0.050rad | Stage 3 | 100% | 100% | 1.533 | none | 0.0 | 0.2842 -> 0.2842 |

Interpretation:

- Stage 1 should remain largely insensitive to the continuity threshold because it does not plan in joint space.
- Stage 2 indicates whether finer Cartesian interpolation is enough to recover validated smooth IK paths.
- Stage 3 indicates whether continuity and collision can both be satisfied at the same resolution before a ladder-graph fallback is needed.
