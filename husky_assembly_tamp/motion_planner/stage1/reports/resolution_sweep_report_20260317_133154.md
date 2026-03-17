# Resolution Sweep Report (20260317_133154)

## Scope

This report compares Stage 1, Stage 2, and Stage 3 across multiple task-space interpolation resolutions.

Run setup:

- Trials per stage/resolution pair: `5` seeds (`0..4`)
- Dist metric: `feature`
- Endpoint IK attempts: `20`
- Joint continuity threshold: `0.2 rad`
- Post-plan refinement: `on`

CSV: `_support/resolution_sweep_20260317_133154.csv`
JSON: `_support/resolution_sweep_20260317_133154.json`

---

## Results

| Resolution | Stage | Validated success | Path found | Avg runtime (s) | Dominant failure | Avg continuity rejects | Avg max dq (coarse -> final) |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| pos=0.050m rot=0.100rad | Stage 1 | 100% | 100% | 0.010 | none | 0.0 | 0.0000 -> 0.0000 |
| pos=0.050m rot=0.100rad | Stage 2 | 20% | 20% | 0.933 | continuity_failure | 42.2 | 0.1981 -> 0.1981 |
| pos=0.050m rot=0.100rad | Stage 3 | 20% | 20% | 0.769 | collision_failure | 41.8 | 0.1981 -> 0.1981 |
| pos=0.030m rot=0.070rad | Stage 1 | 100% | 100% | 0.015 | none | 0.0 | 0.0000 -> 0.0000 |
| pos=0.030m rot=0.070rad | Stage 2 | 0% | 0% | 1.822 | continuity_failure | 32.6 | 0.0000 -> 0.0000 |
| pos=0.030m rot=0.070rad | Stage 3 | 20% | 20% | 1.127 | collision_failure | 25.2 | 0.1735 -> 0.1735 |
| pos=0.020m rot=0.050rad | Stage 1 | 100% | 100% | 0.020 | none | 0.0 | 0.0000 -> 0.0000 |
| pos=0.020m rot=0.050rad | Stage 2 | 20% | 60% | 1.914 | continuity_failure | 11.2 | 0.1813 -> 0.1813 |
| pos=0.020m rot=0.050rad | Stage 3 | 80% | 80% | 0.921 | collision_failure | 6.8 | 0.1686 -> 0.1686 |
| pos=0.015m rot=0.040rad | Stage 1 | 100% | 100% | 0.026 | none | 0.0 | 0.0000 -> 0.0000 |
| pos=0.015m rot=0.040rad | Stage 2 | 40% | 40% | 2.501 | continuity_failure | 6.6 | 0.1634 -> 0.1634 |
| pos=0.015m rot=0.040rad | Stage 3 | 80% | 80% | 1.355 | collision_failure | 6.2 | 0.1591 -> 0.1591 |
| pos=0.010m rot=0.025rad | Stage 1 | 100% | 100% | 0.040 | none | 0.0 | 0.0000 -> 0.0000 |
| pos=0.010m rot=0.025rad | Stage 2 | 20% | 60% | 2.538 | continuity_failure | 2.0 | 0.1390 -> 0.1390 |
| pos=0.010m rot=0.025rad | Stage 3 | 100% | 100% | 1.783 | none | 2.2 | 0.1399 -> 0.1399 |

Interpretation:

- Stage 1 should remain largely insensitive to the continuity threshold because it does not plan in joint space.
- Stage 2 indicates whether finer Cartesian interpolation is enough to recover validated smooth IK paths.
- Stage 3 indicates whether continuity and collision can both be satisfied at the same resolution before a ladder-graph fallback is needed.
