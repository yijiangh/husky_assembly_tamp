# Session Memory

Use this file to resume the current refactor without depending on autocompact.

## Focus Memo

- Primary goal right now: understand why the Stage 3 planner fails for the real-state `G3` case.
- Do not drift into cleanup, naming, refactors, or comfort fixes unless they directly unblock that debugging task.
- Before making a non-debugging code change, ask: "Does this help explain or fix the Stage 3 `G3` failure?" If not, defer it.
- When drift starts happening, explicitly redirect back to:
  - reproduce the `G3` Stage 3 failure
  - inspect endpoint IK / endpoint collision / first failing planner step
  - compare diagnosis mode versus full planner behavior
  - identify the concrete collision pair or planner rejection cause
- Default workflow for this phase:
  1. reproduce
  2. inspect
  3. isolate discrepancy
  4. patch only what is necessary
  5. rerun `G3`
- If I seem to be spending time polishing code structure instead of debugging `G3`, remind me to stop and return to the failure investigation.

## Goal

Restart the motion-planning work from the original Stage 1 design intent in `docs/algorithm_description.pdf`, but do it cleanly:

- new code path
- single-tree `RRT`, not `BiRRT`
- task-space only
- no `robot_setup.py`
- minimal setup: load URDF directly in PyBullet
- no projector, no ladder graph, no `CreateValidConfs` in the planning loop

## Current Stage 1 Implementation

New files:

- `husky_assembly_tamp/motion_planner/stage1/__init__.py`
- `husky_assembly_tamp/motion_planner/stage1/minimal_rrt.py`

What `minimal_rrt.py` does:

- loads the Husky dual-arm URDF directly
- loads start/end joint states from `RobotCellState` JSON
- loads grasp transforms from `GraspTargets` JSON
- reconstructs start/end bar poses from FK
- runs a clean pose-space single-tree `RRT`
- by default checks floating-bar collision against the robot and all other loaded scene bodies except the moving bar and debug ghosts
- visualizes the resulting bar path in PyBullet when GUI is enabled

Important design choices in this file:

- the planner is written in a functional style closer to Caelan's `pybullet_planning` motion-planner modules
- avoid planner classes unless they are clearly justified; prefer small plain functions with explicit arguments and lightweight node reuse from upstream helpers
- prefer reusing existing helpers from `../pybullet_planning` instead of rewriting basic geometry / interpolation / planner utilities locally
- planning is purely in bar pose space
- default success criterion is: find a task-space pose path from start bar pose to goal bar pose
- no joint-space continuity or dual-arm compatibility logic yet
- if both left/right grasp entries exist, Stage 1 computes both reconstructed bar poses and warns if they disagree, but uses the left-arm result as the planning reference
- Stage 1 currently applies a debug-only world-frame start offset `[-0.5, 0.0, 0.5]` to force a non-trivial path; this intentionally breaks robot/bar consistency and is expected to be removed later
- the floating bar is currently modeled as a box with the cylinder's bounding dimensions `(2r, 2r, L)` and feature points are the explicit local box corners

## Docker Feedback Loop

The host sandbox did not have `pybullet` / `pybullet_planning`, so Docker was used for real validation.

Updated files:

- `docker/trajectory_testbench/run.sh`
- `docker/trajectory_testbench/README.md`
- `docker/trajectory_testbench/Dockerfile`
- `docker/trajectory_testbench/docker-compose.yml`
- `docker/trajectory_testbench/start_virtual_desktop.sh`

New runner actions:

- `./run.sh stage1`
- `./run.sh debug-stage1`
- `./run.sh desktop-up`
- `./run.sh desktop-down`
- `./run.sh desktop-status`
- `./run.sh stage1-vnc`
- `./run.sh testbench-vnc`

New headless mode:

- set `HUSKY_DOCKER_HEADLESS=1` to skip XQuartz / host GUI checks

New browser-based GUI mode:

- the container can now host its own X desktop with `Xvfb + fluxbox + x11vnc + noVNC`
- noVNC is exposed at `http://localhost:6080/vnc_lite.html?autoconnect=1&resize=remote&host=localhost&port=6080&path=websockify`
- this is the preferred GUI path on macOS because XQuartz + container OpenGL was not reliable for PyBullet

## Validated Command

This command was run successfully:

```bash
HUSKY_DOCKER_HEADLESS=1 ./docker/trajectory_testbench/run.sh stage1 -- --no-gui --max-time 3 --max-iterations 100 --max-attempts 1 --random-seed 0
```

Observed result:

- container built and started successfully
- `minimal_rrt.py` ran successfully inside the Ubuntu container
- reported:

```text
Running minimal Stage 1 RRT.
start pose: [0.7912 0.0902 0.8623]
goal pose:  [ 0.897  -0.0046  0.3443]
floating collision: off
Found Stage 1 pose path with 12 waypoints.
```

## noVNC Validation

Validated command:

```bash
./docker/trajectory_testbench/run.sh desktop-up
```

Observed result:

- updated image built successfully with the desktop stack packages
- container started successfully
- desktop services started successfully:
  - `xvfb`
  - `fluxbox`
  - `x11vnc`
  - `novnc`
- `curl -I "http://localhost:6080/vnc_lite.html?autoconnect=1&resize=remote&host=localhost&port=6080&path=websockify"` returned `HTTP/1.1 200 OK`

Recommended GUI flow now:

```bash
./docker/trajectory_testbench/run.sh desktop-up
open "http://localhost:6080/vnc_lite.html?autoconnect=1&resize=remote&host=localhost&port=6080&path=websockify"
./docker/trajectory_testbench/run.sh stage1-vnc
```

## Known Constraints

- Local host sandbox here could not import `pybullet` or `pybullet_planning`
- Docker container has the needed runtime and is the preferred validation path
- `run.sh` originally assumed GUI support even for headless runs; that has been fixed with `HUSKY_DOCKER_HEADLESS=1`
- macOS XQuartz forwarding no longer matters for noVNC runs
- `run.sh` now uses `python -B -m ...` so the container ignores stale `.pyc` files and executes the current host-mounted source

## Likely Next Steps

Choose one and continue from there:

1. Add a small Stage 1-specific debug export format for the tree and path.
2. Add Stage 1 CLI knobs for workspace bounds / interpolation resolution.
3. Add a simple floating-obstacle collision mode beyond robot-only collision.
4. Keep Stage 1 fixed and start Stage 2 by introducing projection into `extend`.

## Files To Read First In A New Session

If resuming later, read these first:

- `memory.md`
- `husky_assembly_tamp/motion_planner/stage1/minimal_rrt.py`
- `docker/trajectory_testbench/run.sh`
- `docker/trajectory_testbench/README.md`

## Session Addendum: Stage 1 Preferences And Tooling

- Coding style preference: keep `husky_assembly_tamp/motion_planner/stage1/minimal_rrt.py` focused on the core algorithm plus a thin single-run entrypoint. Avoid planner classes when plain functions are sufficient. Prefer Caelan-style functional structure and reuse helpers from `../pybullet_planning` whenever possible.
- Comment style preference: add short comments ahead of major logic blocks in non-trivial functions so the flow is easier to scan, but avoid noisy line-by-line comments.
- `minimal_rrt.py` now owns the single-run Stage 1 workflow:
  - direct runnable `main()`
  - scene setup / teardown
  - `run_stage1_trial(...)`
  - visualization loop
  - core planner `plan_pose_rrt(...)`
- `husky_assembly_tamp/motion_planner/stage1/debug_runner.py` is the heavier wrapper for benchmarking / reporting. It should call into `minimal_rrt.py`, not the other way around.

## Session Addendum: Stage 1 Planner Decisions

- Replaced local quaternion helper usage with `pybullet_planning` helpers where appropriate:
  - `pp.is_pose_close`
  - `pp.interpolate_poses`
  - `pp.quat_angle_between`
- The floating bar geometry is currently modeled as a box with the cylinder bounding dimensions `(2r, 2r, L)`.
- Feature points are explicit local box corners, transformed by pose; do not recover them from AABB each time.
- Floating collision is enabled by default.
- Floating-body collision now checks the bar against all loaded scene bodies except the moving bar and the debug ghosts, not just the robot.
- Stage 1 currently uses a debug-only start pose offset in world coordinates `[-0.5, 0.0, 0.5]` to force a non-trivial path. This intentionally breaks consistency with the start robot configuration and is expected to be removed later.

## Session Addendum: Exposed Hyperparameters

- `position_res` and `rotation_res` are exposed at the highest level in both Stage 1 entrypoints.
- The CLI/help text should state units explicitly:
  - `position_res`: meters
  - `rotation_res`: radians
- There is a `--lock-renderer-during-search` option for GUI runs. This wraps the search in `pp.LockRenderer()` so live tree redraw is suppressed during planning, then the result can be visualized afterward.

## Session Addendum: Stage 1 Benchmarking / Reports

- `husky_assembly_tamp/motion_planner/stage1/debug_runner.py` now supports batch analysis with `--analysis-trials`.
- By default `debug_runner.py` is headless; use `--gui` to enable PyBullet GUI.
- The runner is intended to generate artifacts similar in coverage to the archived report under `husky_assembly_tamp/motion_planner/zh_archive/reports/debug_report_20260310_121640.md`.
- Current Stage 1 analysis artifacts include:
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
- The benchmarking instructions were added to the root `README.md` under `Stage 1 Benchmarking`.

## Session Addendum: Stage 2/3 Joint Continuity Refinement

- Current limitation: the Stage 2/3 RRT is still fundamentally a task-space tree with a single seed-chained IK branch attached to each edge.
  - In `extend`, the planner only propagates one `current_conf`.
  - `solve_dual_arm_pose_ik(...)` returns one seed-chained dual-arm IK solution per interpolated pose.
  - Joint continuity is therefore not optimized during search; it is only measured afterward in trajectory validation.
- Important conclusion from this: increasing interpolation density inside the planner can help, but it is still only a heuristic because the planner does not compare multiple IK branches.
- First fix implemented before any ladder-graph work:
  - add a dense post-plan refinement pass for Stage 2/3 in `husky_assembly_tamp/motion_planner/stage1/minimal_rrt.py`
  - workflow:
    1. plan a coarse task-space path as before
    2. densify that pose path
    3. re-run seed-chained dual-arm IK along the denser path
    4. keep the refined result only if it improves continuity / remains feasible
- New planner CLI knobs:
  - `--no-refine-after-plan`
  - `--refine-position-res`
  - `--refine-rotation-res`
  - `--refine-max-passes`
- Default refinement policy:
  - enabled for Stage 2/3
  - initial refinement resolution defaults to half of the coarse `position_res` / `rotation_res`
  - each refinement pass halves the step again
  - default `refine_max_passes=2`
- New continuity helper in `minimal_rrt.py`:
  - `summarize_joint_continuity(...)`
  - used to compare coarse and refined joint paths by max wrapped joint delta and first violating step
- Important implementation detail:
  - the refinement pass must recompute IK along the dense path
  - simply inserting extra Cartesian waypoints without re-solving IK does not address branch-jump issues
- Stage 2/3 debug reporting now records refinement behavior in `debug_runner.py`:
  - whether refinement was attempted
  - whether the refined path was accepted
  - coarse vs final max joint delta
  - refinement status / failure reason
  - refinement waypoint counts
- The comparison report now explicitly shows continuity before/after refinement.

## Session Addendum: Observed Refinement Outcomes

- Validated with Docker compare run:

```bash
python -B -m husky_assembly_tamp.motion_planner.stage1.debug_runner \
  --compare-stages \
  --analysis-trials 1 \
  --analysis-seed-start 0 \
  --position-res 0.1 \
  --rotation-res 0.2 \
  --endpoint-ik-attempts 20 \
  --max-time 3 \
  --max-iterations 100 \
  --max-attempts 1
```

- Latest clean report bundle after refinement work:
  - `husky_assembly_tamp/motion_planner/stage1/reports/stage_comparison_report_20260317_101738.md`
  - support artifacts under `husky_assembly_tamp/motion_planner/stage1/reports/_support`
- Concrete result for seed `0`:
  - Stage 2:
    - coarse path had `29` waypoints
    - refinement densified it to `69` waypoints
    - max joint delta improved from `0.6884` to `0.3614 rad`
    - joint continuity changed from fail to pass
  - Stage 3:
    - coarse path remained collision-free but had poor continuity (`1.4064 rad`)
    - dense re-IK refinement failed with `ik_failure_at_waypoint_11`
    - planner correctly fell back to the coarse path
    - Stage 3 therefore still needs a stronger method than local seed-chained refinement
- Interpretation:
  - dense post-plan refinement is a worthwhile first fix and already solves at least some Stage 2 continuity failures
  - Stage 3 demonstrates the limit of this approach under stronger collision/IK constraints
  - the next principled step, if needed, is still the ladder-graph refinement over multiple IK candidates per capsule / waypoint

## Session Addendum: Report / Artifact State

- Reports directory is kept intentionally clean:
  - top-level `husky_assembly_tamp/motion_planner/stage1/reports` contains only the latest Markdown reports
  - all CSV / JSON / PNG / profile dumps live under `_support`
- Validation images from `path_validation.py` are also routed into `_support` and embedded in the generated reports.
- Workspace tree plots across Stage 1/2/3 now use fixed shared axis bounds so the views do not shift between stages.
