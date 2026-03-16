# Session Memory

Use this file to resume the current refactor without depending on autocompact.

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
- optionally checks floating-bar collision against the fixed robot
- visualizes the resulting bar path in PyBullet when GUI is enabled

Important design choices in this file:

- `PoseNode` only stores `pose`, `parent`, and an optional cached feature vector
- planning is purely in bar pose space
- default success criterion is: find a task-space pose path from start bar pose to goal bar pose
- no joint-space continuity or dual-arm compatibility logic yet
- if both left/right grasp entries exist, Stage 1 computes both reconstructed bar poses and warns if they disagree, but uses the left-arm result as the planning reference

## Docker Feedback Loop

The host sandbox did not have `pybullet` / `pybullet_planning`, so Docker was used for real validation.

Updated files:

- `docker/trajectory_testbench/run.sh`
- `docker/trajectory_testbench/README.md`

New runner actions:

- `./run.sh stage1`
- `./run.sh debug-stage1`

New headless mode:

- set `HUSKY_DOCKER_HEADLESS=1` to skip XQuartz / host GUI checks

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

## Known Constraints

- Local host sandbox here could not import `pybullet` or `pybullet_planning`
- Docker container has the needed runtime and is the preferred validation path
- `run.sh` originally assumed GUI support even for headless runs; that has been fixed with `HUSKY_DOCKER_HEADLESS=1`

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
