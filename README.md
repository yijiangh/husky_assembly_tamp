# How to Deploy?

```bash
git clone --branch lzh/dual-arm https://github.com/yijiangh/husky_assembly_tamp.git
cd husky_assembly_tamp
git submodule init
git submodule update --remote
cd ext/husky-assembly-teleop
git submodule init
git submodule update --remote
cd external/compas_fab
pip install -e .
```

# Dual-arm Constrained Planner

This module implements a dual-arm constrained motion planner that maintains a fixed relative pose between the two end-effectors while planning a collision-aware joint-space trajectory.

File: `scripts/motion_planner/trajectory_dual_constrained_solver.py`

## Overview
- Enforces a dual-arm constraint via a projector that keeps a constant transform between the left and right tool links.
- Tries a direct path first; falls back to an RRT-based planner from `pybullet_planning` with custom sampling/extension/distance functions.
- Provides optional visualization of the search and the final path in PyBullet.
- Exports the result as a `compas_fab` `JointTrajectory` JSON.

## Public API

### class TrajectoryDualConstrainedSolver(robot_setup, target_parser, resolution=DEFAULT_RESOLUTION)
- **robot_setup**: A configured `RobotSetup` instance (scene, robot, obstacles).
- **target_parser**: A `TargetParser` to access grasp tools/poses.
- **resolution**: Angular resolution used internally for planning.

Methods:
- **plan(start_conf, target_conf, max_time=600, max_projection_attempts=100, visualization=True) -> Optional[List[np.ndarray]]**
  - Normalizes joint angles to [-π, π].
  - Sets up the dual-arm constraint projector and collision checking.
  - Generates projected start/target configurations, then:
    - Tries a direct connection via a custom extend function; if that fails,
    - Runs RRT with custom `sample_fn`, `extend_fn` (continuous), and `distance_fn`.
  - Post-processes to remove trailing frames where the right arm does not change.
  - Optionally visualizes the search tree and final trajectory in PyBullet.
  - Returns a list of 12-DOF joint arrays or `None` if planning fails.

- **interactive_trajectory_playback(path: List[np.ndarray])**
  - Interactive slider playback of a planned path in PyBullet GUI.

- **@staticmethod initialize_robot_setup_for_planning(robot_name, robot_type, target_cell_state_path, use_scene_parser_gui=True, scene_parser_verbose=True) -> (RobotSetup, np.ndarray, DualArmProjection)**
  - Creates a `RobotSetup` from a robot cell state file, extracts and normalizes the target configuration, and builds the dual-arm projector.

- **generate_start_configuration(projector, delta_pose_point=[0.4, 0.0, 0.75], delta_pose_euler=[-1.5708, 1.5708, 0], tool_index=1, max_attempts=100) -> np.ndarray**
  - Generates a feasible, collision-free start configuration that satisfies the dual-arm constraint using IK plus projection.

## How main() calls the planner
The `main()` function demonstrates end-to-end usage:
1. Compute paths for the design study, case, and `target_cell_state_path` based on `DATA_DIR` and a `target_name`.
2. Initialize the planning context:
   - `robot_setup, target_conf, projector = TrajectoryDualConstrainedSolver.initialize_robot_setup_for_planning("r0", "husky_dual", target_cell_state_path, use_scene_parser_gui=True, scene_parser_verbose=True)`
3. Create a `TargetParser` with the design study root and grasp targets file.
4. Instantiate the solver: `solver = TrajectoryDualConstrainedSolver(robot_setup, target_parser)`.
5. Generate a valid start configuration: `start_conf = solver.generate_start_configuration(projector)`.
6. Plan the trajectory:
   - `path = solver.plan(start_conf, target_conf, max_time=36000, max_projection_attempts=100, visualization=True)`
7. Optionally visualize/play back the path (commented in code).
8. Save the trajectory and clean up (details below).

Minimal runnable example (GUI recommended):
```bash
python scripts/motion_planner/trajectory_dual_constrained_solver.py
```