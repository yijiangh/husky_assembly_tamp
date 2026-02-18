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

File: `husky_assembly_tamp/motion_planner/trajectory_dual_constrained_solver.py`

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
python -m husky_assembly_tamp.motion_planner.trajectory_dual_constrained_solver
```

# Dual-arm Cartesian-based Constrained Planner

This module plans a dual-arm trajectory by interpolating in Cartesian space (tool pose) and projecting to joint space while maintaining a fixed relative pose constraint between tool frames. It organizes intermediate IK solutions into a ladder-graph-like structure and can render this as an SVG.

File: `husky_assembly_tamp/motion_planner/trajectory_dual_cart_constrained_solver.py`

## Overview
- Samples and interpolates end-effector poses between start and goal in Cartesian space, then solves IK with a dual-arm projector to obtain paired joint configs.
- Builds per-step sets of IK solutions (rungs) and records feasible connections between adjacent rungs.
- Attempts a direct connection first via `pybullet_planning.direct_path`; if it fails, can fall back to `rrt_connect` on capsules.
- Can visualize the ladder graph (nodes and edges) and save it as an SVG to `PROJECT_DIR/plots/`.

## Key Concepts
- Capsule: a node representing a set of dual-arm joint configurations (multiple IK solutions) at a Cartesian waypoint. Each capsule stores connections to its parent capsule, forming a ladder graph across the sequence.
- Ladder graph: each rung is the set of IK solutions at a waypoint; edges connect feasible pairs across adjacent rungs.

## Public API

### class TrajectoryDualCartConstrainedSolver(robot_setup, target_parser, projector)
- Initializes the solver with the robot/environment setup, target parser, and a dual-arm constraint projector.

Methods:
- initialize_robot_setup_for_planning(robot_name, robot_type, target_cell_state_path, use_scene_parser_gui=True, scene_parser_verbose=True) -> (RobotSetup, np.ndarray, DualArmProjection)
  - Build `RobotSetup`, compute normalized target joint config, and construct the dual-arm projector from current tool poses.

- cart_linear_interp(q1: Capsule, q2: Capsule, position_res=0.005, rotation_res=0.01) -> List[Tuple[np.ndarray, np.ndarray]]
  - Interpolate tool pose from `q1` to `q2` using linear position and quaternion slerp, returning a list of waypoints as (position, quaternion).

- cart_linear_interp_z(q1: Capsule, q2: Capsule, position_res=0.1, rotation_res=0.1) -> List[Tuple[np.ndarray, np.ndarray]]
  - Not used.

- _get_sample_fn() -> Callable[[], Capsule]
  - Returns a sampler that proposes random Cartesian poses for the bar/tool and solves for valid dual-arm IK, yielding a new `Capsule` when successful.

- _get_extend_fn() -> Callable[[Capsule, Capsule], Iterable[Capsule]]
  - Returns an extension function that interpolates in Cartesian space and at each waypoint projects to valid dual-arm IK, yielding a sequence of `Capsule`s.

- _get_distance_fn() -> Callable[[Capsule, Capsule], float]
  - Returns a distance metric combining the translational difference and orientation error (via right-from-left tool transform) between two capsules.

- _get_collision_fn() -> Callable[[Capsule], bool]
  - Returns a predicate that flags a capsule as in-collision/invalid if it has no configs or lacks a valid connection to its parent.

- generate_start_configuration(projector, delta_pose_point=[0.4, 0.0, 0.75], delta_pose_euler=[π, π/2, π/2], max_attempts=100, delta_angle=π) -> np.ndarray
  - Generates a feasible start configuration satisfying the dual-arm constraint via IK and projection; returns a 12-DOF joint vector.

- try_direct_path(extend_fn, start_capsule: Capsule, target_capsule: Capsule) -> Optional[List[Capsule]]
  - Uses `pybullet_planning.direct_path` with the provided `extend_fn` and internal collision checking to attempt a straight-line connection in capsule space.

### Helper classes and functions

- class Capsule(config, parent=None, robot_setup=None, projector=None)
  - Represents a rung of IK solutions (`config` is a list of 12-DOF joint arrays). Upon creation with a parent, computes feasible connections to the parent rung.
  - Methods:
    - create_connection(robot_setup, projector): populates `connection` lists to parent indices.
    - check_connection() -> bool: whether any valid parent connection exists.
    - retrace() -> List[Capsule]: returns the chain from root to this capsule.

- configs_capsule(nodes: List[Capsule]) -> Optional[Tuple[List[np.ndarray], List[int]]]
  - Given an ordered list of capsules (rungs), selects a feasible sequence of joint configs across rungs by backtracking the stored connections. Returns `None` if no sequence exists.

- plot_capsule_path(capsule_path: List[Capsule], highlight_feasible: bool = False) -> Optional[str]
  - Draws the ladder graph (nodes per rung and edges across adjacent rungs) and saves an SVG to `PROJECT_DIR/plots/ladder_graph_<timestamp>.svg`. Returns the saved file path or `None`.

- rrt_connect_capsule(start: Capsule, goal: Capsule, distance_fn, sample_fn, extend_fn, collision_fn, robot_setup, projector, ...) -> Optional[List[np.ndarray]]
  - Standard RRT-Connect over capsules using the provided primitives; returns a joint-space path if found.

- extend_towards_capsule(tree, target, distance_fn, extend_fn, collision_fn, robot_setup, projector, swap=False, tree_frequency=1, **kwargs)
  - Helper to extend a search tree of capsules toward a target capsule.

- asymmetric_extend(q1, q2, extend_fn, backward=False)
  - Utility to reverse the direction of an extension when needed.

## How to run (GUI recommended)
```bash
python -m husky_assembly_tamp.motion_planner.trajectory_dual_cart_constrained_solver
```

What it does:
- Initializes the robot setup and dual-arm projector from a chosen design study and robot cell state.
- Generates a start configuration, interpolates Cartesian waypoints, and draws them in PyBullet.
- Attempts a direct capsule path; if unavailable, falls back to `rrt_connect_capsule` and prints the found path.
- Saves a ladder-graph SVG to `PROJECT_DIR/plots/` via `plot_capsule_path`.