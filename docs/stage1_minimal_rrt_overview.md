# `minimal_rrt.py` High-Level Notes

## What problem it solves

- Plan a path for a long bar in task space from a start pose to a goal pose.
- Use a clean single-tree `RRT` instead of the older, heavier planning stack.
- Add constraints in layers so debugging is easier.

## Core idea

- The planner grows a tree in bar pose space, not in full robot joint space.
- A node is a candidate bar pose in SE(3).
- Each extension interpolates from one pose toward a sampled target pose.
- The planner stops when a new node is close enough to the goal pose.

## Why the stages exist

- Stage 1: task-space search only.
- Stage 2: same search, but every new pose must admit dual-arm IK.
- Stage 3: same as Stage 2, plus robot collision checking.

## Main pipeline

- Load the Husky dual-arm URDF directly in PyBullet.
- Load start/end robot states and grasp transforms from JSON.
- Reconstruct start and goal bar poses from forward kinematics.
- Build an `RRT` over bar poses.
- Optionally attach dual-arm IK and collision checks during extension.
- Return the pose path, optional joint path, and validation plots.

## Important design choices

- Keep the planner functional and compact.
- Reuse `pybullet_planning` helpers for interpolation, pose comparison, and tree nodes.
- Keep Stage 1 intentionally simple so failures are easier to localize.
- Treat Stage 2 and Stage 3 as incremental constraint layers on the same base algorithm.

## What makes it different from the old stack

- No ladder graph.
- No projector in the Stage 1 search loop.
- No `robot_setup.py`.
- No joint-space search as the primary state space.
- No bi-directional tree.

## One-sentence summary

- `minimal_rrt.py` first solves the geometric "can the bar move through space?" problem, then progressively adds "can both arms realize that motion?" and "can they realize it without collision?".
