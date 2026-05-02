---
title: `minimal_rrt.py` Overview
subtitle: High-level algorithmic ideas
---

# `minimal_rrt.py`

- Goal: move a long bar from start pose to goal pose.
- State space: bar pose in SE(3), not full robot configuration.
- Planner: single-tree `RRT`.
- Philosophy: start simple, then add constraints in layers.

```text
Start bar pose  -->  grow RRT in task space  -->  reach goal bar pose
```

---

# Why This File Exists

- It is a clean restart of the planning logic.
- It strips away older machinery that made debugging harder.
- It isolates the algorithmic question:
- Can task-space exploration work before we add harder constraints?

```text
Old stack: many coupled components
        |
        v
minimal_rrt.py: one planner core + optional constraint layers
```

---

# Main Inputs

- URDF of the dual-arm Husky robot.
- Start and goal robot states from `RobotCellState` JSON.
- Left/right grasp transforms from `GraspTargets` JSON.

```text
RobotCellState + GraspTargets
            |
            v
Forward kinematics
            |
            v
Start bar pose + Goal bar pose
```

---

# Planner State

- Each node stores one candidate bar pose.
- The tree lives in workspace/task space.
- Distance can be measured by:
- feature-point distance on the bar
- or a simpler pose-based metric

```text
Tree node = (bar position, bar orientation)
Edge      = interpolated motion between two bar poses
```

---

# Core RRT Loop

- Sample a random target pose, sometimes the goal.
- Find the nearest existing tree node.
- Interpolate toward the target.
- Stop extension on failure.
- Add valid intermediate poses as new tree nodes.
- Finish when the newest node is close enough to the goal.

```text
sample target
    |
    v
find nearest node
    |
    v
extend by interpolation
    |
    +--> fail: stop this branch
    |
    +--> success: add nodes
                |
                v
            goal reached?
```

---

# Stage 1

- Pure task-space planning.
- No IK in the planning loop.
- Optional floating-bar collision only.
- Best for testing whether the bar path itself is geometrically findable.

```text
pose sample --> pose interpolation --> floating-bar validity
```

---

# Stage 2

- Same task-space tree.
- Every new pose must also admit dual-arm IK.
- IK is seed-chained from the previous valid configuration.
- This tests feasibility of realizing the pose path with both arms.

```text
new bar pose
    |
    v
solve right/left IK with previous joint seed
    |
    +--> IK fail: reject extension
    +--> IK pass: keep node and store joint config
```

---

# Stage 3

- Same as Stage 2.
- Also run robot collision checking on the IK result.
- This is the most realistic mode in this file.

```text
bar pose --> dual-arm IK --> robot collision check --> accept or reject
```

---

# Why Seed-Chained IK Matters

- Nearby poses should usually have nearby joint solutions.
- Reusing the previous solution stabilizes the path.
- It is much cheaper than searching globally from scratch every step.

```text
q(k)  --->  use as seed for pose(k+1)  --->  q(k+1)
```

---

# Collision Logic

- Stage 1: optional floating bar against robot/environment.
- Stage 2: usually collision-off, focused on IK feasibility.
- Stage 3: robot self-collision and robot-vs-environment collision.
- Grasp-adjacent links are masked where contact is expected.

```text
collision layers:
1. bar vs world
2. robot self
3. robot vs static world
```

---

# Path Refinement And Validation

- After planning, the code can reconstruct or refine a joint path.
- It checks:
- collisions
- joint continuity
- left/right end-effector relative-transform drift
- It also writes diagnostic plots.

```text
planned path
    |
    v
joint reconstruction / refinement
    |
    v
validation plots + summaries
```

---

# Big Picture

- First solve: "Can the bar move through space?"
- Then solve: "Can both arms realize each waypoint?"
- Then solve: "Can they do it without collisions?"

```text
Stage 1: task-space connectivity
      ->
Stage 2: task-space + dual-arm IK feasibility
      ->
Stage 3: task-space + IK + collision feasibility
```

---

# One-Slide Summary

- `minimal_rrt.py` is a layered planner.
- The base planner is a simple pose-space `RRT`.
- IK and collisions are not the planner itself.
- They are additional filters on top of the same search skeleton.
- That makes failures easier to diagnose stage by stage.
