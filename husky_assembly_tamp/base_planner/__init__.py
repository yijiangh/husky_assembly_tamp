"""
The monitor's base planner, geometry only: whether a robot base can drive from one floor pose to another, and how.

Copied from husky-assembly-teleop (each file names its source and commit) and kept close to it, so it can be re-synced.

- `lattice.plan_drive`: plan a path in a synced `world.PlanningWorld`.
- ! No ROS, viser, crl_husky or monitor code: only numpy, scipy, pybullet and `scaffolding_assembly_core`.
- ! Import the modules you need: this file imports nothing, so the package costs nothing until used.
"""
