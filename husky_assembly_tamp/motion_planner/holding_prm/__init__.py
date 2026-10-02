"""A roadmap planner for the two-arm bar carry, with the constraints flipped.

The production M1 planner (``dual_arm_task_space_rrt``) samples bar poses,
holds the bar exactly at every waypoint, and fails when the joints jump. This
package does the opposite, as proposed in the 2026-09-29 meeting:

* nodes are sampled in joint space ("forward kinematics for one arm, inverse
  kinematics for the other"), so every node holds the bar exactly;
* nodes are joined by straight lines in joint space, so the joints never jump;
* the price of an edge is how far the two tools drift apart from the bar along
  it -- the bar-holding violation -- and a graph search looks for the path
  whose violation is smallest.

Modules, in the order they were built:

    holding    the violation measure and the forward kinematics it needs
    sampling   the node sampler and the endpoint (start / goal) solver
    roadmap    nearest neighbours, edge checks, edge cost, the roadmap itself
    search     the start-set to goal-set search and its reports
    parallel   the worker processes that do the sampling and edge checks
    store      where results are written and how they are read back
    pilot      the build / query / export drivers the command line calls

The command line is ``scripts/prm_pilot.py``. The design notes are in
``docs/prm_pilot_plan.md``.

! Nothing in this package imports ``dual_arm_task_space_rrt.core``: importing
! that module empties its log file, which a planner batch running in another
! process may be writing to at the same time.
"""
