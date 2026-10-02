# Holding-constraint PRM pilot: discussion digest and implementation plan

Source: Plaud recording "09-29 Meeting: Dual-Arm Motion Planning and PRM Planner
Pilot" (2026-09-29, 32 min, Yijiang and Caelan Garrett). Written 2026-09-30.
Next check-in with Caelan is planned for around 2026-10-13.

---

## Part 1. What was discussed

### The problem as presented

- The M1 transfer planner (`plan_constrained_dual_arm`, task-space BiRRT with an
  IK projection per waypoint) is the bottleneck of the live demo. After the base
  parks and mocap reports the base station, a replan can take about five minutes,
  even on problems that look intuitively solvable.
- The benchmark (see `benchmark/README.md`) showed one start that no seed out
  of ten could solve, and on the failures **74% of tree extensions stopped on the
  continuity gate**, not on collision or IK failure. That benchmark scene has no
  environment: only self-collision and bar-robot collision.
- Why the continuity gate dominates: every tree node stores one IK branch, each
  extension walks a task-space straight line and takes the ssik branch nearest
  the parent, and the extension is cut the moment the nearest branch jumps by
  more than the threshold (about 10 degrees on any joint). The tree therefore
  never leaves the branch sheet its root started on.
- Open worry: with two 6-DOF arms holding one rigid bar, the constrained
  configuration space may simply be disconnected between some start and goal
  branch pairs (elbows in versus elbows out). We cannot currently tell "no
  solution exists" from "the planner is weak". Caelan agreed this is a hard
  problem and some pairs may really be disconnected.
- The start pose is free (three carry anchors: on the back, horizontal in
  front, vertical), so the problem is underdefined. Today a start is derived
  by hand-tuned heuristics and that does not guarantee a good pairing with
  the goal.

### Caelan's suggestions, in the order they came up

1. **Relax constraints one at a time** to see which one produces the jump from
   infeasible to feasible: world collision, self-collision, continuity gate,
   even one arm removed entirely. Whichever relaxation makes the failing cases
   solvable names the real obstacle.
2. **Count queries, not seconds.** Give the planner a very large budget and
   record how many IK calls, collision checks, nodes and edges a solution
   needs. If the answer is "thousands of samples", that justifies GPU
   acceleration later; if not, the CPU is fine. Do not switch to GPU yet.
3. **The PRM with the constraint priorities flipped.** Today's planner enforces
   the bar-holding constraint exactly at every waypoint and fails on joint
   continuity. Build the opposite:
   - Sample nodes primarily in joint space, using "forward kinematics for one
     arm, IK for the other" so each node still holds the bar exactly.
   - Connect nearby nodes with **straight joint-space interpolation**, so
     continuity holds by construction and joint limits are never crossed.
   - Filter edges for collision.
   - Make the **holding-constraint violation along the edge the edge cost**
     (zero within a tolerance, a large penalty beyond it). Optionally add a
     soft cost for being within a centimeter or two of collision.
   - Run Dijkstra on the roadmap. Look at what it returns: if the best path
     has violation slightly above the tolerance, the problem is near-feasible
     and we learn what the gap is; if it is far above, the pair is probably
     disconnected.
   - Roadmap size in the thousands of configurations. FK and the violation
     checks are cheap; IK is the expensive operation (10-20 ms per ssik call
     today).
4. **Why a roadmap and not a tree**: the roadmap keeps many alternative paths,
   so the same graph answers different cost functions and can be re-searched.
5. **Refinement loop for quality.** Take the first (suboptimal, violating)
   path, sample densely near it (either perturb the path or sample the whole
   space and keep what lands close), reconnect or shortcut between pairs of
   configurations on it, and search again. The search and cost code stay the
   same; only the graph changes and it gets much smaller after the first
   round. Repeat until the violation converges.
6. **Changing collision scenes** (different scaffolding around the robot) are
   not a reason to avoid precomputed roadmaps: re-check the edges against the
   new scene. Collision checks are cheap next to IK, especially batched.
7. The mirror image is also worth a try later: plan in task space and make the
   continuity violation the soft cost.

### Second topic: an easy intermediate carry pose

Before entering dense scaffolding the robot could switch from the loading
home pose to another home pose that makes the final approach easy. Caelan's
suggestions: evaluate candidate intermediate poses over a distribution of
sampled problems (does a plan to the manipulation stage exist from there,
with little base travel), or search backward from the contact state, as in
the RSS paper, to find states reasonably far from contact that are good
waypoints. Yijiang's variant: pre-plan for the known bars and partial
structures, then intersect the sets of workable poses. This is a follow-up,
not part of the pilot.

### Decisions and actions

| What | Who | When |
|---|---|---|
| Build a small CPU pilot of the flipped-priority PRM on the benchmark problems | Yijiang | before 2026-10-13 |
| Record IK, FK and collision counts, node and edge counts, violation of the best path | Yijiang | with the pilot |
| Run the constraint-relaxation diagnosis on the unsolved benchmark start | Yijiang | quick, before the pilot |
| Decide on GPU acceleration only from the pilot's sample counts | Yijiang | after the pilot |
| Send the preprints (Reinforcement Gen, DemoStream, the multi-arm conflict-based GPU planner with Philip Huang) | Caelan | when approved |
| Push code toward the open-source Scheduling Stream repo if approvals allow | Caelan | later |

Side note from the meeting: Valentin's last day at CRL was 2026-09-25; he starts
at Hexagon on 2026-09-30.

---

## Part 2. Implementation plan

### Goal of the pilot

Answer two questions on the benchmark problems, especially the start that
failed on all ten seeds:

1. Does a **continuous joint-space path** exist between start and goal whose
   bar-holding violation stays under a small tolerance? If yes, how many
   samples, IK calls and collision checks did it take?
2. If no, how large is the smallest violation the roadmap can achieve, and
   where along the path does it happen? That tells us whether the pair is
   near-feasible or truly disconnected.

The pilot is a research tool first. It does not touch `api.py` until step 7.

### Where the code goes

```
husky_assembly_tamp/motion_planner/holding_prm/
    __init__.py
    holding.py      violation measure and FK helpers (pure functions)
    sampling.py     node sampler: FK one arm, IK the other, collision filter
    roadmap.py      neighbor search, edge evaluation, costs, counters
    search.py       Dijkstra over the roadmap with start and goal sets
    refine.py       resample near a path, rebuild the local graph, re-search
husky_assembly_tamp/benchmark/prm_pilot.py   drives one library problem end to end
scripts/benchmark_cdfm.py                    new subcommand: prm-pilot
```

Libraries: `networkx` and `scipy` are already in the sidecar venv (checked
2026-09-30). `scipy.spatial.cKDTree` for neighbors, `networkx` or
`scipy.sparse.csgraph.dijkstra` for the search. No new dependencies.

### What is reused from the existing code

| Need | Reuse |
|---|---|
| Live scene, robot ids, grasps, template state | `benchmark.scene.SceneCtx` via `open_scene` |
| Bar pose from a configuration | `SceneCtx.bar_pose_at(conf12)` and `SceneCtx.fk_link_pose` |
| All IK branches of one arm at a tool pose | `motion_planner.ssik_ik.ssik_arm_branches(robot, arm, pose, seed6=None)` |
| Production collision verdict | `SceneCtx.collision_fn()` (same compas_fab check the planner uses) |
| Continuity threshold and per-joint step | `core.DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD`, `core.joint_step_deltas` |
| Goals, their alternative branch pairs, surveyed starts | the benchmark library (`library.load_bar_library`) |
| Turning a near-feasible path into a production path | `core.repair_joint_path` / `core.reconstruct_joint_path_for_pose_path` |
| Comparison against today's planner | the existing `run` batches and `variants.py` |

### Build sequence

Built in this order. Each step is runnable and testable on its own before the
next one layers on top.

**Step 1. The violation measure (`holding.py`).** Built first because every
later step scores with it.

- `bar_pose_from_left(conf12)`: bar pose implied by the left tool, which is
  what `SceneCtx.bar_pose_at` already does.
- `holding_violation(conf12) -> (pos_err_m, rot_err_rad)`: where the right
  tool actually is versus where it should be if the bar were where the left
  tool says it is. Expected right tool pose = bar_from_left times grasp_R.
  The error is the position distance and the rotation angle between actual
  and expected right tool0 poses.
- Check: for any configuration that came out of `solve_dual_arm_pose_ik`
  the violation must be at FK-gate level (sub-millimeter, sub-0.01 rad).
- Two FK calls per evaluation; PyBullet FK is microseconds, so this is cheap
  compared with IK, which matches Caelan's cost picture.

**Step 2. Node sampling (`sampling.py`).** Layers on step 1 only for the
sanity check.

- One sample: pick the "driver" arm (alternate left and right), draw six
  joint values for it, FK it to its tool0 pose, derive the bar pose through
  its grasp, derive the other arm's tool0 target through the other grasp,
  call `ssik_arm_branches` with `seed6=None` to get **every** branch of the
  other arm. Each branch gives one candidate node (12 values). Filter each
  with the collision predicate.
- Where to draw the driver arm's joints from. Uniform over the full UR limits
  (shoulder pan is plus or minus two turns) mostly produces wound-up arms
  that are useless. Use a mix: half the draws uniform inside a box spanned
  by the start, goal and home configurations plus a margin of about 90
  degrees per joint; half Gaussian around one of the start, goal or home
  configurations. Record which mode produced each node so we can see later
  which one mattered.
- Always add as nodes: the start configuration, the goal configuration, and
  every collision-free branch pair recorded in the library for the goal pose
  (`free_pairs`) and, if available, for the start pose. This gives the
  search several entry and exit points, which is the cheap version of the
  "multi-source" idea from the meeting.
- Counters: samples drawn, IK calls, branches returned, collision checks,
  nodes kept, kept-by-mode.

**Step 3. Edges and costs (`roadmap.py`).** Layers on steps 1 and 2.

- Neighbors: k nearest in 12-D joint space with raw joint differences (no
  wrapping, same reasoning as `joint_step_deltas`), k around 10 to 15, plus a
  radius cap so far-apart nodes are never joined. Use `cKDTree` with the
  Chebyshev metric so the neighbor distance is the largest joint move.
- Edge check: interpolate in joint space with a step no larger than the
  continuity threshold (so every step would pass today's gate by
  construction) and evaluate at each interior waypoint:
  - collision: hard reject for the pilot (a soft proximity cost is a later
    option; the compas_fab check returns a boolean, not a distance);
  - holding violation from step 1: keep the maximum and the sum along the
    edge.
- Edge cost = joint-space length + `w_hold` times max(0, violation minus
  tolerance), with `w_hold` large (the meeting's "big coefficient").
  Tolerance is a parameter; start at 5 mm and 2 degrees. Store both raw
  numbers on the edge so the same graph can be re-searched with a different
  tolerance or weight without recomputing anything.
- Counters: candidate edges, collision-rejected edges, accepted edges,
  collision checks, FK evaluations, wall time per stage.

**Step 4. Search (`search.py`).** Layers on step 3.

- Dijkstra from the set of start nodes to the set of goal nodes (add a
  virtual source and sink with zero-cost edges, or run multi-source).
- Path report: node list, per-edge and per-waypoint violation profile, the
  maximum violation and where it sits, joint path length, and which start
  and goal branch pair the path used. The last item directly answers the
  branch-pairing question from the meeting.
- Also report the best path under **each** of a few tolerances (e.g. 2 mm,
  5 mm, 1 cm, 3 cm) from the same graph, to show how feasibility changes
  with the tolerance. This is the "what is the gap" plot Caelan asked for.

**Step 5. The pilot driver and CLI (`benchmark/prm_pilot.py`, `prm-pilot`
subcommand).** Layers on steps 1 to 4.

- Inputs: bar id, goal index, start index (the library's numbering), node
  budget, k, tolerance, random seed. Loads the trimmed cell and the library
  entry the same way `cmd_run` does, opens one `SceneCtx`, runs steps 2 to 4,
  and writes one JSON with all counters, the path and the violation profile
  to `HUSKY_BENCH_ROOT/prm_pilot/<bar>/<goal>_<start>_<seed>.json`.
- First target: the start that failed on all ten seeds in the `night1` batch,
  and two starts that succeeded, on the same goal. Then the library's other
  goals for that bar. Bar B3 is the one whose costs are already measured.
- Node budgets to sweep: 500, 1000, 2000, 4000. Record success (path under
  tolerance) per budget. This produces the "how many samples are needed"
  number that decides the GPU question.
- Reuse the profiling context from `benchmark.profiling` so the IK versus
  collision versus FK split is measured, not guessed.

**Step 6. Refinement (`refine.py`).** Layers on step 5.

- Given a path from step 4, draw new nodes by perturbing each path
  configuration (Gaussian, a few degrees per joint) and re-projecting through
  the step 2 sampler (drive with one arm, IK the other) so the new nodes hold
  the bar exactly. Also try direct shortcuts between path nodes two or more
  apart, scored with the step 3 edge check.
- Build the graph over the old path nodes plus the new ones, search again,
  repeat while the maximum violation keeps dropping. Same search and cost
  code, smaller graph each round.
- Stop when the best path's violation is under the tolerance or has not
  improved for three rounds.

**Step 7. Bridge to a production path.** Layers on step 6.

- A path whose violation is under a few millimeters is a joint path that
  nearly holds the bar. Re-solve it exactly with `core.repair_joint_path`
  (per-waypoint IK warm-seeded from the previous waypoint, with the
  production continuity gate and full collision). If the repair passes, the
  result is a path today's planner would accept, and can be handed to
  `plan_constrained_dual_arm`'s callers.
- Compare against the `night1` BiRRT results on the same tasks: solved or
  not, wall time, IK count, path length.
- Only after this works: add `use_prm` (or a variant name) to
  `plan_constrained_dual_arm` and a `variants.py` entry so the benchmark
  runner can batch it.

### Quick diagnosis to run before or alongside step 1

Caelan's first suggestion needs no new code. On the unsolved start, rerun the
benchmark with:

- `--variant v2_nocont` (continuity gate off, already in `variants.py`);
- a scene with the bar-robot and self-collision pairs allowed (needs a
  small variant that widens the ACM; a one-line entry if `ik_options` or the
  template state can carry it, otherwise a short new variant);
- one arm ignored is harder to fake in the current planner and is skipped.

Whichever relaxation first makes the case solvable tells us what the PRM's
cost function must be soft on.

### What to look at when the pilot has run

- Success versus node budget, per tolerance. If 1000 to 2000 nodes solve the
  problems the BiRRT could not, the CPU pilot already beats production and
  GPU work is optional.
- The violation profile of the best path on a "disconnected" pair. A single
  narrow spike means a branch change that a denser local sample can smooth
  out (step 6). A broad plateau means the pair really needs the bar to let
  go, so the answer is a different start pose, not a better planner.
- Which start and goal branch pairs the winning paths used. If they differ
  from what `_ssik_pair_goal_branch_with_home` picked, the pairing rule needs
  to change.
- The IK share of wall time. Expected to dominate at 10-20 ms per call; the
  count is what decides whether batching on a GPU is worth it.

### Follow-ups, not part of the pilot

- Soft collision-proximity cost (needs a distance query, not the boolean
  check we have).
- The mirror variant: task-space roadmap with continuity as the soft cost.
- Re-checking a stored roadmap against a new scaffolding scene instead of
  rebuilding it.
- The intermediate carry pose study (Part 1, second topic): evaluate
  candidate poses over the library's goals, or search backward from contact.
