# Technical Report — `plan_constrained_dual_arm` (M1 constrained dual-arm SE(3) RRT)

Module: `husky_assembly_tamp.motion_planner.api`
Core: `husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.core`
IK adapter: `husky_assembly_tamp.motion_planner.ssik_ik`

Commits under review:
- **`5e226c7`** — *in-process ssik analytical IK backend for the offline planner* (foundation)
- **`55b585a`** — *M1 start-derivation escalations: home variants, corridor shortcut, BiRRT*

> Note on defaults: `55b585a` added `use_birrt=False`; a later commit (`b603a87`, "make bidirectional RRT-Connect the default pose planner") flipped it, so the **current** signature is `use_birrt=True`. This report describes the code as it stands now.

---

## 1. What the two commits changed

### 1.1 `5e226c7` — ssik analytical IK becomes the default backend
The planner's per-waypoint / endpoint IK switched from PyBullet gradient descent to the
analytical **ssik** solver, dispatched everywhere by `ssik_ik.ik_backend()` (env
`HUSKY_IK_BACKEND`, default `"ssik"`, `"gradient"` for the old path).

Changes touching `plan_constrained_dual_arm`'s subtree:
- **`_ik_dual_arm_at_frames`** — ssik branch: each arm takes its analytical branch nearest the seed (collision off), raises `InverseKinematicsError` if a branch set is empty.
- **NEW `_ssik_pair_goal_branch_with_home`** — re-picks the M1 *goal* conf on the IK branch **pair** most compatible with the home (loading) anchor. With real joint limits the goal branch and the home branch chosen independently can be 100–230° apart, so no continuous in-limit motion connects them; this ranks goal branches by distance to the nearest legal home branch and returns the first FK-valid + collision-free combination.
- **`_derive_constrained_start_for_plan`** — on ssik, tries `derive_constrained_start_tracked` (backward tracking from the goal conf) **first**, falling back to the cold `derive_constrained_start` endpoint sweep.
- **Bugfix** — `plan_pose_rrt` diagnostics kwarg is `debug_tree_out`, not `profile_out`; the old name fell into `**_unused_kwargs` and left `info["profile"]` empty.
- Added `info["ik_backend"]`.
- (Also in this commit but on the M2/M3 path, not M1: `_run_dual_arm_cartesian_ik_loop_ssik`.)

### 1.2 `55b585a` — start-derivation escalations + BiRRT switch
- **`home_bar_anchor_variants`** (core) — candidate home anchors beyond the canonical
  orientation: the bar **rolled** about its own long axis and **yawed** about the base
  vertical, smallest-rotation-first. `home_bar_anchor_pose_mb` gains `bar_quat_override`.
  Escape hatch for goals whose branch sheet cannot reach the canonical home orientation
  within joint limits (measured: every goal branch ~176° from every canonical-home branch on the hard "B226" case).
- **`derive_constrained_start_tracked`** — three automatic escalations (below), a 120 s
  soft budget over the whole sweep, and blocked-fraction diagnostics.
- **Direct-connect shortcut** (in `plan_constrained_dual_arm`) — when the derivation
  returns a fully collision-free tracked **corridor**, its reversal *is* the M1 path
  (`info["planner"]="tracked_corridor"`) and the RRT is skipped entirely.
- **`_ssik_pair_goal_branch_with_home`** now evaluates home branches across **all**
  anchor variants, not just canonical.
- **NEW `use_birrt` switch** — dispatch to `plan_pose_birrt` (goal-rooted tree grows out
  of cluttered pockets a single start-rooted tree cannot thread into) vs `plan_pose_rrt`.
  `info["planner"]` records which ran.

---

## 2. Call stack

```
husky_monitor.py  (ROS2 live GUI, M1 button; up to 3 start-retries, max_time=120)
scripts/headless_bar_action_planner.py::_plan_movement  (M1 role, use_birrt/derive_start)
scripts/headless_live_monitor_test.py
        │
        ▼
plan_constrained_dual_arm(planner, start_state, *, active_bar_id, goal_conf|goal_ee_frames,
                          stage=3, use_birrt=True, derive_start=…, …)
│
├─ derive_start=True ─► _derive_constrained_start_for_plan(...)
│   ├─ _ik_dual_arm_at_frames ──────────► ssik_ik.ssik_arm_branches ─► ssik_inprocess.solve
│   ├─ _fk_link_pose_pp (goal FK → world_from_bar_goal, grasp_L/R from authored attachment)
│   ├─ _build_cfab_collision_fn ────────► planner.check_collision (cfab, ACM + attached bar)
│   ├─ _ssik_pair_goal_branch_with_home          [NEW 5e226c7 / extended 55b585a]
│   │    ├─ core.home_bar_anchor_variants  ─► home_bar_anchor_pose_mb
│   │    ├─ ssik_ik.ssik_arm_branches (goal + every home variant)
│   │    ├─ ssik_ik.rebranch_toward_seed  (±2π onto goal branch, limit-checked)
│   │    └─ core.validate_dual_arm_bar_pose  (PyBullet FK ground-truth gate)
│   ├─ core.solve_endpoint_dual_arm_ik  (re-solve goal if colliding)
│   │    └─ _solve_endpoint_dual_arm_ik_ssik ─► ssik_arm_branches
│   ├─ core.derive_constrained_start_tracked     [PRIMARY on ssik; escalations 55b585a]
│   │    ├─ core.home_bar_anchor_variants
│   │    ├─ pp.interpolate_poses (backward track goal → home)
│   │    ├─ core.solve_dual_arm_pose_ik ─► _solve_dual_arm_pose_ik_ssik ─► ssik_arm_branches
│   │    ├─ core.joint_step_exceeds_threshold (continuity gate)
│   │    └─ cfab_collision_fn (arrival must be free; coarse corridor scan)
│   └─ core.derive_constrained_start  (cold endpoint sweep — fallback / gradient backend)
│        └─ auto_compute_home_bar_pose ─► solve_endpoint_dual_arm_ik
│
├─ [direct-connect shortcut]  corridor present → reverse → return (planner="tracked_corridor")  [55b585a]
│
├─ rrt_fn = plan_pose_birrt if use_birrt else plan_pose_rrt      [switch 55b585a]
│   ├─ build_cfab_pose_collision_fn        (floating-bar pose collision)
│   ├─ sample_pose / nearest_node
│   ├─ extend_toward
│   │    ├─ core.solve_dual_arm_pose_ik  (per-waypoint IK, warm-seeded)
│   │    ├─ core.joint_step_exceeds_threshold  (branch-flip reject)
│   │    └─ joint_collision_fn (cfab, stage 3)
│   ├─ (birrt only) _stitch_forward / _stitch_backward ─► solve_dual_arm_pose_ik
│   └─ update_debug_tree
│
└─ smooth.smooth_dual_arm_pose_path        (enable_smoothing, skipped by the corridor shortcut)
```

---

## 3. Top-level control flow — `plan_constrained_dual_arm`

```text
plan_constrained_dual_arm(planner, start_state, *, active_bar_id,
                          goal_conf | goal_ee_frames, stage=3,
                          position_res=0.01, rotation_res=0.025,
                          max_time=30, max_iterations=2000, max_attempts=5,
                          use_birrt=True, enable_smoothing=True,
                          derive_start=False, ...):

    assert exactly one of (goal_conf, goal_ee_frames)
    resolve joint_names_12, arm_joints, tool_link_left/right
    planner.set_robot_cell_state(start_state)          # sync pybullet world
    bar_body, obstacles = bar id, all obstacle puids except the bar

    # ---- Endpoint geometry: start_conf, bar start/goal poses, rigid grasps ----
    if derive_start:                                   # M1 home → approach
        (start_conf, w_bar_start, w_bar_goal, goal_conf_arr,
         grasp_L, grasp_R, derive_info) = _derive_constrained_start_for_plan(...)
        if start_conf is None: return None, derive_info
        planner.set_robot_cell_state(start_state)      # restore cfab cache

        # DIRECT-CONNECT SHORTCUT [55b585a]:
        corridor = derive_info.get("corridor")
        if corridor is not None:
            poses, confs = corridor                    # goal → home order
            return reversed(confs),                    # start → goal path, RRT skipped
                   {planner:"tracked_corridor", stage, ik_backend, path_poses}
    else:                                              # trust start_state config
        start_conf = conf12(start_state)
        FK(start) → tool0_L/R ; w_bar_start = pybullet bar pose
        grasp_{L,R} = inv(w_bar_start) · tool0_{L,R}_start
        goal: from goal_ee_frames (dual-arm IK) OR FK(goal_conf)
        w_bar_goal = tool0_L_goal · inv(grasp_L)

    feature_points ← get_bar_feature_points()
    enable_ik = stage >= 2 ; enforce_collision = stage >= 3
    info ← {stage, ik_backend, planner: "birrt" if use_birrt else "rrt", ...}

    with WorldSaver():
        joint_collision_fn = cfab predicate               if stage 3
        ik_context = {robot, arm_joints, tool links, grasp_L, grasp_R}  if stage >= 2
        rrt_fn = plan_pose_birrt if use_birrt else plan_pose_rrt     # [switch 55b585a]
        path_poses, path_confs = rrt_fn(
            start_pose=w_bar_start, goal_pose=w_bar_goal,
            start_conf=start_conf, goal_conf=goal_conf_arr,
            enable_ik, enable_collision, ik_context, joint_collision_fn,
            position_res, rotation_res, max_time, max_iterations, max_attempts,
            joint_continuity_threshold_rad,
            debug_tree_out=planner_profile)               # [bugfix 5e226c7]
        info["profile"], info["path_poses"] = planner_profile, path_poses
        if path_poses is None:
            return None, info(failure_reason = profile.outcome or "rrt_failed")
        if enable_smoothing and path_confs is not None:
            path_poses, path_confs = smooth_dual_arm_pose_path(...)

    planner.set_robot_cell_state(start_state)              # re-sync cfab cache
    if path_confs is None:
        return None,  info(stage 1 → pose_only_success  else  "no_joint_path")
    return [np.array(q) for q in path_confs], info
```

### 3.1 Start derivation — `_derive_constrained_start_for_plan`

```text
1. goal_conf_arr = dual-arm IK(goal_ee_frames)  OR  conf12(goal_conf)
2. tool0_from_bar ← authored attachment in start_state.rigid_body_states[bar]
3. FK(goal_conf_arr): w_bar_goal, grasp_L, grasp_R
4. world_from_mobile_base ← start_state.robot_base_frame
   cfab_collision_fn ← _build_cfab_collision_fn(...)

   if backend == ssik:                                     # [5e226c7]
       goal_conf_arr = _ssik_pair_goal_branch_with_home(...)   # re-pick goal branch pair
       planner.set_robot_cell_state(start_state)

   if cfab_collision_fn(goal_conf_arr):                    # goal in collision
       goal_conf_arr = solve_endpoint_dual_arm_ik(..., collision_fn)  or  fail "goal_in_collision"

   start_conf = w_bar_start = None ; tracked_info = None
   if backend == ssik:                                     # PRIMARY
       w_bar_start, start_conf, tracked_info =
           derive_constrained_start_tracked(..., goal_conf_arr)
   if start_conf is None:                                  # FALLBACK / gradient backend
       w_bar_start, start_conf = derive_constrained_start(..., seed=goal_conf_arr)
   if start_conf is None: fail "start_derivation_failed"

   info = {derived_start_conf}
   if ssik and "corridor" in tracked_info: info["corridor"] = tracked_info["corridor"]  # [55b585a]
   return start_conf, w_bar_start, w_bar_goal, goal_conf_arr, grasp_L, grasp_R, info
```

### 3.2 Goal/home branch pairing — `_ssik_pair_goal_branch_with_home` (NEW `5e226c7`, extended `55b585a`)

```text
home_poses = [ w·anchor  for _, anchor in home_bar_anchor_variants(...) ]   # all variants [55b585a]
for arm in {left, right}:
    goal_branches = ssik_arm_branches(goal tool pose, seed=None)            # every legal branch
    home_branches = ⋃ over home_poses of ssik_arm_branches(home tool pose)
    if either empty: return goal_conf_fallback                             # keep seeded goal
    for goal_q in goal_branches:
        nearest_home = min over home_q of  max| rebranch(home_q → goal_q) − goal_q |
    rank goal_branches by nearest_home ascending
combos = { top-4 left × top-4 right }, key = worse-arm cross distance, ascending
for (_, conf) in combos:
    if not validate_dual_arm_bar_pose(conf): continue        # PyBullet FK gate
    if cfab_collision_fn(conf): continue
    return conf                                              # best branch-paired goal
return goal_conf_fallback
```

### 3.3 Backward-tracked start — `derive_constrained_start_tracked` (escalations `55b585a`)

```text
anchor_variants = home_bar_anchor_variants(...)      # canonical, then roll±/yaw± growing
deltas = grid_in_box sorted nearest-first
budget = 120 s ; first_start = None ; best_partial = None

for (label, (base_pos, quat)) in anchor_variants:
    variant_deltas = all deltas if canonical else deltas[:60]
    for delta in variant_deltas:
        if time exceeded: break out of the whole sweep
        home = (base_pos + delta, quat)

        # ---- backward track: walk interpolated bar segment GOAL → home ----
        track = [(w_bar_goal, goal_conf)] ; track_ok = True
        for pose in interpolate_poses(w_bar_goal, w_bar_home)[1:]:
            next_conf = solve_dual_arm_pose_ik(pose, seed = track[-1].conf)   # ssik, warm
            if next_conf is None or joint_step_exceeds_threshold(next_conf, track[-1].conf):
                track_ok = False ; break                     # branch flip / unreachable
            track.append((pose, next_conf))

        if track_ok:
            arrival = track[-1].conf
            if cfab_collision_fn(arrival): arrival_collisions++ ; continue

            # ESCALATION 1 — FREE CORRIDOR (coarse-first, every 3rd waypoint)
            first_blocked = scan interior for collision (strides 0,1,2)
            if first_blocked is None:
                info["corridor"] = (poses, confs)            # finished M1 path
                return w_bar_home, arrival, info             # → caller skips RRT
            else:
                blocked_corridors++ ; record blocked_fraction
                if first_start is None: first_start = {w_bar_home, arrival}
                continue
        else:
            track_breaks++
            partial_dist = | track[-1].pos − goal.pos |
            best_partial = max by partial_dist                # farthest-reaching broken walk

# after the sweep:
if first_start: return first_start                            # fully tracked, clean arrival, corridor blocked mid-way
# ESCALATION 3 — PARTIAL start
if best_partial and best_partial.dist >= 2 cm:
    walk best_partial.track back from its farthest waypoint until a collision-free conf ≥ 2 cm from goal
    info["partial"] = True ; return that (pose, conf)         # M0 covers the remaining gap
return None
```
(**Escalation 2**, orientation variants, is the outer `anchor_variants` loop.)

---

## 4. `plan_pose_rrt` — single start-rooted tree (archival)

```text
plan_pose_rrt(start_pose, goal_pose, start_conf, goal_conf, ...):
    rng ← seed ; feature_points
    collision_fn = cfab pose-collision  if enable_collision  else  noop     # floating bar
    # endpoint feasibility
    if joint_collision_fn:  reject if start_conf or goal_conf collides       # stage 3
    else:                   reject if start_pose or goal_pose in bar-collision # stage 1
    extend_stop_reasons = Counter()

    for attempt in range(max_attempts):                 # independent restarts
        root = TreeNode(start_pose) ; nodes = [root]
        node_confs[root] = start_conf   (if IK)
        feature_vecs[root] = feature_vec(start_pose)

        for iteration in range(max_iterations):
            if elapsed >= max_time: break
            target = sample_pose(goal w.p. goal_sample_prob=0.1, else random workspace pose)
            nearest = nearest_node(nodes, target, "feature")
            new_last, reached, stop = extend_toward(nodes, nearest, target, ...)  # grows tree
            extend_stop_reasons[stop]++
            if not reached: continue
            if goal_pose_reached(new_last, goal_pose, position_res, rotation_res):
                update_debug_tree(success)
                path_nodes = new_last.retrace()          # frontier → root, then reversed
                return poses(path_nodes),
                       [node_confs[n] for n in path_nodes]  (if IK)
        # attempt exhausted (time / iteration cap) → restart

    update_debug_tree(failure)
    return None, None
```

### 4.1 `extend_toward` (shared by both planners)

```text
extend_toward(nodes, source, target_pose, ...):
    current, current_conf = source, node_confs[source]
    if enable_ik and (no conf cache / ik_context / current_conf): return source, False, "ik_failure"

    for pose in interpolate_poses(source.config, target_pose)[1:]:   # step (position_res, rotation_res)
        if enable_ik:
            next_conf = solve_dual_arm_pose_ik(pose, seed=current_conf)     # warm
            if next_conf is None:                       stop_reason="ik_failure" ; break
            if joint_step_exceeds_threshold(next_conf, current_conf):
                                                        stop_reason="continuity" ; break   # branch flip
            if joint_collision_fn(next_conf):           stop_reason="collision"  ; break   # stage 3
        if collision_fn(pose):                          stop_reason="collision"  ; break   # floating bar
        append TreeNode(pose, parent=current)           # keep every valid intermediate node
        advance current, current_conf
    return current, reached, stop_reason
```
Key property: intermediate nodes up to the first failure are **kept** in the tree even
when the extension does not reach the target — this is what lets the tree grow into
partially-blocked regions.

---

## 5. `plan_pose_birrt` — bidirectional RRT-Connect (default)

```text
plan_pose_birrt(start_pose, goal_pose, start_conf, goal_conf, ...):
    rng, feature_points, collision_fn
    endpoint feasibility  (identical to plan_pose_rrt)
    make_tree(root_pose, root_conf) → (nodes, node_confs, feature_vecs)

    for attempt in range(max_attempts):
        Ta = make_tree(start_pose, start_conf)          # start-rooted
        Tb = make_tree(goal_pose,  goal_conf)           # goal-rooted  ← threads OUT of pockets

        for iteration in range(max_iterations):
            if elapsed >= max_time: break

            grow_a_first = |Ta| <= |Tb|                 # balance: smaller tree grows
            tree_grow, tree_other = (Ta,Tb) if grow_a_first else (Tb,Ta)

            # sample: w.p. goal_sample_prob=0.05 bias toward the OTHER tree's root; else random
            target = other_root  or  sample_pose(random)

            nearest  = nearest_node(tree_grow, target)
            new_last, _, stop = extend_toward(tree_grow, nearest, target, ...)
            if new_last is nearest: continue            # no progress this iteration

            # ---- CONNECT: extend the other tree ALL THE WAY to the new frontier pose ----
            connect_target       = new_last.config
            connect_nearest      = nearest_node(tree_other, connect_target)
            connect_last, connect_reached, _ =
                extend_toward(tree_other, connect_nearest, connect_target, ...)
            if not connect_reached: continue

            # ---- STITCH into start → goal order, dropping the duplicate seam node ----
            if grow_a_first: start_side = new_last.retrace()     ; goal_side = reversed(connect_last.retrace())
            else:            start_side = connect_last.retrace() ; goal_side = reversed(new_last.retrace())
            goal_side_tail = goal_side[1:]                       # drop seam duplicate
            path_poses = poses(start_side) + poses(goal_side_tail)

            if enable_ik:
                start_confs = [node_confs_a[n] for n in start_side]
                goal_confs  = [node_confs_b[n] for n in goal_side_tail]
                # each tree carries its OWN branch; re-IK across the seam so the whole
                # path lies on ONE IK branch sheet (else the join would branch-flip).
                stitched_goal = _stitch_forward(seed = start_confs[-1])       # rebuild goal side
                if stitched_goal is not None:
                    path_confs = start_confs + stitched_goal
                else:                                                         # fallback
                    rebuilt_start = _stitch_backward(seed = goal-root branch) # rebuild start side
                    if rebuilt_start is None: continue
                    path_confs = rebuilt_start + goal_confs
            update_debug_tree(success)
            return path_poses, path_confs
        # attempt exhausted → restart both trees

    update_debug_tree(failure)
    return None, None
```

### 5.1 Stitch helpers (single-branch-sheet guarantee)

```text
_stitch_forward(seed):            # walk goal-side poses forward from the seam conf
    cur = seed
    for pose in goal_side_poses:
        nxt = solve_dual_arm_pose_ik(pose, seed=cur)
        if nxt is None:                                    reasons["stitch_ik_failure"]++   ; return None
        if joint_step_exceeds_threshold(nxt, cur):         reasons["stitch_continuity_fail"]++; return None
        if joint_collision_fn(nxt):                        reasons["stitch_collision"]++     ; return None
        emit nxt ; cur = nxt
    return goal-side confs

_stitch_backward(seed_at_goal):   # walk start-side poses in reverse, then check start endpoint
    ... same gates, reversed ...
    if joint_step_exceeds_threshold(out[0], start_confs[0]): reasons["stitch_endpoint_mismatch"]++; return None
    return start-side confs
```

### 5.2 Why BiRRT is the default
A single start-rooted tree must *thread into* a goal pose buried in a cluttered pocket —
every extension toward it dies on collision at the pocket mouth. The goal-rooted tree
instead grows **out** of the pocket, where there is free space, and the two frontiers meet
in the open. Both trees stay on one IK branch sheet because (a) the ssik goal/home pairing
and the tracked start put start and goal on compatible branches, and (b) the stitch re-IKs
the seam so the concatenated path passes the same raw-delta continuity gate the RRT uses.

---

## 6. Cross-cutting mechanisms

| Mechanism | Where | Purpose |
|---|---|---|
| **Backend dispatch** | `ssik_ik.ik_backend()` | one env var (`HUSKY_IK_BACKEND`) switches every IK site between analytical ssik (default) and PyBullet gradient. |
| **±2π re-branching** | `ssik_ik.rebranch_toward_seed` | ssik ranks/dedups with wrap-to-π metrics; UR limits exceed ±π, so a legal 2π-twin (wrap dist 0, raw dist 6.28) would look like a huge jump and kill continuity. Shift each joint toward the seed by k·2π when it stays in-limit. |
| **Continuity gate** | `joint_step_exceeds_threshold` (raw L∞, ~10°) | rejects IK branch flips between consecutive waypoints — the single invariant every stage enforces (extend, track, stitch). |
| **FK ground-truth gate** | `validate_dual_arm_bar_pose` | PyBullet FK re-check after ssik, guarding joint-order / frame-convention mismatch. |
| **Collision** | cfab (`planner.check_collision`) | single source of truth (RobotCell ACM + attached bar); no URDF/SRDF file on disk needed. Along-corridor collisions are ignored during tracking (RRT routes around them); only endpoints/arrival must be free. |
| **`allow_rescue=False`** | tracking loops | per-waypoint solves fail fast (~1 ms) instead of ~40 ms numeric rescue whose loose LM solutions can branch-flip. |

---

## 7. Review observations (non-blocking)

1. **`use_birrt` default drift.** `55b585a` shipped `use_birrt=False`; `b603a87` flipped
   it to `True`. Confirm all call sites that omit the arg (`husky_monitor.py`) intend
   BiRRT — they now get it silently.
2. **`profile_out → debug_tree_out` was a real latent bug** (`5e226c7`): before the fix
   `info["profile"]` was always empty, so failure `outcome`/`extend_stop_reasons` never
   surfaced. Anything that parsed `info["profile"]` before this commit saw `{}`.
3. **Corridor shortcut skips smoothing.** The `tracked_corridor` return path bypasses
   `smooth_dual_arm_pose_path`. That is fine (it is already a tracked, resolution-dense
   path), but the returned waypoint density differs from the RRT+smoothed path — downstream
   consumers assuming uniform spacing should be checked.
4. **`_ssik_pair_goal_branch_with_home` cost.** Home branches are now enumerated across
   *all* ~13 anchor variants and every goal branch is ranked against the full union — an
   O(goal × home) scan per arm. It runs once per M1 plan so cost is minor, but the pairing
   quality depends on the variant list staying small.
5. **Coarse corridor scan is only a win on rejection.** The stride-(0,1,2) scan visits
   every interior waypoint when the corridor is clean; the ~⅓ saving materializes only
   when a corridor is blocked early. Matches the intent, worth keeping in mind for timing.
```
