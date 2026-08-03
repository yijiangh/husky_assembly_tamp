# VAMP Manifold-Constrained Planner as an M1 Backend — Feasibility & Integration Design

<!-- * Written 2026-07-22 as the outcome of the integration feasibility study.
     * Sources: vamp @ constrained_planner branch (github.com/CoMMALab/vamp),
     * cricket @ constraints branch, foam (github.com/CoMMALab/foam),
     * paper "Vectorizing Projection in Manifold-Constrained Motion Planning" (arXiv:2604.13323). -->

## 1. What we want

Add VAMP's Constrained RRT-Connect (**C-RRTC**) as a second backend for the **M1**
constrained dual-arm movement (both grippers rigidly holding a bar, free-space move
from home to approach), next to our existing `plan_pose_birrt` in
`husky_assembly_tamp/motion_planner/dual_arm_task_space_rrt/core.py`, and compare
planning time and plan quality on the same problems.

The two planners formulate the same closed-chain problem differently:

| | ours (`plan_pose_birrt`) | VAMP C-RRTC |
|---|---|---|
| search space | bar SE(3) pose (6-DOF task space) | 12 arm joints (joint space) |
| constraint handling | exact — IK per waypoint from the sampled bar pose | projection — Levenberg-Marquardt steps onto the constraint manifold (tolerance band) |
| collision checking | compas_fab PyBullet, exact meshes + ACM | compiled SIMD sphere approximation, no ACM |
| IK | ssik analytical / PyBullet gradient | none needed (joint-space planning) |
| reported speed class | seconds | milliseconds (paper: 30–60x faster than IK-BiRRT on a bimanual KUKA closed-chain task — literally our problem shape) |

## 2. Feasibility findings (verified, not assumed)

1. **Windows native: not possible.** VAMP builds with GCC/Clang + CMake on Linux/macOS
   only; the pip release (`vamp-planner` 0.6.4) ships no Windows wheels and — important —
   **does not contain the constrained planner at all**. The `constrained_planner` branch
   must be built from source.
2. **WSL2 works and is already set up on this machine.** Ubuntu-22.04 (default distro,
   running) has the full required toolchain out of the box: cmake 3.22, gcc 11.4,
   Eigen 3.4, python3.10 + dev headers, AVX2 CPU flag, 16 cores. `pip install ./vamp`
   of the branch builds there (verified in this study). Docker Desktop (linux backend)
   is also available; VAMP's `docker/ubuntu2204.dockerfile` is 8 lines (apt deps +
   `pip install .`), so containerizing the final setup is trivial.
3. **The bimanual closed-chain constraint is a first-class citizen.**
   `vamp_module.BimanualTaskSpaceConstraint(rTl, lb6, ub6)` fixes the right-EE pose
   relative to the left EE (`rTl = [qw,qx,qy,qz,tx,ty,tz]`) within a 6-DOF tolerance
   band — exactly our rigid two-gripper grasp. The planner call is
   `crrtc(start12, goal12, env, crrtc_settings, constraint, sampler)`, and constrained
   shortcutting exists (`simplify_with_constraints`).
4. **Custom robots are compiled, not configured.** VAMP has no URDF loader at runtime;
   each robot is a generated C++ header (SIMD FK + sphere collision + constraint
   kernels). The pipeline is: **foam** (spherize URDF) → **cricket** `fkcc_gen`
   (trace FK/collision/constraint code) → drop header into a vamp fork → rebuild.
   The existing `bimanual_panda` (123k-line generated header) came from exactly this
   pipeline via `cricket/resources/bipanda.json`, which is only: name, spherized URDF,
   SRDF, two end-effector link names, template paths. For any robot with 2 EEs, cricket
   auto-generates the bimanual constraint kernels and vamp's binding template
   auto-exposes `BimanualTaskSpaceConstraint` + `crrtc` in Python (`n_eef > 1`
   compile-time switch in `src/impl/vamp/bindings/robot_helper.hh`).
5. **Our URDF is already the right shape.**
   `asset/husky_urdf/mt_husky_dual_ur5_e_moveit_config/urdf/husky_dual_ur5_e_no_base_joint_All_Calibrated.urdf`
   has exactly the 12 arm revolute joints actuated (wheels/chassis fixed, no gripper
   links — grippers are compas_fab tools), fixed base, plus
   `config/dual_arm_husky.srdf` with 98 disabled-collision pairs that cricket consumes
   directly. foam resolves `package://pkg/...` mesh URIs by stripping the prefix and
   resolving relative to the URDF's folder — and `asset/husky_urdf/` contains all
   referenced package dirs as direct children, so a URDF copy placed there resolves
   with zero mesh surgery.
6. **compas_fab collision checker reuse inside VAMP: not possible.** The SIMD-compiled
   sphere checks *are* the planner's speed; there is no pluggable checker interface.
   Consistency is instead ensured by construction + verification (§3.4).

## 3. Integration design

### 3.1 Architecture: Windows client → WSL2/Docker sidecar server

Mirror the proven ssik sidecar pattern (`keyframe/ssik_client.py` +
`keyframe/ssik_sidecar/serve.py`): a long-lived subprocess speaking newline-delimited
JSON over stdin/stdout with a `{"ready": true}` handshake. The only new twist is the
subprocess command is `wsl -d Ubuntu-22.04 -- <venv-python> <server.py>` (or
`docker run -i husky-vamp-server`). Files:

- `husky_assembly_tamp/motion_planner/vamp_backend/client.py` — Windows side.
  Exports the scene, sends one `plan` request per M1 query, converts the returned
  joint path back to our 12-vec convention (same joint order: left 6 then right 6).
- `husky_assembly_tamp/motion_planner/vamp_backend/serve_vamp.py` — Linux side.
  Imports the forked vamp, builds the `Environment` + `Attachment`s + constraint from
  the request, runs `crrtc` + `simplify_with_constraints`, replies with the path and
  timing. Runs fine under WSL's system python3.10 in the build venv.

Request payload (all geometry expressed **in the robot base frame** — VAMP robots are
compiled at the origin, so the client transforms world data by
`inv(world_from_robot_base)` from `start_state.robot_base_frame`):

```json
{
  "cmd": "plan",
  "start": [12 floats], "goal": [12 floats],
  "rTl": [7 floats], "tolerance": [6 floats],
  "cuboids": [{"center": [3], "euler_xyz": [3], "half_extents": [3]}],
  "spheres": [...], "pointcloud": {"points": [[3]...], "point_radius": 0.0025},
  "attachments": [
    {"eef": 0, "tf_ee_from_att": [16], "spheres": [{"xyz": [3], "r": 0.01}, ...]},
    {"eef": 1, ...}
  ],
  "settings": {"range": 1.0, "num_projection_iterations": 15, "simplify": true}
}
```

### 3.2 Robot compilation (one-time, re-run only when the URDF changes)

1. Copy the calibrated URDF to `asset/husky_urdf/husky_dual_ur5e_vamp.urdf`
   (so `package://` resolves), prune `world_link`/inertial-only links if cricket
   complains, keep the 12 revolute joints.
2. **foam**: `python scripts/generate_sphere_urdf.py <urdf>` → `husky_dual_ur5e_spherized.urdf`.
   Tune per-link sphere budgets (`--branch`, per-link overrides) — start coarse
   (~8 spheres/link, like bimanual_panda's 118 total) and refine where the
   post-validation rejection rate (§3.4) says the approximation is too loose/tight.
3. **cricket** (constraints branch, built via its `Dockerfile.cricket` — Ubuntu Noble +
   robotpkg Pinocchio + CppAD/CppADCodeGen/CGAL; this is where the user-suggested
   Docker container fits): config `bihusky.json` ≙ `bipanda.json` with
   `"end_effectors": ["left_ur_arm_tool0", "right_ur_arm_tool0"]` and
   `"srdf": dual_arm_husky.srdf`. Run `fkcc_gen bihusky.json`, then `fix_robot.py`,
   `clang-format`.
4. **vamp fork**: add `src/impl/vamp/robots/husky_dual_ur5e.hh`, append
   `husky_dual_ur5e;HuskyDualUR5e` to `VAMP_ROBOT_MODULES/STRUCTS` in `pyproject.toml`,
   `pip install .` in the WSL venv. Optionally trim the module list to just ours +
   bimanual_panda to cut the ~20 min build down.
5. Validate the compiled model: `vamp_module.eefk(q)` vs compas_fab FK on a grid of
   configs (tool0 pose agreement), and `vamp_module.validate(q, env)` vs
   `planner.check_collision` on random configs (sphere model must be conservative:
   VAMP-free ⇒ cfab-free ideally; measure the disagreement rate both ways).

### 3.3 Scene transfer (per M1 query, in the client)

| our data (RobotCell.json / state) | VAMP object |
|---|---|
| built bars + jig + static rigid bodies (`rigid_body_models[*].collision_meshes` verts/faces) | `env.add_pointcloud(...)` (CAPT) from surface-sampled meshes; pure boxes → `vamp.Cuboid` exactly |
| held bar (rigid grasp `tool0_from_bar` from `attached_to_link`/`attachment_frame`) | `vamp.Attachment(tf=left_tool0_from_bar)` + line-of-spheres along the bar, `env.attach(a, eef_id=left)` |
| gripper tool geometry (compas_fab tools, not in URDF) | per-EE `Attachment` spheres (spherize the two tool meshes once, offline) |
| ACM / touch_links at the grasp | no ACM in VAMP → *trim* attachment/pointcloud spheres that collide with robot spheres at the (known-valid) start config; valid because the grasp is rigid, so attachment↔robot clearances near the wrists are constant along any constrained path. `filter_self_from_pointcloud` helps for the environment cloud |
| `grasp_bar_from_left/right` | constraint `rTl = pose of right tool0 in left tool0 frame` = `inv(bar_from_left_tool0... )` composition; tolerance ±1e-3 like the vamp example |
| start/goal 12-vec (from existing derive-start + ssik goal IK) | `start`, `goal` args (must satisfy the constraint — ours do exactly, by construction) |

### 3.4 Modeling consistency: verify-at-the-boundary

Since VAMP's collision world is an approximation, every returned path is
**post-validated on the Windows side with the existing compas_fab checker**
(`_build_cfab_collision_fn` from `motion_planner/api.py`, plus the
`joint_step_exceeds_threshold` continuity gate), densely interpolated at the same
resolution our BiRRT uses. Accept → convert via
`trajectory_io` conventions into the movement trajectory exactly like the BiRRT path.
Reject → report which waypoint/pair collided (that feedback drives sphere-budget
tuning in §3.2) and optionally retry with an inflated `point_radius`.
This keeps compas_fab as the single source of truth for what counts as a valid plan —
the same role it already plays for ssik IK results.

### 3.5 Backend selection + comparison

- `plan_constrained_dual_arm(..., m1_backend="birrt"|"vamp")`, CLI
  `--m1-backend vamp` in `scripts/headless_bar_action_planner.py`, mirroring the
  `HUSKY_IK_BACKEND` pattern (explicit validation, no silent fallback; clear error
  pointing at the WSL setup steps if the sidecar can't start).
- `info` dict gains `planner: "vamp_crrtc"`, `vamp_ms` (pure plan time inside the
  server), `roundtrip_ms`, `post_validation` outcome — alongside the existing
  `[timing]` roll-up.
- Comparison script `scripts/compare_m1_backends.py`: for each BarAction problem run
  both backends N seeds; record success rate, wall-clock (incl. IPC), pure planner
  time, waypoint count, joint path length Σ‖Δq‖, bar-pose path length (positional +
  rotational, via FK), max joint step; emit a Markdown table like
  `dual_arm_task_space_rrt/run.py`'s `write_report`.

### 3.6 Sphere-model fidelity — measured findings (2026-07-23)

The sphere approximation is the backend's real engineering surface. What the
first compile iterations established, with numbers:

- foam's defaults produced 62 spheres (bulkhead plate: 4, wrists: 1 each) —
  far too coarse: valid configs near the chassis were phantom-rejected.
  Per-link budget multipliers (CLI kwargs, e.g. ``--dual_arm_bulkhead_link 6
  --left_ur_arm_upper_arm_link 3``) raise the budget where it matters;
  148- and 208-sphere models followed.
- The choke points are geometric, not tunable away: PyBullet closest-point
  measurement shows the UR5e's own ``base_link_inertia``-vs-``upper_arm``
  clearance is intrinsically ~1.3-1.5 cm across normal configs (the SRDF does
  NOT disable that pair), and bar poses near the chassis put the upper arms
  within 0-2 cm of the ``dual_arm_bulkhead_link`` plate.
- Consequence: **test/benchmark endpoints must be generated with explicit
  clearance margins** (closest-point checks per pair class; pp's
  ``max_distance`` is not applied between robot links). The scenario
  generator enumerates ssik branch pairs and keeps the first that clears
  3 cm to chassis / cross-arm and 1 cm on the intrinsic same-arm pairs.
- vamp's per-robot ``debug(q, env)`` returns the colliding sphere-index pairs;
  cricket's fkcc_gen stdout gives the sphere->link map. Together they make
  phantom-collision triage mechanical (see ``~/mcvamp/debug_selfcc.py``).
- If a needed clearance band still can't be met by budgets alone, foam's
  ``--shrinkage <1.0`` trades conservatism for permissiveness — acceptable in
  this architecture because every returned path is post-validated by the
  exact compas_fab checker (§3.4).

## 4. Risks / open items

- **Attachment↔robot false positives at the grasp** (no ACM): handled by the trim
  step in §3.3; verified empirically during §3.2.5 validation.
- **Projection tolerance vs our exact closed chain**: C-RRTC keeps the constraint
  within ±tol (1 mm / ~0.06° class), not exactly zero. Post-validation FK-checks the
  grasp consistency the same way `validate_dual_arm_bar_pose` does; if drift matters
  we re-project waypoints with one ssik IK pass per waypoint (cheap, analytical).
- **cricket build friction** (Pinocchio/CppADCodeGen): mitigated by using their
  `Dockerfile.cricket` unchanged; codegen is one-shot.
- **g1/digit modules bloat the vamp build**: trim `VAMP_ROBOT_MODULES` in our fork.
- **Windows↔WSL IPC overhead**: single JSON round-trip per plan (~ms for our scene
  sizes) against a planner that runs in ms — measure `roundtrip_ms` separately so the
  comparison stays honest.

## 5. Current state (2026-07-23, end of feasibility study)

**Done and verified:**
- vamp `constrained_planner` branch built from source in WSL2. Working install:
  `~/mcvamp/venv208` (pristine venv, wheel-installed — see §6 ops notes).
  Clones: `~/mcvamp/{vamp,cricket,foam}`; docker images `foam-dev`/`cricket-dev`.
- `bimanual_panda` reference demo: C-RRTC solves the 14-DoF closed-chain box
  problem in ~16 ms mean; simplification 57 -> 15 waypoints in 1.6 ms.
- **Custom robot compiled**: `husky_dual_ur5e` (12 DoF, 2 EEs, 208 spheres,
  26 MB generated header) via foam -> cricket -> vamp fork. FK validated exact
  against compas_robots on the calibrated URDF (worst error 1.3e-7 m).
  Joint order = left arm then right, same as the repo's 12-vec convention.
  `eefk()` returns one 4x4 matrix per EE.
- **Sidecar backend E2E works**: `motion_planner/vamp_backend/{serve_vamp,client}.py`,
  Windows -> `wsl` -> venv208 -> C-RRTC -> reply. Warm roundtrip 7.2 ms
  (plan 9 ms inside), cold start ~2 s. Handshake reports `husky_n_spheres`
  to catch stale installs.
- Scenario generator (`scratchpad/gen_tight_case.py`): ssik branch-pair
  enumeration with per-pair-class mesh clearance margins; produces
  ground-truth-valid closed-chain start/goal cases.

**Known limitations to address next:**
1. **Larger displacements don't connect yet.** Short bar displacements
   (~5-8 cm) solve in single-digit ms (often a direct constrained connect);
   a ~10 cm + 8 deg case ran 30+ s / ~90k iterations without a path across a
   settings sweep (range 0.3-2.0, projection iters 15-60, std 0.01-0.1).
   Diagnosis: samples span the FULL +/-2pi UR joint ranges, so nearly all
   projected samples are useless (the paper's robots have +/-3 rad ranges and
   no chassis). Fix: emit `set_lows`/`set_highs` in the cricket template (the
   UR5 module has them; ours doesn't) or clamp joint limits in the URDF prep
   step to a task envelope, e.g. hull(start, goal) padded by ~1 rad.
2. **`simplify_with_constraints` segfaults on trivial 2-waypoint paths**
   (upstream bug; guarded in serve_vamp.py and the test).
3. nanobind refuses row views of `result.path.numpy()` as arguments — pass
   fresh `np.array(..., dtype=np.float32)` copies (fixed in serve_vamp.py).
4. Remaining integration milestones: scene export from the RobotCell state
   (§3.3) → `--m1-backend vamp` in `api.plan_constrained_dual_arm` with cfab
   post-validation (§3.4/3.5) → `scripts/compare_m1_backends.py` on real
   BarActions (needs the data root location).

## 6. WSL operations notes (hard-won)

- Long builds must run detached INSIDE WSL: `nohup setsid bash script.sh <nonce> &`
  with a nonce marker file and `sync` at the end. Never let a Windows-side
  background task own a long WSL build: on task timeout the client detaches and
  the machine accumulated 8 zombie `wsl.exe` processes, ultimately producing a
  WSL VM split-brain (two live instances of the distro with divergent
  filesystem views; only one instance's writes survived `wsl --shutdown`).
- Recovery from weird state: `Stop-Process -Name wsl -Force` + `wsl --shutdown`,
  then verify durable state before rebuilding.
- Always gate on provenance before trusting results: `sha1sum` of the installed
  `_core_ext*.so` + `module.n_spheres()`.
- scikit-build's build dir is in-source (`vamp/build`) and shared by concurrent
  pips — wipe it if two builds may have raced.
