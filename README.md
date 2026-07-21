# husky_assembly_tamp

Task and Motion Planning (TAMP) for the Husky bar-assembly project. Given a
`BarAssemblyAction` exported from Rhino, this package solves the robot base + IK
keyframes and plans full RRT trajectories between them — **all without Rhino
running**. It is the *offline planner* (stage 2) of the three-stage pipeline
(Rhino export → offline planner → live ROS2 execution).

## Package layout

```text
husky_assembly_tamp/
  setup.py                       install spec (pins the whole planning stack)
  husky_assembly_tamp/           the importable Python package
    keyframe/                    base search + dual-arm IK + walkable ground
    motion_planner/              the RRT motion planners
    utils/                       shared helpers
  scripts/                       the two headless entry points (see below)
    headless_bar_action_planner.py   solve a plan
    replay_bar_action_plan.py        inspect / collision-check a plan
```

---

## 1. Install a Python that can run it

Both scripts depend on `compas`, `compas_fab`, `pybullet`, `numpy`,
`matplotlib`, `rs_data_structure`, etc. You need **one** Python that has them.
Pick whichever fits your machine:

### Option A — a clean venv, no Rhino (recommended for headless runs)

This builds a plain Python that has **no connection to Rhino** — it brings its
own copies of every package. This is the right choice for running the planner
headless (on the design machine *or* on Ubuntu), and it is what the rest of this
section assumes.

We use [`uv`](https://docs.astral.sh/uv/) to make the venv and install into it.

**Windows (PowerShell), Python 3.11:**

```powershell
# from the husky_assembly_tamp repo root (the folder with setup.py):

# 1. Create a Python 3.11 venv (uv downloads a standalone CPython 3.11 if you
#    don't have one -- this is NOT Rhino's Python, which is the whole point):
uv venv --python 3.11 .venv

# 2. Install this package + its full planning stack into that venv.
#    (compas_fab and rs_data_structure are fetched from git; pybullet compiles,
#     so the first install takes a few minutes.)
uv pip install -e . --python .venv\Scripts\python.exe
```

**Ubuntu (bash):**

```bash
# from the husky_assembly_tamp repo root:
uv venv --python 3.11 .venv
uv pip install -e . --python .venv/bin/python
```

> **Why 3.11 and not Rhino's 3.9?** Rhino 8 ships CPython **3.9**. This clean
> venv is deliberately a *different* interpreter so it can never accidentally
> pick up Rhino's packages. The scripts import `_rhino_env_bootstrap`, which
> wires Rhino's `py39-rh8` site-envs onto `sys.path` — but that step
> **auto-skips on any non-3.9 interpreter** (Rhino's compiled extensions can't
> load in 3.11 anyway), so a 3.11 venv stays genuinely Rhino-free.

Don't have `uv`? Install it once with `pip install uv` (or see the uv docs), or
use plain `python -m venv .venv` + `pip install -e .` with a system Python 3.11.

The `.venv/` folder is git-ignored — each machine builds its own; never commit it.

### Option B — Rhino 8's Python (Windows design machine only)

On the Windows machine with Rhino 8, you *can* run the scripts with the **Rhino**
Python — the packages already live in its site-envs:

```powershell
C:/Users/yijiangh/.rhinocode/py39-rh8/python.exe scripts/headless_bar_action_planner.py ...
```

Each script calls `_rhino_env_bootstrap.bootstrap_rhino_site_envs()` on import,
which wires the `scaffolding_env` site-env onto `sys.path` for you — you do
**not** have to activate anything. If you see `ModuleNotFoundError: No module
named 'compas'`, you launched it with the wrong Python. (This path only works
under Rhino's py39; a clean venv from Option A is preferred otherwise.)

---

## 2. Point the scripts at your data

Everything is keyed off a **data root** that contains one folder per problem.
The data root is machine-specific (a synced folder), so it is **not** baked into
the code: pass it as the planner's positional `data_root` argument (the replay
tool's `--data-root`), or set the `HUSKY_ASSEMBLY_DATA_ROOT` environment variable
once per machine. The scripts error with instructions when neither is given.

```text
<data_root>/
  <problem>/                       e.g. 2026-05-16_double_kissing_jig_demo
    RobotCell.json                 the robot + tools + all bars/obstacles
    WalkableGround.json            ground meshes (only needed for --base sample)
    BarActions/
      B6.json                      "clean" export from Rhino
      B6.solved_keyframe.json      <- written by --solve-keyframes
      B6.solved_motion.json        <- written by motion planning
```

Set `HUSKY_ASSEMBLY_DATA_ROOT` once so you never repeat the path. Point it at
the folder that *holds* the per-problem subfolders (the parent of `<problem>/`
above), not at a single problem folder.

- **Windows (PowerShell)** — set it for the current session, or persist it:

  ```powershell
  # this session only:
  $env:HUSKY_ASSEMBLY_DATA_ROOT = "C:/Users/yijiangh/Dropbox/0_Projects/2025_husky_assembly/problems"

  # persist for all future sessions (writes to your user environment):
  setx HUSKY_ASSEMBLY_DATA_ROOT "C:/Users/yijiangh/Dropbox/0_Projects/2025_husky_assembly/problems"
  ```

  `setx` only affects **new** terminals, so reopen the shell after running it.

- **Ubuntu (bash)** — set it for the current session, or persist it:

  ```bash
  # this session only:
  export HUSKY_ASSEMBLY_DATA_ROOT="$HOME/husky_assembly/problems"

  # persist: append the same line to your ~/.bashrc, then reopen the shell:
  echo 'export HUSKY_ASSEMBLY_DATA_ROOT="$HOME/husky_assembly/problems"' >> ~/.bashrc
  ```

The paths above are examples — use your machine's actual synced-data folder.

The remaining defaults (baked into `headless_bar_action_planner.py`) are
problem-relative names, not machine paths:

- `--problem` = `2026-05-16_double_kissing_jig_demo`
- `--bar-action` = `B6.json`

A **clean** file is a plain `<bar_id>.json` with no extra dot in the name. The
`*.solved_keyframe.json` / `*.solved_motion.json` sidecars carry an extra dotted
tag, so batch runs (`--all`) skip them and never re-solve a sidecar.

---

## 3. Headless BarAction workflow

Two scripts, two halves of the same pipeline — the planner *produces* a solved
BarAction file, the replay tool *inspects* it. Neither needs Rhino running.

| Script | What it does | Writes |
|---|---|---|
| [`headless_bar_action_planner.py`](scripts/headless_bar_action_planner.py) | **Solves** a BarAction: base + IK keyframes (the headless twin of `RSIKKeyframe`), or full RRT motion planning. | `*.solved_keyframe.json` / `*.solved_motion.json` sidecars |
| [`replay_bar_action_plan.py`](scripts/replay_bar_action_plan.py) | **Replays + checks** an already-produced plan: plots the configs, collision-checks them, and (with `--gui`) lets you scrub them in PyBullet. | `bar_action_keyframes.png` / `bar_action_motion.png` |

> In every example below, **`PYTHON`** stands for the interpreter you chose in
> section 1 — either the clean venv's `.venv\Scripts\python.exe` (Windows) /
> `.venv/bin/python` (Ubuntu), or Rhino's
> `C:/Users/yijiangh/.rhinocode/py39-rh8/python.exe`. Run everything from the
> repo root so the `scripts/...` paths resolve.

### 3a. `headless_bar_action_planner.py` — solve a plan

This script has **two modes**. Which one runs is decided by whether you pass
`--solve-keyframes`.

#### Mode A — Keyframe solve (`--solve-keyframes`)

The headless twin of the `RSIKKeyframe` Rhino command. It solves the
**M1 → M2 → M3 IK chain** (approach → assembled → retreat) to find a robot base
frame plus one config per keyframe, then stamps that base into every movement's
`start_state` and writes a `*.solved_keyframe.json` sidecar. `--movement` is
ignored in this mode.

`--base` picks where the base frame comes from:

- `--base saved` **(actual default)** — use the base frame already stored in the
  BarAction's M1 `start_state`, no sampling. One solve at that base.
- `--base sample` — load `WalkableGround.json`, take the ground(s) named by the
  bar's `walkable_ground_ids`, seed a base under the bar's assembled midpoint,
  then grow the sample radius in rings until the whole chain solves.

> **Note:** the `--base` *help text* says "sample (default)", but the code default
> is `saved`. Pass `--base` explicitly if it matters to you.

Examples:

```bash
# Solve one bar at its saved base (fast, no ground needed):
PYTHON scripts/headless_bar_action_planner.py \
    --solve-keyframes --base saved --bar-action B6.json

# Solve one bar by sampling on its associated WalkableGround:
PYTHON scripts/headless_bar_action_planner.py \
    --solve-keyframes --base sample --bar-action B6.json

# Batch: solve every clean bar in the problem (skips sidecars):
PYTHON scripts/headless_bar_action_planner.py \
    --solve-keyframes --base sample --all
```

With `--gui --base saved`, the run pauses after **each** solved keyframe, pushes
the pose into the PyBullet window, and prints an ASCII "joint value vs joint
limit" plot so you can see *why* a later keyframe fails (e.g. a wrist near its
limit after M1). This pause hook is off in `--base sample` (it would stop on every
sampled base) and off in batch runs.

Output: `BarActions/<bar>.solved_keyframe.json`, with the found base written into
every movement's `start_state.robot_base_frame` and the solved approach/assembled/
retreat configs chained onto M1/M2/M3 (and M3's retreat also filled into M4's
start).

#### Mode B — Motion planning (`--movement …`)

Runs the RRT motion planners to produce full **trajectories** between keyframes.
Requires `--movement`:

- `--movement all` chains `M1 → M2 → M3 → M4` (M0 is left unplanned — the live
  monitor plans it).
- `--movement M1` (or `M2`/`M3`/`M4`) plans a single movement.

`--load` picks the starting file:

- `--load clean` **(default)** — the Rhino export; plan from scratch.
- `--load solved_keyframe` — start from a keyframe-solve sidecar (its solved base
  + keyframe configs), and fill in the trajectories between them. **This is the
  normal way to plan on top of a keyframe solve.**
- `--load solved_motion` — resume this bar's motion sidecar, keeping already-planned
  trajectories and planning only the missing movements.

Examples:

```bash
# Plan all four movements from the clean export, watch it replay:
PYTHON scripts/headless_bar_action_planner.py \
    --movement all --bar-action B6.json --gui

# Plan motion on top of an existing keyframe solve:
PYTHON scripts/headless_bar_action_planner.py \
    --movement all --load solved_keyframe --bar-action B6.json

# Batch-plan every clean bar:
PYTHON scripts/headless_bar_action_planner.py \
    --movement all --all
```

Output: `BarActions/<bar>.solved_motion.json`, snapshotted after **each** solved
movement (so a run that fails on M3 still leaves M1/M2 on disk). With `--gui`
(single bar, no `--all`, no `--no-replay`) it opens a PyBullet slider to scrub the
whole M1..M4 trajectory.

#### Motion-planning debug flags (single bar only)

- `--probe-endpoints` — M1 only. Derive + report whether a feasible, collision-free
  start and goal dual-arm config exist (goal-IK + start-derivation stage), then
  exit **without** running the RRT. Fast feasibility check; also saves
  `m1_endpoint_confs.png`.
- `--diagnosis` — M4 only, needs `--gui`. Draws the birrt search trees live (red =
  forward/start tree, blue = backward/goal tree, gray = raw samples).
- `--max-time 60` — RRT time budget per movement, seconds (not the IK budget).
- `--no-replay` — skip the slider after planning.

#### All flags at a glance

| Flag | Mode | Meaning |
|---|---|---|
| `data_root` (positional) | both | parent folder of problems (falls back to `HUSKY_ASSEMBLY_DATA_ROOT`) |
| `--problem` | both | problem subfolder |
| `--bar-action B6.json` | both | which clean bar (ignored with `--all`) |
| `--all` | both | every clean bar in `BarActions/` |
| `--cell` | both | override `RobotCell.json` path |
| `--gui` | both | open the PyBullet window |
| `--solve-keyframes` | A | switch to base+IK keyframe solve |
| `--base sample\|saved` | A | where the base comes from (default `saved`) |
| `--walkable-ground` | A | `WalkableGround.json` path (for `--base sample`) |
| `--movement M0..M4\|all` | B | which movement(s) to plan (required in mode B) |
| `--load clean\|solved_keyframe\|solved_motion` | B | starting file |
| `--max-time` | B | RRT budget per movement (s) |
| `--probe-endpoints` | B | M1 feasibility only, no RRT |
| `--diagnosis` | B | draw M4 search trees live (`--gui`) |
| `--no-replay` | B | skip the replay slider |

### 3b. `replay_bar_action_plan.py` — inspect / check a plan

Loads a produced BarAction file and **plots + collision-checks** it; with `--gui`
it also scrubs it in PyBullet. It does **not** re-solve — it reads what the planner
already wrote. `--load` picks the file and the view:

- `--load clean` **(default)** / `--load solved_keyframe` — the **keyframe view**:
  plots the discrete configs (M1.start → M2.start → M3.start → M4.target), each
  rendered with that movement's own attachments (bar held in M1/M2, released in
  M3/M4), and full-report collision-checks each. Saves `bar_action_keyframes.png`.
- `--load solved_motion` — the **motion view**: loads every movement's full
  trajectory, collision-checks **every waypoint** (quiet, reported per-movement,
  e.g. `M1: 15/15 collision-free`), and plots joint value vs waypoint index with
  the movement boundaries marked. Saves `bar_action_motion.png`. Falls back to the
  keyframe view if the file has no trajectories.

Examples:

```bash
# Check the clean export's authored keyframes:
PYTHON scripts/replay_bar_action_plan.py \
    --load clean --bar-action B6.json

# Check a keyframe solve, and scrub it in PyBullet:
PYTHON scripts/replay_bar_action_plan.py \
    --load solved_keyframe --bar-action B6.json --gui

# Check every waypoint of a planned motion:
PYTHON scripts/replay_bar_action_plan.py \
    --load solved_motion --bar-action B6.json --gui
```

Flags: `--load {clean,solved_keyframe,solved_motion}`, `--gui`, `--bar-action`,
`--problem`, `--data-root` (falls back to `HUSKY_ASSEMBLY_DATA_ROOT`, same as
the planner).

### 3c. The end-to-end loop

```text
Rhino: export BarAction  ->  BarActions/B6.json  (clean)
              │
              ▼   headless_bar_action_planner.py --solve-keyframes --base sample
        B6.solved_keyframe.json  (base + keyframe configs)
              │
              ▼   replay_bar_action_plan.py --load solved_keyframe --gui   (eyeball / collision-check)
              │
              ▼   headless_bar_action_planner.py --movement all --load solved_keyframe
        B6.solved_motion.json    (full trajectories)
              │
              ▼   replay_bar_action_plan.py --load solved_motion --gui     (check every waypoint)
```

Add `--all` to the two planner steps to run the whole problem instead of one bar.
