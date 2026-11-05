# Running the holding-constraint roadmap (PRM) planner and its dashboards on Ubuntu

A setup-and-run manual for a second machine. It covers: getting the right
branch of the code, building a Python environment that matches the one the
experiments ran in, getting the data across, serving the two dashboards,
running the roadmap planner (precompute, online planning in a real scene, and
the library experiments), and what was changed in the code to make all of this
portable. Nothing here needs Rhino, ROS 2 or a display.

The commands are written for bash on Ubuntu 22.04 or 24.04. Where a Windows
command differs it is shown too, so the same page serves both machines.

---

## 1. What runs where

| piece | what it is | where it lives |
|---|---|---|
| roadmap planner | `husky_assembly_tamp/motion_planner/holding_prm/` | the `yh/holding-prm` branch of this repo |
| its command lines | `scripts/prm_pilot.py`, `prm_carry_search.py`, `prm_bridge.py`, `plan_process_carry.py` and `prm_check_solved.py` | same branch |
| benchmark (BiRRT) | `husky_assembly_tamp/benchmark/`, `scripts/benchmark_cdfm.py` | same branch |
| dashboards | `benchmark/dashboard/` (two pages: `index.html` for the BiRRT benchmark, `prm.html` for roadmap runs) | same branch; copied next to each run when exported |
| data | trimmed scenes, libraries, roadmap runs, benchmark batches | the shared Google Drive folder `2025-03 Husky Assembly/data_experiment/cdfm_planner_benchmark` |
| exported problems | one *process folder* per design study (`process.json` + `actions/` + `robots/` + `meshes/`, schema 2.1, written by Rhino's `RSExportProcess`); section 7 | `2025-03 Husky Assembly/data_design_study/<problem>/` |
| shared core library | `scaffolding_assembly_core`: reads process folders, turns a movement into a scene and a scene into a compas_fab cell | its own repo, vendored by the design workflow as `external/scaffolding_assembly_core`; installed by the `[process]` extra (section 3) |

The branch matters: `main` and the detached checkout used with Rhino do not
have the roadmap planner at all. Everything below assumes `yh/holding-prm`.

---

## 2. Get the code

Clone this repo on its own. It does not need the design-workflow repo that
vendors it on the Windows machine: its source dependencies that live as
siblings there (`compas_fab` and `scaffolding_assembly_core`) are installed
from git by `setup.py`, at the same commits.

```bash
cd ~
git clone -b yh/holding-prm git@github.com:yijiangh/husky_assembly_tamp.git
cd husky_assembly_tamp
```

If the branch has not been pushed since the portability changes (section 8),
push it from the Windows machine first:

```powershell
# Windows, in the worktree that holds the branch
git push origin yh/holding-prm
```

The per-arm analytical IK solvers (`asset/ssik/*.py`) are committed, so a
plain checkout is self-contained. No `ssik build` step is needed.

---

## 3. Python environment

Python 3.11 with the versions in `constraints-py311.txt` (the stack the
roadmap experiments were run with). [`uv`](https://docs.astral.sh/uv/) makes
the venv and downloads a standalone 3.11 if the machine has none.

```bash
# once per machine
curl -LsSf https://astral.sh/uv/install.sh | sh      # installs ~/.local/bin/uv
sudo apt install -y build-essential git              # pybullet compiles from source if no wheel matches

# from the repo root
uv venv --python 3.11 .venv
uv pip install -e ".[ssik]" -c constraints-py311.txt --python .venv/bin/python
```

`-e .` installs this package and pulls `compas_fab` (branch `wip_process`)
from GitHub; the first install takes a few minutes because pybullet is
compiled. `[ssik]` adds the analytical IK
backend (the roadmap's IK calls go through it; it ships Linux wheels for
every version). `-c` pins everything else.

To read the exported process folders (section 7) add the `[process]`
extra: `uv pip install -e ".[ssik,process]" -c constraints-py311.txt --python .venv/bin/python`.
It installs the shared core library `scaffolding_assembly_core` from its
GitHub repo over ssh (the repo is private; the machine needs a key that can
read it) at the commit the design workflow vendors, plus `trimesh`,
`yourdfpy` and `lxml`, the readers it uses for URDF and mesh files. Without
a key, clone the core by hand and point the code at the folder holding the
clone: `export HUSKY_EXTERNAL_DIR=~/src` when the clone is `~/src/scaffolding_assembly_core`
(the Windows machine finds it as a sibling under `external/` and needs no
variable).

Check it:

```bash
.venv/bin/python -c "
import compas, compas_fab, pybullet, ssik, scipy, networkx
from husky_assembly_tamp.motion_planner.holding_prm import pilot, online
from husky_assembly_tamp.keyframe import ssik_inprocess
print('ssik in-process:', ssik_inprocess.available())
from husky_assembly_tamp.benchmark import sibling_packages, process_export
process_export.require_core(); print('core: ok')"
```

It should print `ssik in-process: True` and `core: ok` (the last line only
matters for process folders; its error message says how to get the core). If `import ssik` fails the planner
would still run with `HUSKY_IK_BACKEND=gradient`, but that is a different
(slower, gradient-descent) solver and not what the experiments used.

> **Windows equivalent.** `uv venv --python 3.11 .venv` then
> `uv pip install -e ".[ssik]" -c constraints-py311.txt --python .venv\Scripts\python.exe`.
> Everywhere below, read `.venv/bin/python` as `.venv\Scripts\python.exe`.

### Optional: the dashboard self-checks

Two checks in `benchmark/tools/` run the dashboard's own JavaScript outside a
browser and need Node.js (`sudo apt install nodejs`, version 18 or newer):
`check_modules.mjs` loads every module of an exported page and
`test_fk.mjs` replays PyBullet poses through the page's forward kinematics.
Neither is needed to run or view anything.

---

## 4. Data

### 4a. What to sync

The benchmark data root holds everything the planner reads and writes.
Its sub-folders, with their sizes on 2026-11-02:

| folder | holds | size | needed for |
|---|---|---|---|
| `cells/` | trimmed per-bar scenes, one folder per export (the cache key) | 316 MB | building roadmaps, library experiments, tests |
| `library/` | goal and start libraries `cdfm_v1` and `smoke` | 825 MB | library experiments (`--library`), tests |
| `prm/` | every roadmap run, carry searches, logs, RobArch plans, and `dashboards.json` | 557 MB | roadmap dashboards, online planning on an existing roadmap |
| `assets/` | the robot as glTF per export, read by both dashboards | 6 MB | dashboards (3D view) |
| `runs/` + `runs.json` + `compare/` | the BiRRT benchmark batches (`night1`, ...) | 24 GB | the benchmark dashboard only |

If Insync (or Google Drive for desktop) syncs the shared folder on the Ubuntu
machine, nothing needs copying: the data root is then

```
/home/<user>/Insync/yijiang94817@gmail.com/Google Drive - Shared with me/2025-03 Husky Assembly/data_experiment/cdfm_planner_benchmark
```

Otherwise copy it. To run the roadmap planner and see its dashboards, the
first four rows are enough (about 1.7 GB); add `runs/` only for the BiRRT
benchmark dashboard. For example, from the Windows machine with rsync over
ssh, or with a USB drive:

```bash
rsync -av --exclude runs "<windows data root>/" ubuntu:~/husky_data/cdfm_planner_benchmark/
```

The exported process folders (`data_design_study/<problem>/`, about 90 MB
each: `process.json`, `actions/`, `robots/` with the URDF copies, `meshes/`)
are needed for anything that touches a real scene: `build-cells`, the
roadmap precompute, online planning and `prm_check_solved.py`. Their
`runs/`, `solutions/` and `source/` sub-folders are not read.
`261008_2ModulesDemo_purged_process` is the one the commands below use.

### 4b. Point the code at it

Three variables, set once in `~/.bashrc`:

```bash
# the benchmark data root (cells/, library/, prm/, assets/, runs/)
export HUSKY_BENCH_ROOT="$HOME/Insync/yijiang94817@gmail.com/Google Drive - Shared with me/2025-03 Husky Assembly/data_experiment/cdfm_planner_benchmark"
# the folder holding the exported process folders (the parent of 261008_2ModulesDemo_purged_process/)
export HUSKY_DESIGN_STUDY_DIR="$HOME/Insync/yijiang94817@gmail.com/Google Drive - Shared with me/2025-03 Husky Assembly/data_design_study"
# the standalone M1 runner's data root (dual_arm_task_space_rrt/run.py): the same folder
export HUSKY_ASSEMBLY_DATA_ROOT="$HUSKY_DESIGN_STUDY_DIR"
```

then `source ~/.bashrc`. The paths contain spaces; keep the quotes, and
quote `"$HUSKY_BENCH_ROOT"` wherever it appears on a command line. A fourth
variable, `HUSKY_EXTERNAL_DIR`, is only needed when the sibling source
checkouts (`compas_fab`, `scaffolding_assembly_core`) are not pip-installed
and do not sit next to this repo: it names the folder holding them.

Every script errors with a clear message when `HUSKY_BENCH_ROOT` is unset
rather than guessing a folder. `--root` overrides it per command.

> **Windows equivalent.** `setx HUSKY_BENCH_ROOT "C:\Users\...\cdfm_planner_benchmark"`
> (new terminals only), or `$env:HUSKY_BENCH_ROOT = "..."` for the session.

### 4c. The trimmed-scene cache and its key

`cells/<key>/<bar>/` is written by `benchmark_cdfm.py build-cells` from a
process folder. The key is the folder's name plus a short hash over the
**content hashes** of `process.json` and of its action files (the core's
`content_hash`, which leaves the writer and producer stamps out), so
re-exporting an unchanged design keeps the key, any change to a bar, a grasp
or a robot file changes it, and a synced or copied data root keeps it too.
Every roadmap run and library records the key it was built from. On a
machine where a process folder's cache does not exist yet, run `build-cells`
once for it (seconds per bar):

```bash
.venv/bin/python scripts/benchmark_cdfm.py build-cells \
    "$HUSKY_DESIGN_STUDY_DIR/261008_2ModulesDemo_purged_process" --bars B21
```

It prints the key. When more than one cache exists under `cells/`, the
library commands ask for `--source-key`; the roadmap commands take the key
from the library's `index.json` or compute it from `--problem-dir`, so they
do not. The caches and roadmap runs made from the older RobArch export
(`cells/260920_RobArch_demo_revamp-*`, `prm/260920_RobArch_demo_revamp__*`)
are plain compas_fab files and keep working; the export itself can no longer
be read, since the old format was dropped on 2026-11-02.

---

## 5. Smoke test

The roadmap's tests come in three groups: pure functions (no data), the
smoke scene (needs `library/smoke` and its cache in the data root), and the
process-folder reader (needs `HUSKY_DESIGN_STUDY_DIR/261008_2ModulesDemo_purged_process`
and the core; it skips when either is missing).

```bash
.venv/bin/python -m pytest benchmark/tests/test_holding_prm.py benchmark/tests/test_process_export.py -q
```

Expect every test to pass or skip, none to fail. The second group opens one
PyBullet scene in-process and takes about a minute.

A full end-to-end smoke run (a tiny roadmap, one query, its dashboard page),
writing under `$HUSKY_BENCH_ROOT/prm/smoke_prm/`:

```bash
.venv/bin/python scripts/prm_pilot.py build --name smoke_prm --library smoke --bar B3 --goals 0 --nodes 200 --workers 2
.venv/bin/python scripts/prm_pilot.py query --name smoke_prm --goal 0 --starts free
.venv/bin/python scripts/prm_pilot.py export --name smoke_prm
```

---

## 6. The dashboards

Both dashboards are plain web files that must be **served over HTTP** (a
browser refuses JavaScript modules and `fetch` on `file://` addresses).
Python's built-in server is enough. Serve the **data root**, not a dashboard
folder: the robot geometry in `assets/` is shared by every run and sits
outside any one run's folder.

```bash
.venv/bin/python -m http.server --directory "$HUSKY_BENCH_ROOT" 8000
```

Leave it running and open, in a browser on the same machine:

| page | address |
|---|---|
| roadmap runs (PRM) | `http://localhost:8000/prm/<run>/dashboard/` e.g. `http://localhost:8000/prm/261008_2ModulesDemo_purged_process__B21/dashboard/` |
| BiRRT benchmark batches | `http://localhost:8000/runs/<batch>/dashboard/` e.g. `http://localhost:8000/runs/night1/dashboard/` |

Both pages have a run dropdown under the title (fed by `prm/dashboards.json`
and `runs.json`), so one address reaches every exported run. To view from
another computer on the network, bind all interfaces
(`--bind 0.0.0.0`) and open `http://<ubuntu-host>:8000/...`.

What each page shows is explained in `benchmark/DASHBOARD.md` (benchmark) and
in the report `docs/holding_prm_report.tex` (roadmap page).

### Re-exporting a page

A run's dashboard folder is a *copy* of `benchmark/dashboard/` made when the
run was exported. After pulling a newer dashboard, or after new queries,
refinements or online plans were added to a run, re-export it and reload:

```bash
.venv/bin/python scripts/prm_pilot.py export --name <run>                  # roadmap run
.venv/bin/python scripts/benchmark_cdfm.py export-dashboard --name <batch>  # benchmark batch
```

The copied files carry a content hash in their addresses, so a reload always
picks up the new version; no cache clearing.

### If something looks wrong

| what you see | likely cause |
|---|---|
| header only, blank page | opened from disk instead of through the server, or a script failed: press F12 and read the Console |
| charts fine, 3D view empty | served a dashboard folder instead of the data root; or `assets/<key>/scene.glb` is missing (re-export the run, it writes it) |
| 404 on `prm/<run>/dashboard/` | the run was never exported on this data root: `prm_pilot.py export --name <run>` |
| the dropdown lacks a run | same: export it; the dropdown reads `prm/dashboards.json`, which export updates |

---

## 7. Running the roadmap planner

All commands run from the repository root with `HUSKY_BENCH_ROOT` set. Each
script's `--help` lists every flag; the module docstrings of
`scripts/prm_pilot.py`, `prm_carry_search.py`, `prm_bridge.py` and
`prm_check_solved.py` explain the subcommands. Results go under
`$HUSKY_BENCH_ROOT/prm/<run>/` and never into `runs/` or a problem folder.

### 7a. Precompute a bar's roadmap from a process folder (the production path)

One roadmap per bar, built in a scene with only the robot, its tools and the
carried bar with its halves; then every edge within a score tracked exactly
overnight, so online planning only has to check the real scene.

```bash
P="$HUSKY_DESIGN_STUDY_DIR/261008_2ModulesDemo_purged_process"
.venv/bin/python scripts/benchmark_cdfm.py build-cells "$P" --bars B21        # 0. trim the carry scene (seconds)
export WORKERS=8                                                             # worker processes; use the core count
scripts/prm_runs/process_precompute.sh 261008_2ModulesDemo_purged_process B21   # 1. build, track, export the page
```

`TRACK_MM=0` skips the tracking for a quick look. The recipe runs, per bar:

```bash
.venv/bin/python scripts/prm_pilot.py build --name 261008_2ModulesDemo_purged_process__B21 \
    --problem-dir "$P" --bar B21 --nodes 6000 --starts-file carry/carry_v1/starts.json --workers 8
.venv/bin/python scripts/prm_pilot.py track --name 261008_2ModulesDemo_purged_process__B21 --max-score-mm 15 --workers 8
.venv/bin/python scripts/prm_pilot.py export --name 261008_2ModulesDemo_purged_process__B21
```

`--starts-file` is looked up as given, then under `$HUSKY_BENCH_ROOT/prm/`
(the carry poses from the carry search live there, synced with the data).
The run name `<problem>__<bar>` is what the online planner looks for by
default. On the design machine a 6000-node build with 8 workers took about
10 minutes per bar, and tracking every edge within 15 mm took between half
an hour and two hours per bar (the report's runs); a 600-node build of B21
with 4 workers took 104 s. Windows: `scripts\prm_runs\process_precompute.ps1 -Problem ... -Bars B21`.

### 7b. Plan a bar carry in its real scene, on that roadmap

`plan_process_carry.py` plans the carry movement of one bar in the scene the
core rebuilds for its start (the bars built so far with their halves, the
ground, obstacles, a support robot where the process places one, the acting
robot at its planned base), on the precomputed roadmap or with the BiRRT:

```bash
.venv/bin/python -u scripts/plan_process_carry.py "$P" --bar B21 --cdfm-backend prm --prm-workers 8 --save-tag prm
.venv/bin/python -u scripts/plan_process_carry.py "$P" --bar B21 --cdfm-backend birrt --max-time 300 --save-tag birrt1
```

Flags that matter here: `--prm-run` (another roadmap than `<problem>__<bar>`),
`--prm-tolerances-mm`, `--prm-tube-rounds`, `--prm-no-pretracked` (ignore
the overnight tracking), `--base-shift=x,y,deg` (park the base off its
planned pose; write it with the `=`, since a value starting with a minus
sign would otherwise be read as a flag), `--goal joints` (end exactly at the
authored posture, where the insert starts, instead of in any posture at the
authored flange poses), `--write-solution` (also write
`<process>/solutions/<bar>_J.json`, the file Rhino's `RSLoadSolvedBarAction`
reads). Each run writes, into `$HUSKY_BENCH_ROOT/prm/plans/<problem>/` (the
default `--save-dir`; nothing goes into the process folder):

    <bar>_J.solved_motion.<tag>.json             the path, in the core's solution shape
    <bar>_J.solved_motion.<tag>.plan_info.json   how the plan went (also when it failed)
    <bar>_J_planned_motion.<tag>.png             the joint curves and the holding error

Then check the saved path on its own, with the production repair and the
full collision check (writes one JSON line per file to `--out`):

```bash
.venv/bin/python -u scripts/prm_check_solved.py "$P" --bar B21 --tags prm birrt1 \
    --out "$HUSKY_BENCH_ROOT/prm/plans/261008_2ModulesDemo_purged_process/checks.jsonl"
```

`scripts/prm_runs/process_plans.sh 261008_2ModulesDemo_purged_process B21`
runs the whole T2/T5/T3 set of the report for one bar (roadmap planner,
three BiRRT runs, one-worker roadmap run, four shifted bases, each followed
by its check); `... B21 PRM` runs only the roadmap plan and its check;
Windows: `process_plans.ps1 -Problem ... -Bar B21`. The online plan is also
saved as a query of the roadmap, so the run's page shows it in its scene
(`prm_pilot.py export` again after planning).

How the scene is built, and what that means for the measurement:

- `process.scene_at(carry)` gives the world at the start of the carry, and
  the core's `CompasFabMirror` turns it into the compas_fab cell and state
  the planner API takes: the same path Rhino's commands and the design
  workflow's `plan_process.py` use, so the planner sees what Rhino shows.
  The allowed contacts (a tool with the part it is on, the halves with their
  bar, mated halves with each other) are derived by the core.
- All mounted halves ride with the bar (B21 carries seven: two male, one
  female, four mocap halves). The roadmap's precompute scene has them too,
  so its verdicts and the online plan's agree (`robot_side_signature`
  checks that).
- Both grasps are in the file (the two holders of the bar).
- Planning cells carry no visual shapes (as in production); the robot asset
  of the dashboard and the scene files are drawn from the collision shapes.

### 7c. The library experiments (build, query, refine, bridge, carry)

These are the roadmap pilot's research commands on the benchmark library
(`cdfm_v1`): sample a roadmap for a bar, connect a goal and a set of starts,
search it at several tolerances, track the routes exactly, and compare with
the BiRRT. The report's experiments E1 to E6 and the carry search were run
with the PowerShell queues in `scripts/prm_runs/*.ps1`; each is a list of
command lines like

```bash
.venv/bin/python scripts/prm_pilot.py build --name B18_g2_n4000 --library cdfm_v1 --bar B18 --goals 2 --anchor-starts 40 --nodes 4000 --workers 6
.venv/bin/python scripts/prm_pilot.py query --name B18_g2_n4000 --goal 2 --starts all --tag all --workers 6
.venv/bin/python scripts/prm_pilot.py refine --name B18_g2_n4000 --query all --workers 6
.venv/bin/python scripts/prm_bridge.py validate --name B18_g2_n4000 --refine all_k2.5_p10mm_mid
.venv/bin/python scripts/prm_pilot.py export --name B18_g2_n4000
.venv/bin/python scripts/prm_pilot.py info --name B18_g2_n4000
```

To run one of those queues on Ubuntu, hand its command lines to the portable
runner (section 7d); the `.ps1` files hold the exact flags of each
experiment.

### 7d. Long jobs: the logged runner

`scripts/prm_runs/run_logged.py` runs a list of command lines in order and
appends all their output to `$HUSKY_BENCH_ROOT/prm/logs/<name>.log`, like
`run_logged.ps1` did on Windows, but with no path written into it and with
the same Python that runs it. The first word of each command picks the
script (`build`/`query`/`refine`/`track`/`export`/`info` for `prm_pilot.py`;
`bridge`, `carry`, `plan`, `check`, `bench` for the others).

```bash
# see what would run, without running it
.venv/bin/python scripts/prm_runs/run_logged.py --log test.log --dry-run \
    "build --name B18_g2_n4000 --library cdfm_v1 --bar B18 --goals 2 --nodes 4000 --workers 6" \
    "export --name B18_g2_n4000"

# run it detached, so it survives closing the terminal (or use tmux / screen)
nohup .venv/bin/python scripts/prm_runs/run_logged.py --log e1_20261102.log \
    "build ..." "query ..." "export ..." > /dev/null 2>&1 &

# follow the log
tail -f "$HUSKY_BENCH_ROOT/prm/logs/e1_20261102.log"
```

A failing command does not stop the queue (`--stop-on-error` does); the
runner's exit code is 1 when any command failed, and the log says which.

Notes on resources: each worker process opens its own PyBullet scene and
loads the IK solvers, so `--workers` beyond the physical core count only
adds memory; the workers set `OMP_NUM_THREADS=1` themselves. The scripts use
the `spawn` start method on every platform, so behaviour matches Windows.

---

## 8. What was changed for portability (and what was not)

Changes on `yh/holding-prm`:

- **The old export is gone** (2026-11-02). `RobotCell.json` + `BarActions/`
  (compas_fab objects serialized with `rs_data_structure`) are no longer
  read anywhere: `headless_bar_action_planner.py`, `replay_bar_action_plan.py`,
  the RobArch recipes and the collision-split test went with them, and
  `rs_data_structure` is no longer a dependency. Everything reads the
  schema 2.1 process folder through the shared core
  (`benchmark/process_export.py`); `scripts/plan_process_carry.py` is the
  online planner, `benchmark/plot_motion.py` its figure.
- **Cache key of trimmed scenes** (`benchmark/build_cells.py`,
  `source_key_for`): the content hashes of `process.json` and the action
  files, so a synced or copied data root keeps the same key.
- **Legacy design-study path** (`dual_arm_task_space_rrt/run.py`): the
  hard-coded `/home/yijiangh/Insync/...` default now reads
  `HUSKY_ASSEMBLY_DATA_ROOT` first. (That standalone runner still reads the
  old export and is dead code since the drop.)
- **Dependencies** (`setup.py`): `termcolor`, `scipy`, `networkx` and
  `pyinstrument` were imported but not declared; they are now in
  `install_requires`. The `[process]` extra adds the core and its readers.
  `constraints-py311.txt` records the known-good versions.
- **Sibling checkouts** (`benchmark/sibling_packages.py`): `HUSKY_EXTERNAL_DIR`
  names the folder holding them when the repo is not vendored.
- **Long Windows paths** (`benchmark/paths.py`): the dashboard copy works
  under the Drive folder's long path.
- **Run recipes**: `scripts/prm_runs/run_logged.py` (portable runner, the
  `plan` command word), `process_precompute.{sh,ps1}` and
  `process_plans.{sh,ps1}`, for any problem name. The library queues
  (`queue*.ps1`, `e1_radius160.ps1`) are unchanged.

Already portable before, worth knowing:

- The dashboards are static files served by `http.server`; the robot's glTF
  is written by Python (`benchmark/glb.py`), no Node or external tool.
- Worker pools use `multiprocessing`'s `spawn` context explicitly.
- All data locations come from `HUSKY_BENCH_ROOT`, `HUSKY_ASSEMBLY_DATA_ROOT`,
  `--root`, `--problem-dir` or positional arguments; the ssik solver modules
  resolve to `asset/ssik` inside the checkout.

Still Windows-only or machine-specific:

- The library experiment queues `scripts/prm_runs/queue*.ps1` and
  `e1_radius160.ps1` (their command lines are the record of each
  experiment; run them through `run_logged.py`).
- `--gui` (PyBullet window) needs a display; the planned-motion figure is
  written without one.
- Linux file names are case-sensitive; the data's names (`B21_J.json`,
  `process.json`) must be typed exactly.
- `docs/figures/make_prm_figures.py` regenerates the report's figures and
  tables from the roadmap runs; it needs the runs named in it to be present
  on the data root (they are, if `prm/` was synced in full).

---

## 9. Keeping both machines in step

Code moves through git (push `yh/holding-prm` from whichever machine
changed it, pull on the other); data moves through the shared Drive folder.
Every run's `config.json` records the commit of this repo, whether the tree
was dirty, the package versions and the machine it ran on
(`benchmark/provenance.py`), so a result can always be traced back to the
code that produced it, on either machine.
