# husky_assembly_tamp

Motion planning for the Husky bar-assembly project, without Rhino running. It
reads the process folder that Rhino's `RSExportProcess` writes (schema 2.1:
`process.json`, one `actions/<bar>_<code>.json` per action, the robots as URDF
files, the meshes as OBJ files), rebuilds the scene of any movement through the
shared core library `scaffolding_assembly_core`, and plans in it with the
task-space BiRRT or with a roadmap precomputed per bar.

## Package layout

```text
husky_assembly_tamp/
  setup.py                       install spec (pins the planning stack)
  constraints-py311.txt          the versions the experiments ran with
  husky_assembly_tamp/           the importable package
    keyframe/                    base search + dual-arm IK (shared with Rhino's commands)
    motion_planner/              the BiRRT (dual_arm_task_space_rrt/) and the roadmap planner (holding_prm/)
    benchmark/                   per-bar scenes from a process folder, the BiRRT benchmark, the dashboards' data
  benchmark/dashboard/           the two web pages (BiRRT benchmark, roadmap runs)
  scripts/
    benchmark_cdfm.py            build-cells (trim a process folder into per-bar scenes), the BiRRT benchmark
    prm_pilot.py                 build / query / refine / track / export a roadmap
    plan_process_carry.py        plan one bar's carry in its real scene, on the roadmap or with the BiRRT
    prm_check_solved.py          check a saved carry on its own (repair, collisions, holding error)
    prm_carry_search.py, prm_bridge.py   the roadmap's research tools (benchmark library)
    prm_runs/                    run recipes and the logged job runner
  docs/                          the reports and the setup manual
```

The step-by-step manual, for Ubuntu and Windows, is
[`docs/prm_ubuntu_setup.md`](docs/prm_ubuntu_setup.md): environment, data,
the dashboards, and every stage of the carry experiment on a process folder.

## Environment

One Python 3.11 with the pinned stack, made with [`uv`](https://docs.astral.sh/uv/)
from the repo root:

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[ssik,process]" -c constraints-py311.txt --python .venv/bin/python   # Windows: .venv\Scripts\python.exe
```

`[ssik]` is the analytical IK backend (the committed solver modules live in
`asset/ssik`); `[process]` installs the shared core library and its URDF and
mesh readers. When this repo is vendored in the design workflow the core is
found as the sibling checkout `external/scaffolding_assembly_core` instead; a
clone anywhere else is found through `HUSKY_EXTERNAL_DIR` (the folder holding
it). `.venv/` is git-ignored: each machine builds its own.

## Data

Two folders, named once per machine:

- `HUSKY_BENCH_ROOT`: the benchmark data root (`cells/`, `library/`, `prm/`,
  `assets/`, `runs/`), where every stage reads and writes.
- `HUSKY_DESIGN_STUDY_DIR`: the folder holding the exported process folders,
  one per design study.

Both are machine-specific synced folders and are never written into the code.

## The carry experiment in short

```bash
P="$HUSKY_DESIGN_STUDY_DIR/261008_2ModulesDemo_purged_process"
.venv/bin/python scripts/benchmark_cdfm.py build-cells "$P" --bars B21                   # trim the carry scene
scripts/prm_runs/process_precompute.sh 261008_2ModulesDemo_purged_process B21             # roadmap: build, track, page
.venv/bin/python -u scripts/plan_process_carry.py "$P" --bar B21 --cdfm-backend prm --prm-workers 8 --save-tag prm
.venv/bin/python -u scripts/plan_process_carry.py "$P" --bar B21 --cdfm-backend birrt --max-time 300 --save-tag birrt1
.venv/bin/python -u scripts/prm_check_solved.py "$P" --bar B21 --tags prm birrt1          # check the saved paths
.venv/bin/python scripts/prm_pilot.py export --name 261008_2ModulesDemo_purged_process__B21   # the roadmap page
```

`scripts/prm_runs/process_plans.sh <problem> <bar>` runs the planning and
checking set of the report (T2, T5, T3) in one logged job; the `.ps1` twins do
the same on Windows. The dashboards are served from the data root with
`python -m http.server --directory "$HUSKY_BENCH_ROOT" 8000` (see the manual,
section 6).
