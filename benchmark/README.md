# CDFM transfer planner benchmark

Measures `plan_constrained_dual_arm` — the M1 transfer, where the robot carries
a bar held in both tools from one place to another. It answers: how often does
the planner find a motion, how long does it take, and where does that time go.

It measures the production planner rather than reimplementing it. Reachability,
the start sweep, the straight-line check and the planning itself are all calls
into `husky_assembly_tamp.motion_planner`, and every collision decision goes
through the same compas_fab check the planner uses. A number here describes the
planner we actually ship.

## Running it

Everything goes through one script. Set the data root once:

```bash
export HUSKY_BENCH_ROOT="…/data_experiment/cdfm_planner_benchmark"
PYTHON=external/husky_assembly_tamp/.venv/Scripts/python.exe
```

Then, in order:

```bash
# 1. Trim the exported design study into one small scene per bar. Once per
#    export; takes a couple of minutes.
$PYTHON scripts/benchmark_cdfm.py build-cells "…/data_design_study/260921_motion_sample"

# 2. Find out what a library will cost before committing a night to it.
$PYTHON scripts/benchmark_cdfm.py measure --sweep-steps 0.3,0.2

# 3. Sample goals and survey the starts they can be reached from.
$PYTHON scripts/benchmark_cdfm.py build-library --goals 50 --sweep-step 0.2

# 4. See how big a planning batch would be.
$PYTHON scripts/benchmark_cdfm.py plan --seeds 10 --max-starts-per-goal 3

# 5. Plan everything. Resumable: stop it and run it again.
$PYTHON scripts/benchmark_cdfm.py run --name night1 --seeds 10 \
    --max-starts-per-goal 3 --workers 12

# 6. Aggregate, and write the files the dashboard reads.
$PYTHON scripts/benchmark_cdfm.py summarize --name night1
$PYTHON scripts/benchmark_cdfm.py export-dashboard --name night1
```

The dashboard is a static page. Serve the data root, not the dashboard folder,
so the shared robot geometry resolves:

```bash
$PYTHON -m http.server --directory "$HUSKY_BENCH_ROOT" 8000
# then open http://localhost:8000/runs/night1/dashboard/
```

It has to be served rather than opened from disk: browsers refuse `fetch` and
JavaScript modules on `file://` URLs.

## What a run is

Each run plans one start-to-goal transfer with one random seed, twice:

| pass | step | budget | what it is for |
|---|---|---|---|
| coarse | 0.1 m / 0.1 rad | 30 s | a quick look at whether any route exists |
| fine | 0.002 m / 0.002 rad | 300 s | the resolution the robot executes at |

A run counts as solved only when both passes find a motion. The coarse pass
earns its keep on the failures: hopeless problems are rejected in tens of
seconds rather than five minutes, and on a solvable one it takes about 4 s
against the fine pass's 31 s.

Whether that is a *safe* filter was checked rather than assumed — a coarse
search takes bigger steps, and the planner rejects any step that moves the arms
more than about ten degrees, so it could in principle fail for its own reasons.
Across the sampled problems no case solved by the fine pass was rejected by the
coarse one. Worth re-checking if the continuity threshold ever changes.

## What the numbers cost

Measured on bar B3, one core, on the design machine:

| operation | median |
|---|---|
| collision check (compas_fab, trimmed scene) | 12.9 ms |
| set robot cell state | 1.7 ms |
| analytical IK, one arm, reachable | 1.8 ms |
| analytical IK, one arm, unreachable | 0.8 ms |
| forward-kinematics gate | 0.14 ms |

Collision checking dominates everything, and it is what makes the start survey
expensive: a carry pose the robot can hold costs about one check, but one that
collides costs up to sixty-four, because every way of grasping it has to be
ruled out before the pose can be rejected.

That makes the survey cost about 240 ms per candidate pose, near enough
independent of the grid:

| grid step | offsets | candidate poses | per goal | 800 goals, 12 workers |
|---|---|---|---|---|
| 0.30 m | 27 | 972 | 3.9 min | 4.3 h |
| 0.20 m | 64 | 2 304 | 9.2 min | 10.3 h |
| 0.15 m | 125 | 4 500 | 18 min | 20 h |
| 0.10 m | 343 | 12 348 | 49 min | 55 h |

0.10 m is what the planner itself sweeps. Surveying at that resolution for a
full library is two and a half days of compute, so the grid step is a flag and
`measure` exists to make the trade explicit.

## Picking which starts to plan from

A survey turns up hundreds of workable carry poses per goal — 380 for one
sampled goal of B3. Planning from all of them, ten seeds each, would be
millions of runs, so a batch takes a sample.

Which sample matters more than it might seem. Being able to *hold* the bar at a
carry pose says nothing about being able to *get there* from the goal, and most
poses turn out to be far harder to reach than to occupy. Of those 380: seven
were already joined to the goal by a straight motion, 81 by a straight motion
that passes through something, and 292 could not be reached in a straight line
at all. A sample taken in survey order would be hundreds of variations on one
pose; taken at random it would be almost all of the hardest kind.

So every start is labelled by what the straight-line probe found —

| label | meaning |
|---|---|
| `free` | a straight motion already solves it; the search only has to notice |
| `blocked` | a straight motion reaches the start but passes through something |
| `broken` | a straight motion cannot even reach the start |

— and `--max-starts-per-goal` samples across both these classes and the three
carry anchors, always including the start the planner itself would have picked.
The library keeps every feasible start regardless, so the choice can be redone
without re-running the survey.

## The dashboard

Four views, all state in the URL:

```
#/                                overview: every bar
#/bar/B12                         one bar's goals, drawn around the robot
#/bar/B12/goal/7                  one goal's start survey
#/bar/B12/goal/7/start/42         one start's seeds and motions
```

The robot is drawn from `assets/<source key>/scene.glb`, whose node tree
mirrors the robot's joints, plus `kinematics.json` saying which twelve to turn.
Posing it in the browser rather than shipping a matrix per link per frame is
what keeps a planned motion at forty kilobytes instead of two megabytes.

That the browser's forward kinematics agrees with the simulator is checked
rather than assumed — a robot drawn in the wrong place looks perfectly
plausible:

```bash
$PYTHON benchmark/tools/export_fk_fixtures.py   # record poses from PyBullet
node benchmark/tools/test_fk.mjs                # replay them through the page's own module
```

They currently agree to about 1e-7 m.

Charts are hand-drawn SVG rather than a charting library: six chart types are
less code than vendoring a megabyte, and it works with no network.

## Profiling

Every planning pass is sampled with pyinstrument, producing two things: a flame
graph per run (`profiles/*.speedscope.json`, openable at speedscope.app or any
speedscope build) for reading one run in detail, and a component breakdown —
collision, IK, smoothing and so on — that averages over thousands of runs and
is charted on the overview and start pages. `--no-profile` turns it off.

## Reproducing a result

Each batch writes a `config.json` with the library it planned from, the exact
planner settings, and the commit and cleanliness of every repository involved.
A result from a dirty working tree says so, because it cannot be reproduced
from its commit alone.
