# Reading the benchmark dashboard

A guide to opening the dashboard for a benchmark run and making sense of what it
shows. For how the benchmark itself is built and run, see `README.md`; for what
the first run found, see `analysis_night1.md`.

## Opening it

Everything lives in the benchmark data folder on the shared drive:

```
C:\Users\yijiangh\Insync\yijiang94817@gmail.com\Google Drive - Shared with me\2025-03 Husky Assembly\data_experiment\cdfm_planner_benchmark
```

The dashboard is a set of plain web files, but a browser will only run them when
they come from a web server, not when opened straight from disk. Python's
built-in server is enough. In PowerShell, from `external\husky_assembly_tamp`:

```powershell
$env:HUSKY_BENCH_ROOT = "C:\Users\yijiangh\Insync\yijiang94817@gmail.com\Google Drive - Shared with me\2025-03 Husky Assembly\data_experiment\cdfm_planner_benchmark"
.venv\Scripts\python.exe -m http.server --directory $env:HUSKY_BENCH_ROOT 8000
```

Then open **http://localhost:8000/runs/night1/dashboard/**. Each run has its own
address: replace `night1` with the run's name, or pick the run from the dropdown
under the title. The dropdown lists every run in `runs.json` with its size and
success rate. Switching keeps you on the same bar, goal or start when the other
run has it, and otherwise opens that run's overview. On deeper pages, the
"overview" link next to the dropdown goes back to the run's front page. Leave the window running while
you browse, and press Ctrl+C in it to stop.

Serve the **data folder**, not the dashboard folder inside it. The robot model is
shared by all runs and sits outside any one run's folder; serve the wrong folder
and the charts load but the 3D view stays empty.

### After changing the data

If a run is re-summarised, rebuild its dashboard files, then reload the page:

```powershell
.venv\Scripts\python.exe scripts\benchmark_cdfm.py export-dashboard --name night1 --library cdfm_v1
```

A normal reload always picks up the newest version; no need to clear the cache.

### If something looks wrong

| what you see | likely cause |
|---|---|
| header only, blank below | opened from disk rather than through the server, or a script failed: press F12 and look at the Console |
| charts fine, 3D view empty | served the dashboard folder instead of the data folder |
| "Search trees: not saved for this start" | expected for most starts; see *Search trees* below |

## Four ideas that make the rest readable

**A run** is one attempt to plan one transfer: carry the bar from one start pose
to one goal pose, with one random seed. Each run is planned twice. The **coarse
pass** (10 cm steps, 30 s limit) is a quick look at whether any route exists; the
**fine pass** (2 mm steps, 300 s limit) is the motion the robot would execute,
and runs only if the coarse one succeeded. A run counts as **solved** only when
both succeed. Each goal gets 3 start poses and 10 seeds, so 30 runs.

**Two clocks, never mixed.** *Motion planning time* is time inside the planner
finding a motion; it is what the robot would wait for. *Setup time* is the work
done once, beforehand, to pose the question: goal IK to find a goal the arms can
reach, and the start survey that finds the carry poses a transfer could begin
from. Every chart says which clock it shows.

**Workable versus reachable.** A *workable start* is a carry pose where both arms
can hold the bar without colliding. That says nothing about whether the planner
can get from there to the goal. Each start is therefore also tested with a
straight motion to the goal, and labelled by the result:

| label | meaning |
|---|---|
| clear | a straight motion already reaches it with nothing in the way: the easiest problems |
| blocked | a straight motion reaches it but passes through something |
| cannot reach | a straight motion cannot even get there |

**Seeds.** The search is random, so the same problem is planned with 10 seeds.
One seed failing means little; all ten failing means the problem is genuinely
hard for this planner.

## The pages

Click through from the top; the path at the top of the page shows where you are
and each part of it is a link back. Every address can be bookmarked or shared.

### Overview — every bar

- **Tiles**: total runs, how many were solved, the median motion-planning time of
  solved runs, and setup time per goal.
- **Solved, by bar**: the share of runs solved, per bar, ordered by bar length.
  The whisker is a 95% confidence interval; the hollow circle is how often the
  coarse pass alone succeeded. A big gap between circle and bar means the fine
  pass is where runs fail. Click a bar to open it.
- **Motion planning time, by bar**: solved runs only, on a log scale. The box
  covers the middle 80% of runs, the line inside is the median.
- **Setup time, by bar**: minutes per goal spent before any planning, split into
  goal IK, the start survey, the straight-line tests, and the planner's own start
  search (run for comparison). Measured with many bars in parallel, so roughly
  1.5 to 2 times what one processor core would take.
- **Trying again**: the share of problems solved by at least one of the first *k*
  seeds. A curve still rising at 10 means more restarts would keep helping.
- **Why extensions stopped**: each time the search tries to grow, it stops for a
  reason. *continuity* means the arm solution jumped between IK branches;
  *collision* means something was in the way; *ik failure* means the arms could
  not reach. Names starting *connect* are the two trees trying to join.
- **Where motion planning time went**: what the planner spent its time on.
  "IK while growing the tree" is the arm IK at every search step, not goal IK.
- **Bars**: the same numbers as a table. Click a row to open the bar.

### Bar — one bar's 50 goals

The 3D view shows the robot and all 50 goal poses of the bar, coloured by how
many of their 30 runs were solved; the legend under the robot gives the bands and
how many goals fall in each. Drag to rotate, scroll to zoom, click a goal to open
it.

The **Goals** table has a key underneath explaining every column. The ones worth
knowing: *goal IK solutions* is how many different arm configurations hold the
bar at the goal; *workable starts* is how many carry poses the survey found;
*straight-line starts* is how many of those a straight motion reaches cleanly.

### Goal — the start survey for one goal

The switch under the robot chooses what the 3D view shows:

- **Workable only**: the carry poses the arms can hold, coloured clear, blocked,
  or cannot reach.
- **Include failed**: every pose the survey tried, including rejected ones, drawn
  faintly and coloured by why they were rejected. Click a rejected pose to see
  why.

Below:

- **The start survey**: how many poses were tried and why the rest failed, plus
  how long the survey took for this goal.
- **By carry anchor**: the same, split by the three ways of carrying the bar
  (across the front, upright, over the back).
- **The offset grid**: the survey tries small offsets around each carry anchor.
  One height at a time (slider), one grid per anchor; each cell is coloured by
  its outcome.
- **Starts planned from**: the starts this goal was actually planned from,
  with their straight-line label and how often they were solved. Click one to
  open it.
- **Solved, by straight-line label and by carry anchor** (when more than one
  start was planned): the share of planned starts with at least one solved
  seed, grouped by what the straight-line test said about them and by the
  carry anchor they came from. With a whole goal's starts planned this is the
  planner's ability to connect two configurations, read against how the start
  looked beforehand. The **Planning outcome** switch under the robot colours
  the same starts in 3D: green solved, orange coarse pass only, red failed.
- **What the planner itself would have done**: whether the planner's own start
  search found a start, and whether a straight motion would have solved it
  outright.

### Start — one start–goal pair, all its seeds

The 3D view shows the robot at the start, with **faint bars** marking where the
transfer begins (green) and ends (blue).

- **Search trees** (under the robot, when saved): step through the seeds with the
  arrows and switch between coarse and fine pass. The green tree grew from the
  start, the blue one from the goal; each line is the path of the bar's midpoint
  and each node is a small faint bar. **Click a node** to put the robot at the
  joint configuration the search stored there; the box under the robot lists
  the twelve angles.
- **Failed attempts** (same controls): every extension the search threw away
  is drawn at the waypoint it rejected, with a thin line from the last accepted
  node, coloured by the reason — orange *continuity*, purple *IK failure*, red
  *collision*. The toggle buttons show or hide each kind (and count them); the
  *iteration* slider replays the search in time. Pick one either by clicking
  its bar, or with the **step through** row: choose a reason and press ◀ ▶ to
  walk its stops in the order the search made them; the chosen bar turns
  yellow. Either way, the box under the robot gives the numbers behind it:
  - *continuity*: the parent node's IK solution is drawn on the solid robot
    and the rejected node's on a see-through orange copy, so the jump can be
    seen, with a table of both configurations and each joint's jump, the
    joints over the 10° gate in red. Two buttons swap which is solid. One
    joint over the gate is a branch flip; most joints over it means the step
    itself was too big for the gate.
  - *IK failure*: which arm had no solution, and whether none exists at all
    (*unreachable*), or solutions exist but every one breaks a joint limit
    (*joint limit* — with the offending joints, their values and the allowed
    range, and a button to pose the arm at that out-of-range solution anyway).
  - *collision*: the two parts that touched, lit up red on the robot, and
    whether it was the whole robot or the bar alone that was checked.
- **What the failed attempts say**: the same, summed for the pass on show —
  which joint flips most and by how much, IK failures by arm and cause, and the
  pairs of parts that collide most.
- **Seeds**: every seed's outcome for both passes, iterations, and tree sizes.
- **Motion planning time per seed**: *c* is the coarse pass, *f* the fine pass.
- **Where each seed's motion planning time went**: as on the overview, per seed.
- **Flame graphs**: download one and open it at speedscope.app to see exactly
  which functions a run spent its time in. The file is read in your browser and
  not uploaded.
- **Motion**: for solved seeds, choose a motion and drag the slider under the
  robot to play it; the chart below plots all 12 joint angles along the motion
  (left arm blue, right arm orange) with a cursor at the current point.

### Search trees

The first batch saved only the size of each tree, not the trees. To see them for
a start, re-plan it with its trees saved — its page shows the exact command, for
example:

```powershell
.venv\Scripts\python.exe scripts\benchmark_cdfm.py replay --name night1 --bar B9 --goal 0 --start 0
```

All ten seeds are planned at once. A start whose seeds all fail takes about a
minute; one with solved seeds takes up to about 10 minutes, since a fine pass
can run for 5. Then reload the page.

A replay uses the same seeds and settings, but the search stops on a time limit,
so it can stop at a different point and occasionally end differently from the
original run. The page says when you are looking at a replay and shows both
outcomes.

Batches run since then save the trees and the failed attempts for every run
by default (`run --no-trees` turns that off), so this only applies to
`night1`'s original runs. A replayed start records everything the new pages
show; an older tree file still draws, but its nodes cannot pose the robot and
it has no attempts.

## Known gaps in the first run's data

- **What collided most** says "not available" for `night1`: that library recorded colliding
  links as unreadable memory addresses, a bug fixed since. A new library records
  their names.
- **Workable starts** should be the same for every goal of a bar, and are on only
  5 of 16 bars; on the rest the carry poses shift between goals. This looks like
  a problem in the planner's carry-pose code and is written up in
  `analysis_night1.md`.
- **Setup times** were measured under heavy parallel load and overstate what a
  single run would take.
