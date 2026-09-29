# What the first benchmark run says

Run `night1`: 24,000 planning runs over 16 bars, 800 goal poses, 3 start poses
per goal, 10 random seeds each. 42.9 hours on 12 workers, no crashes and no
errors. Library `cdfm_v1`, built from `260921_motion_sample`.

Nothing here has been acted on. These are observations and the changes they
suggest, for review before anything is touched.

## Two clocks, kept apart

Every time figure in this document is one of two different things, and they
must not be mixed:

- **Motion planning time** is time inside `plan_constrained_dual_arm`: growing
  the search trees, joining them, smoothing. It is what a robot would wait for
  on every transfer.
- **Setup time** is the work done beforehand to pose the benchmark's questions:
  goal IK to find a reachable goal pose, and surveying the carry poses a
  transfer could start from. It happened once, in the 27-hour library build,
  and is not part of any planning run.

The IK that shows up inside motion planning (Finding 2) is the arm IK solved at
every step while the search grows its tree. It is not goal IK.

## The headline

| | |
|---|---|
| Runs solved | 3,906 of 24,000 = **16.3%** (95% interval 15.8 to 16.7) |
| Solved within 10 seeds | **28.0%** of problems |
| Motion planning time, median of solved runs | **143 s** (fine pass); the coarse pass before it, median 21 s |
| Setup per goal, before any planning | about 27 min, almost all start survey (see "Setup cost") |
| Timed out | 368 = 1.5% |

Bar length barely matters. Success runs from 8% (B39, 1.75 m) to 21% (B3,
0.50 m, and B6, and B18) with no clean trend. Median motion-planning time on
solved runs ranges from 100 s (B46) to 227 s (B21), also with no trend in
length. Whatever limits this planner, it is not the size of the thing being
carried.

A correction to an earlier version of this note, which said the median time was
about 36 s for every bar: that figure was the median over *all* runs, and since
most runs stop at the coarse pass's 30-second limit, it only reported that
limit. The numbers above are for solved runs.

## Finding 1: the search is rejected by its own continuity gate, not by obstacles

Every attempt to grow the search tree ends for a reason, and they are lopsided:

| why an extension stopped | share |
|---|---|
| **continuity** (the arm solution jumped) | **75.4%** |
| ik_failure (a waypoint was unreachable) | 18.5% |
| collision (something was in the way) | 3.7% |
| everything else | 2.4% |

Three quarters of all rejections are the branch-flip gate: consecutive
waypoints solved to arm configurations more than about ten degrees apart, so
the step was thrown away. Actual collisions account for one rejection in
twenty-seven.

This is the single most important number in the run. The planner is not mostly
fighting the environment. It is mostly fighting its own inverse kinematics
jumping between solution branches.

## Finding 2: motion-planning time goes where the failures do not

Motion planning time only, split by what the planner was doing:

| component | mean seconds per pass | share |
|---|---|---|
| collision checking | 17.35 | **36.8%** |
| IK while growing the tree | 15.54 | 33.0% |
| joining the two trees | 6.11 | 13.0% |
| smoothing | 4.61 | 9.8% |
| everything else | 3.50 | 7.4% |

Collision checking is the largest cost and causes 3.7% of rejections. IK while
growing the tree is the second largest and causes, through the continuity gate
it feeds, 94% of them. (That IK is solved at each step of the search. No goal
IK happens inside a planning run: goal and start configurations are given.)

A single collision check costs 12.9 ms, measured, because each one clones the
whole cell state and re-synchronises the simulator. That is the number to
attack if the goal is raw speed.

## Finding 3: the search fails on problems that provably have a solution

Every start was probed beforehand by walking the bar in a straight line from the
goal, and labelled by what that found:

| the straight line... | starts | runs solved |
|---|---|---|
| reaches the start, nothing in the way (`free`) | 3.2% | **43.6%** |
| reaches it but passes through something (`blocked`) | 16.4% | 3.4% |
| cannot reach it at all (`broken`) | 80.4% | 1.1% |

The probe is strongly predictive, which is useful in itself. But look at the top
row: on problems where a straight-line motion demonstrably exists, the search
still fails **56% of the time**, and 30% of those start-goal pairs are not
solved by any of ten seeds.

The stop reasons on those problems are the same as everywhere else, 76.5%
continuity. So this is not a hard-search-space problem. The planner is failing
to find a solution it has effectively been handed, and failing for the same
branch-flip reason as everything else.

## Finding 4: the start is chosen without regard to whether it can be reached

Production's cold start sweep takes the first carry pose the arms can hold,
sweeping offsets outward from each anchor. Over 800 goals:

- it chose the **horizontal** anchor **100% of the time**, because that anchor
  is tried first and the very first candidate is usually holdable;
- by success rate, horizontal is the **worst** of the three: 12.0%, against
  21.4% for `back` and **29.6%** for `vertical`;
- its pick is `broken` (no straight line reaches it) in **81%** of cases, which
  is exactly the base rate — the choice carries no information about
  reachability;
- its picks succeed **2.7%** of the time, against **23.1%** for the other starts
  the benchmark planned from.

To be fair to the shipped planner, this cold sweep is its fallback. Its primary
method walks backward from the goal, and does better:

| production's primary start derivation | |
|---|---|
| found a start | 95.9% of goals |
| found a free straight path, skipping the search entirely | **34.9%** |
| fell back to a partial start | 33.2% |
| time spent | median **121 s** (its budget is 120 s) |

So roughly a third of goals are solved outright without searching. But the
survey found that **73% of goals have at least one straight-line-solvable
start** — the tracked method is finding about half of what is there, while
spending its entire time budget. The ones it misses sit at median position 111
in sweep order, far past where anything stops looking.

## Setup cost, separately from motion planning

The 27-hour library build is setup, not planning. Per goal, averaged over the
16 bars:

| setup stage | minutes per goal |
|---|---|
| goal IK: finding a reachable, collision-free goal pose | 0.2 |
| start survey: IK and collision checks on 2,304 carry poses | **17.4** |
| straight-line probes to each workable start | 7.1 |
| the planner's own start search, run for comparison | 1.8 |
| total | about 27 |

These were measured while 16 bars were surveyed at once on 8 cores, with the
machine short of memory at the end, so they are roughly 1.5 to 2 times what one
core would take alone.

Goal IK is cheap. The start survey is the cost, and for half the bars most of it
is repeated work. The carry poses are meant to be built from the grasp and the
three carry anchors only, independent of where the goal is:

| bars | across their 50 goals |
|---|---|
| B3, B6, B9, B12, B15 | stable: the same poses tried (to 10⁻⁵ m), the same verdict on every one |
| B33, B40, B46 | positions the same, but orientations flip by up to 180°; 13–18% of verdicts change (the workable count happens to stay constant) |
| B18, B21, B24, B27, B30, B36, B39, B43 | positions shift by 5–16 cm and orientations by up to 180°; 7–22% of verdicts change; the workable count varies by up to 5% |

For the five stable bars, the survey could run once per bar, with only the cheap
per-goal part -- choosing each start's IK solution nearest the goal's arm
configuration, and probing the straight line -- repeated per goal. That would
remove nearly all of their survey time.

The other eleven are a finding about the planner, not only a cost. The survey
calls the planner's own carry-anchor code, so for those bars **the carry pose
the planner would start from depends on the goal pose in a way it is not meant
to** -- whole rotations of the carry pose, not a small drift. The likely cause
is that the carry orientation is computed by rotating the grasp axis onto an
anchor axis, which is ill-conditioned when the two are nearly opposite, so
floating-point differences between goals pick very different answers. This is
not yet confirmed and deserves its own look before anything relies on it.

One caution in reading the benchmark's own numbers: the workable-start count is
a poor check for this. Three bars keep a constant count while one sixth of their
verdicts change underneath it.

## What I would try, in order

None of this is implemented. Rough order by expected gain per unit of work.

**1. Choose the start by whether it can be reached, not only held.**
The straight-line probe separates 43.6% success from 1.1% and costs about 0.2 s
per candidate. Sampling perhaps twenty candidates across all three anchors and
keeping the best-probing one would cost ~4 s, against the 121 s the current
derivation already spends. This looks like the largest available gain and the
cheapest to get.

**2. Stop the anchor sweep short-circuiting on `horizontal`.**
The loop tries every offset of every anchor but returns at the first success, and
the ordering guarantees that is always `horizontal`. The other two anchors are
roughly twice as good and are never reached. Interleaving anchors before
offsets would be a small change.

**2b. Check the carry-anchor orientation for numerical instability.**
On eleven of sixteen bars the carry pose the planner would start from rotates
between goals by up to 180°, and moves by up to 16 cm, where it should not
depend on the goal at all (see "Setup cost"). If confirmed, that makes the start
the planner picks partly a matter of floating-point noise, which is worth
settling before tuning anything that depends on the start.

**3. Look hard at the continuity gate.**
75% of all rejections. Worth establishing whether the ~10° raw threshold is
simply too tight for the step sizes in use, and whether a rejected step could be
recovered by re-seeding the IK or re-branching, rather than discarded. I would
want a focused experiment sweeping the threshold before changing it, since
loosening it risks admitting genuine branch flips that the execution side cannot
track.

**4. Make the collision check cheaper.**
37% of runtime for 3.7% of rejections, at 12.9 ms a call, most of it cloning
cell state. A cheaper path for the common case would buy a large speedup without
touching search behaviour at all. This makes runs faster; it will not by itself
make more of them succeed.

**5. Consider restarts as a first-class strategy.**
Success rises from 15.5% on one seed to 28.0% on ten, and is still climbing.
The curve suggests the search is high-variance rather than systematically stuck,
which is consistent with findings 1 and 3.

## Things worth knowing about the method

- **The coarse pre-pass is sound and cheap.** 17.1% pass coarse, 16.3% pass
  both; only 208 of 19,726 coarse successes then failed at fine resolution. It
  rejects hopeless problems in ~21 s instead of ~143 s, and it is not throwing
  away solvable ones.
- **Goal sampling is not the bottleneck.** All 800 goals are reachable and
  collision-free by the planner's own endpoint solver, so every failure here is
  a search failure, not a bad problem.
- **The shortcut is rare in this problem set.** Only 3.2% of feasible starts are
  joined to their goal by a free straight line, so the benchmark is genuinely
  measuring the search rather than the shortcut.
- **This is one library, one robot, one export.** The conclusions are about this
  planner on this bar set; re-run before generalising.
