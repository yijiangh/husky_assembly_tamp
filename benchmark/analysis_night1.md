# What the first benchmark run says

Run `night1`: 24,000 planning runs over 16 bars, 800 goal poses, 3 start poses
per goal, 10 random seeds each. 42.9 hours on 12 workers, no crashes and no
errors. Library `cdfm_v1`, built from `260921_motion_sample`.

Nothing here has been acted on. These are observations and the changes they
suggest, for review before anything is touched.

## The headline

| | |
|---|---|
| Runs solved | 3,906 of 24,000 = **16.3%** (95% interval 15.8 to 16.7) |
| Solved within 10 seeds | **28.0%** of problems |
| Median time on a solved run | 143 s fine pass, 21 s coarse |
| Timed out | 368 = 1.5% |

Bar length barely matters. Success runs from 8% (B39, 1.75 m) to 21% (B3,
0.50 m, and B6, and B18) with no clean trend, and the median time is 36 s for
every bar to within a second. Whatever limits this planner, it is not the size
of the thing being carried.

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

## Finding 2: the time goes where the failures do not

| component | mean seconds per pass | share |
|---|---|---|
| collision checking | 17.35 | **36.8%** |
| inverse kinematics | 15.54 | 33.0% |
| stitching the two trees | 6.11 | 13.0% |
| smoothing | 4.61 | 9.8% |
| everything else | 3.50 | 7.4% |

Collision checking is the largest cost and causes 3.7% of rejections. IK is the
second largest and causes, through the continuity gate it feeds, 94% of them.

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
