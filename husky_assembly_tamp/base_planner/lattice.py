# * Copy of husky-assembly-teleop `husky_assembly_teleop/plugins/base_planner/algorithms/lattice.py` at monitor commit
# * b75b9975. Kept close to the original so it can be re-synced: what is not listed here is verbatim.
# * Cut: the viser `Lattice` class, the FOLLOWED layer, every use of crl_husky and of the monitor's `follow.py`
# *      (follower model, xICR range, the simulated follow), `BasePath` and its timing, `PlanResult.hit`.
# * Changed, all marked "Geometry only" or listed here:
# * - `Settings`: `x_icr` and `time_limit` instead of `x_icr_spread` and `follower_model`.
# * - `plan`: no `hit` in its results; the time limit is `settings.time_limit`.
# * - `_plan_one`: one turning centre and no swing; `hits` checks the poses of the turning centre's path instead of
# *   `path.driven`; `follows` checks the returned waypoints instead of simulating the follower; `found` returns
# *   the waypoints as an array (`_points`); `tried` and `found` draw no traces; the last failure's wording.
# * Added at the end: `_points`, `drive_margins`, `plan_drive`.
# * Added: `Settings.one_way_no` and EXHAUSTED (off by default: `plan` then searches both ways as the monitor does).
"""
Hybrid A* over the base's floor pose (x, y, yaw) that drives curves where they fit, and lines up exactly where it must.

- ! Geometry only: the monitor checks every move as its path follower would drive it, for a range of turning
  centres, and simulates the follower along the path found. Here a move is checked where it is planned, for one
  turning centre (`Settings.x_icr`), and no crl_husky code is used.
- Plans the turning centre (`Settings.x_icr` ahead of base_footprint), which moves like an ideal differential drive:
  a turn on the spot turns about that centre. Checks and the path returned are in base_footprint poses.
- Moves: drive straight or along an arc, forwards or backwards, or turn on the spot. Through a narrow place (a portal)
  in one move: lined up exactly before it by a turn-drive-turn, then straight through. The target is reached driving
  straight, from where the robot has room.
- Each move is checked in small steps, with a little extra margin on curves (`_track`, `_Node.slack`).
- Guided by the cost still to go on a coarse map of where the robot's shape fits at each heading (`_CostMap`). The path
  found is shortened, then its waypoints are checked once more (`follows`).
- Narrow passages: moves may come as close as the narrow margin where the margin does not fit; the cost map makes
  such places dear, and the single move and the shortening keep the margin.
- Searches from the tighter end first, then the other way, each with `Settings.time_limit` (so up to twice it in
  all): a search that starts in a narrow place is already lined up with it.
"""

from __future__ import annotations

import heapq
import itertools
from collections import Counter
import math
import threading
import time
from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import fftconvolve
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .path import SETTLE_DISTANCE, densify, steer
from .support import BASE_PLAN_DISTANCE, LEAD, POSITION_STEP, TIME_LIMIT, Debug, PlanResult, end_margins, yaw_step
from .world import CLEARANCE_LIMIT, CLEARANCE_STEP, COLLISION_MARGIN, FLOOR_BOTTOM, PlanningWorld

#: Search grid: the first pose to reach a cell and heading bin keeps it. Cell size, metres (also of the cost map's fine
#: maps), and heading bin, radians.
CELL = 0.05
HEADING_BIN = math.radians(2.5)
#: Straight drives of a move, metres; turns on the spot drive the longer one after.
STEPS = (0.05, 0.10)
#: Arcs: ARC_LENGTH metres at each of ARC_RADII metres, left or right, forwards or backwards.
ARC_RADII = (0.6, 1.2, 2.4)
ARC_LENGTH = 0.2
#: Turns on the spot, degrees, each way: small ones to line up, big ones to turn.
TURNS = (2.5, 5.0, 7.5, 15.0, 30.0, 45.0, 90.0, 135.0, 180.0)
#: After a turn on the spot of BIG_TURN degrees or more, drive at least MIN_DRIVE metres before the next turn, change
#: of direction or stop (`_fragile`): the turn leaves the base off the path when its true xICR is off the model, and
#: the follower may circle the end of a shorter drive for ever (crl_husky pure pursuit, simulated: `follow.followed`).
BIG_TURN = 45.0
MIN_DRIVE = 0.3
#: A move costs the metres driven, REVERSE_WEIGHT more per metre backwards, CURVE_WEIGHT per radian turned while
#: driving, the turn weight per radian turned on the spot plus STOP, and STOP per change of direction.
TURN_WEIGHT = 0.6   # m/rad
CURVE_WEIGHT = 0.2  # m/rad
REVERSE_WEIGHT = 0.2
STOP = 0.5          # m
#: On a curve of radius R the base strays up to TRACK_BASE + TRACK_GAIN * swing / R off it, where swing is how far its
#: turning centre may be off the follower's model (fitted to `follow.followed`; `path.driven` leaves it out).
TRACK_BASE = 0.01   # m
TRACK_GAIN = 0.3
#: The distance still to go counts this many times: 1 finds the cheapest path, more finds one sooner.
GREED = 2.5
#: Where only the narrow margin fits, driving costs NARROW_WEIGHT times more on the cost map.
NARROW_MARGIN = 0.05  # m
NARROW_WEIGHT = 2.0
#: Each search direction stops searching FINISH_RESERVE seconds before its time is up, for the shortening, the check
#: as followed and the planner plugin's measuring (`planner_plugin.narrowed`).
FINISH_RESERVE = 3.0  # s
#: How a search words having tried every pose it can reach (`Settings.one_way_no`).
EXHAUSTED = "no path inside the planning area"
#: Try to finish from every FINISH_EVERY-th node taken, and from every node within FINISH_NEAR metres of the lead's
#: start.
FINISH_EVERY = 10
FINISH_NEAR = 1.0
#: Keep at most this many entries into one grid cell waiting; an entry is only checked once taken, and one that hits
#: frees its place.
MAX_ENTRIES = 4
#: The search stays in the box around start and target, grown by this much per side, metres.
PADDING = 3.0
#: Cost map: where the robot's shape fits, at SHAPE_HEADINGS headings on the search grid, then on a MAP_CELL grid at
#: MAP_HEADINGS headings (fits if it fits at a nearby sample), and the cost to go from there, BLOCKED times where it
#: does not fit, so the search avoids such places but can still pass where the cost map is too coarse.
SHAPE_HEADINGS = 72
MAP_HEADINGS = 36
MAP_CELL = 0.10    # m
BLOCKED = 10.0
#: Room: where the robot fits at fewer than ROOMY_FITS of the SHAPE_HEADINGS headings, the cost map's way costs more,
#: up to TIGHT times where it fits at one. So the cost map leads round a squeeze when a way with room is not much
#: longer (the search finds that sooner, and the follower drives it more surely); a passage that is the only way costs
#: only TIGHT.
ROOMY_FITS = 36
TIGHT = 3.0
#: The shape fits where it overlaps things in fewer than this many search-grid cells: rasterizing it is that rough.
OVERLAP = 3
#: The cost map learns from the search: every move that hits makes its cell LEARN_STEP times dearer (up to BLOCKED),
#: and after every RELEARN_HITS such hits the cost to go is worked out again. So a way the cost map wrongly thinks fits
#: (it is coarse, and leaves out the follower's swing) stops pulling the search in.
LEARN_STEP = 1.0
RELEARN_HITS = 300
#: Cost map moves: grid steps (one way; the other is the same edge), used at headings within MAP_SLIP of their
#: direction.
DIRECTIONS = ((1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2), (-1, 1), (-2, 1))
MAP_SLIP = math.radians(30.0)
#: Portals: narrow places, where the robot fits at PORTAL_FITS or fewer of the SHAPE_HEADINGS headings, the one
#: fitting at fewest per PORTAL_SPACING square, at most PORTAL_COUNT. The search may cross one in a single move: lined
#: up exactly PORTAL_LEAD before, then straight through and PORTAL_PAST beyond.
PORTAL_FITS = 8
PORTAL_SPACING = 0.3          # m
PORTAL_COUNT = 200
PORTAL_LEAD = 1.0             # m: a curve onto it settles (SETTLE_DISTANCE) before the narrow place
PORTAL_PAST = 0.3             # m
PORTAL_REACH = 3.0            # m, try portals whose lead-in is this close
PORTAL_TRIES = 2              # portals tried from one node
#: Centring portals takes at most this share of a search's time: in a cluttered place every one costs hundreds of
#: checks, and none may have a way through.
PORTAL_SHARE = 1.0 / 3.0
#: A portal is centred when first tried: its line turned and shifted sideways to the widest clear room, first ROUGH
#: then FINE (turns in degrees, shifts in metres); it reaches along that line until the base has PORTAL_ROOM to spare.
ROUGH = (np.arange(-36.0, 36.1, 3.0), np.linspace(-0.09, 0.09, 10))
FINE = (np.arange(-2.0, 2.1, 0.5), np.linspace(-0.04, 0.04, 17))
PORTAL_ROOM = 0.10            # m
PORTAL_STEP = 0.05            # m, along the line
#: Things at least FLOOR_BOTTOM high count (`world`). Ones lower than BAND only hit parts of the robot that reach
#: below BAND (chassis, wheels), so an arm passes over them; higher ones hit every part.
BAND = 0.30  # m
#: Give shortening the path at most this long, seconds; it joins poses about ANCHOR_SPACING metres apart.
SHORTEN_TIME = 1.5
ANCHOR_SPACING = 0.3
#: Diagnostics: where the search got stuck, hits within NEAR metres count. At most TOUCHES_DRAWN touch points and
#: FINISHES_DRAWN refused finishes (the last) are drawn.
NEAR = 0.5
TOUCHES_DRAWN = 300
FINISHES_DRAWN = 40
#: Spacing of the poses sent to the follower, metres and radians: it fits a curve through them, which must stay on the
#: plan; a turn on the spot is sent as base_footprint's arc about the turning centre.
SEND_STEP = 0.05
SEND_TURN = 0.05

# * Own colours: the debug view's layers (test_colors skips this file).
#: Debug layers, back to front: name and colour (r, g, b).
FLOOR = "cost map: no heading fits", (206, 212, 218)
LEARNED = "cost map: learned to avoid", (255, 212, 59)
MAP_WAY = "cost map's way", (250, 176, 5)
MAP_TIGHT = "cost map's way: robot does not fit", (232, 89, 12)
LEAD_IN = "lead onto the target", (174, 62, 201)
TAKEN = "moves taken", (134, 142, 150)
HIT = "moves that hit, where the base was", (224, 49, 49)
FINISHES = "finishes refused", (255, 135, 135)
PORTALS = "portals", (12, 166, 120)
PORTAL_LINES = "portal lines, lead-in to far side", (9, 146, 104)
SEARCHED = "search's path", (247, 103, 7)
SHORTENED = "shortened", (47, 158, 68)
CLOSEST = "closest the search got", (132, 94, 247)
TOUCHES = "where the robot touches", (33, 37, 41)
FAILED_AT = "where a check failed", (201, 42, 42)


@dataclass(frozen=True)
class Settings:
    """The lattice search's settings for one search.

    Attributes:
        x_icr: How far base_footprint lies ahead of the robot's turning centre, metres (negative: behind), as
            crl_husky's xICR.
        lead: Straight drive onto the target, metres, from where the robot has room; 0 for none.
        turn_weight: Metres of driving one radian of turning on the spot is worth.
        margin: Keep at least this far from everything else, metres.
        curves: Whether the path may drive curves; else it turns on the spot only.
        narrow_margin: Where nothing keeps `margin`, keep at least this far, metres; at most `margin`.
        time_limit: Seconds per search direction: the first gives up this long after `plan` started, the second
            twice as long after it, each FINISH_RESERVE earlier.
    """

    # * Cindy's turning-centre offset, from crl-husky `crl_husky/config/base_models/cindy_tiles.yaml`.
    x_icr: float = -0.0171
    lead: float = LEAD
    turn_weight: float = TURN_WEIGHT
    margin: float = COLLISION_MARGIN
    curves: bool = True
    narrow_margin: float = NARROW_MARGIN
    time_limit: float = TIME_LIMIT
    # ! Not in the monitor: a direction that tried every pose it can reach ends the plan, the other way is not
    #   searched (EXHAUSTED). For a yes-or-no question asked many times; False plans as the monitor does.
    one_way_no: bool = False


def plan(world: PlanningWorld, own: str, start, goal, abort: threading.Event, settings: Settings = Settings(),
         debug: Debug | None = None) -> PlanResult[np.ndarray]:
    """Plan a collision-free base path from start to goal. Runs on the worker thread.

    Args:
        world: The planning world, already synced.
        own: The robot to plan for, by id.
        start: (x, y, yaw) where it is.
        goal: (x, y, yaw) where it should end up.
        abort: Set from the main thread to end the search early.
        settings: The search's settings.
        debug: Where to record the search's inner workings; None records nothing.

    Returns:
        PlanResult[np.ndarray]: The path's waypoints (x, y, yaw), shape (N, 3), or why there is none.
    """
    debug = debug or Debug()
    for layer in (FLOOR, LEARNED, MAP_WAY, MAP_TIGHT, LEAD_IN, PORTALS, PORTAL_LINES, TAKEN, HIT, FINISHES,
                  CLOSEST, SEARCHED, SHORTENED, FAILED_AT, TOUCHES):
        debug.layer(*layer)
    started = time.time()
    start, goal = tuple(map(float, start)), tuple(map(float, goal))
    narrow = min(settings.margin, settings.narrow_margin)
    for name, pose in (("start", start), ("target", goal)):
        other = world.hit_by(own, pose, narrow)
        if other is not None:
            return PlanResult(None, "cancelled" if abort.is_set() else f"{name} is in collision with {other}",
                              time.time() - started)
    # * From the tighter end first: a search that starts there is lined up with its narrow place already.
    backwards_first = world.clearance(own, [goal])[0] < world.clearance(own, [start])[0]
    reasons = []
    for k, backwards in enumerate((backwards_first, not backwards_first)):
        debug.note(f"search {'back from the target' if backwards else 'from the start'}")
        result = _plan_one(world, own, start, goal, abort, settings, debug,
                           started + settings.time_limit * (k + 1) - FINISH_RESERVE, backwards)
        if result.path is not None or abort.is_set():
            return PlanResult(result.path, result.reason, time.time() - started, direct=result.direct)
        reasons.append(f"{'back from the target' if backwards else 'from the start'}: {result.reason}")
        if settings.one_way_no and EXHAUSTED in result.reason:
            break
    return PlanResult(None, "; ".join(reasons), time.time() - started)


def _plan_one(world: PlanningWorld, own: str, start, goal, abort: threading.Event, settings: Settings, debug: Debug,
              deadline: float, backwards: bool) -> PlanResult[np.ndarray]:
    """`plan` in one direction, searching until `deadline`.

    `backwards`: the search runs from the start of the drive in back to `start`; the path found is reversed, then
    driven in onto `goal` as usual.
    """
    started = phase = time.time()
    # * The narrow margin is the hard one; the margin only where a move need not come closer.
    full = settings.margin
    margin = min(full, settings.narrow_margin)

    def lap(text: str) -> None:
        """Note how long the phase since the last lap took."""
        nonlocal phase
        debug.note(f"{text} ({time.time() - phase:.2f} s)")
        phase = time.time()

    def failed(reason: str) -> PlanResult[np.ndarray]:
        return PlanResult(None, "cancelled" if abort.is_set() else reason, time.time() - started)

    x_model = settings.x_icr
    # * Plan the turning centre: `ahead` metres ahead of base_footprint along its heading.
    ahead = -x_model
    # ! Geometry only: one turning centre where the monitor checks a range of them, so a turn leaves the base nowhere
    #   off the path (no swing).
    icrs = ends = (x_model,)
    swing = 0.0
    turn_step = yaw_step(world, own, icrs)
    # The last check that failed: where, with which xICR, and what happened.
    failure: list = []

    def hits(poses, xicrs, extra: float = 0.0, at: float | None = None) -> str | None:
        """What driving through `poses` (turning-centre poses, yaw unwrapped) comes within the margin plus `extra` of.

        `at` checks against that margin instead of the narrow one.

        For some xICR in `xicrs`: "hits <label>", "cancelled" once aborted, or None if nothing. A path that never turns
        drives the same whatever the xICR.
        """
        turns = any(abs(b[2] - a[2]) > 1e-9 for a, b in zip(poses[:-1], poses[1:]))
        for icr in xicrs if turns else xicrs[:1]:
            # ! Geometry only: base_footprint along the turning centre's path, in small steps. The monitor's
            #   `path.driven` gives these same poses when the follower's model is exact.
            points = [_to_base(pose, ahead) for pose in densify(poses, POSITION_STEP, turn_step)]
            # * The end first: most moves into something end in it, and it costs one check.
            for pose in points[-1:] + points[:-1]:
                if abort.is_set():
                    return "cancelled"
                other = hit_by(pose, (margin if at is None else at) + extra, at is not None)
                if other is not None:
                    failure[:] = [pose, icr, f"hits {other}", (margin if at is None else at) + extra]
                    return failure[2]
        return None

    def roomy(poses, xicrs, extra: float = 0.0) -> str | None:
        """`hits` at the full margin, what the ends are close to keeping its own margin (`measured_from`).

        Leaves `failure` as it was: a hit at the full margin is no failure.
        """
        kept = list(failure)
        try:
            with world.measured_from(full):
                return hits(poses, xicrs, extra, full)
        finally:
            failure[:] = kept

    # Checks done, by pose and clearance: moves share their turns and ends. Clearances are rounded up to
    # CLEARANCE_STEP, so moves with nearly the same slack share them too.
    done: dict[tuple, str | None] = {}

    def hit_by(pose, clearance: float, from_full: bool = False) -> str | None:
        clearance = math.ceil(clearance / CLEARANCE_STEP - 1e-9) * CLEARANCE_STEP
        # * `from_full`: checked from the full margin (`measured_from`), which keeps close objects differently.
        key = (round(pose[0], 4), round(pose[1], 4), round(pose[2], 4), round(clearance, 4), from_full)
        if key not in done:
            done[key] = world.hit_by(own, pose, clearance)
        return done[key]

    def follows(poses) -> bool:
        """Whether the base_footprint waypoints `poses`, in small steps, keep the margin.

        ! Geometry only: the monitor simulates its follower along `poses` here; this checks the path as returned.
        """
        for pose in [poses[0]] + densify(poses, POSITION_STEP, turn_step):
            if abort.is_set():
                return False
            other = hit_by(tuple(pose), margin)
            if other is not None:
                failure[:] = [tuple(pose), x_model, f"hits {other}", margin]
                return False
        return True

    def tried(text: str) -> None:
        """Note and draw why the path last followed failed."""
        lap(f"{text} {failure[2]} with xICR {failure[1]:+.3f} m at ({failure[0][0]:.2f}, {failure[0][1]:.2f})")
        debug.points(FAILED_AT[0], [failure[0]], FAILED_AT[1])
        touched([(failure[0], failure[3])] if failure[3] is not None else [])

    def touched(where) -> None:
        """Draw the points the robot touches, at (pose, clearance) each; only with debug on, as it costs checks."""
        # ? Never reached: `Debug` is always off here, and `PlanningWorld.touch` is not copied.
        if debug.on:
            touches = (world.touch(own, pose, clearance) for pose, clearance in where)
            debug.points(TOUCHES[0], [point[:2] for _, point in filter(None, touches)], TOUCHES[1])

    def room(pose) -> str:
        """How much room the robot has at `pose`: centimetres to the nearest thing and what it is, as text."""
        clear, near = world.nearest(own, pose)
        if near is None:
            return f"over {CLEARANCE_LIMIT * 100:.0f} cm"
        return f"touches {near}" if clear <= 0.0 else f"{clear * 100:.0f} cm to {near}"

    def found(poses, direct: bool = False) -> PlanResult[np.ndarray]:
        lap(f"found: {len(poses)} waypoints, clear with xICR {x_model:+.3f} m; {len(done)} poses checked in all")
        return PlanResult(_points(poses), seconds=time.time() - started, direct=direct)

    def room_along(back) -> tuple[float | None, str]:
        """How far along `back` from the target the robot first has PORTAL_ROOM beyond the margin, or None and why.

        The walk ends at something within the margin: the robot cannot stand there, so no room lies beyond it.
        """
        reach, alongside = 0.0, None
        while reach < PORTAL_REACH:
            pose = (*(goal[:2] + reach * back), goal[2])
            inside = hit_by(pose, margin) if reach > 0.0 else None
            if inside is not None:
                return None, (f"{inside} at {reach:.2f} m" if alongside in (inside, None)
                              else f"{alongside} alongside, then {inside} at {reach:.2f} m")
            near = hit_by(pose, margin + PORTAL_ROOM)
            if near is None:
                return reach, ""
            alongside = alongside or near
            reach += PORTAL_STEP
        return None, f"no room within {PORTAL_REACH:.0f} m, {alongside} alongside"

    def drive_in(back, distance: float, shortest: float = PORTAL_STEP) -> tuple[tuple | None, str]:
        """The start of a straight drive onto the target along `back`, or None and what it hits.

        From `distance` out, shortened in PORTAL_STEPs down to `shortest` until its start has PORTAL_ROOM beyond the
        margin: the robot turns onto the drive there. With room nowhere, the one from `distance`.
        """
        def start_of(along: float) -> tuple[tuple | None, str]:
            x, y = goal[:2] + along * back
            pose = (float(x), float(y), goal[2])
            other = hit_by(pose, margin)
            what = (f"hits {other}" if other is not None
                    else hits([_to_centre(pose, ahead), _to_centre(goal, ahead)], icrs))
            return (pose, "") if what is None else (None, f"{what} from {along:.2f} m out")

        along = distance
        while along >= shortest - 1e-9:
            x, y = goal[:2] + along * back
            if hit_by((float(x), float(y), goal[2]), margin + PORTAL_ROOM) is None:
                pose, _ = start_of(along)
                if pose is not None:
                    return pose, ""
            along -= PORTAL_STEP
        return start_of(distance)

    # * Reach the target driving straight, from the side facing the start first. In a narrow place the drive in starts
    # * `lead` before where the robot has room again, so it lines up there, not in the narrow place; with no such room,
    # * a drive in of just `lead` that keeps the margin. Either is shortened until its start has room to turn onto it.
    # * Only then the far side, which needs a way round the target.
    arrive, refused = goal, []
    if settings.lead > 0.0:
        heading = np.array([math.cos(goal[2]), math.sin(goal[2])])
        behind = 1.0 if np.dot(heading, np.subtract(goal[:2], start[:2])) >= 0.0 else -1.0
        for side in (behind, -behind):
            name, back = "forwards" if side > 0 else "backwards", -side * heading
            reach, why = room_along(back)
            pose = None
            if reach is not None:
                pose, why = drive_in(back, reach + settings.lead, max(reach, PORTAL_STEP))
            if pose is None:
                refused.append(f"{name}: {why}")
                if reach != 0.0:  # * at 0 the short drive in is the one just refused
                    pose, why = drive_in(back, settings.lead)
                    if pose is None:
                        refused.append(f"{name} short: {why}")
            if pose is not None:
                arrive = pose
                break
    tail = [goal] if arrive != goal else []
    debug.path(LEAD_IN[0], [arrive, goal], LEAD_IN[1])
    lead_note = (f"straight in {'forwards' if np.dot(np.subtract(goal[:2], arrive[:2]), heading) > 0 else 'backwards'}"
                 f" over {math.dist(arrive[:2], goal[:2]):.2f} m"
                 if tail else f"no straight drive in ({'; '.join(refused) or 'lead 0'})")
    target_room = room(goal)
    passed = f" ({'; '.join(refused)} refused)" if tail and refused else ""
    lap(f"lead: {lead_note}{passed}. Room (margin {margin * 100:.0f} cm): start {room(start)}, target {target_room}"
        + (f", start of the drive in {room(arrive)}" if tail else ""))

    # * From here on in turning-centre poses.
    start_c, arrive_c, tail_c = (_to_centre(start, ahead), _to_centre(arrive, ahead),
                                 [_to_centre(pose, ahead) for pose in tail])

    # * One move onto the lead's start, if it is clear.
    join = _join(start_c, arrive_c)
    if join:
        poses = [start_c] + join + _continued(tail_c, arrive_c[2], join[-1][2])
        if not roomy(poses, icrs, _track(poses, swing)):
            if follows(_sent(poses, ahead)):
                return found(_sent(poses, ahead), direct=True)
            tried("one move: as followed,")
    if abort.is_set():
        return failed("cancelled")
    if failure:
        lap(f"one move: blocked, last {failure[2]}")
        debug.points(FAILED_AT[0], [failure[0]], FAILED_AT[1])
    else:
        lap("one move: none that the follower drives reliably")

    low = np.minimum.reduce([start[:2], arrive[:2], goal[:2]]) - PADDING
    shape = tuple(int(math.ceil(size / CELL)) + 1 for size in np.maximum.reduce(
        [start[:2], arrive[:2], goal[:2]]) + PADDING - low)
    # * Backwards: the cost map leads to the start, and the search starts at the start of the drive in.
    cost_map = _CostMap(world, own, low, shape, start if backwards else arrive,
                        min(settings.turn_weight, CURVE_WEIGHT) if settings.curves else settings.turn_weight, full,
                        margin)
    debug.points(FLOOR[0], cost_map.nowhere(), FLOOR[1])
    debug.points(PORTALS[0], cost_map.portals, PORTALS[1])
    way = cost_map.way(arrive if backwards else start)
    for (x0, y0, _), (x1, y1, fits) in zip(way[:-1], way[1:]):
        if fits:
            debug.lines(MAP_WAY[0], [((x0, y0), (x1, y1))], MAP_WAY[1])
    tight = [(x, y) for x, y, fits in way if not fits]
    debug.points(MAP_TIGHT[0], tight, MAP_TIGHT[1])
    length = sum(math.dist(a[:2], b[:2]) for a, b in zip(way[:-1], way[1:]))
    lap(f"cost map: {shape[0]} x {shape[1]} cells, {len(cost_map.portals)} portals; its way is {length:.1f} m"
        + (f", and the robot fits nowhere near the heading along {len(tight) * MAP_CELL:.1f} m of it, first at "
           f"({tight[0][0]:.2f}, {tight[0][1]:.2f})" if tight else ", the robot fitting all along"))

    def widest(xy, yaw: float, turns, shifts):
        """The pose near (xy, yaw) with the widest clear room sideways; None if none is clear.

        Tries `turns` (degrees) and `shifts` sideways (metres).
        """
        best = (0, None)
        for turn in turns:
            heading = yaw + math.radians(turn)
            side = np.array([-math.sin(heading), math.cos(heading)])
            run, longest = 0, (0, 0)
            for i, shift in enumerate(shifts):
                clear = hit_by((*(np.asarray(xy) + shift * side), heading), margin) is None
                run = run + 1 if clear else 0
                longest = max(longest, (run, i))
            if longest[0] > best[0]:
                middle = np.interp(longest[1] - (longest[0] - 1) / 2.0, np.arange(len(shifts)), shifts)
                best = (longest[0], (np.asarray(xy) + middle * side, heading))
        return best[1]

    def centred(portal):
        """A portal's lead-in and far-side poses; None if no line through it is clear.

        Its line is turned and moved to the middle of the room there, and runs from where the base has room again
        before it to where it has room again beyond it.
        """
        rough = widest(portal[:2], portal[2], *ROUGH)
        middle = rough and widest(rough[0], rough[1], *FINE)
        if middle is None:
            return None
        xy, heading = middle
        along = np.array([math.cos(heading), math.sin(heading)])

        def reach(sign: float) -> float | None:
            for k in range(1, int(PORTAL_REACH / PORTAL_STEP) + 1):
                if hit_by((*(xy + sign * k * PORTAL_STEP * along), heading), margin + PORTAL_ROOM) is None:
                    return k * PORTAL_STEP
            return None

        behind, ahead = reach(-1.0), reach(1.0)
        if behind is None or ahead is None:
            return None
        lead_in, past = xy - (behind + PORTAL_LEAD) * along, xy + (ahead + PORTAL_PAST) * along
        return (float(lead_in[0]), float(lead_in[1]), heading), (float(past[0]), float(past[1]), heading)

    narrowing = roomy if full > margin + 1e-9 else None
    if backwards:
        corners, reason = _search(arrive_c, start_c, [], ahead, cost_map, centred, hits, failure, touched, ends, swing,
                                  settings, deadline, abort, debug, aim="the start", roomy=narrowing, flip=True)
        # ? Searched backwards, the drift slack and a big turn's drive after it count in search order; the check as
        #   followed below decides in driving order.
        corners = corners and corners[::-1] + tail_c
    else:
        corners, reason = _search(start_c, arrive_c, tail_c, ahead, cost_map, centred, hits, failure, touched, ends,
                                  swing, settings, deadline, abort, debug, roomy=narrowing)
    if corners is None:
        return failed(f"{reason}. Target: {target_room}, {lead_note}{passed}")
    corners = _merged(corners)
    debug.path(SEARCHED[0], [_to_base(pose, ahead) for pose in corners], SEARCHED[1])
    lap(f"search: {reason}; {len(corners)} waypoints")

    shortened = _shortened(corners, bool(tail), roomy, ends, swing, settings, time.time() + SHORTEN_TIME)
    debug.path(SHORTENED[0], [_to_base(pose, ahead) for pose in shortened], SHORTENED[1])
    lap(f"shortened to {len(shortened)} waypoints")

    for candidate in (shortened, corners) if shortened != corners else (corners,):
        if follows(_sent(candidate, ahead)):
            return found(_sent(candidate, ahead))
        if abort.is_set():
            return failed("cancelled")
        tried("path as followed")
    return failed(f"the path found {failure[2]} at ({failure[0][0]:.2f}, {failure[0][1]:.2f}) when checked again")


class _CostMap:
    """Cost still to go from a pose to the lead's start, on a coarse map of where the robot's shape fits per heading.

    Turning costs as in the search; at each heading it moves forwards or backwards in the grid directions within
    MAP_SLIP of the heading, never sideways, but may skip a cell (knight moves).
    """

    def __init__(self, world: PlanningWorld, own: str, low, shape: tuple[int, int], target, turn_weight: float,
                 margin: float = 0.0, narrow: float | None = None):
        """Map where the robot fits, and run Dijkstra over (x, y, heading) from the target.

        Args:
            world: The planning world, already synced.
            own: The robot to plan for, by id.
            low: (x, y) of the centre of the search grid's cell (0, 0), metres.
            shape: The search grid's cells along x and y (CELL apart).
            target: (x, y, yaw) to measure to.
            turn_weight: Metres of driving one radian of turning is worth.
            margin: Grow the robot's shape by this much, metres, to the search-grid cell: as the search checks it.
            narrow: Where only a shape grown by this much fits, driving costs NARROW_WEIGHT times more; None: `margin`.
        """
        heights = world.floor_heights(own, low, CELL, shape, FLOOR_BOTTOM)
        high, low_only = heights >= BAND, np.isfinite(heights) & (heights < BAND)
        top, under = world.robot_heights(own, CELL, FLOOR_BOTTOM)
        centre = top.shape[0] // 2

        def fine(grow: float) -> np.ndarray:
            """Whether the robot grown by `grow` hits something, per heading sample and search-grid cell."""
            shapes = [_grown(np.argwhere(np.isfinite(top)) - centre, grow),
                      _grown(np.argwhere(under < BAND) - centre, grow)]
            size = int(math.ceil(max(np.abs(np.vstack(shapes)).max(), centre) * math.sqrt(2.0))) + 1
            out = np.zeros((SHAPE_HEADINGS, *shape), dtype=bool)
            for n in range(SHAPE_HEADINGS):
                yaw = 2.0 * math.pi * n / SHAPE_HEADINGS
                rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
                for cells, things in zip(shapes, (high | low_only, low_only)):
                    mask = np.zeros((2 * size + 1, 2 * size + 1))
                    i, j = np.round(cells @ rotation.T).astype(int).T + size
                    mask[i, j] = 1.0
                    # The robot at a cell hits whatever lies under its shape there: correlate the map with the shape.
                    out[n] |= fftconvolve(things.astype(float), mask[::-1, ::-1], mode="same") > OVERLAP - 0.5
            return out

        step = int(round(MAP_CELL / CELL))
        nx, ny = (-(-size // step) for size in shape)
        per = SHAPE_HEADINGS // MAP_HEADINGS

        def coarse(fine_hit: np.ndarray) -> np.ndarray:
            """Coarse from fine: hits at a cost map heading and cell unless it fits at a nearby sample and fine cell.

            Fits at a cost map heading if it fits at a sample within one sample of it, and in a coarse cell if it fits
            in one of the search-grid cells there.
            """
            out = np.stack([fine_hit[(k * per + np.arange(-per + 1, per)) % SHAPE_HEADINGS].all(axis=0)
                            for k in range(MAP_HEADINGS)])
            out = np.pad(out, ((0, 0), (0, nx * step - shape[0]), (0, ny * step - shape[1])), constant_values=True)
            return out.reshape(MAP_HEADINGS, nx, step, ny, step).all(axis=(2, 4)).transpose(1, 2, 0)

        hit = fine(margin)
        narrow_hit = hit if narrow is None or narrow >= margin else fine(narrow)
        self.portals = _portals(~narrow_hit, low)
        # Room: how much more driving costs, per search-grid cell, from how many headings fit there.
        room = 1.0 + (TIGHT - 1.0) * np.clip(1.0 - (~hit).sum(axis=0) / ROOMY_FITS, 0.0, 1.0)
        hit, narrow_hit = coarse(hit), (coarse(narrow_hit) if narrow_hit is not hit else None)
        # A coarse cell has the room of its roomiest search-grid cell, as it fits if one of them fits.
        room = np.pad(room, ((0, nx * step - shape[0]), (0, ny * step - shape[1])), constant_values=TIGHT)
        room = room.reshape(nx, step, ny, step).min(axis=(1, 3))
        # * Fits: where the robot fits at the narrow margin; where only there, at NARROW_WEIGHT times the room.
        fits = hit if narrow_hit is None else narrow_hit
        self._low, self._shape, self._hit = np.asarray(low, dtype=float) + (step - 1) * CELL / 2.0, (nx, ny), fits

        index = np.arange(nx * ny * MAP_HEADINGS).reshape(nx, ny, MAP_HEADINGS)
        turn = turn_weight * 2.0 * math.pi / MAP_HEADINGS + 1e-6
        rows, cols, lengths = [], [], []

        def join(a, b, length):
            rows.append(index[a].ravel())
            cols.append(index[b].ravel())
            lengths.append(np.full(rows[-1].shape, length))

        # Grid steps at each heading, both ways, for `way`.
        self._steps = [[] for _ in range(MAP_HEADINGS)]
        for k in range(MAP_HEADINGS):
            for dx, dy in DIRECTIONS:
                if abs(math.remainder(math.atan2(dy, dx) - 2.0 * math.pi * k / MAP_HEADINGS, math.pi)) > MAP_SLIP:
                    continue
                self._steps[k] += [(dx, dy), (-dx, -dy)]
                join(np.s_[max(0, -dx):nx + min(0, -dx), max(0, -dy):ny + min(0, -dy), k],
                     np.s_[max(0, dx):nx + min(0, dx), max(0, dy):ny + min(0, dy), k], MAP_CELL * math.hypot(dx, dy))
        join(np.s_[:, :, :], (slice(None), slice(None), np.roll(np.arange(MAP_HEADINGS), -1)), turn)
        # The edges between grid nodes, with their length before weighting.
        self._edges = tuple(np.concatenate(part) for part in (rows, cols, lengths))
        # A source node joined to the target's cell at every heading, by the turn to the target's yaw.
        i, j = self._cell(target)
        headings = 2.0 * math.pi * np.arange(MAP_HEADINGS) / MAP_HEADINGS
        self._source = (np.full(MAP_HEADINGS, index.size), index[i, j],
                        turn_weight * np.abs(np.angle(np.exp(1j * (headings - target[2])))) + 1e-6)
        # Where the robot does not fit counts BLOCKED times, where it fits with little room more as it has less, and
        # where the search's moves hit, more as they hit.
        self._fits = np.where(fits, BLOCKED, np.where(hit, NARROW_WEIGHT, 1.0) * room[:, :, None])
        self._hits = np.zeros((nx, ny))
        #: Goes up each time the cost to go is worked out again.
        self.version = 0
        self._solve()

    def _solve(self) -> None:
        """Work out the cost to go from every grid node, with the weights as learned so far."""
        nx, ny = self._shape
        weight = np.maximum(self._fits, np.minimum(BLOCKED, 1.0 + LEARN_STEP * self._hits)[:, :, None]).ravel()
        rows, cols, lengths = self._edges
        lengths = lengths * (weight[rows] + weight[cols]) / 2.0
        graph = coo_matrix((np.concatenate([lengths, self._source[2]]),
                            (np.concatenate([rows, self._source[0]]), np.concatenate([cols, self._source[1]]))),
                           shape=(nx * ny * MAP_HEADINGS + 1,) * 2).tocsr()
        self._to_go = dijkstra(graph, directed=False, indices=nx * ny * MAP_HEADINGS)[:-1].reshape(
            nx, ny, MAP_HEADINGS)
        self.version += 1

    def hit(self, pose) -> None:
        """Count a move of the search that hit something with base_footprint at `pose`."""
        i, j = self._cell(pose)
        if 0 <= i < self._shape[0] and 0 <= j < self._shape[1]:
            self._hits[i, j] += 1.0

    def relearn(self) -> None:
        """Work out the cost to go again, making the cells where moves hit dearer."""
        self._solve()

    def learned(self) -> np.ndarray:
        """(x, y) of the cells the cost map learned to avoid: as dear as where the robot does not fit."""
        return self._low + MAP_CELL * np.argwhere(1.0 + LEARN_STEP * self._hits >= BLOCKED)

    def _cell(self, xy) -> tuple[int, int]:
        """The coarse grid index around (x, y), which may lie outside the grid."""
        i, j = np.round((np.asarray(xy[:2], dtype=float) - self._low) / MAP_CELL).astype(int)
        return int(i), int(j)

    def __call__(self, x: float, y: float, yaw: float) -> float:
        """Cost still to go from (x, y, yaw); inf outside the map."""
        i, j = self._cell((x, y))
        if 0 <= i < self._shape[0] and 0 <= j < self._shape[1]:
            return float(self._to_go[i, j, int(round(yaw * MAP_HEADINGS / (2.0 * math.pi))) % MAP_HEADINGS])
        return math.inf

    def way(self, pose) -> list[tuple[float, float, bool]]:
        """The way the cost map sees from `pose` to its target: (x, y, whether the robot fits there at that heading)."""
        i, j = self._cell(pose)
        k = int(round(pose[2] * MAP_HEADINGS / (2.0 * math.pi))) % MAP_HEADINGS
        nx, ny = self._shape
        if not (0 <= i < nx and 0 <= j < ny and math.isfinite(self._to_go[i, j, k])):
            return []
        out = []
        for _ in range(nx * ny):
            out.append((*(self._low + MAP_CELL * np.array([i, j])), not self._hit[i, j, k]))
            options = [(i, j, (k + 1) % MAP_HEADINGS), (i, j, (k - 1) % MAP_HEADINGS)]
            options += [(i + dx, j + dy, k) for dx, dy in self._steps[k] if 0 <= i + dx < nx and 0 <= j + dy < ny]
            best = min(options, key=lambda cell: self._to_go[cell])
            if self._to_go[best] >= self._to_go[i, j, k]:
                break
            i, j, k = best
        return out

    def nowhere(self) -> np.ndarray:
        """(x, y) of the coarse cells where the robot fits at no heading."""
        return self._low + MAP_CELL * np.argwhere(self._hit.all(axis=2))


def _grown(cells: np.ndarray, margin: float) -> np.ndarray:
    """Search-grid cells (offsets) of a shape grown by `margin` metres, to the nearest cell."""
    reach = int(round(margin / CELL))
    if reach <= 0 or not len(cells):
        return cells
    disk = np.array([(i, j) for i in range(-reach, reach + 1) for j in range(-reach, reach + 1)
                     if i * i + j * j <= reach * reach])
    return np.unique((cells[:, None, :] + disk[None, :, :]).reshape(-1, 2), axis=0)


def _portals(fits: np.ndarray, low) -> list[tuple[float, float, float]]:
    """Narrow places, and the way through each.

    Args:
        fits: Bool (heading sample, x, y): whether the robot fits there at that heading.
        low: (x, y) of the centre of cell (0, 0), metres.

    Returns:
        list: (x, y, yaw) per portal: roughly a heading it fits at there, along the narrow place, either way.
    """
    count = fits.sum(axis=0)
    narrow = (count > 0) & (count <= PORTAL_FITS)
    cells = np.argwhere(narrow)
    if not len(cells):
        return []
    squares = np.floor(cells * CELL / PORTAL_SPACING).astype(int)
    counts = count[cells[:, 0], cells[:, 1]]
    order = np.lexsort((counts, squares[:, 1], squares[:, 0]))
    first = np.r_[True, np.any(np.diff(squares[order], axis=0) != 0, axis=1)]
    chosen = order[first]
    chosen = chosen[np.argsort(counts[chosen], kind="stable")][:PORTAL_COUNT]

    portals = []
    for i, j in cells[chosen]:
        yaws = 2.0 * math.pi * np.flatnonzero(fits[:, i, j]) / fits.shape[0]
        # The axis of the headings that fit, either way along it.
        axis = 0.5 * math.atan2(np.mean(np.sin(2.0 * yaws)), np.mean(np.cos(2.0 * yaws)))
        x, y = np.asarray(low, dtype=float) + CELL * np.array([i, j])
        for yaw in (axis, axis + math.pi):
            # Only the way it fits: within one sample of a heading that fits.
            off = np.abs(np.remainder(yaws - yaw + math.pi, 2.0 * math.pi) - math.pi)
            if np.any(off <= 2.0 * math.pi / fits.shape[0]):
                portals.append((float(x), float(y), yaw))
    return portals


@dataclass
class _Node:
    """A pose the search reached, and how.

    Attributes:
        pose: (x, y, yaw), yaw unwrapped from the start.
        cost: Cost from the start.
        parent: Index of the node it came from; -1 for the start.
        via: The move's waypoints after the parent's pose, ending with `pose`.
        backwards: Whether the move's last drive went backwards.
        slack0: How far the base may be off the path at the end of the last turn, metres.
        since: Metres driven since that turn.
        piece: Metres driven since the last turn on the spot or change of direction.
        big: Whether the last turn on the spot was at least BIG_TURN.
    """

    pose: tuple[float, float, float]
    cost: float
    parent: int
    via: tuple
    backwards: bool
    slack0: float
    since: float
    piece: float = 0.0
    big: bool = False

    @property
    def slack(self) -> float:
        """How far the base may still be off the path here, metres (`path.driven` settles it over SETTLE_DISTANCE)."""
        return self.slack0 * max(0.0, 1.0 - self.since / SETTLE_DISTANCE)


def _search(start, arrive, tail, ahead: float, cost_map: _CostMap, centred, hits, failure: list, touched, ends,
            swing: float, settings: Settings, deadline: float, abort: threading.Event, debug: Debug,
            aim: str | None = None, roomy=None, flip: bool = False):
    """Hybrid A* over turning-centre poses from start to the lead's start; each move is checked once taken.

    Args:
        start: (x, y, yaw) to start from, of the turning centre, like all poses here.
        arrive: (x, y, yaw) to finish at, by `_join`.
        tail: Poses after `arrive` (the straight drive in), checked with the finish.
        ahead: How far the turning centre lies ahead of base_footprint, metres.
        cost_map: Cost still to go, and the portals, both for base_footprint.
        centred: `centred(portal)` from `plan`: a portal's base_footprint lead-in and far-side poses, or None.
        hits: `hits(poses, xicrs, extra)` from `plan`.
        failure: `plan`'s record of the last check that hit: where, with which xICR, what, and the clearance.
        touched: `touched(where)` from `plan`: draws where the robot touches, at (pose, clearance) each.
        ends: The xICR values to check moves for.
        swing: How far a turn may leave the base off the path per unit of chord turned, metres.
        settings: The search's settings.
        deadline: Give up at this time.
        abort: Set from the main thread to end the search early.
        debug: Where to record the search.
        aim: What `arrive` is, for the reason; None: the target, or the start of the drive in with a `tail`.
        roomy: `hits` at the full margin, if it is more than the narrow one: moves clear only at the narrow margin cost
            NARROW_WEIGHT times as much. None: every clear move costs the same.
        flip: The search runs back from the target: a move's poses are driven in the other order (`_backwards`).

    Returns:
        tuple[list | None, str]: The poses from start to the end of `tail`, or None; and what happened.
    """
    def base(pose) -> tuple[float, float, float]:
        return _to_base(pose, ahead)

    def to_go(pose) -> float:
        return cost_map(*base(pose))

    # Moves: (kind, amount, drive), the drive signed (negative: backwards). "drive" straight; "turn" on the spot by
    # `amount` radians, then drive; "arc" at curvature `amount`, 1/m; "portal": through portal `amount`, lined up by
    # its join number `drive`.
    moves = [("drive", 0.0, sign * step) for step in STEPS for sign in (1.0, -1.0)]
    moves += [("turn", sign * math.radians(turn), drive) for turn in TURNS
              for sign in ((1.0,) if turn == 180.0 else (1.0, -1.0)) for drive in (STEPS[-1], -STEPS[-1])]
    if settings.curves:
        moves += [("arc", sign / radius, drive) for radius in ARC_RADII for sign in (1.0, -1.0)
                  for drive in (ARC_LENGTH, -ARC_LENGTH)]
    headings = int(round(2.0 * math.pi / HEADING_BIN))

    def cell(pose) -> tuple[int, int, int]:
        return (int(round(pose[0] / CELL)), int(round(pose[1] / CELL)),
                int(round(pose[2] / HEADING_BIN)) % headings)

    # Portals by index: their lead-in and far-side poses once centred (None: no way through), and which were tried
    # from which node.
    through: dict[int, tuple | None] = {}
    tried: set = set()
    # Seconds spent centring portals, and how many it may take (PORTAL_SHARE of the search's time).
    centring = [0.0, PORTAL_SHARE * max(deadline - time.time(), 0.0)]

    def child(parent: _Node, index: int, move) -> _Node:
        kind, amount, drive = move
        x, y, yaw = parent.pose
        if kind == "portal":
            lead_in, past = through[amount]
            join = _join(parent.pose, lead_in)
            via = tuple(join + _continued([past], lead_in[2], join[-1][2]))
        elif kind == "arc":
            via = tuple(_arc(parent.pose, amount, drive, SEND_STEP))
        else:
            yaw += amount
            end = (x + drive * math.cos(yaw), y + drive * math.sin(yaw), yaw)
            via = ((x, y, yaw), end) if kind == "turn" else (end,)
        cost, backwards = _cost([parent.pose, *via], settings.turn_weight, parent.backwards, flip)
        slack0, since = _settled(parent.slack0, parent.since, [parent.pose, *via], swing)
        piece, big = _piece(parent.piece, parent.big, parent.backwards, [parent.pose, *via], flip)
        return _Node(via[-1], parent.cost + cost, index, via, backwards, slack0, since, piece, big)

    def poses_to(index: int) -> list:
        chain = []
        while index > 0:
            chain.append(nodes[index])
            index = chain[-1].parent
        return [start] + [pose for node in reversed(chain) for pose in node.via]

    def portal_moves(node: _Node, index: int) -> list:
        """Moves through up to PORTAL_TRIES portals: lead-in within reach, far side closer to the target."""
        here = to_go(node.pose)
        options = []
        for k, (x, y, yaw) in enumerate(cost_map.portals):
            # Roughly, before centring: the narrow place as long as the robot.
            lead_in = (x - (PORTAL_LEAD + 0.5) * math.cos(yaw), y - (PORTAL_LEAD + 0.5) * math.sin(yaw))
            past = (x + (PORTAL_PAST + 0.5) * math.cos(yaw), y + (PORTAL_PAST + 0.5) * math.sin(yaw), yaw)
            near = math.dist(base(node.pose)[:2], lead_in)
            # ! Skip portals already found to have no way through: they would take the tries from the good ones.
            if (near < PORTAL_REACH and (index, k) not in tried and through.get(k, True) is not None
                    and cost_map(*past) < here):
                options.append((cost_map(*past) + near, k))
        out = []
        for _, k in sorted(options)[:PORTAL_TRIES]:
            tried.add((index, k))
            if k not in through:
                if centring[0] >= centring[1]:
                    continue
                began = time.time()
                line = centred(cost_map.portals[k])
                centring[0] += time.time() - began
                through[k] = line and tuple(_to_centre(pose, ahead) for pose in line)
                crossings[k] = Counter()
            if through[k] is not None and _join(node.pose, through[k][0]):
                out.append(("portal", k, 0.0))
        return out

    nodes = [_Node(start, 0.0, -1, (start,), False, 0.0, 0.0)]
    # Entries: (cost + greed * to go, cost, tie-break, parent index, move); a node is made only once its entry is taken.
    ties = itertools.count()
    # The last item: the extra cost a move that fits only at the narrow margin was given, once checked (0: not yet).
    heap = [(GREED * to_go(start), 0.0, next(ties), -1, None, cost_map.version, 0.0)]
    relearned, since_relearn = 0, 0
    closed: set = set()
    entries: dict = {}
    taken, hit, refused = [], [], []
    # What the moves hit, per hit; and the node with the least left to go, by the cost map.
    blocked: list[str] = []
    # Per hit: the pose that hit and the clearance checked, and whether it was through a portal.
    hit_at: list = []
    closest = (math.inf, 0)
    # Per portal centred: what the moves through it hit.
    crossings: dict[int, Counter] = {}

    budget = deadline - time.time()

    def stuck(why: str) -> str:
        """Why the search ended without a path, and where it got closest."""
        node = nodes[closest[1]]
        x, y, yaw = base(node.pose)
        debug.path(CLOSEST[0], [base(pose) for pose in poses_to(closest[1])], CLOSEST[1])
        debug.points(CLOSEST[0], [(x, y)], CLOSEST[1])
        near = Counter(what for pose, what in zip(hit, blocked) if math.dist(pose[:2], (x, y)) < NEAR)
        where = aim or ("the start of the drive in" if tail else "the target")
        return (f"{why}; closest {math.dist((x, y), base(arrive)[:2]):.1f} m from {where}, at "
                f"({x:.2f}, {y:.2f}, {math.degrees(yaw) % 360:.0f}°)"
                + (f", where moves hit {_most(near)}" if near else "")
                + (f"; in all, moves hit {_most(Counter(blocked))}"
                   if blocked and Counter(blocked).most_common(1)[0][0] not in near else ""))

    try:
        while heap:
            if abort.is_set():
                return None, "cancelled"
            if time.time() > deadline:
                return None, stuck(f"no path found in {budget:.0f} s")
            _, cost, _, parent_index, move, version, dear = heapq.heappop(heap)
            node = nodes[0] if parent_index < 0 else child(nodes[parent_index], parent_index, move)
            if dear:
                node = replace(node, cost=node.cost + dear)
            if version != cost_map.version:
                # * Scored by an older cost map: score it again, and take it when its turn comes.
                heapq.heappush(heap, (cost + GREED * to_go(node.pose), cost, next(ties), parent_index, move,
                                      cost_map.version, dear))
                continue
            key = cell(node.pose)
            if key in closed:
                continue
            if parent_index >= 0:
                parent = nodes[parent_index]
                poses = [parent.pose, *node.via]
                slack = max(parent.slack, _track(poses, swing))
                what = hits(poses, ends, slack)
                if not what and roomy is not None and not dear and roomy(poses, ends, slack):
                    # * Fits only at the narrow margin: NARROW_WEIGHT times as dear, taken when its turn comes.
                    dear = (NARROW_WEIGHT - 1.0) * (node.cost - parent.cost) + 1e-9
                    heapq.heappush(heap, (cost + dear + GREED * to_go(node.pose), cost + dear, next(ties),
                                          parent_index, move, version, dear))
                    continue
                if what:
                    hit.append(base(node.pose))
                    blocked.append(what.removeprefix("hits "))
                    hit_at.append((failure[0], failure[3], move[0] == "portal"))
                    cost_map.hit(failure[0])
                    since_relearn += 1
                    if since_relearn >= RELEARN_HITS:
                        cost_map.relearn()
                        relearned, since_relearn = relearned + 1, 0
                    if move[0] == "portal":
                        crossings[move[1]][what.removeprefix("hits ")] += 1
                    entries[key] -= 1  # another entry may still reach the cell clear
                    continue
                taken.append((base(parent.pose)[:2], base(node.pose)[:2]))
                nodes.append(node)
            index = len(nodes) - 1
            closed.add(key)
            closest = min(closest, (to_go(node.pose), index))

            extra = []
            if len(closed) % FINISH_EVERY == 1 or math.dist(node.pose[:2], arrive[:2]) < FINISH_NEAR:
                join = _join(node.pose, arrive)
                finish = join and [node.pose] + join + _continued(tail, arrive[2], join[-1][2])
                if finish and not _cuts_short(node, finish, flip):
                    if not hits(finish, ends, max(node.slack, _track(finish, swing))):
                        return poses_to(index) + finish[1:], f"{len(closed)} nodes taken, {len(hit)} moves hit"
                    refused.append(finish)
                extra = portal_moves(node, index)

            for move in moves + extra:
                next_node = child(node, index, move)
                next_key = cell(next_node.pose)
                if (next_key in closed or entries.get(next_key, 0) >= MAX_ENTRIES
                        or _cuts_short(node, [node.pose, *next_node.via], flip)):
                    continue
                left = to_go(next_node.pose)
                if math.isfinite(left):
                    entries[next_key] = entries.get(next_key, 0) + 1
                    heapq.heappush(heap, (next_node.cost + GREED * left, next_node.cost, next(ties), index, move,
                                          cost_map.version, 0.0))
        return None, stuck(EXHAUSTED)
    finally:
        debug.lines(TAKEN[0], taken, TAKEN[1])
        debug.points(HIT[0], hit, HIT[1])
        for finish in refused[-FINISHES_DRAWN:]:
            debug.path(FINISHES[0], [base(pose) for pose in finish], FINISHES[1])
        # * Touches near where the search got closest and through portals first, then evenly over the rest.
        near = base(nodes[closest[1]].pose)
        first = [k for k, (pose, _, portal) in enumerate(hit_at) if portal or math.dist(pose[:2], near[:2]) < NEAR]
        chosen_first = set(first)
        rest = [k for k in range(len(hit_at)) if k not in chosen_first]
        chosen = first[:TOUCHES_DRAWN] + rest[::max(1, len(rest) // max(1, TOUCHES_DRAWN - len(first)))]
        touched([hit_at[k][:2] for k in chosen[:TOUCHES_DRAWN]])
        debug.note(f"search: {len(closed)} nodes taken, {len(hit)} moves hit, {len(refused)} finishes refused, "
                   f"{len(through)} portals centred, {sum(v is None for v in through.values())} with no way through")
        if blocked:
            debug.note(f"moves hit {_most(Counter(blocked), 4)}")
        if relearned:
            learned = cost_map.learned()
            debug.points(LEARNED[0], learned, LEARNED[1])
            debug.note(f"cost map relearned {relearned} times from the hits; it now avoids {len(learned)} cells")
        for k, line in through.items():
            x, y, yaw = cost_map.portals[k]
            if line is None:
                debug.note(f"portal at ({x:.2f}, {y:.2f}): no clear line through it")
                continue
            debug.lines(PORTAL_LINES[0], [(base(line[0])[:2], base(line[1])[:2])], PORTAL_LINES[1])
            debug.note(f"portal at ({x:.2f}, {y:.2f}), line {math.degrees(line[0][2]) % 360:.1f}°: moves through it "
                       + (f"hit {_most(crossings[k])}" if crossings[k] else "never hit"))


def _most(counts: Counter, count: int = 2) -> str:
    """The most common things in `counts`, with how often, as text: "a ×12, b ×3"."""
    return ", ".join(f"{what} ×{n}" for what, n in counts.most_common(count))


def _join(q1, q2) -> list[tuple[float, float, float]] | None:
    """Turn-drive-turn from q1 exactly onto q2 (the poses after q1, yaw continued from q1's); None if it is fragile."""
    turns = steer(q1, q2)
    return None if _fragile([q1, *turns]) else turns


def _arc(pose, curvature: float, length: float, step: float) -> list[tuple[float, float, float]]:
    """Drive `length` metres (negative: backwards) at `curvature` (1/m, positive: left) from `pose`.

    Returns:
        list[tuple[float, float, float]]: The poses after `pose`, at most `step` metres apart.
    """
    x, y, yaw = (float(value) for value in pose)
    count = max(int(math.ceil(abs(length) / step)), 1)
    poses = []
    for i in range(1, count + 1):
        s = length * i / count
        if abs(curvature) < 1e-9:
            poses.append((x + s * math.cos(yaw), y + s * math.sin(yaw), yaw))
        else:
            turn = curvature * s
            poses.append((x + (math.sin(yaw + turn) - math.sin(yaw)) / curvature,
                          y - (math.cos(yaw + turn) - math.cos(yaw)) / curvature, yaw + turn))
    return poses


def _continued(poses, was: float, now: float) -> list[tuple[float, float, float]]:
    """`poses` turned by whole turns so their yaw follows on from `now`, where it followed on from `was` before."""
    shift = 2.0 * math.pi * round((now - was) / (2.0 * math.pi))
    return [(pose[0], pose[1], pose[2] + shift) for pose in poses]


def _merged(poses) -> list[tuple[float, float, float]]:
    """The waypoints without repeats, one turn per spot, and no stops between straight drives in the same direction."""
    out = [tuple(poses[0])]
    for pose in map(tuple, poses[1:]):
        if math.dist(pose[:2], out[-1][:2]) < 1e-9 and abs(pose[2] - out[-1][2]) < 1e-9:
            continue
        turns_on = len(out) > 1 and math.dist(out[-2][:2], out[-1][:2]) < 1e-9
        if turns_on and math.dist(out[-1][:2], pose[:2]) < 1e-9:
            out[-1] = pose
        elif (len(out) > 1 and _drive(out[-2], out[-1]) is not None
                and _drive(out[-2], out[-1]) == _drive(out[-1], pose)):
            out[-1] = pose
        else:
            out.append(pose)
    return out


def _drive(a, b) -> tuple[float, bool] | None:
    """The heading and direction (backwards?) of a straight drive from a to b; None if it turns or stays."""
    if abs(b[2] - a[2]) > 1e-9 or math.dist(a[:2], b[:2]) < 1e-9:
        return None
    return round(a[2], 9), _backwards(a, b)


def _backwards(a, b, flip: bool = False) -> bool:
    """Whether moving from a to b drives backwards; `flip`: the base drives from b to a (a search back from the end)."""
    return (math.cos(math.atan2(b[1] - a[1], b[0] - a[0]) - (a[2] + b[2]) / 2.0) < 0.0) != flip


def _cost(poses, turn_weight: float, backwards: bool = False, flip: bool = False) -> tuple[float, bool]:
    """What driving through `poses` costs, as the search counts it, and whether it ends driving backwards.

    Args:
        poses: (x, y, yaw) waypoints, yaw unwrapped.
        turn_weight: Metres one radian of turning on the spot is worth.
        backwards: Whether the base drove backwards before the first pose.
        flip: The base drives `poses` in the other order (`_backwards`).
    """
    cost = 0.0
    for a, b in zip(poses[:-1], poses[1:]):
        turned = abs(b[2] - a[2])
        length = math.dist(a[:2], b[:2])
        if length < 1e-9:
            cost += turn_weight * turned + STOP
            continue
        back = _backwards(a, b, flip)
        cost += length * (1.0 + (REVERSE_WEIGHT if back else 0.0)) + CURVE_WEIGHT * turned
        cost += STOP if back != backwards else 0.0
        backwards = back
    return cost, backwards


def _track(poses, swing: float) -> float:
    """How far the base may stray off the curves among `poses`, metres; 0 if they have none (`TRACK_GAIN`)."""
    worst = 0.0
    for a, b in zip(poses[:-1], poses[1:]):
        length = math.dist(a[:2], b[:2])
        if length > 1e-9 and abs(b[2] - a[2]) > 1e-9:
            worst = max(worst, TRACK_BASE + TRACK_GAIN * swing * abs(b[2] - a[2]) / length)
    return worst


def _piece(piece: float, big: bool, backwards: bool, poses, flip: bool = False) -> tuple[float, bool]:
    """`_Node.piece` and `big` after driving through `poses` from the first.

    Starts from `piece`, `big` and `backwards` (whether the base drove backwards before).
    """
    for a, b in zip(poses[:-1], poses[1:]):
        length = math.dist(a[:2], b[:2])
        if length < 1e-9:
            piece, big = 0.0, abs(b[2] - a[2]) >= math.radians(BIG_TURN) - 1e-9
            continue
        back = _backwards(a, b, flip)
        piece = (piece if back == backwards else 0.0) + length
        backwards = back
    return piece, big


def _fragile(poses) -> bool:
    """Whether a path has a turn on the spot of BIG_TURN or more with a drive shorter than MIN_DRIVE after it."""
    # Drive pieces and turns in order: ("drive", metres) or ("turn", radians).
    parts: list = []
    backwards = None
    for a, b in zip(poses[:-1], poses[1:]):
        length = math.dist(a[:2], b[:2])
        if length < 1e-9:
            if parts and parts[-1][0] == "turn":
                parts[-1] = ("turn", parts[-1][1] + b[2] - a[2])
            else:
                parts.append(("turn", b[2] - a[2]))
            backwards = None
            continue
        back = _backwards(a, b)
        if parts and parts[-1][0] == "drive" and back == backwards:
            parts[-1] = ("drive", parts[-1][1] + length)
        else:
            parts.append(("drive", length))
        backwards = back
    return any(kind == "turn" and abs(amount) >= math.radians(BIG_TURN) - 1e-9 and k + 1 < len(parts)
               and parts[k + 1][0] == "drive" and parts[k + 1][1] < MIN_DRIVE
               for k, (kind, amount) in enumerate(parts))


def _cuts_short(node: _Node, poses, flip: bool = False) -> bool:
    """Whether `poses`, from `node`, turn on the spot or change direction before MIN_DRIVE after a big turn."""
    if not node.big or node.piece >= MIN_DRIVE or len(poses) < 2:
        return False
    a, b = poses[0], poses[1]
    return math.dist(a[:2], b[:2]) < 1e-9 or _backwards(a, b, flip) != node.backwards


def _settled(slack0: float, since: float, poses, swing: float) -> tuple[float, float]:
    """`_Node.slack0` and `since` after driving through `poses` from the first, starting from `slack0` and `since`.

    A turn on the spot adds its swing; a curve keeps at least how far the base may stray on it; both then settle over
    SETTLE_DISTANCE of driving.
    """
    for a, b in zip(poses[:-1], poses[1:]):
        now = slack0 * max(0.0, 1.0 - since / SETTLE_DISTANCE)
        length = math.dist(a[:2], b[:2])
        if length < 1e-9:
            slack0, since = now + swing * 2.0 * math.sin(min(abs(b[2] - a[2]), math.pi) / 2.0), 0.0
        elif abs(b[2] - a[2]) > 1e-9:
            slack0, since = max(now, _track((a, b), swing)), 0.0
        else:
            since += length
    return slack0, since


def _shortened(poses, keep_last: bool, hits, ends, swing: float, settings: Settings, deadline: float) -> list:
    """The path with stretches replaced by one turn-drive-turn (`_join`) where that is cheaper and clear.

    Greedy from the start, longest stretch first, between poses about ANCHOR_SPACING apart. A stretch is checked with
    SETTLE_DISTANCE of the path after it, at the ends of the xICR range, with the slack the search would give it; the
    caller checks the whole path.

    Args:
        poses: The path's waypoints.
        keep_last: Keep the last drive (the straight drive onto the target) as it is.
        hits: `hits(poses, xicrs, extra)` from `plan`.
        ends: The xICR values to check for.
        swing: As in `_search`.
        settings: The search's settings.
        deadline: Stop at this time.
    """
    poses = list(poses)
    i = 0
    while time.time() < deadline:
        anchors = _anchors(poses, len(poses) - (2 if keep_last else 1))
        later = [j for j in anchors if j > i + 1]
        if not later:
            break
        slack0, since = _settled(0.0, 0.0, poses[:i + 1], swing)
        slack = slack0 * max(0.0, 1.0 - since / SETTLE_DISTANCE)
        for j in reversed(later):
            if time.time() > deadline:
                break
            join = _join(poses[i], poses[j])
            if (join is None or _cost([poses[i], *join], settings.turn_weight)[0]
                    >= _cost(poses[i:j + 1], settings.turn_weight)[0] - 1e-6):
                continue
            rest = _continued(poses[j + 1:], poses[j][2], join[-1][2])
            if _fragile(poses[:i + 1] + join + rest):
                continue
            stretch = [poses[i], *join, *_within(rest, join[-1], SETTLE_DISTANCE)]
            if not hits(stretch, ends, max(slack, _track(stretch, swing))):
                poses = _merged(poses[:i + 1] + join + rest)
                break
        i = min((j for j in _anchors(poses, len(poses) - 1) if j > i), default=len(poses))
    return poses


def _anchors(poses, last: int) -> list[int]:
    """Indices of poses about ANCHOR_SPACING apart along the path, and of every turn on the spot, up to `last`."""
    out, along, mark = [0], 0.0, ANCHOR_SPACING
    for i in range(1, last + 1):
        along += math.dist(poses[i - 1][:2], poses[i][:2])
        spot = math.dist(poses[i - 1][:2], poses[i][:2]) < 1e-9
        if along >= mark or spot or i == last:
            out.append(i)
            mark = along + ANCHOR_SPACING
    return out


def _within(poses, start, length: float) -> list:
    """The first of `poses` that lie within `length` metres of driving from `start`, and the one just beyond."""
    out, along, last = [], 0.0, start
    for pose in poses:
        out.append(pose)
        along += math.dist(last[:2], pose[:2])
        last = pose
        if along >= length:
            break
    return out


def _sent(poses, ahead: float) -> list[tuple[float, float, float]]:
    """The base_footprint waypoints to send for turning-centre `poses`: at most SEND_STEP and SEND_TURN apart."""
    poses = [tuple(map(float, pose)) for pose in poses]
    return [_to_base(pose, ahead) for pose in [poses[0]] + densify(poses, SEND_STEP, SEND_TURN)]


def _to_centre(pose, ahead: float) -> tuple[float, float, float]:
    """The turning-centre pose of a base_footprint pose: `ahead` metres along its heading."""
    x, y, yaw = (float(value) for value in pose)
    return x + ahead * math.cos(yaw), y + ahead * math.sin(yaw), yaw


def _to_base(pose, ahead: float) -> tuple[float, float, float]:
    """The base_footprint pose of a turning-centre pose."""
    return _to_centre(pose, -ahead)


# --- --- --- --- --- NOT IN THE MONITOR --- --- --- --- ---


def _points(poses) -> np.ndarray:
    """The waypoints as an (N, 3) array of (x, y, yaw), repeats left out: what the monitor's `BasePath` holds."""
    points = np.array(poses, dtype=float)
    return points[np.concatenate([[True], np.any(np.abs(np.diff(points, axis=0)) > 1e-9, axis=1)])]


def drive_margins(world: PlanningWorld, own: str, start, goal, margin: float) -> tuple[float, dict[str, float]]:
    """The margins a drive from start to goal is planned with, as the monitor's planner plugin sets them.

    Args:
        world: The planning world, already synced.
        own: The robot to plan for, by id.
        start: (x, y, yaw) where it is.
        goal: (x, y, yaw) where it should end up.
        margin: Keep at least this far from everything else where it can, metres.

    Returns:
        tuple[float, dict[str, float]]: The margin kept in narrow places, and object id -> the smaller margin kept
        from what the start or goal is already closer to; both for `world.keeping`.
    """
    narrow = min(margin, NARROW_MARGIN)
    return narrow, end_margins(world.close_objects(own, (start, goal), narrow), narrow < margin)


def plan_drive(world: PlanningWorld, own: str, start, goal, margin: float = BASE_PLAN_DISTANCE,
               time_limit: float = 8.0, one_way_no: bool = True) -> PlanResult[np.ndarray]:
    """Plan a collision-free base path from start to goal (x, y, yaw), the robot's arms and what it holds as synced.

    The monitor's live planning without its `roomier_goal`: the goal is planned to exactly as given.

    Args:
        world: The planning world, already synced.
        own: The robot to plan for, by id.
        start: (x, y, yaw) where it is, metres and radians.
        goal: (x, y, yaw) where it should end up.
        margin: Keep at least this far from everything else where it can, metres; NARROW_MARGIN in narrow places,
            and less only from what the start or goal is already closer to.
        time_limit: `Settings.time_limit`, seconds: with both directions searched, "no path" takes up to about
            twice this.
        one_way_no: `Settings.one_way_no`: a direction that tried every pose it can reach answers for both.

    Returns:
        PlanResult[np.ndarray]: The path's waypoints (x, y, yaw), shape (N, 3), yaw unwrapped; or why there is none.
    """
    narrow, close = drive_margins(world, own, start, goal, margin)
    with world.keeping(own, narrow, close):
        return plan(world, own, start, goal, threading.Event(),
                    Settings(margin=margin, time_limit=time_limit, one_way_no=one_way_no))
