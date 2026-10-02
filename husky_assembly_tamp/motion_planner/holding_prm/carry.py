"""Carry poses: where the robot holds the bar while driving, and how good a candidate is.

A carry (home) pose has a class and two free choices:

    class       horizontal  the bar along the base's y axis, in front of the robot
                vertical    the bar along z, in front of the robot
                back        the bar along x, carried over the base
    midpoint    where the point halfway between the two grasps sits, in the
                mobile-base frame (metres)
    roll        how far the bar is turned about its own long axis (radians)

The bar's orientation follows from the class the same way the planner's
anchors do (``core.HOME_BAR_ANCHORS`` with ``core.bar_orientation_from_grasps``,
copied here because this package must not import the core): the smallest
rotation that turns the bar's right-to-left grasp direction onto the class
axis, then the roll about the bar's local z axis. Unlike the core, the grasp
midpoint is placed AFTER the roll, so the midpoint really is where the
candidate says (the grasps sit 8 cm off the bar's centre line, so a roll
about the bar frame would move it).

The same three numbers give a carry pose for any bar, so a pose found once
can be fixed and reused.

What makes a candidate good (``score_carry_pose``):

* it can be held at all: collision-free branch pairs (``expand_bar_pose``);
* many of its branch pairs are far from the joint limits and from the elbow
  and wrist singularities (``pair_margins``; counting comfortable pairs
  rather than taking the best one, because with up to 64 pairs some pair is
  nearly always comfortable -- measured on the first coarse test);
* it stays holdable when nudged by 5 cm / 15 degrees and by 10 cm / 30 degrees
  (``nudged_poses``), which also keeps it away from the robot body: a pose
  close to the body loses its nudges toward it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from husky_assembly_tamp.motion_planner.holding_prm.holding import HoldingModel, linf
from husky_assembly_tamp.motion_planner.holding_prm.sampling import WorkerTools, expand_bar_pose


# * The three classes. Bar axes as in core.HOME_BAR_ANCHORS; "today" is the
# * midpoint the planner uses now (core.py:78-117), kept for comparison.
CARRY_CLASSES: Dict[str, Dict[str, Any]] = {
    "horizontal": {"bar_axis": [0.0, 1.0, 0.0], "today": [0.3974, -0.0398, 0.8622],
                   "box": [[0.35, 0.95], [-0.3, 0.3], [0.6, 1.2]]},
    "vertical": {"bar_axis": [0.0, 0.0, 1.0], "today": [0.715, 0.0, 0.70],
                 "box": [[0.4, 1.0], [-0.3, 0.3], [0.4, 1.0]]},
    "back": {"bar_axis": [1.0, 0.0, 0.0], "today": [0.10, 0.0, 1.04],
             "box": [[-0.2, 0.4], [-0.3, 0.3], [0.8, 1.4]]},
}

# * Rolls tried, as the library's sweep does: 12 steps of 30 degrees.
ROLL_STEP_DEG = 30.0

# * Margins that count as "comfortable": half a radian (29 degrees) from any
# * joint limit, and |sin| of the elbow and wrist-2 angles at least 0.5
# * (30 degrees from straight). A margin scores min(1, margin / comfortable).
LIMIT_COMFORT_RAD = 0.5
SINGULARITY_COMFORT = 0.5

# * The nudges: shifts along each base axis and turns about each base axis
# * (about the grasp midpoint), both ways, at two sizes: 12 nudges per size.
NUDGE_SIZES = ((0.05, 15.0), (0.10, 30.0))   # (shift m, turn deg)


# -----------------------------------------------------------------------------
# * Geometry
# -----------------------------------------------------------------------------


def axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    """The 3x3 rotation by ``angle`` about a unit ``axis`` (Rodrigues' formula)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    cross = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross


def align_rotation(vector: Sequence[float], target: Sequence[float]) -> np.ndarray:
    """The smallest rotation turning ``vector`` onto ``target`` (as ``core.bar_orientation_from_grasps``).

    Args:
        vector (Sequence[float]): direction to turn.
        target (Sequence[float]): direction it should end up along.

    Returns:
        np.ndarray: 3x3 rotation. For opposite directions, 180 degrees about x
        (the core's choice).
    """
    v = np.asarray(vector, dtype=float) / np.linalg.norm(vector)
    t = np.asarray(target, dtype=float) / np.linalg.norm(target)
    cross = np.cross(v, t)
    sine, cosine = float(np.linalg.norm(cross)), float(np.dot(v, t))
    if sine < 1e-9:
        return np.eye(3) if cosine > 0 else axis_rotation([1.0, 0.0, 0.0], np.pi)
    return axis_rotation(cross / sine, float(np.arctan2(sine, cosine)))


def grasp_points(model: HoldingModel) -> Dict[str, np.ndarray]:
    """The two grasp points and their midpoint, in the bar frame."""
    left, right = model.grasp["left"][:3, 3], model.grasp["right"][:3, 3]
    return {"left": left, "right": right, "mid": 0.5 * (left + right)}


def carry_bar_pose(model: HoldingModel, carry_class: str, midpoint: Sequence[float], roll_rad: float) -> np.ndarray:
    """The bar's world pose for one carry candidate.

    Args:
        model (HoldingModel): the holding model (grasps).
        carry_class (str): a key of ``CARRY_CLASSES``.
        midpoint (Sequence[float]): where the grasp midpoint goes (base frame, m).
        roll_rad (float): turn about the bar's local z axis.

    Returns:
        np.ndarray: the 4x4 bar pose.
    """
    points = grasp_points(model)
    rotation = align_rotation(points["left"] - points["right"], CARRY_CLASSES[carry_class]["bar_axis"])
    rotation = rotation @ axis_rotation([0.0, 0.0, 1.0], roll_rad)
    pose = np.eye(4)
    pose[:3, :3] = rotation
    # The grasp midpoint lands exactly on the requested point, after the roll.
    pose[:3, 3] = np.asarray(midpoint, dtype=float) - rotation @ points["mid"]
    return pose


def nudged_poses(model: HoldingModel, pose: np.ndarray, shift_m: float, turn_deg: float) -> List[np.ndarray]:
    """The candidate shifted along and turned about each base axis, both ways (12 poses).

    Args:
        model (HoldingModel): the holding model (grasps).
        pose (np.ndarray): the candidate's 4x4 bar pose.
        shift_m (float): size of the shifts (m).
        turn_deg (float): size of the turns (deg), about the grasp midpoint.

    Returns:
        List[np.ndarray]: the 12 nudged poses.
    """
    mid_world = pose[:3, :3] @ grasp_points(model)["mid"] + pose[:3, 3]
    out = []
    for axis in np.eye(3):
        for sign in (1.0, -1.0):
            shifted = pose.copy()
            shifted[:3, 3] += sign * shift_m * axis
            out.append(shifted)
            # Turn about the grasp midpoint, so the bar stays where it is on average.
            turn = axis_rotation(axis, sign * np.radians(turn_deg))
            turned = np.eye(4)
            turned[:3, :3] = turn @ pose[:3, :3]
            turned[:3, 3] = mid_world + turn @ (pose[:3, 3] - mid_world)
            out.append(turned)
    return out


# -----------------------------------------------------------------------------
# * Scoring
# -----------------------------------------------------------------------------


def pair_margins(model: HoldingModel, conf12: Sequence[float]) -> Dict[str, float]:
    """How comfortable one branch pair is: joint-limit and singularity margins.

    Args:
        model (HoldingModel): the holding model (limits, home).
        conf12 (Sequence[float]): the twelve joint values.

    Returns:
        Dict[str, float]: ``limit_margin_deg`` (smallest distance of any joint
        to its limit), ``elbow`` and ``wrist`` (smallest |sin| of the two
        elbow and the two wrist-2 angles: 0 = straight, singular), ``home_deg``
        (largest joint difference to the home configuration) and ``comfort``
        (0..1: the worst of the three margins against its comfortable value).
    """
    q = np.asarray(conf12, dtype=float)
    limits = np.vstack([model.limits["left"], model.limits["right"]])
    limit_margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
    elbow = float(min(abs(np.sin(q[2])), abs(np.sin(q[8]))))
    wrist = float(min(abs(np.sin(q[4])), abs(np.sin(q[10]))))
    comfort = min(1.0, limit_margin / LIMIT_COMFORT_RAD, elbow / SINGULARITY_COMFORT, wrist / SINGULARITY_COMFORT)
    return {"limit_margin_deg": float(np.degrees(limit_margin)), "elbow": elbow, "wrist": wrist,
            "home_deg": float(np.degrees(linf(q, model.home_conf12))), "comfort": float(max(0.0, comfort))}


def score_carry_pose(tools: WorkerTools, pose: np.ndarray, with_nudges: bool,
                     counters: Dict[str, float]) -> Dict[str, Any]:
    """Score one carry candidate.

    Args:
        tools (WorkerTools): model and collision check.
        pose (np.ndarray): the bar's 4x4 world pose.
        with_nudges (bool): also test the 12 nudged poses (12 more solves).
        counters (Dict[str, float]): tally.

    Returns:
        Dict[str, Any]: ``n_free`` (collision-free branch pairs),
        ``n_comfortable`` (of those, pairs with comfort 1: every margin at
        least its comfortable value), and when holdable ``best`` (the most
        comfortable pair: its ``conf`` and :func:`pair_margins`); with nudges,
        ``nudge_holdable`` (how many of the 24 nudged poses can still be held)
        and ``nudge_free`` (their free-pair counts, small nudges first).
    """
    model = tools.model
    home = [("home", model.home_conf12.tolist())]
    found = expand_bar_pose(tools, pose, home, counters)
    result: Dict[str, Any] = {"n_free": len(found), "n_comfortable": 0}
    if found:
        scored = [(pair_margins(model, entry["conf"]), entry["conf"]) for entry in found]
        result["n_comfortable"] = int(sum(1 for margins, _conf in scored if margins["comfort"] >= 1.0))
        margins, conf = max(scored, key=lambda item: (item[0]["comfort"], item[0]["limit_margin_deg"]))
        result["best"] = {"conf": [float(v) for v in conf], **margins}
    if with_nudges:
        counts = [len(expand_bar_pose(tools, nudged, home, counters))
                  for shift, turn in NUDGE_SIZES for nudged in nudged_poses(model, pose, shift, turn)]
        result["nudge_holdable"] = int(sum(1 for c in counts if c > 0))
        result["nudge_free"] = [int(c) for c in counts]
    return result


def candidate_grid(step_m: float) -> List[Dict[str, Any]]:
    """Every (class, midpoint, roll) candidate on a grid over each class's box.

    Args:
        step_m (float): grid spacing of the midpoint (m).

    Returns:
        List[Dict[str, Any]]: candidates with ``carry``, ``midpoint`` and ``roll_deg``.
    """
    rolls = np.arange(-180.0, 180.0, ROLL_STEP_DEG)
    out = []
    for carry, spec in CARRY_CLASSES.items():
        axes = [np.round(np.arange(low, high + 1e-9, step_m), 4) for low, high in spec["box"]]
        for x in axes[0]:
            for y in axes[1]:
                for z in axes[2]:
                    for roll in rolls:
                        out.append({"carry": carry, "midpoint": [float(x), float(y), float(z)],
                                    "roll_deg": float(roll)})
    return out
