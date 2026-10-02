"""How well a two-arm configuration holds the bar, and the geometry behind it.

Everything here is plain numpy arithmetic on 4x4 transforms. The only thing
read from the simulator is where each arm's base sits in the world, and that
is read once, when the model is built (:func:`build_holding_model`). After
that no function in this module needs PyBullet, which is what lets the
parent process score edges and paths without opening a scene of its own.

Conventions, shared with the rest of the planner:

* a configuration (``conf12``) is twelve joint values, left arm then right,
  in the canonical UR order;
* a grasp is the pose of a tool flange (``tool0``) in the bar's own frame,
  so ``world_from_tool0 = world_from_bar @ grasp``;
* poses on disk are ``[[x, y, z], [qx, qy, qz, qw]]`` in metres.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from pybullet_planning import get_link_pose, link_from_name, tform_from_pose

from husky_assembly_tamp.keyframe.ssik_inprocess import fk, joint_limits
from husky_assembly_tamp.motion_planner.ssik_ik import ARM_BASE_LINK, ARM_SLICE, rebranch_toward_seed


# * The largest joint move allowed between two waypoints: 10 degrees on any
# * joint, measured on the raw joint values (no wrapping).
# ! Same value as core.DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD (core.py:67).
# ! It is copied rather than imported, because importing core.py empties the
# ! planner's log file (it opens it in "w" mode at import time).
STEP_MAX_RAD = 10.0 * np.pi / 180.0

# The two arms, in the order their joints appear in a 12-vector.
ARMS = ("left", "right")

# The arm that is not the given one.
OTHER_ARM = {"left": "right", "right": "left"}


# -----------------------------------------------------------------------------
# * Small pose helpers (numpy only)
# -----------------------------------------------------------------------------


def matrix_from_pose(pose: Sequence[Sequence[float]]) -> np.ndarray:
    """Turn a ``[[x, y, z], [qx, qy, qz, qw]]`` pose into a 4x4 transform.

    Args:
        pose (Sequence[Sequence[float]]): position and unit quaternion (x, y, z, w).

    Returns:
        np.ndarray: the 4x4 homogeneous transform.
    """
    (x, y, z), (qx, qy, qz, qw) = pose
    # Normalise first, so a quaternion rounded on its way through JSON still
    # gives a proper rotation.
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]
    matrix[:3, 3] = [x, y, z]
    return matrix


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return the (x, y, z, w) unit quaternion of a transform's rotation.

    Args:
        matrix (np.ndarray): a 4x4 transform (or a 3x3 rotation).

    Returns:
        np.ndarray: the quaternion, with w >= 0.
    """
    r = np.asarray(matrix)[:3, :3]
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    # Standard branch choice: divide by the largest of the four candidates so
    # the result stays accurate near 180-degree rotations.
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    q = np.asarray(q, dtype=float)
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def pose_from_matrix(matrix: np.ndarray) -> List[List[float]]:
    """Turn a 4x4 transform into a ``[[x, y, z], [qx, qy, qz, qw]]`` pose.

    Args:
        matrix (np.ndarray): the transform.

    Returns:
        List[List[float]]: position and quaternion as plain lists.
    """
    return [[float(v) for v in matrix[:3, 3]], [float(v) for v in quaternion_from_matrix(matrix)]]


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    """Invert a rigid transform without a general matrix inverse.

    Args:
        matrix (np.ndarray): a 4x4 rigid transform.

    Returns:
        np.ndarray: its inverse.
    """
    inverse = np.eye(4)
    rotation_t = matrix[:3, :3].T
    inverse[:3, :3] = rotation_t
    inverse[:3, 3] = -rotation_t @ matrix[:3, 3]
    return inverse


def pose_gap(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """How far apart two poses are: a distance and an angle.

    Args:
        a (np.ndarray): first 4x4 pose.
        b (np.ndarray): second 4x4 pose.

    Returns:
        Tuple[float, float]: the distance between the origins (metres) and the
        angle of the rotation that takes one frame to the other (radians).
    """
    distance = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    relative = a[:3, :3].T @ b[:3, :3]
    # The angle from both its cosine and its sine. The cosine alone (an
    # arc-cosine) loses most of its digits for nearly equal rotations, which
    # is exactly the case that matters here.
    cosine = (np.trace(relative) - 1.0) / 2.0
    sine = 0.5 * np.linalg.norm([relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0],
                                 relative[1, 0] - relative[0, 1]])
    angle = float(np.arctan2(sine, cosine))
    return distance, angle


# -----------------------------------------------------------------------------
# * The holding model: what the constraint needs to know about one bar
# -----------------------------------------------------------------------------


@dataclass
class HoldingModel:
    """Everything needed to score the bar-holding constraint, with no simulator.

    Attributes:
        world_from_base (Dict[str, np.ndarray]): each arm's base link in the
            world, 4x4. Read once from the scene; it does not depend on the
            arm joints.
        grasp (Dict[str, np.ndarray]): each tool flange in the bar frame, 4x4.
        grasp_inv (Dict[str, np.ndarray]): the inverse of each grasp.
        limits (Dict[str, np.ndarray]): each arm's joint limits, shape (6, 2).
        home_conf12 (np.ndarray): the authored home configuration.
        joint_names_12 (List[str]): the twelve joint names, left arm first.
    """

    world_from_base: Dict[str, np.ndarray]
    grasp: Dict[str, np.ndarray]
    grasp_inv: Dict[str, np.ndarray]
    limits: Dict[str, np.ndarray]
    home_conf12: np.ndarray
    joint_names_12: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Return the model as plain lists, ready for JSON."""
        return {
            "world_from_base": {arm: self.world_from_base[arm].tolist() for arm in ARMS},
            "grasp": {arm: self.grasp[arm].tolist() for arm in ARMS},
            "limits": {arm: self.limits[arm].tolist() for arm in ARMS},
            "home_conf12": [float(v) for v in self.home_conf12],
            "joint_names_12": list(self.joint_names_12),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HoldingModel":
        """Rebuild a model saved with :meth:`to_dict`.

        Args:
            data (Dict[str, Any]): the saved dict.

        Returns:
            HoldingModel: the model.
        """
        grasp = {arm: np.asarray(data["grasp"][arm], dtype=float) for arm in ARMS}
        return cls(
            world_from_base={arm: np.asarray(data["world_from_base"][arm], dtype=float) for arm in ARMS},
            grasp=grasp,
            grasp_inv={arm: invert_transform(grasp[arm]) for arm in ARMS},
            limits={arm: np.asarray(data["limits"][arm], dtype=float) for arm in ARMS},
            home_conf12=np.asarray(data["home_conf12"], dtype=float),
            joint_names_12=list(data["joint_names_12"]),
        )


def build_holding_model(robot_puid: int, problem) -> HoldingModel:
    """Read the arm bases from a live scene and bundle them with the grasps.

    The arm base links sit on the mobile base, which does not move while one
    bar is carried, so their world poses are read once here instead of on
    every inverse-kinematics call (``ssik_ik.target_in_arm_base`` reads them
    every time because it cannot assume that).

    Args:
        robot_puid (int): PyBullet body id of the robot (``SceneCtx.robot_puid``).
        problem: the scene's ``BarProblem`` (grasps, home, joint names).

    Returns:
        HoldingModel: the model.
    """
    world_from_base = {
        arm: tform_from_pose(get_link_pose(robot_puid, link_from_name(robot_puid, ARM_BASE_LINK[arm])))
        for arm in ARMS
    }
    grasp = {
        "left": matrix_from_pose(problem.grasp_bar_from_left),
        "right": matrix_from_pose(problem.grasp_bar_from_right),
    }
    return HoldingModel(
        world_from_base={arm: np.asarray(m, dtype=float) for arm, m in world_from_base.items()},
        grasp=grasp,
        grasp_inv={arm: invert_transform(grasp[arm]) for arm in ARMS},
        limits=arm_limits(),
        home_conf12=np.asarray(problem.home_conf12, dtype=float),
        joint_names_12=list(problem.joint_names_12),
    )


# -----------------------------------------------------------------------------
# * Forward kinematics and the violation measure
# -----------------------------------------------------------------------------


def tool0_world(model: HoldingModel, arm: str, q6: Sequence[float]) -> np.ndarray:
    """Where one arm's tool flange is in the world.

    Uses the ssik solver's own forward kinematics (about 30 microseconds),
    which is built from the calibrated robot description and agrees with
    PyBullet to about 3e-7 m.

    Args:
        model (HoldingModel): the holding model.
        arm (str): "left" or "right".
        q6 (Sequence[float]): that arm's six joint values.

    Returns:
        np.ndarray: the 4x4 world pose of the arm's ``tool0``.
    """
    return model.world_from_base[arm] @ fk(arm, np.asarray(q6, dtype=float))


def bar_from_arm(model: HoldingModel, arm: str, q6: Sequence[float]) -> np.ndarray:
    """Where the bar is, according to one arm.

    Args:
        model (HoldingModel): the holding model.
        arm (str): the arm whose flange places the bar.
        q6 (Sequence[float]): that arm's six joint values.

    Returns:
        np.ndarray: the 4x4 world pose of the bar.
    """
    return tool0_world(model, arm, q6) @ model.grasp_inv[arm]


def tool_target_from_bar(model: HoldingModel, arm: str, world_from_bar: np.ndarray) -> np.ndarray:
    """Where an arm's tool flange must be to hold the bar at a given pose.

    Args:
        model (HoldingModel): the holding model.
        arm (str): "left" or "right".
        world_from_bar (np.ndarray): the bar's 4x4 world pose.

    Returns:
        np.ndarray: the 4x4 world pose the arm's ``tool0`` must reach.
    """
    return world_from_bar @ model.grasp[arm]


def violation_from_tools(model: HoldingModel, left_tool: np.ndarray,
                         right_tool: np.ndarray) -> Tuple[float, float]:
    """The holding violation, given both flange poses.

    The bar placed by the left flange asks the right flange to be at one pose;
    the bar placed by the right flange asks the left flange to be at another.
    Both gaps are measured and the larger kept, so the answer does not depend
    on which arm is taken as the reference. (The two angles are always equal;
    the two distances differ by the lever arm between the grasps.)

    Args:
        model (HoldingModel): the holding model.
        left_tool (np.ndarray): left ``tool0`` world pose.
        right_tool (np.ndarray): right ``tool0`` world pose.

    Returns:
        Tuple[float, float]: position error (m) and rotation error (rad).
    """
    expected_right = left_tool @ model.grasp_inv["left"] @ model.grasp["right"]
    expected_left = right_tool @ model.grasp_inv["right"] @ model.grasp["left"]
    pos_a, rot_a = pose_gap(expected_right, right_tool)
    pos_b, rot_b = pose_gap(expected_left, left_tool)
    return max(pos_a, pos_b), max(rot_a, rot_b)


def holding_violation(model: HoldingModel, conf12: Sequence[float]) -> Tuple[float, float]:
    """How badly a configuration fails to hold the bar rigidly.

    Zero (to about 1e-7) for any configuration that came out of the planner's
    inverse kinematics; grows as the two flanges drift apart from the fixed
    relation the grasps prescribe.

    Args:
        model (HoldingModel): the holding model.
        conf12 (Sequence[float]): the twelve joint values.

    Returns:
        Tuple[float, float]: position error (m) and rotation error (rad).
    """
    q = np.asarray(conf12, dtype=float)
    return violation_from_tools(model, tool0_world(model, "left", q[:6]),
                                tool0_world(model, "right", q[6:]))


def node_geometry(model: HoldingModel, conf12: Sequence[float]) -> Dict[str, Any]:
    """Everything the roadmap stores about one configuration's geometry.

    Args:
        model (HoldingModel): the holding model.
        conf12 (Sequence[float]): the twelve joint values.

    Returns:
        Dict[str, Any]: ``tool_left`` and ``tool_right`` (flange positions),
        ``ee_mid`` (the midpoint between them: where the dashboard draws the
        node), ``bar_pos`` and ``bar_quat`` (the bar as the LEFT flange places
        it, the same convention as ``SceneCtx.bar_pose_at``), and ``viol_pos``
        / ``viol_rot`` (the holding violation).
    """
    q = np.asarray(conf12, dtype=float)
    left = tool0_world(model, "left", q[:6])
    right = tool0_world(model, "right", q[6:])
    bar = left @ model.grasp_inv["left"]
    viol_pos, viol_rot = violation_from_tools(model, left, right)
    return {
        "tool_left": left[:3, 3].copy(),
        "tool_right": right[:3, 3].copy(),
        "ee_mid": 0.5 * (left[:3, 3] + right[:3, 3]),
        "bar_pos": bar[:3, 3].copy(),
        "bar_quat": quaternion_from_matrix(bar),
        "viol_pos": viol_pos,
        "viol_rot": viol_rot,
    }


def geometry_to_lists(geometry: Dict[str, Any]) -> Dict[str, Any]:
    """Turn the arrays of :func:`node_geometry` into plain lists, for JSON and for other processes.

    Args:
        geometry (Dict[str, Any]): what :func:`node_geometry` returned.

    Returns:
        Dict[str, Any]: the same keys, with lists and floats only.
    """
    return {key: (value.tolist() if isinstance(value, np.ndarray) else float(value))
            for key, value in geometry.items()}


# -----------------------------------------------------------------------------
# * Joint-space straight lines and 2*pi turns
# -----------------------------------------------------------------------------


def interpolate_confs(a: Sequence[float], b: Sequence[float], step_max_rad: float = STEP_MAX_RAD,
                      include_ends: bool = False) -> np.ndarray:
    """Evenly spaced configurations on the straight joint-space line a -> b.

    The number of pieces is chosen so no joint moves more than
    ``step_max_rad`` between neighbouring configurations. Joint differences
    are raw (no wrapping), the same rule the production continuity gate uses
    (``core.joint_step_deltas``), so every step of such a line would pass it.

    Args:
        a (Sequence[float]): start configuration.
        b (Sequence[float]): end configuration.
        step_max_rad (float): largest joint move per piece.
        include_ends (bool): also return ``a`` and ``b`` themselves.

    Returns:
        np.ndarray: shape (m, 12). Without the ends, ``m`` is the number of
        pieces minus one, and can be zero for a short line.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    largest_move = float(np.max(np.abs(b - a)))
    pieces = max(1, int(np.ceil(largest_move / step_max_rad - 1e-12)))
    fractions = np.linspace(0.0, 1.0, pieces + 1)
    if not include_ends:
        fractions = fractions[1:-1]
    return a[None, :] + fractions[:, None] * (b - a)[None, :]


def arm_limits() -> Dict[str, np.ndarray]:
    """Both arms' joint limits, (6, 2) each, straight from the ssik artifacts (no scene needed)."""
    return {arm: np.asarray(joint_limits(arm), dtype=float) for arm in ARMS}


def rebranch12(conf12: Sequence[float], reference12: Sequence[float],
               limits: Dict[str, np.ndarray]) -> np.ndarray:
    """Move every joint by whole turns so it sits nearest the reference.

    A UR joint whose range is wider than one turn can reach the same angle in
    two (shoulder pan: three) ways, 2*pi apart. The roadmap measures raw joint
    differences, so two such "twins" look a full turn apart. This picks, joint
    by joint, the twin nearest ``reference12`` among those inside the limits
    (``ssik_ik.rebranch_toward_seed`` does the work).

    Args:
        conf12 (Sequence[float]): the configuration to shift.
        reference12 (Sequence[float]): the configuration to move toward.
        limits (Dict[str, np.ndarray]): each arm's (6, 2) joint limits, e.g.
            ``model.limits`` or :func:`arm_limits`.

    Returns:
        np.ndarray: the shifted copy (same physical pose, same holding).
    """
    q = np.array(conf12, dtype=float)
    ref = np.asarray(reference12, dtype=float)
    for arm in ARMS:
        part = ARM_SLICE[arm]
        q[part] = rebranch_toward_seed(q[part], ref[part], limits[arm])
    return q


def linf(a: Sequence[float], b: Sequence[float]) -> float:
    """The largest raw joint difference between two configurations."""
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def l2(a: Sequence[float], b: Sequence[float]) -> float:
    """The Euclidean length of the raw joint difference between two configurations."""
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def violation_score_mm(viol_pos_m: float, viol_rot_rad: float, rot_deg_per_mm: float) -> float:
    """Fold a position and a rotation error into one number, in millimetres.

    A tolerance is given as one number ``t`` (mm); a configuration passes it
    when the position error is at most ``t`` mm AND the rotation error is at
    most ``t * rot_deg_per_mm`` degrees. The score is the smallest ``t`` that
    passes, so "score <= t" and "passes tolerance t" mean the same thing.

    Args:
        viol_pos_m (float): position error, metres.
        viol_rot_rad (float): rotation error, radians.
        rot_deg_per_mm (float): how many degrees of rotation error count as
            one millimetre (default 0.4: 2 degrees per 5 mm, the pair named in
            the pilot plan).

    Returns:
        float: the score, millimetres.
    """
    return max(viol_pos_m * 1000.0, np.degrees(viol_rot_rad) / rot_deg_per_mm)


# -----------------------------------------------------------------------------
# * Bar poses between two configurations (used by the exact tracker)
# -----------------------------------------------------------------------------

# How a "target" bar pose is read from a configuration that may not hold the
# bar exactly: from the left flange, from the right flange, or halfway between.
BAR_LEADS = ("mid", "left", "right")


def slerp_quaternion(qa: np.ndarray, qb: np.ndarray, t: float) -> np.ndarray:
    """Turn a fraction ``t`` of the way from one rotation to another, the short way round.

    Args:
        qa (np.ndarray): start quaternion (x, y, z, w).
        qb (np.ndarray): end quaternion (x, y, z, w).
        t (float): 0 gives ``qa``, 1 gives ``qb``.

    Returns:
        np.ndarray: the unit quaternion in between.
    """
    qa = np.asarray(qa, dtype=float)
    qb = np.asarray(qb, dtype=float)
    dot = float(np.dot(qa, qb))
    # q and -q are the same rotation; flip one so the turn is the short one.
    if dot < 0.0:
        qb, dot = -qb, -dot
    if dot > 1.0 - 1e-12:
        # Nearly the same rotation: a straight blend is exact enough and avoids
        # dividing by a vanishing sine.
        q = qa + t * (qb - qa)
        return q / np.linalg.norm(q)
    angle = np.arccos(min(1.0, dot))
    q = (np.sin((1.0 - t) * angle) * qa + np.sin(t * angle) * qb) / np.sin(angle)
    return q / np.linalg.norm(q)


def interpolate_pose(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """A pose a fraction ``t`` of the way from ``a`` to ``b``.

    The origin moves on the straight line and the rotation turns about one
    fixed axis (the same rule as ``pybullet_planning.interpolate_poses``, which
    the BiRRT uses).

    Args:
        a (np.ndarray): start 4x4 pose.
        b (np.ndarray): end 4x4 pose.
        t (float): fraction, 0..1.

    Returns:
        np.ndarray: the 4x4 pose in between.
    """
    out = np.eye(4)
    out[:3, 3] = (1.0 - t) * a[:3, 3] + t * b[:3, 3]
    out[:3, :3] = matrix_from_pose([[0.0, 0.0, 0.0],
                                    slerp_quaternion(quaternion_from_matrix(a), quaternion_from_matrix(b), t)])[:3, :3]
    return out


def grasp_midpoint_offset(model: HoldingModel) -> np.ndarray:
    """A shift from the bar's own frame to the point halfway between the two grasps.

    The bar frame of the export sits at one end of the bar, so turning the bar
    about its own origin swings one grasp much more than the other. Blending
    and interpolating bar poses about the grasp midpoint instead keeps the two
    arms' share of any correction about equal.

    Args:
        model (HoldingModel): the holding model (grasps).

    Returns:
        np.ndarray: 4x4 pure translation, expressed in the bar frame.
    """
    offset = np.eye(4)
    offset[:3, 3] = 0.5 * (model.grasp["left"][:3, 3] + model.grasp["right"][:3, 3])
    return offset


def target_bar_pose(model: HoldingModel, conf12: Sequence[float], lead: str = "mid") -> np.ndarray:
    """The bar pose a configuration "means", even when it does not hold the bar exactly.

    At a configuration off the holding surface the two flanges place the bar
    at two slightly different poses. ``lead`` picks which one to aim for:
    ``"left"`` or ``"right"`` (that arm's flange decides, the other arm makes
    the whole correction) or ``"mid"`` (halfway between the two, blended about
    the grasp midpoint, so each arm corrects about half the error). For a
    configuration that holds the bar, all three are the same pose.

    Args:
        model (HoldingModel): the holding model.
        conf12 (Sequence[float]): the twelve joint values.
        lead (str): one of ``BAR_LEADS``.

    Returns:
        np.ndarray: the 4x4 world pose of the bar.
    """
    q = np.asarray(conf12, dtype=float)
    if lead == "left":
        return bar_from_arm(model, "left", q[:6])
    if lead == "right":
        return bar_from_arm(model, "right", q[6:])
    if lead != "mid":
        raise ValueError(f"Unknown bar lead {lead!r}; expected one of {BAR_LEADS}.")
    offset = grasp_midpoint_offset(model)
    from_left = bar_from_arm(model, "left", q[:6]) @ offset
    from_right = bar_from_arm(model, "right", q[6:]) @ offset
    return interpolate_pose(from_left, from_right, 0.5) @ invert_transform(offset)
