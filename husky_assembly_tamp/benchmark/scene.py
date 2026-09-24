"""Build the small, robot-centric planning scene each benchmark problem uses.

The exported design data describes a whole construction site: one robot cell
with a hundred rigid bodies (every bar and every joint half of the structure)
and four tool models. The benchmark does not want any of that. It asks a much
narrower question -- how well does the planner carry ONE bar around an
otherwise empty robot -- so the scene is trimmed to exactly:

    the robot, the two scaffolding tools, the bar being carried, and the two
    joint halves screwed onto its ends

Everything else is dropped. This matters twice over: loading the full cell into
PyBullet is slow, and every collision check clones the cell state, so the fewer
bodies the state carries the faster every single check runs.

What is NOT dropped is the collision bookkeeping. The allowed-collision
information lives in three places, and all three are carried over verbatim from
the bar action's M1 movement so the benchmark collides exactly like production:

    1. ``robot_semantics.disabled_collisions``  robot link pairs that may touch
    2. each body's ``touch_links``              robot links that body may touch
    3. each body's ``touch_bodies``             other bodies it may touch

The one thing deliberately thrown away is the authored robot base frame. The
benchmark is robot-centric: the robot sits at the world origin and goal poses
are sampled around it, so a base frame from the design file would only add an
irrelevant offset.

Units: the planner works in metres, so everything here is metres.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet_planning as pp
from compas.data import json_dump, json_load
from compas.geometry import Frame, Transformation
from compas_fab.backends import PyBulletClient, PyBulletPlanner
from compas_fab.robots import RobotCell, RobotCellState

from compas_fab.backends import CollisionCheckError

from husky_assembly_tamp.motion_planner.api import (
    TOOL_LINK_LEFT,
    TOOL_LINK_RIGHT,
    _arm_joint_names,
    _build_cfab_collision_fn,
    _pp_pose_from_frame,
)


# * The two scaffolding tools the assembly robot carries. The support robots
# * (ObstacleRobotAlice / ObstacleRobotBelle) are modelled as tools too, but
# * they are parked 50 m away in the exports and are irrelevant here, so the
# * trimmed cell keeps only these two.
KEPT_TOOL_IDS = ("AT3L", "AT3R")

# * Which movement of a "__J" (jointing) bar action is the M1 transfer. The
# * export writes a fixed six-movement cycle and index 3 is the constrained
# * dual-arm free movement -- the one this benchmark is about.
CDFM_MOVEMENT_INDEX = 3

# * Which movement of the matching "__R" (release) action carries the authored
# * home configuration in its target_configuration. Index 3 is "free_home".
HOME_MOVEMENT_INDEX = 3

# * The planner's own logger. The benchmark listens to it (that is the only
# * place the search reports its two tree sizes) but does not want its output.
PLANNER_LOGGER_NAME = "dual_arm_task_space_rrt_core"


def quiet_planner_logging() -> None:
    """Stop the planner from writing to the console and to its log file.

    Two reasons, both practical rather than cosmetic:

    The console handler writes to the stdout that existed when the planner was
    imported. Opening a PyBullet client replaces that stream, and every
    subsequent log call then fails with "I/O operation on closed file" and
    prints a stack trace of its own -- pages of noise around real output.

    The file handler opens one fixed path in truncate mode. With a dozen worker
    processes they all truncate and interleave into the same file, which is
    worse than having no log at all.

    Records still reach handlers the benchmark attaches itself, so nothing that
    is actually read is lost.
    """
    logger = logging.getLogger(PLANNER_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # noqa: BLE001 - closing a broken handler may throw
            pass
    # ! A logger with no handlers at all is not silent: Python falls back to
    # ! its own last-resort handler and prints every warning to stderr. During
    # ! a start survey that is thousands of lines about candidates that did not
    # ! work, which is exactly what this function exists to stop. A null
    # ! handler counts as handling them, so the fallback never fires.
    logger.addHandler(logging.NullHandler())
    logger.propagate = False


@dataclass
class BarProblem:
    """Everything about one bar-and-grasp that the benchmark needs.

    This is the durable description of a benchmark problem: it survives to
    JSON, is small, and is enough (together with the trimmed cell) to pose the
    robot anywhere. Poses are PyBullet style, ``[[x, y, z], [qx, qy, qz, qw]]``,
    in metres.

    Attributes:
        bar_id (str): the design-side bar name, e.g. ``"B3"``.
        bar_rb_name (str): the rigid-body name in the cell, e.g. ``"bar_B3"``.
        joint_rb_names (List[str]): the two joint halves attached to the tools.
        bar_length_m (float): length of the bar, from its collision mesh.
        bar_radius_m (float): radius of the bar, from its collision mesh.
        bar_from_midpoint (List[float]): the bar's midpoint in bar-local
            coordinates. The bar frame sits at one end, so this is needed to
            place a bar by its middle.
        attached_link (str): the robot link the bar is attached to.
        tool0_from_bar (List[List[float]]): the grasp, as the bar's pose in the
            attached link's frame. Copied from the action's attachment frame.
        grasp_bar_from_left (List[List[float]]): left tool0's pose in the bar
            frame -- what the planner calls the left grasp.
        grasp_bar_from_right (List[List[float]]): the same for the right tool0.
        home_conf12 (List[float]): the authored dual-arm home configuration.
        joint_names_12 (List[str]): the twelve arm joints, left six then right.
        source_action (str): file name of the bar action this came from.
    """

    bar_id: str
    bar_rb_name: str
    joint_rb_names: List[str]
    bar_length_m: float
    bar_radius_m: float
    bar_from_midpoint: List[float]
    attached_link: str
    tool0_from_bar: List[List[float]]
    grasp_bar_from_left: List[List[float]]
    grasp_bar_from_right: List[List[float]]
    home_conf12: List[float]
    joint_names_12: List[str]
    source_action: str

    def to_dict(self) -> Dict[str, Any]:
        """Return this problem as a plain JSON-ready dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BarProblem":
        """Rebuild a problem from :meth:`to_dict` output.

        Args:
            data (Dict[str, Any]): the dict a previous ``to_dict`` produced.

        Returns:
            BarProblem: the reconstructed problem.
        """
        return cls(**data)


def pose_to_lists(pose) -> List[List[float]]:
    """Turn a PyBullet ``(position, quaternion)`` pose into plain lists.

    Args:
        pose: a ``((x, y, z), (qx, qy, qz, qw))`` pose.

    Returns:
        List[List[float]]: ``[[x, y, z], [qx, qy, qz, qw]]``, ready for JSON.
    """
    position, quaternion = pose
    return [[float(v) for v in position], [float(v) for v in quaternion]]


def pose_from_lists(data: Sequence[Sequence[float]]):
    """Turn ``[[x, y, z], [qx, qy, qz, qw]]`` back into a PyBullet pose.

    Args:
        data (Sequence[Sequence[float]]): the two lists to convert.

    Returns:
        tuple: the ``((x, y, z), (qx, qy, qz, qw))`` pose.
    """
    position, quaternion = data
    return (tuple(float(v) for v in position), tuple(float(v) for v in quaternion))


def select_cdfm_movement(action):
    """Return the M1 constrained dual-arm free movement of a jointing action.

    Args:
        action: a ``BarAssemblyJointingAction`` loaded from a ``*__J.json``.

    Returns:
        The movement object at the fixed CDFM index.

    Raises:
        ValueError: if the action does not have the expected movement there.
            The benchmark refuses to guess -- an export with a different shape
            needs looking at, not a fallback.
    """
    movements = list(action.movements)
    if len(movements) <= CDFM_MOVEMENT_INDEX:
        raise ValueError(
            f"{action.action_id}: expected at least {CDFM_MOVEMENT_INDEX + 1} "
            f"movements, found {len(movements)}. This export does not have the "
            "six-movement jointing shape the benchmark expects."
        )
    movement = movements[CDFM_MOVEMENT_INDEX]
    kind = type(movement).__name__
    if kind != "EndEffectorConstrainedDualArmFreeMovement":
        raise ValueError(
            f"{action.action_id}: movement {CDFM_MOVEMENT_INDEX} is a {kind}, "
            "not the EndEffectorConstrainedDualArmFreeMovement (M1) this "
            "benchmark measures."
        )
    return movement


def mesh_extents(rigid_body) -> Tuple[np.ndarray, np.ndarray]:
    """Return the axis-aligned bounds of a rigid body's collision mesh.

    Args:
        rigid_body: a compas_fab ``RigidBody``.

    Returns:
        Tuple[np.ndarray, np.ndarray]: the ``(minimum, maximum)`` corner, in
        the body's own frame and in metres.

    Raises:
        ValueError: if the body carries no collision mesh.
    """
    meshes = list(rigid_body.collision_meshes or [])
    if not meshes:
        raise ValueError("rigid body has no collision mesh to measure")
    points = []
    for mesh in meshes:
        points.extend(mesh.vertices_attributes("xyz"))
    vertices = np.asarray(points, dtype=float) * float(rigid_body.native_scale)
    return vertices.min(axis=0), vertices.max(axis=0)


def extract_bar_problem(action, release_action, rcell) -> BarProblem:
    """Read one bar-and-grasp out of a pair of exported bar actions.

    The jointing action gives the bar, its two joint halves, the grasp and the
    two tool0 target frames; the matching release action gives the authored
    home configuration. Nothing here touches PyBullet -- it is pure reading.

    The right-hand grasp deserves a note. The export records only ONE
    attachment (the bar hangs off the left tool0), because the two tools hold
    the bar rigidly and the second grasp is implied. It is recovered from the
    two authored tool0 target frames: the transform from the left flange to the
    right one is constant for the whole movement, so composing it with the left
    grasp gives the right one.

    Args:
        action: the ``*__J.json`` jointing action.
        release_action: the matching ``*__R.json`` release action.
        rcell (RobotCell): the full robot cell (for the bar's mesh and the
            joint names).

    Returns:
        BarProblem: the extracted problem.

    Raises:
        ValueError: if the action does not hold the bar in both tools, or the
            release action carries no home configuration.
    """
    movement = select_cdfm_movement(action)
    state = movement.start_state

    # Exactly the bodies that ride on the robot: the bar plus its joint halves.
    attached = {
        name: rb_state
        for name, rb_state in (state.rigid_body_states or {}).items()
        if rb_state.attached_to_link
    }
    bar_rb_name = f"bar_{action.active_bar_id}"
    if bar_rb_name not in attached:
        raise ValueError(
            f"{action.action_id}: {bar_rb_name} is not attached to a robot link "
            "in the M1 start state, so there is no grasp to benchmark."
        )
    joint_rb_names = sorted(name for name in attached if name != bar_rb_name)

    bar_state = attached[bar_rb_name]
    if bar_state.attachment_frame is None:
        raise ValueError(f"{action.action_id}: {bar_rb_name} has no attachment frame.")

    # The grasp, as the planner wants it: the two tool0 poses in the bar frame.
    # tool0_from_bar is the authored attachment; inverting it gives the left
    # grasp, and the constant flange-to-flange transform gives the right one.
    tool0_from_bar = _pp_pose_from_frame(bar_state.attachment_frame)
    grasp_bar_from_left = pp.invert(tool0_from_bar)
    ee_frames = movement.target_ee_frames or {}
    if "left" not in ee_frames or "right" not in ee_frames:
        raise ValueError(
            f"{action.action_id}: M1 has no left/right target_ee_frames, so the "
            "right-hand grasp cannot be recovered."
        )
    world_from_tool0_left = _pp_pose_from_frame(ee_frames["left"])
    world_from_tool0_right = _pp_pose_from_frame(ee_frames["right"])
    left_from_right = pp.multiply(pp.invert(world_from_tool0_left), world_from_tool0_right)
    grasp_bar_from_right = pp.multiply(grasp_bar_from_left, left_from_right)

    # Bar geometry lives only in the mesh; the action never states a length.
    lower, upper = mesh_extents(rcell.rigid_body_models[bar_rb_name])
    size = upper - lower
    bar_length_m = float(size.max())
    bar_radius_m = float(0.5 * float(np.sort(size)[1]))
    bar_from_midpoint = [float(v) for v in 0.5 * (lower + upper)]

    # The planner's own joint ordering: the left arm's six, then the right's.
    left_joint_names, right_joint_names = _arm_joint_names(rcell)
    joint_names_12 = list(left_joint_names) + list(right_joint_names)

    home_movement = list(release_action.movements)[HOME_MOVEMENT_INDEX]
    home_target = getattr(home_movement, "target_configuration", None)
    if home_target is None:
        raise ValueError(
            f"{release_action.action_id}: movement {HOME_MOVEMENT_INDEX} has no "
            "target_configuration, so the authored dual-arm home is missing."
        )
    home_conf12 = [float(home_target[name]) for name in joint_names_12]

    return BarProblem(
        bar_id=action.active_bar_id,
        bar_rb_name=bar_rb_name,
        joint_rb_names=joint_rb_names,
        bar_length_m=bar_length_m,
        bar_radius_m=bar_radius_m,
        bar_from_midpoint=bar_from_midpoint,
        attached_link=bar_state.attached_to_link,
        tool0_from_bar=pose_to_lists(tool0_from_bar),
        grasp_bar_from_left=pose_to_lists(grasp_bar_from_left),
        grasp_bar_from_right=pose_to_lists(grasp_bar_from_right),
        home_conf12=home_conf12,
        joint_names_12=list(joint_names_12),
        source_action=os.path.basename(getattr(action, "_source_path", "")) or action.action_id,
    )


def build_trimmed_cell(rcell, problem: BarProblem) -> RobotCell:
    """Return a robot cell holding only what this benchmark problem needs.

    The robot model and semantics are shared by reference with the source cell
    (they are large and never modified); only the tool and rigid-body
    dictionaries are rebuilt, keeping the two scaffolding tools and the three
    carried bodies.

    Args:
        rcell (RobotCell): the full cell loaded from ``RobotCell.json``.
        problem (BarProblem): which bar and joints to keep.

    Returns:
        RobotCell: the trimmed cell.

    Raises:
        ValueError: if the source cell is missing one of the kept bodies.
    """
    kept_bodies = [problem.bar_rb_name] + list(problem.joint_rb_names)
    missing = [name for name in kept_bodies if name not in rcell.rigid_body_models]
    missing += [name for name in KEPT_TOOL_IDS if name not in rcell.tool_models]
    if missing:
        raise ValueError(
            f"RobotCell.json is missing {missing}. It does not match the bar "
            "action being trimmed -- check that both come from the same export."
        )
    return RobotCell(
        robot_model=rcell.robot_model,
        robot_semantics=rcell.robot_semantics,
        tool_models={name: rcell.tool_models[name] for name in KEPT_TOOL_IDS},
        rigid_body_models={name: rcell.rigid_body_models[name] for name in kept_bodies},
    )


def build_template_state(action, problem: BarProblem, trimmed_cell: RobotCell) -> RobotCellState:
    """Return the cell state every benchmark run starts from.

    This is the M1 start state with the scene trimmed to match the trimmed
    cell, the robot moved to the world origin, and the arm configuration filled
    in (the export leaves it empty for the planner to derive). All the
    allowed-collision bookkeeping -- the tools' and bodies' touch lists and the
    attachment frames -- is carried over untouched, except that references to
    dropped bodies are removed so the record stays honest.

    Args:
        action: the jointing action the state comes from.
        problem (BarProblem): which bodies survive the trim.
        trimmed_cell (RobotCell): the cell this state must match.

    Returns:
        RobotCellState: the template. Callers copy it and write a
        configuration into the copy; see :func:`state_with_conf12`.
    """
    source_state = select_cdfm_movement(action).start_state
    kept_bodies = set([problem.bar_rb_name] + list(problem.joint_rb_names))

    rigid_body_states = {}
    for name in sorted(kept_bodies):
        rb_state = source_state.rigid_body_states[name].copy()
        # Drop allowances that point at bodies no longer in the scene, so the
        # state cannot claim a body may touch something that does not exist.
        rb_state.touch_bodies = [
            other
            for other in (rb_state.touch_bodies or [])
            if other in kept_bodies or other in KEPT_TOOL_IDS
        ]
        rigid_body_states[name] = rb_state

    tool_states = {name: source_state.tool_states[name].copy() for name in KEPT_TOOL_IDS}

    state = RobotCellState(
        # ! Robot-centric by design: the authored base frame is discarded.
        robot_base_frame=Frame.worldXY(),
        robot_configuration=trimmed_cell.zero_full_configuration(),
        tool_states=tool_states,
        rigid_body_states=rigid_body_states,
    )
    # Start the arms at the authored home rather than at all-zeros, which is a
    # self-colliding pose for this robot.
    for name, value in zip(problem.joint_names_12, problem.home_conf12):
        state.robot_configuration[name] = float(value)
    trimmed_cell.assert_cell_state_match(state)
    return state


def state_with_conf12(
    template: RobotCellState, conf12: Sequence[float], joint_names_12: Sequence[str],
) -> RobotCellState:
    """Return a copy of ``template`` with the twelve arm joints set.

    Args:
        template (RobotCellState): the state to copy.
        conf12 (Sequence[float]): twelve joint values, left arm then right.
        joint_names_12 (Sequence[str]): the matching joint names.

    Returns:
        RobotCellState: the copy.
    """
    state = template.copy()
    for name, value in zip(joint_names_12, conf12):
        state.robot_configuration[name] = float(value)
    return state


class SceneCtx:
    """A live PyBullet scene for one bar problem, plus the ids planners want.

    Holding a scene open is expensive (loading the robot and its meshes takes
    seconds), so one of these is built per bar and reused for every goal,
    start and planning run against that bar.

    Attributes:
        client (PyBulletClient): the compas_fab client owning the simulation.
        planner (PyBulletPlanner): the planner bound to that client.
        cell (RobotCell): the trimmed cell.
        template_state (RobotCellState): the state runs start from.
        problem (BarProblem): the bar and grasp being benchmarked.
        robot_puid (int): PyBullet body id of the robot.
        arm_joints (List[int]): PyBullet joint indices of the twelve arm joints.
        tool_link_left (int): left tool0 link index.
        tool_link_right (int): right tool0 link index.
        bar_body (int): PyBullet body id of the carried bar.
    """

    def __init__(self, client, planner, cell, template_state, problem: BarProblem):
        """Wire up a scene context. Use :func:`open_scene` instead of this."""
        self.client = client
        self.planner = planner
        self.cell = cell
        self.template_state = template_state
        self.problem = problem

        self.robot_puid = client.robot_puid
        self.arm_joints = pp.joints_from_names(self.robot_puid, problem.joint_names_12)
        self.tool_link_left = pp.link_from_name(self.robot_puid, TOOL_LINK_LEFT)
        self.tool_link_right = pp.link_from_name(self.robot_puid, TOOL_LINK_RIGHT)
        self.bar_body = client.rigid_bodies_puids[problem.bar_rb_name][0]

    @property
    def grasp_bar_from_left(self):
        """The left grasp as a PyBullet pose."""
        return pose_from_lists(self.problem.grasp_bar_from_left)

    @property
    def grasp_bar_from_right(self):
        """The right grasp as a PyBullet pose."""
        return pose_from_lists(self.problem.grasp_bar_from_right)

    def collision_fn(self, template_state: Optional[RobotCellState] = None):
        """Return the production collision predicate for this scene.

        This is exactly what ``plan_constrained_dual_arm`` uses internally, so
        a configuration the benchmark calls collision-free is one the planner
        would also accept.

        Args:
            template_state (Optional[RobotCellState]): the state to vary; the
                scene's own template by default.

        Returns:
            Callable[[Sequence[float]], bool]: ``conf12 -> True when colliding``.
        """
        state = self.template_state if template_state is None else template_state
        return _build_cfab_collision_fn(self.planner, state, self.problem.joint_names_12)

    def reporting_collision_fn(self, template_state: Optional[RobotCellState] = None):
        """Return a collision predicate that also remembers what collided.

        Same verdicts as :meth:`collision_fn` -- it is the same compas_fab
        check -- but it asks for the full report and hangs the offending body
        pairs off the function as ``last_collision_pairs``. The endpoint IK
        picks them up from there when it is asked for a report.

        Args:
            template_state (Optional[RobotCellState]): the state to vary; the
                scene's own template by default.

        Returns:
            Callable[[Sequence[float]], bool]: ``conf12 -> True when colliding``,
            with a ``last_collision_pairs`` attribute.
        """
        base_state = self.template_state if template_state is None else template_state
        joint_names_12 = self.problem.joint_names_12
        planner = self.planner
        options = {"full_report": True, "verbose": False}

        def _fn(conf12, *_args, **_kwargs):
            """Return True when this configuration collides, recording the pairs."""
            state = state_with_conf12(base_state, conf12, joint_names_12)
            try:
                planner.check_collision(state, options=options)
            except CollisionCheckError as exc:
                _fn.last_collision_pairs = list(getattr(exc, "collision_pairs", []) or [])
                return True
            _fn.last_collision_pairs = []
            return False

        _fn.last_collision_pairs = []
        return _fn

    def set_state(self, state: RobotCellState) -> None:
        """Push a cell state into the simulation.

        Args:
            state (RobotCellState): the state to apply.
        """
        self.planner.set_robot_cell_state(state)

    def fk_link_pose(self, conf12: Sequence[float], link_index: int):
        """Return a link's world pose with the arms at the given configuration.

        The live world is left untouched.

        Args:
            conf12 (Sequence[float]): twelve joint values.
            link_index (int): the PyBullet link index to read.

        Returns:
            tuple: the link's ``(position, quaternion)`` world pose.
        """
        with pp.WorldSaver():
            pp.set_joint_positions(self.robot_puid, self.arm_joints, list(conf12))
            return pp.get_link_pose(self.robot_puid, link_index)

    def bar_pose_at(self, conf12: Sequence[float]):
        """Return where the carried bar sits for a given arm configuration.

        Args:
            conf12 (Sequence[float]): twelve joint values.

        Returns:
            tuple: the bar's ``(position, quaternion)`` world pose.
        """
        world_from_tool0_left = self.fk_link_pose(conf12, self.tool_link_left)
        return pp.multiply(world_from_tool0_left, pp.invert(self.grasp_bar_from_left))

    def close(self) -> None:
        """Disconnect the simulation."""
        try:
            self.client.__exit__(None, None, None)
        except Exception:  # noqa: BLE001 - teardown must never mask a result
            pp.disconnect()


def open_scene(cell: RobotCell, template_state: RobotCellState, problem: BarProblem,
               *, use_gui: bool = False) -> SceneCtx:
    """Load a trimmed cell into PyBullet and return a context for it.

    Args:
        cell (RobotCell): the trimmed cell.
        template_state (RobotCellState): the state to apply once loaded.
        problem (BarProblem): the bar and grasp this scene serves.
        use_gui (bool): open a PyBullet window instead of running headless.

    Returns:
        SceneCtx: the live scene.
    """
    # ! Before the client replaces stdout: see quiet_planner_logging.
    quiet_planner_logging()
    client = PyBulletClient(
        connection_type="gui" if use_gui else "direct",
        verbose=False,
        enable_debug_gui=use_gui,
    )
    client.__enter__()
    # pybullet_planning keeps its own notion of the active client; point it at
    # the one compas_fab just made so both libraries talk to the same world.
    pp.set_client(client.client_id)
    pp.CLIENTS[client.client_id] = True if use_gui else None

    planner = PyBulletPlanner(client)
    with pp.LockRenderer(False):
        planner.set_robot_cell(cell)
        planner.set_robot_cell_state(template_state)
    return SceneCtx(client, planner, cell, template_state, problem)


# ---------------------------------------------------------------------------
# Caching trimmed scenes on disk
# ---------------------------------------------------------------------------


def cell_cache_dir(root: str, source_key: str) -> str:
    """Return the folder holding trimmed cells for one source export.

    Args:
        root (str): the benchmark data root.
        source_key (str): a short key identifying the source export.

    Returns:
        str: the folder path (not created).
    """
    return os.path.join(root, "cells", source_key)


def save_trimmed_problem(root: str, source_key: str, problem: BarProblem,
                         cell: RobotCell, template_state: RobotCellState) -> str:
    """Write one trimmed problem to the cache.

    The cell and state go through compas' JSON encoder (they contain meshes and
    frames); the problem description is written beside them as plain JSON so it
    can be read without compas.

    Args:
        root (str): the benchmark data root.
        source_key (str): a short key identifying the source export.
        problem (BarProblem): the problem being cached.
        cell (RobotCell): its trimmed cell.
        template_state (RobotCellState): its template state.

    Returns:
        str: the folder the three files were written to.
    """
    folder = os.path.join(cell_cache_dir(root, source_key), problem.bar_id)
    os.makedirs(folder, exist_ok=True)
    json_dump(cell, os.path.join(folder, "cell.json"))
    json_dump(template_state, os.path.join(folder, "template_state.json"))
    with open(os.path.join(folder, "problem.json"), "w", encoding="utf-8") as handle:
        json.dump(problem.to_dict(), handle, indent=2)
    return folder


def load_trimmed_problem(root: str, source_key: str, bar_id: str):
    """Read one trimmed problem back from the cache.

    Args:
        root (str): the benchmark data root.
        source_key (str): a short key identifying the source export.
        bar_id (str): which bar to load.

    Returns:
        Tuple[BarProblem, RobotCell, RobotCellState]: the cached trio.

    Raises:
        FileNotFoundError: if this bar has not been cached. The benchmark does
            not silently rebuild it -- that would hide a stale or missing
            source export -- so the message says which stage to run.
    """
    folder = os.path.join(cell_cache_dir(root, source_key), bar_id)
    problem_path = os.path.join(folder, "problem.json")
    if not os.path.isfile(problem_path):
        raise FileNotFoundError(
            f"No trimmed scene for {bar_id} at {folder}. Run the 'build-cells' "
            "stage first."
        )
    with open(problem_path, "r", encoding="utf-8") as handle:
        problem = BarProblem.from_dict(json.load(handle))
    cell = json_load(os.path.join(folder, "cell.json"))
    template_state = json_load(os.path.join(folder, "template_state.json"))
    return problem, cell, template_state
