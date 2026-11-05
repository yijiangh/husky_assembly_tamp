"""The small, robot-centric planning scene each benchmark problem uses, and its cache on disk.

An exported process describes a whole construction site: every bar and every
joint half of the structure, the ground, three robots. The benchmark does not
want any of that. It asks a much narrower question -- how well does the planner
carry ONE bar around an otherwise empty robot -- so the scene is trimmed to:

    the robot, its two scaffolding tools, the bar being carried, and the joint
    halves mounted on it

(``process_export.trimmed_problem`` does the trimming, with the core's
``CompasFabCell``). This matters twice over: loading everything into PyBullet
is slow, and every collision check clones the cell state, so the fewer bodies
the state carries the faster every single check runs.

What is NOT dropped is the collision bookkeeping: the robot's SRDF, and each
body's allowed contacts (``touch_links``, ``touch_bodies``), derived by the
core exactly as the online planner gets them.

The one thing deliberately thrown away is the planned robot base pose. The
benchmark is robot-centric: the robot sits at the world origin and goal poses
are sampled around it, so a base pose from the design would only add an
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
from compas_fab.backends import CollisionCheckError, PyBulletClient, PyBulletPlanner
from compas_fab.robots import RobotCell, RobotCellState

from husky_assembly_tamp.motion_planner.api import (
    TOOL_LINK_LEFT,
    TOOL_LINK_RIGHT,
    _build_cfab_collision_fn,
)


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
        bar_rb_name (str): the rigid-body name in the cell, e.g. ``"bars/B3"``.
        joint_rb_names (List[str]): the joint halves mounted on the bar, riding with it.
        bar_length_m (float): length of the bar, from its collision mesh.
        bar_radius_m (float): radius of the bar, from its collision mesh.
        bar_from_midpoint (List[float]): the bar's midpoint in bar-local
            coordinates. The bar frame sits at one end, so this is needed to
            place a bar by its middle.
        attached_link (str): the robot link the bar is attached to.
        tool0_from_bar (List[List[float]]): the grasp, as the bar's pose in the
            attached link's frame (the left holder's grasp in the process).
        grasp_bar_from_left (List[List[float]]): left tool0's pose in the bar
            frame -- what the planner calls the left grasp.
        grasp_bar_from_right (List[List[float]]): the same for the right tool0.
        home_conf12 (List[float]): the authored dual-arm home configuration.
        joint_names_12 (List[str]): the twelve arm joints, left six then right.
        source_action (str): file name of the jointing action this came from.
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
