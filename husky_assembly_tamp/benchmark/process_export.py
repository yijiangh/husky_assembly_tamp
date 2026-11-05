"""Reading a schema 2.1 process folder for the benchmark, the roadmap planner and the check script.

Rhino's ``RSExportProcess`` writes a *process folder*: ``process.json``, one
``actions/<action>.json`` per scheduled action, the robots as URDF files and
the meshes as OBJ files (the format is described in
``scaffolding_assembly_core/doc/process_format.md``). The shared core library
``scaffolding_assembly_core`` reads such a folder into a ``Process``, turns
any movement into a ``Scene`` (``process.scene_at``) and builds a compas_fab
cell for one robot out of a scene (``mirrors.cell.CompasFabCell``).

This module sits between the core and the planners. It takes a process
folder and hands out what they work with:

* :func:`trimmed_problem`: the ``BarProblem``, the trimmed cell and the
  template state of one bar, for ``build-cells``;
* :func:`carry_scene`: the world at the start of a bar's carry, for the online
  plan and for checking a saved path;
* :func:`carry_goal_frames`: the planned base pose and the goal pose of the
  left flange, for the roadmap's goal region;
* :func:`process_source_key`: the trimmed-scene cache key of a process folder.

! Nothing here imports the core at module level: the benchmark library
! experiments need no process folder, so they run without the core installed.
! :func:`require_core` says how to install it when a process folder is given.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from husky_assembly_tamp.benchmark.scene import BarProblem, mesh_extents
from husky_assembly_tamp.motion_planner.api import TOOL_LINK_LEFT, TOOL_LINK_RIGHT, _arm_joint_names


# * The file that marks a folder as a process folder (schema 2.1).
PROCESS_FILE = "process.json"

# * Action id codes of the format: "<bar>_J" joints a bar, "<bar>_R" releases it.
JOINTING_CODE = "J"
RELEASE_CODE = "R"


# -----------------------------------------------------------------------------
# * Finding and reading a process folder
# -----------------------------------------------------------------------------


def is_process_folder(problem_dir: str) -> bool:
    """Whether a folder is a schema 2.1 process folder (it holds ``process.json``).

    Args:
        problem_dir (str): the folder to look at.

    Returns:
        bool: True for a process folder, False for an old export or anything else.
    """
    return os.path.isfile(os.path.join(problem_dir, PROCESS_FILE))


def require_core() -> None:
    """Make sure the shared core library can be imported.

    Raises:
        ImportError: with install instructions when it cannot. The core is a
            sibling checkout ``external/scaffolding_assembly_core`` when this
            repo is vendored in the design workflow, or a pip install of that
            repository on a stand-alone machine.
    """
    try:
        import scaffolding_assembly_core  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Reading a process folder needs the shared core library "
            "'scaffolding_assembly_core'. Clone git@github.com:yijiangh/scaffolding_assembly_core.git "
            "next to this repo as external/scaffolding_assembly_core, or run "
            "`pip install -e <that clone>` into this environment. Underlying error: " + str(exc)
        ) from exc


def default_plans_dir(problem_dir: str, root: Optional[str]) -> str:
    """Where the carry plans of a process folder go when no ``--save-dir`` is given.

    Args:
        problem_dir (str): the process folder.
        root (Optional[str]): the benchmark data root (``HUSKY_BENCH_ROOT``), or None.

    Returns:
        str: ``<root>/prm/plans/<process folder name>``; without a root,
        ``<process folder>/solutions/plans`` (the core never reads that subfolder).
    """
    name = os.path.basename(os.path.normpath(problem_dir))
    if root:
        return os.path.join(root, "prm", "plans", name)
    return os.path.join(problem_dir, "solutions", "plans")


def read_process(problem_dir: str):
    """Read and validate a process folder.

    Args:
        problem_dir (str): the process folder (holds ``process.json``).

    Returns:
        Process: the process, with every file checked by the core's reader.
    """
    require_core()
    from scaffolding_assembly_core.process import read

    return read(problem_dir)


def process_source_key(problem_dir: str) -> str:
    """The trimmed-scene cache key of a process folder.

    The key is the folder name plus eight hex digits of a hash over the
    content hashes of ``process.json`` and every action file. The core's
    content hash leaves the ``writer`` and ``producer`` stamps out, so
    re-exporting an unchanged design gives the same key, while a moved bar, a
    new grasp or a new robot file gives another one (``process.json`` records
    the hash of each robot's files and names every mesh by its content).

    Args:
        problem_dir (str): the process folder.

    Returns:
        str: the key, e.g. ``"261008_2ModulesDemo_purged_process-3f2a1b9c"``.
    """
    require_core()
    from scaffolding_assembly_core.process.write import content_hash

    folder = Path(problem_dir)
    digest = hashlib.sha256()
    digest.update(content_hash(folder / PROCESS_FILE).encode("utf-8"))
    for path in sorted((folder / "actions").glob("*.json")):
        digest.update(path.name.encode("utf-8"))
        digest.update(content_hash(path).encode("utf-8"))
    return f"{os.path.basename(os.path.normpath(problem_dir))}-{digest.hexdigest()[:8]}"


# -----------------------------------------------------------------------------
# * Ids, actions and movements
# -----------------------------------------------------------------------------


def short_id(bar: str) -> str:
    """The bar name without its prefix: ``"bars/B21"`` -> ``"B21"`` (a bare name is returned as it is)."""
    return bar.split("/", 1)[1] if "/" in bar else bar


def action_id(bar_id: str, code: str) -> str:
    """The id of a bar's action: ``action_id("B21", "J")`` -> ``"B21_J"``."""
    return f"{short_id(bar_id)}_{code}"


def process_bar_ids(process) -> List[str]:
    """The bars a process has both a jointing and a release action for, in schedule order.

    Args:
        process: the process.

    Returns:
        List[str]: short bar ids, e.g. ``["B20", "B21", ...]``.
    """
    bars = []
    for name in process.schedule:
        action = process.actions[name]
        if action.type == "bar_jointing" and action_id(action.bar, RELEASE_CODE) in process.actions:
            bars.append(short_id(action.bar))
    return bars


def jointing_action(process, bar_id: str):
    """A bar's jointing action (``<bar>_J``).

    Args:
        process: the process.
        bar_id (str): the bar, e.g. ``"B21"``.

    Returns:
        Action: the action.

    Raises:
        KeyError: when the process has no jointing action for the bar.
    """
    name = action_id(bar_id, JOINTING_CODE)
    if name not in process.actions:
        raise KeyError(f"{process.folder}: no action {name!r}; the bars with a jointing action are "
                       f"{', '.join(process_bar_ids(process)) or 'none'}")
    return process.actions[name]


def carry_movement(action):
    """The bar carry of a jointing action: its one free, coupled arm movement.

    The format stores no movement roles; what a movement does is in its typed
    fields. The carry (``<bar>_J_transfer``) is the only movement in which both
    arms move freely while keeping their relative pose (``coupled``).

    Args:
        action: the jointing action.

    Returns:
        Movement: the carry.

    Raises:
        ValueError: when the action has no such movement, or more than one.
    """
    found = [m for m in action.movements if m.arms and m.path == "free" and m.coupled]
    if len(found) != 1:
        raise ValueError(f"{action.id}: expected one free, coupled arm movement (the bar carry), found "
                         f"{len(found)}: {[m.id for m in found]}")
    return found[0]


def home_joints(process, bar_id: str, robot: str) -> Dict[str, float]:
    """The authored home of a bar: where the release action's last free arm movement ends.

    Args:
        process: the process.
        bar_id (str): the bar.
        robot (str): the acting robot, e.g. ``"robots/cindy"``.

    Returns:
        Dict[str, float]: joint name -> value.

    Raises:
        ValueError: when the bar has no release action, or none of its free movements names target joints.
    """
    name = action_id(bar_id, RELEASE_CODE)
    release = process.actions.get(name)
    if release is None:
        raise ValueError(f"{bar_id}: no release action {name!r}; the home pose of this export lives there")
    homes = [m for m in release.movements
             if m.arms and m.path == "free" and m.target is not None and m.target.joints.get(robot)]
    if not homes:
        raise ValueError(f"{name}: no free arm movement with target joints for {robot} (the home pose)")
    return dict(homes[-1].target.joints[robot])


def flange_links(movement) -> Tuple[str, str]:
    """The (left, right) flange link names of a dual-arm movement.

    Args:
        movement: a movement whose ``arms`` name two flange links.

    Returns:
        Tuple[str, str]: the left and the right link name, e.g.
        ``("left_ur_arm_tool0", "right_ur_arm_tool0")``.

    Raises:
        ValueError: when the movement does not move exactly those two flanges.
    """
    require_core()
    from scaffolding_assembly_core.ids import split_link_id

    links = {split_link_id(arm)[1] for arm in movement.arms}
    if links != {TOOL_LINK_LEFT, TOOL_LINK_RIGHT}:
        raise ValueError(f"{movement.id}: expected the arms {TOOL_LINK_LEFT!r} and {TOOL_LINK_RIGHT!r}, "
                         f"found {sorted(links)}")
    return TOOL_LINK_LEFT, TOOL_LINK_RIGHT


def holders(movement, bar: str) -> Dict[str, Any]:
    """The two grasps of a carried bar, keyed ``"left"`` and ``"right"``.

    A holder's ``grasp`` is the bar's pose in the holding flange's frame, so
    the planner's grasp (the flange in the bar's frame) is its inverse.

    Args:
        movement: the carry; its start state lists the bar's holders.
        bar (str): the bar id, e.g. ``"bars/B21"``.

    Returns:
        Dict[str, Holder]: the left and the right holder.

    Raises:
        ValueError: when the bar is not held by both flanges at the start.
    """
    found = {}
    for holder in movement.start.attached.get(bar, ()):
        if holder.to.endswith("/" + TOOL_LINK_LEFT):
            found["left"] = holder
        elif holder.to.endswith("/" + TOOL_LINK_RIGHT):
            found["right"] = holder
    if set(found) != {"left", "right"}:
        raise ValueError(f"{movement.id}: {bar} must be held by both {TOOL_LINK_LEFT} and {TOOL_LINK_RIGHT} "
                         f"at the start of the carry; the holders are "
                         f"{[h.to for h in movement.start.attached.get(bar, ())]}")
    return found


def goal_link_poses(movement, robot: str) -> Dict[str, Any]:
    """Where the two flanges must end, keyed ``"left"`` and ``"right"``, as core poses in the world.

    Args:
        movement: the carry.
        robot (str): the acting robot.

    Returns:
        Dict[str, Pose]: the two goal poses.

    Raises:
        ValueError: when the target names no pose for one of the flanges.
    """
    if movement.target is None:
        raise ValueError(f"{movement.id}: no target")
    poses = movement.target.link_poses
    out = {}
    for side, link in (("left", TOOL_LINK_LEFT), ("right", TOOL_LINK_RIGHT)):
        key = f"{robot}/{link}"
        if key not in poses:
            raise ValueError(f"{movement.id}: the target gives no pose for {key}")
        out[side] = poses[key]
    return out


def goal_joints(movement, robot: str, joint_names_12: Sequence[str]) -> List[float]:
    """The authored posture at the end of the carry, in the planner's joint order.

    Args:
        movement: the carry.
        robot (str): the acting robot.
        joint_names_12 (Sequence[str]): the twelve arm joints, left six then right six.

    Returns:
        List[float]: the twelve values.

    Raises:
        ValueError: when the target gives no joints (the keyframes were not solved in Rhino).
    """
    joints = movement.target.joints.get(robot) if movement.target is not None else None
    if not joints:
        raise ValueError(f"{movement.id}: the target gives no joints for {robot} (keyframes not solved)")
    missing = [name for name in joint_names_12 if name not in joints]
    if missing:
        raise ValueError(f"{movement.id}: the target joints lack {missing}")
    return [float(joints[name]) for name in joint_names_12]


# -----------------------------------------------------------------------------
# * Poses
# -----------------------------------------------------------------------------


def pose_lists(pose) -> List[List[float]]:
    """A core pose as the PyBullet-style ``[[x, y, z], [qx, qy, qz, qw]]`` lists the benchmark stores."""
    return [[float(v) for v in pose.position], [float(v) for v in pose.orientation]]


def pose_from_lists(data: Sequence[Sequence[float]]):
    """A core pose from ``[[x, y, z], [qx, qy, qz, qw]]`` (or a PyBullet ``(position, quaternion)`` pose)."""
    require_core()
    from scaffolding_assembly_core.geometry import Pose

    position, quaternion = data
    return Pose.from_arrays(position, quaternion)


def shifted_pose(base, dx: float, dy: float, dyaw_rad: float):
    """A base pose moved on the ground in its own frame: shifted along its x and y, turned about its z.

    Args:
        base (Pose): the planned base pose.
        dx (float): shift along the base's x axis (m).
        dy (float): shift along the base's y axis (m).
        dyaw_rad (float): turn about the base's z axis (rad).

    Returns:
        Pose: the moved base pose.
    """
    require_core()
    from scaffolding_assembly_core.geometry import Pose, compose

    half = 0.5 * float(dyaw_rad)
    return compose(base, Pose((float(dx), float(dy), 0.0), (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))))


def refresh_world_poses(scene) -> None:
    """Recompute every body's world pose after a robot or a placement was changed in place.

    Args:
        scene: the scene, changed in place.
    """
    from scaffolding_assembly_core.kinematics import link_pose
    from scaffolding_assembly_core.scene import world_poses

    scene.world_poses = world_poses(scene.bodies, scene.robots,
                                    lambda robot, link: link_pose(robot.model.urdf, robot.base, robot.joints, link))


# -----------------------------------------------------------------------------
# * Scenes of one bar's carry
# -----------------------------------------------------------------------------


def attach_to_left(scene, process, action, movement) -> None:
    """Make the carried bar and its halves ride on the LEFT flange, whichever holder the file lists first.

    The core attaches a held bar to its first holder. The roadmap backend and
    the benchmark read the grasp from the bar's attachment to the left tool
    link, so a file that lists the right holder first is re-attached here with
    the left grasp; the halves follow by their mount offset.

    Args:
        scene: the scene of the carry's start, changed in place.
        process: the process.
        action: the jointing action.
        movement: the carry.
    """
    from scaffolding_assembly_core.geometry import compose
    from scaffolding_assembly_core.process.relations import mount_offset
    from scaffolding_assembly_core.scene import Attachment

    left_link, _right_link = flange_links(movement)
    bar = scene.bodies[action.bar]
    if isinstance(bar.placement, Attachment) and bar.placement.link == left_link:
        return
    grasp = holders(movement, action.bar)["left"].grasp
    bar.placement = Attachment(action.robot, left_link, grasp)
    for half in process.halves_of(action.bar):
        scene.bodies[half].placement = Attachment(action.robot, left_link,
                                                  compose(grasp, mount_offset(process, half)))
    refresh_world_poses(scene)


def carry_scene(process, bar_id: str, *, joints: Optional[Dict[str, float]] = None, base=None):
    """The world at the start of a bar's carry, ready for the acting robot to plan in.

    The process leaves the carry's start joints open (the planner derives the
    start), so the robot is posed at ``joints``, or at the bar's authored home
    when none are given. Other robots keep the state the process gives them;
    the core's mirror parks the ones whose base or joints are unknown.

    Args:
        process: the process.
        bar_id (str): the bar, e.g. ``"B21"``.
        joints (Optional[Dict[str, float]]): values for the acting robot's
            joints the process leaves open (its known joints keep their value).
        base (Optional[Pose]): a base pose to use instead of the planned one
            (a parking error, see :func:`shifted_pose`).

    Returns:
        Tuple[Scene, Action, Movement]: the scene, the jointing action and the carry.
    """
    require_core()
    from scaffolding_assembly_core.process.scenes import seeded

    action = jointing_action(process, bar_id)
    movement = carry_movement(action)
    robot = action.robot
    scene = process.scene_at(movement)
    if base is not None:
        scene.robots[robot].base = base
        scene.robots[robot].base_tracked = True
    # ? `seeded` fills in the open joints and recomputes the world poses.
    seeded(scene, robot, dict(joints) if joints else home_joints(process, bar_id, robot))
    attach_to_left(scene, process, action, movement)
    return scene, action, movement


def trimmed_scene(scene, action):
    """A copy of a carry scene holding only the acting robot and what rides on it, with the base at the origin.

    This is the roadmap's precompute scene: the robot, its two tools, the bar
    and the halves mounted on it. Allowances toward bodies that are no longer
    in the scene are dropped, so the record stays honest.

    Args:
        scene: the scene of the carry's start (see :func:`carry_scene`); not changed.
        action: the jointing action.

    Returns:
        Scene: the small scene.
    """
    from scaffolding_assembly_core.geometry import Pose
    from scaffolding_assembly_core.scene import Attachment

    robot = action.robot
    out = scene.copy()
    for other in [name for name in out.robots if name != robot]:
        del out.robots[other]
    carried = {name for name, body in out.bodies.items()
               if body.enabled and isinstance(body.placement, Attachment) and body.placement.parent == robot}
    for name in [name for name in out.bodies if name not in carried]:
        out.remove(name)
    tools = {tool.id for tool in out.robots[robot].model.tools.values()}
    for name in carried:
        body = out.bodies[name]
        body.touches = tuple(other for other in body.touches
                             if other in carried or other in tools or other == robot or other.startswith(robot + "/"))
    # ! Robot-centric by design: the planned base pose is discarded.
    out.robots[robot].base = Pose()
    refresh_world_poses(out)
    return out


def trimmed_problem(process, bar_id: str):
    """One bar of a process as the benchmark caches it: its problem, trimmed cell and template state.

    The cell is built by the core's ``CompasFabCell`` from the trimmed carry
    scene, so the attachments and the allowed contacts are exactly what the
    online planner sees.

    Args:
        process: the process.
        bar_id (str): the bar, e.g. ``"B21"``.

    Returns:
        Tuple[BarProblem, RobotCell, RobotCellState]: the trio ``build-cells`` writes.
    """
    require_core()
    from scaffolding_assembly_core.geometry import invert
    from scaffolding_assembly_core.mirrors.cell import CompasFabCell

    scene, action, movement = carry_scene(process, bar_id)
    robot = action.robot
    small = trimmed_scene(scene, action)

    # * Planning cells carry no visual shapes (as in production); the robot
    # * asset for the dashboard falls back to the collision shapes.
    builder = CompasFabCell(robot, visual=False)
    builder.update(small)
    cell = builder.cell
    template = builder.state(small)

    left_names, right_names = _arm_joint_names(cell)
    joint_names_12 = list(left_names) + list(right_names)
    home = home_joints(process, bar_id, robot)
    home_conf12 = [float(home[name]) for name in joint_names_12]
    for name, value in zip(joint_names_12, home_conf12):
        template.robot_configuration[name] = value
    cell.assert_cell_state_match(template)

    held = holders(movement, action.bar)
    left_link, _right_link = flange_links(movement)
    lower, upper = mesh_extents(cell.rigid_body_models[action.bar])
    size = upper - lower
    problem = BarProblem(
        bar_id=bar_id,
        bar_rb_name=action.bar,
        joint_rb_names=sorted(name for name in small.bodies if name != action.bar),
        bar_length_m=float(size.max()),
        bar_radius_m=float(0.5 * float(np.sort(size)[1])),
        bar_from_midpoint=[float(v) for v in 0.5 * (lower + upper)],
        attached_link=left_link,
        # The authored attachment: the bar in the left flange's frame.
        tool0_from_bar=pose_lists(held["left"].grasp),
        # The planner's grasps: each flange in the bar's frame.
        grasp_bar_from_left=pose_lists(invert(held["left"].grasp)),
        grasp_bar_from_right=pose_lists(invert(held["right"].grasp)),
        home_conf12=home_conf12,
        joint_names_12=joint_names_12,
        source_action=f"{action.id}.json",
    )
    return problem, cell, template


def carry_goal_frames(process, bar_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """Where the design puts the robot and the left flange at the end of a bar's carry.

    Args:
        process: the process.
        bar_id (str): the bar.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 4x4 transforms ``world_from_base`` (the
        planned base pose) and ``world_from_left_tool0`` (the goal pose of the
        left flange).

    Raises:
        ValueError: when the process does not place the base or give the flange its goal pose.
    """
    action = jointing_action(process, bar_id)
    movement = carry_movement(action)
    start = movement.start.robots.get(action.robot)
    if start is None or start.base is None:
        raise ValueError(f"{movement.id}: the process does not place {action.robot}'s base at the carry")
    goal = goal_link_poses(movement, action.robot)
    return start.base.matrix(), goal["left"].matrix()
