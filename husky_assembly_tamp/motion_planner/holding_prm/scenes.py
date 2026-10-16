"""Turning an exported bar action into the scenes the roadmap planner needs.

Two scenes matter for one bar-carry movement:

* the **precompute scene**: the robot, its two tools and the carried bodies,
  with the robot at the world origin. ``benchmark build-cells`` makes it and
  caches it under ``<root>/cells/<source_key>/<bar>/`` (see
  ``benchmark/scene.py``); the roadmap is built in it.
* the **real scene**: the same, plus everything visible and stationary in the
  movement's start state (built bars and joints, a support robot), with the
  robot at its real base pose. The online planner checks routes in it.

A full exported cell holds every body of the whole assembly (hundreds of
megabytes); a worker process must not load that. :func:`build_scene_cell`
keeps only what a movement's scene actually shows, and the pair it returns is
small enough to write to disk for each worker to open.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from compas.data import json_dump, json_load
from compas.geometry import Frame, Transformation
from compas_fab.robots import RobotCell, RobotCellState

from husky_assembly_tamp.benchmark.build_cells import source_key_for
from husky_assembly_tamp.benchmark.glb import GlbBuilder, frame_to_translation_rotation, mesh_arrays
from husky_assembly_tamp.benchmark.scene import (
    HOME_MOVEMENT_INDEX,
    BarProblem,
    select_cdfm_movement,
)
from husky_assembly_tamp.motion_planner.api import _pp_pose_from_frame
from husky_assembly_tamp.motion_planner.holding_prm.holding import invert_transform, matrix_from_pose
from husky_assembly_tamp.motion_planner.holding_prm.sampling import is_carried


# -----------------------------------------------------------------------------
# * Reading an export
# -----------------------------------------------------------------------------


def action_paths(problem_dir: str, bar_id: str) -> Tuple[str, str]:
    """The jointing and release action files of one bar in an export.

    Args:
        problem_dir (str): the exported problem folder.
        bar_id (str): the bar, e.g. ``"B3"``.

    Returns:
        Tuple[str, str]: ``<bar>__J.json`` and ``<bar>__R.json``.

    Raises:
        FileNotFoundError: when either is missing.
    """
    folder = os.path.join(problem_dir, "BarActions")
    paths = (os.path.join(folder, f"{bar_id}__J.json"), os.path.join(folder, f"{bar_id}__R.json"))
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No bar action {path}. Check the problem folder and the bar id.")
    return paths


def load_carry_movement(problem_dir: str, bar_id: str):
    """The bar-carry movement of one bar, and the authored home configuration.

    Args:
        problem_dir (str): the exported problem folder.
        bar_id (str): the bar.

    Returns:
        Tuple: the jointing action, its bar-carry movement (movement
        ``CDFM_MOVEMENT_INDEX``, an ``EndEffectorConstrainedDualArmFreeMovement``)
        and the home target configuration read from the release action.

    Raises:
        ValueError: when the release action carries no home configuration.
    """
    jointing_path, release_path = action_paths(problem_dir, bar_id)
    action = json_load(jointing_path)
    release = json_load(release_path)
    movement = select_cdfm_movement(action)
    home = getattr(list(release.movements)[HOME_MOVEMENT_INDEX], "target_configuration", None)
    if home is None:
        raise ValueError(f"{release_path}: movement {HOME_MOVEMENT_INDEX} has no target_configuration (the home).")
    return action, movement, home


def frame_matrix(frame) -> np.ndarray:
    """A compas ``Frame`` as a 4x4 transform."""
    position, quaternion = _pp_pose_from_frame(frame)
    return matrix_from_pose([list(position), list(quaternion)])


def planned_goal(movement, problem: BarProblem) -> Dict[str, np.ndarray]:
    """Where the design puts the bar at the end of the carry, in the world and as seen from the robot.

    Args:
        movement: the bar-carry movement (its ``target_ee_frames`` and the
            base pose in its ``start_state``).
        problem (BarProblem): the bar's grasps.

    Returns:
        Dict[str, np.ndarray]: 4x4 transforms ``world_from_base`` (the planned
        base pose), ``world_from_bar`` and ``base_from_bar`` (the goal as seen
        from the robot: what a roadmap in the robot's frame works with).
    """
    world_from_base = frame_matrix(movement.start_state.robot_base_frame)
    world_from_left = frame_matrix(movement.target_ee_frames["left"])
    world_from_bar = world_from_left @ invert_transform(matrix_from_pose(problem.grasp_bar_from_left))
    return {"world_from_base": world_from_base, "world_from_bar": world_from_bar,
            "base_from_bar": invert_transform(world_from_base) @ world_from_bar}


def shifted_base(world_from_base: np.ndarray, dx: float, dy: float, dyaw: float) -> np.ndarray:
    """A base pose moved on the ground: shifted in its own x/y and turned about its own z.

    Args:
        world_from_base (np.ndarray): the base pose, 4x4.
        dx (float): shift along the base's x axis (m).
        dy (float): shift along the base's y axis (m).
        dyaw (float): turn about the base's z axis (rad).

    Returns:
        np.ndarray: the moved base pose, 4x4.
    """
    move = np.eye(4)
    move[:3, :3] = [[np.cos(dyaw), -np.sin(dyaw), 0.0], [np.sin(dyaw), np.cos(dyaw), 0.0], [0.0, 0.0, 1.0]]
    move[:3, 3] = [dx, dy, 0.0]
    return world_from_base @ move


# -----------------------------------------------------------------------------
# * The real scene of one movement, small enough for a worker
# -----------------------------------------------------------------------------


def scene_body_names(state: RobotCellState) -> Dict[str, List[str]]:
    """Which bodies and tools a state actually shows.

    Args:
        state (RobotCellState): a movement's start state.

    Returns:
        Dict[str, List[str]]: ``carried`` (rigid bodies riding on the robot),
        ``stationary`` (visible rigid bodies that do not), ``arm_tools`` and
        ``stationary_tools`` (visible tools not on the robot, e.g. a support robot).
    """
    bodies = state.rigid_body_states or {}
    tools = state.tool_states or {}
    return {
        "carried": sorted(n for n, b in bodies.items() if is_carried(b) and not b.is_hidden),
        "stationary": sorted(n for n, b in bodies.items() if not is_carried(b) and not b.is_hidden),
        "arm_tools": sorted(n for n, t in tools.items() if t.attached_to_group),
        "stationary_tools": sorted(n for n, t in tools.items() if not t.attached_to_group and not t.is_hidden),
    }


def build_scene_cell(rcell: RobotCell, state: RobotCellState) -> Tuple[RobotCell, RobotCellState]:
    """Cut a cell and a state down to what the state shows.

    Hidden bodies and hidden tools are dropped; everything visible is kept
    with its state untouched (base pose, attachments, allowances). The robot
    model and semantics are shared with the source cell.

    Args:
        rcell (RobotCell): the full cell.
        state (RobotCellState): the movement's start state (its configuration
            must be filled in).

    Returns:
        Tuple[RobotCell, RobotCellState]: the small cell and its matching state.
    """
    names = scene_body_names(state)
    kept_bodies = names["carried"] + names["stationary"]
    kept_tools = names["arm_tools"] + names["stationary_tools"]
    cell = RobotCell(
        robot_model=rcell.robot_model,
        robot_semantics=rcell.robot_semantics,
        tool_models={name: rcell.tool_models[name] for name in kept_tools},
        rigid_body_models={name: rcell.rigid_body_models[name] for name in kept_bodies},
    )
    visible = set(kept_bodies) | set(kept_tools)
    body_states = {}
    for name in kept_bodies:
        body_state = state.rigid_body_states[name].copy()
        # An allowance toward a body that is not in the scene means nothing here.
        body_state.touch_bodies = [other for other in (body_state.touch_bodies or []) if other in visible]
        body_states[name] = body_state
    small = RobotCellState(
        robot_base_frame=state.robot_base_frame.copy(),
        robot_configuration=state.robot_configuration.copy(),
        tool_states={name: state.tool_states[name].copy() for name in kept_tools},
        rigid_body_states=body_states,
    )
    cell.assert_cell_state_match(small)
    return cell, small


def shape_print(body_model) -> List[Any]:
    """A cheap summary of a rigid body's collision shape, to notice when it changed.

    Args:
        body_model: a compas_fab ``RigidBody``.

    Returns:
        List[Any]: per collision mesh its vertex count, face count, and the
        sum and the sum of absolute values of its vertex coordinates (rounded).
        A reshaped or resized body gives other numbers; hashing the whole
        mesh would cost as much as writing the scene.
    """
    out = []
    for mesh in body_model.collision_meshes or []:
        points = np.asarray([mesh.vertex_coordinates(key) for key in mesh.vertices()], dtype=float).reshape(-1, 3)
        out.append([len(points), mesh.number_of_faces(),
                    [round(float(v), 4) for v in points.sum(axis=0)],
                    [round(float(v), 4) for v in np.abs(points).sum(axis=0)]])
    return out


def short_hash(parts: Any) -> str:
    """A 16-character hash of plain data (lists, numbers, names)."""
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def geometry_key(cell: RobotCell) -> str:
    """A short key for what a scene cell is made of: which bodies and tools, and the bodies' shapes.

    Where things stand is not part of it, so the same structure seen from
    another base pose has the same key (and its cell file is reused).

    Args:
        cell (RobotCell): the (small) scene cell.

    Returns:
        str: a hash.
    """
    bodies = [[name, shape_print(cell.rigid_body_models[name])] for name in sorted(cell.rigid_body_models)]
    return short_hash([bodies, sorted(cell.tool_models)])


def state_key(state: RobotCellState, bar_rb_name: str) -> str:
    """A short key for where everything is: the base pose, each body and tool's place and allowances.

    Args:
        state (RobotCellState): the (small) scene state.
        bar_rb_name (str): the carried bar.

    Returns:
        str: a hash; the arm joint values are not part of it.
    """
    parts: List[Any] = [bar_rb_name, [round(float(v), 6) for v in list(state.robot_base_frame.point)
                                      + list(state.robot_base_frame.quaternion)]]
    for name in sorted(state.rigid_body_states):
        body = state.rigid_body_states[name]
        frame = body.attachment_frame if is_carried(body) else body.frame
        parts.append([name, body.attached_to_link, body.attached_to_tool,
                      [round(float(v), 6) for v in list(frame.point) + list(frame.quaternion)],
                      sorted(body.touch_links or []), sorted(body.touch_bodies or [])])
    for name in sorted(state.tool_states):
        tool = state.tool_states[name]
        frame = tool.frame
        parts.append([name, tool.attached_to_group, sorted(tool.touch_links or []),
                      None if frame is None else [round(float(v), 6) for v in list(frame.point) + list(frame.quaternion)],
                      None if tool.configuration is None else [round(float(v), 6) for v in tool.configuration.joint_values]])
    return short_hash(parts)


def save_scene(cache_dir: str, cell: RobotCell, state: RobotCellState, bar_rb_name: str) -> Tuple[str, str, str]:
    """Write a scene for worker processes to open.

    The cell (robot model, meshes: the large part) is written once per
    geometry and reused; the state (where things stand: small) is written for
    every base pose and scene arrangement.

    Args:
        cache_dir (str): the folder that holds cached scenes.
        cell (RobotCell): the small cell.
        state (RobotCellState): its state.
        bar_rb_name (str): the carried bar.

    Returns:
        Tuple[str, str, str]: the cell file, the state file, and the scene's
        key (``<geometry key>/<state key>``).
    """
    geometry, placement = geometry_key(cell), state_key(state, bar_rb_name)
    folder = os.path.join(cache_dir, geometry)
    os.makedirs(folder, exist_ok=True)
    cell_path = os.path.join(folder, "cell.json")
    state_path = os.path.join(folder, f"state_{placement}.json")
    # Written under another name first, so a worker never opens half a file
    # (the process id keeps two planners from sharing that name).
    if not os.path.isfile(cell_path):
        temporary = f"{cell_path}.{os.getpid()}.tmp"
        json_dump(cell, temporary)
        os.replace(temporary, cell_path)
    temporary = f"{state_path}.{os.getpid()}.tmp"
    json_dump(state, temporary)
    os.replace(temporary, state_path)
    return cell_path, state_path, f"{geometry}/{placement}"


def filled_start_state(state: RobotCellState, rcell: RobotCell, joint_names_12: Sequence[str],
                       conf12: Sequence[float]) -> RobotCellState:
    """A copy of a start state with a full robot configuration (the export leaves it empty).

    Args:
        state (RobotCellState): the exported start state.
        rcell (RobotCell): the cell (for the robot's full joint list).
        joint_names_12 (Sequence[str]): the twelve arm joints.
        conf12 (Sequence[float]): their values (e.g. home).

    Returns:
        RobotCellState: the copy.
    """
    filled = state.copy()
    if filled.robot_configuration is None:
        filled.robot_configuration = rcell.zero_full_configuration()
    for name, value in zip(joint_names_12, conf12):
        filled.robot_configuration[name] = float(value)
    return filled


def problem_source_key(problem_dir: str) -> str:
    """The trimmed-scene cache key of an export (see ``benchmark.build_cells``)."""
    return source_key_for(problem_dir)


# -----------------------------------------------------------------------------
# * The scene as a 3D file for the roadmap page
# -----------------------------------------------------------------------------

# Colours (red, green, blue, alpha) of what stands around the robot. The robot
# itself and the carried bar come from the robot asset (benchmark/export_assets).
SCENE_COLOURS: Dict[str, Tuple[float, float, float, float]] = {
    "built_bar": (0.55, 0.58, 0.62, 1.0),
    "joint": (0.85, 0.72, 0.25, 1.0),
    "carried_joint": (0.93, 0.55, 0.20, 1.0),
    "support_robot": (0.42, 0.44, 0.48, 1.0),
}


def as_transformation(frame) -> Transformation:
    """A compas frame as a transformation (the identity when the frame is missing)."""
    return Transformation() if frame is None else Transformation.from_frame(frame)


def export_scene_glb(cell: RobotCell, state: RobotCellState, bar_rb_name: str, out_path: str) -> Dict[str, Any]:
    """Write what stands around the robot, and what rides on its arms, as one glb file.

    Everything is written in the robot's base frame, because the roadmap page
    draws the robot (and the roadmap's nodes) with the base at the origin.
    Bodies standing in the scene (built bars, their joint halves, parked
    support robots posed in their own configuration) are placed once. Bodies
    riding on a robot link (the joint halves on the carried bar)
    are written as top-level nodes named ``C-<link name>-<number>`` whose
    transform is their attachment frame, so the page can place them on the
    posed link; the carried bar itself is left out, because the page draws it
    from the roadmap's grasp.

    Args:
        cell (RobotCell): the (small) scene cell.
        state (RobotCellState): its state.
        bar_rb_name (str): the carried bar's rigid-body name (not written).
        out_path (str): the glb file to write.

    Returns:
        Dict[str, Any]: counts of what was written and the file size.

    Raises:
        ValueError: a body rides on a tool rather than on a robot link, which
            the page cannot follow.
    """
    builder = GlbBuilder()
    materials = {name: builder.add_material(name, colour) for name, colour in SCENE_COLOURS.items()}
    roots: List[int] = []
    counts = {"standing": 0, "carried": 0, "support_robots": 0, "triangles": 0}
    # From the world to the robot's base frame: applied to everything that stands in the world.
    base_from_world = as_transformation(state.robot_base_frame).inverse()

    def add_mesh_node(name: str, mesh, scale: float, frame, material: int) -> None:
        """One mesh at one world (or link) frame."""
        vertices, faces = mesh_arrays(mesh)
        if faces.size == 0:
            return
        vertices = vertices * np.float32(scale)
        translation, rotation = frame_to_translation_rotation(frame)
        counts["triangles"] += int(faces.shape[0])
        roots.append(builder.add_node(name, translation=translation, rotation=rotation,
                                      mesh=builder.add_mesh(vertices, faces, material)))

    # * Rigid bodies: standing ones at their frame, carried ones at their attachment frame.
    for name, body in cell.rigid_body_models.items():
        body_state = state.rigid_body_states[name]
        if body_state.is_hidden or name == bar_rb_name:
            continue
        meshes = list(body.visual_meshes or body.collision_meshes or [])
        if is_carried(body_state):
            if not body_state.attached_to_link:
                raise ValueError(f"{name} rides on tool {body_state.attached_to_tool!r}; the roadmap page can "
                                 "only follow bodies attached to a robot link.")
            counts["carried"] += 1
            for number, mesh in enumerate(meshes):
                add_mesh_node(f"C-{body_state.attached_to_link}-{counts['carried']}_{number}", mesh,
                              body.native_scale, body_state.attachment_frame, materials["carried_joint"])
        else:
            counts["standing"] += 1
            colour = "joint" if name.startswith("joint_") else "built_bar"
            frame = Frame.from_transformation(base_from_world * as_transformation(body_state.frame))
            for number, mesh in enumerate(meshes):
                add_mesh_node(f"S-{counts['standing']}_{number}", mesh, body.native_scale, frame, materials[colour])

    # * Support robots: tools that are not on an arm, posed in their own configuration.
    for name, tool in cell.tool_models.items():
        tool_state = state.tool_states[name]
        if tool_state.attached_to_group or tool_state.is_hidden or tool_state.frame is None:
            continue
        counts["support_robots"] += 1
        world_from_root = base_from_world * as_transformation(tool_state.frame)
        joint_state = tool_state.configuration if tool_state.configuration is not None else {}
        moved = tool.compute_transformations(joint_state) if joint_state else {}
        # Each link's frame in the tool's root frame: the root link sits at the
        # origin, every other link at its parent joint's posed origin.
        link_in_root: Dict[str, Transformation] = {tool.root.name: Transformation()}
        for joint in tool.iter_joints():
            link_in_root[joint.child.link] = moved.get(joint.name, Transformation()) * as_transformation(
                joint.current_origin)
        for link in tool.links:
            if link.name not in link_in_root:
                continue
            for index, visual in enumerate(link.visual or []):
                shape = visual.geometry.shape
                scale = getattr(shape, "scale", None)
                world_from_visual = world_from_root * link_in_root[link.name] * as_transformation(visual.origin)
                frame = Frame.from_transformation(world_from_visual)
                for mesh_index, mesh in enumerate(list(getattr(shape, "meshes", None) or [])):
                    vertices, faces = mesh_arrays(mesh)
                    if faces.size == 0:
                        continue
                    if scale is not None:
                        vertices = vertices * np.asarray(scale, dtype=np.float32)
                    translation, rotation = frame_to_translation_rotation(frame)
                    counts["triangles"] += int(faces.shape[0])
                    roots.append(builder.add_node(
                        f"R-{counts['support_robots']}_{link.name}_{index}_{mesh_index}",
                        translation=translation, rotation=rotation,
                        mesh=builder.add_mesh(vertices, faces, materials["support_robot"])))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    counts["bytes"] = builder.write(out_path, roots)
    counts["path"] = out_path
    return counts
