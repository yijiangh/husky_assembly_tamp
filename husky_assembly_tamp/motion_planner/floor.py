"""A floor under the robot for the held transfer: a slab whose top is level with the robot's base frame.

The process's ground is a thin sheet that ground-mated joint halves may touch, so a carried bar could pass through
it. The slab is checked against the arms, their tools and everything they carry; the rest of the robot stands on it.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence, Tuple

from compas.datastructures import Mesh
from compas.geometry import Frame
from compas_fab.robots import RigidBody, RigidBodyState, RobotCell, RobotCellState

# * The slab's name in a cell, and its size: wide enough for any carry, thick enough that nothing gets under it.
FLOOR = "floor"
FLOOR_SIDE = 12.0
FLOOR_THICKNESS = 0.5
# * A robot link with this in its name belongs to an arm, so it is checked against the slab.
ARM_LINK_MARK = "ur_arm"


def floor_model() -> RigidBody:
    """The slab as a rigid body: its top face is the plane z = 0 of its frame."""
    half = FLOOR_SIDE / 2.0
    corners = [[x, y, z] for z in (-FLOOR_THICKNESS, 0.0) for x, y in ((-half, -half), (half, -half), (half, half),
                                                                      (-half, half))]
    faces = [[3, 2, 1, 0], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return RigidBody.from_mesh(Mesh.from_vertices_and_faces(corners, faces))


def floor_state(state: RobotCellState, link_names: Sequence[str]) -> RigidBodyState:
    """The slab under the robot of a state, with what may stand on it.

    Args:
        state: The scene; its robot base frame gives the slab's place and height.
        link_names: Every link of the robot.

    Returns:
        RigidBodyState: The slab, level, its top at the base frame's height; the robot's links outside the arms and
        the tools that are not on this robot (another robot standing in the scene) may touch it.
    """
    base = state.robot_base_frame
    standing = [name for name in link_names if ARM_LINK_MARK not in name]
    others = [name for name, tool in (state.tool_states or {}).items() if not tool.attached_to_group]
    return RigidBodyState(Frame([base.point.x, base.point.y, base.point.z], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
                          touch_links=standing, touch_bodies=others)


def cell_with_floor(cell: RobotCell, state: RobotCellState) -> Tuple[RobotCell, RobotCellState]:
    """Add the slab to a cell that is not loaded yet.

    Args:
        cell: The cell; gets the slab's model.
        state: Its state; not changed.

    Returns:
        Tuple[RobotCell, RobotCellState]: The cell, and a copy of the state with the slab in it.
    """
    cell.rigid_body_models[FLOOR] = floor_model()
    with_floor = state.copy()
    with_floor.rigid_body_states[FLOOR] = floor_state(state, [link.name for link in cell.robot_model.links])
    return cell, with_floor


@contextmanager
def floor_added(planner, state: RobotCellState) -> Iterator[RobotCellState]:
    """Put the slab into a planner's loaded cell for the length of a plan.

    ! Plan with the state this gives, not the caller's: only it knows the slab.

    Args:
        planner: A compas_fab PyBullet planner with its cell loaded.
        state: The plan's start state; not changed.

    Yields:
        RobotCellState: A copy of the state with the slab in it.
    """
    client = planner.client
    cell = client.robot_cell
    if FLOOR in cell.rigid_body_models:
        # Already there (a cell opened with `cell_with_floor`): only the state may lack it.
        with_floor = state
        if FLOOR not in state.rigid_body_states:
            with_floor = state.copy()
            with_floor.rigid_body_states[FLOOR] = floor_state(state, [link.name for link in cell.robot_model.links])
        yield with_floor
        return
    model = floor_model()
    cell.rigid_body_models[FLOOR] = model
    client._add_rigid_body(FLOOR, model)
    with_floor = state.copy()
    with_floor.rigid_body_states[FLOOR] = floor_state(state, [link.name for link in cell.robot_model.links])
    try:
        yield with_floor
    finally:
        client._remove_rigid_body(FLOOR)
        del cell.rigid_body_models[FLOOR]
        planner.set_robot_cell_state(state)
