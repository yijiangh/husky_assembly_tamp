"""The base planner copied from the monitor (``husky_assembly_tamp.base_planner``) and the drive-in check built on it.

The robot is a box with a bar on a turning link, from a URDF written to a temporary folder: no process folder, no
ROS. No case depends on how fast the machine is: it ends before any search, or accepts both ways a search can end.
The tests need the shared core library and are skipped without it. Run from the repository root::

    python -m pytest benchmark/tests/test_base_planner.py -q -p no:cacheprovider
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pybullet_planning as pp
import pytest

# * Importing the benchmark package first puts the sibling checkout of the core on the path.
from husky_assembly_tamp.benchmark.process_export import refresh_world_poses

pytest.importorskip("scaffolding_assembly_core", reason="needs scaffolding_assembly_core")

from scaffolding_assembly_core.geometry import Pose, box_geometry  # noqa: E402
from scaffolding_assembly_core.robot import RobotObject, robot_model  # noqa: E402
from scaffolding_assembly_core.scene import Attachment, Body, Scene  # noqa: E402

import husky_assembly_tamp.base_planner as base_planner  # noqa: E402
from husky_assembly_tamp.base_planner.lattice import NARROW_MARGIN, plan_drive  # noqa: E402
from husky_assembly_tamp.base_planner.path import densify  # noqa: E402
from husky_assembly_tamp.base_planner.world import PlanningWorld  # noqa: E402
from husky_assembly_tamp.benchmark import drive_in  # noqa: E402
from husky_assembly_tamp.benchmark.drive_in import DriveIn  # noqa: E402

ROBOT = "robots/box"
#: A chassis 0.9 m long and 0.6 m wide, and a link 1 m up that turns about the vertical: the "arm" the bar rides on.
URDF = """<?xml version="1.0"?>
<robot name="box">
  <link name="base_link">
    <collision><origin xyz="0 0 0.3"/><geometry><box size="0.9 0.6 0.4"/></geometry></collision>
  </link>
  <link name="arm_link"/>
  <joint name="arm_joint" type="revolute">
    <parent link="base_link"/><child link="arm_link"/>
    <origin xyz="0 0 1.0"/><axis xyz="0 0 1"/>
    <limit lower="-3.2" upper="3.2" effort="1" velocity="1"/>
  </joint>
</robot>
"""
JOINTS = ["arm_joint"]
#: The bar across the robot (it reaches 1.2 m to either side), and turned along it.
ACROSS, ALONG = [0.0], [math.pi / 2.0]
#: The drive: 1.6 m west, facing west all the way.
START, GOAL = (0.0, 0.0, math.pi), (-1.6, 0.0, math.pi)
#: A post beside the drive: clear of the chassis, but in the way of the bar held across.
POST = Body("test/post", box_geometry((0.1, 0.1, 2.0)), Pose((-1.0, 1.1, 1.0)), label="post")
#: A post the bar held across is inside of at the start.
POST_IN_BAR = Body("test/post", box_geometry((0.1, 0.1, 2.0)), Pose((0.0, 1.0, 1.0)), label="post")
#: Time limit of a search that should find its way, seconds: far more than the second or two it takes.
LIMIT = 20.0


@pytest.fixture(scope="module", autouse=True)
def other_world():
    """Another PyBullet world with one box, opened before the planner's: what a carry planner would live in."""
    client = p.connect(p.DIRECT)
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], physicsClientId=client)
    box = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape, basePosition=(1.0, 2.0, 3.0),
                            physicsClientId=client)
    yield SimpleNamespace(client=client, box=box, pp_client=pp.get_client())
    p.disconnect(physicsClientId=client)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """The box robot's model, its URDF written once."""
    urdf = tmp_path_factory.mktemp("box_robot") / "box.urdf"
    urdf.write_text(URDF)
    return robot_model("box", urdf, None)


@pytest.fixture(scope="module")
def world():
    """One planning world for the module; each test syncs the scene it needs."""
    world = PlanningWorld()
    yield world
    world.close()


def _scene(model, bodies=(), arm=ACROSS, bar: bool = False, base: Pose = Pose()) -> Scene:
    """The box robot at `base` with its arm at `arm`, the standing `bodies`, and with `bar` a 2.4 m bar on the arm."""
    scene = Scene(bodies={body.id: body.copy() for body in bodies},
                  robots={ROBOT: RobotObject(ROBOT, model, base, dict(zip(JOINTS, arm)))})
    if bar:
        scene.bodies["test/bar"] = Body("test/bar", box_geometry((0.1, 2.4, 0.1)),
                                        Attachment(ROBOT, "arm_link", Pose()), label="bar")
    refresh_world_poses(scene)
    return scene


def _drive_in(monkeypatch, model, bodies, **options) -> DriveIn:
    """A `DriveIn` from START to GOAL whose carry scene is the box robot at GOAL holding the bar, among `bodies`."""
    at_goal = Pose((GOAL[0], GOAL[1], 0.0), (0.0, 0.0, 1.0, 0.0))
    scene = _scene(model, bodies, bar=True, base=at_goal)
    monkeypatch.setattr(drive_in, "carry_scene",
                        lambda process, bar_id, base=None: (scene, SimpleNamespace(robot=ROBOT), None))
    return DriveIn(None, "B1", JOINTS, START, **options)


def _assert_drives_clear(world: PlanningWorld, path: np.ndarray) -> None:
    """The path runs from START to GOAL in small steps, keeps the narrow margin, and touches nothing in between."""
    assert path.ndim == 2 and path.shape[1] == 3
    np.testing.assert_allclose(path[0], START, atol=1e-9)
    np.testing.assert_allclose(path[-1][:2], GOAL[:2], atol=1e-9)
    assert math.cos(path[-1][2] - GOAL[2]) == pytest.approx(1.0)
    # * 5 cm steps of the turning centre: base_footprint, a little off it, moves slightly more on a curve.
    assert np.hypot(*np.diff(path[:, :2], axis=0).T).max() < 0.06
    for pose in path:
        assert world.hit_by(ROBOT, tuple(pose), NARROW_MARGIN - 0.005) is None, pose
    for pose in densify(path, 0.01, 0.01):
        assert world.hit_by(ROBOT, pose, 0.0) is None, pose


# -----------------------------------------------------------------------------
# * The search (plan_drive)
# -----------------------------------------------------------------------------


def test_free_line_is_direct(world, model):
    """Nothing in the way: the straight move, no search, as waypoints from the start to the goal."""
    world.sync(_scene(model))
    result = plan_drive(world, ROBOT, START, GOAL)
    assert result.path is not None and result.direct, result.reason
    _assert_drives_clear(world, result.path)


def test_post_beside_the_line_is_in_the_way_only_of_a_bar_held_across(world, model):
    """The chassis passes the post, and so does a bar held along it; a bar held across needs a detour, or is refused."""
    for scene in (_scene(model, [POST]), _scene(model, [POST], arm=ALONG, bar=True)):
        world.sync(scene)
        assert plan_drive(world, ROBOT, START, GOAL).direct
    world.sync(_scene(model, [POST], bar=True))
    result = plan_drive(world, ROBOT, START, GOAL, time_limit=LIMIT)
    assert not result.direct
    if result.path is None:  # * only on a machine too busy to search: no proof, and it says so
        assert "no path found in" in result.reason
    else:
        _assert_drives_clear(world, result.path)
        assert np.hypot(*np.diff(result.path[:, :2], axis=0).T).sum() > 1.6 + 0.1, "longer than the straight line"


def test_start_inside_a_post_is_refused_at_once(world, model):
    """The bar is inside a post where the drive starts: refused before any search, and the reason names the post."""
    world.sync(_scene(model, [POST_IN_BAR], bar=True))
    result = plan_drive(world, ROBOT, START, GOAL)
    assert result.path is None and result.reason == "start is in collision with post"


# -----------------------------------------------------------------------------
# * The drive-in check (DriveIn)
# -----------------------------------------------------------------------------


def test_drive_in_keeps_its_answers_and_reuses_a_path_for_another_posture(monkeypatch, model):
    """Asked again, a posture costs nothing; a posture a little off drives the path already found."""
    drive = _drive_in(monkeypatch, model, [POST])
    try:
        assert drive.goal == pytest.approx(GOAL)
        assert drive(ALONG) == "" and drive(ALONG) == ""
        assert (drive.asks, drive.reused, len(drive._paths)) == (2, 0, 1)
        nearly = [ALONG[0] + 0.01]
        assert drive(nearly) == ""
        assert (drive.asks, drive.reused, len(drive._paths)) == (3, 1, 1)
        summary = drive.summary(nearly)
        assert summary["direct"] and summary["length_m"] == pytest.approx(1.6)
        assert summary["points"] == drive.summary(ALONG)["points"] and summary["points"][0] == pytest.approx(START)
        assert (summary["station"], summary["goal"]) == (list(START), pytest.approx(GOAL))
        assert (summary["asks"], summary["reused"], summary["refused"]) == (3, 1, [])
        assert "points" not in drive.summary() and "points" not in drive.summary(ACROSS)
    finally:
        drive.close()


def test_drive_in_with_the_bar_across_finds_a_detour_or_says_it_ran_out_of_time(monkeypatch, model):
    """The posture decides: the bar held across cannot drive the straight line past the post."""
    drive = _drive_in(monkeypatch, model, [POST], time_limit=LIMIT)
    try:
        verdict = drive(ACROSS)
        summary = drive.summary(ACROSS)
        if verdict:  # * only on a machine too busy to search
            assert summary["refused"][0]["timed_out"] and "points" not in summary
        else:
            assert not summary["direct"] and summary["length_m"] > 1.6 + 0.1 and summary["refused"] == []
            _assert_drives_clear(drive.world, np.array(summary["points"]))
    finally:
        drive.close()


def test_drive_in_refuses_a_bar_inside_a_post_at_the_station_at_once(monkeypatch, model):
    """No search for a station in collision: the reason names the station and the post, and is no time-out."""
    drive = _drive_in(monkeypatch, model, [POST_IN_BAR])
    try:
        assert drive(ACROSS) == "at the station, bar is in collision with post"
        (refused,) = drive.summary()["refused"]
        assert refused["reason"] == drive(ACROSS) and refused["timed_out"] is False
        assert drive._paths == [] and len(drive.summary()["refused"]) == 1, "asked twice, refused once"
    finally:
        drive.close()


def test_drive_in_records_a_search_out_of_time_as_such(monkeypatch, model):
    """With no time to search, the bar held across is refused as a time-out, not as proven impossible."""
    drive = _drive_in(monkeypatch, model, [POST], time_limit=0.0)
    try:
        assert drive(ACROSS).startswith("no drive path from the station")
        (refused,) = drive.summary()["refused"]
        assert refused["timed_out"] is True and refused["seconds"] > 0.0
    finally:
        drive.close()


def test_drive_in_scene_leaves_out_the_ground_and_what_is_overhead_and_allows_no_contact(monkeypatch, model):
    """The ground sheet and a beam far overhead are left out; a contact the process allows the bar still blocks."""
    ground = Body("ground/floor", box_geometry((12.0, 12.0, 0.002)), Pose((0.0, 0.0, -0.001)))
    beam = Body("test/beam", box_geometry((12.0, 12.0, 0.3)), Pose((0.0, 0.0, 6.15)))
    lintel = Body("test/lintel", box_geometry((0.2, 0.2, 0.1)), Pose((5.0, 5.0, 1.3)))  # 0.2 m above the bar's top
    receiver = Body("test/receiver", box_geometry((0.1, 0.1, 0.1)), Pose((0.0, 1.0, 1.0)), touches=("test/bar",),
                    label="receiver")
    drive = _drive_in(monkeypatch, model, [ground, beam, lintel, receiver])
    try:
        bodies = drive.scene.bodies
        assert bodies["test/receiver"].touches == ()
        assert drive(ACROSS) == "at the station, bar is in collision with receiver"
        assert drive(ALONG) == "" and drive.summary(ALONG)["direct"]
        assert not bodies["ground/floor"].enabled and not bodies["test/beam"].enabled
        assert bodies["test/lintel"].enabled and bodies["test/receiver"].enabled
    finally:
        drive.close()


# -----------------------------------------------------------------------------
# * What the copy must not do
# -----------------------------------------------------------------------------


def test_no_ros_viser_or_monitor_imports():
    """The copied package and the drive-in check import nothing of ROS, viser, crl_husky or the monitor."""
    banned = {"rclpy", "ament_index_python", "viser", "crl_husky", "husky_assembly_teleop"}
    files = sorted(Path(base_planner.__file__).parent.glob("*.py")) + [Path(drive_in.__file__)]
    assert len(files) == 7
    for file in files:
        tree = ast.parse(file.read_text())
        names = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        names += [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
        assert not {name.split(".")[0] for name in names} & banned, file.name
    assert not ast.parse(Path(base_planner.__file__).read_text()).body[1:], "the package's __init__ imports nothing"


def test_other_pybullet_worlds_are_left_alone(other_world, world, model):
    """A search in the planner's own world leaves another world, and pybullet_planning's client, as they were."""
    world.sync(_scene(model, [POST], bar=True))
    assert plan_drive(world, ROBOT, START, GOAL, time_limit=0.0).path is None
    assert world.mirror.client_id != other_world.client
    assert p.getNumBodies(physicsClientId=other_world.client) == 1
    assert p.getBasePositionAndOrientation(other_world.box, physicsClientId=other_world.client)[0] == (1.0, 2.0, 3.0)
    assert pp.get_client() == other_world.pp_client
