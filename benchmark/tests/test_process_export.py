"""Reading a schema 2.1 process folder as the benchmark reads an old export.

The tests need the two-modules demo process folder
(``<HUSKY_DESIGN_STUDY_DIR>/261008_2ModulesDemo_purged_process``) and the
shared core library (``scaffolding_assembly_core``, a sibling checkout or a
pip install); they are skipped when either is missing. Run from the
repository root::

    python -m pytest benchmark/tests/test_process_export.py -v
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from husky_assembly_tamp.benchmark import process_export
from husky_assembly_tamp.benchmark.build_cells import find_bar_ids, source_key_for
from husky_assembly_tamp.benchmark.scene import open_scene
from husky_assembly_tamp.motion_planner.holding_prm.scenes import carry_goal
from husky_assembly_tamp.motion_planner.holding_prm.sampling import robot_side_signature

# * The process folder the tests read, and the bar they look at.
STUDY_DIR = os.environ.get("HUSKY_DESIGN_STUDY_DIR", "")
PROCESS_DIR = os.path.join(STUDY_DIR, "261008_2ModulesDemo_purged_process")
BAR = "B21"


def core_present() -> bool:
    """Whether the shared core library can be imported."""
    try:
        process_export.require_core()
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not (STUDY_DIR and process_export.is_process_folder(PROCESS_DIR) and core_present()),
    reason="needs HUSKY_DESIGN_STUDY_DIR/261008_2ModulesDemo_purged_process and scaffolding_assembly_core",
)


@pytest.fixture(scope="module")
def process():
    """The process, read once for the module."""
    return process_export.read_process(PROCESS_DIR)


# -----------------------------------------------------------------------------
# * Finding things in the folder
# -----------------------------------------------------------------------------


def test_source_key_is_stable_and_content_based(process):
    """The cache key depends on the folder name and the files' content, not on their dates."""
    key = source_key_for(PROCESS_DIR)
    assert key.startswith("261008_2ModulesDemo_purged_process-")
    assert len(key.split("-")[-1]) == 8
    assert key == process_export.process_source_key(PROCESS_DIR)


def test_bar_ids_follow_the_schedule(process):
    """Every bar with a jointing and a release action, in schedule order."""
    bars = find_bar_ids(PROCESS_DIR)
    assert bars[:3] == ["B20", "B21", "B22"]
    assert bars == process_export.process_bar_ids(process)
    assert all(f"{bar}_R" in process.actions for bar in bars)


def test_carry_is_the_coupled_free_movement(process):
    """The carry is the transfer: both arms move freely, coupled, from an open start to authored joints."""
    action = process_export.jointing_action(process, BAR)
    movement = process_export.carry_movement(action)
    assert movement.id == f"{BAR}_J_transfer"
    assert movement.start.robots[action.robot].joints is None
    assert len(process_export.goal_joints(movement, action.robot, sorted(movement.target.joints[action.robot]))) == 12
    poses = process_export.goal_link_poses(movement, action.robot)
    assert set(poses) == {"left", "right"}
    held = process_export.holders(movement, action.bar)
    assert held["left"].to.endswith("left_ur_arm_tool0") and held["right"].to.endswith("right_ur_arm_tool0")


def test_home_joints_come_from_the_release(process):
    """The home is the release's last free movement's target, all twelve arm joints."""
    action = process_export.jointing_action(process, BAR)
    home = process_export.home_joints(process, BAR, action.robot)
    assert len(home) == 12
    assert all("ur_arm" in name for name in home)


# -----------------------------------------------------------------------------
# * The trimmed problem, as build-cells caches it
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trimmed(process):
    """The problem, trimmed cell and template state of the bar."""
    return process_export.trimmed_problem(process, BAR)


def test_trimmed_problem_reads_the_bar(trimmed):
    """The bar's size comes from its cylinder, the halves ride along, the grasps are the holders' inverses."""
    problem, cell, template = trimmed
    assert problem.bar_id == BAR and problem.bar_rb_name == f"bars/{BAR}"
    assert abs(problem.bar_length_m - 2.0) < 1e-6 and abs(problem.bar_radius_m - 0.01) < 1e-6
    assert len(problem.joint_rb_names) == 7 and all(name.startswith("joints/") for name in problem.joint_rb_names)
    assert problem.attached_link == "left_ur_arm_tool0"
    assert len(problem.joint_names_12) == 12 and len(problem.home_conf12) == 12
    # The cell holds exactly the robot's two tools and the carried bodies.
    assert set(cell.tool_models) == {"tools/cindy/AT3L", "tools/cindy/AT3R"}
    assert set(cell.rigid_body_models) == {problem.bar_rb_name, *problem.joint_rb_names}
    # The template stands at the origin, at the home, with the bar on the left flange.
    assert np.allclose(list(template.robot_base_frame.point), [0.0, 0.0, 0.0])
    bar_state = template.rigid_body_states[problem.bar_rb_name]
    assert bar_state.attached_to_link == "left_ur_arm_tool0"
    assert np.allclose(list(bar_state.attachment_frame.point), problem.tool0_from_bar[0])
    for name, value in zip(problem.joint_names_12, problem.home_conf12):
        assert template.robot_configuration[name] == pytest.approx(value)
    # The tools may touch the bar they hold (the core's part rule).
    assert "tools/cindy/AT3L" in bar_state.touch_bodies


def test_grasp_reproduces_the_bar_pose(trimmed):
    """Posed at the home, PyBullet puts the bar where the left flange and the recorded grasp say (build-cells' check)."""
    from husky_assembly_tamp.benchmark.build_cells import verify_grasp

    problem, cell, template = trimmed
    scene = open_scene(cell, template, problem)
    try:
        errors = verify_grasp(scene, problem)
    finally:
        scene.close()
    assert errors["position_error_m"] < 1e-6 and errors["orientation_error"] < 1e-6


def test_carry_scene_matches_the_trimmed_template(process, trimmed):
    """The online scene carries the same bodies, attached the same way, as the roadmap's precompute scene."""
    from scaffolding_assembly_core.mirrors.cell import CompasFabCell

    problem, _cell, template = trimmed
    scene, action, _movement = process_export.carry_scene(process, BAR)
    builder = CompasFabCell(action.robot, visual=False)
    builder.update(scene)
    live = builder.state(scene)
    assert robot_side_signature(live, problem.bar_rb_name) == robot_side_signature(template, problem.bar_rb_name)
    # The real scene also shows what was built before: B20 and its halves, and the ground.
    shown = {name for name, body in live.rigid_body_states.items() if not body.is_hidden}
    assert "bars/B20" in shown and "ground/WG0" in shown and "bars/B22" not in shown


def test_goal_frames_and_shifted_base(process, trimmed):
    """The goal as seen from the robot moves with the base; the shift is in the base's own frame."""
    problem, _cell, _template = trimmed
    goal = carry_goal(PROCESS_DIR, BAR, problem)
    assert goal["world_from_bar"].shape == (4, 4)
    world_from_base, _world_from_left = process_export.carry_goal_frames(process, BAR)
    base = process_export.pose_from_lists([world_from_base[:3, 3], [0, 0, 0, 1]])
    moved = process_export.shifted_pose(base, 0.1, 0.0, 0.0)
    assert np.allclose(np.subtract(moved.position, base.position), [0.1, 0.0, 0.0])
    turned = process_export.shifted_pose(base, 0.0, 0.0, np.pi / 2)
    assert np.allclose(turned.position, base.position)
    assert abs(turned.orientation[2] - np.sin(np.pi / 4)) < 1e-9


def test_trimmed_cell_survives_a_round_trip_in_another_process(tmp_path, trimmed):
    """A worker process reads the cached cell back (its meshes carry the core's mesh class)."""
    from compas.data import json_dump

    problem, cell, template = trimmed
    json_dump(cell, str(tmp_path / "cell.json"))
    json_dump(template, str(tmp_path / "template_state.json"))
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from husky_assembly_tamp.benchmark import sibling_packages\n"
        "from compas.data import json_load\n"
        "cell = json_load(%r); state = json_load(%r)\n"
        "cell.assert_cell_state_match(state)\n"
        "print(len(cell.rigid_body_models), len(cell.robot_model.links))\n"
    ) % (os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
         str(tmp_path / "cell.json"), str(tmp_path / "template_state.json"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split()[0] == str(1 + len(problem.joint_rb_names))
