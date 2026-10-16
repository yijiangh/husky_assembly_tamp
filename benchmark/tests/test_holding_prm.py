"""Checks for the holding-constraint roadmap pilot (``motion_planner.holding_prm``).

The first group needs no robot. The second opens the smoke cell of the
benchmark data root in this process (no files are written there) and is
skipped when the root is not available (set ``HUSKY_BENCH_ROOT``). The last
group compares collision checks on an exported problem with built bars around
the robot; it is skipped when that export or its trimmed scenes are missing
(``HUSKY_EXPORT_PROBLEM_DIR`` points at another export than the default). Run with::

    .venv/Scripts/python.exe -m pytest benchmark/tests/test_holding_prm.py -q
"""
from __future__ import annotations

import glob
import os
import struct
import subprocess
import sys
from json import loads
from types import SimpleNamespace

import numpy as np
import pytest
from compas.datastructures import Mesh
from compas.geometry import Box, Frame
from compas_fab.robots import RigidBody, RigidBodyState, RobotCellState, ToolState
from pybullet_planning import tform_from_pose

from husky_assembly_tamp.benchmark.asset_versions import stamp_module_imports, stamp_page_assets
from husky_assembly_tamp.benchmark.scene import load_trimmed_problem, open_scene
from husky_assembly_tamp.motion_planner.api import plan_constrained_dual_arm
from husky_assembly_tamp.motion_planner.holding_prm.holding import (
    STEP_MAX_RAD,
    HoldingModel,
    arm_limits,
    bar_from_arm,
    build_holding_model,
    continuous_poses,
    holding_violation,
    interpolate_confs,
    interpolate_pose,
    invert_transform,
    matrix_from_pose,
    pose_from_matrix,
    pose_gap,
    target_bar_pose,
    tool0_world,
    tool_target_from_bar,
    violation_score_mm,
)
from husky_assembly_tamp.motion_planner.holding_prm.online import RoadmapMismatch, check_roadmap_fits
from husky_assembly_tamp.motion_planner.holding_prm.pilot import (
    choose_starts,
    load_library_entry,
    spread_starts,
    wrapped_match,
)
from husky_assembly_tamp.motion_planner.holding_prm.roadmap import (
    bisection_order,
    candidate_pairs,
    edge_cost,
    evaluate_edge,
)
from husky_assembly_tamp.motion_planner.holding_prm.sampling import (
    SamplerConfig,
    WorkerTools,
    arm_branches,
    expand_bar_pose,
    make_collision_fn,
    new_counters,
    robot_side_signature,
    sample_chunk,
)
from husky_assembly_tamp.motion_planner.holding_prm.scenes import (
    export_scene_glb,
    geometry_key,
    problem_source_key,
    state_key,
)
from husky_assembly_tamp.motion_planner.holding_prm.search import (
    EdgeTable,
    build_graph,
    lazy_search,
    minimax_to_goals,
    minimax_to_goals_nx,
    route_from,
    shortest_between_sets,
)
from husky_assembly_tamp.motion_planner.holding_prm.refine import Tracker, spread_size
from husky_assembly_tamp.motion_planner.holding_prm.store import read_json, write_json_atomic
from husky_assembly_tamp.motion_planner.holding_prm.tracking import TrackSettings, track_edge


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# * ---- Pieces that need no robot ------------------------------------------


def test_interpolation_never_moves_a_joint_more_than_the_step():
    """Every piece of a joint-space line would pass the production continuity gate."""
    a = np.zeros(12)
    b = np.linspace(-2.0, 3.0, 12)
    points = interpolate_confs(a, b, STEP_MAX_RAD, include_ends=True)
    steps = np.max(np.abs(np.diff(points, axis=0)), axis=1)
    assert steps.max() <= STEP_MAX_RAD + 1e-12
    assert np.allclose(points[0], a) and np.allclose(points[-1], b)
    # Interior points only, and none at all for a line shorter than one step.
    assert len(interpolate_confs(a, a + 0.01, STEP_MAX_RAD)) == 0


def test_bisection_order_visits_every_index_once_middle_first():
    for n in (1, 2, 5, 16):
        order = bisection_order(n)
        assert sorted(order) == list(range(n))
    assert bisection_order(5)[0] == 2


def test_pose_round_trip_and_small_angles():
    pose = [[0.1, -0.2, 0.3], [0.1826, 0.3651, 0.5477, 0.7303]]
    again = pose_from_matrix(matrix_from_pose(pose))
    assert np.allclose(matrix_from_pose(again), matrix_from_pose(pose), atol=1e-9)
    # A rotation of one micro-radian must read as one micro-radian, not as
    # the arc-cosine's noise floor.
    angle = 1e-6
    turned = np.eye(4)
    turned[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    _, measured = pose_gap(np.eye(4), turned)
    assert measured == pytest.approx(angle, rel=1e-3)


def test_path_poses_keep_their_quaternion_sign():
    """A flipped sign is the same orientation, but an interpolation by numbers reads it as a full turn."""
    poses = [[[0, 0, 0], [0, 0, 0, 1]], [[0, 0, 1], [0, 0, -0.01, -1]], [[0, 0, 2], [0, 0, 0.02, 1]]]
    fixed = continuous_poses(poses)
    assert [pose[0] for pose in fixed] == [[0, 0, 0], [0, 0, 1], [0, 0, 2]]
    assert np.allclose(fixed[1][1], [0, 0, 0.01, 1]) and np.allclose(fixed[2][1], [0, 0, 0.02, 1])
    for before, after in zip(fixed[:-1], fixed[1:]):
        assert np.dot(before[1], after[1]) > 0


def test_score_and_cost_mean_the_same_tolerance():
    """An edge within tolerance costs its length; one over it pays the penalty."""
    assert violation_score_mm(0.002, 0.0, 0.4) == pytest.approx(2.0)
    assert violation_score_mm(0.0, np.radians(2.0), 0.4) == pytest.approx(5.0)
    costs = edge_cost(np.array([1.0, 1.0]), np.array([4.0, 10.0]), tolerance_mm=5.0, w_hold=10.0)
    assert costs[0] == pytest.approx(1.0)
    assert costs[1] == pytest.approx(1.0 + 10.0 * (10.0 / 5.0 - 1.0))


def test_candidate_pairs_are_unique_symmetric_and_respect_the_radius():
    confs = np.zeros((4, 12))
    confs[1, 0] = 0.1
    confs[2, 0] = 0.2
    confs[3, 0] = 5.0  # far from everything
    pairs = candidate_pairs(confs, k=3, radius_rad=1.0)
    assert all(i < j for i, j in pairs)
    assert len({tuple(p) for p in pairs}) == len(pairs)
    assert not any(3 in p for p in pairs)


def _table(edges):
    """An edge table from (i, j, length, pos_mm, free, checked) tuples."""
    columns = {"i": [], "j": [], "len_l2": [], "viol_pos_max": [], "viol_rot_max": [], "free": [], "checked": []}
    for i, j, length, pos_mm, free, checked in edges:
        for name, value in zip(columns, (i, j, length, pos_mm / 1000.0, 0.0, free, checked)):
            columns[name].append(value)
    return EdgeTable.from_columns([columns], rot_deg_per_mm=0.4)


def test_minimax_and_shortest_search_between_sets():
    """Two starts (0, 1), one goal (4). Start 0 is short but bad; start 1 is long but good."""
    table = _table([
        (0, 4, 1.0, 20.0, True, True),   # direct, 20 mm
        (1, 2, 1.0, 1.0, True, True),
        (2, 3, 1.0, 2.0, True, True),
        (3, 4, 1.0, 1.5, True, True),
        (0, 2, 1.0, 50.0, False, True),  # blocked: must never be used
    ])
    gap, parent = minimax_to_goals(5, table, table.usable(), goal_ids=[4])
    assert gap[1] == pytest.approx(2.0)
    assert gap[0] == pytest.approx(20.0)
    assert route_from(1, parent) == [1, 2, 3, 4]
    within = table.usable() & (table.score <= 5.0)
    path, _cost = shortest_between_sets(build_graph(table, within, table.len_l2, [0, 1], [4]))
    assert path == [1, 2, 3, 4]
    path, _cost = shortest_between_sets(build_graph(table, table.usable(), table.len_l2, [0, 1], [4]))
    assert path == [0, 4]


def test_fast_minimax_matches_the_networkx_reference():
    """The scipy minimax gives the same gaps as the networkx one, duplicates and blocked edges included."""
    rng = np.random.default_rng(3)
    n = 60
    edges = [(int(a), int(b), 1.0, float(rng.uniform(0.5, 40.0)), bool(rng.random() > 0.1), True)
             for a, b in rng.integers(0, n, size=(240, 2)) if a != b]
    edges += [(edges[0][0], edges[0][1], 1.0, 0.1, True, True)]  # a cheaper duplicate of one pair
    table = _table(edges)
    goals = [0, 5, 9]
    fast_gap, fast_parent = minimax_to_goals(n, table, table.usable(), goals)
    slow_gap, _slow_parent = minimax_to_goals_nx(n, table, table.usable(), goals)
    assert np.array_equal(np.isfinite(fast_gap), np.isfinite(slow_gap))
    finite = np.isfinite(slow_gap)
    assert np.allclose(fast_gap[finite], slow_gap[finite], atol=1e-9)
    # Every parent pointer leads to a goal through edges no worse than the gap.
    for node in np.flatnonzero(finite):
        assert route_from(int(node), fast_parent)[-1] in goals


def test_lazy_search_drops_edges_that_turn_out_blocked():
    table = _table([(0, 1, 1.0, 1.0, True, False), (1, 2, 1.0, 1.0, True, False),
                    (0, 3, 2.0, 1.0, True, False), (3, 2, 2.0, 1.0, True, False)])

    def find_path(mask):
        found = shortest_between_sets(build_graph(table, mask, table.len_l2, [0], [2]))
        return found[0] if found else None

    def check_rows(rows):
        # Edge 1-2 collides; everything else is free.
        return [{"free": not (table.i[r] == 1 and table.j[r] == 2)} for r in rows]

    path, tally = lazy_search(find_path, table, check_rows)
    assert path == [0, 3, 2]
    assert tally["edges_blocked"] == 1


def test_small_batches_are_spread_over_all_workers():
    assert spread_size(8, 4, 4) == 2      # 8 edges, 4 workers: 2 each, not 4 + 4 on two workers
    assert spread_size(3, 8, 4) == 1      # fewer edges than workers: one each
    assert spread_size(500, 8, 4) == 4    # a big batch keeps the usual task size
    assert spread_size(0, 8, 4) == 1


def test_start_selection():
    starts = [{"start_index": k, "rank": k, "anchor": ("back", "horizontal")[k % 2],
               "difficulty": ("free", "blocked", "broken")[k % 3]} for k in range(30)]
    goal = {"goal_index": 0, "starts": starts}
    assert [s["start_index"] for s in choose_starts(goal, "0,12", None)] == [0, 12]
    assert all(s["difficulty"] == "free" for s in choose_starts(goal, "free", None))
    spread = spread_starts(starts, 6)
    assert len(spread) == 6 and spread[0]["start_index"] == 0
    assert {s["difficulty"] for s in spread} == {"free", "blocked", "broken"}
    with pytest.raises(ValueError):
        choose_starts(goal, "easy", None)


def test_twin_configurations_match_physically():
    a = np.zeros(12)
    b = a.copy()
    b[0] = 2 * np.pi
    assert wrapped_match(a, b)
    b[1] = 0.01
    assert not wrapped_match(a, b)


def test_json_files_round_trip(tmp_path):
    for name in ("x.json", "x.json.gz"):
        path = str(tmp_path / name)
        write_json_atomic(path, {"a": np.arange(3), "b": np.inf, "c": np.float32(1.5)})
        assert read_json(path) == {"a": [0, 1, 2], "b": None, "c": 1.5}


def test_copied_dashboard_stamps_every_import_with_a_content_hash(tmp_path):
    """A change deep in the module tree must change the address of every module above it.

    This is the cache bug of 2026-10-13: the entry script was stamped, but its
    bare import of charts.js was not, so the browser ran an old cached copy.
    """
    js_dir = tmp_path / "static" / "js"
    js_dir.mkdir(parents=True)
    (js_dir / "leaf.js").write_text("export const a = 1;\n", encoding="utf-8")
    (js_dir / "middle.js").write_text('import { a } from "./leaf.js";\nexport const b = a;\n', encoding="utf-8")
    (js_dir / "entry.js").write_text("import { b } from './middle.js';\nimport * as THREE from \"three\";\n", encoding="utf-8")
    (tmp_path / "index.html").write_text('<script type="module" src="./static/js/entry.js"></script>\n', encoding="utf-8")

    stamp_page_assets(str(tmp_path), ("index.html",), ("static/js/entry.js", "static/css/missing.css"))
    first = {name: (js_dir / name).read_text(encoding="utf-8") for name in ("leaf.js", "middle.js", "entry.js")}
    assert '"./leaf.js?v=' in first["middle.js"]
    assert "'./middle.js?v=" in first["entry.js"]
    assert 'from "three"' in first["entry.js"]  # third-party imports are left alone
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert 'src="./static/js/entry.js?v=' in html and "missing.css" not in html

    # * Changing the leaf changes the stamp in the middle module, which changes the entry's.
    (js_dir / "leaf.js").write_text("export const a = 2;\n", encoding="utf-8")
    (js_dir / "middle.js").write_text('import { a } from "./leaf.js";\nexport const b = a;\n', encoding="utf-8")
    (js_dir / "entry.js").write_text("import { b } from './middle.js';\nimport * as THREE from \"three\";\n", encoding="utf-8")
    (tmp_path / "index.html").write_text('<script type="module" src="./static/js/entry.js"></script>\n', encoding="utf-8")
    stamp_page_assets(str(tmp_path), ("index.html",), ("static/js/entry.js",))
    second = {name: (js_dir / name).read_text(encoding="utf-8") for name in ("middle.js", "entry.js")}
    assert second["middle.js"] != first["middle.js"] and second["entry.js"] != first["entry.js"]
    assert (tmp_path / "index.html").read_text(encoding="utf-8") != html

    # * A cycle has no consistent stamping order and is reported, not guessed.
    (js_dir / "leaf.js").write_text('import { b } from "./middle.js";\nexport const a = 1;\n', encoding="utf-8")
    with pytest.raises(ValueError, match="cycle"):
        stamp_module_imports(str(js_dir))


def test_the_package_never_imports_the_planner_core():
    """Importing the core empties its log file, which a running batch may be writing."""
    code = ("import sys; import husky_assembly_tamp.motion_planner.holding_prm.pilot; "
            "import husky_assembly_tamp.motion_planner.holding_prm.refine; "
            "import husky_assembly_tamp.motion_planner.holding_prm.online; "
            "import husky_assembly_tamp.motion_planner.holding_prm.scenes; "
            "print([m for m in sys.modules if 'dual_arm_task_space_rrt' in m])")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip().splitlines()[-1] == "[]"


# * ---- The exact tracker, on a synthetic two-arm model --------------------
# Both arm bases at the world origin and a collision check that never fires:
# enough to test the geometry and the chain of solves without a scene (the
# analytical solvers and their forward kinematics need no simulator).


def _synthetic_model() -> HoldingModel:
    """Two arms on one base; the right flange holds the bar 15 cm along the bar's y axis."""
    grasp_left = np.eye(4)
    grasp_right = np.eye(4)
    grasp_right[:3, 3] = [0.0, 0.15, 0.0]
    return HoldingModel(world_from_base={"left": np.eye(4), "right": np.eye(4)},
                        grasp={"left": grasp_left, "right": grasp_right},
                        grasp_inv={"left": invert_transform(grasp_left), "right": invert_transform(grasp_right)},
                        limits=arm_limits(), home_conf12=np.zeros(12), joint_names_12=[f"j{n}" for n in range(12)])


def _exact_node(model: HoldingModel, q_left: np.ndarray, seed_right: np.ndarray) -> np.ndarray:
    """Drive the left arm, solve the right arm (branch nearest the seed): a node that holds exactly."""
    world_from_bar = bar_from_arm(model, "left", q_left)
    branches = arm_branches(model, "right", tool_target_from_bar(model, "right", world_from_bar),
                            seed_right, allow_rescue=False, counters={})
    assert branches, "the synthetic pose should be reachable by the right arm"
    return np.concatenate([q_left, branches[0]])


@pytest.fixture(scope="module")
def synthetic():
    """The synthetic model, tools that never collide, and two exact nodes about 8 degrees apart."""
    model = _synthetic_model()
    tools = WorkerTools(model, lambda conf: False)
    q_left = np.array([0.3, -1.2, 1.5, -1.8, -1.57, 0.2])
    qa = _exact_node(model, q_left, q_left)
    qb = _exact_node(model, q_left + np.radians([6.0, -5.0, 8.0, 4.0, -3.0, 5.0]), qa[6:])
    return model, tools, qa, qb


def test_pose_interpolation_ends_and_middle():
    a = np.eye(4)
    b = matrix_from_pose([[0.2, 0.0, 0.0], [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]])  # 90 deg about z
    assert np.allclose(interpolate_pose(a, b, 0.0), a) and np.allclose(interpolate_pose(a, b, 1.0), b)
    middle = interpolate_pose(a, b, 0.5)
    distance, angle = pose_gap(a, middle)
    assert distance == pytest.approx(0.1) and angle == pytest.approx(np.pi / 4)


def test_target_bar_pose_agrees_for_a_holding_configuration(synthetic):
    model, _tools, qa, _qb = synthetic
    poses = [target_bar_pose(model, qa, lead) for lead in ("mid", "left", "right")]
    for pose in poses[1:]:
        distance, angle = pose_gap(poses[0], pose)
        assert distance < 1e-6 and angle < 1e-6


def test_tracker_walks_an_edge_exactly(synthetic):
    """Every waypoint holds the bar, neighbours pass the gate, and the chain starts at the start node."""
    model, tools, qa, qb = synthetic
    record = track_edge(tools, qa, qb, TrackSettings(), counters={})
    assert record["free"] and record["reason"] == "ok"
    confs = np.asarray(record["confs"])
    assert len(confs) == record["n_waypoints"] >= 3
    for conf in confs:
        pos, rot = holding_violation(model, conf)
        assert pos < 1e-6 and rot < 1e-6
    assert np.max(np.abs(np.diff(confs, axis=0))) <= STEP_MAX_RAD
    assert np.max(np.abs(confs[0] - qa)) < 1e-6
    # The bar is never more than one position step from where the edge says it is.
    assert record["max_dev_rad"] < np.radians(10.0)


def test_tracker_does_not_need_exact_seeds(synthetic):
    """Seeded a few degrees off the start node, the tracker still returns exact waypoints."""
    model, tools, qa, qb = synthetic
    rng = np.random.default_rng(1)
    seed = qa + np.radians(rng.uniform(-3.0, 3.0, 12))
    record = track_edge(tools, qa, qb, TrackSettings(), seed=seed, counters={})
    assert record["free"]
    for conf in record["confs"]:
        pos, rot = holding_violation(model, conf)
        assert pos < 1e-6 and rot < 1e-6
    # Nodes only (no knots): a straight bar move between the two nodes.
    straight = track_edge(tools, qa, qb, TrackSettings(knot_step_deg=0.0), counters={})
    assert straight["free"] and straight["n_knots"] == 2


def test_tracker_reports_a_collision(synthetic):
    model, _tools, qa, qb = synthetic
    always = WorkerTools(model, lambda conf: True)
    record = track_edge(always, qa, qb, TrackSettings(), counters={})
    assert not record["free"] and record["reason"] == "collision" and record["confs"] == []


def test_tube_draws_stay_near_the_path(synthetic):
    model, tools, qa, qb = synthetic
    points = interpolate_confs(qa, qb, np.radians(2.5), include_ends=True)
    config = SamplerConfig(anchors=points, anchor_roles=["tube"] * len(points), mode_mix={"path_tube": 1.0},
                           box_margin_rad=0.0, gauss_sigma_rad=0.0, line_sigma_rad=0.0, max_branches=1,
                           tube_points=points, tube_weights=np.ones(len(points)), tube_sigma_rad=np.radians(2.0))
    again = SamplerConfig.from_dict(config.to_dict())  # what a worker receives
    nodes, counters = sample_chunk(tools, again, seed=0, chunk_index=0, n_draws=20)
    assert counters["draws"] == 20 and nodes
    for node in nodes:
        conf = np.asarray(node["conf"])
        pos, rot = holding_violation(model, conf)
        assert pos < 1e-4 and rot < 1e-4
        assert node["mode"] == "path_tube" and node["anchor"] == -1
        part = slice(0, 6) if node["driver"] == "left" else slice(6, 12)
        # The driven arm is drawn within a few sigma of some path point.
        assert np.min(np.max(np.abs(points[:, part] - conf[part]), axis=1)) < np.radians(12.0)


def test_assembly_reads_reversed_edges_and_skips_the_shared_node():
    """Edges tracked i -> j are read backwards when a route walks them j -> i."""
    table = _table([(0, 1, 1.0, 1.0, True, True), (2, 1, 1.0, 1.0, True, True)])
    tracker = Tracker.__new__(Tracker)  # no workers: assembly must not need a re-track here
    tracker.table = table
    tracker.row_of = table.index_of()
    tracker.track_settings = TrackSettings()
    tracker.retracks = 0
    tracker.seeded = {}
    a, b, c = np.zeros(12), np.full(12, 0.01), np.full(12, 0.02)
    tracker.confs = np.stack([a, b, c])  # the node configurations (node 0, 1, 2)
    # Row 1 is stored 2 -> 1 with its own walk 1 -> 2 (not just the same waypoints reversed).
    tracker.records = {0: {"free": True, "confs": [a.tolist(), ((a + b) / 2).tolist(), b.tolist()]},
                       1: {"free": True, "confs": [c.tolist(), b.tolist()],
                           "confs_back": [b.tolist(), ((b + c) / 2 + 0.001).tolist(), c.tolist()]}}
    confs, info = tracker.assemble([0, 1, 2])
    assert info["joins"] == 0 and info["failed_row"] is None
    assert np.allclose(confs[0], a) and np.allclose(confs[-1], c) and len(confs) == 5
    assert np.allclose(confs[3], (b + c) / 2 + 0.001)  # the route used the 1 -> 2 walk

    # * The first edge now ends on "another branch" (0.5 rad off node 1): the
    # * second edge must be continued from there, by a seeded re-track (here
    # * already remembered, so no worker is needed).
    off = b + 0.5
    tracker.records[0]["confs"][-1] = off.tolist()
    elsewhere = [off.tolist(), (off + 0.01).tolist()]
    tracker.seeded[(1, 1, 2, tuple(np.round(off, 6)))] = {"free": True, "confs": elsewhere}
    confs, info = tracker.assemble([0, 1, 2])
    assert info["joins"] == 1 and np.allclose(confs[-1], off + 0.01)

    # * If that continuation fails, the edge that left the branch (row 0) is blamed, not row 1.
    tracker.seeded[(1, 1, 2, tuple(np.round(off, 6)))] = {"free": False, "confs": []}
    confs, info = tracker.assemble([0, 1, 2])
    assert confs is None and info["failed_row"] == 1 and info["blame_row"] == 0

    # * An edge that tracked only the other way cannot be used in this direction: it is blamed itself.
    tracker.records[0]["confs"][-1] = b.tolist()
    tracker.records[1]["confs_back"] = []
    confs, info = tracker.assemble([0, 1, 2])
    assert confs is None and info["failed_row"] == 1 and info["blame_row"] == 1


# * ---- Planning in a changing scene, without a scene ------------------------
# The roadmap's verdicts cover only the robot, its tools and the carried bar.
# These tests use made-up waypoints and a stand-in for the worker processes.


class _FakeScenePool:
    """Stands in for the workers: a configuration "touches the scene" when its first joint is above 0.5."""

    workers = 2

    def run(self, job, tasks, label, progress=None):
        """Answer scene-check tasks the way ``parallel.task_check`` would."""
        return [{"chunk": task["chunk"], "counters": new_counters(),
                 "collides": [conf[0] > 0.5 for conf in task["confs"]]} for task in tasks]


def _scene_tracker(table, records):
    """A tracker whose edges all come from the overnight file and still wait for the scene check."""
    tracker = Tracker.__new__(Tracker)
    tracker.table = table
    tracker.row_of = table.index_of()
    tracker.pool = _FakeScenePool()
    tracker.counters = {}
    tracker.records = records
    tracker.scene_check = True
    tracker.scene_pending = set(records)
    tracker.scene_blocked = []
    tracker.asked_rows = set()
    return tracker


def test_scene_check_drops_only_the_direction_that_touches_the_scene():
    clear, touching = [0.0] * 12, [0.9] + [0.0] * 11
    table = _table([(0, 1, 1.0, 1.0, True, False), (1, 2, 1.0, 1.0, True, False)])
    # Row 0: the walk 0 -> 1 is clear, the walk back touches the scene. Row 1: both touch.
    records = {0: {"row": 0, "free": True, "confs": [clear, clear], "confs_back": [clear, touching, clear]},
               1: {"row": 1, "free": True, "confs": [clear, touching], "confs_back": [touching, clear]}}
    tracker = _scene_tracker(table, records)
    tracker.scene_check_rows([0, 1])
    assert records[0]["free"] and records[0]["confs"] and records[0]["confs_back"] == []
    assert not records[1]["free"] and records[1]["reason"] == "scene_collision"
    assert tracker.scene_blocked == [1] and tracker.scene_pending == set()
    assert list(table.free) == [True, False] and all(table.checked)


def test_lazy_search_goes_around_an_edge_the_scene_blocks():
    """The short route 0-1-2 was fine overnight, but a built bar now sits on edge 1-2."""
    clear, touching = [0.0] * 12, [0.9] + [0.0] * 11
    table = _table([(0, 1, 1.0, 1.0, True, False), (1, 2, 1.0, 1.0, True, False),
                    (0, 3, 2.0, 1.0, True, False), (3, 2, 2.0, 1.0, True, False)])
    records = {row: {"row": row, "free": True, "confs": [clear, clear], "confs_back": [clear, clear]}
               for row in range(4)}
    records[1]["confs"] = [clear, touching, clear]
    records[1]["confs_back"] = [clear, touching, clear]
    tracker = _scene_tracker(table, records)

    def find_path(mask):
        found = shortest_between_sets(build_graph(table, mask, table.len_l2, [0], [2]))
        return found[0] if found else None

    path, tally = lazy_search(find_path, table, tracker.track_rows)
    assert path == [0, 3, 2]
    assert tally["edges_blocked"] == 1 and tracker.scene_blocked == [1]
    # Only the edges a route wanted were sent to the scene check.
    assert tally["edges_checked"] == 4 and tracker.scene_pending == set()
    assert tracker.asked_rows == {0, 1, 2, 3}


def _carrying_state(grasp_shift: float = 0.0) -> RobotCellState:
    """A state with a bar on the left flange, a joint half riding along, an arm tool and a built bar."""
    grasp = Frame([grasp_shift, 0.0, 0.1], [1, 0, 0], [0, 1, 0])
    bodies = {
        "bar_B1": RigidBodyState(Frame.worldXY(), attached_to_link="left_ur_arm_tool0", attachment_frame=grasp,
                                 touch_links=["left_ur_arm_wrist_3_link"], touch_bodies=["joint_a", "bar_B0"]),
        "joint_a": RigidBodyState(Frame.worldXY(), attached_to_link="left_ur_arm_tool0",
                                  attachment_frame=Frame.worldXY(), touch_bodies=["bar_B1"]),
        "bar_B0": RigidBodyState(Frame.worldXY()),  # built, stands in the scene
    }
    tools = {"gripper_left": ToolState(Frame.worldXY(), attached_to_group="left_arm",
                                       touch_links=["left_ur_arm_wrist_3_link"])}
    return RobotCellState(Frame.worldXY(), None, tools, bodies)


def test_a_roadmap_built_for_another_load_is_refused():
    model = SimpleNamespace(grasp={"right": np.eye(4)})
    signature = robot_side_signature(_carrying_state(), "bar_B1")
    # The built bar is not part of the signature (it belongs to the scene, not to the robot).
    assert sorted(signature["bodies"]) == ["bar_B1", "joint_a"]
    assert signature["bodies"]["bar_B1"]["touch_bodies"] == ["joint_a"]
    roadmap = SimpleNamespace(meta={"export": {"signature": signature}}, model=model)

    # Same load: accepted, also after the scene around the robot changed.
    check_roadmap_fits(roadmap, _carrying_state(), "bar_B1", np.eye(4))
    grown = _carrying_state()
    grown.rigid_body_states["bar_B7"] = RigidBodyState(Frame.worldXY())
    check_roadmap_fits(roadmap, grown, "bar_B1", None)

    # The bar is grasped 2 cm further along: refused, saying what differs.
    with pytest.raises(RoadmapMismatch, match="bar_B1: attachment frame differs"):
        check_roadmap_fits(roadmap, _carrying_state(grasp_shift=0.02), "bar_B1", None)
    # Another right grasp than the roadmap's: refused.
    other_grasp = np.eye(4)
    other_grasp[0, 3] = 0.01
    with pytest.raises(RoadmapMismatch, match="right grasp differs"):
        check_roadmap_fits(roadmap, _carrying_state(), "bar_B1", other_grasp)
    # A joint half is missing from what the robot carries: refused.
    lighter = _carrying_state()
    del lighter.rigid_body_states["joint_a"]
    with pytest.raises(RoadmapMismatch, match="joint_a: missing in the scene"):
        check_roadmap_fits(roadmap, lighter, "bar_B1", None)
    # A roadmap from a benchmark library carries no signature: refused rather than trusted.
    with pytest.raises(RoadmapMismatch):
        check_roadmap_fits(SimpleNamespace(meta={}, model=model), _carrying_state(), "bar_B1", None)


def test_scene_cache_keys_tell_geometry_from_placement():
    """The large cell file is reused when only the base moved; a reshaped bar gets a new one."""
    def cell_with_bar(length: float) -> SimpleNamespace:
        bar = RigidBody.from_mesh(Mesh.from_shape(Box(length, 0.02, 0.02)))
        return SimpleNamespace(rigid_body_models={"bar_B1": bar}, tool_models={"gripper_left": None})

    state = _carrying_state()
    moved = _carrying_state()
    moved.robot_base_frame = Frame([0.05, 0.0, 0.0], [1, 0, 0], [0, 1, 0])
    assert geometry_key(cell_with_bar(1.0)) == geometry_key(cell_with_bar(1.0))
    assert geometry_key(cell_with_bar(1.0)) != geometry_key(cell_with_bar(1.2))
    assert state_key(state, "bar_B1") == state_key(_carrying_state(), "bar_B1")
    assert state_key(state, "bar_B1") != state_key(moved, "bar_B1")


def test_an_unknown_planning_backend_is_refused():
    """The check comes before any planner or scene is touched, so none is needed here."""
    with pytest.raises(ValueError, match="backend must be one of"):
        plan_constrained_dual_arm(None, None, active_bar_id="B1", goal_conf=np.zeros(12), backend="rrt-connect")
    with pytest.raises(ValueError, match="stage 3"):
        plan_constrained_dual_arm(None, None, active_bar_id="B1", goal_conf=np.zeros(12), backend="prm", stage=2)


# * ---- Pieces that need the smoke scene -----------------------------------

ROOT = os.environ.get("HUSKY_BENCH_ROOT")
SMOKE_LIBRARY = os.path.join(ROOT or "", "library", "smoke", "bars", "B3.json")
needs_scene = pytest.mark.skipif(not (ROOT and os.path.isfile(SMOKE_LIBRARY)),
                                 reason="needs HUSKY_BENCH_ROOT with the smoke library")


@pytest.fixture(scope="module")
def smoke():
    """The smoke B3 scene, its holding model, and its library entry."""
    entry = load_library_entry(ROOT, "smoke", "B3")
    # The trimmed scene the smoke library was built from (several exports may be cached).
    source_key = read_json(os.path.join(ROOT, "library", "smoke", "index.json"))["source_key"]
    problem, cell, template = load_trimmed_problem(ROOT, source_key, "B3")
    scene = open_scene(cell, template, problem)
    tools = WorkerTools(build_holding_model(scene.robot_puid, problem), make_collision_fn(scene))
    yield scene, tools, entry
    scene.close()


def _library_confs(entry):
    goal = entry["goals"][0]
    confs = [goal["primary_conf"]] + [p["conf"] for p in goal["free_branch_pairs"]]
    return np.asarray(confs + [s["conf"] for s in goal["starts"][:10]], dtype=float)


@needs_scene
def test_library_configurations_hold_the_bar(smoke):
    _scene, tools, entry = smoke
    for conf in _library_confs(entry):
        pos, rot = holding_violation(tools.model, conf)
        assert pos < 1e-4 and rot < 1e-4


@needs_scene
def test_numpy_forward_kinematics_matches_pybullet(smoke):

    scene, tools, entry = smoke
    for conf in _library_confs(entry)[:5]:
        for arm, link, part in (("left", scene.tool_link_left, slice(0, 6)),
                                ("right", scene.tool_link_right, slice(6, 12))):
            expected = np.asarray(tform_from_pose(scene.fk_link_pose(conf, link)))
            assert np.abs(tool0_world(tools.model, arm, conf[part]) - expected).max() < 1e-5


@needs_scene
def test_endpoint_solver_finds_the_library_goal_pairs(smoke):
    _scene, tools, entry = smoke
    goal = entry["goals"][0]
    found = expand_bar_pose(tools, matrix_from_pose(goal["world_from_bar"]),
                            [("given", goal["primary_conf"]), ("home", entry["problem"]["home_conf12"])], {})
    for pair in goal["free_branch_pairs"]:
        assert any(wrapped_match(f["conf"], pair["conf"]) for f in found)


@needs_scene
def test_sampled_nodes_hold_the_bar_and_edges_are_measured(smoke):
    _scene, tools, entry = smoke
    anchors = _library_confs(entry)
    config = SamplerConfig(anchors=anchors, anchor_roles=["goal"] * 2 + ["start"] * (len(anchors) - 2),
                           mode_mix={"anchor_gauss": 1.0}, box_margin_rad=1.0, gauss_sigma_rad=0.3,
                           line_sigma_rad=0.1, max_branches=1)
    nodes, counters = sample_chunk(tools, config, seed=0, chunk_index=0, n_draws=12)
    again, _ = sample_chunk(tools, config, seed=0, chunk_index=0, n_draws=12)
    assert [n["conf"] for n in nodes] == [n["conf"] for n in again]  # same seed, same nodes
    assert counters["draws"] == 12 and len(nodes) == counters["kept"]
    for node in nodes:
        pos, rot = holding_violation(tools.model, node["conf"])
        assert pos < 1e-4 and rot < 1e-4
    if len(nodes) >= 2:
        record = evaluate_edge(tools, 0, 1, np.asarray(nodes[0]["conf"]), np.asarray(nodes[1]["conf"]),
                               collision_step_rad=STEP_MAX_RAD, viol_step_rad=np.radians(2.5),
                               rot_deg_per_mm=0.4, check_collision=True, counters={})
        assert record["checked"] and record["viol_pos_max"] >= 0.0


# * ---- The collision split, on an exported problem with built bars ----------
# One process holds one simulator scene, so each scene is opened by the helper
# program next to this file and its verdicts are read back.

EXPORT_DIR = os.environ.get("HUSKY_EXPORT_PROBLEM_DIR") or os.path.normpath(os.path.join(
    ROOT or "", "..", "..", "data_design_study", "260920_RobArch_demo_revamp"))
SPLIT_BAR = "B4"  # its scene has two built bars, their joints and a support robot


def _has_export() -> bool:
    """Whether the export and the trimmed scene of the test bar are on this machine."""
    if not (ROOT and os.path.isfile(os.path.join(EXPORT_DIR, "RobotCell.json"))):
        return False
    return os.path.isdir(os.path.join(ROOT, "cells", problem_source_key(EXPORT_DIR), SPLIT_BAR))


def _verdicts(mode: str, count: int):
    """Run the helper program for one scene and read its verdicts."""
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "collision_split_helper.py")
    result = subprocess.run([sys.executable, helper, mode, EXPORT_DIR, SPLIT_BAR, str(count)],
                            cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    line = [row for row in result.stdout.splitlines() if row.startswith("RESULT ")][-1]
    return loads(line[len("RESULT "):])


@pytest.mark.skipif(not _has_export(), reason="needs the exported problem and its trimmed scenes (build-cells)")
def test_collision_split_adds_up_to_the_full_check():
    """What the roadmap reuse rests on, checked on a real scene.

    1. The full check says "collides" exactly when the robot-side check or the
       scene-only check does (nothing is lost or invented by splitting).
    2. The scene the roadmap is precomputed in (base at the origin, nothing
       built) gives the same verdict as the robot-side check in the real
       scene at the real base pose (so precomputed verdicts carry over).
    """
    real = _verdicts("real", 80)
    trimmed = _verdicts("trimmed", 80)
    assert [a or b for a, b in zip(real["robot_side"], real["scene_only"])] == real["full"]
    assert trimmed["full"] == real["robot_side"]
    # The sample must exercise both parts, or the test proves nothing.
    assert any(real["robot_side"]) and any(real["scene_only"]) and not all(real["full"])


# * ---- The scene file the roadmap page draws -------------------------------

ROBARCH_B4_RUN = os.path.join(ROOT or "", "prm", "260920_RobArch_demo_revamp__B4")


def _a_cached_scene():
    """The cell and state files of one scene an online plan was made in, or None."""
    for state_path in sorted(glob.glob(os.path.join(ROBARCH_B4_RUN, "scenes", "*", "state_*.json"))):
        cell_path = os.path.join(os.path.dirname(state_path), "cell.json")
        if os.path.isfile(cell_path):
            return cell_path, state_path
    return None


@pytest.mark.skipif(not (ROOT and _a_cached_scene()), reason="needs a cached scene of the B4 roadmap run")
def test_scene_glb_holds_the_bodies_in_the_robots_frame(tmp_path):
    """The glb lists every standing body, the carried joint halves by their link, and the support robots."""
    from compas.data import json_load

    cell_path, state_path = _a_cached_scene()
    cell, state = json_load(cell_path), json_load(state_path)
    out = str(tmp_path / "scene.glb")
    counts = export_scene_glb(cell, state, "bar_B4", out)
    standing = [n for n, b in state.rigid_body_states.items() if not (b.attached_to_link or b.attached_to_tool)]
    carried = [n for n, b in state.rigid_body_states.items() if b.attached_to_link and n != "bar_B4"]
    assert counts["standing"] == len(standing) and counts["carried"] == len(carried)
    assert counts["support_robots"] == sum(1 for t in state.tool_states.values() if not t.attached_to_group)
    # The glb's JSON chunk names one node per carried mesh, with the link it rides on.
    with open(out, "rb") as handle:
        handle.read(12)
        length = struct.unpack("<I", handle.read(4))[0]
        handle.read(4)
        scene_json = loads(handle.read(length).decode("utf-8"))
    names = [node["name"] for node in scene_json["nodes"]]
    links = {state.rigid_body_states[n].attached_to_link for n in carried}
    assert all(any(name.startswith(f"C-{link}-") for name in names) for link in links)
    assert sum(name.startswith("S-") for name in names) >= len(standing)
