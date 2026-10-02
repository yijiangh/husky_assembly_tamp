"""Checks for the holding-constraint roadmap pilot (``motion_planner.holding_prm``).

The first group needs no robot. The second opens the smoke cell of the
benchmark data root in this process (no files are written there) and is
skipped when the root is not available (set ``HUSKY_BENCH_ROOT``). Run with::

    .venv/Scripts/python.exe -m pytest benchmark/tests/test_holding_prm.py -q
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
from pybullet_planning import tform_from_pose

from husky_assembly_tamp.benchmark.scene import load_trimmed_problem, open_scene
from husky_assembly_tamp.motion_planner.holding_prm.holding import (
    STEP_MAX_RAD,
    HoldingModel,
    arm_limits,
    bar_from_arm,
    build_holding_model,
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
    sample_chunk,
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
from husky_assembly_tamp.motion_planner.holding_prm.refine import Tracker
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


def test_the_package_never_imports_the_planner_core():
    """Importing the core empties its log file, which a running batch may be writing."""
    code = ("import sys; import husky_assembly_tamp.motion_planner.holding_prm.pilot; "
            "import husky_assembly_tamp.motion_planner.holding_prm.refine; "
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


# * ---- Pieces that need the smoke scene -----------------------------------

ROOT = os.environ.get("HUSKY_BENCH_ROOT")
SMOKE_LIBRARY = os.path.join(ROOT or "", "library", "smoke", "bars", "B3.json")
needs_scene = pytest.mark.skipif(not (ROOT and os.path.isfile(SMOKE_LIBRARY)),
                                 reason="needs HUSKY_BENCH_ROOT with the smoke library")


@pytest.fixture(scope="module")
def smoke():
    """The smoke B3 scene, its holding model, and its library entry."""
    entry = load_library_entry(ROOT, "smoke", "B3")
    cells = os.path.join(ROOT, "cells")
    problem, cell, template = load_trimmed_problem(ROOT, sorted(os.listdir(cells))[0], "B3")
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
