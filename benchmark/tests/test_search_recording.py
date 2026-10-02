"""Does a recorded search come out whole: every node with a configuration,
every failed attempt pointing at a real node and carrying its cause?

The first tests need no robot. The last one plans a few seconds on the smoke
cell of the benchmark data root and is skipped when that root is not around
(set ``HUSKY_BENCH_ROOT``). Run with::

    .venv/Scripts/python.exe -m pytest benchmark/tests -q
"""
from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pytest

from husky_assembly_tamp.benchmark.cdfm import attempts_path_for, rounded
from husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.core import (
    SearchRecorder,
    collision_pair_names,
    joint_step_deltas,
    joint_step_exceeds_threshold,
)


# * ---- Pieces that need no robot ------------------------------------------

def test_gate_and_report_use_the_same_distance():
    """The joint the report blames must be the one that tripped the gate."""
    current = np.zeros(12)
    candidate = np.zeros(12)
    candidate[7] = 0.3  # one joint jumps by more than 10 degrees
    deltas = joint_step_deltas(candidate, current)
    assert int(np.argmax(deltas)) == 7
    assert joint_step_exceeds_threshold(candidate, current, 0.1745)
    assert not joint_step_exceeds_threshold(candidate, current, 0.5)
    # No wrapping: a full turn is a jump, not a no-op.
    assert joint_step_deltas([2 * np.pi], [0.0])[0] == pytest.approx(2 * np.pi)


def test_collision_pairs_are_named_through_the_cell():
    """Links carry their name; tools and rigid bodies are found by identity."""
    class Link:
        def __init__(self, name):
            self.name = name

    class RigidBody:
        pass

    class Cell:
        tool_models = {}
        rigid_body_models = {}

    bar = RigidBody()
    cell = Cell()
    cell.rigid_body_models = {"bar_7": bar}
    names = collision_pair_names([(Link("left_wrist"), bar)], cell)
    assert names == [["left_wrist", "bar_7"]]
    # Without the cell the body cannot be named, only described.
    fallback = collision_pair_names([(Link("left_wrist"), bar)])
    assert fallback[0][0] == "left_wrist"
    assert isinstance(fallback[0][1], str) and fallback[0][1] != "bar_7"
    assert collision_pair_names(None) == []


def test_recorder_turns_node_identity_into_tree_indices():
    """Attempts refer to nodes by id while searching, by index afterwards."""
    recorder = SearchRecorder()
    recorder.add({"reason": "continuity", "source_id": 11, "last_id": 12, "n_added": 1},
                 tree="a", phase="extend", iteration=3)
    recorder.add({"reason": "collision", "source_id": 21, "last_id": 21, "n_added": 0},
                 tree="b", phase="connect", iteration=4)
    rows = recorder.export({"a": {11: 0, 12: 1}, "b": {21: 5}})
    assert rows[0]["from_node"] == 0 and rows[0]["last_node"] == 1
    assert rows[1]["from_node"] == 5 and rows[1]["last_node"] == 5
    assert rows[1]["phase"] == "connect" and rows[1]["iteration"] == 4
    assert "source_id" not in rows[0]
    recorder.reset()
    assert recorder.attempts == []


def test_attempts_file_sits_next_to_the_tree_file():
    assert attempts_path_for("x/B3_g000_s0000_k00_coarse.tree.json") == (
        "x/B3_g000_s0000_k00_coarse.attempts.json.gz")
    with pytest.raises(ValueError):
        attempts_path_for("x/not_a_tree.json")
    assert rounded({"a": [0.123456, 1], "b": "s"}) == {"a": [0.1235, 1], "b": "s"}


# * ---- A real short search on the smoke cell ------------------------------

def _bench_root():
    """The benchmark data root, or None when this machine has none."""
    root = os.environ.get("HUSKY_BENCH_ROOT")
    return root if root and os.path.isdir(root) else None


@pytest.mark.skipif(_bench_root() is None, reason="HUSKY_BENCH_ROOT is not set")
def test_a_short_pass_records_trees_and_attempts(tmp_path):
    """Plan four seconds and check everything the dashboard relies on."""
    from husky_assembly_tamp.benchmark.cdfm import RunTask, plan_one_pass
    from husky_assembly_tamp.benchmark.cli import resolve_source_key
    from husky_assembly_tamp.benchmark.library import load_bar_library
    from husky_assembly_tamp.benchmark.scene import load_trimmed_problem, open_scene

    root = _bench_root()
    entry = load_bar_library(root, "smoke", "B3")
    goal = entry["goals"][0]
    start = goal["starts"][0]
    task = RunTask(bar_id="B3", goal_index=0, start_index=start["start_index"], seed=0,
                   goal_conf=goal["primary_conf"], start_conf=start["conf"])
    problem, cell, template = load_trimmed_problem(root, resolve_source_key(root, None), "B3")
    scene = open_scene(cell, template, problem)
    tree_path = str(tmp_path / "B3_g000_s0000_k00_coarse.tree.json")
    try:
        record = plan_one_pass(scene, task, "coarse", enable_profiling=False,
                               store_path=False, tree_path=tree_path,
                               overrides={"max_time": 4.0})
    finally:
        scene.close()

    tree = json.load(open(tree_path, encoding="utf-8"))
    assert tree["version"] == 2
    for name, expected in (("forward", record["tree_a_nodes"]), ("backward", record["tree_b_nodes"])):
        part = tree[name]
        assert len(part["points"]) == expected
        assert len(part["confs"]) == expected and all(len(c) == 12 for c in part["confs"])
        assert len(part["midpoints"]) == expected
        for parent, child in part["edges"]:
            assert 0 <= parent < expected and 0 <= child < expected

    with gzip.open(attempts_path_for(tree_path), "rt", encoding="utf-8") as handle:
        attempts = json.load(handle)
    assert len(attempts) == record["n_attempts"] > 0
    # Every recorded stop is also in the planner's own count.
    counted = sum(n for key, n in record["extend_stop_reasons"].items()
                  if not key.endswith("reached") and not key.startswith("stitch"))
    assert len(attempts) == counted
    for attempt in attempts:
        part = tree["forward" if attempt["tree"] == "a" else "backward"]
        assert 0 <= attempt["from_node"] < len(part["points"])
        assert 0 <= attempt["last_node"] < len(part["points"])
        assert attempt["reason"] in ("continuity", "ik_failure", "collision")
        detail = attempt["detail"]
        if attempt["reason"] == "continuity":
            assert detail["joint"] == int(np.argmax(detail["delta_rad"]))
            assert detail["delta_rad"][detail["joint"]] > detail["threshold_rad"]
            assert len(attempt["rejected_conf"]) == 12
        elif attempt["reason"] == "collision":
            assert detail["check"] in ("robot", "bar_only")
            assert detail["pairs"] is None or all(len(pair) == 2 for pair in detail["pairs"])
        else:
            assert detail["arm"] in ("left", "right", "both", "unknown")
            assert detail["cause"] in ("unreachable", "joint_limit", "fk_gate")
            assert attempt["rejected_conf"] is None
