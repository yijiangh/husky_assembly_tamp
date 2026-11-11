"""Checks for the rounded corners of a held bar's path (``dual_arm_task_space_rrt.smooth``).

No robot: the geometry is checked on its own, and the joint re-solve of
``round_dual_arm_pose_path`` is replaced by a stand-in. Run with::

    .venv/bin/python -m pytest benchmark/tests/test_round_corners.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt import smooth
from husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.smooth import (
    pose_path_corners,
    round_dual_arm_pose_path,
    round_pose_path,
)

IDENTITY = (0.0, 0.0, 0.0, 1.0)
FEATURES = [np.array([sx * 0.02, sy * 0.02, sz * 0.5]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


def about_z(angle: float):
    """A quaternion (xyzw) turning by ``angle`` about z."""
    return (0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0))


def dense(points, quaternions, step: float = 0.004):
    """Straight pieces through the given poses, cut every ``step`` metres or less."""
    out = [(tuple(points[0]), tuple(quaternions[0]))]
    for (p0, q0), (p1, q1) in zip(zip(points[:-1], quaternions[:-1]), zip(points[1:], quaternions[1:])):
        n = max(1, int(math.ceil(np.linalg.norm(np.subtract(p1, p0)) / step)))
        for k in range(1, n + 1):
            s = k / n
            out.append((tuple(np.add(p0, s * np.subtract(p1, p0))), tuple(smooth._slerp(np.array(q0), np.array(q1), s))))
    return out


def l_path():
    """Half a metre along x, then half a metre along y, turning 60 degrees about z on the second piece."""
    return dense([(0, 0, 1), (0.5, 0, 1), (0.5, 0.5, 1)], [IDENTITY, IDENTITY, about_z(math.radians(60))])


def directions(poses):
    """Each step's direction in (position, rotation vector) space, unit length."""
    out = []
    for (p0, q0), (p1, q1) in zip(poses[:-1], poses[1:]):
        turn = smooth._quat_angle(np.array(q0), np.array(q1))
        step = np.concatenate([np.subtract(p1, p0), [turn * 0.5]])  # the bar's half length turns the corner points
        norm = np.linalg.norm(step)
        if norm > 1e-9:
            out.append(step / norm)
    return out


def test_corners_of_an_l_path():
    poses = l_path()
    corners = pose_path_corners(poses, FEATURES, tolerance=0.001)
    assert corners[0] == 0 and corners[-1] == len(poses) - 1
    assert len(corners) == 3
    assert np.allclose(poses[corners[1]][0], (0.5, 0, 1))


def test_coarse_tolerance_drops_a_small_kink():
    # A 5 mm kink in the middle of a straight metre is a corner at 1 mm, not at 10 mm.
    poses = dense([(0, 0, 1), (0.5, 0.005, 1), (1.0, 0, 1)], [IDENTITY] * 3)
    assert len(pose_path_corners(poses, FEATURES, tolerance=0.001)) == 3
    assert len(pose_path_corners(poses, FEATURES, tolerance=0.01)) == 2


def test_rounded_path_keeps_its_ends_and_resolution():
    poses = l_path()
    corners = pose_path_corners(poses, FEATURES, tolerance=0.001)
    rounded, owner = round_pose_path(poses, corners, [0.1], position_res=0.004, rotation_res=0.01,
                                     feature_points=FEATURES)
    assert np.allclose(rounded[0][0], poses[0][0]) and np.allclose(rounded[-1][0], poses[-1][0])
    assert np.allclose(np.abs(np.dot(rounded[-1][1], poses[-1][1])), 1.0)
    assert len(owner) == len(rounded) and {-1, 0} == set(owner)
    for (p0, q0), (p1, q1) in zip(rounded[:-1], rounded[1:]):
        assert np.linalg.norm(np.subtract(p1, p0)) <= 0.004 + 1e-9
        assert smooth._quat_angle(np.array(q0), np.array(q1)) <= 0.01 + 1e-9


def test_rounded_path_has_no_corner():
    poses = l_path()
    corners = pose_path_corners(poses, FEATURES, tolerance=0.001)
    sharp, _ = round_pose_path(poses, corners, [0.0], position_res=0.004, rotation_res=0.01, feature_points=FEATURES)
    rounded, _ = round_pose_path(poses, corners, [0.1], position_res=0.004, rotation_res=0.01,
                                 feature_points=FEATURES)

    def sharpest(path):
        d = directions(path)
        return max(math.degrees(math.acos(np.clip(np.dot(a, b), -1, 1))) for a, b in zip(d[:-1], d[1:]))

    assert sharpest(sharp) > 45.0
    # The curve turns the same 90 degrees in many small steps, and joins the straight pieces without a kink.
    assert sharpest(rounded) < 5.0


def test_the_curve_stays_near_the_corner():
    poses = l_path()
    corners = pose_path_corners(poses, FEATURES, tolerance=0.001)
    rounded, owner = round_pose_path(poses, corners, [0.1], position_res=0.004, rotation_res=0.01,
                                     feature_points=FEATURES)
    on_curve = [pose for pose, k in zip(rounded, owner) if k == 0]
    assert on_curve
    # Within the radius of the corner (feature points move at least as far as the bar's centre).
    assert max(np.linalg.norm(np.subtract(p, (0.5, 0, 1))) for p, _q in on_curve) <= 0.1 + 1e-6


def test_radius_is_capped_at_half_a_piece():
    poses = l_path()
    corners = pose_path_corners(poses, FEATURES, tolerance=0.001)
    big, owner = round_pose_path(poses, corners, [10.0], position_res=0.004, rotation_res=0.01,
                                 feature_points=FEATURES)
    # Starts at most half way along the first piece: the first quarter metre stays straight along x.
    straight = [p for (p, _q), k in zip(big, owner) if k == -1]
    assert straight and all(abs(p[1]) < 1e-9 for p in straight if p[0] <= 0.25 - 1e-6)


class FakeRepair:
    """Stands in for ``repair_joint_path``: fails on the waypoints ``bad`` says, else maps poses to joints."""

    def __init__(self, bad):
        self.bad = bad
        self.calls = 0

    def __call__(self, scene, poses, start, goal, **_kwargs):
        self.calls += 1
        for index, pose in enumerate(poses):
            if self.bad(pose):
                return None, {"reason": "collision", "forward_reason": f"collision_at_waypoint_{index}",
                              "backward_reason": None}
        confs = [np.concatenate([pose[0], pose[1], np.zeros(5)]) for pose in poses]
        confs[0], confs[-1] = np.asarray(start), np.asarray(goal)
        return confs, {"success": True}


def joints_for(poses):
    return [np.concatenate([p, q, np.zeros(5)]) for p, q in poses]


def test_round_dual_arm_rounds_when_the_resolve_passes(monkeypatch):
    poses = l_path()
    fake = FakeRepair(lambda pose: False)
    monkeypatch.setattr(smooth, "repair_joint_path", fake)
    profile = {}
    out_poses, out_confs = round_dual_arm_pose_path(poses, joints_for(poses), scene={}, feature_points=FEATURES,
                                                    position_res=0.004, rotation_res=0.01, radius=0.1,
                                                    tolerance=0.001, profile_out=profile)
    assert profile["rounded"] and profile["radii"] == [0.1] and fake.calls == 1
    assert len(out_poses) == len(out_confs)
    assert np.array_equal(out_confs[0], joints_for(poses)[0]) and np.array_equal(out_confs[-1], joints_for(poses)[-1])


def test_a_failing_curve_shrinks_until_it_passes(monkeypatch):
    # Anything more than 3 cm inside the corner collides: the 10 cm curve cuts in further, the small one does not.
    def bad(pose):
        x, y, _z = pose[0]
        return x < 0.5 - 1e-6 and y > 1e-6 and (0.5 - x) + y > 0.03 and x > 0.3 and y < 0.2

    monkeypatch.setattr(smooth, "repair_joint_path", FakeRepair(bad))
    profile = {}
    round_dual_arm_pose_path(l_path(), joints_for(l_path()), scene={}, feature_points=FEATURES,
                             position_res=0.004, rotation_res=0.01, radius=0.1, tolerance=0.001,
                             min_radius=0.005, profile_out=profile)
    assert profile["rounded"]
    assert 0.0 < profile["radii"][0] < 0.1
    assert all(failure["shrunk"] == [0] for failure in profile["failures"])


def test_nothing_rounds_and_the_path_comes_back(monkeypatch):
    poses = l_path()
    monkeypatch.setattr(smooth, "repair_joint_path", FakeRepair(lambda pose: 0.2 < pose[0][0] < 0.5))
    profile = {}
    out_poses, out_confs = round_dual_arm_pose_path(poses, joints_for(poses), scene={}, feature_points=FEATURES,
                                                    position_res=0.004, rotation_res=0.01, radius=0.1,
                                                    tolerance=0.001, profile_out=profile)
    assert not profile["rounded"] and profile["reason"]
    assert out_poses == list(poses) and len(out_confs) == len(poses)


def test_a_nearly_straight_path_becomes_straight(monkeypatch):
    poses = dense([(0, 0, 1), (0.5, 0.005, 1), (1.0, 0, 1)], [IDENTITY] * 3)
    monkeypatch.setattr(smooth, "repair_joint_path", FakeRepair(lambda pose: False))
    profile = {}
    out_poses, _ = round_dual_arm_pose_path(poses, joints_for(poses), scene={}, feature_points=FEATURES,
                                            position_res=0.004, rotation_res=0.01, tolerance=0.01, profile_out=profile)
    assert profile["rounded"] and profile["n_corners"] == 0
    assert max(abs(p[1]) for p, _q in out_poses) < 1e-9


def test_failing_straight_pieces_look_for_corners_more_finely(monkeypatch):
    # The straight line cuts through y < 2 mm in the middle; the path itself bends around it at 5 mm.
    poses = dense([(0, 0, 1), (0.5, 0.005, 1), (1.0, 0, 1)], [IDENTITY] * 3)
    monkeypatch.setattr(smooth, "repair_joint_path", FakeRepair(lambda pose: 0.45 < pose[0][0] < 0.55
                                                                and pose[0][1] < 0.002))
    profile = {}
    round_dual_arm_pose_path(poses, joints_for(poses), scene={}, feature_points=FEATURES, position_res=0.004,
                             rotation_res=0.01, radius=0.01, tolerance=0.01, min_radius=0.002, profile_out=profile)
    assert profile["rounded"] and profile["tolerance"] < 0.01 and profile["n_corners"] == 1


def test_radius_zero_is_off(monkeypatch):
    poses = l_path()
    fake = FakeRepair(lambda pose: False)
    monkeypatch.setattr(smooth, "repair_joint_path", fake)
    profile = {}
    out_poses, _ = round_dual_arm_pose_path(poses, joints_for(poses), scene={}, feature_points=FEATURES,
                                            radius=0.0, profile_out=profile)
    assert fake.calls == 0 and out_poses == list(poses) and not profile["rounded"]


@pytest.mark.parametrize("reason, expected", [("collision_at_waypoint_17", 17), ("continuity_at_waypoint_3", 3),
                                              ("missing_right_grasp", None), (None, None)])
def test_failed_index(reason, expected):
    assert smooth._failed_index(reason) == expected
