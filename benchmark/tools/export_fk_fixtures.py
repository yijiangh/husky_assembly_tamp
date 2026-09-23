"""Record simulator poses for the dashboard's forward-kinematics test.

The dashboard poses the robot itself, in the browser, from the exported
kinematics description. That has to agree with the simulator the planner used,
or every 3D view shows the robot somewhere it never was -- and a wrong pose
looks perfectly plausible on screen.

This writes what the simulator says for a handful of configurations, so
``test_fk.mjs`` can check the browser's answer against it without needing
Python, PyBullet or a browser.

    python benchmark/tools/export_fk_fixtures.py --root <data root> --bar B3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pybullet_planning as pp

# Run from a source checkout without installing anything.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from husky_assembly_tamp.benchmark.glb import matrix_to_quaternion  # noqa: E402
from husky_assembly_tamp.benchmark.scene import (  # noqa: E402
    load_trimmed_problem, open_scene,
)


# Which links to check. The tool flanges are what matters -- the bar hangs off
# them -- and a wrist is included so an error in the middle of the arm is not
# hidden by one that cancels out at the end.
CHECKED_LINKS = (
    "left_ur_arm_tool0",
    "right_ur_arm_tool0",
    "left_ur_arm_wrist_1_link",
)


def export_fixtures(root: str, source_key: str, bar_id: str, out_path: str,
                    *, n_random: int = 6, seed: int = 0) -> Dict[str, Any]:
    """Write the fixtures file.

    Args:
        root (str): the benchmark data root.
        source_key (str): which trimmed-scene cache to use.
        bar_id (str): which bar's scene to load; the robot is the same in all.
        out_path (str): where to write the fixtures.
        n_random (int): how many random configurations to include, on top of
            the all-zero and home ones.
        seed (int): the random seed, so the fixtures are reproducible.

    Returns:
        Dict[str, Any]: the fixtures that were written.
    """
    problem, cell, template = load_trimmed_problem(root, source_key, bar_id)
    kinematics_path = os.path.join(root, "assets", source_key, "kinematics.json")
    if not os.path.isfile(kinematics_path):
        raise FileNotFoundError(
            f"No kinematics description at {kinematics_path}. Export the robot "
            "assets first."
        )
    with open(kinematics_path, "r", encoding="utf-8") as handle:
        kinematics = json.load(handle)

    # The link tree, exactly as the exported scene arranges it: each link with
    # its parent and the fixed offset between them.
    model = cell.robot_model
    links: List[Dict[str, Any]] = []
    parent_of = {joint.child.link: joint.parent.link for joint in model.joints}
    origin_of = {joint.child.link: joint.origin for joint in model.joints}
    for link in model.links:
        frame = origin_of.get(link.name)
        if frame is None:
            translation, rotation = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        else:
            translation = [float(v) for v in frame.point]
            rotation = matrix_to_quaternion(np.array([
                [frame.xaxis[0], frame.yaxis[0], frame.zaxis[0]],
                [frame.xaxis[1], frame.yaxis[1], frame.zaxis[1]],
                [frame.xaxis[2], frame.yaxis[2], frame.zaxis[2]],
            ], dtype=float))
        links.append({
            "name": link.name,
            "parent": parent_of.get(link.name),
            "translation": translation,
            "rotation": rotation,
        })

    scene = open_scene(cell, template, problem)
    try:
        rng = np.random.default_rng(seed)
        cases = [("zero", np.zeros(12)),
                 ("home", np.asarray(problem.home_conf12, dtype=float))]
        for index in range(n_random):
            cases.append((f"random{index}", rng.uniform(-2.5, 2.5, 12)))

        samples = []
        for name, angles in cases:
            entry: Dict[str, Any] = {
                "name": name,
                "angles": [float(v) for v in angles],
                "links": {},
            }
            for link_name in CHECKED_LINKS:
                link_index = pp.link_from_name(scene.robot_puid, link_name)
                with pp.WorldSaver():
                    pp.set_joint_positions(
                        scene.robot_puid, scene.arm_joints, list(angles))
                    position, quaternion = pp.get_link_pose(
                        scene.robot_puid, link_index)
                entry["links"][link_name] = {
                    "position": [float(v) for v in position],
                    "quaternion": [float(v) for v in quaternion],
                }
            samples.append(entry)
    finally:
        scene.close()

    fixtures = {
        "source_key": source_key,
        "bar_id": bar_id,
        "kinematics": kinematics,
        "links": links,
        "samples": samples,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(fixtures, handle, indent=1)
    print(f"[fixtures] {len(samples)} configuration(s), "
          f"{len(CHECKED_LINKS)} link(s) each -> {out_path}")
    return fixtures


def main() -> int:
    """Parse the command line and write the fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.environ.get("HUSKY_BENCH_ROOT"))
    parser.add_argument("--source-key", default=None)
    parser.add_argument("--bar", default="B3")
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures",
        "fk_fixtures.json"))
    args = parser.parse_args()

    if not args.root:
        raise SystemExit("Pass --root or set HUSKY_BENCH_ROOT.")
    source_key = args.source_key
    if not source_key:
        keys = sorted(os.listdir(os.path.join(args.root, "cells")))
        if len(keys) != 1:
            raise SystemExit(f"Say which cache with --source-key: {keys}")
        source_key = keys[0]

    export_fixtures(args.root, source_key, args.bar, os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
