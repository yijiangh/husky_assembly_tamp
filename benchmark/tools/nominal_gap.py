"""Measure how far the nominal UR5e solver's tool pose sits from the calibrated model.

    .venv\\Scripts\\python.exe benchmark\\tools\\nominal_gap.py [--library cdfm_v1] [--bars B3,B9] [--random 2000]

The ``v1_nominal`` planner variant searches with ssik's prebuilt nominal
solver but checks every solution against the CALIBRATED robot's forward
kinematics (the FK gate). The gate's tolerance therefore has to be wider than
the gap between the two geometries, or every waypoint is thrown away as a
"fk_gate" failure. This prints that gap -- position and orientation, per arm --
over the library's start and goal configurations (the poses a plan actually
visits) and over random in-limit configurations, and suggests tolerances of
about twice the largest gap seen. Set them in ``benchmark/variants.py``.

The gap is measured between the two solver modules' own forward kinematics
in the arm's base-link frame, which is exactly what the FK gate compares
(both solvers get their target in that frame from the live base link pose).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

# The package lives two folders up (external/husky_assembly_tamp); make it
# importable when this script is run directly from the repo checkout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from husky_assembly_tamp.keyframe import ssik_inprocess  # noqa: E402


def pose_gap(t_a: np.ndarray, t_b: np.ndarray) -> tuple:
    """Return the position (m) and rotation (rad) difference of two 4x4 poses."""
    position = float(np.linalg.norm(t_a[:3, 3] - t_b[:3, 3]))
    relative = t_a[:3, :3].T @ t_b[:3, :3]
    cosine = max(-1.0, min(1.0, (np.trace(relative) - 1.0) / 2.0))
    return position, float(math.acos(cosine))


def library_confs(root: str, library_name: str, bars) -> list:
    """Collect every start and goal 12-vector of the chosen bars in a library."""
    confs = []
    folder = os.path.join(root, "library", library_name, "bars")
    names = bars or [name[:-5] for name in sorted(os.listdir(folder)) if name.endswith(".json")]
    for bar_id in names:
        with open(os.path.join(folder, f"{bar_id}.json"), "r", encoding="utf-8") as handle:
            entry = json.load(handle)
        for goal in entry.get("goals", []):
            confs.append(goal["primary_conf"])
            for start in goal.get("starts", []):
                confs.append(start["conf"])
    return confs


def main(argv=None) -> int:
    """Measure the gap and print suggested FK-gate tolerances.

    Args:
        argv: command-line arguments (``None`` = ``sys.argv``).

    Returns:
        int: the exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=None, help="benchmark data root (or HUSKY_BENCH_ROOT)")
    parser.add_argument("--library", default="cdfm_v1")
    parser.add_argument("--bars", default=None, help="comma-separated bar ids (default: all)")
    parser.add_argument("--random", type=int, default=2000,
                        help="random in-limit configurations to add (default 2000)")
    parser.add_argument("--nominal-set", default="nominal", help="artifact set to compare")
    args = parser.parse_args(argv)
    root = args.root or os.environ.get("HUSKY_BENCH_ROOT")
    if not root:
        parser.error("no benchmark data root: pass --root or set HUSKY_BENCH_ROOT")
    bars = [part.strip() for part in args.bars.split(",")] if args.bars else None

    confs = library_confs(root, args.library, bars)
    rng = np.random.default_rng(0)
    limits = {arm: ssik_inprocess.joint_limits(arm) for arm in ("left", "right")}

    def gaps_for(arm: str, joint_sets) -> np.ndarray:
        """Position and rotation gap for every 6-vector given, as an (n, 2) array."""
        rows = []
        for q6 in joint_sets:
            calibrated = ssik_inprocess.fk(arm, q6)
            nominal = ssik_inprocess.fk(arm, q6, args.nominal_set)
            rows.append(pose_gap(calibrated, nominal))
        return np.asarray(rows, dtype=float).reshape(-1, 2)

    worst_pos = 0.0
    worst_rot = 0.0
    for arm, offset in (("left", 0), ("right", 6)):
        from_library = gaps_for(arm, [np.asarray(c, float)[offset:offset + 6] for c in confs])
        random_q = [
            [rng.uniform(lo, hi) for lo, hi in limits[arm]] for _ in range(args.random)
        ]
        from_random = gaps_for(arm, random_q)
        for label, gaps in (("library endpoints", from_library), ("random in-limit", from_random)):
            if not len(gaps):
                continue
            pos, rot = gaps[:, 0], gaps[:, 1]
            print(f"{arm:5s} {label:18s} n={len(gaps):5d}  position gap: median {1000 * np.median(pos):.2f} mm, "
                  f"p99 {1000 * np.percentile(pos, 99):.2f} mm, max {1000 * pos.max():.2f} mm | "
                  f"rotation gap: median {math.degrees(np.median(rot)):.3f} deg, max {math.degrees(rot.max()):.3f} deg")
            worst_pos = max(worst_pos, float(pos.max()))
            worst_rot = max(worst_rot, float(rot.max()))

    print(f"\nlargest gap seen: {1000 * worst_pos:.2f} mm, {math.degrees(worst_rot):.3f} deg")
    print(f"suggested FK-gate tolerances (2x): fk_pos_tolerance = {2 * worst_pos:.4f} m, "
          f"fk_ori_tolerance = {2 * worst_rot:.4f} rad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
