"""Keep the carries of a plan set next to their roadmaps, as the carries the monitor loads the bars for.

For every ``<bar>_J.solved_motion.<tag>.json`` of a plan set (``plan_process_carry.py``) it writes
``<root>/prm/<process>__<bar>/carry.json`` (``benchmark/saved_carry.py``), with the station and the drive path from
the plan summary next to the solved file. The same file ``plan_process_carry.py --save-carry`` writes at plan time.

Run from the repository root::

    .venv/bin/python scripts/prm_save_carries.py <root>/prm/plans/<process>/<set> --process <process folder name>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from husky_assembly_tamp.benchmark.saved_carry import carry_from_plan, write_carry  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    """Parse the command line and write one carry per solved file; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("plans_dir", help="the plan set's folder (holds <bar>_J.solved_motion.<tag>.json)")
    parser.add_argument("--process", required=True, help="the process folder's name (the roadmaps' run prefix)")
    parser.add_argument("--tag", default="prm", help="the plans' tag in the file names")
    parser.add_argument("--prm-root", default="",
                        help="the benchmark data root holding prm/<run> (default: HUSKY_BENCH_ROOT)")
    parser.add_argument("--allow-unchecked", action="store_true",
                        help="also keep carries whose loading posture was not checked for the drive in")
    args = parser.parse_args(argv)

    root = args.prm_root or os.environ.get("HUSKY_BENCH_ROOT")
    if not root:
        print("[X] give --prm-root (or set HUSKY_BENCH_ROOT): the folder holding prm/<run>.")
        return 1
    solved_files = sorted(glob.glob(os.path.join(args.plans_dir, f"*.solved_motion.{args.tag}.json")))
    if not solved_files:
        print(f"[X] no *.solved_motion.{args.tag}.json in {args.plans_dir}")
        return 1
    kept = 0
    for solved in solved_files:
        with open(solved, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        summary_file = os.path.splitext(solved)[0] + ".plan_info.json"
        summary = None
        if os.path.isfile(summary_file):
            with open(summary_file, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
        carry = carry_from_plan(record, summary, args.process)
        bar = carry["action"].rsplit("_", 1)[0]
        roadmap_dir = os.path.join(root, "prm", f"{args.process}__{bar}")
        if not os.path.isdir(roadmap_dir):
            print(f"[!] {bar}: no roadmap folder {roadmap_dir}; skipped")
            continue
        if carry["station"] is None and not args.allow_unchecked:
            print(f"[!] {bar}: its plan has no drive-in check (no station); skipped (--allow-unchecked keeps it)")
            continue
        station = "no station" if carry["station"] is None else "station ({:.2f}, {:.2f}) m, {:.0f} deg".format(
            *carry["station"])
        print(f"[save] {bar}: {len(carry['path'])} rows, {station} -> {write_carry(roadmap_dir, carry)}")
        kept += 1
    print(f"{kept} of {len(solved_files)} carries kept")
    return 0


if __name__ == "__main__":
    sys.exit(main())
