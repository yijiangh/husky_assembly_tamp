"""Explain a batch's continuity stops from the command line.

    .venv\\Scripts\\python.exe benchmark\\tools\\e0_continuity.py --name pilot1_e0

Reads ``runs/<name>/trees/*.attempts.json.gz`` under the benchmark data root
(``--root`` or ``HUSKY_BENCH_ROOT``), prints the three-way classification and
writes ``runs/<name>/continuity_diag.json``. The same thing is available as
``benchmark_cdfm.py diagnose-continuity`` once the command line grows it.
"""
from __future__ import annotations

import argparse
import os
import sys

# The package lives two folders up (external/husky_assembly_tamp); make it
# importable when this script is run directly from the repo checkout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from husky_assembly_tamp.benchmark.continuity_diag import (  # noqa: E402
    diagnose_run,
    format_report,
    write_diagnosis,
)


def main(argv=None) -> int:
    """Run the diagnosis for one batch and print it.

    Args:
        argv: command-line arguments (``None`` = ``sys.argv``).

    Returns:
        int: the exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=None, help="benchmark data root (or HUSKY_BENCH_ROOT)")
    parser.add_argument("--name", required=True, help="the batch to diagnose")
    args = parser.parse_args(argv)
    root = args.root or os.environ.get("HUSKY_BENCH_ROOT")
    if not root:
        parser.error("no benchmark data root: pass --root or set HUSKY_BENCH_ROOT")
    run_dir = os.path.join(root, "runs", args.name)
    diag = diagnose_run(run_dir)
    print(format_report(diag))
    print("written:", write_diagnosis(run_dir, diag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
