"""Command-line entry point for the CDFM planner benchmark.

Run ``python scripts/benchmark_cdfm.py --help`` for the stages. Everything
lives in ``husky_assembly_tamp.benchmark``; this file only makes the package
importable when the repo is used as a source checkout rather than installed.
"""
import os
import sys

# The package sits one level up from scripts/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from husky_assembly_tamp.benchmark.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
