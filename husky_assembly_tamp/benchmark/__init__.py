"""Benchmark harness for the M1 constrained dual-arm transfer planner.

This package measures ``plan_constrained_dual_arm`` -- it does not reimplement
any part of it. Every feasibility decision (IK, the FK gate, collision) and the
motion planning itself are calls into the production functions in
``husky_assembly_tamp.motion_planner``, so a benchmark number always describes
the planner we actually ship.

The work splits into stages, each one a CLI subcommand (see ``cli.py``):

    build-cells    trim the shared robot cell down to one bar per problem
    build-library  sample reachable, collision-free goal bar poses
    sweep-starts   survey every home-anchor candidate for a start config
    run            plan start -> goal many times, with several random seeds
    summarize      aggregate the runs into tables
    export-dashboard  write the files the web dashboard reads

Stages are ordered: each one reads what the previous wrote.
"""
from husky_assembly_tamp.benchmark.sibling_packages import ensure_sibling_packages

# ! Runs on import, before anything here imports compas_fab or
# ! rs_data_structure, so the sibling source checkouts win over any older
# ! copies installed in the venv. See sibling_packages.py for why.
ensure_sibling_packages()
