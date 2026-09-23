"""Time the benchmark's inner loops, so a batch can be sized before it is run.

Deciding how thorough a start survey to do, or how many goals to sample, is a
decision about hours of compute. This stage measures the few operations that
dominate -- collision checking above all -- and then times a real survey at
several grid resolutions, so the cost of a whole library build is extrapolated
from measurement rather than guessed at.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

import numpy as np
import pybullet_planning as pp

from husky_assembly_tamp.benchmark.scene import load_trimmed_problem, open_scene
from husky_assembly_tamp.benchmark.sweep import DEFAULT_SWEEP_BOX, sweep_goal_starts
from husky_assembly_tamp.motion_planner import ssik_ik
from husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt import core


def median_ms(function, repeats: int) -> float:
    """Return the median wall time of repeated calls, in milliseconds.

    The median rather than the mean, so one scheduling hiccup does not skew
    a number that whole-batch estimates are built on.

    Args:
        function: the callable to time; called with no arguments.
        repeats (int): how many times to call it.

    Returns:
        float: the median duration in milliseconds.
    """
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        function()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def measure_costs(root: str, source_key: str, bar_id: str, *,
                  sweep_steps: str = "0.3,0.2", n_goals: int = 1,
                  repeats: int = 40) -> Dict[str, Any]:
    """Time the operations a benchmark batch is made of, and report.

    Args:
        root (str): the benchmark data root.
        source_key (str): which trimmed-scene cache to use.
        bar_id (str): which bar to measure on.
        sweep_steps (str): comma-separated start-grid steps to time, metres.
        n_goals (int): how many goals to survey at each step.
        repeats (int): how many times to repeat each per-call measurement.

    Returns:
        Dict[str, Any]: the measured per-call costs and per-step survey costs.
    """
    problem, cell, template = load_trimmed_problem(root, source_key, bar_id)
    t0 = time.time()
    scene = open_scene(cell, template, problem)
    open_s = time.time() - t0

    try:
        goal_conf = np.asarray(problem.home_conf12, dtype=float)
        world_from_bar_goal = scene.bar_pose_at(goal_conf)
        plain = scene.collision_fn()
        reporting = scene.reporting_collision_fn()

        reachable = pp.multiply(world_from_bar_goal, scene.grasp_bar_from_left)
        far_away = ((10.0, 10.0, 10.0), (0.0, 0.0, 0.0, 1.0))

        per_call = {
            "check_collision_ms": median_ms(lambda: plain(goal_conf), repeats),
            "check_collision_full_report_ms": median_ms(
                lambda: reporting(goal_conf), repeats),
            "set_robot_cell_state_ms": median_ms(
                lambda: scene.set_state(template), repeats),
            "ssik_branches_reachable_ms": median_ms(
                lambda: ssik_ik.ssik_arm_branches(
                    scene.robot_puid, "left", reachable, goal_conf[0:6],
                    max_solutions=8), repeats),
            "ssik_branches_unreachable_ms": median_ms(
                lambda: ssik_ik.ssik_arm_branches(
                    scene.robot_puid, "left", far_away, goal_conf[0:6],
                    max_solutions=8, allow_rescue=False), repeats),
            "validate_bar_pose_ms": median_ms(
                lambda: core.validate_dual_arm_bar_pose(
                    robot=scene.robot_puid, arm_joints=scene.arm_joints,
                    tool_link_left=scene.tool_link_left,
                    tool_link_right=scene.tool_link_right,
                    full_conf=goal_conf, bar_pose=world_from_bar_goal,
                    grasp_bar_from_left=scene.grasp_bar_from_left,
                    grasp_bar_from_right=scene.grasp_bar_from_right), repeats),
        }

        print(f"[measure] bar {bar_id}, scene opened in {open_s:.2f} s")
        print("[measure] per call, median of "
              f"{repeats}:")
        for name, value in per_call.items():
            print(f"    {name:34s} {value:7.2f} ms")

        steps = [float(part) for part in sweep_steps.split(",") if part.strip()]
        surveys: List[Dict[str, Any]] = []
        print("\n[measure] start survey of one goal, per grid step:")
        print(f"    {'step':>6} {'candidates':>11} {'wall_s':>9} {'ms each':>9} "
              f"{'starts':>8}")
        for step in steps:
            scene.set_state(template)
            survey = sweep_goal_starts(
                scene, goal_conf, world_from_bar_goal,
                sweep_step_m=step, sweep_box=DEFAULT_SWEEP_BOX,
                check_direct_paths=False,
            )
            surveys.append({
                "sweep_step_m": step,
                "n_candidates": survey["n_candidates"],
                "wall_s": survey["wall_s"],
                "ms_per_candidate": survey["wall_s"] / max(1, survey["n_candidates"]) * 1000,
                "n_starts": survey["n_starts"],
                "counts": survey["counts"],
            })
            row = surveys[-1]
            print(f"    {step:6.2f} {row['n_candidates']:11d} {row['wall_s']:9.1f} "
                  f"{row['ms_per_candidate']:9.1f} {row['n_starts']:8d}")

        # The cost is very nearly linear in the number of candidates, so the
        # untimed steps can be extrapolated from the timed ones.
        if surveys:
            ms_each = float(np.median([row["ms_per_candidate"] for row in surveys]))
            print(f"\n[measure] {ms_each:.0f} ms per candidate; a survey costs "
                  "that times the number of candidates")
            print(f"    {'step':>6} {'candidates':>11} {'per goal':>10} "
                  f"{'800 goals, 12 workers':>22}")
            for step in (0.3, 0.2, 0.15, 0.1):
                n_offsets = int(round(0.6 / step) + 1) ** 3
                n_candidates = n_offsets * 3 * 12
                per_goal_s = n_candidates * ms_each / 1000.0
                print(f"    {step:6.2f} {n_candidates:11d} {per_goal_s / 60:9.1f}m "
                      f"{per_goal_s * 800 / 3600 / 12:21.1f}h")
    finally:
        scene.close()

    return {"open_scene_s": open_s, "per_call": per_call, "surveys": surveys}
