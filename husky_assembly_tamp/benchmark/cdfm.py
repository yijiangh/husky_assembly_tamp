"""Plan one start-to-goal transfer and write down everything about it.

This is the measurement itself. It calls ``plan_constrained_dual_arm`` -- the
production planner, unmodified -- and records what happened.

Each run is planned twice, coarse then fine:

    coarse  10 cm / 0.1 rad steps, 30 s. A quick look at whether any route
            exists at all. Cheap, because every step of the search costs an IK
            solve and a collision check, and coarse steps mean far fewer of
            them.
    fine    2 mm / 0.002 rad steps, 300 s. The resolution the robot actually
            executes at.

A run counts as a success only when both passes find a path. The coarse pass
earns its keep on the failures: a problem with no route at all is rejected in
tens of seconds instead of five minutes. It is not a filter that can wrongly
reject a solvable problem in principle -- a coarse path can always be refined
in principle -- but in practice coarse steps can also miss narrow gaps, so the
two passes are recorded separately and the coarse pass's own success rate is
reported alongside the fine one.

Note that the resolution is not only a step size: the planner also uses it as
the tolerance for having arrived at the goal, so the coarse pass genuinely
solves a looser problem.
"""
from __future__ import annotations

import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from husky_assembly_tamp.benchmark.profiling import profiled
from husky_assembly_tamp.benchmark.scene import SceneCtx, pose_to_lists, state_with_conf12
from husky_assembly_tamp.motion_planner.api import plan_constrained_dual_arm


# * The two passes. `max_iterations` is set absurdly high on purpose: we want
# * wall-clock to be the only thing that stops a search, so timings from
# * different bars stay comparable. `max_attempts=1` for the same reason --
# * restarts are the benchmark's ten seeds, recorded individually, rather than
# * hidden inside one call.
PASS_SETTINGS: Dict[str, Dict[str, float]] = {
    "coarse": {
        "position_res": 0.1,
        "rotation_res": 0.1,
        "max_time": 30.0,
    },
    "fine": {
        "position_res": 0.002,
        "rotation_res": 0.002,
        "max_time": 300.0,
    },
}
PASS_ORDER = ("coarse", "fine")

# The planner logs its two tree sizes and nothing else exposes them, so they
# are read back out of the log line.
TREE_SIZE_PATTERN = re.compile(r"tree_a=(\d+).*?tree_b=(\d+)")

# The logger the planner's search writes to.
PLANNER_LOGGER_NAME = "dual_arm_task_space_rrt_core"


@dataclass
class RunTask:
    """One thing to plan: a bar, a goal, a start and a random seed.

    Attributes:
        bar_id (str): which bar.
        goal_index (int): which sampled goal.
        start_index (int): which feasible start.
        seed (int): the random seed for the search.
        goal_conf (List[float]): the goal configuration.
        start_conf (List[float]): the start configuration.
    """

    bar_id: str
    goal_index: int
    start_index: int
    seed: int
    goal_conf: List[float]
    start_conf: List[float]

    @property
    def run_id(self) -> str:
        """A stable, sortable, filename-safe id for this run."""
        return (f"{self.bar_id}/g{self.goal_index:03d}"
                f"/s{self.start_index:04d}/k{self.seed:02d}")

    @property
    def file_stem(self) -> str:
        """The run id flattened into a single filename component."""
        return (f"g{self.goal_index:03d}_s{self.start_index:04d}"
                f"_k{self.seed:02d}")


class TreeSizeCapture(logging.Handler):
    """Collect the planner's tree sizes out of its log lines.

    The search reports how big each of its two trees grew, but only in a log
    message -- the returned data has the two merged. Rather than change the
    planner, listen to its logger.
    """

    def __init__(self):
        """Start with no sizes captured."""
        super().__init__(level=logging.INFO)
        self.tree_a: Optional[int] = None
        self.tree_b: Optional[int] = None
        self.lines: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Remember any log line that mentions the two tree sizes."""
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a bad log line must not fail a run
            return
        self.lines.append(message)
        match = TREE_SIZE_PATTERN.search(message)
        if match:
            self.tree_a = int(match.group(1))
            self.tree_b = int(match.group(2))


def path_lengths(path: Sequence[Sequence[float]]) -> Dict[str, float]:
    """Measure how much motion a path contains.

    Args:
        path (Sequence[Sequence[float]]): the configurations along the path.

    Returns:
        Dict[str, float]: ``joint_rad`` (total joint travel, summed over all
        twelve) and ``max_step_rad`` (the largest single-step jump, which
        should stay under the planner's continuity threshold).
    """
    if not path or len(path) < 2:
        return {"joint_rad": 0.0, "max_step_rad": 0.0}
    array = np.asarray(path, dtype=float)
    steps = np.abs(np.diff(array, axis=0))
    return {
        "joint_rad": float(steps.sum()),
        "max_step_rad": float(steps.max()),
    }


def plan_one_pass(scene: SceneCtx, task: RunTask, pass_name: str, *,
                  profile_path: Optional[str] = None,
                  enable_profiling: bool = True,
                  store_path: bool = True,
                  overrides: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Run the planner once, at one resolution, and record the result.

    Args:
        scene (SceneCtx): the live scene for this bar.
        task (RunTask): what to plan.
        pass_name (str): ``"coarse"`` or ``"fine"``.
        profile_path (Optional[str]): where to write the flame graph.
        enable_profiling (bool): profile this pass at all.
        store_path (bool): keep the planned configurations in the result.
        overrides (Optional[Dict[str, float]]): replace individual pass
            settings, e.g. a shorter ``max_time`` for a smoke test.

    Returns:
        Dict[str, Any]: the pass record -- see the module docstring for what
        goes into it.
    """
    settings = dict(PASS_SETTINGS[pass_name])
    settings.update(overrides or {})

    start_state = state_with_conf12(
        scene.template_state, task.start_conf, scene.problem.joint_names_12,
    )
    scene.set_state(start_state)

    # Seed the module-level generators too. The planner seeds its own from
    # `random_seed`, but anything it calls that reaches for a global generator
    # should land in the same place on a replay.
    random.seed(task.seed)
    np.random.seed(task.seed % (2 ** 32))

    capture = TreeSizeCapture()
    planner_logger = logging.getLogger(PLANNER_LOGGER_NAME)
    planner_logger.addHandler(capture)

    record: Dict[str, Any] = {"pass": pass_name, "settings": settings}
    t0 = time.perf_counter()
    try:
        with profiled(profile_path, enabled=enable_profiling) as profile:
            path, info = plan_constrained_dual_arm(
                scene.planner,
                start_state,
                active_bar_id=scene.problem.bar_rb_name,
                # ! A plain list, not an array: the planner accepts either a
                # ! named configuration or a bare 12-vector, and it tells them
                # ! apart by indexing with a joint name and catching the error.
                # ! A numpy array raises a different error there and slips
                # ! through as neither.
                goal_conf=[float(value) for value in task.goal_conf],
                stage=3,
                position_res=settings["position_res"],
                rotation_res=settings["rotation_res"],
                max_time=settings["max_time"],
                # Wall-clock is the only stop control; see the module docstring.
                max_iterations=10 ** 7,
                max_attempts=1,
                use_birrt=True,
                enable_smoothing=True,
                random_seed=task.seed,
                derive_start=False,
            )
    finally:
        planner_logger.removeHandler(capture)
    record["wall_s"] = time.perf_counter() - t0

    record["success"] = path is not None
    record["failure_reason"] = info.get("failure_reason")
    record["planner"] = info.get("planner")
    record["ik_backend"] = info.get("ik_backend")
    record["tree_a_nodes"] = capture.tree_a
    record["tree_b_nodes"] = capture.tree_b
    record["profile"] = {
        "breakdown": profile.get("breakdown", {}),
        "speedscope": (os.path.basename(profile["speedscope"])
                       if profile.get("speedscope") else None),
    }

    search_profile = info.get("profile") or {}
    record["iterations"] = search_profile.get("iterations")
    record["extend_stop_reasons"] = dict(search_profile.get("extend_stop_reasons") or {})
    tree = search_profile.get("tree1") or {}
    record["tree_points"] = len(tree.get("points") or [])
    record["tree_edges"] = len(tree.get("edges") or [])

    if path is not None:
        record["n_waypoints"] = len(path)
        record["path_lengths"] = path_lengths(path)
        if store_path:
            record["path"] = [[float(v) for v in conf] for conf in path]
    else:
        record["n_waypoints"] = 0
        record["path_lengths"] = {"joint_rad": 0.0, "max_step_rad": 0.0}
    return record


def run_one(scene: SceneCtx, task: RunTask, *,
            profile_dir: Optional[str] = None,
            enable_profiling: bool = True,
            store_paths: bool = True,
            pass_overrides: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
    """Plan one start-to-goal transfer, coarse pass then fine.

    Args:
        scene (SceneCtx): the live scene for this bar.
        task (RunTask): what to plan.
        profile_dir (Optional[str]): folder for the flame graphs.
        enable_profiling (bool): profile the passes at all.
        store_paths (bool): keep the planned configurations in the result.
        pass_overrides (Optional[Dict[str, Dict[str, float]]]): per-pass
            setting overrides, e.g. ``{"fine": {"max_time": 10}}``.

    Returns:
        Dict[str, Any]: the run record -- the identifying fields, ``success``
        (both passes found a path), ``failed_pass`` when one did not, the two
        pass records under ``passes``, and the total wall time.
    """
    overrides = pass_overrides or {}
    result: Dict[str, Any] = {
        "run_id": task.run_id,
        "bar_id": task.bar_id,
        "goal_index": task.goal_index,
        "start_index": task.start_index,
        "seed": task.seed,
        "status": "ok",
        "passes": {},
        "started_at": time.time(),
        "worker_pid": os.getpid(),
    }

    total_t0 = time.perf_counter()
    failed_pass: Optional[str] = None
    for pass_name in PASS_ORDER:
        profile_path = None
        if profile_dir:
            profile_path = os.path.join(
                profile_dir, f"{task.bar_id}_{task.file_stem}_{pass_name}.speedscope.json",
            )
        record = plan_one_pass(
            scene, task, pass_name,
            profile_path=profile_path,
            enable_profiling=enable_profiling,
            store_path=store_paths,
            overrides=overrides.get(pass_name),
        )
        result["passes"][pass_name] = record
        if not record["success"]:
            # ! No point refining a route that does not exist at this
            # ! resolution -- stop here and let the coarse pass be the verdict.
            failed_pass = pass_name
            break

    result["wall_s"] = time.perf_counter() - total_t0
    result["success"] = failed_pass is None
    result["failed_pass"] = failed_pass
    return result
