"""Worker processes for the roadmap: sampling, edge checks and endpoint solves.

All three jobs are "the same small computation, thousands of times, each
independent of the others", so they are split into chunks and handed to a
pool of processes. Each process opens its OWN headless PyBullet scene once,
when it starts, and keeps it for every chunk it is given.

Why processes and not threads: ``pybullet_planning`` keeps one global
"current simulation" and the collision check writes joint values into the
simulation, so two threads would trample each other. Separate processes each
have their own simulation.

Why ``ProcessPoolExecutor`` (as ``benchmark.library.build_library`` uses)
rather than a hand-written queue: when a worker dies inside PyBullet's C++
code the executor reports a broken pool at once, instead of waiting forever
for an answer that will never come.

! Everything that crosses into a worker is plain lists, numbers and strings.
! Scenes, closures and numpy generators cannot be sent to another process on
! Windows (processes are started fresh, not copied).
"""
from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from compas.data import json_load

from husky_assembly_tamp.benchmark.scene import BarProblem, load_trimmed_problem, open_scene, quiet_planner_logging
from husky_assembly_tamp.motion_planner.holding_prm.carry import carry_bar_pose, score_carry_pose
from husky_assembly_tamp.motion_planner.holding_prm.holding import (
    HoldingModel,
    build_holding_model,
    geometry_to_lists,
    matrix_from_pose,
    node_geometry,
    pose_from_matrix,
)
from husky_assembly_tamp.motion_planner.holding_prm.roadmap import evaluate_edge
from husky_assembly_tamp.motion_planner.holding_prm.sampling import (
    SamplerConfig,
    WorkerTools,
    expand_bar_pose,
    make_collision_fn,
    make_collision_fns,
    new_counters,
    sample_chunk,
)
from husky_assembly_tamp.motion_planner.holding_prm.tracking import TrackSettings, fine_segment, track_edge_landing
from husky_assembly_tamp.motion_planner.ssik_ik import verify_ssik_joint_order


# * The one scene this process owns, set up by init_worker.
_WORKER: Dict[str, Any] = {}


def init_worker(root: str, source_key: str, bar_id: str, sampler: Optional[Dict[str, Any]],
                edge_settings: Dict[str, float]) -> None:
    """Open this process's scene. Runs once, when the process starts.

    Args:
        root (str): the benchmark data root.
        source_key (str): which trimmed-scene cache to load.
        bar_id (str): which bar's scene.
        sampler (Optional[Dict[str, Any]]): ``SamplerConfig.to_dict()``, or
            None when this pool does not sample.
        edge_settings (Dict[str, float]): ``collision_step_rad``,
            ``viol_step_rad`` and ``rot_deg_per_mm``.
    """
    # Six workers each starting a pool of maths-library threads, next to a
    # benchmark batch that may be running too, would only fight over cores.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    quiet_planner_logging()
    problem, cell, template = load_trimmed_problem(root, source_key, bar_id)
    scene = open_scene(cell, template, problem)
    # The artifact joint order must match the robot's; checked once here.
    verify_ssik_joint_order(scene.robot_puid)
    _WORKER.clear()
    _WORKER.update({
        "scene": scene,
        "tools": WorkerTools(build_holding_model(scene.robot_puid, problem), make_collision_fn(scene)),
        "sampler": SamplerConfig.from_dict(sampler) if sampler else None,
        "edge_settings": dict(edge_settings),
    })


def init_worker_scene(cell_path: str, state_path: str, problem: Dict[str, Any],
                      edge_settings: Dict[str, float]) -> None:
    """Open a real scene in this process, from files. Runs once, when the process starts.

    Unlike :func:`init_worker`, which opens a bar's precompute scene from the
    benchmark cache, this opens whatever cell and state it is pointed at: the
    robot at its real base pose with the built bars around it (see
    ``scenes.build_scene_cell``). The worker then has both the full collision
    check and the scene-only one.

    Args:
        cell_path (str): the scene's cell file (compas JSON).
        state_path (str): its state file.
        problem (Dict[str, Any]): the bar's ``BarProblem`` as a dict (grasps,
            home, joint names).
        edge_settings (Dict[str, float]): see :func:`init_worker`.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    quiet_planner_logging()
    bar_problem = BarProblem.from_dict(problem)
    scene = open_scene(json_load(cell_path), json_load(state_path), bar_problem)
    verify_ssik_joint_order(scene.robot_puid)
    checks = make_collision_fns(scene.planner, scene.template_state, bar_problem.joint_names_12)
    _WORKER.clear()
    _WORKER.update({
        "scene": scene,
        "tools": WorkerTools(build_holding_model(scene.robot_puid, bar_problem), checks["full"],
                             checks["scene_only"]),
        "sampler": None,
        "edge_settings": dict(edge_settings),
    })


def use_tools_in_process(tools: WorkerTools, edge_settings: Dict[str, float]) -> None:
    """Make this process the single worker, on a scene the caller already has open.

    Args:
        tools (WorkerTools): model and collision checks built on the caller's planner.
        edge_settings (Dict[str, float]): see :func:`init_worker`.
    """
    _WORKER.clear()
    _WORKER.update({"scene": None, "tools": tools, "sampler": None, "edge_settings": dict(edge_settings)})


def worker_model() -> HoldingModel:
    """The holding model of this process's scene (used by the parent in single-process mode)."""
    return _WORKER["tools"].model


# -----------------------------------------------------------------------------
# * The jobs a worker can do. Each takes and returns plain data.
# -----------------------------------------------------------------------------


def task_model(task: Dict[str, Any]) -> Dict[str, Any]:
    """Report the holding model, so the parent never has to open a scene itself."""
    return {"chunk": task["chunk"], "model": _WORKER["tools"].model.to_dict(), "counters": {}}


def task_sample(task: Dict[str, Any]) -> Dict[str, Any]:
    """Make the nodes of one chunk of draws.

    Args:
        task (Dict[str, Any]): ``chunk``, ``seed`` and ``n_draws``.

    Returns:
        Dict[str, Any]: ``chunk``, ``nodes`` and ``counters``.
    """
    start = perf_counter()
    nodes, counters = sample_chunk(_WORKER["tools"], _WORKER["sampler"], task["seed"],
                                   task["chunk"], task["n_draws"])
    counters["seconds_worker"] = perf_counter() - start
    return {"chunk": task["chunk"], "nodes": nodes, "counters": counters}


def task_nodes(task: Dict[str, Any]) -> Dict[str, Any]:
    """Check given configurations (anchors): holding, geometry and collision.

    Args:
        task (Dict[str, Any]): ``chunk`` and ``confs`` (lists of 12 values).

    Returns:
        Dict[str, Any]: ``chunk``, ``nodes`` (geometry plus ``collides`` for
        each configuration, in order) and ``counters``.
    """
    start = perf_counter()
    tools: WorkerTools = _WORKER["tools"]
    counters = new_counters()
    nodes = []
    for conf in task["confs"]:
        conf = np.asarray(conf, dtype=float)
        geometry = geometry_to_lists(node_geometry(tools.model, conf))
        nodes.append({"conf": conf.tolist(), "collides": tools.timed_collides(conf, counters), **geometry})
    counters["seconds_worker"] = perf_counter() - start
    return {"chunk": task["chunk"], "nodes": nodes, "counters": counters}


def task_edges(task: Dict[str, Any]) -> Dict[str, Any]:
    """Measure one chunk of edges.

    Args:
        task (Dict[str, Any]): ``chunk``, ``pairs`` ([[i, j], ...]), ``confs_a``,
            ``confs_b`` (the matching configurations) and ``check_collision``.

    Returns:
        Dict[str, Any]: ``chunk``, ``edges`` (one record per pair, in order)
        and ``counters``.
    """
    start = perf_counter()
    settings = _WORKER["edge_settings"]
    counters = new_counters()
    edges = [
        evaluate_edge(_WORKER["tools"], i, j, np.asarray(qa), np.asarray(qb),
                      collision_step_rad=settings["collision_step_rad"],
                      viol_step_rad=settings["viol_step_rad"],
                      rot_deg_per_mm=settings["rot_deg_per_mm"],
                      check_collision=task["check_collision"], counters=counters)
        for (i, j), qa, qb in zip(task["pairs"], task["confs_a"], task["confs_b"])
    ]
    counters["seconds_worker"] = perf_counter() - start
    return {"chunk": task["chunk"], "edges": edges, "counters": counters}


def task_expand(task: Dict[str, Any]) -> Dict[str, Any]:
    """Solve every collision-free branch pair at one bar pose (a query endpoint).

    Args:
        task (Dict[str, Any]): ``chunk``, ``world_from_bar`` (pose lists) and
            ``turn_references`` ([[name, conf12], ...]).

    Returns:
        Dict[str, Any]: ``chunk``, ``confs`` (see ``sampling.expand_bar_pose``),
        their geometry, and ``counters``.
    """
    start = perf_counter()
    tools: WorkerTools = _WORKER["tools"]
    counters = new_counters()
    found = expand_bar_pose(tools, matrix_from_pose(task["world_from_bar"]),
                            [(name, conf) for name, conf in task["turn_references"]], counters)
    for entry in found:
        entry.update(geometry_to_lists(node_geometry(tools.model, np.asarray(entry["conf"]))))
    counters["seconds_worker"] = perf_counter() - start
    return {"chunk": task["chunk"], "confs": found, "counters": counters}


def task_track(task: Dict[str, Any]) -> Dict[str, Any]:
    """Track one chunk of edges exactly (see ``tracking.track_edge_landing``).

    Args:
        task (Dict[str, Any]): ``chunk``, ``rows`` (the edges' table rows, only
            passed back), ``confs_a`` and ``confs_b`` (start and end of each
            edge, in the direction to walk), optional ``seeds`` (one per edge,
            or None per edge: seed the first solve from ``confs_a``), and
            ``settings`` (``TrackSettings.to_dict()``).

    Returns:
        Dict[str, Any]: ``chunk``, ``records`` (one per edge, in order, each
        with its ``row``) and ``counters``.
    """
    start = perf_counter()
    settings = TrackSettings.from_dict(task["settings"])
    seeds = task.get("seeds") or [None] * len(task["rows"])
    counters = new_counters()
    records = []
    for row, qa, qb, seed in zip(task["rows"], task["confs_a"], task["confs_b"], seeds):
        record = track_edge_landing(_WORKER["tools"], np.asarray(qa, dtype=float), np.asarray(qb, dtype=float),
                                    settings, seed=None if seed is None else np.asarray(seed, dtype=float),
                                    counters=counters)
        record["row"] = int(row)
        records.append(record)
    counters["seconds_worker"] = perf_counter() - start
    return {"chunk": task["chunk"], "records": records, "counters": counters}


def task_check(task: Dict[str, Any]) -> Dict[str, Any]:
    """Collision-check a batch of configurations, with the scene-only check or the full one.

    Args:
        task (Dict[str, Any]): ``chunk``, ``confs`` (lists of 12 values) and
            ``which`` ("scene": only against what does not ride on the robot;
            "full": everything).

    Returns:
        Dict[str, Any]: ``chunk``, ``collides`` (one bool per configuration)
        and ``counters``.
    """
    start = perf_counter()
    tools: WorkerTools = _WORKER["tools"]
    counters = new_counters()
    check = tools.timed_scene_collides if task["which"] == "scene" else tools.timed_collides
    collides = [bool(check(np.asarray(conf, dtype=float), counters)) for conf in task["confs"]]
    counters["seconds_worker"] = perf_counter() - start
    return {"chunk": task["chunk"], "collides": collides, "counters": counters}


def task_fine(task: Dict[str, Any]) -> Dict[str, Any]:
    """Walk consecutive exact waypoints at a fine step and collision-check what lies between.

    Args:
        task (Dict[str, Any]): ``chunk``, ``confs`` (consecutive exact
            waypoints of a path; each neighbouring pair is one step),
            ``first_index`` (the path index of ``confs[0]``, only passed back),
            ``position_res_m`` and ``rotation_res_rad`` (the fine step).

    Returns:
        Dict[str, Any]: ``chunk``, ``first_index``, ``steps`` (one record per
        pair, see ``tracking.fine_segment``) and ``counters``.
    """
    start = perf_counter()
    tools: WorkerTools = _WORKER["tools"]
    counters = new_counters()
    confs = [np.asarray(conf, dtype=float) for conf in task["confs"]]
    steps = [fine_segment(tools, a, b, task["position_res_m"], task["rotation_res_rad"], counters)
             for a, b in zip(confs[:-1], confs[1:])]
    counters["seconds_worker"] = perf_counter() - start
    return {"chunk": task["chunk"], "first_index": task["first_index"], "steps": steps, "counters": counters}


def task_carry(task: Dict[str, Any]) -> Dict[str, Any]:
    """Score one chunk of carry-pose candidates (see ``carry.score_carry_pose``).

    Args:
        task (Dict[str, Any]): ``chunk``, ``candidates`` (each with ``carry``,
            ``midpoint`` and ``roll_deg``) and ``with_nudges``.

    Returns:
        Dict[str, Any]: ``chunk``, ``scores`` (one per candidate, in order,
        with the candidate's fields and its bar pose) and ``counters``.
    """
    start = perf_counter()
    tools: WorkerTools = _WORKER["tools"]
    counters = new_counters()
    scores = []
    for candidate in task["candidates"]:
        pose = carry_bar_pose(tools.model, candidate["carry"], candidate["midpoint"],
                              float(np.radians(candidate["roll_deg"])))
        result = score_carry_pose(tools, pose, task["with_nudges"], counters)
        scores.append({**candidate, **result, "world_from_bar": pose_from_matrix(pose)})
    counters["seconds_worker"] = perf_counter() - start
    return {"chunk": task["chunk"], "scores": scores, "counters": counters}


def task_sample_with(task: Dict[str, Any]) -> Dict[str, Any]:
    """Make the nodes of one chunk of draws, with sampling settings sent in the task.

    Unlike :func:`task_sample`, whose settings are fixed when the pool starts,
    this lets one pool sample around a different path each round (the
    refinement's tube draws).

    Args:
        task (Dict[str, Any]): ``chunk`` (position in this batch, for sorting),
            ``chunk_index`` (seeds the random source), ``seed``, ``n_draws``
            and ``sampler`` (``SamplerConfig.to_dict()``).

    Returns:
        Dict[str, Any]: ``chunk``, ``nodes`` and ``counters``.
    """
    start = perf_counter()
    nodes, counters = sample_chunk(_WORKER["tools"], SamplerConfig.from_dict(task["sampler"]), task["seed"],
                                   task["chunk_index"], task["n_draws"])
    counters["seconds_worker"] = perf_counter() - start
    return {"chunk": task["chunk"], "nodes": nodes, "counters": counters}


# -----------------------------------------------------------------------------
# * The pool
# -----------------------------------------------------------------------------


class WorkerPool:
    """A fixed set of processes, each holding one open scene of the same bar.

    Attributes:
        workers (int): how many processes (1 = everything runs in this process,
            which gives full tracebacks when debugging).
        verbose (bool): print one progress line per finished chunk.
    """

    def __init__(self, root: str, source_key: str, bar_id: str, workers: int,
                 sampler: Optional[Dict[str, Any]], edge_settings: Dict[str, float],
                 verbose: bool = True):
        """Start the processes (each opens its scene; this takes a few seconds).

        Args:
            root (str): the benchmark data root.
            source_key (str): trimmed-scene cache key.
            bar_id (str): the bar.
            workers (int): how many processes.
            sampler (Optional[Dict[str, Any]]): sampling settings, or None.
            edge_settings (Dict[str, float]): see :func:`init_worker`.
            verbose (bool): print progress.
        """
        self.workers = max(1, int(workers))
        self.verbose = verbose
        # Whether this pool opened the in-process scene (and so must close it).
        self.owns_scene = True
        self._start(init_worker, (root, source_key, bar_id, sampler, edge_settings))

    def _start(self, initializer: Callable[..., None], init_args: tuple) -> None:
        """Start the processes, or open the scene in this process when there is one worker."""
        start = perf_counter()
        if self.workers == 1:
            initializer(*init_args)
            self.executor = None
        else:
            self.executor = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=get_context("spawn"),
                initializer=initializer, initargs=init_args,
            )
        self.startup_s = perf_counter() - start

    @classmethod
    def from_scene_files(cls, cell_path: str, state_path: str, problem: Dict[str, Any], workers: int,
                         edge_settings: Dict[str, float], verbose: bool = True) -> "WorkerPool":
        """A pool whose workers each open one real scene from files (see :func:`init_worker_scene`).

        Args:
            cell_path (str): the scene's cell file.
            state_path (str): its state file.
            problem (Dict[str, Any]): the bar's problem as a dict.
            workers (int): how many processes (at least 2: a single worker
                would open a second scene in the caller's process; use
                :meth:`in_process` for that).
            edge_settings (Dict[str, float]): see :func:`init_worker`.
            verbose (bool): print progress.

        Returns:
            WorkerPool: the pool.
        """
        if int(workers) < 2:
            raise ValueError("A pool from scene files needs at least 2 workers; use in_process for one.")
        pool = cls.__new__(cls)
        pool.workers, pool.verbose, pool.owns_scene = int(workers), verbose, True
        pool._start(init_worker_scene, (cell_path, state_path, problem, edge_settings))
        return pool

    @classmethod
    def in_process(cls, tools: WorkerTools, edge_settings: Dict[str, float], verbose: bool = True) -> "WorkerPool":
        """A one-worker pool on a scene the caller already has open (never opens a second one).

        Args:
            tools (WorkerTools): model and collision checks on the caller's planner.
            edge_settings (Dict[str, float]): see :func:`init_worker`.
            verbose (bool): print progress.

        Returns:
            WorkerPool: the pool.
        """
        pool = cls.__new__(cls)
        pool.workers, pool.verbose, pool.owns_scene = 1, verbose, False
        pool._start(use_tools_in_process, (tools, edge_settings))
        return pool

    def run(self, job: Callable[[Dict[str, Any]], Dict[str, Any]], tasks: List[Dict[str, Any]],
            label: str, progress: Optional[Callable[[Dict[str, Any]], str]] = None) -> List[Dict[str, Any]]:
        """Run one job over many tasks and return the results in task order.

        Args:
            job (Callable): one of the ``task_*`` functions of this module.
            tasks (List[Dict[str, Any]]): one dict per chunk, each with ``chunk``.
            label (str): shown in the progress lines.
            progress (Optional[Callable]): turns a result into extra progress
                text (e.g. how many nodes so far).

        Returns:
            List[Dict[str, Any]]: the results, sorted by ``chunk``.
        """
        start = perf_counter()
        results: List[Dict[str, Any]] = []
        last_print = 0.0

        def report(result: Dict[str, Any]) -> None:
            nonlocal last_print
            results.append(result)
            # The progress callback sees every result (it may keep a running
            # count); only the printing is rationed.
            extra = f", {progress(result)}" if progress else ""
            now = perf_counter() - start
            # At most one line every two seconds, plus the last one.
            if self.verbose and (now - last_print > 2.0 or len(results) == len(tasks)):
                last_print = now
                print(f"  [{label}] {len(results)}/{len(tasks)} chunks, {now:.0f} s{extra}", flush=True)

        if self.executor is None:
            for task in tasks:
                report(job(task))
        else:
            pending = {self.executor.submit(job, task) for task in tasks}
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    # A worker exception is re-raised here, with its traceback.
                    report(future.result())
        return sorted(results, key=lambda result: result["chunk"])

    def warm_up(self) -> float:
        """Make every worker start and open its scene now; return how long that took (s).

        The executor starts its processes only when work arrives, so without
        this the start-up cost hides inside the first phase's time. A planner
        that runs for a whole session would pay it once, so timings meant to
        show the cost of one query should leave it out.
        """
        start = perf_counter()
        if self.executor is not None:
            # More tasks than workers, so every process gets at least one.
            self.run(task_model, [{"chunk": n} for n in range(2 * self.workers)], "warm up")
        return perf_counter() - start

    def close(self) -> None:
        """Stop the processes (or close the in-process scene)."""
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        elif "scene" in _WORKER:
            # A scene the caller lent us stays open; one this pool opened is closed.
            if self.owns_scene and _WORKER["scene"] is not None:
                _WORKER["scene"].close()
            _WORKER.clear()

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def default_workers() -> int:
    """Six processes, or fewer on a small machine.

    Six leaves room on the 16-thread workstation for a planner batch that
    uses eight, which the other benchmark sessions routinely run.
    """
    return max(1, min(6, (os.cpu_count() or 2) - 1))

