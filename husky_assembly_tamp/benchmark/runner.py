"""Run many planning tasks across several processes, and survive their crashes.

Three things make this more than a thread pool.

PyBullet allows one simulation per process, so parallelism has to be by
process, and each worker pays the cost of loading the scene once. Workers
therefore keep their scene open and are fed tasks for the same bar in a row.

A PyBullet crash takes the whole process down with it -- it is a C++ library
and a bad geometry query can abort rather than raise. A pool that hands out
work through a shared queue would simply hang. So the pool tracks which task
each worker is holding, notices when a worker dies, records that task as
crashed, and starts a replacement.

An overnight batch will be interrupted. Every result is written to its own file
as it is produced, and starting again skips the tasks already on disk, so a
batch can be stopped and resumed freely.

Tasks are ordered so that stopping early still leaves a usable dataset: all of
seed 0 across every bar and goal first, then all of seed 1, and so on. Half a
batch is then a complete benchmark at fewer seeds, not a complete benchmark of
half the bars.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from husky_assembly_tamp.benchmark.cdfm import PASS_SETTINGS, RunTask, run_one


# Seconds of slack on top of the planning budget before a worker is assumed
# hung. The budget is both passes' search time; the margin has to cover what
# runs after each search -- the strict repair of the found path and the
# smoothing, each a full walk of the fine path's thousands of waypoints and
# together up to a couple of minutes -- plus scene loading and writing
# results. (A planner variant with a wide coarse gate hit the old 120 s
# margin on a solved run, which was then recorded as a timeout.)
TIMEOUT_MARGIN_S = 300.0

# How often the parent checks on its workers.
POLL_INTERVAL_S = 0.5


def default_task_timeout_s(pass_overrides: Optional[Dict[str, Dict[str, float]]] = None) -> float:
    """Return how long one task may take before it is called hung.

    Args:
        pass_overrides (Optional[Dict[str, Dict[str, float]]]): per-pass
            setting overrides in force for this batch.

    Returns:
        float: the timeout in seconds.
    """
    overrides = pass_overrides or {}
    budget = 0.0
    for name, settings in PASS_SETTINGS.items():
        budget += float(overrides.get(name, {}).get("max_time", settings["max_time"]))
    return budget + TIMEOUT_MARGIN_S


def result_path(runs_dir: str, task: RunTask) -> str:
    """Return the file one task's result is written to.

    Args:
        runs_dir (str): the run's ``runs/`` folder.
        task (RunTask): the task.

    Returns:
        str: the JSON path.
    """
    return os.path.join(runs_dir, task.bar_id, f"{task.file_stem}.json")


def write_result(runs_dir: str, task: RunTask, result: Dict[str, Any]) -> str:
    """Write one result to disk so that it is never seen half-written.

    Args:
        runs_dir (str): the run's ``runs/`` folder.
        task (RunTask): the task the result belongs to.
        result (Dict[str, Any]): the result record.

    Returns:
        str: the path written.
    """
    path = result_path(runs_dir, task)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(result, handle)
    os.replace(temporary, path)
    return path


def order_tasks(tasks: Iterable[RunTask], start_ranks: Dict[Tuple[str, int, int], int]) -> List[RunTask]:
    """Put tasks in the order a batch should work through them.

    Seed first, then how production-like the start is, then goal, then bar. A
    batch stopped part-way has then covered every bar and goal at the lower
    seeds, rather than every seed of the first few bars.

    Args:
        tasks (Iterable[RunTask]): the tasks to order.
        start_ranks (Dict[Tuple[str, int, int], int]): how each start ranked in
            the sweep, keyed by ``(bar_id, goal_index, start_index)``. Rank 0
            is the start production itself would have picked.

    Returns:
        List[RunTask]: the ordered tasks.
    """
    def key(task: RunTask):
        rank = start_ranks.get((task.bar_id, task.goal_index, task.start_index), 10 ** 6)
        return (task.seed, rank, task.goal_index, task.bar_id)

    return sorted(tasks, key=key)


def pending_tasks(tasks: Sequence[RunTask], runs_dir: str) -> List[RunTask]:
    """Drop the tasks that already have a result on disk.

    Args:
        tasks (Sequence[RunTask]): all the tasks in the batch.
        runs_dir (str): the run's ``runs/`` folder.

    Returns:
        List[RunTask]: the ones still to do.
    """
    return [task for task in tasks if not os.path.isfile(result_path(runs_dir, task))]


def _worker_main(task_queue, result_queue, config: Dict[str, Any]) -> None:
    """Run tasks off the queue until told to stop. Runs in a child process.

    The scene is kept open between tasks and only rebuilt when the bar
    changes, because opening one costs seconds and a task costs tens of
    seconds.

    Args:
        task_queue: queue of ``RunTask``; a None means stop.
        result_queue: queue the parent reads ``(kind, payload)`` messages from.
        config (Dict[str, Any]): the batch settings (paths, profiling, and so
            on), passed by value because this is a spawned process.
    """
    # Imported here, not at module import: this runs in a fresh interpreter
    # that has not imported the benchmark package yet.
    from husky_assembly_tamp.benchmark.scene import (
        load_trimmed_problem, open_scene, quiet_planner_logging,
    )

    # A dozen workers would otherwise all truncate and interleave into the
    # planner's one log file.
    quiet_planner_logging()

    scene = None
    scene_bar: Optional[str] = None
    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            result_queue.put(("started", {"run_id": task.run_id, "pid": os.getpid()}))
            try:
                if scene_bar != task.bar_id:
                    if scene is not None:
                        scene.close()
                    problem, cell, template = load_trimmed_problem(
                        config["root"], config["source_key"], task.bar_id,
                    )
                    scene = open_scene(cell, template, problem)
                    scene_bar = task.bar_id

                result = run_one(
                    scene, task,
                    profile_dir=config.get("profile_dir"),
                    enable_profiling=config.get("enable_profiling", True),
                    store_paths=config.get("store_paths", True),
                    tree_dir=config.get("tree_dir"),
                    pass_overrides=config.get("pass_overrides"),
                    # Which planner variant the batch runs; None = the baseline.
                    # Its solver sets were checked in the parent before any
                    # worker started, so a bad name never reaches here.
                    variant=config.get("variant"),
                )
            except Exception:  # noqa: BLE001 - a bad task must not kill the batch
                result = {
                    "run_id": task.run_id,
                    "bar_id": task.bar_id,
                    "goal_index": task.goal_index,
                    "start_index": task.start_index,
                    "seed": task.seed,
                    "status": "error",
                    "success": False,
                    "error": traceback.format_exc(limit=8),
                    "worker_pid": os.getpid(),
                }
                # The scene may be in an unknown state after an error.
                if scene is not None:
                    scene.close()
                scene = None
                scene_bar = None
            result_queue.put(("done", {"task": task, "result": result}))
    finally:
        if scene is not None:
            scene.close()


class BatchRunner:
    """Work through a list of planning tasks on several processes.

    Attributes:
        tasks (List[RunTask]): what is left to do, in order.
        runs_dir (str): where results are written.
        n_workers (int): how many processes to use.
    """

    def __init__(self, tasks: Sequence[RunTask], *, runs_dir: str, config: Dict[str, Any],
                 n_workers: int = 8, task_timeout_s: Optional[float] = None,
                 verbose: bool = True):
        """Set up a batch. Call :meth:`run` to start it.

        Args:
            tasks (Sequence[RunTask]): the tasks, already ordered.
            runs_dir (str): the run's ``runs/`` folder.
            config (Dict[str, Any]): settings passed to each worker.
            n_workers (int): how many processes.
            task_timeout_s (Optional[float]): how long one task may take;
                derived from the planning budget when not given.
            verbose (bool): print progress.
        """
        self.tasks = list(tasks)
        self.runs_dir = runs_dir
        self.config = config
        self.n_workers = max(1, int(n_workers))
        self.task_timeout_s = task_timeout_s or default_task_timeout_s(
            config.get("pass_overrides"),
        )
        self.verbose = verbose

    def run(self) -> Dict[str, Any]:
        """Work through every task and return a summary.

        Returns:
            Dict[str, Any]: counts of what happened and the total wall time.
        """
        if not self.tasks:
            return {"n_tasks": 0, "done": 0, "succeeded": 0, "crashed": 0,
                    "errored": 0, "timed_out": 0, "wall_s": 0.0}

        context = mp.get_context("spawn")
        task_queue = context.Queue()
        result_queue = context.Queue()

        remaining = list(self.tasks)
        for task in remaining:
            task_queue.put(task)

        workers: List[Any] = []
        # What each worker is currently holding, so a death can be attributed.
        in_flight: Dict[int, Tuple[RunTask, float]] = {}
        # Tasks a worker died on; retried once each.
        retried: set = set()

        def spawn_worker():
            """Start one worker process and remember it."""
            process = context.Process(
                target=_worker_main,
                args=(task_queue, result_queue, self.config),
                daemon=True,
            )
            process.start()
            workers.append(process)
            return process

        for _ in range(self.n_workers):
            spawn_worker()
        # One stop signal per worker, queued behind the real work.
        for _ in range(self.n_workers):
            task_queue.put(None)

        counts = {"done": 0, "succeeded": 0, "crashed": 0, "errored": 0, "timed_out": 0}
        n_total = len(remaining)
        wall_times: List[float] = []
        t0 = time.time()
        pid_to_run: Dict[int, str] = {}
        run_id_to_task = {task.run_id: task for task in remaining}

        while counts["done"] < n_total:
            try:
                kind, payload = result_queue.get(timeout=POLL_INTERVAL_S)
            except queue.Empty:
                kind, payload = None, None

            if kind == "started":
                pid_to_run[payload["pid"]] = payload["run_id"]
                task = run_id_to_task.get(payload["run_id"])
                if task is not None:
                    in_flight[payload["pid"]] = (task, time.time())
            elif kind == "done":
                task, result = payload["task"], payload["result"]
                in_flight.pop(result.get("worker_pid", -1), None)
                write_result(self.runs_dir, task, result)
                counts["done"] += 1
                if result.get("status") == "error":
                    counts["errored"] += 1
                elif result.get("success"):
                    counts["succeeded"] += 1
                if "wall_s" in result:
                    wall_times.append(result["wall_s"])
                if self.verbose and counts["done"] % 10 == 0:
                    self._report(counts, n_total, wall_times, t0)

            # A worker that died takes its task with it: record the loss and
            # put a replacement in its place.
            for process in list(workers):
                if process.is_alive():
                    continue
                workers.remove(process)
                lost = in_flight.pop(process.pid, None)
                if lost is not None:
                    task, _started = lost
                    counts["done"] += 1
                    counts["crashed"] += 1
                    write_result(self.runs_dir, task, {
                        "run_id": task.run_id,
                        "bar_id": task.bar_id,
                        "goal_index": task.goal_index,
                        "start_index": task.start_index,
                        "seed": task.seed,
                        "status": "crashed",
                        "success": False,
                        "note": "the worker process died while planning this task",
                    })
                if counts["done"] < n_total:
                    spawn_worker()
                    task_queue.put(None)

            # A worker still holding a task well past the planning budget is
            # stuck; stopping it turns a hang into a recorded timeout.
            now = time.time()
            for pid, (task, started) in list(in_flight.items()):
                if now - started <= self.task_timeout_s:
                    continue
                for process in list(workers):
                    if process.pid != pid:
                        continue
                    process.terminate()
                    in_flight.pop(pid, None)
                    counts["done"] += 1
                    counts["timed_out"] += 1
                    write_result(self.runs_dir, task, {
                        "run_id": task.run_id,
                        "bar_id": task.bar_id,
                        "goal_index": task.goal_index,
                        "start_index": task.start_index,
                        "seed": task.seed,
                        "status": "timeout",
                        "success": False,
                        "note": (f"no result after {self.task_timeout_s:.0f} s, "
                                 "well past the planning budget"),
                    })

        for process in workers:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()

        summary = {"n_tasks": n_total, "wall_s": time.time() - t0, **counts}
        if self.verbose:
            self._report(counts, n_total, wall_times, t0, final=True)
        return summary

    def _report(self, counts: Dict[str, int], n_total: int,
                wall_times: List[float], t0: float, *, final: bool = False) -> None:
        """Print one progress line.

        Args:
            counts (Dict[str, int]): running tallies.
            n_total (int): how many tasks the batch has.
            wall_times (List[float]): per-task wall times so far.
            t0 (float): when the batch started.
            final (bool): this is the closing line.
        """
        done = counts["done"]
        elapsed = time.time() - t0
        mean_wall = sum(wall_times) / len(wall_times) if wall_times else 0.0
        remaining = n_total - done
        eta_s = remaining * mean_wall / self.n_workers if mean_wall else 0.0
        tag = "[run] done" if final else "[run]"
        print(f"{tag} {done}/{n_total}  ok {counts['succeeded']}  "
              f"crashed {counts['crashed']}  timeout {counts['timed_out']}  "
              f"error {counts['errored']}  "
              f"mean {mean_wall:.1f} s  elapsed {elapsed / 60:.1f} min"
              + ("" if final else f"  eta {eta_s / 60:.1f} min"), flush=True)
