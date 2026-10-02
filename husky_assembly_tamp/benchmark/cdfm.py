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

What a pass leaves on disk, when trees are kept (the default for batches):

    <bar>_<stem>_<pass>.tree.json         both search trees. ``forward`` grew
        from the start, ``backward`` from the goal; each has ``points``
        (bar-frame origin), ``quaternions``, ``midpoints`` (what is drawn),
        ``confs`` (the 12 joint angles holding the bar there) and ``edges``
        as [parent, child] indices into that tree.
    <bar>_<stem>_<pass>.attempts.json.gz  every extension that stopped short,
        gzipped. Per attempt: ``tree`` (a = start tree, b = goal tree),
        ``phase`` (extend / connect), ``iteration``, ``reason``, ``from_node``
        and ``last_node`` (indices into that tree), ``n_added``, ``target``,
        ``rejected_pose``, ``rejected_conf`` and a ``detail`` dict: the joint
        that jumped and by how much (continuity), the arm without a solution
        and whether one exists past a joint limit (ik_failure), or the parts
        that touched (collision).
"""
from __future__ import annotations

import gzip
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from husky_assembly_tamp.benchmark.profiling import profiled
from husky_assembly_tamp.benchmark.scene import SceneCtx, pose_to_lists, state_with_conf12
from husky_assembly_tamp.benchmark.variants import json_safe, resolve_variant
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


def attempts_path_for(tree_path: str) -> str:
    """The attempts file that goes with a tree file.

    Args:
        tree_path (str): ``.../<name>.tree.json``.

    Returns:
        str: ``.../<name>.attempts.json.gz`` in the same folder.
    """
    if not tree_path.endswith(".tree.json"):
        raise ValueError(f"expected a *.tree.json path, got {tree_path!r}")
    return tree_path[:-len(".tree.json")] + ".attempts.json.gz"


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


def rotate(quaternion: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    """Rotate a vector by an ``(x, y, z, w)`` quaternion.

    Args:
        quaternion (Sequence[float]): the rotation.
        vector (Sequence[float]): the vector to rotate.

    Returns:
        np.ndarray: the rotated vector.
    """
    x, y, z, w = (float(v) for v in quaternion)
    u = np.array([x, y, z])
    v = np.asarray(vector, dtype=float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def tree_for_dashboard(tree: Dict[str, Any], bar_from_midpoint: Sequence[float]) -> Dict[str, Any]:
    """Round one exported tree and add the bar midpoints the dashboard draws.

    Each node is the bar's pose at that point of the search, plus the joint
    configuration that holds it there. The bar frame sits at one end of the
    bar, so the midpoint is worked out as well: it is where the bar actually
    is, and what the tree lines are drawn through.

    Args:
        tree (Dict[str, Any]): the planner's ``{"points", "quaternions",
            "confs", "edges"}`` export of one tree.
        bar_from_midpoint (Sequence[float]): the midpoint in bar coordinates.

    Returns:
        Dict[str, Any]: the same keys plus ``midpoints``, all rounded.
    """
    points = tree.get("points") or []
    quaternions = tree.get("quaternions") or []
    return {
        "points": [[round(v, 5) for v in p] for p in points],
        "quaternions": [[round(v, 6) for v in q] for q in quaternions],
        "midpoints": [[round(float(v), 5) for v in
                       np.asarray(p) + rotate(q, bar_from_midpoint)]
                      for p, q in zip(points, quaternions)],
        "confs": [None if c is None else [round(v, 4) for v in c]
                  for c in (tree.get("confs") or [])],
        "edges": tree.get("edges") or [],
    }


def rounded(value: Any, places: int = 4) -> Any:
    """Round every float inside nested lists and dicts, leaving the rest alone.

    Args:
        value (Any): a number, list, dict, or anything else.
        places (int): decimal places to keep.

    Returns:
        Any: the same shape with floats rounded.
    """
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, list):
        return [rounded(v, places) for v in value]
    if isinstance(value, dict):
        return {k: rounded(v, places) for k, v in value.items()}
    return value


def write_attempts(path: str, attempts: List[Dict[str, Any]]) -> None:
    """Write the failed extension attempts of one pass as gzipped JSON.

    A coarse pass makes about two thousand attempts, each with a few poses and
    a configuration; gzip shrinks that from hundreds of KB to tens. The
    dashboard unpacks the file itself in the browser.

    Args:
        path (str): where to write; conventionally ``*.attempts.json.gz``.
        attempts (List[Dict[str, Any]]): the planner's ``attempts`` export.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(rounded(attempts), handle, separators=(",", ":"))


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
                  tree_path: Optional[str] = None,
                  overrides: Optional[Dict[str, float]] = None,
                  variant: Optional[str] = None) -> Dict[str, Any]:
    """Run the planner once, at one resolution, and record the result.

    Args:
        scene (SceneCtx): the live scene for this bar.
        task (RunTask): what to plan.
        pass_name (str): ``"coarse"`` or ``"fine"``.
        profile_path (Optional[str]): where to write the flame graph.
        enable_profiling (bool): profile this pass at all.
        store_path (bool): keep the planned configurations in the result.
        tree_path (Optional[str]): when given, save the search's two trees
            there, whether it solved the problem or not.
        overrides (Optional[Dict[str, float]]): replace individual pass
            settings, e.g. a shorter ``max_time`` for a smoke test.
        variant (Optional[str]): which planner variant to run (see
            ``benchmark.variants``); ``None`` is the baseline.

    Returns:
        Dict[str, Any]: the pass record -- see the module docstring for what
        goes into it.
    """
    settings = dict(PASS_SETTINGS[pass_name])
    settings.update(overrides or {})
    # The variant's settings for THIS pass (a variant may treat the coarse
    # and fine resolutions differently, e.g. a wider gate for the big steps).
    variant_entry = resolve_variant(variant, pass_name)

    start_state = state_with_conf12(
        scene.template_state, task.start_conf, scene.problem.joint_names_12,
    )
    scene.set_state(start_state)

    # Seed the module-level generators too. The planner seeds its own from
    # `random_seed`, but anything it calls that reaches for a global generator
    # should land in the same place on a replay.
    random.seed(task.seed)
    np.random.seed(task.seed % (2 ** 32))

    record: Dict[str, Any] = {
        "pass": pass_name, "settings": settings, "variant": variant_entry["name"],
    }
    t0 = time.perf_counter()
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
            # Every failed extension with its cause, when trees are wanted.
            record_search=bool(tree_path),
            # The variant's solver, gate and repair settings (see variants.py).
            **variant_entry["planner_kwargs"],
        )
    record["wall_s"] = time.perf_counter() - t0

    record["success"] = path is not None
    # * The search alone, before the strict repair of the found path; the
    # * repair's verdict is what separates the two (see api.plan_constrained_dual_arm).
    record["search_success"] = bool(info.get("search_success", path is not None))
    record["postprocess"] = json_safe(info.get("repair"))
    record["failure_reason"] = info.get("failure_reason")
    record["planner"] = info.get("planner")
    record["ik_backend"] = info.get("ik_backend")
    record["profile"] = {
        "breakdown": profile.get("breakdown", {}),
        "speedscope": (os.path.basename(profile["speedscope"])
                       if profile.get("speedscope") else None),
    }

    search_profile = info.get("profile") or {}
    record["iterations"] = search_profile.get("iterations")
    record["extend_stop_reasons"] = dict(search_profile.get("extend_stop_reasons") or {})
    # How many dual-arm IK solves the search made (tree growth, connects and
    # stitches), so the profiler's IK seconds can be turned into ms per solve.
    record["n_ik_solves"] = search_profile.get("n_ik_solves")
    # The planner hands back its two trees separately, with every node's joint
    # configuration. An empty profile means it never got as far as searching
    # (start or goal already in collision), which is a result, not an error.
    tree_a = search_profile.get("tree_a") or {"points": [], "edges": []}
    tree_b = search_profile.get("tree_b") or {"points": [], "edges": []}
    record["tree_a_nodes"] = len(tree_a["points"])
    record["tree_b_nodes"] = len(tree_b["points"])
    record["tree_points"] = record["tree_a_nodes"] + record["tree_b_nodes"]
    record["tree_edges"] = len(tree_a["edges"]) + len(tree_b["edges"])

    if tree_path:
        trees = {
            "version": 2,
            "forward": tree_for_dashboard(tree_a, scene.problem.bar_from_midpoint),
            "backward": tree_for_dashboard(tree_b, scene.problem.bar_from_midpoint),
            "run_id": task.run_id, "pass": pass_name, "success": path is not None,
            "start_pose": pose_to_lists(scene.bar_pose_at(task.start_conf)),
            "goal_pose": pose_to_lists(scene.bar_pose_at(task.goal_conf)),
        }
        os.makedirs(os.path.dirname(tree_path), exist_ok=True)
        with open(tree_path, "w", encoding="utf-8") as handle:
            json.dump(trees, handle, separators=(",", ":"))
        record["tree_file"] = os.path.basename(tree_path)

        # The failed attempts go next to the trees, gzipped: see write_attempts.
        attempts = search_profile.get("attempts")
        if attempts is None and search_profile:
            raise RuntimeError(
                "The planner searched but recorded no attempts although "
                "record_search was on; look into plan_constrained_dual_arm.")
        attempts_path = attempts_path_for(tree_path)
        write_attempts(attempts_path, attempts or [])
        record["attempts_file"] = os.path.basename(attempts_path)
        record["n_attempts"] = len(attempts or [])
        # * How often a growth walk added no node at all, i.e. failed on its
        # * very first waypoint. Growth walks are the "extend" attempts plus
        # * the walks that reached their sample (those are not attempts).
        grow_attempts = [a for a in (attempts or []) if a.get("phase") == "extend"]
        record["n_first_step_failures"] = sum(
            1 for a in grow_attempts if int(a.get("n_added") or 0) == 0)
        record["n_growth_walks"] = (
            len(grow_attempts) + int(record["extend_stop_reasons"].get("reached", 0)))

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
            tree_dir: Optional[str] = None,
            pass_overrides: Optional[Dict[str, Dict[str, float]]] = None,
            variant: Optional[str] = None) -> Dict[str, Any]:
    """Plan one start-to-goal transfer, coarse pass then fine.

    Args:
        scene (SceneCtx): the live scene for this bar.
        task (RunTask): what to plan.
        profile_dir (Optional[str]): folder for the flame graphs.
        enable_profiling (bool): profile the passes at all.
        store_paths (bool): keep the planned configurations in the result.
        tree_dir (Optional[str]): folder to save each pass's search trees in;
            not saved when None.
        pass_overrides (Optional[Dict[str, Dict[str, float]]]): per-pass
            setting overrides, e.g. ``{"fine": {"max_time": 10}}``.
        variant (Optional[str]): which planner variant to run, for both
            passes; ``None`` is the baseline.

    Returns:
        Dict[str, Any]: the run record -- the identifying fields, ``success``
        (both passes found a path), ``search_success`` (both searches found
        one, before the repair), ``failed_pass`` when one did not, the two
        pass records under ``passes``, and the total wall time.
    """
    overrides = pass_overrides or {}
    result: Dict[str, Any] = {
        "run_id": task.run_id,
        "bar_id": task.bar_id,
        "goal_index": task.goal_index,
        "start_index": task.start_index,
        "seed": task.seed,
        "variant": resolve_variant(variant)["name"],
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
        tree_path = None
        if tree_dir:
            tree_path = os.path.join(
                tree_dir, f"{task.bar_id}_{task.file_stem}_{pass_name}.tree.json")
        record = plan_one_pass(
            scene, task, pass_name,
            profile_path=profile_path,
            enable_profiling=enable_profiling,
            store_path=store_paths,
            tree_path=tree_path,
            overrides=overrides.get(pass_name),
            variant=variant,
        )
        result["passes"][pass_name] = record
        if not record["success"]:
            # ! No point refining a route that does not exist at this
            # ! resolution -- stop here and let the coarse pass be the verdict.
            failed_pass = pass_name
            break

    result["wall_s"] = time.perf_counter() - total_t0
    result["success"] = failed_pass is None
    result["search_success"] = all(
        record.get("search_success") for record in result["passes"].values()
    ) and bool(result["passes"])
    result["failed_pass"] = failed_pass
    return result
