"""Where roadmap results live on disk, and how they are written and read.

Everything goes under ``<root>/prm/``, next to (never inside) the benchmark's
own ``runs/`` folder, so a roadmap run can never overwrite a planner batch or
the batch catalogue (``runs.json``) that the benchmark dashboard reads:

    <root>/prm/index.json                    every roadmap run (only this tool writes it)
    <root>/prm/<run>/config.json             build settings, anchors, tallies, timings, provenance
    <root>/prm/<run>/roadmap.json.gz         the roadmap: model, nodes, edges
    <root>/prm/<run>/queries/index.json      one line per query that was run
    <root>/prm/<run>/queries/<tag>.json.gz   one query: endpoints, their edges, gaps, paths
    <root>/prm/<run>/dashboard/              the roadmap page and its scripts

Every write goes to a temporary file first and is then renamed into place,
so a reader (the dashboard, a sync client) never sees half a file.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
from time import time
from typing import Any, Dict, List, Optional

import numpy as np

from husky_assembly_tamp.benchmark.asset_versions import stamp_page_assets
from husky_assembly_tamp.benchmark.provenance import provenance
from husky_assembly_tamp.keyframe.ssik_inprocess import describe_artifact_set
from husky_assembly_tamp.motion_planner.holding_prm.holding import HoldingModel
from husky_assembly_tamp.motion_planner.holding_prm.roadmap import EDGE_COLUMNS, NODE_COLUMNS, Roadmap


SCHEMA_VERSION = 1

# * How many decimals each stored column keeps. Joint values to a micro-radian
# * and positions to a hundredth of a millimetre are far below anything the
# * planner can tell apart, and the rounding roughly halves the file.
NODE_DECIMALS = {"conf": 6, "tool_left": 5, "tool_right": 5, "ee_mid": 5, "bar_pos": 5,
                 "bar_quat": 6, "viol_pos": 8, "viol_rot": 8}
EDGE_DECIMALS = {"len_l2": 5, "len_linf": 5, "t_collide": 3, "viol_pos_max": 7,
                 "viol_rot_max": 7, "t_viol_max": 3}


# -----------------------------------------------------------------------------
# * Paths
# -----------------------------------------------------------------------------


def prm_root(root: str) -> str:
    """The folder all roadmap runs live in: ``<root>/prm``."""
    return os.path.join(root, "prm")


def run_dir(root: str, run_name: str) -> str:
    """The folder of one roadmap run: ``<root>/prm/<run_name>``."""
    return os.path.join(prm_root(root), run_name)


def roadmap_path(root: str, run_name: str) -> str:
    """The roadmap file of a run."""
    return os.path.join(run_dir(root, run_name), "roadmap.json.gz")


def query_path(root: str, run_name: str, tag: str) -> str:
    """The file of one query of a run."""
    return os.path.join(run_dir(root, run_name), "queries", f"{tag}.json.gz")


# -----------------------------------------------------------------------------
# * Reading and writing JSON
# -----------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """Turn numpy values (anywhere inside lists and dicts) into plain Python ones."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        # JSON has no infinity; the dashboard reads null as "cannot reach".
        return None if not np.isfinite(value) else value
    return value


def write_json_atomic(path: str, payload: Any) -> None:
    """Write JSON (gzipped when the name ends in .gz) through a temporary file.

    Args:
        path (str): destination.
        payload (Any): anything JSON can hold; numpy values are converted.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    text = json.dumps(_plain(payload), separators=(",", ":"))
    if path.endswith(".gz"):
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
            handle.write(text)
    else:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(text)
    os.replace(temporary, path)


def read_json(path: str) -> Any:
    """Read a JSON file, gzipped or not.

    Args:
        path (str): the file.

    Returns:
        Any: its contents.

    Raises:
        FileNotFoundError: with the path, when the file is missing.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def rounded_columns(columns: Dict[str, list], decimals: Dict[str, int]) -> Dict[str, Any]:
    """Round the numeric columns that have a precision set; pass the rest through.

    Args:
        columns (Dict[str, list]): column name -> values.
        decimals (Dict[str, int]): column name -> decimals to keep.

    Returns:
        Dict[str, Any]: the columns, ready for JSON.
    """
    out = {}
    for name, values in columns.items():
        if name in decimals and len(values):
            out[name] = np.round(np.asarray(values, dtype=float), decimals[name]).tolist()
        else:
            out[name] = _plain(values)
    return out


# -----------------------------------------------------------------------------
# * The roadmap file
# -----------------------------------------------------------------------------


def save_roadmap(root: str, run_name: str, roadmap: Roadmap) -> str:
    """Write a roadmap to ``<root>/prm/<run>/roadmap.json.gz``.

    Args:
        root (str): the benchmark data root.
        run_name (str): the run.
        roadmap (Roadmap): what to write.

    Returns:
        str: the file written.
    """
    path = roadmap_path(root, run_name)
    write_json_atomic(path, {
        "schema_version": SCHEMA_VERSION,
        "meta": roadmap.meta,
        "config": roadmap.config,
        "model": roadmap.model.to_dict(),
        "anchors": roadmap.anchors,
        "nodes": rounded_columns(roadmap.nodes, NODE_DECIMALS),
        "edges": rounded_columns(roadmap.edges, EDGE_DECIMALS),
        "counters": roadmap.counters,
        "timings": roadmap.timings,
    })
    return path


def load_roadmap(root: str, run_name: str) -> Roadmap:
    """Read a roadmap written by :func:`save_roadmap`.

    Args:
        root (str): the benchmark data root.
        run_name (str): the run.

    Returns:
        Roadmap: the roadmap.
    """
    data = read_json(roadmap_path(root, run_name))
    roadmap = Roadmap(model=HoldingModel.from_dict(data["model"]))
    roadmap.nodes = {name: list(data["nodes"].get(name, [])) for name in NODE_COLUMNS}
    roadmap.edges = {name: list(data["edges"].get(name, [])) for name in EDGE_COLUMNS}
    roadmap.anchors = data.get("anchors", [])
    roadmap.config = data.get("config", {})
    roadmap.counters = data.get("counters", {})
    roadmap.timings = data.get("timings", {})
    roadmap.meta = data.get("meta", {})
    return roadmap


# -----------------------------------------------------------------------------
# * Indexes: the list of runs, and each run's list of queries
# -----------------------------------------------------------------------------


def update_index(path: str, key: str, entry: Dict[str, Any], list_name: str) -> None:
    """Add or replace one entry in a small JSON index file.

    Args:
        path (str): the index file.
        key (str): the field that identifies an entry (e.g. ``run_name``).
        entry (Dict[str, Any]): the entry.
        list_name (str): the list inside the file (``runs`` or ``queries``).
    """
    index = read_json(path) if os.path.isfile(path) else {"schema_version": SCHEMA_VERSION, list_name: []}
    rows = [row for row in index.get(list_name, []) if row.get(key) != entry.get(key)]
    rows.append(entry)
    index[list_name] = sorted(rows, key=lambda row: str(row.get(key)))
    write_json_atomic(path, index)


def update_run_index(root: str, run_name: str, entry: Dict[str, Any]) -> None:
    """Record a run in ``<root>/prm/index.json``."""
    update_index(os.path.join(prm_root(root), "index.json"), "run_name",
                 {"run_name": run_name, "updated": time(), **entry}, "runs")


def update_query_index(root: str, run_name: str, entry: Dict[str, Any]) -> None:
    """Record a query in ``<root>/prm/<run>/queries/index.json``."""
    update_index(os.path.join(run_dir(root, run_name), "queries", "index.json"), "tag",
                 {"updated": time(), **entry}, "queries")


def list_queries(root: str, run_name: str) -> List[Dict[str, Any]]:
    """The queries recorded for a run (empty when none has been run)."""
    path = os.path.join(run_dir(root, run_name), "queries", "index.json")
    return read_json(path).get("queries", []) if os.path.isfile(path) else []


# -----------------------------------------------------------------------------
# * Provenance
# -----------------------------------------------------------------------------


def provenance_record() -> Dict[str, Any]:
    """Where a result came from: commits, dirty flags, versions, machine.

    Uses the benchmark's own ``provenance`` (standard library only, so safe to
    import). It shells out to git, so a failure is recorded rather than
    allowed to stop a long run.

    Returns:
        Dict[str, Any]: the record, or ``{"error": ...}``.
    """
    try:
        return provenance({"calibrated": describe_artifact_set(None)})
    except Exception as error:  # noqa: BLE001 - provenance must never kill a run
        return {"error": f"provenance unavailable: {error!r}"}


# -----------------------------------------------------------------------------
# * The dashboard page
# -----------------------------------------------------------------------------


def dashboard_source_dir() -> str:
    """The checked-in dashboard folder (``<repo>/benchmark/dashboard``)."""
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(os.path.dirname(package_dir), "benchmark", "dashboard")


def copy_dashboard(root: str, run_name: str) -> str:
    """Copy the roadmap page and the shared scripts into the run's dashboard folder.

    The page ``prm.html`` is copied twice, as itself and as ``index.html``,
    so the folder's own address opens it and the module checker
    (``benchmark/tools/check_modules.mjs``, which looks for ``index.html``)
    can check it. The script and stylesheet addresses, and the imports inside
    the copied modules, get a content hash appended, so a browser never runs
    a stale cached copy (see ``benchmark/asset_versions.py``).

    Args:
        root (str): the benchmark data root.
        run_name (str): the run.

    Returns:
        str: the dashboard folder.

    Raises:
        FileNotFoundError: when the checked-in dashboard files are missing.
    """
    source = dashboard_source_dir()
    page = os.path.join(source, "prm.html")
    if not os.path.isfile(page):
        raise FileNotFoundError(
            f"The roadmap page is missing from {source}. It belongs to the same "
            "checkout as this package (benchmark/dashboard/prm.html)."
        )
    out_dir = os.path.join(run_dir(root, run_name), "dashboard")
    os.makedirs(out_dir, exist_ok=True)
    shutil.copytree(os.path.join(source, "static"), os.path.join(out_dir, "static"), dirs_exist_ok=True)
    for name in ("prm.html", "index.html"):
        shutil.copyfile(page, os.path.join(out_dir, name))
    # * Content hashes on the page's assets AND on the imports inside the copied
    # * modules: the entry script's hash alone left the browser free to reuse an
    # * old cached charts.js (seen 2026-10-13: "svg.setCursor is not a function").
    stamp_page_assets(out_dir, ("prm.html", "index.html"), ("static/css/index.css", "static/js/prm-viewer.js"))
    return out_dir


def optional_float(value: Optional[float]) -> Optional[float]:
    """A float, or None for a missing or infinite value (JSON has no infinity)."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None
