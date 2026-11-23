"""A bar's carry planned ahead and kept next to its roadmap: ``<roadmap_dir>/carry.json``.

The monitor moves the empty arms to its first row before the bar is mounted (the loading posture), and plans the
live carry with its path as a seed (``prm_options["seed_path"]``).
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from husky_assembly_tamp.benchmark.provenance import repo_state
from husky_assembly_tamp.motion_planner.holding_prm.store import write_json_atomic

CARRY_FILE = "carry.json"
CARRY_FORMAT = "husky_carry/1"


def carry_path(roadmap_dir: str) -> str:
    """The carry file of a roadmap folder."""
    return os.path.join(roadmap_dir, CARRY_FILE)


def carry_from_plan(record: Dict[str, Any], plan_info: Optional[Dict[str, Any]], process_name: str) -> Dict[str, Any]:
    """A carry file's content from a solved-motion record and its plan summary (``plan_process_carry.py``).

    Args:
        record (Dict[str, Any]): the solved file (one solved movement, the carry).
        plan_info (Optional[Dict[str, Any]]): the plan summary next to it; gives the station, the drive path and
            the joints the insert and the retreat end in.
        process_name (str): the process folder's name.

    Returns:
        Dict[str, Any]: ``format``, ``process``, ``action``, ``movement``, ``robot``, ``joint_names``, ``path``
        (twelve joints a row, the loading posture first), ``base`` (the base pose it was planned at: position and
        quaternion, seven numbers), ``loading_class`` (the loading pose's class: "horizontal", "vertical" or
        "back"; None when not known), ``station`` ([x, y, yaw in degrees], or None when the drive in was not
        checked), ``drive_path`` ([[x, y, yaw in radians], ...]), ``insert_joints``, ``retreat_joints``,
        ``solved_against``, ``tamp`` (its commit) and ``saved_at``.

    Raises:
        ValueError: when the record does not hold exactly one solved movement.
    """
    solved = {name: result for name, result in record["movements"].items() if result.get("status") == "solved"}
    if len(solved) != 1:
        raise ValueError(f"{record.get('action')}: expected one solved movement, found {sorted(solved)}")
    movement, result = next(iter(solved.items()))
    trajectory = result["trajectory"]
    robot = trajectory["robot"]
    info = plan_info or {}
    drive = info.get("drive_in") or {}
    follow = info.get("follow_on") or {}
    station = drive.get("station")
    start = (info.get("prm") or {}).get("start") or {}
    return {
        "format": CARRY_FORMAT, "process": process_name, "action": record["action"], "movement": movement,
        "robot": robot, "joint_names": list(trajectory["joint_names"]),
        "path": [[float(value) for value in row] for row in trajectory["positions"]],
        "base": [float(value) for value in result["bases"][robot]],
        "loading_class": start.get("carry") or None,
        "station": None if station is None else [float(station[0]), float(station[1]),
                                                 math.degrees(float(station[2])) % 360.0],
        "drive_path": drive.get("points"),
        "insert_joints": follow.get("insert_joints"), "retreat_joints": follow.get("retreat_joints"),
        "solved_against": record.get("solved_against"),
        "tamp": repo_state(os.path.dirname(os.path.abspath(__file__))),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_carry(roadmap_dir: str, carry: Dict[str, Any]) -> str:
    """Write a carry next to its roadmap; an earlier one is replaced.

    Returns:
        str: the file written.
    """
    path = carry_path(roadmap_dir)
    write_json_atomic(path, carry)
    return path


def read_carry(path: str) -> Dict[str, Any]:
    """Read a carry file.

    Raises:
        FileNotFoundError: no such file.
        ValueError: not a carry file of this format.
    """
    with open(path, "r", encoding="utf-8") as handle:
        carry = json.load(handle)
    if carry.get("format") != CARRY_FORMAT:
        raise ValueError(f"{path}: not a carry file ({carry.get('format')!r}, expected {CARRY_FORMAT!r})")
    return carry
