"""Windows-side client for the VAMP constrained-planner sidecar.

Launches ``serve_vamp.py`` inside the WSL2 distro where the forked vamp (with
the ``husky_dual_ur5e`` robot module) is installed, and talks newline-JSON over
stdin/stdout -- the same transport pattern as ``keyframe/ssik_client.py``.

Environment knobs (all optional):
    HUSKY_VAMP_WSL_DISTRO   WSL distro name        (default "Ubuntu-22.04")
    HUSKY_VAMP_PYTHON       Linux-side python path (default "~/mcvamp/venv208/bin/python")

The server file itself is this package's ``serve_vamp.py``; it is reached from
Linux through WSL's /mnt/<drive> mount, so no file copying is needed.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Optional

# * module-level cache: one live sidecar per interpreter (like the ssik client)
_PROC: Optional[subprocess.Popen] = None

_SERVER_PATH = Path(__file__).with_name("serve_vamp.py")


def _windows_to_wsl_path(path: Path) -> str:
    """Translate ``C:\\foo\\bar`` into ``/mnt/c/foo/bar`` for use inside WSL.

    Args:
        path (Path): absolute Windows path.

    Returns:
        str: the same location as a WSL mount path.
    """
    p = path.resolve()
    drive = p.drive.rstrip(":").lower()
    rest = p.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{rest}"


def _launch() -> subprocess.Popen:
    """Start the sidecar and block until its ready handshake.

    Returns:
        subprocess.Popen: live server process with text pipes.

    Raises:
        RuntimeError: when the process dies before the handshake, with its
            stderr attached (no silent fallback).
    """
    distro = os.environ.get("HUSKY_VAMP_WSL_DISTRO", "Ubuntu-22.04")
    linux_python = os.environ.get("HUSKY_VAMP_PYTHON",
                                  "/home/yijiangh/mcvamp/venv208/bin/python")
    cmd = ["wsl", "-d", distro, "--", linux_python,
           _windows_to_wsl_path(_SERVER_PATH)]

    # scrub inherited python config so the Linux interpreter starts clean
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONUTF8"] = "1"

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1, env=env)

    # ! handshake watchdog: a wedged `wsl` session can hang forever without
    # ! producing a byte -- kill it after the timeout so callers get a clear
    # ! error instead of an infinite readline block
    timeout_s = float(os.environ.get("HUSKY_VAMP_HANDSHAKE_TIMEOUT", "120"))
    watchdog = threading.Timer(timeout_s, proc.kill)
    watchdog.start()
    try:
        line = proc.stdout.readline()
    finally:
        watchdog.cancel()
    try:
        handshake = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        handshake = None
    if not handshake or not handshake.get("ready"):
        err = proc.stderr.read() if proc.poll() is not None else ""
        proc.kill()
        raise RuntimeError(
            "vamp sidecar failed to start.\n"
            f"  command: {' '.join(cmd)}\n"
            f"  first line: {line!r}\n"
            f"  stderr: {err[-2000:]}\n"
            "Check that the WSL venv has the forked vamp installed "
            "(see docs/vamp_constrained_planner.md).")
    return proc


def _get_proc() -> subprocess.Popen:
    """Return the cached live sidecar, launching it on first use."""
    global _PROC
    if _PROC is None or _PROC.poll() is not None:
        _PROC = _launch()
    return _PROC


def request(payload: dict) -> dict:
    """Send one JSON request line and read one JSON reply line.

    Args:
        payload (dict): request as documented in ``serve_vamp.py``.

    Returns:
        dict: the server's reply.

    Raises:
        RuntimeError: when the sidecar dies mid-request or replies with
            ``ok: false``.
    """
    proc = _get_proc()
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        err = proc.stderr.read() if proc.poll() is not None else ""
        raise RuntimeError(f"vamp sidecar died mid-request; stderr: {err[-2000:]}")
    reply = json.loads(line)
    if not reply.get("ok", False):
        raise RuntimeError(f"vamp sidecar error: {reply.get('error')}\n"
                           f"{reply.get('traceback', '')}")
    return reply


def shutdown():
    """Ask the sidecar to exit and drop the cached handle."""
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        try:
            _PROC.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            _PROC.stdin.flush()
            _PROC.wait(timeout=5)
        except Exception:  # noqa: BLE001 -- best effort on teardown
            _PROC.kill()
    _PROC = None
