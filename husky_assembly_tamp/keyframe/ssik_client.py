"""Client for the ssik analytical-IK sidecar (works inside Rhino AND headless).

ssik needs Python 3.11+ and ships compiled extensions, so it cannot be imported
into Rhino's CPython 3.9 (or an arbitrary offline venv). Instead we run it in a
small sidecar process (a Python 3.11 venv, see ``keyframe/ssik_sidecar/serve.py``)
and talk to it over stdin/stdout using newline-delimited JSON. This module is the
caller-side half: it spawns and caches the sidecar, sends solve requests, and
shuts it down.

Only the standard library + ``keyframe.config`` are used here, so it imports
cleanly inside Rhino without pulling in the heavy compas / pybullet stack.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from husky_assembly_tamp.keyframe import config


# Reuse Rhino's per-session sticky cache when available (so the sidecar survives
# across script runs in one Rhino session); fall back to a module dict outside
# Rhino. Same pattern the design repo's core.robot_cell uses.
try:
    import scriptcontext as _sc  # type: ignore
    _STICKY = _sc.sticky
except ImportError:
    _STICKY: dict = {}

# Session-cache key holding the live sidecar ``subprocess.Popen`` handle.
_STICKY_SIDECAR = "bar_joint:ssik_sidecar"


def _flatten_row_major(matrix) -> list:
    """Flatten a 4x4 transform into 16 row-major floats (no numpy dependency).

    Args:
        matrix: a 4x4 array-like (numpy array or nested list) transform.

    Returns:
        list[float]: the 16 entries, row 0 first.
    """
    return [float(matrix[i][j]) for i in range(4) for j in range(4)]


def _spawn_sidecar():
    """Launch the ssik sidecar in the Python 3.11 venv and wait for its handshake.

    Resolves the venv interpreter, the sidecar script, and the artifact directory
    (env var -> host-repo default -> error; see ``keyframe.config``), verifies they
    all exist, then starts the process and blocks until it prints its
    ``{"ready": true}`` handshake line.

    Returns:
        subprocess.Popen: the running sidecar (text-mode pipes on stdin/stdout).

    Raises:
        RuntimeError: if the venv / sidecar / artifacts are missing, or the sidecar
            dies or fails to hand off before it is ready. The message tells the user
            how to set the environment up.
    """
    # Resolved at spawn time (not import time) so a missing venv only errors when
    # the ssik backend is actually used.
    python = config.get_ssik_venv_python()
    script = config.get_ssik_sidecar_script()
    artifact_dir = config.get_ssik_artifact_dir()

    # * First-use notice: this spawn only runs the FIRST time the ssik backend is
    # * called in a session -- the live sidecar handle is cached afterwards, so
    # * later solves reuse it and this block never runs again. Print the Python 3.11
    # * env we launch the sidecar from so you can confirm the right venv is used
    # * (handy when IK results or startup look off, or when an env var override is
    # * meant to be picking a different venv).
    print(f"[ssik] launching sidecar (first ssik call this session)")
    print(f"[ssik]   python env : {python}")
    print(f"[ssik]   artifacts  : {artifact_dir}")

    # Clear, actionable errors instead of a silent fallback to the wrong backend.
    if not os.path.isfile(python):
        raise RuntimeError(
            f"ssik sidecar Python not found at {python}. Create the venv and install "
            "ssik (see keyframe/ssik_sidecar/README.md), point HUSKY_SSIK_VENV_DIR at an "
            "existing venv, or set HUSKY_IK_BACKEND=gradient."
        )
    if not os.path.isfile(script):
        raise RuntimeError(f"ssik sidecar script missing at {script}.")
    if not os.path.isdir(artifact_dir):
        raise RuntimeError(
            f"ssik artifact dir missing at {artifact_dir}. Run the one-time `ssik build` "
            "for both arms (see keyframe/ssik_sidecar/README.md), or point "
            "HUSKY_SSIK_ARTIFACT_DIR at the built modules."
        )

    # On Windows, hide the console window the child would otherwise flash.
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    child_env = dict(os.environ)
    # ! CRITICAL: strip PYTHONPATH / PYTHONHOME. The parent (Rhino's CPython 3.9,
    # or a headless run that wired in Rhino's site-envs) has these pointing at
    # 3.9-built packages. If the 3.11 venv child inherits them it imports the wrong
    # numpy (a 3.9 C-extension) and dies with "No module named
    # numpy._core._multiarray_umath". The venv is self-contained, so it needs
    # neither -- drop them so the child uses ONLY its own site-packages.
    child_env.pop("PYTHONPATH", None)
    child_env.pop("PYTHONHOME", None)
    # Force UTF-8 in the child: ssik logs Unicode (e.g. "→") to stderr, and the
    # Windows default cp1252 console encoding would raise UnicodeEncodeError on it.
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        [python, script, "--artifact-dir", artifact_dir],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,  # line-buffered
        creationflags=creationflags,
        env=child_env,
    )

    # Block until the sidecar reports it has imported the arm modules and is ready.
    # Any line that is not the handshake (stray prints) is skipped; EOF = it died.
    while True:
        line = proc.stdout.readline()
        if line == "":
            err = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"ssik sidecar exited before it was ready.\n{err}")
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue  # non-JSON stray output -> ignore
        if msg.get("ready"):
            return proc
        if "error" in msg:
            raise RuntimeError(f"ssik sidecar failed to start: {msg['error']}")


def _get_sidecar():
    """Return the cached sidecar, spawning it on first use (and if it has died)."""
    proc = _STICKY.get(_STICKY_SIDECAR)
    if proc is not None and proc.poll() is None:
        return proc
    proc = _spawn_sidecar()
    _STICKY[_STICKY_SIDECAR] = proc
    return proc


def solve(arm: str, target_in_base, *, max_solutions: int = 8) -> list:
    """Ask the sidecar for every ssik IK branch of one arm.

    Args:
        arm (str): "left" or "right" (selects the built solver module).
        target_in_base: 4x4 tool0 target transform in the arm's base-link frame,
            meters (array-like; numpy array or nested list).
        max_solutions (int): cap on branches returned.

    Returns:
        list[list[float]]: one joint-value list (radians, kinematic order) per
        branch; empty if the pose is unreachable.

    Raises:
        RuntimeError: if the sidecar is unavailable or reports a solve error.
    """
    module = config.SSIK_ARM_BUILD[arm][2]
    request = {
        "arm": arm,
        "module": module,
        "T": _flatten_row_major(target_in_base),
        "max_solutions": int(max_solutions),
    }

    proc = _get_sidecar()
    try:
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        reply_line = proc.stdout.readline()
    except (BrokenPipeError, OSError) as exc:
        raise RuntimeError(f"ssik sidecar communication failed ({exc}).")

    if reply_line == "":
        err = proc.stderr.read() if proc.stderr else ""
        # Drop the dead handle so the next call respawns.
        _STICKY.pop(_STICKY_SIDECAR, None)
        raise RuntimeError(f"ssik sidecar died during solve.\n{err}")

    reply = json.loads(reply_line)
    if not reply.get("ok"):
        raise RuntimeError(f"ssik solve error ({arm}): {reply.get('error')}")
    return reply["solutions"]


def shutdown():
    """Terminate the sidecar (if running) and clear the cache. Safe to call always."""
    proc = _STICKY.pop(_STICKY_SIDECAR, None)
    if proc is None:
        return
    try:
        if proc.poll() is None:
            # Ask it to quit gracefully, then hard-stop if it lingers.
            try:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
    except Exception:
        # Shutdown must never raise into stop_pb_client's cleanup path.
        try:
            proc.terminate()
        except Exception:
            pass
