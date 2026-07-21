"""ssik analytical-IK sidecar server (runs in a Python 3.11 venv with ssik).

Rhino's CPython 3.9 (and any offline venv without 3.11) cannot import ssik, so
this tiny server runs in a separate 3.11 interpreter and answers IK requests over
stdin/stdout as newline-delimited JSON. See ``keyframe/ssik_client.py`` for the
caller side and ``keyframe/ssik_sidecar/README.md`` for the one-time setup.

Protocol (one JSON object per line):
  request : {"arm": "left", "module": "left_ur_arm_ik",
             "T": [16 floats row-major, meters], "max_solutions": 8}
  reply   : {"ok": true, "solutions": [[j1..j6], ...], "residuals": [...]}
            {"ok": false, "error": "<message>"}
  quit    : {"cmd": "quit"}  ->  server exits
On startup the server prints exactly one handshake line: {"ready": true}.

All ssik / library output is redirected to stderr so stdout carries ONLY our JSON
replies (the client reads stdout one line at a time).
"""

import argparse
import json
import os
import sys
from importlib import import_module

import numpy as np


# Keep stdout pristine for JSON replies: hand the real stdout to `_reply` and point
# sys.stdout at stderr so any stray library prints cannot corrupt the protocol.
_OUT = sys.stdout
sys.stdout = sys.stderr

# Cache of imported per-arm solver modules, keyed by module name.
_MODULES = {}


def _reply(obj):
    """Write one JSON reply line to the real stdout and flush it."""
    _OUT.write(json.dumps(obj) + "\n")
    _OUT.flush()


def _get_module(module_name: str):
    """Import (and cache) a built ssik arm module by name.

    Args:
        module_name (str): e.g. ``"left_ur_arm_ik"`` (a file in the artifact dir).

    Returns:
        module: the imported solver module, exposing ``solve(T, ...)``.
    """
    if module_name not in _MODULES:
        _MODULES[module_name] = import_module(module_name)
    return _MODULES[module_name]


def _handle(request: dict) -> dict:
    """Run one solve request and return the reply dict.

    Args:
        request (dict): the parsed request object (see module docstring).

    Returns:
        dict: the reply object (``ok``/``solutions``/``residuals`` or ``ok=False``).
    """
    module = _get_module(request["module"])
    target = np.asarray(request["T"], dtype=float).reshape(4, 4)
    max_solutions = int(request.get("max_solutions", 8))

    sols = module.solve(target, max_solutions=max_solutions)

    solutions, residuals = [], []
    for sol in sols:
        # Each Solution carries q (joint vector) and fk_residual (‖FK(q) − T‖).
        solutions.append([float(v) for v in sol.q])
        residuals.append(float(getattr(sol, "fk_residual", 0.0)))
    return {"ok": True, "solutions": solutions, "residuals": residuals}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Folder holding the generated <arm>_ik.py solver modules.",
    )
    args = parser.parse_args()

    # Make the built arm modules importable by name.
    if args.artifact_dir not in sys.path:
        sys.path.insert(0, os.path.abspath(args.artifact_dir))

    # Handshake: the loop is ready to accept requests.
    _reply({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError as exc:
            _reply({"ok": False, "error": f"bad JSON: {exc}"})
            continue

        if request.get("cmd") == "quit":
            break

        try:
            _reply(_handle(request))
        except Exception as exc:  # never let one bad request kill the server
            _reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
