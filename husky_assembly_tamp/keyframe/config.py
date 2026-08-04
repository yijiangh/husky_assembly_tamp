"""Solver tuning constants + ssik backend wiring for the keyframe solvers.

Moved here from the design front-end's ``scripts/core/config.py`` so the solver
owns its own tuning: the Rhino front-end re-imports these values (one
definition), and the headless planner never has to reach into the Rhino
script tree for a parameter.

Everything DESIGN-specific (home configurations, tool names, layer names, ...)
stays on the Rhino side and reaches the solvers either through the exported
JSON files or through explicit function arguments -- never through this module.

All distances are in millimetres; all angles are in radians (matching the
design front-end convention). The gradient-polish tolerances are the exception:
compas_fab consumes SI, so they are meters / radians as noted inline.

Environment variables (all optional overrides):
  - ``HUSKY_IK_BACKEND``        "ssik" (default) or "gradient"
  - ``HUSKY_SSIK_VENV_DIR``     folder of the sidecar venv that has ssik (Rhino-3.9
                                fallback only; the offline planner runs ssik in-process)
  - ``HUSKY_SSIK_ARTIFACT_DIR`` folder holding the built <arm>_ik.py modules
"""

from __future__ import annotations

import os
import sys


# ---------------------------------------------------------------------------
# Base sampling (the expanding-radius search in walkable_ground)
# ---------------------------------------------------------------------------

IK_BASE_SAMPLE_RADIUS = 1000.0  # mm
IK_BASE_SAMPLE_MAX_ITER = 20
# How far BEHIND the bar the SEED mobile base stands (mm). The seed is offset
# from the bar's ground projection by this distance, OPPOSITE the average
# male-joint insertion direction (the way the bar is pushed to mate), so the base
# faces along the insertion direction and moving forward carries the bar into the
# assembly. Only a starting point: the expanding-radius search samples around this
# seed and IK validates each. Tunable.
IK_BASE_STANDOFF_MM = 700.0  # mm

# ---------------------------------------------------------------------------
# Dual-arm IK solver tuning
# ---------------------------------------------------------------------------

# Max gradient-descent steps the solver takes toward the target within a single
# seed before giving up on that seed.
IK_MAX_DESCEND_ITERATIONS = 200

# Dual-arm IK random-restart budget for a COLD solve (no good warm-start, e.g. the
# first movement M1). Each restart resamples a random dual-arm config to descend
# from. Warm-started solves (M2/M3, seeded from the previous keyframe) force this to 1.
# It is an early-exit CEILING -- easy poses solve in a few restarts; only a tight pose
# (or a genuinely unreachable one) spends the whole budget. Measured M1 success on the
# tight double-kissing-jig B6 at base (0,0,0): 50 -> ~1/8, 150 -> ~4/8, 400 -> 8/8.
IK_MAX_RESTART_ITER = 100
IK_TOLERANCE_POSITION = 1e-5  # m (compas_fab uses SI; converted at the call site if needed)
IK_TOLERANCE_ORIENTATION = 1e-5  # rad

# ---------------------------------------------------------------------------
# IK backend selection
# ---------------------------------------------------------------------------
# * Which inverse-kinematics engine ``dual_arm_ik.solve_dual_arm_ik`` uses:
# *   "ssik"     -> DEFAULT. Analytical IK from ssik (github.com/personalrobotics/ssik).
# *                 ssik builds a closed-form solver straight from our CALIBRATED
# *                 URDF, so there is nothing to tune. ssik 4.1+ ships a cp310 wheel and
# *                 supports Python >= 3.10, so the offline planner (this ROS2 Py3.10
# *                 venv) imports it and solves IN-PROCESS -- no subprocess. Code:
# *                 ``keyframe.ssik_inprocess`` (native solve) via
# *                 ``keyframe.dual_arm_ik.solve_dual_arm_ik_ssik``. Rhino's CPython 3.9
# *                 cannot import ssik, so THERE it falls back to a sidecar process over
# *                 stdio (``keyframe.ssik_client`` + ``keyframe/ssik_sidecar/``); the
# *                 client short-circuits to the in-process path whenever ssik imports.
# *   "gradient" -> the original PyBullet damped-least-squares descent with random
# *                 restarts. Kept only for benchmarking against ssik and as an
# *                 archival fallback; not used in the normal workflow.
IK_BACKEND = os.environ.get("HUSKY_IK_BACKEND", "ssik")
if IK_BACKEND not in ("ssik", "gradient"):
    # ! No silent fallback: a typo in the env var must not quietly pick a backend.
    raise RuntimeError(
        f"HUSKY_IK_BACKEND={IK_BACKEND!r} is not a known IK backend; "
        "use 'ssik' or 'gradient' (or unset it for the ssik default)."
    )

# Per-arm build recipe: (base_link, ee_link, generated module name). Documents the
# one-time `ssik build <urdf> --base <base_link> --ee <ee_link>` for each arm and
# tells the client/sidecar which module to import. Each arm is a plain 6R chain.
SSIK_ARM_BUILD = {
    "left": ("left_ur_arm_base_link", "left_ur_arm_tool0", "left_ur_arm_ik"),
    "right": ("right_ur_arm_base_link", "right_ur_arm_tool0", "right_ur_arm_ik"),
}


# ---------------------------------------------------------------------------
# ssik path resolution
# ---------------------------------------------------------------------------
# The built per-arm solver modules are COMMITTED IN THIS REPO at ``asset/ssik``
# so the package runs standalone (no host repo needed). The ssik sidecar venv
# (a Python 3.10+ env with ssik installed) is only needed by Rhino's CPython 3.9 and is a
# machine-local resource in the HOST repo -- the repo that vendors this package
# as ``external/husky_assembly_tamp`` (the design repo has it at
# ``external/ssik_env``). Each path is resolved at USE time (never at import
# time) with the order: environment variable -> packaged/host default -> clear
# error.

_KEYFRAME_DIR = os.path.dirname(os.path.abspath(__file__))
# <checkout>/husky_assembly_tamp/keyframe -> <checkout> (the tamp repo root)
_TAMP_REPO_ROOT = os.path.dirname(os.path.dirname(_KEYFRAME_DIR))


def _host_repo_root():
    """The repo that vendors this tamp checkout under ``external/``, if any.

    Returns:
        str | None: the host repo root (e.g. the design repo) when this checkout
        sits at ``<host>/external/husky_assembly_tamp``, else ``None`` (a
        pip-installed or standalone checkout -- env vars are then required).
    """
    parent = os.path.dirname(_TAMP_REPO_ROOT)
    if os.path.basename(parent) == "external":
        return os.path.dirname(parent)
    return None


def get_ssik_venv_python() -> str:
    """Path of the interpreter inside the ssik SIDECAR venv (Rhino-3.9 fallback only).

    Only Rhino's CPython 3.9 needs this: it cannot import ssik, so it shells out to a
    sidecar venv that can. The offline planner imports ssik in-process and never calls
    this. Resolution order: ``HUSKY_SSIK_VENV_DIR`` env var -> ``<host>/external/ssik_env``
    -> error. A venv stores its interpreter in a different place per OS (Windows:
    ``Scripts\\python.exe``, macOS/Linux: ``bin/python``), so the right one is
    picked for the current platform.

    Returns:
        str: the interpreter path (existence is checked by the spawner, which
        raises with setup instructions if the file is missing).

    Raises:
        RuntimeError: when no venv folder can even be determined (env var unset
            and this checkout is not vendored inside a host repo).
    """
    venv_dir = os.environ.get("HUSKY_SSIK_VENV_DIR")
    if not venv_dir:
        host = _host_repo_root()
        if host is None:
            raise RuntimeError(
                "Cannot locate the ssik venv: HUSKY_SSIK_VENV_DIR is not set and this "
                f"husky_assembly_tamp checkout ({_TAMP_REPO_ROOT}) is not vendored under a "
                "host repo's external/ folder. Create the venv (see "
                "keyframe/ssik_sidecar/README.md) and set HUSKY_SSIK_VENV_DIR to it, or "
                "set HUSKY_IK_BACKEND=gradient."
            )
        venv_dir = os.path.join(host, "external", "ssik_env")
    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def get_ssik_artifact_dir() -> str:
    """Folder holding the per-arm solver modules generated by ``ssik build``.

    Resolution order: ``HUSKY_SSIK_ARTIFACT_DIR`` env var -> ``<this repo>/asset/ssik``
    (the artifacts are committed in this repo, so a plain checkout is
    self-contained) -> ``<host>/asset/ssik`` (older layout, when this checkout
    is vendored under a host repo) -> error.

    Returns:
        str: the artifact folder path (existence is checked by the caller /
        spawner).

    Raises:
        RuntimeError: when no candidate folder exists and the env var is unset.
    """
    artifact_dir = os.environ.get("HUSKY_SSIK_ARTIFACT_DIR")
    if artifact_dir:
        return artifact_dir
    # Packaged default: the artifacts committed in THIS repo.
    in_repo = os.path.join(_TAMP_REPO_ROOT, "asset", "ssik")
    if os.path.isdir(in_repo):
        return in_repo
    # Older layout: artifacts in the host repo that vendors this checkout.
    host = _host_repo_root()
    if host is not None:
        host_dir = os.path.join(host, "asset", "ssik")
        if os.path.isdir(host_dir):
            return host_dir
    raise RuntimeError(
        "Cannot locate the ssik artifacts: HUSKY_SSIK_ARTIFACT_DIR is not set and "
        f"{in_repo} does not exist. Run the one-time `ssik build` for both arms "
        "(see keyframe/ssik_sidecar/README.md) and set HUSKY_SSIK_ARTIFACT_DIR to the "
        "output folder, or set HUSKY_IK_BACKEND=gradient."
    )


def get_ssik_sidecar_script() -> str:
    """Path of the sidecar server script (packaged with this code, no env var)."""
    return os.path.join(_KEYFRAME_DIR, "ssik_sidecar", "serve.py")
