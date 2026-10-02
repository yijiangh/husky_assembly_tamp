"""In-process loader for the built ssik solver modules (Python 3.11+ only).

The ssik sidecar (``ssik_client`` + ``ssik_sidecar/serve.py``) exists because
Rhino runs CPython 3.9, which cannot import ssik. But the OFFLINE planner runs
in a Python 3.11 venv where ssik installs cleanly -- shelling out to a second
3.11 venv from there is pointless. This module imports the SAME committed
artifact modules (``<arm>_ik.py`` from ``config.get_ssik_artifact_dir()``)
directly into the current interpreter, so a solve is a plain function call.

Extra win over the sidecar: the artifact ``solve()`` supports ``q_seed``
(nearest-branch / trajectory-tracking mode), which the sidecar wire protocol
never exposed. The motion planner leans on this heavily.

This is the ONLY module that touches ``sys.path`` / ``importlib`` for the
artifact modules. Only stdlib + numpy + ``keyframe.config`` are imported here,
so importing this module is always safe -- availability of ssik itself is
probed lazily via :func:`available`.

Artifact SETS (benchmark experiments only): every function takes an optional
``artifact_set`` name from ``config.SSIK_ARTIFACT_SETS``. ``None`` is the
committed calibrated solver, loaded exactly as before. Any other set is loaded
from its file under a unique module name, so several sets can be used in one
process at the same time (the motion search on one, the re-solve of the found
path on another). See :func:`_load_artifact_module` for what is overridden in
such a module to make it behave like ours.
"""

from __future__ import annotations

import hashlib
import os
import sys
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from husky_assembly_tamp.keyframe import config


# * The name of the committed calibrated solver set (the production solver).
DEFAULT_ARTIFACT_SET = "calibrated"

# * Lazy caches: the artifact modules, keyed by (set name, arm), and the "can
# * we import ssik at all in this interpreter" probe result. None = not probed.
_MODULES: Dict[Tuple[str, str], Any] = {}
_AVAILABLE: Optional[bool] = None


def _set_name(artifact_set: Optional[str]) -> str:
    """Normalise a set name: ``None`` means the calibrated default."""
    return artifact_set or DEFAULT_ARTIFACT_SET


def available() -> bool:
    """Whether ssik can be imported in the CURRENT interpreter (cached probe).

    Returns:
        bool: True when ``import ssik`` succeeds (a Python 3.11+ env with ssik
        installed), False otherwise (e.g. Rhino's CPython 3.9 -- callers should
        then fall back to the sidecar).
    """
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import ssik  # noqa: F401  # probe only; the artifacts import from it
            _AVAILABLE = True
        except ImportError:
            _AVAILABLE = False
    return _AVAILABLE


def _require_ssik() -> None:
    """Raise a readable error when ssik cannot be imported here.

    Raises:
        RuntimeError: the message carries the install command and the gradient
            escape hatch (no silent fallback, matching the repo's philosophy).
    """
    if available():
        return
    raise RuntimeError(
        "ssik in-process backend unavailable (`import ssik` failed in this "
        f"interpreter, Python {sys.version_info.major}.{sys.version_info.minor}). "
        "ssik 4.1+ ships a cp310 wheel and supports Python >= 3.10, so "
        "install it directly into this env (no sidecar / 3.11 venv needed):\n"
        "    pip install \"ssik>=4.1,<5\"\n"
        "It pulls numpy>=1.26; in a ROS2 venv keep numpy on the 1.x line "
        "(e.g. `pip install numpy==1.26.4`), NOT numpy 2.x. Rhino's 3.9 "
        "still uses the sidecar. Or set HUSKY_IK_BACKEND=gradient for the "
        "PyBullet fallback."
    )


def _load_artifact_module(set_name: str, arm: str):
    """Load one arm's solver module of a NON-default artifact set from its file.

    The module is loaded under its own name (``husky_ssik_<set>_<arm>``), never
    through ``import_module``: the calibrated modules are called
    ``left_ur_arm_ik`` and a nominal build would carry the same name, and
    Python keeps one module per name. A separate copy per arm also lets the
    two arms carry different joint names.

    Two things in the loaded module are then overridden so the solver behaves
    like ours (see ``config.SSIK_ARTIFACT_SETS``):

      joint names -> our URDF's names for that arm (ssik's prebuilt UR5e
                     module has the generic ``shoulder_pan_joint`` and so on,
                     which the joint-order guard would not find in PyBullet)
      limits      -> the calibrated set's joint limits, so the limit filter
                     keeps exactly the branches production would keep. The
                     filter reads the module's ``_KB`` object, which is built
                     once at import from ``_JOINT_LIMITS``, so it is rebuilt
                     here after the override.

    Args:
        set_name (str): the artifact set (not the default).
        arm (str): "left" or "right".

    Returns:
        module: the loaded and adjusted solver module.
    """
    entry = config.get_ssik_artifact_set(set_name)
    path = entry["files"][arm]
    module_name = f"husky_ssik_{set_name}_{arm}"
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"ssik artifact set {set_name!r}: cannot load {path}")
    mod = module_from_spec(spec)
    # Registered so the module's own relative lookups (dataclasses, pickling)
    # find it; the name is unique to this set and arm.
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    if entry.get("joint_names") == "husky":
        mod._JOINT_NAMES = config.husky_joint_names(arm)
    if entry.get("limits") == "calibrated":
        mod._JOINT_LIMITS = [tuple(pair) for pair in _get_module(arm)._JOINT_LIMITS]
    mod._KB = mod._build_kb()
    print(f"[ssik-inprocess] loaded artifact set {set_name!r} for the {arm} arm "
          f"from {path} ({getattr(mod, 'SOLVER_NAME', '?')})")
    return mod


def _get_module(arm: str, artifact_set: Optional[str] = None):
    """Import (once) and return the built solver module for one arm.

    Args:
        arm (str): "left" or "right" (selects the module via
            ``config.SSIK_ARM_BUILD``).
        artifact_set (Optional[str]): which solver set; ``None`` is the
            committed calibrated one, loaded exactly as before.

    Returns:
        module: the imported artifact module (exposes ``solve``,
        ``_JOINT_NAMES``, ``_JOINT_LIMITS``, ...).

    Raises:
        RuntimeError: when ssik cannot be imported in this interpreter, or
            the set's module file is missing.
    """
    set_name = _set_name(artifact_set)
    mod = _MODULES.get((set_name, arm))
    if mod is None:
        _require_ssik()
        if set_name == DEFAULT_ARTIFACT_SET:
            # * The artifact folder holds plain .py modules generated by `ssik
            # * build`; putting it on sys.path is how they become importable
            # * (same trick the sidecar server uses).
            artifact_dir = config.get_ssik_artifact_dir()
            if artifact_dir not in sys.path:
                sys.path.insert(0, artifact_dir)
            module_name = config.SSIK_ARM_BUILD[arm][2]
            mod = import_module(module_name)
            # One-line breadcrumb so a log makes clear the sidecar was NOT used.
            print(f"[ssik-inprocess] loaded {module_name} from {artifact_dir}")
        else:
            mod = _load_artifact_module(set_name, arm)
        _MODULES[(set_name, arm)] = mod
    return mod


def solve(
    arm: str,
    target_in_base,
    *,
    max_solutions: Optional[int] = None,
    q_seed=None,
    allow_rescue: bool = True,
    respect_limits: bool = True,
    artifact_set: Optional[str] = None,
) -> List[Tuple[np.ndarray, float]]:
    """Solve one arm's analytical IK in-process.

    Args:
        arm (str): "left" or "right".
        target_in_base: 4x4 tool0 target transform expressed in the arm's
            base-link frame, meters (array-like).
        max_solutions (Optional[int]): cap on returned branches (None = all).
        q_seed: optional 6-vector; when given, ssik orders branches
            nearest-to-seed first (trajectory-tracking mode).
        allow_rescue (bool): let ssik run its (slow, ~40x) numeric rescue when
            the exact solver finds nothing -- worth it for one-off endpoint
            solves, but turn it OFF in tracking loops: an unreachable waypoint
            should just fail fast (~1 ms), and rescued solutions are loose LM
            refinements that may sit on a different branch anyway.
        respect_limits (bool): drop every branch that cannot be wrapped into
            the joint limits (the default, and what any real plan must use).
            Pass False only to find out WHY a pose had no solution: an empty
            list with limits on can mean "no solution" or "solutions exist but
            all break a limit", and this tells the two apart.
        artifact_set (Optional[str]): which solver set to use; ``None`` is the
            calibrated production solver.

    Returns:
        List[Tuple[np.ndarray, float]]: one ``(q6, fk_residual)`` pair per
        branch (joints in kinematic base->tool0 order, radians); empty list =
        pose unreachable.
    """
    module = _get_module(arm, artifact_set)
    sols = module.solve(
        np.asarray(target_in_base, dtype=float),
        max_solutions=max_solutions,
        q_seed=None if q_seed is None else np.asarray(q_seed, dtype=float),
        allow_rescue=allow_rescue,
        respect_limits=respect_limits,
    )
    return [(np.asarray(s.q, dtype=float), float(s.fk_residual)) for s in sols]


def fk(arm: str, q, artifact_set: Optional[str] = None) -> np.ndarray:
    """Forward kinematics: tool0 pose in the arm's ``*_ur_arm_base_link`` frame.

    Cheap enough (~16 us/call) for live per-tick use, e.g. streaming the flange
    pose while an operator free-drives the arm.

    Args:
        arm (str): "left" or "right".
        q: 6 joint angles in radians, kinematic base->tool0 order (same order
            as the driver's reordered joint state, ``arm_joint_pose``).
        artifact_set (Optional[str]): which solver set's geometry to use.

    Returns:
        np.ndarray: 4x4 homogeneous transform, meters (the set's geometry:
        calibrated by default).
    """
    return np.asarray(_get_module(arm, artifact_set).fk(np.asarray(q, dtype=float)))


def joint_names(arm: str, artifact_set: Optional[str] = None) -> List[str]:
    """The arm's 6 joint names in the solver's (kinematic) order.

    Args:
        arm (str): "left" or "right".
        artifact_set (Optional[str]): which solver set.

    Returns:
        List[str]: URDF joint names, base->tool0 order.
    """
    return list(_get_module(arm, artifact_set)._JOINT_NAMES)


def joint_limits(arm: str, artifact_set: Optional[str] = None) -> List[Tuple[float, float]]:
    """The arm's joint limits in the solver's (kinematic) order.

    Args:
        arm (str): "left" or "right".
        artifact_set (Optional[str]): which solver set.

    Returns:
        List[Tuple[float, float]]: ``(lower, upper)`` per joint, radians.
    """
    return [(float(lo), float(hi)) for lo, hi in _get_module(arm, artifact_set)._JOINT_LIMITS]


def describe_artifact_set(artifact_set: Optional[str] = None) -> Dict[str, Any]:
    """Say which solver files a set uses, for a batch's provenance record.

    Args:
        artifact_set (Optional[str]): which solver set (``None`` = calibrated).

    Returns:
        Dict[str, Any]: per arm: the module file, its sha1, ssik's solver name
        and tier, the expected milliseconds per solve, and the joint names and
        limits in force after any override.
    """
    out: Dict[str, Any] = {}
    for arm in ("left", "right"):
        mod = _get_module(arm, artifact_set)
        path = getattr(mod, "__file__", None)
        digest = None
        if path and os.path.isfile(path):
            with open(path, "rb") as handle:
                digest = hashlib.sha1(handle.read()).hexdigest()
        out[arm] = {
            "file": path,
            "sha1": digest,
            "solver_name": getattr(mod, "SOLVER_NAME", None),
            "solver_tier": getattr(mod, "SOLVER_TIER", None),
            "expected_ms_median": getattr(mod, "EXPECTED_MS_MEDIAN", None),
            "joint_names": list(mod._JOINT_NAMES),
            "joint_limits": [[float(lo), float(hi)] for lo, hi in mod._JOINT_LIMITS],
        }
    return out


def preload_artifact_sets(names: Iterable[Optional[str]]) -> None:
    """Load the given sets for both arms now, so a missing file fails up front.

    Args:
        names (Iterable[Optional[str]]): set names (``None`` = calibrated).
    """
    for name in names:
        for arm in ("left", "right"):
            _get_module(arm, name)
