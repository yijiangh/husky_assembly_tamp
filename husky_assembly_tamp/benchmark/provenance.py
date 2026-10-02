"""Record which code produced a set of results.

The whole point of this benchmark is to compare planner versions, which only
works if every result says which version it came from. That means the commit of
each repository involved, and a flag when a repository had uncommitted changes
-- a result produced from a dirty tree cannot be reproduced from its commit
alone, and saying so is better than pretending otherwise.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional


def repo_state(path: str) -> Dict[str, Any]:
    """Return the commit and cleanliness of one git repository.

    Args:
        path (str): a folder inside the repository.

    Returns:
        Dict[str, Any]: ``commit``, ``dirty`` and ``branch``; the values are
        None when the folder is not a git repository or git is unavailable.
    """
    def git(*arguments: str) -> Optional[str]:
        """Run a git command in `path` and return its output, or None."""
        try:
            out = subprocess.run(
                ["git", "-C", path, *arguments],
                capture_output=True, text=True, timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        return out.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def package_versions() -> Dict[str, Optional[str]]:
    """Return the versions of the packages the results depend on.

    Returns:
        Dict[str, Optional[str]]: version strings, None where unavailable.
    """
    versions: Dict[str, Optional[str]] = {"python": sys.version.split()[0]}
    for name in ("numpy", "compas", "compas_fab", "compas_robots", "pybullet",
                 "ssik", "pyinstrument"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except ImportError:
            versions[name] = None
    return versions


# * Environment variables that change which solver or data a batch used. They
# * are recorded because a result produced with a different artifact folder or
# * IK backend is a different planner, even at the same commit.
RECORDED_ENV_VARS = ("HUSKY_SSIK_ARTIFACT_DIR", "HUSKY_IK_BACKEND", "HUSKY_BENCH_ROOT")


def provenance(artifacts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return everything needed to say where a set of results came from.

    Args:
        artifacts (Optional[Dict[str, Any]]): a description of every solver
            artifact set the batch will use (file, solver name and tier, joint
            limits), as returned by ``ssik_inprocess.describe_artifact_set``,
            keyed by set name. ``None`` records nothing about artifacts.

    Returns:
        Dict[str, Any]: the repositories' git state, package versions, the
        machine the batch ran on, the recorded environment variables and the
        solver artifacts.
    """
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tamp_repo = os.path.dirname(package_dir)
    external_dir = os.path.dirname(tamp_repo)
    workflow_repo = os.path.dirname(external_dir)

    repos: Dict[str, Dict[str, Any]] = {"husky_assembly_tamp": repo_state(tamp_repo)}
    for name in ("compas_fab", "rs_data_structure"):
        folder = os.path.join(external_dir, name)
        if os.path.isdir(folder):
            repos[name] = repo_state(folder)
    if os.path.isdir(os.path.join(workflow_repo, ".git")):
        repos["bar_joint_rhino_design_workflow"] = repo_state(workflow_repo)

    return {
        "repos": repos,
        "packages": package_versions(),
        "machine": {
            "node": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        # Unset variables are recorded as None so "was it set?" stays answerable.
        "env": {name: os.environ.get(name) for name in RECORDED_ENV_VARS},
        "artifacts": dict(artifacts or {}),
    }
