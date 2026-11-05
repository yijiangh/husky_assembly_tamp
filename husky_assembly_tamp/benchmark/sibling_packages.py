"""Put the sibling source checkouts ahead of anything pip installed.

When this repo is vendored inside the design workflow repo, two of its
dependencies live beside it as source checkouts rather than as installed
packages:

    external/compas_fab/src             the planning backend we actually develop
    external/scaffolding_assembly_core  the process format reader

A venv may also have older copies of those installed. The source checkouts are
the ones the exported data was written with, so they go first on the path.

* When this repo is not vendored (a git worktree, a plain clone on another
* machine), ``HUSKY_EXTERNAL_DIR`` names the folder holding those checkouts
* instead; it is looked at before the folder next to this repo.

The benchmark package is imported, not run, so the wiring lives here and
``__init__`` calls it once; the scripts call it before anything else.
"""
from __future__ import annotations

import os
import sys
from typing import List


def sibling_source_dirs() -> List[str]:
    """Return the sibling source folders that exist next to this repo.

    Returns:
        List[str]: absolute paths, most important first. Empty when this repo
        is not vendored (a plain pip install, where the installed packages are
        the right ones anyway).
    """
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_dir = os.path.dirname(package_dir)
    # The folder next to this repo, and the one named by HUSKY_EXTERNAL_DIR (first).
    external_dirs = [os.path.dirname(repo_dir)]
    named = os.environ.get("HUSKY_EXTERNAL_DIR")
    if named:
        external_dirs.insert(0, named)
    candidates = []
    for external_dir in external_dirs:
        candidates += [
            os.path.join(external_dir, "compas_fab", "src"),
            os.path.join(external_dir, "scaffolding_assembly_core"),
        ]
    found = []
    for path in candidates:
        # ? The same package from two folders: the first one wins.
        if os.path.isdir(path) and os.path.basename(path.rstrip(os.sep)) not in [
                os.path.basename(p.rstrip(os.sep)) for p in found]:
            found.append(path)
    return found


def ensure_sibling_packages() -> List[str]:
    """Prepend the sibling source folders to ``sys.path``.

    Safe to call more than once: a folder already on the path is left where it
    is rather than being added again.

    Returns:
        List[str]: the folders that were added by this call.
    """
    added = []
    for path in reversed(sibling_source_dirs()):
        if path not in sys.path:
            sys.path.insert(0, path)
            added.append(path)
    return added
