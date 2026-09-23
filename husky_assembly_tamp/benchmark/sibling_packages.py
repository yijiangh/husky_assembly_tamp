"""Put the sibling source checkouts ahead of anything pip installed.

When this repo is vendored inside the design workflow repo, two of its
dependencies live beside it as source checkouts rather than as installed
packages:

    external/compas_fab/src        the planning backend we actually develop
    external/rs_data_structure     the bar action / movement classes

A venv may also have older copies of those installed. The source checkouts are
the ones the exported data was written with -- reading a bar action with a
stale ``rs_data_structure`` fails with an unhelpful decoder error -- so they go
first on the path.

``scripts/headless_bar_action_planner.py`` does the same thing inline. The
benchmark package cannot: it is imported, not run, so the wiring lives here and
``__init__`` calls it once.
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
    external_dir = os.path.dirname(repo_dir)
    candidates = [
        os.path.join(external_dir, "compas_fab", "src"),
        os.path.join(external_dir, "rs_data_structure"),
    ]
    return [path for path in candidates if os.path.isdir(path)]


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
