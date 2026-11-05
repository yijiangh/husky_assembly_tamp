"""File paths that work on both machines.

Windows limits a plain path to 260 characters unless the machine has long
paths switched on. The data root lives in a Google Drive folder with a long
name, a run's dashboard copies a vendored JavaScript tree several folders
deep, and a process folder's run name is longer than the old ones: together
that crosses the limit (seen 2026-11-02 on `261008_2ModulesDemo_purged_process__B21`,
copying `static/vendor/three-r185/examples/jsm/utils/BufferGeometryUtils.js`).
Windows accepts any length when the path carries the ``\\\\?\\`` prefix, which
:func:`long_path` adds. On Linux and macOS it returns the path unchanged.
"""
from __future__ import annotations

import os
import sys

# * The prefix that lifts Windows' path length limit, and its form for network shares.
LONG_PREFIX = "\\\\?\\"
UNC_PREFIX = "\\\\?\\UNC\\"


def long_path(path: str) -> str:
    """The path in a form Windows accepts at any length; unchanged elsewhere.

    Args:
        path (str): a file or folder path, relative or absolute.

    Returns:
        str: on Windows the absolute path with the ``\\\\?\\`` prefix (or
        ``\\\\?\\UNC\\`` for a network share); on other systems ``path`` as given.
    """
    if sys.platform != "win32":
        return path
    absolute = os.path.abspath(path)
    if absolute.startswith(LONG_PREFIX):
        return absolute
    if absolute.startswith("\\\\"):
        return UNC_PREFIX + absolute[2:]
    return LONG_PREFIX + absolute
