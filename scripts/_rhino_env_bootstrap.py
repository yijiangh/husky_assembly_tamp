"""Add Rhino's `# venv: scaffolding_env` site-env to `sys.path` from outside Rhino.

When a `# venv: <name>` script runs inside Rhino's ScriptEditor, packages
declared with `# r:` end up in
`~/.rhinocode/py39-rh8/site-envs/<name>-<hash>/`, which is added to
`sys.path` only by the ScriptEditor invocation. Plain `python.exe` does
NOT include those site-envs, so `import compas` fails.

This helper finds the active `scaffolding_env-*` directory (and falls
back to other Rhino site-envs if absent) and registers them via
`site.addsitedir`, which also processes any `.pth` files inside.

Idempotent. No-op when called from inside Rhino (where `scriptcontext`
already imports cleanly). Call it once at the top of any pure-Python
test / debug script that needs the Rhino site-env packages.
"""

from __future__ import annotations

import glob
import os
import site
import sys


def bootstrap_rhino_site_envs(env_name: str = "scaffolding_env", verbose: bool = True) -> list[str]:
    """Add `~/.rhinocode/py39-rh8/site-envs/<env_name>-*/` to sys.path.

    Returns the list of paths added. Empty list = nothing matched, in
    which case the caller's imports will probably still fail and the
    script should print a helpful message.
    """
    if "scriptcontext" in sys.modules:
        # Already inside Rhino's interpreter; site-envs are wired by ScriptEditor.
        return []

    # * Rhino 8's site-envs live under `py39-rh8` and hold packages compiled for
    # * CPython 3.9. Their binary extensions (numpy, pybullet, ...) cannot load
    # * in any other Python, so on a non-3.9 interpreter -- e.g. a clean 3.11
    # * venv meant to run WITHOUT Rhino -- we must not wire them in at all.
    # * Skip here so such an env stays genuinely Rhino-free and relies only on
    # * its own installed packages.
    if sys.version_info[:2] != (3, 9):
        if verbose:
            print(
                f"_rhino_env_bootstrap: interpreter is Python "
                f"{sys.version_info.major}.{sys.version_info.minor}, not 3.9; "
                "skipping Rhino py39 site-envs (this env must bring its own packages)."
            )
        return []

    home = os.path.expanduser("~")
    rhinocode_root = os.path.join(home, ".rhinocode", "py39-rh8")
    site_envs_root = os.path.join(rhinocode_root, "site-envs")
    if not os.path.isdir(site_envs_root):
        if verbose:
            print(
                f"_rhino_env_bootstrap: no Rhino site-envs root at {site_envs_root}; "
                "ensure Rhino 8 is installed."
            )
        return []

    # Primary: the named env we want.
    primary = sorted(glob.glob(os.path.join(site_envs_root, f"{env_name}-*")))
    # Secondary fallback: any other site-env (e.g. `default-*`) so generic deps
    # like numpy / scipy / pybullet that may live there are still findable.
    fallback = [
        path for path in sorted(glob.glob(os.path.join(site_envs_root, "*-*")))
        if path not in primary
    ]

    added: list[str] = []
    for path in primary + fallback:
        if not os.path.isdir(path):
            continue
        # `addsitedir` adds the directory AND processes its .pth files.
        site.addsitedir(path)
        if path in sys.path:
            added.append(path)

    if verbose:
        if added:
            print(f"_rhino_env_bootstrap: added {len(added)} Rhino site-env(s) to sys.path:")
            for p in added:
                print(f"  - {p}")
        else:
            print(
                f"_rhino_env_bootstrap: no '{env_name}-*' site-env found under "
                f"{site_envs_root}. Run a `# venv: {env_name}` script inside Rhino "
                "first to materialise it."
            )
    return added
