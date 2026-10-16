"""Make a copied dashboard ask for its scripts by content, so no stale cache is run.

A browser holds on to a cached script hard, and a fix that is on disk but not in
the cache looks exactly like a fix that did not work -- which has cost real
debugging time here (twice). Appending a hash of the file's contents to the
address means a changed file is a different address, so the browser fetches it,
and an unchanged one still comes from cache.

Two places need the stamp:

* the page's own ``<script src>`` and ``<link href>`` addresses, and
* the ``import ... from "./other.js"`` lines inside the copied modules. The
  entry script's stamp alone is not enough: the browser resolves the bare
  ``"./charts.js"`` import to the same address as before and reuses its old
  cached copy, which is how a page ended up calling a function that only the
  new ``charts.js`` has.

Modules are stamped leaf first, so a module's hash already includes the stamps
of the modules it imports (a change deep in the tree reaches every address
above it).
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Dict, List, Sequence

# * An import of one of the dashboard's own modules: `from "./name.js"` (with
# * either quote style), possibly carrying a stamp from an earlier export
# * (`"./name.js?v=1a2b3c4d"`), which is replaced. Third-party imports such as
# * "three" or the vendored examples live in other folders and are left alone.
LOCAL_IMPORT_PATTERN = re.compile(r'''from\s+(["'])\./([A-Za-z0-9_\-]+\.js)(\?v=[0-9a-f]+)?\1''')


def content_digest(path: str) -> str:
    """The short content hash that is appended to an address.

    Args:
        path (str): the file to hash.

    Returns:
        str: the first 8 hex digits of the file's SHA-1.
    """
    with open(path, "rb") as handle:
        return hashlib.sha1(handle.read()).hexdigest()[:8]


def stamp_module_imports(js_dir: str) -> Dict[str, str]:
    """Rewrite the local imports of every module in a folder to carry a content hash.

    Each ``from "./name.js"`` becomes ``from "./name.js?v=<hash>"`` where the hash
    is of ``name.js`` *after* its own imports were stamped. Modules are processed
    leaf first; a cycle between modules is an error because no order would make
    the hashes consistent.

    Args:
        js_dir (str): the folder holding the copied modules (``static/js``).

    Returns:
        Dict[str, str]: module file name to its final content hash.

    Raises:
        ValueError: when the modules import each other in a cycle, or a module
            imports a local file that is not in the folder.
    """
    names = sorted(name for name in os.listdir(js_dir) if name.endswith(".js"))
    sources: Dict[str, str] = {}
    imports: Dict[str, List[str]] = {}
    for name in names:
        with open(os.path.join(js_dir, name), "r", encoding="utf-8") as handle:
            sources[name] = handle.read()
        imports[name] = [match.group(2) for match in LOCAL_IMPORT_PATTERN.finditer(sources[name])]
        missing = [target for target in imports[name] if target not in names]
        if missing:
            raise ValueError(f"{name} imports {missing}, which are not in {js_dir}; look into the dashboard copy.")

    # ! Leaf first: a module is ready once every module it imports has its hash.
    digests: Dict[str, str] = {}
    pending = list(names)
    while pending:
        ready = [name for name in pending if all(target in digests for target in imports[name])]
        if not ready:
            raise ValueError(f"The dashboard modules import each other in a cycle: {pending}; look into static/js.")
        for name in ready:
            # ? The replacement keeps the quote style and drops any old stamp.
            text = LOCAL_IMPORT_PATTERN.sub(
                lambda match: f"from {match.group(1)}./{match.group(2)}?v={digests[match.group(2)]}{match.group(1)}",
                sources[name],
            )
            path = os.path.join(js_dir, name)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            digests[name] = content_digest(path)
            pending.remove(name)
    return digests


def stamp_page_assets(out_dir: str, page_names: Sequence[str], assets: Sequence[str]) -> None:
    """Stamp a copied dashboard: its modules' imports, then the page's own asset addresses.

    Args:
        out_dir (str): the dashboard folder (holds the page and ``static/``).
        page_names (Sequence[str]): the html files to rewrite (``index.html`` ...).
        assets (Sequence[str]): the page's asset addresses relative to the folder,
            such as ``static/css/index.css`` and ``static/js/prm-viewer.js``.
            An asset that is not on disk is skipped (the page does not use it).
    """
    stamp_module_imports(os.path.join(out_dir, "static", "js"))
    for page_name in page_names:
        page_path = os.path.join(out_dir, page_name)
        with open(page_path, "r", encoding="utf-8") as handle:
            html = handle.read()
        for relative in assets:
            path = os.path.join(out_dir, relative.replace("/", os.sep))
            if not os.path.isfile(path):
                continue
            html = html.replace(f'"./{relative}"', f'"./{relative}?v={content_digest(path)}"')
        with open(page_path, "w", encoding="utf-8") as handle:
            handle.write(html)
