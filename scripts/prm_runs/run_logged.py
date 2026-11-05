"""Run roadmap-tool commands in order, appending all their output to one log.

The portable twin of ``run_logged.ps1``: it works the same on Ubuntu and on
Windows, takes the benchmark data root from ``HUSKY_BENCH_ROOT`` (or
``--root``) instead of a path written into the file, and runs every command
with the same Python interpreter that runs this script.

Each positional argument is one command line. Its first word picks the script:

    "build ..." / "query ..." / "refine ..." / "track ..." / "export ..."
                                               -> scripts/prm_pilot.py
    "bridge validate ..." / "bridge birrt ..."  -> scripts/prm_bridge.py
    "carry stage1 ..." / "carry starts ..."     -> scripts/prm_carry_search.py
    "plan ..."                                  -> scripts/plan_process_carry.py
    "check ..."                                 -> scripts/prm_check_solved.py
    "bench ..."                                 -> scripts/benchmark_cdfm.py

Everything a command prints (stdout and stderr) is appended to
``<root>/prm/logs/<log name>``, with a ``=== <command>`` line before it and
the exit code after it, so the log reads like the PowerShell one did. A
failing command does not stop the queue (same as before: a night of
experiments should not be lost to one bad flag); the queue's own exit code is
1 when any command failed. ``--stop-on-error`` changes that.

Examples, from the repository root::

    # Ubuntu, in the foreground:
    .venv/bin/python scripts/prm_runs/run_logged.py --log e2_refine.log \\
        "refine --name B18_g2_n4000 --query all --workers 6"

    # Ubuntu, detached (survives closing the terminal):
    nohup .venv/bin/python scripts/prm_runs/run_logged.py --log e2_refine.log \\
        "build --name ..." "track --name ..." "export --name ..." > /dev/null 2>&1 &

    # Windows (PowerShell):
    .venv\\Scripts\\python.exe scripts\\prm_runs\\run_logged.py --log e2_refine.log "refine --name B18_g2_n4000 --query all"

``--dry-run`` prints the resolved command lines and exits without running them.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from typing import List, Tuple

# * Where this file lives: <repo>/scripts/prm_runs/run_logged.py
PRM_RUNS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(PRM_RUNS_DIR)
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)

# * First word of a command -> the script that runs the rest of it. A word that
# * is not in this table means a prm_pilot.py subcommand (build, query, ...).
SCRIPT_BY_FIRST_WORD = {
    "bridge": "prm_bridge.py",
    "carry": "prm_carry_search.py",
    "plan": "plan_process_carry.py",
    "check": "prm_check_solved.py",
    "bench": "benchmark_cdfm.py",
}
DEFAULT_SCRIPT = "prm_pilot.py"


def resolve_root(value: str) -> str:
    """The benchmark data root, from ``--root`` or ``HUSKY_BENCH_ROOT``.

    Args:
        value (str): the flag's value, possibly empty.

    Returns:
        str: the folder, as an absolute path.

    Raises:
        SystemExit: when neither is set.
    """
    root = value or os.environ.get("HUSKY_BENCH_ROOT", "")
    if not root:
        raise SystemExit("No benchmark data root. Pass --root, or set HUSKY_BENCH_ROOT.")
    return os.path.abspath(root)


def split_command(command: str) -> Tuple[str, str]:
    """Pick the script for one command line and strip its dispatch word.

    Args:
        command (str): one command line, e.g. ``"bridge validate --name X"``.

    Returns:
        Tuple[str, str]: the script path, and the rest of the command line
        (for ``prm_pilot.py`` the whole line, since its first word is the
        subcommand itself).
    """
    first, _, rest = command.strip().partition(" ")
    if first in SCRIPT_BY_FIRST_WORD:
        return os.path.join(SCRIPTS_DIR, SCRIPT_BY_FIRST_WORD[first]), rest.strip()
    return os.path.join(SCRIPTS_DIR, DEFAULT_SCRIPT), command.strip()


def build_invocation(script: str, rest: str):
    """The argument list (or, on Windows, the command string) for one command.

    On Linux and macOS the command line is split the way a shell would, so
    quoted paths with spaces survive. On Windows the line is handed over as
    one string, because that is what the system's process creation parses
    (splitting it here would strip the quotes the child needs).

    Args:
        script (str): the script to run.
        rest (str): its command line.

    Returns:
        list | str: what ``subprocess.run`` takes.
    """
    if os.name == "nt":
        return f'"{sys.executable}" -u "{script}" {rest}'
    return [sys.executable, "-u", script, *shlex.split(rest)]


def main(argv: List[str]) -> int:
    """Run the queue.

    Args:
        argv (List[str]): the command-line arguments.

    Returns:
        int: 0 when every command succeeded, else 1.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True, help="log file name, written under <root>/prm/logs/")
    parser.add_argument("--root", default="", help="benchmark data root (default: HUSKY_BENCH_ROOT)")
    parser.add_argument("--stop-on-error", action="store_true", help="stop the queue at the first failing command")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved commands, run nothing")
    parser.add_argument("commands", nargs="+", help="one command line per argument (quote each)")
    args = parser.parse_args(argv)

    root = resolve_root(args.root)
    # The scripts read the root from the environment, so pass it on explicitly
    # when it came from --root.
    os.environ["HUSKY_BENCH_ROOT"] = root
    log_dir = os.path.join(root, "prm", "logs")
    log_path = os.path.join(log_dir, args.log)

    invocations = []
    for command in args.commands:
        script, rest = split_command(command)
        invocations.append((command, build_invocation(script, rest)))

    if args.dry_run:
        print(f"log: {log_path}")
        for command, invocation in invocations:
            shown = invocation if isinstance(invocation, str) else " ".join(shlex.quote(part) for part in invocation)
            print(f"  {command}\n    -> {shown}")
        return 0

    os.makedirs(log_dir, exist_ok=True)
    failed = 0
    with open(log_path, "a", encoding="utf-8") as log:
        for command, invocation in invocations:
            log.write(f"=== {command}\n=== started {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            log.flush()
            # ! stdout and stderr both go to the log, in order, like "2>&1" did.
            result = subprocess.run(invocation, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            log.write(f"=== exit {result.returncode} at {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            log.flush()
            print(f"[run_logged] exit {result.returncode}: {command}")
            if result.returncode != 0:
                failed += 1
                if args.stop_on_error:
                    log.write("STOPPED ON ERROR\n")
                    break
        log.write("ALL DONE\n" if not failed else f"DONE WITH {failed} FAILURE(S)\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
