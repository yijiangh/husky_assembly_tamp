"""Figures and tables for docs/holding_prm_report.tex, made from saved roadmap runs.

Every number in the report's tables comes from here, read from the files the
pilot wrote under ``<root>/prm/`` (and, for the comparison with the task-space
BiRRT, from the benchmark batch ``runs/20260929_1756``). Nothing is typed in
by hand.

    .venv\\Scripts\\python.exe docs\\figures\\make_prm_figures.py

Writes ``docs/figures/prm_*.pdf`` and ``docs/figures/prm_*.tex`` (table bodies
and a few number macros the report inputs).
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402 - the backend must be chosen first
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, ScalarFormatter  # noqa: E402

from husky_assembly_tamp.motion_planner.holding_prm.pilot import find_goal, load_library_entry, wrapped_match  # noqa: E402
from husky_assembly_tamp.motion_planner.holding_prm.refine import refine_path  # noqa: E402
from husky_assembly_tamp.motion_planner.holding_prm.store import list_queries, load_roadmap, query_path, read_json  # noqa: E402


# * Which runs the report is about.
ROOT = os.environ.get("HUSKY_BENCH_ROOT", "")
OUT = os.path.dirname(os.path.abspath(__file__))
BUDGETS = (500, 1000, 2000, 4000)
B18_RUN = "B18_g2_n{}"
B3_RUN = "B3_g4-11-16_n2000"
BIRRT_RUN = "20260929_1756"   # experiment 2: every start of B18 goal 2, two seeds
TOLERANCES = ("2.0", "5.0", "10.0", "30.0")
CLASS_COLOURS = {"free": "#2e9e5b", "blocked": "#d98324", "broken": "#b23a48"}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False})


# -----------------------------------------------------------------------------
# * Loading
# -----------------------------------------------------------------------------


def load_query(run: str, tag: str) -> Optional[Dict[str, Any]]:
    """A saved query, or None when it was not run."""
    path = query_path(ROOT, run, tag)
    return read_json(path) if os.path.isfile(path) else None


def birrt_solved_starts() -> Dict[int, int]:
    """Experiment 2: how many of each start's two BiRRT seeds solved B18 goal 2."""
    solved: Dict[int, int] = defaultdict(int)
    for path in glob.glob(os.path.join(ROOT, "runs", BIRRT_RUN, "runs", "B18", "g002_*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        solved[int(record["start_index"])] += int(bool(record.get("success")))
    return dict(solved)


def fmt(value: Optional[float], digits: int = 1) -> str:
    """A number for a table, or a dash."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return f"{value:.{digits}f}"


def fmt_gap(value: Optional[float], digits: int = 1) -> str:
    """A gap for a table. A missing gap means the node reaches no goal: infinity."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return r"$\infty$"
    return f"{value:.{digits}f}"


def write(name: str, text: str) -> None:
    """Write one generated .tex piece."""
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"  wrote {name}")


def sideways(name: str, corner: str, columns: List[str], rows: List[Tuple[str, List[str]]],
             label_width_cm: float = 6.8) -> None:
    """Write a whole table turned sideways: one row per quantity, one column per run.

    Rows can then carry a full description with its unit, which a column
    header has no room for. Rows whose label is ``None`` become a rule.

    Args:
        name (str): file name under docs/figures.
        corner (str): the top-left cell (what the columns are).
        columns (List[str]): one header per run.
        rows (List[Tuple[str, List[str]]]): (description with unit, one cell per run).
        label_width_cm (float): width of the description column.
    """
    spec = f"@{{}}>{{\\raggedright\\arraybackslash}}p{{{label_width_cm}cm}} " + "r" * len(columns) + "@{}"
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
             " & ".join([corner] + columns) + r" \\", "\\midrule"]
    for label, cells in rows:
        lines.append("\\midrule" if label is None else " & ".join([label] + cells) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(name, "\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# * Tables
# -----------------------------------------------------------------------------


def build_rows(roadmaps: List[Any]) -> List[Tuple[str, List[str]]]:
    """The size-and-cost rows of a sideways roadmap table, one cell per roadmap."""
    def cells(get) -> List[str]:
        return [get(r) for r in roadmaps]

    def count(phase: str, key: str):
        return lambda r: f"{int(r.counters[phase].get(key, 0))}"

    def summary(key: str):
        return lambda r: f"{r.meta['summary'][key]}"

    def anchors(role: str):
        return lambda r: (f"{r.meta['summary']['anchors_in_largest'].get(role, 0)} of "
                          f"{r.meta['summary']['anchors_total'].get(role, 0)}")

    return [
        ("sampled nodes kept: the node budget (count)", cells(lambda r: f"{r.config['nodes']}")),
        ("all nodes, anchors included (count)", cells(lambda r: f"{r.n_nodes}")),
        ("driver-arm draws, one IK call each (count)", cells(count("sample", "draws"))),
        ("collision checks, sampling and edges together (count)",
         cells(lambda r: f"{int(r.counters['sample'].get('collision_checks', 0) + r.counters['edges'].get('collision_checks', 0))}")),
        ("candidate edges measured (count)", cells(lambda r: f"{r.n_edges}")),
        ("collision-free edges (count)", cells(lambda r: f"{int(np.sum(np.asarray(r.edges['free'], dtype=bool)))}")),
        ("components: separate islands of the free graph (count)", cells(summary("n_components"))),
        ("nodes in the largest component (count)", cells(summary("largest_component"))),
        ("goal anchors in the largest component", cells(anchors("goal"))),
        ("start anchors in the largest component", cells(anchors("start"))),
        (None, []),
        ("sampling time (s, wall clock)", cells(lambda r: fmt(r.timings["sample_s"], 0))),
        ("edge measuring time (s, wall clock)", cells(lambda r: fmt(r.timings["edges_s"], 0))),
        ("whole build (s, wall clock)", cells(lambda r: fmt(r.timings["total_s"], 0))),
        ("edge phase speed-up (worker-seconds per wall-second)",
         cells(lambda r: fmt(r.meta["parallel"]["edges"]["speedup"], 1))),
    ]


def build_table(runs: Dict[int, Any]) -> None:
    """Size and cost of each B18 roadmap, sideways (one column per node budget)."""
    sideways("prm_table_build.tex", "node budget", [str(b) for b in runs], build_rows(list(runs.values())))


def query_rows(run_names: List[str]) -> List[Tuple[str, List[str]]]:
    """What each roadmap answers for starts 0 and 12 and for all 807 starts, as sideways rows."""
    pairs = [load_query(run, "s0_s12") for run in run_names]
    everys = [load_query(run, "all") for run in run_names]

    def start_gap(index: int, key: str) -> List[str]:
        out = []
        for pair in pairs:
            rows = {row["start_index"]: row for row in (pair or {}).get("starts", [])}
            out.append(fmt_gap(rows.get(index, {}).get(key)) if pair else "--")
        return out

    def every_cells(get) -> List[str]:
        return [get(e) if e else "--" for e in everys]

    rows = [
        ("start 0: gap, best over its branch pairs and turns (mm)", start_gap(0, "best_gap_mm")),
        ("start 0: gap of its library configuration alone (mm)", start_gap(0, "library_conf_gap_mm")),
        ("start 12: gap, best over its branch pairs and turns (mm)", start_gap(12, "best_gap_mm")),
        ("start 12: gap of its library configuration alone (mm)", start_gap(12, "library_conf_gap_mm")),
        (None, []),
        ("all 807 starts: smallest gap (mm)", every_cells(lambda e: fmt_gap(e["headline"]["best_gap_mm"]))),
        ("starts that reach the goal set at all (count of 807)",
         every_cells(lambda e: f"{e['by_difficulty']['all']['n_reachable']}")),
        ("median gap of the starts that reach it (mm)",
         every_cells(lambda e: fmt(e["by_difficulty"]["all"]["median_gap_mm"]))),
    ]
    for k, tol in enumerate(TOLERANCES):
        rows.append((f"starts with a gap of at most {float(tol):g} mm (count)",
                     every_cells(lambda e, k=k: f"{e['tolerances'][k]['starts_connected']}")))
    rows.append(("all-starts query time (s, wall clock)", every_cells(lambda e: fmt(e["timings"]["total_s"], 0))))
    return rows


def query_table(runs: Dict[int, Any]) -> None:
    """What each B18 roadmap answers, sideways (one column per node budget)."""
    sideways("prm_table_query.tex", "node budget", [str(b) for b in runs],
             query_rows([B18_RUN.format(b) for b in runs]))


def class_table(query: Dict[str, Any], solved: Dict[int, int]) -> None:
    """Per library class (and BiRRT outcome): starts, reachable, median gap, connected per tolerance."""
    groups = {"free": [], "blocked": [], "broken": [], "all": []}
    birrt = {"BiRRT solved (exp. 2)": [], "BiRRT never solved": []}
    for row in query["starts"]:
        gap = row["best_gap_mm"] if row["best_gap_mm"] is not None else np.inf
        groups[row["difficulty"]].append(gap)
        groups["all"].append(gap)
        birrt["BiRRT solved (exp. 2)" if solved.get(row["start_index"], 0) > 0 else "BiRRT never solved"].append(gap)
    rows = []
    for label, gaps in list(groups.items()) + list(birrt.items()):
        gaps = np.asarray(gaps)
        finite = gaps[np.isfinite(gaps)]
        rows.append(" & ".join([label, f"{len(gaps)}", f"{len(finite)}",
                                fmt(float(np.median(finite)) if len(finite) else None),
                                *[f"{int(np.sum(gaps <= float(t)))}" for t in TOLERANCES]]) + r" \\")
    write("prm_table_classes.tex", "\n".join(rows) + "\n")


def carry_table(query: Dict[str, Any], solved: Dict[int, int]) -> None:
    """Per carry anchor: starts, reachable, median gap, connected within 10 and 30 mm, BiRRT-solved starts."""
    rows = []
    for carry in ("back", "horizontal", "vertical"):
        members = [row for row in query["starts"] if row["carry"] == carry]
        gaps = np.array([row["best_gap_mm"] if row["best_gap_mm"] is not None else np.inf for row in members])
        finite = gaps[np.isfinite(gaps)]
        n_solved = sum(1 for row in members if solved.get(row["start_index"], 0) > 0)
        rows.append(" & ".join([carry, f"{len(members)}", f"{len(finite)}",
                                fmt(float(np.median(finite)) if len(finite) else None),
                                f"{int(np.sum(gaps <= 10.0))}", f"{int(np.sum(gaps <= 30.0))}",
                                f"{n_solved}"]) + r" \\")
    write("prm_table_carry.tex", "\n".join(rows) + "\n")


def turn_table(query: Dict[str, Any]) -> None:
    """Which turn the best node of each reachable start is on, and how often it beats the library configuration."""
    rows = []
    reachable = [row for row in query["starts"] if row["best_gap_mm"] is not None]
    for turns in ("given", "home", "given+home"):
        members = [row for row in reachable if row["best_turns"] == turns]
        rows.append(" & ".join([turns.replace("+", " and "), f"{len(members)}"]) + r" \\")
    write("prm_table_turns.tex", "\n".join(rows) + "\n")


def comparison_table(name: str, variants: Tuple[Tuple[str, str], ...]) -> None:
    """Roadmap variants side by side: size and cost rows, then what they answer.

    Args:
        name (str): output file name.
        variants (Tuple[Tuple[str, str], ...]): (column header, run name);
            runs missing on disk are left out.
    """
    present = [(label, run) for label, run in variants
               if os.path.isfile(os.path.join(ROOT, "prm", run, "roadmap.json.gz"))]
    roadmaps = [load_roadmap(ROOT, run) for _label, run in present]
    rows = build_rows(roadmaps) + [(None, [])] + query_rows([run for _label, run in present])
    sideways(name, "roadmap", [label for label, _run in present], rows, label_width_cm=6.2)


def ablation_table() -> None:
    """The 2000-node roadmap: three seeds, and two sampler changes."""
    comparison_table("prm_table_ablation.tex", (
        (r"\shortstack{seed 0\\(default)}", "B18_g2_n2000"), ("seed 1", "B18_g2_n2000_seed1"),
        ("seed 2", "B18_g2_n2000_seed2"), (r"\shortstack{every IK\\branch kept}", "B18_g2_n2000_allbranches"),
        (r"\shortstack{no line\\draws}", "B18_g2_n2000_noline")))


def experiment_e1_e4_table() -> None:
    """E1 (160-degree edges) and E4 (every IK branch with the default number of draws) next to their references."""
    comparison_table("prm_table_e1_e4.tex", (
        (r"\shortstack{4000 nodes\\$90^\circ$ (ref.)}", "B18_g2_n4000"),
        (r"\shortstack{4000 nodes\\$160^\circ$ (E1)}", "B18_g2_n4000_r160"),
        (r"\shortstack{every branch\\2272 draws}", "B18_g2_n2000_allbranches"),
        (r"\shortstack{every branch\\10k draws (E4)}", "B18_g2_allbranches_d10k"),
        (r"\shortstack{one branch\\10k draws (ref.)}", "B18_g2_n2000")))


def load_refine(run: str, tag: str) -> Optional[Dict[str, Any]]:
    """A saved refinement, or None when it was not run."""
    path = refine_path(ROOT, run, tag)
    return read_json(path) if os.path.isfile(path) else None


def load_side(run: str, name: str) -> Optional[Dict[str, Any]]:
    """A bridge result next to a refinement (``<tag>_validate.json``, ``<tag>_birrt_chains.json``)."""
    path = os.path.join(ROOT, "prm", run, "refine", name)
    return read_json(path) if os.path.isfile(path) else None


def refine_table(name: str, variants: Tuple[Tuple[str, str, str], ...]) -> None:
    """Refinements side by side: what was tracked, and how many starts got an exact path.

    Args:
        name (str): output file name.
        variants (Tuple[Tuple[str, str, str], ...]): (column header, run, refinement tag).
    """
    present = [(label, load_refine(run, tag), load_side(run, f"{tag}_validate.json"))
               for label, run, tag in variants]
    present = [item for item in present if item[1]]
    if not present:
        return

    def cells(get) -> List[str]:
        return [get(record, validate) for _label, record, validate in present]

    def reasons(record: Dict[str, Any], reason: str) -> str:
        return f"{sum(1 for r in record['edges']['reason'] if r == reason)}"

    rows = [
        ("starts within 30 mm on the roadmap: the ones refined (count)",
         cells(lambda r, v: f"{r['n_starts_considered']}")),
        ("edges tracked (count)", cells(lambda r, v: f"{len(r['edges']['row'])}")),
        ("\\quad of which the tracker turned into exact waypoints", cells(lambda r, v: reasons(r, "ok"))),
        ("\\quad failed: no IK solution at some waypoint", cells(lambda r, v: reasons(r, "ik_failure"))),
        ("\\quad failed: a joint jumped more than $10^\\circ$", cells(lambda r, v: reasons(r, "continuity"))),
        ("\\quad failed: a corrected waypoint collides", cells(lambda r, v: reasons(r, "collision"))),
        ("\\quad failed: changes elbow branch, so it cannot meet the next edge",
         cells(lambda r, v: reasons(r, "branch_change"))),
        ("edges re-tracked from a route's last waypoint at a node (count)", cells(lambda r, v: f"{r['retracks']}")),
        ("edges dropped because a route could not continue (count)",
         cells(lambda r, v: f"{len(r['join_blocked'])}")),
        (None, []),
    ]
    for k, tol in enumerate(TOLERANCES):
        rows.append((f"starts with an exact path from a route within {float(tol):g} mm, of those the "
                     f"roadmap brings within {float(tol):g} mm",
                     cells(lambda r, v, k=k: f"{r['tolerances'][k]['starts_exact']} of "
                                             f"{r['tolerances'][k]['starts_roadmap']}")))
    rows += [
        (None, []),
        ("best start: smallest exact gap (mm)", cells(lambda r, v: fmt(r["headline"].get("best_exact_gap_mm"), 2))),
        ("\\quad its start index", cells(lambda r, v: f"{r['headline'].get('best_start_index', '--')}")),
        ("\\quad waypoints of its exact path (count)", cells(lambda r, v: f"{r['headline'].get('n_waypoints', '--')}")),
        ("\\quad largest joint step between waypoints (deg)",
         cells(lambda r, v: fmt(r["headline"].get("max_step_deg"), 2))),
        ("\\quad holding error of its waypoints (mm)",
         cells(lambda r, v: f"{r['headline'].get('max_viol_pos_mm', float('nan')):.0e}")),
        ("refinement time (s, wall clock)", cells(lambda r, v: fmt(r["timings"]["total_s"], 0))),
        ("stored exact paths (one per start, plus the best and the shortest ones) passing the production repair at 2 mm steps",
         cells(lambda r, v: f"{v['n_ok']} of {len(v['results'])}" if v else "--")),
    ]
    sideways(name, "refinement", [label for label, _r, _v in present], rows, label_width_cm=7.6)


def validation_table(run: str, tag: str, solved: Dict[int, int]) -> None:
    """The production check of every stored exact path, by carry anchor, next to the BiRRT's solved starts."""
    record, check = load_refine(run, tag), load_side(run, f"{tag}_validate.json")
    if not record or not check:
        return
    carry_of = {row["start_index"]: row["carry"] for row in record["starts"]}
    verdict = {}
    for result in check["results"]:
        if result["label"].startswith("start_"):
            verdict[int(result["label"].split("_")[1])] = result["info"]
    columns = ("back", "horizontal", "vertical", "all")

    def per_carry(select) -> List[str]:
        return [f"{sum(1 for s in select() if carry == 'all' or carry_of.get(s) == carry)}" for carry in columns]

    birrt = {s for s, n in solved.items() if n > 0}
    passed = {s for s, info in verdict.items() if info["success"]}
    rows = [
        ("starts within 30 mm on the roadmap (count)", per_carry(lambda: carry_of.keys())),
        ("starts with an exact path from the roadmap (count)", per_carry(lambda: verdict.keys())),
        ("\\quad passing the production repair at 2 mm steps", per_carry(lambda: passed)),
        ("\\quad failing it on a collision between the 1 cm waypoints",
         per_carry(lambda: [s for s, i in verdict.items() if not i["success"] and i["reason"] == "collision"])),
        ("\\quad failing it otherwise", per_carry(lambda: [s for s, i in verdict.items()
                                                          if not i["success"] and i["reason"] != "collision"])),
        (None, []),
        ("starts the BiRRT solved at least once in experiment 2 (count)",
         [f"{sum(1 for s in birrt if carry == 'all' or carry_of.get(s) == carry)}" for carry in columns]),
        ("\\quad of which the roadmap also gives a production-checked path", per_carry(lambda: passed & birrt)),
    ]
    sideways("prm_table_validation.tex", "carry anchor", list(columns), rows, label_width_cm=8.0)


def refine_macros(run: str, tag: str, solved: Dict[int, int]) -> List[str]:
    """Numbers the refinement section quotes, as LaTeX macros (empty when the run is missing)."""
    record, check = load_refine(run, tag), load_side(run, f"{tag}_validate.json")
    if not record:
        return []
    answers = {f"{a['tolerance_mm']:g}": a for a in record["tolerances"]}
    edges = record["edges"]
    reasons: Dict[str, int] = defaultdict(int)
    for reason in edges["reason"]:
        reasons[reason] += 1
    exact = {row["start_index"]: row for row in record["starts"] if row["exact_gap_mm"] is not None}
    birrt = {s for s, n in solved.items() if n > 0}
    head = record["headline"]
    names = {"2": "Two", "5": "Five", "10": "Ten", "30": "Thirty"}
    lines = []
    for key, word in names.items():
        if key in answers:
            lines.append(f"\\newcommand{{\\PrmExact{word}}}{{{answers[key]['starts_exact']}}}")
            lines.append(f"\\newcommand{{\\PrmRoadmap{word}}}{{{answers[key]['starts_roadmap']}}}")
    lines += [
        f"\\newcommand{{\\PrmRefineEdges}}{{{len(edges['row'])}}}",
        f"\\newcommand{{\\PrmRefineEdgesOk}}{{{reasons['ok']}}}",
        f"\\newcommand{{\\PrmRefineBranchChange}}{{{reasons['branch_change']}}}",
        f"\\newcommand{{\\PrmRefineContinuity}}{{{reasons['continuity']}}}",
        f"\\newcommand{{\\PrmRefineIk}}{{{reasons['ik_failure']}}}",
        f"\\newcommand{{\\PrmRefineCollision}}{{{reasons['collision']}}}",
        f"\\newcommand{{\\PrmRefineSeconds}}{{{record['timings']['total_s']:.0f}}}",
        f"\\newcommand{{\\PrmRefineRounds}}{{{record['exact_gap_rounds']}}}",
        f"\\newcommand{{\\PrmExactHorizontal}}{{{sum(1 for r in exact.values() if r['carry'] == 'horizontal')}}}",
        f"\\newcommand{{\\PrmExactBack}}{{{sum(1 for r in exact.values() if r['carry'] == 'back')}}}",
        f"\\newcommand{{\\PrmExactBirrtOverlap}}{{{len(birrt & set(exact))}}}",
        f"\\newcommand{{\\PrmExactBestStart}}{{{head.get('best_start_index', '--')}}}",
        f"\\newcommand{{\\PrmExactBestWaypoints}}{{{head.get('n_waypoints', '--')}}}",
        f"\\newcommand{{\\PrmExactBestStep}}{{{head.get('max_step_deg', float('nan')):.1f}}}",
    ]
    if check:
        passed = {int(r["label"].split("_")[1]) for r in check["results"]
                  if r["label"].startswith("start_") and r["info"]["success"]}
        checked = [r for r in check["results"] if r["label"].startswith("start_")]
        lines += [
            f"\\newcommand{{\\PrmCheckPaths}}{{{len(checked)}}}",
            f"\\newcommand{{\\PrmCheckPassed}}{{{len(passed)}}}",
            f"\\newcommand{{\\PrmCheckPassedHorizontal}}{{{sum(1 for s in passed if exact.get(s, {}).get('carry') == 'horizontal')}}}",
            f"\\newcommand{{\\PrmCheckPassedBirrt}}{{{len(passed & birrt)}}}",
            f"\\newcommand{{\\PrmCheckSeconds}}{{{check['wall_s']:.0f}}}",
        ]
    return lines


def fig_trackability(record: Dict[str, Any], name: str) -> None:
    """Which edges the tracker turns into exact waypoints, against their roadmap score."""
    edges = record["edges"]
    score = np.asarray(edges["score_mm"])
    free = np.asarray(edges["free"], dtype=bool)
    dev = np.asarray(edges["max_dev_deg"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.6, 2.4))
    bins = np.array([0, 2, 5, 10, 15, 20, 30, 50])
    centres, share, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (score >= lo) & (score < hi)
        if sel.sum() >= 5:
            centres.append(f"{lo:g}-{hi:g}")
            share.append(100.0 * free[sel].mean())
            counts.append(int(sel.sum()))
    left.bar(range(len(share)), share, color="#2e9e5b")
    for x, (s, n) in enumerate(zip(share, counts)):
        left.text(x, s + 1, f"{n}", ha="center", fontsize=6)
    left.set_xticks(range(len(share)), centres, fontsize=6)
    left.set_ylim(0, 110)
    left.set_xlabel("roadmap edge score: worst holding error on the joint line (mm)")
    left.set_ylabel("edges tracked exactly (%)\n(number above: edges in the bin)")
    right.scatter(score[free], dev[free], s=3, alpha=0.4, color="#1f84bd", linewidths=0, label="tracked")
    right.set_xlabel("roadmap edge score (mm)")
    right.set_ylabel("largest joint correction (deg)")
    right.set_xlim(0, min(50, score.max() + 1) if len(score) else 50)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print(f"  wrote {name}")


CARRY_TAG = "carry_v1"
CARRY_COLOURS = {"horizontal": "#1f84bd", "vertical": "#d98324", "back": "#2e9e5b"}


def carry_tables() -> None:
    """The carry-pose search (E5): stage 1 per class, stage 2 for today's anchor and the best candidate."""
    folder = os.path.join(ROOT, "prm", "carry", CARRY_TAG)
    if not (os.path.isfile(os.path.join(folder, "stage1.json.gz")) and os.path.isfile(os.path.join(folder, "stage2.json"))):
        return
    stage1 = read_json(os.path.join(folder, "stage1.json.gz"))
    stage2 = read_json(os.path.join(folder, "stage2.json"))
    classes = ("back", "horizontal", "vertical")
    grid = stage1["grid"]

    def today(carry: str) -> Dict[str, Any]:
        return next((c for c in stage2["candidates"] if c["carry"] == carry and c["today"]), {})

    def best(carry: str) -> Dict[str, Any]:
        return next((c for c in stage2["ranking"][carry] if not c["today"]), {})

    def nudges(entry: Dict[str, Any]) -> str:
        row = next((r for r in stage1["finalists"] if r["carry"] == entry.get("carry")
                    and r["midpoint"] == entry.get("midpoint") and r["roll_deg"] == entry.get("roll_deg")), None)
        return f"{row.get('min_nudge_holdable', 0)} of 24" if row else "--"

    def place(entry: Dict[str, Any]) -> str:
        if not entry:
            return "--"
        x, y, z = entry["midpoint"]
        return f"({x:.2f}, {y:.2f}, {z:.2f}), ${entry['roll_deg']:g}^\\circ$"

    n_goals = max((c["n_goals"] for c in stage2["candidates"]), default=0)
    rows = [
        ("grid candidates holdable on both bars (of 4116 per class)",
         [f"{sum(1 for r in grid if r['carry'] == c and r['holdable_all'])}" for c in classes]),
        (None, []),
        ("today's anchor at its best roll: grasp midpoint (m), roll", [place(today(c)) for c in classes]),
        ("\\quad nudges survived on its worst bar", [nudges(today(c)) for c in classes]),
        (f"\\quad goals with an exact path from a route within 10 mm (of {n_goals})",
         [f"{today(c).get('exact_within', '--')}" for c in classes]),
        ("\\quad goals with any exact path (route within 30 mm)", [f"{today(c).get('exact_any', '--')}" for c in classes]),
        (None, []),
        ("best candidate: grasp midpoint (m), roll", [place(best(c)) for c in classes]),
        ("\\quad nudges survived on its worst bar", [nudges(best(c)) for c in classes]),
        (f"\\quad goals with an exact path from a route within 10 mm (of {n_goals})",
         [f"{best(c).get('exact_within', '--')}" for c in classes]),
        ("\\quad goals with any exact path (route within 30 mm)", [f"{best(c).get('exact_any', '--')}" for c in classes]),
        ("\\quad median exact gap where it has one (mm)", [fmt(best(c).get("median_exact_gap_mm")) for c in classes]),
    ]
    sideways("prm_table_carry_search.tex", "carry class", list(classes), rows, label_width_cm=6.0)


def fig_carry() -> None:
    """Where the carry candidates sit: side view (x, z) and top view (x, y) of the grasp midpoints."""
    folder = os.path.join(ROOT, "prm", "carry", CARRY_TAG)
    if not os.path.isfile(os.path.join(folder, "stage2.json")):
        return
    stage2 = read_json(os.path.join(folder, "stage2.json"))
    fig, (side, top) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for axis, (a, b), labels in ((side, (0, 2), ("x forward (m)", "z up (m)")), (top, (0, 1), ("x forward (m)", "y left (m)"))):
        for carry, colour in CARRY_COLOURS.items():
            members = [c for c in stage2["candidates"] if c["carry"] == carry]
            for c in members:
                marker = "*" if c["today"] else "o"
                size = 60 if c["today"] else 14 + 12 * c["exact_within"]
                axis.scatter(c["midpoint"][a], c["midpoint"][b], s=size, marker=marker, color=colour,
                             alpha=0.9 if c["today"] else 0.55, edgecolors="k" if c["today"] else "none", linewidths=0.4)
            pick = next((c for c in stage2["ranking"][carry] if not c["today"]), None)
            if pick:
                axis.scatter(pick["midpoint"][a], pick["midpoint"][b], s=90, facecolors="none", edgecolors=colour,
                             linewidths=1.2)
        # The two arm bases (robot description: x 0.1225 m, y +-0.149 m).
        if b == 1:
            axis.scatter([0.1225, 0.1225], [0.149, -0.149], marker="s", color="#555", s=18)
        axis.set_xlabel(labels[0])
        axis.set_ylabel(labels[1])
        axis.set_aspect("equal", adjustable="datalim")
    handles = [plt.Line2D([], [], marker="o", ls="", color=col, label=name) for name, col in CARRY_COLOURS.items()]
    handles += [plt.Line2D([], [], marker="*", ls="", color="#777", markeredgecolor="k", markersize=9, label="today's anchor"),
                plt.Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="#333", markersize=9,
                           label="current pick"),
                plt.Line2D([], [], marker="s", ls="", color="#555", label="arm bases")]
    top.legend(handles=handles, frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "prm_carry.pdf"))
    plt.close(fig)
    print("  wrote prm_carry.pdf")


def e3_tables(solved: Dict[int, int]) -> None:
    """E3: tube draws (way 2) and the seeded BiRRT (way 1) on what the plain refinement left."""
    run = "B18_g2_n4000"
    plain = load_refine(run, "all_k2.5_p10mm_mid")
    tube = load_refine(run, "all_k2.5_p10mm_mid_tube3")
    if not (plain and tube):
        return
    answers = lambda r, t: next(a for a in r["tolerances"] if a["tolerance_mm"] == t)  # noqa: E731
    rows = [(f"starts with an exact path from a route within {t:g} mm (count)",
             [f"{answers(plain, t)['starts_exact']}", f"{answers(tube, t)['starts_exact']}"])
            for t in (2.0, 5.0, 10.0, 30.0)]
    rows += [
        ("nodes added by tube draws (count)", ["0", f"{sum(r['nodes'] for r in tube['tube'])}"]),
        ("tube rounds run (count)", ["0", f"{len(tube['tube'])}"]),
        ("target starts improved, summed over rounds (count)",
         ["--", f"{sum(len(r['improved']) for r in tube['tube'])}"]),
        ("refinement time (s, wall clock)", [fmt(plain["timings"]["total_s"], 0), fmt(tube["timings"]["total_s"], 0)]),
    ]
    checks = [load_side(run, f"{tag}_validate.json") for tag in ("all_k2.5_p10mm_mid", "all_k2.5_p10mm_mid_tube3")]
    rows.append(("stored exact paths passing the production repair at 2 mm steps",
                 [f"{c['n_ok']} of {len(c['results'])}" if c else "--" for c in checks]))
    sideways("prm_table_tube.tex", "refinement", ["without tube draws", "with 3 tube rounds"], rows, label_width_cm=7.4)

    columns, cells = [], []
    for label, name in (("seeded with the exact chains", "chains"), ("from the two ends only", "nochains")):
        record = load_side(run, f"all_k2.5_p10mm_mid_tube3_birrt_{name}.json")
        if not record:
            continue
        results = record["results"]
        found = [r for r in results if r["repair"] is not None]
        ok = [r for r in found if r["repair"]["success"]]
        times = [r["info"]["wall_s"] for r in ok]
        solves = [r["info"]["n_ik_solves"] for r in ok]
        columns.append(label)
        cells.append([
            f"{len(results)}", f"{len(found)}", f"{len(ok)}",
            fmt(float(np.median(times)), 0) if times else "--",
            f"{int(np.median(solves))}" if solves else "--",
            f"{sum(1 for r in results if solved.get(r['start_index'], 0) > 0)}",
            f"{record['options']['max_time']:g}",
        ])
    if columns:
        labels = ("unresolved starts tried (count)", "a path found (count)",
                  "\\quad passing the production repair (count)",
                  "median time of a passing run (s)", "median IK solves of a passing run (count)",
                  "of the starts tried, solved by the plain BiRRT in experiment 2 (count)",
                  "time limit per start (s)")
        rows = [(label, [c[k] for c in cells]) for k, label in enumerate(labels)]
        sideways("prm_table_birrt.tex", "seeded BiRRT", columns, rows, label_width_cm=7.4)


HELD_OUT_GOALS = (2, 10, 20, 31, 47)


def precompute_table(name: str, run: str) -> None:
    """E6: what the night costs, and what one online query for an unseen goal costs and answers."""
    if not os.path.isfile(os.path.join(ROOT, "prm", run, "roadmap.json.gz")):
        return
    roadmap = load_roadmap(ROOT, run)
    tracked_path = os.path.join(ROOT, "prm", run, "tracked_10mm.json.gz")
    tracked = read_json(tracked_path) if os.path.isfile(tracked_path) else None
    queries = [(goal, load_query(run, f"online_g{goal:03d}"),
                load_refine(run, f"online_g{goal:03d}_k2.5_p10mm_mid")) for goal in HELD_OUT_GOALS]
    queries = [(g, q, r) for g, q, r in queries if q]
    if not queries:
        return
    summary = roadmap.meta["summary"]
    night = [
        ("night: nodes, of which anchors (count)",
         f"{roadmap.n_nodes}, {len(roadmap.anchors)}"),
        ("night: goals the roadmap is anchored on (count)", f"{len(roadmap.config.get('goals') or [])}"),
        ("night: build time (s)", fmt(roadmap.timings["total_s"], 0)),
        ("night: collision-free edges, of which within 10 mm (count)",
         f"{int(np.sum(np.asarray(roadmap.edges['free'], dtype=bool)))}, {len(tracked['records']) if tracked else '--'}"),
        ("night: tracking the edges within 10 mm (s)", fmt(tracked["wall_s"], 0) if tracked else "--"),
        ("night: components of the free graph (count)", f"{summary['n_components']}"),
    ]

    def cells(get) -> List[str]:
        return [get(q, r) for _g, q, r in queries]

    def timing(record: Optional[Dict[str, Any]], key: str) -> str:
        return fmt(record["timings"].get(key), 1) if record else "--"

    rows = [(label, [value] + [""] * (len(queries) - 1)) for label, value in night]
    rows += [
        (None, []),
        ("online query: start the workers (s)", cells(lambda q, r: timing(q, "pool_startup_s"))),
        ("\\quad solve the goal and the 3 carry poses (s)", cells(lambda q, r: timing(q, "endpoints_s"))),
        ("\\quad holding error of their edges (s)", cells(lambda q, r: timing(q, "endpoint_edges_s"))),
        ("\\quad gaps, with lazy collision checks (s)", cells(lambda q, r: timing(q, "gaps_s"))),
        ("\\quad whole query (s)", cells(lambda q, r: timing(q, "total_s"))),
        ("online refinement with the night's verdicts (s)", cells(lambda q, r: timing(r, "total_s"))),
        ("\\quad of which starting its workers (s)", cells(lambda q, r: timing(r, "pool_startup_s"))),
        (None, []),
        ("best gap from the carry poses to the goal (mm)",
         cells(lambda q, r: fmt_gap(q["headline"]["best_gap_mm"]))),
        ("carry poses with an exact path within 10 mm (of 3)",
         cells(lambda q, r: f"{r['tolerances'][-1]['starts_exact']}" if r else "--")),
    ]
    sideways(name, "unseen goal", [str(g) for g, _q, _r in queries], rows, label_width_cm=6.4)


def b3_table() -> None:
    """B3 goals 4, 11, 16: the night1 starts."""
    night1 = {4: {0: 0, 13: 4, 42: 8}, 11: {0: 0, 13: 2, 221: 2}, 16: {0: 0, 13: 8, 23: 3}}
    rows = []
    for goal, starts in night1.items():
        query = load_query(B3_RUN, f"g{goal}_night1")
        if not query:
            continue
        by_start = {row["start_index"]: row for row in query["starts"]}
        for start, successes in starts.items():
            row = by_start.get(start, {})
            rows.append(" & ".join([f"{goal}", f"{start}", str(row.get("carry", "")), str(row.get("difficulty", "")),
                                    f"{successes}/10", fmt_gap(row.get("best_gap_mm")),
                                    fmt_gap(row.get("library_conf_gap_mm")),
                                    str(row.get("best_pair", "--"))]) + r" \\")
    write("prm_table_b3.tex", "\n".join(rows) + "\n")


def macros(runs: Dict[int, Any], query: Optional[Dict[str, Any]], solved: Dict[int, int]) -> None:
    """A few single numbers the report's text quotes, as LaTeX macros."""
    largest = max(runs)
    roadmap = runs[largest]
    edges = roadmap.edge_arrays()
    free = edges["free"].astype(bool)
    score = np.maximum(edges["viol_pos_max"] * 1000.0, np.degrees(edges["viol_rot_max"]) / 0.4)
    lengths = np.degrees(edges["len_linf"])
    lines = [
        f"\\newcommand{{\\PrmLargestBudget}}{{{largest}}}",
        f"\\newcommand{{\\PrmNodesLargest}}{{{roadmap.n_nodes}}}",
        f"\\newcommand{{\\PrmFreeEdgeShareLargest}}{{{100.0 * free.mean():.0f}}}",
        f"\\newcommand{{\\PrmMedianScoreShort}}{{{np.median(score[free & (lengths < 20)]):.1f}}}",
        f"\\newcommand{{\\PrmMedianScoreLong}}{{{np.median(score[free & (lengths >= 60)]):.1f}}}",
        f"\\newcommand{{\\PrmBirrtSolvedStarts}}{{{sum(1 for v in solved.values() if v > 0)}}}",
        f"\\newcommand{{\\PrmBirrtStarts}}{{{len(solved)}}}",
    ]
    if query and query["headline"]["best_gap_mm"] is not None:
        head = query["headline"]
        route = head["minimax_route"]
        # The goal pair in the LIBRARY's numbering (the query numbers pairs in
        # the solver's own order): match the route's last configuration.
        goal = find_goal(load_library_entry(ROOT, "cdfm_v1", "B18"), 2)
        goal_conf = np.asarray(route["dense_confs"][-1])
        library_pair = next((p["pair_index"] for p in goal["free_branch_pairs"]
                             if wrapped_match(goal_conf, p["conf"], tol=1e-4)), None)
        better_branch = sum(1 for row in query["starts"]
                            if row["best_gap_mm"] is not None and (row["library_conf_gap_mm"] is None
                                                                   or row["best_gap_mm"] < 0.99 * row["library_conf_gap_mm"]))
        reachable = sum(1 for row in query["starts"] if row["best_gap_mm"] is not None)
        solved_gaps = [row["best_gap_mm"] for row in query["starts"]
                       if solved.get(row["start_index"], 0) > 0 and row["best_gap_mm"] is not None]
        lines += [
            f"\\newcommand{{\\PrmBestGoalLibraryPair}}{{{library_pair if library_pair is not None else '--'}}}",
            f"\\newcommand{{\\PrmBestStartIsLibraryConf}}{{{'yes' if route['start']['library_conf'] else 'no'}}}",
            f"\\newcommand{{\\PrmBetterBranchStarts}}{{{better_branch}}}",
            f"\\newcommand{{\\PrmReachableStarts}}{{{reachable}}}",
            f"\\newcommand{{\\PrmSolvedMedianGap}}{{{np.median(solved_gaps):.1f}}}" if solved_gaps else "",
        ]
        lines += [
            f"\\newcommand{{\\PrmBestGap}}{{{head['best_gap_mm']:.2f}}}",
            f"\\newcommand{{\\PrmBestStart}}{{{head['best_start_index']}}}",
            f"\\newcommand{{\\PrmBestRouteNodes}}{{{len(route['nodes'])}}}",
            f"\\newcommand{{\\PrmBestRouteLength}}{{{route['length_rad']:.2f}}}",
            f"\\newcommand{{\\PrmBestRouteMaxStep}}{{{np.degrees(route['max_step_rad']):.2f}}}",
            f"\\newcommand{{\\PrmBestStartPair}}{{{route['start']['pair']}}}",
            f"\\newcommand{{\\PrmBestStartTurn}}{{{route['start']['turns']}}}",
            f"\\newcommand{{\\PrmBestGoalPair}}{{{route['goal']['pair']}}}",
            f"\\newcommand{{\\PrmGoalNodes}}{{{query['goal']['n_goal_nodes']}}}",
            f"\\newcommand{{\\PrmGoalLibraryPairs}}{{{query['goal']['n_library_pairs']}}}",
            f"\\newcommand{{\\PrmGoalLibraryPairsFound}}{{{query['goal']['n_library_pairs_found']}}}",
            f"\\newcommand{{\\PrmStartNodes}}{{{query['n_start_nodes']}}}",
            f"\\newcommand{{\\PrmVerifyRounds}}{{{query['verify_rounds']}}}",
            f"\\newcommand{{\\PrmLazyChecked}}{{{int(query['counters']['lazy_checks'].get('edges_checked', 0))}}}",
            f"\\newcommand{{\\PrmEndpointEdges}}{{{len(query['endpoint_edges']['i'])}}}",
        ]
    lines += refine_macros(B18_RUN.format(largest), "all_k2.5_p10mm_mid", solved)
    write("prm_macros.tex", "\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# * Figures
# -----------------------------------------------------------------------------


def plain_log_axis(axis, ticks) -> None:
    """A log axis labelled with plain numbers at the given ticks only (no 3x10^3 clutter)."""
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def fig_violation_vs_length(roadmap) -> None:
    """How the holding error of a joint-space line grows with its length."""
    edges = roadmap.edge_arrays()
    free = edges["free"].astype(bool)
    length = np.degrees(edges["len_linf"][free])
    score = np.maximum(edges["viol_pos_max"][free] * 1000.0, np.degrees(edges["viol_rot_max"][free]) / 0.4)
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.scatter(length, score, s=2, alpha=0.15, color="#555", linewidths=0)
    bins = np.arange(0, 95, 10)
    centres, medians, low, high = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (length >= lo) & (length < hi)
        if sel.sum() >= 5:
            centres.append((lo + hi) / 2)
            medians.append(np.median(score[sel]))
            low.append(np.percentile(score[sel], 25))
            high.append(np.percentile(score[sel], 75))
    ax.plot(centres, medians, "o-", color="#b23a48", ms=3, label="median per 10$^\\circ$ bin")
    ax.fill_between(centres, low, high, color="#b23a48", alpha=0.2, label="25--75 %")
    for tol in (2, 5, 10, 30):
        ax.axhline(tol, color="#2e9e5b", lw=0.6, ls=":")
    ax.set_yscale("log")
    plain_log_axis(ax.yaxis, [1, 2, 5, 10, 30, 100, 300])
    ax.set_xlabel("edge length: largest joint move (deg)")
    ax.set_ylabel("worst holding error on the edge (mm score)")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(os.path.join(OUT, "prm_viol_vs_length.pdf"))
    plt.close(fig)
    print("  wrote prm_viol_vs_length.pdf")


def fig_budget(runs: Dict[int, Any]) -> None:
    """Best gap and connected starts against the node budget."""
    budgets = list(runs)
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.6, 2.4))
    series = defaultdict(list)
    for budget in budgets:
        pair = load_query(B18_RUN.format(budget), "s0_s12")
        every = load_query(B18_RUN.format(budget), "all")
        gaps = {row["start_index"]: row["best_gap_mm"] for row in (pair or {}).get("starts", [])}
        series["start 0 (BiRRT 0/10 in night1)"].append(gaps.get(0))
        series["start 12 (BiRRT 6/10 in night1)"].append(gaps.get(12))
        series["best of all 807 starts"].append(every["headline"]["best_gap_mm"] if every else None)
        for k, tol in enumerate(TOLERANCES):
            series[f"within {float(tol):g} mm"].append(
                100.0 * every["tolerances"][k]["starts_connected"] / len(every["starts"]) if every else None)
    for name, marker in (("start 0 (BiRRT 0/10 in night1)", "s"), ("start 12 (BiRRT 6/10 in night1)", "^"),
                         ("best of all 807 starts", "o")):
        values = [np.nan if v is None else v for v in series[name]]
        left.plot(budgets, values, marker=marker, ms=3, label=name)
    left.set_xscale("log")
    left.set_yscale("log")
    plain_log_axis(left.xaxis, budgets)
    plain_log_axis(left.yaxis, [2, 3, 5, 10, 20, 50, 100])
    left.set_xlabel("sampled nodes")
    left.set_ylabel("gap to the goal set (mm score)")
    left.legend(frameon=False)
    for tol in TOLERANCES:
        values = [np.nan if v is None else v for v in series[f"within {float(tol):g} mm"]]
        right.plot(budgets, values, marker="o", ms=3, label=f"within {float(tol):g} mm")
    right.set_xscale("log")
    plain_log_axis(right.xaxis, budgets)
    right.set_xlabel("sampled nodes")
    right.set_ylabel("starts connected to the goal set (%)")
    right.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "prm_budget.pdf"))
    plt.close(fig)
    print("  wrote prm_budget.pdf")


def fig_gap_by_class(query: Dict[str, Any], solved: Dict[int, int]) -> None:
    """Distribution of per-start gaps, by library class, with the BiRRT's solved starts marked."""
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for label, colour in CLASS_COLOURS.items():
        gaps = np.sort([row["best_gap_mm"] for row in query["starts"]
                        if row["difficulty"] == label and row["best_gap_mm"] is not None])
        n = sum(1 for row in query["starts"] if row["difficulty"] == label)
        if n:
            ax.step(gaps, np.arange(1, len(gaps) + 1) / n, where="post", color=colour, label=f"{label} ({n})")
    solved_gaps = [row["best_gap_mm"] for row in query["starts"]
                   if solved.get(row["start_index"], 0) > 0 and row["best_gap_mm"] is not None]
    ax.plot(solved_gaps, np.full(len(solved_gaps), -0.04), "|", color="k", ms=8,
            label="starts the BiRRT solved (exp. 2)")
    ax.set_xscale("log")
    plain_log_axis(ax.xaxis, [2, 5, 10, 20, 50, 100, 300])
    ax.set_xlabel("gap to the goal set (mm score)")
    ax.set_ylabel("share of starts")
    ax.legend(frameon=False, loc="lower right", fontsize=6)
    fig.savefig(os.path.join(OUT, "prm_gap_by_class.pdf"))
    plt.close(fig)
    print("  wrote prm_gap_by_class.pdf")


def fig_profile(query: Dict[str, Any]) -> None:
    """Holding error along the best route and along the shortest path at the best gap."""
    fig, ax = plt.subplots(figsize=(6.6, 2.2))
    head = query["headline"]
    for key, label, colour in (("minimax_route", "best (minimax) route", "#b23a48"),
                               ("shortest_at_best_gap", "shortest path within the best gap", "#1f84bd")):
        report = head.get(key)
        if not report:
            continue
        s = np.asarray(report["dense_s"])
        ax.plot(np.degrees(s), report["profile_score_mm"], color=colour, lw=1,
                label=f"{label}: {len(report['nodes'])} nodes, {np.degrees(report['length_rad']):.0f} deg")
        # Mark the nodes: where the profile is back at zero.
        nodes_at = np.flatnonzero(np.asarray(report["profile_score_mm"]) < 1e-3)
        ax.plot(np.degrees(s[nodes_at]), np.zeros(len(nodes_at)), "o", color=colour, ms=2)
    ax.axhline(head["best_gap_mm"], color="#555", lw=0.6, ls=":")
    # Room above the curves for the legend.
    ax.set_ylim(0.0, 1.45 * head["best_gap_mm"])
    ax.set_xlabel("joint-space distance travelled (deg, Euclidean over 12 joints)")
    ax.set_ylabel("holding error (mm score)")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.savefig(os.path.join(OUT, "prm_profile.pdf"))
    plt.close(fig)
    print("  wrote prm_profile.pdf")


def fig_cost(runs: Dict[int, Any]) -> None:
    """Where the build time goes, and the counts behind it."""
    budgets = list(runs)
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.6, 2.3))
    phases = (("pool_startup_s", "worker start"), ("anchors_s", "anchors"), ("sample_s", "sampling"),
              ("knn_s", "neighbours"), ("edges_s", "edges"))
    bottom = np.zeros(len(budgets))
    x = np.arange(len(budgets))
    for key, label in phases:
        values = np.array([runs[b].timings.get(key, 0.0) for b in budgets])
        left.bar(x, values, bottom=bottom, label=label, width=0.6)
        bottom += values
    left.set_xticks(x, [str(b) for b in budgets])
    left.set_xlabel("sampled nodes")
    left.set_ylabel("wall-clock time (s), 8 workers")
    left.legend(frameon=False, fontsize=6)
    ik = [runs[b].counters["sample"].get("ik_calls", 0) for b in budgets]
    cc_sample = [runs[b].counters["sample"].get("collision_checks", 0) for b in budgets]
    cc_edges = [runs[b].counters["edges"].get("collision_checks", 0) for b in budgets]
    fk = [runs[b].counters["edges"].get("fk_pairs", 0) for b in budgets]
    right.plot(budgets, ik, "o-", ms=3, label="IK calls (sampling)")
    right.plot(budgets, cc_sample, "s-", ms=3, label="collision checks (sampling)")
    right.plot(budgets, cc_edges, "^-", ms=3, label="collision checks (edges)")
    right.plot(budgets, fk, "d-", ms=3, label="holding-error points (edges)")
    right.set_xscale("log")
    right.set_yscale("log")
    plain_log_axis(right.xaxis, budgets)
    right.set_xlabel("sampled nodes")
    right.set_ylabel("count")
    right.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "prm_cost.pdf"))
    plt.close(fig)
    print("  wrote prm_cost.pdf")


def main() -> None:
    """Make every figure and table the report uses."""
    if not ROOT:
        raise SystemExit("Set HUSKY_BENCH_ROOT to the benchmark data root.")
    runs = {}
    for budget in BUDGETS:
        name = B18_RUN.format(budget)
        if os.path.isfile(os.path.join(ROOT, "prm", name, "roadmap.json.gz")):
            runs[budget] = load_roadmap(ROOT, name)
    if not runs:
        raise SystemExit("No B18 roadmap runs found under <root>/prm.")
    solved = birrt_solved_starts()
    largest = max(runs)
    query = load_query(B18_RUN.format(largest), "all")
    build_table(runs)
    query_table(runs)
    if query:
        class_table(query, solved)
        carry_table(query, solved)
        turn_table(query)
        fig_gap_by_class(query, solved)
        fig_profile(query)
    b3_table()
    ablation_table()
    experiment_e1_e4_table()
    refine_table("prm_table_refine.tex", (
        (r"\shortstack{4000 nodes\\knots $2.5^\circ$}", "B18_g2_n4000", "all_k2.5_p10mm_mid"),
        (r"\shortstack{4000 nodes\\end nodes only}", "B18_g2_n4000", "all_nodes_p10mm_mid"),
        (r"\shortstack{2000 nodes\\seed 0}", "B18_g2_n2000", "all_k2.5_p10mm_mid"),
        (r"\shortstack{2000 nodes\\seed 1}", "B18_g2_n2000_seed1", "all_k2.5_p10mm_mid"),
        (r"\shortstack{2000 nodes\\seed 2}", "B18_g2_n2000_seed2", "all_k2.5_p10mm_mid")))
    validation_table("B18_g2_n4000", "all_k2.5_p10mm_mid", solved)
    carry_tables()
    fig_carry()
    e3_tables(solved)
    precompute_table("prm_table_precompute_6k.tex", "B18_overnight")
    precompute_table("prm_table_precompute_30k.tex", "B18_night30k")
    refined = load_refine("B18_g2_n4000", "all_k2.5_p10mm_mid")
    if refined:
        fig_trackability(refined, "prm_trackability.pdf")
    macros(runs, query, solved)
    fig_violation_vs_length(runs[largest])
    fig_budget(runs)
    fig_cost(runs)
    print("  queries of the largest run:", [q["tag"] for q in list_queries(ROOT, B18_RUN.format(largest))])


if __name__ == "__main__":
    main()
