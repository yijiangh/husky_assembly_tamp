// The dashboard: four views, from the whole batch down to one motion.
//
// State lives entirely in the URL hash, so any view can be linked to or
// reloaded. The 3D panel and the loaded robot are created once and shared,
// because rebuilding them on every navigation would be far slower than
// anything else the page does.
//
//   #/                                      the whole batch
//   #/bar/B12                               one bar's goals
//   #/bar/B12/goal/7                        one goal's start survey
//   #/bar/B12/goal/7/start/42               one start's seeds and motions

import { loadJson, loadJsonGz, loadTrajectory, paths } from "./data.js";
import {
  COLOURS, BREAKDOWN_ORDER, BREAKDOWN_LABELS, SETUP_ORDER, SETUP_LABELS,
  SETUP_COLOURS, STOP_REASON_ORDER, STOP_REASON_LABELS, STOP_REASON_COLOURS,
  groupStopReasons, stopReasonVerdict,
  barChart, boxChart, heatmap, lineChart, stackedBarChart, copyAsCsv,
} from "./charts.js";
import { OUTCOME_COLOURS, RobotViewer } from "./robot-viewer.js";

const main = document.querySelector("[data-main]");
const breadcrumb = document.querySelector("[data-breadcrumb]");
const viewerHost = document.querySelector("[data-viewer]");
const viewerCaption = document.querySelector("[data-viewer-caption]");
const stageControls = document.querySelector("[data-stage-controls]");

let viewer = null;
let runInfo = null;

// Every exported run, newest first, read from the catalogue the exporter keeps
// at the data root (runs.json). Null until loaded. If the catalogue cannot be
// read, this holds the reason instead, and the header says so.
let runList = null;
let runListError = null;

// The catalogue sits at the data root, three folders above this page
// (runs/<name>/dashboard/), the same place the shared robot assets are found.
const DATA_ROOT = "../../..";
let robotReady = null;
const layout = document.querySelector("[data-layout]");

// Which carry poses the goal page draws: "workable" or "all". Kept for the
// session, so stepping from goal to goal does not reset the choice.
let goalPoseMode = "workable";

// ---------------------------------------------------------------------------
// Small helpers for building the page
// ---------------------------------------------------------------------------

function element(tag, attributes = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of [].concat(children)) {
    node.append(child instanceof Node ? child : document.createTextNode(child));
  }
  return node;
}

function panel(title, subtitle = null) {
  const head = [element("h2", {}, title)];
  if (subtitle) head.push(element("p", { class: "muted" }, subtitle));
  return element("section", { class: "panel" }, head);
}

function withCsv(section, rows) {
  const button = element("button", { class: "csv" }, "copy as CSV");
  button.addEventListener("click", () => copyAsCsv(rows, button));
  section.appendChild(button);
  return section;
}

function table(headers, rows, { onRowClick = null } = {}) {
  const head = element("tr", {}, headers.map((h) => element("th", {}, h)));
  const body = rows.map((row) => {
    const tr = element("tr", {}, row.map((cell) =>
      element("td", {}, cell instanceof Node ? cell : String(cell))));
    if (onRowClick) {
      tr.className = "clickable";
      tr.addEventListener("click", () => onRowClick(row));
    }
    return tr;
  });
  return element("table", { class: "grid" },
    [element("thead", {}, [head]), element("tbody", {}, body)]);
}

// The trail under the title: run / bar / goal / start.
//
// The first entry is always the run. It is drawn as a dropdown of every
// exported run, so switching data sets does not mean editing the address.
// Because a dropdown cannot also be a link, a separate "overview" link follows
// it on the deeper pages, standing in for what clicking the run name used to do.
function setBreadcrumb(parts) {
  const [runPart, ...rest] = parts;
  const nodes = [runPicker()];
  if (runPart.href) {
    nodes.push(element("span", { class: "sep" }, "/"),
      element("a", { href: runPart.href }, "overview"));
  }
  for (const part of rest) {
    const node = part.href
      ? element("a", { href: part.href }, part.label)
      : element("span", {}, part.label);
    nodes.push(element("span", { class: "sep" }, "/"), node);
  }
  breadcrumb.replaceChildren(...nodes);
}

// ---------------------------------------------------------------------------
// Switching between runs
// ---------------------------------------------------------------------------

// One line per run in the dropdown: its name, size and how often it solved.
function runOptionLabel(entry) {
  const count = Number(entry.n_runs || 0).toLocaleString();
  const solved = entry.success_rate == null
    ? "?" : (100 * entry.success_rate).toFixed(1);
  return `${entry.run_name}  (${count} runs, ${solved}% solved)`;
}

// Builds the run dropdown, or a plain label plus the reason when the list of
// runs could not be read, so a missing catalogue is visible rather than silent.
function runPicker() {
  if (!runList) {
    const reason = runListError || "the list of runs is still loading";
    return element("span", { title: reason }, [
      runInfo.run_name,
      element("span", { class: "picker-note" }, ` (no run list: ${reason})`),
    ]);
  }
  const select = element("select", {
    class: "run-picker",
    "aria-label": "Benchmark run to show",
  }, runList.map((entry) => {
    const option = element("option", { value: entry.run_name },
      runOptionLabel(entry));
    if (entry.run_name === runInfo.run_name) option.selected = true;
    return option;
  }));
  select.addEventListener("change", () => {
    const entry = runList.find((row) => row.run_name === select.value);
    if (entry) switchRun(entry, select);
  });
  return select;
}

// The data file the current view depends on most deeply, relative to a run's
// dashboard folder, or null on the overview. Used to ask whether another run
// has the same bar, goal or start before jumping straight to it.
function deepestFileFor(parts) {
  if (parts[0] !== "bar" || parts.length < 2) return null;
  if (parts[4] === "start") {
    return paths.start(parts[1], Number(parts[3]), Number(parts[5]));
  }
  if (parts[2] === "goal") return paths.goal(parts[1], Number(parts[3]));
  return paths.bar(parts[1]);
}

// Opens the same view in another run. Runs do not all share the same bars and
// goals (the smoke tests used only a few), so the page first checks that the
// file behind the current view exists in the chosen run; if not, it opens that
// run's overview instead of an error page.
async function switchRun(entry, select) {
  const target = `${DATA_ROOT}/${entry.path}/`;
  const hash = window.location.hash.replace(/^#\/?/, "");
  const parts = hash.split("/").filter(Boolean);
  const file = deepestFileFor(parts);
  let keepView = !file;
  select.disabled = true;
  if (file) {
    try {
      const response = await fetch(target + file, { method: "HEAD" });
      keepView = response.ok;
    } catch (error) {
      keepView = false;
    }
  }
  window.location.href = target + (keepView ? `#/${parts.join("/")}` : "#/");
}

// Reads the catalogue of runs. Failure is recorded, not thrown: the run being
// viewed still works without it, and the header shows why the dropdown is
// missing (usually the server was started on this run's folder instead of the
// data folder, see DASHBOARD.md).
async function loadRunList() {
  try {
    const catalog = await loadJson(`${DATA_ROOT}/runs.json`);
    runList = (catalog.runs || []).slice();
    // The run on screen must be choosable even if the catalogue predates it.
    if (!runList.some((row) => row.run_name === runInfo.run_name)) {
      runList.unshift({ run_name: runInfo.run_name, path: "." });
    }
  } catch (error) {
    runListError = String(error && error.message ? error.message : error);
  }
}

async function ensureRobot() {
  if (!viewer) viewer = new RobotViewer(viewerHost);
  if (!robotReady) {
    const assets = runInfo.assets || "../../../assets";
    robotReady = viewer.loadRobot(
      `${assets}/scene.glb`, `${assets}/kinematics.json`);
  }
  await robotReady;
  return viewer;
}

function showViewer(caption) {
  layout.classList.add("split");
  viewerHost.parentElement.hidden = false;
  viewerCaption.textContent = caption;
  if (viewer) viewer.resize();
}

function hideViewer() {
  layout.classList.remove("split");
  viewerHost.parentElement.hidden = true;
  clearStageControls();
}

// Whether a recorded collision pair holds real names.
//
// Libraries built before a recording fix stored each robot link as its object
// address ("<compas_robots.model.link.Link object at 0x...>"), which names
// nothing. Those are recognised and left out rather than displayed.
function isReadablePair(pair) {
  return pair.every((part) => !/ object at 0x[0-9a-f]+>$/i.test(String(part)));
}

// A small coloured square for a legend entry.
function swatch(hex) {
  const node = element("i");
  node.style.background = cssColour(hex);
  return node;
}

// Anything that drives the 3D view belongs directly under it, not further down
// the page: a scrubber you cannot see the robot from is no use.
function setStageControls(...nodes) {
  stageControls.replaceChildren(...nodes);
  stageControls.hidden = nodes.length === 0;
}

function clearStageControls() {
  stageControls.replaceChildren();
  stageControls.hidden = true;
}

// ---------------------------------------------------------------------------
// The overview: how the planner did across every bar
// ---------------------------------------------------------------------------

async function viewOverview() {
  hideViewer();
  setBreadcrumb([{ label: runInfo.run_name }]);
  const summary = await loadJson(paths.summary());
  const bars = summary.per_bar.slice()
    .sort((a, b) => a.bar_length_m - b.bar_length_m);

  main.replaceChildren();

  const totals = summary.totals;
  const setupTotal = setupSeconds(totals.setup_s_per_goal);
  // Two different clocks, kept apart on purpose. Motion planning is what the
  // planner spends finding a motion. Setup is the work done beforehand to
  // pose the question -- finding the goal and the start poses -- and is not
  // part of any planning run.
  main.appendChild(element("section", { class: "panel tiles" }, [
    tile("runs", totals.n_runs),
    tile("solved", `${(totals.success_rate * 100).toFixed(0)}%`),
    tile("motion planning, median of solved runs",
      `${totals.fine_wall_s.p50.toFixed(0)} s`),
    tile("setup per goal, before planning",
      setupTotal ? `${(setupTotal / 60).toFixed(0)} min` : "-"),
    tile("bars", bars.length),
  ]));

  // 1. How often each bar's transfers were solved.
  const successPanel = panel(
    "Solved, by bar",
    "Bars ordered by length. The whisker is a 95% interval; the hollow " +
    "marker is how often the coarse pass alone succeeded.");
  const successChart = element("div");
  successPanel.appendChild(successChart);
  main.appendChild(withCsv(successPanel, [
    ["bar", "length_m", "runs", "success_rate", "ci_low", "ci_high", "coarse_rate"],
    ...bars.map((b) => [b.bar_id, b.bar_length_m, b.n_runs,
      b.success_rate, b.success_rate_ci95[0], b.success_rate_ci95[1],
      b.coarse_success_rate]),
  ]));
  barChart(successChart, {
    yLabel: "solved", yMax: 1,
    format: (v) => `${(v * 100).toFixed(0)}%`,
    data: bars.map((b) => ({
      label: `${b.bar_id}\n${b.bar_length_m.toFixed(2)}m`,
      value: b.success_rate,
      low: b.success_rate_ci95[0], high: b.success_rate_ci95[1],
      marker: b.coarse_success_rate,
      colour: COLOURS.fine,
    })),
    onClick: (d) => { window.location.hash = `#/bar/${d.label.split("\n")[0]}`; },
  });

  // 2. How long a solved transfer took.
  const timePanel = panel(
    "Motion planning time, by bar",
    "Time inside the planner finding a motion: the fine pass of each solved " +
    "run, on a log scale. The box spans the 10th to 90th percentile, the line " +
    "is the median. It does not include finding the goal or start poses; " +
    "that is setup, shown next.");
  const timeChart = element("div");
  timePanel.appendChild(timeChart);
  main.appendChild(withCsv(timePanel, [
    ["bar", "p10_s", "p50_s", "p90_s", "max_s"],
    ...bars.map((b) => [b.bar_id, b.fine_wall_s.p10, b.fine_wall_s.p50,
      b.fine_wall_s.p90, b.fine_wall_s.max]),
  ]));
  boxChart(timeChart, {
    yLabel: "seconds",
    data: bars.map((b) => ({ label: b.bar_id, ...b.fine_wall_s })),
  });

  // 2b. Setup: the work done before any motion planning.
  const setupPanel = panel(
    "Setup time, by bar (not motion planning)",
    "Minutes per goal spent posing the benchmark's questions, before any " +
    "motion planning: goal IK finds a reachable goal pose; the start survey " +
    "runs IK and collision checks on every candidate carry pose; the " +
    "straight-line probes test each workable start; and the planner's own " +
    "start search is run once for comparison. Measured while 16 bars were " +
    "surveyed in parallel, so roughly 1.5 to 2 times single-core time.");
  const setupChart = element("div");
  setupPanel.appendChild(setupChart);
  main.appendChild(withCsv(setupPanel, [
    ["bar", ...SETUP_ORDER.map((k) => `${k}_s_per_goal`)],
    ...bars.map((b) => [b.bar_id,
      ...SETUP_ORDER.map((k) => (b.setup_s_per_goal || {})[k] || 0)]),
  ]));
  stackedBarChart(setupChart, {
    keys: SETUP_ORDER, labels: SETUP_LABELS, colours: SETUP_COLOURS,
    normalize: false, yLabel: "minutes per goal",
    format: (v) => `${v.toFixed(1)}`,
    data: bars.map((b) => ({
      label: b.bar_id,
      values: Object.fromEntries(SETUP_ORDER.map(
        (k) => [k, ((b.setup_s_per_goal || {})[k] || 0) / 60])),
    })),
  });

  // 3. Does trying again help?
  const restartPanel = panel(
    "Trying again",
    "How often a problem is solved by at least one of the first few seeds. " +
    "A curve that keeps climbing means the planner is worth restarting.");
  const restartChart = element("div");
  restartPanel.appendChild(restartChart);
  main.appendChild(restartPanel);
  lineChart(restartChart, {
    xLabel: "seeds tried", yLabel: "problems solved",
    series: [{
      points: (totals.restart_curve || []).map((value, index) => [index + 1, value]),
      colour: COLOURS.fine, width: 2,
    }],
  });

  // 4. Why the searches stopped. The headline sentence is over the whole run;
  //    the chart is the same thing per bar.
  const runReasons = groupStopReasons(totals.extend_stop_reasons);
  if (Object.values(runReasons).some(Boolean)) {
    const reasonPanel = panel(
      "Why extensions stopped",
      "Each attempt to grow a search tree ends for a reason. A search whose "
      + "attempts mostly die on the continuity gate is rejecting its own arm "
      + "solutions; one that mostly collides is being stopped by the scene.");
    reasonPanel.appendChild(element("p", { class: "verdict" },
      stopReasonVerdict(runReasons)));
    const reasonChart = element("div");
    reasonPanel.appendChild(reasonChart);
    main.appendChild(reasonPanel);
    stackedBarChart(reasonChart, {
      keys: STOP_REASON_ORDER, labels: STOP_REASON_LABELS,
      colours: STOP_REASON_COLOURS, yLabel: "share of stops",
      data: bars.map((b) => ({
        label: b.bar_id, values: groupStopReasons(b.extend_stop_reasons),
      })),
      format: countOrShare,
    });
  }

  // 5. Where motion planning time went (inside the planner only).
  const breakdownPanel = panel(
    "Where motion planning time went",
    "Share of time inside the planner, by what it was doing. 'IK while " +
    "growing the tree' is the arm IK solved at every step of the search; it " +
    "is not the goal IK, which happens during setup.");
  const breakdownChart = element("div");
  breakdownPanel.appendChild(breakdownChart);
  main.appendChild(withCsv(breakdownPanel, [
    ["bar", ...BREAKDOWN_ORDER],
    ...bars.map((b) => [b.bar_id,
      ...BREAKDOWN_ORDER.map((k) => (b.mean_breakdown_s || {})[k] || 0)]),
  ]));
  stackedBarChart(breakdownChart, {
    keys: BREAKDOWN_ORDER, labels: BREAKDOWN_LABELS, yLabel: "share of time",
    data: bars.map((b) => ({ label: b.bar_id, values: b.mean_breakdown_s || {} })),
  });

  // 6. The bars themselves, as a table.
  const listPanel = panel("Bars", "Click a row to open it.");
  listPanel.appendChild(table(
    ["bar", "length", "runs", "solved", "coarse only",
      "planning s, median solved", "setup min per goal"],
    bars.map((b) => [b.bar_id, `${b.bar_length_m.toFixed(2)} m`, b.n_runs,
      `${(b.success_rate * 100).toFixed(0)}%`,
      `${(b.coarse_success_rate * 100).toFixed(0)}%`,
      b.n_success ? b.fine_wall_s.p50.toFixed(0) : "-",
      (setupSeconds(b.setup_s_per_goal) / 60).toFixed(1)]),
    { onRowClick: (row) => { window.location.hash = `#/bar/${row[0]}`; } },
  ));
  main.appendChild(listPanel);

  // 7. Comparisons of several batches (planner variants), when any exist.
  // The index lives at the data root; most data roots have none, and a
  // missing file is the normal case, not an error.
  let comparisons = [];
  try {
    comparisons = (await loadJson(paths.compareIndex())).comparisons || [];
  } catch (error) {
    if (!String(error && error.message).includes("HTTP 404")) throw error;
  }
  if (comparisons.length) {
    const comparePanel = panel(
      "Comparisons",
      "Several batches, usually planner variants, measured on the runs they " +
      "have in common. Click a row to open it.");
    comparePanel.appendChild(table(
      ["comparison", "baseline", "batches", "runs in common", "problems"],
      comparisons.map((c) => [c.name, c.baseline, c.run_names.join(", "),
        c.n_common_runs, c.n_problems]),
      { onRowClick: (row) => { window.location.hash = `#/compare/${row[0]}`; } },
    ));
    main.appendChild(comparePanel);
  }
}

// Total setup seconds from a per-stage breakdown; zero when there is none.
function setupSeconds(stages) {
  if (!stages) return 0;
  return SETUP_ORDER.reduce((sum, key) => sum + (stages[key] || 0), 0);
}

function tile(label, value) {
  return element("div", { class: "tile" }, [
    element("strong", {}, String(value)),
    element("span", {}, label),
  ]);
}

// ---------------------------------------------------------------------------
// One bar: its sampled goals, drawn around the robot
// ---------------------------------------------------------------------------

async function viewBar(barId) {
  clearStageControls();
  setBreadcrumb([
    { label: runInfo.run_name, href: "#/" },
    { label: barId },
  ]);
  const bar = await loadJson(paths.bar(barId));
  const problem = bar.problem;

  main.replaceChildren();
  showViewer(`${barId}: ${bar.goals.length} goal poses, coloured by the share `
    + "of their 30 planning runs that found a motion. Click one to open it.");

  const view = await ensureRobot();
  view.clearPoseClouds();
  view.clearTrees();
  view.clearCarriedBar();
  view.pose(problem.home_conf12);
  view.setTrajectoryLine(null);
  view.addPoseCloud(bar.goals.map((goal) => ({
    world_from_bar: goal.world_from_bar,
    colour: successColour(goal.success_rate),
    id: goal.goal_index,
  })), {
    length: problem.bar_length_m,
    radius: problem.bar_radius_m,
    bar_from_midpoint: problem.bar_from_midpoint,
    opacity: 0.8,
  });
  view.onPick((goalIndex) => {
    window.location.hash = `#/bar/${barId}/goal/${goalIndex}`;
  });

  // What the colours mean, with how many goals fall in each band.
  const bandCounts = SUCCESS_BANDS.map(() => 0);
  for (const goal of bar.goals) bandCounts[successBand(goal.success_rate)] += 1;
  setStageControls(element("div", { class: "legend" },
    SUCCESS_BANDS.map((band, index) => element("span", { class: "legend-item" }, [
      swatch(band.colour), `${band.label} ${bandCounts[index]}`,
    ]))));

  const summaryPanel = panel(
    `${barId}, ${problem.bar_length_m.toFixed(2)} m`,
    `Held ${describeGrasp(problem)}.`);
  main.appendChild(summaryPanel);

  const goalsPanel = panel("Goals", "Click a row to open its start survey.");
  goalsPanel.appendChild(table(
    ["goal", "runs", "solved", "planning s", "setup min",
      "goal IK solutions", "workable starts", "straight-line starts"],
    bar.goals.map((goal) => [
      goal.goal_index, goal.n_runs,
      `${(goal.success_rate * 100).toFixed(0)}%`,
      goal.planning_median_s != null ? goal.planning_median_s.toFixed(0) : "-",
      goal.setup_s ? (goal.setup_s.total / 60).toFixed(1) : "-",
      goal.n_free_branch_pairs, goal.n_starts_found,
      (goal.difficulty_counts || {}).free || 0,
    ]),
    { onRowClick: (row) => {
      window.location.hash = `#/bar/${barId}/goal/${row[0]}`;
    } },
  ));
  goalsPanel.appendChild(columnKey([
    ["runs", "Planning runs for this goal: 3 start poses, 10 random seeds each."],
    ["solved", "Share of those runs where both the coarse and the fine pass "
      + "found a motion."],
    ["planning s", "Median motion-planning time of the solved runs (fine "
      + "pass). Only the time inside the planner; setup is not included."],
    ["setup min", "Time spent on this goal before any planning: goal IK, the "
      + "start survey, the straight-line probes and the planner's own start "
      + "search."],
    ["goal IK solutions", "How many different arm configurations hold the bar "
      + "at the goal without colliding (pairs of left and right IK branches, "
      + "at most 64). Planning aims for the one the planner itself would pick."],
    ["workable starts", "Carry poses the start survey found where both arms "
      + "can hold the bar without colliding. Meant to be the same for every "
      + "goal of a bar, since carry poses come from the grasp and the three "
      + "carry anchors, not from the goal. On 11 of the 16 bars they shift "
      + "between goals instead, a suspected numerical problem in the "
      + "planner's carry-pose code; see the analysis."],
    ["straight-line starts", "Workable starts that a straight motion from the "
      + "goal reaches with nothing in the way: the easiest problems. This "
      + "varies a great deal from goal to goal."],
  ]));
  main.appendChild(goalsPanel);
}

// A short glossary under a table, one line per column. Column names have to
// be short to fit; this is where they are spelled out.
function columnKey(entries) {
  return element("dl", { class: "key" }, entries.flatMap(([term, text]) => [
    element("dt", {}, term), element("dd", {}, text),
  ]));
}

function describeGrasp(problem) {
  const left = problem.grasp_bar_from_left[0];
  const right = problem.grasp_bar_from_right[0];
  return `at ${Math.abs(left[2]).toFixed(2)} m and `
    + `${Math.abs(right[2]).toFixed(2)} m along its length`;
}

// Bands for colouring a goal by how often it was solved.
//
// Set from the data rather than from round numbers: overall only about one run
// in six succeeds, so a scale whose top band starts at two thirds leaves
// nearly every goal the same colour and says nothing.
const SUCCESS_BANDS = [
  { label: "never solved", colour: 0xb23a48, below: 1e-9 },
  { label: "under 10%", colour: 0xd98324, below: 0.10 },
  { label: "10 to 25%", colour: 0xe0b33a, below: 0.25 },
  { label: "25 to 50%", colour: 0x8cbf5a, below: 0.50 },
  { label: "50% or more", colour: 0x2e9e5b, below: Infinity },
];

function successBand(rate) {
  const value = rate || 0;
  return SUCCESS_BANDS.findIndex((band) => value < band.below);
}

function successColour(rate) {
  if (rate == null) return 0xb8bec2;
  return SUCCESS_BANDS[successBand(rate)].colour;
}

// ---------------------------------------------------------------------------
// One goal: every carry pose the survey tried
// ---------------------------------------------------------------------------

async function viewGoal(barId, goalIndex) {
  clearStageControls();
  setBreadcrumb([
    { label: runInfo.run_name, href: "#/" },
    { label: barId, href: `#/bar/${barId}` },
    { label: `goal ${goalIndex}` },
  ]);
  const [bar, goal] = await Promise.all([
    loadJson(paths.bar(barId)),
    loadJson(paths.goal(barId, goalIndex)),
  ]);
  const problem = bar.problem;

  main.replaceChildren();
  showViewer("");

  const view = await ensureRobot();
  view.setTrajectoryLine(null);
  view.clearTrees();
  view.pose(goal.primary_conf);
  view.setCarriedBar({
    length: problem.bar_length_m, radius: problem.bar_radius_m,
    linkName: problem.attached_link, grasp: problem.grasp_bar_from_left,
    bar_from_midpoint: problem.bar_from_midpoint,
  });

  // Draws the carry poses for the chosen mode, with a legend to match.
  //
  // "Workable" shows only poses the arms can hold, coloured by how hard they
  // look to reach from the goal -- the question is which starts are good.
  // "All tried" adds every pose the survey rejected, coloured by why -- the
  // question is where the robot cannot hold the bar at all. The two modes
  // colour differently on purpose, so each has its own legend.
  const failed = goal.candidates.filter((c) => c.reason !== "feasible");
  const drawPoses = () => {
    view.clearPoseClouds();
    const shape = {
      length: problem.bar_length_m, radius: problem.bar_radius_m,
      bar_from_midpoint: problem.bar_from_midpoint,
    };
    let legendItems;
    if (goalPoseMode === "outcome") {
      // What the planner made of each start it was given: solved outright,
      // only the coarse pass, or neither. Unplanned starts stay faint so the
      // spread of the planned ones is still visible in context.
      const outcomeOf = (start) => (start.success_rate > 0 ? "solved"
        : start.coarse_success_rate > 0 ? "coarse_only" : "failed");
      const outcomeColours = { solved: OUTCOME_COLOURS.free,
        coarse_only: OUTCOME_COLOURS.blocked, failed: OUTCOME_COLOURS.broken };
      const planned = goal.starts.filter((start) => start.planned);
      view.addPoseCloud(goal.starts.filter((start) => !start.planned).map((start) => ({
        world_from_bar: start.world_from_bar, colour: 0xb8bec2,
        id: { kind: "start", start },
      })), { ...shape, opacity: 0.12 });
      view.addPoseCloud(planned.map((start) => ({
        world_from_bar: start.world_from_bar,
        colour: outcomeColours[outcomeOf(start)],
        id: { kind: "start", start },
      })), { ...shape, opacity: 0.6 });
      const byOutcome = { solved: 0, coarse_only: 0, failed: 0 };
      for (const start of planned) byOutcome[outcomeOf(start)] += 1;
      legendItems = [
        ["solved", outcomeColours.solved, byOutcome.solved],
        ["coarse pass only", outcomeColours.coarse_only, byOutcome.coarse_only],
        ["failed", outcomeColours.failed, byOutcome.failed],
        ["not planned", 0xb8bec2, goal.starts.length - planned.length],
      ];
      viewerCaption.textContent = `Goal ${goalIndex} in blue. The `
        + `${planned.length} starts the planner was given, coloured by what it `
        + "made of them. Click one to open its runs.";
    } else if (goalPoseMode === "all") {
      // Rejected poses are faint, so the workable ones still read on top.
      view.addPoseCloud(failed.map((c) => ({
        world_from_bar: c.world_from_bar,
        colour: OUTCOME_COLOURS[c.reason] || 0xb8bec2,
        id: { kind: "failed", candidate: c },
      })), { ...shape, opacity: 0.18 });
      view.addPoseCloud(goal.starts.map((start) => ({
        world_from_bar: start.world_from_bar,
        colour: OUTCOME_COLOURS.feasible,
        id: { kind: "start", start },
      })), { ...shape, opacity: 0.6 });
      const byReason = {};
      for (const c of failed) byReason[c.reason] = (byReason[c.reason] || 0) + 1;
      legendItems = [
        ["workable", OUTCOME_COLOURS.feasible, goal.starts.length],
        ...Object.entries(byReason)
          .sort((a, b) => b[1] - a[1])
          .map(([reason, count]) => [
            reason.replace("ik_unreachable_", "no reach, "),
            OUTCOME_COLOURS[reason] || 0xb8bec2, count]),
      ];
      viewerCaption.textContent = `Goal ${goalIndex} in blue. All `
        + `${goal.candidates.length} carry poses the survey tried: workable ones `
        + "in green, rejected ones faint and coloured by why.";
    } else {
      view.addPoseCloud(goal.starts.map((start) => ({
        world_from_bar: start.world_from_bar,
        colour: OUTCOME_COLOURS[start.difficulty] || 0xb8bec2,
        id: { kind: "start", start },
      })), { ...shape, opacity: 0.5 });
      const byDifficulty = {};
      for (const s of goal.starts) {
        byDifficulty[s.difficulty] = (byDifficulty[s.difficulty] || 0) + 1;
      }
      legendItems = [
        ["straight line clear", OUTCOME_COLOURS.free, byDifficulty.free || 0],
        ["straight line blocked", OUTCOME_COLOURS.blocked, byDifficulty.blocked || 0],
        ["straight line breaks", OUTCOME_COLOURS.broken, byDifficulty.broken || 0],
      ];
      viewerCaption.textContent = `Goal ${goalIndex} in blue. The `
        + `${goal.starts.length} carry poses the arms can hold, coloured by `
        + "whether a straight motion reaches them from the goal. Click a "
        + "planned one to open its runs.";
    }
    legendHost.replaceChildren(...legendItems.map(([label, colour, count]) =>
      element("span", { class: "legend-item" }, [
        swatch(colour), `${label} ${count}`,
      ])));
  };

  // A two-way switch rather than a checkbox: both states are views in their
  // own right, and the pressed button says which one is showing.
  const modeButtons = [["workable", "Workable only"], ["all", "Include failed"],
    ["outcome", "Planning outcome"]]
    .map(([mode, label]) => {
      const button = element("button", {
        type: "button", "aria-pressed": String(goalPoseMode === mode),
      }, label);
      button.addEventListener("click", () => {
        goalPoseMode = mode;
        for (const other of modeButtons) {
          other.setAttribute("aria-pressed", String(other === button));
        }
        drawPoses();
      });
      return button;
    });
  const legendHost = element("div", { class: "legend" });
  setStageControls(element("div", { class: "controls" }, [
    element("div", { class: "toggle", role: "group",
      "aria-label": "which carry poses to show" }, modeButtons),
    legendHost,
  ]));
  drawPoses();

  view.onPick((picked) => {
    if (!picked) return;
    if (picked.kind === "failed") {
      // A rejected pose has no arm configuration to show; say why instead.
      const c = picked.candidate;
      viewerCaption.textContent = `Rejected pose, ${c.anchor} anchor: `
        + `${c.reason.replace(/_/g, " ")}.`
        + (c.collision_pairs && c.collision_pairs.length
          && isReadablePair(c.collision_pairs[0])
          ? ` First collision: ${c.collision_pairs[0].join(" and ")}.` : "");
      return;
    }
    const start = picked.start;
    if (start.planned) {
      window.location.hash =
        `#/bar/${barId}/goal/${goalIndex}/start/${start.start_index}`;
    } else {
      view.pose(start.conf || goal.primary_conf);
    }
  });

  // Why candidates failed.
  const counts = goal.sweep_counts || {};
  const setup = goal.setup_s || {};
  const reasonPanel = panel(
    "The start survey (setup, not motion planning)",
    `${goal.candidates.length} carry poses tried, each with IK for both arms `
    + "and a collision check. A pose fails either because an arm cannot reach "
    + "it, or because holding it would collide."
    + (setup.start_survey
      ? ` The survey took ${(setup.start_survey / 60).toFixed(1)} min for this `
        + `goal; the whole setup took ${(setup.total / 60).toFixed(1)} min, of `
        + `which goal IK was ${setup.goal_ik.toFixed(0)} s.`
      : ""));
  const reasonChart = element("div");
  reasonPanel.appendChild(reasonChart);
  main.appendChild(withCsv(reasonPanel,
    [["reason", "count"], ...Object.entries(counts)]));
  barChart(reasonChart, {
    yLabel: "poses", format: (v) => v.toFixed(0),
    data: Object.entries(counts).map(([reason, count]) => ({
      label: reason.replace("ik_unreachable_", "no reach "),
      value: count,
      colour: cssColour(OUTCOME_COLOURS[reason] || 0xb8bec2),
    })),
  });

  // Which anchors produced workable poses.
  const byAnchor = {};
  for (const candidate of goal.candidates) {
    const row = byAnchor[candidate.anchor] || (byAnchor[candidate.anchor] = {});
    row[candidate.reason] = (row[candidate.reason] || 0) + 1;
  }
  const anchorPanel = panel(
    "By carry anchor",
    "The three ways the robot can hold a bar while moving it.");
  const anchorChart = element("div");
  anchorPanel.appendChild(anchorChart);
  main.appendChild(anchorPanel);
  stackedBarChart(anchorChart, {
    keys: Object.keys(counts),
    colours: Object.fromEntries(Object.keys(counts).map(
      (k) => [k, cssColour(OUTCOME_COLOURS[k] || 0xb8bec2)])),
    yLabel: "share of poses",
    data: Object.entries(byAnchor).map(([anchor, values]) => ({
      label: anchor, values,
    })),
  });

  // What was in the way, when something was.
  const pairCounts = {};
  let unreadablePairs = 0;
  for (const candidate of goal.candidates) {
    for (const pair of candidate.collision_pairs || []) {
      if (!isReadablePair(pair)) { unreadablePairs += 1; continue; }
      const key = pair.join(" ↔ ");
      pairCounts[key] = (pairCounts[key] || 0) + 1;
    }
  }
  const topPairs = Object.entries(pairCounts)
    .sort((a, b) => b[1] - a[1]).slice(0, 10);
  if (topPairs.length) {
    const pairPanel = panel("What collided most", "Across every rejected pose.");
    pairPanel.appendChild(table(["pair", "times"], topPairs));
    main.appendChild(withCsv(pairPanel, [["pair", "times"], ...topPairs]));
  } else if (unreadablePairs) {
    // Say so plainly rather than show a table of memory addresses.
    main.appendChild(panel(
      "What collided most",
      "Not available for this library. The colliding pairs were recorded as "
      + "object addresses instead of link names, a bug fixed since. Rebuilding "
      + "the library, or re-checking just its collision rejections, restores "
      + "them."));
  }

  // A slice through the offset grid.
  const anchors = Array.from(new Set(goal.candidates.map((c) => c.anchor)));
  const zValues = Array.from(new Set(goal.candidates.map((c) => round(c.delta[2]))))
    .sort((a, b) => a - b);
  const slicePanel = panel(
    "The offset grid",
    "The planner sweeps offsets around each carry anchor. One height slice "
    + "at a time; hover a cell for its reason.");
  const sliceControls = element("div", { class: "controls" });
  const sliceChart = element("div", { class: "heatmaps" });
  slicePanel.append(sliceControls, sliceChart);
  main.appendChild(slicePanel);

  let sliceZ = zValues[Math.floor(zValues.length / 2)];
  const drawSlice = () => {
    sliceChart.replaceChildren();
    for (const anchor of anchors) {
      const cells = goal.candidates
        .filter((c) => c.anchor === anchor && round(c.delta[2]) === sliceZ)
        .map((c) => ({ x: round(c.delta[0]), y: round(c.delta[1]), candidate: c }));
      const xValues = Array.from(new Set(cells.map((c) => c.x))).sort((a, b) => a - b);
      const yValues = Array.from(new Set(cells.map((c) => c.y))).sort((a, b) => a - b);
      const holder = element("div", { class: "heatmap" },
        [element("h3", {}, anchor)]);
      const target = element("div");
      holder.appendChild(target);
      sliceChart.appendChild(holder);
      heatmap(target, {
        cells, xValues, yValues,
        xLabel: `offset x (m), height ${sliceZ.toFixed(2)} m`,
        yLabel: "offset y (m)",
        colourOf: (cell) => cssColour(
          OUTCOME_COLOURS[cell.candidate.reason] || 0xb8bec2),
        titleOf: (cell) =>
          `${cell.candidate.reason} (${(cell.candidate.wall_s * 1000).toFixed(0)} ms)`,
      });
    }
  };
  const slider = element("input", {
    type: "range", min: 0, max: String(zValues.length - 1),
    value: String(zValues.indexOf(sliceZ)),
  });
  slider.addEventListener("input", () => {
    sliceZ = zValues[Number(slider.value)];
    drawSlice();
  });
  sliceControls.append(element("label", {}, "height"), slider);
  drawSlice();

  // The starts that were actually planned from.
  const planned = goal.starts.filter((start) => start.planned);
  const startPanel = panel(
    "Starts planned from",
    `${planned.length} of ${goal.starts.length} workable poses were used. `
    + "Click a row to see its runs.");
  startPanel.appendChild(table(
    ["start", "anchor", "straight line", "solved", "planning s"],
    planned.map((start) => [
      start.start_index, start.anchor, DIFFICULTY_LABELS[start.difficulty] || "-",
      `${((start.success_rate || 0) * 100).toFixed(0)}%`,
      start.planning_median_s != null ? start.planning_median_s.toFixed(0) : "-",
    ]),
    { onRowClick: (row) => {
      window.location.hash = `#/bar/${barId}/goal/${goalIndex}/start/${row[0]}`;
    } },
  ));
  startPanel.appendChild(columnKey([
    ["anchor", "Which of the three ways of carrying the bar this start uses."],
    ["straight line", "What a straight motion from the goal to this start "
      + "found: clear, blocked by something, or unable to reach it at all."],
    ["solved", "Share of this start's 10 seeded runs that found a motion."],
    ["planning s", "Median motion-planning time of the solved runs (fine "
      + "pass), inside the planner only."],
  ]));
  main.appendChild(startPanel);

  // ! Where the planner succeeds and where it does not, across the starts it
  // ! was given. With a whole goal's worth of starts planned, this is the
  // ! planner's ability to connect two configurations, read by how the
  // ! start looked beforehand: whether a straight motion reached it, and
  // ! which way the bar was carried.
  if (planned.length >= 2) {
    const outcomePanel = panel(
      "Solved, by straight-line label and by carry anchor",
      "Share of planned starts with at least one solved seed, grouped by what "
      + "the straight-line test said about the start (left) and by the carry "
      + "anchor it came from (right). The count is the starts in each group.");
    const groupRate = (keyOf, labelOf, colourOf) => {
      const groups = new Map();
      for (const start of planned) {
        const key = keyOf(start);
        const group = groups.get(key) || { n: 0, solved: 0 };
        group.n += 1;
        if (start.success_rate > 0) group.solved += 1;
        groups.set(key, group);
      }
      return [...groups.entries()].map(([key, group]) => ({
        label: `${labelOf(key)} (${group.n})`, value: group.solved / group.n,
        colour: colourOf(key),
      }));
    };
    const charts = element("div", { class: "heatmaps" });
    const byLine = element("div", { class: "heatmap" });
    const byAnchor = element("div", { class: "heatmap" });
    charts.append(byLine, byAnchor);
    outcomePanel.appendChild(charts);
    main.appendChild(outcomePanel);
    barChart(byLine, {
      yLabel: "share solved", format: (v) => `${(v * 100).toFixed(0)}%`,
      data: groupRate((s) => s.difficulty || "?",
        (k) => DIFFICULTY_LABELS[k] || k, (k) => OUTCOME_COLOURS[k] || 0xb8bec2),
    });
    barChart(byAnchor, {
      yLabel: "share solved", format: (v) => `${(v * 100).toFixed(0)}%`,
      data: groupRate((s) => s.anchor || "?", (k) => k, () => COLOURS.fine),
    });
  }

  const derivation = goal.production_derivation || {};
  main.appendChild(panel(
    "What the planner itself would have done",
    // What the start search found, followed by how long it took.
    (derivation.found
      ? (derivation.direct_path_free
        ? "Found a carry pose joined to the goal by a straight motion, so it "
          + "would have skipped searching entirely."
        : "Found a carry pose, but not one a straight motion reaches, so it "
          + "would have had to search.")
      : "Found no carry pose at all within its time budget.")
    + (derivation.wall_s
      ? ` Its start search took ${derivation.wall_s.toFixed(0)} s; that is `
        + "setup, and is not counted in any planning time here."
      : "")));
}

// Number format for the stop-reason charts: the axis shows shares (0, 0.5, 1)
// and the hover shows raw counts, so whole numbers print as-is and fractions
// with two decimals.
function countOrShare(value) {
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

// Where a transfer begins and ends, drawn faintly on the start page.
const GHOST_COLOURS = { start: 0x2e9e5b, goal: 0x1f84bd };

// Plain names for what the straight-line probe found.
const DIFFICULTY_LABELS = {
  free: "clear",
  blocked: "blocked",
  broken: "cannot reach",
};

function round(value) {
  return Math.round(value * 1000) / 1000;
}

function cssColour(hex) {
  return `#${hex.toString(16).padStart(6, "0")}`;
}

// A joint name short enough for a table: "left shoulder pan".
function shortJointName(jointOrder, index) {
  const name = (jointOrder || [])[index];
  if (!name) return `joint ${index}`;
  return name.replace("_ur_arm_", " ").replace("_joint", "").replace(/_/g, " ");
}

// Rotates a vector by an [x, y, z, w] quaternion.
function rotateByQuaternion(q, v) {
  const [x, y, z, w] = q;
  // t = 2 * cross(q.xyz, v); result = v + w * t + cross(q.xyz, t)
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [
    v[0] + w * tx + (y * tz - z * ty),
    v[1] + w * ty + (z * tx - x * tz),
    v[2] + w * tz + (x * ty - y * tx),
  ];
}

// How many edges from the root to a node, given [parent, child] edges.
function nodeDepth(edges, index) {
  const parentOf = new Map(edges.map(([parent, child]) => [child, parent]));
  let depth = 0;
  let node = index;
  while (parentOf.has(node) && depth < 100000) {
    node = parentOf.get(node);
    depth += 1;
  }
  return depth;
}

// A small table for the detail box; rows flagged in `over` are drawn in red.
function jointTable(rows, headers, over = []) {
  const grid = table(headers, rows);
  grid.querySelectorAll("tbody tr").forEach((tr, i) => {
    if (over[i]) tr.classList.add("over");
  });
  return grid;
}

// ---------------------------------------------------------------------------
// One start: the seeds, and the motions they found
// ---------------------------------------------------------------------------

async function viewStart(barId, goalIndex, startIndex) {
  setBreadcrumb([
    { label: runInfo.run_name, href: "#/" },
    { label: barId, href: `#/bar/${barId}` },
    { label: `goal ${goalIndex}`, href: `#/bar/${barId}/goal/${goalIndex}` },
    { label: `start ${startIndex}` },
  ]);
  const [bar, goal, start] = await Promise.all([
    loadJson(paths.bar(barId)),
    loadJson(paths.goal(barId, goalIndex)),
    loadJson(paths.start(barId, goalIndex, startIndex)),
  ]);
  const problem = bar.problem;
  const startRecord = goal.starts.find((s) => s.start_index === startIndex);

  main.replaceChildren();
  showViewer(`Start ${startIndex} of goal ${goalIndex}.`);

  const view = await ensureRobot();
  view.clearPoseClouds();
  view.setTrajectoryLine(null);
  view.clearTrees();
  // Clicks here must not reach the goal page's handler, which would still be
  // registered from the previous view.
  view.onPick(null);
  view.setCarriedBar({
    length: problem.bar_length_m, radius: problem.bar_radius_m,
    linkName: problem.attached_link, grasp: problem.grasp_bar_from_left,
    bar_from_midpoint: problem.bar_from_midpoint,
  });
  if (startRecord && startRecord.conf) view.pose(startRecord.conf);

  // Faint copies of the bar where the transfer begins and ends, as fixed
  // references while the carried bar moves or the search trees are drawn.
  const barShape = {
    length: problem.bar_length_m, radius: problem.bar_radius_m,
    bar_from_midpoint: problem.bar_from_midpoint, opacity: 0.28,
  };
  if (startRecord) {
    view.addPoseCloud([{ world_from_bar: startRecord.world_from_bar,
      colour: GHOST_COLOURS.start, id: null }], barShape);
  }
  view.addPoseCloud([{ world_from_bar: goal.world_from_bar,
    colour: GHOST_COLOURS.goal, id: null }], barShape);
  const ghostLegend = element("div", { class: "legend" }, [
    element("span", { class: "legend-item" },
      [swatch(GHOST_COLOURS.start), "start: bar and its tree"]),
    element("span", { class: "legend-item" },
      [swatch(GHOST_COLOURS.goal), "goal: bar and its tree"]),
  ]);
  setStageControls(ghostLegend);

  // The seeds, side by side.
  const seedPanel = panel(
    "Seeds",
    "The same problem, planned with different random seeds. Each run is "
    + "planned coarse first, then fine; both must succeed.");
  const seedRows = start.seeds.map((seed) => {
    const coarse = seed.passes.coarse || {};
    const fine = seed.passes.fine || {};
    return [
      seed.seed,
      coarse.success ? `yes ${coarse.wall_s.toFixed(1)}s` : "no",
      fine.success ? `yes ${fine.wall_s.toFixed(1)}s` : (seed.failed_pass === "coarse" ? "-" : "no"),
      fine.iterations || coarse.iterations || 0,
      `${coarse.tree_a_nodes || 0}/${coarse.tree_b_nodes || 0}`,
      fine.n_waypoints || 0,
    ];
  });
  seedPanel.appendChild(table(
    ["seed", "coarse", "fine", "iterations", "tree a/b", "waypoints"], seedRows));
  main.appendChild(withCsv(seedPanel,
    [["seed", "coarse", "fine", "iterations", "trees", "waypoints"], ...seedRows]));

  // Time per seed, coarse against fine.
  const timePanel = panel(
    "Motion planning time per seed",
    "Time inside the planner for each seed: c is the coarse pass, f the fine "
    + "pass (run only when the coarse one succeeds). None of the setup time "
    + "spent finding this goal and start is included.");
  const timeChart = element("div");
  timePanel.appendChild(timeChart);
  main.appendChild(timePanel);
  barChart(timeChart, {
    yLabel: "seconds", format: (v) => v.toFixed(0),
    data: start.seeds.flatMap((seed) => {
      const rows = [];
      if (seed.passes.coarse) {
        rows.push({
          label: `${seed.seed}c`, value: seed.passes.coarse.wall_s,
          colour: COLOURS.coarse,
        });
      }
      if (seed.passes.fine) {
        rows.push({
          label: `${seed.seed}f`, value: seed.passes.fine.wall_s,
          colour: COLOURS.fine,
        });
      }
      return rows;
    }),
  });

  // Where each seed's time went.
  const breakdownPanel = panel(
    "Where each seed's motion planning time went",
    "Share of time inside the planner, by what it was doing. 'IK while "
    + "growing the tree' is the arm IK at every search step, not goal IK.");
  const breakdownChart = element("div");
  breakdownPanel.appendChild(breakdownChart);
  main.appendChild(breakdownPanel);
  stackedBarChart(breakdownChart, {
    keys: BREAKDOWN_ORDER, labels: BREAKDOWN_LABELS, yLabel: "share of time",
    data: start.seeds.flatMap((seed) =>
      Object.entries(seed.passes).map(([name, record]) => ({
        label: `${seed.seed}${name[0]}`, values: record.breakdown || {},
      }))),
  });

  // ! Why each seed's search kept being cut short.
  // This is what separates a slow search from a blocked one. Every attempt to
  // grow a tree ends for a reason; a seed whose attempts mostly die on the
  // continuity gate is rejecting its own arm solutions, not hitting the scene.
  // The table underneath adds how much tree each attempt actually bought.
  const passRows = start.seeds.flatMap((seed) =>
    Object.entries(seed.passes).map(([name, record]) => ({
      seed: seed.seed, name, record,
      grouped: groupStopReasons(record.extend_stop_reasons),
    })));
  const startReasons = groupStopReasons(
    ...passRows.map((row) => row.record.extend_stop_reasons));
  if (Object.values(startReasons).some(Boolean)) {
    const reasonPanel = panel(
      "Why each seed's extensions stopped",
      "Share of tree-growing attempts, by how each one ended. c is the coarse "
      + "pass, f the fine pass. The sentence is over every pass of every seed "
      + "on this start.");
    reasonPanel.appendChild(element("p", { class: "verdict" },
      stopReasonVerdict(startReasons)));
    const reasonChart = element("div");
    reasonPanel.appendChild(reasonChart);
    stackedBarChart(reasonChart, {
      keys: STOP_REASON_ORDER, labels: STOP_REASON_LABELS,
      colours: STOP_REASON_COLOURS, yLabel: "share of stops",
      data: passRows.map((row) => ({
        label: `${row.seed}${row.name[0]}`, values: row.grouped,
      })),
      format: countOrShare,
    });

    // How much each attempt bought: attempts per second says how fast the
    // search ran, nodes per 100 attempts says how much of that was wasted.
    const percent = (count, total) => total ? `${(count / total * 100).toFixed(0)}%` : "-";
    const yieldRows = passRows.map((row) => {
      const attempts = STOP_REASON_ORDER.reduce((sum, key) => sum + row.grouped[key], 0);
      const nodes = (row.record.tree_a_nodes || 0) + (row.record.tree_b_nodes || 0);
      const wall = row.record.wall_s;
      return [
        row.seed, row.name, attempts,
        wall ? (attempts / wall).toFixed(0) : "-",
        nodes,
        attempts ? (nodes / attempts * 100).toFixed(1) : "-",
        percent(row.grouped.continuity, attempts),
        percent(row.grouped.ik_failure, attempts),
        percent(row.grouped.collision, attempts),
        percent(row.grouped.reached, attempts),
      ];
    });
    const yieldHeaders = ["seed", "pass", "attempts", "attempts / s", "tree nodes",
      "nodes per 100 attempts", "continuity", "IK failure", "collision", "reached"];
    reasonPanel.appendChild(table(yieldHeaders, yieldRows));
    reasonPanel.appendChild(columnKey([
      ["attempts", "Walks toward a target pose: growing a tree, trying to join "
        + "the other tree, or re-solving the seam where they met."],
      ["attempts / s", "How fast the search ran. Low means each attempt was "
        + "expensive: many IK solves and collision checks per waypoint."],
      ["tree nodes", "Both trees together at the end of the pass."],
      ["nodes per 100 attempts", "How much tree the search kept. Near zero "
        + "means almost every attempt was thrown away before adding anything."],
      ["reached", "Attempts that walked all the way to their target."],
    ]));
    main.appendChild(withCsv(reasonPanel, [yieldHeaders, ...yieldRows]));
  }

  // Flame graphs, for reading one run in detail.
  const profiles = start.seeds.flatMap((seed) =>
    Object.entries(seed.passes)
      .filter(([, record]) => record.speedscope)
      .map(([name, record]) => element("a", {
        class: "flame",
        href: `../profiles/${record.speedscope}`,
        download: record.speedscope,
      }, `seed ${seed.seed} ${name}`)));
  if (profiles.length) {
    const profilePanel = panel(
      "Flame graphs",
      "Download and open at speedscope.app to see exactly where a run spent "
      + "its time.");
    profilePanel.appendChild(element("div", { class: "flames" }, profiles));
    main.appendChild(profilePanel);
  }

  // Everything under the robot: the tree pager, the motion scrubber and the
  // legend, each present only when there is something for it to show.
  const stageParts = { trees: null, playback: null };
  const refreshStage = () => setStageControls(
    ...[stageParts.trees, stageParts.playback, ghostLegend].filter(Boolean));

  // ! The search itself: both trees node by node, and every extension that
  // ! failed, with its cause. This is what explains a failure: where each
  // ! tree got to, what it kept trying, and what threw each attempt away.
  const withTrees = start.seeds.filter((seed) =>
    seed.trees && seed.trees.passes && Object.keys(seed.trees.passes).length);
  const treePanel = panel(
    "Search trees",
    "The two trees the search grew: one from the start bar (green), one from "
    + "the goal bar (blue), each node a small bar you can click to pose the "
    + "robot there. Failed attempts are drawn at the waypoint that was "
    + "rejected, coloured by why; click one to see the numbers behind it. "
    + "Page through the seeds under the robot.");
  main.appendChild(treePanel);

  // The failed attempts of the seed and pass on show, summarised: which joint
  // keeps flipping, which arm keeps losing its IK solution and why, and what
  // keeps colliding. Filled in by the pager below.
  const attemptPanel = panel(
    "What the failed attempts say",
    "For the seed and pass shown under the robot. Continuity stops are "
    + "counted by the joint that jumped the most; IK failures by arm and by "
    + "whether a solution exists at all; collisions by the pair of parts.");
  const attemptBody = element("div");
  attemptPanel.appendChild(attemptBody);
  main.appendChild(attemptPanel);

  if (!withTrees.length) {
    const command = `python scripts/benchmark_cdfm.py replay --name ${runInfo.run_name} `
      + `--bar ${barId} --goal ${goalIndex} --start ${startIndex}`;
    treePanel.append(
      element("p", {}, "Not saved for this start. The batch kept only the size "
        + "of each tree. To see them, re-plan this start with its trees saved "
        + "(same seeds and settings, a few minutes), then reload:"),
      element("pre", { class: "command" }, command));
    attemptBody.appendChild(element("p", { class: "muted" },
      "Nothing recorded for this start."));
  } else {
    const treeInfo = element("p", { class: "tree-info" }, "");
    treePanel.appendChild(treeInfo);
    let position = 0;
    let passName = "coarse";

    // --- Row 1 under the robot: which seed and pass.
    const previous = element("button", { type: "button" }, "◀ previous seed");
    const next = element("button", { type: "button" }, "next seed ▶");
    const seedLabel = element("span", { class: "muted" }, "");
    const passButtons = ["coarse", "fine"].map((name) => {
      const button = element("button", { type: "button" }, `${name} pass`);
      button.addEventListener("click", () => { passName = name; drawTree(); });
      return button;
    });

    // --- Row 2: what to draw. The toggles carry their colour, so they are the
    // legend as well. Counts are filled in once a pass is loaded.
    const shown = { nodes: true, continuity: true, ik_failure: true, collision: true };
    // The chart colours are CSS strings; the 3D view wants numbers.
    const reasonHex = Object.fromEntries(Object.entries(STOP_REASON_COLOURS)
      .map(([key, css]) => [key, parseInt(css.slice(1), 16)]));
    const toggleColours = {
      nodes: GHOST_COLOURS.start,
      continuity: reasonHex.continuity,
      ik_failure: reasonHex.ik_failure,
      collision: reasonHex.collision,
    };
    const toggleLabels = { nodes: "tree nodes", continuity: "continuity",
      ik_failure: "IK failure", collision: "collision" };
    const toggles = {};
    for (const key of Object.keys(shown)) {
      const button = element("button", { type: "button", "aria-pressed": "true" },
        [swatch(toggleColours[key]), toggleLabels[key]]);
      button.addEventListener("click", () => {
        shown[key] = !shown[key];
        button.setAttribute("aria-pressed", String(shown[key]));
        drawSearch();
      });
      toggles[key] = button;
    }
    // Scrub the search in time: only attempts up to this iteration are drawn.
    const iterationSlider = element("input", { type: "range", min: "0", max: "0", value: "0" });
    const iterationLabel = element("span", { class: "muted" }, "");
    iterationSlider.addEventListener("input", () => { drawSearch(); stepIndex = -1; updateStepLabel(); });

    // --- Row 3: step through the failed attempts one at a time. Clicking a
    // bar in the 3D view is fiddly with thousands of them; this walks them in
    // search order for one reason at a time, and the 3D view follows.
    const reasonChooser = element("select", {}, [
      element("option", { value: "continuity" }, "continuity stops"),
      element("option", { value: "ik_failure" }, "IK failure stops"),
      element("option", { value: "collision" }, "collision stops"),
    ]);
    const stepBack = element("button", { type: "button" }, "\u25c0");
    const stepForward = element("button", { type: "button" }, "\u25b6");
    const stepLabel = element("span", { class: "muted" }, "");
    let stepIndex = -1;
    // The stops of the chosen reason, in the order the search made them,
    // limited to what the iteration slider lets through.
    const stepList = () => attempts
      .filter((a) => a.reason === reasonChooser.value && a.iteration <= Number(iterationSlider.value))
      .sort((x, y) => x.iteration - y.iteration);
    const updateStepLabel = () => {
      const list = stepList();
      stepBack.disabled = stepIndex <= 0;
      stepForward.disabled = stepIndex >= list.length - 1;
      stepLabel.textContent = list.length
        ? (stepIndex >= 0 ? `${stepIndex + 1} of ${list.length}` : `${list.length}, pick one \u25b6`)
        : "none";
    };
    const stepTo = (index) => {
      const list = stepList();
      if (!list.length) return;
      stepIndex = Math.max(0, Math.min(list.length - 1, index));
      updateStepLabel();
      showAttempt(list[stepIndex]);
    };
    stepBack.addEventListener("click", () => stepTo(stepIndex - 1));
    stepForward.addEventListener("click", () => stepTo(stepIndex + 1));
    reasonChooser.addEventListener("change", () => { stepIndex = -1; updateStepLabel(); });

    // --- Row 4: the outcome line, then what was clicked.
    const stageInfo = element("div", { class: "muted tree-stage-info" }, "");
    const detail = element("div", { class: "search-detail" });
    detail.hidden = true;
    stageParts.trees = element("div", {}, [
      element("div", { class: "controls" }, [
        element("div", { class: "toggle" }, [previous, next]),
        seedLabel,
        element("div", { class: "toggle", role: "group",
          "aria-label": "which pass" }, passButtons),
      ]),
      element("div", { class: "controls" }, [
        element("div", { class: "toggle", role: "group",
          "aria-label": "what to draw" }, Object.values(toggles)),
        element("label", {}, "iteration"), iterationSlider, iterationLabel,
      ]),
      element("div", { class: "controls" }, [
        element("label", {}, "step through"), reasonChooser,
        element("div", { class: "toggle" }, [stepBack, stepForward]),
        stepLabel,
      ]),
      stageInfo,
      detail,
    ]);

    const outcome = (success, wall) =>
      `${success ? "solved" : "failed"}${wall != null ? ` in ${wall.toFixed(0)} s` : ""}`;
    const barShape = {
      length: problem.bar_length_m, radius: problem.bar_radius_m,
      bar_from_midpoint: problem.bar_from_midpoint,
    };
    const jointName = (index) => shortJointName(runInfo.joint_order, index);
    const degrees = (rad) => `${(rad * 180 / Math.PI).toFixed(1)}°`;

    // What is loaded for the pass on show.
    let tree = null;        // { forward, backward } with points, quaternions, confs, edges
    let attempts = [];      // the failed extensions, in search order
    let attemptsNote = "";  // why there are none, when there are none
    let drawn = { nodeClouds: [], attemptClouds: [], lines: [] };
    // attempt -> { cloud, index }: which instanced bar stands for it.
    let markerOf = new Map();
    let lit = null;  // the marker currently drawn in the "selected" colour
    const treeOf = (attempt) => (attempt.tree === "a" ? tree.forward : tree.backward);

    // The bar's midpoint at a stored pose, for drawing lines between bars.
    const midpointAt = (pose) => {
      const [p, q] = pose;
      const m = rotateByQuaternion(q, problem.bar_from_midpoint);
      return [p[0] + m[0], p[1] + m[1], p[2] + m[2]];
    };

    // Draws the nodes and attempts for the current toggles and iteration cap.
    const drawSearch = () => {
      for (const cloud of [...drawn.nodeClouds, ...drawn.attemptClouds]) view.removePoseCloud(cloud);
      for (const line of drawn.lines) view.removeOverlay(line);
      drawn = { nodeClouds: [], attemptClouds: [], lines: [] };
      markerOf = new Map();
      lit = null;
      if (!tree) return;

      if (shown.nodes) {
        for (const [name, data, colour] of [["forward", tree.forward, GHOST_COLOURS.start],
                                            ["backward", tree.backward, GHOST_COLOURS.goal]]) {
          const cloud = view.addPoseCloud((data.points || []).map((point, index) => ({
            world_from_bar: [point, data.quaternions[index]], colour,
            id: { kind: "node", tree: name, index },
          })), { ...barShape, opacity: 0.35 });
          if (cloud) drawn.nodeClouds.push(cloud);
        }
      }

      const cap = Number(iterationSlider.value);
      iterationLabel.textContent = `${cap} of ${iterationSlider.max}`;
      const visible = attempts.filter((a) =>
        a.iteration <= cap && shown[a.reason] && a.rejected_pose);
      for (const reason of ["continuity", "ik_failure", "collision"]) {
        const ofReason = visible.filter((a) => a.reason === reason);
        const colour = reasonHex[reason];
        const cloud = view.addPoseCloud(ofReason.map((a) => ({
          world_from_bar: a.rejected_pose, colour, id: { kind: "attempt", attempt: a },
        })), { ...barShape, opacity: 0.5 });
        if (cloud) {
          drawn.attemptClouds.push(cloud);
          ofReason.forEach((a, index) => markerOf.set(a, { cloud, index }));
        }
        // A line from the last accepted node to the rejected waypoint: the
        // step that failed.
        const segments = ofReason.map((a) => {
          const nodes = treeOf(a);
          const last = nodes.points[a.last_node];
          if (!last) return null;
          return [midpointAt([last, nodes.quaternions[a.last_node]]), midpointAt(a.rejected_pose)];
        }).filter(Boolean);
        const lines = view.addLines(segments, { colour, opacity: 0.45 });
        if (lines) drawn.lines.push(lines);
      }
      for (const [key, button] of Object.entries(toggles)) {
        const count = key === "nodes"
          ? (tree.forward.points.length + tree.backward.points.length)
          : attempts.filter((a) => a.reason === key).length;
        button.lastChild.textContent = `${toggleLabels[key]} ${count}`;
      }
    };

    // Puts the robot at a configuration and says where it is. The ghost robot
    // (a second configuration, see-through) is hidden unless one is given.
    const poseRobot = (conf, what, ghost = null) => {
      view.clearHighlight();
      if (ghost && ghost.conf) view.showGhost(ghost.conf, ghost.colour);
      else view.hideGhost();
      if (!conf) return;
      view.pose(conf);
      viewerCaption.textContent = `Robot at ${what}`
        + (ghost && ghost.conf ? `; the see-through robot is at ${ghost.what}.` : ".");
    };

    // Marks the chosen attempt's bar in the selection colour, undoing the last.
    const lightMarker = (attempt) => {
      if (lit) view.setInstanceColour(lit.cloud, lit.index, lit.colour);
      lit = null;
      const marker = markerOf.get(attempt);
      if (!marker) return;
      lit = { ...marker, colour: reasonHex[attempt.reason] };
      view.setInstanceColour(marker.cloud, marker.index, OUTCOME_COLOURS.selected);
    };

    const showNode = ({ tree: name, index }) => {
      const data = name === "forward" ? tree.forward : tree.backward;
      const conf = data.confs[index];
      const depth = nodeDepth(data.edges, index);
      if (lit) { view.setInstanceColour(lit.cloud, lit.index, lit.colour); lit = null; }
      poseRobot(conf, `${name === "forward" ? "start" : "goal"} tree node ${index}`);
      detail.hidden = false;
      detail.replaceChildren(
        element("p", {}, [element("strong", {},
          `${name === "forward" ? "Start" : "Goal"} tree, node ${index}`),
          ` — ${depth} step${depth === 1 ? "" : "s"} from the root.`]),
        conf ? jointTable(conf.map((v, i) => [jointName(i), degrees(v)]), ["joint", "angle"])
          : element("p", { class: "muted" }, "No joint configuration stored for this node."),
      );
    };

    const showAttempt = (a) => {
      const nodes = treeOf(a);
      const lastConf = nodes.confs[a.last_node];
      const treeLabel = a.tree === "a" ? "start tree" : "goal tree";
      const phase = a.phase === "connect"
        ? "trying to join the other tree" : "growing toward a random sample";
      const header = element("p", {}, [
        element("strong", {}, `${STOP_REASON_LABELS[a.reason] || a.reason}`),
        ` — iteration ${a.iteration}, ${treeLabel} ${phase}, from node ${a.from_node}; `
        + `${a.n_added} waypoint${a.n_added === 1 ? "" : "s"} accepted before the stop.`,
      ]);
      const buttons = element("div", { class: "buttons" });
      const poseButton = (label, conf, what) => {
        const button = element("button", { type: "button" }, label);
        button.disabled = !conf;
        button.addEventListener("click", () => poseRobot(conf, what));
        buttons.appendChild(button);
        return button;
      };
      lightMarker(a);
      const parentWhat = `the parent node's IK solution (node ${a.last_node}, last accepted)`;
      const rejectedWhat = "the rejected node's IK solution";
      const parts = [header, buttons];
      const d = a.detail || {};
      if (a.reason === "continuity") {
        // Both configurations at once: the parent node's solution on the
        // solid robot, the rejected node's on the see-through one, and the
        // two side by side in the table with the joints over the gate in red.
        const ghostColour = reasonHex.continuity;
        const parentGhost = { conf: lastConf, colour: ghostColour, what: parentWhat };
        const rejectedGhost = { conf: a.rejected_conf, colour: ghostColour, what: rejectedWhat };
        const solidParent = element("button", { type: "button" }, "solid: parent, ghost: rejected");
        const solidRejected = element("button", { type: "button" }, "solid: rejected, ghost: parent");
        solidParent.addEventListener("click", () => poseRobot(lastConf, parentWhat, rejectedGhost));
        solidRejected.addEventListener("click", () => poseRobot(a.rejected_conf, rejectedWhat, parentGhost));
        solidRejected.disabled = !a.rejected_conf;
        buttons.append(solidParent, solidRejected);
        const rows = (d.delta_rad || []).map((delta, i) => ({
          cells: [jointName(i),
            lastConf ? degrees(lastConf[i]) : "-",
            a.rejected_conf ? degrees(a.rejected_conf[i]) : "-",
            degrees(delta)],
          over: delta > d.threshold_rad,
        }));
        parts.push(element("p", {}, `The gate allows ${degrees(d.threshold_rad || 0)} per joint `
          + `between waypoints; ${jointName(d.joint)} moved ${degrees((d.delta_rad || [])[d.joint] || 0)}. `
          + `Solid robot: parent node ${a.last_node}. See-through robot: the rejected node.`));
        parts.push(jointTable(rows.map((r) => r.cells), ["joint", "parent", "rejected", "jump"],
          rows.map((r) => r.over)));
        poseRobot(lastConf, parentWhat, rejectedGhost);
      } else if (a.reason === "collision") {
        poseButton("robot at last accepted", lastConf, parentWhat);
        poseButton("robot at rejected", a.rejected_conf, rejectedWhat);
        const pairs = d.pairs;
        parts.push(element("p", {}, (d.check === "bar_only"
          ? "The bar alone, at that pose, touches something. "
          : "The whole robot at that configuration touches something. ")
          + (pairs === null || pairs === undefined
            ? "Which parts is not recorded for this run."
            : pairs.length ? "" : "The re-check found no pair to name.")));
        if (pairs && pairs.length) {
          parts.push(jointTable(pairs.map((pair) => [pair[0], pair[1]]), ["part", "touches"]));
        }
        poseRobot(a.rejected_conf || lastConf, "the rejected waypoint");
        // Light up the parts. Anything that is not a robot link is the bar or
        // a tool, so the carried bar is tinted for those.
        const names = [...new Set((pairs || []).flat())];
        const notLinks = view.highlightParts(names);
        if (notLinks.length) view.highlightCarriedBar();
      } else if (a.reason === "ik_failure") {
        poseButton("robot at last accepted", lastConf, parentWhat);
        const causeText = { unreachable: "no arm solution exists at all",
          joint_limit: "solutions exist, but every one breaks a joint limit",
          fk_gate: "both arms solved but the FK check disagreed with the solver" };
        parts.push(element("p", {}, `${d.arm === "both" ? "Both arms" : `The ${d.arm} arm`}: `
          + `${causeText[d.cause] || d.cause || "no detail recorded"}.`));
        for (const [arm, info] of Object.entries(d.arms || {})) {
          if (info.cause !== "joint_limit") continue;
          const violations = info.violations || [];
          parts.push(element("p", {}, `${arm} arm: nearest solution past a limit `
            + `(${info.n_candidates} candidate${info.n_candidates === 1 ? "" : "s"}); `
            + (violations.length ? "joints out of range:" : "every joint could be wrapped into range, "
              + "so the limit that failed is the solver's own reachability filter.")));
          if (violations.length) {
            parts.push(jointTable(violations.map((v) => [
              jointName(v.joint), degrees(v.value), `${degrees(v.lower)} … ${degrees(v.upper)}`,
            ]), ["joint", "value", "allowed"], violations.map(() => true)));
          }
          // The candidate is 6 values for one arm; put it into the last
          // accepted configuration so the whole robot can be posed with it.
          if (lastConf && info.candidate) {
            const conf = lastConf.slice();
            const first = arm === "left" ? 0 : 6;
            info.candidate.forEach((v, i) => { conf[first + i] = v; });
            poseButton(`robot at ${arm} arm's out-of-limit solution`, conf,
              `the ${arm} arm's nearest solution, past its joint limit`);
          }
        }
        if (d.cause === "fk_gate" && d.candidate) {
          poseButton("robot at the rejected FK-gate solution", d.candidate, "the FK-gate solution");
        }
        poseRobot(lastConf, `the last accepted waypoint (node ${a.last_node})`);
      }
      detail.hidden = false;
      detail.replaceChildren(...parts);
    };

    view.onPick((picked) => {
      if (!picked || !tree) return;
      if (picked.kind === "node") showNode(picked);
      else if (picked.kind === "attempt") {
        reasonChooser.value = picked.attempt.reason;
        stepIndex = stepList().indexOf(picked.attempt);
        updateStepLabel();
        showAttempt(picked.attempt);
      }
    });

    // The summary panel for the pass on show.
    const summariseAttempts = () => {
      attemptBody.replaceChildren();
      if (!attempts.length) {
        attemptBody.appendChild(element("p", { class: "muted" },
          attemptsNote || "No failed attempts recorded for this pass."));
        return;
      }
      const count = (items, keyOf) => {
        const counts = new Map();
        for (const item of items) {
          const key = keyOf(item);
          counts.set(key, (counts.get(key) || 0) + 1);
        }
        return [...counts.entries()].sort((x, y) => y[1] - x[1]);
      };
      const share = (n) => `${(n / attempts.length * 100).toFixed(0)}%`;
      const continuity = attempts.filter((a) => a.reason === "continuity");
      const ik = attempts.filter((a) => a.reason === "ik_failure");
      const collision = attempts.filter((a) => a.reason === "collision");
      const sections = [];

      if (continuity.length) {
        const byJoint = count(continuity, (a) => a.detail.joint);
        const median = (values) => {
          const sorted = values.slice().sort((x, y) => x - y);
          return sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0;
        };
        // How many joints were over the gate at once: one joint flipping is a
        // branch change; all twelve moving is just a step too big for the gate.
        const overCounts = continuity.map((a) =>
          (a.detail.delta_rad || []).filter((v) => v > a.detail.threshold_rad).length);
        sections.push(element("h3", {}, `Continuity: ${continuity.length} (${share(continuity.length)})`));
        sections.push(element("p", { class: "muted" },
          `Joints over the gate per stop: median ${median(overCounts)} of 12. `
          + "A single joint means a branch flip; many at once means the step "
          + "itself was too large for the gate at this resolution."));
        sections.push(table(["joint that jumped most", "stops", "median jump"],
          byJoint.map(([joint, n]) => [jointName(joint), n,
            degrees(median(continuity.filter((a) => a.detail.joint === joint)
              .map((a) => a.detail.delta_rad[joint])))])));
      }
      if (ik.length) {
        sections.push(element("h3", {}, `IK failure: ${ik.length} (${share(ik.length)})`));
        sections.push(table(["arm", "cause", "stops"],
          count(ik, (a) => `${a.detail.arm || "?"}|${a.detail.cause || "?"}`)
            .map(([key, n]) => [...key.split("|"), n])));
      }
      if (collision.length) {
        sections.push(element("h3", {}, `Collision: ${collision.length} (${share(collision.length)})`));
        const pairs = collision.flatMap((a) => (a.detail.pairs || []).map((p) => `${p[0]}|${p[1]}`));
        sections.push(table(["part", "touches", "times"],
          count(pairs, (p) => p).slice(0, 10).map(([key, n]) => [...key.split("|"), n])));
        const unnamed = collision.filter((a) => !a.detail.pairs).length;
        if (unnamed) {
          sections.push(element("p", { class: "muted" },
            `${unnamed} collision stop(s) have no named pair (older run).`));
        }
      }
      attemptBody.replaceChildren(...sections);
    };

    const drawTree = async () => {
      const seed = withTrees[position];
      const passes = seed.trees.passes;
      if (!passes[passName]) passName = Object.keys(passes)[0];
      for (const button of passButtons) {
        const name = button.textContent.split(" ")[0];
        button.setAttribute("aria-pressed", String(name === passName));
        button.disabled = !passes[name];
      }
      previous.disabled = position === 0;
      next.disabled = position === withTrees.length - 1;
      seedLabel.textContent = `seed ${seed.seed} (${position + 1} of ${withTrees.length})`;

      const pass = passes[passName];
      const replay = seed.trees.source === "replay";
      const original = seed.passes && seed.passes[passName];
      const summary = `${passName} pass ${outcome(pass.success, pass.wall_s)}, `
        + `start tree ${pass.tree_a_nodes ?? "?"} nodes, `
        + `goal tree ${pass.tree_b_nodes ?? "?"} nodes`;
      stageInfo.textContent = summary + (replay ? " (replayed)" : "");
      detail.hidden = true;
      view.clearHighlight();
      view.hideGhost();
      stepIndex = -1;

      // Where the search spent its extension attempts, largest first, folded
      // into the same four groups the "why extensions stopped" panel uses.
      const reasons = Object.entries(groupStopReasons(pass.extend_stop_reasons))
        .filter(([, n]) => n > 0)
        .sort((a, b) => b[1] - a[1]);
      const total = reasons.reduce((sum, [, n]) => sum + n, 0) || 1;
      treeInfo.textContent = `Seed ${seed.seed}: ${summary}. `
        + (replay
          ? `This is a replay: same seed and settings, but the search stops on a `
            + `wall-clock limit, so it can end differently from the original, which `
            + `${original ? outcome(original.success, original.wall_s) : "did not run this pass"}. `
          : "")
        + (reasons.length
          ? `Its extensions stopped because of: ${reasons.slice(0, 3)
            .map(([reason, n]) => `${reason.replace(/_/g, " ")} ${(n / total * 100).toFixed(0)}%`)
            .join(", ")}.`
          : "");

      tree = null;
      attempts = [];
      attemptsNote = "";
      view.clearTrees();
      drawSearch();
      if (pass.tree_error) {
        stageInfo.textContent = `${summary}. Trees not drawn: ${pass.tree_error}`;
        summariseAttempts();
        return;
      }
      const loaded = await loadJson(pass.tree);
      if (loaded.error) {
        stageInfo.textContent = `${summary}. Trees not drawn: ${loaded.error}`;
        summariseAttempts();
        return;
      }
      // Older tree files have no joint configurations; the nodes still draw,
      // but clicking one cannot pose the robot.
      for (const part of [loaded.forward, loaded.backward]) {
        if (!part.confs) part.confs = part.points.map(() => null);
      }
      tree = loaded;
      view.setTrees([
        { tree: tree.forward, colour: GHOST_COLOURS.start },
        { tree: tree.backward, colour: GHOST_COLOURS.goal },
      ]);
      if (pass.attempts) {
        attempts = await loadJsonGz(pass.attempts);
      } else {
        attemptsNote = "This pass was planned before failed attempts were "
          + "recorded. Re-plan it (the replay command on this page) to get them.";
      }
      const lastIteration = attempts.reduce((m, a) => Math.max(m, a.iteration), 0);
      iterationSlider.max = String(lastIteration);
      iterationSlider.value = String(lastIteration);
      drawSearch();
      updateStepLabel();
      summariseAttempts();
    };

    previous.addEventListener("click", () => { position -= 1; drawTree(); });
    next.addEventListener("click", () => { position += 1; drawTree(); });
    await drawTree();
  }
  refreshStage();

  // The motions themselves, scrubbable.
  const solved = start.seeds.filter((seed) =>
    Object.values(seed.passes || {}).some((record) => record.trajectory));
  if (!solved.length) {
    main.appendChild(panel("Motion", "No seed solved this one, so there is "
      + "nothing to play back."));
    return;
  }

  const playbackPanel = panel(
    "Motion", "The scrubber sits under the robot at the top of the page.");
  const chooser = element("select", {});
  for (const seed of solved) {
    for (const [name, record] of Object.entries(seed.passes)) {
      if (!record.trajectory) continue;
      chooser.appendChild(element("option",
        { value: record.trajectory }, `seed ${seed.seed}, ${name} pass`));
    }
  }
  const slider = element("input", { type: "range", min: "0", max: "1", value: "0" });
  const readout = element("span", { class: "muted" }, "");
  const jointChart = element("div");
  playbackPanel.append(jointChart);
  main.appendChild(playbackPanel);
  stageParts.playback = element("div", { class: "controls" }, [chooser, slider, readout]);
  refreshStage();

  let path = null;
  const drawJoints = (index) => {
    lineChart(jointChart, {
      height: 220, xLabel: "waypoint", yLabel: "joint angle (rad)",
      cursor: index,
      series: path[0].map((_, joint) => ({
        points: path.map((conf, i) => [i, conf[joint]]),
        colour: joint < 6 ? COLOURS.fine : COLOURS.blocked,
        width: 1, opacity: 0.85,
      })),
      onCursor: (value) => {
        const target = Math.round(value);
        slider.value = String(target);
        show(target);
      },
    });
  };
  const show = (index) => {
    const clamped = Math.max(0, Math.min(path.length - 1, index));
    view.pose(path[clamped]);
    readout.textContent = `waypoint ${clamped + 1} of ${path.length}`;
    drawJoints(clamped);
  };
  const load = async () => {
    path = await loadTrajectory(chooser.value);
    slider.max = String(path.length - 1);
    slider.value = "0";
    const [mx, my, mz] = problem.bar_from_midpoint;
    const points = path.map((conf) => {
      view.pose(conf);
      const e = view.carriedBar.matrix.elements;  // column-major 4x4
      return [
        e[0] * mx + e[4] * my + e[8] * mz + e[12],
        e[1] * mx + e[5] * my + e[9] * mz + e[13],
        e[2] * mx + e[6] * my + e[10] * mz + e[14],
      ];
    });
    view.setTrajectoryLine(points);
    show(0);
  };
  chooser.addEventListener("change", load);
  slider.addEventListener("input", () => show(Number(slider.value)));
  await load();
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Several batches side by side: planner variants on the same problems
// ---------------------------------------------------------------------------

// Where another batch's dashboard lives, relative to this one.
function otherDashboard(runName) {
  return `../../${runName}/dashboard/`;
}

function percent(value) {
  return `${(value * 100).toFixed(0)}%`;
}

// One colour per variant, in the order the comparison lists them.
const VARIANT_PALETTE = [
  COLOURS.fine, COLOURS.coarse, "#2e9e5b", "#8a6fb0", "#d98324", "#4fa3a1", "#b23a48",
];

// A comparison is written by `benchmark_cdfm.py compare`: for several batches
// it keeps only the runs (bar, goal, start, seed) they all have, and measures
// each batch on those. Everything on this page is therefore like for like.
async function viewCompare(name) {
  hideViewer();
  setBreadcrumb([{ label: runInfo.run_name, href: "#/" }, { label: `comparison: ${name}` }]);
  const comparison = await loadJson(paths.compare(name));
  main.replaceChildren();

  const variants = comparison.variants;
  const labelOf = (v) => v.variant || v.run_name;
  const colourOf = (index) => VARIANT_PALETTE[index % VARIANT_PALETTE.length];
  const baseline = variants.find((v) => v.is_baseline);

  main.appendChild(element("section", { class: "panel tiles" }, [
    tile("batches compared", variants.length),
    tile("runs in common", comparison.n_common_runs),
    tile("problems", comparison.n_problems),
    tile("seeds", comparison.seeds.length),
    tile("baseline", baseline ? labelOf(baseline) : comparison.baseline),
  ]));

  // 1. The headline numbers, one row per variant.
  const headPanel = panel(
    "Variants",
    "Every number is over the runs the batches have in common. 'Search only' " +
    "counts a path the search found even when the strict repair then rejected " +
    "it. Problems solved shows, against the baseline, how many problems only " +
    "this variant solved (+) and how many only the baseline solved (−), with " +
    "an exact McNemar p-value. Time ratios are on runs both solved; the " +
    "bracket is a 95% bootstrap interval of the median ratio.");
  headPanel.appendChild(table(
    ["batch", "variant", "solved runs", "95% interval", "problems solved",
      "search only", "planning s, median both-solved", "time ratio vs baseline",
      "repair failures", "IK ms per solve", "first-step failures", "timeouts"],
    variants.map((v) => {
      const totals = v.totals;
      const ratio = v.time_ratio_vs_baseline;
      const paired = v.paired_vs_baseline;
      const repairFailures = Object.values(totals.postprocess_reasons || {})
        .reduce((sum, n) => sum + n, 0);
      return [
        element("a", { href: otherDashboard(v.run_name) }, v.run_name),
        labelOf(v),
        percent(totals.success_rate),
        `${percent(totals.success_rate_ci95[0])}–${percent(totals.success_rate_ci95[1])}`,
        paired
          ? `${v.n_solved_problems}/${v.n_problems}  (+${paired.variant_only} / −${paired.baseline_only}, p=${paired.mcnemar_p.toFixed(2)})`
          : `${v.n_solved_problems}/${v.n_problems}`,
        percent(totals.search_success_rate),
        v.fine_wall_s_both_solved.p50 ? v.fine_wall_s_both_solved.p50.toFixed(0) : "-",
        ratio && ratio.median_ratio != null
          ? `${ratio.median_ratio.toFixed(2)} [${ratio.ci95[0].toFixed(2)}–${ratio.ci95[1].toFixed(2)}], n=${ratio.n}`
          : (v.is_baseline ? "baseline" : "-"),
        repairFailures,
        totals.ik_ms_per_solve ? totals.ik_ms_per_solve.toFixed(1) : "-",
        percent(totals.first_step_failure_rate)
          + (totals.first_step_failure_rate_is_lower_bound ? " (at least)" : ""),
        v.timeouts,
      ];
    })));
  main.appendChild(headPanel);

  // 2. Solved runs, with the search-only rate as the hollow marker.
  const successPanel = panel(
    "Solved runs, by variant",
    "The whisker is a 95% interval. The hollow marker is the search alone, " +
    "before the strict repair on the calibrated model.");
  const successChart = element("div");
  successPanel.appendChild(successChart);
  main.appendChild(withCsv(successPanel, [
    ["variant", "success_rate", "ci_low", "ci_high", "search_success_rate"],
    ...variants.map((v) => [labelOf(v), v.totals.success_rate,
      v.totals.success_rate_ci95[0], v.totals.success_rate_ci95[1],
      v.totals.search_success_rate]),
  ]));
  barChart(successChart, {
    yLabel: "solved", yMax: 1, format: (x) => percent(x),
    data: variants.map((v, i) => ({
      label: labelOf(v), value: v.totals.success_rate, colour: colourOf(i),
      low: v.totals.success_rate_ci95[0], high: v.totals.success_rate_ci95[1],
      marker: v.totals.search_success_rate,
    })),
  });

  // 3. Trying again: the restart curve of every variant on one chart.
  const restartPanel = panel(
    "Trying again",
    "Share of problems solved by at least one of the first k seeds, per variant.");
  const restartChart = element("div");
  restartPanel.appendChild(restartChart);
  lineChart(restartChart, {
    yLabel: "problems solved", xLabel: "seeds tried",
    series: variants.map((v, i) => ({
      label: labelOf(v), colour: colourOf(i),
      points: (v.totals.restart_curve || []).map((y, k) => [k + 1, y]),
    })),
  });
  restartPanel.appendChild(element("p", { class: "muted" },
    variants.flatMap((v, i) => [swatch(colourOf(i)), ` ${labelOf(v)}  `])));
  main.appendChild(restartPanel);

  // 4. Planning time on the runs both the variant and the baseline solved.
  const timePanel = panel(
    "Motion planning time, runs both solved",
    "Fine pass, repair included, log scale: the box spans the 10th to 90th " +
    "percentile, the line is the median. Only runs solved by both the variant " +
    "and the baseline, so a slower day cannot pass for a slower planner.");
  const timeChart = element("div");
  timePanel.appendChild(timeChart);
  boxChart(timeChart, {
    yLabel: "seconds",
    data: variants.map((v, i) => ({
      label: labelOf(v), colour: colourOf(i), ...v.fine_wall_s_both_solved,
    })),
  });
  main.appendChild(timePanel);

  // 5. Why extensions stopped, per variant.
  const stopPanel = panel(
    "Why extensions stopped",
    "Growth, connect and stitch stops folded into the same groups as the overview.");
  const stopChart = element("div");
  stopPanel.appendChild(stopChart);
  stackedBarChart(stopChart, {
    keys: STOP_REASON_ORDER, labels: STOP_REASON_LABELS,
    colours: STOP_REASON_COLOURS, yLabel: "share of stops",
    data: variants.map((v) => ({
      label: labelOf(v), values: groupStopReasons(v.totals.extend_stop_reasons),
    })),
  });
  main.appendChild(stopPanel);

  // 6. Where the time went, per variant (the repair is its own slice).
  const breakdownPanel = panel(
    "Where motion planning time went",
    "Share of time inside the planner, per variant. 'Re-solving the path' is " +
    "the strict repair on the calibrated model that every variant runs.");
  const breakdownChart = element("div");
  breakdownPanel.appendChild(breakdownChart);
  stackedBarChart(breakdownChart, {
    keys: BREAKDOWN_ORDER, labels: BREAKDOWN_LABELS, yLabel: "share of time",
    data: variants.map((v) => ({
      label: labelOf(v), values: v.totals.mean_breakdown_s || {},
    })),
  });
  main.appendChild(breakdownPanel);

  // 7. Why the repair rejected searched paths, when it did.
  const reasonKeys = [...new Set(variants.flatMap(
    (v) => Object.keys(v.totals.postprocess_reasons || {})))];
  if (reasonKeys.length) {
    const repairPanel = panel(
      "Why the repair rejected searched paths",
      "Counts over both passes. A path the search found but the strict " +
      "re-solve could not follow -- a branch flip it could not undo, a " +
      "waypoint without a solution, or a collision on the calibrated model.");
    const repairChart = element("div");
    repairPanel.appendChild(repairChart);
    stackedBarChart(repairChart, {
      keys: reasonKeys,
      labels: Object.fromEntries(reasonKeys.map((k) => [k, k])),
      colours: Object.fromEntries(reasonKeys.map((k, i) => [k, colourOf(i)])),
      yLabel: "repair failures", normalize: false, format: (x) => x.toFixed(0),
      data: variants.map((v) => ({
        label: labelOf(v), values: v.totals.postprocess_reasons || {},
      })),
    });
    main.appendChild(repairPanel);
  }

  // 8. Every problem against every variant: seeds solved, coloured.
  const nSeeds = comparison.seeds.length;
  const matrixPanel = panel(
    "Problem by variant",
    `Seeds solved out of ${nSeeds}, per problem. Click a cell to open that ` +
    "run's start page in its own batch.");
  const rows = comparison.problems.map((p) => {
    const cells = variants.map((v) => {
      const solved = p.solved_seeds[v.run_name] || 0;
      const cell = element("a", {
        href: `${otherDashboard(v.run_name)}#/bar/${p.bar_id}/goal/${p.goal_index}/start/${p.start_index}`,
        class: "matrix-cell",
      }, `${solved}/${nSeeds}`);
      cell.style.display = "inline-block";
      cell.style.minWidth = "3em";
      cell.style.textAlign = "center";
      cell.style.padding = "2px 6px";
      cell.style.borderRadius = "3px";
      cell.style.background = solved ? cssColour(successColour(solved / nSeeds)) : "#f2f4f5";
      cell.style.color = solved ? "white" : "inherit";
      return cell;
    });
    return [p.key, DIFFICULTY_LABELS[p.difficulty] || p.difficulty || "-",
      p.anchor || "-", ...cells];
  });
  matrixPanel.appendChild(withCsv(
    element("div"),
    [["problem", "start", "carry", ...variants.map(labelOf)],
      ...comparison.problems.map((p) => [p.key, p.difficulty, p.anchor,
        ...variants.map((v) => p.solved_seeds[v.run_name] || 0)])],
  ));
  matrixPanel.appendChild(table(
    ["problem", "start", "carry", ...variants.map(labelOf)], rows));
  main.appendChild(matrixPanel);
}

async function route() {
  const hash = window.location.hash.replace(/^#\/?/, "");
  const parts = hash.split("/").filter(Boolean);
  main.scrollTop = 0;
  layout.scrollTop = 0;
  try {
    if (!parts.length) await viewOverview();
    else if (parts[0] === "compare" && parts.length === 2) await viewCompare(parts[1]);
    else if (parts[0] === "bar" && parts.length === 2) await viewBar(parts[1]);
    else if (parts[0] === "bar" && parts[2] === "goal" && parts.length === 4) {
      await viewGoal(parts[1], Number(parts[3]));
    } else if (parts[0] === "bar" && parts[4] === "start") {
      await viewStart(parts[1], Number(parts[3]), Number(parts[5]));
    } else {
      await viewOverview();
    }
  } catch (error) {
    main.replaceChildren(element("section", { class: "panel error" }, [
      element("h2", {}, "Something went wrong"),
      element("pre", {}, String(error && error.message ? error.message : error)),
    ]));
  }
}

async function start() {
  runInfo = await loadJson(paths.run());
  await loadRunList();
  document.querySelector("[data-title]").textContent =
    `${runInfo.run_name} — CDFM planner benchmark`;
  window.addEventListener("hashchange", route);
  await route();
}

start();
