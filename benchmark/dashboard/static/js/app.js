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

import { loadJson, loadTrajectory, paths } from "./data.js";
import {
  COLOURS, BREAKDOWN_ORDER, BREAKDOWN_LABELS, SETUP_ORDER, SETUP_LABELS,
  SETUP_COLOURS, barChart, boxChart, heatmap, lineChart,
  stackedBarChart, copyAsCsv,
} from "./charts.js";
import { OUTCOME_COLOURS, RobotViewer } from "./robot-viewer.js";

const main = document.querySelector("[data-main]");
const breadcrumb = document.querySelector("[data-breadcrumb]");
const viewerHost = document.querySelector("[data-viewer]");
const viewerCaption = document.querySelector("[data-viewer-caption]");
const stageControls = document.querySelector("[data-stage-controls]");

let viewer = null;
let runInfo = null;
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

function setBreadcrumb(parts) {
  breadcrumb.replaceChildren(...parts.flatMap((part, index) => {
    const node = part.href
      ? element("a", { href: part.href }, part.label)
      : element("span", {}, part.label);
    return index ? [element("span", { class: "sep" }, "/"), node] : [node];
  }));
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

  // 4. Why the searches stopped.
  const reasonKeys = Array.from(new Set(
    bars.flatMap((b) => Object.keys(b.extend_stop_reasons || {}))));
  if (reasonKeys.length) {
    const reasonPanel = panel(
      "Why extensions stopped",
      "Each attempt to grow the search tree ends for a reason. " +
      "'continuity' means the arm solution jumped; 'collision' means " +
      "something was in the way.");
    const reasonChart = element("div");
    reasonPanel.appendChild(reasonChart);
    main.appendChild(reasonPanel);
    stackedBarChart(reasonChart, {
      keys: reasonKeys,
      yLabel: "share of stops",
      data: bars.map((b) => ({ label: b.bar_id, values: b.extend_stop_reasons || {} })),
      format: (v) => v.toFixed(2),
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
    if (goalPoseMode === "all") {
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
  const modeButtons = [["workable", "Workable only"], ["all", "Include failed"]]
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

  const derivation = goal.production_derivation || {};
  main.appendChild(panel(
    "What the planner itself would have done",
    derivation.found
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

  // The search trees, seed by seed. These are what explain a failure: where
  // each tree got to, and where it stopped.
  const withTrees = start.seeds.filter((seed) =>
    seed.trees && seed.trees.passes && Object.keys(seed.trees.passes).length);
  const treePanel = panel(
    "Search trees",
    "The two trees the search grew: one from the start bar, one from the goal "
    + "bar, drawn as the path of the bar's midpoint. A solved run is where "
    + "they met. Page through the seeds under the robot.");
  main.appendChild(treePanel);

  if (!withTrees.length) {
    const command = `python scripts/benchmark_cdfm.py replay --name ${runInfo.run_name} `
      + `--bar ${barId} --goal ${goalIndex} --start ${startIndex}`;
    treePanel.append(
      element("p", {}, "Not saved for this start. The batch kept only the size "
        + "of each tree. To see them, re-plan this start with its trees saved "
        + "(same seeds and settings, a few minutes), then reload:"),
      element("pre", { class: "command" }, command));
  } else {
    const treeInfo = element("p", { class: "tree-info" }, "");
    treePanel.appendChild(treeInfo);
    let position = 0;
    let passName = "coarse";

    const previous = element("button", { type: "button" }, "◀ previous seed");
    const next = element("button", { type: "button" }, "next seed ▶");
    const seedLabel = element("span", { class: "muted" }, "");
    const passButtons = ["coarse", "fine"].map((name) => {
      const button = element("button", { type: "button" }, `${name} pass`);
      button.addEventListener("click", () => { passName = name; drawTree(); });
      return button;
    });
    const stageInfo = element("div", { class: "muted tree-stage-info" }, "");
    stageParts.trees = element("div", {}, [
      element("div", { class: "controls" }, [
        element("div", { class: "toggle" }, [previous, next]),
        seedLabel,
        element("div", { class: "toggle", role: "group",
          "aria-label": "which pass" }, passButtons),
      ]),
      stageInfo,
    ]);

    const outcome = (success, wall) =>
      `${success ? "solved" : "failed"}${wall != null ? ` in ${wall.toFixed(0)} s` : ""}`;

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

      // Where the search spent its extension attempts, largest first.
      const reasons = Object.entries(pass.extend_stop_reasons || {})
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

      if (pass.tree_error) {
        view.clearTrees();
        stageInfo.textContent = `${summary}. Trees not drawn: ${pass.tree_error}`;
        return;
      }
      const tree = await loadJson(pass.tree);
      if (tree.error) {
        view.clearTrees();
        stageInfo.textContent = `${summary}. Trees not drawn: ${tree.error}`;
        return;
      }
      view.setTrees([
        { tree: tree.forward, colour: GHOST_COLOURS.start },
        { tree: tree.backward, colour: GHOST_COLOURS.goal },
      ]);
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

async function route() {
  const hash = window.location.hash.replace(/^#\/?/, "");
  const parts = hash.split("/").filter(Boolean);
  main.scrollTop = 0;
  layout.scrollTop = 0;
  try {
    if (!parts.length) await viewOverview();
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
  document.querySelector("[data-title]").textContent =
    `${runInfo.run_name} — CDFM planner benchmark`;
  window.addEventListener("hashchange", route);
  await route();
}

start();
