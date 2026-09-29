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
  COLOURS, BREAKDOWN_ORDER, barChart, boxChart, heatmap, lineChart,
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
  viewerHost.parentElement.hidden = false;
  viewerCaption.textContent = caption;
  if (viewer) viewer.resize();
}

function hideViewer() {
  viewerHost.parentElement.hidden = true;
  clearStageControls();
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
  main.appendChild(element("section", { class: "panel tiles" }, [
    tile("runs", totals.n_runs),
    tile("solved", `${(totals.success_rate * 100).toFixed(0)}%`),
    tile("median time", `${totals.fine_wall_s.p50.toFixed(1)} s`),
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
    "Planning time, by bar",
    "Solved runs only, on a log scale. The box spans the 10th to 90th " +
    "percentile, the line inside is the median.");
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

  // 5. Where the time went.
  const breakdownPanel = panel(
    "Where the time went",
    "Mean seconds per run, split by what the planner was doing.");
  const breakdownChart = element("div");
  breakdownPanel.appendChild(breakdownChart);
  main.appendChild(withCsv(breakdownPanel, [
    ["bar", ...BREAKDOWN_ORDER],
    ...bars.map((b) => [b.bar_id,
      ...BREAKDOWN_ORDER.map((k) => (b.mean_breakdown_s || {})[k] || 0)]),
  ]));
  stackedBarChart(breakdownChart, {
    keys: BREAKDOWN_ORDER, yLabel: "share of time",
    data: bars.map((b) => ({ label: b.bar_id, values: b.mean_breakdown_s || {} })),
  });

  // 6. The bars themselves, as a table.
  const listPanel = panel("Bars", "Click a row to open it.");
  listPanel.appendChild(table(
    ["bar", "length", "runs", "solved", "coarse", "median s"],
    bars.map((b) => [b.bar_id, `${b.bar_length_m.toFixed(2)} m`, b.n_runs,
      `${(b.success_rate * 100).toFixed(0)}%`,
      `${(b.coarse_success_rate * 100).toFixed(0)}%`,
      b.fine_wall_s.p50.toFixed(1)]),
    { onRowClick: (row) => { window.location.hash = `#/bar/${row[0]}`; } },
  ));
  main.appendChild(listPanel);
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
  showViewer(`${barId}: ${bar.goals.length} goal poses, coloured by how often `
    + "they were solved. Click one to open it.");

  const view = await ensureRobot();
  view.clearPoseClouds();
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

  const summaryPanel = panel(
    `${barId}, ${problem.bar_length_m.toFixed(2)} m`,
    `Held ${describeGrasp(problem)}.`);
  main.appendChild(summaryPanel);

  const goalsPanel = panel("Goals", "Click a row to open its start survey.");
  goalsPanel.appendChild(table(
    ["goal", "runs", "solved", "median s", "starts found", "grasps", "easy starts"],
    bar.goals.map((goal) => [
      goal.goal_index, goal.n_runs,
      `${(goal.success_rate * 100).toFixed(0)}%`,
      goal.median_wall_s ? goal.median_wall_s.toFixed(1) : "-",
      goal.n_starts_found, goal.n_free_branch_pairs,
      (goal.difficulty_counts || {}).free || 0,
    ]),
    { onRowClick: (row) => {
      window.location.hash = `#/bar/${barId}/goal/${row[0]}`;
    } },
  ));
  main.appendChild(goalsPanel);
}

function describeGrasp(problem) {
  const left = problem.grasp_bar_from_left[0];
  const right = problem.grasp_bar_from_right[0];
  return `at ${Math.abs(left[2]).toFixed(2)} m and `
    + `${Math.abs(right[2]).toFixed(2)} m along its length`;
}

function successColour(rate) {
  if (rate == null) return 0xb8bec2;
  if (rate >= 0.66) return OUTCOME_COLOURS.free;
  if (rate > 0) return OUTCOME_COLOURS.blocked;
  return OUTCOME_COLOURS.broken;
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
  showViewer(`Goal ${goalIndex} in blue, and every carry pose the survey found `
    + "workable, coloured by how hard it looks to reach. Click one to open it.");

  const view = await ensureRobot();
  view.clearPoseClouds();
  view.setTrajectoryLine(null);
  view.pose(goal.primary_conf);
  view.setCarriedBar({
    length: problem.bar_length_m, radius: problem.bar_radius_m,
    linkName: problem.attached_link, grasp: problem.grasp_bar_from_left,
    bar_from_midpoint: problem.bar_from_midpoint,
  });
  view.addPoseCloud(goal.starts.map((start) => ({
    world_from_bar: start.world_from_bar,
    colour: OUTCOME_COLOURS[start.difficulty] || 0xb8bec2,
    id: start.start_index,
  })), {
    length: problem.bar_length_m, radius: problem.bar_radius_m,
    bar_from_midpoint: problem.bar_from_midpoint, opacity: 0.5,
  });
  view.onPick((startIndex) => {
    const start = goal.starts.find((s) => s.start_index === startIndex);
    if (start && start.planned) {
      window.location.hash = `#/bar/${barId}/goal/${goalIndex}/start/${startIndex}`;
    } else if (start) {
      view.pose(start.conf || goal.primary_conf);
    }
  });

  // Why candidates failed.
  const counts = goal.sweep_counts || {};
  const reasonPanel = panel(
    "The start survey",
    `${goal.candidates.length} carry poses tried. A pose fails either because `
    + "an arm cannot reach it, or because holding it would collide.");
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
  for (const candidate of goal.candidates) {
    for (const pair of candidate.collision_pairs || []) {
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
    ["start", "anchor", "difficulty", "solved", "median s"],
    planned.map((start) => [
      start.start_index, start.anchor, start.difficulty || "-",
      `${((start.success_rate || 0) * 100).toFixed(0)}%`,
      start.median_wall_s ? start.median_wall_s.toFixed(1) : "-",
    ]),
    { onRowClick: (row) => {
      window.location.hash = `#/bar/${barId}/goal/${goalIndex}/start/${row[0]}`;
    } },
  ));
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
      : "Found no carry pose at all within its time budget."));
}

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
  view.setCarriedBar({
    length: problem.bar_length_m, radius: problem.bar_radius_m,
    linkName: problem.attached_link, grasp: problem.grasp_bar_from_left,
    bar_from_midpoint: problem.bar_from_midpoint,
  });
  if (startRecord && startRecord.conf) view.pose(startRecord.conf);

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
  const timePanel = panel("Time per seed", null);
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
  const breakdownPanel = panel("Where each seed's time went", null);
  const breakdownChart = element("div");
  breakdownPanel.appendChild(breakdownChart);
  main.appendChild(breakdownPanel);
  stackedBarChart(breakdownChart, {
    keys: BREAKDOWN_ORDER, yLabel: "share of time",
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

  // The motions themselves, scrubbable.
  const solved = start.seeds.filter((seed) =>
    Object.values(seed.passes).some((record) => record.trajectory));
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
  setStageControls(element("div", { class: "controls" },
    [chooser, slider, readout]));

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
    const points = path.map((conf) => {
      view.pose(conf);
      const matrix = view.carriedBar.matrix;
      return [matrix.elements[12], matrix.elements[13], matrix.elements[14]];
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
