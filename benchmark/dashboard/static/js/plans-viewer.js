// The page of a set of bar-carry plans: every bar of a process in build
// order, and each plan in 3D in the scene it was made in.
//
// The front page lists the bars with what their plan cost, phase by phase.
// A bar's page poses the robot along the planned path with a slider (the
// carried bar and its joint halves follow the arms; the built bars, the
// obstacles and the other robots stand where the scene of that bar's turn has
// them), draws the joint values and how well the bar is held along the path
// with a cursor at the waypoint on show, and shows where the planning time
// went and what the checks found.
//
// The files come from scripts/prm_plans_page.py: plans.json (one row per
// bar) and bars/<bar>.json (the path, the scene's address, timings, counters).
//
//   #/            every bar
//   #/bar/<id>    one bar's plan

import * as THREE from "three";
import { loadJson } from "./data.js";
import { JOINT_COLOURS, horizontalBarChart, lineChart, markIcon, stackedBarChart } from "./charts.js";
import { RobotViewer, poseToMatrix } from "./robot-viewer.js";

const main = document.querySelector("[data-main]");
const breadcrumb = document.querySelector("[data-breadcrumb]");
const viewerHost = document.querySelector("[data-viewer]");
const viewerCaption = document.querySelector("[data-viewer-caption]");
const stageControls = document.querySelector("[data-stage-controls]");

// * The phases of one plan, in the order they run, and how the front page groups them.
const PHASES = [
  ["load_s", "read the roadmap", "setup"],
  ["worker_start_s", "start the workers", "setup"],
  ["scene_setup_s", "write and open the scene for the workers", "setup"],
  ["goal_s", "end postures: IK and collision check at the goal poses", "ends"],
  ["starts_s", "start set: scene check of the roadmap's carry poses", "ends"],
  ["connect_s", "join new end nodes to the roadmap", "ends"],
  ["search_s", "search: tracking edges (holding the bar) and their scene checks", "search"],
  ["walk_in_s", "fallback 1: walk the goal postures to loading poses of the three classes", "walk"],
  ["via_s", "fallback 2: waypoints grown from the goal out of the structure", "walk"],
  ["drive_in_s", "drive-in check: a base path from the station with the arms as loaded", "drive"],
  ["goal_accept_s", "end posture: the insert and the retreat planned from it", "follow"],
  ["final_check_s", "final check of the route at the fine step", "final"],
  ["post_process_s", "after the route: smoothing and corner rounding", "post"],
  ["dense_check_s", "dense check of the finished path between its waypoints", "final"],
];
const GROUPS = ["setup", "ends", "search", "walk", "drive", "follow", "final", "post", "other"];
const GROUP_LABELS = {
  setup: "roadmap, workers, scene", ends: "start and end sets", search: "search (tracking + scene checks)",
  walk: "fallbacks (loading poses, waypoints)", drive: "drive-in check",
  follow: "insert and retreat check", final: "final and dense checks", post: "smoothing and rounding",
  other: "rest of the call",
};
const GROUP_COLOURS = {
  setup: "#b8bcc2", ends: "#8a6fb0", search: "#1f84bd", walk: "#17a2a2", drive: "#e0b030", follow: "#2e9e5b",
  final: "#d98324", post: "#b23a48", other: "#5a5a6e",
};
const BUILD_PHASES = {
  pool_startup_s: "start the workers", anchors_s: "anchors: goal and carry postures", sample_s: "sample nodes",
  knn_s: "pick candidate edges", edges_s: "edges: collision and holding error",
};

// * Points per step of the path at which the holding of the bar is measured between waypoints (joints interpolated
// * linearly), and the most points measured for one path (longer paths get fewer per step).
const BETWEEN_POINTS = 8;
const BETWEEN_CAP = 8000;
const AXIS_COLOURS = ["#b23a48", "#2e9e5b", "#1f84bd"];

// `held` is how the bar is held along the path on show (see measurePath); `pathCharts` are the svgs of the path
// panel, kept so the cursor moves without a redraw.
const state = { index: null, bar: null, data: null, step: 0, playing: null, held: null, pathCharts: [] };
let viewer = null;
let robotReady = null;
let grasps = null;
let rightBar = null;

// ---------------------------------------------------------------------------
// * Small helpers
// ---------------------------------------------------------------------------

function element(tag, attributes = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of [].concat(children)) {
    if (child == null) continue;
    node.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function table(headers, rows, { onRowClick = null } = {}) {
  const head = element("tr", {}, headers.map((h) => element("th", {}, h)));
  const body = rows.map((row, index) => {
    const tr = element("tr", {}, row.map((cell) => element("td", {}, cell instanceof Node ? cell : String(cell))));
    if (onRowClick) {
      tr.className = "clickable";
      tr.addEventListener("click", () => onRowClick(index));
    }
    return tr;
  });
  return element("table", { class: "grid" }, [element("thead", {}, [head]), element("tbody", {}, body)]);
}

const panel = (title, children) => element("section", { class: "panel" }, [element("h2", {}, title), ...children]);
const tiles = (items) => element("div", { class: "tiles" }, items.map(([label, value]) =>
  element("div", { class: "tile" }, [element("strong", {}, String(value)), element("span", {}, label)])));
const seconds = (v) => (v == null ? "—" : v >= 100 ? `${v.toFixed(0)} s` : `${v.toFixed(1)} s`);
const number = (v, digits = 0, unit = "") => (v == null || !Number.isFinite(v) ? "—" : `${v.toFixed(digits)}${unit}`);
const median = (values) => {
  const sorted = values.filter((v) => v != null).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

// One plan's time by group, the rest of the call (scene sync, start derivation, saving) as "other".
function groupSeconds(row) {
  const out = Object.fromEntries(GROUPS.map((g) => [g, 0]));
  const timings = row.timings || {};
  for (const [key, , group] of PHASES) out[group] += timings[key] || 0;
  const timed = PHASES.reduce((sum, [key]) => sum + (timings[key] || 0), 0);
  out.other = Math.max(0, (row.plan_s || 0) - timed);
  return out;
}

// Why a saved path failed its own check: the check's reason, else what the fine re-solve hit.
function checkFailure(check) {
  if (check.reason) return check.reason;
  if (check.repair_success === false) return `${check.repair_reason || "no path"} when re-solved at 2 mm`;
  if (check.saved_waypoints_colliding) return `${check.saved_waypoints_colliding} waypoint(s) in collision`;
  return "?";
}

const plannerName = (backend) => (backend === "prm" ? "roadmap" : backend === "birrt" ? "BiRRT" : backend || "?");

// Which fallback, if any, a roadmap plan needed: "none" (a precomputed carry pose, found by the first search),
// "walk" (a precomputed carry pose, reached by walking the goal posture to it), "loading pose" (a pose of the three
// classes rederived for the bar) or "waypoints" (waypoints out of the structure as well).
function fallbackOf(row) {
  const start = row.start || {};
  if (!row.ok || row.backend !== "prm") return null;
  const after = start.found_after || "carry poses";
  if (after.startsWith("waypoints")) return "waypoints";
  if (start.source === "grid" || start.source === "derived") return "loading pose";
  return after === "carry poses" ? "none" : "walk";
}

// Where a carry starts: the class of the loading pose and where the pose came from.
function loadingPose(row, long = false) {
  const start = row.start || {};
  const kind = fallbackOf(row);
  if (!kind || !start.carry) return "—";
  const rederived = start.source === "grid" || start.source === "derived";
  const origin = !rederived ? (kind === "none" ? "precomputed"
    : long ? "a precomputed carry pose, found by walking the goal posture straight to it" : "precomputed, walked to")
    : start.source === "derived" ? (long ? "rederived by an earlier plan and saved" : "rederived (saved)") : "rederived";
  const via = kind === "waypoints" ? (long ? ", reached over waypoints out of the structure" : " + waypoints") : "";
  const where = long && start.midpoint ? `, grasp midpoint at [${start.midpoint.map((v) => v.toFixed(2)).join(", ")}] m `
    + `in the base frame, roll ${number(start.roll_deg, 0, "°")}` : "";
  return `${start.carry}: ${origin}${via}${where}`;
}

// * The mark of a loading pose under a bar of the time chart: the shape says the class, the colour whether the pose
// * was precomputed or rederived for the bar.
const CLASS_SHAPES = { horizontal: "wide", vertical: "tall", back: "dot" };
const ORIGIN_COLOURS = { precomputed: "#33383b", rederived: "#e8590c" };

function loadingMark(row) {
  const start = row.start || {};
  if (!fallbackOf(row) || !CLASS_SHAPES[start.carry]) return null;
  const rederived = start.source === "grid" || start.source === "derived";
  return { shape: CLASS_SHAPES[start.carry], colour: ORIGIN_COLOURS[rederived ? "rederived" : "precomputed"],
           title: `${row.bar}, loading pose ${loadingPose(row)}` };
}

// The drive in with the loading posture, in a few words: what the check found for the plan's start.
function driveText(row) {
  const drive = row.drive_in;
  if (!row.planned || row.backend !== "prm") return "—";
  if (!drive) return "not checked";
  const refused = row.starts_rejected ? `, ${row.starts_rejected} refused first` : "";
  // ! No drive path on a failed plan says nothing about the postures: some may have passed the check.
  if (!drive.found) return row.ok ? "passed, path not recorded" : `no plan; ${drive.refused} posture(s) refused`;
  return `${drive.direct ? "straight in" : "path"}, ${number(drive.length_m, 1, " m")}${refused}`;
}

function resultText(row) {
  if (!row.planned) return "not planned";
  if (!row.ok) return `no path: ${row.failure_reason || "?"}`;
  if (row.check && row.check.ok === false) return `path, check FAILED: ${checkFailure(row.check)}`;
  return "path found";
}

// ---------------------------------------------------------------------------
// * The front page: every bar
// ---------------------------------------------------------------------------

function renderOverview() {
  const bars = state.index.bars;
  const planned = bars.filter((b) => b.planned);
  const solved = planned.filter((b) => b.ok);
  const checked = solved.filter((b) => b.check && b.check.ok);
  const workers = [...new Set(planned.map((b) => b.workers).filter((w) => w != null))].join(", ");
  const onRoadmap = solved.filter((b) => b.backend === "prm");
  const fallback = solved.filter((b) => b.first_try);
  const children = [
    panel(`${state.index.problem}: ${bars.length} bars in build order`, [
      tiles([
        ["bars with a path on their roadmap", `${onRoadmap.length} / ${bars.length}`],
        ["of these, from a precomputed carry pose, first search", onRoadmap.filter((b) => fallbackOf(b) === "none").length],
        ["from a precomputed carry pose, walked to from the goal", onRoadmap.filter((b) => fallbackOf(b) === "walk").length],
        ["from a loading pose rederived for the bar", onRoadmap.filter((b) => fallbackOf(b) === "loading pose").length],
        ["over waypoints out of the structure", onRoadmap.filter((b) => fallbackOf(b) === "waypoints").length],
        ...(fallback.length ? [["more with the BiRRT, after the roadmap found no route", fallback.length]] : []),
        ["paths passing their own check", `${checked.length} / ${solved.length}`],
        ["median planning time on the roadmap", seconds(median(onRoadmap.map((b) => b.plan_s)))],
        ["slowest on the roadmap", seconds(Math.max(0, ...onRoadmap.map((b) => b.plan_s || 0)))],
        ["worker processes per roadmap plan", workers || "—"],
      ]),
      element("p", { class: "muted" }, "Each bar's carry is planned on that bar's own roadmap, in the scene of its "
        + "turn: the bars built before it, the obstacles and the other robots where the process has them, and a "
        + "floor level with the robot's base. The end posture is resolved by IK at the authored tool poses; it is "
        + "kept only when the insert and the retreat can be planned from it. A plan starts from the precomputed "
        + "carry poses; when none connects, the goal postures are walked to other loading poses of the same three "
        + "classes (horizontal, vertical, back), and when the structure blocks those walks, waypoints are grown "
        + "from the goal outward. Click a row for the plan in 3D."),
    ]),
    panel("Where the planning time went, by bar", [
      element("p", { class: "muted" }, "Wall time of each plan (seconds), split by phase, the bars in build order; "
        + "click one for its page. The roadmap's precompute (building it, once per bar) is not in these bars; it is "
        + "in the table and on each bar's page."),
      element("div", { class: "chart", "data-chart-time": "" }),
      element("div", { class: "legend" }, [
        element("span", { class: "legend-item" }, "Under each bar, the loading pose its carry starts from:"),
        ...Object.entries(CLASS_SHAPES).map(([name, shape]) =>
          element("span", { class: "legend-item" }, [markIcon(shape, "#7a8288"), name])),
        ...Object.entries(ORIGIN_COLOURS).map(([name, colour]) => element("span", { class: "legend-item" }, [
          element("i", { style: `background:${colour}` }),
          name === "rederived" ? "rederived for the bar by a fallback" : "precomputed"])),
      ]),
    ]),
    panel("Bars", [table(
      ["#", "bar", "built before", "planner", "result", "loading pose", "drive-in", "plan", "search", "fallbacks",
       "ends", "follow-on", "post", "roadmap nodes",
       "build", "waypoints", "bar travel", "lowest above floor", "end posture"],
      bars.map((b) => {
        const groups = groupSeconds(b);
        const check = b.check || {};
        const follow = b.follow_on;
        const posture = !b.ok || !follow ? "—"
          : follow.end_is_authored_posture ? `authored${follow.end_off_authored_deg > 1 ? " (whole turns off)" : ""}`
            : `IK-resolved${follow.rejected ? `, ${follow.rejected} refused first` : ""}`;
        const planner = !b.planned ? "—" : b.first_try
          ? `${plannerName(b.backend)} (roadmap: ${b.first_try.failure_reason}`
            + `${b.first_try.best_gap_left_mm != null ? `, ${b.first_try.best_gap_left_mm.toFixed(0)} mm short` : ""})`
          : plannerName(b.backend);
        return [b.order, b.bar, b.built_before, planner, resultText(b), loadingPose(b), driveText(b),
                b.planned ? seconds(b.plan_s) : "—", b.planned ? seconds(groups.search) : "—",
                b.planned && groups.walk ? seconds(groups.walk) : "—", b.planned ? seconds(groups.ends) : "—",
                b.planned ? seconds(groups.follow) : "—", b.planned ? seconds(groups.post) : "—",
                b.roadmap ? b.roadmap.n_nodes : "—", seconds(b.build_s),
                b.ok ? b.n_waypoints : "—",
                b.ok && check.bar_travel_m != null ? `${check.bar_travel_m.toFixed(2)} m / ${number(check.bar_turn_deg, 0, "°")}` : "—",
                check.lowest_z_above_base_frame_m != null ? number(check.lowest_z_above_base_frame_m * 1000, 0, " mm") : "—",
                posture];
      }),
      { onRowClick: (index) => { location.hash = `#/bar/${bars[index].bar}`; } })]),
  ];
  main.replaceChildren(...children);
  stackedBarChart(main.querySelector("[data-chart-time]"), {
    data: planned.map((b) => ({ label: b.bar.replace(/^B/, ""), values: groupSeconds(b), mark: loadingMark(b),
                                onClick: () => { location.hash = `#/bar/${b.bar}`; } })),
    keys: GROUPS, colours: GROUP_COLOURS, labels: GROUP_LABELS, normalize: false, yLabel: "seconds",
    format: (v) => v.toFixed(0),
  });
}

// ---------------------------------------------------------------------------
// * One bar: the 3D view
// ---------------------------------------------------------------------------

async function showBarScene() {
  const data = state.data;
  if (!viewer) viewer = new RobotViewer(viewerHost);
  if (!robotReady) {
    robotReady = viewer.loadRobot(`${state.index.assets}/scene.glb`, `${state.index.assets}/kinematics.json`);
  }
  await robotReady;
  const problem = data.problem;
  const left = poseToMatrix(problem.grasp_bar_from_left);
  const right = poseToMatrix(problem.grasp_bar_from_right);
  grasps = { left, right, leftInv: left.clone().invert(), rightInv: right.clone().invert() };
  viewer.setCarriedBar({
    length: problem.bar_length_m, radius: problem.bar_radius_m, linkName: problem.attached_link,
    grasp: problem.grasp_bar_from_left, bar_from_midpoint: problem.bar_from_midpoint,
  });
  if (rightBar) { viewer.scene.remove(rightBar); rightBar.geometry.dispose(); rightBar = null; }
  if (data.scene) await viewer.loadStaticScene(data.scene);
  else viewer.clearStaticScene();
  state.held = measurePath(data.positions, problem);
  viewer.setTrajectoryLine(state.held.midpoints);
  drawDrive((data.drive_in || {}).line);
}

// The base's way in from the station, on the floor: the path, and a short stroke the way the robot faces every
// few poses. The points are (x, y, yaw) in the frame of the base pose the drive ends at, which is the view's frame.
function drawDrive(line) {
  viewer.clearOverlays();
  if (!line || line.length < 2) return;
  const z = 0.02;
  const path = line.slice(1).map((point, i) => [[line[i][0], line[i][1], z], [point[0], point[1], z]]);
  const every = Math.max(1, Math.round(line.length / 40));
  const facing = line.filter((_, i) => i % every === 0).map(([x, y, yaw]) =>
    [[x, y, z], [x + 0.25 * Math.cos(yaw), y + 0.25 * Math.sin(yaw), z]]);
  viewer.addLines(path, { colour: 0x7a5c00, opacity: 1 });
  viewer.addLines(facing, { colour: 0xe0b030, opacity: 0.9 });
}

// The right flange as seen from the left one, at the posed robot: fixed while both arms hold the bar.
function rightFromLeft() {
  const left = viewer.kinematics.linkMatrix("left_ur_arm_tool0");
  const right = viewer.kinematics.linkMatrix("right_ur_arm_tool0");
  return left && right ? left.clone().invert().multiply(right) : null;
}

// How far a flange-to-flange transform is from a reference one: the distance in mm, then per axis (x, y, z) the
// angle in degrees between that axis of the frame now and in the reference.
function driftFrom(reference, now) {
  const a = reference.elements;
  const b = now.elements;
  const out = [Math.hypot(b[12] - a[12], b[13] - a[13], b[14] - a[14]) * 1000];
  for (const c of [0, 4, 8]) {
    const u = new THREE.Vector3(a[c], a[c + 1], a[c + 2]);
    const v = new THREE.Vector3(b[c], b[c + 1], b[c + 2]);
    out.push(Math.atan2(u.clone().cross(v).length(), u.dot(v)) * 180 / Math.PI);
  }
  return out;
}

// Poses every waypoint once and reads off the path: the line the bar's midpoint takes, and how the bar is held
// against the first waypoint, `at` the waypoints ([mm, x°, y°, z°] each) and `between` them (per step of the path,
// [where the step is on the waypoint axis, then the largest mm, x°, y°, z° on it], joints interpolated linearly).
function measurePath(positions, problem) {
  const mid = new THREE.Vector3(...(problem.bar_from_midpoint || [0, 0, 0]));
  const held = { midpoints: [], at: [], between: [], perStep: 1 };
  held.perStep = Math.max(1, Math.min(BETWEEN_POINTS, Math.floor(BETWEEN_CAP / Math.max(1, positions.length))));
  let reference = null;
  for (let k = 0; k < positions.length; k += 1) {
    viewer.kinematics.pose(positions[k]);
    const flange = viewer.kinematics.linkMatrix(problem.attached_link);
    const now = rightFromLeft();
    if (!flange || !now) break;
    const p = mid.clone().applyMatrix4(flange.clone().multiply(grasps.leftInv));
    held.midpoints.push([p.x, p.y, p.z]);
    reference = reference || now;
    held.at.push(driftFrom(reference, now));
    if (k + 1 === positions.length) break;
    const worst = [0, 0, 0, 0];
    for (let s = 1; s <= held.perStep; s += 1) {
      const t = s / (held.perStep + 1);
      viewer.kinematics.pose(positions[k].map((v, j) => v + (positions[k + 1][j] - v) * t));
      driftFrom(reference, rightFromLeft()).forEach((value, i) => { worst[i] = Math.max(worst[i], value); });
    }
    held.between.push([k + 0.5, ...worst]);
  }
  return held;
}

// The holding error at the posed robot: how far each flange is from where the other one puts it.
function holdingError() {
  const leftLink = viewer.kinematics.linkMatrix("left_ur_arm_tool0");
  const rightLink = viewer.kinematics.linkMatrix("right_ur_arm_tool0");
  if (!leftLink || !rightLink) return null;
  const expectedRight = leftLink.clone().multiply(grasps.leftInv).multiply(grasps.right);
  const a = new THREE.Vector3().setFromMatrixPosition(expectedRight);
  const b = new THREE.Vector3().setFromMatrixPosition(rightLink);
  return a.distanceTo(b);
}

function showStep(index) {
  const positions = state.data.positions;
  if (!positions.length) return;
  state.step = Math.max(0, Math.min(positions.length - 1, index));
  viewer.pose(positions[state.step]);
  const slider = stageControls.querySelector("[data-path-slider]");
  if (slider) slider.value = String(state.step);
  // Only the cursors move; the charts stay as drawn.
  for (const svg of state.pathCharts) svg.setCursor(state.step);
  const error = holdingError();
  const onRoadmap = ((state.data.plan || {}).backend || "prm") === "prm";
  const row = state.index.bars.find((b) => b.bar === state.data.bar) || {};
  const where = state.step === 0 ? (onRoadmap ? ` (start: the loading pose, ${loadingPose(row)})` : " (start: derived by the search)")
    : state.step === positions.length - 1 ? " (end: the authored tool poses, where the insert starts)" : "";
  viewerCaption.textContent = `Waypoint ${state.step + 1} of ${positions.length}${where}.`
    + (error != null ? ` Holding error here: ${(error * 1000).toFixed(2)} mm.` : "")
    + " Yellow line: the bar's midpoint along the path."
    + ((state.data.drive_in || {}).line ? " Brown line on the floor: the base's way in from the station, with the "
      + "arms in the loading posture (short strokes: the way the robot faces)." : "")
    + " Drag to orbit, scroll to zoom; Space plays and pauses, the arrow keys step.";
}

function stopPlaying() {
  if (state.playing) { clearInterval(state.playing); state.playing = null; }
  const button = stageControls.querySelector("[data-play]");
  if (button) button.textContent = "Play";
}

function renderStageControls() {
  stopPlaying();
  const positions = state.data.positions;
  if (!positions.length) {
    stageControls.hidden = true;
    viewerCaption.textContent = "No path was found for this bar: the scene of its turn is shown with the robot "
      + "at its zero posture.";
    viewer.pose(new Array(12).fill(0));
    return;
  }
  const slider = element("input", { type: "range", min: "0", max: String(positions.length - 1), step: "1",
                                    "data-path-slider": "" });
  slider.addEventListener("input", () => { stopPlaying(); showStep(Number(slider.value)); });
  const play = element("button", { "data-play": "", title: "Space plays and pauses; the arrow keys step" }, "Play");
  play.addEventListener("click", () => {
    if (state.playing) { stopPlaying(); return; }
    if (state.step >= positions.length - 1) showStep(0);
    play.textContent = "Pause";
    state.playing = setInterval(() => {
      if (state.step >= positions.length - 1) stopPlaying();
      else showStep(state.step + 1);
    }, 50);
  });
  const jump = (label, to) => element("button", { onclick: () => { stopPlaying(); showStep(to()); } }, label);
  stageControls.replaceChildren(element("div", { class: "controls" }, [
    play, jump("Start", () => 0), jump("‹", () => state.step - 1), slider, jump("›", () => state.step + 1),
    jump("End", () => positions.length - 1),
  ]));
  stageControls.hidden = false;
  showStep(0);
}

// ---------------------------------------------------------------------------
// * One bar: the panels
// ---------------------------------------------------------------------------

function planPanel(row, data) {
  const check = data.check || {};
  const prm = data.prm || {};
  const start = prm.start || {};
  const lines = [];
  if (data.first_try) {
    lines.push(`The roadmap found no path for this bar (${data.first_try.failure_reason} after ${seconds(data.first_try.plan_s)}`
      + (data.first_try.best_gap_left_mm != null ? `; its best route broke the holding by ${data.first_try.best_gap_left_mm.toFixed(0)} mm` : "")
      + `). The path shown is the ${plannerName(row.backend)}'s, which derives its own start next to the goal `
      + "instead of starting from one of the roadmap's carry poses; its end posture was checked for the insert "
      + "and the retreat after planning.");
  } else if (row.ok) {
    lines.push(`Route of ${row.route_nodes} nodes within ${number(prm.tolerance_mm, 0, " mm")} of holding the bar; `
      + `${prm.n_waypoints_1cm ?? "?"} waypoints at 1 cm, ${row.n_waypoints} after smoothing and rounding.`);
    lines.push(`Loading pose (where the carry starts): ${loadingPose(row, true)}.`);
    const dense = prm.dense_check;
    if (dense) lines.push(`Dense check between the waypoints (straight joint moves, ${dense.step_deg}° per joint, `
      + `${dense.n_points} points): the ${dense.used} path is free`
      + (dense.failed.length ? `; ${dense.failed.map((f) => f.path).join(" and ")} collided and was set aside.` : "."));
  } else {
    lines.push(`No path: ${row.failure_reason}.`
      + (prm.best_gap_left_mm != null ? ` The best route left breaks the holding by ${prm.best_gap_left_mm.toFixed(1)} mm.` : "")
      + ((prm.goals_rejected || []).length ? ` ${prm.goals_rejected.length} end posture(s) were refused.` : ""));
    const strict = (data.plan || {}).strict_contacts || {};
    const pairs = (list) => (list || []).map((pair) => pair.join(" and ")).join("; ");
    if (strict.goal_hits) lines.push(`At the carry's end pose these parts overlap: ${pairs(strict.goal_hits)}. `
      + "They belong to the mate the bar is about to be joined to, which a carry may not touch: the insert's offset "
      + "in the export is too small for this bar.");
    for (const failed of (prm.dense_check || {}).failed || []) {
      lines.push(`Dense check, ${failed.path} path: collision after waypoint ${failed.segment + 1} `
        + `(${pairs(failed.pairs)}).`);
    }
  }
  const starts = prm.starts || {};
  if (row.backend === "prm") lines.push(`Carry poses free in this scene: ${starts.free_in_scene ?? "—"} of ${starts.carry_nodes ?? "—"}. `
    + `Edges looked at: ${prm.edges_looked_at ?? "—"} (${Object.entries(prm.edge_verdicts || {}).map(([k, v]) => `${v} ${k}`).join(", ") || "none"}); `
    + `blocked by the scene: ${prm.scene_blocked_edges ?? "—"}; taken from the precomputed tracking: ${prm.pretracked_edges ?? 0} on file.`);
  const drive = data.drive_in;
  if (row.backend === "prm" && !data.first_try) {
    const deg = (v) => (v * 180 / Math.PI).toFixed(0);
    lines.push(!drive ? "Drive-in: not checked for this plan (no station given)."
      : !drive.points ? `Drive-in from the station at [${drive.station[0].toFixed(2)}, ${drive.station[1].toFixed(2)}] m, `
        + `${deg(drive.station[2])}°: no drive path is shown, as no carry was planned. `
        + (drive.accepted === undefined ? "" : `${drive.accepted} loading posture(s) can drive in; `)
        + `${drive.refused.length} refused`
        + (drive.refused.length ? ` (${drive.refused.filter((r) => r.timed_out).length} of them because the search `
          + `ran out of time, which is not a proof)` : "")
        + ` in ${seconds(drive.seconds)}.`
        : `Drive-in: with the arms in this loading posture the base can drive from the station at `
          + `[${drive.station[0].toFixed(2)}, ${drive.station[1].toFixed(2)}] m, ${deg(drive.station[2])}° to this bar's `
          + `base pose: ${drive.direct ? "a straight line" : "a planned path"} of ${number(drive.length_m, 2, " m")}, `
          + `kept ${number(drive.margin * 100, 0, " cm")} from everything except at its two ends `
          + `(${drive.asks} posture(s) asked in ${seconds(drive.seconds)}).`);
  }
  const rejected = prm.starts_rejected || [];
  const rejectedTable = !rejected.length ? [] : [
    element("h3", {}, `Loading postures refused by the drive-in check: ${rejected.length}`),
    table(["class", "pose", "from", "why", "other postures of the pose dropped with it"],
          rejected.map((r) => [r.carry || "—", r.label || `node ${r.node}`, r.source || "—", r.reason,
                               r.dropped_with_it || 0])),
  ];
  const stages = (prm.walk_in || {}).stages || [];
  const stageTable = !stages.length ? [] : [
    element("p", { class: "muted" }, "The precomputed carry poses gave no route, so the fallbacks ran. Each walk "
      + "moves the bar on a straight line from a goal posture with both arms following, so what it reaches is on "
      + "that posture's own arm configuration."),
    table(["stage", "poses tried", "walks", "arrive collision-free", "free all the way", "refused: cannot drive in",
           "starts added", "waypoints kept"],
          stages.map((s) => [s.stage, s.poses ?? "—", s.walks ?? "—", s.arrived_free ?? "—",
                             s.free_all_the_way ?? "—", s.refused_drive ?? "—", s.starts ?? s.starts_reached ?? "—",
                             s.waypoints ?? "—"])),
  ];
  return panel(`${row.bar}: bar ${row.order} of ${state.index.bars.length}, ${row.built_before} built before it`, [
    tiles([
      ["result", row.ok ? (check.ok === false ? "check failed" : "path found") : "no path"],
      ["planning time", seconds(row.plan_s)],
      ["waypoints", row.ok ? row.n_waypoints : "—"],
      ["bar travel", check.bar_travel_m != null ? `${check.bar_travel_m.toFixed(2)} m` : "—"],
      ["bar turn", number(check.bar_turn_deg, 0, "°")],
      ["lowest point above the floor", check.lowest_z_above_base_frame_m != null
        ? number(check.lowest_z_above_base_frame_m * 1000, 0, " mm") : "—"],
    ]),
    ...lines.map((line) => element("p", { class: "muted" }, line)),
    ...rejectedTable,
    ...stageTable,
    element("p", {}, [element("a", { href: data.run_page }, "This bar's roadmap page"),
                      " (the roadmap's nodes and edges, and this plan's route on it)."]),
  ]);
}

// The largest value of one column of rows, for the sentences over the path charts.
const largest = (rows, column) => rows.reduce((most, row) => Math.max(most, row[column]), 0);

function pathPanel(data) {
  if (!data.positions.length || !state.held.at.length) return null;
  const { at, between, perStep } = state.held;
  const rotation = (rows, first) => Math.max(...[0, 1, 2].map((axis) => largest(rows, first + axis)));
  const gap = "margin-top:16px";
  return panel("Along the path", [
    element("p", { class: "muted" }, "The yellow line is the waypoint on show in the 3D view and moves with the "
      + "slider; click in a chart to jump there. Dots are the waypoints of the saved path."),
    element("h3", { style: gap }, "Joint values"),
    element("div", { class: "chart", "data-chart-joints": "" }),
    element("h3", { style: gap }, "Holding the bar: position against the first waypoint"),
    element("p", { class: "muted" }, "Where the right flange is as seen from the left one, and how far that is from "
      + `the first waypoint. Largest at a waypoint: ${largest(at, 0).toFixed(3)} mm. The line is the largest `
      + `between each waypoint and the next, with the joints interpolated linearly (${perStep} points per step): `
      + `up to ${largest(between, 1).toFixed(3)} mm.`),
    element("div", { class: "chart", "data-chart-held-position": "" }),
    element("h3", { style: gap }, "Holding the bar: rotation against the first waypoint, per axis"),
    element("p", { class: "muted" }, "The angle between each axis of the right flange's frame, as seen from the "
      + `left flange, and the same axis at the first waypoint. Largest at a waypoint: ${rotation(at, 1).toFixed(4)}°; `
      + `the lines are the largest between each waypoint and the next: up to ${rotation(between, 2).toFixed(4)}°.`),
    element("div", { class: "chart", "data-chart-held-rotation": "" }),
  ]);
}

// Draws the three charts of the path panel; they share the x axis (waypoint index) and the cursor.
function drawPathCharts() {
  state.pathCharts = [];
  const jointsHolder = main.querySelector("[data-chart-joints]");
  if (!jointsHolder) return;
  const positions = state.data.positions;
  const { at, between } = state.held;
  const names = state.data.joint_names || [];
  const onCursor = (x) => { stopPlaying(); showStep(Math.round(x)); };
  const shared = { xLabel: "waypoint", cursor: state.step, onCursor };
  state.pathCharts.push(lineChart(jointsHolder, {
    series: JOINT_COLOURS.map((colour, j) => ({
      points: positions.map((conf, k) => [k, conf[j] * 180 / Math.PI]), colour, width: 1.2, dots: 1.8,
      label: (names[j] || `joint ${j}`).replace("_ur_arm_", " ").replace("_joint", "").replace(/_/g, " "),
    })),
    yLabel: "joint value (deg)", height: 320, yTickCount: 5, yFormat: (v) => v.toFixed(0), ...shared,
  }));
  state.pathCharts.push(lineChart(main.querySelector("[data-chart-held-position]"), {
    series: [
      { points: between.map((row) => [row[0], row[1]]), colour: "#1f84bd", width: 1.2,
        label: "largest between a waypoint and the next" },
      { points: at.map((row, k) => [k, row[0]]), colour: "#b23a48", width: 0, dots: 2.2, label: "at the waypoints" },
    ],
    yLabel: "position error (mm)", height: 200, yFormat: (v) => v.toFixed(3), ...shared,
  }));
  state.pathCharts.push(lineChart(main.querySelector("[data-chart-held-rotation]"), {
    series: ["x", "y", "z"].map((axis, i) => ({
      points: between.map((row) => [row[0], row[2 + i]]), colour: AXIS_COLOURS[i], width: 1.2,
      label: `${axis} axis`,
    })).concat(["x", "y", "z"].map((axis, i) => ({
      points: at.map((row, k) => [k, row[1 + i]]), colour: AXIS_COLOURS[i], width: 0, dots: 2.2,
    }))),
    yLabel: "rotation error (deg)", height: 200, yFormat: (v) => v.toFixed(3), ...shared,
  }));
}

// The rows of the two time charts: this plan's phases, and the roadmap's precompute.
function planRows(row, data) {
  const timings = (data.prm || {}).timings || {};
  const rows = PHASES.filter(([key]) => timings[key] != null)
    .map(([key, label, group]) => ({ label, value: timings[key], colour: GROUP_COLOURS[group] }));
  const round = ((data.plan || {}).post || {}).round;
  if (round && round.wall_s != null) rows.push({ label: "of which corner rounding", value: round.wall_s, colour: "#d9a0a8" });
  rows.push({ label: "rest of the call (scene sync, saving)", value: groupSeconds(row).other, colour: GROUP_COLOURS.other });
  rows.push({ label: "whole planning call (wall time)", value: row.plan_s || 0, colour: "#33383b" });
  return rows;
}

function buildRows(data) {
  const pre = data.precompute || {};
  const rows = Object.entries(pre.build_timings || {}).filter(([key, value]) => key !== "total_s" && value != null)
    .map(([key, value]) => ({ label: BUILD_PHASES[key] || key, value, colour: "#8a8f96" }));
  if ((pre.build_timings || {}).total_s != null) {
    rows.push({ label: "whole build (wall time)", value: pre.build_timings.total_s, colour: "#33383b" });
  }
  for (const t of pre.tracking || []) {
    rows.push({ label: `track every edge up to ${t.max_score_mm} mm (${t.n_records} edges)`, value: t.wall_s, colour: "#8a6fb0" });
  }
  return rows;
}

function timingPanel(row, data) {
  const pre = data.precompute || {};
  return panel("Computation time", [
    element("h3", {}, row.backend === "prm" ? `This plan (${row.workers ?? "?"} worker processes)`
      : `This plan (${plannerName(row.backend)}, one process)`),
    element("p", { class: "muted" }, row.backend === "prm"
      ? "What the operator would wait for. Tracking an edge solves the arms' exact path "
        + "along it so the bar stays held, then checks it against the scene; collision checks and IK run in the "
        + "worker processes, the smoothing and rounding in one."
      : "What the operator would wait for. The search records no phases; only the corner rounding is timed apart."),
    element("div", { class: "chart", "data-chart-plan": "" }),
    element("h3", {}, `Precompute, once per bar (${(pre.settings || {}).workers ?? "?"} worker processes)`),
    element("p", { class: "muted" }, `The roadmap: ${pre.n_nodes ?? "—"} nodes, ${pre.n_edges ?? "—"} edges `
      + `(${pre.n_edges_free ?? "—"} collision-free), built with only the robot, its tools, the carried bar and the `
      + "floor. Not part of the planning time above."),
    element("div", { class: "chart", "data-chart-build": "" }),
  ]);
}

function countersPanel(data) {
  const counters = (data.prm || {}).counters || {};
  const pick = (name, key) => (counters[name] || {})[key];
  const rows = [
    ["end postures at the goal", pick("endpoints", "ik_calls"), pick("endpoints", "collision_checks"), pick("endpoints", "seconds_worker")],
    ["carry poses checked in the scene", null, pick("nodes", "scene_checks"), pick("nodes", "seconds_worker")],
    ["edges tracked", pick("track", "ik_calls"), pick("track", "collision_checks"), pick("track", "seconds_worker")],
    ["new nodes drawn around the best route", pick("tube_sample", "ik_calls"), pick("tube_sample", "collision_checks"), pick("tube_sample", "seconds_worker")],
    ["fallback walks from the goal postures", pick("walk", "ik_calls"), pick("walk", "collision_checks"), pick("walk", "seconds_worker")],
    ["final check", pick("final", "ik_calls"), pick("final", "collision_checks"), pick("final", "seconds_worker")],
  ].filter((row) => row.slice(1).some((v) => v != null));
  const track = counters.track || {};
  return panel("Work done", [
    element("p", { class: "muted" }, "IK calls and collision checks by phase, and the processor time the workers "
      + "spent on them (summed over the workers, so it can exceed the wall time above)."
      + (track.edges_tracked != null ? ` Edges tracked: ${track.edges_tracked} (${track.track_ok ?? 0} fine, `
        + `${track.track_collision ?? 0} in collision, ${track.track_continuity ?? 0} with a joint jump).` : "")),
    table(["phase", "IK calls", "collision checks", "worker seconds"],
          rows.map(([label, ik, cc, s]) => [label, ik ?? "—", cc ?? "—", s != null ? s.toFixed(1) : "—"])),
  ]);
}

function followPanel(data) {
  const follow = (data.plan || {}).follow_on;
  if (!follow) return null;
  const children = [];
  if (follow.end_joints) {
    children.push(element("p", { class: "muted" }, (follow.end_is_authored_posture
      ? "The path ends in the authored arm posture"
      : "The path ends in another arm posture than the authored one, at the same tool poses")
      + ` (largest joint ${number(follow.end_off_authored_deg, 1, "°")} from the authored value). `
      + (follow.end_refused ? `! The returned end posture was refused: ${follow.end_refused}.`
        : `${follow.insert} and ${follow.retreat} were planned from it as straight moves with the scene's `
          + "collisions and the joint-step limit; the joints they end in are below.")));
    const names = Object.keys(follow.end_joints);
    const deg = (v) => (v == null ? "—" : (v * 180 / Math.PI).toFixed(1));
    children.push(table(["joint", "carry ends (°)", "after the insert (°)", "after the retreat (°)"],
      names.map((name) => [name.replace("_ur_arm", "").replace("_joint", ""), deg(follow.end_joints[name]),
                           deg((follow.insert_joints || {})[name]), deg((follow.retreat_joints || {})[name])])));
  }
  const rejected = follow.rejected || [];
  if (rejected.length) {
    children.push(element("h3", {}, `End postures refused before this one: ${rejected.length}`));
    children.push(table(["roadmap node", "why"], rejected.map((r) => [r.node, r.reason])));
  }
  children.push(element("p", { class: "muted" }, `Time spent on these checks: ${seconds(follow.seconds)}.`));
  return panel("End posture, and what follows it", children);
}

function checkPanel(data) {
  const check = data.check;
  if (!check || check.reason === "file not found") return null;
  const rows = [
    ["verdict", check.ok ? "ok" : `FAILED: ${checkFailure(check)}`],
    ["waypoints of the saved path in collision", check.saved_waypoints_colliding
      + (check.saved_first_hit ? ` (first: waypoint ${check.saved_first_hit.waypoint + 1}, `
        + `${(check.saved_first_hit.pairs || []).map((pair) => pair.join(" and ")).join("; ")})` : "")],
    ["between the waypoints (straight joint moves, 0.1° per joint)", check.dense_ok == null ? "not checked"
      : check.dense_ok ? `free at all ${check.dense_points} points`
        : `collision after waypoint ${check.dense_first_hit.segment + 1}: `
          + `${(check.dense_first_hit.pairs || []).map((pair) => pair.join(" and ")).join("; ")}`],
    ["largest joint step between waypoints", number(check.saved_max_step_deg, 2, "°")],
    ["holding error between waypoints", `${number(check.holding_between_pos_mm, 3, " mm")}, ${number(check.holding_between_rot_deg, 4, "°")}`],
    ["re-solved at 2 mm with the production repair", check.repair_success ? `ok (${seconds(check.repair_s)})` : `failed: ${check.repair_reason}`],
    ["lowest point", `${number((check.lowest_z_above_base_frame_m ?? NaN) * 1000, 0, " mm")} above the floor, `
      + `${check.lowest_body || "?"} at waypoint ${check.lowest_at_waypoint ?? "?"}`],
    ["joint travel, reversals", `${number(check.joint_travel_deg, 0, "°")}, ${check.joint_reversals ?? "—"}`],
  ];
  return panel("The saved path, checked on its own", [
    element("p", { class: "muted" }, "prm_check_solved.py reads the saved file back and measures it without the "
      + "planner. Its collision checks are strict: nothing the arms carry or wear may touch a standing body, also "
      + "not the receiver the bar is about to be joined to."),
    table(["what", "found"], rows),
  ]);
}

async function renderBar(barId) {
  const bars = state.index.bars;
  const at = bars.findIndex((b) => b.bar === barId);
  const row = bars[at];
  if (!row || !row.planned) {
    main.replaceChildren(panel(barId, [element("p", {}, "No plan was saved for this bar.")]));
    return;
  }
  state.data = await loadJson(row.file);
  const data = state.data;
  await showBarScene();
  // The bars before and after this one in build order (the process's schedule), with their place in it.
  const link = (offset, text) => {
    const other = bars[at + offset];
    return other ? element("a", { href: `#/bar/${other.bar}` }, text(`${other.order}. ${other.bar}`)) : null;
  };
  const nav = element("p", {}, [link(-1, (name) => `‹ ${name}`), "  ·  ", element("a", { href: "#/" }, "every bar"),
                                "  ·  ", link(1, (name) => `${name} ›`),
                                element("span", { class: "muted" }, "   build order, from the process's schedule")]);
  main.replaceChildren(...[nav, planPanel(row, data), pathPanel(data), timingPanel(row, data), countersPanel(data),
                           followPanel(data), checkPanel(data)].filter(Boolean));
  drawPathCharts();
  const format = { format: seconds, xLabel: "seconds (wall time)", labelWidth: 380 };
  horizontalBarChart(main.querySelector("[data-chart-plan]"), { data: planRows(row, data), ...format });
  horizontalBarChart(main.querySelector("[data-chart-build]"), { data: buildRows(data), ...format });
  renderStageControls();
}

// ---------------------------------------------------------------------------
// * Routing
// ---------------------------------------------------------------------------

function renderBreadcrumb(barId) {
  const home = element("a", { href: "#/" }, `${state.index.problem} · ${state.index.set} · ${state.index.tag}`);
  const picker = element("select", { class: "run-picker" }, [
    element("option", { value: "" }, "every bar"),
    ...state.index.bars.map((b) => element("option", { value: b.bar },
      `${b.order}. ${b.bar}: ${b.planned ? (b.ok ? `path, ${seconds(b.plan_s)}` : "no path") : "not planned"}`)),
  ]);
  picker.value = barId || "";
  picker.addEventListener("change", () => { location.hash = picker.value ? `#/bar/${picker.value}` : "#/"; });
  breadcrumb.replaceChildren(home, element("span", { class: "sep" }, "›"), picker);
}

async function route() {
  stopPlaying();
  const match = /^#\/bar\/([^/]+)/.exec(location.hash);
  const barId = match ? decodeURIComponent(match[1]) : null;
  renderBreadcrumb(barId);
  // The front page has no robot to show: it takes the full width.
  document.querySelector("[data-layout]").classList.toggle("split", Boolean(barId));
  document.querySelector(".stage").hidden = !barId;
  try {
    if (barId) await renderBar(barId);
    else renderOverview();
  } catch (error) {
    main.replaceChildren(panel("Something went wrong", [element("p", {}, String(error && error.message || error))]));
    console.error(error);
  }
}

async function start() {
  state.index = await loadJson("plans.json");
  for (const node of document.querySelectorAll("[data-title]")) {
    node.textContent = `Bar carry plans: ${state.index.problem}`;
  }
  window.addEventListener("hashchange", route);
  window.addEventListener("keydown", (event) => {
    if (!state.data || !state.data.positions || !state.data.positions.length || !/^#\/bar\//.test(location.hash)) return;
    const tag = event.target ? event.target.tagName : "";
    if (event.key === " ") {
      // Space plays and pauses wherever the focus is, except in the bar picker, where it opens the list.
      if (tag === "SELECT") return;
      event.preventDefault();
      if (!event.repeat) stageControls.querySelector("[data-play]").click();
      return;
    }
    if (/INPUT|SELECT/.test(tag)) return;
    if (event.key === "ArrowRight") { stopPlaying(); showStep(state.step + 1); }
    if (event.key === "ArrowLeft") { stopPlaying(); showStep(state.step - 1); }
  });
  // Releasing Space would also press a focused button: on a bar's page Space has already done its work.
  window.addEventListener("keyup", (event) => {
    const onButton = event.target && event.target.tagName === "BUTTON";
    if (event.key === " " && onButton && /^#\/bar\//.test(location.hash)) event.preventDefault();
  });
  await route();
}

start();
