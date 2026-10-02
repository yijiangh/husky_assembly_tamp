// The roadmap pilot's page: one roadmap, and the queries run against it.
//
// Each roadmap node is drawn as a small ball at the midpoint between the two
// tool flanges -- a stand-in for "where the bar is" that is the same for
// every node, whatever the bar's orientation. Clicking a ball poses the robot
// at that node's twelve joints and lists its holding error and its edges.
//
// Edges are the straight lines in joint space the planner may follow. Their
// colour says how far the bar-holding constraint is broken along them,
// measured against the tolerance slider under the 3D view: green inside the
// tolerance, amber up to three times it, red beyond.
//
// With a query loaded, every node also knows its "gap": the smallest
// tolerance at which it can reach the goal set, and the next node on that
// best route. Clicking any node therefore also shows its best route to the
// goals, which the stepper under the 3D view walks through.
//
// Nothing in the shared dashboard files is changed: this page only imports
// them. The robot viewer's click handling is reused by registering the balls
// in its list of pickable clouds.
//
//   #/                       the roadmap alone
//   #/query/<tag>            one query
//   #/query/<tag>/node/<id>  one query, one node selected

import * as THREE from "three";
import { loadJson, loadJsonGz } from "./data.js";
import { lineChart } from "./charts.js";
import { RobotViewer, poseToMatrix } from "./robot-viewer.js";

const main = document.querySelector("[data-main]");
const breadcrumb = document.querySelector("[data-breadcrumb]");
const viewerHost = document.querySelector("[data-viewer]");
const viewerCaption = document.querySelector("[data-viewer-caption]");
const stageControls = document.querySelector("[data-stage-controls]");

// * Colours. Node kinds, and the three bands of the tolerance scale.
const KIND_COLOURS = {
  sampled: 0x8a8f96,
  anchor: 0x8a6fb0,
  start: 0x2e9e5b,
  goal: 0x1f84bd,
  selected: 0xf2c14e,
};
const BAND_COLOURS = { ok: 0x2e9e5b, near: 0xd98324, far: 0xb23a48, unknown: 0xb8bcc2, blocked: 0x5a5a6e };
const BALL_RADIUS = { sampled: 0.008, anchor: 0.012, start: 0.011, goal: 0.018 };

// * Page state. Everything is rebuilt from these on each change.
const state = {
  info: null,        // prm.json
  roadmap: null,     // roadmap.json.gz
  query: null,       // the loaded query, or null
  queryTag: null,
  nodes: null,       // combined node table (roadmap + query endpoints)
  edges: null,       // combined edge table
  adjacency: null,   // node id -> edge rows
  tolerance: 5,      // mm
  colourMode: "kind",
  shown: { sampled: true, anchor: true, start: true, goal: true, ok: true, near: true, far: false,
           unknown: false, blocked: false },
  selected: null,    // node id
  path: null,        // { name, nodes, dense, profile }
  step: 0,
};

let viewer = null;
let robotReady = null;
let drawn = { clouds: [], lines: [] };
let rightBar = null;   // the bar as the RIGHT flange places it (see updateRightBar)
let grasps = null;     // the two grasps as matrices, and their inverses

// ---------------------------------------------------------------------------
// * Small helpers for building the page
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

function table(headers, rows, { onRowClick = null, highlight = [] } = {}) {
  const head = element("tr", {}, headers.map((h) => element("th", {}, h)));
  const body = rows.map((row, index) => {
    const tr = element("tr", {}, row.map((cell) => element("td", {}, cell instanceof Node ? cell : String(cell))));
    if (onRowClick) {
      tr.className = "clickable";
      tr.addEventListener("click", () => onRowClick(index));
    }
    if (highlight[index]) tr.classList.add("over");
    return tr;
  });
  return element("table", { class: "grid" }, [element("thead", {}, [head]), element("tbody", {}, body)]);
}

function panel(title, children) {
  return element("section", { class: "panel" }, [element("h2", {}, title), ...children]);
}

const degrees = (rad) => `${(rad * 180 / Math.PI).toFixed(1)}°`;
const mm = (value) => (value == null || !Number.isFinite(value) ? "—" : `${value.toFixed(2)} mm`);
const cssColour = (hex) => `#${hex.toString(16).padStart(6, "0")}`;

function shortJointName(index) {
  const name = (state.info.problem.joint_names_12 || [])[index];
  return name ? name.replace("_ur_arm_", " ").replace("_joint", "").replace(/_/g, " ") : `joint ${index}`;
}

// The violation score of an edge or a point: position error in mm, or
// rotation error in degrees divided by the degrees-per-mm ratio, whichever
// is larger. "score <= t" means "within tolerance t".
function score(posM, rotRad) {
  return Math.max(posM * 1000, (rotRad * 180 / Math.PI) / state.info.rot_deg_per_mm);
}

function band(value) {
  if (value == null || !Number.isFinite(value)) return "far";
  if (value <= state.tolerance) return "ok";
  if (value <= 3 * state.tolerance) return "near";
  return "far";
}

// ---------------------------------------------------------------------------
// * Loading and joining the data
// ---------------------------------------------------------------------------

// One node table for the roadmap plus the query's own endpoint nodes.
function joinNodes() {
  const r = state.roadmap.nodes;
  const q = state.query ? state.query.endpoint_nodes : null;
  const columns = ["conf", "ee_mid", "tool_left", "tool_right", "viol_pos", "viol_rot", "kind", "mode",
                   "driver", "anchor"];
  const nodes = {};
  for (const name of columns) nodes[name] = (r[name] || []).concat(q ? (q[name] || []) : []);
  const n = nodes.conf.length;
  // Endpoint roles: which starts and goals each node stands for.
  nodes.roles = Array.from({ length: n }, () => []);
  if (state.query) {
    const e = state.query.endpoint;
    for (let row = 0; row < e.id.length; row += 1) {
      nodes.roles[e.id[row]].push({
        role: e.role[row], start_index: e.start_index[row], carry: e.carry[row],
        difficulty: e.difficulty[row], pair: e.pair[row], turns: e.turns[row],
        library_conf: e.library_conf[row],
      });
    }
  }
  nodes.category = nodes.kind.map((kind, id) => {
    const roles = nodes.roles[id];
    if (roles.some((x) => x.role === "goal")) return "goal";
    if (roles.some((x) => x.role === "start")) return "start";
    return kind === "anchor" ? "anchor" : "sampled";
  });
  nodes.gap = state.query ? state.query.node_gap_mm : null;
  nodes.parent = state.query ? state.query.node_parent : null;
  return nodes;
}

// One edge table, with the final collision verdicts of the query when there is one.
function joinEdges() {
  const r = state.roadmap.edges;
  const q = state.query ? state.query.endpoint_edges : null;
  const edges = {};
  for (const name of ["i", "j", "len_l2", "len_linf", "viol_pos_max", "viol_rot_max", "t_viol_max", "t_collide",
                      "free", "checked"]) {
    edges[name] = (r[name] || []).concat(q ? (q[name] || []) : []);
  }
  if (state.query && state.query.edge_free) {
    edges.free = state.query.edge_free;
    edges.checked = state.query.edge_checked;
  }
  edges.score = edges.i.map((_, row) => score(edges.viol_pos_max[row], edges.viol_rot_max[row]));
  return edges;
}

function buildAdjacency(n, edges) {
  const adjacency = Array.from({ length: n }, () => []);
  for (let row = 0; row < edges.i.length; row += 1) {
    adjacency[edges.i[row]].push(row);
    adjacency[edges.j[row]].push(row);
  }
  return adjacency;
}

async function loadQuery(tag) {
  // ! Only skip when the tables are already built for this choice. On the first
  // ! visit both the requested and the loaded query are "none", and the node and
  // ! edge tables still have to be made.
  if (state.nodes && tag === state.queryTag) return;
  if (!tag) {
    state.query = null;
    state.queryTag = null;
  } else {
    const entry = state.info.queries.find((q) => q.tag === tag);
    if (!entry) throw new Error(`no query called ${tag} in this run`);
    state.query = await loadJsonGz(entry.file);
    state.queryTag = tag;
  }
  state.nodes = joinNodes();
  state.edges = joinEdges();
  state.adjacency = buildAdjacency(state.nodes.conf.length, state.edges);
  state.selected = null;
  state.path = null;
  if (state.query) state.colourMode = "gap";
}

// ---------------------------------------------------------------------------
// * The 3D view
// ---------------------------------------------------------------------------

async function ensureRobot() {
  if (!viewer) viewer = new RobotViewer(viewerHost);
  if (!robotReady) {
    robotReady = viewer.loadRobot(`${state.info.assets}/scene.glb`, `${state.info.assets}/kinematics.json`);
  }
  await robotReady;
  const problem = state.info.problem;
  if (!grasps) {
    const left = poseToMatrix(problem.grasp_bar_from_left);
    const right = poseToMatrix(problem.grasp_bar_from_right);
    grasps = { left, right, leftInv: left.clone().invert(), rightInv: right.clone().invert() };
    viewer.setCarriedBar({
      length: problem.bar_length_m, radius: problem.bar_radius_m, linkName: problem.attached_link,
      grasp: problem.grasp_bar_from_left, bar_from_midpoint: problem.bar_from_midpoint,
    });
    // A second, see-through bar placed by the RIGHT flange. When the robot
    // holds the bar exactly the two bars coincide; along an edge that breaks
    // the constraint they visibly come apart.
    const geometry = new THREE.CylinderGeometry(problem.bar_radius_m * 1.3, problem.bar_radius_m * 1.3,
                                                problem.bar_length_m, 12, 1);
    geometry.rotateX(Math.PI / 2);
    const mid = problem.bar_from_midpoint || [0, 0, problem.bar_length_m / 2];
    geometry.translate(mid[0], mid[1], mid[2]);
    rightBar = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
      color: 0xd98324, transparent: true, opacity: 0.45, depthWrite: false }));
    rightBar.matrixAutoUpdate = false;
    viewer.scene.add(rightBar);
  }
  return viewer;
}

function toolLinks() {
  const links = (viewer.kinematicsSpec && viewer.kinematicsSpec.tool_links) || {};
  return { left: links.left || "left_ur_arm_tool0", right: links.right || "right_ur_arm_tool0" };
}

// Poses the robot and returns its holding error there, computed from the
// posed flanges exactly as the planner computes it (the larger of the two
// "where the other flange should be" gaps).
function poseAndMeasure(conf) {
  viewer.pose(conf);
  const links = toolLinks();
  const left = viewer.kinematics.linkMatrix(links.left);
  const right = viewer.kinematics.linkMatrix(links.right);
  if (!left || !right) return null;
  if (rightBar) rightBar.matrix.multiplyMatrices(right, grasps.rightInv);
  const expectedRight = left.clone().multiply(grasps.leftInv).multiply(grasps.right);
  const expectedLeft = right.clone().multiply(grasps.rightInv).multiply(grasps.left);
  const gap = (a, b) => {
    const pa = new THREE.Vector3().setFromMatrixPosition(a);
    const pb = new THREE.Vector3().setFromMatrixPosition(b);
    // The relative rotation's angle from both its sine and cosine: three.js's
    // angleTo uses an arc-cosine, which is too coarse for nearly equal frames.
    const qa = new THREE.Quaternion().setFromRotationMatrix(a);
    const qb = new THREE.Quaternion().setFromRotationMatrix(b);
    const q = qa.invert().multiply(qb);
    const rot = 2 * Math.atan2(Math.hypot(q.x, q.y, q.z), Math.abs(q.w));
    return { pos: pa.distanceTo(pb), rot };
  };
  const a = gap(expectedRight, right);
  const b = gap(expectedLeft, left);
  return { pos: Math.max(a.pos, b.pos), rot: Math.max(a.rot, b.rot) };
}

function showPose(conf, what) {
  const error = poseAndMeasure(conf);
  viewerCaption.textContent = `Robot at ${what}.` + (error
    ? ` Holding error here: ${(error.pos * 1000).toFixed(2)} mm, ${degrees(error.rot)}`
      + ` (score ${score(error.pos, error.rot).toFixed(2)} mm). Orange bar: where the right flange puts the bar.`
    : "");
}

// A cloud of clickable balls. The mesh goes into the viewer's own list of
// pickable clouds, so its existing click handler reports which ball was hit.
function addBallCloud(points, colours, ids, radius) {
  if (!points.length) return null;
  const mesh = new THREE.InstancedMesh(new THREE.SphereGeometry(radius, 10, 8),
    new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.0 }), points.length);
  const matrix = new THREE.Matrix4();
  const colour = new THREE.Color();
  points.forEach((p, index) => {
    matrix.makeTranslation(p[0], p[1], p[2]);
    mesh.setMatrixAt(index, matrix);
    mesh.setColorAt(index, colour.setHex(colours[index]));
  });
  mesh.instanceMatrix.needsUpdate = true;
  mesh.instanceColor.needsUpdate = true;
  mesh.userData.ids = ids;
  viewer.scene.add(mesh);
  viewer.poseClouds.push(mesh);
  return mesh;
}

// Many coloured segments as one object (one colour per segment end).
function addSegments(segments, colours, opacity) {
  if (!segments.length) return null;
  const positions = new Float32Array(segments.length * 6);
  const vertexColours = new Float32Array(segments.length * 6);
  const colour = new THREE.Color();
  segments.forEach(([p, q], index) => {
    positions.set([p[0], p[1], p[2], q[0], q[1], q[2]], index * 6);
    colour.setHex(colours[index]);
    vertexColours.set([colour.r, colour.g, colour.b, colour.r, colour.g, colour.b], index * 6);
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(vertexColours, 3));
  const lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: opacity < 1, opacity }));
  viewer.scene.add(lines);
  viewer.overlays.push(lines);
  return lines;
}

function nodeColour(id) {
  const category = state.nodes.category[id];
  if (state.colourMode === "gap" && state.nodes.gap) {
    const gap = state.nodes.gap[id];
    return gap == null ? BAND_COLOURS.unknown : BAND_COLOURS[band(gap)];
  }
  return KIND_COLOURS[category];
}

function edgeBand(row) {
  const e = state.edges;
  if (!e.free[row]) return "blocked";
  if (!e.checked[row]) return "unknown";
  return band(e.score[row]);
}

function draw() {
  for (const cloud of drawn.clouds) viewer.removePoseCloud(cloud);
  for (const line of drawn.lines) viewer.removeOverlay(line);
  drawn = { clouds: [], lines: [] };
  const nodes = state.nodes;
  for (const category of ["sampled", "anchor", "start", "goal"]) {
    if (!state.shown[category]) continue;
    const ids = [];
    nodes.category.forEach((c, id) => { if (c === category) ids.push(id); });
    const cloud = addBallCloud(ids.map((id) => nodes.ee_mid[id]), ids.map(nodeColour),
      ids.map((id) => ({ kind: "node", index: id })), BALL_RADIUS[category]);
    if (cloud) {
      cloud.userData.nodeIds = ids;
      drawn.clouds.push(cloud);
    }
  }
  const e = state.edges;
  const segments = [];
  const colours = [];
  for (let row = 0; row < e.i.length; row += 1) {
    const which = edgeBand(row);
    if (!state.shown[which]) continue;
    segments.push([nodes.ee_mid[e.i[row]], nodes.ee_mid[e.j[row]]]);
    colours.push(BAND_COLOURS[which]);
  }
  const lines = addSegments(segments, colours, 0.55);
  if (lines) drawn.lines.push(lines);
  markSelected();
  drawPath();
}

// Paints the selected node's ball in the selection colour.
function markSelected() {
  if (state.selected == null) return;
  for (const cloud of drawn.clouds) {
    const index = cloud.userData.nodeIds.indexOf(state.selected);
    if (index >= 0) viewer.setInstanceColour(cloud, index, KIND_COLOURS.selected);
  }
}

// ---------------------------------------------------------------------------
// * Paths: stored query paths, a node's best route, a single edge
// ---------------------------------------------------------------------------

function interpolate(a, b, stepRad) {
  const largest = Math.max(...a.map((v, k) => Math.abs(b[k] - v)));
  const pieces = Math.max(1, Math.ceil(largest / stepRad - 1e-12));
  const out = [];
  for (let s = 0; s <= pieces; s += 1) out.push(a.map((v, k) => v + (b[k] - v) * (s / pieces)));
  return out;
}

// Sets the path on show: the dense joint path through the given nodes, and
// its holding error at every point (measured by posing the robot).
function setPath(name, nodeIds) {
  if (!nodeIds || nodeIds.length < 1) { state.path = null; drawPath(); renderPathPanel(); return; }
  const step = state.roadmap.config.edge_settings.viol_step_rad;
  const confs = state.nodes.conf;
  let dense = [confs[nodeIds[0]]];
  for (let k = 1; k < nodeIds.length; k += 1) dense = dense.concat(interpolate(confs[nodeIds[k - 1]], confs[nodeIds[k]], step).slice(1));
  const profile = dense.map((conf) => {
    const error = poseAndMeasure(conf);
    return error ? score(error.pos, error.rot) : 0;
  });
  state.path = { name, nodes: nodeIds, dense, profile };
  state.step = 0;
  drawPath();
  renderPathPanel();
  showStep(0);
}

function drawPath() {
  if (!viewer) return;
  if (!state.path) { viewer.setTrajectoryLine(null); return; }
  viewer.setTrajectoryLine(state.path.nodes.map((id) => state.nodes.ee_mid[id]));
}

function showStep(index) {
  if (!state.path) return;
  state.step = Math.max(0, Math.min(state.path.dense.length - 1, index));
  const slider = stageControls.querySelector("[data-path-slider]");
  if (slider) slider.value = String(state.step);
  const label = stageControls.querySelector("[data-path-label]");
  if (label) label.textContent = `${state.step + 1} / ${state.path.dense.length}`;
  showPose(state.path.dense[state.step], `${state.path.name}, point ${state.step + 1} of ${state.path.dense.length}`);
  const chart = main.querySelector("[data-path-chart]");
  if (chart) drawProfile(chart);
}

function drawProfile(container) {
  const p = state.path;
  lineChart(container, {
    series: [{ points: p.profile.map((v, k) => [k, v]), colour: "#b23a48" },
             { points: [[0, state.tolerance], [p.profile.length - 1, state.tolerance]], colour: "#2e9e5b", width: 1 }],
    yLabel: "holding error score (mm)", xLabel: "point along the path",
    cursor: state.step, onCursor: (x) => showStep(Math.round(x)), height: 190,
  });
}

// A node's best route to the goal set, following the query's tree.
function routeFrom(id) {
  const parent = state.nodes.parent;
  if (!parent) return null;
  const route = [id];
  while (parent[route[route.length - 1]] >= 0 && route.length < 100000) route.push(parent[route[route.length - 1]]);
  return route;
}

// ---------------------------------------------------------------------------
// * Selecting a node
// ---------------------------------------------------------------------------

function selectNode(id, { follow = true } = {}) {
  state.selected = id;
  draw();
  const conf = state.nodes.conf[id];
  showPose(conf, `node ${id}`);
  renderNodePanel();
  if (follow && state.query) {
    const gap = state.nodes.gap[id];
    if (gap != null) setPath(`best route from node ${id}`, routeFrom(id));
  }
  const tagPart = state.queryTag ? `#/query/${state.queryTag}` : "#";
  history.replaceState(null, "", `${tagPart}/node/${id}`);
}

function describeNode(id) {
  const nodes = state.nodes;
  const parts = [];
  const kind = nodes.kind[id];
  if (kind === "sampled") {
    parts.push(`Sampled node (${nodes.mode[id]}, the ${nodes.driver[id]} arm drove the draw).`);
  } else if (kind === "anchor") {
    const anchor = state.roadmap.anchors[nodes.anchor[id]] || {};
    const what = anchor.role === "goal" ? `goal ${anchor.goal_index}, branch pair ${anchor.pair_index}`
      : anchor.role === "start" ? `start ${anchor.start_index} (${anchor.carry}, ${anchor.difficulty}) of goal ${anchor.goal_index}`
        : "the home configuration";
    parts.push(`Anchor: ${what}, ${anchor.turn} turn.`);
  } else {
    parts.push("Endpoint node added by the query.");
  }
  for (const role of nodes.roles[id]) {
    parts.push(role.role === "goal"
      ? `Goal set: branch pair ${role.pair}, turn ${role.turns}${role.library_conf ? " (the library's primary goal configuration)" : ""}.`
      : `Start set: start ${role.start_index} (${role.carry}, ${role.difficulty}), branch pair ${role.pair}, `
        + `turn ${role.turns}${role.library_conf ? " (the library's own start configuration)" : ""}.`);
  }
  return parts;
}

function renderNodePanel() {
  const holder = main.querySelector("[data-node]");
  if (!holder) return;
  const id = state.selected;
  if (id == null) {
    holder.replaceChildren(panel("Node", [element("p", { class: "muted" },
      "Click a ball in the 3D view (or a row in the starts table) to see its configuration.")]));
    return;
  }
  const nodes = state.nodes;
  const conf = nodes.conf[id];
  const gap = nodes.gap ? nodes.gap[id] : null;
  const lines = describeNode(id).map((text) => element("p", {}, text));
  lines.push(element("p", {}, `Holding error at the node itself: ${(nodes.viol_pos[id] * 1000).toFixed(4)} mm, `
    + `${degrees(nodes.viol_rot[id])} (nodes hold the bar exactly by construction).`));
  if (state.query) {
    lines.push(element("p", {}, gap == null ? "This node cannot reach the goal set through free edges."
      : `Gap to the goal set: ${mm(gap)} -- the smallest tolerance at which a route of free edges joins it to a goal node.`));
  }
  const edgeRows = state.adjacency[id].slice().sort((a, b) => state.edges.score[a] - state.edges.score[b]);
  const e = state.edges;
  const other = (row) => (e.i[row] === id ? e.j[row] : e.i[row]);
  const edgeTable = table(["to node", "category", "largest move", "worst error", "collision", ""],
    edgeRows.map((row) => [
      other(row), state.nodes.category[other(row)], degrees(e.len_linf[row]), mm(e.score[row]),
      !e.checked[row] ? "not checked" : e.free[row] ? "free" : `blocked at ${(e.t_collide[row] * 100).toFixed(0)}%`,
      element("button", { type: "button", onclick: (event) => { event.stopPropagation(); setPath(`edge ${id} to ${other(row)}`, [id, other(row)]); } }, "walk"),
    ]),
    { onRowClick: (index) => selectNode(other(edgeRows[index])) });
  holder.replaceChildren(element("section", { class: "panel search-detail" }, [
    element("h2", {}, `Node ${id}`),
    ...lines,
    table(["joint", "angle"], conf.map((v, k) => [shortJointName(k), degrees(v)])),
    element("h3", {}, `Edges (${edgeRows.length}), best first; click a row to go to that node, "walk" to step along the edge`),
    edgeTable,
  ]));
}

// ---------------------------------------------------------------------------
// * The panels on the right
// ---------------------------------------------------------------------------

function tiles(items) {
  return element("div", { class: "tiles" }, items.map(([label, value]) =>
    element("div", { class: "tile" }, [element("strong", {}, String(value)), element("span", {}, label)])));
}

function renderRoadmapPanel() {
  const r = state.roadmap;
  const s = r.counters.sample || {};
  const e = r.counters.edges || {};
  const free = r.edges.free.filter(Boolean).length;
  const summary = r.meta.summary || {};
  return panel(`Roadmap ${state.info.run_name} — bar ${state.info.bar_id}`, [
    tiles([
      ["nodes", r.nodes.conf.length], ["edges (free)", `${r.edges.i.length} (${free})`],
      ["IK calls", s.ik_calls || 0], ["collision checks", (s.collision_checks || 0) + (e.collision_checks || 0)],
      ["components", summary.n_components], ["largest", summary.largest_component],
      ["build time", `${Math.round(r.timings.total_s)} s`],
    ]),
    element("p", { class: "muted" }, `Library ${r.meta.library}, goals ${JSON.stringify(r.config.goals)}, `
      + `${r.anchors.length} anchors, k = ${r.config.k}, edge radius ${r.config.radius_deg}°, `
      + `collision every ${r.config.collision_step_deg}°, holding error every ${r.config.viol_step_deg}°, `
      + `${r.config.workers} workers, seed ${r.config.seed}${r.config.lazy ? ", lazy edges" : ""}.`),
  ]);
}

function renderQueryPanel() {
  const choices = [element("option", { value: "" }, "roadmap only")].concat(
    state.info.queries.map((q) => element("option", { value: q.tag }, `${q.tag} (goal ${q.goal_index}, ${q.n_starts} starts)`)));
  const select = element("select", {}, choices);
  select.value = state.queryTag || "";
  select.addEventListener("change", () => { window.location.hash = select.value ? `#/query/${select.value}` : "#/"; });
  const children = [element("div", { class: "controls" }, [element("label", {}, "Query "), select])];
  const q = state.query;
  if (q) {
    const h = q.headline;
    children.push(element("p", {}, h.best_gap_mm == null
      ? "No start reaches the goal set through free edges at any tolerance."
      : `Best gap ${mm(h.best_gap_mm)}, from start ${h.best_start_index}. The goal set has ${q.goal.n_goal_nodes} nodes `
        + `(${q.goal.n_library_pairs} branch pairs in the library, ${q.goal.n_library_pairs_found} found again); `
        + `${q.n_start_nodes} start nodes for ${q.starts.length} starts.`));
    const pathButton = (name, report) => (report
      ? element("button", { type: "button", onclick: () => setPath(name, report.nodes) }, "show") : "—");
    const rows = [];
    if (h.minimax_route) rows.push(["best route (minimax)", "—", h.minimax_route.nodes.length, mm(h.minimax_route.max_score_mm), pathButton("best route", h.minimax_route)]);
    if (h.shortest_at_best_gap) rows.push(["shortest at best gap", mm(h.best_gap_mm), h.shortest_at_best_gap.nodes.length, mm(h.shortest_at_best_gap.max_score_mm), pathButton("shortest at best gap", h.shortest_at_best_gap)]);
    for (const a of q.tolerances) {
      rows.push([`within ${a.tolerance_mm} mm (${a.starts_connected}/${q.starts.length} starts)`, mm(a.tolerance_mm),
        a.filtered ? a.filtered.nodes.length : "none", a.filtered ? mm(a.filtered.max_score_mm) : "—",
        pathButton(`shortest within ${a.tolerance_mm} mm`, a.filtered)]);
      rows.push([`penalised above ${a.tolerance_mm} mm`, mm(a.tolerance_mm), a.penalised ? a.penalised.nodes.length : "none",
        a.penalised ? mm(a.penalised.max_score_mm) : "—", pathButton(`penalised above ${a.tolerance_mm} mm`, a.penalised)]);
    }
    children.push(table(["path", "tolerance", "nodes", "worst error", ""], rows));
    const byClass = Object.entries(q.by_difficulty).map(([label, s]) => [label, s.n, s.n_reachable, mm(s.median_gap_mm),
      Object.entries(s.connected).map(([t, n]) => `${t}: ${n}`).join(", ")]);
    children.push(element("h3", {}, "Starts by library class"));
    children.push(table(["class", "starts", "reach goals", "median gap", "connected within (mm: starts)"], byClass));
  }
  return panel("Query", children);
}

function renderStartsPanel() {
  const q = state.query;
  if (!q) return null;
  const rows = q.starts.slice().sort((a, b) => (a.best_gap_mm ?? Infinity) - (b.best_gap_mm ?? Infinity));
  const shown = rows.slice(0, 300);
  return panel(`Starts (${rows.length}), smallest gap first`, [
    element("p", { class: "muted" }, "Gap: the smallest tolerance at which the start reaches the goal set, over all of "
      + "its branch pairs. Library conf gap: the same for the start's own library configuration only. Click a row to "
      + "select the start's best node and show its route."),
    table(["start", "carry", "class", "direct line free", "nodes", "gap", "library conf gap", "best pair / turn"],
      shown.map((s) => [s.start_index, s.carry, s.difficulty, s.direct_path_free ? "yes" : "no", s.n_nodes,
        mm(s.best_gap_mm), mm(s.library_conf_gap_mm), s.best_pair == null ? "—" : `${s.best_pair} / ${s.best_turns}`]),
      { onRowClick: (index) => { if (shown[index].best_node != null) selectNode(shown[index].best_node); } }),
    rows.length > shown.length ? element("p", { class: "muted" }, `First ${shown.length} shown.`) : null,
  ]);
}

function renderPathPanel() {
  const holder = main.querySelector("[data-path]");
  if (!holder) return;
  renderStageControls();
  if (!state.path) {
    holder.replaceChildren(panel("Path", [element("p", { class: "muted" },
      "Choose a path in the query table, click a node to follow its best route, or walk one edge.")]));
    return;
  }
  const p = state.path;
  const worst = Math.max(...p.profile);
  const chart = element("div", { class: "chart", "data-path-chart": "" });
  holder.replaceChildren(panel(`Path: ${p.name}`, [
    element("p", {}, `${p.nodes.length} nodes, ${p.dense.length} points. Worst holding error along it: ${mm(worst)} `
      + `(green line: the tolerance). Joint moves between points are at most `
      + `${degrees(state.roadmap.config.edge_settings.viol_step_rad)} per joint by construction.`),
    chart,
  ]));
  drawProfile(chart);
}

function renderStageControls() {
  const slider = element("input", { type: "range", min: "0", max: "100", step: "1" });
  const toMm = (v) => Math.pow(10, -0.3 + (Number(v) / 100) * 2.3);  // 0.5 mm .. 100 mm, log scale
  slider.value = String(Math.round(((Math.log10(state.tolerance) + 0.3) / 2.3) * 100));
  const label = element("span", {}, `${state.tolerance.toFixed(1)} mm`);
  slider.addEventListener("input", () => { state.tolerance = toMm(slider.value); label.textContent = `${state.tolerance.toFixed(1)} mm`; });
  slider.addEventListener("change", () => { draw(); if (state.path) { const c = main.querySelector("[data-path-chart]"); if (c) drawProfile(c); } });

  const mode = element("select", {}, [element("option", { value: "kind" }, "colour nodes by kind"),
    element("option", { value: "gap" }, "colour nodes by gap to goals")]);
  mode.value = state.colourMode;
  mode.disabled = !state.query;
  mode.addEventListener("change", () => { state.colourMode = mode.value; draw(); });

  const toggleLabels = { sampled: "sampled", anchor: "anchors", start: "starts", goal: "goals",
    ok: "edges within", near: "edges up to 3x", far: "edges beyond", unknown: "unchecked", blocked: "blocked" };
  const toggleColours = { ...KIND_COLOURS, ...BAND_COLOURS };
  const toggles = element("div", { class: "toggle" }, Object.keys(toggleLabels).map((key) => {
    const button = element("button", { type: "button", "aria-pressed": String(state.shown[key]) },
      [element("i"), toggleLabels[key]]);
    button.firstChild.style.background = cssColour(toggleColours[key]);
    button.addEventListener("click", () => {
      state.shown[key] = !state.shown[key];
      button.setAttribute("aria-pressed", String(state.shown[key]));
      draw();
    });
    return button;
  }));

  const rows = [element("div", { class: "controls" }, [element("label", {}, "Tolerance "), slider, label, mode]), toggles];
  if (state.path) {
    const stepper = element("input", { type: "range", min: "0", max: String(state.path.dense.length - 1), step: "1",
      "data-path-slider": "" });
    stepper.value = String(state.step);
    stepper.addEventListener("input", () => showStep(Number(stepper.value)));
    rows.push(element("div", { class: "controls" }, [element("label", {}, "Along the path "), stepper,
      element("span", { "data-path-label": "" }, `${state.step + 1} / ${state.path.dense.length}`)]));
  }
  stageControls.replaceChildren(...rows);
  stageControls.hidden = false;
}

function renderPage() {
  // replaceChildren would print a null as the text "null", so only real parts go in.
  const crumbs = [element("a", { href: "#/" }, `prm / ${state.info.run_name}`)];
  if (state.queryTag) crumbs.push(element("span", {}, ` / ${state.queryTag}`));
  breadcrumb.replaceChildren(...crumbs);
  main.replaceChildren(
    renderRoadmapPanel(),
    renderQueryPanel(),
    element("div", { "data-path": "" }),
    element("div", { "data-node": "" }),
    renderStartsPanel(),
  );
  renderNodePanel();
  renderPathPanel();
}

// ---------------------------------------------------------------------------
// * Routing
// ---------------------------------------------------------------------------

async function route() {
  const parts = window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  let tag = null;
  let node = null;
  for (let k = 0; k < parts.length - 1; k += 2) {
    if (parts[k] === "query") tag = decodeURIComponent(parts[k + 1]);
    if (parts[k] === "node") node = Number(parts[k + 1]);
  }
  try {
    await loadQuery(tag);
    await ensureRobot();
    renderPage();
    draw();
    if (node != null && node < state.nodes.conf.length) selectNode(node);
    else if (state.query && state.query.headline.minimax_route) setPath("best route", state.query.headline.minimax_route.nodes);
    else showPose(state.roadmap.model.home_conf12, "the home configuration");
  } catch (error) {
    main.replaceChildren(element("section", { class: "panel error" }, [element("h2", {}, "Could not show this page"),
      element("pre", {}, String(error && error.stack ? error.stack : error))]));
  }
}

// A handle for the browser console (and for automated checks of this page):
// __prm.state is the page state, __prm.selectNode(id) selects a node,
// __prm.viewer() is the 3D viewer.
window.__prm = { state, selectNode: (id) => selectNode(id), setPath: (name, ids) => setPath(name, ids),
                 viewer: () => viewer };

async function start() {
  state.info = await loadJson("prm.json");
  state.roadmap = await loadJsonGz(state.info.roadmap);
  document.title = `Roadmap ${state.info.run_name}`;
  for (const node of document.querySelectorAll("[data-title]")) node.textContent = `Roadmap pilot — ${state.info.run_name}`;
  await ensureRobot();
  viewer.onPick((picked) => { if (picked && picked.kind === "node") selectNode(picked.index); });
  window.addEventListener("hashchange", route);
  await route();
}

start().catch((error) => {
  main.replaceChildren(element("section", { class: "panel error" }, [element("h2", {}, "Could not load the roadmap"),
    element("pre", {}, String(error && error.stack ? error.stack : error))]));
});
