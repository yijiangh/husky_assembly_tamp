// Charts, drawn as plain SVG.
//
// A charting library would be a megabyte of vendored code for six chart types,
// and would still need styling to match the 3D view. These are small enough to
// write, they work with no network, and the colours are shared with the
// viewer so a category looks the same everywhere.

export const COLOURS = {
  free: "#2e9e5b",
  blocked: "#d98324",
  broken: "#b23a48",
  feasible: "#2e9e5b",
  collision: "#b23a48",
  ik_unreachable_left: "#8a6fb0",
  ik_unreachable_right: "#6f8ab0",
  ik_unreachable_both: "#5a5a6e",
  no_valid_branch_pair: "#b0a06f",
  coarse: "#9ec5df",
  fine: "#1f84bd",
  accent: "#f2c14e",
  grid: "#dfe3e5",
  text: "#33383b",
  muted: "#7a8288",
};

// Time breakdown components, in the order they are stacked.
export const BREAKDOWN_ORDER = [
  "collision", "ik", "fk_gate", "interpolation", "nearest_node",
  "state_sync", "smoothing", "stitching", "repair", "other",
];

export const BREAKDOWN_LABELS = {
  collision: "collision checks",
  ik: "IK while growing the tree",
  fk_gate: "FK check",
  interpolation: "interpolation",
  nearest_node: "nearest-node search",
  state_sync: "scene sync",
  smoothing: "smoothing",
  stitching: "joining the two trees",
  repair: "re-solving the path on the calibrated model",
  other: "other",
};

// The stages of setup, before any motion planning happens.
export const SETUP_ORDER = [
  "goal_ik", "start_survey", "straight_line_probes", "planner_start_derivation",
];

export const SETUP_LABELS = {
  goal_ik: "goal IK",
  start_survey: "start survey (IK + collision)",
  straight_line_probes: "straight-line probes",
  planner_start_derivation: "planner's own start search",
};

export const SETUP_COLOURS = {
  goal_ik: "#1f84bd",
  start_survey: "#b23a48",
  straight_line_probes: "#d98324",
  planner_start_derivation: "#8a6fb0",
};

const BREAKDOWN_COLOURS = {
  collision: "#b23a48", ik: "#1f84bd", fk_gate: "#2e9e5b",
  interpolation: "#d98324", nearest_node: "#8a6fb0", state_sync: "#6f8ab0",
  smoothing: "#4fa3a1", stitching: "#b0a06f", repair: "#c46fa0", other: "#b8bec2",
};

// ! Why a tree extension stopped.
// Every attempt to grow a search tree walks toward a target pose and ends for
// one reason. The planner counts eleven raw keys: the plain reason for the
// growth walk, the same reason with a `connect_` prefix for the walk that tries
// to join the other tree, and a `stitch_` prefix for re-solving the seam where
// the trees meet. Read together they are four things, and those four are what
// the report's "why extensions stop" line is made of.
export const STOP_REASON_ORDER = [
  "continuity", "ik_failure", "collision", "reached", "other",
];

export const STOP_REASON_LABELS = {
  continuity: "continuity: the arm solution jumped",
  ik_failure: "IK failure: a waypoint was unreachable",
  collision: "collision: something was in the way",
  reached: "reached its target",
  other: "other",
};

export const STOP_REASON_COLOURS = {
  continuity: "#d98324", ik_failure: "#8a6fb0", collision: "#b23a48",
  reached: "#2e9e5b", other: "#b8bec2",
};

// Raw planner key -> one of the four groups above. Anything not listed here
// counts as "other" (today that is only the rare stitch endpoint mismatch).
const STOP_REASON_GROUP = {
  continuity: "continuity",
  connect_continuity: "continuity",
  stitch_continuity_fail: "continuity",
  ik_failure: "ik_failure",
  connect_ik_failure: "ik_failure",
  stitch_ik_failure: "ik_failure",
  collision: "collision",
  connect_collision: "collision",
  stitch_collision: "collision",
  reached: "reached",
  connect_reached: "reached",
};

// Folds one or more raw stop-reason histograms into the four groups.
//
// Pass several histograms to add them up, e.g. every pass of every seed on a
// start page. Missing or empty histograms are fine and add nothing.
export function groupStopReasons(...histograms) {
  const grouped = Object.fromEntries(STOP_REASON_ORDER.map((key) => [key, 0]));
  for (const histogram of histograms) {
    for (const [rawKey, count] of Object.entries(histogram || {})) {
      grouped[STOP_REASON_GROUP[rawKey] || "other"] += count || 0;
    }
  }
  return grouped;
}

// One sentence in the style of the report: the three shares that matter, then
// a verdict on what is holding the search back.
//
// The verdict only looks at the three ways a walk can be rejected, so a run
// where most walks reach their target still names whichever rejection is the
// most common one.
export function stopReasonVerdict(grouped) {
  const total = STOP_REASON_ORDER.reduce((sum, key) => sum + grouped[key], 0);
  if (!total) return "No extension attempts were recorded.";
  const share = (key) => `${(grouped[key] / total * 100).toFixed(1)} %`;
  const rejections = ["continuity", "ik_failure", "collision"];
  const main = rejections.reduce((best, key) => grouped[key] > grouped[best] ? key : best);
  const verdict = {
    continuity: "The continuity gate is the main brake, not the environment.",
    ik_failure: "Reachability is the main brake: the arms could not solve the waypoints.",
    collision: "The environment is the main brake: the search kept running into something.",
  }[main];
  return `Why extensions stop: continuity ${share("continuity")}, `
    + `IK failure ${share("ik_failure")}, collision ${share("collision")}. ${verdict}`;
}

const NS = "http://www.w3.org/2000/svg";

function el(name, attributes = {}, text = null) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) {
    node.setAttribute(key, value);
  }
  if (text != null) node.textContent = text;
  return node;
}

// An SVG at its natural size, allowed to shrink but never to grow.
//
// A width of 100% with a viewBox scales the whole drawing to the container,
// text included, so on a wide screen an 11px label is rendered at 30px and the
// chart looks like a poster. Giving real pixel dimensions and capping the width
// in CSS keeps text at the size it was designed at, and still lets a narrow
// window scale the chart down.
function makeSvg(width, height) {
  return el("svg", {
    viewBox: `0 0 ${width} ${height}`,
    width: String(width),
    height: String(height),
    // ! Inline, not in the stylesheet: this is the one rule that must hold, and
    // ! an inline style cannot be overridden by a stale cached stylesheet.
    style: `width:${width}px;height:${height}px;max-width:100%`,
    preserveAspectRatio: "xMidYMid meet",
    class: "chart",
  });
}

// Draws a tick label, breaking it into stacked lines on newlines.
//
// SVG text ignores newlines, so a two-line label would otherwise render as one
// long line and collide with its neighbours -- which is what a category label
// carrying both a name and a measurement does on every chart here.
function tickLabel(x, y, text, { anchor = "middle", size = 11 } = {}) {
  const lines = String(text).split("\n");
  const node = el("text", {
    x, y, "text-anchor": anchor, "font-size": size, fill: COLOURS.muted,
  });
  lines.forEach((line, index) => {
    node.appendChild(el("tspan", {
      x, dy: index === 0 ? 0 : size + 2,
    }, line));
  });
  return node;
}

function axes(svg, box, { xLabel = "", yLabel = "", yTicks = [], xTicks = [] }) {
  const { left, top, width, height } = box;
  svg.appendChild(el("line", {
    x1: left, y1: top + height, x2: left + width, y2: top + height,
    stroke: COLOURS.muted, "stroke-width": 1,
  }));
  for (const tick of yTicks) {
    svg.appendChild(el("line", {
      x1: left, y1: tick.y, x2: left + width, y2: tick.y,
      stroke: COLOURS.grid, "stroke-width": 1,
    }));
    svg.appendChild(el("text", {
      x: left - 6, y: tick.y + 4, "text-anchor": "end",
      "font-size": 11, fill: COLOURS.muted,
    }, tick.label));
  }
  for (const tick of xTicks) {
    svg.appendChild(tickLabel(tick.x, top + height + 16, tick.label));
  }
  if (yLabel) {
    svg.appendChild(el("text", {
      x: 12, y: top + height / 2, "text-anchor": "middle",
      transform: `rotate(-90 12 ${top + height / 2})`,
      "font-size": 11, fill: COLOURS.muted,
    }, yLabel));
  }
  if (xLabel) {
    svg.appendChild(el("text", {
      x: left + width / 2, y: top + height + 48, "text-anchor": "middle",
      "font-size": 11, fill: COLOURS.muted,
    }, xLabel));
  }
}

// Bars with optional error bars, and optional hollow markers for a second
// series (used to show the coarse pass alongside the fine one).
export function barChart(container, { data, yLabel = "", xLabel = "",
                                      yMax = null, format = (v) => v.toFixed(2),
                                      onClick = null }) {
  const width = 820;
  const height = 210;
  const box = { left: 48, top: 12, width: width - 72, height: height - 74 };
  const svg = makeSvg(width, height);

  const maximum = yMax != null ? yMax : Math.max(
    0.001, ...data.map((d) => Math.max(d.value, d.high || 0, d.marker || 0)));
  const toY = (value) => box.top + box.height - (value / maximum) * box.height;

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((fraction) => ({
    y: toY(fraction * maximum),
    label: format(fraction * maximum),
  }));
  const slot = box.width / Math.max(1, data.length);
  const barWidth = Math.min(38, slot * 0.62);
  const xTicks = data.map((d, i) => ({
    x: box.left + slot * (i + 0.5), label: d.label,
  }));
  axes(svg, box, { yLabel, xLabel, yTicks, xTicks });

  data.forEach((d, index) => {
    const centre = box.left + slot * (index + 0.5);
    const y = toY(d.value);
    const rect = el("rect", {
      x: centre - barWidth / 2, y, width: barWidth,
      height: Math.max(1, box.top + box.height - y),
      fill: d.colour || COLOURS.fine, rx: 2,
    });
    rect.appendChild(el("title", {}, `${d.label}: ${format(d.value)}`));
    if (onClick) {
      rect.style.cursor = "pointer";
      rect.addEventListener("click", () => onClick(d));
    }
    svg.appendChild(rect);

    if (d.low != null && d.high != null) {
      svg.appendChild(el("line", {
        x1: centre, y1: toY(d.low), x2: centre, y2: toY(d.high),
        stroke: COLOURS.text, "stroke-width": 1.5,
      }));
      for (const bound of [d.low, d.high]) {
        svg.appendChild(el("line", {
          x1: centre - 5, y1: toY(bound), x2: centre + 5, y2: toY(bound),
          stroke: COLOURS.text, "stroke-width": 1.5,
        }));
      }
    }
    if (d.marker != null) {
      const marker = el("circle", {
        cx: centre, cy: toY(d.marker), r: 4,
        fill: "white", stroke: COLOURS.coarse, "stroke-width": 2,
      });
      marker.appendChild(el("title", {}, `coarse pass: ${format(d.marker)}`));
      svg.appendChild(marker);
    }
  });

  container.replaceChildren(svg);
  return svg;
}

// Horizontal bars, one per row, with the row's label on the left and its
// value written at the end of the bar. Used for time breakdowns by phase,
// where the labels are sentences ("edges: collision and holding error") that
// would not fit under a vertical bar. Rows are { label, value, colour, note }:
// `note` is extra text after the value (a count, a percentage).
export function horizontalBarChart(container, { data, xLabel = "",
                                                format = (v) => v.toFixed(1),
                                                labelWidth = 230, rowHeight = 22 }) {
  const width = 820;
  const height = rowHeight * Math.max(1, data.length) + 44;
  const box = { left: labelWidth, top: 8, width: width - labelWidth - 90, height: rowHeight * data.length };
  const svg = makeSvg(width, height);

  const maximum = Math.max(0.001, ...data.map((d) => d.value || 0));
  const toX = (value) => box.left + (value / maximum) * box.width;

  // Light vertical grid at quarter steps, labelled under the axis.
  for (const fraction of [0, 0.25, 0.5, 0.75, 1]) {
    const x = toX(fraction * maximum);
    svg.appendChild(el("line", { x1: x, y1: box.top, x2: x, y2: box.top + box.height,
                                 stroke: COLOURS.grid, "stroke-width": 1 }));
    svg.appendChild(tickLabel(x, box.top + box.height + 16, format(fraction * maximum)));
  }
  if (xLabel) {
    svg.appendChild(el("text", { x: box.left + box.width / 2, y: height - 4, "text-anchor": "middle",
                                 "font-size": 11, fill: COLOURS.muted }, xLabel));
  }

  data.forEach((d, index) => {
    const y = box.top + rowHeight * index;
    const barHeight = rowHeight * 0.64;
    const value = d.value || 0;
    svg.appendChild(el("text", { x: box.left - 8, y: y + rowHeight / 2 + 4, "text-anchor": "end",
                                 "font-size": 11, fill: COLOURS.text }, d.label));
    const rect = el("rect", { x: box.left, y: y + (rowHeight - barHeight) / 2,
                              width: Math.max(1, toX(value) - box.left), height: barHeight,
                              fill: d.colour || COLOURS.fine, rx: 2 });
    rect.appendChild(el("title", {}, `${d.label}: ${format(value)}${d.note ? ` ${d.note}` : ""}`));
    svg.appendChild(rect);
    svg.appendChild(el("text", { x: toX(value) + 6, y: y + rowHeight / 2 + 4, "font-size": 11,
                                 fill: COLOURS.text }, `${format(value)}${d.note ? ` ${d.note}` : ""}`));
  });

  container.replaceChildren(svg);
  return svg;
}

// Stacked bars, for compositions such as failure reasons or time breakdowns.
export function stackedBarChart(container, { data, keys, colours = {},
                                             labels = {},
                                             yLabel = "", normalize = true,
                                             format = (v) => v.toFixed(2) }) {
  const width = 820;
  const height = 210;
  const box = { left: 48, top: 12, width: width - 72, height: height - 74 };
  const svg = makeSvg(width, height);

  const totals = data.map((d) => keys.reduce((sum, key) => sum + (d.values[key] || 0), 0));
  const maximum = normalize ? 1 : Math.max(0.001, ...totals);
  const toY = (value) => box.top + box.height - (value / maximum) * box.height;

  const yTicks = [0, 0.5, 1].map((fraction) => ({
    y: toY(fraction * maximum), label: format(fraction * maximum),
  }));
  const slot = box.width / Math.max(1, data.length);
  const barWidth = Math.min(38, slot * 0.62);
  axes(svg, box, {
    yLabel, yTicks,
    xTicks: data.map((d, i) => ({ x: box.left + slot * (i + 0.5), label: d.label })),
  });

  data.forEach((d, index) => {
    const centre = box.left + slot * (index + 0.5);
    const total = totals[index] || 1;
    let cursor = 0;
    for (const key of keys) {
      const raw = d.values[key] || 0;
      if (!raw) continue;
      const value = normalize ? raw / total : raw;
      const yTop = toY(cursor + value);
      const yBottom = toY(cursor);
      const rect = el("rect", {
        x: centre - barWidth / 2, y: yTop, width: barWidth,
        height: Math.max(1, yBottom - yTop),
        fill: colours[key] || BREAKDOWN_COLOURS[key] || COLOURS.muted,
      });
      rect.appendChild(el("title", {},
        `${d.label} — ${labels[key] || key}: ${format(raw)}`));
      svg.appendChild(rect);
      cursor += value;
    }
  });

  container.replaceChildren(svg, legend(keys, colours, labels));
  return svg;
}

function legend(keys, colours = {}, labels = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = "legend";
  for (const key of keys) {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("i");
    swatch.style.background = colours[key] || BREAKDOWN_COLOURS[key] || COLOURS.muted;
    item.append(swatch, document.createTextNode(labels[key] || key));
    wrapper.appendChild(item);
  }
  return wrapper;
}

// Box plot of a distribution per category, on a log scale because planning
// times span orders of magnitude.
export function boxChart(container, { data, yLabel = "" }) {
  const width = 820;
  const height = 210;
  const box = { left: 48, top: 12, width: width - 72, height: height - 74 };
  const svg = makeSvg(width, height);

  const values = data.flatMap((d) => [d.p10, d.p50, d.p90, d.max].filter((v) => v > 0));
  const maximum = Math.max(0.1, ...values);
  const minimum = Math.max(0.01, Math.min(...values, maximum));
  const logMin = Math.log10(minimum);
  const logMax = Math.log10(maximum * 1.2);
  const toY = (value) => {
    const clamped = Math.max(minimum, value || minimum);
    return box.top + box.height
      - ((Math.log10(clamped) - logMin) / (logMax - logMin)) * box.height;
  };

  const yTicks = [];
  for (let exponent = Math.floor(logMin); exponent <= Math.ceil(logMax); exponent += 1) {
    yTicks.push({ y: toY(10 ** exponent), label: `${10 ** exponent}s` });
  }
  const slot = box.width / Math.max(1, data.length);
  axes(svg, box, {
    yLabel, yTicks,
    xTicks: data.map((d, i) => ({ x: box.left + slot * (i + 0.5), label: d.label })),
  });

  data.forEach((d, index) => {
    if (!d.p50) return;
    const centre = box.left + slot * (index + 0.5);
    const barWidth = Math.min(30, slot * 0.5);
    svg.appendChild(el("line", {
      x1: centre, y1: toY(d.p10), x2: centre, y2: toY(d.p90),
      stroke: COLOURS.text, "stroke-width": 1,
    }));
    const rect = el("rect", {
      x: centre - barWidth / 2, y: toY(d.p90), width: barWidth,
      height: Math.max(2, toY(d.p10) - toY(d.p90)),
      fill: d.colour || COLOURS.fine, opacity: 0.75, rx: 2,
    });
    rect.appendChild(el("title", {},
      `${d.label}: median ${d.p50.toFixed(1)}s, ` +
      `10th–90th ${d.p10.toFixed(1)}–${d.p90.toFixed(1)}s`));
    svg.appendChild(rect);
    svg.appendChild(el("line", {
      x1: centre - barWidth / 2, y1: toY(d.p50),
      x2: centre + barWidth / 2, y2: toY(d.p50),
      stroke: COLOURS.text, "stroke-width": 2,
    }));
  });

  container.replaceChildren(svg);
  return svg;
}

// Line chart, used for the restart curve and for the curves along a path
// (holding error, joint angles).
//
// Each series is { points: [[x, y], ...], colour, width, opacity, dots, label }.
// `dots` (a radius in px) also marks every data point with a filled circle, so
// the reader can see how dense the data is and where the line is only a guess
// between two points. `label` puts the series in a legend under the chart.
// `yTickCount` and `yFormat` shape the y axis labels (joint angles want whole
// degrees, errors want two decimals).
//
// The returned svg carries `setCursor(x)`, which moves the vertical cursor
// without redrawing the chart: with a thousand dots per series a full redraw
// on every slider tick would make the stepper sluggish.
export function lineChart(container, { series, yLabel = "", xLabel = "",
                                       cursor = null, onCursor = null,
                                       height = 200, yTickCount = 3,
                                       yFormat = (v) => v.toFixed(2) }) {
  const width = 820;
  const box = { left: 48, top: 12, width: width - 72, height: height - 58 };
  const svg = makeSvg(width, height);

  // The data range, found with plain loops: spreading tens of thousands of
  // points into Math.min/Math.max overflows the call stack.
  let yMin = 0; let yMax = 0.001; let xMin = 0; let xMax = 1;
  for (const s of series) {
    for (const p of s.points) {
      if (p[0] < xMin) xMin = p[0];
      if (p[0] > xMax) xMax = p[0];
      if (p[1] < yMin) yMin = p[1];
      if (p[1] > yMax) yMax = p[1];
    }
  }
  const toX = (value) => box.left + ((value - xMin) / (xMax - xMin || 1)) * box.width;
  const toY = (value) => box.top + box.height
    - ((value - yMin) / (yMax - yMin || 1)) * box.height;

  // Evenly spaced y ticks from the bottom to the top of the data.
  const yFractions = [];
  for (let k = 0; k < yTickCount; k += 1) yFractions.push(k / Math.max(1, yTickCount - 1));
  axes(svg, box, {
    yLabel, xLabel,
    yTicks: yFractions.map((f) => ({
      y: toY(yMin + f * (yMax - yMin)),
      label: yFormat(yMin + f * (yMax - yMin)),
    })),
    xTicks: [0, 0.5, 1].map((f) => ({
      x: toX(xMin + f * (xMax - xMin)),
      label: Math.round(xMin + f * (xMax - xMin)).toString(),
    })),
  });

  for (const line of series) {
    if (!line.points.length) continue;
    const colour = line.colour || COLOURS.fine;
    // A width of 0 means "dots only, no line".
    const strokeWidth = line.width != null ? line.width : 1.5;
    if (strokeWidth > 0) {
      const path = line.points
        .map((p, i) => `${i ? "L" : "M"}${toX(p[0]).toFixed(1)},${toY(p[1]).toFixed(1)}`)
        .join(" ");
      svg.appendChild(el("path", {
        d: path, fill: "none",
        stroke: colour,
        "stroke-width": strokeWidth,
        opacity: line.opacity != null ? line.opacity : 1,
      }));
    }
    // One small filled circle per data point, when asked for.
    if (line.dots) {
      const group = el("g", { fill: colour, opacity: line.opacity != null ? line.opacity : 1 });
      for (const p of line.points) {
        group.appendChild(el("circle", { cx: toX(p[0]).toFixed(1), cy: toY(p[1]).toFixed(1), r: line.dots }));
      }
      svg.appendChild(group);
    }
  }

  // The cursor is one line that later calls move; it is created on first use.
  let cursorLine = null;
  svg.setCursor = (value) => {
    if (value == null) { if (cursorLine) cursorLine.remove(); cursorLine = null; return; }
    if (!cursorLine) {
      cursorLine = el("line", { y1: box.top, y2: box.top + box.height, stroke: COLOURS.accent, "stroke-width": 2 });
      svg.appendChild(cursorLine);
    }
    cursorLine.setAttribute("x1", toX(value));
    cursorLine.setAttribute("x2", toX(value));
  };
  if (cursor != null) svg.setCursor(cursor);
  if (onCursor) {
    svg.style.cursor = "crosshair";
    svg.addEventListener("click", (event) => {
      const rect = svg.getBoundingClientRect();
      const fraction = (event.clientX - rect.left) / rect.width;
      const value = xMin + (fraction * width - box.left) / box.width * (xMax - xMin);
      onCursor(Math.max(xMin, Math.min(xMax, value)));
    });
  }

  // A legend for the series that carry a label, under the drawing.
  const labelled = series.filter((s) => s.label);
  if (labelled.length) {
    const keys = labelled.map((s) => s.label);
    const colours = Object.fromEntries(labelled.map((s) => [s.label, s.colour || COLOURS.fine]));
    container.replaceChildren(svg, legend(keys, colours));
  } else {
    container.replaceChildren(svg);
  }
  return svg;
}

// Grid of coloured cells, for one slice of the start-offset sweep.
export function heatmap(container, { cells, xValues, yValues, xLabel, yLabel,
                                     colourOf, titleOf }) {
  const cellSize = 26;
  const width = 60 + xValues.length * cellSize;
  const height = 40 + yValues.length * cellSize;
  const svg = makeSvg(width, height);

  yValues.forEach((yValue, row) => {
    xValues.forEach((xValue, column) => {
      const cell = cells.find((c) => c.x === xValue && c.y === yValue);
      const rect = el("rect", {
        x: 50 + column * cellSize, y: 10 + row * cellSize,
        width: cellSize - 2, height: cellSize - 2, rx: 2,
        fill: cell ? colourOf(cell) : "#f2f4f5",
        stroke: "white", "stroke-width": 1,
      });
      if (cell) rect.appendChild(el("title", {}, titleOf(cell)));
      svg.appendChild(rect);
    });
    svg.appendChild(el("text", {
      x: 44, y: 10 + row * cellSize + cellSize / 2 + 4, "text-anchor": "end",
      "font-size": 10, fill: COLOURS.muted,
    }, yValue.toFixed(2)));
  });
  xValues.forEach((xValue, column) => {
    svg.appendChild(el("text", {
      x: 50 + column * cellSize + (cellSize - 2) / 2, y: height - 14,
      "text-anchor": "middle", "font-size": 10, fill: COLOURS.muted,
    }, xValue.toFixed(2)));
  });
  svg.appendChild(el("text", {
    x: width / 2, y: height - 2, "text-anchor": "middle",
    "font-size": 11, fill: COLOURS.muted,
  }, xLabel));

  container.replaceChildren(svg);
  return svg;
}

// Turns any chart's data into a CSV the user can paste elsewhere.
export function copyAsCsv(rows, button) {
  const text = rows.map((row) => row.join(",")).join("\n");
  navigator.clipboard.writeText(text).then(() => {
    const original = button.textContent;
    button.textContent = "copied";
    setTimeout(() => { button.textContent = original; }, 1200);
  });
}
