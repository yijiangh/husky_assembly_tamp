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
  "state_sync", "smoothing", "stitching", "other",
];

const BREAKDOWN_COLOURS = {
  collision: "#b23a48", ik: "#1f84bd", fk_gate: "#2e9e5b",
  interpolation: "#d98324", nearest_node: "#8a6fb0", state_sync: "#6f8ab0",
  smoothing: "#4fa3a1", stitching: "#b0a06f", other: "#b8bec2",
};

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
  const width = 900;
  const height = 280;
  const box = { left: 52, top: 16, width: width - 76, height: height - 82 };
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

// Stacked bars, for compositions such as failure reasons or time breakdowns.
export function stackedBarChart(container, { data, keys, colours = {},
                                             yLabel = "", normalize = true,
                                             format = (v) => v.toFixed(2) }) {
  const width = 900;
  const height = 280;
  const box = { left: 52, top: 16, width: width - 76, height: height - 82 };
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
      rect.appendChild(el("title", {}, `${d.label} — ${key}: ${format(raw)}`));
      svg.appendChild(rect);
      cursor += value;
    }
  });

  container.replaceChildren(svg, legend(keys, colours));
  return svg;
}

function legend(keys, colours = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = "legend";
  for (const key of keys) {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("i");
    swatch.style.background = colours[key] || BREAKDOWN_COLOURS[key] || COLOURS.muted;
    item.append(swatch, document.createTextNode(key));
    wrapper.appendChild(item);
  }
  return wrapper;
}

// Box plot of a distribution per category, on a log scale because planning
// times span orders of magnitude.
export function boxChart(container, { data, yLabel = "" }) {
  const width = 900;
  const height = 280;
  const box = { left: 52, top: 16, width: width - 76, height: height - 82 };
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

// Line chart, used for the restart curve and for joint angles along a path.
export function lineChart(container, { series, yLabel = "", xLabel = "",
                                       cursor = null, onCursor = null,
                                       height = 260 }) {
  const width = 900;
  const box = { left: 52, top: 16, width: width - 76, height: height - 62 };
  const svg = makeSvg(width, height);

  const allY = series.flatMap((s) => s.points.map((p) => p[1]));
  const allX = series.flatMap((s) => s.points.map((p) => p[0]));
  const yMin = Math.min(0, ...allY);
  const yMax = Math.max(0.001, ...allY);
  const xMin = Math.min(...allX, 0);
  const xMax = Math.max(...allX, 1);
  const toX = (value) => box.left + ((value - xMin) / (xMax - xMin || 1)) * box.width;
  const toY = (value) => box.top + box.height
    - ((value - yMin) / (yMax - yMin || 1)) * box.height;

  axes(svg, box, {
    yLabel, xLabel,
    yTicks: [0, 0.5, 1].map((f) => ({
      y: toY(yMin + f * (yMax - yMin)),
      label: (yMin + f * (yMax - yMin)).toFixed(2),
    })),
    xTicks: [0, 0.5, 1].map((f) => ({
      x: toX(xMin + f * (xMax - xMin)),
      label: Math.round(xMin + f * (xMax - xMin)).toString(),
    })),
  });

  for (const line of series) {
    if (!line.points.length) continue;
    const path = line.points
      .map((p, i) => `${i ? "L" : "M"}${toX(p[0]).toFixed(1)},${toY(p[1]).toFixed(1)}`)
      .join(" ");
    svg.appendChild(el("path", {
      d: path, fill: "none",
      stroke: line.colour || COLOURS.fine,
      "stroke-width": line.width || 1.5,
      opacity: line.opacity != null ? line.opacity : 1,
    }));
  }

  if (cursor != null) {
    svg.appendChild(el("line", {
      x1: toX(cursor), y1: box.top, x2: toX(cursor), y2: box.top + box.height,
      stroke: COLOURS.accent, "stroke-width": 2,
    }));
  }
  if (onCursor) {
    svg.style.cursor = "crosshair";
    svg.addEventListener("click", (event) => {
      const rect = svg.getBoundingClientRect();
      const fraction = (event.clientX - rect.left) / rect.width;
      const value = xMin + (fraction * width - box.left) / box.width * (xMax - xMin);
      onCursor(Math.max(xMin, Math.min(xMax, value)));
    });
  }

  container.replaceChildren(svg);
  return svg;
}

// Grid of coloured cells, for one slice of the start-offset sweep.
export function heatmap(container, { cells, xValues, yValues, xLabel, yLabel,
                                     colourOf, titleOf }) {
  const cellSize = 34;
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
