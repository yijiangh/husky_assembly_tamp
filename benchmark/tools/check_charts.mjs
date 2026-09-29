// Do the charts actually fit their labels?
//
// A chart that renders without error can still be unreadable, and the usual way
// to find out is to look at it. This checks the two failures that are worth
// catching mechanically: text scaled away from the size it was designed at,
// and category labels wide enough to collide with their neighbours.
//
//   node benchmark/tools/check_charts.mjs [summary.json]

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dashboard = resolve(here, "..", "dashboard");

// The chart module builds DOM nodes, so give it just enough of one to run.
// Recording attributes and children is all the checks below need.
class FakeNode {
  constructor(name) {
    this.nodeName = name;
    this.attributes = {};
    this.children = [];
    this.textContent = "";
    this.style = {};
    this.className = "";
  }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  getAttribute(key) { return this.attributes[key]; }
  appendChild(child) { this.children.push(child); return child; }
  append(...kids) { this.children.push(...kids); }
  replaceChildren(...kids) { this.children = kids; }
  addEventListener() {}
  querySelector() { return null; }
}

globalThis.document = {
  createElementNS: (_ns, name) => new FakeNode(name),
  createElement: (name) => new FakeNode(name),
  createTextNode: (text) => ({ nodeName: "#text", textContent: text }),
};
globalThis.Node = FakeNode;

const charts = await import(
  pathToFileURL(resolve(dashboard, "static/js/charts.js")).href);

const summaryPath = process.argv[2];
let bars;
if (summaryPath) {
  const summary = JSON.parse(readFileSync(summaryPath, "utf-8"));
  bars = summary.per_bar.slice().sort((a, b) => a.bar_length_m - b.bar_length_m);
} else {
  // Sixteen categories with two-line labels: the shape that broke before.
  bars = Array.from({ length: 16 }, (_, i) => ({
    bar_id: `B${i * 3 + 3}`,
    bar_length_m: 0.5 + i * 0.1,
    success_rate: 0.15,
    success_rate_ci95: [0.13, 0.17],
    coarse_success_rate: 0.17,
  }));
}

const container = new FakeNode("div");
const svg = charts.barChart(container, {
  yLabel: "solved", yMax: 1,
  format: (v) => `${(v * 100).toFixed(0)}%`,
  data: bars.map((b) => ({
    label: `${b.bar_id}\n${b.bar_length_m.toFixed(2)}m`,
    value: b.success_rate,
    low: b.success_rate_ci95[0], high: b.success_rate_ci95[1],
    marker: b.coarse_success_rate,
  })),
});

const problems = [];

// 1. The drawing must not be stretched to the container, or the text goes with
//    it. A pixel width alongside the viewBox is what prevents that.
const width = Number(svg.getAttribute("width"));
const viewBoxWidth = Number(svg.getAttribute("viewBox").split(" ")[2]);
if (!width || width !== viewBoxWidth) {
  problems.push(
    `svg width is "${svg.getAttribute("width")}" but the viewBox is ` +
    `${viewBoxWidth} wide; the browser will scale the drawing, and the text ` +
    `with it`);
}

// 2. Category labels must fit their slot. Each is its own text node, so the
//    slot width comes from how many there are.
const labels = svg.children.filter(
  (child) => child.nodeName === "text" && child.children.length > 0);
const slot = (viewBoxWidth - 76) / bars.length;
// Rough advance width for this font at this size; generous, so a pass here
// means real text fits too.
const CHAR_WIDTH = 0.62;
for (const label of labels) {
  const size = Number(label.getAttribute("font-size")) || 11;
  for (const line of label.children) {
    const estimated = line.textContent.length * size * CHAR_WIDTH;
    if (estimated > slot) {
      problems.push(
        `label "${line.textContent}" needs about ${estimated.toFixed(0)}px ` +
        `but its slot is ${slot.toFixed(0)}px wide`);
    }
  }
}

// 3. A label written as two lines must render as two lines.
const multiline = labels.filter((label) => label.children.length > 1);
if (bars.length && !multiline.length) {
  problems.push(
    "two-line labels were requested but every label rendered as one line; " +
    "SVG text ignores newlines unless they are split into tspans");
}

console.log(
  `checked ${labels.length} category label(s) at ${viewBoxWidth}px wide, ` +
  `${slot.toFixed(0)}px per slot`);
if (problems.length) {
  console.error(`\n${problems.length} problem(s):`);
  for (const problem of new Set(problems)) console.error(`  ${problem}`);
  process.exit(1);
}
console.log("PASS: charts render at their intended size and labels fit");
