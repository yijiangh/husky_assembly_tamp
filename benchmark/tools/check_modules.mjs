// Does every module the dashboard imports actually exist?
//
// A single missing file anywhere in the import graph makes the browser abandon
// the whole graph, and the page renders as a blank area with nothing in the
// console except a 404 for a file nobody mentioned. That is exactly what
// happened once already: the vendored GLTFLoader imports two utility modules
// that were not copied alongside it.
//
// This walks every import from the entry module, resolving bare specifiers
// through the page's own import map, and reports anything that is not on disk.
//
//   node benchmark/tools/check_modules.mjs [dashboard folder]

import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = process.argv[2] || resolve(here, "..", "dashboard");
const indexPath = resolve(root, "index.html");

if (!existsSync(indexPath)) {
  console.error(`No index.html at ${indexPath}`);
  process.exit(1);
}

const html = readFileSync(indexPath, "utf-8");

// The import map tells the browser what a bare specifier such as "three"
// means; resolution here has to agree with it or the check is meaningless.
const mapMatch = html.match(/<script type="importmap">([\s\S]*?)<\/script>/);
const importMap = mapMatch ? JSON.parse(mapMatch[1]).imports || {} : {};

// Entry points: every module script the page loads.
const entries = [...html.matchAll(/<script[^>]*type="module"[^>]*src="([^"]+)"/g)]
  .map((match) => resolve(root, match[1].replace(/^\.\//, "")));

if (!entries.length) {
  console.error("index.html loads no module scripts; nothing to check.");
  process.exit(1);
}

const seen = new Set();
const missing = [];

// Only real import statements count. Matching a bare `from "x"` anywhere would
// also catch documentation and ordinary strings -- the vendored three.js files
// contain both a JSDoc line reading `import ... from 'three/addons/...'` and a
// warning message containing `from "srgb-linear"`, neither of which the browser
// ever loads. Anchoring on the import or export keyword, and stripping block
// comments first, leaves only the real thing.
const IMPORT_PATTERNS = [
  // import ... from "x"   /   export ... from "x"
  /\b(?:import|export)\b[^;]*?\bfrom\s*["']([^"']+)["']/g,
  // import "x"  (side-effect only)
  /\bimport\s+["']([^"']+)["']/g,
  // import("x")  (dynamic)
  /\bimport\s*\(\s*["']([^"']+)["']\s*\)/g,
];

// Removes /* ... */ comments, where the misleading JSDoc lives. Line comments
// are left alone: stripping them safely would mean parsing strings, and the
// keyword anchoring above already handles what they contain.
function stripBlockComments(source) {
  return source.replace(/\/\*[\s\S]*?\*\//g, "");
}

function resolveSpecifier(specifier, fromFile) {
  if (specifier.startsWith("./") || specifier.startsWith("../")) {
    return resolve(dirname(fromFile), specifier);
  }
  if (specifier.startsWith("/")) {
    return resolve(root, specifier.slice(1));
  }
  // A bare specifier: only the import map can resolve it in a browser.
  const mapped = importMap[specifier];
  if (mapped) return resolve(root, mapped.replace(/^\.\//, ""));
  return null; // unmapped bare specifier
}

function walk(file) {
  if (seen.has(file)) return;
  seen.add(file);
  if (!existsSync(file)) {
    missing.push({ file, importedBy: "(entry)" });
    return;
  }
  const source = stripBlockComments(readFileSync(file, "utf-8"));
  const specifiers = new Set();
  for (const pattern of IMPORT_PATTERNS) {
    for (const match of source.matchAll(pattern)) specifiers.add(match[1]);
  }
  for (const specifier of specifiers) {
    if (/^https?:/.test(specifier)) continue;
    const target = resolveSpecifier(specifier, file);
    if (target === null) {
      // A bare specifier the import map does not cover. The browser would
      // fail on it, so it counts as missing.
      missing.push({ file: specifier, importedBy: file, bare: true });
      continue;
    }
    if (!existsSync(target)) {
      missing.push({ file: target, importedBy: file });
      continue;
    }
    walk(target);
  }
}

for (const entry of entries) walk(entry);

console.log(`checked ${seen.size} module(s) reachable from ${entries.length} entry point(s)`);
if (missing.length) {
  console.error(`\n${missing.length} missing:`);
  for (const item of missing) {
    console.error(`  ${item.file}`);
    console.error(`    imported by ${item.importedBy}`);
  }
  console.error("\nFAIL: the browser would abandon the module graph and render a blank page");
  process.exit(1);
}
console.log("PASS: every import resolves to a file on disk");
