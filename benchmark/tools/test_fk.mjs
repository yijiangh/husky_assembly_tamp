// Does the browser's forward kinematics agree with the simulator's?
//
// The dashboard draws the robot by posing a node tree from kinematics.json. If
// that disagrees with the simulator, every 3D view is quietly wrong -- the
// robot would be shown somewhere the planner never put it, and nobody would
// notice from the picture alone.
//
// This runs the real kinematics module against poses recorded from PyBullet.
// No browser: three.js runs fine in Node, and the module touches no DOM.
//
//   node --experimental-import-meta-resolve test_fk.mjs <fixtures.json>
//
// Fixtures come from the benchmark's fk_check; see the README.

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dashboard = resolve(here, "..", "dashboard");

const threeUrl = pathToFileURL(
  resolve(dashboard, "static/vendor/three-r185/build/three.module.min.js"));
const THREE = await import(threeUrl.href);

// The browser resolves the bare specifier "three" through the import map in
// index.html. Node has no import map, so the module's source is loaded with
// that one specifier rewritten to the vendored file. The code under test is
// otherwise byte for byte what the page runs.
async function importBrowserModule(path) {
  const source = readFileSync(path, "utf-8")
    .replace(/from\s+"three"/g, `from "${threeUrl.href}"`);
  return import(
    `data:text/javascript;base64,${Buffer.from(source, "utf-8").toString("base64")}`);
}

const fixturesPath = process.argv[2]
  || resolve(here, "..", "tests", "fixtures", "fk_fixtures.json");
const fixtures = JSON.parse(readFileSync(fixturesPath, "utf-8"));

// Build the node tree the exported scene would have, straight from the
// kinematics description, and pose it exactly as the dashboard does.
const kinematics = fixtures.kinematics;
const nodesByLink = new Map();
const root = new THREE.Object3D();
root.name = "root";

for (const link of fixtures.links) {
  const node = new THREE.Object3D();
  node.name = `L_${link.name}`;
  node.position.fromArray(link.translation);
  node.quaternion.fromArray(link.rotation);
  nodesByLink.set(link.name, node);
}
for (const link of fixtures.links) {
  const node = nodesByLink.get(link.name);
  const parent = link.parent ? nodesByLink.get(link.parent) : root;
  parent.add(node);
}

const { RobotKinematics } = await importBrowserModule(
  resolve(dashboard, "static/js/kinematics.js"));
const robot = new RobotKinematics(kinematics, root);

let worst = 0;
for (const sample of fixtures.samples) {
  robot.pose(sample.angles);
  for (const [linkName, expected] of Object.entries(sample.links)) {
    const matrix = robot.linkMatrix(linkName);
    if (!matrix) throw new Error(`no node for link ${linkName}`);
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    matrix.decompose(position, quaternion, scale);
    const positionError = position.distanceTo(
      new THREE.Vector3().fromArray(expected.position));

    // Compare the quaternions component by component rather than as an angle.
    // Turning a dot product into an angle with acos is unstable when the two
    // rotations nearly agree, which is exactly the case here: it turns float
    // noise of 1e-8 into an apparent error of 1e-4 and would fail a test that
    // the position error shows to be fine.
    const expectedQuaternion = new THREE.Quaternion().fromArray(expected.quaternion);
    // q and -q are the same rotation, so line the signs up first.
    const sign = quaternion.dot(expectedQuaternion) < 0 ? -1 : 1;
    const rotationError = Math.max(
      Math.abs(quaternion.x * sign - expectedQuaternion.x),
      Math.abs(quaternion.y * sign - expectedQuaternion.y),
      Math.abs(quaternion.z * sign - expectedQuaternion.z),
      Math.abs(quaternion.w * sign - expectedQuaternion.w),
    );
    worst = Math.max(worst, positionError, rotationError);
    console.log(
      `case ${sample.name.padEnd(8)} ${linkName.padEnd(24)} ` +
      `position ${positionError.toExponential(2)} rotation ${rotationError.toExponential(2)}`);
  }
}

console.log(`\nworst disagreement: ${worst.toExponential(2)}`);
if (worst > 1e-4) {
  console.error("FAIL: the browser would draw the robot in the wrong place");
  process.exit(1);
}
console.log("PASS");
