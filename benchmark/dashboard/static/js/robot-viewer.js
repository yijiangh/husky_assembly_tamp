// The 3D panel: the robot, the bar it carries, and clouds of candidate poses.
//
// One of these is created once and reused by every page, because setting up
// WebGL and loading three megabytes of robot is far too slow to redo on each
// navigation.
//
// Candidate poses are drawn as instanced cylinders rather than separate
// objects. A goal page shows fifty bars and a start page can show several
// hundred, and instancing keeps that to one draw call instead of hundreds.

import * as THREE from "three";
import { OrbitControls } from "../vendor/three-r185/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "../vendor/three-r185/examples/jsm/loaders/GLTFLoader.js";
import { RobotKinematics, findNode } from "./kinematics.js";

// Colours shared with the charts, so a bar that is orange in a plot is orange
// in the 3D view too.
export const OUTCOME_COLOURS = {
  free: 0x2e9e5b,
  blocked: 0xd98324,
  broken: 0xb23a48,
  feasible: 0x2e9e5b,
  collision: 0xb23a48,
  ik_unreachable_left: 0x8a6fb0,
  ik_unreachable_right: 0x6f8ab0,
  ik_unreachable_both: 0x5a5a6e,
  no_valid_branch_pair: 0xb0a06f,
  goal: 0x1f84bd,
  selected: 0xf2c14e,
};

export class RobotViewer {
  // container: the element to render into.
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xeef0f1);

    this.camera = new THREE.PerspectiveCamera(40, 1, 0.05, 100);
    // The robot's world is Z-up, and three.js defaults to Y-up. Telling the
    // camera which way is up is the whole fix; the geometry stays as exported.
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(2.6, -2.6, 2.0);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0, 0.8);
    this.controls.enableDamping = true;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 2.0));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(3, -4, 6);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xfff0e0, 1.0);
    fill.position.set(-4, 3, 2);
    this.scene.add(fill);

    this.scene.add(new THREE.GridHelper(6, 12, 0xc8ccce, 0xdfe3e5)
      .rotateX(Math.PI / 2));

    this.robotRoot = null;
    this.kinematics = null;
    this.carriedBar = null;
    this.poseClouds = [];
    this.trajectoryLine = null;
    this.treeObjects = [];
    // Extra lines drawn over the scene (the failed attempts), and the parts
    // currently tinted to point at a collision.
    this.overlays = [];
    this.highlighted = [];

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.onPickCallback = null;
    this.renderer.domElement.addEventListener("click", (event) => this.handleClick(event));

    this.resize();
    // Follow the box, not just the window. In the split layout the viewer's
    // height changes whenever the controls under it change, and a canvas
    // sized once would then be squashed or cropped.
    new ResizeObserver(() => this.resize()).observe(container);
    this.animate();
  }

  resize() {
    const width = this.container.clientWidth || 640;
    const height = this.container.clientHeight || 420;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  // Loads the robot once. Later calls resolve immediately.
  async loadRobot(glbUrl, kinematicsUrl) {
    if (this.robotRoot) return;
    const [gltf, kinematics] = await Promise.all([
      new Promise((resolve, reject) => {
        new GLTFLoader().load(glbUrl, resolve, undefined, reject);
      }),
      fetch(kinematicsUrl).then((response) => {
        if (!response.ok) throw new Error(`kinematics.json: HTTP ${response.status}`);
        return response.json();
      }),
    ]);
    this.robotRoot = gltf.scene;
    this.scene.add(this.robotRoot);
    this.kinematicsSpec = kinematics;
    this.kinematics = new RobotKinematics(kinematics, this.robotRoot);
    this.pose(new Array(this.kinematics.joints.length).fill(0));
  }

  // A second, see-through robot, for showing one configuration next to
  // another: the rejected waypoint of a continuity stop beside the accepted
  // one it was supposed to follow. Built once from a copy of the robot; the
  // joint rest rotations come from kinematics.json, so the copy poses
  // exactly like the original regardless of how the original stands.
  showGhost(angles, colour = 0xd98324) {
    if (!this.robotRoot) return;
    if (!this.ghostRoot) {
      this.ghostRoot = this.robotRoot.clone(true);
      this.ghostRoot.traverse((child) => {
        if (!child.isMesh) return;
        child.material = new THREE.MeshStandardMaterial({
          color: colour, transparent: true, opacity: 0.38, depthWrite: false,
          roughness: 0.6, metalness: 0.0,
        });
      });
      this.ghostKinematics = new RobotKinematics(this.kinematicsSpec, this.ghostRoot);
      this.scene.add(this.ghostRoot);
    }
    this.ghostRoot.traverse((child) => {
      if (child.isMesh) child.material.color.setHex(colour);
    });
    this.ghostRoot.visible = true;
    this.ghostKinematics.pose(angles);
  }

  hideGhost() {
    if (this.ghostRoot) this.ghostRoot.visible = false;
  }

  // Recolours one bar of an instanced cloud, e.g. to mark the selected one.
  setInstanceColour(mesh, index, hex) {
    if (!mesh || index == null) return;
    mesh.setColorAt(index, new THREE.Color(hex));
    mesh.instanceColor.needsUpdate = true;
  }

  pose(angles) {
    if (!this.kinematics) return;
    this.kinematics.pose(angles);
    if (this.carriedBar) this.updateCarriedBar();
  }

  // Attaches a bar to a tool flange, so it follows the arm when posed.
  //
  // grasp is the flange's pose in the bar's own frame, as the benchmark
  // stores it: [[x, y, z], [qx, qy, qz, qw]]. Inverting it gives where the bar
  // sits relative to the flange.
  setCarriedBar({ length, radius, linkName, grasp, bar_from_midpoint }) {
    if (this.carriedBar) {
      this.scene.remove(this.carriedBar);
      this.carriedBar.geometry.dispose();
    }
    const geometry = barGeometry(length, radius, bar_from_midpoint);
    const material = new THREE.MeshStandardMaterial({
      color: OUTCOME_COLOURS.goal, roughness: 0.45, metalness: 0.1,
    });
    this.carriedBar = new THREE.Mesh(geometry, material);
    this.carriedBar.matrixAutoUpdate = false;
    this.scene.add(this.carriedBar);

    const flangeInBar = poseToMatrix(grasp);
    this.barInFlange = new THREE.Matrix4().copy(flangeInBar).invert();
    this.carriedBarLink = linkName;
    this.updateCarriedBar();
  }

  updateCarriedBar() {
    const flange = this.kinematics.linkMatrix(this.carriedBarLink);
    if (!flange) return;
    this.carriedBar.matrix.multiplyMatrices(flange, this.barInFlange);
  }

  clearCarriedBar() {
    if (!this.carriedBar) return;
    this.scene.remove(this.carriedBar);
    this.carriedBar.geometry.dispose();
    this.carriedBar = null;
  }

  // Draws many bar poses at once, coloured by outcome.
  //
  // poses: [{ world_from_bar, colour, id }]
  addPoseCloud(poses, { length, radius, bar_from_midpoint, opacity = 0.85 }) {
    if (!poses.length) return null;
    const geometry = barGeometry(length, radius, bar_from_midpoint);
    const material = new THREE.MeshStandardMaterial({
      roughness: 0.6, metalness: 0.0, transparent: opacity < 1, opacity,
    });
    const mesh = new THREE.InstancedMesh(geometry, material, poses.length);
    mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
    const colour = new THREE.Color();
    poses.forEach((pose, index) => {
      mesh.setMatrixAt(index, poseToMatrix(pose.world_from_bar));
      colour.setHex(pose.colour);
      mesh.setColorAt(index, colour);
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.userData.ids = poses.map((pose) => pose.id);
    this.scene.add(mesh);
    this.poseClouds.push(mesh);
    return mesh;
  }

  clearPoseClouds() {
    for (const mesh of this.poseClouds) {
      this.scene.remove(mesh);
      mesh.geometry.dispose();
      mesh.dispose();
    }
    this.poseClouds = [];
  }

  // Removes one cloud, leaving the others. Used when a toggle hides one kind
  // of marker without redrawing everything else.
  removePoseCloud(mesh) {
    if (!mesh) return;
    const index = this.poseClouds.indexOf(mesh);
    if (index < 0) return;
    this.poseClouds.splice(index, 1);
    this.scene.remove(mesh);
    mesh.geometry.dispose();
    mesh.dispose();
  }

  // Draws straight segments over the scene: segments is [[[x,y,z], [x,y,z]], ...].
  // Returns the object, so it can be removed on its own with removeOverlay.
  addLines(segments, { colour, opacity = 0.6 }) {
    if (!segments.length) return null;
    const positions = [];
    for (const [p, q] of segments) positions.push(p[0], p[1], p[2], q[0], q[1], q[2]);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    const lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
      color: colour, transparent: opacity < 1, opacity,
    }));
    this.scene.add(lines);
    this.overlays.push(lines);
    return lines;
  }

  removeOverlay(object) {
    if (!object) return;
    const index = this.overlays.indexOf(object);
    if (index >= 0) this.overlays.splice(index, 1);
    this.scene.remove(object);
    object.geometry.dispose();
    object.material.dispose();
  }

  clearOverlays() {
    for (const object of this.overlays.slice()) this.removeOverlay(object);
  }

  // Tints robot links so a collision pair stands out. Names are link names as
  // the planner reports them; the exported robot has a node "L_<name>" per
  // link. Returns the names that are not links, so the caller can decide
  // those are the bar or a tool and tint the carried bar instead.
  highlightParts(names, colour = 0xb23a48) {
    this.clearHighlight();
    const missing = [];
    for (const name of names) {
      const node = this.robotRoot && findNode(this.robotRoot, `L_${name}`);
      if (!node) { missing.push(name); continue; }
      node.traverse((child) => {
        if (!child.isMesh) return;
        this.highlighted.push({ mesh: child, material: child.material });
        child.material = child.material.clone();
        child.material.emissive = new THREE.Color(colour);
        child.material.emissiveIntensity = 0.55;
      });
    }
    return missing;
  }

  highlightCarriedBar(colour = 0xb23a48) {
    if (!this.carriedBar) return;
    this.highlighted.push({ mesh: this.carriedBar, material: this.carriedBar.material });
    this.carriedBar.material = this.carriedBar.material.clone();
    this.carriedBar.material.emissive = new THREE.Color(colour);
    this.carriedBar.material.emissiveIntensity = 0.55;
  }

  clearHighlight() {
    for (const { mesh, material } of this.highlighted) {
      mesh.material.dispose();
      mesh.material = material;
    }
    this.highlighted = [];
  }

  // Draws the two trees a search grew, as the paths of the bar's midpoint.
  //
  // Each tree is { midpoints: [[x, y, z]...], edges: [[parent, child]...] }.
  // The start tree and the goal tree take the colours of the start and goal
  // bars, so it is plain which end each branch grew from; nodes are dots, so
  // a tree that barely left its root is still visible.
  setTrees(trees) {
    this.clearTrees();
    for (const { tree, colour } of trees) {
      if (!tree || !tree.midpoints || !tree.midpoints.length) continue;
      const positions = [];
      for (const [a, b] of tree.edges || []) {
        const p = tree.midpoints[a];
        const q = tree.midpoints[b];
        if (p && q) positions.push(p[0], p[1], p[2], q[0], q[1], q[2]);
      }
      if (positions.length) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position",
          new THREE.Float32BufferAttribute(positions, 3));
        const lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
          color: colour, transparent: true, opacity: 0.75,
        }));
        this.scene.add(lines);
        this.treeObjects.push(lines);
      }
      const nodes = new THREE.BufferGeometry();
      nodes.setAttribute("position",
        new THREE.Float32BufferAttribute(tree.midpoints.flat(), 3));
      const dots = new THREE.Points(nodes, new THREE.PointsMaterial({
        color: colour, size: 0.012,
      }));
      this.scene.add(dots);
      this.treeObjects.push(dots);
    }
  }

  clearTrees() {
    for (const object of this.treeObjects) {
      this.scene.remove(object);
      object.geometry.dispose();
      object.material.dispose();
    }
    this.treeObjects = [];
  }

  // Draws the path the bar's midpoint takes along a trajectory.
  setTrajectoryLine(points) {
    if (this.trajectoryLine) {
      this.scene.remove(this.trajectoryLine);
      this.trajectoryLine.geometry.dispose();
      this.trajectoryLine = null;
    }
    if (!points || points.length < 2) return;
    const geometry = new THREE.BufferGeometry().setFromPoints(
      points.map((p) => new THREE.Vector3(p[0], p[1], p[2])));
    this.trajectoryLine = new THREE.Line(
      geometry, new THREE.LineBasicMaterial({ color: OUTCOME_COLOURS.selected }));
    this.scene.add(this.trajectoryLine);
  }

  onPick(callback) {
    this.onPickCallback = callback;
  }

  handleClick(event) {
    if (!this.onPickCallback || !this.poseClouds.length) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.poseClouds, false);
    if (!hits.length) return;
    const hit = hits[0];
    const ids = hit.object.userData.ids || [];
    if (hit.instanceId != null && ids[hit.instanceId] != null) {
      this.onPickCallback(ids[hit.instanceId]);
    }
  }
}

// A cylinder standing along the bar's own long axis.
//
// three.js cylinders are built along Y and centred, while the exported bars
// run along Z with their origin at one end, so the geometry is rotated and
// shifted to match before anything is ever positioned.
function barGeometry(length, radius, barFromMidpoint) {
  const geometry = new THREE.CylinderGeometry(radius, radius, length, 12, 1);
  geometry.rotateX(Math.PI / 2);
  const midpoint = barFromMidpoint || [0, 0, length / 2];
  geometry.translate(midpoint[0], midpoint[1], midpoint[2]);
  return geometry;
}

// Turns a stored [[x, y, z], [qx, qy, qz, qw]] pose into a matrix.
export function poseToMatrix(pose) {
  const [position, quaternion] = pose;
  return new THREE.Matrix4().compose(
    new THREE.Vector3(position[0], position[1], position[2]),
    new THREE.Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3]),
    new THREE.Vector3(1, 1, 1),
  );
}
