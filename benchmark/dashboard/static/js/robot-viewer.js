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
import { RobotKinematics } from "./kinematics.js";

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
    this.kinematics = new RobotKinematics(kinematics, this.robotRoot);
    this.pose(new Array(this.kinematics.joints.length).fill(0));
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
