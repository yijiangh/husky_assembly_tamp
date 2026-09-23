// Posing the robot in the browser.
//
// The exported scene arranges the robot's geometry as its own joint tree: each
// link is a node sitting at a fixed offset from its parent. That means posing
// the robot is just setting the rotation of the twelve nodes that correspond
// to arm joints, and letting the renderer work out the rest. Nothing here has
// to touch geometry, and a trajectory is only a list of joint angles rather
// than a matrix for every link of every frame.
//
// Verified against the simulator's own forward kinematics to within about
// 3e-7 metres; see the benchmark's fk_check.

import * as THREE from "three";

// Reused across calls so posing a robot allocates nothing.
const scratchQuaternion = new THREE.Quaternion();
const scratchAxis = new THREE.Vector3();

export class RobotKinematics {
  // Binds a kinematics description to the nodes of a loaded scene.
  //
  // kinematics: the parsed kinematics.json.
  // root: the THREE.Object3D the scene was loaded into.
  constructor(kinematics, root) {
    this.jointOrder = kinematics.joint_order;
    this.toolLinks = kinematics.tool_links || {};
    this.joints = [];

    for (const joint of kinematics.joints) {
      const node = findNode(root, joint.node);
      if (!node) {
        throw new Error(
          `the scene has no node called ${joint.node}, so joint ` +
          `${joint.name} cannot be moved`
        );
      }
      // The node's rest rotation is the joint's fixed offset. Posing composes
      // the joint angle onto it, so the rest value has to be kept.
      this.joints.push({
        name: joint.name,
        node,
        axis: new THREE.Vector3(joint.axis[0], joint.axis[1], joint.axis[2]).normalize(),
        restQuaternion: new THREE.Quaternion(
          joint.rotation[0], joint.rotation[1], joint.rotation[2], joint.rotation[3]
        ),
        lower: joint.lower,
        upper: joint.upper,
      });
    }
    this.root = root;
  }

  // Poses the robot. `angles` is twelve numbers in the exported joint order:
  // the left arm's six, then the right arm's six.
  pose(angles) {
    if (!angles || angles.length !== this.joints.length) {
      throw new Error(
        `expected ${this.joints.length} joint angles, got ` +
        `${angles ? angles.length : 0}`
      );
    }
    for (let i = 0; i < this.joints.length; i += 1) {
      const joint = this.joints[i];
      scratchAxis.copy(joint.axis);
      scratchQuaternion.setFromAxisAngle(scratchAxis, angles[i]);
      joint.node.quaternion.copy(joint.restQuaternion).multiply(scratchQuaternion);
    }
    this.root.updateMatrixWorld(true);
  }

  // The world matrix of one link, after posing. Used to hang the carried bar
  // off a tool flange.
  linkMatrix(linkName) {
    const node = findNode(this.root, `L_${linkName}`);
    return node ? node.matrixWorld : null;
  }
}

// Finds a node by name, allowing for the loader having rewritten it.
//
// The glTF loader strips characters it uses for animation paths (dots, slashes
// and brackets) out of node names. The exporter avoids those characters, but
// the fallback costs nothing and turns a silent blank screen into a match.
export function findNode(root, name) {
  let node = root.getObjectByName(name);
  if (node) return node;
  const sanitized = name.replace(/[\s.:/[\]]/g, "_");
  return root.getObjectByName(sanitized) || null;
}
