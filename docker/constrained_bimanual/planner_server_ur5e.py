"""
Constrained bimanual planner server for dual UR5e arms on Husky.

Adapted from the IIWA 7-DOF version. Key differences:
- UR5e is 6-DOF: no continuous self-motion parameter (psi)
- Up to 8 discrete IK solution branches per target pose
- Planning happens in 6D (left arm joints); right arm is resolved via
  analytical IK with branch selection at each configuration

The planner:
1. Plans in 6D (left arm joint space)
2. At each sampled configuration, tries all IK branches for the right arm
3. Selects the best collision-free branch (closest to previous config)
4. Outputs both 6D and 12D paths

Usage (from inside container):
    cd /opt/proj
    python host_scripts/planner_server_ur5e.py
"""

import json
import os
import sys
import time

import numpy as np

# -- Drake imports (available in the container) --
from pydrake.all import (
    CollisionCheckerParams,
    Parser,
    RigidTransform,
    RobotDiagramBuilder,
    SceneGraphCollisionChecker,
)

# -- Repo imports (available at /opt/proj) --
sys.path.insert(0, "/opt/proj")
sys.path.insert(0, "/opt/proj/host_scripts")
from src.rrt import BiRRT, RRTOptions
from src.shortcut import shortcut

# Local UR5e analytical IK wrapper
from ur5e_analytic_ik import AnalyticIK_UR5e, UR5E_LIMITS_LOWER, UR5E_LIMITS_UPPER

EXCHANGE_DIR = "/opt/exchange"
REQUEST_FILE = f"{EXCHANGE_DIR}/request.json"
RESPONSE_FILE = f"{EXCHANGE_DIR}/response.json"

# ROS package directories mounted into the container
ROS_PACKAGES_DIR = "/opt/ros_packages"


def setup_scene(extra_collision_objects=None):
    """
    Build the Drake collision checker with the Husky dual UR5e robot.

    The full Husky URDF is loaded (with 45-degree tilted arm mounts already
    encoded in the fixed joints). Only the 12 arm joints are configured as
    planning DOFs; all other joints (wheels, etc.) are locked.

    Args:
        extra_collision_objects: optional list of dicts describing additional
            collision objects to add (from scene_exporter). Each dict has:
            {type: "box"|"cylinder"|"sphere", dims: [...], pose: 4x4, name: str}
    """
    params = CollisionCheckerParams()
    builder = RobotDiagramBuilder(time_step=0.0)
    plant = builder.plant()
    parser = Parser(plant)

    # Register all ROS packages so Drake can resolve package:// URIs
    for pkg_name in os.listdir(ROS_PACKAGES_DIR):
        pkg_path = os.path.join(ROS_PACKAGES_DIR, pkg_name)
        if os.path.isdir(pkg_path):
            pkg_xml = os.path.join(pkg_path, "package.xml")
            if os.path.exists(pkg_xml):
                parser.package_map().AddPackageXml(pkg_xml)
            else:
                # Some packages don't have package.xml; register by name
                parser.package_map().Add(pkg_name, pkg_path)

    # Load the full Husky + dual UR5e URDF
    urdf_path = os.path.join(
        ROS_PACKAGES_DIR,
        "mt_husky_dual_ur5_e_moveit_config",
        "urdf",
        "husky_dual_ur5_e_no_base_joint_All_Calibrated.urdf",
    )
    model_instances = parser.AddModels(urdf_path)
    husky_instance = model_instances[0]

    # Weld world_link to Drake's world frame (prevents floating base)
    world_body = plant.GetBodyByName("world_link", husky_instance)
    plant.WeldFrames(
        plant.world_frame(),
        world_body.body_frame(),
        RigidTransform(),
    )

    # Add extra collision objects (environment obstacles from pybullet)
    if extra_collision_objects:
        _add_collision_objects(plant, extra_collision_objects)

    # The robot model instance contains all joints; we mark it for planning
    params.robot_model_instances = [husky_instance]

    plant.Finalize()
    params.model = builder.Build()
    params.edge_step_size = 0.01
    checker = SceneGraphCollisionChecker(params)

    # Identify the arm joint indices in the plant's position vector
    arm_joint_names_left = [
        "left_ur_arm_shoulder_pan_joint",
        "left_ur_arm_shoulder_lift_joint",
        "left_ur_arm_elbow_joint",
        "left_ur_arm_wrist_1_joint",
        "left_ur_arm_wrist_2_joint",
        "left_ur_arm_wrist_3_joint",
    ]
    arm_joint_names_right = [
        "right_ur_arm_shoulder_pan_joint",
        "right_ur_arm_shoulder_lift_joint",
        "right_ur_arm_elbow_joint",
        "right_ur_arm_wrist_1_joint",
        "right_ur_arm_wrist_2_joint",
        "right_ur_arm_wrist_3_joint",
    ]

    # Get position indices for the arm joints
    left_indices = []
    for name in arm_joint_names_left:
        joint = plant.GetJointByName(name)
        # For revolute joints, position_start() gives the index
        left_indices.append(joint.position_start())

    right_indices = []
    for name in arm_joint_names_right:
        joint = plant.GetJointByName(name)
        right_indices.append(joint.position_start())

    total_positions = plant.num_positions()

    return checker, left_indices, right_indices, total_positions


def _add_collision_objects(plant, objects):
    """Add static collision objects (boxes, cylinders, spheres) to the plant."""
    from pydrake.all import (
        Box,
        Cylinder,
        Sphere,
        CoulombFriction,
        ProximityProperties,
    )
    from pydrake.math import RigidTransform as RT, RollPitchYaw

    for i, obj in enumerate(objects):
        name = obj.get("name", f"obstacle_{i}")
        obj_type = obj["type"]
        pose = np.array(obj["pose"])

        # Create rigid body welded to world
        body = plant.AddRigidBody(
            name,
            plant.world_frame().body().model_instance(),
        )

        # Create shape
        if obj_type == "box":
            shape = Box(*obj["dims"])
        elif obj_type == "cylinder":
            shape = Cylinder(obj["dims"][0], obj["dims"][1])
        elif obj_type == "sphere":
            shape = Sphere(obj["dims"][0])
        else:
            print(f"Warning: Unknown collision object type '{obj_type}', skipping")
            continue

        # Register collision geometry
        proximity_props = ProximityProperties()
        proximity_props.AddProperty("material", "coulomb_friction",
                                    CoulombFriction(1.0, 1.0))
        plant.RegisterCollisionGeometry(
            body, RT(pose), shape, name + "_collision", proximity_props
        )

        # Weld to world at the specified pose
        plant.WeldFrames(
            plant.world_frame(),
            body.body_frame(),
            RT(pose),
        )


def build_parameterization(
    grasp_distance,
    grasp_angle_deg,
    base_offset,
    left_indices,
    right_indices,
    total_positions,
):
    """
    Build the parameterization that maps 6D left arm config -> 12D full config.

    For UR5e (6-DOF), there is no continuous self-motion parameter. Instead,
    we select from discrete IK solution branches. The validity checker tries
    all branches and picks the best collision-free one.

    Args:
        grasp_distance: distance between gripper frames along grasped object
        grasp_angle_deg: gripper rotation angle (degrees)
        base_offset: 3D offset from left arm base to right arm base [x, y, z]
        left_indices: position indices for left arm joints in Drake plant
        right_indices: position indices for right arm joints in Drake plant
        total_positions: total number of positions in Drake plant
    """
    ik = AnalyticIK_UR5e()

    # Precompute rotation matrices for grasp constraint
    ang = (180 - 2.0 * grasp_angle_deg) * np.pi / 180.0
    c, s = np.cos(ang), np.sin(ang)
    rot_flip = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
    rot_angle = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    offset_dir = np.array([0, 0, -grasp_distance])
    base_off = np.array(base_offset)

    def q_to_ee_target(q_left):
        """Compute right arm EE target from left arm config via grasp constraint."""
        tf = ik.FK(q_left).copy()
        tf[:3, :3] = tf[:3, :3] @ rot_flip @ rot_angle
        tf[:3, 3] = tf[:3, 3] + tf[:3, :3] @ offset_dir + base_off
        return tf

    def parameterization(q_left, q_right):
        """Map 6D left + 6D right -> full Drake position vector."""
        q_full = np.zeros(total_positions)
        for i, idx in enumerate(left_indices):
            q_full[idx] = q_left[i]
        for i, idx in enumerate(right_indices):
            q_full[idx] = q_right[i]
        return q_full

    def find_valid_branches(q_left, q_right_prev=None):
        """
        Find all valid IK branches for right arm given left arm config.

        Returns:
            List of (q_right, branch_index) tuples, sorted by closeness
            to q_right_prev if provided.
        """
        try:
            tf_goal = q_to_ee_target(q_left)
        except Exception:
            return []

        solutions = ik.IK_all_with_branches(tf_goal)
        if not solutions:
            return []

        if q_right_prev is not None:
            # Sort by distance to previous config for branch consistency
            solutions.sort(key=lambda s: np.linalg.norm(s[0] - q_right_prev))

        return solutions

    def is_reachable(q_left):
        """Check if any IK solution exists for the right arm."""
        try:
            tf_goal = q_to_ee_target(q_left)
            solutions = ik.IK_all(tf_goal)
            return len(solutions) > 0
        except Exception:
            return False

    return ik, parameterization, find_valid_branches, is_reachable, q_to_ee_target


def plan(request, checker, left_indices, right_indices, total_positions):
    """
    Run BiRRT planner with the constrained bimanual parameterization for UR5e.

    Request fields:
        start_config_left_6d:  6D left arm start config
        start_config_right_6d: 6D right arm start config
        goal_config_left_6d:   6D left arm goal config
        goal_config_right_6d:  6D right arm goal config
        grasp_distance:        distance between grippers along grasped object
        grasp_angle_deg:       (optional) gripper angle, default 68.0
        base_offset:           (optional) right arm base offset from left [x,y,z]
        rrt_step_size:         (optional) RRT step size, default 0.2
        rrt_max_iters:         (optional) max RRT iterations, default 100000
        rrt_timeout:           (optional) timeout in seconds, default 60
        shortcut_tries:        (optional) shortcut iterations, default 100
    """
    start_left = np.array(request["start_config_left_6d"])
    start_right = np.array(request["start_config_right_6d"])
    goal_left = np.array(request["goal_config_left_6d"])
    goal_right = np.array(request["goal_config_right_6d"])
    grasp_distance = request["grasp_distance"]
    grasp_angle_deg = request.get("grasp_angle_deg", 68.0)

    # Base offset: right arm base relative to left arm base
    # Default computed from URDF: left at y=+0.14891, right at y=-0.14891
    # In the arm's own frame, this offset depends on the mounting geometry.
    # This should be measured or computed from the URDF transforms.
    base_offset = request.get("base_offset", [0, -0.2978, 0])

    # Load extra collision objects if provided
    extra_objects = request.get("collision_objects", None)

    ik, parameterization_fn, find_valid_branches, is_reachable, _ = (
        build_parameterization(
            grasp_distance, grasp_angle_deg, base_offset,
            left_indices, right_indices, total_positions,
        )
    )

    # Domain bounds: UR5e joint limits for left arm (planning space)
    domain_lower = UR5E_LIMITS_LOWER.copy()
    domain_upper = UR5E_LIMITS_UPPER.copy()

    def random_config():
        return np.random.uniform(low=domain_lower, high=domain_upper)

    # Track the best right-arm config at each visited node for branch consistency
    # We use a simple cache keyed by the left-arm config (discretized)
    right_arm_cache = {}

    def get_cache_key(q_left):
        return tuple(np.round(q_left, decimals=4))

    # Store initial right arm configs
    right_arm_cache[get_cache_key(start_left)] = start_right
    right_arm_cache[get_cache_key(goal_left)] = goal_right

    def validity_checker(q_left):
        """Check if q_left leads to any valid, collision-free full config."""
        if not is_reachable(q_left):
            return False

        # Find the closest cached right-arm config for branch consistency
        key = get_cache_key(q_left)
        q_right_prev = right_arm_cache.get(key)

        # Try all branches, prefer closest to previous
        branches = find_valid_branches(q_left, q_right_prev)
        for q_right, branch_idx in branches:
            q_full = parameterization_fn(q_left, q_right)
            if checker.CheckConfigCollisionFree(q_full):
                # Cache the successful right-arm config
                right_arm_cache[key] = q_right
                return True

        return False

    # RRT options
    step_size = request.get("rrt_step_size", 0.2)
    max_iters = int(request.get("rrt_max_iters", 100000))
    timeout = request.get("rrt_timeout", 60.0)

    options = RRTOptions(
        step_size=step_size,
        check_size=0.01,
        max_vertices=int(1e4),
        max_iters=max_iters,
        goal_sample_frequency=0.05,
        timeout=timeout,
    )

    print(f"Planning with BiRRT (step={step_size}, max_iters={max_iters}, timeout={timeout}s)...")
    t0 = time.time()
    rrt = BiRRT(random_config, validity_checker)
    path_6d = rrt.plan(start_left, goal_left, options)
    planning_time = time.time() - t0

    if len(path_6d) == 0:
        return {
            "success": False,
            "path_left_6d": [],
            "path_right_6d": [],
            "path_12d": [],
            "planning_time": planning_time,
            "shortcut_time": 0.0,
            "message": "BiRRT failed to find a path",
        }

    print(f"Raw path: {len(path_6d)} waypoints in {planning_time:.3f}s")

    # Shortcut
    shortcut_tries = int(request.get("shortcut_tries", 100))
    t1 = time.time()
    path_6d = shortcut(
        path_6d, validity_checker, num_tries=shortcut_tries, check_size=0.01
    )
    shortcut_time = time.time() - t1

    # Resolve right arm configs for the final path with branch consistency
    path_left = []
    path_right = []
    path_12d = []
    q_right_prev = start_right

    for q_left in path_6d:
        branches = find_valid_branches(q_left, q_right_prev)
        # Pick closest collision-free branch
        q_right_best = None
        for q_right, _ in branches:
            q_full = parameterization_fn(q_left, q_right)
            if checker.CheckConfigCollisionFree(q_full):
                q_right_best = q_right
                break

        if q_right_best is None:
            # Fallback: use cached value
            key = get_cache_key(q_left)
            q_right_best = right_arm_cache.get(key, q_right_prev)

        path_left.append(q_left.tolist())
        path_right.append(q_right_best.tolist())
        path_12d.append(q_left.tolist() + q_right_best.tolist())
        q_right_prev = q_right_best

    print(f"Shortcut path: {len(path_6d)} waypoints, shortcut took {shortcut_time:.3f}s")

    return {
        "success": True,
        "path_left_6d": path_left,
        "path_right_6d": path_right,
        "path_12d": path_12d,
        "planning_time": planning_time,
        "shortcut_time": shortcut_time,
        "num_waypoints": len(path_6d),
        "message": "OK",
    }


def main():
    print("Setting up Drake scene with Husky dual UR5e...")

    # Check if extra collision objects are in the request
    extra_objects = None
    if os.path.exists(REQUEST_FILE):
        with open(REQUEST_FILE, "r") as f:
            request = json.load(f)
        extra_objects = request.get("collision_objects", None)
    else:
        request = None

    checker, left_indices, right_indices, total_positions = setup_scene(extra_objects)
    print(f"Scene ready. Total positions: {total_positions}")
    print(f"Left arm indices: {left_indices}")
    print(f"Right arm indices: {right_indices}")

    if request is None:
        print(f"No request file found at {REQUEST_FILE}")
        return

    result = plan(request, checker, left_indices, right_indices, total_positions)

    with open(RESPONSE_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Response written to {RESPONSE_FILE}")
    print(f"  success: {result['success']}")
    print(f"  waypoints: {result.get('num_waypoints', 0)}")
    print(f"  planning_time: {result['planning_time']:.3f}s")


if __name__ == "__main__":
    main()
