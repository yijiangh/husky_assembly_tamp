"""
Standalone example for the constrained bimanual planner with dual UR5e on Husky.
Runs inside the Docker container without needing Jupyter.

Usage (from host):
    docker exec -it constrained-bimanual-planner \
        python3 /opt/proj/host_scripts/run_example_ur5e.py [--meshcat]

This script:
1. Sets up the Drake scene with the Husky dual UR5e robot
2. Verifies the analytical IK wrapper works correctly
3. Runs a constrained bimanual BiRRT plan
4. Optionally visualizes via Meshcat
"""

import sys
import os
import time

import numpy as np

sys.path.insert(0, "/opt/proj")
sys.path.insert(0, "/opt/proj/host_scripts")

from pydrake.all import (
    StartMeshcat,
    CollisionCheckerParams,
    RobotDiagramBuilder,
    MeshcatVisualizerParams,
    Role,
    MeshcatVisualizer,
    Parser,
    RigidTransform,
    SceneGraphCollisionChecker,
)

import src.rrt as rrt
import src.shortcut as shortcut_mod
from ur5e_analytic_ik import AnalyticIK_UR5e, UR5E_LIMITS_LOWER, UR5E_LIMITS_UPPER

# ============================================================
# Configuration
# ============================================================
grasp_distance = 0.3  # Distance between grippers along grasped bar
grasp_angle_deg = 68.0

# Example UR5e home-ish configs (will need adjustment for actual workspace)
# These are placeholder configs - adjust to your actual start/goal
q_left_start = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
q_left_goal = np.array([0.5, -1.2, 1.2, -1.0, -1.57, 0.3])

USE_MESHCAT = "--meshcat" in sys.argv

ROS_PACKAGES_DIR = "/opt/ros_packages"

# ============================================================
# Scene setup
# ============================================================
print("Setting up Drake scene with Husky dual UR5e...")
t0 = time.time()

if USE_MESHCAT:
    meshcat = StartMeshcat()
    print(f"Meshcat URL: {meshcat.web_url()}")

params = CollisionCheckerParams()
builder = RobotDiagramBuilder(time_step=0.0)

if USE_MESHCAT:
    meshcat_visual_params = MeshcatVisualizerParams()
    meshcat_visual_params.delete_on_initialization_event = False
    meshcat_visual_params.role = Role.kIllustration
    meshcat_visual_params.prefix = "visual"
    MeshcatVisualizer.AddToBuilder(
        builder.builder(), builder.scene_graph(), meshcat, meshcat_visual_params
    )

plant = builder.plant()
parser = Parser(plant)

# Register ROS packages
for pkg_name in os.listdir(ROS_PACKAGES_DIR):
    pkg_path = os.path.join(ROS_PACKAGES_DIR, pkg_name)
    if os.path.isdir(pkg_path):
        pkg_xml = os.path.join(pkg_path, "package.xml")
        if os.path.exists(pkg_xml):
            parser.package_map().AddPackageXml(pkg_xml)
        else:
            parser.package_map().Add(pkg_name, pkg_path)

# Load Husky dual UR5e URDF
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

params.robot_model_instances = [husky_instance]
plant.Finalize()

builder.builder().ExportInput(plant.get_actuation_input_port(), "actuation")
builder.builder().ExportOutput(plant.get_state_output_port(), "state")

diagram = builder.Build()
params.model = diagram
params.edge_step_size = 0.01
checker = SceneGraphCollisionChecker(params)

# Get arm joint indices
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

left_indices = [plant.GetJointByName(n).position_start() for n in arm_joint_names_left]
right_indices = [plant.GetJointByName(n).position_start() for n in arm_joint_names_right]
total_positions = plant.num_positions()

print(f"Scene setup: {time.time() - t0:.2f}s")
print(f"Total plant positions: {total_positions}")
print(f"Left arm indices: {left_indices}")
print(f"Right arm indices: {right_indices}")

# ============================================================
# Test analytical IK
# ============================================================
print("\n" + "=" * 50)
print("Testing UR5e analytical IK...")
print("=" * 50)

ik = AnalyticIK_UR5e()

# Test FK
pose_start = ik.FK(q_left_start)
print(f"\nFK of start config:")
print(f"  Position: {pose_start[:3, 3]}")

# Test IK roundtrip
solutions = ik.IK_all(pose_start)
print(f"\nIK solutions for start pose: {len(solutions)} found")
for i, sol in enumerate(solutions):
    # Verify FK(IK(pose)) ≈ pose
    pose_check = ik.FK(sol)
    pos_err = np.linalg.norm(pose_check[:3, 3] - pose_start[:3, 3])
    rot_err = np.linalg.norm(pose_check[:3, :3] - pose_start[:3, :3])
    print(f"  Branch {i}: pos_err={pos_err:.6f}m, rot_err={rot_err:.6f}")

# ============================================================
# Build parameterization
# ============================================================
print("\n" + "=" * 50)
print("Building constrained parameterization...")
print("=" * 50)

# Grasp constraint geometry
ang = (180 - 2.0 * grasp_angle_deg) * np.pi / 180.0
c, s = np.cos(ang), np.sin(ang)
rot_flip = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
rot_angle = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
offset_dir = np.array([0, 0, -grasp_distance])
# Base offset: approximate from URDF (left y=+0.14891, right y=-0.14891 relative to bulkhead)
base_offset = np.array([0, -0.2978, 0])


def q_to_ee_target(q_left):
    """Compute right arm EE target from left arm config."""
    tf = ik.FK(q_left).copy()
    tf[:3, :3] = tf[:3, :3] @ rot_flip @ rot_angle
    tf[:3, 3] = tf[:3, 3] + tf[:3, :3] @ offset_dir + base_offset
    return tf


def make_full_config(q_left, q_right):
    """Create full Drake position vector from arm configs."""
    q_full = np.zeros(total_positions)
    for i, idx in enumerate(left_indices):
        q_full[idx] = q_left[i]
    for i, idx in enumerate(right_indices):
        q_full[idx] = q_right[i]
    return q_full


# Find right arm config for start
tf_right_start = q_to_ee_target(q_left_start)
right_solutions = ik.IK_all(tf_right_start)
print(f"\nRight arm IK solutions at start: {len(right_solutions)}")

if len(right_solutions) == 0:
    print("ERROR: No IK solution for right arm at start config!")
    print("The start config may not be compatible with the grasp constraint.")
    print("Try adjusting q_left_start or grasp parameters.")
    sys.exit(1)

# Pick first valid solution
q_right_start = right_solutions[0]
print(f"Using right arm branch 0: {q_right_start}")

# Verify collision-free
q_full_start = make_full_config(q_left_start, q_right_start)
start_free = checker.CheckConfigCollisionFree(q_full_start)
print(f"Start config collision-free: {start_free}")

if USE_MESHCAT:
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, q_full_start)
    diagram.ForcedPublish(context)
    print(f"\nStart config shown in Meshcat at {meshcat.web_url()}")
    print("Check the visualization before continuing.")
    input("Press Enter to continue...")

# Find right arm config for goal
tf_right_goal = q_to_ee_target(q_left_goal)
right_solutions_goal = ik.IK_all(tf_right_goal)
print(f"\nRight arm IK solutions at goal: {len(right_solutions_goal)}")

if len(right_solutions_goal) == 0:
    print("ERROR: No IK solution for right arm at goal config!")
    sys.exit(1)

q_right_goal = right_solutions_goal[0]
q_full_goal = make_full_config(q_left_goal, q_right_goal)
goal_free = checker.CheckConfigCollisionFree(q_full_goal)
print(f"Goal config collision-free: {goal_free}")

if not start_free or not goal_free:
    print("WARNING: Start or goal in collision. Planning may fail.")

# ============================================================
# Validity checker for BiRRT (plans in 6D left arm space)
# ============================================================
print("\n" + "=" * 50)
print("Setting up validity checker...")
print("=" * 50)

domain_lower = UR5E_LIMITS_LOWER.copy()
domain_upper = UR5E_LIMITS_UPPER.copy()

right_arm_cache = {}
right_arm_cache[tuple(np.round(q_left_start, 4))] = q_right_start
right_arm_cache[tuple(np.round(q_left_goal, 4))] = q_right_goal


def RandomConfig():
    return np.random.uniform(low=domain_lower, high=domain_upper)


def ValidityChecker(q_left):
    key = tuple(np.round(q_left, 4))

    try:
        tf_goal = q_to_ee_target(q_left)
    except Exception:
        return False

    solutions = ik.IK_all(tf_goal)
    if not solutions:
        return False

    # Sort by closeness to cached config
    q_prev = right_arm_cache.get(key)
    if q_prev is not None:
        solutions.sort(key=lambda s: np.linalg.norm(s - q_prev))

    for q_right in solutions:
        q_full = make_full_config(q_left, q_right)
        if checker.CheckConfigCollisionFree(q_full):
            right_arm_cache[key] = q_right
            return True

    return False


# Verify
print(f"Start valid: {ValidityChecker(q_left_start)}")
print(f"Goal valid:  {ValidityChecker(q_left_goal)}")

# ============================================================
# BiRRT planning (in 6D left arm space)
# ============================================================
print("\n" + "=" * 50)
print("Running BiRRT planner in 6D left arm space...")
print("=" * 50)

rrt_options = rrt.RRTOptions(
    step_size=2e-1,
    check_size=1e-2,
    max_vertices=int(1e4),
    max_iters=int(1e5),
    goal_sample_frequency=0.05,
    always_swap=False,
)
rrt_planner = rrt.BiRRT(RandomConfig, ValidityChecker)

np.random.seed(0)
t0 = time.time()
path_left = rrt_planner.plan(q_left_start, q_left_goal, rrt_options)
planning_time = time.time() - t0

if len(path_left) == 0:
    print(f"BiRRT FAILED to find a path ({planning_time:.2f}s)")
    sys.exit(1)

print(f"BiRRT found path: {len(path_left)} waypoints in {planning_time:.2f}s")

# Shortcut
print("\nRunning shortcutting...")
np.random.seed(0)
t0 = time.time()
path_left = shortcut_mod.shortcut(
    path_left.copy(), ValidityChecker, num_tries=100, check_size=rrt_options.check_size
)
shortcut_time = time.time() - t0
print(f"Shortcut: {len(path_left)} waypoints in {shortcut_time:.2f}s")

# ============================================================
# Resolve full 12D path with branch consistency
# ============================================================
print("\nResolving right arm configs along path...")
full_path = []
q_right_prev = q_right_start

for q_left in path_left:
    tf_target = q_to_ee_target(q_left)
    solutions = ik.IK_all(tf_target)
    # Sort by closeness to previous
    solutions.sort(key=lambda s: np.linalg.norm(s - q_right_prev))

    q_right_best = None
    for q_right in solutions:
        q_full = make_full_config(q_left, q_right)
        if checker.CheckConfigCollisionFree(q_full):
            q_right_best = q_right
            break

    if q_right_best is None:
        q_right_best = q_right_prev  # fallback

    full_path.append(make_full_config(q_left, q_right_best))
    q_right_prev = q_right_best

print(f"\n{'=' * 50}")
print(f"RESULTS")
print(f"{'=' * 50}")
print(f"  Planning time:  {planning_time:.2f}s")
print(f"  Shortcut time:  {shortcut_time:.2f}s")
print(f"  Total time:     {planning_time + shortcut_time:.2f}s")
print(f"  Path length:    {len(path_left)} waypoints (6D left arm)")
print(f"  Full path:      {len(full_path)} waypoints ({total_positions}D plant)")

# ============================================================
# Visualize in Meshcat (if enabled)
# ============================================================
if USE_MESHCAT:
    print("\nVisualizing in Meshcat...")
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    for q_full in full_path:
        plant.SetPositions(plant_context, q_full)
        diagram.ForcedPublish(context)
        time.sleep(0.1)

    print("Done! Check Meshcat visualization.")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
else:
    print("\nRun with --meshcat to visualize (Meshcat at localhost:7001)")
    print("Done!")
