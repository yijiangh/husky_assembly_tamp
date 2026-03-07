"""
Standalone script equivalent of the key parts of main.ipynb.
Runs inside the Docker container without needing Jupyter.

Usage (from host):
    docker exec -it constrained-bimanual-planner python3 /opt/proj/run_example.py

This runs the constrained bimanual BiRRT planner on the IIWA dual-arm setup
and optionally visualizes via Meshcat.
"""

import sys
import os
import time

import numpy as np

sys.path.insert(0, "/opt/proj")

from pydrake.all import (
    StartMeshcat,
    CollisionCheckerParams,
    RobotDiagramBuilder,
    MeshcatVisualizerParams,
    Role,
    MeshcatVisualizer,
    Parser,
    LoadModelDirectives,
    ProcessModelDirectives,
    SceneGraphCollisionChecker,
    PiecewisePolynomial,
    CompositeTrajectory,
)

import src.iiwa_analytic_ik as iiwa_analytic_ik
import src.common as common
import src.rrt as rrt
import src.shortcut as shortcut_mod

# ============================================================
# Configuration
# ============================================================
grasp_distance = 0.6
GC2, GC4, GC6 = 1, 1, -1

q_tilde_bottom = np.array([
    -0.643, 1.916, -1.797, 1.295, -0.024, -0.877, -1.704, 1.45
])
q_tilde_top = np.array([
    -0.199, 0.914, -2.237, 0.524, 0.800, -1.358, -1.015, 2.41
])

USE_MESHCAT = "--meshcat" in sys.argv

# ============================================================
# Scene setup
# ============================================================
print("Setting up Drake scene...")
t0 = time.time()

if USE_MESHCAT:
    meshcat = StartMeshcat()
    print(f"Meshcat URL: {meshcat.web_url()}")

directives_file = os.path.join(common.RepoDir(), "models/old_shelves.dmd.yaml")

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
package_xml_path = os.path.join(common.RepoDir(), "package.xml")
parser.package_map().AddPackageXml(package_xml_path)
directives = LoadModelDirectives(directives_file)
ProcessModelDirectives(directives, parser)

params.robot_model_instances = [
    plant.GetModelInstanceByName("iiwa_left"),
    plant.GetModelInstanceByName("iiwa_right"),
]

plant.Finalize()

builder.builder().ExportInput(plant.get_actuation_input_port(), "actuation")
builder.builder().ExportOutput(plant.get_state_output_port(), "state")

diagram = builder.Build()
params.model = diagram
params.edge_step_size = 0.01
checker = SceneGraphCollisionChecker(params)

print(f"Scene setup: {time.time() - t0:.2f}s")

# ============================================================
# Analytic IK + parameterization
# ============================================================
analytic_ik = iiwa_analytic_ik.Analytic_IK_7DoF()


def q_to_ee_target(q):
    tf_goal = analytic_ik.FK(q).copy()
    ang = (180 - 2.0 * 68.0) * np.pi / 180.0
    c, s = np.cos(ang), np.sin(ang)
    tf_goal[:3, :3] = (
        tf_goal[:3, :3]
        @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    )
    tf_goal[:3, 3] += tf_goal[:3, :3] @ np.array([0, 0, -grasp_distance])
    tf_goal[:3, 3] += np.array([0, -0.765, 0])
    return tf_goal


def parameterization(q_tilde):
    q_full = np.zeros(14)
    q_full[:7] = q_tilde[:7]
    tf_goal = q_to_ee_target(q_tilde[:7])
    from pydrake.all import RigidTransform

    q_full[7:] = analytic_ik.IK(RigidTransform(tf_goal), [GC2, GC4, GC6], q_tilde[7])
    return q_full


def unclipped_vals(q_tilde):
    tf_goal = q_to_ee_target(q_tilde[:7])
    from pydrake.all import RigidTransform

    return analytic_ik.IK(
        RigidTransform(tf_goal), [GC2, GC4, GC6], q_tilde[7],
        return_unclipped_vals=True,
    )


# ============================================================
# Validity checker
# ============================================================
domain_lower = np.hstack((iiwa_analytic_ik.iiwa_limits_lower, [0.0]))
domain_upper = np.hstack((iiwa_analytic_ik.iiwa_limits_upper, [2.0 * np.pi]))


def RandomConfig():
    return np.random.uniform(low=domain_lower, high=domain_upper)


def ValidityChecker(q_tilde):
    uv = unclipped_vals(q_tilde)
    if np.any(np.abs(uv) > 1.0):
        return False

    q_full = parameterization(q_tilde)
    q_sub = q_full[7:]
    if np.any(q_sub < iiwa_analytic_ik.iiwa_limits_lower):
        return False
    if np.any(q_sub > iiwa_analytic_ik.iiwa_limits_upper):
        return False
    if np.any(np.abs(q_sub[[1, 3, 5]]) < 1e-2):
        return False

    return checker.CheckConfigCollisionFree(q_full)


# ============================================================
# Verify start/goal validity
# ============================================================
print("\nVerifying start/goal configs...")
start_valid = ValidityChecker(q_tilde_bottom)
goal_valid = ValidityChecker(q_tilde_top)
print(f"  Start valid: {start_valid}")
print(f"  Goal valid:  {goal_valid}")

if not start_valid or not goal_valid:
    print("ERROR: Start or goal is invalid! Exiting.")
    sys.exit(1)

# Show start config in meshcat
if USE_MESHCAT:
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, parameterization(q_tilde_bottom))
    diagram.ForcedPublish(context)

# ============================================================
# BiRRT planning
# ============================================================
print("\n" + "=" * 50)
print("Running BiRRT planner...")
print("=" * 50)

rrt_options = rrt.RRTOptions(
    step_size=2e-1,
    check_size=1e-2,
    max_vertices=int(1e4),
    max_iters=int(1e6),
    goal_sample_frequency=0.01,
    always_swap=False,
)
rrt_planner = rrt.BiRRT(RandomConfig, ValidityChecker)

np.random.seed(0)
t0 = time.time()
path = rrt_planner.plan(q_tilde_bottom, q_tilde_top, rrt_options)
planning_time = time.time() - t0

if len(path) == 0:
    print(f"BiRRT FAILED to find a path ({planning_time:.2f}s)")
    sys.exit(1)

print(f"BiRRT found path: {len(path)} waypoints in {planning_time:.2f}s")

# ============================================================
# Shortcutting
# ============================================================
print("\nRunning shortcutting...")
np.random.seed(0)
t0 = time.time()
shortcut_path = shortcut_mod.shortcut(
    path.copy(), ValidityChecker, num_tries=100, check_size=rrt_options.check_size
)
shortcut_time = time.time() - t0
print(f"Shortcut: {len(shortcut_path)} waypoints in {shortcut_time:.2f}s")

# ============================================================
# Convert to full 14D path
# ============================================================
full_path = [parameterization(q) for q in shortcut_path]

print(f"\n{'=' * 50}")
print(f"RESULTS")
print(f"{'=' * 50}")
print(f"  Planning time:  {planning_time:.2f}s")
print(f"  Shortcut time:  {shortcut_time:.2f}s")
print(f"  Total time:     {planning_time + shortcut_time:.2f}s")
print(f"  Path length:    {len(shortcut_path)} waypoints (8D)")
print(f"  Full path:      {len(full_path)} waypoints (14D)")

# ============================================================
# Visualize in Meshcat (if enabled)
# ============================================================
if USE_MESHCAT:
    print("\nVisualizing in Meshcat...")
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    # Animate through the path
    for i, q_full in enumerate(full_path):
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
