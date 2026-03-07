"""
Visualize dual UR5e arm configurations on the Husky robot in Drake Meshcat.

Allows interactive exploration of start/goal configs, IK solutions,
and planned paths. Use the sliders in Meshcat to adjust joint angles.

Usage (from host):
    docker exec -it constrained-bimanual-planner \
        python3 /opt/proj/host_scripts/visualize_ur5e.py

Then open http://localhost:7001 in your browser.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/opt/proj")
sys.path.insert(0, "/opt/proj/host_scripts")

from pydrake.all import (
    StartMeshcat,
    RobotDiagramBuilder,
    MeshcatVisualizerParams,
    Role,
    MeshcatVisualizer,
    Parser,
    RigidTransform,
)

from ur5e_analytic_ik import AnalyticIK_UR5e, UR5E_LIMITS_LOWER, UR5E_LIMITS_UPPER

ROS_PACKAGES_DIR = "/opt/ros_packages"

# ============================================================
# Arm joint names
# ============================================================
LEFT_JOINT_NAMES = [
    "left_ur_arm_shoulder_pan_joint",
    "left_ur_arm_shoulder_lift_joint",
    "left_ur_arm_elbow_joint",
    "left_ur_arm_wrist_1_joint",
    "left_ur_arm_wrist_2_joint",
    "left_ur_arm_wrist_3_joint",
]
RIGHT_JOINT_NAMES = [
    "right_ur_arm_shoulder_pan_joint",
    "right_ur_arm_shoulder_lift_joint",
    "right_ur_arm_elbow_joint",
    "right_ur_arm_wrist_1_joint",
    "right_ur_arm_wrist_2_joint",
    "right_ur_arm_wrist_3_joint",
]


def register_ros_packages(parser):
    """Register all ROS packages from the mounted directory."""
    for pkg_name in os.listdir(ROS_PACKAGES_DIR):
        pkg_path = os.path.join(ROS_PACKAGES_DIR, pkg_name)
        if os.path.isdir(pkg_path):
            pkg_xml = os.path.join(pkg_path, "package.xml")
            if os.path.exists(pkg_xml):
                parser.package_map().AddPackageXml(pkg_xml)
            else:
                parser.package_map().Add(pkg_name, pkg_path)


def build_scene(meshcat):
    """Build the Drake scene with Husky dual UR5e and Meshcat visualization."""
    builder = RobotDiagramBuilder(time_step=0.0)

    meshcat_params = MeshcatVisualizerParams()
    meshcat_params.delete_on_initialization_event = False
    meshcat_params.role = Role.kIllustration
    meshcat_params.prefix = "visual"
    MeshcatVisualizer.AddToBuilder(
        builder.builder(), builder.scene_graph(), meshcat, meshcat_params
    )

    plant = builder.plant()
    parser = Parser(plant)
    register_ros_packages(parser)

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

    plant.Finalize()

    # Get joint indices
    left_indices = []
    for name in LEFT_JOINT_NAMES:
        joint = plant.GetJointByName(name)
        left_indices.append(joint.position_start())

    right_indices = []
    for name in RIGHT_JOINT_NAMES:
        joint = plant.GetJointByName(name)
        right_indices.append(joint.position_start())

    total_positions = plant.num_positions()

    diagram = builder.Build()
    return diagram, plant, left_indices, right_indices, total_positions


def make_full_config(q_left, q_right, left_indices, right_indices, total_positions):
    """Create full Drake position vector from arm configs."""
    q_full = np.zeros(total_positions)
    for i, idx in enumerate(left_indices):
        q_full[idx] = q_left[i]
    for i, idx in enumerate(right_indices):
        q_full[idx] = q_right[i]
    return q_full


def main():
    parser = argparse.ArgumentParser(description="Visualize dual UR5e on Husky")
    parser.add_argument(
        "--path-file",
        default=None,
        help="JSON file with planned path (response from planner)",
    )
    parser.add_argument(
        "--grasp-distance", type=float, default=0.3,
        help="Distance between grippers along grasped bar",
    )
    parser.add_argument(
        "--grasp-angle", type=float, default=68.0,
        help="Gripper approach angle in degrees",
    )
    args = parser.parse_args()

    # ============================================================
    # Start Meshcat
    # ============================================================
    meshcat = StartMeshcat()
    print(f"\nMeshcat URL: {meshcat.web_url()}")
    print("From host browser: http://localhost:7001")

    # ============================================================
    # Build scene
    # ============================================================
    print("\nBuilding Drake scene with Husky dual UR5e...")
    t0 = time.time()
    diagram, plant, left_indices, right_indices, total_positions = build_scene(meshcat)
    print(f"Scene built in {time.time() - t0:.2f}s")
    print(f"Total plant positions: {total_positions}")
    print(f"Left arm indices: {left_indices}")
    print(f"Right arm indices: {right_indices}")

    # ============================================================
    # Analytical IK setup
    # ============================================================
    ik = AnalyticIK_UR5e()

    grasp_distance = args.grasp_distance
    grasp_angle_deg = args.grasp_angle
    ang = (180 - 2.0 * grasp_angle_deg) * np.pi / 180.0
    c, s = np.cos(ang), np.sin(ang)
    rot_flip = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
    rot_angle = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    offset_dir = np.array([0, 0, -grasp_distance])
    base_offset = np.array([0, -0.2978, 0])

    def q_to_ee_target(q_left):
        tf = ik.FK(q_left).copy()
        tf[:3, :3] = tf[:3, :3] @ rot_flip @ rot_angle
        tf[:3, 3] = tf[:3, 3] + tf[:3, :3] @ offset_dir + base_offset
        return tf

    # ============================================================
    # Add sliders for left arm joints
    # ============================================================
    joint_short_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
    slider_names = []
    for i, name in enumerate(joint_short_names):
        slider_name = f"L_{name}"
        meshcat.AddSlider(
            slider_name,
            min=float(UR5E_LIMITS_LOWER[i]),
            max=float(UR5E_LIMITS_UPPER[i]),
            step=0.01,
            value=0.0,
        )
        slider_names.append(slider_name)

    # Branch selector
    meshcat.AddSlider("IK_branch", min=0.0, max=7.0, step=1.0, value=0.0)

    # Mode selector: 0 = interactive, 1 = start, 2 = goal, 3 = animate path
    meshcat.AddSlider("mode", min=0.0, max=3.0, step=1.0, value=0.0)

    # If path file provided, add trajectory slider
    path_data = None
    if args.path_file and os.path.exists(args.path_file):
        with open(args.path_file, "r") as f:
            path_data = json.load(f)
        n_waypoints = len(path_data.get("path_left_6d", []))
        if n_waypoints > 0:
            meshcat.AddSlider(
                "trajectory_time",
                min=0.0, max=1.0,
                step=1.0 / max(n_waypoints - 1, 1),
                value=0.0,
            )
            print(f"\nLoaded path: {n_waypoints} waypoints")

    # ============================================================
    # Example configs
    # ============================================================
    # Default start/goal (can be overridden from path file)
    q_left_start = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    q_left_goal = np.array([0.5, -1.2, 1.2, -1.0, -1.57, 0.3])

    if path_data:
        if "path_left_6d" in path_data and len(path_data["path_left_6d"]) > 0:
            q_left_start = np.array(path_data["path_left_6d"][0])
            q_left_goal = np.array(path_data["path_left_6d"][-1])
        elif "start_config_left_6d" in path_data:
            q_left_start = np.array(path_data["start_config_left_6d"])
            q_left_goal = np.array(path_data["goal_config_left_6d"])

    # Set initial slider values to start config
    for i, name in enumerate(slider_names):
        meshcat.SetSliderValue(name, float(q_left_start[i]))

    # ============================================================
    # Main loop
    # ============================================================
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    print("\n" + "=" * 60)
    print("CONTROLS:")
    print("  mode=0: Interactive - drag L_* sliders to move left arm")
    print("  mode=1: Show START config")
    print("  mode=2: Show GOAL config")
    print("  mode=3: Animate path (if loaded)")
    print("  IK_branch: Select which IK branch (0-7) for right arm")
    print("=" * 60)
    print("\nPress Ctrl+C to exit.")

    prev_q_left = None
    prev_branch = None
    prev_mode = None

    try:
        while True:
            mode = int(meshcat.GetSliderValue("mode"))
            branch = int(meshcat.GetSliderValue("IK_branch"))

            if mode == 0:
                # Interactive mode: read sliders
                q_left = np.array([meshcat.GetSliderValue(n) for n in slider_names])
            elif mode == 1:
                q_left = q_left_start.copy()
            elif mode == 2:
                q_left = q_left_goal.copy()
            elif mode == 3 and path_data:
                t = meshcat.GetSliderValue("trajectory_time")
                path_left = path_data.get("path_left_6d", [])
                path_right = path_data.get("path_right_6d", [])
                if len(path_left) > 0:
                    idx = int(round(t * (len(path_left) - 1)))
                    idx = max(0, min(idx, len(path_left) - 1))
                    q_left = np.array(path_left[idx])
                    # Use the pre-computed right arm config
                    if idx < len(path_right):
                        q_right = np.array(path_right[idx])
                        q_full = make_full_config(
                            q_left, q_right, left_indices, right_indices, total_positions
                        )
                        plant.SetPositions(plant_context, q_full)
                        diagram.ForcedPublish(context)
                        prev_q_left = q_left
                        prev_mode = mode
                        time.sleep(0.02)
                        continue
                q_left = q_left_start.copy()
            else:
                q_left = q_left_start.copy()

            # Check if anything changed
            if (prev_q_left is not None
                    and np.allclose(q_left, prev_q_left, atol=1e-4)
                    and branch == prev_branch
                    and mode == prev_mode):
                time.sleep(0.02)
                continue

            # Compute right arm via grasp constraint + analytical IK
            try:
                tf_right = q_to_ee_target(q_left)
                solutions = ik.IK_all(tf_right)
                if len(solutions) > 0:
                    actual_branch = min(branch, len(solutions) - 1)
                    q_right = solutions[actual_branch]
                    status = f"branch {actual_branch}/{len(solutions)}"
                else:
                    q_right = np.zeros(6)
                    status = "NO IK SOLUTION"
            except Exception as e:
                q_right = np.zeros(6)
                status = f"IK error: {e}"

            # Build full config and display
            q_full = make_full_config(
                q_left, q_right, left_indices, right_indices, total_positions
            )
            plant.SetPositions(plant_context, q_full)
            diagram.ForcedPublish(context)

            if mode != prev_mode or branch != prev_branch or prev_q_left is None:
                mode_names = {0: "Interactive", 1: "START", 2: "GOAL", 3: "Path"}
                print(f"  [{mode_names.get(mode, '?')}] Left: {np.round(q_left, 3)} | "
                      f"Right: {np.round(q_right, 3)} | {status}")

            prev_q_left = q_left.copy()
            prev_branch = branch
            prev_mode = mode
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        for name in slider_names:
            meshcat.DeleteSlider(name)
        meshcat.DeleteSlider("IK_branch")
        meshcat.DeleteSlider("mode")
        if path_data:
            try:
                meshcat.DeleteSlider("trajectory_time")
            except Exception:
                pass


if __name__ == "__main__":
    main()
