"""
Visualize a planned constrained bimanual path in Meshcat.

Reads the planned path from the exchange directory and animates both IIWA arms
through the trajectory using Drake's Meshcat visualizer.

Prerequisites:
    cd external/husky_assembly_tamp/docker/constrained_bimanual && ./run.sh up

Usage (from host):
    docker exec -it constrained-bimanual-planner \
        python3 /opt/proj/host_scripts/visualize_path.py

Then open http://localhost:7001 in your browser.

Options:
    --file PATH     Path to JSON file with path_14d (default: /opt/exchange/response.json)
    --interpolate N Interpolate N intermediate configs between waypoints (default: 10)
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/opt/proj")

from pydrake.all import (
    LoadModelDirectives,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    ProcessModelDirectives,
    Role,
    RobotDiagramBuilder,
    StartMeshcat,
)

import src.common as common


def load_path(filepath):
    """Load path_14d from a JSON file (response.json or last_path.json)."""
    with open(filepath, "r") as f:
        data = json.load(f)

    if "path_14d" not in data:
        print(f"ERROR: No 'path_14d' key in {filepath}")
        sys.exit(1)

    path = [np.array(q) for q in data["path_14d"]]
    print(f"Loaded {len(path)} waypoints from {filepath}")
    return path


def interpolate_path(path, n_interp):
    """Linearly interpolate between waypoints for smoother animation."""
    if n_interp <= 0 or len(path) < 2:
        return path
    smooth = []
    for i in range(len(path) - 1):
        for t in np.linspace(0, 1, n_interp + 1, endpoint=False):
            smooth.append(path[i] * (1 - t) + path[i + 1] * t)
    smooth.append(path[-1])
    return smooth


def build_scene(meshcat):
    """Build the Drake scene with both IIWAs and Meshcat visualization."""
    directives_file = os.path.join(common.RepoDir(), "models/old_shelves.dmd.yaml")

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
    package_xml_path = os.path.join(common.RepoDir(), "package.xml")
    parser.package_map().AddPackageXml(package_xml_path)
    directives = LoadModelDirectives(directives_file)
    ProcessModelDirectives(directives, parser)

    plant.Finalize()

    builder.builder().ExportInput(plant.get_actuation_input_port(), "actuation")
    builder.builder().ExportOutput(plant.get_state_output_port(), "state")

    diagram = builder.Build()
    return diagram, plant


def main():
    parser = argparse.ArgumentParser(description="Visualize constrained bimanual path")
    parser.add_argument(
        "--file",
        default="/opt/exchange/response.json",
        help="JSON file with path_14d",
    )
    parser.add_argument(
        "--interpolate",
        type=int,
        default=10,
        help="Interpolation steps between waypoints (0 to disable)",
    )
    args = parser.parse_args()

    # Load path
    path = load_path(args.file)
    n_waypoints = len(path)

    # Interpolate for smooth scrubbing
    smooth_path = interpolate_path(path, args.interpolate)
    n_frames = len(smooth_path)
    print(f"Path: {n_waypoints} waypoints -> {n_frames} frames after interpolation")

    # Start Meshcat
    meshcat = StartMeshcat()
    print(f"\nMeshcat URL: {meshcat.web_url()}")
    print("From host browser: http://localhost:7001")

    # Build scene
    print("\nBuilding Drake scene...")
    t0 = time.time()
    diagram, plant = build_scene(meshcat)
    print(f"Scene built in {time.time() - t0:.2f}s")

    # Add trajectory time slider
    slider_name = "trajectory_time"
    meshcat.AddSlider(
        slider_name,
        min=0.0,
        max=1.0,
        step=1.0 / max(n_frames - 1, 1),
        value=0.0,
    )

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    # Show start config
    plant.SetPositions(plant_context, smooth_path[0])
    diagram.ForcedPublish(context)

    print("\nSlider ready. Drag 'trajectory_time' in Meshcat to scrub through the path.")
    print(f"  0.0 = start config, 1.0 = goal config")
    print(f"  {n_waypoints} original waypoints, {n_frames} interpolated frames")
    print("Press Ctrl+C to exit.")

    prev_idx = 0
    try:
        while True:
            t = meshcat.GetSliderValue(slider_name)
            idx = int(round(t * (n_frames - 1)))
            idx = max(0, min(idx, n_frames - 1))

            if idx != prev_idx:
                plant.SetPositions(plant_context, smooth_path[idx])
                diagram.ForcedPublish(context)
                prev_idx = idx

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        meshcat.DeleteSlider(slider_name)


if __name__ == "__main__":
    main()
