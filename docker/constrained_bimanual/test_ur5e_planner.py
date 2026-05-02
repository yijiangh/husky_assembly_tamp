"""
Test script for the UR5e constrained bimanual planner running in Docker.

This script:
1. Exports the current PyBullet collision scene (if available)
2. Writes a planning request to the exchange directory
3. Invokes the planner_server_ur5e.py inside the Docker container
4. Reads and validates the response

Prerequisites:
    cd external/husky_assembly_tamp/docker/constrained_bimanual && ./run.sh up
    # Install ur-analytic-ik in container:
    docker exec -it constrained-bimanual-planner pip install ur-analytic-ik

Usage:
    python external/husky_assembly_tamp/docker/constrained_bimanual/test_ur5e_planner.py
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
EXCHANGE_DIR = os.path.join(REPO_ROOT, "data", "planner_exchange")
REQUEST_FILE = os.path.join(EXCHANGE_DIR, "request.json")
RESPONSE_FILE = os.path.join(EXCHANGE_DIR, "response.json")

CONTAINER_NAME = "constrained-bimanual-planner"

# Example UR5e configs (6D: 6 joints per arm)
# These are home-ish positions - adjust for your workspace
EXAMPLE_LEFT_START = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
EXAMPLE_LEFT_GOAL = [0.5, -1.2, 1.2, -1.0, -1.57, 0.3]

# Right arm configs will be computed by the planner via analytical IK,
# but we provide initial guesses for the start/goal
EXAMPLE_RIGHT_START = [0.0, -1.57, -1.57, -1.57, 1.57, 0.0]
EXAMPLE_RIGHT_GOAL = [-0.5, -1.2, -1.2, -1.0, 1.57, -0.3]

EXAMPLE_GRASP_DISTANCE = 0.3


def check_container_running():
    """Check if the Docker container is running."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() == "true"
    except FileNotFoundError:
        print("ERROR: docker command not found. Is Docker installed?")
        return False


def ensure_ur_analytic_ik():
    """Check if ur-analytic-ik is installed in the container."""
    result = subprocess.run(
        ["docker", "exec", CONTAINER_NAME,
         "python3", "-c", "import ur_analytic_ik; print('OK')"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Installing ur-analytic-ik in container...")
        subprocess.run(
            ["docker", "exec", CONTAINER_NAME,
             "pip", "install", "ur-analytic-ik"],
            check=True,
        )
        print("Installed successfully.")


def send_planning_request(request):
    """Write request JSON and invoke planner inside container."""
    os.makedirs(EXCHANGE_DIR, exist_ok=True)

    if os.path.exists(RESPONSE_FILE):
        os.remove(RESPONSE_FILE)

    with open(REQUEST_FILE, "w") as f:
        json.dump(request, f, indent=2)
    print(f"Request written to {REQUEST_FILE}")

    print(f"Invoking UR5e planner in container '{CONTAINER_NAME}'...")
    t0 = time.time()

    result = subprocess.run(
        [
            "docker", "exec", CONTAINER_NAME,
            "python3", "/opt/proj/host_scripts/planner_server_ur5e.py",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )

    wall_time = time.time() - t0

    print(f"Container stdout:\n{result.stdout}")
    if result.stderr:
        print(f"Container stderr:\n{result.stderr}")

    if result.returncode != 0:
        print(f"ERROR: planner exited with code {result.returncode}")
        return None, wall_time

    if not os.path.exists(RESPONSE_FILE):
        print(f"ERROR: Response file not found at {RESPONSE_FILE}")
        return None, wall_time

    with open(RESPONSE_FILE, "r") as f:
        response = json.load(f)

    return response, wall_time


def validate_path(response):
    """Basic validation of the planned path."""
    path_12d = np.array(response["path_12d"])
    path_left = np.array(response["path_left_6d"])
    path_right = np.array(response["path_right_6d"])

    print(f"\n--- Path Validation ---")
    print(f"  Waypoints: {len(path_12d)}")
    print(f"  Left arm dim: {path_left.shape[1] if len(path_left) > 0 else 0}")
    print(f"  Right arm dim: {path_right.shape[1] if len(path_right) > 0 else 0}")

    if len(path_left) > 1:
        diffs = np.diff(path_left, axis=0)
        max_step = np.max(np.abs(diffs))
        mean_step = np.mean(np.linalg.norm(diffs, axis=1))
        print(f"  Left arm max single-joint step: {max_step:.4f} rad")
        print(f"  Left arm mean waypoint distance: {mean_step:.4f} rad")

    if len(path_right) > 1:
        diffs_r = np.diff(path_right, axis=0)
        max_step_r = np.max(np.abs(diffs_r))
        print(f"  Right arm max single-joint step: {max_step_r:.4f} rad")

    print(f"------------------------")


def main():
    print("=" * 60)
    print("UR5e Constrained Bimanual Planner - Docker Test")
    print("=" * 60)

    if not check_container_running():
        print(
            f"\nContainer '{CONTAINER_NAME}' is not running."
            f"\nStart it with: cd external/husky_assembly_tamp/docker/constrained_bimanual && ./run.sh up"
        )
        sys.exit(1)

    print(f"\nContainer '{CONTAINER_NAME}' is running.")
    ensure_ur_analytic_ik()

    # Build request
    request = {
        "start_config_left_6d": EXAMPLE_LEFT_START,
        "start_config_right_6d": EXAMPLE_RIGHT_START,
        "goal_config_left_6d": EXAMPLE_LEFT_GOAL,
        "goal_config_right_6d": EXAMPLE_RIGHT_GOAL,
        "grasp_distance": EXAMPLE_GRASP_DISTANCE,
        "grasp_angle_deg": 68.0,
        "base_offset": [0, -0.2978, 0],
        "rrt_step_size": 0.2,
        "rrt_max_iters": 100000,
        "rrt_timeout": 60.0,
        "shortcut_tries": 100,
    }

    # Optionally load collision scene from file
    collision_file = os.path.join(EXCHANGE_DIR, "collision_scene.json")
    if os.path.exists(collision_file):
        print(f"\nLoading collision scene from {collision_file}")
        with open(collision_file, "r") as f:
            collision_data = json.load(f)
        request["collision_objects"] = collision_data.get("collision_objects", [])
        print(f"  Loaded {len(request['collision_objects'])} collision objects")

    print(f"\nStart left:  {EXAMPLE_LEFT_START}")
    print(f"Start right: {EXAMPLE_RIGHT_START}")
    print(f"Goal left:   {EXAMPLE_LEFT_GOAL}")
    print(f"Goal right:  {EXAMPLE_RIGHT_GOAL}")
    print(f"Grasp dist:  {EXAMPLE_GRASP_DISTANCE}")

    # Send request
    print()
    response, wall_time = send_planning_request(request)

    if response is None:
        print("\nPlanning FAILED (no response)")
        sys.exit(1)

    # Report results
    print(f"\n{'=' * 40}")
    print(f"Result: {'SUCCESS' if response['success'] else 'FAILED'}")
    print(f"Message: {response['message']}")
    print(f"Planning time: {response['planning_time']:.3f}s")
    print(f"Shortcut time: {response.get('shortcut_time', 0):.3f}s")
    print(f"Total wall time: {wall_time:.3f}s")

    if response["success"]:
        print(f"Waypoints: {response.get('num_waypoints', 0)}")
        validate_path(response)

        # Save path
        output_file = os.path.join(EXCHANGE_DIR, "last_path_ur5e.json")
        with open(output_file, "w") as f:
            json.dump(response, f, indent=2)
        print(f"\nPath saved to {output_file}")

    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
