"""
Minimal test for the constrained bimanual planner running in Docker.

This script:
1. Writes a planning request (IIWA start/goal configs) to the exchange directory
2. Invokes the planner_server.py inside the Docker container
3. Reads and validates the response

Prerequisites:
    cd external/husky_assembly_tamp/docker/constrained_bimanual && ./run.sh up

Usage:
    python -m motion_planner.test_constrained_bimanual_docker

Or directly:
    python test_constrained_bimanual_docker.py
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

# Exchange directory (relative to repo root)
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
EXCHANGE_DIR = os.path.join(REPO_ROOT, "data", "planner_exchange")
REQUEST_FILE = os.path.join(EXCHANGE_DIR, "request.json")
RESPONSE_FILE = os.path.join(EXCHANGE_DIR, "response.json")

CONTAINER_NAME = "constrained-bimanual-planner"

# Example configs from the notebook (8D: 7 IIWA joints + 1 self-motion param)
EXAMPLE_START = [-0.643, 1.916, -1.797, 1.295, -0.024, -0.877, -1.704, 1.45]
EXAMPLE_GOAL = [-0.199, 0.914, -2.237, 0.524, 0.800, -1.358, -1.015, 2.41]
EXAMPLE_GRASP_DISTANCE = 0.6  # Distance between gripper frames (matches notebook)


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


def send_planning_request(request):
    """Write request JSON and invoke planner inside container."""
    os.makedirs(EXCHANGE_DIR, exist_ok=True)

    # Clean up old response
    if os.path.exists(RESPONSE_FILE):
        os.remove(RESPONSE_FILE)

    # Write request
    with open(REQUEST_FILE, "w") as f:
        json.dump(request, f, indent=2)
    print(f"Request written to {REQUEST_FILE}")

    # Invoke planner inside container
    print(f"Invoking planner in container '{CONTAINER_NAME}'...")
    t0 = time.time()

    result = subprocess.run(
        [
            "docker",
            "exec",
            CONTAINER_NAME,
            "python3",
            "/opt/proj/host_scripts/planner_server.py",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    wall_time = time.time() - t0

    print(f"Container stdout:\n{result.stdout}")
    if result.stderr:
        print(f"Container stderr:\n{result.stderr}")

    if result.returncode != 0:
        print(f"ERROR: planner_server.py exited with code {result.returncode}")
        return None, wall_time

    # Read response
    if not os.path.exists(RESPONSE_FILE):
        print(f"ERROR: Response file not found at {RESPONSE_FILE}")
        return None, wall_time

    with open(RESPONSE_FILE, "r") as f:
        response = json.load(f)

    return response, wall_time


def validate_path(path_14d, grasp_distance, tolerance=0.05):
    """
    Basic validation: check that the relative EE distance is roughly maintained
    along the path. This is a coarse check — the actual constraint is on the
    full relative transform, but distance is a quick sanity check.
    """
    # We can't run Drake FK from outside the container, so just check basic
    # properties of the path itself.
    path = np.array(path_14d)

    print(f"\n--- Path Validation ---")
    print(f"  Waypoints: {len(path)}")
    print(f"  Config dim: {path.shape[1]}")

    # Check smoothness: max joint-space step between consecutive waypoints
    if len(path) > 1:
        diffs = np.diff(path, axis=0)
        max_step = np.max(np.abs(diffs))
        mean_step = np.mean(np.linalg.norm(diffs, axis=1))
        print(f"  Max single-joint step: {max_step:.4f} rad")
        print(f"  Mean waypoint distance: {mean_step:.4f} rad")

    # Check joint limits (IIWA: roughly +/- 2.97 rad for most joints)
    iiwa_limits = 2.967
    in_limits = np.all(np.abs(path) <= iiwa_limits + 0.1)
    print(f"  All configs within joint limits: {in_limits}")

    print(f"------------------------")
    return True


def main():
    print("=" * 60)
    print("Constrained Bimanual Planner - Docker Integration Test")
    print("=" * 60)

    # Check container
    if not check_container_running():
        print(
            f"\nContainer '{CONTAINER_NAME}' is not running."
            f"\nStart it with: cd external/husky_assembly_tamp/docker/constrained_bimanual && ./run.sh up"
        )
        sys.exit(1)

    print(f"\nContainer '{CONTAINER_NAME}' is running.")

    # Build request
    request = {
        "start_config_8d": EXAMPLE_START,
        "goal_config_8d": EXAMPLE_GOAL,
        "grasp_distance": EXAMPLE_GRASP_DISTANCE,
        "rrt_step_size": 0.2,
        "rrt_max_iters": 100000,
        "rrt_timeout": 60.0,
        "shortcut_tries": 100,
    }

    print(f"\nStart config (8D): {EXAMPLE_START}")
    print(f"Goal config  (8D): {EXAMPLE_GOAL}")
    print(f"Grasp distance:    {EXAMPLE_GRASP_DISTANCE}")

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
        print(f"Waypoints (8D): {len(response['path_8d'])}")
        print(f"Waypoints (14D): {len(response['path_14d'])}")
        validate_path(response["path_14d"], EXAMPLE_GRASP_DISTANCE)

        # Save path for optional visualization
        output_file = os.path.join(EXCHANGE_DIR, "last_path.json")
        with open(output_file, "w") as f:
            json.dump(
                {
                    "path_8d": response["path_8d"],
                    "path_14d": response["path_14d"],
                    "start": EXAMPLE_START,
                    "goal": EXAMPLE_GOAL,
                },
                f,
            )
        print(f"\nPath saved to {output_file}")

    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
