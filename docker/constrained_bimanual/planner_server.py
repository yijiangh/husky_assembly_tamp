"""
Planner server that runs INSIDE the cohnt/constrained-bimanual-planning-example
Docker container. Reads a planning request JSON from /opt/exchange/request.json,
runs the constrained bimanual BiRRT planner, and writes the result to
/opt/exchange/response.json.

Usage (from inside container):
    cd /opt/proj
    python planner_server.py

The request/response exchange is file-based for simplicity.
"""

import json
import os
import sys
import time

import numpy as np

# -- Drake imports (available in the container) --
from pydrake.all import (
    CollisionCheckerParams,
    LoadModelDirectives,
    Parser,
    ProcessModelDirectives,
    RigidTransform,
    RobotDiagramBuilder,
    SceneGraphCollisionChecker,
)

# -- Repo imports (available at /opt/proj) --
sys.path.insert(0, "/opt/proj")
from src.common import RepoDir
from src.iiwa_analytic_ik import Analytic_IK_7DoF
from src.rrt import BiRRT, RRTOptions
from src.shortcut import shortcut

EXCHANGE_DIR = "/opt/exchange"
REQUEST_FILE = f"{EXCHANGE_DIR}/request.json"
RESPONSE_FILE = f"{EXCHANGE_DIR}/response.json"


def setup_scene():
    """Build the Drake collision checker and scene, matching the notebook."""
    params = CollisionCheckerParams()
    builder = RobotDiagramBuilder(time_step=0.0)
    plant = builder.plant()
    parser = Parser(plant)

    repo_dir = RepoDir()
    package_xml_path = os.path.join(repo_dir, "package.xml")
    parser.package_map().AddPackageXml(package_xml_path)
    directives = LoadModelDirectives(f"{repo_dir}/models/old_shelves.dmd.yaml")
    ProcessModelDirectives(directives, parser)

    params.robot_model_instances = [
        plant.GetModelInstanceByName("iiwa_left"),
        plant.GetModelInstanceByName("iiwa_right"),
    ]

    plant.Finalize()
    params.model = builder.Build()
    params.edge_step_size = 0.01
    checker = SceneGraphCollisionChecker(params)
    return checker


def build_parameterization(grasp_distance, grasp_angle_deg=68.0):
    """
    Build the parameterization function that maps 8D q_tilde -> 14D q_full.

    Parameters:
        grasp_distance: distance between the two gripper frames along the bar
        grasp_angle_deg: gripper rotation angle (default matches notebook: 68 deg)
    """
    ik = Analytic_IK_7DoF()

    ang = (180 - 2.0 * grasp_angle_deg) * np.pi / 180.0
    c, s = np.cos(ang), np.sin(ang)
    rot_flip = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
    rot_angle = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    offset_dir = np.array([0, 0, -grasp_distance])
    base_offset = np.array([0, -0.765, 0])  # right arm base offset from left

    # Global configuration for follower arm (from notebook: GC2=1, GC4=1, GC6=-1)
    GC = [1, 1, -1]

    def q_to_ee_target(q7):
        """Compute follower (right) arm EE target from controlled (left) arm config."""
        tf = ik.FK(q7).copy()
        tf[:3, :3] = tf[:3, :3] @ rot_flip @ rot_angle
        tf[:3, 3] = tf[:3, 3] + tf[:3, :3] @ offset_dir + base_offset
        return tf

    def parameterization(q_tilde):
        """Map 8D parameterized config -> 14D full config."""
        q_full = np.zeros(14)
        q_full[:7] = q_tilde[:7]
        tf_goal = q_to_ee_target(q_tilde[:7])
        psi = q_tilde[7]
        q_full[7:] = ik.IK(RigidTransform(tf_goal), GC, psi)
        return q_full

    def is_reachable(q_tilde):
        """Check if the parameterized config maps to a valid full config."""
        try:
            tf_goal = q_to_ee_target(q_tilde[:7])
            q_right = ik.IK(RigidTransform(tf_goal), GC, q_tilde[7])
            # Check joint limits
            limits_lower = ik.limits_lower
            limits_upper = ik.limits_upper
            if np.any(q_right < limits_lower) or np.any(q_right > limits_upper):
                return False
            return True
        except Exception:
            return False

    return ik, parameterization, is_reachable, q_to_ee_target


def plan(request, checker):
    """
    Run BiRRT planner with the constrained bimanual parameterization.

    Request fields:
        start_config_8d: 8D parameterized start config (7 left joints + 1 psi)
        goal_config_8d:  8D parameterized goal config
        grasp_distance:  distance between grippers along the grasped object
        grasp_angle_deg: (optional) gripper angle, default 68.0
        rrt_step_size:   (optional) RRT step size, default 0.2
        rrt_max_iters:   (optional) max RRT iterations, default 100000
        rrt_timeout:     (optional) timeout in seconds, default 60
        shortcut_tries:  (optional) shortcut iterations, default 100
    """
    start_8d = np.array(request["start_config_8d"])
    goal_8d = np.array(request["goal_config_8d"])
    grasp_distance = request["grasp_distance"]
    grasp_angle_deg = request.get("grasp_angle_deg", 68.0)

    ik, parameterization, is_reachable, _ = build_parameterization(
        grasp_distance, grasp_angle_deg
    )

    # Domain bounds: IIWA joint limits + psi in [0, 2*pi]
    domain_lower = np.hstack((ik.limits_lower, [0.0]))
    domain_upper = np.hstack((ik.limits_upper, [2.0 * np.pi]))

    def random_config():
        return np.random.uniform(low=domain_lower, high=domain_upper)

    def validity_checker(q_tilde):
        if not is_reachable(q_tilde):
            return False
        q_full = parameterization(q_tilde)
        return checker.CheckConfigCollisionFree(q_full)

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
    path_8d = rrt.plan(start_8d, goal_8d, options)
    planning_time = time.time() - t0

    if len(path_8d) == 0:
        return {
            "success": False,
            "path_8d": [],
            "path_14d": [],
            "planning_time": planning_time,
            "shortcut_time": 0.0,
            "message": "BiRRT failed to find a path",
        }

    print(f"Raw path: {len(path_8d)} waypoints in {planning_time:.3f}s")

    # Shortcut
    shortcut_tries = int(request.get("shortcut_tries", 100))
    t1 = time.time()
    path_8d = shortcut(
        path_8d, validity_checker, num_tries=shortcut_tries, check_size=0.01
    )
    shortcut_time = time.time() - t1

    # Convert to 14D
    path_14d = [parameterization(q).tolist() for q in path_8d]
    path_8d_list = [q.tolist() for q in path_8d]

    print(f"Shortcut path: {len(path_8d)} waypoints, shortcut took {shortcut_time:.3f}s")

    return {
        "success": True,
        "path_8d": path_8d_list,
        "path_14d": path_14d,
        "planning_time": planning_time,
        "shortcut_time": shortcut_time,
        "num_waypoints": len(path_8d),
        "message": "OK",
    }


def main():
    print("Setting up Drake scene...")
    checker = setup_scene()
    print("Scene ready.")

    print(f"Reading request from {REQUEST_FILE}...")
    with open(REQUEST_FILE, "r") as f:
        request = json.load(f)

    result = plan(request, checker)

    with open(RESPONSE_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Response written to {RESPONSE_FILE}")
    print(f"  success: {result['success']}")
    print(f"  waypoints: {result.get('num_waypoints', 0)}")
    print(f"  planning_time: {result['planning_time']:.3f}s")


if __name__ == "__main__":
    main()
