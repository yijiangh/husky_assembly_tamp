"""
Planner backend abstraction for the trajectory testbench.

Allows swapping between different dual-arm motion planners via a common
interface. Each backend wraps a specific planner implementation.

Usage:
    backend = get_backend("birrt")
    path = backend.plan(start_conf, goal_conf, robot_setup=..., projector=..., ...)
"""

import json
import os
import subprocess
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class PlannerBackend(ABC):
    """Abstract base class for testbench planner backends."""

    name: str = "base"
    description: str = "Abstract planner backend"

    @abstractmethod
    def plan(
        self,
        start_conf: np.ndarray,
        goal_conf: np.ndarray,
        robot_setup=None,
        projector=None,
        **kwargs,
    ) -> Optional[List[np.ndarray]]:
        """Plan a path from start_conf to goal_conf.

        Args:
            start_conf: Start joint configuration (12D for dual UR5e)
            goal_conf:  Goal joint configuration
            robot_setup: RobotSetup instance (for collision checking, IK, etc.)
            projector: DualArmProjection instance (for constraint projection)
            **kwargs: Planner-specific options

        Returns:
            List of joint configurations (path), or None if planning fails.
        """
        ...


class BiRRTBackend(PlannerBackend):
    """Wraps the existing TrajectoryDualCartConstrainedSolver (capsule-space BiRRT)."""

    name = "birrt"
    description = "Capsule-space BiRRT with dual-arm projection (UR5e)"

    def plan(
        self,
        start_conf: np.ndarray,
        goal_conf: np.ndarray,
        robot_setup=None,
        projector=None,
        **kwargs,
    ) -> Optional[List[np.ndarray]]:
        from husky_assembly_tamp.motion_planner.trajectory_dual_cart_constrained_solver import (
            TrajectoryDualCartConstrainedSolver,
        )

        solver = TrajectoryDualCartConstrainedSolver(robot_setup, None, projector)

        return solver.plan(
            start_conf=start_conf,
            target_conf=goal_conf,
            max_time=kwargs.get("max_time", 60),
            max_iterations=kwargs.get("max_iterations", 5000),
            max_attempts=kwargs.get("max_attempts", 10),
            use_draw=kwargs.get("use_draw", True),
            verbose=kwargs.get("verbose", True),
            enable_ik=kwargs.get("enable_ik", True),
            enable_collision=kwargs.get("enable_collision", True),
            dist_metric=kwargs.get("dist_metric", "feature"),
            ladder_search=kwargs.get("ladder_search", "shortest"),
            ladder_expand_delta=kwargs.get("ladder_expand_delta", None),
            start_goal_delta=kwargs.get("start_goal_delta", None),
            goal_sample_prob=kwargs.get("goal_sample_prob", None),
            return_task_path=kwargs.get("return_task_path", False),
            guide_poses=kwargs.get("guide_poses", None),
            warm_start_path=kwargs.get("warm_start_path", None),
            warm_start_first=kwargs.get("warm_start_first", True),
        )


class ConstrainedBimanualDockerBackend(PlannerBackend):
    """
    Calls the cohnt constrained bimanual planner via Docker.

    IMPORTANT: This planner uses KUKA IIWA-14 robots (7-DOF each), NOT UR5e.
    The returned path is in IIWA joint space (14D) and cannot be played back
    on the UR5e robot. This backend is useful for:
    - Validating the constrained bimanual planning approach
    - Timing comparisons (planning time, not execution)
    - Understanding the algorithm before adapting for UR5e
    """

    name = "constrained_bimanual"
    description = "Constrained BiRRT via Docker (IIWA-14, approach validation only)"

    CONTAINER_NAME = "constrained-bimanual-planner"

    def __init__(self, exchange_dir=None):
        if exchange_dir is None:
            repo_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
            )
            exchange_dir = os.path.join(repo_root, "data", "planner_exchange")
        self.exchange_dir = exchange_dir
        self.request_file = os.path.join(exchange_dir, "request.json")
        self.response_file = os.path.join(exchange_dir, "response.json")

    def is_available(self) -> bool:
        """Check if the Docker container is running."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", self.CONTAINER_NAME],
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() == "true"
        except FileNotFoundError:
            return False

    def plan(
        self,
        start_conf: np.ndarray,
        goal_conf: np.ndarray,
        robot_setup=None,
        projector=None,
        **kwargs,
    ) -> Optional[List[np.ndarray]]:
        """
        Plan using the Docker-based constrained bimanual planner.

        Note: start_conf and goal_conf are ignored (they're UR5e configs).
        Instead, uses IIWA configs from kwargs or defaults.

        Kwargs:
            start_config_8d: 8D IIWA parameterized start (default: notebook example)
            goal_config_8d:  8D IIWA parameterized goal (default: notebook example)
            grasp_distance:  distance between grippers (default: 0.765)
            rrt_timeout:     planning timeout in seconds (default: 60)
        """
        if not self.is_available():
            print(
                f"ERROR: Docker container '{self.CONTAINER_NAME}' is not running.\n"
                f"Start it: cd external/husky_assembly_tamp/docker/constrained_bimanual && ./run.sh up"
            )
            return None

        # Use IIWA configs (UR5e configs can't be used with this planner)
        request = {
            "start_config_8d": kwargs.get(
                "start_config_8d",
                [-0.643, 1.916, -1.797, 1.295, -0.024, -0.877, -1.704, 1.45],
            ),
            "goal_config_8d": kwargs.get(
                "goal_config_8d",
                [-0.199, 0.914, -2.237, 0.524, 0.800, -1.358, -1.015, 2.41],
            ),
            "grasp_distance": kwargs.get("grasp_distance", 0.6),
            "rrt_step_size": kwargs.get("rrt_step_size", 0.2),
            "rrt_max_iters": kwargs.get("rrt_max_iters", 100000),
            "rrt_timeout": kwargs.get("rrt_timeout", 60.0),
            "shortcut_tries": kwargs.get("shortcut_tries", 100),
        }

        os.makedirs(self.exchange_dir, exist_ok=True)

        # Clean old response
        if os.path.exists(self.response_file):
            os.remove(self.response_file)

        with open(self.request_file, "w") as f:
            json.dump(request, f, indent=2)

        print(f"Sending planning request to Docker container...")
        t0 = time.time()

        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    self.CONTAINER_NAME,
                    "python3",
                    "/opt/proj/host_scripts/planner_server.py",
                ],
                capture_output=True,
                text=True,
                timeout=int(request["rrt_timeout"]) + 30,
            )
        except subprocess.TimeoutExpired:
            print("ERROR: Docker planner timed out")
            return None

        wall_time = time.time() - t0

        if result.stdout:
            print(f"[Docker] {result.stdout.strip()}")
        if result.returncode != 0:
            print(f"ERROR: planner_server.py failed (exit code {result.returncode})")
            if result.stderr:
                print(f"[Docker stderr] {result.stderr.strip()}")
            return None

        if not os.path.exists(self.response_file):
            print("ERROR: No response file from container")
            return None

        with open(self.response_file, "r") as f:
            response = json.load(f)

        if not response.get("success", False):
            print(f"Planner returned failure: {response.get('message', 'unknown')}")
            return None

        print(
            f"Docker planner: {response['num_waypoints']} waypoints, "
            f"plan={response['planning_time']:.3f}s, "
            f"shortcut={response.get('shortcut_time', 0):.3f}s, "
            f"wall={wall_time:.3f}s"
        )

        # Return 14D IIWA path (NOTE: not UR5e-compatible for playback)
        path = [np.array(q) for q in response["path_14d"]]
        return path


# Registry of available backends
BACKENDS = {
    "birrt": BiRRTBackend,
    "constrained_bimanual": ConstrainedBimanualDockerBackend,
}


def get_backend(name: str) -> PlannerBackend:
    """Get a planner backend by name."""
    if name not in BACKENDS:
        available = ", ".join(BACKENDS.keys())
        raise ValueError(f"Unknown planner backend '{name}'. Available: {available}")
    return BACKENDS[name]()


def list_backends() -> List[str]:
    """List available backend names."""
    return list(BACKENDS.keys())
