"""Replay an exported Stage 3 JointTrajectory in the Stage 1 scene with a slider."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet
import pybullet_planning as pp

from husky_assembly_tamp.motion_planner.stage1.minimal_rrt import (
    setup_planning_scene,
    teardown_planning_scene,
)
from husky_assembly_tamp.motion_planner.stage1.trajectory_io import load_joint_trajectory_as_path


PoseLike = Tuple[np.ndarray, np.ndarray]


def load_metadata(json_path: str) -> Dict[str, Any]:
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Metadata JSON not found: {json_path}")
    with open(json_path) as f:
        return json.load(f)


def dict_to_pose(data: Dict[str, Sequence[float]]) -> PoseLike:
    return (np.asarray(data["position"], dtype=float), np.asarray(data["quaternion"], dtype=float))


def metadata_to_scene_spec(metadata: Dict[str, Any]) -> Dict[str, Any]:
    raw_spec = metadata.get("scene_spec")
    if not raw_spec:
        raise ValueError("Metadata JSON missing 'scene_spec' block.")
    scene_spec: Dict[str, Any] = dict(raw_spec)
    for key in ("mobile_base_from_tool0_left_home", "world_from_bar_start", "world_from_bar_goal"):
        if key in scene_spec:
            scene_spec[key] = dict_to_pose(scene_spec[key])
    for key in ("start_joint_values", "end_joint_values"):
        if key in scene_spec:
            scene_spec[key] = np.asarray(scene_spec[key], dtype=float)
    if "grasp_targets" in scene_spec:
        scene_spec["grasp_targets"] = [
            (dict_to_pose(bar_pose), dict_to_pose(tool_pose))
            for bar_pose, tool_pose in scene_spec["grasp_targets"]
        ]
    for built_bar in scene_spec.get("built_bars", []):
        if "pose" in built_bar:
            built_bar["pose"] = dict_to_pose(built_bar["pose"])
    return scene_spec


def reconstruct_bar_path(scene: Dict[str, Any], joint_path: Sequence[np.ndarray]) -> List[PoseLike]:
    poses: List[PoseLike] = []
    grasp_bar_from_left = scene["grasp_bar_from_left"]
    for conf in joint_path:
        pp.set_joint_positions(scene["robot"], scene["arm_joints"], conf)
        world_from_left = pp.get_link_pose(scene["robot"], scene["tool_link_left"])
        bar_pose = pp.multiply(world_from_left, pp.invert(grasp_bar_from_left))
        poses.append((np.asarray(bar_pose[0], dtype=float), np.asarray(bar_pose[1], dtype=float)))
    return poses


def run_slider_loop(scene: Dict[str, Any], joint_path: Sequence[np.ndarray], pose_path: Sequence[PoseLike]) -> None:
    if not joint_path:
        raise ValueError("Trajectory is empty.")

    slider = pybullet.addUserDebugParameter("Waypoint", 0, len(joint_path) - 1, 0, physicsClientId=scene["cid"])
    current_idx = -1
    while True:
        try:
            idx = int(round(pybullet.readUserDebugParameter(slider, physicsClientId=scene["cid"])))
            idx = max(0, min(idx, len(joint_path) - 1))
            if idx != current_idx:
                current_idx = idx
                pp.set_joint_positions(scene["robot"], scene["arm_joints"], joint_path[idx])
                pp.set_pose(scene["bar_body"], pose_path[idx])
            time.sleep(0.01)
        except KeyboardInterrupt:
            return


def parse_args():
    parser = argparse.ArgumentParser(description="Replay an exported Stage 3 JointTrajectory with a waypoint slider")
    parser.add_argument("--trajectory-json", type=str, required=True, help="Path to exported JointTrajectory JSON")
    parser.add_argument("--metadata-json", type=str, required=True, help="Sidecar metadata JSON with scene_spec and pose path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata_json)
    scene = setup_planning_scene(
        scene_spec=metadata_to_scene_spec(metadata),
        use_gui=True,
    )
    try:
        joint_path = load_joint_trajectory_as_path(args.trajectory_json)
        pose_path: Optional[List[PoseLike]] = None
        if metadata and metadata.get("pose_path"):
            pose_path = [dict_to_pose(item) for item in metadata["pose_path"]]
        if pose_path is None or len(pose_path) != len(joint_path):
            pose_path = reconstruct_bar_path(scene, joint_path)

        start_pose = scene["start_pose"]
        goal_pose = pose_path[-1]
        if metadata and metadata.get("start_pose"):
            start_pose = dict_to_pose(metadata["start_pose"])
        if metadata and metadata.get("goal_pose"):
            goal_pose = dict_to_pose(metadata["goal_pose"])

        pp.set_pose(scene["ghost_start"], start_pose)
        pp.set_pose(scene["ghost_goal"], goal_pose)
        pp.add_text("Replay Start", start_pose[0], color=(0.0, 0.8, 0.0, 1.0))
        pp.add_text("Replay Goal", goal_pose[0], color=(0.8, 0.0, 0.0, 1.0))
        pp.set_joint_positions(scene["robot"], scene["arm_joints"], joint_path[0])
        pp.set_pose(scene["bar_body"], pose_path[0])
        run_slider_loop(scene, joint_path, pose_path)
    finally:
        teardown_planning_scene()


if __name__ == "__main__":
    main()
