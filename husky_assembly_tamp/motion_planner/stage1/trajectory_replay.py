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
    TOOL_LINK_LEFT,
    build_default_paths,
    setup_stage1_scene,
    teardown_stage1_scene,
)
from husky_assembly_tamp.motion_planner.stage1.trajectory_io import load_joint_trajectory_as_path


PoseLike = Tuple[np.ndarray, np.ndarray]


def load_metadata(json_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if json_path is None or not os.path.isfile(json_path):
        return None
    with open(json_path) as f:
        return json.load(f)


def dict_to_pose(data: Dict[str, Sequence[float]]) -> PoseLike:
    return (np.asarray(data["position"], dtype=float), np.asarray(data["quaternion"], dtype=float))


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
    default_grasp_json, default_start_state, default_end_state = build_default_paths()
    parser = argparse.ArgumentParser(description="Replay an exported Stage 3 JointTrajectory with a waypoint slider")
    parser.add_argument("--trajectory-json", type=str, required=True, help="Path to exported JointTrajectory JSON")
    parser.add_argument("--metadata-json", type=str, default=None, help="Optional sidecar metadata JSON from goal_pose_study.py")
    parser.add_argument("--grasp-json", type=str, default=default_grasp_json, help="Path to grasp JSON file")
    parser.add_argument("--start-state", type=str, default=default_start_state, help="Path to start RobotCellState JSON")
    parser.add_argument("--end-state", type=str, default=default_end_state, help="Path to baseline end RobotCellState JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata_json)
    scene = setup_stage1_scene(args.grasp_json, args.start_state, args.end_state, use_gui=True)
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
        teardown_stage1_scene()


if __name__ == "__main__":
    main()
