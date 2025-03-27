import argparse
import json
import os
import random
import re
import signal
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import pybullet as p
import pybullet_planning as pp
import yaml

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import utils.load_multi_tangent as load_multi_tangent
from model.scene_parse import SceneParser
from multi_tangent.collision import create_collision_bodies
from multi_tangent.convert import flatten_list
from robot.robot_setup import RobotSetup
from utils.collision import init_pb
from utils.params import *
from utils.utils import HideOutput, SetSeeds


def load_scene_config(scene_path: str) -> Dict:
    """加载场景配置文件"""
    with open(scene_path, "r") as f:
        return yaml.safe_load(f)


def save_scene_config(config: SceneParser, save_path: str):
    """保存场景配置文件"""
    scene_dir = os.path.dirname(save_path)
    scene_name = os.path.basename(os.path.dirname(scene_dir))
    task_name = os.path.basename(save_path)
    task_id = task_name.split(".")[0]

    with open(save_path, "w") as f:
        f.write(f"# Generated from: {os.path.basename(config.scene_file)}\n")
        f.write(f"# Generated time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f'scene_name: "{scene_name}_{task_id}"\n')
        f.write(f"num_elements: {config.scene_data.num_elements}\n\n")

        f.write("robot:\n")
        f.write(f"  start: {config.robot_info.start}\n")
        f.write(f"  target: {config.robot_info.target}\n")
        f.write(f"  grasp_offset: {config.robot_info.grasp_offset}\n")
        f.write(f"  pose_2d: {config.robot_info.pose_2d}\n\n")

        f.write("channels_info:\n")
        for channel in config.scene_data.channels_info:
            f.write(f"  - center: {channel.center}\n")
            f.write(f"    direction: {channel.direction}\n")
            f.write(f'    type: "{channel.type}"\n')
            f.write(f"    size: {channel.size}\n")
            f.write(f"    thickness: {channel.thickness}\n\n")

        f.write("elements:\n")
        for element in config.scene_data.elements:
            f.write(f'  - id: "{element.id}"\n')
            f.write(f"    position: {element.position}\n")
            f.write(f"    orientation: {element.orientation}\n")
            f.write("    shape:\n")
            f.write(f'      type: "{element.shape.type}"\n')
            f.write("      parameters:\n")
            f.write(f"        radius: {element.shape.parameters.radius}\n")
            f.write(f"        height: {element.shape.parameters.height}\n")
            f.write("    sphere_fit:\n")
            f.write(f"      radius: {element.sphere_fit.radius}\n")
            f.write(f"      count: {element.sphere_fit.count}\n\n")


def randomize_robot_config(
    config: Dict, joint_range: float, position_range: float, yaw_range: float, grasp_offset_range: float
) -> Dict:
    """随机化机器人配置"""
    new_config = deepcopy(config)

    # 随机化起始关节角
    new_config.robot_info.start = [
        angle + random.uniform(-joint_range, joint_range) for angle in config.robot_info.start
    ]

    # 随机化目标关节角
    new_config.robot_info.target = [
        angle + random.uniform(-joint_range, joint_range) for angle in config.robot_info.target
    ]

    # 随机化pose_2d
    new_config.robot_info.pose_2d = [
        val + random.uniform(-position_range, position_range) for val in config.robot_info.pose_2d
    ]
    new_config.robot_info.pose_2d[2] += random.uniform(-yaw_range, yaw_range)

    # 随机化grasp_offset的y轴
    new_config.robot_info.grasp_offset = config.robot_info.grasp_offset.copy()
    new_config.robot_info.grasp_offset[1] += random.uniform(-grasp_offset_range, grasp_offset_range)

    return new_config


def main():
    parser = argparse.ArgumentParser(description="Scene Configuration Generator")
    parser.add_argument("--scene", type=str, default="cuboid_1", help="Scene name")
    parser.add_argument("--task", type=str, default="task_1", help="Task ID")
    args = parser.parse_args()

    init_pb()

    # 加载基准配置文件
    scene_parser = SceneParser(os.path.join(HERE, "model", "scenes", f"{args.scene}", f"{args.task}.yml"))
    scene_parser.load_scene()

    line_pts_flattened, radius_per_edge = scene_parser.get_element_info()
    element_bodies = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)

    rb = RobotSetup("r0")
    robot_pose_2d = scene_parser.get_robot_pose_2d()
    pp.set_pose(
        rb.robot, pp.Pose(point=[robot_pose_2d[0], robot_pose_2d[1], 0], euler=pp.Euler(0, 0, robot_pose_2d[2]))
    )
    rb_shadow = RobotSetup("r0_shadow")
    robot_pose_2d = scene_parser.get_robot_pose_2d()
    pp.set_pose(
        rb_shadow.robot, pp.Pose(point=[robot_pose_2d[0], robot_pose_2d[1], 0], euler=pp.Euler(0, 0, robot_pose_2d[2]))
    )
    pp.set_color(rb_shadow.robot, (0, 0, 1, 0.5))

    line_pts_grasped = [np.array([0, 0, 0]), np.array([0, 0, 1])]
    grasped_element = create_collision_bodies(line_pts_grasped, [0.01], viewer=True)[0]
    pp.set_pose(
        grasped_element,
        pp.multiply(
            pp.get_link_pose(rb.robot, rb.tool_link),
            pp.Pose(point=scene_parser.get_robot_grasp_offset(), euler=pp.Euler(1.5708, 0, 0)),
        ),
    )

    grasped_element_shadow = create_collision_bodies(line_pts_grasped, [0.01], viewer=True)[0]
    pp.set_pose(
        grasped_element_shadow,
        pp.multiply(
            pp.get_link_pose(rb_shadow.robot, rb_shadow.tool_link),
            pp.Pose(point=scene_parser.get_robot_grasp_offset(), euler=pp.Euler(1.5708, 0, 0)),
        ),
    )

    # 创建滑动条控制随机范围
    joint_range_slider = p.addUserDebugParameter("Joint Angle Random Range", 0.0, np.pi, 0.1)
    position_range_slider = p.addUserDebugParameter("Position Random Range", 0.0, 0.5, 0.1)
    yaw_range_slider = p.addUserDebugParameter("Yaw Random Range", 0.0, np.pi, 0.1)
    grasp_offset_slider = p.addUserDebugParameter("Grasp Offset Y Range", 0.0, 0.2, 0.1)
    generate_button = p.addUserDebugParameter("Generate New Config", 1, 0, 0)
    save_button = p.addUserDebugParameter("Save Config", 1, 0, 0)

    prev_generate_value = 0
    prev_save_value = 0
    current_config = deepcopy(scene_parser)
    base_config = deepcopy(scene_parser)

    while True:
        joint_range = p.readUserDebugParameter(joint_range_slider)
        position_range = p.readUserDebugParameter(position_range_slider)
        yaw_range = p.readUserDebugParameter(yaw_range_slider)
        grasp_offset_range = p.readUserDebugParameter(grasp_offset_slider)
        generate_value = p.readUserDebugParameter(generate_button)
        save_value = p.readUserDebugParameter(save_button)

        if generate_value > prev_generate_value:
            current_config = randomize_robot_config(
                base_config, joint_range, position_range, yaw_range, grasp_offset_range
            )
            print("New configuration generated")

            rb.set_joint_positions(rb.arm_joints, current_config.robot_info.start)
            rb_shadow.set_joint_positions(rb_shadow.arm_joints, current_config.robot_info.target)
            rb.set_base_pose_2d(*current_config.robot_info.pose_2d)
            rb_shadow.set_base_pose_2d(*current_config.robot_info.pose_2d)

            pp.set_pose(
                grasped_element,
                pp.multiply(
                    pp.get_link_pose(rb.robot, rb.tool_link),
                    pp.Pose(point=current_config.robot_info.grasp_offset, euler=pp.Euler(1.5708, 0, 0)),
                ),
            )

            pp.set_pose(
                grasped_element_shadow,
                pp.multiply(
                    pp.get_link_pose(rb_shadow.robot, rb_shadow.tool_link),
                    pp.Pose(point=current_config.robot_info.grasp_offset, euler=pp.Euler(1.5708, 0, 0)),
                ),
            )

        if save_value > prev_save_value:
            save_dir = os.path.dirname(scene_parser.scene_file)
            os.makedirs(save_dir, exist_ok=True)

            task_files = [f for f in os.listdir(save_dir) if f.startswith("task_") and f.endswith(".yml")]

            task_numbers = []
            for file in task_files:
                match = re.search(r"task_(\d+)", file)
                if match:
                    task_numbers.append(int(match.group(1)))

            new_task_number = 1
            if task_numbers:
                new_task_number = max(task_numbers) + 1

            save_path = os.path.join(save_dir, f"task_{new_task_number}.yml")
            save_scene_config(current_config, save_path)
            print(f"Configuration saved to: {save_path}")

        prev_generate_value = generate_value
        prev_save_value = save_value
        time.sleep(1.0 / 240)


if __name__ == "__main__":
    main()
