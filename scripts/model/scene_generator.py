import argparse
import os
import random
import re
import sys
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

from model.scene_parse import SceneParser
from robot.robot_setup import RobotSetup
from utils.params import *


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
    config: SceneParser, rb: RobotSetup, rb_shadow: RobotSetup, grasped_element, grasped_element_shadow, element_bodies, joint_range: float, position_range: float, yaw_range: float, grasp_offset_range: float, max_attempts: int = 100
) -> Dict:
    """随机化机器人配置并检查碰撞"""
    new_config = deepcopy(config)

    # 为被抓握物体创建附着关系
    grasp_offset = config.get_robot_grasp_offset()

    grasp_attachment = pp.create_attachment(rb.robot, rb.tool_link, grasped_element)
    grasp_attachment_shadow = pp.create_attachment(rb_shadow.robot, rb_shadow.tool_link, grasped_element_shadow)

    # 创建碰撞检测函数 - 起始构型
    extra_disabled_collisions = [
        (
            (rb.robot, pp.link_from_name(rb.robot, "ur_arm_wrist_3_link")),
            (rb.ee_attachment.child, pp.BASE_LINK),
        ),
        (
            (rb.ee_attachment.child, pp.BASE_LINK),
            (grasp_attachment.child, pp.BASE_LINK),
        ),
    ]

    collision_fn_start = pp.get_collision_fn(
        rb.robot,
        rb.arm_joints,
        obstacles=element_bodies,
        attachments=[grasp_attachment, rb.ee_attachment] + rb.attachments,
        self_collisions=True,
        disabled_collisions=rb.disabled_collisions,
        extra_disabled_collisions=extra_disabled_collisions,
        max_distance=0.0,
    )

    # 创建碰撞检测函数 - 终止构型
    extra_disabled_collisions_shadow = [
        (
            (rb_shadow.robot, pp.link_from_name(rb_shadow.robot, "ur_arm_wrist_3_link")),
            (rb_shadow.ee_attachment.child, pp.BASE_LINK),
        ),
        (
            (rb_shadow.ee_attachment.child, pp.BASE_LINK),
            (grasp_attachment_shadow.child, pp.BASE_LINK),
        ),
    ]

    collision_fn_target = pp.get_collision_fn(
        rb_shadow.robot,
        rb_shadow.arm_joints,
        obstacles=element_bodies,
        attachments=[grasp_attachment_shadow, rb_shadow.ee_attachment] + rb_shadow.attachments,
        self_collisions=True,
        disabled_collisions=rb_shadow.disabled_collisions,
        extra_disabled_collisions=extra_disabled_collisions_shadow,
        max_distance=0.0,
    )

    # 尝试生成无碰撞配置
    attempt = 0
    found_valid_config = False

    while not found_valid_config and attempt < max_attempts:
        attempt += 1

        # 随机化参数
        new_start = [angle + random.uniform(-joint_range, joint_range) for angle in config.robot_info.start]
        new_target = [angle + random.uniform(-joint_range, joint_range) for angle in config.robot_info.target]
        new_pose_2d = [val + random.uniform(-position_range, position_range) for val in config.robot_info.pose_2d]
        new_pose_2d[2] += random.uniform(-yaw_range, yaw_range)
        new_grasp_offset = config.robot_info.grasp_offset.copy()
        new_grasp_offset[1] += random.uniform(-grasp_offset_range, grasp_offset_range)

        # 设置配置并检查
        rb.set_base_pose_2d(*new_pose_2d)
        rb_shadow.set_base_pose_2d(*new_pose_2d)

        # 检查起始位置碰撞
        rb.set_joint_positions(rb.arm_joints, new_start)
        pp.set_pose(
            grasped_element,
            pp.multiply(
                pp.get_link_pose(rb.robot, rb.tool_link),
                pp.Pose(point=tuple(new_grasp_offset), euler=pp.Euler(1.5708, 0, 0)),
            ),
        )
        grasp_attachment.assign()

        start_has_collision = collision_fn_start(new_start)

        # 检查目标位置碰撞
        rb_shadow.set_joint_positions(rb_shadow.arm_joints, new_target)
        pp.set_pose(
            grasped_element_shadow,
            pp.multiply(
                pp.get_link_pose(rb_shadow.robot, rb_shadow.tool_link),
                pp.Pose(point=tuple(new_grasp_offset), euler=pp.Euler(1.5708, 0, 0)),
            ),
        )
        grasp_attachment_shadow.assign()

        target_has_collision = collision_fn_target(new_target)

        # 如果两个配置都没有碰撞，则采用这个配置
        if not start_has_collision and not target_has_collision:
            found_valid_config = True
            new_config.robot_info.start = new_start
            new_config.robot_info.target = new_target
            new_config.robot_info.pose_2d = new_pose_2d
            new_config.robot_info.grasp_offset = new_grasp_offset
            print(f"找到无碰撞配置! 尝试次数: {attempt}")
            break

    if not found_valid_config:
        print(f"警告：未能找到无碰撞配置，使用最后一次尝试的配置。")
        # 使用最后生成的配置（可能有碰撞）
        new_config.robot_info.start = new_start
        new_config.robot_info.target = new_target
        new_config.robot_info.pose_2d = new_pose_2d
        new_config.robot_info.grasp_offset = new_grasp_offset

    return new_config


def main():
    parser = argparse.ArgumentParser(description="Scene Configuration Generator")
    parser.add_argument("--scene", type=str, default="cuboid_1", help="Scene name")
    parser.add_argument("--task", type=str, default="task_1", help="Task ID")
    parser.add_argument("--max-attempts", type=int, default=100, help="最大尝试次数来寻找无碰撞配置")
    args = parser.parse_args()

    init_pb()

    # 加载基准配置文件
    scene_parser = SceneParser(os.path.join(HERE, "model", "scenes", f"{args.scene}", f"{args.task}.yml"))
    scene_parser._load_scene()

    line_pts_flattened, radius_per_edge = scene_parser.get_element_info()
    element_bodies = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)

    rb = RobotSetup("r0")
    robot_pose_2d = scene_parser.get_robot_pose_2d()
    pp.set_pose(rb.robot, pp.Pose(point=[robot_pose_2d[0], robot_pose_2d[1], 0], euler=pp.Euler(0, 0, robot_pose_2d[2])))
    rb.set_joint_positions(rb.arm_joints, scene_parser.get_robot_start_pose())

    rb_shadow = RobotSetup("r0_shadow")
    pp.set_pose(rb_shadow.robot, pp.Pose(point=[robot_pose_2d[0], robot_pose_2d[1], 0], euler=pp.Euler(0, 0, robot_pose_2d[2])))
    pp.set_color(rb_shadow.robot, (0, 0, 1, 0.5))
    rb_shadow.set_joint_positions(rb_shadow.arm_joints, scene_parser.get_robot_target_pose())

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
    max_attempts_slider = p.addUserDebugParameter("Max Attempts", 10, 500, args.max_attempts)

    prev_generate_value = 0
    prev_save_value = 0
    current_config = deepcopy(scene_parser)
    base_config = deepcopy(scene_parser)

    pp.wait_for_user()

    while True:
        joint_range = p.readUserDebugParameter(joint_range_slider)
        position_range = p.readUserDebugParameter(position_range_slider)
        yaw_range = p.readUserDebugParameter(yaw_range_slider)
        grasp_offset_range = p.readUserDebugParameter(grasp_offset_slider)
        generate_value = p.readUserDebugParameter(generate_button)
        save_value = p.readUserDebugParameter(save_button)
        max_attempts = int(p.readUserDebugParameter(max_attempts_slider))

        if generate_value > prev_generate_value:
            current_config = randomize_robot_config(base_config, rb, rb_shadow, grasped_element, grasped_element_shadow, element_bodies, joint_range, position_range, yaw_range, grasp_offset_range, max_attempts)
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
