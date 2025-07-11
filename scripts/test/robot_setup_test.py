#!/usr/bin/env python3
"""
示例脚本：展示如何使用重构后的 RobotSetup 类与 SceneParser
"""

import os
import sys

import pybullet_planning as pp

# 添加父目录到路径以导入 RobotSetup
HERE = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(HERE)
sys.path.append(PARENT_DIR)

from robot.robot_setup import RobotSetup


def example():
    """使用 SceneParser 加载机器人的示例"""
    print("=== 使用 SceneParser 的 RobotSetup 示例 ===")

    # 定义路径（与 reconstruct_test.py 中相同）
    design_study_path = os.path.join(HERE, "..", "..", "data", "husky_assembly_design_study")
    design_case = "250707_RobotX_box_demo"
    robot_cell_state_path = os.path.join(design_study_path, design_case, "RobotCellStates", "robotx_box_A3-A_RobotCellState.json")

    # 使用 SceneParser 创建 RobotSetup
    robot_setup = RobotSetup(robot_name="husky_with_scene", robot_type="husky_dual", robot_cell_state_path=robot_cell_state_path, use_scene_parser_gui=True, scene_parser_verbose=True)

    print("机器人设置成功！")
    print(f"机器人 ID: {robot_setup.robot}")
    print(f"使用 SceneParser: {robot_setup.is_using_scene_parser()}")
    print(f"末端执行器附件: {robot_setup.ee_attachment}")

    pp.wait_for_user()

    # 清理资源
    robot_setup.cleanup()


def main():
    """主函数"""
    print("RobotSetup 与 SceneParser 集成示例")
    print("=====================================")

    example()


if __name__ == "__main__":
    main()
