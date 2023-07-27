import sys, os, argparse
from single_bar_grasp import load_robot
import pybullet_planning as pp
import pybullet as p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teleopt_target', action='store_true',
                        help='')
    parser.add_argument('--connect_to_mocap', action='store_true',
                        help='connect to mocap.')
    parser.add_argument('--connect_to_hw', action='store_true',
                        help='connect to robot hardware.')
    parser.add_argument('--debug', action='store_true',
                        help='')
    args = parser.parse_args()

    # * start pybullet simulator
    pp.connect(use_gui=True, shadows=True, color=[0.9, 0.9, 1.0])
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=pp.CLIENT)
    # pp.set_camera(92.0, -85, 5.20)

    shadow_robot, shadow_ee_attachment, _, _ = load_robot()
    shadow_color = [0.5, 0.5, 0.5, 0.7]
    pp.set_color(shadow_robot, shadow_color)
    pp.set_color(shadow_ee_attachment.child, shadow_color)

    with pp.LockRenderer():
        # for link in pp.get_links(shadow_robot):
        for link_name in ['ur_arm_shoulder_link']:
            link = pp.link_from_name(shadow_robot, link_name)
            link_pose = pp.get_link_pose(shadow_robot, link)
            pp.set_color(shadow_robot, link=link, color=pp.apply_alpha(pp.BLUE, 0.7))
            pp.draw_pose(link_pose, length=0.1)
            pp.add_text(link_name, (0, 0, 0.05), parent=shadow_robot, parent_link=link)
    pp.wait_if_gui('Press enter to continue')

if __name__ == "__main__":
    main()