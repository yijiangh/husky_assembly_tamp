if __name__ == "__main__":
    import os
    import sys
    import pybullet_planning as pp
    import numpy as np

    HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(HERE)

    import utils.load_multi_tangent as load_multi_tangent
    from multi_tangent.collision import create_collision_bodies
    from multi_tangent.convert import flatten_list
    from robot.robot_setup import RobotSetup
    from utils.collision import Element, create_couplers, init_pb

    urdf_path = (
        "/home/jeong/summer_research/eth/husky_assembly/data/husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e.urdf"
    )

    init_pb()

    rb = RobotSetup("r0")
    rb.set_joint_positions(rb.arm_joints, np.array([0] * 6))

    pp.draw_pose(pp.get_pose(rb.ee_attachment.child))

    pp.wait_for_user()
