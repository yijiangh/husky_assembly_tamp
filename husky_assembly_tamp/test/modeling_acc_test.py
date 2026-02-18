import pybullet_planning as pp
import numpy as np
import robotic as ry
import matplotlib.pyplot as plt
from scipy import interpolate
import time


def generate_smooth_trajectory(joint_limits, num_waypoints=5, trajectory_length=100):
    """Generate a random smooth joint trajectory"""
    low = np.array(joint_limits[0])
    high = np.array(joint_limits[1])
    n_joints = len(low)

    if n_joints == 0:
        raise ValueError(f"Joint limits are empty! Cannot generate trajectory.")

    if len(high) != n_joints:
        raise ValueError(f"Joint limits mismatch: low has {n_joints} elements, high has {len(high)} elements")

    # Generate random waypoints within joint limits
    waypoints = []
    for i in range(num_waypoints):
        waypoint = []
        for j in range(n_joints):
            # Generate random value within joint limits with some margin
            min_val, max_val = low[j], high[j]
            margin = 0.1 * (max_val - min_val)
            waypoint.append(np.random.uniform(min_val + margin, max_val - margin))
        waypoints.append(waypoint)

    waypoints = np.array(waypoints)

    # Create time points for waypoints
    waypoint_times = np.linspace(0, 1, num_waypoints)

    # Interpolate to create smooth trajectory
    trajectory_times = np.linspace(0, 1, trajectory_length)
    trajectory = np.zeros((trajectory_length, n_joints))

    for j in range(n_joints):
        # Use cubic spline interpolation for smoothness
        cs = interpolate.interp1d(waypoint_times, waypoints[:, j], kind="cubic", bounds_error=False, fill_value="extrapolate")
        trajectory[:, j] = cs(trajectory_times)

    return trajectory


def execute_trajectory_pybullet(robot, trajectory, left_tool_name="ur_arm_tool0", right_tool_name=None):
    """Execute trajectory in PyBullet and record tool0 positions"""
    positions_left = []
    positions_right = [] if right_tool_name else None

    for i, q in enumerate(trajectory):
        # Set joint positions
        pp.set_joint_positions(robot, pp.get_movable_joints(robot), q)

        # Get left tool0 position
        left_tool_pose = pp.get_link_pose(robot, pp.link_from_name(robot, left_tool_name))
        positions_left.append(left_tool_pose[0])  # position part of pose

        # Get right tool0 position if dual arm
        if right_tool_name:
            right_tool_pose = pp.get_link_pose(robot, pp.link_from_name(robot, right_tool_name))
            positions_right.append(right_tool_pose[0])

        if i % 20 == 0:  # Print progress
            print(f"PyBullet step {i}/{len(trajectory)}")

    if right_tool_name:
        return np.array(positions_left), np.array(positions_right)
    else:
        return np.array(positions_left)


def execute_trajectory_komo(C, trajectory, left_tool_name="ur_arm_tool0", right_tool_name=None):
    """Execute trajectory in Komo and record tool0 positions"""
    positions_left = []
    positions_right = [] if right_tool_name else None

    for i, q in enumerate(trajectory):
        # Set joint state
        q_new = np.hstack((q[6:], q[:6]))
        C.setJointState(q_new)

        # Get left tool0 position
        left_tool_pos = C.getFrame(left_tool_name).getPosition()
        positions_left.append(left_tool_pos)

        # Get right tool0 position if dual arm
        if right_tool_name:
            right_tool_pos = C.getFrame(right_tool_name).getPosition()
            positions_right.append(right_tool_pos)

        if i % 20 == 0:  # Print progress
            print(f"Komo step {i}/{len(trajectory)}")

    if right_tool_name:
        return np.array(positions_left), np.array(positions_right)
    else:
        return np.array(positions_left)


if __name__ == "__main__":
    pp.connect(True)

    # Dual arm robot paths
    dual_g_path = "/home/jeong/summer_research/husky_assembly/ext/husky-assembly-teleop/data/husky_urdf/mt_husky_dual_ur5_e_moveit_config/urdf/husky_dual_ur5_e_no_base_joint_standalone.g"
    dual_urdf_path = "/home/jeong/summer_research/husky_assembly/ext/husky-assembly-teleop/data/husky_urdf/mt_husky_dual_ur5_e_moveit_config/urdf/husky_dual_ur5_e_no_base_joint.urdf"

    # Load dual arm robot in PyBullet
    robot = pp.load_model(dual_urdf_path, pose=([0, 0, 0], [0, 0, 0, 1]))

    # Load dual arm config into Komo
    C = ry.Config()
    C.addFile(dual_g_path)

    print("PyBullet robot loaded")
    print("Komo config loaded")

    # C.view()

    # pp.wait_for_user()

    # for i in range(12):

    #     print("=" * 50)
    #     print(f"stage {i+1}")
    #     print("=" * 50)

    #     q = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     q_komo = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     q[i] = 1
    #     q_komo[(i + 6) % 12] = 1

    #     C.setJointState(q_komo)
    #     pp.set_joint_positions(robot, pp.get_movable_joints(robot), q)

    #     C.view()

    #     pp.wait_for_user()

    # Check joint state to verify joints are loaded
    joint_state = C.getJointState()
    print(f"Komo joint state shape: {np.array(joint_state).shape}")
    print(f"Komo joint state: {joint_state}")

    # Get movable joints from PyBullet to ensure we match
    pb_movable_joints = pp.get_movable_joints(robot)
    print(f"PyBullet movable joints: {pb_movable_joints}, count: {len(pb_movable_joints)}")

    # Get joint limits for trajectory generation
    low, high = C.getJointLimits()
    print(f"Komo joint limits - low shape: {np.array(low).shape}, high shape: {np.array(high).shape}")
    print(f"Komo joint limits - low: {low}")
    print(f"Komo joint limits - high: {high}")

    # Check if joint limits are empty or have wrong dimensions
    low = np.array(low)
    high = np.array(high)

    if len(low) == 0 or len(high) == 0:
        raise ValueError(f"Joint limits are empty! low: {low}, high: {high}")

    if len(low) != len(high):
        raise ValueError(f"Joint limits mismatch: low has {len(low)} elements, high has {len(high)} elements")

    if len(low) != len(pb_movable_joints):
        print(f"Warning: Komo has {len(low)} joints but PyBullet has {len(pb_movable_joints)} movable joints")
        print("This might cause issues. Make sure the models match.")

    # Prepare joint limits in format expected by generate_smooth_trajectory
    dual_joint_limits = (low, high)

    # Generate random smooth trajectory
    print("Generating trajectory...")

    # Dual arm trajectory (12 joints - 6 for left, 6 for right)
    dual_trajectory = generate_smooth_trajectory(dual_joint_limits, num_waypoints=8, trajectory_length=200)
    print(f"Generated dual arm trajectory with {len(dual_trajectory)} points, {dual_trajectory.shape[1]} joints")
    print(f"Trajectory shape: {dual_trajectory.shape}")
    if len(dual_trajectory) > 0:
        print(f"First trajectory point: {dual_trajectory[0]}")
    else:
        raise ValueError("Generated trajectory is empty!")

    # Execute trajectory in both environments
    print("Executing dual arm robot...")
    start_time = time.time()
    pb_left_pos, pb_right_pos = execute_trajectory_pybullet(robot, dual_trajectory, "left_ur_arm_tool0", "right_ur_arm_tool0")
    pb_dual_time = time.time() - start_time

    start_time = time.time()
    komo_left_pos, komo_right_pos = execute_trajectory_komo(C, dual_trajectory, "left_ur_arm_tool0", "right_ur_arm_tool0")
    komo_dual_time = time.time() - start_time

    print(f"Dual arm - PyBullet: {pb_dual_time:.3f}s, Komo: {komo_dual_time:.3f}s")

    # Create 6 subplots for dual end-effectors
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Dual-Arm Robot End-Effector Position Comparison (PyBullet vs Komo)', fontsize=16)

    # Left arm plots
    axes[0, 0].plot(pb_left_pos[:, 0], label='PyBullet', linewidth=2)
    axes[0, 0].plot(komo_left_pos[:, 0], label='Komo', linewidth=2, linestyle='--')
    axes[0, 0].set_title('Left Arm - X Position')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('X Position (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(pb_left_pos[:, 1], label='PyBullet', linewidth=2)
    axes[0, 1].plot(komo_left_pos[:, 1], label='Komo', linewidth=2, linestyle='--')
    axes[0, 1].set_title('Left Arm - Y Position')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Y Position (m)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[0, 2].plot(pb_left_pos[:, 2], label='PyBullet', linewidth=2)
    axes[0, 2].plot(komo_left_pos[:, 2], label='Komo', linewidth=2, linestyle='--')
    axes[0, 2].set_title('Left Arm - Z Position')
    axes[0, 2].set_xlabel('Time Step')
    axes[0, 2].set_ylabel('Z Position (m)')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Right arm plots
    axes[1, 0].plot(pb_right_pos[:, 0], label='PyBullet', linewidth=2)
    axes[1, 0].plot(komo_right_pos[:, 0], label='Komo', linewidth=2, linestyle='--')
    axes[1, 0].set_title('Right Arm - X Position')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('X Position (m)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(pb_right_pos[:, 1], label='PyBullet', linewidth=2)
    axes[1, 1].plot(komo_right_pos[:, 1], label='Komo', linewidth=2, linestyle='--')
    axes[1, 1].set_title('Right Arm - Y Position')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Y Position (m)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    axes[1, 2].plot(pb_right_pos[:, 2], label='PyBullet', linewidth=2)
    axes[1, 2].plot(komo_right_pos[:, 2], label='Komo', linewidth=2, linestyle='--')
    axes[1, 2].set_title('Right Arm - Z Position')
    axes[1, 2].set_xlabel('Time Step')
    axes[1, 2].set_ylabel('Z Position (m)')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig('dual_arm_modeling_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plot saved as 'dual_arm_modeling_accuracy_comparison.png'")

    # Calculate and print position differences for both arms
    def analyze_accuracy(pb_pos, komo_pos, arm_name):
        position_diff = pb_pos - komo_pos
        max_diff = np.max(np.abs(position_diff), axis=0)
        rms_diff = np.sqrt(np.mean(position_diff**2, axis=0))
        print(f"\n{arm_name} Position accuracy analysis:")
        print(f"Max differences (X, Y, Z): {max_diff}")
        print(f"RMS differences (X, Y, Z): {rms_diff}")

    analyze_accuracy(pb_left_pos, komo_left_pos, "Left Arm")
    analyze_accuracy(pb_right_pos, komo_right_pos, "Right Arm")

    plt.show()
