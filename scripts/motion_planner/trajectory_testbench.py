"""
Trajectory Dual-Arm Constrained Solver - Testbench

Standalone testbench for the dual-arm constrained motion planner.
Loads the robot directly from URDF (no compas_fab), provides a GUI
for configuring start/end bar poses and grasps, and includes timing
instrumentation for bottleneck identification.

Usage:
    cd external/husky_assembly_tamp/scripts
    python -m motion_planner.trajectory_testbenc
"""

import os
import sys
import time

import numpy as np
import pybullet
import pybullet_planning as pp

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from robot.dual_arm_projection import DualArmProjection
from robot.robot_setup import (
    HUSKY_DUAL_ARM_JOINT_NAMES,
    HUSKY_DUAL_INIT_ARM_JOINT_ANGLES,
    HUSKY_DUAL_URDF_PATH,
    RobotSetup,
)
import motion_planner.trajectory_dual_cart_constrained_solver as solver_mod
from motion_planner.trajectory_dual_cart_constrained_solver import (
    TrajectoryDualCartConstrainedSolver,
)
from utils.util import normalize_angles


# ---------------------------------------------------------------------------
# GUI helpers
# ---------------------------------------------------------------------------

class Button:
    def __init__(self, name, cid=0):
        self.cid = cid
        self.dbg_param = pybullet.addUserDebugParameter(name, 1.0, 0.0, 0.0, physicsClientId=cid)
        self.prev_value = pybullet.readUserDebugParameter(self.dbg_param, physicsClientId=cid)

    def pressed(self):
        new_value = pybullet.readUserDebugParameter(self.dbg_param, physicsClientId=self.cid)
        if new_value != self.prev_value:
            self.prev_value = new_value
            return True
        return False


class PoseSliders:
    """Six sliders for position (x, y, z) and orientation (roll, pitch, yaw)."""

    def __init__(self, prefix, cid=0, defaults=None, pos_range=2.0, angle_range=np.pi):
        self.cid = cid
        if defaults is None:
            defaults = [0.4, 0.0, 0.75, np.pi, np.pi / 2, np.pi / 2]

        labels = ["x", "y", "z", "roll", "pitch", "yaw"]
        mins = [-pos_range, -pos_range, 0.0, -angle_range, -angle_range, -angle_range]
        maxs = [pos_range, pos_range, 2.0, angle_range, angle_range, angle_range]

        self.sliders = []
        for i, label in enumerate(labels):
            s = pybullet.addUserDebugParameter(
                f"{prefix} {label}", mins[i], maxs[i], defaults[i], physicsClientId=cid
            )
            self.sliders.append(s)
        self._prev = list(defaults)

    def read(self):
        vals = [pybullet.readUserDebugParameter(s, physicsClientId=self.cid) for s in self.sliders]
        return vals

    def changed(self):
        vals = self.read()
        if not np.allclose(vals, self._prev, atol=1e-6):
            self._prev = list(vals)
            return True
        return False

    def as_pose(self):
        vals = self.read()
        return pp.Pose(point=vals[:3], euler=pp.Euler(*vals[3:]))


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self.records = {}

    def start(self, name):
        self.records[name] = time.time()

    def stop(self, name):
        elapsed = time.time() - self.records[name]
        self.records[name] = elapsed
        return elapsed

    def summary(self):
        print("\n--- Timing Summary ---")
        for name, val in self.records.items():
            print(f"  {name}: {val:.3f}s")
        print("----------------------\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    timer = Timer()

    # ------------------------------------------------------------------
    # 1. Direct URDF loading (no compas_fab)
    # ------------------------------------------------------------------
    timer.start("robot_loading")

    cid = pp.connect(use_gui=True)
    # pp.connect disables the debug GUI panel — re-enable it for sliders/buttons
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 1, physicsClientId=cid)
    pybullet.resetDebugVisualizerCamera(
        cameraDistance=2.5, cameraYaw=45, cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5], physicsClientId=cid,
    )

    with pp.LockRenderer():
        with pp.HideOutput():
            robot = pp.load_pybullet(HUSKY_DUAL_URDF_PATH, fixed_base=True)

    arm_joints = pp.joints_from_names(robot, HUSKY_DUAL_ARM_JOINT_NAMES)
    pp.set_joint_positions(robot, arm_joints, HUSKY_DUAL_INIT_ARM_JOINT_ANGLES)

    # Create a simple bar (box primitive)
    bar = pp.create_box(0.5, 0.03, 0.03, color=(0.8, 0.4, 0.1, 1.0))

    # Place bar at a reasonable initial position relative to robot
    init_bar_pose = pp.Pose(point=[0.4, 0.0, 0.75], euler=pp.Euler(np.pi, np.pi / 2, np.pi / 2))
    pp.set_pose(bar, init_bar_pose)

    # Create RobotSetup via robot_data dict (bypasses SceneParser/compas_fab)
    robot_setup = RobotSetup(
        robot_name="r0",
        robot_type="husky_dual",
        robot_data={
            "robot_id": robot,
            "obstacles": [],
            "target_bar": bar,
            "joint_values": list(HUSKY_DUAL_INIT_ARM_JOINT_ANGLES),
        },
    )

    load_time = timer.stop("robot_loading")
    print(f"Robot loaded in {load_time:.3f}s (direct URDF, no compas_fab)")

    # ------------------------------------------------------------------
    # 2. Create GUI
    # ------------------------------------------------------------------
    print("\n=== GUI Controls ===")
    print("  Sliders: Start bar pose, End bar pose, Grasp offset")
    print("  Buttons: Set Grasp, Solve Start IK, Solve End IK, Plan Path")
    print("====================\n")

    start_sliders = PoseSliders("Start", cid=cid, defaults=[0.4, 0.0, 0.75, np.pi, np.pi / 2, np.pi / 2])
    end_sliders = PoseSliders("End", cid=cid, defaults=[0.3, 0.2, 0.6, np.pi, np.pi / 2, 0.0])
    grasp_sliders = PoseSliders(
        "Grasp",
        cid=cid,
        defaults=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        pos_range=0.3,
        angle_range=np.pi,
    )

    btn_set_grasp = Button("Set Grasp From Current", cid=cid)
    btn_solve_start = Button("Solve Start IK", cid=cid)
    btn_solve_end = Button("Solve End IK", cid=cid)
    btn_plan = Button("Plan Path", cid=cid)

    # Mode selector: which bar pose to show
    mode_slider = pybullet.addUserDebugParameter("Show (0=start, 1=end)", 0, 1, 0, physicsClientId=cid)

    # ------------------------------------------------------------------
    # 3. State
    # ------------------------------------------------------------------
    grasp_bar_from_right = None
    grasp_bar_from_left = None
    projector = None
    solver = None

    start_confs = None
    end_confs = None
    start_conf_idx_slider = None
    end_conf_idx_slider = None

    path = None
    path_slider = None
    path_current_idx = -1

    # Ghost bars for showing start (green) and end (red) simultaneously
    ghost_start = pp.create_box(0.5, 0.03, 0.03, color=(0.0, 0.8, 0.0, 0.4))
    ghost_end = pp.create_box(0.5, 0.03, 0.03, color=(0.8, 0.0, 0.0, 0.4))

    # ------------------------------------------------------------------
    # 4. Main loop
    # ------------------------------------------------------------------
    print("Ready. Adjust sliders and press buttons in the PyBullet GUI.")

    while True:
        try:
            # Read current slider poses
            start_pose = start_sliders.as_pose()
            end_pose = end_sliders.as_pose()

            # Show ghost bars for start and end
            pp.set_pose(ghost_start, start_pose)
            pp.set_pose(ghost_end, end_pose)

            # Move the actual bar to whichever mode is selected
            mode = int(round(pybullet.readUserDebugParameter(mode_slider, physicsClientId=cid)))
            if mode == 0:
                pp.set_pose(bar, start_pose)
            else:
                pp.set_pose(bar, end_pose)

            # ---- Set Grasp ----
            if btn_set_grasp.pressed():
                # Compute grasp from current right/left tool pose relative to bar
                world_from_right = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_right)
                world_from_left = pp.get_link_pose(robot_setup.robot, robot_setup.tool_link_left)
                world_from_bar = pp.get_pose(bar)

                grasp_bar_from_right = pp.multiply(pp.invert(world_from_bar), world_from_right)
                grasp_bar_from_left = pp.multiply(pp.invert(world_from_bar), world_from_left)

                # Compute relative transform for dual-arm projection
                desired_right_from_left = pp.multiply(pp.invert(world_from_right), world_from_left)
                projector = DualArmProjection(robot_setup, desired_right_from_left)

                # Set globals for the solver
                solver_mod.bar_from_right = grasp_bar_from_right
                solver_mod.bar_from_left = grasp_bar_from_left

                print(f"Grasp set!")
                print(f"  bar_from_right: pos={np.round(grasp_bar_from_right[0], 4)}, "
                      f"quat={np.round(grasp_bar_from_right[1], 4)}")
                print(f"  bar_from_left:  pos={np.round(grasp_bar_from_left[0], 4)}, "
                      f"quat={np.round(grasp_bar_from_left[1], 4)}")

                # Reset path since grasp changed
                path = None
                start_confs = None
                end_confs = None

            # ---- Solve Start IK ----
            if btn_solve_start.pressed():
                if projector is None or grasp_bar_from_right is None:
                    print("Set grasp first!")
                else:
                    timer.start("solve_start_ik")
                    with pp.LockRenderer():
                        start_confs = projector.create_valid_confs(
                            robot_setup.ik_solver_right,
                            start_pose,
                            grasp_bar_from_right,
                            delta=np.pi,
                            max_attempts=20,
                            collision_fn=robot_setup.create_collision_fn(obstacle_bodies=robot_setup.obstacles),
                        )
                    elapsed = timer.stop("solve_start_ik")

                    if start_confs is not None:
                        start_confs = normalize_angles(start_confs)
                        print(f"Start IK: {len(start_confs)} solutions found in {elapsed:.3f}s")
                        # Show first solution
                        robot_setup.set_joint_positions(robot_setup.arm_joints, start_confs[0])
                        if len(start_confs) > 1:
                            start_conf_idx_slider = pybullet.addUserDebugParameter(
                                "Start conf idx", 0, len(start_confs) - 1, 0, physicsClientId=cid
                            )
                    else:
                        print(f"Start IK: no solution found ({elapsed:.3f}s)")

            # ---- Solve End IK ----
            if btn_solve_end.pressed():
                if projector is None or grasp_bar_from_right is None:
                    print("Set grasp first!")
                else:
                    timer.start("solve_end_ik")
                    with pp.LockRenderer():
                        end_confs = projector.create_valid_confs(
                            robot_setup.ik_solver_right,
                            end_pose,
                            grasp_bar_from_right,
                            delta=np.pi,
                            max_attempts=20,
                            collision_fn=robot_setup.create_collision_fn(obstacle_bodies=robot_setup.obstacles),
                        )
                    elapsed = timer.stop("solve_end_ik")

                    if end_confs is not None:
                        end_confs = normalize_angles(end_confs)
                        print(f"End IK: {len(end_confs)} solutions found in {elapsed:.3f}s")
                        robot_setup.set_joint_positions(robot_setup.arm_joints, end_confs[0])
                        if len(end_confs) > 1:
                            end_conf_idx_slider = pybullet.addUserDebugParameter(
                                "End conf idx", 0, len(end_confs) - 1, 0, physicsClientId=cid
                            )
                    else:
                        print(f"End IK: no solution found ({elapsed:.3f}s)")

            # ---- Plan Path ----
            if btn_plan.pressed():
                if start_confs is None or end_confs is None:
                    print("Solve start and end IK first!")
                elif projector is None:
                    print("Set grasp first!")
                else:
                    start_conf = start_confs[0]
                    end_conf = end_confs[0]

                    # Create solver (uses cached collision fn)
                    solver = TrajectoryDualCartConstrainedSolver(robot_setup, None, projector)

                    print("Planning path...")
                    timer.start("plan_path")
                    path = solver.plan(
                        start_conf=start_conf,
                        target_conf=end_conf,
                        max_time=60,
                        max_iterations=5000,
                        max_attempts=10,
                        use_draw=True,
                        verbose=True,
                    )
                    elapsed = timer.stop("plan_path")

                    if path is not None:
                        print(f"Path found! {len(path)} waypoints in {elapsed:.3f}s")
                        path_slider = pybullet.addUserDebugParameter(
                            "Path idx", 0, len(path) - 1, 0, physicsClientId=cid
                        )
                        path_current_idx = -1
                    else:
                        print(f"No path found ({elapsed:.3f}s)")

                    timer.summary()

            # ---- Browse start configs ----
            if start_confs is not None and start_conf_idx_slider is not None and path is None:
                idx = int(pybullet.readUserDebugParameter(start_conf_idx_slider, physicsClientId=cid))
                idx = min(idx, len(start_confs) - 1)
                robot_setup.set_joint_positions(robot_setup.arm_joints, start_confs[idx])

            # ---- Browse end configs ----
            if end_confs is not None and end_conf_idx_slider is not None and path is None:
                if mode == 1:
                    idx = int(pybullet.readUserDebugParameter(end_conf_idx_slider, physicsClientId=cid))
                    idx = min(idx, len(end_confs) - 1)
                    robot_setup.set_joint_positions(robot_setup.arm_joints, end_confs[idx])

            # ---- Path playback ----
            if path is not None and path_slider is not None:
                idx = int(pybullet.readUserDebugParameter(path_slider, physicsClientId=cid))
                idx = min(idx, len(path) - 1)
                if idx != path_current_idx:
                    path_current_idx = idx
                    robot_setup.set_joint_positions(robot_setup.arm_joints, path[idx])

            time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nExiting...")
            break

    pp.disconnect()


if __name__ == "__main__":
    main()
