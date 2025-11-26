import numpy as np
import pybullet_planning as pp
import robotic as ry
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm

print("The path where model files are pre-installed:\n", ry.raiPath(""))

C = ry.Config()

CYLINDER_SIZE = [1.0, 0.01]
PROTRUSION_OFFSET = 0.15
VERTICAL_DISTANCE = 0.04
HALF_LENGTH = 0.5
VERTICAL_Z = 0.5
HORIZONTAL_Z = [0.75, 0.77, 0.79]
ROBOT_DISTANCE = -0.25

v1_pos = [0.5, 0.0, VERTICAL_Z]
v2_pos = [-0.25, 0.433, VERTICAL_Z]
v3_pos = [-0.25, -0.433, VERTICAL_Z]


def horizontal_cylinder_quaternion(direction):
    dir_norm = direction / np.linalg.norm(direction)
    z_axis = np.array([0, 0, 1])
    rot_axis = np.cross(z_axis, dir_norm)
    if np.linalg.norm(rot_axis) < 1e-6:
        rot_axis = np.array([1, 0, 0])
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    angle = np.pi / 2
    return [np.cos(angle / 2), rot_axis[0] * np.sin(angle / 2), rot_axis[1] * np.sin(angle / 2), rot_axis[2] * np.sin(angle / 2)]


def normalize_vector(vec, default=None):
    if default is None:
        default = np.array([1, 0, 0])
    return vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 1e-6 else default


def create_horizontal_element(config: ry.Config, name: str, v_start: np.ndarray, v_end: np.ndarray, protrusion_target: np.ndarray, z_pos: float, color: list[float]) -> tuple[np.ndarray, np.ndarray]:
    edge_dir = np.array([v_end[0] - v_start[0], v_end[1] - v_start[1], 0])
    edge_mid = np.array([(v_start[0] + v_end[0]) / 2, (v_start[1] + v_end[1]) / 2, z_pos])
    protrusion_dir = np.array([protrusion_target[0] - edge_mid[0], protrusion_target[1] - edge_mid[1], 0])
    protrusion_dir = normalize_vector(protrusion_dir) * PROTRUSION_OFFSET
    element_pos = edge_mid + protrusion_dir

    config.addFrame(name).setShape(ry.ST.cylinder, CYLINDER_SIZE).setPosition(element_pos.tolist()).setQuaternion(horizontal_cylinder_quaternion(edge_dir)).setColor(color).setContact(1)

    return element_pos, edge_dir


def calculate_element_end(element_pos, edge_dir, target_pos):
    edge_dir_norm = normalize_vector(edge_dir)
    dir_to_target = np.array([target_pos[0] - element_pos[0], target_pos[1] - element_pos[1], 0])
    dir_to_target_norm = normalize_vector(dir_to_target, edge_dir_norm)
    dot = np.dot(edge_dir_norm, dir_to_target_norm)
    return element_pos + edge_dir_norm * HALF_LENGTH * (1 if dot > 0 else -1)


def position_vertical_element(element_end, initial_v_pos):
    dir_to_v = np.array([initial_v_pos[0] - element_end[0], initial_v_pos[1] - element_end[1], 0])
    dir_to_v_norm = normalize_vector(dir_to_v)
    v_pos = (element_end + dir_to_v_norm * VERTICAL_DISTANCE).tolist()
    v_pos[2] = VERTICAL_Z
    return v_pos


def create_vertical_element(config: ry.Config, name: str, position: list[float], color: list[float]) -> None:
    config.addFrame(name).setShape(ry.ST.cylinder, CYLINDER_SIZE).setPosition(position).setColor(color).setContact(1)


def calculate_robot_position(element_pos, edge_dir, distance):
    edge_dir_norm = normalize_vector(edge_dir)
    perp_dir = np.array([-edge_dir_norm[1], edge_dir_norm[0], 0])
    element_xy = np.array([element_pos[0], element_pos[1], 0.0])
    robot_pos = element_xy + perp_dir * distance
    return robot_pos.tolist()


if __name__ == "__main__":
    element_4_pos, edge_23_dir = create_horizontal_element(C, "element_4", v2_pos, v3_pos, v1_pos, HORIZONTAL_Z[0], [1, 1, 0])
    element_5_pos, edge_31_dir = create_horizontal_element(C, "element_5", v3_pos, v1_pos, v2_pos, HORIZONTAL_Z[1], [1, 0, 1])
    element_6_pos, edge_12_dir = create_horizontal_element(C, "element_6", v1_pos, v2_pos, v3_pos, HORIZONTAL_Z[2], [0, 1, 1])

    element_4_end = calculate_element_end(element_4_pos, edge_23_dir, v1_pos)
    element_5_end = calculate_element_end(element_5_pos, edge_31_dir, v2_pos)
    element_6_end = calculate_element_end(element_6_pos, edge_12_dir, v3_pos)
    edge_12_dir_norm = normalize_vector(edge_12_dir)
    dir_to_v3 = np.array([v3_pos[0] - element_6_pos[0], v3_pos[1] - element_6_pos[1], 0])
    dot_6 = np.dot(edge_12_dir_norm, normalize_vector(dir_to_v3, edge_12_dir_norm))
    element_6_other_end = element_6_pos - edge_12_dir_norm * HALF_LENGTH * (1 if dot_6 > 0 else -1)

    v1_pos = position_vertical_element(element_4_end, v1_pos)
    v2_pos = position_vertical_element(element_5_end, v2_pos)
    v3_pos = position_vertical_element(element_6_other_end, v3_pos)

    create_vertical_element(C, "element_1", v1_pos, [1, 0, 0])
    create_vertical_element(C, "element_2", v2_pos, [0, 1, 0])
    create_vertical_element(C, "element_3", v3_pos, [0, 0, 1])

    robot_1_pos = calculate_robot_position(element_4_pos, edge_23_dir, ROBOT_DISTANCE)
    robot_2_pos = calculate_robot_position(element_5_pos, edge_31_dir, ROBOT_DISTANCE)
    robot_3_pos = calculate_robot_position(element_6_pos, edge_12_dir, ROBOT_DISTANCE)

    C.addFile(ry.raiPath("panda/panda.g"), "r1_").setPosition(robot_1_pos).setQuaternion([1, 0, 0, 1])
    C.addFile(ry.raiPath("panda/panda.g"), "r2_").setPosition(robot_2_pos).setQuaternion([1, 0, 0, 1])
    C.addFile(ry.raiPath("panda/panda.g"), "r3_").setPosition(robot_3_pos).setQuaternion([1, 0, 0, 1])

    # C.computeCollisions()
    # print("\ncollisions: ", C.getCollisions())
    # print("\naccumulatedCollisions: ", C.eval(ry.FS.accumulatedCollisions, []))
    # print("\ntotalPenetration: ", C.getCollisionsTotalPenetration())

    C.view()
    # pp.wait_for_user()

    def draw_pose(config, frame_name, pose_name_prefix, length=0.1):
        """Draw pose using marker frames"""
        frame = config.getFrame(frame_name)
        pos = frame.getPosition()
        quat = frame.getQuaternion()

        config.addFrame(f"{pose_name_prefix}_marker").setShape(ry.ST.marker, [length]).setPosition(pos).setQuaternion(quat).setColor([1, 1, 0])

    def generate_random_initial_state(config):
        """Generate a random initial joint state"""
        q0 = config.getJointState()
        # q_random = q0 + np.random.uniform(-0.5, 0.5, size=len(q0))
        q_random = np.random.uniform(-np.pi, np.pi, size=len(q0))
        return q_random

    def check_gripper_constraints(config, q_state, eps=1e-3):
        """Manually check gripper constraints using eval"""
        config.setJointState(q_state)

        eq_constraints = []
        ineq_constraints = []

        scalar_product_xy_4 = config.eval(ry.FS.scalarProductXY, ["element_4", "r1_gripper"])
        scalar_product_yy_4 = config.eval(ry.FS.scalarProductYY, ["element_4", "r1_gripper"])
        val_xy_4 = float(scalar_product_xy_4[0][0])
        val_yy_4 = float(scalar_product_yy_4[0][0])
        eq_constraints.append(("scalarProductXY_4", val_xy_4, 0))
        eq_constraints.append(("scalarProductYY_4", val_yy_4, 0))

        scalar_product_xy_5 = config.eval(ry.FS.scalarProductXY, ["element_5", "r2_gripper"])
        scalar_product_yy_5 = config.eval(ry.FS.scalarProductYY, ["element_5", "r2_gripper"])
        val_xy_5 = float(scalar_product_xy_5[0][0])
        val_yy_5 = float(scalar_product_yy_5[0][0])
        eq_constraints.append(("scalarProductXY_5", val_xy_5, 0))
        eq_constraints.append(("scalarProductYY_5", val_yy_5, 0))

        scalar_product_xy_6 = config.eval(ry.FS.scalarProductXY, ["element_6", "r3_gripper"])
        scalar_product_yy_6 = config.eval(ry.FS.scalarProductYY, ["element_6", "r3_gripper"])
        val_xy_6 = float(scalar_product_xy_6[0][0])
        val_yy_6 = float(scalar_product_yy_6[0][0])
        eq_constraints.append(("scalarProductXY_6", val_xy_6, 0))
        eq_constraints.append(("scalarProductYY_6", val_yy_6, 0))

        position_rel_1 = config.eval(ry.FS.positionRel, ["r1_gripper", "element_4"])
        pos_rel_1 = position_rel_1[0]
        val_pos_1_z = float(pos_rel_1[2])
        ineq_constraints.append(("positionRel_1_z", val_pos_1_z, 0.5, -0.5))

        position_rel_2 = config.eval(ry.FS.positionRel, ["r2_gripper", "element_5"])
        pos_rel_2 = position_rel_2[0]
        val_pos_2_z = float(pos_rel_2[2])
        ineq_constraints.append(("positionRel_2_z", val_pos_2_z, 0.5, -0.5))

        position_rel_3 = config.eval(ry.FS.positionRel, ["r3_gripper", "element_6"])
        pos_rel_3 = position_rel_3[0]
        val_pos_3_z = float(pos_rel_3[2])
        ineq_constraints.append(("positionRel_3_z", val_pos_3_z, 0.5, -0.5))

        accumulated_collisions = config.eval(ry.FS.accumulatedCollisions, [])
        val_collisions = float(accumulated_collisions[0][0])
        eq_constraints.append(("accumulatedCollisions", val_collisions, 0))

        all_eq_satisfied = all(abs(val - target) < eps for _, val, target in eq_constraints)
        all_ineq_satisfied = all(val <= upper and val >= lower for _, val, upper, lower in ineq_constraints)

        return all_eq_satisfied and all_ineq_satisfied, eq_constraints, ineq_constraints

    def solve_komo_problem(komo, max_attempts, C, view=False, mult=3, offset=-1.5, damping=None, wolfe=None, initial_state=None, eps=1e-3):
        for num_attempt in range(max_attempts):
            if num_attempt == 0 and initial_state is not None:
                komo.initWithConstant(initial_state)
            elif num_attempt > 0:
                dim = len(C.getJointState())
                x_init = np.random.rand(dim) * mult + offset
                komo.initWithConstant(x_init)

            solver = ry.NLP_Solver(komo.nlp(), verbose=0)

            if damping is not None:
                solver.setOptions(damping=damping)
            if wolfe is not None:
                solver.setOptions(wolfe=wolfe)

            retval = solver.solve()
            retval = retval.dict()

            if view:
                print(retval)
                komo.view(True, "IK solution")

            if retval["feasible"]:
                keyframes = komo.getPath()
                if keyframes is not None and len(keyframes) > 0:
                    is_feasible, eq_vals, ineq_vals = check_gripper_constraints(C, keyframes[0], eps=eps)
                    if is_feasible:
                        return retval, keyframes

        return retval, None

    def solve_with_weights(joint_weight, gripper_weight, initial_state, max_attempts=10, view=False, damping=None, wolfe=None):
        komo = ry.KOMO(C, 1, 1, 0, True)

        komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [joint_weight], initial_state)

        komo.addObjective([], ry.FS.scalarProductXY, ["element_4", "r1_gripper"], ry.OT.eq, [gripper_weight], [0])
        komo.addObjective([], ry.FS.scalarProductYY, ["element_4", "r1_gripper"], ry.OT.eq, [gripper_weight], [0])
        komo.addObjective([], ry.FS.positionRel, ["r1_gripper", "element_4"], ry.OT.ineq, [gripper_weight], [0, 0, 0.5])
        komo.addObjective([], ry.FS.positionRel, ["r1_gripper", "element_4"], ry.OT.ineq, [-gripper_weight], [0, 0, -0.5])

        komo.addObjective([], ry.FS.scalarProductXY, ["element_5", "r2_gripper"], ry.OT.eq, [gripper_weight], [0])
        komo.addObjective([], ry.FS.scalarProductYY, ["element_5", "r2_gripper"], ry.OT.eq, [gripper_weight], [0])
        komo.addObjective([], ry.FS.positionRel, ["r2_gripper", "element_5"], ry.OT.ineq, [gripper_weight], [0, 0, 0.5])
        komo.addObjective([], ry.FS.positionRel, ["r2_gripper", "element_5"], ry.OT.ineq, [-gripper_weight], [0, 0, -0.5])

        komo.addObjective([], ry.FS.scalarProductXY, ["element_6", "r3_gripper"], ry.OT.eq, [gripper_weight], [0])
        komo.addObjective([], ry.FS.scalarProductYY, ["element_6", "r3_gripper"], ry.OT.eq, [gripper_weight], [0])
        komo.addObjective([], ry.FS.positionRel, ["r3_gripper", "element_6"], ry.OT.ineq, [gripper_weight], [0, 0, 0.5])
        komo.addObjective([], ry.FS.positionRel, ["r3_gripper", "element_6"], ry.OT.ineq, [-gripper_weight], [0, 0, -0.5])

        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)

        ret_dict, keyframes = solve_komo_problem(komo, max_attempts, C, view=view, damping=damping, wolfe=wolfe, initial_state=initial_state)

        class RetWrapper:
            def __init__(self, ret_dict):
                self.feasible = ret_dict.get("feasible", False)
                self.eq = ret_dict.get("eq", float("inf"))
                self.ineq = ret_dict.get("ineq", float("inf"))
                self.sos = ret_dict.get("sos", float("inf"))

        ret = RetWrapper(ret_dict)

        return ret, komo

    joint_weight_start = 1e-2
    joint_weight_end = 1e1
    joint_weight_num = 25
    joint_weights = np.logspace(np.log10(joint_weight_start), np.log10(joint_weight_end), joint_weight_num)

    gripper_weight_start = 1e-2
    gripper_weight_end = 1e3
    gripper_weight_num = 25
    gripper_weights = np.logspace(np.log10(gripper_weight_start), np.log10(gripper_weight_end), gripper_weight_num)

    num_initial_states = 20
    print(f"Generating {num_initial_states} random initial states")
    print(f"For each initial state, searching {len(joint_weights)} x {len(gripper_weights)} = {len(joint_weights) * len(gripper_weights)} weight combinations")
    print(f"Joint weights range: {joint_weight_start} to {joint_weight_end}")
    print(f"Gripper weights range: {gripper_weight_start} to {gripper_weight_end}")

    np.random.seed(42)

    initial_states = []
    for i in range(num_initial_states):
        q_init = generate_random_initial_state(C)
        initial_states.append(q_init)
        print(f"Initial state {i+1}: {q_init[:5]}...")  # Print first 5 values

    pp.wait_for_user()

    # Initialize count matrices for heatmaps
    unique_joint_weights = sorted(set(joint_weights))
    unique_gripper_weights = sorted(set(gripper_weights))
    count_matrix_before = np.zeros((len(unique_gripper_weights), len(unique_joint_weights)))  # Before check_gripper_constraints
    count_matrix_after = np.zeros((len(unique_gripper_weights), len(unique_joint_weights)))  # After check_gripper_constraints

    all_results = []

    for state_idx, initial_state in enumerate(initial_states):
        print(f"\n{'='*60}")
        print(f"Initial State {state_idx + 1}/{num_initial_states}")
        print(f"{'='*60}")

        C.setJointState(initial_state)

        feasible_config = None
        feasible_ret = None
        feasible_komo = None

        feasible_configs_for_state = []

        total_combinations = len(joint_weights) * len(gripper_weights)
        with tqdm(total=total_combinations, desc=f"  State {state_idx + 1}: Weight combinations", leave=False, unit="combo") as pbar:
            for joint_weight in joint_weights:
                for gripper_weight in gripper_weights:
                    # print(f"Trying joint_weight={joint_weight:.3e}, gripper_weight={gripper_weight:.3e}", end=" ... ")
                    ret, komo = solve_with_weights(joint_weight, gripper_weight, initial_state, max_attempts=1)
                    # print(f"feasible={ret.feasible}, eq={ret.eq:.3e}, ineq={ret.ineq:.3e}, sos={ret.sos:.3e}")

                    # Get indices for count matrices
                    joint_idx = unique_joint_weights.index(joint_weight)
                    gripper_idx = unique_gripper_weights.index(gripper_weight)

                    if ret.feasible:
                        # Count before check_gripper_constraints
                        count_matrix_before[gripper_idx, joint_idx] += 1

                        q = komo.getPath()
                        is_feasible, eq_vals, ineq_vals = check_gripper_constraints(C, q[0], eps=1e-3)

                        if is_feasible:
                            # Count after check_gripper_constraints
                            count_matrix_after[gripper_idx, joint_idx] += 1
                            # print(f"✓ Found feasible configuration: joint_weight={joint_weight:.3e}, gripper_weight={gripper_weight:.3e}")
                            feasible_configs_for_state.append(
                                {
                                    "joint_weight": joint_weight,
                                    "gripper_weight": gripper_weight,
                                    "q": q[0].tolist(),
                                    "ret_eq": float(ret.eq),
                                    "ret_ineq": float(ret.ineq),
                                    "ret_sos": float(ret.sos),
                                    "eq_constraints": {name: float(val) for name, val, _ in eq_vals},
                                    "ineq_constraints": {name: float(val) for name, val, _, _ in ineq_vals},
                                }
                            )
                        else:
                            # print(f"  Filtered out: constraints not satisfied")
                            pass

                    pbar.update(1)

                # q = komo.getPath()
                # C.setJointState(q[0])
                # C.view()

        if len(feasible_configs_for_state) > 0:
            feasible_config = (feasible_configs_for_state[0]["joint_weight"], feasible_configs_for_state[0]["gripper_weight"])
            feasible_q = feasible_configs_for_state[0]["q"]
        else:
            feasible_config = None
            feasible_q = None

        result = {
            "state_idx": state_idx,
            "initial_state": initial_state.tolist(),
            "feasible": feasible_config is not None,
            "feasible_configs": feasible_configs_for_state,
            "joint_weight": feasible_config[0] if feasible_config else None,
            "gripper_weight": feasible_config[1] if feasible_config else None,
            "q": feasible_q,
        }
        all_results.append(result)

        if feasible_config is None:
            print(f"✗ No feasible configuration found for initial state {state_idx + 1}")
        else:
            print(f"✓ Feasible configuration found for initial state {state_idx + 1}")

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    feasible_count = sum(1 for r in all_results if r["feasible"])
    total_feasible_configs = sum(len(r["feasible_configs"]) for r in all_results)
    print(f"Feasible solutions found: {feasible_count}/{num_initial_states} initial states")
    print(f"Total feasible configurations: {total_feasible_configs}")

    all_feasible_configs = []
    for r in all_results:
        for cfg in r["feasible_configs"]:
            all_feasible_configs.append(
                {
                    "state_idx": r["state_idx"],
                    "joint_weight": cfg["joint_weight"],
                    "gripper_weight": cfg["gripper_weight"],
                    "q": cfg["q"],
                    "eq": cfg["ret_eq"],
                    "ineq": cfg["ret_ineq"],
                    "sos": cfg["ret_sos"],
                    "eq_constraints": cfg["eq_constraints"],
                    "ineq_constraints": cfg["ineq_constraints"],
                }
            )

    output_dir = "komo_results"
    os.makedirs(output_dir, exist_ok=True)

    if len(all_feasible_configs) > 0:
        print(f"\nSaving {len(all_feasible_configs)} feasible configurations...")

        results_file = os.path.join(output_dir, "feasible_configs.json")
        with open(results_file, "w") as f:
            json.dump(all_feasible_configs, f, indent=2)
        print(f"Saved to {results_file}")

    print("\nPlotting weight heatmaps...")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    # Heatmap 1: Before check_gripper_constraints (only ret.feasible)
    im1 = ax1.imshow(count_matrix_before, aspect="auto", cmap="YlOrRd", origin="lower", interpolation="nearest", vmin=0, vmax=num_initial_states)

    ax1.set_xticks(range(len(unique_joint_weights)))
    ax1.set_xticklabels([f"{w:.2e}" for w in unique_joint_weights], rotation=45, ha="right")
    ax1.set_yticks(range(len(unique_gripper_weights)))
    ax1.set_yticklabels([f"{w:.2e}" for w in unique_gripper_weights])

    ax1.set_xlabel("Joint Weight", fontsize=12)
    ax1.set_ylabel("Gripper Weight", fontsize=12)
    ax1.set_title(f"Before check_gripper_constraints (Total: {int(count_matrix_before.sum())} configs)", fontsize=14)

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Number of Feasible Configurations", fontsize=11)

    for i in range(len(unique_gripper_weights)):
        for j in range(len(unique_joint_weights)):
            if count_matrix_before[i, j] > 0:
                text = ax1.text(j, i, int(count_matrix_before[i, j]), ha="center", va="center", color="black", fontsize=8)

    # Heatmap 2: After check_gripper_constraints (filtered)
    im2 = ax2.imshow(count_matrix_after, aspect="auto", cmap="YlOrRd", origin="lower", interpolation="nearest", vmin=0, vmax=num_initial_states)

    ax2.set_xticks(range(len(unique_joint_weights)))
    ax2.set_xticklabels([f"{w:.2e}" for w in unique_joint_weights], rotation=45, ha="right")
    ax2.set_yticks(range(len(unique_gripper_weights)))
    ax2.set_yticklabels([f"{w:.2e}" for w in unique_gripper_weights])

    ax2.set_xlabel("Joint Weight", fontsize=12)
    ax2.set_ylabel("Gripper Weight", fontsize=12)
    ax2.set_title(f"After check_gripper_constraints (Total: {len(all_feasible_configs)} configs)", fontsize=14)

    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("Number of Feasible Configurations", fontsize=11)

    for i in range(len(unique_gripper_weights)):
        for j in range(len(unique_joint_weights)):
            if count_matrix_after[i, j] > 0:
                text = ax2.text(j, i, int(count_matrix_after[i, j]), ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, "feasible_weights_heatmap.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {plot_file}")
    plt.close()

    if len(all_feasible_configs) > 0:
        print("\nFeasible configurations summary:")
        for r in all_results:
            if r["feasible"]:
                print(f"  State {r['state_idx']+1}: {len(r['feasible_configs'])} feasible config(s)")

        first_feasible = next((r for r in all_results if r["feasible"]), None)
        if first_feasible:
            print(f"\nUsing first feasible solution (State {first_feasible['state_idx']+1}) for visualization")
            q = np.array(first_feasible["q"])
            C.setJointState(q)

            # draw_pose(C, "r1_gripper", "r1_gripper_pose", length=0.1)
            # draw_pose(C, "element_4", "element_4_pose", length=0.1)
            # draw_pose(C, "r2_gripper", "r2_gripper_pose", length=0.1)
            # draw_pose(C, "element_5", "element_5_pose", length=0.1)
            # draw_pose(C, "r3_gripper", "r3_gripper_pose", length=0.1)
            # draw_pose(C, "element_6", "element_6_pose", length=0.1)

            C.view()
            pp.wait_for_user()
    else:
        print("\nNo feasible configurations found for any initial state!")
        C.view()
        pp.wait_for_user()
