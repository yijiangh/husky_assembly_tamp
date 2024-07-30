import os
import json
import argparse
import time
from collections import defaultdict

import numpy as np
import pybullet as p
import pybullet_planning as pp

from load_pddlstream import HERE
from parse_symbolic import MT_DATA_PATH, PDDL_FOLDERS

from multi_tangent.collision import create_collision_bodies, create_swivel_coupler
from multi_tangent.convert import flatten_list, list_to_pairs
from multi_tangent.contact import compute_closest_t_between_lines

def parse_mt_geometric(mt_json_file_name):
    file_path = os.path.join(MT_DATA_PATH, mt_json_file_name)
    with open(file_path, 'r') as f:
        json_data = json.load(f)

    line_pt_pairs = json_data['line_pt_pairs']
    contact_id_pairs = json_data['contact_id_pairs']
    beam_ids = [f'b{i}' for i in range(len(line_pt_pairs))]

    if 'opt_parameters' in json_data:
        bar_radius = json_data['opt_parameters'].get('bar_radius', 0.01)
    else:
        bar_radius = 0.01

    return line_pt_pairs, contact_id_pairs, bar_radius

def parse_plan_file(mt_json_file_name, symbolic_planner, case_number):
    mt_name = mt_json_file_name.split('.')[0]
    plan_file_name = f'result_{mt_name}_{symbolic_planner}.json'
    plan_file_path = os.path.join(HERE, PDDL_FOLDERS[case_number - 1], plan_file_name)

    with open(plan_file_path, 'r') as f:
        json_data = json.load(f)

    return json_data


def init_pb():
    # * start pybullet simulator
    pp.connect(use_gui=True, shadows=True, color=[0.9, 0.9, 1.0])

    # * y-up to be consistent with mocap
    p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1, physicsClientId=pp.CLIENT)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=pp.CLIENT)
    # pp.set_camera(np.deg2rad(92.0), np.deg2rad(-85), 5.20)
    # pp.set_camera(92.0, -85, 5.20)

def create_couplers(line_pts_flattened, contact_id_pairs):
    contact_ts = []
    for ei, ej in contact_id_pairs:
        t1, t2 = compute_closest_t_between_lines(line_pts_flattened[ei*2], line_pts_flattened[ei*2+1], line_pts_flattened[ej*2], line_pts_flattened[ej*2+1])
        contact_ts.extend([t1,t2])

    node_pairs = list_to_pairs(line_pts_flattened)
    contact_t_pairs = list_to_pairs(contact_ts)
    half_couplers = defaultdict(list)
    with pp.LockRenderer():
        for contact_idp, contact_tp in zip(contact_id_pairs, contact_t_pairs):
            e0, e1 = contact_idp
            # collision checking between clamps and bars performed inside
            coupler_pair = create_swivel_coupler(node_pairs, e0, e1, *contact_tp)
            half_couplers[frozenset([e0, e1])] = coupler_pair
    return half_couplers

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Problem info (Input)
    parser.add_argument('--mt_file_name', default='one_tet_MT_layer_0_contact.json',
                        help='The name of the multi tangent file to solve (json file\'s name, e.g. "box_MT_layer_1.json")')
    parser.add_argument('--symbolic_planner', default='fd')

    # Planning Problem Scope
    parser.add_argument('--planning_case', type=int, help='Which planning case to parse')

    args = parser.parse_args()

    # Load process file
    mt_file_name = args.mt_file_name
    line_pt_pairs, contact_id_pairs, bar_radius = parse_mt_geometric(mt_file_name)
    line_pts_flattened = flatten_list(np.array(line_pt_pairs))
    radius_per_edge = [bar_radius] * int(len(line_pts_flattened)/2)

    init_pb()
    element_bodies = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)
    half_coupler_from_contact_pair = create_couplers(line_pts_flattened, contact_id_pairs)
    # pp.wait_if_gui('Preview')

    # move all elements to a place far away to hide
    element_poses = {}
    for i, e in enumerate(element_bodies):
        element_poses[i] = pp.get_pose(e)
        pp.set_pose(e, pp.Pose(pp.Point(0,0,100)))

    # half_coupler_poses = defaultdict(list)
    # for contact_pair, half_couplers in half_coupler_from_contact_pair.items():
    #     for half_coupler in half_couplers:
    #         half_coupler_poses[contact_pair].append(pp.get_pose(half_coupler.body))
    #         pp.set_pose(half_coupler, pp.Pose(pp.Point(0,0,100)))

    plan_dict = parse_plan_file(mt_file_name, args.symbolic_planner, args.planning_case)
    max_action_n = len(plan_dict)

    # * Control UI
    #  For a button, the value of getUserDebugParameter for a button increases 1 at each button press.
    next_button = p.addUserDebugParameter("Next", 1, 0, 0)
    old_next_button_value = p.readUserDebugParameter(next_button)

    current_step = -1 
    old_currrent_step = -1
    while True:
        current_next_value = p.readUserDebugParameter(next_button)
        if current_next_value > old_next_button_value:
            current_step += 1
        old_next_button_value = current_next_value

        if current_step != old_currrent_step and current_step < max_action_n:
            # update drawing
            current_action = plan_dict[current_step]
            current_action_name = current_action['action_name']
            current_action_args = current_action['args']
            # strip away 'b' from current_action_args[0]
            current_element_id = int(current_action_args[0][1:])
            print(f'{current_step}', current_action_name, current_action_args, current_element_id)

            if current_action_name.startswith('assemble_beam'):
                pp.set_pose(element_bodies[current_element_id], element_poses[current_element_id])
                if current_action_name.endswith('grounded'):
                    pp.set_color(element_bodies[current_element_id], pp.RED)
                if current_action_name.endswith('and_hold'):
                    pp.set_color(element_bodies[current_element_id], pp.BLUE)

            if current_action_name.startswith('release_hold'):
                pp.set_color(element_bodies[current_element_id], pp.RED)

            old_currrent_step = current_step

        time.sleep(0.01)
