import os
import json

from load_pddlstream import HERE
from parse_symbolic import MT_DATA_PATH, PDDL_FOLDERS

def parse_mt_geometric(mt_json_file_name):
    file_path = os.path.join(MT_DATA_PATH, mt_json_file_name)
    with open(file_path, 'r') as f:
        json_data = json.load(f)

    line_pt_pairs = json_data['line_pt_pairs']
    # z up to y up
    # line_pt_pairs = [[[pt_pair[0][1], pt_pair[0][2], pt_pair[0][0]],
    #                   [pt_pair[1][1], pt_pair[1][2], pt_pair[1][0]]] for pt_pair in line_pt_pairs]

    contact_id_pairs = json_data['contact_id_pairs']

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

