import os
import json

from termcolor import colored
import argparse

import load_pddlstream

from pddlstream.utils import read, write
from pddlstream.language.constants import And, Equal, TOTAL_COST, Not
from pddlstream.language.temporal import parse_domain

from load_pddlstream import HERE
from utils import LOGGER
from export_pddl_utils import pddl_problem_with_original_names

MT_DATA_PATH = os.path.join(HERE, '..', '..', 'data', 'multi_tangent_data', 'mt_results')

############################################

PDDL_FOLDERS = ['01_joint_only', 
                ]
DOMAIN_NAMES = ['joint_only', 
                ]

############################################

def parse_mt(mt_json_file_name):
    file_path = os.path.join(MT_DATA_PATH, mt_json_file_name)
    with open(file_path, 'r') as f:
        json_data = json.load(f)

    line_pt_pairs = json_data['line_pt_pairs']
    contact_id_pairs = json_data['contact_id_pairs']
    beam_ids = [f'b{i}' for i in range(len(line_pt_pairs))]

    if 'opt_parameters' in json_data:
        bar_radius = json_data['opt_parameters'].get('bar_radius', 0.01)
        clamp_gap = json_data['opt_parameters'].get('clamp_gap', 0.016)
    else:
        bar_radius = 0.01
        clamp_gap = 0.016

    return beam_ids, contact_id_pairs

############################################

def init_with_cost(manipulate_cost=5.0):
    init = [
        Equal(('Cost',), manipulate_cost),
        Equal((TOTAL_COST,), 0)
    ]
    return init

def add_robots_to_init_goal(
        init=[], goal=[],
        number_of_robots=2):
    for i in range(number_of_robots):
        init.extend([
            ('Robot', f'r{i}'),
            ('RobotFree', f'r{i}'),
        ])
        goal.extend([
            # Not(('RobotHold', f'r{i}')),
            ('RobotFree', f'r{i}'),
        ])
    return init, goal

def extract_pddl_domain_name(pddl_folder):
    domain_pddl = read(os.path.join(pddl_folder, 'domain.pddl'))
    domain_name = parse_domain(domain_pddl).pddl
    return domain_name

def mt_to_init_goal_beams(
        beam_ids,
        init=[], goal=[],
        declare_static=False,
):
    # * All Beams
    for i, beam_id in enumerate(beam_ids):
        # Declare init and goal predicates
        init.extend([
            ('BeamAtStorage', beam_id),
        ])
        goal.extend([
            ('BeamAtAssembled', beam_id),
        ])
        # Declare static predicate of beam
        if declare_static:
            init.extend([
                ('Beam', beam_id),
            ])

    # * Grounded Beams
    # ! hardcoded for now
    if True:
        init.extend([
            ('GroundedBeam', 'b0'),
            ('GroundedBeam', 'b1'),
        ])

    return init, goal

def mt_to_init_goal_joints(
        contact_id_pairs,
        init=[], goal=[],
        num_elements_to_export=-1,
):
    # * Joints
    for joint_pair in contact_id_pairs:
        c0, c1 = joint_pair
        init.extend([
            ('Joint', f'b{c0}', f'b{c1}'),
            ('Joint', f'b{c1}', f'b{c0}'),
        ])

    return init, goal

# Utility functions for parsing

def export_pddl(domain_name, init, goal, pddl_folder, problem_name):
    """export PDDL domain file
    """
    # parse domain pddl to make sure the domain and problem have consistent names

    # [parsed_domain_name] = re.findall(r'\(domain ([^ ]+)\)', domain_name)
    # problem_pddl_str = pddl_problem_with_original_names(problem_name, parsed_domain_name, init, goal)
    problem_pddl_str = pddl_problem_with_original_names(
        problem_name, domain_name, init, goal)

    pddl_problem_path = os.path.join(
        HERE, pddl_folder, 'problem_' + problem_name + '.pddl')
    write(pddl_problem_path, problem_pddl_str)
    LOGGER.info(colored('Exported PDDL domain file to {}'.format(
        pddl_problem_path), 'green'))

def mt_to_init_goal_by_case(
        beam_ids, contact_id_pairs,
        case_number: int,
        init=[], goal=[],
):
    # Extract init and goal
    if case_number == 1:
        init, goal = add_robots_to_init_goal(init, goal, 
                                             number_of_robots=2)
        init, goal = mt_to_init_goal_beams(
            beam_ids, init, goal)
        init, goal = mt_to_init_goal_joints(
            contact_id_pairs, init, goal)

    unioned_goal = And(*goal)
    return init, unioned_goal

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Problem info (Input)
    parser.add_argument('--mt_file_name', default='one_tet_MT_layer_0_contact.json',
                        help='The name of the multi tangent file to solve (json file\'s name, e.g. "box_MT_layer_1.json")')

    # Planning Problem Scope
    parser.add_argument('--planning_cases', metavar='N', type=int, nargs='+',
                        help='Which planning case to parse')

    args = parser.parse_args()

    # Load process file
    mt_file_name = args.mt_file_name
    mt = parse_mt(mt_file_name)

    mt_name = os.path.splitext(os.path.basename(mt_file_name))[0]
    problem_name = mt_name

    # Hard coded domain names and folder names
    pddl_folders = PDDL_FOLDERS
    domain_names = DOMAIN_NAMES

    # Create PDDL problem from process and export
    for case_number in range(1, 6):
        if case_number in args.planning_cases:
            # Extract init and goal
            init, unioned_goal = mt_to_init_goal_by_case(
                *mt, case_number, [], [])
            # Export PDDL domain file
            export_pddl(domain_names[case_number - 1], init,
                        unioned_goal, pddl_folders[case_number - 1], problem_name)
