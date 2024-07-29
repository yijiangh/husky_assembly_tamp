import os
import json
from more_itertools import last
from termcolor import colored
import argparse

import load_pddlstream

from pddlstream.utils import read, write
from pddlstream.language.constants import And, Equal, TOTAL_COST
from pddlstream.language.temporal import parse_domain

from load_pddlstream import HERE
from utils import LOGGER

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
        ])

    return init, goal

def mt_to_init_goal_joints(
        contact_id_pairs,
        init=[], goal=[],
        num_elements_to_export=-1,
):
    # * Joints
    for c0, c1 in contact_id_pairs:
        init.extend([
            ('Joint', c0, c1),
            ('Joint', c1, c0),
        ])

    return init, goal