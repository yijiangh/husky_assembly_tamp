import argparse
import time

import numpy as np
import pybullet as p
import pybullet_planning as pp

from multi_tangent.collision import create_collision_bodies
from multi_tangent.convert import flatten_list

from parse import parse_mt_geometric, parse_plan_file
from collision import init_pb, create_couplers

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
