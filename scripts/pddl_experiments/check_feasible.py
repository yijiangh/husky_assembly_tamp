import argparse

import numpy as np
import pybullet_planning as pp
from multi_tangent.collision import create_collision_bodies
from multi_tangent.convert import flatten_list

from parse import parse_mt_geometric
from collision import init_pb, create_couplers, Element
from stream import get_place_gen_fn
from robot_setup import RobotSetup

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Problem info (Input)
    parser.add_argument('--mt_file_name', default='one_tet_MT_layer_0_contact.json',
                        help='The name of the multi tangent file to solve (json file\'s name, e.g. "box_MT_layer_1.json")')

    args = parser.parse_args()

    # Load process file
    mt_file_name = args.mt_file_name
    line_pt_pairs, contact_id_pairs, bar_radius = parse_mt_geometric(mt_file_name)
    line_pts_flattened = flatten_list(np.array(line_pt_pairs))

    min_z = np.min(line_pts_flattened, axis=0)[2]
    line_pts_flattened = [np.array([0, 0, -min_z]) + point for point in line_pts_flattened]

    radius_per_edge = [bar_radius] * int(len(line_pts_flattened)/2)

    init_pb()
    with pp.LockRenderer():
        element_bodies = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)
        half_coupler_from_contact_pair = create_couplers(line_pts_flattened, contact_id_pairs)
        # pp.wait_if_gui('Preview')

        robot_setup0 = RobotSetup('r0')

    element_from_index = {i: Element(i, e, pp.get_pose(e), [line_pts_flattened[2*i], line_pts_flattened[2*i+1]]) for i, e in enumerate(element_bodies)}
    place_gen = get_place_gen_fn(robot_setup0, element_from_index, [], collisions=False, teleops=True)

    for i in element_from_index:
        command, = next(place_gen(i, assembled=list(range(i)), diagnosis=False))
        print(command)
        pp.wait_if_gui()
    