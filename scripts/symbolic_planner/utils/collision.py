from collections import defaultdict, namedtuple

import pybullet as p
import pybullet_planning as pp

import utils.load_multi_tangent as load_multi_tangent
from multi_tangent.collision import create_swivel_coupler
from multi_tangent.convert import flatten_list, list_to_pairs
from multi_tangent.contact import compute_closest_t_between_lines

Element = namedtuple('Element', ['index', 'body', 'init_pose', 'goal_pose', 'axis_endpoints'])

class Grasp(object):
    def __init__(self, element, gripper_from_object):
        self.element = element # bar vertex key
        self.gripper_from_object = gripper_from_object
    def __repr__(self):
        return '{}(E{})'.format(self.__class__.__name__, self.element)

def init_pb():
    # * start pybullet simulator
    pp.connect(use_gui=True, shadows=True, color=[0.9, 0.9, 1.0])

    # * y-up to be consistent with mocap
    # p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1, physicsClientId=pp.CLIENT)

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

