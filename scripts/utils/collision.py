from collections import defaultdict, namedtuple

import pybullet as p
import pybullet_planning as pp

import utils.load_multi_tangent as load_multi_tangent
from multi_tangent.collision import create_swivel_coupler
from multi_tangent.convert import flatten_list, list_to_pairs
from multi_tangent.contact import compute_closest_t_between_lines

Element = namedtuple("Element", ["index", "body", "init_pose", "goal_pose", "axis_endpoints"])

# {link_name: [infos: [offset, radius, visual_id], joint_name, weight]}
collision_info = {
    "base_link": [
        [
            [(0.3160000145435333, 0.2529999911785126, 0.20000000298023224), 0.2, -1],
            [(0.3160000145435333, 0.0, 0.20000000298023224), 0.2, -1],
            [(0.3160000145435333, -0.2529999911785126, 0.20000000298023224), 0.2, -1],
            [(-0.3160000145435333, 0.2529999911785126, 0.20000000298023224), 0.2, -1],
            [(-0.3160000145435333, 0.0, 0.20000000298023224), 0.2, -1],
            [(-0.3160000145435333, -0.2529999911785126, 0.20000000298023224), 0.2, -1],
            [(0.0, 0.2529999911785126, 0.20000000298023224), 0.2, -1],
            [(0.0, 0.0, 0.20000000298023224), 0.2, -1],
            [(0.0, -0.2529999911785126, 0.20000000298023224), 0.2, -1],
        ],
        "base_joint",
        10.0,
    ],
    "ur_arm_base_link_inertia": [
        [[(0.3889999985694885, 0.0, 0.4099999964237213), 0.075, -1]],
        "base_joint",
        5.0,
    ],
    "ur_arm_shoulder_link": [
        [[(0.0, -0.000299990177154541, -0.010500013828277588), 0.075, -1]],
        "ur_arm_shoulder_pan_joint",
        5.0,
    ],
    "ur_arm_upper_arm_link": [
        [
            [(0.010500013828277588, -6.3721117271597905e-09, 0.1373000144958496), 0.075, -1],
            [(-0.08950001001358032, -6.3721117271597905e-09, 0.1373000144958496), 0.05, -1],
            [(-0.18950003385543823, -6.3721117271597905e-09, 0.1373000144958496), 0.05, -1],
            [(-0.28949999809265137, -6.3721117271597905e-09, 0.1373000144958496), 0.05, -1],
            [(-0.40950000286102295, -6.3721117271597905e-09, 0.1373000144958496), 0.075, -1],
        ],
        "ur_arm_shoulder_lift_joint",
        3.0,
    ],
    "ur_arm_forearm_link": [
        [
            [(0.015500009059906006, -5.244373824098147e-10, 0.011299997568130493), 0.075, -1],
            [(-0.0845000147819519, -5.244373824098147e-10, 0.011299997568130493), 0.05, -1],
            [(-0.18450003862380981, -5.244373824098147e-10, 0.011299997568130493), 0.05, -1],
            [(-0.2844999432563782, -5.244373824098147e-10, 0.011299997568130493), 0.05, -1],
            [(-0.3844999670982361, -5.244373824098147e-10, 0.011299997568130493), 0.05, -1],
            [(-0.3844999670982361, 1.796069071247075e-09, -0.03870001435279846), 0.05, -1],
        ],
        "ur_arm_elbow_joint",
        1.0,
    ],
    "ur_arm_wrist_1_link": [
        [[(0.007700085639953613, -6.325699075659941e-09, 0.0029999613761901855), 0.05, -1]],
        "ur_arm_wrist_1_joint",
        1.0,
    ],
    "ur_arm_wrist_2_link": [
        [[(0.007700085639953613, 0.0029999613761901855, 0.00030000507831573486), 0.05, -1]],
        "ur_arm_wrist_2_joint",
        1.0,
    ],
    "ur_arm_wrist_3_link": [
        [[(0.0, 0.0, 0.0), 0.05, -1]],
        "ur_arm_wrist_3_joint",
        1.0,
    ],
    "gripper_link": [
        [[(0.0, 0.0, 0.1), 0.075, -1]],
        "ur_arm_wrist_3_joint",
        1.0,
    ],
}


class Grasp(object):
    def __init__(self, element, gripper_from_object):
        self.element = element  # bar vertex key
        self.gripper_from_object = gripper_from_object

    def __repr__(self):
        return "{}(E{})".format(self.__class__.__name__, self.element)


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
        t1, t2 = compute_closest_t_between_lines(
            line_pts_flattened[ei * 2],
            line_pts_flattened[ei * 2 + 1],
            line_pts_flattened[ej * 2],
            line_pts_flattened[ej * 2 + 1],
        )
        contact_ts.extend([t1, t2])

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
