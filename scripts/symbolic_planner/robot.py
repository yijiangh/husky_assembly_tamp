from typing import Dict, List, Tuple
from copy import deepcopy

import numpy as np
import pybullet_planning as pp
from collision import Element
from pybullet_planning import Attachment, Euler, Point, Pose, get_distance, interpolate_poses, invert, multiply
from robot_setup import RobotSetup
from stream import get_pick_gen_fn, get_place_gen_fn, get_transfer_gen_fn, get_transit_gen_fn
from utils import CounterModule, CounterValue


class PathItem(object):
    def __init__(self, element_index: int, attach: Attachment) -> None:
        self.conf = []
        self.mask = []
        self.element_index = element_index
        self.attach = attach

    def Append(self, confs, masks):
        self.conf.extend(confs)
        self.mask.extend(masks)

    def LeftAppend(self, confs, masks):
        self.conf = confs + self.conf
        self.mask = masks + self.mask


class Robot(object):
    def __init__(
        self,
        index: int,
        robot_setup: RobotSetup,
        element_from_index: Dict[int, Element],
        attachments: List[Attachment] = [],
    ) -> None:
        self.index = index
        self.robot_setup = robot_setup
        self.counter_handle = CounterModule()
        self.place_counter_handle = self.counter_handle.create_handle("place")
        self.pick_counter_handle = self.counter_handle.create_handle("pick")
        self.transfer_counter_handle = self.counter_handle.create_handle("transfer")

        self.element_from_index = element_from_index
        self.attachments = attachments
        self.path = []
        self.last_pose2d = np.array([5, 5, 0])

        self.place_gen = get_place_gen_fn(
            self.robot_setup,
            element_from_index,
            [],
            verbose=False,
            collisions=True,
            teleops=False,
            allow_failure=True,
            max_grasp=20,
        )

        self.pick_gen = get_pick_gen_fn(
            self.robot_setup, element_from_index, [], verbose=False, collisions=True, teleops=False, allow_failure=True
        )

        self.transfer_gen = get_transfer_gen_fn(
            self.robot_setup,
            element_from_index,
            [],
            verbose=False,
            collisions=True,
            teleops=False,
            allow_failure=True,
            max_attempts=10,
        )

        self.transit_gen = get_transit_gen_fn(
            self.robot_setup, element_from_index, [], verbose=True, collisions=True, teleops=False
        )

        self.planner_fn = self.DefaultPlan
        # self.planner_fn = self.MotionPlan

    def DefaultPlan(
        self,
        element_index: int,
        assembled_index_list: List[int],
        unassembled_index_list: List[int],
        attachment_list=List[Attachment],
    ) -> bool:
        return True

    def MotionPlan(
        self,
        element_index: int,
        assembled_index_list: List[int],
        unassembled_index_list: List[int],
        attachment_list=List[Attachment],
    ) -> bool:

        for assembled_element_index in assembled_index_list:
            pp.set_pose(
                self.element_from_index[assembled_element_index].body,
                self.element_from_index[assembled_element_index].goal_pose,
            )

        # -------------------- Place --------------------#
        place_cmd, place_grasp_mask, grasp_attach, grasp, pregrasp_pose = self.PlacePathPlan(
            element_index, assembled_index_list, unassembled_index_list, attachment_list
        )
        if place_cmd is None:
            return False
        path_obj = PathItem(element_index, grasp_attach)
        base_path_obj = PathItem(element_index, grasp_attach)

        # -------------------- Pick --------------------#
        self.UpdateElementsRobot(element_index)
        pick_cmd, pick_grasp_mask = self.PickPathPlan(
            element_index, assembled_index_list, unassembled_index_list, attachment_list, grasp
        )
        if pick_cmd is None:
            return False

        # -------------------- Transfer --------------------#
        transfer_cmd, transfer_grasp_mask = self.TransferPathPlan(
            element_index,
            assembled_index_list,
            unassembled_index_list,
            attachment_list,
            grasp_attach,
            pick_cmd[-1],
            place_cmd[0],
        )
        if transfer_cmd is None:
            return False

        # -------------------- Transit --------------------#
        transit_cmd, transit_grasp_mask = self.TransitPathPlan(
            element_index,
            assembled_index_list,
            attachment_list,
            self.last_pose2d,
            place_cmd[0][:3],
        )
        if transit_cmd is None:
            return False

        # print(transfer_cmd)
        self.last_pose2d = place_cmd[0][:3]

        base_path_obj.Append(transit_cmd, transit_grasp_mask)
        path_obj.Append(pick_cmd, pick_grasp_mask)
        path_obj.Append(transfer_cmd, transfer_grasp_mask)
        path_obj.Append(place_cmd, place_grasp_mask)

        self.path.append((base_path_obj, path_obj))
        return True

    def UpdateElementsRobot(self, element_index: int):
        # 设置当前element到机器人上的托盘
        with pp.LockRenderer():
            ipad_link_pose = pp.get_link_pose(
                self.robot_setup.robot, pp.link_from_name(self.robot_setup.robot, "ipad_rack_link")
            )
            delta_pose = Pose(point=[0, 0, 0.5], euler=Euler(roll=-np.pi / 2, pitch=0, yaw=0))
            bar_pose = multiply(ipad_link_pose, delta_pose)
            pp.set_pose(self.element_from_index[element_index].body, bar_pose)

    def PlacePathPlan(
        self,
        element_index: int,
        assembled_index_list: List[int],
        unassembled_index_list: List[int],
        attachment_list: List[Attachment],
    ) -> Tuple[List[np.ndarray], List[int], Attachment, Tuple, Tuple]:
        """
        @brief: compute place path\n
        ---
        @param:\n
            : \n
        ---
        @return:\n
            command: List[np.ndarray]\n
            grasp_mask: List[int]\n
            grasp_attach: Attachment\n
            grasp: Pose\n
            pregrasp_pose: Pose\n
        """
        with pp.LockRenderer():
            place_cmd, place_grasp_mask, grasp_attach, grasp, pregrasp_pose = next(
                self.place_gen(
                    element_index,
                    assembled=assembled_index_list,
                    unassembled=unassembled_index_list,
                    attachments=attachment_list,
                    counter=self.place_counter_handle,
                )
            )
        return place_cmd, place_grasp_mask, grasp_attach, grasp, pregrasp_pose

    def PickPathPlan(
        self,
        element_index: int,
        assembled_index_list: List[int],
        unassembled_index_list: List[int],
        attachment_list: List[Attachment],
        grasp: Tuple[Tuple, Tuple],
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        @brief: compute pick path\n
        ---
        @param:\n
            : \n
        ---
        @return:\n
            command: List[np.ndarray]\n
            grasp_mask: List[int]\n
        """
        unassembled_index_list = deepcopy(unassembled_index_list)
        with pp.LockRenderer():
            pick_cmd, pick_grasp_mask = next(
                self.pick_gen(
                    element_index,
                    grasp,
                    assembled_index_list,
                    unassembled_index_list + [element_index],
                    attachment_list,
                    self.pick_counter_handle,
                )
            )

        return pick_cmd, pick_grasp_mask

    def TransferPathPlan(
        self,
        element_index: int,
        assembled_index_list: List[int],
        unassembled_index_list: List[int],
        attachment_list: List[Attachment],
        grasp_attach: Attachment,
        start_conf: np.ndarray,
        target_conf: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        @brief: compute transfer path\n
        ---
        @param:\n
            : \n
        ---
        @return:\n
            command: List[np.ndarray]\n
            grasp_mask: List[int]\n
        """
        if element_index != 0:
            diagnosis = True
        else:
            diagnosis = False
        diagnosis = False
        with pp.LockRenderer():
            transfer_cmd, transfer_grasp_mask = next(
                self.transfer_gen(
                    element_index,
                    grasp_attach,
                    start_conf,
                    target_conf,
                    assembled=assembled_index_list,
                    unassembled=unassembled_index_list,
                    attachments=attachment_list,
                    counter=self.transfer_counter_handle,
                    diagnosis=diagnosis,
                )
            )

        return transfer_cmd, transfer_grasp_mask

    def TransitPathPlan(
        self,
        element_index: int,
        assembled_index_list: List[int],
        attachment_list: List[Attachment],
        start_pose2d: np.ndarray,
        target_pose2d: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        @brief: compute transfer path\n
        ---
        @param:\n
            : \n
        ---
        @return:\n
            command: List[np.ndarray]\n
            grasp_mask: List[int]\n
        """
        pp.set_pose(self.element_from_index[element_index].body, self.element_from_index[element_index].goal_pose)
        with pp.LockRenderer():
            transit_cmd, transit_grasp_mask = next(
                self.transit_gen(
                    element_index,
                    start_pose2d,
                    target_pose2d,
                    assembled_index_list,
                    attachment_list,
                )
            )

        return transit_cmd, transit_grasp_mask
