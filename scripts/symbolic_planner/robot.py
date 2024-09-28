from typing import Dict, List, Tuple

import numpy as np
import pybullet_planning as pp
from collision import Element
from pybullet_planning import Attachment, Euler, Point, Pose, get_distance, interpolate_poses, invert, multiply
from robot_setup import RobotSetup
from stream import get_pick_gen_fn, get_place_gen_fn, get_transfer_gen_fn, get_transit_gen_fn
from utils import CounterModule, CounterValue


class Robot(object):
    def __init__(self, index: int, robot_setup: RobotSetup, element_from_index: Dict[int, Element]) -> None:
        self.index = index
        self.robot_setup = robot_setup
        self.counter_handle = CounterModule()
        self.place_counter_handle = self.counter_handle.create_handle("place")
        self.pick_counter_handle = self.counter_handle.create_handle("pick")
        self.transfer_counter_handle = self.counter_handle.create_handle("transfer")

        self.element_from_index = element_from_index

        self.place_gen = get_place_gen_fn(
            self.robot_setup, element_from_index, [], verbose=False, collisions=True, teleops=False, allow_failure=True
        )

        # self.planner_fn = self.DefaultPlan
        self.planner_fn = self.MotionPlan

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
            pp.set_pose(self.element_from_index[assembled_element_index].body, self.element_from_index[assembled_element_index].goal_pose)

        place_cmd, place_grasp_mask, grasp_attach, grasp, pregrasp_pose = self.PlacePathPlan(
            element_index, assembled_index_list, unassembled_index_list, attachment_list
        )
        if place_cmd is None:
            return False
        
        # print(place_cmd)

        return True

    def PlacePathPlan(
        self,
        element_index: int,
        assembled_index_list: List[int],
        unassembled_index_list: List[int],
        attachment_list=List[Attachment],
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
