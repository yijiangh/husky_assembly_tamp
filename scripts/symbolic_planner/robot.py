from collections import namedtuple
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import pybullet_planning as pp
from collision import Element
from motion_planner.place import get_place_gen_fn
from pybullet_planning import Attachment, Euler, Point, Pose, get_distance, interpolate_poses, invert, multiply
from robot_setup import RobotSetup
from stream import get_pick_gen_fn, get_transfer_gen_fn, get_transit_gen_fn
from utils import CounterModule, CounterValue

ConcretePath = namedtuple("ConcretePath", ["base_path", "manipulator_path"])


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


class PathWithIndex(object):
    def __init__(self) -> None:
        self.storage = {}

    def add_manipulator(self, index: int, path: PathItem):
        if index in self.storage.keys():
            last: ConcretePath = self.storage[index]
            self.storage[index] = ConcretePath(last.base_path, path)
        else:
            self.storage[index] = ConcretePath(None, path)

    def add_base(self, index: int, path: PathItem):
        if index in self.storage.keys():
            last: ConcretePath = self.storage[index]
            self.storage[index] = ConcretePath(path, last.manipulator_path)
        else:
            self.storage[index] = ConcretePath(path, None)

    def get(self, index: int) -> ConcretePath:
        try:
            return self.storage[index]
        except:
            return None, None


class Robot(object):
    def __init__(
        self,
        index: int,
        robot_setup: RobotSetup,
        element_from_index: Dict[int, Element],
        attachments: List[Attachment] = [],
        storage: PathWithIndex = None,
    ) -> None:
        self.index = index
        self.robot_setup = robot_setup
        self.counter_handle = CounterModule()
        self.place_counter_handle = self.counter_handle.create_handle("place")
        self.pick_counter_handle = self.counter_handle.create_handle("pick")
        self.transfer_counter_handle = self.counter_handle.create_handle("transfer")

        self.element_from_index = element_from_index
        self.attachments = attachments
        self.path_storage = storage
        self.last_pose2d = np.array([5, 5, 0])

        self.place_gen = get_place_gen_fn(
            self.robot_setup,
            element_from_index,
            [],
            verbose=False,
            collisions=True,
            teleops=False,
            allow_failure=True,
        )

        # self.pick_gen = get_pick_gen_fn(
        #     self.robot_setup, element_from_index, [], verbose=False, collisions=True, teleops=False, allow_failure=True
        # )

        # self.transfer_gen = get_transfer_gen_fn(
        #     self.robot_setup,
        #     element_from_index,
        #     [],
        #     verbose=True,
        #     collisions=True,
        #     teleops=False,
        #     allow_failure=True,
        #     max_attempts=15,
        # )

        # self.transit_gen = get_transit_gen_fn(
        #     self.robot_setup,
        #     element_from_index,
        #     [],
        #     verbose=True,
        #     collisions=True,
        #     teleops=False,
        #     allow_failure=True,
        # )

        # self.manipulator_planner_fn = self.DefaultPlan
        self.manipulator_planner_fn = self.ManipulatorMotionPlan

    def SaveLog(self, path: str, suffix=""):
        self.counter_handle.save(path, f"r{self.index}{suffix}.log")

    def DefaultPlan(
        self,
        element_index: int,
        assembled_index_list: List[int],
        unassembled_index_list: List[int],
        attachment_list=List[Attachment],
    ) -> bool:
        return True

    def ManipulatorMotionPlan(
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
            element_index, assembled_index_list, unassembled_index_list, attachment_list, pp_show=False
        )
        if place_cmd is None:
            return False
        path_obj = PathItem(element_index, grasp_attach)

        # -------------------- Pick --------------------#
        # self.UpdateElementsRobot(element_index)
        # pick_cmd, pick_grasp_mask = self.PickPathPlan(
        #     element_index, assembled_index_list, unassembled_index_list, attachment_list, grasp
        # )
        # if pick_cmd is None:
        #     return False

        # -------------------- Transfer --------------------#
        # transfer_cmd, transfer_grasp_mask = self.TransferPathPlan(
        #     element_index,
        #     assembled_index_list,
        #     unassembled_index_list,
        #     attachment_list,
        #     grasp_attach,
        #     pick_cmd[-1],
        #     place_cmd[0],
        # )
        # if transfer_cmd is None:
        #     return False

        # path_obj.Append(pick_cmd, pick_grasp_mask)
        # path_obj.Append(transfer_cmd, transfer_grasp_mask)
        path_obj.Append(place_cmd, place_grasp_mask)

        if self.path_storage is not None:
            self.path_storage.add_manipulator(element_index, path_obj)
        return True

    # def BaseMotionPlan(self, path: List[List[int]]):
    #     assembled = []
    #     for index_list in path:
    #         for i in self.element_from_index.keys():
    #             if i in assembled:
    #                 pp.set_pose(self.element_from_index[i].body, self.element_from_index[i].goal_pose)
    #             else:
    #                 pp.set_pose(self.element_from_index[i].body, pp.Pose(point=(5, 0, 0), euler=pp.Euler(0, 1.5708, 0)))

    #         index = index_list[0]
    #         base_path_obj = PathItem(index, None)
    #         # -------------------- Transit --------------------#
    #         # TODO attachment list
    #         transit_cmd, transit_grasp_mask = self.TransitPathPlan(
    #             index, assembled, [], self.last_pose2d, self.path_storage.get(index).manipulator_path.conf[-1][:3]
    #         )
    #         self.last_pose2d = self.path_storage.get(index).manipulator_path.conf[-1][:3]
    #         base_path_obj.Append(transit_cmd, transit_grasp_mask)
    #         self.path_storage.add_base(index, base_path_obj)
    #         assembled.append(index)

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
        pp_show: bool = False,
    ) -> Tuple[List[np.ndarray], List[int], Attachment, Tuple, Tuple]:
        """
        Compute place path.

        Params:
            element_index (int): index of plance element
            assembled_index_list ([int]): indices of assembled elements
            unassembled_index_list ([int]): indices of unassembled elements excluding current element
            attachment_list ([Attachment]): list of attachments
            pp_show (bool, False): whether show in pybullet GUI while planning

        Returns:
            command ([np.ndarray]): list of robot confs
            mask ([bool]): list of attachment masks
            grasp_attachment (Attachment): attachment between robot tool0 and element
            grasp (pp.Pose): gripper_from_body
            pregrasp (pp.Pose): world_from_element, used in transfer motion plan as target conf
        """
        if pp_show:
            command, mask, grasp_attachment, grasp, pregrasp = next(
                self.place_gen(
                    element_index,
                    assembled=assembled_index_list,
                    unassembled=unassembled_index_list,
                    attachments=attachment_list,
                    counter=self.place_counter_handle,
                )
            )
        else:
            with pp.LockRenderer():
                command, mask, grasp_attachment, grasp, pregrasp = next(
                    self.place_gen(
                        element_index,
                        assembled=assembled_index_list,
                        unassembled=unassembled_index_list,
                        attachments=attachment_list,
                        counter=self.place_counter_handle,
                    )
                )
        return command, mask, grasp_attachment, grasp, pregrasp

    # def PickPathPlan(
    #     self,
    #     element_index: int,
    #     assembled_index_list: List[int],
    #     unassembled_index_list: List[int],
    #     attachment_list: List[Attachment],
    #     grasp: Tuple[Tuple, Tuple],
    # ) -> Tuple[List[np.ndarray], List[int]]:
    #     """
    #     @brief: compute pick path\n
    #     ---
    #     @param:\n
    #         : \n
    #     ---
    #     @return:\n
    #         command: List[np.ndarray]\n
    #         grasp_mask: List[int]\n
    #     """
    #     unassembled_index_list = deepcopy(unassembled_index_list)
    #     with pp.LockRenderer():
    #         pick_cmd, pick_grasp_mask = next(
    #             self.pick_gen(
    #                 element_index,
    #                 grasp,
    #                 assembled_index_list,
    #                 unassembled_index_list + [element_index],
    #                 attachment_list,
    #                 self.pick_counter_handle,
    #             )
    #         )

    #     return pick_cmd, pick_grasp_mask

    # def TransferPathPlan(
    #     self,
    #     element_index: int,
    #     assembled_index_list: List[int],
    #     unassembled_index_list: List[int],
    #     attachment_list: List[Attachment],
    #     grasp_attach: Attachment,
    #     start_conf: np.ndarray,
    #     target_conf: np.ndarray,
    # ) -> Tuple[List[np.ndarray], List[int]]:
    #     """
    #     @brief: compute transfer path\n
    #     ---
    #     @param:\n
    #         : \n
    #     ---
    #     @return:\n
    #         command: List[np.ndarray]\n
    #         grasp_mask: List[int]\n
    #     """
    #     if element_index != 0:
    #         diagnosis = True
    #     else:
    #         diagnosis = False
    #     diagnosis = False
    #     with pp.LockRenderer():
    #         transfer_cmd, transfer_grasp_mask = next(
    #             self.transfer_gen(
    #                 element_index,
    #                 grasp_attach,
    #                 start_conf,
    #                 target_conf,
    #                 assembled=assembled_index_list,
    #                 unassembled=unassembled_index_list,
    #                 attachments=attachment_list,
    #                 counter=self.transfer_counter_handle,
    #                 diagnosis=diagnosis,
    #             )
    #         )

    #     return transfer_cmd, transfer_grasp_mask

    # def TransitPathPlan(
    #     self,
    #     element_index: int,
    #     assembled_index_list: List[int],
    #     attachment_list: List[Attachment],
    #     start_pose2d: np.ndarray,
    #     target_pose2d: np.ndarray,
    # ) -> Tuple[List[np.ndarray], List[int]]:
    #     """
    #     @brief: compute transfer path\n
    #     ---
    #     @param:\n
    #         : \n
    #     ---
    #     @return:\n
    #         command: List[np.ndarray]\n
    #         grasp_mask: List[int]\n
    #     """
    #     pp.set_pose(self.element_from_index[element_index].body, self.element_from_index[element_index].goal_pose)
    #     with pp.LockRenderer():
    #         transit_cmd, transit_grasp_mask = next(
    #             self.transit_gen(
    #                 element_index,
    #                 start_pose2d,
    #                 target_pose2d,
    #                 assembled_index_list,
    #                 attachment_list,
    #             )
    #         )

    #     return transit_cmd, transit_grasp_mask
