import itertools
import os
import sys
import types
from collections import namedtuple
from copy import deepcopy
from functools import partial
from typing import Dict, List, Tuple, Union

from termcolor import cprint
from utils.params import *

sys.path.append(PROJECT_DIR)

import numpy as np
import pybullet_planning as pp
from motion_planner.pick import get_pick_gen_fn
from motion_planner.place import get_place_gen_fn
from motion_planner.transfer import get_transfer_gen_fn
from pybullet_planning import Attachment, Euler, Point, Pose, get_distance, interpolate_poses, invert, multiply
from robot.robot_setup import ONBOARD_LINK, ONBOARD_POSE, RobotSetup
from utils.collision import Element
from utils.utils import CounterModule, timeit_decorator_counter

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
        counter: CounterModule,
        attachments: List[Attachment] = [],
        storage: Union[PathWithIndex, None] = None,
        planner: str = MANIPULATOR_PLANNER,
    ) -> None:
        self.index = index
        self.robot_setup = robot_setup
        self.counter_handle = counter
        self.place_counter_handle = self.counter_handle.create_handle("place")
        self.pick_counter_handle = self.counter_handle.create_handle("pick")
        self.transfer_counter_handle = self.counter_handle.create_handle("transfer")
        self.others_counter_handle = self.counter_handle.create_handle("others")

        self.element_from_index = element_from_index
        self.attachments = attachments
        self.path_storage = storage
        self.last_pose2d = np.array([5, 5, 0])

        self.place_gen = get_place_gen_fn(
            self.robot_setup,
            element_from_index,
            [],
            verbose=PLACE_VERBOSE,
            collisions=True,
            teleops=False,
            allow_failure=True,
        )

        self.pick_gen = get_pick_gen_fn(
            self.robot_setup,
            element_from_index,
            [],
            verbose=PICK_VERBOSE,
            collisions=True,
            teleops=False,
            allow_failure=True,
        )

        self.transfer_gen = get_transfer_gen_fn(
            self.robot_setup,
            element_from_index,
            [],
            verbose=TRANSFER_VERBOSE,
            collisions=True,
            teleops=False,
            allow_failure=True,
        )

        # self.transit_gen = get_transit_gen_fn(
        #     self.robot_setup,
        #     element_from_index,
        #     [],
        #     verbose=True,
        #     collisions=True,
        #     teleops=False,
        #     allow_failure=True,
        # )

        if planner == "normal":
            self.manipulator_planner_fn = partial(self.ManipulatorMotionPlan, verbose=MNIPULATOR_PLAN_SHOW)
        else:
            self.manipulator_planner_fn = self.DefaultPlan

    # def SaveLog(self, path: str, suffix=""):
    #     if self.counter_handle is not None:
    #         self.counter_handle.save(path, f"r{self.index}{suffix}.json")

    @timeit_decorator_counter("others_counter_handle")
    def DefaultPlan(
        self,
        element_index: int,
        assembled_index_list: List[int],
        unassembled_index_list: List[int],
        attachment_list: List[Attachment] = [],
        max_attempts: int = 2,
        verbose: bool = False,
        output_path: bool = False,
    ) -> Union[bool, Tuple[bool, Union[PathItem, None]]]:
        if output_path:
            return True, None
        else:
            return True

    @timeit_decorator_counter("others_counter_handle")
    def ManipulatorMotionPlan(
        self,
        element_index: int,
        assembled_index_list: List[int],
        unassembled_index_list: List[int],
        attachment_list: List[Attachment] = [],
        max_attempts: int = 2,
        verbose: bool = False,
        output_path: bool = False,
    ) -> Union[bool, Tuple[bool, Union[PathItem, None]]]:
        """
        Plan manipulator motion to assemble an element.

        Params:
            element_index (int): the index of the element to assemble
            assembled_index_list (List[int]): indices of assembled elements
            unassembled_index_list (List[int]): indices of unassembled elements excluding current element
            attachment_list (List[Attachment], [], [not used]): not used
            max_attempts (int, 2): max attempts to plan
            verbose (bool, False): whether print plan information
            output_path (bool, False): whether return planned manipulator path

        Returns:
            bool: True if successful, False otherwise
            PathItem: (optional) planned manipulator path
        """

        for attempt in range(max_attempts):

            if verbose:
                cprint(f"Manipulator plan attempt: {attempt+1}", "magenta")

            for assembled_element_index in assembled_index_list:
                pp.set_pose(
                    self.element_from_index[assembled_element_index].body,
                    self.element_from_index[assembled_element_index].goal_pose,
                )

            # -------------------- Place --------------------#
            place_cmd, place_grasp_mask, grasp_attachment, grasp, pregrasp_pose = self.PlacePathPlan(
                element_index, assembled_index_list, unassembled_index_list, attachment_list, pp_show=PLACE_SHOW
            )
            if place_cmd is None:
                continue
            path_obj = PathItem(element_index, grasp_attachment)

            # -------------------- Pick --------------------#
            # self.UpdateElementsRobot(element_index)
            pick_cmd, pick_grasp_mask = self.PickPathPlan(
                element_index, assembled_index_list, unassembled_index_list, attachment_list, grasp, pp_show=PICK_SHOW
            )
            if pick_cmd is None:
                continue

            # -------------------- Transfer --------------------#
            transfer_cmd, transfer_grasp_mask = self.TransferPathPlan(
                element_index,
                assembled_index_list,
                unassembled_index_list,
                attachment_list,
                grasp_attachment,
                pick_cmd[-1],
                place_cmd[0],
                pp_show=TRANSFER_SHOW,
            )
            if transfer_cmd is None:
                continue

            path_obj.Append(pick_cmd, pick_grasp_mask)
            path_obj.Append(transfer_cmd, transfer_grasp_mask)
            path_obj.Append(place_cmd, place_grasp_mask)

            if self.path_storage is not None:
                self.path_storage.add_manipulator(element_index, path_obj)

            if output_path:
                return True, path_obj
            else:
                return True

        if output_path:
            return False, None
        else:
            return False

    @staticmethod
    def ManipulatorGroupMotionPlan(
        robots: List["Robot"],
        task: List[int],
        assembled_index_list: List[int],
        unassembled_index_list: List[int],
        attachment_list: List[Attachment] = [],
        max_attempts: int = 2,
        verbose: bool = False,
    ) -> Tuple[bool, List[int]]:

        task_all = list(itertools.permutations(task))
        task_all = [list(temp) for temp in task_all]
        mertic_all = [0.0] * len(task_all)
        plan_path_obj_all = [[]] * len(task_all)

        # -------------------- find all solutions --------------------#
        for current_id, current_task in enumerate(task_all):

            cprint(f"\n========== Cooperation plan on task {current_task} ==========\n", "light_magenta")

            current_assembled_index_list = deepcopy(assembled_index_list)
            current_unassembled_index_list = deepcopy(unassembled_index_list)
            current_plan_time = 0.0
            current_plan_status = True
            current_plan_path_obj = []

            for element_index, robot in zip(current_task, robots):
                if MANIPULATOR_PLANNER == "normal":
                    plan_func = robot.ManipulatorMotionPlan
                else:
                    plan_func = robot.DefaultPlan

                # remove current element from unassembled list
                if element_index in current_unassembled_index_list:
                    current_unassembled_index_list.remove(element_index)

                plan_status, plan_path_obj = plan_func(
                    element_index,
                    current_assembled_index_list,
                    current_unassembled_index_list,
                    [],
                    verbose=True,
                    output_path=True,
                )

                # store path object
                current_plan_path_obj.append(plan_path_obj)

                # if plan succeed, add current element to assembled list
                if plan_status and element_index not in current_assembled_index_list:
                    current_assembled_index_list.append(element_index)

                if plan_status:
                    # if plan succeed, accumulate total plan time
                    timer_handle = robot.others_counter_handle.add_counter_value("total time")
                    current_plan_time += timer_handle.last_update
                else:
                    # if plan failed, set current_plan_status to False and break
                    current_plan_status = False
                    break

            mertic_all[current_id] = current_plan_time if current_plan_status else np.inf
            plan_path_obj_all[current_id] = current_plan_path_obj

        print("\n")

        # -------------------- return a best solution --------------------#
        best_index = mertic_all.index(min(mertic_all))
        best_task = task_all[best_index]
        best_metric = mertic_all[best_index]
        best_path_obj_list = plan_path_obj_all[best_index]

        # if best metric equal to inf, motion plan failed
        if best_metric == np.inf:
            return False, task

        # store planned path to global path_storage
        for robot, path_obj, index in zip(robots, best_path_obj_list, best_task):
            robot.path_storage.add_manipulator(index, path_obj)

        return True, best_task

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
                self.robot_setup.robot, pp.link_from_name(self.robot_setup.robot, ONBOARD_LINK)
            )
            delta_pose = Pose(
                point=ONBOARD_POSE[:3], euler=Euler(roll=ONBOARD_POSE[3], pitch=ONBOARD_POSE[4], yaw=ONBOARD_POSE[5])
            )
            bar_pose = multiply(ipad_link_pose, delta_pose)
            pp.set_pose(self.element_from_index[element_index].body, bar_pose)

    def PlacePathPlan(
        self,
        element_index: int,
        assembled_index_list: List[int],
        unassembled_index_list: List[int],
        attachment_list: List[Attachment],
        pp_show: bool = False,
    ) -> Tuple[
        List[np.ndarray], List[bool], Attachment, Tuple[Tuple[float], Tuple[float]], Tuple[Tuple[float], Tuple[float]]
    ]:
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
                    diagnosis=PLACE_DIAGNOSIS,
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
                        diagnosis=PLACE_DIAGNOSIS,
                    )
                )
        return command, mask, grasp_attachment, grasp, pregrasp

    def PickPathPlan(
        self,
        element_index: int,
        assembled_index_list: List[int],
        unassembled_index_list: List[int],
        attachment_list: List[Attachment],
        grasp: Tuple[Tuple[float], Tuple[float]],
        pp_show: bool = False,
    ) -> Tuple[List[np.ndarray], List[bool]]:
        """
        Compute pick path.

        Params:
            element_index (int): index of plance element
            assembled_index_list ([int]): indices of assembled elements
            unassembled_index_list ([int]): indices of unassembled elements excluding current element
            attachment_list ([Attachment]): list of attachments
            grasp (pp.Pose): gripper_from_body
            pp_show (bool, False): whether show in pybullet GUI while planning

        Returns:
            command ([np.ndarray]): list of robot confs
            mask ([bool]): list of attachment masks
        """
        if pp_show:
            pick_cmd, pick_grasp_mask = next(
                self.pick_gen(
                    element_index,
                    grasp,
                    assembled_index_list,
                    unassembled_index_list,
                    attachment_list,
                    self.pick_counter_handle,
                    diagnosis=PICK_DIAGNOSIS,
                )
            )
        else:
            with pp.LockRenderer():
                pick_cmd, pick_grasp_mask = next(
                    self.pick_gen(
                        element_index,
                        grasp,
                        assembled_index_list,
                        unassembled_index_list,
                        attachment_list,
                        self.pick_counter_handle,
                        diagnosis=PICK_DIAGNOSIS,
                    )
                )

        return pick_cmd, pick_grasp_mask

    def TransferPathPlan(
        self,
        element_index: int,
        assembled_index_list: List[int],
        unassembled_index_list: List[int],
        attachment_list: List[Attachment],
        grasp_attachment: Attachment,
        start_conf: np.ndarray,
        target_conf: np.ndarray,
        pp_show: bool = False,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Compute transfer path.

        Params:
            element_index (int): index of plance element
            assembled_index_list ([int]): indices of assembled elements
            unassembled_index_list ([int]): indices of unassembled elements excluding current element
            attachment_list ([Attachment]): list of attachments
            grasp_attachment (Attachment): grasp attachment of current element
            start_conf (np.ndarray): start conf of robot
            target_conf (np.ndarray): target conf of robot
            pp_show (bool, False): whether show in pybullet GUI while planning

        Returns:
            command ([np.ndarray]): list of robot confs
            mask ([bool]): list of attachment masks
        """
        if pp_show:
            transfer_cmd, transfer_grasp_mask = next(
                self.transfer_gen(
                    element_index,
                    start_conf,
                    target_conf,
                    grasp_attachment,
                    assembled=assembled_index_list,
                    unassembled=unassembled_index_list,
                    attachments=attachment_list,
                    counter=self.transfer_counter_handle,
                    diagnosis=TRANSFER_DIAGNOSIS,
                )
            )
        else:
            with pp.LockRenderer():
                transfer_cmd, transfer_grasp_mask = next(
                    self.transfer_gen(
                        element_index,
                        start_conf,
                        target_conf,
                        grasp_attachment,
                        assembled=assembled_index_list,
                        unassembled=unassembled_index_list,
                        attachments=attachment_list,
                        counter=self.transfer_counter_handle,
                        diagnosis=TRANSFER_DIAGNOSIS,
                    )
                )

        return transfer_cmd, transfer_grasp_mask

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
