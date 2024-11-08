import itertools
import os
import sys
import time
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
from motion_planner.back import get_back_gen_fn
from motion_planner.pick import get_pick_gen_fn
from motion_planner.place import get_place_gen_fn
from motion_planner.transfer import get_transfer_gen_fn
from pybullet_planning import Attachment, Euler, Point, Pose, get_distance, interpolate_poses, invert, multiply
from robot.robot_setup import INIT_ARM_JOINT_ANGLES, ONBOARD_LINK, ONBOARD_POSE, RobotSetup
from utils.collision import Element
from utils.utils import CounterModule, TermPrint, timeit_decorator_counter

ConcretePath = namedtuple("ConcretePath", ["base_path", "manipulator_path", "robot_index", "attachment"])


class PathItem(object):
    def __init__(self) -> None:
        self.conf = []
        self.mask = []

    def Append(self, confs: Union[List[np.ndarray], None], masks: Union[List[bool], None]):
        if confs is not None:
            self.conf.extend(confs)
        if masks is not None:
            self.mask.extend(masks)


class PathWithIndex(object):
    def __init__(self) -> None:
        self.storage = {}

    def update_manipulator(
        self,
        element_index: int,
        robot_index: int,
        path: Union[PathItem, None] = None,
        attachment: Union[Attachment, None] = None,
    ):
        if element_index in self.storage.keys():
            last: ConcretePath = self.storage[element_index]

            base_path_new = last.base_path
            manipulator_path_new = last.manipulator_path if path is None else path
            attachment_new = last.attachment if attachment is None else attachment

            self.storage[element_index] = ConcretePath(base_path_new, manipulator_path_new, robot_index, attachment_new)
        else:
            self.storage[element_index] = ConcretePath(None, path, robot_index, attachment)

    def add_base(self, index: int, path: PathItem, robot_index: int):
        if index in self.storage.keys():
            last: ConcretePath = self.storage[index]
            self.storage[index] = ConcretePath(path, last.manipulator_path, robot_index)
        else:
            self.storage[index] = ConcretePath(path, None, robot_index)

    def get(self, index: int) -> ConcretePath:
        try:
            return self.storage[index]
        except:
            return None


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

        self.back_gen = get_back_gen_fn(
            self.robot_setup,
            element_from_index,
            [],
            verbose=BACK_VERBOSE,
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

        self.planner = planner

    @timeit_decorator_counter("others_counter_handle")
    def ManipulatorMotionPlan(
        self,
        element_index: int,
        assembled_index_list: List[int],
        unassembled_index_list: List[int],
        attachment_list: List[Attachment] = [],
        other_obstacles: List[int] = [],
        max_attempts: int = 3,
        verbose: bool = False,
        rich_output: bool = False,
        plan_back: bool = True,
    ) -> Union[
        bool,
        Tuple[
            bool,
            Union[PathItem, None],
            Union[np.ndarray, None],
            Union[Attachment, None],
            Union[Tuple[Tuple[float], Tuple[float]], None],
        ],
    ]:
        """
        Plan manipulator motion to assemble an element.

        Params:
            element_index (int): the index of the element to assemble
            assembled_index_list (List[int]): indices of assembled elements
            unassembled_index_list (List[int]): indices of unassembled elements excluding current element
            attachment_list (List[Attachment], [], [not used]): not used
            other_obstacles ([int], []): other obstacles, e.g. other robots
            max_attempts (int, 3): max attempts to plan
            verbose (bool, False): whether print plan information
            rich_output (bool, False): whether return planned manipulator path, hold conf and grasp attachment
            plan_back (bool, True): whether plan back motion

        Returns:
            bool: True if successful, False otherwise
            PathItem: (optional) planned manipulator path
            np.ndarray: (optional) hold conf of robot
            Attachment: (optional) grasp attachment
            pp.Pose: (optional) gripper_from_body

        """

        for attempt in range(max_attempts):

            if verbose:
                TermPrint.print(f"Manipulator plan attempt: {attempt+1}", "magenta", blank_f=True)

            for assembled_element_index in assembled_index_list:
                pp.set_pose(
                    self.element_from_index[assembled_element_index].body,
                    self.element_from_index[assembled_element_index].goal_pose,
                )

            path_item = PathItem()

            # -------------------- Place --------------------#
            plan_status, place_cmd, place_grasp_mask, grasp_attachment, grasp, pregrasp_pose, hold_conf = (
                self.PlacePathPlan(
                    element_index,
                    assembled_index_list,
                    unassembled_index_list,
                    attachment_list,
                    other_obstacles=other_obstacles,
                    pp_show=PLACE_SHOW,
                )
            )
            if not plan_status:
                continue

            # -------------------- Pick --------------------#
            plan_status, pick_cmd, pick_grasp_mask = self.PickPathPlan(
                element_index,
                assembled_index_list,
                unassembled_index_list,
                attachment_list,
                grasp,
                other_obstacles=other_obstacles,
                pp_show=PICK_SHOW,
            )
            if not plan_status:
                continue

            # -------------------- Transfer --------------------#
            plan_status, transfer_cmd, transfer_grasp_mask = self.TransferPathPlan(
                element_index,
                assembled_index_list,
                unassembled_index_list,
                attachment_list,
                grasp_attachment,
                pick_cmd[-1],
                place_cmd[0],
                other_obstacles=other_obstacles,
                pp_show=TRANSFER_SHOW,
            )
            if not plan_status:
                continue

            if plan_back:
                # -------------------- Back --------------------#
                plan_status, back_cmd, back_grasp_mask = self.BackPathPlan(
                    element_index,
                    hold_conf,
                    grasp,
                    assembled_index_list,
                    unassembled_index_list,
                    attachment_list,
                    other_obstacles=other_obstacles,
                    pp_show=BACK_SHOW,
                )
                if not plan_status:
                    continue

            path_item.Append(pick_cmd, pick_grasp_mask)
            path_item.Append(transfer_cmd, transfer_grasp_mask)
            path_item.Append(place_cmd, place_grasp_mask)
            if plan_back:
                path_item.Append(back_cmd, back_grasp_mask)

            if self.path_storage is not None:
                self.path_storage.update_manipulator(element_index, self.index, path_item, grasp_attachment)

            if rich_output:
                return True, path_item, hold_conf, grasp_attachment, grasp
            else:
                return True

        if rich_output:
            return False, None, None, None, None
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
        verbose: bool = True,
    ) -> Tuple[bool, List[int]]:

        task_all = list(itertools.permutations(task))
        task_all = [list(temp) for temp in task_all]
        mertic_all = [0.0] * len(task_all)
        plan_path_obj_all = [[]] * len(task_all)
        plan_attachment_all = [[]] * len(task_all)
        total_plan_time = 0.0

        total_start = time.time()

        # -------------------- find all solutions --------------------#
        for current_id, current_task in enumerate(task_all):

            TermPrint.print(
                f"========== Cooperation plan on task {current_task} ==========",
                "light_blue",
                blank_f=True,
                blank_b=True,
            )

            current_assembled_index_list = deepcopy(assembled_index_list)
            current_unassembled_index_list = deepcopy(unassembled_index_list)
            current_plan_time = 0.0
            current_plan_status = True
            current_plan_path_obj = []
            current_plan_obstacles = []
            current_plan_hold_confs = []
            current_plan_grasp = []
            current_plan_attachment = []

            # **************************************************************************
            # plan motion without back
            # **************************************************************************

            for element_index, robot in zip(current_task, robots):

                # -------------------- remove current element from unassembled list --------------------#
                if element_index in current_unassembled_index_list:
                    current_unassembled_index_list.remove(element_index)

                plan_status, plan_path_obj, hold_conf, grasp_attachment, grasp = robot.ManipulatorMotionPlan(
                    element_index,  # element_index
                    current_assembled_index_list,  # assembled_index_list
                    current_unassembled_index_list,  # unassembled_index_list
                    attachment_list=[],  # attachment_list
                    other_obstacles=current_plan_obstacles,  # other_obstacles
                    max_attempts=max_attempts,
                    verbose=verbose,
                    rich_output=True,
                    plan_back=False,
                )

                # -------------------- store path object --------------------#
                current_plan_path_obj.append(plan_path_obj)

                # -------------------- if plan succeed, add current element to assembled list --------------------#
                if plan_status and element_index not in current_assembled_index_list:
                    current_assembled_index_list.append(element_index)

                if plan_status:
                    # -------------------- if plan succeed, accumulate total plan time --------------------#
                    timer_handle = robot.others_counter_handle.add_counter_value("total time")
                    current_plan_time += timer_handle.last_update

                    # -------------------- if plan succeed, set robot joint position and add robot to obstacles --------------------#
                    if hold_conf is not None:
                        robot.robot_setup.set_joint_positions(robot.robot_setup.control_joints, hold_conf)
                        grasp_attachment.assign()
                        current_plan_obstacles.append(robot.robot_setup.robot)
                        current_plan_hold_confs.append(hold_conf)

                    # -------------------- if plan succeed, add grasp to list --------------------#
                    if grasp is not None:
                        current_plan_grasp.append(grasp)

                    # -------------------- if plan succeed, add attachment to list --------------------#
                    if grasp_attachment is not None:
                        current_plan_attachment.append(grasp_attachment)

                else:
                    # -------------------- if plan failed, set current_plan_status to False and break --------------------#
                    current_plan_status = False
                    break

            # **************************************************************************
            # plan back motion in the reverse sequence
            # **************************************************************************

            if current_plan_status:
                for element_index, robot, hold_conf, grasp in zip(
                    current_task[::-1], robots[::-1], current_plan_hold_confs[::-1], current_plan_grasp[::-1]
                ):

                    current_assembled_index_list = deepcopy(assembled_index_list) + task
                    current_unassembled_index_list = deepcopy(unassembled_index_list)

                    # -------------------- remove current element from unassembled list --------------------#
                    if element_index in current_unassembled_index_list:
                        current_unassembled_index_list.remove(element_index)

                    fail_flag = True
                    for attempt in range(max_attempts):

                        if verbose:
                            TermPrint.print(f"Manipulator plan attempt: {attempt+1}", "magenta", blank_f=True)

                        start_time = time.time()

                        plan_status, back_cmd, back_grasp_mask = robot.BackPathPlan(
                            element_index,
                            hold_conf,
                            grasp,
                            current_assembled_index_list,
                            current_unassembled_index_list,
                            [],  # attachment_list
                            other_obstacles=current_plan_obstacles,
                            pp_show=BACK_SHOW,
                        )

                        end_time = time.time()

                        if plan_status:
                            # -------------------- if plan succeed, accumulate total plan time --------------------#
                            timer_handle = robot.others_counter_handle.add_counter_value("total time")
                            timer_handle.increment(end_time - start_time)
                            current_plan_time += end_time - start_time

                            fail_flag = False
                            if robot.path_storage is not None:
                                path = robot.path_storage.get(element_index)
                                manipulator_path: PathItem = path.manipulator_path
                                manipulator_path.Append(back_cmd, back_grasp_mask)
                                robot.path_storage.update_manipulator(element_index, robot.index, manipulator_path)
                            robot.robot_setup.set_joint_positions(robot.robot_setup.arm_joints, INIT_ARM_JOINT_ANGLES)
                            break

                    if fail_flag:
                        # -------------------- if plan failed, set current_plan_status to False and break --------------------#
                        current_plan_status = False
                        break

            mertic_all[current_id] = current_plan_time if current_plan_status else np.inf
            plan_path_obj_all[current_id] = current_plan_path_obj
            plan_attachment_all[current_id] = current_plan_attachment

        # -------------------- return a best solution --------------------#
        best_index = mertic_all.index(min(mertic_all))
        best_task = task_all[best_index]
        best_metric = mertic_all[best_index]
        best_path_obj_list = plan_path_obj_all[best_index]
        best_attachment_list = plan_attachment_all[best_index]

        total_plan_time = time.time() - total_start
        TermPrint.print(
            f"========== Cooperation plan on task {best_task} finished in {total_plan_time}s! ==========",
            "light_blue",
            blank_f=True,
            blank_b=True,
        )

        # -------------------- if best metric equal to inf, motion plan failed --------------------#
        if best_metric == np.inf:
            return False, task

        # -------------------- store planned path to global path_storage --------------------#
        for robot, path_obj, index, attachment in zip(robots, best_path_obj_list, best_task, best_attachment_list):
            robot.path_storage.update_manipulator(index, robot.index, path_obj, attachment)

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
        other_obstacles: List[int] = [],
        pp_show: bool = False,
    ) -> Tuple[
        bool,
        List[np.ndarray],
        List[bool],
        Attachment,
        Tuple[Tuple[float], Tuple[float]],
        Tuple[Tuple[float], Tuple[float]],
        np.ndarray,
    ]:
        """
        Compute place path.

        Params:
            element_index (int): index of plance element
            assembled_index_list ([int]): indices of assembled elements
            unassembled_index_list ([int]): indices of unassembled elements excluding current element
            attachment_list ([Attachment]): list of attachments
            other_obstacles ([int], []): other obstacles, e.g. other robots
            pp_show (bool, False): whether show in pybullet GUI while planning

        Returns:
            plan_status (bool): plan status
            command ([np.ndarray]): list of robot confs
            mask ([bool]): list of attachment masks
            grasp_attachment (Attachment): attachment between robot tool0 and element
            grasp (pp.Pose): gripper_from_body
            pregrasp (pp.Pose): world_from_element, used in transfer motion plan as target conf
            hold_conf (np.ndarray): conf of robot to hold this element at target pose
        """
        if self.planner == "normal":
            if pp_show:
                command, mask, grasp_attachment, grasp, pregrasp = next(
                    self.place_gen(
                        element_index,
                        assembled=assembled_index_list,
                        unassembled=unassembled_index_list,
                        attachments=attachment_list,
                        other_obstacles=other_obstacles,
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
                            other_obstacles=other_obstacles,
                            counter=self.place_counter_handle,
                            diagnosis=PLACE_DIAGNOSIS,
                        )
                    )
            if command is not None:
                plan_status = True
                last_true_index = len(mask) - mask[::-1].index(True) - 1
                hold_conf = command[last_true_index]
            else:
                plan_status = False
                hold_conf = None
        else:
            plan_status = True
            command = None
            mask = None
            grasp_attachment = None
            grasp = None
            pregrasp = None
            hold_conf = None

        return plan_status, command, mask, grasp_attachment, grasp, pregrasp, hold_conf

    def PickPathPlan(
        self,
        element_index: int,
        assembled_index_list: List[int],
        unassembled_index_list: List[int],
        attachment_list: List[Attachment],
        grasp: Tuple[Tuple[float], Tuple[float]],
        other_obstacles: List[int] = [],
        pp_show: bool = False,
    ) -> Tuple[bool, List[np.ndarray], List[bool]]:
        """
        Compute pick path.

        Params:
            element_index (int): index of plance element
            assembled_index_list ([int]): indices of assembled elements
            unassembled_index_list ([int]): indices of unassembled elements excluding current element
            attachment_list ([Attachment]): list of attachments
            grasp (pp.Pose): gripper_from_body
            other_obstacles ([int], []): other obstacles, e.g. other robots
            pp_show (bool, False): whether show in pybullet GUI while planning

        Returns:
            plan_status (bool): plan status
            command ([np.ndarray]): list of robot confs
            mask ([bool]): list of attachment masks
        """
        if self.planner == "normal":
            if pp_show:
                pick_cmd, pick_grasp_mask = next(
                    self.pick_gen(
                        element_index,
                        grasp,
                        assembled=assembled_index_list,
                        unassembled=unassembled_index_list,
                        attachments=attachment_list,
                        other_obstacles=other_obstacles,
                        counter=self.pick_counter_handle,
                        diagnosis=PICK_DIAGNOSIS,
                    )
                )
            else:
                with pp.LockRenderer():
                    pick_cmd, pick_grasp_mask = next(
                        self.pick_gen(
                            element_index,
                            grasp,
                            assembled=assembled_index_list,
                            unassembled=unassembled_index_list,
                            attachments=attachment_list,
                            other_obstacles=other_obstacles,
                            counter=self.pick_counter_handle,
                            diagnosis=PICK_DIAGNOSIS,
                        )
                    )
            if pick_cmd is not None:
                plan_status = True
            else:
                plan_status = False
        else:
            plan_status = True
            pick_cmd = None
            pick_grasp_mask = None

        return plan_status, pick_cmd, pick_grasp_mask

    def TransferPathPlan(
        self,
        element_index: int,
        assembled_index_list: List[int],
        unassembled_index_list: List[int],
        attachment_list: List[Attachment],
        grasp_attachment: Attachment,
        start_conf: np.ndarray,
        target_conf: np.ndarray,
        other_obstacles: List[int] = [],
        pp_show: bool = False,
    ) -> Tuple[bool, List[np.ndarray], List[int]]:
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
            other_obstacles ([int], []): other obstacles, e.g. other robots
            pp_show (bool, False): whether show in pybullet GUI while planning

        Returns:
            plan_status (bool): plan status
            command ([np.ndarray]): list of robot confs
            mask ([bool]): list of attachment masks
        """
        if self.planner == "normal":
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
                        other_obstacles=other_obstacles,
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
                            other_obstacles=other_obstacles,
                            counter=self.transfer_counter_handle,
                            diagnosis=TRANSFER_DIAGNOSIS,
                        )
                    )
            if transfer_cmd is not None:
                plan_status = True
            else:
                plan_status = False
        else:
            plan_status = True
            transfer_cmd = None
            transfer_grasp_mask = None

        return plan_status, transfer_cmd, transfer_grasp_mask

    def BackPathPlan(
        self,
        element_index: int,
        start_conf: np.ndarray,
        grasp: Tuple[Tuple[float], Tuple[float]],
        assembled_index_list: List[int],
        unassembled_index_list: List[int],
        attachment_list: List[Attachment],
        other_obstacles: List[int] = [],
        pp_show: bool = False,
    ) -> Tuple[bool, List[np.ndarray], List[bool]]:
        """
        Compute back path.

        Params:
            element_index (int): index of plance element
            start_conf (np.ndarray): start conf of robot
            grasp (pp.Pose): gripper_from_body
            assembled_index_list ([int]): indices of assembled elements
            unassembled_index_list ([int]): indices of unassembled elements excluding current element
            attachment_list ([Attachment]): list of attachments
            other_obstacles ([int], []): other obstacles, e.g. other robots
            pp_show (bool, False): whether show in pybullet GUI while planning

        Returns:
            plan_status (bool): plan status
            command ([np.ndarray]): list of robot confs
            mask ([bool]): list of attachment masks
        """
        if self.planner == "normal":
            if pp_show:
                command, mask = next(
                    self.back_gen(
                        element_index,
                        start_conf,
                        grasp,
                        assembled=assembled_index_list,
                        unassembled=unassembled_index_list,
                        attachments=attachment_list,
                        other_obstacles=other_obstacles,
                        counter=self.place_counter_handle,
                        diagnosis=PLACE_DIAGNOSIS,
                    )
                )
            else:
                with pp.LockRenderer():
                    command, mask = next(
                        self.back_gen(
                            element_index,
                            start_conf,
                            grasp,
                            assembled=assembled_index_list,
                            unassembled=unassembled_index_list,
                            attachments=attachment_list,
                            other_obstacles=other_obstacles,
                            counter=self.place_counter_handle,
                            diagnosis=PLACE_DIAGNOSIS,
                        )
                    )
            if command is not None:
                plan_status = True
            else:
                plan_status = False
        else:
            plan_status = True
            command = None
            mask = None

        return plan_status, command, mask

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
