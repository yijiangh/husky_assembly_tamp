import operator
from collections import deque
from copy import deepcopy
from typing import Tuple

import numpy as np
from collision import Element
from element_object import ElementObject, ElementStatus
from heuristic import BasicHeuristic, CenterDistanceHeuristic, GroundedChainHeuristic, GroundedHeightHeuristic
from robot import Robot
from utils import flatten

# TODO  * trace back for Search
#       * trace back for SearchRobotCooperation
#       * plan transit/transfer/pick motion
#       * store planned path

class Planner(object):
    def __init__(self, robot_num: int, robots: list[Robot]) -> None:
        self.robot_num = robot_num
        self.robots = robots

    def Plan(
        self, element_from_index: dict[Element], contact_id_pairs: list[list], grounded_elements_index: list
    ) -> list:
        # -------------------- Generate element objects --------------------#
        element_object_list = Planner.GetElementObjects(element_from_index, contact_id_pairs, grounded_elements_index)

        # GroundedChainHeuristic.Update(element_object_list)
        GroundedHeightHeuristic.Update(element_object_list)
        # CenterDistanceHeuristic.Update(element_object_list)

        path_index = self.Search(element_object_list)
        # path_index = self.BackwardSearchWithoutMotionPlan(element_object_list)
        return path_index

    def Search(self, element_object_list: list[ElementObject]) -> list:
        # -------------------- init --------------------#
        not_visited_index = deque([obj.index for obj in element_object_list])
        visited_index = deque([])
        path_index = deque([])
        blacklist_index = deque([])

        # -------------------- loop --------------------#
        while len(not_visited_index) != 0:
            # -------------------- pop --------------------#
            element_object_index, _ = Planner.FindMin(list(not_visited_index), element_object_list)
            not_visited_index.remove(element_object_index)

            # -------------------- assemble --------------------#
            visited_index_list = list(visited_index)
            Planner.Assemble(element_object_index, visited_index_list, element_object_list)
            status = element_object_list[element_object_index].status

            # -------------------- decide what to do --------------------#
            if status == ElementStatus.fixed:
                plan_status = self.robots[0].planner_fn(
                    element_object_index, list(visited_index), list(not_visited_index), []
                )

                if plan_status:
                    visited_index_list = list(visited_index)
                    Planner.MultiDisassemble(list(blacklist_index), visited_index_list, element_object_list)
                    not_visited_index.extend(blacklist_index)
                    blacklist_index.clear()
                    visited_index.append(element_object_index)
                    path_index.append([element_object_index])
                else:
                    visited_index_list = list(visited_index)
                    Planner.Disassemble(element_object_index, visited_index_list, element_object_list)
                    blacklist_index.append(element_object_index)

            elif status == ElementStatus.float:
                visited_index_list = list(visited_index)
                Planner.Disassemble(element_object_index, visited_index_list, element_object_list)
                blacklist_index.append(element_object_index)

            elif status == ElementStatus.rotate:
                state, task = self.SearchRobotCooperation(
                    element_object_list, list(visited_index), list(not_visited_index), [element_object_index]
                )
                if state:
                    visited_index_list = list(visited_index)
                    Planner.MultiDisassemble(list(blacklist_index), visited_index_list, element_object_list)
                    not_visited_index.extend(blacklist_index)
                    blacklist_index.clear()
                    for temp_element_index in task:  # remove task from not_visited
                        not_visited_index = deque(
                            [
                                element_temp_index
                                for element_temp_index in not_visited_index
                                if element_temp_index != temp_element_index
                            ]
                        )
                    visited_index.extend(task)
                    path_index.append(task)
                else:
                    visited_index_list = list(visited_index)
                    Planner.MultiDisassemble(task, visited_index_list, element_object_list)
                    Planner.Disassemble(element_object_index, visited_index_list, element_object_list)
                    blacklist_index.append(element_object_index)
                    
            else:
                raise RuntimeError("This status is not possible!")

            #-------------------- Check if backtracking is necessary --------------------#
            if len(not_visited_index) == 0 and len(blacklist_index) != 0:
                print("********** Dead end reached, need to backtrack! **********")
                # not_visited_index.extend(blacklist_index)
                # blacklist_index.clear()
                # # -------------------- pop --------------------#
                # element_object_index_backtrack, _ = Planner.FindMax(list(visited_index), element_object_list)
                # backtrack_index_list = Planner.GetBacktrackElementsFromPath(element_object_index_backtrack, path_index)
                # for index_temp in backtrack_index_list:
                #     visited_index.remove(index_temp)
                # path_index.remove(backtrack_index_list)
                # blacklist_index.extend(backtrack_index_list)

        return list(path_index)

    def SearchRobotCooperation(
        self,
        element_object_list: list[ElementObject],
        visited_index_list: list,
        not_visited_index_list: list,
        hold_index_list: list,
    ) -> Tuple[bool, list[ElementObject]]:
        visited_index_list = deepcopy(visited_index_list)
        not_visited_index_list = deepcopy(not_visited_index_list)
        visited_index = deque(visited_index_list)
        not_visited_index = deque(not_visited_index_list)
        hold_index = deque(hold_index_list)
        blacklist_index = deque()

        if len(hold_index_list) == self.robot_num:
            return False, hold_index_list

        # -------------------- loop --------------------#
        while len(not_visited_index) != 0:
            # -------------------- pop --------------------#
            element_object_index, _ = Planner.FindMin(list(not_visited_index), element_object_list)
            not_visited_index.remove(element_object_index)

            # -------------------- assemble --------------------#
            visited_hold_index_list = list(visited_index) + list(hold_index)
            Planner.Assemble(element_object_index, visited_hold_index_list, element_object_list)
            status = element_object_list[element_object_index].status

            if status == ElementStatus.fixed and not Planner.ElementsStatusCheck(list(hold_index), element_object_list):
                visited_hold_index_list = list(visited_index) + list(hold_index)
                Planner.Disassemble(element_object_index, visited_hold_index_list, element_object_list)
                blacklist_index.append(element_object_index)

            elif status == ElementStatus.fixed and Planner.ElementsStatusCheck(list(hold_index), element_object_list):
                hold_index.append(element_object_index)
                
                plan_status = True
                for i, hold_element_index in enumerate(hold_index):
                    robot = self.robots[i]
                    plan_status = robot.planner_fn(hold_element_index, list(visited_index), list(not_visited_index), [])
                    if plan_status == False:
                        break
                        # return False, hold_index_list
                
                if plan_status:
                    return True, list(hold_index)
                else:
                    visited_hold_index_list = list(visited_index) + list(hold_index)
                    Planner.Disassemble(element_object_index, visited_hold_index_list, element_object_list)
                    Planner.MultiDisassemble(list(hold_index), list(visited_index), element_object_list)
                    Planner.MultiAssemble(list(hold_index), list(visited_index), element_object_list)
                    blacklist_index.append(element_object_index)
                    continue

            elif status == ElementStatus.rotate:
                state, task = self.SearchRobotCooperation(
                    element_object_list,
                    list(visited_index),
                    list(not_visited_index),
                    [element_object_index] + list(hold_index),
                )
                if state:
                    return state, task
                else:
                    visited_hold_index_list = list(visited_index) + list(hold_index)
                    Planner.MultiDisassemble(task, list(visited_index), element_object_list)
                    Planner.Disassemble(element_object_index, visited_hold_index_list, element_object_list)
                    Planner.MultiDisassemble(list(hold_index), list(visited_index), element_object_list)
                    Planner.MultiAssemble(list(hold_index), list(visited_index), element_object_list)
                    blacklist_index.append(element_object_index)

            elif status == ElementStatus.float:
                visited_hold_index_list = list(visited_index) + list(hold_index)
                Planner.Disassemble(element_object_index, visited_hold_index_list, element_object_list)
                Planner.MultiDisassemble(list(hold_index), list(visited_index), element_object_list)
                Planner.MultiAssemble(list(hold_index), list(visited_index), element_object_list)
                blacklist_index.append(element_object_index)

            else:
                raise RuntimeError("This status is not possible!")

        return False, []

    def BackwardSearchWithoutMotionPlan(self, element_object_list: list[ElementObject]) -> list:
        # -------------------- init --------------------#
        not_visited_index = deque([obj.index for obj in element_object_list])
        visited_index = deque([])
        path_index = deque([])
        blacklist_index = deque([])

        # -------------------- preprocess --------------------#
        Planner.MultiAssemble(list(not_visited_index), list(visited_index), element_object_list)

        # -------------------- loop --------------------#
        while len(not_visited_index) != 0:
            # -------------------- pop --------------------#
            element_object_index, _ = Planner.FindMax(list(not_visited_index), element_object_list)
            not_visited_index.remove(element_object_index)

            # -------------------- disassemble --------------------#
            assembled_index_list = list(not_visited_index) + list(blacklist_index)
            Planner.Disassemble(element_object_index, assembled_index_list, element_object_list)
            rotate_element_cnt = Planner.ElementsStatusCount(
                assembled_index_list, element_object_list, ElementStatus.rotate
            )

            if rotate_element_cnt == 0:
                not_visited_index.extend(blacklist_index)
                blacklist_index.clear()
                visited_index.append(element_object_index)
                path_index.append([element_object_index])

            elif rotate_element_cnt <= self.robot_num - 1:
                not_visited_index.extend(blacklist_index)
                blacklist_index.clear()

                multi_disassemble_index_list = Planner.GetElementIndexBYStatus(
                    list(not_visited_index), element_object_list, ElementStatus.rotate
                )
                for index in multi_disassemble_index_list:
                    if index in not_visited_index:
                        not_visited_index.remove(index)

                visited_index.extend([element_object_index] + multi_disassemble_index_list)
                path_index.append([element_object_index] + multi_disassemble_index_list)

                Planner.MultiDisassemble(multi_disassemble_index_list, list(not_visited_index), element_object_list)

            else:
                Planner.Assemble(
                    element_object_index, list(not_visited_index) + list(blacklist_index), element_object_list
                )
                blacklist_index.append(element_object_index)

        return list(path_index)[::-1]

    @staticmethod
    def GetBacktrackElementsFromPath(element_index: int, path: deque) -> list:
        for path_step in path:
            if element_index in path_step:
                return list(path_step)
        return []

    @staticmethod
    def GetElementObjects(
        element_from_index: dict[Element], contact_id_pairs: list[list], grounded_elements_index: list
    ) -> list[ElementObject]:
        element_object_list = []
        for index, element in element_from_index.items():
            element: Element
            element_object_list.append(
                ElementObject(
                    element.index,
                    element.body,
                    element.init_pose,
                    element.goal_pose,
                    element.axis_endpoints,
                    coupled_elements=ElementObject.GetCoupledElements(element.index, contact_id_pairs),
                    is_grounded=True if index in grounded_elements_index else False,
                )
            )
        element_object_list.sort(key=operator.attrgetter("index"))
        return element_object_list

    @staticmethod
    def UpdateElements(assembled_list: list, element_object_list: list[ElementObject]):
        for element_object in element_object_list:
            element_object.UpdateConstrain(assembled_list)
        for element_object in element_object_list:
            element_object.UpdateStatus(element_object_list)

    @staticmethod
    def ElementsStatusCheck(index_list: list, element_object_list: list[ElementObject]) -> bool:
        for index in index_list:
            if element_object_list[index].status != ElementStatus.fixed:
                return False
        return True

    @staticmethod
    def ElementsStatusCount(index_list: list, element_object_list: list[ElementObject], status: ElementStatus) -> int:
        cnt = 0
        for index in index_list:
            if element_object_list[index].status == status:
                cnt += 1
        return cnt

    @staticmethod
    def GetElementIndexBYStatus(
        index_list: list, element_object_list: list[ElementObject], status: ElementStatus
    ) -> list:
        index_list_rtn = []
        for index in index_list:
            if element_object_list[index].status == status:
                index_list_rtn.append(index)
        return index_list_rtn

    @staticmethod
    def Assemble(index: int, visited_index_list: list, element_object_list: list[ElementObject]):
        element_object_list[index].Assemble(visited_index_list)
        visited_index_list.append(index)
        Planner.UpdateElements(visited_index_list, element_object_list)

    @staticmethod
    def Disassemble(index: int, visited_index_list: list, element_object_list: list[ElementObject]):
        element_object_list[index].Disassemble()
        Planner.UpdateElements(visited_index_list, element_object_list)

    @staticmethod
    def MultiDisassemble(index_list: list, visited_index_list: list, element_object_list: list[ElementObject]):
        for index in index_list:
            Planner.Disassemble(index, visited_index_list, element_object_list)

    @staticmethod
    def MultiAssemble(index_list: list, visited_index_list: list, element_object_list: list[ElementObject]):
        for index in index_list:
            Planner.Assemble(index, visited_index_list, element_object_list)

    @staticmethod
    def SetHold(index_list: list, element_object_list: list[ElementObject]):
        for index in index_list:
            element_object_list[index].status = ElementStatus.fixed

    @staticmethod
    def FindMin(
        index_list: list, element_object_list: list[ElementObject], key: str = "heuristic_value"
    ) -> Tuple[int, int]:
        """
        @brief: find min in index_list\n
        ---
        @return:\n
            element_index\n
            index in index_list\n
        """
        min_value = np.inf
        min_index = -1
        for element_index in index_list:
            if getattr(element_object_list[element_index], key) < min_value:
                min_value = getattr(element_object_list[element_index], key)
                min_index = element_index
        return min_index, index_list.index(min_index)

    @staticmethod
    def FindMax(
        index_list: list, element_object_list: list[ElementObject], key: str = "heuristic_value"
    ) -> Tuple[int, int]:
        """
        @brief: find max in index_list\n
        ---
        @return:\n
            element_index\n
            index in index_list\n
        """
        max_value = -np.inf
        max_index = -1
        for element_index in index_list:
            if getattr(element_object_list[element_index], key) > max_value:
                max_value = getattr(element_object_list[element_index], key)
                max_index = element_index
        return max_index, index_list.index(max_index)
