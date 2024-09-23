import operator
from collections import deque

from collision import Element
from element_object import ElementObject, ElementStatus
from heuristic import BasicHeuristic, CenterDistanceHeuristic, GroundedChainHeuristic, GroundedHeightHeuristic
from utils import flatten
from typing import Tuple
from copy import deepcopy


class Planner(object):
    def __init__(self, robot_num: int = 3) -> None:
        self.robot_num = robot_num

    def Plan(
        self, element_from_index: dict[Element], contact_id_pairs: list[list], grounded_elements_index: list
    ) -> list:
        # -------------------- Generate element objects --------------------#
        element_object_list = Planner.GetElementObjects(element_from_index, contact_id_pairs, grounded_elements_index)

        # GroundedChainHeuristic.Update(element_object_list)
        GroundedHeightHeuristic.Update(element_object_list)
        # CenterDistanceHeuristic.Update(element_object_list)
        element_object_list.sort(key=operator.attrgetter("heuristic_value"))

        path_index = self.PlanWithoutMotionPlan(element_object_list)
        return path_index

        # return [element_object.index for element_object in element_object_list]

    def PlanWithoutMotionPlan(self, element_object_list: list[ElementObject]) -> list:
        # -------------------- init --------------------#
        element_object_list.sort(key=operator.attrgetter("heuristic_value"))
        not_visited_index = deque([obj.index for obj in element_object_list])
        visited_index = deque([])
        path_index = deque([])
        blacklist_index = deque([])

        # -------------------- loop --------------------#
        while len(not_visited_index) != 0:
            # -------------------- sort --------------------#
            element_object_list.sort(key=operator.attrgetter("index"))
            not_visited_list = [element_object_list[not_id] for not_id in list(not_visited_index)]
            not_visited_list.sort(key=operator.attrgetter("heuristic_value"))
            not_visited_index = deque([obj.index for obj in not_visited_list])
            element_object_list.sort(key=operator.attrgetter("heuristic_value"))

            # -------------------- pop --------------------#
            element_object_index = not_visited_index.popleft()

            # -------------------- assemble --------------------#
            visited_index_list = list(visited_index)
            Planner.Assemble(element_object_index, visited_index_list, element_object_list)
            element_object_list.sort(key=operator.attrgetter("index"))
            status = element_object_list[element_object_index].status
            element_object_list.sort(key=operator.attrgetter("heuristic_value"))

            # -------------------- decide what to do --------------------#
            if status == ElementStatus.fixed:
                visited_index_list = list(visited_index)
                Planner.MultiDisassemble(list(blacklist_index), visited_index_list, element_object_list)
                not_visited_index.extend(blacklist_index)
                blacklist_index.clear()
                visited_index.append(element_object_index)
                path_index.append([element_object_index])

            elif status == ElementStatus.float:
                visited_index_list = list(visited_index)
                Planner.Disassemble(element_object_index, visited_index_list, element_object_list)
                blacklist_index.append(element_object_index)

            elif status == ElementStatus.rotate:
                state, task = self.PlanRobotCooperationWithoutMotionPlan(
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

        return list(path_index)

    def PlanRobotCooperationWithoutMotionPlan(
        self,
        element_object_list: list[ElementObject],
        visited_index_list: list,
        not_visited_index_list: list,
        hold_index_list: list,
    ) -> Tuple[bool, list[ElementObject]]:
        pass

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
            # -------------------- sort --------------------#
            element_object_list.sort(key=operator.attrgetter("index"))
            not_visited_list = [element_object_list[not_id] for not_id in list(not_visited_index)]
            not_visited_list.sort(key=operator.attrgetter("heuristic_value"))
            not_visited_index = deque([obj.index for obj in not_visited_list])
            element_object_list.sort(key=operator.attrgetter("heuristic_value"))

            # -------------------- pop --------------------#
            element_object_index = not_visited_index.popleft()

            # -------------------- assemble --------------------#
            visited_hold_index_list = list(visited_index) + list(hold_index)
            Planner.Assemble(element_object_index, visited_hold_index_list, element_object_list)
            element_object_list.sort(key=operator.attrgetter("index"))
            status = element_object_list[element_object_index].status
            element_object_list.sort(key=operator.attrgetter("heuristic_value"))

            if status == ElementStatus.fixed and not Planner.ElementsStatusCheck(list(hold_index), element_object_list):
                visited_hold_index_list = list(visited_index) + list(hold_index)
                Planner.Disassemble(element_object_index, visited_hold_index_list, element_object_list)
                blacklist_index.append(element_object_index)

            elif status == ElementStatus.fixed and Planner.ElementsStatusCheck(list(hold_index), element_object_list):
                hold_index.append(element_object_index)
                return True, list(hold_index)

            elif status == ElementStatus.rotate:
                state, task = self.PlanRobotCooperationWithoutMotionPlan(
                    element_object_list, list(visited_index), list(not_visited_index), [element_object_index] + list(hold_index)
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
        element_object_list.sort(key=operator.attrgetter("index"))
        for index in index_list:
            if element_object_list[index].status != ElementStatus.fixed:
                element_object_list.sort(key=operator.attrgetter("heuristic_value"))
                return False
        element_object_list.sort(key=operator.attrgetter("heuristic_value"))
        return True

    @staticmethod
    def Assemble(index: int, visited_index_list: list, element_object_list: list[ElementObject]):
        element_object_list.sort(key=operator.attrgetter("index"))
        element_object_list[index].Assemble(visited_index_list)
        visited_index_list.append(index)
        Planner.UpdateElements(visited_index_list, element_object_list)
        element_object_list.sort(key=operator.attrgetter("heuristic_value"))

    @staticmethod
    def Disassemble(index: int, visited_index_list: list, element_object_list: list[ElementObject]):
        element_object_list.sort(key=operator.attrgetter("index"))
        element_object_list[index].Disassemble()
        Planner.UpdateElements(visited_index_list, element_object_list)
        element_object_list.sort(key=operator.attrgetter("heuristic_value"))

    @staticmethod
    def MultiDisassemble(
        index_list: list, visited_index_list: list, element_object_list: list[ElementObject]
    ) -> list[ElementObject]:
        for index in index_list:
            Planner.Disassemble(index, visited_index_list, element_object_list)

    @staticmethod
    def MultiAssemble(
        index_list: list, visited_index_list: list, element_object_list: list[ElementObject]
    ) -> list[ElementObject]:
        for index in index_list:
            Planner.Assemble(index, visited_index_list, element_object_list)

    @staticmethod
    def SetHold(index_list: list, element_object_list: list[ElementObject]):
        element_object_list.sort(key=operator.attrgetter("index"))
        for index in index_list:
            element_object_list[index].status = ElementStatus.fixed
        element_object_list.sort(key=operator.attrgetter("heuristic_value"))

    # @staticmethod
    # def UpdateList(input_list: list[ElementObject], element_object_list: list[ElementObject]) -> list[ElementObject]:
    #     element_object_list.sort(key=operator.attrgetter("index"))
    #     new_list = []
    #     for obj in input_list:
    #         new_list.append(element_object_list[obj.index])
    #     element_object_list.sort(key=operator.attrgetter("heuristic_value"))
    #     return new_list
