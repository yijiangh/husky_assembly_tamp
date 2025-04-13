import operator
import os
import sys
import time
from collections import deque
from copy import deepcopy
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import pybullet_planning as pp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from robot.robot import PathItem, Robot
from symbolic_planner.element_object import ElementObject
from symbolic_planner.element_status import ElementStatus
from symbolic_planner.heuristic import (
    BasicHeuristic,
    CenterDistanceHeuristic,
    GroundedChainHeuristic,
    GroundedHeightHeuristic,
)
from termcolor import cprint
from utils.collision import Element
from utils.util import TermPrint, flatten, timeit_decorator_counter


class PlanState(object):
    _instances = {}

    def __new__(cls, assembled, unassembled, blacklist, last_step=None):
        sorted_id = tuple(sorted(assembled))
        if sorted_id not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[sorted_id] = instance
        return cls._instances[sorted_id]

    def __init__(
        self,
        assembled: List[int],
        unassembled: List[int],
        blacklist: List[int],
        last_step: Union[List[int], None] = None,
    ) -> None:
        if not hasattr(self, "initialized"):
            self._assembled = assembled
            self._unassembled = unassembled
            self._blacklist = blacklist
            self._last_step = last_step

            self._father = None
            self.is_deadend = False
            self.initialized = True

            self.path = []

    def __repr__(self):
        return f"PlanState(assembled={self._assembled}, unassembled={self._unassembled}, blacklist={self._blacklist})"

    def UpdateBlacklist(self, index_list: List[int]):
        for index in index_list:
            if index in self._assembled:
                self._assembled.remove(index)
            if index in self._unassembled:
                self._unassembled.remove(index)
            if index not in self._blacklist:
                self._blacklist.append(index)

    def UpdateFatherState(self):
        cur_state = self
        while cur_state.deadend and cur_state.father != None:
            new_blacklist_list = PlanState.Difference(cur_state.assembled, cur_state.father.assembled)
            cur_state.father.UpdateBlacklist(new_blacklist_list)
            cur_state = cur_state.father

    def UnassembledRemove(self, index_list: List[int]):
        rtn = self.unassembled
        for index in index_list:
            if index in rtn:
                rtn.remove(index)
        return rtn

    def FindNextIndex(self, fn_handle, **kwargs):
        return fn_handle(list(self._unassembled), **kwargs)

    def SetFather(self, father: "PlanState"):
        self._father = father

    def TraceBack(self) -> "PlanState":
        self.UpdateFatherState()
        cur_state = self
        while cur_state.deadend and cur_state.father != None:
            cur_state = cur_state.father
        return cur_state

    @property
    def finished(self) -> bool:
        if len(self.blacklist) == 0 and len(self.unassembled) == 0:
            return True
        return False

    @property
    def deadend(self) -> bool:
        if len(self._unassembled) == 0 and len(self._blacklist) != 0:
            self.is_deadend = True
        else:
            self.is_deadend = False
        return self.is_deadend

    @property
    def assembled(self) -> List[int]:
        return list(deepcopy(self._assembled))

    @property
    def unassembled(self) -> List[int]:
        return list(deepcopy(self._unassembled))

    @property
    def blacklist(self) -> List[int]:
        return list(deepcopy(self._blacklist))

    @property
    def last_step(self) -> List[int]:
        return list(deepcopy(self._last_step))

    @property
    def father(self) -> "PlanState":
        return self._father

    @staticmethod
    def GenerateNextState(current_state: "PlanState", index_list: List[int]) -> "PlanState":
        """
        @brief: 产生下一个状态，清空blacklist\n
        """
        assembled = deepcopy(current_state._assembled)
        unassembled = deepcopy(current_state._unassembled)
        blacklist = deepcopy(current_state._blacklist)

        # push element from index_list to assembled
        for index in index_list:
            if index not in assembled:
                assembled.append(index)
            if index in unassembled:
                unassembled.remove(index)

        # push element from blacklist to unassembled
        for index in blacklist:
            if index not in unassembled:
                unassembled.append(index)

        rtn = PlanState(assembled, unassembled, [], last_step=index_list)
        rtn.SetFather(current_state)

        return rtn

    @staticmethod
    def Difference(minuend: List, subtrahend: List):
        return list(set(minuend) - set(subtrahend))

    @staticmethod
    def GetPath(end_state: "PlanState") -> List[int]:
        path_invert = []
        cur_state = end_state
        while cur_state.father != None:
            # step = PlanState.Difference(cur_state.assembled, cur_state.father.assembled)
            step = cur_state.last_step
            path_invert.append(step)
            cur_state = cur_state.father
        return path_invert[::-1]


class CooperationPlanState(object):
    _instances = {}

    def __new__(
        cls,
        assembled: List[int],
        unassembled: List[int],
        blacklist: List[int],
        hold: List[int],
        last_step: Union[List[int], None] = None,
        root: Union[PlanState, None] = None,
    ):
        sorted_id = (tuple(sorted(assembled)), tuple(sorted(hold)))
        if sorted_id not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[sorted_id] = instance
        return cls._instances[sorted_id]

    def __init__(
        self,
        assembled: List[int],
        unassembled: List[int],
        blacklist: List[int],
        hold: List[int],
        last_step: Union[List[int], None] = None,
        root: Union[PlanState, None] = None,
    ):
        if not hasattr(self, "initialized"):
            self._assembled = assembled
            self._unassembled = unassembled
            self._blacklist = blacklist

            self._father = None
            self.is_deadend = False
            self.initialized = True

            self._hold = hold
            self._last_step = last_step
            self._root = root

    def __repr__(self):
        return f"CooperationPlanState(assembled={self._assembled}, unassembled={self._unassembled}, blacklist={self._blacklist}, hold={self._hold})"

    def UpdateBlacklist(self, index_list: List[int]):
        for index in index_list:
            if index in self._hold:
                self._hold.remove(index)
            if index in self._unassembled:
                self._unassembled.remove(index)
            if index not in self.blacklist:
                self._blacklist.append(index)

    def UpdateFatherState(self):
        cur_state = self
        while cur_state.deadend and cur_state.father != None:
            new_blacklist_list = CooperationPlanState.Difference(cur_state.hold, cur_state.father.hold)
            cur_state.father.UpdateBlacklist(new_blacklist_list)
            cur_state = cur_state.father

    def UnassembledRemove(self, index_list: List[int]) -> List[int]:
        rtn = self.unassembled
        for index in index_list:
            if index in rtn:
                rtn.remove(index)
        return rtn

    def FindNextIndex(self, fn_handle, **kwargs):
        return fn_handle(list(self._unassembled), **kwargs)

    def SetFather(self, father: "CooperationPlanState"):
        self._father = father

    def TraceBack(self) -> "CooperationPlanState":
        self.UpdateFatherState()
        cur_state = self
        while cur_state.deadend and cur_state.father != None:
            cur_state = cur_state.father
        return cur_state

    def TraceBack2Root(self) -> PlanState:
        state = self.TraceBack()
        return state.root

    @property
    def deadend(self) -> bool:
        if len(self._unassembled) == 0 and len(self._blacklist) != 0:
            self.is_deadend = True
        else:
            self.is_deadend = False
        return self.is_deadend

    @property
    def assembled(self) -> List[int]:
        return list(deepcopy(self._assembled))

    @property
    def unassembled(self) -> List[int]:
        return list(deepcopy(self._unassembled))

    @property
    def blacklist(self) -> List[int]:
        return list(deepcopy(self._blacklist))

    @property
    def hold(self) -> List[int]:
        return list(deepcopy(self._hold))

    @property
    def last_step(self) -> List[int]:
        return list(deepcopy(self._last_step))

    @property
    def father(self) -> "CooperationPlanState":
        return self._father

    @property
    def root(self) -> PlanState:
        return self._root

    @staticmethod
    def GenerateNextState(current_state: "CooperationPlanState", index_list: List[int]) -> "CooperationPlanState":
        """
        @brief: 产生下一个状态，清空blacklist\n
        """
        assembled = current_state.assembled
        unassembled = current_state.unassembled
        blacklist = current_state.blacklist
        hold = current_state.hold

        # push element from index_list to hold
        for index in index_list:
            if index not in hold:
                hold.append(index)
            if index in unassembled:
                unassembled.remove(index)

        # push element from blacklist to unassembled
        for index in blacklist:
            if index not in unassembled:
                unassembled.append(index)

        rtn = CooperationPlanState(assembled, unassembled, [], hold, last_step=index_list)
        rtn.SetFather(current_state)

        return rtn

    @staticmethod
    def Difference(minuend: List, subtrahend: List):
        return list(set(minuend) - set(subtrahend))


class Planner(object):
    def __init__(self, robot_num: int, robots: List[Robot]) -> None:
        self.robot_num = robot_num
        self.robots = robots

    def Plan(
        self, element_from_index: Dict[int, Element], contact_id_pairs: List[List], grounded_elements_index: List[int]
    ) -> List[List[int]]:
        # -------------------- Generate element objects --------------------#
        element_object_list = Planner.GetElementObjects(element_from_index, contact_id_pairs, grounded_elements_index)

        # GroundedChainHeuristic.Update(element_object_list)
        GroundedHeightHeuristic.Update(element_object_list)
        # CenterDistanceHeuristic.Update(element_object_list)

        path_index = self.Search(element_object_list)
        # path_index = self.BackwardSearchWithoutMotionPlan(element_object_list)

        # TODO: 多机优化
        # self.robots[0].BaseMotionPlan(path_index)
        return path_index

    @timeit_decorator_counter(verbose=True)
    def Search(self, element_object_list: List[ElementObject]) -> List[List[int]]:
        # -------------------- init --------------------#
        current_state = PlanState([], [obj.index for obj in element_object_list], [])
        root_state = current_state

        # -------------------- loop --------------------#
        while not root_state.deadend:

            last_time = time.time()

            for element_obj in element_object_list:
                if element_obj.index in current_state.assembled:
                    pp.set_pose(element_obj.body, element_obj.goal_pose)
                else:
                    pp.set_pose(element_obj.body, pp.Pose(point=(5, 0, 0), euler=pp.Euler(0, 1.5708, 0)))

            # -------------------- pop --------------------#
            element_object_index, _ = current_state.FindNextIndex(
                Planner.FindMin, element_object_list=element_object_list
            )

            TermPrint.print(
                "-------------------------------------------------------------------------------------------------------",
                blank_f=True,
            )
            TermPrint.print(f"start plan {element_object_index}: {current_state}")
            TermPrint.print(f"current path {PlanState.GetPath(current_state)}")
            TermPrint.print(
                "-------------------------------------------------------------------------------------------------------",
                blank_b=True,
            )

            # -------------------- assemble --------------------#
            Planner.Assemble(element_object_index, current_state.assembled, element_object_list)
            status = element_object_list[element_object_index].status

            # -------------------- decide what to do --------------------#
            if status == ElementStatus.fixed:
                plan_status = self.robots[0].ManipulatorMotionPlan(
                    element_object_index,
                    current_state.assembled,
                    current_state.UnassembledRemove([element_object_index]),
                    [],  # TODO: add attachment list
                )

                if plan_status:
                    Planner.MultiDisassemble(current_state.blacklist, current_state.assembled, element_object_list)
                    cur_time = time.time()
                    TermPrint.print(
                        f"========== plan {element_object_index}: {current_state} success {cur_time-last_time}s ==========",
                        "green",
                        blank_f=True,
                        blank_b=True,
                    )
                    last_time = cur_time
                    next_state = PlanState.GenerateNextState(current_state, [element_object_index])
                    current_state = next_state
                else:
                    Planner.Disassemble(element_object_index, current_state.assembled, element_object_list)
                    TermPrint.print(
                        f"********** plan {element_object_index}: {current_state} failed **********",
                        "red",
                        blank_f=True,
                        blank_b=True,
                    )
                    current_state.UpdateBlacklist([element_object_index])

            elif status == ElementStatus.float:
                Planner.Disassemble(element_object_index, current_state.assembled, element_object_list)
                TermPrint.print(
                    f"********** {element_object_index}: {current_state} is float **********",
                    "red",
                    blank_f=True,
                    blank_b=True,
                )
                current_state.UpdateBlacklist([element_object_index])

            elif status == ElementStatus.rotate:
                current_coop_state = CooperationPlanState(
                    current_state.assembled,
                    current_state.UnassembledRemove([element_object_index]) + current_state.blacklist,
                    [],
                    [element_object_index],
                    current_state,
                )
                solve_status, task = self.SearchRobotCooperation(element_object_list, current_coop_state)
                if solve_status:
                    Planner.MultiDisassemble(current_state.blacklist, current_state.assembled, element_object_list)
                    cur_time = time.time()
                    TermPrint.print(
                        f"========== plan {element_object_index}: {task} success {cur_time-last_time}s ==========",
                        "green",
                        blank_f=True,
                        blank_b=True,
                    )
                    last_time = cur_time
                    next_state = PlanState.GenerateNextState(current_state, task)
                    current_state = next_state
                else:
                    Planner.MultiDisassemble(task, current_state.assembled, element_object_list)
                    # Planner.Disassemble(element_object_index, current_state.assembled, element_object_list)
                    TermPrint.print(
                        f"********** cooperation plan {element_object_index}: {current_state} not found **********",
                        "red",
                        blank_f=True,
                        blank_b=True,
                    )
                    current_state.UpdateBlacklist([element_object_index])

            else:
                raise RuntimeError("This status is not possible!")

            # -------------------- Check if goal is reached --------------------#
            if current_state.finished:
                TermPrint.print(
                    "==================== Finished! ====================",
                    "green",
                    blank_f=True,
                    blank_b=True,
                )
                return PlanState.GetPath(current_state)

            # -------------------- Check if backtracking is necessary --------------------#
            if current_state.deadend:
                TermPrint.print(
                    "****************************** Deadend reached, need to traceback! *********************************",
                    "cyan",
                    blank_f=True,
                )
                TermPrint.print(f"current state: {current_state}", "cyan")
                current_state = current_state.TraceBack()
                TermPrint.print(f"traceback state: {current_state}", "cyan")
                TermPrint.print(
                    "****************************************************************************************************\n",
                    "cyan",
                    blank_b=True,
                )

        TermPrint.print(
            "******************** Plan failed! ********************",
            "red",
            blank_f=True,
            blank_b=True,
        )
        return []

    def SearchRobotCooperation(
        self, element_object_list: List[ElementObject], cur_state: CooperationPlanState
    ) -> Tuple[bool, List[int]]:
        current_state = cur_state
        current_root = cur_state

        if len(current_state.hold) == self.robot_num:
            return False, current_state.hold

        # -------------------- loop --------------------#
        while not current_root.deadend:
            # -------------------- pop --------------------#
            element_object_index, _ = current_state.FindNextIndex(
                Planner.FindMin, element_object_list=element_object_list
            )

            # -------------------- assemble --------------------#
            Planner.Assemble(element_object_index, current_state.assembled + current_state.hold, element_object_list)
            status = element_object_list[element_object_index].status

            if status == ElementStatus.fixed and not Planner.ElementsStatusCheck(
                current_state.hold, element_object_list
            ):
                Planner.Disassemble(
                    element_object_index, current_state.assembled + current_state.hold, element_object_list
                )
                current_state.UpdateBlacklist([element_object_index])

            elif status == ElementStatus.fixed and Planner.ElementsStatusCheck(current_state.hold, element_object_list):
                plan_status, plan_task = Robot.ManipulatorGroupMotionPlan(
                    self.robots,
                    current_state.hold + [element_object_index],
                    cur_state.assembled,
                    cur_state.unassembled,
                    [],  # TODO: add attachment list and consider different robot
                )
                if plan_status:
                    return True, plan_task
                else:
                    # visited_hold_index_list = list(visited_index) + list(hold_index)
                    Planner.Disassemble(
                        element_object_index, current_state.assembled + current_state.hold, element_object_list
                    )
                    # Planner.MultiDisassemble(list(hold_index), list(visited_index), element_object_list)
                    # Planner.MultiAssemble(list(hold_index), list(visited_index), element_object_list)
                    current_state.UpdateBlacklist([element_object_index])

            elif status == ElementStatus.rotate:
                next_state = CooperationPlanState.GenerateNextState(current_state, [element_object_index])
                solve_status, task = self.SearchRobotCooperation(element_object_list, next_state)
                if solve_status:
                    return solve_status, task
                else:
                    Planner.Disassemble(
                        element_object_index, current_state.assembled + current_state.hold, element_object_list
                    )
                    current_state.UpdateBlacklist([element_object_index])

            elif status == ElementStatus.float:
                Planner.Disassemble(
                    element_object_index, current_state.assembled + current_state.hold, element_object_list
                )
                current_state.UpdateBlacklist([element_object_index])

            else:
                raise RuntimeError("This status is not possible!")

            # -------------------- Check if backtracking is necessary --------------------#
            # if current_state.deadend:
            #     print("********** SearchRobotCooperation: Deadend reached, need to traceback! **********")
            #     current_state = current_state.TraceBack()

        # print("***** Cooperation search not found! *****")
        return False, current_root.hold

    def BackwardSearchWithoutMotionPlan(self, element_object_list: List[ElementObject]) -> List[List[int]]:
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
    def GetElementObjects(
        element_from_index: Dict[int, Element], contact_id_pairs: List[List[int]], grounded_elements_index: List[int]
    ) -> List[ElementObject]:
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
                    checker="algebraic",
                    is_grounded=True if index in grounded_elements_index else False,
                )
            )
        element_object_list.sort(key=operator.attrgetter("index"))
        return element_object_list

    @staticmethod
    def UpdateElements(assembled_list: List[int], element_object_list: List[ElementObject]):
        for element_object in element_object_list:
            element_object.UpdateConstrain(assembled_list)
        for element_object in element_object_list:
            element_object.UpdateStatus(assembled_list, element_object_list)

    @staticmethod
    def ElementsStatusCheck(index_list: List[int], element_object_list: List[ElementObject]) -> bool:
        for index in index_list:
            if element_object_list[index].status != ElementStatus.fixed:
                return False
        return True

    @staticmethod
    def ElementsStatusCount(
        index_list: List[int], element_object_list: List[ElementObject], status: ElementStatus
    ) -> int:
        cnt = 0
        for index in index_list:
            if element_object_list[index].status == status:
                cnt += 1
        return cnt

    @staticmethod
    def GetElementIndexBYStatus(
        index_list: List[int], element_object_list: List[ElementObject], status: ElementStatus
    ) -> List[int]:
        index_list_rtn = []
        for index in index_list:
            if element_object_list[index].status == status:
                index_list_rtn.append(index)
        return index_list_rtn

    @staticmethod
    def Assemble(index: int, visited_index_list: List[int], element_object_list: List[ElementObject]):
        element_object_list[index].Assemble(visited_index_list)
        visited_index_list.append(index)
        Planner.UpdateElements(visited_index_list, element_object_list)

    @staticmethod
    def Disassemble(index: int, visited_index_list: List[int], element_object_list: List[ElementObject]):
        element_object_list[index].Disassemble()
        Planner.UpdateElements(visited_index_list, element_object_list)

    @staticmethod
    def MultiDisassemble(
        index_list: List[int], visited_index_list: List[int], element_object_list: List[ElementObject]
    ):
        for index in index_list:
            Planner.Disassemble(index, visited_index_list, element_object_list)

    @staticmethod
    def MultiAssemble(index_list: List[int], visited_index_list: List[int], element_object_list: List[ElementObject]):
        for index in index_list:
            Planner.Assemble(index, visited_index_list, element_object_list)

    @staticmethod
    def SetHold(index_list: List[int], element_object_list: List[ElementObject]):
        for index in index_list:
            element_object_list[index].status = ElementStatus.fixed

    @staticmethod
    def FindMin(
        index_list: List[int], element_object_list: List[ElementObject], key: str = "heuristic_value"
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
        index_list: List[int], element_object_list: List[ElementObject], key: str = "heuristic_value"
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


if __name__ == "__main__":
    # state1 = PlanState([1, 2, 3], [4, 5, 6], [0])
    # state2 = PlanState.GenerateNextState(state1, [5])
    # print(state1)
    # print(state2)
    # print(state2._father)
    # state1.UpdateBlacklist([5])
    # print(state2._father)

    f_state1 = PlanState([1, 2, 3], [4, 5, 6], [0])

    state1 = CooperationPlanState([1, 2, 3], [4, 5, 6], [0], [7, 8], [])
    state11 = CooperationPlanState([3, 2, 1], [4, 5, 6], [0], [7, 8, 9], [])
    state12 = CooperationPlanState([1, 2, 3], [4, 5, 6], [0], [8, 7], [])

    print(state11 is state1)
    print(state12 is state1)
    print(f_state1 is state1)

    print(state1)
    print(state11)
    print(state12)
