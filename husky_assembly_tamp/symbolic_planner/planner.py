import operator
import os
import sys
import time
from collections import deque
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pybullet_planning as pp

from husky_assembly_tamp.robot.robot import PathItem, Robot
from husky_assembly_tamp.symbolic_planner.element_object import ElementObject
from husky_assembly_tamp.symbolic_planner.element_status import ElementStatus
from husky_assembly_tamp.symbolic_planner.heuristic import (
    BasicHeuristic,
    CenterDistanceHeuristic,
    GroundedChainHeuristic,
    GroundedHeightHeuristic,
)
from termcolor import cprint
from husky_assembly_tamp.utils.collision import Element
from husky_assembly_tamp.utils.util import TermPrint, flatten, timeit_decorator_counter

# Import MultiPhaseKomoSolver if available
try:
    from husky_assembly_tamp.solver.komo_multi_frame_solver import MultiPhaseKomoSolver, RobotPositionCalculator, GeometryCalculator
    KOMO_SOLVER_AVAILABLE = True
except ImportError:
    KOMO_SOLVER_AVAILABLE = False
    MultiPhaseKomoSolver = None
    RobotPositionCalculator = None
    GeometryCalculator = None


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
        self, 
        element_from_index: Dict[int, Element], 
        contact_id_pairs: List[List], 
        grounded_elements_index: List[int],
        config=None,
        robot_gripper_frames: Optional[List[List[str]]] = None,
        robot_base_frame_names: Optional[List[str]] = None,
    ) -> Tuple[List[List[int]], List[Optional[List[np.ndarray]]]]:
        # -------------------- Generate element objects --------------------#
        element_object_list = Planner.GetElementObjects(element_from_index, contact_id_pairs, grounded_elements_index)

        # GroundedChainHeuristic.Update(element_object_list)
        GroundedHeightHeuristic.Update(element_object_list)
        # CenterDistanceHeuristic.Update(element_object_list)

        # path_index = self.Search(element_object_list)
        path_index, keyframes_list = self.BackwardSearchWithoutMotionPlan(
            element_object_list,
            config=config,
            robot_gripper_frames=robot_gripper_frames,
            robot_base_frame_names=robot_base_frame_names,
        )

        # TODO: 多机优化
        # self.robots[0].BaseMotionPlan(path_index)
        return path_index, keyframes_list

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

    def _arrange_frames_for_step(
        self,
        element_indices: List[int],
        robot_gripper_frames: List[List[str]],
        robot_base_frame_names: List[str],
    ) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
        """
        Arrange elements into frames/phases based on robot availability.
        This checks whether a step can be arranged with available robots.
        
        Args:
            element_indices: List of element indices to arrange
            robot_gripper_frames: List of gripper frame names for each robot
            robot_base_frame_names: List of base frame names for each robot
        
        Returns:
            Tuple of (robot_names_phases, target_names_phases, baselink_names_phases)
            Returns empty lists if arrangement is not possible
        """
        num_elements = len(element_indices)
        
        if num_elements == 0:
            return [], [], []
        
        if len(robot_gripper_frames) == 0:
            return [], [], []
        
        if num_elements == 1:
            # Single element: single frame (single robot grasps single element)
            element_idx = element_indices[0]
            element_name = f"element_{element_idx + 1}"
            
            # Use first available robot
            robot_idx = 0
            grippers = robot_gripper_frames[robot_idx]
            
            # For dual-arm robots, use both left and right tools for the same element
            if len(grippers) > 1:
                # Dual-arm: use both arms
                robot_gripper_names = grippers
                baselink_names = [robot_base_frame_names[robot_idx], robot_base_frame_names[robot_idx]]
            else:
                # Single-arm: use the only gripper
                robot_gripper_names = [grippers[0]]
                baselink_names = [robot_base_frame_names[robot_idx]]
            
            # Single phase: robot(s) grasp one element
            robot_names_phases = [robot_gripper_names]
            target_names_phases = [[element_name] * len(robot_gripper_names)]
            baselink_names_phases = [baselink_names]
            
            return robot_names_phases, target_names_phases, baselink_names_phases
        
        # Multiple elements: arrange into frames
        robot_num = len(robot_gripper_frames)
        
        # Identify single-arm and dual-arm robots
        single_arm_robot_indices = []
        dual_arm_robot_indices = []
        for i, grippers in enumerate(robot_gripper_frames):
            if len(grippers) == 1:
                single_arm_robot_indices.append(i)
            else:
                dual_arm_robot_indices.append(i)
        
        dual_robot_num = len(dual_arm_robot_indices)
        
        # Check if robot_num >= len(element_indices): arrange as single frame
        if robot_num >= num_elements:
            # Enough robots: one robot per element, single frame
            robot_names_phases = [[]]
            target_names_phases = [[]]
            baselink_names_phases = [[]]
            
            for i, element_idx in enumerate(element_indices):
                element_name = f"element_{element_idx + 1}"
                grippers = robot_gripper_frames[i]
                
                # For dual-arm robots, use both left and right tools for the same element
                if len(grippers) > 1:
                    # Dual-arm: use both arms
                    robot_gripper_names = grippers
                    robot_names_phases[0].extend(robot_gripper_names)
                    target_names_phases[0].extend([element_name, element_name])
                    baselink_names_phases[0].extend([robot_base_frame_names[i], robot_base_frame_names[i]])
                else:
                    # Single-arm: use the only gripper
                    robot_gripper_name = grippers[0]
                    robot_names_phases[0].append(robot_gripper_name)
                    target_names_phases[0].append(element_name)
                    baselink_names_phases[0].append(robot_base_frame_names[i])
            
            return robot_names_phases, target_names_phases, baselink_names_phases
        
        # Check if robot_num < len(element_indices) and dual_robot_num + robot_num >= len(element_indices)
        # Arrange as dual-frames task
        if robot_num < num_elements and dual_robot_num + robot_num >= num_elements:
            # Calculate number of frames needed (at least 2 frames)
            num_frames = 2
            
            # Determine which elements will be held unchanged (by single-arm robots)
            num_unchanged_elements = min(len(single_arm_robot_indices), num_elements - robot_num)
            unchanged_element_indices = element_indices[:num_unchanged_elements] if num_unchanged_elements > 0 else []
            new_element_indices = element_indices[num_unchanged_elements:] if num_unchanged_elements > 0 else element_indices
            
            robot_names_phases = []
            target_names_phases = []
            baselink_names_phases = []
            
            # Distribute new elements across frames
            elements_per_frame = len(new_element_indices) // num_frames if num_frames > 0 else 0
            remaining_elements = len(new_element_indices) % num_frames if num_frames > 0 else 0
            
            new_element_idx = 0
            for frame_idx in range(num_frames):
                frame_robots = []
                frame_targets = []
                frame_baselinks = []
                
                # First, assign unchanged elements (held by single-arm robots in all frames)
                for unchanged_idx, unchanged_element_idx in enumerate(unchanged_element_indices):
                    if unchanged_idx < len(single_arm_robot_indices):
                        robot_idx = single_arm_robot_indices[unchanged_idx]
                        element_name = f"element_{unchanged_element_idx + 1}"
                        grippers = robot_gripper_frames[robot_idx]
                        
                        # Single-arm robot: use the only gripper
                        robot_gripper_name = grippers[0]
                        
                        frame_robots.append(robot_gripper_name)
                        frame_targets.append(element_name)
                        frame_baselinks.append(robot_base_frame_names[robot_idx])
                
                # Then, assign new elements for this frame
                num_new_elements_this_frame = elements_per_frame + (1 if frame_idx < remaining_elements else 0)
                
                # Use remaining robots (dual-arm or unused single-arm) for new elements
                available_robot_indices = list(range(len(robot_gripper_frames)))
                # Remove robots already used for unchanged elements
                for unchanged_idx in range(len(unchanged_element_indices)):
                    if unchanged_idx < len(single_arm_robot_indices):
                        if single_arm_robot_indices[unchanged_idx] in available_robot_indices:
                            available_robot_indices.remove(single_arm_robot_indices[unchanged_idx])
                
                # Assign new elements to available robots
                for _ in range(num_new_elements_this_frame):
                    if new_element_idx >= len(new_element_indices) or len(available_robot_indices) == 0:
                        break
                    
                    element_idx = new_element_indices[new_element_idx]
                    element_name = f"element_{element_idx + 1}"
                    robot_idx = available_robot_indices.pop(0)
                    grippers = robot_gripper_frames[robot_idx]
                    
                    # For dual-arm robots, use both left and right tools for the same element
                    if len(grippers) > 1:
                        # Dual-arm: use both arms
                        robot_gripper_names = grippers
                        frame_robots.extend(robot_gripper_names)
                        frame_targets.extend([element_name, element_name])
                        frame_baselinks.extend([robot_base_frame_names[robot_idx], robot_base_frame_names[robot_idx]])
                    else:
                        # Single-arm: use the only gripper
                        robot_gripper_name = grippers[0]
                        frame_robots.append(robot_gripper_name)
                        frame_targets.append(element_name)
                        frame_baselinks.append(robot_base_frame_names[robot_idx])
                    
                    new_element_idx += 1
                
                if len(frame_robots) > 0:
                    robot_names_phases.append(frame_robots)
                    target_names_phases.append(frame_targets)
                    baselink_names_phases.append(frame_baselinks)
            
            return robot_names_phases, target_names_phases, baselink_names_phases
        
        # Fallback: More elements than robots, arrange into multiple frames
        # Calculate number of frames needed
        num_frames = (num_elements + robot_num - 1) // robot_num  # Ceiling division
        if num_frames < 2:
            num_frames = 2  # At least 2 frames
        
        # Determine which elements will be held unchanged (by single-arm robots)
        num_unchanged_elements = min(len(single_arm_robot_indices), num_elements - robot_num)
        unchanged_element_indices = element_indices[:num_unchanged_elements] if num_unchanged_elements > 0 else []
        new_element_indices = element_indices[num_unchanged_elements:] if num_unchanged_elements > 0 else element_indices
        
        robot_names_phases = []
        target_names_phases = []
        baselink_names_phases = []
        
        # Distribute new elements across frames
        elements_per_frame = len(new_element_indices) // num_frames if num_frames > 0 else 0
        remaining_elements = len(new_element_indices) % num_frames if num_frames > 0 else 0
        
        new_element_idx = 0
        for frame_idx in range(num_frames):
            frame_robots = []
            frame_targets = []
            frame_baselinks = []
            
            # First, assign unchanged elements (held by single-arm robots in all frames)
            for unchanged_idx, unchanged_element_idx in enumerate(unchanged_element_indices):
                if unchanged_idx < len(single_arm_robot_indices):
                    robot_idx = single_arm_robot_indices[unchanged_idx]
                    element_name = f"element_{unchanged_element_idx + 1}"
                    grippers = robot_gripper_frames[robot_idx]
                    
                    # Single-arm robot: use the only gripper
                    robot_gripper_name = grippers[0]
                    
                    frame_robots.append(robot_gripper_name)
                    frame_targets.append(element_name)
                    frame_baselinks.append(robot_base_frame_names[robot_idx])
            
            # Then, assign new elements for this frame
            num_new_elements_this_frame = elements_per_frame + (1 if frame_idx < remaining_elements else 0)
            
            # Use remaining robots (dual-arm or unused single-arm) for new elements
            available_robot_indices = list(range(len(robot_gripper_frames)))
            # Remove robots already used for unchanged elements
            for unchanged_idx in range(len(unchanged_element_indices)):
                if unchanged_idx < len(single_arm_robot_indices):
                    if single_arm_robot_indices[unchanged_idx] in available_robot_indices:
                        available_robot_indices.remove(single_arm_robot_indices[unchanged_idx])
            
            # Assign new elements to available robots
            for _ in range(num_new_elements_this_frame):
                if new_element_idx >= len(new_element_indices) or len(available_robot_indices) == 0:
                    break
                
                element_idx = new_element_indices[new_element_idx]
                element_name = f"element_{element_idx + 1}"
                robot_idx = available_robot_indices.pop(0)
                grippers = robot_gripper_frames[robot_idx]
                
                # For dual-arm robots, use both left and right tools for the same element
                if len(grippers) > 1:
                    # Dual-arm: use both arms
                    robot_gripper_names = grippers
                    frame_robots.extend(robot_gripper_names)
                    frame_targets.extend([element_name, element_name])
                    frame_baselinks.extend([robot_base_frame_names[robot_idx], robot_base_frame_names[robot_idx]])
                else:
                    # Single-arm: use the only gripper
                    robot_gripper_name = grippers[0]
                    frame_robots.append(robot_gripper_name)
                    frame_targets.append(element_name)
                    frame_baselinks.append(robot_base_frame_names[robot_idx])
                
                new_element_idx += 1
            
            if len(frame_robots) > 0:
                robot_names_phases.append(frame_robots)
                target_names_phases.append(frame_targets)
                baselink_names_phases.append(frame_baselinks)
        
        return robot_names_phases, target_names_phases, baselink_names_phases

    def _setup_solver_for_step(
        self,
        element_indices: List[int],
        config,
        robot_gripper_frames: List[List[str]],
        robot_base_frame_names: List[str],
        element_object_list: List[ElementObject],
    ) -> Optional[Tuple[MultiPhaseKomoSolver, np.ndarray]]:
        """
        Set up MultiPhaseKomoSolver for a given step.
        First checks if elements can be arranged, then sets up the solver.
        
        Args:
            element_indices: List of element indices for this step
            config: ry.Config object
            robot_gripper_frames: List of gripper frame names for each robot
            robot_base_frame_names: List of base frame names for each robot
            element_object_list: List of ElementObject instances to access element positions
            
        Returns:
            Tuple of (solver, initial_state_path) if setup successful, None otherwise
        """
        if not KOMO_SOLVER_AVAILABLE or config is None:
            return None
            
        num_elements = len(element_indices)
        if num_elements == 0:
            return None
        
        # First, check if elements can be arranged with available robots
        robot_names_phases, target_names_phases, baselink_names_phases = self._arrange_frames_for_step(
            element_indices, robot_gripper_frames, robot_base_frame_names
        )
        
        # If arrangement failed (empty lists), return None
        if len(robot_names_phases) == 0 or len(target_names_phases) == 0:
            return None
        
        # Helper function to calculate pose for a target element
        def calculate_pose_for_element(target_name: str) -> Tuple[np.ndarray, List[float]]:
            """Calculate base pose facing a target element."""
            default_far_away_pos = np.array([10.0, 10.0, 0.0])
            default_quat = [1.0, 0.0, 0.0, 0.0]
            far_away_distance = -1.0  # Negative distance means away from element but facing it
            
            # Extract element index from target name (e.g., "element_1" -> 0)
            try:
                element_idx = int(target_name.split("_")[1]) - 1
            except (ValueError, IndexError):
                return (default_far_away_pos.copy(), default_quat)
            
            # Get element object
            if element_idx >= len(element_object_list):
                return (default_far_away_pos.copy(), default_quat)
            
            element_obj = element_object_list[element_idx]
            
            # Get element position from goal_pose
            element_pos = None
            if hasattr(element_obj, 'goal_pose') and element_obj.goal_pose:
                goal_point = element_obj.goal_pose[0]
                if isinstance(goal_point, (list, tuple)):
                    element_pos = np.array(goal_point[:3])
                elif isinstance(goal_point, np.ndarray):
                    element_pos = goal_point[:3]
            
            if element_pos is None:
                return (default_far_away_pos.copy(), default_quat)
            
            # Get element direction (edge direction from vertices)
            edge_dir = None
            if hasattr(element_obj, 'vertices') and element_obj.vertices:
                # Calculate direction from vertices (endpoints)
                vertices = element_obj.vertices
                if len(vertices) >= 2:
                    v1 = np.array(vertices[0][:3]) if isinstance(vertices[0], (list, tuple)) else vertices[0][:3]
                    v2 = np.array(vertices[1][:3]) if isinstance(vertices[1], (list, tuple)) else vertices[1][:3]
                    dir_vec = v2 - v1
                    if np.linalg.norm(dir_vec) > 1e-6:
                        edge_dir = dir_vec[:2] / np.linalg.norm(dir_vec[:2])  # Use only x, y components
            
            # Default direction if not available
            if edge_dir is None:
                edge_dir = np.array([1.0, 0.0])
            
            # Calculate pose facing target element (far away but facing it)
            if RobotPositionCalculator is not None:
                base_pos, base_quat = RobotPositionCalculator.calculate_pose_toward_target(
                    element_pos,
                    edge_dir,
                    far_away_distance,
                    element_pos  # Look at element position
                )
                return (base_pos, base_quat)
            
            return (default_far_away_pos.copy(), default_quat)
        
        # Prepare initial state path based on number of phases
        if len(robot_names_phases) == 1:
            # Single phase: calculate poses for all robots in this phase
            phase_robots = robot_names_phases[0]
            phase_targets = target_names_phases[0]
            phase_baselinks = baselink_names_phases[0]
            
            # Map baselink to target for this phase
            baselink_to_target = {}
            for robot_name, target_name, baselink_name in zip(phase_robots, phase_targets, phase_baselinks):
                if baselink_name not in baselink_to_target:
                    baselink_to_target[baselink_name] = target_name
            
            # Create base_poses list in order of robot_base_frame_names
            base_poses = []
            base_frame_names_ordered = []
            default_far_away_pos = np.array([10.0, 10.0, 0.0])
            default_quat = [1.0, 0.0, 0.0, 0.0]
            
            for baselink_name in robot_base_frame_names:
                if baselink_name in baselink_to_target:
                    pose = calculate_pose_for_element(baselink_to_target[baselink_name])
                    base_poses.append(pose)
                    base_frame_names_ordered.append(baselink_name)
                else:
                    base_poses.append((default_far_away_pos.copy(), default_quat))
                    base_frame_names_ordered.append(baselink_name)
            
            # Apply base poses to config
            if len(base_frame_names_ordered) > 0 and RobotPositionCalculator is not None:
                RobotPositionCalculator.apply_base_poses(config, base_frame_names_ordered, base_poses)
            
            # Get initial joint state after applying base poses
            try:
                initial_state = config.getJointState()
            except:
                initial_state = np.zeros(config.getJointDimension())
            
            # Set specific joint initial positions
            if len(initial_state) > 20:
                initial_state[3] = 0
                initial_state[4] = -np.pi / 2 - np.pi / 4
                initial_state[5:10] = 0
                initial_state[10] = -np.pi / 2 + np.pi / 4
                initial_state[18] = 0
                initial_state[19] = -np.pi / 2
                initial_state[20:] = 0
            
            initial_state_path = initial_state
        else:
            # Multiple phases: generate different base poses for each phase
            initial_state_path = []
            
            for phase_idx in range(len(robot_names_phases)):
                phase_robots = robot_names_phases[phase_idx]
                phase_targets = target_names_phases[phase_idx]
                phase_baselinks = baselink_names_phases[phase_idx]
                
                # Map baselink to target for this phase
                baselink_to_target = {}
                for robot_name, target_name, baselink_name in zip(phase_robots, phase_targets, phase_baselinks):
                    if baselink_name not in baselink_to_target:
                        baselink_to_target[baselink_name] = target_name
                
                # Create base_poses list in order of robot_base_frame_names for this phase
                base_poses = []
                base_frame_names_ordered = []
                default_far_away_pos = np.array([10.0, 10.0, 0.0])
                default_quat = [1.0, 0.0, 0.0, 0.0]
                
                for baselink_name in robot_base_frame_names:
                    if baselink_name in baselink_to_target:
                        pose = calculate_pose_for_element(baselink_to_target[baselink_name])
                        base_poses.append(pose)
                        base_frame_names_ordered.append(baselink_name)
                    else:
                        base_poses.append((default_far_away_pos.copy(), default_quat))
                        base_frame_names_ordered.append(baselink_name)
                
                # Apply base poses to config for this phase
                if len(base_frame_names_ordered) > 0 and RobotPositionCalculator is not None:
                    RobotPositionCalculator.apply_base_poses(config, base_frame_names_ordered, base_poses)
                
                # Get initial joint state after applying base poses for this phase
                try:
                    phase_initial_state = config.getJointState()
                except:
                    phase_initial_state = np.zeros(config.getJointDimension())
                
                # Set specific joint initial positions
                if len(phase_initial_state) > 20:
                    phase_initial_state[3] = 0
                    phase_initial_state[4] = -np.pi / 2 - np.pi / 4
                    phase_initial_state[5:10] = 0
                    phase_initial_state[10] = -np.pi / 2 + np.pi / 4
                    phase_initial_state[18] = 0
                    phase_initial_state[19] = -np.pi / 2
                    phase_initial_state[20:] = 0
                
                initial_state_path.append(phase_initial_state)
            
            # Convert to numpy array
            initial_state_path = np.array(initial_state_path)
        
        # Create solver
        try:
            solver = MultiPhaseKomoSolver(
                config=config,
                robot_names_phases=robot_names_phases,
                target_names_phases=target_names_phases,
                joint_weight=0.1,
                gripper_weight=5.11,
                position_rel_z_bounds=(0.45, -0.45),
                constraint_eps=1e-3,
                freeze_arm_joints=False,
                collision_weight=1.0,
                pose_rel_weight=0.0,
                enable_constraint_verification=False,
                baselink_names_phases=baselink_names_phases,
                baselink_distance_weight=1.0,
                baselink_distance_target=1.4,
            )
            
            return solver, initial_state_path
        except Exception as e:
            TermPrint.print(f"Failed to create solver for step {element_indices}: {e}", "yellow")
            return None

    def BackwardSearchWithoutMotionPlan(
        self, 
        element_object_list: List[ElementObject],
        config=None,
        robot_gripper_frames: Optional[List[List[str]]] = None,
        robot_base_frame_names: Optional[List[str]] = None,
    ) -> Tuple[List[List[int]], Dict[Tuple[int, ...], List[np.ndarray]]]:
        # -------------------- init --------------------#
        not_visited_index = deque([obj.index for obj in element_object_list])
        visited_index = deque([])
        path_index = deque([])
        blacklist_index = deque([])
        # Store keyframes for each step (keyed by tuple of element indices)
        keyframes_dict = {}

        # -------------------- preprocess --------------------#
        Planner.MultiAssemble(list(not_visited_index), list(visited_index), element_object_list)

        # -------------------- loop --------------------#
        while len(not_visited_index) != 0:
            # -------------------- pop --------------------#
            element_object_index, _ = Planner.FindMax(list(not_visited_index), element_object_list)
            # element_object_index, _ = Planner.FindMin(list(not_visited_index), element_object_list)
            not_visited_index.remove(element_object_index)

            # -------------------- disassemble --------------------#
            assembled_index_list = list(not_visited_index) + list(blacklist_index)
            Planner.Disassemble(element_object_index, assembled_index_list, element_object_list)
            rotate_element_cnt = Planner.ElementsStatusCount(
                assembled_index_list, element_object_list, ElementStatus.rotate
            )

            # Determine step elements based on rotate count
            step_elements = []
            
            if rotate_element_cnt == 0:
                step_elements = [element_object_index]
            elif rotate_element_cnt <= self.robot_num - 1:
                multi_disassemble_index_list = Planner.GetElementIndexBYStatus(
                    list(not_visited_index), element_object_list, ElementStatus.rotate
                )
                step_elements = [element_object_index] + multi_disassemble_index_list
            else:
                # Too many rotate elements, add to blacklist
                Planner.Assemble(
                    element_object_index, list(not_visited_index) + list(blacklist_index), element_object_list
                )
                blacklist_index.append(element_object_index)
                continue
            
            # Check if step can be arranged with available robots
            arrangement_valid = True
            if robot_gripper_frames is not None and robot_base_frame_names is not None:
                robot_names_phases, target_names_phases, baselink_names_phases = self._arrange_frames_for_step(
                    step_elements, robot_gripper_frames, robot_base_frame_names
                )
                
                # If arrangement failed (empty lists), mark step as invalid
                if len(robot_names_phases) == 0 or len(target_names_phases) == 0:
                    arrangement_valid = False
                    TermPrint.print(
                        f"✗ Step {step_elements} cannot be arranged with available robots",
                        "red"
                    )
            
            # If arrangement check failed, mark step as failed and add to blacklist
            if not arrangement_valid:
                for idx in step_elements:
                    Planner.Assemble(idx, list(not_visited_index) + list(blacklist_index), element_object_list)
                    if idx not in blacklist_index:
                        blacklist_index.append(idx)
                TermPrint.print(
                    f"Step {step_elements} marked as failed (arrangement invalid) and added to blacklist",
                    "yellow"
                )
                continue
            
            # Step can be arranged, proceed with solver validation if config is available
            solver_success = True
            if config is not None and robot_gripper_frames is not None and robot_base_frame_names is not None:
                solver_result = self._setup_solver_for_step(
                    step_elements, config, robot_gripper_frames, robot_base_frame_names, element_object_list
                )
                
                if solver_result is not None:
                    solver, initial_state_path = solver_result
                    try:
                        ret, komo = solver.solve(initial_state_path, view=False)
                        
                        if ret.feasible and ret.keyframes is not None:
                            # Solver succeeded - save configuration
                            TermPrint.print(
                                f"✓ Solver success for step {step_elements} (eq={ret.eq:.3e}, ineq={ret.ineq:.3e})",
                                "green"
                            )
                            # Save keyframes for this step (keyed by tuple of element indices for hashability)
                            step_key = tuple(sorted(step_elements))
                            keyframes_dict[step_key] = ret.keyframes
                            solver_success = True
                        else:
                            # Solver failed
                            TermPrint.print(
                                f"✗ Solver failed for step {step_elements} (eq={ret.eq:.3e}, ineq={ret.ineq:.3e})",
                                "red"
                            )
                            solver_success = False
                    except Exception as e:
                        TermPrint.print(f"✗ Solver exception for step {step_elements}: {e}", "red")
                        solver_success = False
                else:
                    # Solver setup failed, but continue with original logic
                    solver_success = True  # Don't fail if solver setup fails
            
            # Handle step based on solver result
            if not solver_success:
                # Mark step as failed: assemble elements and update blacklist
                for idx in step_elements:
                    Planner.Assemble(idx, list(not_visited_index) + list(blacklist_index), element_object_list)
                    if idx not in blacklist_index:
                        blacklist_index.append(idx)
                TermPrint.print(
                    f"Step {step_elements} marked as failed and added to blacklist",
                    "yellow"
                )
                continue
            
            # Solver succeeded or not used - proceed with original logic
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

        # Reverse path_index to get forward order
        reversed_path_index = list(path_index)[::-1]
        
        # Create keyframes list matching the path_index order
        keyframes_list = []
        for step_elements in reversed_path_index:
            step_key = tuple(sorted(step_elements))
            if step_key in keyframes_dict:
                keyframes_list.append(keyframes_dict[step_key])
            else:
                keyframes_list.append(None)

        return reversed_path_index, keyframes_list

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
