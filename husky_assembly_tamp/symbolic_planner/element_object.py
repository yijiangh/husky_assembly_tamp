import operator
from collections import deque
from copy import deepcopy
from typing import List, Tuple

import pybullet_planning as pp
from husky_assembly_tamp.symbolic_planner.element_status import ElementStatus


class ElementObject(object):
    def __init__(
        self,
        index: int,
        body: int,
        init_pose: Tuple[Tuple[float], Tuple[float]],
        goal_pose: Tuple[Tuple[float], Tuple[float]],
        vertices: List[List[float]],
        coupled_elements: List = [],
        checker: str = "default",
        is_grounded=False,
    ) -> None:

        from symbolic_planner.status_checker import (
            BasicChecker,
            DefaultChecker,
            GroundedChecker,
            TwoFixConstrainChecker,
            AlgebraicChecker,
        )

        self.index = index
        self.body = body
        self.cur_pose = init_pose
        self.goal_pose = goal_pose
        self.vertices = vertices
        self.coupled_elements = coupled_elements  # bars that should be connected with self
        self.is_grounded = is_grounded

        self.heuristic_value = 0
        # self.assigned_couplers = []  # installed couplers
        self.assembled_elements = []  # connected elements index
        self.status = ElementStatus.unassembled
        self.checker = checker
        if checker == "two-fix":
            self.status_checker = TwoFixConstrainChecker()
        elif checker == "algebraic":
            self.status_checker = AlgebraicChecker()
        else:
            self.status_checker = DefaultChecker()

    def __str__(self) -> str:
        return f"index: {self.index}, status: {self.status.name}, assembled: {self.assembled_elements}"

    def __repr__(self) -> str:
        # return f"index: {self.index}"
        return f"index: {self.index}, status: {self.status.name}"

    def Assemble(self, assembled_list: List[int]):
        self.status = ElementStatus.float
        self.UpdateConstrain(assembled_list)

    def Disassemble(self):
        self.status = ElementStatus.unassembled
        self.assembled_elements = []

    def UpdateConstrain(self, assembled_list: List[int]):
        if self.status != ElementStatus.unassembled:
            for assembled_index in assembled_list:
                if assembled_index in self.coupled_elements:
                    self.AddAssembleElement(assembled_index)

    def UpdateStatus(self, assembled: List[int], element_object_list: List["ElementObject"]):
        if self.checker == "algebraic":
            self.status = self.status_checker.Check(self.index, assembled, element_object_list)
        else:
            self.status = self.status_checker.Check(self.index, element_object_list)
        # print(f"index {self.index}: ", self.status.name)

    def AddAssembleElement(self, index: int):
        if index not in self.assembled_elements:
            self.assembled_elements.append(index)

    def SetHeuristicValue(self, value: float):
        self.heuristic_value = value

    @staticmethod
    def GetCoupledElements(element_index: int, contact_id_pairs: List[List[int]]) -> List[int]:
        contact_id_pairs = deepcopy(contact_id_pairs)
        coupled_elements = []
        for contact_id_pair in contact_id_pairs:
            if element_index in contact_id_pair:
                contact_id_pair.remove(element_index)
                coupled_elements.append(contact_id_pair[0])
        return coupled_elements

    @staticmethod
    def GetCouplers(assembled: List[int], element_object_list: List["ElementObject"]) -> List[Tuple[int]]:
        """
        Get couplers of assembled substructure.

        Params:
            assembled ([int]): indices of assembled elements
            element_object_list ([ElementObject]): list of ElementObject

        Returns:
            List[Tuple[int]]: [(index_1, index_2), ...] couplers
        """
        couplers = []
        for index in assembled:
            for next_index in element_object_list[index].assembled_elements:
                coupler = [index, next_index]
                coupler.sort()
                coupler = tuple(coupler)
                if coupler not in couplers:
                    couplers.append(coupler)

        return couplers
