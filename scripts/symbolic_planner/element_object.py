import operator
from collections import deque
from copy import deepcopy
from enum import Enum
from typing import Tuple

import pybullet_planning as pp


class ElementStatus(Enum):
    unassembled = 0
    float = 1
    rotate = 2
    fixed = 3

# TODO consider hold

class ElementObject(object):
    def __init__(
        self,
        index: int,
        body: int,
        init_pose: Tuple[Tuple, Tuple],
        goal_pose: Tuple[Tuple, Tuple],
        vertices: list[list, list],
        coupled_elements: list = [],
        is_grounded=False,
    ) -> None:
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
        self.status_checker = TwoFixConstrainChecker()

    def __str__(self) -> str:
        return f"index: {self.index}, status: {self.status.name}, assembled: {self.assembled_elements}"

    def __repr__(self) -> str:
        # return f"index: {self.index}"
        return f"index: {self.index}, status: {self.status.name}"

    def Assemble(self, assembled_list: list):
        self.status = ElementStatus.float
        self.UpdateConstrain(assembled_list)

    def Disassemble(self):
        self.status = ElementStatus.unassembled
        self.assembled_elements = []

    def UpdateConstrain(self, assembled_list: list):
        if self.status != ElementStatus.unassembled:
            for assembled_index in assembled_list:
                if assembled_index in self.coupled_elements:
                    self.AddAssembleElement(assembled_index)

    def UpdateStatus(self, element_object_list: list):
        self.status = self.status_checker.Check(self.index, element_object_list)
        # print(f"index {self.index}: ", self.status.name)

    def AddAssembleElement(self, index: int):
        if index not in self.assembled_elements:
            self.assembled_elements.append(index)

    def SetHeuristicValue(self, value: float):
        self.heuristic_value = value

    @staticmethod
    def GetCoupledElements(element_index: int, contact_id_pairs: list[list]) -> list:
        contact_id_pairs = deepcopy(contact_id_pairs)
        coupled_elements = []
        for contact_id_pair in contact_id_pairs:
            if element_index in contact_id_pair:
                contact_id_pair.remove(element_index)
                coupled_elements.append(contact_id_pair[0])
        return coupled_elements


class BasicChecker(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def Check(index: int, element_object_list: list[ElementObject]) -> ElementStatus:
        # -------------------- first judge assemble state --------------------#
        if element_object_list[index].status == ElementStatus.unassembled:
            return ElementStatus.unassembled

        # -------------------- second judge ground state --------------------#
        if element_object_list[index].is_grounded:
            return ElementStatus.fixed

        if len(element_object_list[index].assembled_elements) == 0:
            return ElementStatus.float
        elif len(element_object_list[index].assembled_elements) == 1:
            return ElementStatus.rotate
        else:
            return ElementStatus.fixed


class GroundedChecker(object):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def Check(index: int, element_object_list: list[ElementObject]) -> ElementStatus:

        basic_status = BasicChecker.Check(index, element_object_list)

        if basic_status == ElementStatus.fixed or basic_status == ElementStatus.unassembled:
            return basic_status

        queue = deque([index])
        visited = set([index])
        predecessor = {index: None}

        # path = []
        is_grounded = False
        while queue:
            node_index = queue.popleft()
            if element_object_list[node_index].is_grounded:
                # while node_index is not None:
                #     path.append(node_index)
                #     node_index = predecessor[node_index]
                is_grounded = True
                break
            for neighbor_index in element_object_list[node_index].assembled_elements:
                if neighbor_index not in visited:
                    queue.append(neighbor_index)
                    visited.add(neighbor_index)
                    predecessor[neighbor_index] = node_index
        # if len(path) == 0:
        #     is_grounded = False
        # else:
        #     is_grounded = True

        if is_grounded:
            return basic_status
        else:
            return ElementStatus.float

    @staticmethod
    def CheckGroundNum(index: int, element_object_list: list[ElementObject]) -> int:
        queue = deque([index])
        visited = set([index])
        ground_num = 0
        while queue:
            node_index = queue.popleft()
            if element_object_list[node_index].is_grounded:
                ground_num += 1
            for neighbor_index in element_object_list[node_index].assembled_elements:
                if neighbor_index not in visited:
                    queue.append(neighbor_index)
                    visited.add(neighbor_index)
        return ground_num

    @staticmethod
    def GetGroundPath(index: int, element_object_list: list[ElementObject]) -> list[ElementObject]:
        queue = deque([index])
        visited = set([index])
        predecessor = {index: None}

        path = []
        while queue:
            node_index = queue.popleft()
            if element_object_list[node_index].is_grounded:
                while node_index is not None:
                    path.append(node_index)
                    node_index = predecessor[node_index]
                return path[::-1]
            for neighbor_index in element_object_list[node_index].assembled_elements:
                if neighbor_index not in visited:
                    queue.append(neighbor_index)
                    visited.add(neighbor_index)
                    predecessor[neighbor_index] = node_index
        return []

    @staticmethod
    def GetTrueGroundPath(index: int, element_object_list: list[ElementObject]) -> list[ElementObject]:
        queue = deque([index])
        visited = set([index])
        predecessor = {index: None}

        path = []
        while queue:
            node_index = queue.popleft()
            if element_object_list[node_index].is_grounded:
                while node_index is not None:
                    path.append(node_index)
                    node_index = predecessor[node_index]
                return path[::-1]
            for neighbor_index in element_object_list[node_index].coupled_elements:
                if neighbor_index not in visited:
                    queue.append(neighbor_index)
                    visited.add(neighbor_index)
                    predecessor[neighbor_index] = node_index
        return []


class TwoFixConstrainChecker(object):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def Check(index: int, element_object_list: list[ElementObject], visited=[]) -> ElementStatus:
        # -------------------- grounded: only fixed is cannot be determined --------------------#
        grounded_status = GroundedChecker.Check(index, element_object_list)
        if (
            grounded_status == ElementStatus.unassembled
            or grounded_status == ElementStatus.float
            or grounded_status == ElementStatus.rotate
        ):
            return grounded_status

        # -------------------- directly grounded --------------------#
        if element_object_list[index].is_grounded:
            return ElementStatus.fixed

        # -------------------- judge the fixed constrain num --------------------#
        fix_constrain_num = 0
        element_object = element_object_list[index]
        for neighbor_index in element_object.assembled_elements:
            if neighbor_index in visited:
                continue
            neighbor_status = TwoFixConstrainChecker.Check(neighbor_index, element_object_list, visited + [index])
            if neighbor_status == ElementStatus.fixed:
                fix_constrain_num += 1
        
        if fix_constrain_num >= 2:
            return ElementStatus.fixed
        else:
            return ElementStatus.rotate
