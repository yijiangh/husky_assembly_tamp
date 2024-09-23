import numpy as np
from element_object import ElementObject, ElementStatus, GroundedChecker


class BasicHeuristic(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def Update(element_object_list: list[ElementObject]):
        for element_object in element_object_list:
            heuristic_val = element_object.index
            element_object.SetHeuristicValue(heuristic_val)


class GroundedChainHeuristic(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def Update(element_object_list: list[ElementObject]):
        for element_object in element_object_list:
            grounded_path = GroundedChecker.GetTrueGroundPath(element_object.index, element_object_list)
            if len(grounded_path) != 0:
                heuristic_val = len(grounded_path)
            else:
                heuristic_val = np.inf
            element_object.SetHeuristicValue(heuristic_val)


class GroundedHeightHeuristic(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def Update(element_object_list: list[ElementObject]):
        for element_object in element_object_list:
            vertices = np.array(element_object.vertices)
            heuristic_val = vertices.mean(axis=0)[-1]  # height
            element_object.SetHeuristicValue(heuristic_val)


class CenterDistanceHeuristic(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def Update(element_object_list: list[ElementObject]):
        center = CenterDistanceHeuristic.CalculateCenter(element_object_list)
        for element_object in element_object_list:
            vertices = np.array(element_object.vertices)
            heuristic_val = CenterDistanceHeuristic.Point2SegmentDist(center, vertices[0], vertices[1])
            element_object.SetHeuristicValue(heuristic_val)

    @staticmethod
    def CalculateCenter(element_object_list: list[ElementObject]) -> np.ndarray:
        all_vertices = []
        for element_object in element_object_list:
            all_vertices.extend(element_object.vertices)
        center = np.array(all_vertices).mean(axis=0)
        return center

    @staticmethod
    def Point2SegmentDist(P, A, B):
        P = np.array(P)
        A = np.array(A)
        B = np.array(B)

        AB = B - A
        AP = P - A

        t = np.dot(AP, AB) / np.dot(AB, AB)

        if t < 0.0:
            closest_point = A
        elif t > 1.0:
            closest_point = B
        else:
            closest_point = A + t * AB

        distance = np.linalg.norm(P - closest_point)
        return distance
