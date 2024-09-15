from collision import Element
from element_object import ElementObject
import operator


def Plan(element_from_index: dict[Element], contact_id_pairs: list[list], grounded_elements_index: list) -> list:
    # TODO finish planner

    # -------------------- Generate element objects --------------------#
    element_object_list = GetElementObjects(element_from_index, contact_id_pairs, grounded_elements_index)

    return list(element_from_index.keys())


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


def UpdateElements(assembled_list: list, element_object_list: list[ElementObject]):
    for element_object in element_object_list:
        element_object.UpdateConstrain(assembled_list)
    for element_object in element_object_list:
        element_object.UpdateStatus(element_object_list)
