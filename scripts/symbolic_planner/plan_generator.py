import argparse
import os

import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))

# -------------------- self-defined modules --------------------#
import load_multi_tangent
import pybullet_planning as pp
from collision import Element, create_couplers, init_pb
from element_object import ElementObject, ElementStatus
from multi_tangent.collision import create_collision_bodies
from multi_tangent.convert import flatten_list
from parse import parse_mt_geometric
from planner import Planner


class UnionFind:
    def __init__(self, n):
        # 初始化父节点数组和秩(rank)数组
        self.parent = list(range(n))  # 每个节点的初始父节点为它自己
        self.rank = [0] * n  # 每个节点的初始秩为0

    def find(self, u):
        # 路径压缩：递归地查找u的根节点并压缩路径
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # 找到根节点并直接连接
        return self.parent[u]

    def union(self, u, v):
        # 查找两个节点u和v的根节点
        root_u = self.find(u)
        root_v = self.find(v)

        # 如果两个根节点不相同，合并两个集合
        if root_u != root_v:
            # 根据rank合并小树到大树，保证效率
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1
            return False  # 没有形成环
        return True  # 形成环


def count_cycles(element_objects):
    # 假设 element_objects 是一个包含所有刚性杆对象的列表
    n = len(element_objects)  # 节点的数量
    uf = UnionFind(n)  # 初始化并查集
    cycle_count = 0  # 用于记录回路的数量

    # 遍历每个刚性杆和它连接的其他刚性杆
    for element in element_objects:
        for neighbor in element.assembled_elements:
            # 只处理 element.index < neighbor 的情况，避免重复计算
            if element.index < neighbor:
                if uf.union(element.index, neighbor):
                    cycle_count += 1  # 如果union返回True，表示找到了一个回路

    return cycle_count


if __name__ == "__main__":
    with pp.HideOutput():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--mt_file_name",
            # default="one_tet_MT_layer_0_contact.json",
            default="tower_integral_one_len_MT_layer_0.json",
            help='The name of the multi tangent file to solve (json file\'s name, e.g. "tower_integral_one_len_MT_layer_0.json")',
        )
        args = parser.parse_args()

        # -------------------- Load process file --------------------#
        mt_file_name = args.mt_file_name
        line_pt_pairs, contact_id_pairs, bar_radius = parse_mt_geometric(mt_file_name)
        line_pt_pairs: list[list[list]]  # bar list
        contact_id_pairs: list[list]  # contact pairs
        bar_radius: float
        line_pts_flattened: list[np.ndarray] = flatten_list(np.array(line_pt_pairs))  # numpy points list
        vertices: list[list] = flatten_list(line_pt_pairs)  # points list

        # -------------------- Eliminate Z-axis deviation --------------------#
        min_z = np.min(line_pts_flattened, axis=0)[2]
        line_pts_flattened = [np.array([0, 0, -min_z]) + point for point in line_pts_flattened]

        radius_per_edge = [bar_radius] * int(len(line_pts_flattened) / 2)

        # -------------------- Init --------------------#
        init_pb()
        with pp.LockRenderer():
            element_bodies = create_collision_bodies(line_pts_flattened, radius_per_edge, viewer=True)
            half_coupler_from_contact_pair = create_couplers(line_pts_flattened, contact_id_pairs)
            for i, e in enumerate(element_bodies):
                pp.add_text(str(i), pp.get_point(e))
                pp.set_color(e, (1, 0, 0, 0.1))
        element_from_index = {
            i: Element(i, e, pp.get_pose(e), pp.get_pose(e), [line_pts_flattened[2 * i], line_pts_flattened[2 * i + 1]])
            for i, e in enumerate(element_bodies)
        }

        grounded_elements_index = [0, 1, 4, 19]  # tower_integral_one_len_MT_layer_0
        # grounded_elements_index = [0]  # one_tet_MT_layer_0_contact
        # for index in grounded_elements_index:
        #     pp.set_color(element_from_index[index].body, pp.BLACK)

        # -------------------- Plan --------------------#
        planner = Planner(robot_num=2)
        path_index = planner.Plan(element_from_index, contact_id_pairs, grounded_elements_index)
        element_object_list = Planner.GetElementObjects(element_from_index, contact_id_pairs, grounded_elements_index)
        # path_index = [[0], [1], [4], [19], [6], [7], [14], [8], [12], [13]]
        # path_index = [[0], [1], [2], [3], [4], [5]]

        # -------------------- Visualization --------------------#
        assembled = []
        for step_num, index_list in enumerate(path_index):
            for index in index_list:
                element_object_list[index].Assemble(assembled)
                assembled.append(index)
                Planner.UpdateElements(assembled, element_object_list)
            with pp.LockRenderer():
                for element_obj in element_object_list:
                    if element_obj.status == ElementStatus.float:
                        pp.set_color(element_obj.body, pp.YELLOW)
                    elif element_obj.status == ElementStatus.rotate:
                        pp.set_color(element_obj.body, pp.GREEN)
                    elif element_obj.status == ElementStatus.fixed:
                        pp.set_color(element_obj.body, pp.RED)
                    else:
                        pp.set_color(element_obj.body, (1, 0, 0, 0.1))
                for index in index_list:
                    pp.set_color(element_object_list[index].body, pp.GREY)
            pp.wait_for_user(
                f"step: {step_num+1}/{len(path_index)} ,cur update index: {index_list}/{len(path_index)-1}"
            )
