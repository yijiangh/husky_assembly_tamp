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


# 假设element_object的结构如下：
class ElementObject:
    def __init__(self, index, assembled_elements):
        self.index = index  # 当前杆的ID
        self.assembled_elements = assembled_elements  # 与当前杆连接的其他杆的ID列表


# 示例用法
if __name__ == "__main__":
    # 创建结构体，每个刚性杆和它连接的杆的列表
    element_objects = [
        ElementObject(0, [1, 2]),  # 杆0连接杆1和杆2
        ElementObject(1, [0, 2]),  # 杆1连接杆0和杆2
        ElementObject(2, [0, 1]),  # 杆2连接杆0和杆1（形成一个三角形回路）
        ElementObject(3, [4]),     # 杆3连接杆4
        ElementObject(4, [3]),     # 杆4连接杆3（不形成回路）
    ]

    # 计算回路的个数
    cycle_count = count_cycles(element_objects)
    print("回路的个数:", cycle_count)  # 输出应为1
