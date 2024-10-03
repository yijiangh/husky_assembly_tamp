class TreeNode:
    _instances = {}

    def __new__(cls, identifier1: list[int], data_list_1: list[int], data_list_2: list[int], identifier4: list[int]):
        # 将 identifier1 和 identifier4 排序并组合成元组，以确保唯一性
        sorted_id = (tuple(sorted(identifier1)), tuple(sorted(identifier4)))
        if sorted_id not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[sorted_id] = instance
        return cls._instances[sorted_id]

    def __init__(self, identifier1: list[int], data_list_1: list[int], data_list_2: list[int], identifier4: list[int]):
        # 防止重复初始化
        if not hasattr(self, 'initialized'):
            self.identifier1 = identifier1  # 用于唯一标识节点的列表1
            self.data_list_1 = data_list_1   # 其他数据列表1
            self.data_list_2 = data_list_2   # 其他数据列表2
            self.identifier4 = identifier4    # 用于唯一标识节点的列表4
            self.father = None
            self.children = []
            self.initialized = True  # 标记已初始化

    def set_father(self, father: 'TreeNode'):
        self.father = father
        father.children.append(self)

    def __repr__(self):
        return (f"TreeNode(identifier1={self.identifier1}, "
                f"data_list_1={self.data_list_1}, data_list_2={self.data_list_2}, "
                f"identifier4={self.identifier4})")


class ExtendedTreeNode(TreeNode, object):
    _instances_extended = {}

    def __new__(cls, identifier1: list[int], data_list_1: list[int], data_list_2: list[int], identifier4: list[int], identifier5: list[int]):
        # 将 identifier1、identifier4 和 identifier5 排序并组合成元组，以确保唯一性
        sorted_id = (tuple(sorted(identifier1)), tuple(sorted(identifier4)), tuple(sorted(identifier5)))
        if sorted_id not in cls._instances_extended:
            instance = super().__new__(cls)
            cls._instances_extended[sorted_id] = instance
        return cls._instances_extended[sorted_id]

    def __init__(self, identifier1: list[int], data_list_1: list[int], data_list_2: list[int], identifier4: list[int], identifier5: list[int]):
        # 调用父类构造函数
        super().__init__(identifier1, data_list_1, data_list_2, identifier4)
        # 只在第一次初始化时设置 identifier5
        if not hasattr(self, 'initialized_extended'):
            self.identifier5 = identifier5  # 新的标识符
            self.initialized_extended = True  # 标记已初始化

    def __repr__(self):
        return (f"ExtendedTreeNode(identifier1={self.identifier1}, "
                f"data_list_1={self.data_list_1}, data_list_2={self.data_list_2}, "
                f"identifier4={self.identifier4}, identifier5={self.identifier5})")


# 创建两个 ExtendedTreeNode 实例
extended_node_1 = ExtendedTreeNode([1, 2, 3], [10, 20], [30, 40], [7, 8, 9], [100, 200])
extended_node_2 = ExtendedTreeNode([1, 2, 3], [50, 60], [70, 80], [7, 8, 9], [300, 400])

# 显示 ExtendedTreeNode 的信息
print(extended_node_1)
print(extended_node_2)

# 检查是否指向同一实例
print(extended_node_1 is extended_node_2)  # 输出: False
