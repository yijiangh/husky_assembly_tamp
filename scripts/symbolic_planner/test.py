from collections import deque

def bfs_path(graph, start, target):
    # 创建一个队列来存储要访问的节点，并初始化路径信息
    queue = deque([start])
    # 用于记录访问过的节点，防止重复访问
    visited = set([start])
    # 前驱节点字典，用于记录每个节点是从哪个节点访问过来的
    predecessor = {start: None}
    
    while queue:
        # 弹出队列中的第一个节点
        node = queue.popleft()

        # 如果找到了目标节点，重建路径并返回
        if node == target:
            path = []
            while node is not None:
                path.append(node)
                node = predecessor[node]
            return path[::-1]  # 反转路径，因为我们是从目标到起点进行回溯的

        # 遍历该节点的邻居
        for neighbor in graph[node]:
            if neighbor not in visited:
                # 将未访问的邻居加入队列
                queue.append(neighbor)
                # 标记该邻居为已访问
                visited.add(neighbor)
                # 记录前驱节点
                predecessor[neighbor] = node

    # 如果没有找到目标节点，返回空列表
    return []

# 定义一个图，使用字典的邻接表表示
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

# 从节点 'A' 到节点 'F' 执行广度优先搜索，返回路径
path = bfs_path(graph, 'A', 'F')
print("Path from A to F:", path)
