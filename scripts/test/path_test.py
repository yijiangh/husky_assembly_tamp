import numpy as np
import copy
from scipy.interpolate import splprep, splev # 用于 B 样条

# --- 关键：碰撞检测占位符 ---
# 你需要根据你的实际情况实现这个函数
# 它检查从配置 q1 到 q2 的直线路径是否与 obstacles 碰撞
def is_collision(q1, q2, obstacles):
    """
    Placeholder for collision checking between two configurations.

    Args:
        q1: Starting configuration (e.g., list or numpy array of joint angles or coordinates).
        q2: Ending configuration.
        obstacles: Representation of obstacles in the environment.

    Returns:
        bool: True if collision occurs, False otherwise.
    """
    # --- 在这里替换为你的真实碰撞检测逻辑 ---
    # 例如：
    # 1. 在 q1 和 q2 之间进行插值采样得到中间点
    # 2. 检查每个采样点（或插值段）是否与 obstacles 碰撞
    # print(f"      (Collision Check between {q1} and {q2}) -> False (Placeholder)") # 调试用
    return False # 假设总是无碰撞（仅为示例）
# -----------------------------

def shortcut_path(path, obstacles, iterations=100):
    """
    使用路径缩短（Shortcut Smoothing）优化路径。

    Args:
        path (list): 原始路径，包含一系列配置点 (e.g., [[x1, y1], [x2, y2], ...]).
        obstacles: 环境中的障碍物.
        iterations (int): 尝试缩短的次数.

    Returns:
        list: 优化后的路径.
    """
    if not path or len(path) < 3:
        return path # 路径太短，无法缩短

    print("Starting Path Shortcutting...")
    optimized_path = copy.deepcopy(path) # 操作副本以防万一
    n = len(optimized_path)

    for k in range(iterations):
        # 随机选择两个不相邻的索引
        i = np.random.randint(0, n - 1)
        j = np.random.randint(i + 2, n) # 确保 j > i + 1

        q_i = optimized_path[i]
        q_j = optimized_path[j]

        # 检查 q_i 和 q_j 之间是否存在直接无碰撞路径
        if not is_collision(q_i, q_j, obstacles):
            # 如果无碰撞，移除中间的节点
            # print(f"  Shortcut found between index {i} and {j}. Removing {j - i - 1} points.")
            # 更新路径：保留 i 之前的部分 + q_i + q_j + j 之后的部分
            del optimized_path[i+1:j] # 直接在列表上操作
            n = len(optimized_path) # 更新路径长度
            if n < 3: # 如果路径变得太短，停止优化
                break
        # else:
            # print(f"  No shortcut between index {i} and {j}.")

        if n < 3: break # 提前退出

    print(f"Path Shortcutting finished. Original length: {len(path)}, Optimized length: {len(optimized_path)}")
    return optimized_path

def smooth_path_bspline(path, obstacles, num_points=100, smoothing_factor=0):
    """
    使用 B 样条拟合平滑路径，并进行碰撞检测。

    Args:
        path (list): 路径，包含一系列配置点 (e.g., [[x1, y1], [x2, y2], ...]).
        obstacles: 环境中的障碍物.
        num_points (int): 在生成的样条曲线上采样用于碰撞检测和平滑路径表示的点数。
        smoothing_factor (float): B样条拟合的平滑因子 (s)。
                                    s=0: 样条曲线将通过所有原始点（插值）。
                                    s>0: 样条曲线会更平滑，但可能不通过所有原始点（逼近）。

    Returns:
        list or None: 平滑后的路径（包含 num_points 个配置点），如果样条路径与障碍物碰撞则返回 None。
    """
    if not path or len(path) < 2:
        print("Path too short for B-spline smoothing.")
        return path

    print("Starting B-spline smoothing...")
    path_np = np.array(path)
    dims = path_np.shape[1] # 获取配置空间的维度

    # splprep 需要将坐标按维度分开
    # tck 是包含节点向量、系数和次数的元组
    # u 是每个原始点对应的参数值
    try:
        # k 是样条次数，通常为 3 (cubic)
        tck, u = splprep([path_np[:, d] for d in range(dims)], s=smoothing_factor, k=min(3, len(path)-1))
    except ValueError as e:
        print(f"Error during splprep: {e}. Path might be too simple or co-linear.")
        return path # 无法生成样条，返回原始路径

    # 在参数范围 [0, 1] 内均匀生成 num_points 个参数点
    u_new = np.linspace(u.min(), u.max(), num_points)

    # 使用 splev 计算样条曲线上这些参数点对应的配置
    # der=0 表示计算位置
    new_points_coords = splev(u_new, tck, der=0)

    # 将分开的坐标重新组合成配置点列表
    # new_points_coords 是一个包含每个维度坐标列表的元组，需要转置
    smooth_path = np.vstack(new_points_coords).T.tolist()

    # --- 非常重要：检查生成的样条路径是否碰撞 ---
    print("Checking collisions along the smoothed spline path...")
    for i in range(len(smooth_path) - 1):
        if is_collision(smooth_path[i], smooth_path[i+1], obstacles):
            print(f"Collision detected on smoothed path between point {i} and {i+1}.")
            print("B-spline smoothing failed due to collision.")
            return None # 或者可以返回原始缩短后的路径

    print("B-spline smoothing successful and collision-free.")
    return smooth_path

# === 示例用法 ===
if __name__ == "__main__":
    # 假设 BiRRT 生成的路径 (例如在 2D 空间)
    # 这个路径故意包含一些冗余和不必要的弯曲
    dummy_birrt_path = [
        [0, 0], [1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [4, 3],
        [4, 2], [4, 1], [5, 1], [6, 2], [7, 3], [8, 4], [9, 5], [10, 5]
    ]

    # 假设的环境障碍物 (这里只是示意，实际应为具体几何对象或数据)
    dummy_obstacles = ["obstacle1_data", "obstacle2_data"]

    print("Original Path:", dummy_birrt_path)
    print("-" * 30)

    # 1. 使用路径缩短进行优化
    shortcutted_path = shortcut_path(dummy_birrt_path, dummy_obstacles, iterations=200)
    print("Shortcutted Path:", shortcutted_path)
    print("-" * 30)

    # 2. 对缩短后的路径进行 B 样条平滑
    # 注意：如果 is_collision 检测到碰撞，这里会返回 None
    # smoothing_factor=0 会尝试通过所有点，可能会不够平滑
    # 增大 smoothing_factor 会更平滑，但可能偏离原始安全点，更需要仔细的碰撞检测
    smoothed_path = smooth_path_bspline(shortcutted_path, dummy_obstacles, num_points=50, smoothing_factor=0.1)

    if smoothed_path:
        print("Smoothed Path (B-spline):", smoothed_path)
        # 你可以在这里添加代码来可视化路径进行比较
        try:
            import matplotlib.pyplot as plt
            path_orig_np = np.array(dummy_birrt_path)
            path_short_np = np.array(shortcutted_path)
            path_smooth_np = np.array(smoothed_path)

            plt.figure()
            plt.plot(path_orig_np[:, 0], path_orig_np[:, 1], 'ro-', label='Original BiRRT')
            plt.plot(path_short_np[:, 0], path_short_np[:, 1], 'bo-', label='Shorted')
            plt.plot(path_smooth_np[:, 0], path_smooth_np[:, 1], 'g-', label='Smoothed (Spline)')
            plt.title('Path Optimization')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            plt.show()
        except ImportError:
            print("\nMatplotlib not found. Skipping visualization.")
            print("To visualize, install matplotlib: pip install matplotlib")

    else:
        print("Could not generate a collision-free smoothed path.")

    print("-" * 30)
    print("Note: For spline smoothing, ensure your 'is_collision' function is robust!")
    print("Checking only the segments between sampled points on the spline is an approximation.")
    print("Consider more rigorous continuous collision detection if necessary.")
