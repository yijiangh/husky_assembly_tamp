import numpy as np
from scipy.spatial.transform import Rotation as R

# 定义向量 p1 和旋转向量 rot_vector
p1 = np.array([1, 1, 1])
rot_vector = np.array([0, 0, np.pi/2])  # 旋转向量（Rodrigues 向量）

# 创建旋转对象
rotation = R.from_rotvec(rot_vector)

# 应用旋转
p1_rotated = rotation.apply(p1)

print(p1_rotated)
