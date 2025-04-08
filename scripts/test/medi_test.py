import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

def generate_shape(size):
    """生成一个简单的形状（例如，矩形和圆形的组合）"""
    shape = np.zeros(size, dtype=np.uint8)
    
    # 创建矩形
    shape[20:80, 20:80] = 1
    
    # 创建圆形
    y, x = np.ogrid[:size[0], :size[1]]
    mask = (x - 50)**2 + (y - 50)**2 <= 15**2
    shape[mask] = 1
    
    return shape

def medial_axis_thinning(shape):
    """使用收缩法计算中值骨架"""
    # 复制原始形状
    skeleton = shape.copy()
    
    # 迭代收缩
    while True:
        # 进行一次腐蚀操作
        eroded = binary_erosion(skeleton)
        
        # 计算骨架变化
        skeleton_change = skeleton ^ eroded & skeleton  # 使用异或和与操作，确保只保留被移除的像素
        
        # 更新骨架
        skeleton = eroded
        
        # 如果没有变化，停止迭代
        if not skeleton_change.any():
            break
            
    return skeleton

def visualize_shape_and_skeleton(shape, skeleton):
    """可视化原始形状和中值骨架"""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Shape")
    plt.imshow(shape, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Medial Axis Skeleton")
    plt.imshow(skeleton, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 生成形状
    shape = generate_shape((100, 100))
    
    # 计算中值骨架
    skeleton = medial_axis_thinning(shape)
    
    # 可视化结果
    visualize_shape_and_skeleton(shape, skeleton)
