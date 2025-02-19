import pybullet as p
import pybullet_data
import time

# 连接到GUI模式
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 加载平面和一个物体（例如r2d2）
planeId = p.loadURDF("plane.urdf")
objId = p.loadURDF("r2d2.urdf", [0, 0, 0.2])

# 设置仿真时间步长
timeStep = 1.0 / 240.0
p.setTimeStep(timeStep)

# 创建一个用于“影子”的视觉形状（小球，红色，半径0.05）
shadow_visual_shape = p.createVisualShape(
    shapeType=p.GEOM_SPHERE,
    radius=0.05,
    rgbaColor=[1, 0, 0, 1]
)

# 影子生成的时间间隔（单位秒）
shadow_interval = 0.5  
next_shadow_time = time.time() + shadow_interval

# 主仿真循环
while p.isConnected():
    p.stepSimulation()
    time.sleep(timeStep)
    
    current_time = time.time()
    if current_time >= next_shadow_time:
        # 获取物体当前位置
        pos, orn = p.getBasePositionAndOrientation(objId)
        # 在当前位置创建一个影子（静态小球，多体的质量设为0，不受动力学影响）
        p.createMultiBody(
            baseMass=0,  # 静态物体
            baseVisualShapeIndex=shadow_visual_shape,
            basePosition=pos
        )
        # 更新下次生成影子的时间
        next_shadow_time += shadow_interval
