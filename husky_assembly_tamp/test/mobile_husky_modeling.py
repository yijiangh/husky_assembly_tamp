import time

import numpy as np
import pybullet_planning as pp
import robotic as ry

print("The path where model files are pre-installed:\n", ry.raiPath(""))
# you could overwrite this with:
# ry.setRaiPath('/home/mtoussai/git/rai-robotModels/')

# C = ry.Config()
# C.addFile(ry.raiPath("panda/panda.g")).setPosition([0.0, 0.0, 0.0]).setQuaternion([1, 0, 0, 0])
# C.addFile(ry.raiPath("robotiq/robotiq.g")).setParent(C.getFrame("panda_joint7")).setRelativePosition([0.0, 0.0, 0.15])

# C.addFile(ry.raiPath("panda/panda.g"), "r_").setPosition([0.5, 0.0, 0.0]).setQuaternion([1, 0, 0, 0])

# C.addFile(ry.raiPath("pr2/pr2.g")).setPosition([1.0, 0.0, 0.0])

# C.addFile(ry.raiPath("ranger/ranger_clean.g")).setPosition([2.0, 0.0, 0.0]).setQuaternion([1, 0, 0, 0])

# C.addFile(ry.raiPath("ranger/ranger.g")).setPosition([2.5, 0.0, 0.0]).setQuaternion([1, 0, 0, 0])

# C.addFile(ry.raiPath("ur10/ur10.g")).setPosition([3.0, 0.0, 0.0]).setQuaternion([1, 0, 0, 0])

# C.addFile("/home/jeong/summer_research/husky_assembly/ext/husky-assembly-teleop/data/husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e_standalone.g").setPosition([0.0, 0.0, 0.0]).setQuaternion([1, 0, 0, 0])

# C.addFile("/home/jeong/summer_research/husky_assembly/ext/husky-assembly-teleop/data/husky_urdf/mt_husky_moveit_config/urdf/husky_ur5_e_standalone.g").setPosition([2.0, 0.0, 0.0]).setQuaternion([1, 0, 0, 0])

# print("Joint state: ", C.getJointState())

# C.setJointState([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# C.view()

# pp.wait_for_user()

# while True:
#     pass
