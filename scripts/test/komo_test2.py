import time

import numpy as np
import pybullet_planning as pp
import robotic as ry

print("The path where model files are pre-installed:\n", ry.raiPath(""))
# you could overwrite this with:
# ry.setRaiPath('/home/mtoussai/git/rai-robotModels/')

C = ry.Config()
C.addFile(ry.raiPath("panda/panda.g")).setPosition([1.0, 0.0, 0.0]).setQuaternion([1, 0, 0, 1])
C.addFile(ry.raiPath("panda/panda.g"), "r_").setPosition([1.5, 0.0, 0.0]).setQuaternion([1, 0, 0, 1])
C.addFile(ry.raiPath("pr2/pr2.g")).setPosition([-1.0, 0.0, 0.0])
# C.addFile(ry.raiPath("baxter/baxter.g")).setPosition([0, 0.0, 1.0]).setQuaternion([1, 0, 0, 1])
C.addFile(ry.raiPath("robotiq/robotiq.g")).setParent(C.getFrame("panda_joint7")).setRelativePosition([0.0, 0.0, 0.15])
C.addFile(ry.raiPath("ur10/ur10_clean.g")).setParent(C.getFrame("panda_joint7")).setRelativePosition([0.0, 0.0, 0.15])
C.view()

pp.wait_for_user()

# while True:
#     pass