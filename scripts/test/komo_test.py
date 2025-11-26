import time

import numpy as np
from numpy.core.shape_base import block
import robotic

import pybullet_planning as pp

# C = robotic.Config()
# C.addFile(robotic.raiPath("scenarios/pandaSingle.g"))
# C.view()

# urdf_string = C.writeURDF()
# urdf_file_path = "pandaSingle.urdf"
# with open(urdf_file_path, 'w') as f:
#     f.write(urdf_string)
# print(f"URDF saved to {urdf_file_path}")

# exit()

# C.addFrame("box").setPosition([-0.25, 0.1, 1.0]).setShape(robotic.ST.ssBox, size=[0.06, 0.06, 0.06, 0.005]).setColor([1, 0.5, 0]).setContact(1)
# C.view()

# # pp.wait_for_user()

# qHome = C.getJointState()

# komo = robotic.KOMO(C, 1, 1, 0, False)
# komo.addObjective(times=[], feature=robotic.FS.jointState, frames=[], type=robotic.OT.sos, scale=[1e-1], target=qHome)
# komo.addObjective([], robotic.FS.positionDiff, ["l_gripper", "box"], robotic.OT.eq, [1e1])

# ret = robotic.NLP_Solver(komo.nlp(), verbose=4).solve()
# print(ret)

# komo.view(False, "IK solution")

# q = komo.getPath()
# print(type(q), len(q))

# del komo  # also closes komo view
# C.setJointState(q[0])
# C.view()

# pp.wait_for_user()

# # komo = robotic.KOMO(C, 1,1,0, True)
# # komo.addObjective([], robotic.FS.jointState, [], robotic.OT.sos, [1e-1], qHome)
# # komo.addObjective([], robotic.FS.accumulatedCollisions, [], robotic.OT.eq)
# # komo.addObjective([], robotic.FS.jointLimits, [], robotic.OT.ineq)
# # komo.addObjective([], robotic.FS.positionDiff, ['l_gripper', 'box'], robotic.OT.eq, [1e1])
# # komo.addObjective([], robotic.FS.scalarProductXX, ['l_gripper', 'box'], robotic.OT.eq, [1e1], [0])
# # komo.addObjective([], robotic.FS.scalarProductXZ, ['l_gripper', 'box'], robotic.OT.eq, [1e1], [0])
# # komo.addObjective([], robotic.FS.distance, ['l_palm', 'box'], robotic.OT.ineq, [1e1])

# # ret = robotic.NLP_Solver(komo.nlp(), verbose=0 ) .solve()
# # print(ret)
# # if ret.feasible:
# #     print('-- Always check feasibility flag of NLP solver return')
# # else:
# #     print('-- THIS IS INFEASIBLE!')
    
# # q = komo.getPath()
# # C.setJointState(q[0])
# # C.view(False, "IK solution")

# # box = C.getFrame('box')
# # box.setPosition([-.25,.1,1.])
# # p0 = box.getPosition() # memory the start box position

# # for t in range(10):
# #     box.setPosition(p0 + .2 * np.random.randn(3)) # randomize box position
# #     komo.updateRootObjects(C) # only works for root objects (the 'box' is one)
# #     ret = robotic.NLP_Solver(komo.nlp(), verbose=0 ) .solve()
# #     print(ret)
# #     q = komo.getPath()
# #     C.setJointState(q[0])
# #     C.view(False, 'IK solution - ' + ('*** INFEASIBLE ***' if not ret.feasible else 'feasible'))
# #     time.sleep(1.)
    
# pp.wait_for_user()

# # del komo
# komo = []
# for k in range(3):
#     komo.append(robotic.KOMO(C, 1,1,0, True))
#     komo[k].addObjective([], robotic.FS.jointState, [], robotic.OT.sos, [1e-1], qHome)
#     komo[k].addObjective([], robotic.FS.accumulatedCollisions, [], robotic.OT.eq)
#     komo[k].addObjective([], robotic.FS.jointLimits, [], robotic.OT.ineq)
#     komo[k].addObjective([], robotic.FS.positionDiff, ['l_gripper', 'box'], robotic.OT.eq, [1e1])
#     komo[k].addObjective([], robotic.FS.distance, ['l_palm', 'box'], robotic.OT.ineq, [1e1])

# komo[0].addObjective([], robotic.FS.scalarProductXY, ['l_gripper', 'box'], robotic.OT.eq, [1e1], [0])
# komo[0].addObjective([], robotic.FS.scalarProductXZ, ['l_gripper', 'box'], robotic.OT.eq, [1e1], [0])

# komo[1].addObjective([], robotic.FS.scalarProductXX, ['l_gripper', 'box'], robotic.OT.eq, [1e1], [0])
# komo[1].addObjective([], robotic.FS.scalarProductXZ, ['l_gripper', 'box'], robotic.OT.eq, [1e1], [0])

# komo[2].addObjective([], robotic.FS.scalarProductXX, ['l_gripper', 'box'], robotic.OT.eq, [1e1], [0])
# komo[2].addObjective([], robotic.FS.scalarProductXY, ['l_gripper', 'box'], robotic.OT.eq, [1e1], [0])

# box = C.getFrame('box')
# box.setPosition([-.25,.1,1.])
# p0 = box.getPosition() # memory the start box position

# for t in range(10):
#     box.setPosition(p0 + .2 * np.random.randn(3))
#     box.setQuaternion(np.random.randn(4)) # also set random orientation (quaternions get internally normalized)

#     score = []
#     for k in range(3):
#         komo[k].updateRootObjects(C)
#         ret = robotic.NLP_Solver(komo[k].nlp(), verbose=0 ) .solve()
#         score.append( 100.*(ret.eq+ret.ineq) + ret.sos )

#     k = np.argmin(score)
#     C.setJointState(komo[k].getPath()[0])
#     C.view(False, f'IK solution {k} - ' + ('*** INFEASIBLE ***' if not ret.feasible else 'feasible'))
#     pp.wait_for_user()
    
# pp.wait_for_user()

# **************************************************************************
# Path
# **************************************************************************

import robotic as ry

C = ry.Config()
# C.addFile(ry.raiPath('scenarios/pandaSingle.g'))
print(ry.raiPath('scenarios/pandaSingle.g'))
C.addFile("/home/jeong/summer_research/husky_assembly/rai_robot_models/scenarios/huskySingle.g")
C.addFrame('way1').setShape(ry.ST.marker, [.1]).setPosition([.4, .2, 1.])
C.addFrame('way2').setShape(ry.ST.marker, [.1]).setPosition([.4, .2, 1.4])
C.addFrame('way3').setShape(ry.ST.marker, [.1]).setPosition([-.4, .2, 1.])
C.addFrame('way4').setShape(ry.ST.marker, [.1]).setPosition([-.4, .2, 1.4])
C.view()

pp.wait_for_user()

exit()

qHome = C.getJointState()

komo = ry.KOMO(C, phases=4, slicesPerPhase=10, kOrder=1, enableCollisions=False)
komo.addControlObjective([], 0, 1e-1)
komo.addControlObjective([], 1, 1e0)
komo.addObjective([1], ry.FS.positionDiff, ['l_gripper', 'way1'], ry.OT.eq, [1e1])
komo.addObjective([2], ry.FS.positionDiff, ['l_gripper', 'way2'], ry.OT.eq, [1e1])
komo.addObjective([3], ry.FS.positionDiff, ['l_gripper', 'way3'], ry.OT.eq, [1e1])
komo.addObjective([4], ry.FS.positionDiff, ['l_gripper', 'way4'], ry.OT.eq, [1e1])

ret = ry.NLP_Solver(komo.nlp(), verbose=0 ) .solve()
print(ret)
q = komo.getPath()
print(q)

for t in range(len(q)):
    C.setJointState(q[t])
    C.view(False, f'waypoint {t}')
    # time.sleep(1)
    
C.setJointState(qHome)
komo = ry.KOMO(C, 4, 10, 2, False)
komo.addControlObjective([], 0, 1e-1) # what happens if you change weighting to 1e0? why?
komo.addControlObjective([], 2, 1e0)
komo.addObjective([1], ry.FS.positionDiff, ['l_gripper', 'way1'], ry.OT.eq, [1e1])
komo.addObjective([2], ry.FS.positionDiff, ['l_gripper', 'way2'], ry.OT.eq, [1e1])
komo.addObjective([3], ry.FS.positionDiff, ['l_gripper', 'way3'], ry.OT.eq, [1e1])
komo.addObjective([4], ry.FS.positionDiff, ['l_gripper', 'way4'], ry.OT.eq, [1e1])
komo.addObjective([4], ry.FS.jointState, [], ry.OT.eq, [1e1], [], order=1)

ret = ry.NLP_Solver(komo.nlp(), verbose=0 ) .solve()
print(ret)
q = komo.getPath()
print('size of path:', q.shape)

for t in range(q.shape[0]):
    C.setJointState(q[t])
    C.view(False, f'waypoint {t}')
    time.sleep(.1)

pp.wait_for_user()