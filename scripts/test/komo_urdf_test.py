import pybullet_planning as pp

pp.connect(True)

pp.load_model("/home/jeong/summer_research/pandaSingle.urdf")

pp.wait_for_user()