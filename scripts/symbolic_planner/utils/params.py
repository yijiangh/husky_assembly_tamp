# Structure
MT_FILE_NAME = "one_tet_MT_contact" # one_tet_MT_contact/box_MT_contact
GROUNDED_ELEMENTS_INDEX = [0, 1, 2]  # one_tet_MT_contact[0, 1, 2]/box_MT_contact[0, 1, 2]

# Place Module

## robot pose sampler
SAMPLE_MAX_DISTANCE = 1.55  # dist in 2d plane
SAFETY_DISTANCE = 0.90  # safty dist in 2d plane
REACH_DISTANCE = 1.10  # dist in 3d space

## grasp sampler
SAMPLE_RANGE = 0.10
REACHABLE_MARGIN = 0.20
GRASP_METHOD = "robot"  # robot/cylinder
REDIRECT_METHOD = "robot"  # robot/preview/none(only for cylinder)

# Pick Module

PICK_DIRECTION = "left"  # left/behind

# Transfer Module
