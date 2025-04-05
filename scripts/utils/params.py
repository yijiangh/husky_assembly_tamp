import os
import sys

# -------------------- Structure --------------------#

# one_tet_MT_contact/box_MT_contact/triangle_reciprocal_MT_contact
MT_FILE_NAME = "box_MT_contact"

# one_tet_MT_contact[0, 1, 2]/box_MT_contact[0, 1, 2]/triangle_reciprocal_MT_contact[0, 1, 2]
GROUNDED_ELEMENTS_INDEX = [0, 1, 2]

# -------------------- Motion Plan --------------------#

ROBOT_NUM = 1
MANIPULATOR_PLANNER = "normal"  # normal/default

# Place Module

## robot pose sampler
SAMPLE_MAX_DISTANCE = 1.55  # dist in 2d plane
SAFETY_DISTANCE = 0.95  # safty dist in 2d plane
REACH_DISTANCE = 1.15  # dist in 3d space

## grasp sampler
SAMPLE_RANGE = 0.10
REACHABLE_MARGIN = 0.20
GRASP_METHOD = "robot"  # robot/cylinder
REDIRECT_METHOD = "preview"  # robot/preview/none(only for cylinder)

# Pick Module

PICK_DIRECTION = "left"  # left/behind

# Transfer Module

# -------------------- Switch Config --------------------#

## Place Module
PLACE_VERBOSE = True
PLACE_DIAGNOSIS = False
PLACE_SHOW = True or PLACE_DIAGNOSIS

## Pick Module
PICK_VERBOSE = False
PICK_DIAGNOSIS = False
PICK_SHOW = False or PICK_DIAGNOSIS

## Transfer Module
TRANSFER_VERBOSE = False
TRANSFER_DIAGNOSIS = False
TRANSFER_SHOW = False or TRANSFER_DIAGNOSIS

## Back Module
BACK_VERBOSE = True
BACK_DIAGNOSIS = True
BACK_SHOW = False or TRANSFER_DIAGNOSIS

## Robot
MNIPULATOR_PLAN_SHOW = True

# -------------------- Path --------------------#

HERE = os.path.dirname(os.path.dirname(__file__))  # scripts
PROJECT_DIR = os.path.dirname(HERE)  # husky_assembly
DATA_DIR = os.path.join(PROJECT_DIR, "data")  # data
PACKAGE_DIR = os.path.join(DATA_DIR, "husky_urdf")  # husky_urdf
EXT_DIR = os.path.join(PROJECT_DIR, "ext")  # ext
LOG_DIR = os.path.join(HERE, "logs")  # ompl

URDF_PATH = os.path.join(PACKAGE_DIR, "mt_husky_moveit_config", "urdf", "husky_ur5_e.urdf")
