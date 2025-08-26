import os
import sys

# **************************************************************************
# Path
# **************************************************************************

HERE = os.path.dirname(os.path.dirname(__file__))  # scripts
PROJECT_DIR = os.path.dirname(HERE)  # husky_assembly
EXT_DIR = os.path.join(PROJECT_DIR, "ext")  # ext
DATA_DIR = os.path.join(EXT_DIR, "husky-assembly-teleop", "data")  # data
HUSKY_DATA_DIR = os.path.join(DATA_DIR, "husky_urdf")  # husky_urdf
HUSKY_URDF_PATH = os.path.join(HUSKY_DATA_DIR, "mt_husky_moveit_config", "urdf", "husky_ur5_e.urdf")  # exact husky_urdf path
LOG_DIR = os.path.join(HERE, "logs")