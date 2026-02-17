import os
import sys

from termcolor import cprint
from husky_assembly_tamp.utils.params import PROJECT_DIR

multi_tangent_path = os.path.abspath(os.path.join(PROJECT_DIR, 'ext', 'FrameX', 'python'))
sys.path.append(multi_tangent_path)

import multi_tangent

# cprint("Using FrameX multi_tangent from {}".format(os.path.dirname(multi_tangent.__file__)), 'yellow')
