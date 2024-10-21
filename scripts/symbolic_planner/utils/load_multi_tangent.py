import os, sys
from termcolor import cprint

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
multi_tangent_path = os.path.abspath(os.path.join(HERE, '..', '..', 'ext', 'FrameX', 'python'))
sys.path.append(multi_tangent_path)

import multi_tangent
cprint("Using FrameX multi_tangent from {}".format(os.path.dirname(multi_tangent.__file__)), 'yellow')
