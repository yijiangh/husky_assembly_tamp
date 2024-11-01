import os
import sys
from typing import Callable, Dict, List, Set, Tuple, Union

import numpy as np
import pybullet_planning as pp
from termcolor import cprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pybullet_planning import Attachment
from robot.robot_setup import RobotSetup
from utils.collision import Element
from utils.utils import CounterModule, normalize_angles, get_custom_limits

# collision check threshold
MAX_DISTANCE = 0.0

# collision check enable
ENABLE_SELF_COLLISIONS = True


def get_transit_gen_fn(
    robot_setup: RobotSetup,
    element_from_index: Dict,
    fixed_obstacles: List[int],
    max_attempts: int = 10,
    collisions: bool = True,
    allow_failure: bool = False,
    verbose: bool = False,
    teleops: bool = False,
):
    def gen_fn(
        element_index: int,
        start_pose_2d: np.ndarray,
        tar_pose_2d: np.ndarray,
        assembled: List[int] = [],
        attachments: List[Attachment] = [],
        diagnosis: bool = False,
    ):
        robot_setup.update_attachments(attachments)
        # -------------------- obstacles --------------------#
        assambled_element_obstacles = set({element_from_index[e].body for e in list(assembled)})

        obstacles = set(fixed_obstacles) | assambled_element_obstacles
        if not collisions:
            obstacles = set()

        for attempt in range(max_attempts):
            if verbose:
                print(f"attempt from {start_pose_2d} to {tar_pose_2d}: ", attempt)

            command = compute_transit_path(
                robot_setup,
                start_pose_2d,
                tar_pose_2d,
                obstacles,
                verbose=verbose,
                diagnosis=diagnosis,
                teleops=teleops,
            )
            if command is None:
                continue

            cprint("Transit E#{} | Attempts: {} | Command: {}".format(element_index, attempt, len(command)), "green")

            yield command, [0] * len(command)
            break
        else:
            if verbose:
                cprint("E#{} | Attempts: {} | Max attempts exceeded!".format(element_index, max_attempts), "red")

            if allow_failure:
                yield None, None
            else:
                return None, None

    return gen_fn
