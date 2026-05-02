import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import pybullet as p

# PyBullet planning imports
import pybullet_planning as pp
from compas import json_dump, json_load

# COMPAS imports
from compas.geometry import Frame

# COMPAS FAB imports
from compas_fab.backends import PyBulletClient, PyBulletPlanner

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)


def pose_from_frame(frame, scale=1.0):
    """Returns a PyBullet pose from a frame.

    Parameters
    ----------
    frame : :class:`compas.geometry.Frame`

    Returns
    -------
    point, quaternion : tuple
    """
    return ([v * scale for v in frame.point], frame.quaternion.xyzw)


def frame_from_pose(pose, scale=1.0):
    """Returns a frame from a PyBullet pose.

    Parameters
    ----------
    point, quaternion : tuple

    Returns
    -------
    :class:`compas.geometry.Frame`
    """
    point, (x, y, z, w) = pose
    return Frame.from_quaternion([w, x, y, z], point=[v * scale for v in point])


class SceneParser:
    """A class to parse and reconstruct scenes from robot cell state files."""

    def __init__(self, robot_cell_state_path: str, use_gui: bool = True, verbose: bool = False):
        """Initialize the SceneParser.

        Parameters
        ----------
        robot_cell_state_path : str
            Path to the robot cell state JSON file
        use_gui : bool, optional
            Whether to use PyBullet GUI, by default True
        verbose : bool, optional
            Whether to print verbose output, by default False
        """
        self.robot_cell_state_path = robot_cell_state_path
        self.use_gui = use_gui
        self.verbose = verbose

        # Derived paths
        self.robot_cell_state_dir = os.path.dirname(robot_cell_state_path)
        self.design_case_dir = os.path.dirname(self.robot_cell_state_dir)
        self.robot_cell_json_path = os.path.join(self.design_case_dir, "RobotCell.json")

        # Loaded data
        self.robot_cell = None
        self.robot_cell_state = None

        # PyBullet objects
        self.client = None
        self.planner = None

        # Load the data
        self._load_data()

    def _load_data(self):
        """Load robot cell and robot cell state data from JSON files."""
        if not os.path.exists(self.robot_cell_json_path):
            raise FileNotFoundError(f"Robot cell JSON not found: {self.robot_cell_json_path}")

        if not os.path.exists(self.robot_cell_state_path):
            raise FileNotFoundError(f"Robot cell state JSON not found: {self.robot_cell_state_path}")

        if self.verbose:
            print(f"Loading robot cell from: {self.robot_cell_json_path}")
            print(f"Loading robot cell state from: {self.robot_cell_state_path}")

        # Load robot cell
        with open(self.robot_cell_json_path, "r") as f:
            self.robot_cell = json_load(f)

        # Load robot cell state
        with open(self.robot_cell_state_path, "r") as f:
            self.robot_cell_state = json_load(f)

    def reconstruct_scene(self) -> Tuple[PyBulletClient, PyBulletPlanner]:
        """Reconstruct the scene in PyBullet.

        Returns
        -------
        Tuple[PyBulletClient, PyBulletPlanner]
            The PyBullet client and planner objects
        """
        if self.verbose:
            print("Starting scene reconstruction...")

        # Create PyBullet client and planner
        self.client = PyBulletClient(connection_type="gui" if self.use_gui else "direct", verbose=self.verbose, enable_debug_gui=True)
        self.client.__enter__()  # Enter the context manager
        

        pp.CLIENTS[self.client.client_id] = self.use_gui
        self.planner = PyBulletPlanner(self.client)

        # Set up the scene
        start = time.time()
        self.planner.set_robot_cell(self.robot_cell)
        self.planner.set_robot_cell_state(self.robot_cell_state)

        setup_time = time.time() - start
        if self.verbose:
            print(f"Setting robot cell and state took {setup_time:.3f} seconds")

        # Update robot base pose
        robot_base_frame = self.robot_cell_state.robot_base_frame
        robot_base_pose = pose_from_frame(robot_base_frame)
        pp.set_pose(self.client.robot_puid, robot_base_pose)

        if self.verbose:
            print("Scene reconstruction completed!")

        return self.client, self.planner

    def cleanup(self):
        """Clean up PyBullet resources."""
        if self.client is not None:
            try:
                self.client.__exit__(None, None, None)  # Exit the context manager
            except:
                pass
            self.client = None
        self.planner = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    @classmethod
    def from_design_case(cls, design_study_path: str, design_case: str, robot_cell_state_filename: str, use_gui: bool = True, verbose: bool = False) -> "SceneParser":
        """Create SceneParser from design case parameters.

        Parameters
        ----------
        design_study_path : str
            Path to the design study directory
        design_case : str
            Name of the design case
        robot_cell_state_filename : str
            Filename of the robot cell state JSON
        use_gui : bool, optional
            Whether to use PyBullet GUI, by default True
        verbose : bool, optional
            Whether to print verbose output, by default False

        Returns
        -------
        SceneParser
            The SceneParser instance
        """
        robot_cell_state_path = os.path.join(design_study_path, design_case, "RobotCellStates", robot_cell_state_filename)

        return cls(robot_cell_state_path, use_gui, verbose)
