import json
import os
import sys

import numpy as np
import pybullet_planning as pp
from compas.data import Data
from compas.geometry import Transformation

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

from utils.params import DATA_DIR


class GraspTarget(Data):
    def __init__(self, target_type, **kwargs):
        super(GraspTarget, self).__init__()
        self.type = target_type
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def __data__(self):
        data = {"type": self.type}
        for k, v in self.__dict__.items():
            if k not in ["type", "_guid", "_name"]:
                data[k] = v
        return data

    @classmethod
    def __from_data__(cls, data):
        target_type = data.pop("type")
        return cls(target_type, **data)


def parse_transformation(data):
    """Parse a transformation from a dict with a 'matrix' key."""
    if isinstance(data, dict) and "matrix" in data:
        return Transformation(data["matrix"])
    return data


def parse_grasp_target_dict(d):
    """Parse a dict (from JSON) into a GraspTarget, converting transformations."""
    target_type = d.get("type")
    kwargs = {}
    for k, v in d.items():
        if k == "type":
            continue
        # Handle nested 'data' for transformations
        if isinstance(v, dict) and v.get("dtype", "").endswith("Transformation"):
            kwargs[k] = parse_transformation(v["data"])
        else:
            kwargs[k] = v
    return GraspTarget(target_type, **kwargs)


class TargetParser:
    def __init__(self, file_path, state_name):
        self.targets = self.load_grasp_targets(file_path, state_name)
        self.parse_targets()
        self.poses = self.parse_targets()

    def load_grasp_targets(self, file_path, state_name):
        in_path = os.path.join(file_path, "RobotCellStates", state_name + "_GraspTargets.json")
        with open(in_path, "r") as f:
            raw = json.load(f)

        targets = []
        for item in raw:
            data = item["data"] if "data" in item else item
            targets.append(parse_grasp_target_dict(data))

        return targets

    def parse_targets(self):
        poses = []
        for target in self.targets:
            world_from_bar: Transformation = target.world_from_bar
            world_from_tool0: Transformation = target.world_from_tool0
            world_from_bar = pp.pose_from_tform(np.array(world_from_bar.matrix))
            world_from_tool0 = pp.pose_from_tform(np.array(world_from_tool0.matrix))
            tool0_from_bar = pp.multiply(pp.invert(world_from_tool0), world_from_bar)
            poses.append(tool0_from_bar)
        return poses


if __name__ == "__main__":
    file_path = os.path.join(DATA_DIR, "husky_assembly_design_study", "250707_RobotX_box_demo")
    state_name = "robotx_box_A0-G"
    target_parser = TargetParser(file_path, state_name)
