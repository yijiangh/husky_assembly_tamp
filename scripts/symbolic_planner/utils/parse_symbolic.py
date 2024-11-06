import argparse
import json
import os

from termcolor import colored
from utils.utils import LOGGER, PROJECT_DIR

MT_DATA_PATH = os.path.join(PROJECT_DIR, "ext", "FrameX", "data", "mt_results")

############################################

PDDL_FOLDERS = ["01_joint_only"]
DOMAIN_NAMES = ["joint_only"]

############################################


def parse_mt_symbolic(mt_json_file_name):
    file_path = os.path.join(MT_DATA_PATH, mt_json_file_name)
    with open(file_path, "r") as f:
        json_data = json.load(f)

    line_pt_pairs = json_data["line_pt_pairs"]
    contact_id_pairs = json_data["contact_id_pairs"]
    beam_ids = [f"b{i}" for i in range(len(line_pt_pairs))]

    return beam_ids, contact_id_pairs


############################################
