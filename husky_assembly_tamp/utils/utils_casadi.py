import os
import sys
from typing import Dict, List, Tuple, Union

import casadi as ca
import numpy as np

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)


def eval(
    name: str,
    obj: ca.MX,
    sym: List[ca.MX],
    data: List[np.ndarray],
    verbose: bool = False,
    full: bool = True,
) -> np.ndarray:
    obj_cur = obj
    for sym_cur, data_cur in zip(sym, data):
        obj_cur = ca.substitute(obj_cur, sym_cur, data_cur)

    if full:
        val = ca.evalf(obj_cur).toarray()
    else:
        val = obj_cur

    if verbose:
        print(name, "\n", val)

    return val
