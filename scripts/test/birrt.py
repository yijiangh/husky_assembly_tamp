import time
from typing import List, Optional, Union

import numpy as np




class TreeNode(object):
    def __init__(self, config, parent=None):
        self.config = config
        self.parent = parent

    def retrace(self):
        sequence = []
        node = self
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]

    def __str__(self):
        return "TreeNode(" + str(self.config) + ")"

    __repr__ = __str__

class BiRRT:
    def __init__(self, start, target, sample_fn, invalid_fn, extend_fn, distance_fn, max_time):
        self.start = start
        self.target = target
        self.sample_fn = sample_fn
        self.invalid_fn = invalid_fn
        self.extend_fn = extend_fn
        self.distance_fn = distance_fn
        self.max_time = max_time

    def plan(self) -> Union[np.ndarray, None]:
        start_time = time.time()

        if self.invalid_fn(self.start) or self.invalid_fn(self.target):
            return None

        nodes1, nodes2 = [TreeNode(self.start)], [TreeNode(self.target)]
        for iteration in irange(self.max_time):
            if self.max_time <= elapsed_time(start_time):
                break
