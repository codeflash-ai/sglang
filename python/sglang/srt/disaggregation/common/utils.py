import threading
from collections import deque
from typing import List, Tuple

import numpy as np
import numpy.typing as npt


class FastQueue:
    def __init__(self):
        self._buf = deque()
        self._cond = threading.Condition()

    def put(self, item):
        with self._cond:
            self._buf.append(item)
            # wake up a thread of wait()
            self._cond.notify()

    def get(self):
        with self._cond:
            # if queue is empty  ,block until is notified()
            while not self._buf:
                self._cond.wait()
            return self._buf.popleft()


def group_concurrent_contiguous(
    src_indices: npt.NDArray[np.int32], dst_indices: npt.NDArray[np.int32]
) -> Tuple[List[npt.NDArray[np.int32]], List[npt.NDArray[np.int32]]]:
    """Vectorised NumPy implementation."""
    if src_indices.size == 0:
        return [], []

    brk = np.flatnonzero((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1)) + 1
    starts = np.concatenate(([0], brk))
    ends = np.concatenate((brk, [src_indices.size]))

    src_groups = [src_indices[s:e].tolist() for s, e in zip(starts, ends)]
    dst_groups = [dst_indices[s:e].tolist() for s, e in zip(starts, ends)]

    return src_groups, dst_groups
