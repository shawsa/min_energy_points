"""
Python classes for holding point clouds and point generation.
"""

import numpy as np
from scipy.spatial import KDTree
from typing import Callable

from .kernels import RepulsionKernel


class PointCloud:
    def __init__(
        self,
        all_points: np.ndarray[float],
        num_fixed: int = 0,
        num_ghost: int = 0,
    ):
        self.all_points = all_points
        self.num_fixed = num_fixed
        self.num_ghost = num_ghost

        self.num_mutable = len(all_points) - num_fixed - num_ghost
        assert self.num_mutable > 0

    @property
    def mutable_points(self):
        return self.points[: self.num_mutable]

    @mutable_points.setter
    def mutable_points(self, value):
        self.points[: self.num_mutable] = value

    @property
    def points(self):
        return self.all_points[: self.num_mutable + self.num_fixed]

    def settle(
        self,
        *,
        kernel: RepulsionKernel,
        rate: float,
        num_neighbors: int,
        force: Callable = None,
        repeat=1,
    ):
        kdt = KDTree(self.points)
        _, neighbors_indices = kdt.query(self.mutable_points, num_neighbors + 1)
        neighbors = self.points[neighbors_indices][:, 1:]
        update = np.average(
            kernel(neighbors - self.mutable_points[:, np.newaxis, :]), axis=1
        )
        self.mutable_points -= rate * update
        if force is not None:
            self.mutable_points += force(self.mutable_points)
