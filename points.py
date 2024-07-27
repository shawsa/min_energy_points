"""
Python classes for holding point clouds and point generation.
"""

import numpy as np
from scipy.spatial import KDTree
from typing import Callable

from kernels import RepulsionKernel


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

    @property
    def fixed_points(self):
        return self.all_points[self.num_mutable: self.num_mutable + self.num_fixed]

    @property
    def ghost_points(self):
        N = self.num_mutable + self.num_fixed
        return self.all_points[N: N + self.num_ghost]

    def settle(
        self,
        *,
        kernel: RepulsionKernel,
        rate: float,
        num_neighbors: int,
        force: Callable = None,
        projection: Callable[[np.ndarray[float]], np.ndarray[float]] = None,
    ):
        kdt = KDTree(self.points)
        _, neighbors_indices = kdt.query(self.mutable_points, num_neighbors + 1)
        neighbors = self.points[neighbors_indices][:, 1:]
        update = -rate * np.average(
            kernel(neighbors - self.mutable_points[:, np.newaxis, :]), axis=1
        )
        if force is not None:
            update += force(self.mutable_points)
        self.mutable_points += update
        if projection is not None:
            self.mutable_points = projection(self.mutable_points)
