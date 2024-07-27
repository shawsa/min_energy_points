"""
A simple module for placing points on a sphere.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from tqdm import tqdm

from .points import PointCloud
from .kernels import GaussianRepulsionKernel, ConstRepulsionKernel


class TorrusPoints(PointCloud):
    def projection(self, points: np.ndarray[float]) -> np.ndarray[float]:
        ts = np.arctan2(points[:, 1], points[:, 0])
        xs, ys = R * np.cos(ts), R * np.sin(ts)
        my_points = points.copy()
        my_points[:, 0] -= xs
        my_points[:, 1] -= ys
        my_points *= r / la.norm(my_points, axis=-1)[:, np.newaxis]
        my_points[:, 0] += xs
        my_points[:, 1] += ys
        return my_points

    def __init__(
        self,
        N: int,
        R: float = 3,
        r: float = 1,
        auto_settle: bool = True,
        verbose: bool = False,
        tqdm_kwargs={},
    ):
        self.N = N
        self.R = R
        self.r = r
        self.verbose = verbose
        self.tqdm_kwargs = tqdm_kwargs
        points = self.projection(
            1.3 * (self.R + self.r) * 2 * (np.random.random((N, 3)) - 0.5)
        )
        super().__init__(
            all_points=points,
            num_fixed=0,
            num_ghost=0,
        )

        area_per_point = (4 * np.pi**2 * self.R * self.r) / self.N
        self.h = np.sqrt(area_per_point / np.pi)
        self.const_kernel = ConstRepulsionKernel(self.h)
        self.repulsion_kernel = GaussianRepulsionKernel(
            height=2 * self.h, shape=self.h
        )

        if auto_settle:
            self.auto_settle()

    @property
    def coords(self) -> tuple[np.ndarray[float]]:
        return (*self.points.T,)

    def settle(
        self,
        rate: float,
        repeat: int = 1,
        verbose: bool = False,
        tqdm_kwargs={},
    ):
        num_neighbors = 18
        my_iter = range(repeat)
        if verbose:
            my_iter = tqdm(my_iter, **tqdm_kwargs)
            my_iter.set_description(f"Settling {self.N} points")
        for _ in my_iter:
            super().settle(
                kernel=self.repulsion_kernel,
                rate=rate / num_neighbors,
                num_neighbors=num_neighbors,
                projection=self.projection,
            )

    def jostle(self, repeat: int = 1, verbose: bool = False, tqdm_kwargs={}):
        num_neighbors = 3
        my_iter = range(repeat)
        if verbose:
            my_iter = tqdm(my_iter, **tqdm_kwargs)
            my_iter.set_description(f"Jiggling {self.N} points")
        for _ in my_iter:
            super().settle(
                kernel=self.const_kernel,
                rate=1 / num_neighbors,
                num_neighbors=num_neighbors,
                projection=self.projection,
            )

    def auto_settle(self):
        self.jostle(repeat=100, verbose=self.verbose, tqdm_kwargs=self.tqdm_kwargs)
        self.settle(
            rate=1,
            repeat=100,
            verbose=self.verbose,
            tqdm_kwargs=self.tqdm_kwargs,
        )


if __name__ == "__main__":
    N = 3000
    R, r = 3, 1

    torrus = TorrusPoints(N, auto_settle=False)
    plt.ion()
    ax = plt.figure().add_subplot(projection="3d")
    (scatter,) = ax.plot(*torrus.points.T, "k.")
    plt.axis("equal")

    index = 0
    ax.plot(*torrus.points[index], "go")
    from scipy.spatial import KDTree
    tree = KDTree(torrus.points)
    _, nb_id = tree.query(torrus.points[index], k=18)
    ax.plot(*torrus.points[nb_id[1:]].T, "b.")

    for _ in tqdm(range(100)):
        torrus.jostle(repeat=1, verbose=False)
        scatter.set_data_3d(*torrus.points.T)
        plt.pause(0.1)

    for _ in tqdm(range(100)):
        torrus.settle(rate=1, repeat=1, verbose=False)
        scatter.set_data_3d(*torrus.points.T)
        plt.pause(0.1)
