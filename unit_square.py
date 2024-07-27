from math import ceil
import numpy as np
from tqdm import tqdm

from .force_functions import RectifiedLinear
from .hex_limit import hex_limit_covering_radius
from .kernels import ConstRepulsionKernel, GaussianRepulsionKernel
from .geometry import PolygonalBoundary, PlanarBoundary
from .points import PointCloud


def generate_boundary_points(n: int, boundary_points: np.ndarray[float] = None):
    """
    Generate 4n points on the boundary of the unit square with.
    Note that there will be n+1 points on each side since each corner is shared
    by two sides.

    If the boundary_points array is supplied then the result is placed there
    in addition to being returned.
    """

    if boundary_points is None:
        boundary_points = np.empty((4 * n, 2), dtype=float)
    side = np.linspace(0, 1, n, endpoint=False)
    boundary_points[:n][:, 0] = side
    boundary_points[:n][:, 1] = 0
    boundary_points[n : 2 * n][:, 0] = 1
    boundary_points[n : 2 * n][:, 1] = side
    boundary_points[2 * n : 3 * n][:, 0] = 1 - side
    boundary_points[2 * n : 3 * n][:, 1] = 1
    boundary_points[3 * n :][:, 0] = 0
    boundary_points[3 * n :][:, 1] = 1 - side
    return boundary_points


class UnitSquare(PointCloud):
    """
    Generates N points in unit square [0, 1] x [0, 1].
    """

    def __init__(
        self,
        N: int,
        auto_settle: bool = True,
        edge_cluster: bool = True,
        verbose: bool = False,
        tqdm_kwargs={},
    ):
        self.verbose = verbose
        self.tqdm_kwargs = tqdm_kwargs
        self.N = N
        self.h = hex_limit_covering_radius(N)
        self.n = ceil(1 / self.h)

        num_boundary = self.n * 4
        num_interior = self.N - num_boundary
        num_ghost = num_boundary + 4
        assert self.n >= 2

        points = np.empty((self.N + num_ghost, 2))

        boundary_points = points[num_interior : num_interior + num_boundary]
        generate_boundary_points(self.n, boundary_points)

        ghost_points = points[self.N :]
        generate_boundary_points(self.n + 1, ghost_points)
        ghost_points *= 1 + self.h * np.sqrt(2)
        ghost_points -= self.h * np.sqrt(2) / 2

        points[:num_interior] = self.h + (1 - 2 * self.h) * np.random.random(
            (self.N - 4 * self.n, 2)
        )

        super().__init__(
            points,
            num_fixed=num_boundary,
            num_ghost=num_ghost,
        )

        self.const_kernel = ConstRepulsionKernel(self.h / 2)
        self.repulsion_kernel = GaussianRepulsionKernel(self.h / 2, self.h)

        force_func = RectifiedLinear(self.h / 2)
        self.boundary_force = PolygonalBoundary(
            force_func,
            *[
                PlanarBoundary(np.array(offset), np.array(normal))
                for offset, normal in [
                    [(0, 0), (0, 1)],  # bottom
                    [(0, 0), (1, 0)],  # left
                    [(1, 1), (0, -1)],  # top
                    [(1, 1), (-1, 0)],  # right
                ]
            ],
        )

        if auto_settle:
            self.auto_settle()

        if edge_cluster:
            self.edge_cluster()

    def edge_cluster(self):
        shift_points = self.mutable_points - 0.5
        edge_distance = 0.5 - np.max(np.abs(shift_points))
        factor = (0.5 - edge_distance / 2) / (0.5 - edge_distance)
        self.mutable_points = shift_points * factor + 0.5

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
                force=self.boundary_force,
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
                force=self.boundary_force,
            )

    def auto_settle(self):
        self.jostle(repeat=100, verbose=self.verbose, tqdm_kwargs=self.tqdm_kwargs)
        self.settle(
            rate=1, repeat=100, verbose=self.verbose, tqdm_kwargs=self.tqdm_kwargs
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.spatial import Delaunay

    plt.ion()
    N = 10000
    unit_square = UnitSquare(N, auto_settle=False, edge_cluster=False)

    plt.figure()
    (scatter,) = plt.plot(*unit_square.mutable_points.T, "k.")
    plt.plot(*unit_square.fixed_points.T, "bs")
    plt.plot(*unit_square.ghost_points.T, "or")
    plt.axis("equal")

    for _ in tqdm(range(100)):
        unit_square.jostle(repeat=1)
        scatter.set_data(*unit_square.mutable_points.T)
        plt.pause(1e-3)

    for _ in tqdm(range(100)):
        unit_square.settle(rate=1)
        scatter.set_data(*unit_square.mutable_points.T)
        plt.pause(1e-3)

    unit_square.edge_cluster()
    scatter.set_data(*unit_square.mutable_points.T)
    plt.pause(1e-3)

    mesh = Delaunay(unit_square.points)
    plt.triplot(*unit_square.points.T, mesh.simplices)
