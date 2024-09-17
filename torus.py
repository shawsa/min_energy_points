"""
A simple module for placing points on a sphere.
"""

from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from tqdm import tqdm

from .points import PointCloud
from .kernels import GaussianRepulsionKernel, ConstRepulsionKernel


class TorusPoints(PointCloud):
    def __init__(
        self,
        N: int,
        R: float = 3,
        r: float = 1,
        auto_settle: bool = True,
        verbose: bool = False,
        tqdm_kwargs={},
        init_points: np.ndarray[float] = None,
    ):
        self.N = N
        self.R = R
        self.r = r
        self.verbose = verbose
        self.tqdm_kwargs = tqdm_kwargs
        if init_points is not None:
            assert len(init_points) == N
            points = init_points
        else:
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
            height=self.h, shape=1.5 * self.h
        )

        if auto_settle:
            self.auto_settle()

        self.init_normals()

    def projection(self, points: np.ndarray[float]) -> np.ndarray[float]:
        ts = np.arctan2(points[:, 1], points[:, 0])
        xs, ys = self.R * np.cos(ts), self.R * np.sin(ts)
        my_points = points.copy()
        my_points[:, 0] -= xs
        my_points[:, 1] -= ys
        my_points *= self.r / la.norm(my_points, axis=-1)[:, np.newaxis]
        my_points[:, 0] += xs
        my_points[:, 1] += ys
        return my_points

    def init_normals(self) -> np.ndarray[float]:
        ts = np.arctan2(self.points[:, 1], self.points[:, 0])
        normals = self.points.copy()
        normals[:, 0] -= self.R * np.cos(ts)
        normals[:, 1] -= self.R * np.sin(ts)
        normals /= la.norm(normals, axis=-1)[:, np.newaxis]
        self.normals = normals

    def implicit_surf(self, points: np.ndarray[float]) -> np.ndarray[float]:
        ts = np.arctan2(points[:, 1], points[:, 0])
        my_points = points.copy()
        my_points[:, 0] -= self.R * np.cos(ts)
        my_points[:, 1] -= self.R * np.sin(ts)
        return self.r - la.norm(my_points, axis=-1)

    @property
    def coords(self) -> tuple[np.ndarray[float]]:
        return (*self.points.T,)

    def settle(
        self,
        rate: float,
        repeat: int = 1,
        verbose: bool = False,
        tqdm_kwargs={},
        num_neighbors=18,
    ):
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


class SpiralTorus(TorusPoints):
    def __init__(
        self,
        N: int,
        R: float = 3.0,
        r: float = 1.0,
        num_wraps: int = None,
    ):
        self.num_wraps = num_wraps
        if self.num_wraps is None:
            self.num_wraps = ceil(np.sqrt(3) * R / r)
        self.num_spirals = int(
            np.sqrt(N / self.num_wraps / np.sqrt(1 + (self.num_wraps * r / R) ** 2))
        )
        self.points_per_spiral = int(N / self.num_spirals)
        self.points_per_spiral = self.num_wraps * (
            int(self.points_per_spiral / self.num_wraps)
        )

        N = self.num_spirals * self.points_per_spiral

        points = np.zeros((N, 3))
        thetas_base = np.linspace(0, 2 * np.pi, self.points_per_spiral, endpoint=False)
        phis = np.linspace(
            0, 2 * np.pi * self.num_wraps, self.points_per_spiral, endpoint=False
        )

        theta_shift = 2 * np.pi / self.num_wraps / self.num_spirals

        for spiral_index in range(self.num_spirals):
            thetas = thetas_base + (spiral_index * theta_shift)
            start = spiral_index * self.points_per_spiral
            stop = (spiral_index + 1) * self.points_per_spiral
            points[start:stop, 0] = np.cos(thetas) * (R + r * np.cos(phis))
            points[start:stop, 1] = np.sin(thetas) * (R + r * np.cos(phis))
            points[start:stop, 2] = r * np.sin(phis)

        super().__init__(N=N, R=R, r=r, auto_settle=False, init_points=points)


if __name__ == "__main__":
    from .local_voronoi import LocalSurfaceVoronoi

    import pyvista as pv
    from scipy.spatial import KDTree

    N = 30_000
    R, r = 3, 1

    torus = TorusPoints(N, R=R, r=r, auto_settle=False)
    plt.ion()
    ax = plt.figure().add_subplot(projection="3d")
    (scatter,) = ax.plot(*torus.points.T, "k.")
    plt.axis("equal")

    for _ in tqdm(range(100)):
        torus.jostle(repeat=1, verbose=False)
        scatter.set_data_3d(*torus.points.T)
        plt.pause(0.1)

    for _ in tqdm(range(100)):
        torus.settle(rate=1, repeat=1, verbose=False)
        scatter.set_data_3d(*torus.points.T)
        plt.pause(0.1)

    torus = SpiralTorus(N, R=R, r=r)
    plotter = pv.Plotter(off_screen=False)
    plotter.add_mesh(
        pv.PolyData(torus.points),
        style="points",
        point_size=10,
        render_points_as_spheres=True,
    )
    # plotter.set_focus(torus.points[2185])
    plotter.show()

    tree = KDTree(torus.points)
    ds, _ = tree.query(torus.points, k=2)
    ds = ds[:, 1]
    print(f"min h = {np.min(ds)}")
    print(f"average h = {np.average(ds)}")
    print(f"max h = {np.max(ds)}")
    plt.figure()
    plt.hist(ds, bins=100)

    vor = LocalSurfaceVoronoi(torus.points, torus.normals, torus.implicit_surf)
    surf = pv.PolyData(torus.points, [(3, *f) for f in vor.triangles])
    plotter = pv.Plotter(off_screen=False)
    plotter.add_mesh(
        surf,
        show_vertices=True,
        show_edges=True,
        show_scalar_bar=False,
    )
    plotter.show()
