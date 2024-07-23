import numpy as np
from tqdm import tqdm

from force_functions import RectifiedLinear
from kernels import ConstRepulsionKernel, GaussianRepulsionKernel
from geometry import (
    PlanarBoundary,
    PolygonalBoundary
)
from points import PointCloud

import matplotlib.pyplot as plt


boundary_force = PolygonalBoundary(
    RectifiedLinear(0),
    PlanarBoundary([1, 1], [-1, 0]),
    PlanarBoundary([0, 1], [0, -1]),
    PlanarBoundary([-1, 1], [1, 0]),
    PlanarBoundary([-.5, -1], [0, 1]),
    PlanarBoundary([0, 0], [-10, 5]),
    PlanarBoundary([0, 0], [10, 5]),
    PlanarBoundary([0.5, -1], [0, 1]),
)

plt.ion()


N = 2000
points = np.random.random((N, 2)) * 6 - 3
pc = PointCloud(points)

plt.close()
scatter, *_ = plt.plot(*pc.points.T, "k.")
plt.axis("equal")

for bnd in boundary_force.boundaries:
    plt.plot(*np.c_[bnd.offset, bnd.offset+.2*bnd.normal], "g-")

const_kernel = ConstRepulsionKernel(6 / N)
gauss_kernel = GaussianRepulsionKernel(1, 1 / (9 * np.pi))


def jostle(points: PointCloud, rate=1):
    num_neighbors = 3
    points.settle(
        kernel=const_kernel,
        rate=rate / num_neighbors,
        num_neighbors=num_neighbors,
        force=boundary_force,
    )


def settle(points: PointCloud, rate: float = 1):
    num_neighbors = 19
    points.settle(
        kernel=gauss_kernel,
        rate=rate / num_neighbors,
        num_neighbors=num_neighbors,
        force=boundary_force,
    )


for _ in tqdm(range(400)):
    jostle(pc, rate=20)
    scatter.set_data(*pc.mutable_points.T)
    plt.pause(1e-3)

for _ in tqdm(range(400)):
    settle(pc, rate=1)
    scatter.set_data(*pc.mutable_points.T)
    plt.pause(1e-3)
