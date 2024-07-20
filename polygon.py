import matplotlib.plyplot as plt
import numpy as np
from tqdm import tqdm

from force_functions import RectifiedLinear
from kernels import ConstRepulsionKernel, GaussianRepulsionKernel
from geometry import (
    MultipleBoundaryForce,
    PlanarBoundary,
    PolygonalBoundary,
)
from points import PointCloud

outside_num_sides = 8
inside_num_sides = 5

thetas1 = np.linspace(-np.pi, np.pi, outside_num_sides, endpoint=False)
vecs1 = np.c_[np.cos(thetas1), np.sin(thetas1)]

thetas2 = np.linspace(-np.pi, np.pi, inside_num_sides, endpoint=False)
vecs2 = np.c_[np.cos(thetas2), np.sin(thetas2)]

force = MultipleBoundaryForce(
    PolygonalBoundary(
        RectifiedLinear(0),
        *[PlanarBoundary(2*vec, -vec) for vec in vecs1],
    ),
    PolygonalBoundary(
        RectifiedLinear(0),
        *[PlanarBoundary(vec, vec) for vec in vecs2],
    ),
)


plt.ion()

N = 2000
points = np.random.random((N, 2)) * 6 - 3
pc = PointCloud(points)
plt.close()
scatter, *_ = plt.plot(*pc.points.T, "k.")
plt.axis("equal")

const_kernel = ConstRepulsionKernel(6 / N)
gauss_kernel = GaussianRepulsionKernel(1, 1 / (9 * np.pi))


def jostle(points: PointCloud, rate=1):
    num_neighbors = 3
    points.settle(
        kernel=const_kernel,
        rate=rate / num_neighbors,
        num_neighbors=num_neighbors,
        force=force,
    )


def settle(points: PointCloud, rate: float = 1):
    num_neighbors = 19
    points.settle(
        kernel=gauss_kernel,
        rate=rate / num_neighbors,
        num_neighbors=num_neighbors,
        force=force,
    )


for _ in tqdm(range(400)):
    jostle(pc, rate=20)
    scatter.set_data(*pc.mutable_points.T)
    plt.pause(1e-3)

for _ in tqdm(range(400)):
    settle(pc, rate=10)
    scatter.set_data(*pc.mutable_points.T)
    plt.pause(1e-3)
