import numpy as np
from tqdm import tqdm

from force_functions import RectifiedLinear
from kernels import ConstRepulsionKernel, GaussianRepulsionKernel
from geometry import BoundaryForce, MultipleBoundaryForce, CircularBoundary
from points import PointCloud
from .hex_limit import hex_limit_covering_radius

import matplotlib.pyplot as plt


outer_circ = CircularBoundary(np.r_[0.0, 0.0], 3.0)
inner_circ = CircularBoundary(np.r_[1.0, 0.0], 1.0, invert=True)

area = np.pi * (outer_circ.radius**2 - inner_circ.radius**2)

boundary_force = MultipleBoundaryForce(
    BoundaryForce(RectifiedLinear(0), outer_circ),
    BoundaryForce(RectifiedLinear(0), inner_circ),
)

N = 2000
points = np.random.random((N, 2)) * 6 - 3
pc = PointCloud(points)

h = np.sqrt(area) * hex_limit_covering_radius(N)
const_kernel = ConstRepulsionKernel(h)
gauss_kernel = GaussianRepulsionKernel(h, h)


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


plt.close()
scatter, *_ = plt.plot(*pc.points.T, "k.")
plt.axis("equal")

for _ in tqdm(range(100)):
    jostle(pc, rate=1)
    scatter.set_data(*pc.mutable_points.T)
    plt.pause(1e-3)

for _ in tqdm(range(100)):
    settle(pc, rate=1)
    scatter.set_data(*pc.mutable_points.T)
    plt.pause(1e-3)
