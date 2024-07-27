import numpy as np
from tqdm import tqdm

from force_functions import RectifiedLinear
from kernels import ConstRepulsionKernel, GaussianRepulsionKernel
from geometry import (
    BoundaryForce,
    MultipleBoundaryForce,
    IntersectionBoundary,
    CircularBoundary,
    PlanarBoundary,
    PolygonalBoundary
)
from .points import PointCloud
from .hex_limit import hex_limit_covering_radius

import matplotlib.pyplot as plt


outer_circ = CircularBoundary(np.r_[0.0, 0.0], 3.0)

cutout = "square"
# cutout = "ell"
# cutout = "lune"
# cutout = "circ_in_square"

circ_area = np.pi * outer_circ.radius**2

match cutout:
    case "square":
        boundary_force = MultipleBoundaryForce(
            BoundaryForce(RectifiedLinear(0), outer_circ),
            PolygonalBoundary(
                RectifiedLinear(0),
                PlanarBoundary(*2*(np.r_[0, 1],)),
                PlanarBoundary(*2*(np.r_[1, 0],)),
                PlanarBoundary(*2*(np.r_[0, -1],)),
                PlanarBoundary(*2*(np.r_[-1, 0],)),
            )
        )
        area = circ_area - 4
    case "ell":
        boundary_force = MultipleBoundaryForce(
            BoundaryForce(RectifiedLinear(0), outer_circ),
            PolygonalBoundary(
                RectifiedLinear(0),
                PlanarBoundary(np.r_[0, 0], np.r_[-1, 0]),
                PlanarBoundary(np.r_[0, 0], np.r_[0, -1]),
                PlanarBoundary(np.r_[10, 10], np.r_[1, 1]),
            )
        )
        area = circ_area / 4
    case "lune":
        boundary_force = MultipleBoundaryForce(
            BoundaryForce(RectifiedLinear(0), outer_circ) ,
            BoundaryForce(RectifiedLinear(0), CircularBoundary(np.r_[2, 0], 2, invert=True)),
        )
        area = circ_area / 6  # just a guess
    case "circ_in_square":
        boundary_force = MultipleBoundaryForce(
            BoundaryForce(
                RectifiedLinear(0),
                CircularBoundary(np.r_[0, 0], 1.0, invert=True),
            ),
            *[
                BoundaryForce(RectifiedLinear(0), bnd)
                for bnd in (
                    PlanarBoundary(np.r_[0,  2], np.r_[0, -2]),
                    PlanarBoundary(np.r_[2,  0], np.r_[-2, 0]),
                    PlanarBoundary(np.r_[0, -2], np.r_[0, 2]),
                    PlanarBoundary(np.r_[-2, 0], np.r_[2, 0]),
                )
            ]
        )
        area = 4**2 - np.pi

N = 2000
points = np.random.random((N, 2)) * 6 - 3
pc = PointCloud(points)

plt.close()
scatter, *_ = plt.plot(*pc.points.T, "k.")
plt.axis("equal")

h = hex_limit_covering_radius(N) * np.sqrt(area)
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
    num_neighbors = 18
    points.settle(
        kernel=gauss_kernel,
        rate=rate / num_neighbors,
        num_neighbors=num_neighbors,
        force=boundary_force,
    )


for _ in tqdm(range(400)):
    jostle(pc, rate=1)
    scatter.set_data(*pc.mutable_points.T)
    plt.pause(1e-3)

for _ in tqdm(range(400)):
    settle(pc, rate=1)
    scatter.set_data(*pc.mutable_points.T)
    plt.pause(1e-3)
