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
from points import PointCloud

import matplotlib.pyplot as plt


outer_circ = CircularBoundary(np.r_[0.0, 0.0], 3.0)

cutout = "square"
# cutout = "ell"
# cutout = "lune"
# cutout = "circ_in_square"

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
    case "lune":
        boundary_force = MultipleBoundaryForce(
            BoundaryForce(RectifiedLinear(0), outer_circ) ,
            BoundaryForce(RectifiedLinear(0), CircularBoundary(np.r_[2, 0], 2, invert=True)),
        )
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

thetas = np.linspace(-np.pi, np.pi, 201)

plt.ion()
plt.plot(
    outer_circ.radius * np.cos(thetas) + outer_circ.center[0],
    outer_circ.radius * np.sin(thetas) + outer_circ.center[1],
    "k-",
)

plt.axis("equal")

N = 2000
points = np.random.random((N, 2)) * 6 - 3
pc = PointCloud(points)

scatter, *_ = plt.plot(*pc.points.T, "k.")


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
    settle(pc, rate=10)
    scatter.set_data(*pc.mutable_points.T)
    plt.pause(1e-3)
