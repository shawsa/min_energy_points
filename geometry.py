"""
Forces defined by geometric objects.
"""

from abc import ABC, abstractmethod
from functools import reduce
import numpy as np
import numpy.linalg as la

from force_functions import ForceFunction


class Boundary(ABC):
    @abstractmethod
    def contains(self, points: np.ndarray[float]) -> np.ndarray[bool]:
        ...

    def __not__(self):
        return NegatedBoundary(self)


class NegatedBoundary(Boundary):
    def __init__(self, boundary):
        self.boundary = boundary

    def contains(self, points: np.ndarray[float]) -> np.ndarray[bool]:
        return ~self.boundary.contains(points)


class SimpleBoundary(Boundary):
    @abstractmethod
    def signed_distance(self, points: np.ndarray[float]):
        ...

    @abstractmethod
    def direction(self, points: np.ndarray[float]):
        ...

    def contains(self, points: np.ndarray[float]) -> np.ndarray[bool]:
        return self.signed_distance(points) >= 0


class IntersectionBoundary(Boundary):
    def __init__(self, *boundaries):
        self.boundaries = boundaries

    def contains(self, points: np.ndarray[float]) -> np.ndarray[bool]:
        return reduce(np.logical_and, (bnd.contains(points) for bnd in self.boundaries))


class UnionBoundary(Boundary):
    def __init__(self, *boundaries):
        self.boundaries = boundaries

    def contains(self, points: np.ndarray[float]) -> np.ndarray[bool]:
        return reduce(np.logical_or, (bnd.contains(points) for bnd in self.boundaries))


class BoundaryForce:
    def __init__(self, force: ForceFunction, boundary: SimpleBoundary):
        self.force = force
        self.boundary = boundary

    def __call__(self, points: np.ndarray[float]) -> np.ndarray[float]:
        return (
            self.boundary.direction(points)
            * self.force(self.boundary.signed_distance(points))[:, np.newaxis]
        )


class MultipleBoundaryForce(BoundaryForce):
    """Sum multiple boundary forces."""

    def __init__(self, *forces: tuple[BoundaryForce]):
        self.forces = list(forces)

    def __call__(self, points: np.ndarray[float]) -> np.ndarray[float]:
        return sum(force(points) for force in self.forces)


class RestrictedBoundaryForce(BoundaryForce):
    """Only apply boundary force if inside restricted area."""

    def __init__(self, force: BoundaryForce, restriction: Boundary):
        self.force = force
        self.restriction = restriction

    def __call__(self, points: np.ndarray[float]) -> np.ndarray[float]:
        mask = self.restriction.contains(points)
        ret = np.zeros_like(points)
        ret[mask] = self.force(points[mask])
        return ret


class PlanarBoundary(SimpleBoundary):
    def __init__(self, offset: np.ndarray[float], normal: np.ndarray[float]):
        self.offset = offset
        self.normal = normal / la.norm(normal)

    def signed_distance(self, points: np.ndarray[float]):
        return np.dot(points - self.offset, self.normal)

    def direction(self, points: np.ndarray[float]):
        return self.normal


class PolygonalEdgeBoundaryForce(RestrictedBoundaryForce):
    def __init__(
        self,
        force: ForceFunction,
        boundary: PlanarBoundary,
        neighbors: tuple[PlanarBoundary],
    ):
        boundary_force = BoundaryForce(force, boundary)
        restrictions = []
        for bnd in neighbors:
            offset = la.solve(
                np.array([boundary.normal, bnd.normal]),
                np.r_[
                    np.dot(boundary.normal, boundary.offset),
                    np.dot(bnd.normal, bnd.offset),
                ],
            )
            normal = boundary.normal - bnd.normal
            restrictions.append(PlanarBoundary(offset, normal))
        super().__init__(boundary_force, IntersectionBoundary(*restrictions))


class PolygonalBoundary(MultipleBoundaryForce):
    def __init__(
        self,
        force: ForceFunction,
        *boundaries: tuple[PlanarBoundary],
    ):
        self.force = force
        self.boundaries = boundaries

        super().__init__(
            *[
                PolygonalEdgeBoundaryForce(force, bnd, (n1, n2))
                for bnd, n1, n2 in zip(
                    boundaries,
                    boundaries[-1:] + boundaries[:-1],
                    boundaries[1:] + boundaries[:1],
                )
            ]
        )


class CircularBoundary(SimpleBoundary):
    def __init__(self, center: np.ndarray[float], radius: float, invert=False):
        """
        If invert is False the interior is inside the circle, else it's outside.
        """
        self.center = center
        self.radius = radius
        self.orientation = 1.0
        if invert:
            self.orientation = -1.0

    def signed_distance(self, points: np.ndarray[float]):
        return self.orientation * (self.radius - la.norm(points - self.center, axis=-1))

    def direction(self, points: np.ndarray[float]):
        my_dir = (points - self.center) * self.signed_distance(points)[:, np.newaxis]
        my_dir /= la.norm(my_dir, axis=-1)[:, np.newaxis]
        return self.orientation * my_dir


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .force_functions import RectifiedLinear

    threshold = 0.3
    slope = 1

    boundaries = [
        PlanarBoundary(np.array(normal, dtype=float), np.array(offset, dtype=float))
        for normal, offset in [
            ((0, 1), (0, -1)),  # bottom
            ((0, -1), (0, 1)),  # top
            ((1, 0), (-1, 0)),  # left
            ((-1, 0), (1, 0)),  # right
        ]
    ]

    boundary_force = RectifiedLinear(threshold=threshold, slope=slope)
    force = MultipleBoundaryForce(
        *(BoundaryForce(bnd, boundary_force) for bnd in boundaries)
    )

    thetas = np.linspace(-np.pi, np.pi, 101)
    points = np.c_[2 * np.cos(thetas), 2 * np.sin(thetas)]
    change = force(points)
    plt.plot(*points.T, "k.")
    for pnt, vec in zip(points, change):
        plt.plot(*np.c_[pnt, pnt + vec], "r-")

    plt.axis("equal")
