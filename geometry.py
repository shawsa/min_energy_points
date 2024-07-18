"""
Forces defined by geometric objects.
"""

from abc import ABC, abstractmethod
import numpy as np

from .force_functions import ForceFunction


class Boundary(ABC):
    @abstractmethod
    def signed_distance(self, points: np.ndarray[float]):
        ...

    @abstractmethod
    def direction(self, points: np.ndarray[float]):
        ...


class BoundaryForce:
    def __init__(self, boundary: Boundary, force: ForceFunction):
        self.boundary = boundary
        self.force = force

    def __call__(self, points: np.ndarray[float]) -> np.ndarray[float]:
        return (
            self.boundary.direction(points)
            * self.force(self.boundary.signed_distance(points))[:, np.newaxis]
        )


class MultipleBoundaryForce:
    def __init__(self, *forces: tuple[BoundaryForce]):
        self.forces = list(forces)

    def __call__(self, points: np.ndarray[float]) -> np.ndarray[float]:
        return sum(force(points) for force in self.forces)


class PlanarBoundary(Boundary):
    def __init__(self, offset: np.ndarray[float], normal: np.ndarray[float]):
        self.offset = offset
        self.normal = normal

    def signed_distance(self, points: np.ndarray[float]):
        return np.dot(points - self.offset, self.normal)

    def direction(self, points: np.ndarray[float]):
        return self.normal


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
    points = np.c_[2*np.cos(thetas), 2*np.sin(thetas)]
    change = force(points)
    plt.plot(*points.T, "k.")
    for pnt, vec in zip(points, change):
        plt.plot(*np.c_[pnt, pnt + vec], "r-")

    plt.axis("equal")
