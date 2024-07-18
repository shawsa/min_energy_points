from math import ceil
import numpy as np


def hex_limit_covering_radius(N: int):
    """
    Find the approximate covering radius for a hex grid with N points per unit area.
    """
    unit_density = 4 / (3 * np.sqrt(3))
    return np.sqrt(unit_density / N)


def hex_limit_density(h: float):
    """
    Find the approximate number of points per unit area for a hex grid with the
    covering radius h.
    """
    unit_density = 4 / (3 * np.sqrt(3))
    return ceil(1 / (unit_density * h**2))
