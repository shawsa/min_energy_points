"""
Some functions of signed distance used to define boundary forces.
"""

import numpy as np
from typing import Callable

ForceFunction = Callable[[np.ndarray[float]], np.ndarray[float]]


class RectifiedLinear(ForceFunction):

    def __init__(self, threshold: float, slope: float = 1):
        self.threshold = threshold
        self.slope = slope

    def __call__(self, distance: np.ndarray[float]) -> np.ndarray[float]:
        shifted = distance - self.threshold
        return -self.slope * shifted * np.heaviside(-shifted, 0.5)
