"""
Functions of distance used to repel points.
"""

from abc import ABC, abstractmethod
import numpy as np
import numpy.linalg as la


class RepulsionKernel(ABC):
    """Used to redistribute nodes to make them more regular."""

    @abstractmethod
    def __call__(self, displacement: np.ndarray):
        raise NotImplementedError


class GaussianRepulsionKernel(RepulsionKernel):
    def __init__(self, height: float, shape: float):
        self.shape = shape
        self.height = height

    def __call__(self, displacement: np.ndarray):
        ret = displacement.copy()
        mags = la.norm(ret, axis=-1)
        mags = self.height * np.exp(-((mags / self.shape) ** 2)) / mags
        ret[..., 0] *= mags
        ret[..., 1] *= mags
        return ret


class ConstRepulsionKernel(RepulsionKernel):
    def __init__(self, const: float):
        self.const = const

    def __call__(self, displacement: np.ndarray):
        ret = displacement.copy()
        mags = la.norm(ret, axis=-1)
        ret[..., 0] /= mags
        ret[..., 1] /= mags
        return self.const * ret
