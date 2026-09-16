"""Pairwise Euclidean distances on the GPU.

The heavy lifting is a CUDA kernel in ``_core``; this module only adds the
NumPy-friendly conveniences (``y`` defaulting to ``x``, dtype and layout
normalisation) that are easier to write in Python than in C++.
"""

import numpy as np

from gpu_pairwise._core import device_name
from gpu_pairwise._core import pairwise_distances as _pairwise_distances

__all__ = ["device_name", "pairwise_distances"]


def pairwise_distances(x, y=None):
    """Return the ``(n, m)`` matrix of Euclidean distances between rows of ``x`` and ``y``.

    ``x`` has shape ``(n, d)`` and ``y`` shape ``(m, d)``; ``y`` defaults to ``x``.
    Inputs are promoted to float32 unless either is already float64.
    """
    x = np.asarray(x)
    y = x if y is None else np.asarray(y)
    # float64 only if the caller opted in; everything else (ints, float16, ...)
    # goes through the float32 kernel.
    dtype = np.float64 if np.float64 in (x.dtype, y.dtype) else np.float32
    x = np.ascontiguousarray(x, dtype=dtype)
    y = np.ascontiguousarray(y, dtype=dtype)
    return _pairwise_distances(x, y)
