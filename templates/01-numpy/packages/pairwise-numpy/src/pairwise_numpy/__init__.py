"""Pairwise Euclidean distances on the CPU with NumPy.

This is the baseline the GPU stages are measured against. The API (``y``
defaulting to ``x``, the dtype rules, the output shape) is identical in
every stage so that the implementations are drop-in replacements for each
other.
"""

import numpy as np

__all__ = ["pairwise_distances"]

# Rows of ``x`` handled per iteration. The broadcast intermediate below has
# shape (chunk, m, d); at n = m = 8000 and d = 128 the unchunked (n, m, d)
# float32 version would be 33 GB, so bounding it is what makes this usable.
_CHUNK = 256


def pairwise_distances(x, y=None):
    """Return the ``(n, m)`` matrix of Euclidean distances between rows of ``x`` and ``y``.

    ``x`` has shape ``(n, d)`` and ``y`` shape ``(m, d)``; ``y`` defaults to ``x``.
    Inputs are promoted to float32 unless either is already float64.
    """
    x = np.asarray(x)
    y = x if y is None else np.asarray(y)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2-dimensional")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same number of columns (features)")
    # float64 only if the caller opted in; everything else (ints, float16, ...)
    # goes through float32.
    dtype = np.float64 if np.float64 in (x.dtype, y.dtype) else np.float32
    x = np.ascontiguousarray(x, dtype=dtype)
    y = np.ascontiguousarray(y, dtype=dtype)

    out = np.empty((x.shape[0], y.shape[0]), dtype=dtype)
    for start in range(0, x.shape[0], _CHUNK):
        # (chunk, 1, d) - (1, m, d) -> (chunk, m, d), reduced over d.
        diff = x[start : start + _CHUNK, None, :] - y[None, :, :]
        out[start : start + _CHUNK] = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    return out
