"""Pairwise Euclidean distances on the GPU with JAX.

No kernel is written here. The same broadcast-and-reduce that stage 1 does
with NumPy is expressed in ``jax.numpy``, and XLA compiles it for whichever
accelerator jaxlib was built for. The API matches the other stages so the
packages are drop-in replacements for each other.
"""

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["device_name", "pairwise_distances"]


@jax.jit
def _pairwise(x, y):
    # One row of x against all of y: a (m, d) subtract, square, reduce that XLA
    # fuses into a single kernel, so the (n, m, d) intermediate that forces
    # stage 1 to chunk is never materialised.
    def row(carry, xi):
        diff = xi[None, :] - y  # (m, d)
        return carry, jnp.sqrt(jnp.sum(diff * diff, axis=-1))  # (m,)

    # A scan over rows rather than the more obvious vmap. Both give the same
    # numbers, but for the vmap version XLA's GPU autotuner spends minutes on
    # the fused reduction at n = m = 8000, d = 128 (its own slow-compile alarm
    # fires), while this one compiles in well under a second. See the book.
    return jax.lax.scan(row, None, x)[1]  # (n, m)


def device_name():
    """Return the name of the device JAX will run on."""
    return jax.devices()[0].device_kind


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
    # goes through float32. JAX defaults to float32 and has to be told to
    # keep float64 at all.
    dtype = np.float64 if np.float64 in (x.dtype, y.dtype) else np.float32
    if dtype == np.float64:
        jax.config.update("jax_enable_x64", True)
    out = _pairwise(jnp.asarray(x, dtype=dtype), jnp.asarray(y, dtype=dtype))
    # Block until the device is done and hand back an ordinary NumPy array,
    # so the timing and the return type match the other stages.
    return np.asarray(out.block_until_ready())
