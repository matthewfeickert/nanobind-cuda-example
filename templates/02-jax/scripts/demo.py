"""The smallest end-to-end use of the package."""

import pairwise_jax
import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=(5, 3))

print(f"GPU: {pairwise_jax.device_name()}")
print(np.array2string(pairwise_jax.pairwise_distances(x), precision=3))
