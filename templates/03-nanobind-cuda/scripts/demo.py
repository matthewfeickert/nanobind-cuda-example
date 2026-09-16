"""The smallest end-to-end use of the packaged extension."""

import gpu_pairwise
import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=(5, 3))

print(f"GPU: {gpu_pairwise.device_name()}")
print(np.array2string(gpu_pairwise.pairwise_distances(x), precision=3))
