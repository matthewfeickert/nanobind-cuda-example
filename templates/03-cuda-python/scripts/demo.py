"""The smallest end-to-end use of the package."""

import numpy as np
import pairwise_cuda_python

rng = np.random.default_rng(0)
x = rng.normal(size=(5, 3))

print(f"GPU: {pairwise_cuda_python.device_name()}")
print(np.array2string(pairwise_cuda_python.pairwise_distances(x), precision=3))
