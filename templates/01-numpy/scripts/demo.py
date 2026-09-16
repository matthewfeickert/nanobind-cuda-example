"""The smallest end-to-end use of the package."""

import numpy as np
import pairwise_numpy

rng = np.random.default_rng(0)
x = rng.normal(size=(5, 3))

print(np.array2string(pairwise_numpy.pairwise_distances(x), precision=3))
