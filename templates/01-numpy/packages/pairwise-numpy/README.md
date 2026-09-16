# pairwise-numpy

The pairwise Euclidean distance matrix computed with NumPy broadcasting, chunked over rows so the intermediate stays bounded.

```python
import numpy as np
import pairwise_numpy

x = np.random.default_rng(0).normal(size=(1000, 8))
D = pairwise_numpy.pairwise_distances(x)  # (1000, 1000) float32
```

The conda package is built by the `pixi-build-python` backend using the manifest in `pixi.toml`; the workspace that consumes it lives two levels up.
