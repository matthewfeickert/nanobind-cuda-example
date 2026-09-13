# gpu-pairwise

A CUDA kernel for the pairwise Euclidean distance matrix, exposed to NumPy
through [nanobind](https://nanobind.readthedocs.io/) and built with
[scikit-build-core](https://scikit-build-core.readthedocs.io/).

```python
import numpy as np
import gpu_pairwise

x = np.random.default_rng(0).normal(size=(1000, 8))
D = gpu_pairwise.pairwise_distances(x)   # (1000, 1000) float32
```

The conda package is built by the `pixi-build-python` backend using the
manifest in `pixi.toml`; the workspace that consumes it lives two levels up.
