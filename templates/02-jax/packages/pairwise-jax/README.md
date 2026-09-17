# pairwise-jax

The pairwise Euclidean distance matrix computed with [JAX](https://docs.jax.dev/): the NumPy broadcast of stage 1 written in `jax.numpy`, vectorised with `vmap`, and compiled by XLA for the GPU. No kernel is written.

```python
import numpy as np
import pairwise_jax

x = np.random.default_rng(0).normal(size=(1000, 8))
D = pairwise_jax.pairwise_distances(x)  # (1000, 1000) float32
```

The conda package is built by the `pixi-build-python` backend using the manifest in `pixi.toml`; the workspace that consumes it lives two levels up.
