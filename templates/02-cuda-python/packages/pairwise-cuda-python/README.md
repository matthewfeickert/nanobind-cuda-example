# pairwise-cuda-python

The pairwise Euclidean distance matrix computed by a CUDA kernel that is compiled at first use with NVRTC and launched through [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/). The package is pure Python; the kernel source ships as `pairwise.cu` next to the module.

```python
import numpy as np
import pairwise_cuda_python

x = np.random.default_rng(0).normal(size=(1000, 8))
D = pairwise_cuda_python.pairwise_distances(x)  # (1000, 1000) float32
```

The conda package is built by the `pixi-build-python` backend using the manifest in `pixi.toml`; the workspace that consumes it lives two levels up.
