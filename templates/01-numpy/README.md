# Stage 1: NumPy only

The starting point. `pairwise_numpy.pairwise_distances` computes the full Euclidean distance matrix with NumPy broadcasting, chunked over rows of `x` so the `(chunk, m, d)` intermediate stays bounded. It is packaged as a `noarch: python` conda package by [Pixi Build](https://pixi.prefix.dev/latest/build/) with the `pixi-build-python` backend and hatchling. Nothing is compiled and nothing mentions a GPU.

```console
pixi run demo     # prints a 5x5 distance matrix
pixi run bench    # NumPy broadcasting vs scipy.spatial.distance.cdist
pixi run test     # pytest against scipy.spatial.distance.cdist
pixi publish --path src/pairwise-numpy --target-channel local_channel   # a noarch .conda file in a local channel
```

The same function is reimplemented on the GPU in [stage 2](../02-cuda-python) and [stage 3](../03-nanobind-cuda) with an identical signature, so the three packages are drop-in replacements for each other.
