# Stage 3: a CUDA kernel compiled ahead of time and bound with nanobind

The destination of the three stages. The kernel from [stage 2](../02-cuda-python) is compiled by `nvcc` at package build time, exposed to NumPy through [nanobind](https://nanobind.readthedocs.io/), driven by [scikit-build-core](https://scikit-build-core.readthedocs.io/), and turned into a conda package by [Pixi Build](https://pixi.prefix.dev/latest/build/). The package's CUDA runtime requirement is derived from conda-forge run-exports rather than written by hand.

```console
pixi run demo     # builds gpu-pairwise on first run, then prints a 5x5 distance matrix
pixi run bench    # NumPy vs SciPy vs scikit-learn vs the CUDA kernel
pixi run test     # pytest against scipy.spatial.distance.cdist
pixi build --path src/gpu-pairwise -o dist   # a standalone .conda file
```

Every file in this directory is walked through in the [book](https://matthewfeickert.github.io/nanobind-cuda-example/), starting from the workspace chapter. Copy the directory out of the repository to use it as a starting point for your own extension.
