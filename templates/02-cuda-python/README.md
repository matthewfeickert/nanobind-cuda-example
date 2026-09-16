# Stage 2: the kernel in Python, compiled at runtime with cuda-python

The first GPU port. The hot loop from [stage 1](../01-numpy) becomes a CUDA kernel in `pairwise.cu`, but the kernel is shipped as source inside a pure Python package and compiled on first use by NVRTC through [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/), the Pythonic layer of NVIDIA's [cuda-python](https://github.com/NVIDIA/cuda-python) project. The package is still `noarch: python`, and it has to declare its CUDA requirement by hand.

```console
pixi run demo     # prints the GPU name and a 5x5 distance matrix
pixi run bench    # scipy.spatial.distance.cdist vs the NVRTC-compiled kernel
pixi run test     # pytest against scipy.spatial.distance.cdist
pixi publish --path src/pairwise-cuda-python --target-channel ./local_channel   # a noarch .conda file in a local channel
```

[Stage 3](../03-nanobind-cuda) takes the same kernel and compiles it ahead of time with `nvcc` inside a nanobind extension module.
