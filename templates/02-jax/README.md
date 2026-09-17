# Stage 2: JAX, the GPU without a kernel

The first GPU port asks whether a kernel is needed at all. The broadcast from [stage 1](../01-numpy) is rewritten in `jax.numpy`, vectorised over rows with `vmap`, and compiled by XLA for the GPU that [jaxlib](https://docs.jax.dev/) was built for. The package is `noarch: python`, but it has to say by hand that it wants the CUDA build of jaxlib.

```console
pixi run demo     # prints the GPU name and a 5x5 distance matrix
pixi run bench    # scipy.spatial.distance.cdist vs the XLA-compiled broadcast
pixi run test     # pytest against scipy.spatial.distance.cdist
pixi publish --path packages/pairwise-jax --target-channel ./local_channel   # a noarch .conda file in a local channel
```

[Stage 3](../03-cuda-python) and [stage 4](../04-nanobind-cuda) write the kernel by hand instead.
