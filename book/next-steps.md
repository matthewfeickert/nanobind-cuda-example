# Next steps

The example was kept small on purpose.
This chapter lists what it leaves out, roughly in the order a real project would run into them.

## Things this example does not do

The kernel is the naive one.
A production pairwise distance uses the expansion `||x - y||² = ||x||² + ||y||² - 2 x · y` and hands the dot products to cuBLAS, which is what scikit-learn does on the CPU and why it was competitive in the benchmark.
Adding `libcublas-dev` as a host dependency and linking against it is a one line change in each of the package manifest and `CMakeLists.txt`.

The kernel is compiled for every major architecture, and only ahead of time.
Stage 3 showed the other choice, compiling with NVRTC on the user's machine for the GPU that is present, which is what libraries such as CuPy do for user-supplied kernels.
A package can also do both, shipping a fat binary and falling back to runtime compilation for an architecture newer than its build.

The inputs and outputs live on the host.
Every call copies both arrays to the device and the result back, which is a large fraction of the measured time at small sizes.
nanobind's `ndarray` supports the DLPack and CUDA array interface protocols, so the same binding can accept CuPy or PyTorch arrays already on the device by changing `nb::device::cpu` to `nb::device::cuda` and dropping the copies.

There is no CPU fallback.
Importing `gpu_pairwise` on a machine without an NVIDIA driver succeeds, but the first call raises a `RuntimeError` from `cudaMalloc`.
A package meant for mixed hardware would either ship a CPU implementation behind the same function or be published as a `cuda` variant alongside a `cpu` variant, which is the pattern conda-forge uses for PyTorch and JAX.

The package is built for one Python version.
Distributing to users on Python 3.12 or 3.13 means building once per version.
Pixi Build variants can express that as a matrix, or the extension can target the stable ABI with `abi3 = true` in the backend configuration and nanobind's `STABLE_ABI` option so a single build serves every Python from 3.12 onward.

## Building without a GPU

The `cuda = "13"` rich platform means that resolving the workspace requires a CUDA driver to be present.
Continuous integration runners usually do not have one.
Setting `CONDA_OVERRIDE_CUDA=13` in the environment tells Pixi to assume the driver exists, which is enough to resolve, build, and package.
Only running the kernel needs real hardware, and the tests would have to be skipped or moved to a GPU runner.

## Further reading

- [Pixi Build documentation](https://pixi.prefix.dev/latest/build/getting_started/), including the [`pixi-build-python`](https://pixi.prefix.dev/latest/build/backends/pixi-build-python/) and [`pixi-build-cmake`](https://pixi.prefix.dev/latest/build/backends/pixi-build-cmake/) backend references.
- [nanobind documentation](https://nanobind.readthedocs.io/), in particular the [`ndarray`](https://nanobind.readthedocs.io/en/latest/ndarray.html) chapter on array exchange with NumPy, CuPy, PyTorch, and JAX.
- [scikit-build-core documentation](https://scikit-build-core.readthedocs.io/) for the CMake integration and configuration options.
- [Reproducible CUDA Accelerated Workflows for Scientists with Pixi](https://matthewfeickert-talks.github.io/reproducible-cuda-workflows-with-pixi-scipy-2026/), the SciPy 2026 tutorial this example grew out of, which covers the CUDA conda packaging ecosystem in more depth.
