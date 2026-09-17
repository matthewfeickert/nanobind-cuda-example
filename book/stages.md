# Four ways to build the same package

Speeding up a Python package with a GPU rarely happens in one jump.
The `templates/` directory holds four versions of the same package, one per stage, so the differences in code and in packaging can be read side by side.
Each directory is a complete Pixi workspace with its own `pixi.toml`, lock file, scripts, tests, and one source package, and each can be copied out of the repository as a starting point.

```console
$ cp -r templates/02-jax my-project
$ cd my-project && pixi run test
```

## One API, four implementations

All four packages expose the same function.

```python
pairwise_distances(x, y=None)
```

`x` has shape `(n, d)` and `y` shape `(m, d)`, `y` defaults to `x`, the result has shape `(n, m)`, and inputs are promoted to float32 unless either is already float64.
Because the signature and the dtype rules match, the four packages are drop-in replacements for each other, the same test file checks all of them against `scipy.spatial.distance.cdist`, and one benchmark can time them in a single process.

| Stage | Package | Where the work happens | When the kernel is compiled | Package type | Worth it when |
|---|---|---|---|---|---|
| 1 | `pairwise-numpy` | CPU, NumPy broadcasting | never | `noarch: python` | the data is small or no GPU is available |
| 2 | `pairwise-jax` | GPU, array program compiled by XLA | at first call, by XLA, for the GPU present | `noarch: python` | the problem fits array operations and JAX is already a dependency |
| 3 | `pairwise-cuda-python` | GPU, kernel launched by `cuda.core` | at first call, by NVRTC, for the GPU present | `noarch: python` | the package is used by its authors on machines they control |
| 4 | `gpu-pairwise` | GPU, kernel launched from C++ | at package build, by `nvcc`, for all major architectures | `linux-64`, CPython ABI specific | the package is shipped to others and must fail at build time, not at first call |

The three GPU stages are alternatives, not steps that every project takes in turn.
Stage 2 asks whether a kernel is needed at all; stages 3 and 4 write one and differ in when it is compiled, and the [stage 4 chapter](./stage-nanobind.md) opens with the trade-off between those two.

## The umbrella workspace

The `pixi.toml` at the top of the repository is not one of the stages.
It depends on all four packages as source dependencies so that they share one environment, and it owns the book.

```{literalinclude} ../pixi.toml
:filename: pixi.toml
:language: toml
:linenos:
```

Its `bench` task runs the cross-stage benchmark.

```{literalinclude} ../scripts/bench.py
:filename: scripts/bench.py
:language: python
```

Each row times the full `n × n` distance matrix for `n` points in `d` dimensions, held as a float32 array of shape `(n, d)`, so the work grows as `n² · d`.
All three GPU columns include the host to device and device to host copies.
The NumPy baseline is skipped above `n = 8000`, where it would take minutes, and the two largest rows exist to separate the fixed per-call cost of each GPU binding from the kernel.

```{code} text
:filename: pixi run bench
GPU: NVIDIA GeForce RTX 4060 Laptop GPU

      n    d |     numpy     scipy   sklearn | cuda-python  nanobind       jax | nanobind vs scipy
   1000   16 |    0.015s    0.004s    0.003s |      0.003s    0.001s    0.010s |    4.3x
   4000   16 |    0.333s    0.083s    0.056s |      0.040s    0.034s    0.052s |    2.5x
   8000   16 |    1.296s    0.336s    0.281s |      0.158s    0.123s    0.144s |    2.7x
   8000  128 |    8.086s    2.461s    0.354s |      0.256s    0.278s    0.160s |    8.8x
  16000   16 |   skipped    1.359s    1.393s |      0.551s    0.464s    0.383s |    2.9x
  16000  128 |   skipped   10.708s    1.510s |      1.361s    1.092s    0.562s |    9.8x
```

Four things stand out.
The NumPy baseline is the slowest column by a wide margin, and SciPy's compiled loop is a fair bit faster than it, so "just use NumPy" is not where a hot loop ends.
Stages 3 and 4 run the same kernel and stay within about 20% of each other at every size, which says the speedup comes from the kernel, not from how it is bound to Python.
Stage 2, which never wrote a kernel, is the fastest GPU column at `d = 128` and within a few tens of milliseconds of the hand-written kernels everywhere else, because XLA's fused reduction is a better kernel than the naive one in stages 3 and 4.
scikit-learn closes the gap at `d = 128` because it reformulates the problem as a matrix product handed to a multithreaded BLAS, which is the same trick that would make every GPU version far faster still.

## Where the two kernel bindings differ

The larger rows show the cuda-python and nanobind columns crossing over, and the reasons are instructive.

At `d = 16` the kernel does little work per output element and the run is dominated by moving the `n²` result back to the host, a gigabyte at `n = 16000`.
Stage 3 is the slower one there.
Its result lands in a pinned host buffer owned by `cuda.core` and is then copied a second time into the NumPy array that is returned, because the pinned buffer is freed on the way out.
Stage 4's binding function copies from the device straight into the array it hands to NumPy, so the extra host to host pass never happens.

At `d = 128` the kernel dominates and stage 3 is the faster one, by roughly ten percent.
The reason is which GPU code runs.
NVRTC in stage 3 compiles for the compute capability of the GPU that is present, `sm_89` on this laptop.
Stage 4 was compiled for `CMAKE_CUDA_ARCHITECTURES=all-major`, which on CUDA 13 embeds `sm_75`, `sm_80`, `sm_90`, `sm_100`, `sm_110`, and `sm_120` but nothing for `sm_89`, so the driver picks the `sm_80` binary, which is compatible but not tuned.
Building stage 4 with `native` instead, as the comment in its `CMakeLists.txt` shows, closes that gap at the price of a package that only runs on the GPU it was built on.

Neither gap is about nanobind or `cuda.core` as such.
What separates the two stages is what the package needs at build time, what it needs at runtime, and what its metadata is able to say about that, which the next chapters take in turn.

## Where JAX differs from both

The JAX column is timed last in each row, and the [stage 2 chapter](./stage-jax.md) explains why: JAX's runtime keeps every byte of device memory it has touched, so running it first would starve the two kernel stages at the larger sizes.
That is the shape of the stage 2 trade in one sentence.
The kernel, its fusion, and its speed came for free, and so did the memory policy, the compile-time behaviour, and everything else XLA decided.
Stages 3 and 4 are slower here because their kernel is naive, and a better kernel is a matter of editing one file; nothing in stage 2 can be edited to change what XLA does.
