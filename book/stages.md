# Three ways to build the same package

Speeding up a Python package with a GPU rarely happens in one jump.
The `templates/` directory holds three versions of the same package, one per stage, so the differences in code and in packaging can be read side by side.
Each directory is a complete Pixi workspace with its own `pixi.toml`, lock file, scripts, tests, and one source package, and each can be copied out of the repository as a starting point.

```console
$ cp -r templates/02-cuda-python my-project
$ cd my-project && pixi run test
```

## One API, three implementations

All three packages expose the same function.

```python
pairwise_distances(x, y=None)
```

`x` has shape `(n, d)` and `y` shape `(m, d)`, `y` defaults to `x`, the result has shape `(n, m)`, and inputs are promoted to float32 unless either is already float64.
Because the signature and the dtype rules match, the three packages are drop-in replacements for each other, the same test file checks all of them against `scipy.spatial.distance.cdist`, and one benchmark can time them in a single process.

| Stage | Package | Where the work happens | When the kernel is compiled | Package type | Worth it when |
|---|---|---|---|---|---|
| 1 | `pairwise-numpy` | CPU, NumPy broadcasting | never | `noarch: python` | the data is small or no GPU is available |
| 2 | `pairwise-cuda-python` | GPU, kernel launched by `cuda.core` | at first call, by NVRTC, for the GPU present | `noarch: python` | the package is used by its authors on machines they control |
| 3 | `gpu-pairwise` | GPU, kernel launched from C++ | at package build, by `nvcc`, for all major architectures | `linux-64`, CPython ABI specific | the package is shipped to others and must fail at build time, not at first call |

Stages 2 and 3 are alternatives, not steps that every project takes in turn.
The [stage 3 chapter](./stage-nanobind.md) opens with the trade-off between them.

## The umbrella workspace

The `pixi.toml` at the top of the repository is not one of the stages.
It depends on all three packages as source dependencies so that they share one environment, and it owns the book.

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
Both GPU columns include the host to device and device to host copies.
The NumPy baseline is skipped above `n = 8000`, where it would take minutes, and the three largest rows exist to separate the fixed per-call cost of each GPU binding from the kernel.

```{code} text
:filename: pixi run bench
GPU: NVIDIA GeForce RTX 4060 Laptop GPU

      n    d |     numpy     scipy   sklearn | cuda-python  nanobind | nanobind vs scipy
   1000   16 |    0.014s    0.004s    0.003s |      0.003s    0.001s |    4.5x
   4000   16 |    0.312s    0.081s    0.050s |      0.036s    0.030s |    2.7x
   8000   16 |    1.221s    0.329s    0.274s |      0.133s    0.118s |    2.8x
   8000  128 |    7.577s    2.376s    0.287s |      0.252s    0.276s |    8.6x
  16000   16 |   skipped    1.343s    1.278s |      0.535s    0.466s |    2.9x
  16000  128 |   skipped   10.116s    1.412s |      1.002s    1.104s |    9.2x
  32000   16 |   skipped    5.307s    5.153s |      2.170s    1.752s |    3.0x
```

Three things stand out.
The NumPy baseline is the slowest column by a wide margin, and SciPy's compiled loop is a fair bit faster than it, so "just use NumPy" is not where a hot loop ends.
Stages 2 and 3 run the same kernel and stay within about 20% of each other at every size, which says the speedup comes from the kernel, not from how it is bound to Python.
scikit-learn closes the gap at `d = 128` because it reformulates the problem as a matrix product handed to a multithreaded BLAS, which is the same trick that would make either GPU version far faster still.

## Where the two bindings differ

The larger rows show the two GPU columns crossing over, and the reasons are instructive.

At `d = 16` the kernel does little work per output element and the run is dominated by moving the `n²` result back to the host, four gigabytes at `n = 32000`.
Stage 2 is the slower one there.
Its result lands in a pinned host buffer owned by `cuda.core` and is then copied a second time into the NumPy array that is returned, because the pinned buffer is freed on the way out.
Stage 3's binding function copies from the device straight into the array it hands to NumPy, so the extra host to host pass never happens.

At `d = 128` the kernel dominates and stage 2 is the faster one, by roughly ten percent.
The reason is which GPU code runs.
NVRTC in stage 2 compiles for the compute capability of the GPU that is present, `sm_89` on this laptop.
Stage 3 was compiled for `CMAKE_CUDA_ARCHITECTURES=all-major`, which on CUDA 13 embeds `sm_75`, `sm_80`, `sm_90`, `sm_100`, `sm_110`, and `sm_120` but nothing for `sm_89`, so the driver picks the `sm_80` binary, which is compatible but not tuned.
Building stage 3 with `native` instead, as the comment in its `CMakeLists.txt` shows, closes that gap at the price of a package that only runs on the GPU it was built on.

Neither gap is about nanobind or `cuda.core` as such.
What separates the two stages is what the package needs at build time, what it needs at runtime, and what its metadata is able to say about that, which the next chapters take in turn.
