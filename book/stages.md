# The three stages

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

| Stage | Package | Where the work happens | When the kernel is compiled | Package type |
|---|---|---|---|---|
| 1 | `pairwise-numpy` | CPU, NumPy broadcasting | never | `noarch: python` |
| 2 | `pairwise-cuda-python` | GPU, kernel launched by `cuda.core` | at first call, by NVRTC, for the GPU present | `noarch: python` |
| 3 | `gpu-pairwise` | GPU, kernel launched from C++ | at package build, by `nvcc`, for all major architectures | `linux-64`, CPython ABI specific |

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

```{code} text
:filename: pixi run bench
GPU: NVIDIA GeForce RTX 4060 Laptop GPU

      n    d |     numpy     scipy   sklearn | cuda-python  nanobind | nanobind vs scipy
   1000   16 |    0.014s    0.004s    0.003s |      0.003s    0.001s |    4.5x
   4000   16 |    0.323s    0.083s    0.050s |      0.037s    0.031s |    2.7x
   8000   16 |    1.342s    0.326s    0.264s |      0.147s    0.116s |    2.8x
   8000  128 |    7.863s    2.370s    0.340s |      0.253s    0.276s |    8.6x
```

Three things stand out.
The NumPy baseline is the slowest column by a wide margin, and SciPy's compiled loop is a fair bit faster than it, so "just use NumPy" is not where a hot loop ends.
Stages 2 and 3 run the same kernel and land within a few tens of milliseconds of each other, which says the speedup comes from the kernel, not from how it is bound to Python.
scikit-learn closes the gap at `d = 128` because it reformulates the problem as a matrix product handed to a multithreaded BLAS, which is the same trick that would make either GPU version far faster still.

What separates stages 2 and 3 is therefore not speed.
It is what the package needs at build time, what it needs at runtime, and what its metadata is able to say about that, which the next chapters take in turn.
