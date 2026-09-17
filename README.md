# nanobind-cuda-example 🐍⚡

How [Pixi](https://pixi.prefix.dev/) and CUDA fit together, shown as the four stages a Python package might pass through on its way to a GPU. The example computes the full pairwise Euclidean distance matrix. The kernel is deliberately naive, one CUDA thread per output element, because the packaging story is the point, not the kernel.

Each stage lives under [`templates/`](templates) as a self-contained Pixi workspace that can be copied out and used as a starting point. All four expose the same `pairwise_distances(x, y=None)` function.

| Stage | Directory | What it is | Package type |
|---|---|---|---|
| 1 | [`templates/01-numpy`](templates/01-numpy) | NumPy broadcasting on the CPU. The baseline and the loop to port. | `noarch: python` |
| 2 | [`templates/02-jax`](templates/02-jax) | The same broadcast written in `jax.numpy` and compiled by XLA through [JAX](https://docs.jax.dev/) for the GPU. No kernel is written. | `noarch: python`, CUDA build of jaxlib pinned by hand |
| 3 | [`templates/03-cuda-python`](templates/03-cuda-python) | The CUDA kernel shipped as source and compiled at first use by NVRTC through [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) from [cuda-python](https://github.com/NVIDIA/cuda-python). | `noarch: python`, CUDA pin written by hand |
| 4 | [`templates/04-nanobind-cuda`](templates/04-nanobind-cuda) | The same kernel compiled by `nvcc` at build time into a [nanobind](https://nanobind.readthedocs.io/) extension, built with [scikit-build-core](https://scikit-build-core.readthedocs.io/) and packaged by [Pixi Build](https://pixi.prefix.dev/latest/build/). | `linux-64`, CUDA pin derived from run-exports |

In stage 4, `nvcc`, the CUDA runtime, the C++ compiler, CMake, Ninja and Python all come from conda-forge. Nobody has to install a system CUDA toolkit, and the resulting package records exactly which CUDA runtime it needs.

## Layout

```
nanobind-cuda-example/
├── pixi.toml                    # umbrella workspace: all four stages + the book
├── scripts/bench.py             # times the four stages against SciPy and scikit-learn
├── book/                        # MyST Jupyter Book walking through every stage
└── templates/                   # one self-contained Pixi workspace per stage
    ├── 01-numpy/
    │   ├── pixi.toml            #   workspace: plain linux-64, no GPU
    │   ├── scripts/             #   demo.py, bench.py
    │   └── packages/
    │       └── pairwise-numpy/  #   noarch package: pixi-build-python + hatchling
    ├── 02-jax/
    │   ├── pixi.toml            #   workspace: linux-64 with a CUDA 13 driver
    │   ├── scripts/
    │   └── packages/
    │       └── pairwise-jax/    #   noarch package that pins the CUDA build of jaxlib
    ├── 03-cuda-python/
    │   ├── pixi.toml            #   workspace: linux-64 with a CUDA 13 driver
    │   ├── scripts/
    │   └── packages/
    │       └── pairwise-cuda-python/
    │           └── src/pairwise_cuda_python/
    │               ├── __init__.py  # cuda.core: NVRTC compile, buffers, launch
    │               └── pairwise.cu  # the kernel, shipped as source
    └── 04-nanobind-cuda/
        ├── pixi.toml            #   workspace: CUDA platform + build variants
        ├── scripts/
        └── packages/
            └── gpu-pairwise/    #   pixi-build-python package
                ├── pixi.toml        # package manifest: backend, compilers, host deps
                ├── pyproject.toml   # name / version / runtime deps (scikit-build-core)
                ├── CMakeLists.txt   # nanobind_add_module(_core ... pairwise.cu)
                ├── src/pairwise.cu  # the CUDA kernel + nanobind bindings
                ├── src/gpu_pairwise/# the Python package (thin NumPy wrapper)
                └── tests/           # pytest checks against scipy.spatial.distance.cdist
```

## Run it

Inside any template directory:

```console
pixi run demo     # builds the package on first run, then prints a 5x5 distance matrix
pixi run bench    # times that stage against the CPU references
pixi run test     # pytest against scipy.spatial.distance.cdist, in the `test` environment
```

At the top of the repository, the umbrella workspace pulls all four packages into one environment:

```console
pixi run bench    # all four stages side by side
pixi run test     # all four test suites
pixi run lint     # the pre-commit hooks, via prek, on every file
```

The first invocation of stage 4 compiles the extension (Pixi downloads the toolchain, builds, and caches). Subsequent runs are instant unless a `.cu`, `.py`, or CMake file changes. Stage 3 compiles its kernel with NVRTC on the first call instead.

Representative output from the root `pixi run bench` on an RTX 4060 Laptop GPU. Each row times the full `n × n` distance matrix for `n` points in `d` dimensions (a float32 array of shape `(n, d)`), so the work grows as `n² · d`. All GPU columns include host to device copies. The NumPy baseline is skipped above `n = 8000` because it would take minutes.

```
      n    d |     numpy     scipy   sklearn | cuda-python  nanobind       jax | nanobind vs scipy
   1000   16 |    0.015s    0.004s    0.003s |      0.003s    0.001s    0.010s |    4.3x
   4000   16 |    0.333s    0.083s    0.056s |      0.040s    0.034s    0.052s |    2.5x
   8000   16 |    1.296s    0.336s    0.281s |      0.158s    0.123s    0.144s |    2.7x
   8000  128 |    8.086s    2.461s    0.354s |      0.256s    0.278s    0.160s |    8.8x
  16000   16 |   skipped    1.359s    1.393s |      0.551s    0.464s    0.383s |    2.9x
  16000  128 |   skipped   10.708s    1.510s |      1.361s    1.092s    0.562s |    9.8x
```

Stage 2 never wrote a kernel and is the fastest GPU column at `d = 128`, because XLA's fused reduction beats the naive hand-written kernel; it is timed last in each row because JAX's runtime keeps the device memory it has touched, which would otherwise starve the other two stages. Stages 3 and 4 run the same kernel and stay within about 20% of each other, but the larger rows show where each binding pays. At `d = 16` the run is dominated by copying the `n²` result back to the host, and cuda-python is slower because the result passes through a pinned buffer and then a second copy into a NumPy array, while nanobind copies straight into the array it returns. At `d = 128` the kernel dominates and cuda-python is faster, because NVRTC compiled for this GPU's exact `sm_89` while the nanobind build for `all-major` carries no `sm_89` code and runs its `sm_80` binary instead. Neither gap is about the binding itself; both are choices that the [book](https://matthewfeickert.github.io/nanobind-cuda-example/) discusses.

## Build a distributable package

```console
pixi publish --path templates/04-nanobind-cuda/packages/gpu-pairwise --target-channel ./local_channel
```

builds the package and writes it into `local_channel/linux-64/gpu-pairwise-0.1.0-<hash>_0.conda` alongside a `repodata.json` index, so `local_channel/` is a complete conda channel that another workspace can list next to conda-forge. Publishing to a local filesystem channel is how Pixi builds a package for inspection; the older `pixi build` command is deprecated in favour of it. The same command with `templates/03-cuda-python/packages/pairwise-cuda-python` produces the stage 3 package, whose `depends` list contains only what its manifests wrote down. Its recorded runtime dependencies are the interesting part: the single `cuda-cudart-dev` host dependency became `cuda-cudart` and `cuda-version` pins through conda-forge run-exports, and the Python build became a `python_abi` pin.

```json
"depends": [
    "python >=3.11",
    "numpy >=2.3",
    "libstdcxx >=15",
    "libgcc >=15",
    "cuda-version >=13.1,<14",
    "__glibc >=2.28,<3.0.a0",
    "cuda-cudart >=13.4.49,<14.0a0",
    "python_abi 3.14.* *_cp314"
]
```

To read a built package back without unpacking it, install [rattler-build](https://rattler.build/) as a global tool and point its inspect command at the file:

```console
pixi global install rattler-build
rattler-build package inspect $(find ./local_channel/ -type f -iname '*pairwise*.conda')
```

Pointing `--target-channel` at a prefix.dev channel instead publishes the same package for real, and `pixi global install` can install it from either.

## How the pieces fit

- `templates/04-nanobind-cuda/packages/gpu-pairwise/pixi.toml` declares the `pixi-build-python` backend with `config.compilers = ["cxx", "cuda"]`. The backend runs `uv pip install --no-build-isolation`, so scikit-build-core and nanobind are listed as host dependencies rather than fetched from PyPI.
- The stage 4 workspace's `[workspace.build-variants]` table pins which conda-forge package `cuda` resolves to (`cuda-nvcc` 13.1), so the build is reproducible.
- The rich platform entry `{ name = "linux-64-cuda", platform = "linux-64", cuda = "13" }` tells the solver that this machine has a CUDA 13 driver, which is what lets it pick GPU-enabled builds.
- `CMakeLists.txt` uses `nanobind_add_module` on a `.cu` source. CMake's CUDA language support drives `nvcc`; nanobind handles the Python side.

## Read the book

The `book/` directory is a [MyST](https://mystmd.org/) Jupyter Book that walks through every stage, every file, and every command output in this repository, so it can be followed without a GPU. Build it with `pixi run docs-build`, which writes static HTML to `book/_build/html`, or serve it locally with live reload using `pixi run docs-start`.

## Requirements

Linux (x86-64) with an NVIDIA GPU and a CUDA 13 driver for stages 2 to 4 and the root workspace. Stage 1 runs anywhere. No system CUDA toolkit needed.

To resolve or build on a machine without an NVIDIA driver (e.g. CI):

```console
CONDA_OVERRIDE_CUDA=13 pixi install
```
