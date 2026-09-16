# nanobind-cuda-example 🐍⚡

How [Pixi](https://pixi.prefix.dev/) and CUDA fit together, shown as the three stages a Python package might pass through on its way to a GPU. The example computes the full pairwise Euclidean distance matrix. The kernel is deliberately naive, one CUDA thread per output element, because the packaging story is the point, not the kernel.

Each stage lives under [`templates/`](templates) as a self-contained Pixi workspace that can be copied out and used as a starting point. All three expose the same `pairwise_distances(x, y=None)` function.

| Stage | Directory | What it is | Package type |
|---|---|---|---|
| 1 | [`templates/01-numpy`](templates/01-numpy) | NumPy broadcasting on the CPU. The baseline and the loop to port. | `noarch: python` |
| 2 | [`templates/02-cuda-python`](templates/02-cuda-python) | The CUDA kernel shipped as source and compiled at first use by NVRTC through [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/) from [cuda-python](https://github.com/NVIDIA/cuda-python). | `noarch: python`, CUDA pin written by hand |
| 3 | [`templates/03-nanobind-cuda`](templates/03-nanobind-cuda) | The same kernel compiled by `nvcc` at build time into a [nanobind](https://nanobind.readthedocs.io/) extension, built with [scikit-build-core](https://scikit-build-core.readthedocs.io/) and packaged by [Pixi Build](https://pixi.prefix.dev/latest/build/). | `linux-64`, CUDA pin derived from run-exports |

In stage 3, `nvcc`, the CUDA runtime, the C++ compiler, CMake, Ninja and Python all come from conda-forge. Nobody has to install a system CUDA toolkit, and the resulting package records exactly which CUDA runtime it needs.

## Layout

```
nanobind-cuda-example/
├── pixi.toml                    # umbrella workspace: all three stages + the book
├── scripts/bench.py             # times the three stages against SciPy and scikit-learn
├── book/                        # MyST Jupyter Book walking through every stage
└── templates/                   # one self-contained Pixi workspace per stage
    ├── 01-numpy/
    │   ├── pixi.toml            #   workspace: plain linux-64, no GPU
    │   ├── scripts/             #   demo.py, bench.py
    │   └── packages/
    │       └── pairwise-numpy/  #   noarch package: pixi-build-python + hatchling
    ├── 02-cuda-python/
    │   ├── pixi.toml            #   workspace: linux-64 with a CUDA 13 driver
    │   ├── scripts/
    │   └── packages/
    │       └── pairwise-cuda-python/
    │           └── src/pairwise_cuda_python/
    │               ├── __init__.py  # cuda.core: NVRTC compile, buffers, launch
    │               └── pairwise.cu  # the kernel, shipped as source
    └── 03-nanobind-cuda/
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

At the top of the repository, the umbrella workspace pulls all three packages into one environment:

```console
pixi run bench    # all three stages side by side
pixi run test     # all three test suites
```

The first invocation of stage 3 compiles the extension (Pixi downloads the toolchain, builds, and caches). Subsequent runs are instant unless a `.cu`, `.py`, or CMake file changes. Stage 2 compiles its kernel with NVRTC on the first call instead.

Representative output from the root `pixi run bench` on an RTX 4060 Laptop GPU. Each row times the full `n × n` distance matrix for `n` points in `d` dimensions (a float32 array of shape `(n, d)`), so the work grows as `n² · d`. Both GPU columns include host to device copies. The NumPy baseline is skipped above `n = 8000` because it would take minutes.

```
      n    d |     numpy     scipy   sklearn | cuda-python  nanobind | nanobind vs scipy
   1000   16 |    0.014s    0.004s    0.003s |      0.003s    0.001s |    4.5x
   4000   16 |    0.312s    0.081s    0.050s |      0.036s    0.030s |    2.7x
   8000   16 |    1.221s    0.329s    0.274s |      0.133s    0.118s |    2.8x
   8000  128 |    7.577s    2.376s    0.287s |      0.252s    0.276s |    8.6x
  16000   16 |   skipped    1.343s    1.278s |      0.535s    0.466s |    2.9x
  16000  128 |   skipped   10.116s    1.412s |      1.002s    1.104s |    9.2x
  32000   16 |   skipped    5.307s    5.153s |      2.170s    1.752s |    3.0x
```

Stages 2 and 3 run the same kernel and stay within about 20% of each other, but the larger rows show where each binding pays. At `d = 16` the run is dominated by copying the `n²` result back to the host, and cuda-python is slower because the result passes through a pinned buffer and then a second copy into a NumPy array, while nanobind copies straight into the array it returns. At `d = 128` the kernel dominates and cuda-python is faster, because NVRTC compiled for this GPU's exact `sm_89` while the nanobind build for `all-major` carries no `sm_89` code and runs its `sm_80` binary instead. Neither gap is about the binding itself; both are choices that the [book](https://matthewfeickert.github.io/nanobind-cuda-example/) discusses.

## Build a distributable package

```console
pixi publish --path templates/03-nanobind-cuda/packages/gpu-pairwise --target-channel ./local_channel
```

builds the package and writes it into `local_channel/linux-64/gpu-pairwise-0.1.0-<hash>_0.conda` alongside a `repodata.json` index, so `local_channel/` is a complete conda channel that another workspace can list next to conda-forge. Publishing to a local filesystem channel is how Pixi builds a package for inspection; the older `pixi build` command is deprecated in favour of it. The same command with `templates/02-cuda-python/packages/pairwise-cuda-python` produces the stage 2 package, whose `depends` list contains only what its manifests wrote down. Its recorded runtime dependencies are the interesting part: the single `cuda-cudart-dev` host dependency became `cuda-cudart` and `cuda-version` pins through conda-forge run-exports, and the Python build became a `python_abi` pin.

```json
"depends": [
    "python >=3.11",
    "numpy >=2.3,<3",
    "libstdcxx >=15",
    "libgcc >=15",
    "cuda-version >=13.1,<14",
    "__glibc >=2.28,<3.0.a0",
    "cuda-cudart >=13.4.49,<14.0a0",
    "python_abi 3.14.* *_cp314"
]
```

Pointing `--target-channel` at a prefix.dev channel instead publishes the same package for real, and `pixi global install` can install it from either.

## How the pieces fit

- `templates/03-nanobind-cuda/packages/gpu-pairwise/pixi.toml` declares the `pixi-build-python` backend with `config.compilers = ["cxx", "cuda"]`. The backend runs `uv pip install --no-build-isolation`, so scikit-build-core and nanobind are listed as host dependencies rather than fetched from PyPI.
- The stage 3 workspace's `[workspace.build-variants]` table pins which conda-forge package `cuda` resolves to (`cuda-nvcc` 13.1), so the build is reproducible.
- The rich platform entry `{ name = "linux-64-cuda", platform = "linux-64", cuda = "13" }` tells the solver that this machine has a CUDA 13 driver, which is what lets it pick GPU-enabled builds.
- `CMakeLists.txt` uses `nanobind_add_module` on a `.cu` source. CMake's CUDA language support drives `nvcc`; nanobind handles the Python side.

## Read the book

The `book/` directory is a [MyST](https://mystmd.org/) Jupyter Book that walks through every stage, every file, and every command output in this repository, so it can be followed without a GPU. Build it with `pixi run docs-build`, which writes static HTML to `book/_build/html`, or serve it locally with live reload using `pixi run docs-start`.

## Requirements

Linux (x86-64) with an NVIDIA GPU and a CUDA 13 driver for stages 2 and 3 and the root workspace. Stage 1 runs anywhere. No system CUDA toolkit needed.

To resolve or build on a machine without an NVIDIA driver (e.g. CI):

```console
CONDA_OVERRIDE_CUDA=13 pixi install
```
