# nanobind-cuda-example 🐍⚡

One Pixi workspace that builds a **Python extension module wrapping a CUDA kernel** as a conda package, using [Pixi Build](https://pixi.prefix.dev/latest/build/).

The package, [`gpu-pairwise`](src/gpu-pairwise), computes the full pairwise Euclidean distance matrix on the GPU. The kernel is deliberately naive: one CUDA thread per output element. The point of the example is not the kernel but the packaging story it demonstrates:

- You have a hot loop in Python and a small CUDA kernel that replaces it.
- You expose it to NumPy with [nanobind](https://nanobind.readthedocs.io/) and build it with [scikit-build-core](https://scikit-build-core.readthedocs.io/).
- Pixi builds it into a conda package where `nvcc`, the CUDA runtime, the C++ compiler, CMake, Ninja and Python all come from conda-forge. Nobody has to install a system CUDA toolkit, and the resulting package records exactly which CUDA runtime it needs.

## Layout

```
nanobind-cuda-example/
├── pixi.toml                    # the workspace: dependencies + tasks
├── scripts/
│   ├── demo.py                  # smallest end-to-end use
│   └── bench.py                 # NumPy vs SciPy vs scikit-learn vs CUDA
└── src/
    └── gpu-pairwise/            # pixi-build-python package
        ├── pixi.toml            #   package manifest: backend, compilers, host deps
        ├── pyproject.toml       #   name / version / runtime deps (scikit-build-core)
        ├── CMakeLists.txt       #   nanobind_add_module(_core ... pairwise.cu)
        ├── src/pairwise.cu      #   the CUDA kernel + nanobind bindings
        ├── src/gpu_pairwise/    #   the Python package (thin NumPy wrapper)
        └── tests/               #   pytest checks against scipy.spatial.distance.cdist
```

## Run it

```console
pixi run demo     # builds gpu-pairwise on first run, then prints a 5x5 distance matrix
pixi run bench    # times the CPU references against the CUDA kernel
pixi run test     # pytest, in the `test` environment
```

The first invocation compiles the extension (Pixi downloads the toolchain, builds, and caches). Subsequent runs are instant unless a `.cu`, `.py`, or CMake file changes.

Representative output from `pixi run bench` on an RTX 4060 Laptop GPU. Each row times the full `n × n` distance matrix for `n` points in `d` dimensions (a float32 array of shape `(n, d)`), so the work grows as `n² · d`. The CUDA times include host to device copies. The NumPy broadcasting reference is skipped for the larger `n` because its `(n, n, d)` intermediate would need tens of GB of memory.

```
      n    d |     numpy     scipy   sklearn      cuda | speedup vs scipy
   1000   16 |    0.049s    0.006s    0.013s    0.001s |    6.2x
   4000   16 |    3.256s    0.118s    0.157s    0.042s |    2.8x
   8000   16 |   skipped    1.609s    1.181s    0.161s |   10.0x
   8000  128 |   skipped    4.442s    0.674s    0.321s |   13.8x
```

## Build a distributable package

```console
pixi build --path src/gpu-pairwise -o dist
```

produces `dist/gpu-pairwise-0.1.0-<hash>_0.conda`. Its recorded runtime dependencies are the interesting part: the single `cuda-cudart-dev` host dependency became `cuda-cudart` and `cuda-version` pins through conda-forge run-exports, and the Python build became a `python_abi` pin.

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

From here the package can be `pixi publish`ed to a prefix.dev channel or installed elsewhere with `pixi global install`.

## How the pieces fit

- `src/gpu-pairwise/pixi.toml` declares the `pixi-build-python` backend with `config.compilers = ["cxx", "cuda"]`. The backend runs `uv pip install --no-build-isolation`, so scikit-build-core and nanobind are listed as host dependencies rather than fetched from PyPI.
- The workspace `[workspace.build-variants]` table pins which conda-forge package `cuda` resolves to (`cuda-nvcc` 13.1), so the build is reproducible.
- The rich platform entry `{ platform = "linux-64", cuda = "13" }` tells the solver that this machine has a CUDA 13 driver, which is what lets it pick GPU-enabled builds.
- `CMakeLists.txt` uses `nanobind_add_module` on a `.cu` source. CMake's CUDA language support drives `nvcc`; nanobind handles the Python side.

## Read the book

The `book/` directory is a [MyST](https://mystmd.org/) Jupyter Book that walks through every file and every command output in this repository, so it can be followed without a GPU. Build it with `pixi run docs-build`, which writes static HTML to `book/_build/html`, or serve it locally with live reload using `pixi run docs-start`.

## Requirements

Linux (x86-64) with an NVIDIA GPU and a CUDA 13 driver. No system CUDA toolkit needed.

To resolve or build on a machine without an NVIDIA driver (e.g. CI):

```console
CONDA_OVERRIDE_CUDA=13 pixi install
```
