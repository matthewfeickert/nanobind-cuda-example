# Packaging a CUDA kernel for Python with Pixi Build

This is the companion book for the [`nanobind-cuda-example`](https://github.com/matthewfeickert/nanobind-cuda-example) repository.
It steps through one small but complete example of a very common situation in research computing.
You have a hot loop in Python.
You have written, or are about to write, a CUDA kernel that replaces it.
Now you need to hand that kernel to a collaborator, a student, or a cluster, without a hand-written "first install the CUDA toolkit" document.

The example package is called `gpu-pairwise`.
It computes the full matrix of Euclidean distances between the rows of two arrays, which is the first step of nearest neighbour search, clustering, and kernel methods.
The kernel is deliberately naive, with one CUDA thread per output element, because the packaging is the point of the example, not the kernel.

Three tools do the work.

- [nanobind](https://nanobind.readthedocs.io/) exposes the C++ and CUDA code to Python as an extension module that accepts and returns NumPy arrays.
- [scikit-build-core](https://scikit-build-core.readthedocs.io/) is the Python build backend that drives CMake for that extension.
- [Pixi Build](https://pixi.prefix.dev/latest/build/) wraps the whole thing into a conda package where `nvcc`, the CUDA runtime, the C++ compiler, CMake, Ninja, and Python all come from conda-forge.

The result is a `.conda` file whose metadata records exactly which CUDA runtime and which Python ABI it needs.
Nobody who installs it ever installs a system CUDA toolkit.

## How to read this book

Every code listing is included directly from the repository, so what you read is what gets built.
Every command output is captured from a real run on a laptop with an NVIDIA RTX 4060 GPU and a CUDA 13 driver.
That means you can follow the whole story without running anything.
If you do have an NVIDIA GPU and [Pixi](https://pixi.prefix.dev/latest/installation/) installed, cloning the repository and running `pixi run demo` reproduces the first chapter's output.

:::{note} Preview feature
Pixi Build is currently a [preview feature](https://pixi.prefix.dev/latest/reference/pixi_manifest/#preview-features).
Its manifest surface is still evolving, so the backends are pinned in the manifests and small changes between Pixi versions should be expected.
:::

## Repository layout

```
nanobind-cuda-example/
├── pixi.toml                    # the workspace: dependencies + tasks
├── book/                        # this book
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

The chapters follow the layout from the outside in.
[The workspace](./workspace.md) is what you `pixi run`.
[The package](./package.md) is what Pixi turns into a `.conda` file.
[The kernel and its bindings](./kernel.md) are the code being packaged.
[Running it](./running.md) shows the build, the demo, the tests, and the benchmark.
[Distributing it](./distributing.md) builds the standalone package and reads its metadata.
[Next steps](./next-steps.md) lists what this example leaves out and where to go from here.
