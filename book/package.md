# The package

Everything under `packages/gpu-pairwise` describes one conda package.
Three files split the job.
The Pixi package manifest says which build backend to use and what the build needs.
The `pyproject.toml` carries the Python metadata and runtime dependencies.
The `CMakeLists.txt` compiles the extension.

## The package manifest

```{literalinclude} ../templates/03-nanobind-cuda/packages/gpu-pairwise/pixi.toml
:filename: templates/03-nanobind-cuda/packages/gpu-pairwise/pixi.toml
:language: toml
:linenos:
```

There is no `[package]` table with a name and version because the [`pixi-build-python`](https://pixi.prefix.dev/latest/build/backends/pixi-build-python/) backend reads those from `pyproject.toml`.
Only the `[package.build]` table and the dependency tables are needed.

Three configuration keys matter.

- `config.compilers = ["cxx", "cuda"]` puts a C++ compiler and `nvcc` into the build environment, and switches the package from `noarch` to platform specific.
- `config.ignore-pypi-mapping = false` maps the PyPI names in `[project.dependencies]` onto conda-forge package names, so `numpy` on the Python side becomes the conda-forge `numpy` in the package's runtime dependencies.
- `config.extra-input-globs` lists files beyond the Python sources that should trigger a rebuild when they change, which here means the kernel and the CMake configuration.

## Build, host, and run dependencies

Conda packaging distinguishes three dependency roles, and the manifest uses two of them.
Build dependencies are tools that run on the build machine, which here means CMake and Ninja.
Host dependencies are libraries and headers the package is compiled against, and it is where the CUDA runtime enters.

The Python build backend runs `uv pip install --no-build-isolation`.
Without build isolation, `pip` does not fetch scikit-build-core and nanobind from PyPI, so they have to be present in the host environment already.
That is why they are listed as host dependencies alongside `cuda-cudart-dev`.

The third role, run dependencies, is not written down anywhere in this manifest.
The [distributing chapter](./distributing.md) shows how they are derived from the host dependencies through conda-forge's run-export metadata.

## The Python project

```{literalinclude} ../templates/03-nanobind-cuda/packages/gpu-pairwise/pyproject.toml
:filename: templates/03-nanobind-cuda/packages/gpu-pairwise/pyproject.toml
:language: toml
:linenos:
```

This is a completely ordinary scikit-build-core project.
Nothing in it mentions Pixi or conda, and `pip install .` inside an environment with a CUDA toolchain would work just as well.
That is the design goal: Pixi Build packages the project you already have rather than asking you to restructure it.

The `[tool.scikit-build]` table stops scikit-build-core from fetching its own CMake and Ninja wheels from PyPI, because the conda build environment already provides both.
It also passes `-GNinja` to CMake explicitly.
scikit-build-core prefers Ninja only when no `CMAKE_GENERATOR` is set, and rattler-build's build environment sets it to Unix Makefiles, so without this line the extension would be compiled with Make even though Ninja is sitting in the build environment; see [rattler-build issue 2487](https://github.com/prefix-dev/rattler-build/issues/2487) and the [scikit-build Pixi guide](https://scikit-build.org/SIMPLE-Py/pixi-build/).

## The CMake configuration

```{literalinclude} ../templates/03-nanobind-cuda/packages/gpu-pairwise/CMakeLists.txt
:filename: templates/03-nanobind-cuda/packages/gpu-pairwise/CMakeLists.txt
:language: cmake
:linenos:
```

Declaring `CUDA` as a project language is all CMake needs to drive `nvcc` for any `.cu` source.
The `Development.Module` component asks only for Python headers, never for `libpython`, which is the correct choice for an extension module.
The `nanobind_add_module` call takes the `.cu` file directly, and `NB_STATIC` links nanobind's small support library statically so the extension has no extra shared library to ship.

Setting `CMAKE_CUDA_ARCHITECTURES` to `all-major` compiles device code for every major GPU generation nvcc knows about.
That makes the package portable at the cost of compile time, and the comment shows how to override it with `native` while iterating locally.
