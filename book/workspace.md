# The workspace

Pixi uses one manifest format, `pixi.toml`, for two different roles.
A manifest with a `[workspace]` table describes your development environment: channels, platforms, dependencies, and tasks.
A manifest with a `[package]` table and no `[workspace]` table describes how to build one distributable conda package.
The [Pixi manifest reference](https://pixi.prefix.dev/latest/reference/pixi_manifest/) covers both roles in full.

This chapter is about the workspace.
It lives at the top of the repository.

```{literalinclude} ../pixi.toml
:filename: pixi.toml
:language: toml
:linenos:
```

## Declaring a CUDA platform

The `platforms` entry is not the plain string `"linux-64"` but a table.
The `cuda = "13"` field declares that machines using this workspace have a CUDA 13 capable driver, which Pixi exposes to the solver as the `__cuda` [virtual package](https://pixi.prefix.dev/latest/workspace/multi_platform_configuration/#declaring-virtual-packages-per-platform).
Without it, the solver would have no reason to pick GPU enabled builds of anything, and conda-forge's CUDA packages would be unsatisfiable.
The lock file records the resulting environment for the platform name `linux-64-cuda-13`.

## Source dependencies

The line `gpu-pairwise = { path = "src/gpu-pairwise" }` is what makes this a Pixi Build workspace rather than an ordinary one.
It tells Pixi that `gpu-pairwise` is not to be downloaded from a channel but built from the manifest in that directory.
Every `pixi install` and every `pixi run` checks whether the package's input files have changed and rebuilds it if so.
This is why the workspace needs `preview = ["pixi-build"]`.

The remaining dependencies, NumPy, SciPy, and scikit-learn, are the CPU references used by the benchmark and the tests.
Python itself is not pinned, and the solver picked Python 3.14 in the lock file.

## Tasks and environments

Three tasks give the reader something to run: `demo`, `bench`, and `test`.
The `test` task lives in a `test` feature with pytest as an extra dependency and is exposed through the `test` environment, so pytest never ends up in the default environment.
The `docs` feature does the same for this book with [mystmd](https://mystmd.org/).

## Build variants

The last table is the one that ties the workspace to the CUDA toolchain.

```{literalinclude} ../pixi.toml
:filename: pixi.toml
:language: toml
:start-at: workspace.build-variants
```

The package manifest in the next chapter asks for a `cuda` compiler by name, and the build backend has to resolve that name to a concrete conda-forge package.
Build variants pin that resolution.
On conda-forge, the CUDA 12 and later compilers live in the `cuda-nvcc` packages, and `cuda_compiler_version` pins which release is used.
Changing this one line is how you would rebuild the same package against a different CUDA toolkit.
