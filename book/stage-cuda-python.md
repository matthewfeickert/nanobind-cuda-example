# Stage 3: cuda-python, the kernel compiled at runtime

The first hand-written kernel keeps the package pure Python.
[Stage 2](./stage-jax.md) got onto the GPU without one; here the hot loop becomes a CUDA kernel, but the kernel ships as source text and is compiled on the user's machine the first time the function is called.
The tool that makes this possible is [cuda-python](https://github.com/NVIDIA/cuda-python), NVIDIA's own Python bindings for CUDA, and specifically its `cuda.core` layer, which wraps NVRTC, device memory, streams, and kernel launches in Python objects.

## The kernel

```{literalinclude} ../templates/03-cuda-python/packages/pairwise-cuda-python/src/pairwise_cuda_python/pairwise.cu
:filename: templates/03-cuda-python/packages/pairwise-cuda-python/src/pairwise_cuda_python/pairwise.cu
:language: cpp
:linenos:
```

This file is the kernel and nothing else.
Each thread computes one output element by looping over the feature dimension, the same one-thread-per-element formulation that stage 1 expressed as a broadcast.
It is byte for byte the kernel that stage 4 will compile with `nvcc`; stage 4 adds the bindings around it, not a different kernel.

## Driving it from Python

```{literalinclude} ../templates/03-cuda-python/packages/pairwise-cuda-python/src/pairwise_cuda_python/__init__.py
:filename: templates/03-cuda-python/packages/pairwise-cuda-python/src/pairwise_cuda_python/__init__.py
:language: python
:linenos:
```

The `_kernel` helper is the compile step.
On first use it reads `pairwise.cu` from the installed package, hands it to NVRTC through a `Program` with the compute capability of the GPU that is actually present, and asks for both template instantiations by name.
The compiled module is cached for the rest of the process, so the cost is paid once and the benchmark's warm-up call absorbs it.

The body of `pairwise_distances` is what the C++ binding function in stage 4 does, written out in Python.
`cuda.core` copies between its own `Buffer` objects, so the NumPy inputs are first written into page-locked host buffers, which NumPy can view through DLPack, and then copied to device buffers on a stream.
The launch uses the same `(32, 8)` thread block as stage 4, the result comes back through another pinned buffer, and a `finally` block closes every allocation whether or not the launch succeeded.

## The package

```{literalinclude} ../templates/03-cuda-python/packages/pairwise-cuda-python/pyproject.toml
:filename: templates/03-cuda-python/packages/pairwise-cuda-python/pyproject.toml
:language: toml
:linenos:
```

```{literalinclude} ../templates/03-cuda-python/packages/pairwise-cuda-python/pixi.toml
:filename: templates/03-cuda-python/packages/pairwise-cuda-python/pixi.toml
:language: toml
:linenos:
```

The `pyproject.toml` gained one dependency, `cuda-core`, which the backend maps to the conda-forge package of the same name.
The Pixi manifest is still a noarch package with hatchling as its only host dependency.
It gained a `[package.run-dependencies]` table, and that table is the lesson of this stage.

A pure Python package links against nothing, so nothing tells conda which generation of CUDA it needs.
Without the `cuda-version = "13.*"` line the solver would be free to pick the CUDA 12 build of `cuda-core`, whose NVRTC could not target a driver from the CUDA 13 era, and the failure would only show up at first call.
The pin has to be written by hand.
Keep this in mind for the [distributing chapter](./distributing.md), where stage 4 gets the equivalent constraint for free.

## The workspace

```{literalinclude} ../templates/03-cuda-python/pixi.toml
:filename: templates/03-cuda-python/pixi.toml
:language: toml
:linenos:
```

Relative to stage 1 the `platforms` entry became the rich table `{ name = "linux-64-cuda", platform = "linux-64", cuda = "13" }`.
That declares the CUDA driver as the `__cuda` virtual package, without which `cuda-core`'s dependency on `cuda-version` could not be satisfied.
There is still no `[workspace.build-variants]` table, because there is still no compiler to pin.

## Running it

```{code} text
:filename: pixi run demo
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
[[0.    0.491 2.088 1.592 3.094]
 [0.491 0.    2.184 1.41  2.931]
 [2.088 2.184 0.    3.102 3.85 ]
 [1.592 1.41  3.102 0.    1.716]
 [3.094 2.931 3.85  1.716 0.   ]]
```

```{code} text
:filename: pixi run bench
GPU: NVIDIA GeForce RTX 4060 Laptop GPU

      n    d |     scipy cuda-python | speedup vs scipy
   1000   16 |    0.004s      0.003s |    1.6x
   4000   16 |    0.083s      0.036s |    2.3x
   8000   16 |    0.331s      0.140s |    2.4x
   8000  128 |    2.464s      0.254s |    9.7x
```

The naive kernel on a laptop GPU is already ahead of SciPy at every size and nearly an order of magnitude ahead at the largest, with the host transfers included.
The tests pass against `cdist` at the same tolerances as the other stages.

## What the package says it needs

```{code} console
:filename: shell
$ pixi publish --path packages/pairwise-cuda-python --target-channel ./local_channel
```

```{code} text
:filename: pixi publish (excerpt)
Resolved run dependencies(pairwise-cuda-python-0.1.0-pyh4616a5c_0):
╭──────────────────┬──────────────────────────╮
│ Name             ┆ Spec                     │
╞══════════════════╪══════════════════════════╡
│ Run dependencies ┆                          │
│ cuda-core        ┆ >=1.2                    │
│ cuda-version     ┆ 13.*                     │
│ numpy            ┆ >=2.3                    │
│ python           ┆ >=3.11                   │
│                  ┆ * (RE of [host: python]) │
╰──────────────────┴──────────────────────────╯

Files in package:
  ├─ site-packages/pairwise_cuda_python/__init__.py (4.07 KiB)
  ├─ site-packages/pairwise_cuda_python/pairwise.cu (555 B)
  ├─ site-packages/pairwise_cuda_python-0.1.0.dist-info/METADATA (1009 B)
  ├─ ...
  ├─ info/index.json (326 B)
  └─ info/paths.json (2.54 KiB)

Package statistics: 17 files (11 content, 6 metadata), total size: 11.33 KiB

📦 Publishing 1 package(s) to channel file:///tmp/nanobind-cuda-example/templates/03-cuda-python/local_channel
✔ Successfully published 1 package(s) to channel file:///tmp/nanobind-cuda-example/templates/03-cuda-python/local_channel
  - pairwise-cuda-python-0.1.0-pyh4616a5c_0.conda
```

The `file://` URL is the absolute path of the `local_channel` directory, reported back by Pixi; it starts with `/tmp` here only because the repository was cloned into `/tmp` when these outputs were captured.

The same [`rattler-build package inspect`](./distributing.md#what-the-package-says-it-needs) command the stage 4 chapter uses reads the metadata back from the archive.

```{code} console
:filename: shell
$ rattler-build package inspect $(find ./local_channel/ -type f -iname '*pairwise*.conda')
```

```{code} text
:filename: rattler-build package inspect (excerpt)
 Package: ./local_channel/noarch/pairwise-cuda-python-0.1.0-pyh4616a5c_0.conda (5.94 KiB)

 Run dependencies:
 ╭───────────────────╮
 │ Package           │
 ╞═══════════════════╡
 │ cuda-version 13.* │
 │ python >=3.11     │
 │ python *          │
 │ numpy >=2.3       │
 │ cuda-core >=1.2   │
 ╰───────────────────╯
```

The full `info/index.json` it summarises is short enough to read whole.

```{code} json
:filename: info/index.json
{
    "build": "pyh4616a5c_0",
    "build_number": 0,
    "depends": [
        "cuda-version 13.*",
        "python >=3.11",
        "python *",
        "numpy >=2.3",
        "cuda-core >=1.2"
    ],
    "license": "MIT",
    "name": "pairwise-cuda-python",
    "noarch": "python",
    "subdir": "noarch",
    "version": "0.1.0"
}
```

The package is eleven kilobytes, it is `noarch`, and the `.cu` file is sitting inside it as text.
Every entry in `depends` was written by a human in one of the two manifests, and the `RE of` column that fills the equivalent table for stage 4 is empty here apart from the Python pin.

This is a complete, working GPU package.
Whether to go on to [stage 4](./stage-nanobind.md) depends on who it is for, and that chapter opens with the trade-off.
