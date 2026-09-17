# Stage 2: JAX, the GPU without a kernel

Before writing a kernel it is worth asking whether one is needed.
[JAX](https://docs.jax.dev/) offers the NumPy API on the GPU: the same broadcast that stage 1 evaluates on the CPU is handed to XLA, which compiles it into GPU kernels of its own choosing.
The package stays pure Python and the author never sees a line of CUDA.

## The implementation

```{literalinclude} ../templates/02-jax/packages/pairwise-jax/src/pairwise_jax/__init__.py
:filename: templates/02-jax/packages/pairwise-jax/src/pairwise_jax/__init__.py
:language: python
:linenos:
```

The input handling is the stage 1 wrapper again, with one JAX-specific line: JAX computes in float32 unless `jax_enable_x64` is switched on, so the wrapper switches it on when the caller passed float64.
The result is copied back into an ordinary NumPy array, after `block_until_ready`, so the return type and the timings are comparable with the other stages.

The `_pairwise` function is where the port happens.
One row of `x` against all of `y` is a subtract, a square, and a reduction over the feature axis, and XLA fuses the three into a single kernel so the `(n, m, d)` intermediate that forces stage 1 to chunk never exists on the device.
That fusion is the whole speedup, and it is XLA's, not ours.

## What XLA decides for you

The row loop is written as `lax.scan` rather than the more obvious `vmap`.
Both produce the same numbers, and the vmap version is the one most JAX users would write first.
On this laptop, though, XLA's GPU autotuner spent 42 seconds compiling the vmap version for the smallest benchmark size and was still compiling the `n = m = 8000`, `d = 128` case after four minutes, at which point its own "very slow compile" alarm had fired and the GPU had sat idle throughout.
The scan version compiles the same reduction for one row at a time and takes a third of a second.

That is the trade stage 2 makes.
The kernel is free, but so is the choice of how it is compiled, and when XLA's heuristics go wrong the remedies are indirect: a different formulation, as here, or an `XLA_FLAGS` environment variable such as `--xla_gpu_autotune_level=0` that a library cannot reasonably set for its users.
Stages 3 and 4 write the kernel by hand and give up the convenience for control.

Two more things a first-time JAX user notices, both visible in `nvtop` before a single kernel has run.
When its CUDA backend starts, JAX preallocates 75% of device memory, six of the eight gigabytes here; `XLA_PYTHON_CLIENT_PREALLOCATE=false` turns that off, and the umbrella workspace's `bench` task sets it so the other stages can get at the device.
Even then, JAX's runtime keeps every byte it has touched for the life of the process: after a call whose result is four gigabytes, those four gigabytes stay allocated with no live array to show for it, and the cross-stage benchmark has to time JAX last in each row for the kernel stages to fit.
The first call at each new input shape also triggers a compile, which is why every benchmark in this book warms up before timing.

## The package

```{literalinclude} ../templates/02-jax/packages/pairwise-jax/pyproject.toml
:filename: templates/02-jax/packages/pairwise-jax/pyproject.toml
:language: toml
:linenos:
```

```{literalinclude} ../templates/02-jax/packages/pairwise-jax/pixi.toml
:filename: templates/02-jax/packages/pairwise-jax/pixi.toml
:language: toml
:linenos:
```

The `pyproject.toml` depends on `jax`, which on both PyPI and conda-forge is the pure Python front end.
The GPU support lives in `jaxlib`, which conda-forge builds once per CUDA generation and once for the CPU, and nothing in `pyproject.toml` can say which of those a package wants.
So the Pixi manifest says it, in a `[package.run-dependencies]` table, as a build string constraint on `jaxlib` plus the same `cuda-version` pin stage 3 will need.
This is the first stage where the package has to know about CUDA, and it has to be told by hand.

## The workspace

```{literalinclude} ../templates/02-jax/pixi.toml
:filename: templates/02-jax/pixi.toml
:language: toml
:linenos:
```

Relative to stage 1 the `platforms` entry became the rich table with `cuda = "13"`.
That declares the CUDA driver as the `__cuda` virtual package, which jaxlib's CUDA builds require, and it is the same line stages 3 and 4 carry.

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

      n    d |     scipy         jax | speedup vs scipy
   1000   16 |    0.004s      0.008s |    0.6x
   4000   16 |    0.083s      0.046s |    1.8x
   8000   16 |    0.345s      0.131s |    2.6x
   8000  128 |    2.510s      0.174s |   14.4x
```

At the smallest size the fixed cost of dispatching through JAX and copying to and from the device outweighs the work and SciPy wins.
At `d = 128` XLA's fused reduction is the fastest thing in this book, faster than the hand-written kernels of stages 3 and 4, which the [comparison chapter](./stages.md) shows side by side.

## What the package says it needs

```{code} console
:filename: shell
$ pixi publish --path packages/pairwise-jax --target-channel ./local_channel
$ rattler-build package inspect $(find ./local_channel/ -type f -iname '*pairwise*.conda')
```

```{code} text
:filename: rattler-build package inspect (excerpt)
 Package: ./local_channel/noarch/pairwise-jax-0.1.0-pyh4616a5c_0.conda (4.81 KiB)

 Run dependencies:
 ╭───────────────────╮
 │ Package           │
 ╞═══════════════════╡
 │ jaxlib * cuda13*  │
 │ cuda-version 13.* │
 │ python >=3.11     │
 │ python *          │
 │ numpy >=2.3       │
 │ jax >=0.10        │
 ╰───────────────────╯
```

The package is five kilobytes and `noarch`.
Its two CUDA constraints, the jaxlib build string and the `cuda-version` pin, were both typed into the manifest by hand, which is the pattern stage 3 repeats and stage 4 does away with.
