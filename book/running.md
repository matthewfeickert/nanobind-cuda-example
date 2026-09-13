# Running it

This chapter shows what happens when the workspace is used.
All outputs were captured on a laptop with an NVIDIA GeForce RTX 4060 Laptop GPU, driver 595.84, and Pixi 0.80.0.
Lines that only repeat information have been trimmed and the trimming is marked.

## The first `pixi install`

Installing the workspace resolves the environment and, because `gpu-pairwise` is a source dependency, builds it.
Pixi hands the build to the `pixi-build-python` backend, which in turn uses [rattler-build](https://rattler.build/).
The first thing the log shows is the three resolved dependency sets.

```{code} text
:filename: pixi install (excerpt)
Resolved build dependencies(gpu-pairwise-0.1.0-hb4504ce_0):
╭───────────────────────────────┬───────────────────────────┬──────────────┬──────────────────────┬─────────────┬────────────╮
│ Package                       ┆ Spec                      ┆ Version      ┆ Build                ┆ Channel     ┆       Size │
╞═══════════════════════════════╪═══════════════════════════╪══════════════╪══════════════════════╪═════════════╪════════════╡
│ cmake                         ┆ cmake >=3.26              ┆ 4.4.3        ┆ hff7ac6b_0           ┆ conda-forge ┆  24.87 MiB │
│ cuda-nvcc_linux-64            ┆ cuda-nvcc_linux-64 13.1.* ┆ 13.1.115     ┆ hb2fc203_2           ┆ conda-forge ┆  26.87 KiB │
│ gxx_linux-64                  ┆ gxx_linux-64              ┆ 15.3.0       ┆ h8c4937a_1           ┆ conda-forge ┆  27.50 KiB │
│ ninja                         ┆ ninja                     ┆ 1.13.2       ┆ h171cf75_1           ┆ conda-forge ┆ 182.39 KiB │
│ sysroot_linux-64              ┆ sysroot_linux-64 2.28.*   ┆ 2.28         ┆ h4ee821c_9           ┆ conda-forge ┆  22.90 MiB │
│ cuda-cccl_linux-64            ┆                           ┆ 13.1.115     ┆ ha770c72_0           ┆ conda-forge ┆   1.22 MiB │
│ cuda-cudart-dev_linux-64      ┆                           ┆ 13.1.80      ┆ h376f20c_0           ┆ conda-forge ┆ 383.26 KiB │
│ cuda-nvcc-tools               ┆                           ┆ 13.1.115     ┆ he02047a_0           ┆ conda-forge ┆  31.02 MiB │
│ cuda-nvvm-tools               ┆                           ┆ 13.1.115     ┆ h4bc722e_0           ┆ conda-forge ┆  24.45 MiB │
│ gcc_impl_linux-64             ┆                           ┆ 15.3.0       ┆ h6f13dc8_4           ┆ conda-forge ┆  79.01 MiB │
│ ...                           ┆                           ┆              ┆                      ┆             ┆            │
╰───────────────────────────────┴───────────────────────────┴──────────────┴──────────────────────┴─────────────┴────────────╯
Resolved host dependencies(gpu-pairwise-0.1.0-hb4504ce_0):
╭─────────────────────────────┬─────────────────────────────────────────────────────────────┬────────────┬──────────────────────┬─────────────┬────────────╮
│ Package                     ┆ Spec                                                        ┆ Version    ┆ Build                ┆ Channel     ┆       Size │
╞═════════════════════════════╪═════════════════════════════════════════════════════════════╪════════════╪══════════════════════╪═════════════╪════════════╡
│ cuda-cudart-dev             ┆ cuda-cudart-dev                                             ┆ 13.4.49    ┆ hd2095e1_0           ┆ conda-forge ┆  24.13 KiB │
│ cuda-version                ┆ cuda-version >=13.1,<14 (RE of [build: cuda-nvcc_linux-64]) ┆ 13.4       ┆ hfacf21a_3           ┆ conda-forge ┆  20.89 KiB │
│ libgcc                      ┆ libgcc >=15 (RE of [build: gxx_linux-64])                   ┆ 16.2.0     ┆ ha9f2e26_4           ┆ conda-forge ┆   1.01 MiB │
│ libstdcxx                   ┆ libstdcxx >=15 (RE of [build: gxx_linux-64])                ┆ 16.2.0     ┆ h934c35e_4           ┆ conda-forge ┆   6.31 MiB │
│ nanobind                    ┆ nanobind >=2.5                                              ┆ 3.0.1      ┆ pyhd8ed1ab_0         ┆ conda-forge ┆ 228.23 KiB │
│ python                      ┆ python >=3.11                                               ┆ 3.14.7     ┆ hcd007b5_106_cp314   ┆ conda-forge ┆  25.26 MiB │
│ scikit-build-core           ┆ scikit-build-core >=0.11                                    ┆ 1.0.3      ┆ pyh04d0eab_0         ┆ conda-forge ┆ 331.64 KiB │
│ uv                          ┆ uv                                                          ┆ 0.12.13    ┆ h841d291_0           ┆ conda-forge ┆  16.56 MiB │
│ ...                         ┆                                                             ┆            ┆                      ┆             ┆            │
╰─────────────────────────────┴─────────────────────────────────────────────────────────────┴────────────┴──────────────────────┴─────────────┴────────────╯
Resolved run dependencies(gpu-pairwise-0.1.0-hb4504ce_0):
╭──────────────────┬───────────────────────────────────────────────────╮
│ Name             ┆ Spec                                              │
╞══════════════════╪═══════════════════════════════════════════════════╡
│ __glibc          ┆ >=2.28,<3.0.a0 (RE of [build: sysroot_linux-64])  │
│ cuda-cudart      ┆ >=13.4.49,<14.0a0 (RE of [host: cuda-cudart-dev]) │
│ cuda-version     ┆ >=13.1,<14 (RE of [build: cuda-nvcc_linux-64])    │
│ libgcc           ┆ >=15 (RE of [build: gxx_linux-64])                │
│ libstdcxx        ┆ >=15 (RE of [build: gxx_linux-64])                │
│ numpy            ┆ >=2.3,<3                                          │
│ python           ┆ >=3.11                                            │
│ python_abi       ┆ 3.14.* *_cp314 (RE of [host: python])             │
╰──────────────────┴───────────────────────────────────────────────────╯
```

Three things are worth pausing on.

The build dependencies contain the `cuda-nvcc_linux-64` metapackage at exactly the `13.1.*` pinned by the workspace build variants, and it pulls the rest of the CUDA compiler chain with it.
The host dependencies contain the `cuda-cudart-dev`, `nanobind`, `scikit-build-core`, and `python` packages the manifest asked for, plus entries marked `RE of [...]`.
Those are run-exports: conda-forge packages that declare "anything built against me needs this at runtime".

The run dependencies table is entirely derived.
Only `numpy` and `python` come from `pyproject.toml`; every other entry was contributed by a run-export of a build or host dependency.
The `cuda-cudart-dev` host dependency became a `cuda-cudart` run dependency, and the `python` host dependency became a `python_abi` pin.

After resolution, CMake configures the extension using the compilers from the build environment and the Python from the host environment.

```{code} text
:filename: pixi install (excerpt)
-- The CXX compiler identification is GNU 15.3.0
-- The CUDA compiler identification is NVIDIA 13.1.115 with host compiler GNU 15.3.0
-- Check for working CXX compiler: $BUILD_PREFIX/bin/x86_64-conda-linux-gnu-c++ - skipped
-- Check for working CUDA compiler: $BUILD_PREFIX/bin/nvcc - skipped
-- Found Python: $PREFIX/bin/python (found suitable version "3.14.7", minimum required is "3.11") found components: Interpreter Development.Module
```

Both compilers are found under `$BUILD_PREFIX`, which is the conda build environment.
No path outside the Pixi managed directories is consulted.
On this laptop the compile for all major GPU architectures took about 25 seconds, and the whole first install about 30 seconds.

## `pixi run demo`

```{literalinclude} ../scripts/demo.py
:filename: scripts/demo.py
:language: python
```

```{code} text
:filename: pixi run demo
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
[[0.    0.491 2.088 1.592 3.094]
 [0.491 0.    2.184 1.41  2.931]
 [2.088 2.184 0.    3.102 3.85 ]
 [1.592 1.41  3.102 0.    1.716]
 [3.094 2.931 3.85  1.716 0.   ]]
```

The matrix is symmetric with a zero diagonal, which is the quickest sanity check for a distance matrix.

## `pixi run test`

```{code} text
:filename: pixi run test
============================= test session starts ==============================
platform linux -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0 -- /tmp/nanobind-cuda-example/.pixi/envs/test/bin/python3.14
rootdir: /tmp/nanobind-cuda-example/src/gpu-pairwise
configfile: pyproject.toml
collecting ... collected 9 items

src/gpu-pairwise/tests/test_pairwise.py::test_matches_scipy[1-1-1-float32] PASSED [ 11%]
src/gpu-pairwise/tests/test_pairwise.py::test_matches_scipy[1-1-1-float64] PASSED [ 22%]
src/gpu-pairwise/tests/test_pairwise.py::test_matches_scipy[7-5-3-float32] PASSED [ 33%]
src/gpu-pairwise/tests/test_pairwise.py::test_matches_scipy[7-5-3-float64] PASSED [ 44%]
src/gpu-pairwise/tests/test_pairwise.py::test_matches_scipy[257-130-16-float32] PASSED [ 55%]
src/gpu-pairwise/tests/test_pairwise.py::test_matches_scipy[257-130-16-float64] PASSED [ 66%]
src/gpu-pairwise/tests/test_pairwise.py::test_y_defaults_to_x PASSED     [ 77%]
src/gpu-pairwise/tests/test_pairwise.py::test_non_contiguous_and_integer_inputs PASSED [ 88%]
src/gpu-pairwise/tests/test_pairwise.py::test_feature_mismatch_raises PASSED [100%]

============================== 9 passed in 0.53s ===============================
```

The tests run in the separate `test` environment, which is why the interpreter path contains `.pixi/envs/test`.
Pixi built the package once and installed the same artifact into both environments.

## `pixi run bench`

```{literalinclude} ../scripts/bench.py
:filename: scripts/bench.py
:language: python
```

Each row times the full `n × n` distance matrix for `n` points in `d` dimensions, held as a float32 array of shape `(n, d)`, so the work grows as `n² · d`.
The CUDA column includes the host to device and device to host copies.
The NumPy broadcasting reference is skipped for the larger `n` because its `(n, n, d)` intermediate would need tens of GB of memory.

```{code} text
:filename: pixi run bench
GPU: NVIDIA GeForce RTX 4060 Laptop GPU

      n    d |     numpy     scipy   sklearn      cuda | speedup vs scipy
   1000   16 |    0.049s    0.006s    0.013s    0.001s |    6.2x
   4000   16 |    3.256s    0.118s    0.157s    0.042s |    2.8x
   8000   16 |   skipped    1.609s    1.181s    0.161s |   10.0x
   8000  128 |   skipped    4.442s    0.674s    0.321s |   13.8x
```

A naive kernel on a laptop GPU beats SciPy's compiled C loop by an order of magnitude at the larger sizes, and it does so with the memory traffic included.
scikit-learn is closer because it reformulates the problem as a matrix product and hands it to a multithreaded BLAS, which is the same trick that would make the CUDA version far faster still.
