# Distributing it

So far the package has only existed inside the workspace's environments.
This chapter builds it as a standalone `.conda` file and reads back what Pixi recorded in it.

## `pixi build`

```{code} console
:filename: shell
$ pixi build --path src/gpu-pairwise -o dist
```

The output is the same build the workspace ran on first install, followed by the packaging step.

```{code} text
:filename: pixi build (excerpt)
Files in package:
  ├─ lib/python3.14/site-packages/gpu_pairwise/__init__.py (1.10 KiB)
  ├─ lib/python3.14/site-packages/gpu_pairwise/__pycache__/__init__.cpython-314.pyc (1.48 KiB)
  ├─ lib/python3.14/site-packages/gpu_pairwise/_core.cpython-314-x86_64-linux-gnu.so (992.05 KiB)
  ├─ lib/python3.14/site-packages/gpu_pairwise-0.1.0.dist-info/INSTALLER (6 B)
  ├─ lib/python3.14/site-packages/gpu_pairwise-0.1.0.dist-info/METADATA (817 B)
  ├─ lib/python3.14/site-packages/gpu_pairwise-0.1.0.dist-info/RECORD (991 B)
  ├─ lib/python3.14/site-packages/gpu_pairwise-0.1.0.dist-info/REQUESTED (0 B)
  ├─ lib/python3.14/site-packages/gpu_pairwise-0.1.0.dist-info/WHEEL (108 B)
  ├─ lib/python3.14/site-packages/gpu_pairwise-0.1.0.dist-info/direct_url.json (74 B)
  ├─ lib/python3.14/site-packages/gpu_pairwise-0.1.0.dist-info/licenses/LICENSE (1.05 KiB)
  ├─ lib/python3.14/site-packages/gpu_pairwise-0.1.0.dist-info/uv_build.json (2 B)
  ├─ lib/python3.14/site-packages/gpu_pairwise-0.1.0.dist-info/uv_cache.json (194 B)
  ├─ info/about.json (238 B)
  ├─ info/hash_input.json (145 B)
  ├─ info/index.json (447 B)
  ├─ info/paths.json (2.89 KiB)
  └─ info/tests/tests.yaml (3 B)

Package statistics: 17 files (12 content, 5 metadata), total size: 1001.52 KiB

📦 Publishing 1 package(s) to directory dist
✔ Successfully published 1 package(s) to directory dist
  - gpu-pairwise-0.1.0-hb4504ce_0.conda
```

The package is under one megabyte and almost all of it is the extension module, which carries device code for every major GPU architecture.
The layout is a normal conda Python package: a `site-packages` directory with the module and its wheel metadata, plus the `info` directory that conda tooling reads.

## What the package says it needs

The `info/index.json` file inside the archive is the metadata every conda compatible installer reads before installing.
Its `depends` list is the payoff for the whole exercise.

```{code} json
:filename: info/index.json
{
    "arch": "x86_64",
    "build": "hb4504ce_0",
    "build_number": 0,
    "depends": [
        "python >=3.11",
        "numpy >=2.3,<3",
        "libstdcxx >=15",
        "libgcc >=15",
        "cuda-version >=13.1,<14",
        "__glibc >=2.28,<3.0.a0",
        "cuda-cudart >=13.4.49,<14.0a0",
        "python_abi 3.14.* *_cp314"
    ],
    "license": "MIT",
    "name": "gpu-pairwise",
    "platform": "linux",
    "subdir": "linux-64",
    "timestamp": 1789279400235,
    "version": "0.1.0"
}
```

Read it as a contract.
The package needs the CUDA 13 runtime library, at least the version it was linked against, but not the toolkit and not `nvcc`.
It needs a CUDA driver compatible with CUDA 13.1 or later, which `cuda-version` expresses and which the solver checks against the `__cuda` virtual package on the installing machine.
It needs CPython 3.14 specifically, because the extension module was compiled against that ABI.
It needs a glibc no older than the sysroot it was built with.

None of that was written by hand.
The manifest listed one CUDA host dependency and let conda-forge's run-exports do the rest.
Rebuilding for a different Python or CUDA version is a matter of changing the pins in the workspace and running `pixi build` again.

## Where the package can go

The same `.conda` file can be published to a channel with [`pixi publish`](https://pixi.prefix.dev/latest/build/publishing/), for example a channel on [prefix.dev](https://prefix.dev/), and then depended on from any other Pixi workspace or conda environment.
It can also be installed as a tool with `pixi global install`, which creates an isolated environment that satisfies exactly the dependency list above.
Either way, the person on the other end gets a working CUDA extension without a toolkit install, because the runtime library is itself a conda-forge package.
