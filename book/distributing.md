# Distributing it

So far the package has only existed inside the workspace's environments.
This chapter builds it as a `.conda` file in a local channel and reads back what Pixi recorded in it.

## `pixi publish` to a local channel

Pixi has one command for turning a package manifest into a `.conda` file, and it is [`pixi publish`](https://pixi.prefix.dev/latest/reference/cli/pixi/publish/).
The older `pixi build` is deprecated in its favour.
The target can be a hosted channel, but for inspecting a build it can also be a [directory on the local filesystem](https://pixi.prefix.dev/latest/reference/cli/pixi/publish/#publishing-to-a-local-filesystem-channel), which Pixi creates and indexes as a real conda channel.
The name `local_channel` follows the direction of [pixi issue #6600](https://github.com/prefix-dev/pixi/issues/6600), which proposes it as the default, and the repository's `.gitignore` already excludes it.

```{code} console
:filename: shell
$ pixi publish --path templates/03-nanobind-cuda/packages/gpu-pairwise --target-channel ./local_channel
```

The output is the same build the workspace ran on first install, followed by the packaging and indexing steps.

```{code} text
:filename: pixi publish (excerpt)
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

📦 Publishing 1 package(s) to channel file:///tmp/nanobind-cuda-example/local_channel
✔ Successfully published 1 package(s) to channel file:///tmp/nanobind-cuda-example/local_channel
  - gpu-pairwise-0.1.0-hb4504ce_0.conda
```

:::{note} The path in the output is yours, not ours
The relative path `./local_channel` is resolved against the current directory, and Pixi reports it back as an absolute `file://` URL.
The outputs in this book were captured after cloning the repository into `/tmp`, which is why they show `file:///tmp/nanobind-cuda-example/local_channel`.
On your machine the URL will be the absolute path of the `local_channel` directory wherever you cloned the repository, and that is the URL to use when pointing another workspace at the channel below.
:::

```{code} text
:filename: local_channel/
local_channel/
├── linux-64/
│   ├── gpu-pairwise-0.1.0-hb4504ce_0.conda
│   ├── repodata.json
│   ├── repodata.json.zst
│   ├── repodata_shards.msgpack.zst
│   └── shards/
└── noarch/
    └── repodata.json
```

The package went into the `linux-64` subdirectory because it is platform specific; the stage 2 package, being `noarch`, lands under `noarch/` instead.
Next to it Pixi wrote `repodata.json`, the index that every conda compatible solver reads, in both its plain and its sharded form.
That is what makes the directory a channel rather than a folder of files.

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
Compare the `depends` list of the [stage 2 package](./stage-cuda-python.md), where the only CUDA constraint is the one its author remembered to type.
Rebuilding for a different Python or CUDA version is a matter of changing the pins in the workspace and running `pixi publish` again.

## Installing from the local channel

Because `local_channel` is a real channel, any other Pixi workspace can list it next to conda-forge and depend on the package as if it had been downloaded.
The workspace still has to declare the CUDA driver, because the package's `cuda-version` constraint is checked against the `__cuda` virtual package like any other.

The channel URL below is the absolute path to the `local_channel` directory created in the previous section.
It only matches if you cloned the repository into `/tmp`, so substitute the `file://` URL that `pixi publish` printed for you, or build it from `$(pwd)/local_channel` while standing in the repository root.

```{code} console
:filename: shell
$ pixi init --channel file:///tmp/nanobind-cuda-example/local_channel --channel conda-forge consumer
$ cd consumer
$ sed -i 's/platforms = \["linux-64"\]/platforms = [{ name = "linux-64-cuda", platform = "linux-64", cuda = "13" }]/' pixi.toml
$ pixi add gpu-pairwise
✔ Added gpu-pairwise >=0.1.0,<0.2
$ pixi run python -c "import numpy as np, gpu_pairwise; print(gpu_pairwise.pairwise_distances(np.eye(3)))"
[[0.         1.41421356 1.41421356]
 [1.41421356 0.         1.41421356]
 [1.41421356 1.41421356 0.        ]]
```

Nothing in the consuming workspace mentions nanobind, CMake, or `nvcc`.
The solver read the `depends` list above, pulled `cuda-cudart` and the rest from conda-forge, and installed the extension module.

## Where the package can go

Replacing the local path with a hosted channel, for example one on [prefix.dev](https://prefix.dev/), publishes the same `.conda` file for real; the [publishing guide](https://pixi.prefix.dev/latest/build/publishing/) lists the supported targets.
The package can also be installed as a tool with `pixi global install`, which creates an isolated environment that satisfies exactly the dependency list above.
Either way, the person on the other end gets a working CUDA extension without a toolkit install, because the runtime library is itself a conda-forge package.
