# Templates

Three self-contained Pixi workspaces, one per stage of speeding up the same Python package. Each has its own `pixi.toml`, lock file, scripts, tests, and one source package, and each exposes the same `pairwise_distances(x, y=None)` function so the three are drop-in replacements for each other.

| Stage | Directory | What it is | Package type |
|---|---|---|---|
| 1 | [`01-numpy`](01-numpy) | NumPy broadcasting on the CPU. The baseline and the loop to port. | `noarch: python` |
| 2 | [`02-cuda-python`](02-cuda-python) | The CUDA kernel shipped as source, compiled at first use by NVRTC through `cuda.core`. | `noarch: python`, CUDA pin written by hand |
| 3 | [`03-nanobind-cuda`](03-nanobind-cuda) | The same kernel compiled by `nvcc` at build time into a nanobind extension. | `linux-64`, CUDA pin derived from run-exports |

Copy a directory out of the repository to start from that stage:

```console
cp -r templates/02-cuda-python my-project
cd my-project && pixi run test
```

Inside each directory, `pixi run demo`, `pixi run bench`, and `pixi run test` do the same thing, and `pixi publish --path packages/<package> --target-channel ./local_channel` builds the package into a local conda channel that any other Pixi workspace can install from.
