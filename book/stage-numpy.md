# Stage 1: NumPy, the CPU baseline

The starting point is a pure Python package with NumPy as its only dependency.
It is where most scientific Python projects begin, and it is also the first packaging story, because a package that ships nothing but `.py` files is the simplest thing Pixi Build can produce.

## The implementation

```{literalinclude} ../templates/01-numpy/packages/pairwise-numpy/src/pairwise_numpy/__init__.py
:filename: templates/01-numpy/packages/pairwise-numpy/src/pairwise_numpy/__init__.py
:language: python
:linenos:
```

The distance matrix is a broadcast subtraction, `(chunk, 1, d) - (1, m, d)`, squared and summed over the last axis.
Done in one shot the intermediate would have shape `(n, m, d)`, which at `n = m = 8000` and `d = 128` is 33 GB of float32, so the function walks `x` in chunks of rows and only ever materialises `(256, m, d)`.
That bound is what makes the baseline usable at the sizes the benchmark uses.

The input handling at the top of the function, `y` defaulting to `x`, the float32 or float64 rule, and the C-contiguous copy, is repeated verbatim in the later stages.
There it feeds a kernel with strict expectations; here it only keeps the behaviour identical.

## The package

```{literalinclude} ../templates/01-numpy/packages/pairwise-numpy/pyproject.toml
:filename: templates/01-numpy/packages/pairwise-numpy/pyproject.toml
:language: toml
:linenos:
```

```{literalinclude} ../templates/01-numpy/packages/pairwise-numpy/pixi.toml
:filename: templates/01-numpy/packages/pairwise-numpy/pixi.toml
:language: toml
:linenos:
```

The `pyproject.toml` is an ordinary [hatchling](https://hatch.pypa.io/latest/) project.
The Pixi package manifest next to it is shorter than the one in stage 4 in two ways.
There is no `config.compilers` line, and without one the `pixi-build-python` backend produces a `noarch: python` package that installs on any platform.
There is also no `cuda-cudart-dev` host dependency, because there is nothing to link against; the only host dependency is `hatchling` itself, which has to be there because the backend installs with `--no-build-isolation` and so never fetches the build backend from PyPI.

## The workspace

```{literalinclude} ../templates/01-numpy/pixi.toml
:filename: templates/01-numpy/pixi.toml
:language: toml
:linenos:
```

The `platforms` entry is the plain string `"linux-64"`.
Nothing in this workspace needs a GPU driver, so there is no `cuda = "13"` and no `__cuda` virtual package for the solver to consider.
Compare this with the [stage 3 workspace](./stage-cuda-python.md), which is one line different.

## Running it

```{literalinclude} ../templates/01-numpy/scripts/demo.py
:filename: templates/01-numpy/scripts/demo.py
:language: python
```

```{code} text
:filename: pixi run demo
[[0.    0.491 2.088 1.592 3.094]
 [0.491 0.    2.184 1.41  2.931]
 [2.088 2.184 0.    3.102 3.85 ]
 [1.592 1.41  3.102 0.    1.716]
 [3.094 2.931 3.85  1.716 0.   ]]
```

This matrix is the reference the GPU stages have to reproduce, and their demos print exactly the same numbers.

```{code} text
:filename: pixi run bench
      n    d |     numpy     scipy | numpy / scipy
   1000   16 |    0.013s    0.004s |    3.1x
   4000   16 |    0.309s    0.082s |    3.8x
   8000   16 |    1.310s    0.328s |    4.0x
   8000  128 |    7.735s    2.387s |    3.2x
```

NumPy broadcasting is three to four times slower than `scipy.spatial.distance.cdist`, which is a compiled C loop over the same data.
That is the hot loop the next three stages move to the GPU.
