# The kernel and its bindings

The whole extension is one file, `src/pairwise.cu`, plus a small Python wrapper.
This chapter walks through it in four pieces.

## Error handling and device memory

```{literalinclude} ../templates/03-nanobind-cuda/packages/gpu-pairwise/src/pairwise.cu
:filename: templates/03-nanobind-cuda/packages/gpu-pairwise/src/pairwise.cu
:language: cpp
:lines: 22-36
```

CUDA runtime calls report failure through return codes.
Wrapping them in `cuda_check` turns any failure into a C++ exception, which nanobind translates into a Python `RuntimeError` with the CUDA error string in its message.
The `DeviceBuffer` type frees its allocation in its destructor, so an exception thrown part way through a call cannot leak device memory.

## The kernel

```{literalinclude} ../templates/03-nanobind-cuda/packages/gpu-pairwise/src/pairwise.cu
:filename: templates/03-nanobind-cuda/packages/gpu-pairwise/src/pairwise.cu
:language: cpp
:lines: 38-53
```

The kernel is templated on the floating point type so that one definition serves both float32 and float64.
Each thread computes one entry of the output by looping over the feature dimension.
This is the textbook formulation and it is far from optimal, since every thread re-reads its row of `x` and column of `y` from global memory.
It is enough to show a real speedup over the CPU references, and the [next steps](./next-steps.md) chapter points at the better formulations.

## The binding function

```{literalinclude} ../templates/03-nanobind-cuda/packages/gpu-pairwise/src/pairwise.cu
:filename: templates/03-nanobind-cuda/packages/gpu-pairwise/src/pairwise.cu
:language: cpp
:lines: 55-83
```

The argument types are the most important lines in the file.
A nanobind `ndarray` annotated with `nb::ndim<2>`, `nb::c_contig`, and `nb::device::cpu` tells nanobind to accept only two dimensional, row-major, host resident arrays of the given element type.
Anything else raises a `TypeError` before the C++ body runs, so the kernel never sees a stride it does not understand.

The body copies both inputs to the device, launches a two dimensional grid of thread blocks, waits for it to finish, and copies the result back into a freshly allocated host buffer.
The `nb::capsule` hands ownership of that buffer to the returned NumPy array, so Python frees it when the array is garbage collected.

## The module definition

```{literalinclude} ../templates/03-nanobind-cuda/packages/gpu-pairwise/src/pairwise.cu
:filename: templates/03-nanobind-cuda/packages/gpu-pairwise/src/pairwise.cu
:language: cpp
:lines: 87-101
```

Registering the float and double instantiations under the same name gives one Python function that dispatches on the input dtype.
The `device_name` helper exists so the demo and benchmark can say which GPU they ran on.

## The Python wrapper

```{literalinclude} ../templates/03-nanobind-cuda/packages/gpu-pairwise/src/gpu_pairwise/__init__.py
:filename: templates/03-nanobind-cuda/packages/gpu-pairwise/src/gpu_pairwise/__init__.py
:language: python
:linenos:
```

The strict argument types in C++ are only comfortable to use because this thin wrapper normalises inputs first.
It lets `y` default to `x`, promotes integer and float16 inputs to float32, preserves float64 when the caller opted into it, and makes both arrays C-contiguous.
Doing this in Python keeps the C++ side simple and the behaviour easy to test.

## The tests

```{literalinclude} ../templates/03-nanobind-cuda/packages/gpu-pairwise/tests/test_pairwise.py
:filename: templates/03-nanobind-cuda/packages/gpu-pairwise/tests/test_pairwise.py
:language: python
:linenos:
```

The tests compare the kernel against `scipy.spatial.distance.cdist` for both dtypes and a few shapes, including the degenerate single point case.
The tolerances differ by dtype because float32 accumulation over the feature dimension loses precision that float64 does not.
The random number generator is seeded so the tests are deterministic.
