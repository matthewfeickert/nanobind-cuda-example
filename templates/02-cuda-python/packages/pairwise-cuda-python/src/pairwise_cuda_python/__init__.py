"""Pairwise Euclidean distances on the GPU, driven from Python with cuda.core.

The CUDA kernel lives next to this file in ``pairwise.cu`` as plain source.
Nothing compiles it when the package is built; on first use it is handed to
NVRTC, the CUDA runtime compiler, for the GPU that is actually present. All
memory management and the kernel launch go through ``cuda.core``, the
Pythonic layer of NVIDIA's cuda-python project.
"""

import importlib.resources

import numpy as np
from cuda.core import (
    Device,
    LaunchConfig,
    LegacyPinnedMemoryResource,
    Program,
    ProgramOptions,
    launch,
)

__all__ = ["device_name", "pairwise_distances"]

_KERNEL_SOURCE = importlib.resources.files(__package__).joinpath("pairwise.cu").read_text()
# Same launch geometry as the nanobind stage: block.x walks columns of the
# output, block.y walks rows.
_BLOCK = (32, 8)

_kernels = {}


def _kernel(dtype):
    """Compile the kernel once per process and return the instantiation for ``dtype``."""
    if not _kernels:
        dev = Device()
        dev.set_current()
        # NVRTC compiles for this GPU only (sm_89 on an RTX 4060); the nanobind
        # stage compiles for every major architecture ahead of time instead.
        options = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
        program = Program(_KERNEL_SOURCE, code_type="c++", options=options)
        names = {np.float32: "pairwise_kernel<float>", np.float64: "pairwise_kernel<double>"}
        module = program.compile("cubin", name_expressions=tuple(names.values()))
        for key, name in names.items():
            _kernels[np.dtype(key)] = module.get_kernel(name)
    return _kernels[np.dtype(dtype)]


def device_name():
    """Return the name of the GPU the kernel runs on."""
    return Device().name


def pairwise_distances(x, y=None):
    """Return the ``(n, m)`` matrix of Euclidean distances between rows of ``x`` and ``y``.

    ``x`` has shape ``(n, d)`` and ``y`` shape ``(m, d)``; ``y`` defaults to ``x``.
    Inputs are promoted to float32 unless either is already float64.
    """
    x = np.asarray(x)
    y = x if y is None else np.asarray(y)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2-dimensional")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same number of columns (features)")
    # float64 only if the caller opted in; everything else (ints, float16, ...)
    # goes through the float32 kernel.
    dtype = np.float64 if np.float64 in (x.dtype, y.dtype) else np.float32
    x = np.ascontiguousarray(x, dtype=dtype)
    y = np.ascontiguousarray(y, dtype=dtype)
    n, d = x.shape
    m = y.shape[0]

    kernel = _kernel(dtype)
    dev = Device()
    dev.set_current()
    stream = dev.create_stream()
    pinned = LegacyPinnedMemoryResource()
    buffers = []
    try:
        # cuda.core copies between its own Buffer objects, so the NumPy arrays go
        # through page-locked host buffers that NumPy can view via DLPack.
        def to_device(arr):
            host = pinned.allocate(arr.nbytes)
            buffers.append(host)
            np.from_dlpack(host).view(dtype)[:] = arr.ravel()
            device = dev.allocate(arr.nbytes, stream=stream)
            buffers.append(device)
            host.copy_to(device, stream=stream)
            return device

        dx = to_device(x)
        dy = to_device(y)
        out_nbytes = n * m * np.dtype(dtype).itemsize
        dout = dev.allocate(out_nbytes, stream=stream)
        buffers.append(dout)

        grid = (-(-m // _BLOCK[0]), -(-n // _BLOCK[1]))
        config = LaunchConfig(grid=grid, block=_BLOCK)
        launch(stream, config, kernel, dx, dy, dout, np.int32(n), np.int32(m), np.int32(d))

        host_out = pinned.allocate(out_nbytes)
        buffers.append(host_out)
        dout.copy_to(host_out, stream=stream)
        stream.sync()
        # Copy out of the pinned buffer so the result outlives its deallocation.
        return np.from_dlpack(host_out).view(dtype).reshape(n, m).copy()
    finally:
        for buffer in buffers:
            buffer.close(stream)
        stream.sync()
        stream.close()
