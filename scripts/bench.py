"""Time the full (n, n) distance matrix across the four stages.

Columns: the NumPy baseline (stage 1), SciPy and scikit-learn as CPU
references, the XLA-compiled broadcast in JAX (stage 2), the NVRTC-compiled
kernel driven by cuda.core (stage 3), and the nanobind extension compiled
ahead of time (stage 4). All GPU timings include host<->device copies; the
warm-up call absorbs the XLA and NVRTC compiles.

JAX is timed last in each row on purpose. Its runtime keeps every byte of
device memory it has touched for the life of the process, so if it ran first
its result buffer would still be held when the other two stages asked for
theirs, and at the largest sizes they would fail to allocate.

The larger sizes exist to separate the fixed per-call overhead of each GPU
binding from the kernel itself. The NumPy baseline is skipped there because
it would take minutes and its trend is already clear. n stops at 16000
because SciPy's float64 result and cuda-python's pinned buffers at n = 32000
need 8 GB of host memory each, which turns the row into a swap benchmark on
a laptop with other work open.
"""

import time

import gpu_pairwise
import numpy as np
import pairwise_cuda_python
import pairwise_jax
import pairwise_numpy
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances as sk_pairwise


def timeit(fn, *args, repeat=3):
    fn(*args)  # warm up: XLA and NVRTC compiles, first CUDA context, caches
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    rng = np.random.default_rng(20260912)
    print(f"GPU: {gpu_pairwise.device_name()}\n")
    header = f"{'n':>7} {'d':>4} | {'numpy':>9} {'scipy':>9} {'sklearn':>9} | {'cuda-python':>11} {'nanobind':>9} {'jax':>9} | nanobind vs scipy"
    print(header)
    for n, d in [
        (1_000, 16),
        (4_000, 16),
        (8_000, 16),
        (8_000, 128),
        (16_000, 16),
        (16_000, 128),
    ]:
        x = rng.normal(size=(n, d)).astype(np.float32)
        t_np = timeit(pairwise_numpy.pairwise_distances, x) if n <= 8_000 else None
        t_sp = timeit(cdist, x, x)
        t_sk = timeit(sk_pairwise, x)
        t_cp = timeit(pairwise_cuda_python.pairwise_distances, x)
        t_nb = timeit(gpu_pairwise.pairwise_distances, x)
        t_jx = timeit(pairwise_jax.pairwise_distances, x)
        t_np = f"{t_np:8.3f}s" if t_np is not None else f"{'skipped':>9}"
        print(
            f"{n:>7} {d:>4} | {t_np} {t_sp:>8.3f}s {t_sk:>8.3f}s "
            f"| {t_cp:>10.3f}s {t_nb:>8.3f}s {t_jx:>8.3f}s | {t_sp / t_nb:>6.1f}x"
        )


if __name__ == "__main__":
    main()
