"""Time the full (n, n) distance matrix across the three stages.

Columns: the NumPy baseline (stage 1), SciPy and scikit-learn as CPU
references, the NVRTC-compiled kernel driven by cuda.core (stage 2), and the
nanobind extension compiled ahead of time (stage 3). Both GPU timings include
host<->device copies; the warm-up call absorbs stage 2's NVRTC compile.

The larger sizes exist to separate the fixed per-call overhead of each GPU
binding from the kernel itself. The NumPy baseline is skipped there because
it would take minutes and its trend is already clear.
"""

import time

import gpu_pairwise
import numpy as np
import pairwise_cuda_python
import pairwise_numpy
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances as sk_pairwise


def timeit(fn, *args, repeat=3):
    fn(*args)  # warm up: NVRTC compile, first CUDA context, caches
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    rng = np.random.default_rng(20260912)
    print(f"GPU: {gpu_pairwise.device_name()}\n")
    header = f"{'n':>7} {'d':>4} | {'numpy':>9} {'scipy':>9} {'sklearn':>9} | {'cuda-python':>11} {'nanobind':>9} | nanobind vs scipy"
    print(header)
    for n, d in [
        (1_000, 16),
        (4_000, 16),
        (8_000, 16),
        (8_000, 128),
        (16_000, 16),
        (16_000, 128),
        (32_000, 16),
    ]:
        x = rng.normal(size=(n, d)).astype(np.float32)
        t_np = timeit(pairwise_numpy.pairwise_distances, x) if n <= 8_000 else None
        t_sp = timeit(cdist, x, x)
        t_sk = timeit(sk_pairwise, x)
        t_cp = timeit(pairwise_cuda_python.pairwise_distances, x)
        t_nb = timeit(gpu_pairwise.pairwise_distances, x)
        t_np = f"{t_np:8.3f}s" if t_np is not None else f"{'skipped':>9}"
        print(
            f"{n:>7} {d:>4} | {t_np} {t_sp:>8.3f}s {t_sk:>8.3f}s "
            f"| {t_cp:>10.3f}s {t_nb:>8.3f}s | {t_sp / t_nb:>6.1f}x"
        )


if __name__ == "__main__":
    main()
