"""Time the full (n, n) distance matrix: SciPy's C loop vs the XLA-compiled broadcast.

The timing includes host<->device copies. The warm-up call also absorbs the
one-off XLA compilation.
"""

import time

import numpy as np
import pairwise_jax
from scipy.spatial.distance import cdist


def timeit(fn, *args, repeat=3):
    fn(*args)  # warm up: XLA compile, first CUDA context, caches
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    rng = np.random.default_rng(20260912)
    print(f"GPU: {pairwise_jax.device_name()}\n")
    print(f"{'n':>7} {'d':>4} | {'scipy':>9} {'jax':>11} | speedup vs scipy")
    for n, d in [(1_000, 16), (4_000, 16), (8_000, 16), (8_000, 128)]:
        x = rng.normal(size=(n, d)).astype(np.float32)
        t_sp = timeit(cdist, x, x)
        t_gpu = timeit(pairwise_jax.pairwise_distances, x)
        print(f"{n:>7} {d:>4} | {t_sp:>8.3f}s {t_gpu:>10.3f}s | {t_sp / t_gpu:>6.1f}x")


if __name__ == "__main__":
    main()
