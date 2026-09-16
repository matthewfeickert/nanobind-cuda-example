"""Time the full (n, n) distance matrix: NumPy broadcasting vs SciPy's C loop."""

import time

import numpy as np
import pairwise_numpy
from scipy.spatial.distance import cdist


def timeit(fn, *args, repeat=3):
    fn(*args)  # warm up caches
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    rng = np.random.default_rng(20260912)
    print(f"{'n':>7} {'d':>4} | {'numpy':>9} {'scipy':>9} | numpy / scipy")
    for n, d in [(1_000, 16), (4_000, 16), (8_000, 16), (8_000, 128)]:
        x = rng.normal(size=(n, d)).astype(np.float32)
        t_np = timeit(pairwise_numpy.pairwise_distances, x)
        t_sp = timeit(cdist, x, x)
        print(f"{n:>7} {d:>4} | {t_np:>8.3f}s {t_sp:>8.3f}s | {t_np / t_sp:>6.1f}x")


if __name__ == "__main__":
    main()
