"""Time the full (n, n) distance matrix: CPU references vs the CUDA kernel.

The kernel is the naive one-thread-per-output version and the timing includes
host<->device copies, so this is a fair "what does a first GPU port buy you"
comparison, not a peak-throughput measurement.
"""

import time

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances as sk_pairwise

import gpu_pairwise


def numpy_broadcast(x):
    # (n, 1, d) - (1, n, d) -> (n, n, d): simple, and memory hungry.
    return np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))


def timeit(fn, *args, repeat=3):
    fn(*args)  # warm up: JIT, first CUDA context, caches
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    rng = np.random.default_rng(20260912)
    print(f"GPU: {gpu_pairwise.device_name()}\n")
    print(f"{'n':>7} {'d':>4} | {'numpy':>9} {'scipy':>9} {'sklearn':>9} {'cuda':>9} | speedup vs scipy")
    for n, d in [(1_000, 16), (4_000, 16), (8_000, 16), (8_000, 128)]:
        x = rng.normal(size=(n, d)).astype(np.float32)
        t_np = timeit(numpy_broadcast, x) if n <= 4_000 else float("nan")
        t_sp = timeit(cdist, x, x)
        t_sk = timeit(sk_pairwise, x)
        t_gpu = timeit(gpu_pairwise.pairwise_distances, x)
        t_np = f"{t_np:8.3f}s" if t_np == t_np else f"{'skipped':>9}"
        print(f"{n:>7} {d:>4} | {t_np} {t_sp:>8.3f}s {t_sk:>8.3f}s {t_gpu:>8.3f}s | {t_sp / t_gpu:>6.1f}x")


if __name__ == "__main__":
    main()
