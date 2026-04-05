#!/usr/bin/env python3
"""
Grid sweep of (N, n) for timing benchmarks.
Outputs timing_grid.json for use by plot_timing_3d.py.

Usage:
    python3 bench_timing_grid.py              # full grid
    python3 bench_timing_grid.py --quick      # smaller grid for testing
"""

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from conditional_poisson_numpy import ConditionalPoisson
from bench_samplers import sequential_pi
from bench_timing import (dp_forward_Z, dp_loo_pi, tree_loo_pi,
                          run_r_benchmark, time_fn)

import torch
from torch_fft_prototype import forward_log_Z, compute_pi


def run_grid(quick=False):
    if quick:
        Ns = [50, 100, 200, 500, 1000]
        ns = [5, 10, 20, 50, 100, 200, 400]
    else:
        Ns = [50, 100, 200, 500, 1000, 2000, 5000]
        ns = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]

    seed = 42
    rng = np.random.RandomState(seed)
    results = []

    def add(method, experiment, N, n, ms):
        results.append({"method": method, "experiment": experiment,
                        "N": N, "n": n, "time_ms": round(ms, 4)})

    for N in Ns:
        valid_ns = [n for n in ns if n < N]
        print(f"\nN={N}, n in {valid_ns}", file=sys.stderr)
        w = rng.exponential(1.0, N)
        theta = torch.tensor(np.log(w), dtype=torch.float64)

        for n in valid_ns:
            reps = max(3, min(20, 500 // max(1, N // 50)))
            print(f"  n={n}...", file=sys.stderr, end="", flush=True)

            # Z
            ms = time_fn(lambda: dp_forward_Z(w, n), reps=reps)
            add("DP forward", "Z", N, n, ms)

            ms = time_fn(lambda: ConditionalPoisson.from_weights(n, w).log_normalizer, reps=reps)
            add("NumPy tree", "Z", N, n, ms)

            ms = time_fn(lambda: forward_log_Z(theta, n), reps=reps)
            add("PyTorch FFT", "Z", N, n, ms)

            # Pi
            ms = time_fn(lambda: sequential_pi(w, n), reps=reps)
            add("Fwd-bwd DP", "pi", N, n, ms)

            ms = time_fn(lambda: ConditionalPoisson.from_weights(n, w).pi, reps=reps)
            add("NumPy tree", "pi", N, n, ms)

            ms = time_fn(lambda: compute_pi(theta, n), reps=reps)
            add("PyTorch FFT + autograd", "pi", N, n, ms)

            if N <= 500 and n <= 100:
                ms = time_fn(lambda: dp_loo_pi(w, n), reps=3, warmup=1)
                add("N×DP loo", "pi", N, n, ms)

                ms = time_fn(lambda: tree_loo_pi(w, n), reps=3, warmup=1)
                add("N×Tree loo", "pi", N, n, ms)

            # R benchmarks (pi + samples)
            r_results = run_r_benchmark(N, n, seed, reps)
            for r in r_results:
                if "(pi)" in r["method"]:
                    add(r["method"], "pi", N, n, r["time_ms"])
                else:
                    add(r["method"], "samples", N, n, r["time_ms"])

            # Sampling benchmarks
            cp = ConditionalPoisson.from_weights(n, w)
            sample_rng = np.random.RandomState(seed)
            ms = time_fn(lambda: cp.sample(1, rng=sample_rng), reps=reps)
            add("NumPy tree (1 sample)", "samples", N, n, ms)

            ms = time_fn(lambda: cp.sample(10_000, rng=sample_rng),
                         reps=max(1, reps // 2), warmup=1)
            add("NumPy tree (10k samples)", "samples", N, n, ms)

            print(f" done", file=sys.stderr)

    return results


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    results = run_grid(quick=quick)
    out_path = os.path.join(os.path.dirname(__file__), "timing_grid.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} results to {out_path}", file=sys.stderr)
