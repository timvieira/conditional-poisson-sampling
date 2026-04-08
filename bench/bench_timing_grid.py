#!/usr/bin/env python3
"""
Grid sweep of (N, n) for timing benchmarks.
Outputs timing_grid.json for use by the 3D widget in the blog post.

Usage:
    python3 bench_timing_grid.py              # full grid
    python3 bench_timing_grid.py --quick      # smaller grid for testing
"""

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from conditional_poisson.numpy import ConditionalPoissonNumPy
from conditional_poisson.sequential_numpy import ConditionalPoissonSequentialNumPy
from bench_timing import run_r_benchmark, time_fn

import torch
from conditional_poisson.torch import ConditionalPoissonTorch


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
            sample_rng = np.random.default_rng(seed)
            print(f"  n={n}...", file=sys.stderr, end="", flush=True)

            # Z
            ms = time_fn(lambda: ConditionalPoissonSequentialNumPy.from_weights(n, w).log_normalizer, reps=reps)
            add("Sequential DP", "Z", N, n, ms)

            ms = time_fn(lambda: ConditionalPoissonNumPy.from_weights(n, w).log_normalizer, reps=reps)
            add("NumPy tree", "Z", N, n, ms)

            ms = time_fn(lambda: ConditionalPoissonTorch(n, theta).log_normalizer, reps=reps)
            add("PyTorch FFT", "Z", N, n, ms)

            # Pi
            ms = time_fn(lambda: ConditionalPoissonSequentialNumPy.from_weights(n, w).incl_prob, reps=reps)
            add("Sequential DP", "pi", N, n, ms)

            ms = time_fn(lambda: ConditionalPoissonNumPy.from_weights(n, w).incl_prob, reps=reps)
            add("NumPy tree", "pi", N, n, ms)

            ms = time_fn(lambda: ConditionalPoissonTorch(n, theta).incl_prob, reps=reps)
            add("PyTorch FFT + autograd", "pi", N, n, ms)

            # Fitting
            pi_target = ConditionalPoissonSequentialNumPy.from_weights(n, w).incl_prob
            ms = time_fn(lambda: ConditionalPoissonNumPy.fit(pi_target, n), reps=reps)
            add("NumPy tree (L-BFGS)", "fit", N, n, ms)

            # R benchmarks (pi + fit + samples)
            r_results = run_r_benchmark(N, n, seed, reps)
            for r in r_results:
                if "(pi)" in r["method"]:
                    add(r["method"], "pi", N, n, r["time_ms"])
                elif "(fit)" in r["method"]:
                    add(r["method"], "fit", N, n, r["time_ms"])
                else:
                    add(r["method"], "samples", N, n, r["time_ms"])

            # Sampling (excl. precomputation)
            cp = ConditionalPoissonNumPy.from_weights(n, w)
            cp.sample(rng=sample_rng)  # warmup
            ms = time_fn(lambda: cp.sample(rng=sample_rng), reps=reps)
            add("NumPy tree (1 sample)", "samples", N, n, ms)

            cpt = ConditionalPoissonTorch.from_weights(n, w)
            cpt.sample(rng=sample_rng)  # warmup
            ms = time_fn(lambda: cpt.sample(rng=sample_rng), reps=reps)
            add("PyTorch tree (1 sample)", "samples", N, n, ms)

            cp_seq = ConditionalPoissonSequentialNumPy.from_weights(n, w)
            cp_seq.sample(rng=sample_rng)  # warmup
            ms = time_fn(lambda: cp_seq.sample(rng=sample_rng), reps=reps)
            add("Sequential (1 sample)", "samples", N, n, ms)

            print(f" done", file=sys.stderr)

    return results


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    results = run_grid(quick=quick)
    out_path = os.path.join(os.path.dirname(__file__), "results/timing_grid.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} results to {out_path}", file=sys.stderr)
