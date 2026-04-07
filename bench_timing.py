#!/usr/bin/env python3
"""
Timing benchmarks for conditional Poisson sampling.

Benchmarks all centralized implementations across a range of
problem sizes. Outputs timing_data.json for use by plot_timing.py.

Usage:
    python3 bench_timing.py              # full benchmark
    python3 bench_timing.py --quick      # smaller sizes for testing
"""

import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from conditional_poisson_numpy import ConditionalPoissonNumPy


def time_fn(fn, reps=5, warmup=2):
    """Time fn, return median ms. Runs warmup iters first."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


# ── R subprocess ─────────────────────────────────────────────────────────────

def find_rscript():
    """Find Rscript binary — check cps-r conda env first, then PATH."""
    conda = shutil.which("conda")
    if conda:
        try:
            result = subprocess.run(
                ["conda", "run", "-n", "cps-r", "which", "Rscript"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return shutil.which("Rscript")


def run_r_benchmark(N, n, seed, reps):
    """Run bench_timing_r.R and return list of result dicts."""
    rscript = find_rscript()
    if not rscript:
        print("  R not available, skipping R benchmarks", file=sys.stderr)
        return []

    script = os.path.join(os.path.dirname(__file__), "bench_timing_r.R")
    try:
        result = subprocess.run(
            ["conda", "run", "-n", "cps-r", "Rscript", script,
             str(N), str(n), str(seed), str(reps)],
            capture_output=True, text=True,
            timeout=300  # 5 min max per size
        )
        if result.returncode != 0:
            print(f"  R benchmark failed: {result.stderr[:200]}", file=sys.stderr)
            return []
        results = []
        for line in result.stdout.strip().split("\n"):
            if line.startswith("{"):
                results.append(json.loads(line))
        return results
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  R benchmark error: {e}", file=sys.stderr)
        return []


# ── Main benchmark driver ────────────────────────────────────────────────────

def run_benchmarks(quick=False):
    import torch
    from conditional_poisson_torch import ConditionalPoissonTorch, forward_log_Z, compute_pi
    from conditional_poisson_sequential_numpy import ConditionalPoissonSequentialNumPy

    if quick:
        sizes = [(50, 20), (100, 40), (200, 80)]
    else:
        sizes = [(50, 20), (100, 40), (200, 80), (500, 200),
                 (1000, 400), (2000, 800), (5000, 2000)]

    seed = 42
    rng = np.random.RandomState(seed)
    results = []

    def add(method, experiment, N, n, ms):
        results.append({
            "method": method,
            "experiment": experiment,
            "N": N, "n": n,
            "time_ms": round(ms, 4)
        })

    for N, n in sizes:
        print(f"\n=== N={N}, n={n} ===", file=sys.stderr)
        w = rng.exponential(1.0, N)
        theta = torch.tensor(np.log(w), dtype=torch.float64)
        sample_rng = np.random.default_rng(seed)

        reps = max(3, 200 // max(1, N // 50))

        # ── Z benchmarks ────────────────────────────────────────────
        print("  Z: Sequential DP...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: ConditionalPoissonSequentialNumPy.from_weights(n, w).log_normalizer, reps=reps)
        add("Sequential DP", "Z", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        print("  Z: NumPy tree...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: ConditionalPoissonNumPy.from_weights(n, w).log_normalizer, reps=reps)
        add("NumPy tree", "Z", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        print("  Z: PyTorch FFT...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: forward_log_Z(theta, n).item(), reps=reps)
        add("PyTorch FFT", "Z", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        # ── Pi benchmarks ───────────────────────────────────────────
        print("  pi: Sequential DP...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: ConditionalPoissonSequentialNumPy.from_weights(n, w).incl_prob, reps=reps)
        add("Sequential DP", "pi", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        print("  pi: NumPy tree...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: ConditionalPoissonNumPy.from_weights(n, w).incl_prob, reps=reps)
        add("NumPy tree", "pi", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        print("  pi: PyTorch FFT...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: compute_pi(theta, n).detach(), reps=reps)
        add("PyTorch FFT + autograd", "pi", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        # ── Fitting benchmarks ───────────────────────────────────────
        pi_target = ConditionalPoissonSequentialNumPy.from_weights(n, w).incl_prob

        print("  fit: NumPy tree (L-BFGS)...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: ConditionalPoissonNumPy.fit(pi_target, n), reps=reps)
        add("NumPy tree (L-BFGS)", "fit", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        # ── R benchmarks (pi + fit + samples) ──────────────────────
        print("  R benchmarks...", file=sys.stderr, end="", flush=True)
        r_results = run_r_benchmark(N, n, seed, reps)
        for r in r_results:
            if "(pi)" in r["method"]:
                experiment = "pi"
            elif "(fit)" in r["method"]:
                experiment = "fit"
            else:
                experiment = "samples"
            add(r["method"], experiment, N, n, r["time_ms"])
        print(f" {len(r_results)} results", file=sys.stderr)

        # ── Sampling benchmarks (excl. precomputation) ────────────
        # NumPy tree
        cp = ConditionalPoissonNumPy.from_weights(n, w)
        cp.sample(1, rng=sample_rng)  # warmup
        print("  samples: NumPy tree...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: cp.sample(1, rng=sample_rng), reps=reps)
        add("NumPy tree (1 sample)", "samples", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        # PyTorch tree
        cpt = ConditionalPoissonTorch.from_weights(n, w)
        cpt.sample(1, rng=sample_rng)  # warmup
        print("  samples: PyTorch tree...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: cpt.sample(1, rng=sample_rng), reps=reps)
        add("PyTorch tree (1 sample)", "samples", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        # Sequential (NumPy)
        cp_seq = ConditionalPoissonSequentialNumPy.from_weights(n, w)
        cp_seq.sample(1, rng=sample_rng)  # warmup
        print("  samples: Sequential...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: cp_seq.sample(1, rng=sample_rng), reps=reps)
        add("Sequential (1 sample)", "samples", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

    return results


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    results = run_benchmarks(quick=quick)

    out_path = os.path.join(os.path.dirname(__file__), "timing_data.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} results to {out_path}", file=sys.stderr)
