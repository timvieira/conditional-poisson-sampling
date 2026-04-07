#!/usr/bin/env python3
"""
Timing benchmarks for conditional Poisson sampling.

Benchmarks four operations (Z, pi, fit, samples) across multiple methods and
problem sizes. Outputs timing_data.json for use by plot_timing.py.

Usage:
    python3 bench_timing.py              # run all benchmarks
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
from bench_samplers import sequential_pi

# ── Helpers ──────────────────────────────────────────────────────────────────

def time_fn(fn, reps=5, warmup=2):
    """Time fn() over reps, return median ms. Includes warmup calls."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


# ── Z methods ────────────────────────────────────────────────────────────────

def dp_forward_Z(w, n):
    """O(Nn) DP forward pass to compute Z (normalizing constant)."""
    N = len(w)
    # e[k] = Z(w[0:m], k) — we only need a 1D array, updated in place
    e = np.zeros(n + 1)
    e[0] = 1.0
    for m in range(N):
        # Process in reverse to avoid using updated values
        for k in range(min(n, m + 1), 0, -1):
            e[k] += w[m] * e[k - 1]
    return e[n]


def numpy_tree_Z(w, n):
    """O(N log^2 N) product tree Z via ConditionalPoissonNumPy."""
    cp = ConditionalPoissonNumPy.from_weights(n, w)
    return cp.log_normalizer


def fft_Z(theta, n):
    """O(N log^2 n) FFT product tree Z via PyTorch."""
    from conditional_poisson_torch import forward_log_Z
    return forward_log_Z(theta, n).item()


# ── Pi methods ───────────────────────────────────────────────────────────────

def dp_loo_pi(w, n):
    """N x O(Nn) leave-one-out DP for inclusion probabilities.

    For each item i, run a full O(Nn) DP on w_{-i} to get Z_{-i}(n-1),
    then pi_i = w_i * Z_{-i}(n-1) / Z(n).  Total cost: O(N^2 n).
    """
    N = len(w)
    Z = dp_forward_Z(w, n)
    pi = np.empty(N)
    for i in range(N):
        w_loo = np.delete(w, i)
        Z_loo = dp_forward_Z(w_loo, n - 1)
        pi[i] = w[i] * Z_loo / Z
    return pi


def tree_loo_pi(w, n):
    """N x O(N log^2 n) leave-one-out product tree for inclusion probabilities.

    For each item i, build a product tree on w_{-i}, extract Z_{-i}(n-1),
    then pi_i = w_i * Z_{-i}(n-1) / Z(n).  Total cost: O(N^2 log^2 n).
    """
    N = len(w)
    cp_full = ConditionalPoissonNumPy.from_weights(n, w)
    log_Z = cp_full.log_normalizer
    pi = np.empty(N)
    for i in range(N):
        w_loo = np.delete(w, i)
        cp_loo = ConditionalPoissonNumPy.from_weights(n - 1, w_loo)
        pi[i] = w[i] * np.exp(cp_loo.log_normalizer - log_Z)
    return pi


def fwd_bwd_dp_pi(w, n):
    """O(Nn) forward-backward DP for inclusion probabilities."""
    return sequential_pi(w, n)


def numpy_tree_pi(w, n):
    """O(N log^2 N) product tree + backprop for inclusion probabilities."""
    cp = ConditionalPoissonNumPy.from_weights(n, w)
    return cp.incl_prob


def fft_autograd_pi(theta, n):
    """O(N log^2 n) FFT product tree + autograd for inclusion probabilities."""
    from conditional_poisson_torch import compute_pi
    return compute_pi(theta, n).detach().numpy()


# ── Sampling methods ─────────────────────────────────────────────────────────



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

        reps = max(3, 200 // max(1, N // 50))

        # ── Z benchmarks ────────────────────────────────────────────
        print("  Z: DP forward...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: dp_forward_Z(w, n), reps=reps)
        add("DP forward", "Z", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        print("  Z: NumPy tree...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: numpy_tree_Z(w, n), reps=reps)
        add("NumPy tree", "Z", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        print("  Z: PyTorch FFT...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: fft_Z(theta, n), reps=reps)
        add("PyTorch FFT", "Z", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        # ── Pi benchmarks ───────────────────────────────────────────
        if N <= 500:
            loo_reps = 3  # leave-one-out is O(N^2), keep reps low
            print("  pi: N×DP loo...", file=sys.stderr, end="", flush=True)
            ms = time_fn(lambda: dp_loo_pi(w, n), reps=loo_reps, warmup=1)
            add("N×DP (leave-one-out)", "pi", N, n, ms)
            print(f" {ms:.1f}ms", file=sys.stderr)

            print("  pi: N×Tree loo...", file=sys.stderr, end="", flush=True)
            ms = time_fn(lambda: tree_loo_pi(w, n), reps=loo_reps, warmup=1)
            add("N×Tree (leave-one-out)", "pi", N, n, ms)
            print(f" {ms:.1f}ms", file=sys.stderr)

        print("  pi: Fwd-bwd DP...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: fwd_bwd_dp_pi(w, n), reps=reps)
        add("Forward-backward DP", "pi", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        print("  pi: NumPy tree...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: numpy_tree_pi(w, n), reps=reps)
        add("NumPy tree", "pi", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        print("  pi: PyTorch FFT...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: fft_autograd_pi(theta, n), reps=reps)
        add("PyTorch FFT + autograd", "pi", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        # ── Fitting benchmarks ───────────────────────────────────────
        # Use pi from current weights as the fitting target
        pi_target = sequential_pi(w, n)

        print("  fit: NumPy tree...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: ConditionalPoissonNumPy.fit(pi_target, n), reps=reps)
        add("NumPy tree (Newton-CG)", "fit", N, n, ms)
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
        from conditional_poisson_torch import ConditionalPoissonTorch
        from conditional_poisson_sequential_numpy import ConditionalPoissonSequentialNumPy

        sample_rng = np.random.default_rng(seed)

        # NumPy tree
        cp = ConditionalPoissonNumPy.from_weights(n, w)
        cp.sample(1, rng=sample_rng)  # warmup (builds tree + CDFs)
        print("  samples: NumPy tree (1)...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: cp.sample(1, rng=sample_rng), reps=reps)
        add("NumPy tree (1 sample)", "samples", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        # PyTorch tree
        cpt = ConditionalPoissonTorch.from_weights(n, w)
        cpt.sample(1, rng=sample_rng)  # warmup
        print("  samples: PyTorch tree (1)...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: cpt.sample(1, rng=sample_rng), reps=reps)
        add("PyTorch tree (1 sample)", "samples", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        # Sequential (NumPy)
        cp_seq = ConditionalPoissonSequentialNumPy.from_weights(n, w)
        cp_seq.sample(1, rng=sample_rng)  # warmup (builds q table)
        print("  samples: Sequential (1)...", file=sys.stderr, end="", flush=True)
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
