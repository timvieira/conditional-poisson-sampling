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
from bisect import bisect_left

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

def numpy_tree_sample(cp, M, rng):
    """O(n log N) per sample via product tree quota splitting."""
    return cp.sample(M, rng=rng)


def torch_tree_sample(cpt, M, rng_seed):
    """O(n log N) per sample via PyTorch product tree quota splitting."""
    return cpt.sample(M, rng=rng_seed)


def sequential_sample_from_q(q, rng):
    """O(N) sequential scan sampler, matching R's UPMEsfromq."""
    N, n_cols = q.shape
    s = np.empty(n_cols, dtype=np.int32)
    k = n_cols
    cursor = 0
    for i in range(N):
        if k == 0:
            break
        if rng.random() < q[i, k - 1]:
            s[cursor] = i
            cursor += 1
            k -= 1
    return s


def build_tree_cdfs(Pc, S, N, n):
    """Precompute CDFs for every (node, quota) pair in the tree.

    Returns cdfs: dict mapping node -> list of length n+1,
    where cdfs[node][k] is the CDF array for quota k (or None if impossible).
    """
    cdfs = [None] * (2 * S)
    for node in range(1, S):
        L, R = Pc[2 * node], Pc[2 * node + 1]
        max_k = min(n, len(L) - 1 + len(R) - 1)
        node_cdfs = [None] * (max_k + 1)
        for k in range(1, max_k + 1):
            cdf = []
            total = 0.0
            for j in range(k + 1):
                r = k - j
                lv = L[j] if j < len(L) else 0.0
                rv = R[r] if r < len(R) else 0.0
                total += max(lv, 0.0) * max(rv, 0.0)
                cdf.append(total)
            if total > 0:
                node_cdfs[k] = [c / total for c in cdf]
        cdfs[node] = node_cdfs
    return cdfs


def simple_tree_sample(cdfs, S, N, n, rng):
    """O(n log N) single-sample tree sampler with precomputed CDFs."""
    selected = []
    stack = [(1, n)]
    while stack:
        node, k = stack.pop()
        if k == 0:
            continue
        if node >= S:
            if node - S < N:
                selected.append(node - S)
            continue
        cdf = cdfs[node][k]
        j = bisect_left(cdf, rng.random())
        stack.append((2 * node + 1, k - j))
        stack.append((2 * node, j))
    selected.sort()
    return selected


def build_q_table(w, n):
    """O(Nn) DP to compute sequential conditional probabilities q[i, k].

    Matches R's UPMEqfromw: q[i, k] is the probability of including item i
    given that k items still need to be selected from items i..N-1.
    """
    N = len(w)
    # expa[i, k] = e_k(w[i:N]), the k-th elementary symmetric polynomial
    expa = np.zeros((N, n))
    for i in range(N):
        expa[i, 0] = np.sum(w[i:N])
    for i in range(N - n, N):
        expa[i, N - i - 1] = np.prod(w[i:N])
    for i in range(N - 3, -1, -1):
        for k in range(1, min(N - i - 1, n)):
            expa[i, k] = w[i] * expa[i + 1, k - 1] + expa[i + 1, k]
    # q[i, k] = w[i] * expa[i+1, k-1] / expa[i, k]
    q = np.zeros((N, n))
    for i in range(N - 1, -1, -1):
        q[i, 0] = w[i] / expa[i, 0]
    for i in range(N - n, N):
        q[i, N - i - 1] = 1.0
    for i in range(N - 3, -1, -1):
        for k in range(1, min(N - i - 1, n)):
            q[i, k] = w[i] * expa[i + 1, k - 1] / expa[i, k]
    return q


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

        # ── Sampling benchmarks ─────────────────────────────────────
        cp = ConditionalPoissonNumPy.from_weights(n, w)
        sample_rng = np.random.RandomState(seed)

        print("  samples: NumPy tree (1, incl. build)...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: ConditionalPoissonNumPy.from_weights(n, w).sample(1, rng=sample_rng), reps=reps)
        add("NumPy tree (1 sample, incl. build)", "samples", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        print("  samples: NumPy tree (1)...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: numpy_tree_sample(cp, 1, sample_rng), reps=reps)
        add("NumPy tree (1 sample)", "samples", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        # PyTorch tree sampling
        from conditional_poisson_torch import ConditionalPoissonTorch
        cpt = ConditionalPoissonTorch.from_weights(n, w)

        print("  samples: PyTorch tree (1, incl. build)...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: ConditionalPoissonTorch.from_weights(n, w).sample(1, rng=seed), reps=reps)
        add("PyTorch tree (1 sample, incl. build)", "samples", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        print("  samples: PyTorch tree (1)...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: torch_tree_sample(cpt, 1, seed), reps=reps)
        add("PyTorch tree (1 sample)", "samples", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        # Simple tree sampler
        Pc = cp._get_p_tree()[0]
        S_tree = cp._get_p_tree()[2]
        tree_cdfs = build_tree_cdfs(Pc, S_tree, N, n)
        simple_rng = np.random.default_rng(seed)

        print("  samples: Simple tree (1)...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: simple_tree_sample(tree_cdfs, S_tree, N, n, simple_rng), reps=reps)
        add("Simple tree (1 sample)", "samples", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        # Sequential scan sampling (matching R's UPMEsfromq)
        q_table = build_q_table(w, n)
        seq_rng = np.random.default_rng(seed)

        print("  samples: Sequential (1, incl. DP)...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: sequential_sample_from_q(build_q_table(w, n), seq_rng), reps=reps)
        add("Sequential (1 sample, incl. DP)", "samples", N, n, ms)
        print(f" {ms:.1f}ms", file=sys.stderr)

        print("  samples: Sequential (1)...", file=sys.stderr, end="", flush=True)
        ms = time_fn(lambda: sequential_sample_from_q(q_table, seq_rng), reps=reps)
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
