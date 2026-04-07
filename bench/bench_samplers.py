"""
Benchmark: tree-based vs sequential CPS samplers.

Sequential sampler uses ConditionalPoissonSequentialNumPy (O(Nn) DP).
Tree-based sampler uses ConditionalPoissonNumPy (O(N log^2 N) product tree).

This script compares:
  1. Numerical accuracy of inclusion probabilities (tree vs sequential)
  2. Sampling speed (end-to-end and amortised)
  3. Build time (DP table vs P-tree)
"""

import numpy as np
import time
from itertools import combinations
from conditional_poisson.numpy import ConditionalPoissonNumPy
from conditional_poisson.sequential_numpy import ConditionalPoissonSequentialNumPy


# ── Brute force (small instances) ────────────────────────────────────────────

def brute_force_pi(w, n):
    """Exact pi by enumerating all (N choose n) subsets."""
    N = len(w)
    log_w = np.log(w)
    all_S = list(combinations(range(N), n))
    log_probs = np.array([log_w[list(s)].sum() for s in all_S])
    log_Z = np.log(np.exp(log_probs - log_probs.max()).sum()) + log_probs.max()
    probs = np.exp(log_probs - log_Z)
    pi = np.zeros(N)
    for k, s in enumerate(all_S):
        for i in s:
            pi[i] += probs[k]
    return pi


# ── Benchmarks ────────────────────────────────────────────────────────────────

def bench_accuracy():
    """Compare numerical accuracy: tree vs sequential vs brute force."""
    rng = np.random.default_rng(0)

    print("=== Accuracy: tree vs sequential pi ===")
    print("(brute force reference where feasible)")
    print()
    print(f"{'N':>5s} {'n':>4s}  {'tree |sum-n|':>14s}  {'seq |sum-n|':>14s}  "
          f"{'max|tree-seq|':>14s}  {'max|tree-bf|':>14s}  {'max|seq-bf|':>13s}")
    print("-" * 90)

    for N, n in [
        (8, 3), (15, 5), (20, 8),                        # brute-force feasible
        (50, 10), (100, 20), (200, 40), (200, 80),       # moderate
        (500, 50), (500, 100), (500, 200), (500, 250),   # medium
        (1000, 100), (1000, 200), (1000, 500),            # large
        (2000, 200), (2000, 500), (2000, 1000),           # large
        (3000, 500), (5000, 1000), (5000, 2000),          # stress
    ]:
        w = rng.exponential(1.0, N)

        pi_tree = ConditionalPoissonNumPy.from_weights(n, w).incl_prob
        pi_seq = ConditionalPoissonSequentialNumPy.from_weights(n, w).incl_prob

        tree_sum_err = abs(pi_tree.sum() - n)
        seq_sum_err = abs(pi_seq.sum() - n)
        diff = np.max(np.abs(pi_tree - pi_seq))

        # Brute force for small instances
        if N <= 20:
            pi_bf = brute_force_pi(w, n)
            tree_bf = f"{np.max(np.abs(pi_tree - pi_bf)):>14.2e}"
            seq_bf = f"{np.max(np.abs(pi_seq - pi_bf)):>13.2e}"
        else:
            tree_bf = f"{'':>14s}"
            seq_bf = f"{'':>13s}"

        flag = ""
        if tree_sum_err > 0.01:
            flag += " TREE-FAIL"
        if seq_sum_err > 0.01:
            flag += " SEQ-FAIL"

        print(f"{N:>5d} {n:>4d}  {tree_sum_err:>14.2e}  {seq_sum_err:>14.2e}  "
              f"{diff:>14.2e}  {tree_bf}  {seq_bf}{flag}")
    print()


def bench_build():
    """Compare build times: DP table vs P-tree."""
    rng = np.random.default_rng(0)
    print("=== Build time: Sequential DP vs P-tree ===")
    print(f"{'N':>6s}  {'n':>6s}  {'seq (ms)':>10s}  {'tree (ms)':>10s}  {'seq/tree':>8s}")
    print("-" * 50)

    for N, n in [(50, 10), (100, 20), (200, 40), (500, 100), (1000, 200),
                 (2000, 400), (500, 50), (1000, 50), (1000, 500)]:
        w = rng.exponential(1.0, N)
        reps = max(3, int(5000 / (N * max(n, 1))))

        t0 = time.perf_counter()
        for _ in range(reps):
            ConditionalPoissonSequentialNumPy.from_weights(n, w).log_normalizer
        dp_ms = (time.perf_counter() - t0) / reps * 1000

        t0 = time.perf_counter()
        for _ in range(reps):
            ConditionalPoissonNumPy.from_weights(n, w).log_normalizer
        tree_ms = (time.perf_counter() - t0) / reps * 1000

        print(f"{N:>6d}  {n:>6d}  {dp_ms:>10.2f}  {tree_ms:>10.2f}  {dp_ms/tree_ms:>8.2f}")
    print()


def bench_sampling():
    """Compare end-to-end sampling speed."""
    print("=== End-to-end sampling (build + draw), M=10000 ===")
    print(f"{'N':>6s}  {'n':>6s}  {'seq (ms)':>10s}  {'tree (ms)':>10s}  {'seq/tree':>9s}")
    print("-" * 50)

    M = 10_000
    for N, n in [(50, 10), (100, 20), (200, 40), (500, 100), (1000, 200),
                 (2000, 400), (100, 5), (500, 10), (1000, 10), (1000, 500)]:
        w = np.random.default_rng(0).exponential(1.0, N)

        cp_seq = ConditionalPoissonSequentialNumPy.from_weights(n, w)
        t0 = time.perf_counter()
        cp_seq.sample(M, rng=np.random.default_rng(42))
        seq_ms = (time.perf_counter() - t0) * 1000

        cp = ConditionalPoissonNumPy.from_weights(n, w)
        t0 = time.perf_counter()
        cp.sample(M, rng=np.random.default_rng(42))
        tree_ms = (time.perf_counter() - t0) * 1000

        print(f"{N:>6d}  {n:>6d}  {seq_ms:>10.1f}  {tree_ms:>10.1f}  {seq_ms/tree_ms:>9.2f}")
    print()


def bench_varying_M():
    """Amortisation over M samples."""
    print("=== Varying M (N=500, n=100) ===")
    print(f"{'M':>8s}  {'seq (ms)':>10s}  {'tree (ms)':>10s}  {'seq/tree':>9s}")
    print("-" * 42)

    N, n = 500, 100
    w = np.random.default_rng(0).exponential(1.0, N)

    for M in [1, 10, 100, 1000, 10_000, 100_000]:
        cp_seq = ConditionalPoissonSequentialNumPy.from_weights(n, w)
        t0 = time.perf_counter()
        cp_seq.sample(M, rng=np.random.default_rng(42))
        seq_ms = (time.perf_counter() - t0) * 1000

        cp = ConditionalPoissonNumPy.from_weights(n, w)
        t0 = time.perf_counter()
        cp.sample(M, rng=np.random.default_rng(42))
        tree_ms = (time.perf_counter() - t0) * 1000

        print(f"{M:>8d}  {seq_ms:>10.1f}  {tree_ms:>10.1f}  {seq_ms/tree_ms:>9.2f}")
    print()


def bench_verify_sampling():
    """Verify both samplers produce correct distributions."""
    print("=== Sampling verification (M=200k) ===")
    M = 200_000
    for label, N, n, w_fn in [
        ("small", 8, 3, lambda r: r.exponential(1.0, 8)),
        ("extreme", 20, 5, lambda r: np.exp(r.uniform(-10, 10, 20))),
        ("large N", 100, 10, lambda r: r.exponential(1.0, 100)),
    ]:
        rng = np.random.default_rng(42)
        w = w_fn(rng)

        pi_tree = ConditionalPoissonNumPy.from_weights(n, w).incl_prob

        # Sequential
        cp_seq = ConditionalPoissonSequentialNumPy.from_weights(n, w)
        S_seq = cp_seq.sample(M, rng=np.random.default_rng(0))
        pi_seq_emp = np.bincount(S_seq.ravel(), minlength=N) / M

        # Tree
        cp = ConditionalPoissonNumPy.from_weights(n, w)
        S_tree = cp.sample(M, rng=np.random.default_rng(0))
        pi_tree_emp = np.bincount(S_tree.ravel(), minlength=N) / M

        err_seq = np.max(np.abs(pi_seq_emp - pi_tree))
        err_tree = np.max(np.abs(pi_tree_emp - pi_tree))

        print(f"  {label:>10s}  N={N:>3d} n={n:>2d}  "
              f"seq err={err_seq:.4f}  tree err={err_tree:.4f}")
    print()


if __name__ == "__main__":
    bench_accuracy()
    bench_build()
    bench_sampling()
    bench_varying_M()
    bench_verify_sampling()
