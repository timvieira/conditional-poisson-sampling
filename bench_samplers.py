"""
Benchmark: tree-based vs sequential CPS samplers.

Sequential sampler (Kulesza & Taskar 2011):
  Build O(NK) DP table, scan items deciding include/exclude.

Tree-based sampler (this library):
  Build P-tree in O(N (log N)^2), sample by top-down quota splitting.

This script compares:
  1. Numerical accuracy of inclusion probabilities (tree vs sequential)
  2. Sampling speed (end-to-end and amortised)
  3. Build time (DP table vs P-tree)
"""

import numpy as np
import time
from itertools import combinations
from conditional_poisson_numpy import ConditionalPoissonNumPy


# ── Sequential O(NK) DP ──────────────────────────────────────────────────────

def _build_dp_table(q, n):
    """
    Weighted Pascal DP table with row-wise scaling.

    True value: Z(q[0:i] choose k) = W[i, k] * exp(ls[i])  for k >= 1
    W[i, 0] = 1 always (unscaled).

    Recurrence:  Z[i, k] = Z[i-1, k] + q[i-1] * Z[i-1, k-1]
    """
    N = len(q)
    W = np.zeros((N + 1, n + 1))
    ls = np.zeros(N + 1)
    W[0, 0] = 1.0
    for i in range(1, N + 1):
        row = np.empty(n + 1)
        row[0] = 1.0
        if n >= 1:
            # k=1: W[i-1,1]*exp(ls[i-1]) + q[i-1] (unscaled k=0 term)
            row[1] = W[i-1, 1] + q[i-1] * np.exp(-ls[i-1])
        if n >= 2:
            row[2:] = W[i-1, 2:] + q[i-1] * W[i-1, 1:n]
        mx = np.max(np.abs(row[1:])) if n >= 1 else 1.0
        if mx > 0:
            row[1:] /= mx
            ls[i] = ls[i-1] + np.log(mx)
        else:
            ls[i] = ls[i-1]
        W[i] = row
    return W, ls


def sequential_pi(w, n):
    """
    Compute inclusion probabilities via forward-backward DP.

    Forward:  F[i, k] = Z(q[0:i] choose k)
    Backward: B[i, k] = Z(q[i:N] choose k)
    pi_i = q[i] * sum_j F[i,j] * B[i+1, n-1-j] / Z

    O(Nn) time, O(Nn) space.  Numerically stable via row-wise scaling
    and log-sum-exp accumulation.
    """
    N = len(w)
    log_gm = np.mean(np.log(w))
    q = w / np.exp(log_gm)

    # Forward table
    F, Fls = _build_dp_table(q, n)

    # Backward table
    B = np.zeros((N + 1, n + 1))
    Bls = np.zeros(N + 1)
    B[N, 0] = 1.0
    for i in range(N - 1, -1, -1):
        row = np.empty(n + 1)
        row[0] = 1.0
        if n >= 1:
            row[1] = B[i+1, 1] + q[i] * np.exp(-Bls[i+1])
        if n >= 2:
            row[2:] = B[i+1, 2:] + q[i] * B[i+1, 1:n]
        mx = np.max(np.abs(row[1:])) if n >= 1 else 1.0
        if mx > 0:
            row[1:] /= mx
            Bls[i] = Bls[i+1] + np.log(mx)
        else:
            Bls[i] = Bls[i+1]
        B[i] = row

    log_Z = np.log(abs(F[N, n])) + Fls[N]

    pi = np.zeros(N)
    for i in range(N):
        max_j = min(n, i + 1)
        total_log = -np.inf
        for j in range(max_j):
            k = n - 1 - j
            if k < 0 or k > N - i - 1:
                continue
            f_val, b_val = F[i, j], B[i+1, k]
            if f_val == 0 or b_val == 0:
                continue
            log_f = np.log(abs(f_val)) + (Fls[i] if j >= 1 else 0.0)
            log_b = np.log(abs(b_val)) + (Bls[i+1] if k >= 1 else 0.0)
            log_term = log_f + log_b
            if total_log == -np.inf:
                total_log = log_term
            else:
                mx = max(total_log, log_term)
                total_log = mx + np.log(
                    np.exp(total_log - mx) + np.exp(log_term - mx)
                )
        pi[i] = q[i] * np.exp(total_log - log_Z)
    return pi


def sequential_sample(w, n, M, rng):
    """
    Draw M samples of size n using the sequential scan algorithm.

    Scan items i = N..1, include item i with probability
        q[i-1] * W[i-1, k-1] / W[i, k]
    """
    N = len(w)
    log_gm = np.mean(np.log(w))
    q = w / np.exp(log_gm)
    W, ls = _build_dp_table(q, n)

    out = np.empty((M, n), dtype=np.int32)
    k = np.full(M, n, dtype=np.int32)
    cursors = np.full(M, n - 1, dtype=np.int32)

    for i in range(N, 0, -1):
        active = k > 0
        if not active.any():
            break
        ki = k
        numer_coeff = np.where(ki >= 2, W[i-1, np.maximum(ki - 1, 0)], 1.0)
        numer_ls = np.where(ki >= 2, ls[i-1], 0.0)
        prob = np.where(
            active,
            q[i-1] * numer_coeff / W[i, ki] * np.exp(numer_ls - ls[i]),
            0.0,
        )
        np.clip(prob, 0.0, 1.0, out=prob)
        include = rng.random(M) < prob
        idx = np.where(include)[0]
        if len(idx) > 0:
            out[idx, cursors[idx]] = i - 1
            cursors[idx] -= 1
            k[idx] -= 1
    return out


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

        cp = ConditionalPoissonNumPy.from_weights(n, w)
        pi_tree = cp.incl_prob
        pi_seq = sequential_pi(w, n)

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
    print("=== Build time: DP table vs P-tree ===")
    print(f"{'N':>6s}  {'n':>6s}  {'DP (ms)':>10s}  {'tree (ms)':>10s}  {'DP/tree':>8s}")
    print("-" * 50)

    for N, n in [(50, 10), (100, 20), (200, 40), (500, 100), (1000, 200),
                 (2000, 400), (500, 50), (1000, 50), (1000, 500)]:
        w = rng.exponential(1.0, N)
        log_gm = np.mean(np.log(w))
        q = w / np.exp(log_gm)
        reps = max(3, int(5000 / (N * max(n, 1))))

        t0 = time.perf_counter()
        for _ in range(reps):
            _build_dp_table(q, n)
        dp_ms = (time.perf_counter() - t0) / reps * 1000

        cp = ConditionalPoissonNumPy.from_weights(n, w)
        t0 = time.perf_counter()
        for _ in range(reps):
            cp._cache.clear()
            cp._get_p_tree()
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

        t0 = time.perf_counter()
        sequential_sample(w, n, M, np.random.default_rng(42))
        seq_ms = (time.perf_counter() - t0) * 1000

        cp = ConditionalPoissonNumPy.from_weights(n, w)
        cp._cache.clear()
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
        t0 = time.perf_counter()
        sequential_sample(w, n, M, np.random.default_rng(42))
        seq_ms = (time.perf_counter() - t0) * 1000

        cp = ConditionalPoissonNumPy.from_weights(n, w)
        cp._cache.clear()
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
        cp = ConditionalPoissonNumPy.from_weights(n, w)
        exact = cp.incl_prob

        tree_pi = np.bincount(cp.sample(M, rng=0).ravel(), minlength=N) / M
        seq_pi = np.bincount(
            sequential_sample(w, n, M, np.random.default_rng(0)).ravel(),
            minlength=N,
        ) / M

        print(f"  {label:10s}  N={N:3d} n={n:2d}  "
              f"tree_err={np.max(np.abs(tree_pi - exact)):.4f}  "
              f"seq_err={np.max(np.abs(seq_pi - exact)):.4f}")
    print()


if __name__ == "__main__":
    bench_verify_sampling()
    bench_accuracy()
    bench_build()
    bench_sampling()
    bench_varying_M()
