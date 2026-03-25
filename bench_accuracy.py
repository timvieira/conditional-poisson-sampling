"""
Accuracy comparison: tree-based vs sequential forward-backward pi computation.

The tree computes pi via the polynomial product tree + downward pass.
The sequential method computes pi via forward-backward DP tables.
Brute-force enumeration is used as ground truth for small instances.
"""

import numpy as np
import time
from itertools import combinations
from conditional_poisson import ConditionalPoisson


# ── Sequential forward-backward DP ────────────────────────────────────────────

def _build_fwd_bwd_table(q, n):
    """
    Build forward DP table with row-wise scaling.

    True value: Z(q[0:i] choose k) = W[i, k] * exp(ls[i])  for k >= 1
    W[i, 0] = 1 always (unscaled).
    """
    N = len(q)
    W = np.zeros((N + 1, n + 1))
    ls = np.zeros(N + 1)
    W[0, 0] = 1.0
    for i in range(1, N + 1):
        row = np.empty(n + 1)
        row[0] = 1.0
        if n >= 1:
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

    O(Nn) time, O(Nn) space.
    """
    N = len(w)
    log_gm = np.mean(np.log(w))
    q = w / np.exp(log_gm)

    F, Fls = _build_fwd_bwd_table(q, n)

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


# ── Brute force ───────────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(0)

    cases = [
        # (N, n) — brute force feasible
        (8, 3), (15, 5), (20, 8),
        # moderate
        (50, 10), (100, 20), (200, 40), (200, 80),
        # medium
        (500, 50), (500, 100), (500, 200), (500, 250),
        # large
        (1000, 100), (1000, 200), (1000, 500),
        # stress
        (2000, 200), (2000, 500), (2000, 1000),
        (3000, 500), (5000, 1000), (5000, 2000),
    ]

    print("Accuracy: tree vs sequential pi")
    print("(brute force reference where feasible)")
    print()
    print(f"{'N':>5s} {'n':>4s}  "
          f"{'tree |sum-n|':>14s}  {'seq |sum-n|':>14s}  "
          f"{'max|tree-seq|':>14s}  "
          f"{'max|tree-bf|':>14s}  {'max|seq-bf|':>13s}  "
          f"{'tree ms':>8s}  {'seq ms':>8s}")
    print("-" * 115)

    for N, n in cases:
        w = rng.exponential(1.0, N)

        t0 = time.perf_counter()
        cp = ConditionalPoisson.from_weights(n, w)
        pi_tree = cp.pi
        tree_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        pi_seq = sequential_pi(w, n)
        seq_ms = (time.perf_counter() - t0) * 1000

        tree_sum_err = abs(pi_tree.sum() - n)
        seq_sum_err = abs(pi_seq.sum() - n)
        diff = np.max(np.abs(pi_tree - pi_seq))

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

        print(f"{N:>5d} {n:>4d}  "
              f"{tree_sum_err:>14.2e}  {seq_sum_err:>14.2e}  "
              f"{diff:>14.2e}  "
              f"{tree_bf}  {seq_bf}  "
              f"{tree_ms:>8.1f}  {seq_ms:>8.1f}{flag}")


if __name__ == "__main__":
    main()
