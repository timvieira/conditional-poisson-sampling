"""
Accuracy comparison: tree-based vs sequential forward-backward pi computation.

The tree computes pi via the polynomial product tree + downward pass.
The sequential method computes pi via forward-backward DP tables.
Brute-force enumeration is used as ground truth for small instances.
"""

import numpy as np
import time
from itertools import combinations
from conditional_poisson_numpy import ConditionalPoisson


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
        pi_tree = cp.incl_prob
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


def r_accuracy():
    """Compare R sampling::UPmaxentropy accuracy against our implementations.

    Tests moderate weights (Exp(1)) and extreme weights (exp(Uniform(-6,6)))
    across a range of N.  The R package's DP runs in linear space without
    log-scaling, so it produces NaN at large N or extreme weight ranges.
    Our product tree and FFT implementations use log-space arithmetic and
    contour scaling, maintaining machine-epsilon accuracy throughout.

    Requires: R with the 'sampling' package installed.
    """
    import subprocess, json, shutil

    # Check if R is available
    rscript = shutil.which("Rscript")
    if not rscript:
        # Try conda env
        try:
            result = subprocess.run(
                ["conda", "run", "-n", "cps-r", "which", "Rscript"],
                capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                rscript = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    if not rscript:
        print("R not available — skipping R accuracy comparison.")
        print("Install R and the 'sampling' package to run this test.")
        return

    import torch
    from conditional_poisson_torch import compute_pi

    # Write a temporary R script that computes pi and prints JSON
    r_script = """\
library(sampling)
args <- commandArgs(trailingOnly = TRUE)
N <- as.integer(args[1])
n <- as.integer(args[2])
seed <- as.integer(args[3])
weight_type <- args[4]  # "moderate" or "extreme"

set.seed(seed)
if (weight_type == "moderate") {
    w <- rexp(N)
} else {
    w <- exp(runif(N, -6, 6))
}

q <- tryCatch(UPMEqfromw(w, n), error = function(e) NULL)
if (is.null(q)) {
    cat('{"status":"error","msg":"UPMEqfromw failed"}\\n')
} else {
    pik <- tryCatch(UPMEpikfromq(q), error = function(e) NULL)
    if (is.null(pik) || any(is.nan(pik))) {
        cat(sprintf('{"status":"nan","sum_pi":%.6f}\\n', sum(pik)))
    } else {
        # Output pi as JSON array
        cat(sprintf('{"status":"ok","sum_pi":%.15e,"pi":[%s]}\\n',
            sum(pik), paste(sprintf("%.15e", pik), collapse=",")))
    }
}
# Also output the weights so Python can replicate
set.seed(seed)
if (weight_type == "moderate") {
    w <- rexp(N)
} else {
    w <- exp(runif(N, -6, 6))
}
cat(sprintf('{"weights":[%s]}\\n', paste(sprintf("%.15e", w), collapse=",")))
"""
    r_script_path = "/tmp/bench_r_accuracy.R"
    with open(r_script_path, "w") as f:
        f.write(r_script)

    cases = [
        (100, 40), (200, 80), (500, 200),
        (1000, 400), (2000, 800), (5000, 2000),
    ]
    weight_types = ["moderate", "extreme"]
    seed = 42

    print()
    print("=" * 100)
    print("Accuracy comparison: R sampling vs product tree vs PyTorch FFT")
    print("=" * 100)
    print()
    print(f"{'N':>5s} {'n':>5s} {'weights':>8s}  "
          f"{'R sum_err':>12s}  {'tree sum_err':>12s}  {'FFT sum_err':>12s}  "
          f"{'max|R-tree|':>12s}  {'max|R-FFT|':>12s}  {'R status':>10s}")
    print("-" * 100)

    for N, n in cases:
        for wt in weight_types:
            # Run R
            try:
                result = subprocess.run(
                    ["conda", "run", "-n", "cps-r", "Rscript", r_script_path,
                     str(N), str(n), str(seed), wt],
                    capture_output=True, text=True, timeout=120)
                lines = [l for l in result.stdout.strip().split("\n") if l.startswith("{")]
            except (subprocess.TimeoutExpired, FileNotFoundError):
                lines = []

            if len(lines) < 2:
                print(f"{N:>5d} {n:>5d} {wt:>8s}  {'':>12s}  {'':>12s}  {'':>12s}  "
                      f"{'':>12s}  {'':>12s}  {'R-FAILED':>10s}")
                continue

            r_result = json.loads(lines[0])
            w = np.array(json.loads(lines[1])["weights"])

            # Our implementations
            cp = ConditionalPoisson.from_weights(n, w)
            pi_tree = cp.incl_prob
            tree_sum_err = abs(pi_tree.sum() - n)

            theta = torch.tensor(np.log(w), dtype=torch.float64)
            pi_fft = compute_pi(theta, n).detach().numpy()
            fft_sum_err = abs(pi_fft.sum() - n)

            if r_result["status"] == "ok":
                pi_r = np.array(r_result["pi"])
                r_sum_err = abs(pi_r.sum() - n)
                r_tree_diff = np.max(np.abs(pi_r - pi_tree))
                r_fft_diff = np.max(np.abs(pi_r - pi_fft))
                print(f"{N:>5d} {n:>5d} {wt:>8s}  "
                      f"{r_sum_err:>12.2e}  {tree_sum_err:>12.2e}  {fft_sum_err:>12.2e}  "
                      f"{r_tree_diff:>12.2e}  {r_fft_diff:>12.2e}  {'ok':>10s}")
            else:
                status = r_result["status"]
                print(f"{N:>5d} {n:>5d} {wt:>8s}  "
                      f"{'NaN':>12s}  {tree_sum_err:>12.2e}  {fft_sum_err:>12.2e}  "
                      f"{'---':>12s}  {'---':>12s}  {status:>10s}")


if __name__ == "__main__":
    main()
    r_accuracy()
