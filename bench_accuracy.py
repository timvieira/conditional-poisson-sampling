"""
Accuracy comparison: tree-based vs sequential forward-backward pi computation.

The tree computes pi via the polynomial product tree + downward pass.
The sequential method computes pi via forward-backward DP tables.
Brute-force enumeration is used as ground truth for small instances.
"""

import numpy as np
import time
from itertools import combinations
from conditional_poisson_numpy import ConditionalPoissonNumPy
from conditional_poisson_sequential_numpy import ConditionalPoissonSequentialNumPy


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
        pi_tree = ConditionalPoissonNumPy.from_weights(n, w).incl_prob
        tree_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        pi_seq = ConditionalPoissonSequentialNumPy.from_weights(n, w).incl_prob
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
        return

    import torch
    from conditional_poisson_torch import compute_pi

    r_script = """\
library(sampling)
args <- commandArgs(trailingOnly = TRUE)
N <- as.integer(args[1])
n <- as.integer(args[2])
seed <- as.integer(args[3])
weight_type <- args[4]

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
        cat(sprintf('{"status":"ok","sum_pi":%.15e,"pi":[%s]}\\n',
            sum(pik), paste(sprintf("%.15e", pik), collapse=",")))
    }
}
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

            pi_tree = ConditionalPoissonNumPy.from_weights(n, w).incl_prob
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
