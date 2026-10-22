#!/usr/bin/env python3
"""Evaluations needed to fit target inclusion probabilities, for the article's
"Comparison with R" paragraph.

Compares our fit (L-BFGS; conditional_poisson/_fit.py) with the fixed-point
iteration of R's UPMEpiktildefrompik (package sampling), which starts from
p~_i = pi*_i and repeats

    w_i = p~_i / (1 - p~_i),    p~_i <- p~_i + pi*_i - pi_i(w),

counting evaluations of pi until max_i |pi_i - pi*_i| <= TOL.  Both methods get
pi from ConditionalPoissonSequentialNumPy, so the counts compare the two
iterations, not their implementations.  The fixed-point iteration here is a
copy of R's update; check_against_r runs R's own functions on a few targets
and asserts that they take the same number of iterations.

Writes content/figures/fit_comparison.json.

Usage:
    python bench/bench_fit.py
"""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import warnings
from datetime import date
from pathlib import Path

import numpy as np
from scipy.special import logit

from conditional_poisson.sequential_numpy import ConditionalPoissonSequentialNumPy as DP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_timing import find_rscript  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "content" / "figures" / "fit_comparison.json"

TOL = 1e-10            # stop when max_i |pi_i - pi*_i| <= TOL
MAX_EVALS = 2000       # give up on the fixed-point iteration after this many
ARTICLE_W = [0.68, 1.02, 0.55, 1.63, 0.67, 2.82]   # the article's running example


def lbfgs_evals(target, n):
    """Evaluations our fit makes to reach TOL (its verbose mode prints one line
    per evaluation), or None if it stops short."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DP.fit(target, n, tol=TOL, verbose=True)
    return None if caught else out.getvalue().count("max|pi-pi*|")


def fixed_point_evals(target, n):
    """Evaluations R's fixed-point iteration makes to reach TOL, or None if it
    does not within MAX_EVALS."""
    pt = np.array(target, float)
    for k in range(1, MAX_EVALS + 1):
        if not np.all((pt > 0) & (pt < 1)):         # the weights are no longer valid
            return None
        pi = DP(n, logit(pt)).incl_prob
        if np.max(np.abs(pi - target)) <= TOL:
            return k
        pt = pt + target - pi
    return None


R_TRACE = """
library(sampling)
args <- commandArgs(trailingOnly = TRUE)
pik <- scan(args[1], quiet = TRUE)
n <- as.integer(round(sum(pik)))
pikt <- pik
for (k in 1:as.integer(args[2])) {
  # The body of UPMEpiktildefrompik's loop, with our stopping rule.
  w <- pikt / (1 - pikt)
  pi <- UPMEpikfromq(UPMEqfromw(w, n))
  if (max(abs(pi - pik)) <= as.numeric(args[3])) { cat(k, "\\n"); quit() }
  pikt <- pikt + pik - pi
}
cat("NA\\n")
"""


def check_against_r(rscript):
    """R's own iteration takes the same number of steps as fixed_point_evals."""
    cases = [("article example", np.array(ARTICLE_W), 3),
             ("Exp(1), N=10", np.random.default_rng(0).exponential(1.0, 10), 4),
             ("lognormal sigma=2, N=50", np.exp(np.random.default_rng(0).normal(0, 2, 50)), 20),
             ("lognormal sigma=10, N=8", np.exp(np.random.default_rng(0).normal(0, 10, 8)), 3)]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        script, tfile = Path(tmp) / "trace.R", Path(tmp) / "target.txt"
        script.write_text(R_TRACE)
        for label, w, n in cases:
            target = DP.from_weights(n, w).incl_prob
            np.savetxt(tfile, target, fmt="%.17g")
            proc = subprocess.run([rscript, str(script), str(tfile), str(MAX_EVALS), repr(TOL)],
                                  capture_output=True, text=True, timeout=3600, check=True)
            r = proc.stdout.strip()
            r = None if r == "NA" else int(r)
            ours = fixed_point_evals(target, n)
            print(f"  {label:26s} R: {r}   copy: {ours}")
            assert r == ours, f"{label}: R took {r} iterations, the copy took {ours}"
            rows.append(dict(label=label, N=len(w), n=n, r=r, copy=ours))
    return rows


def typical():
    """The article example, and Exp(1) weights with n = 0.4 N."""
    cases = [("article example", np.array(ARTICLE_W), 3)]
    for N in [10, 50, 200, 1000]:
        cases.append((f"Exp(1), N={N}", np.random.default_rng(0).exponential(1.0, N), int(0.4 * N)))
    rows = []
    for label, w, n in cases:
        target = DP.from_weights(n, w).incl_prob
        row = dict(label=label, N=len(w), n=n, lbfgs=lbfgs_evals(target, n),
                   fixed_point=fixed_point_evals(target, n))
        print(f"  {label:20s} L-BFGS: {row['lbfgs']}   fixed point: {row['fixed_point']}")
        rows.append(row)
    return rows


def uneven(per_sigma=100, seed=1):
    """Random targets from lognormal(0, sigma^2) weights, with N in {3, ..., 300}
    and n uniform on 1, ..., N - 1.  Targets with some pi* rounding to 0 or 1
    are skipped, since no finite weights produce them."""
    rng = np.random.default_rng(seed)
    out = {}
    for sigma in [0.5, 2, 5, 10]:
        lb, fp, skipped, unfinished = [], [], 0, []
        while len(lb) < per_sigma:
            N = int(rng.choice([3, 5, 10, 30, 100, 300]))
            n = int(rng.integers(1, N))
            target = DP.from_weights(n, np.exp(rng.normal(0, sigma, N))).incl_prob
            if not np.all((target > 0) & (target < 1)):
                skipped += 1
                continue
            lb.append(lbfgs_evals(target, n))
            fp.append(fixed_point_evals(target, n))
            if fp[-1] is None:
                unfinished.append(dict(N=N, n=n))
        stats = {}
        for name, counts in [("lbfgs", lb), ("fixed_point", fp)]:
            done = [c for c in counts if c is not None]
            # Count a failure as MAX_EVALS + 1: it needed more than MAX_EVALS.
            stats[name] = dict(median=float(np.median([c or MAX_EVALS + 1 for c in counts])),
                               max=max(done) if done else None,
                               not_converged=len(counts) - len(done))
        out[f"{sigma:g}"] = dict(targets=per_sigma, skipped=skipped, **stats,
                                 fixed_point_unfinished=unfinished)
        print(f"  sigma={sigma:<4g} L-BFGS: {stats['lbfgs']}   fixed point: {stats['fixed_point']}"
              f"   unfinished: {unfinished}")
    return out


def main():
    rscript = find_rscript()
    results: dict = dict(date=str(date.today()), tol=TOL, max_evals=MAX_EVALS)
    if rscript:
        print("R's iteration vs the copy here:")
        results["r_check"] = check_against_r(rscript)
    else:
        print("warning: Rscript not found; skipping the check against R (see `make install-r`)",
              file=sys.stderr)
    print("Typical targets:")
    results["typical"] = typical()
    print("Uneven targets:")
    results["uneven"] = uneven()
    OUT.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
