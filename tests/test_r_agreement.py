#!/usr/bin/env python3
"""
Tests that our implementations agree with R's sampling package on key quantities.

Compares:
  1. Inclusion probabilities (π from weights)
  2. Fitting (target π → weights → π roundtrip)
  3. Sampling distribution (chi-squared test on empirical frequencies)

Requires R with the 'sampling' package installed (conda env cps-r).
Skips gracefully if R is not available.
"""

import numpy as np
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from conditional_poisson.tree_numpy import ConditionalPoissonNumPy


def find_rscript():
    """Find Rscript, checking conda cps-r env first."""
    try:
        result = subprocess.run(
            ["conda", "run", "-n", "cps-r", "which", "Rscript"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return ["conda", "run", "-n", "cps-r", "Rscript"]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    if shutil.which("Rscript"):
        return ["Rscript"]
    return None


RSCRIPT = find_rscript()


def run_r(code, timeout=60):
    """Run R code and return stdout."""
    if RSCRIPT is None:
        raise unittest.SkipTest("R not available")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                RSCRIPT + [f.name],
                capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                raise RuntimeError(f"R failed: {result.stderr[:500]}")
            return result.stdout
        finally:
            os.unlink(f.name)


def r_compute_pi(w, n):
    """Compute π from weights using R's UPMEqfromw + UPMEpikfromq."""
    w_str = ",".join(f"{x:.15e}" for x in w)
    code = f"""\
library(sampling)
w <- c({w_str})
n <- {n}L
q <- UPMEqfromw(w, n)
pik <- UPMEpikfromq(q)
cat(paste(sprintf("%.15e", pik), collapse=","))
"""
    out = run_r(code)
    return np.array([float(x) for x in out.strip().split(",")])


def r_fit(pi_target, n):
    """Fit weights from target π using R's UPMEpiktildefrompik."""
    pi_str = ",".join(f"{x:.15e}" for x in pi_target)
    code = f"""\
library(sampling)
pik <- c({pi_str})
n <- {n}L
pikt <- UPMEpiktildefrompik(pik)
w <- pikt / (1 - pikt)
# Recompute pi from fitted weights to check roundtrip
q <- UPMEqfromw(w, n)
pik_recomputed <- UPMEpikfromq(q)
cat(paste(sprintf("%.15e", w), collapse=","))
cat("\\n")
cat(paste(sprintf("%.15e", pikt), collapse=","))
cat("\\n")
cat(paste(sprintf("%.15e", pik_recomputed), collapse=","))
"""
    out = run_r(code)
    lines = out.strip().split("\n")
    w = np.array([float(x) for x in lines[0].split(",")])
    ptilde = np.array([float(x) for x in lines[1].split(",")])
    pi_roundtrip = np.array([float(x) for x in lines[2].split(",")])
    return w, ptilde, pi_roundtrip


def r_sample(w, n, n_samples, seed=42):
    """Draw samples using R's UPMEqfromw + UPMEsfromq (from weights, no fitting)."""
    w_str = ",".join(f"{x:.15e}" for x in w)
    code = f"""\
library(sampling)
set.seed({seed}L)
w <- c({w_str})
n <- {n}L
q <- UPMEqfromw(w, n)
results <- matrix(0L, nrow={n_samples}L, ncol=length(w))
for (i in 1:{n_samples}L) {{
    results[i,] <- UPMEsfromq(q)
}}
# Output as one line per sample, space-separated 0/1
for (i in 1:{n_samples}L) {{
    cat(paste(results[i,], collapse=" "))
    cat("\\n")
}}
"""
    out = run_r(code, timeout=120)
    lines = out.strip().split("\n")
    samples = np.array([[int(x) for x in line.split()] for line in lines])
    return samples


class TestRPiAgreement(unittest.TestCase):
    """Test that R and our implementations agree on π computed from weights."""

    def _check(self, w, n, tol=1e-10):
        pi_r = r_compute_pi(w, n)
        cp = ConditionalPoissonNumPy.from_weights(n, w)
        pi_tree = cp.incl_prob
        self.assertAlmostEqual(pi_r.sum(), n, places=8,
                               msg=f"R pi doesn't sum to n")
        self.assertAlmostEqual(pi_tree.sum(), n, places=8,
                               msg=f"Tree pi doesn't sum to n")
        max_diff = np.max(np.abs(pi_r - pi_tree))
        self.assertLess(max_diff, tol,
                        msg=f"N={len(w)}, n={n}: max|pi_R - pi_tree| = {max_diff:.2e}")

    def test_small(self):
        """N=4, n=2: our micro-example."""
        self._check(np.array([1.5, 3.2, 0.8, 4.5]), 2)

    def test_moderate_uniform(self):
        """N=50, n=20: moderate size, Exp(1) weights."""
        rng = np.random.RandomState(42)
        self._check(rng.exponential(1.0, 50), 20)

    def test_moderate_varied(self):
        """N=100, n=40: moderate size, varied weights."""
        rng = np.random.RandomState(42)
        self._check(rng.exponential(1.0, 100), 40)

    def test_large(self):
        """N=500, n=200: large, near R's stability limit."""
        rng = np.random.RandomState(42)
        self._check(rng.exponential(1.0, 500), 200, tol=1e-8)

    def test_extreme_weights_small(self):
        """N=50, n=20: weights spanning several orders of magnitude."""
        rng = np.random.RandomState(42)
        w = np.exp(rng.uniform(-4, 4, 50))
        self._check(w, 20)

    def test_n_equals_1(self):
        """n=1: reduces to categorical.  R's UPMEqfromw crashes on n=1
        (subscript out of bounds), so we skip this case."""
        self.skipTest("R's UPMEqfromw has a bug for n=1")

    def test_n_equals_N_minus_1(self):
        """n=N-1: only one item excluded."""
        self._check(np.array([1.5, 3.2, 0.8, 4.5, 2.0]), 4)


class TestRFittingAgreement(unittest.TestCase):
    """Test that R and our fitting procedures agree."""

    def _check_roundtrip(self, pi_target, n, tol=1e-5):
        """Both R and our fitter should recover the target π."""
        N = len(pi_target)

        # R fitting
        w_r, ptilde_r, pi_r_roundtrip = r_fit(pi_target, n)
        r_err = np.max(np.abs(pi_r_roundtrip - pi_target))

        # Our fitting
        cp = ConditionalPoissonNumPy.fit(pi_target, n)
        pi_ours = cp.incl_prob
        our_err = np.max(np.abs(pi_ours - pi_target))

        self.assertLess(r_err, tol,
                        msg=f"R roundtrip error: {r_err:.2e}")
        self.assertLess(our_err, tol,
                        msg=f"Our roundtrip error: {our_err:.2e}")

    def _check_weights_agree(self, pi_target, n, tol=1e-4):
        """R and our fitted weights should give the same π."""
        w_r, _, pi_r = r_fit(pi_target, n)
        cp = ConditionalPoissonNumPy.fit(pi_target, n)

        # Both should recover the target
        max_diff_r = np.max(np.abs(pi_r - pi_target))
        max_diff_ours = np.max(np.abs(cp.incl_prob - pi_target))
        self.assertLess(max_diff_r, tol)
        self.assertLess(max_diff_ours, tol)

        # The fitted π values should agree (weights may differ by a global
        # scale due to different parameterizations, but π is unique)
        pi_from_r_weights = ConditionalPoissonNumPy.from_weights(n, w_r).incl_prob
        max_pi_diff = np.max(np.abs(pi_from_r_weights - cp.incl_prob))
        self.assertLess(max_pi_diff, tol,
                        msg=f"π from R weights vs ours: {max_pi_diff:.2e}")

    def test_uniform_target(self):
        """Target π = (n/N, ..., n/N): uniform inclusion."""
        N, n = 20, 8
        pi_target = np.full(N, n / N)
        self._check_roundtrip(pi_target, n)
        self._check_weights_agree(pi_target, n)

    def test_varied_target(self):
        """Target π from a known weight vector."""
        w_true = np.array([1.5, 3.2, 0.8, 4.5, 2.0, 1.0, 3.5, 0.5])
        n = 3
        pi_target = ConditionalPoissonNumPy.from_weights(n, w_true).incl_prob
        self._check_roundtrip(pi_target, n)
        self._check_weights_agree(pi_target, n)

    def test_moderate(self):
        """N=50, n=20: moderate size fitting."""
        rng = np.random.RandomState(42)
        w_true = rng.exponential(1.0, 50)
        n = 20
        pi_target = ConditionalPoissonNumPy.from_weights(n, w_true).incl_prob
        self._check_roundtrip(pi_target, n)
        self._check_weights_agree(pi_target, n)


class TestRSamplingDistribution(unittest.TestCase):
    """Test that R's sampler produces samples from the correct distribution."""

    def test_empirical_pi(self):
        """Empirical inclusion frequencies from R samples should match π."""
        w = np.array([1.5, 3.2, 0.8, 4.5, 2.0, 1.0])
        n = 3
        n_samples = 10000

        # Get true π
        cp = ConditionalPoissonNumPy.from_weights(n, w)
        pi_true = cp.incl_prob

        # Draw samples from R
        samples = r_sample(w, n, n_samples)

        # Check all samples have size n
        sizes = samples.sum(axis=1)
        self.assertTrue(np.all(sizes == n),
                        msg=f"Not all R samples have size {n}: {np.unique(sizes)}")

        # Empirical inclusion frequencies
        pi_empirical = samples.mean(axis=0)

        # Each pi_empirical[i] ~ Binomial(n_samples, pi_true[i]) / n_samples
        # std ≈ sqrt(pi*(1-pi)/n_samples), use 4-sigma tolerance
        for i in range(len(w)):
            std = np.sqrt(pi_true[i] * (1 - pi_true[i]) / n_samples)
            self.assertAlmostEqual(
                pi_empirical[i], pi_true[i], delta=4 * std,
                msg=f"Item {i}: empirical={pi_empirical[i]:.4f} vs "
                    f"true={pi_true[i]:.4f} (4σ={4*std:.4f})")

    def test_empirical_subset_frequencies(self):
        """Empirical subset frequencies should match P(S) (small N for enumeration)."""
        w = np.array([1.5, 3.2, 0.8, 4.5])
        n = 2
        n_samples = 20000

        # True distribution
        cp = ConditionalPoissonNumPy.from_weights(n, w)
        from itertools import combinations
        all_S = list(combinations(range(len(w)), n))
        log_w = np.log(w)
        log_probs = np.array([log_w[list(s)].sum() for s in all_S])
        log_Z = np.log(np.exp(log_probs - log_probs.max()).sum()) + log_probs.max()
        true_probs = np.exp(log_probs - log_Z)

        # Draw samples from R
        samples = r_sample(w, n, n_samples)

        # Count subset frequencies
        counts = np.zeros(len(all_S))
        for row in samples:
            subset = frozenset(np.where(row == 1)[0])
            for k, s in enumerate(all_S):
                if frozenset(s) == subset:
                    counts[k] += 1
                    break

        empirical_probs = counts / n_samples

        # Chi-squared test
        expected = true_probs * n_samples
        chi2 = np.sum((counts - expected) ** 2 / expected)
        df = len(all_S) - 1
        # p-value from chi2 distribution; reject at p < 0.001
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df)
        self.assertGreater(p_value, 0.001,
                           msg=f"Chi-squared test failed: χ²={chi2:.2f}, "
                               f"df={df}, p={p_value:.4f}")


if __name__ == "__main__":
    if RSCRIPT is None:
        print("R not available — skipping all R agreement tests.", file=sys.stderr)
        print("Install R with the 'sampling' package to run these tests.",
              file=sys.stderr)
        sys.exit(0)
    unittest.main(verbosity=2)
