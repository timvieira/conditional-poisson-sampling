#!/usr/bin/env python3
"""
Correctness tests for conditional_poisson_torch.py.

Tests verify against brute-force enumeration of the distribution P(S)
and known mathematical identities — not against another implementation.
"""

import unittest
import math
import numpy as np
import torch
from itertools import combinations
from collections import Counter

from conditional_poisson.tree_torch import ConditionalPoissonTorch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def Z_bf(w, k):
    """Brute-force Z(w, k) = e_k(w) = sum over size-k subsets of product."""
    N = len(w)
    if k < 0 or k > N:
        return 0.0
    if k == 0:
        return 1.0
    return float(sum(math.prod(w[i] for i in s)
                     for s in combinations(range(N), k)))


def all_probs_bf(w, n):
    """Return dict {tuple(S): P(S)} by brute force."""
    N = len(w)
    Z = Z_bf(w, n)
    return {s: math.prod(w[i] for i in s) / Z
            for s in combinations(range(N), n)}


def pi_bf(w, n):
    """Brute-force inclusion probabilities."""
    N = len(w)
    probs = all_probs_bf(w, n)
    pi = np.zeros(N)
    for S, p in probs.items():
        for i in S:
            pi[i] += p
    return pi


def cov_bf(w, n):
    """Brute-force covariance matrix Cov[1_S]."""
    N = len(w)
    pi = pi_bf(w, n)
    probs = all_probs_bf(w, n)
    cov = np.zeros((N, N))
    for S, p in probs.items():
        indicator = np.zeros(N)
        for i in S:
            indicator[i] = 1.0
        cov += p * np.outer(indicator - pi, indicator - pi)
    return cov


def tv_distance(empirical_counts, exact_probs, M):
    """Total variation distance between empirical and exact distributions.

    empirical_counts: dict {subset: count}
    exact_probs: dict {subset: probability}
    M: total number of samples
    """
    all_subsets = set(empirical_counts) | set(exact_probs)
    return 0.5 * sum(abs(empirical_counts.get(s, 0) / M - exact_probs.get(s, 0))
                     for s in all_subsets)


RNG = np.random.default_rng(42)
W_SMALL = RNG.exponential(1.0, 8)
N_SMALL = 3
W_MED = RNG.exponential(1.0, 12)
N_MED = 5


# ===========================================================================
# Tests
# ===========================================================================

class TestLogZ(unittest.TestCase):
    """Test log_normalizer against brute-force."""

    def test_small(self):
        theta = torch.tensor(np.log(W_SMALL), dtype=torch.float64)
        log_Z = ConditionalPoissonTorch(N_SMALL, theta).log_normalizer
        expected = math.log(Z_bf(W_SMALL, N_SMALL))
        self.assertAlmostEqual(log_Z, expected, places=10)

    def test_medium(self):
        theta = torch.tensor(np.log(W_MED), dtype=torch.float64)
        log_Z = ConditionalPoissonTorch(N_MED, theta).log_normalizer
        expected = math.log(Z_bf(W_MED, N_MED))
        self.assertAlmostEqual(log_Z, expected, places=10)

    def test_uniform_weights(self):
        """Z(1, n) = C(N, n)."""
        N, n = 10, 4
        theta = torch.zeros(N, dtype=torch.float64)
        log_Z = ConditionalPoissonTorch(n, theta).log_normalizer
        self.assertAlmostEqual(log_Z, math.log(math.comb(N, n)), places=10)

    def test_n_equals_N(self):
        """Z(w, N) = prod(w)."""
        theta = torch.tensor(np.log(W_SMALL), dtype=torch.float64)
        log_Z = ConditionalPoissonTorch(len(W_SMALL), theta).log_normalizer
        self.assertAlmostEqual(log_Z, np.sum(np.log(W_SMALL)), places=10)

    def test_n_equals_1(self):
        """Z(w, 1) = sum(w)."""
        theta = torch.tensor(np.log(W_SMALL), dtype=torch.float64)
        log_Z = ConditionalPoissonTorch(1, theta).log_normalizer
        self.assertAlmostEqual(log_Z, math.log(np.sum(W_SMALL)), places=10)

    def test_n_equals_0(self):
        """Z(w, 0) = 1."""
        theta = torch.tensor(np.log(W_SMALL), dtype=torch.float64)
        log_Z = ConditionalPoissonTorch(0, theta).log_normalizer
        self.assertAlmostEqual(log_Z, 0.0, places=10)

    def test_extreme_weights(self):
        """Contour scaling handles large dynamic range."""
        rng = np.random.default_rng(99)
        w = np.exp(rng.uniform(-10, 10, 15))
        n = 5
        theta = torch.tensor(np.log(w), dtype=torch.float64)
        log_Z = ConditionalPoissonTorch(n, theta).log_normalizer
        expected = math.log(Z_bf(w, n))
        self.assertAlmostEqual(log_Z, expected, places=8)


class TestInclusionProbabilities(unittest.TestCase):
    """Test incl_prob against brute-force."""

    def test_small(self):
        theta = torch.tensor(np.log(W_SMALL), dtype=torch.float64)
        pi = ConditionalPoissonTorch(N_SMALL, theta).incl_prob.numpy()
        np.testing.assert_allclose(pi, pi_bf(W_SMALL, N_SMALL), atol=1e-10)

    def test_medium(self):
        theta = torch.tensor(np.log(W_MED), dtype=torch.float64)
        pi = ConditionalPoissonTorch(N_MED, theta).incl_prob.numpy()
        np.testing.assert_allclose(pi, pi_bf(W_MED, N_MED), atol=1e-10)

    def test_sums_to_n(self):
        for w, n in [(W_SMALL, N_SMALL), (W_MED, N_MED)]:
            theta = torch.tensor(np.log(w), dtype=torch.float64)
            pi = ConditionalPoissonTorch(n, theta).incl_prob
            self.assertAlmostEqual(pi.sum().item(), n, places=10)

    def test_in_zero_one(self):
        theta = torch.tensor(np.log(W_SMALL), dtype=torch.float64)
        pi = ConditionalPoissonTorch(N_SMALL, theta).incl_prob
        self.assertTrue((pi > 0).all() and (pi < 1).all())

    def test_uniform_weights(self):
        """pi_i = n/N when all weights are equal."""
        N, n = 10, 4
        theta = torch.zeros(N, dtype=torch.float64)
        pi = ConditionalPoissonTorch(n, theta).incl_prob.numpy()
        np.testing.assert_allclose(pi, n / N, atol=1e-10)

    def test_is_marginal_of_P(self):
        """pi_i = sum_{S ∋ i} P(S), verified against brute-force P(S)."""
        w = W_SMALL
        n = N_SMALL
        probs = all_probs_bf(w, n)
        pi_from_probs = np.zeros(len(w))
        for S, p in probs.items():
            for i in S:
                pi_from_probs[i] += p

        theta = torch.tensor(np.log(w), dtype=torch.float64)
        pi = ConditionalPoissonTorch(n, theta).incl_prob.numpy()
        np.testing.assert_allclose(pi, pi_from_probs, atol=1e-10)



class TestLogProb(unittest.TestCase):
    """Test log_prob against brute-force P(S) for every subset."""

    def test_all_subsets(self):
        """log_prob matches brute-force for every size-n subset."""
        w = W_SMALL
        n = N_SMALL
        cp = ConditionalPoissonTorch.from_weights(n, w)
        probs = all_probs_bf(w, n)

        for S, p_bf in probs.items():
            lp = cp.log_prob(torch.tensor(S))
            self.assertAlmostEqual(lp, math.log(p_bf), places=10,
                                   msg=f"S={S}")

    def test_normalizes(self):
        """sum P(S) = 1 over all size-n subsets."""
        w = W_SMALL
        n = N_SMALL
        cp = ConditionalPoissonTorch.from_weights(n, w)
        total = sum(math.exp(cp.log_prob(torch.tensor(S)))
                    for S in combinations(range(len(w)), n))
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_batch(self):
        w = W_SMALL
        n = N_SMALL
        cp = ConditionalPoissonTorch.from_weights(n, w)
        probs = all_probs_bf(w, n)
        subsets = list(probs.keys())[:5]
        S = torch.tensor(subsets, dtype=torch.long)
        lps = cp.log_prob(S)
        for i, s in enumerate(subsets):
            self.assertAlmostEqual(lps[i].item(), math.log(probs[s]), places=10)

    def test_bool_mask(self):
        w = W_SMALL
        n = N_SMALL
        cp = ConditionalPoissonTorch.from_weights(n, w)
        probs = all_probs_bf(w, n)
        S = list(probs.keys())[0]
        mask = torch.zeros(len(w), dtype=torch.bool)
        for i in S:
            mask[i] = True
        lp = cp.log_prob(mask)
        self.assertAlmostEqual(lp, math.log(probs[S]), places=10)


class TestSampling(unittest.TestCase):
    """Test the tree-based sampler against brute-force P(S)."""

    def test_sample_shape(self):
        cp = ConditionalPoissonTorch.from_weights(N_SMALL, W_SMALL)
        rng = np.random.default_rng(0)
        samples = torch.stack([cp.sample(rng=rng) for _ in range(100)])
        self.assertEqual(samples.shape, (100, N_SMALL))

    def test_sample_no_duplicates(self):
        cp = ConditionalPoissonTorch.from_weights(N_SMALL, W_SMALL)
        rng = np.random.default_rng(0)
        samples = torch.stack([cp.sample(rng=rng) for _ in range(1000)])
        for row in samples:
            self.assertEqual(len(row.unique()), N_SMALL)

    def test_sample_sorted(self):
        cp = ConditionalPoissonTorch.from_weights(N_SMALL, W_SMALL)
        rng = np.random.default_rng(0)
        samples = torch.stack([cp.sample(rng=rng) for _ in range(100)])
        for row in samples:
            self.assertTrue(torch.all(row[1:] > row[:-1]))

    def test_sample_deterministic(self):
        cp = ConditionalPoissonTorch.from_weights(N_SMALL, W_SMALL)
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        s1 = torch.stack([cp.sample(rng=rng1) for _ in range(50)])
        s2 = torch.stack([cp.sample(rng=rng2) for _ in range(50)])
        self.assertTrue(torch.equal(s1, s2))

    def test_sample_distribution_small(self):
        """Empirical subset frequencies match exact P(S) (TV distance)."""
        w = W_SMALL
        n = N_SMALL
        cp = ConditionalPoissonTorch.from_weights(n, w)
        exact = all_probs_bf(w, n)

        M = 50_000
        rng = np.random.default_rng(0)
        samples = torch.stack([cp.sample(rng=rng) for _ in range(M)])
        counts = Counter(tuple(row.tolist()) for row in samples)

        tv = tv_distance(counts, exact, M)
        self.assertLess(tv, 0.02,
                        f"TV distance = {tv:.4f}, expected < 0.02")

    def test_sample_distribution_extreme_weights(self):
        """Full distribution test with extreme weight ratios."""
        rng = np.random.default_rng(99)
        w = np.exp(rng.uniform(-5, 5, 10))
        n = 3
        cp = ConditionalPoissonTorch.from_weights(n, w)
        exact = all_probs_bf(w, n)

        M = 50_000
        rng2 = np.random.default_rng(0)
        samples = torch.stack([cp.sample(rng=rng2) for _ in range(M)])
        counts = Counter(tuple(row.tolist()) for row in samples)

        tv = tv_distance(counts, exact, M)
        self.assertLess(tv, 0.02,
                        f"TV distance = {tv:.4f}, expected < 0.02")

    def test_sample_every_subset_appears(self):
        """Every possible subset is sampled at least once (small N)."""
        w = RNG.exponential(1.0, 6)
        n = 2
        cp = ConditionalPoissonTorch.from_weights(n, w)
        num_subsets = math.comb(6, 2)  # 15

        M = 10_000
        rng = np.random.default_rng(0)
        samples = torch.stack([cp.sample(rng=rng) for _ in range(M)])
        seen = set(tuple(row.tolist()) for row in samples)
        self.assertEqual(len(seen), num_subsets,
                         f"saw {len(seen)}/{num_subsets} subsets")

    def test_sample_n_zero(self):
        cp = ConditionalPoissonTorch.from_weights(0, W_SMALL)
        rng = np.random.default_rng(0)
        samples = torch.stack([cp.sample(rng=rng) for _ in range(10)])
        self.assertEqual(samples.shape, (10, 0))

    def test_sample_n_equals_N(self):
        cp = ConditionalPoissonTorch.from_weights(len(W_SMALL), W_SMALL)
        rng = np.random.default_rng(0)
        samples = torch.stack([cp.sample(rng=rng) for _ in range(5)])
        expected = torch.arange(len(W_SMALL))
        for row in samples:
            self.assertTrue(torch.equal(row, expected))


class TestFit(unittest.TestCase):
    """Test fitting to target inclusion probabilities."""

    def test_fit_recovers_distribution(self):
        """Fit to pi from known weights, verify full P(S) matches."""
        w = W_SMALL
        n = N_SMALL
        exact = all_probs_bf(w, n)
        pi_target = pi_bf(w, n)

        cp_fit = ConditionalPoissonTorch.fit(pi_target, n, tol=1e-8)
        for S, p_bf in exact.items():
            lp = cp_fit.log_prob(torch.tensor(S))
            self.assertAlmostEqual(math.exp(lp), p_bf, places=6,
                                   msg=f"S={S}")

    def test_fit_uniform(self):
        """Fit to uniform pi = n/N; P(S) should be 1/C(N,n) for all S."""
        N, n = 8, 3
        pi_star = torch.full((N,), n / N, dtype=torch.float64)
        cp = ConditionalPoissonTorch.fit(pi_star, n, tol=1e-8)
        expected_p = 1.0 / math.comb(N, n)
        for S in combinations(range(N), n):
            lp = cp.log_prob(torch.tensor(S))
            self.assertAlmostEqual(math.exp(lp), expected_p, places=7,
                                   msg=f"S={S}")


class TestEdgeCases(unittest.TestCase):

    def test_from_weights_rejects_nonpositive(self):
        with self.assertRaises(ValueError):
            ConditionalPoissonTorch.from_weights(2, [1.0, -1.0, 2.0])

    def test_from_weights_rejects_inf(self):
        with self.assertRaises(ValueError):
            ConditionalPoissonTorch.from_weights(1, [1.0, float('inf')])

    def test_invalid_n(self):
        with self.assertRaises(AssertionError):
            ConditionalPoissonTorch(10, torch.zeros(5))
        with self.assertRaises(AssertionError):
            ConditionalPoissonTorch(-1, torch.zeros(5))

    def test_N_equals_1(self):
        """Trivial case: N=1, n=1."""
        cp = ConditionalPoissonTorch.from_weights(1, [3.0])
        self.assertAlmostEqual(math.exp(cp.log_prob(torch.tensor([0]))), 1.0)
        self.assertAlmostEqual(cp.incl_prob[0].item(), 1.0)

    def test_N_equals_2_n_equals_1(self):
        """N=2, n=1: P({i}) = w_i / (w_0 + w_1)."""
        w = np.array([2.0, 3.0])
        cp = ConditionalPoissonTorch.from_weights(1, w)
        self.assertAlmostEqual(math.exp(cp.log_prob(torch.tensor([0]))),
                               2.0 / 5.0, places=10)
        self.assertAlmostEqual(math.exp(cp.log_prob(torch.tensor([1]))),
                               3.0 / 5.0, places=10)


if __name__ == '__main__':
    unittest.main()
