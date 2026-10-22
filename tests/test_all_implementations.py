"""Parameterized tests for all ConditionalPoisson implementations.

Every test runs on all four implementations. No class-specific test files.
"""

import math
from fractions import Fraction
import numpy as np
import pytest
from itertools import combinations

from conditional_poisson.tree_numpy import ConditionalPoissonNumPy
from conditional_poisson.sequential_numpy import ConditionalPoissonSequentialNumPy
from conditional_poisson.tree_torch import ConditionalPoissonTorch
from conditional_poisson.sequential_torch import ConditionalPoissonSequentialTorch

ALL_CLASSES = [
    ConditionalPoissonNumPy,
    ConditionalPoissonSequentialNumPy,
    ConditionalPoissonTorch,
    ConditionalPoissonSequentialTorch,
]


def to_numpy(x):
    if hasattr(x, 'numpy'):
        return x.detach().numpy()
    return np.asarray(x)


def brute_force(w, n):
    """Brute-force Z, P(S), pi for small instances."""
    N = len(w)
    all_S = list(combinations(range(N), n))
    scores = np.array([np.prod(w[list(s)]) for s in all_S])
    Z = scores.sum()
    probs = dict(zip(all_S, scores / Z))
    pi = np.zeros(N)
    for s, p in probs.items():
        for i in s:
            pi[i] += p
    return Z, probs, pi


@pytest.fixture(params=ALL_CLASSES, ids=lambda c: c.__name__)
def cls(request):
    return request.param


# ── Inclusion probabilities ──────────────────────────────────────────────────

class TestInclProb:
    def test_sums_to_n(self, cls):
        w = np.random.default_rng(0).exponential(1.0, 15)
        cp = cls.from_weights(5, w)
        pi = to_numpy(cp.incl_prob)
        assert abs(pi.sum() - 5) < 1e-6

    def test_in_unit_interval(self, cls):
        w = np.random.default_rng(0).exponential(1.0, 20)
        cp = cls.from_weights(7, w)
        pi = to_numpy(cp.incl_prob)
        assert np.all(pi > 0) and np.all(pi < 1)

    def test_matches_brute_force(self, cls):
        w = np.random.default_rng(42).exponential(1.0, 8)
        n = 3
        _, _, pi_bf = brute_force(w, n)
        cp = cls.from_weights(n, w)
        np.testing.assert_allclose(to_numpy(cp.incl_prob), pi_bf, atol=1e-8)

    def test_uniform_weights(self, cls):
        cp = cls.from_weights(3, np.ones(10))
        np.testing.assert_allclose(to_numpy(cp.incl_prob), 0.3, atol=1e-8)


# ── Log normalizer ───────────────────────────────────────────────────────────

class TestLogNormalizer:
    def test_finite(self, cls):
        w = np.random.default_rng(0).exponential(1.0, 15)
        cp = cls.from_weights(5, w)
        assert np.isfinite(cp.log_normalizer)

    def test_matches_brute_force(self, cls):
        w = np.random.default_rng(42).exponential(1.0, 8)
        n = 3
        Z_bf, _, _ = brute_force(w, n)
        cp = cls.from_weights(n, w)
        assert abs(cp.log_normalizer - np.log(Z_bf)) < 1e-8

    def test_uniform_is_log_comb(self, cls):
        cp = cls.from_weights(4, np.ones(10))
        assert abs(cp.log_normalizer - np.log(math.comb(10, 4))) < 1e-8

    def test_n_equals_N(self, cls):
        w = np.array([1.5, 2.0, 3.0])
        cp = cls.from_weights(3, w)
        assert abs(cp.log_normalizer - np.log(np.prod(w))) < 1e-8

    def test_n_equals_1(self, cls):
        w = np.array([1.5, 2.0, 3.0])
        cp = cls.from_weights(1, w)
        assert abs(cp.log_normalizer - np.log(np.sum(w))) < 1e-8

    def test_independent_of_call_order(self, cls):
        # Some classes compute log Z by a cheaper path when nothing else is cached.
        w = np.random.default_rng(3).exponential(1.0, 40)
        alone = cls.from_weights(15, w).log_normalizer
        cp = cls.from_weights(15, w)
        cp.incl_prob
        assert abs(cp.log_normalizer - alone) < 1e-12 * abs(alone)


# ── Log probability ──────────────────────────────────────────────────────────

class TestLogProb:
    def test_normalizes(self, cls):
        w = np.array([1.5, 2.0, 0.5, 3.0])
        cp = cls.from_weights(2, w)
        total = sum(np.exp(cp.log_prob(np.array(S))) for S in combinations(range(4), 2))
        assert abs(total - 1.0) < 1e-10

    def test_matches_brute_force(self, cls):
        w = np.random.default_rng(42).exponential(1.0, 7)
        n = 3
        _, probs_bf, _ = brute_force(w, n)
        cp = cls.from_weights(n, w)
        for S, p in probs_bf.items():
            assert abs(cp.log_prob(np.array(S)) - np.log(p)) < 1e-8


# ── Sampling ─────────────────────────────────────────────────────────────────

class TestSampling:
    def test_correct_shape(self, cls):
        cp = cls.from_weights(3, np.random.default_rng(0).exponential(1.0, 10))
        S = np.stack([to_numpy(cp.sample()) for _ in range(10)])
        assert S.shape == (10, 3)

    def test_sorted(self, cls):
        cp = cls.from_weights(3, np.random.default_rng(0).exponential(1.0, 10))
        for _ in range(20):
            s = to_numpy(cp.sample())
            assert np.all(np.diff(s) > 0)

    def test_empirical_pi_matches(self, cls):
        N, n = 10, 4
        w = np.random.default_rng(0).exponential(1.0, N)
        cp = cls.from_weights(n, w)
        pi = to_numpy(cp.incl_prob)
        M = 5_000
        S = np.stack([to_numpy(cp.sample()) for _ in range(M)])
        pi_emp = np.bincount(S.ravel(), minlength=N) / M
        assert np.max(np.abs(pi_emp - pi)) < 0.05

    def test_distribution_matches_brute_force(self, cls):
        N, n = 6, 2
        w = np.random.default_rng(42).exponential(1.0, N)
        cp = cls.from_weights(n, w)
        _, probs_bf, _ = brute_force(w, n)

        M = 10_000
        counts = {}
        for _ in range(M):
            s = tuple(sorted(to_numpy(cp.sample())))
            counts[s] = counts.get(s, 0) + 1

        max_err = max(abs(counts.get(s, 0) / M - probs_bf[s]) for s in probs_bf)
        assert max_err < 3 / np.sqrt(M)


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_n_equals_1(self, cls):
        cp = cls.from_weights(1, np.random.default_rng(1).exponential(1.0, 10))
        pi = to_numpy(cp.incl_prob)
        assert abs(pi.sum() - 1.0) < 1e-10
        S = np.stack([to_numpy(cp.sample()) for _ in range(50)])
        assert S.shape == (50, 1)

    def test_n_equals_N_minus_1(self, cls):
        N = 6
        cp = cls.from_weights(N - 1, np.random.default_rng(2).exponential(1.0, N))
        assert abs(to_numpy(cp.incl_prob).sum() - (N - 1)) < 1e-10

    def test_small_N(self, cls):
        cp = cls.from_weights(1, np.array([2.0, 3.0]))
        pi = to_numpy(cp.incl_prob)
        assert abs(pi[0] - 0.4) < 1e-10
        assert abs(pi[1] - 0.6) < 1e-10


# ── Fitting ──────────────────────────────────────────────────────────────────

class TestFit:
    def test_fit_recovers_pi(self, cls):
        w = np.random.default_rng(0).exponential(1.0, 10)
        n = 4
        cp_ref = cls.from_weights(n, w)
        target = to_numpy(cp_ref.incl_prob)
        cp_fit = cls.fit(target, n)
        pi_fit = to_numpy(cp_fit.incl_prob)
        assert np.max(np.abs(pi_fit - target)) < 1e-6

    @pytest.mark.parametrize("np_cls", [ConditionalPoissonNumPy, ConditionalPoissonSequentialNumPy],
                             ids=lambda c: c.__name__)
    def test_fit_reaches_tol(self, np_cls):
        # The README example.  L-BFGS stops at 2e-10 with the NumPy tree, because
        # the line search rejects a better point on rounding error in the loss;
        # fit must return the best point it evaluated.
        pi_star = np.array([0.6, 0.4, 0.8, 0.3, 0.9])
        cp = np_cls.fit(pi_star, n=3, tol=1e-10)
        assert np.max(np.abs(cp.incl_prob - pi_star)) < 1e-10

    def test_fit_uniform(self, cls):
        N, n = 8, 3
        target = np.full(N, n / N)
        cp = cls.fit(target, n)
        expected_p = 1.0 / math.comb(N, n)
        for S in combinations(range(N), n):
            assert abs(np.exp(cp.log_prob(np.array(S))) - expected_p) < 1e-6


# ── Stability ────────────────────────────────────────────────────────────────

class TestStability:
    def test_extreme_weight_ratio(self, cls):
        w = np.exp(np.random.default_rng(99).uniform(-10, 10, 15))
        cp = cls.from_weights(5, w)
        pi = to_numpy(cp.incl_prob)
        assert np.isfinite(pi).all()
        assert np.all(pi > 0) and np.all(pi < 1)
        assert abs(pi.sum() - 5) < 1e-6

    def test_large_uniform(self, cls):
        cp = cls.from_weights(20, np.ones(100))
        pi = to_numpy(cp.incl_prob)
        assert abs(pi.sum() - 20) < 1e-6
        np.testing.assert_allclose(pi, 0.2, atol=1e-8)

    def test_no_overflow_large_N(self, cls):
        # Z = C(N, n) c^n is far beyond the float64 range (about 10^1100 here).
        N, n, c = 3000, 1200, 2.0
        cp = cls.from_weights(n, np.full(N, c))
        log_Z = math.lgamma(N + 1) - math.lgamma(n + 1) - math.lgamma(N - n + 1) + n * math.log(c)
        assert abs(cp.log_normalizer - log_Z) < 1e-9 * abs(log_Z)
        np.testing.assert_allclose(to_numpy(cp.incl_prob), n / N, atol=1e-10)

    def test_many_ordinary_weights_exact(self, cls):
        # Ordinary weights, but many of them: the largest coefficient of
        # prod_i (1 + w_i z) is about 10^240 times the one at z^n, so an FFT
        # without rescaling loses it.  Checked against exact rational arithmetic.
        N, n = 1000, 10
        w = np.random.default_rng(0).exponential(1.0, N)
        W = [Fraction(float(wi)) for wi in w]
        E = [Fraction(1)] + [Fraction(0)] * n
        for wi in W:
            for k in range(n, 0, -1):
                E[k] += wi * E[k - 1]
        # pi_i = w_i e_{n-1}(w without i) / e_n(w), where removing item i gives
        # e_k(w without i) = e_k(w) - w_i e_{k-1}(w without i), e_0 = 1.
        pi_exact = []
        for wi in W:
            e = Fraction(1)
            for k in range(1, n):
                e = E[k] - wi * e
            pi_exact.append(float(wi * e / E[n]))
        log_Z = math.log(E[n].numerator) - math.log(E[n].denominator)
        cp = cls.from_weights(n, w)
        assert abs(cp.log_normalizer - log_Z) < 1e-12 * abs(log_Z)
        np.testing.assert_allclose(to_numpy(cp.incl_prob), pi_exact, rtol=1e-10, atol=1e-15)

    def test_extreme_weights_exact(self, cls):
        # Weights spanning e^{\pm 300}, checked against exact rational arithmetic.
        N, n = 200, 50
        w = np.exp(np.random.default_rng(7).uniform(-300, 300, N))
        E = [Fraction(1)] + [Fraction(0)] * n
        for wi in w:
            wi = Fraction(float(wi))
            for k in range(n, 0, -1):
                E[k] += wi * E[k - 1]
        log_Z = math.log(E[n].numerator) - math.log(E[n].denominator)
        cp = cls.from_weights(n, w)
        assert abs(cp.log_normalizer - log_Z) < 1e-12 * abs(log_Z)
        pi = to_numpy(cp.incl_prob)
        assert np.isfinite(pi).all() and abs(pi.sum() - n) < 1e-9
        S = to_numpy(cp.sample())
        assert len(set(S.tolist())) == n


# ── Input validation ─────────────────────────────────────────────────────────

class TestValidation:
    def test_rejects_nonpositive_weights(self, cls):
        with pytest.raises((ValueError, AssertionError)):
            cls.from_weights(2, np.array([1.0, -1.0, 2.0]))

    def test_rejects_inf_weights(self, cls):
        with pytest.raises((ValueError, AssertionError)):
            cls.from_weights(2, np.array([np.inf, 1.0, 2.0]))

    def test_rejects_zero_weights(self, cls):
        with pytest.raises((ValueError, AssertionError)):
            cls.from_weights(2, np.array([0.0, 1.0, 2.0]))


# ── Cross-implementation agreement ───────────────────────────────────────────

class TestAgreement:
    def test_pi_agrees(self):
        w = np.random.default_rng(0).exponential(1.0, 20)
        results = {}
        for c in ALL_CLASSES:
            cp = c.from_weights(7, w)
            results[c.__name__] = to_numpy(cp.incl_prob)
        ref = list(results.values())[0]
        for name, pi in results.items():
            assert np.max(np.abs(pi - ref)) < 1e-8, f"{name} disagrees"

    def test_log_Z_agrees(self):
        w = np.random.default_rng(0).exponential(1.0, 20)
        results = {}
        for c in ALL_CLASSES:
            cp = c.from_weights(7, w)
            results[c.__name__] = cp.log_normalizer
        ref = list(results.values())[0]
        for name, lz in results.items():
            assert abs(lz - ref) < 1e-8, f"{name} disagrees: {lz} vs {ref}"
