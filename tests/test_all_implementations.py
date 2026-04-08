"""Parameterized tests for all ConditionalPoisson implementations.

Every implementation must pass the same battery of correctness tests:
- inclusion probabilities sum to n and lie in (0, 1)
- log normalizer is finite and consistent with brute-force
- samples have the correct size and match the true distribution
- basic properties (uniform weights, small N, n=1, n=N-1)
"""

import numpy as np
import pytest
from itertools import combinations

from conditional_poisson.numpy import ConditionalPoissonNumPy
from conditional_poisson.sequential_numpy import ConditionalPoissonSequentialNumPy

# Torch imports may fail if torch not installed
torch_classes = []
try:
    from conditional_poisson.torch import ConditionalPoissonTorch
    torch_classes.append(ConditionalPoissonTorch)
except ImportError:
    pass
try:
    from conditional_poisson.sequential_torch import ConditionalPoissonSequentialTorch
    torch_classes.append(ConditionalPoissonSequentialTorch)
except ImportError:
    pass

ALL_CLASSES = [ConditionalPoissonNumPy, ConditionalPoissonSequentialNumPy] + torch_classes


def to_numpy(x):
    """Convert result to numpy array regardless of backend."""
    if hasattr(x, 'numpy'):
        return x.detach().numpy()
    return np.asarray(x)


@pytest.fixture(params=ALL_CLASSES, ids=lambda c: c.__name__)
def cls(request):
    return request.param


# ── Inclusion probabilities ──────────────────────────────────────────────────

class TestInclProb:
    def test_sums_to_n(self, cls):
        w = np.random.default_rng(0).exponential(1.0, 20)
        cp = cls.from_weights(7, w)
        pi = to_numpy(cp.incl_prob)
        assert abs(pi.sum() - 7) < 1e-6

    def test_in_unit_interval(self, cls):
        w = np.random.default_rng(0).exponential(1.0, 20)
        cp = cls.from_weights(7, w)
        pi = to_numpy(cp.incl_prob)
        assert np.all(pi > 0) and np.all(pi < 1)

    def test_uniform_weights(self, cls):
        cp = cls.from_weights(3, np.ones(10))
        pi = to_numpy(cp.incl_prob)
        assert np.allclose(pi, 0.3, atol=1e-8)

    def test_small_N(self, cls):
        cp = cls.from_weights(1, np.array([2.0, 3.0]))
        pi = to_numpy(cp.incl_prob)
        assert abs(pi[0] - 2.0 / 5.0) < 1e-10
        assert abs(pi[1] - 3.0 / 5.0) < 1e-10

    def test_matches_brute_force(self, cls):
        """Verify pi against brute-force enumeration for small instance."""
        N, n = 8, 3
        w = np.random.default_rng(42).exponential(1.0, N)
        cp = cls.from_weights(n, w)
        pi = to_numpy(cp.incl_prob)

        # Brute-force
        log_w = np.log(w)
        all_S = list(combinations(range(N), n))
        log_probs = np.array([log_w[list(s)].sum() for s in all_S])
        log_Z = np.log(np.exp(log_probs - log_probs.max()).sum()) + log_probs.max()
        probs = np.exp(log_probs - log_Z)
        pi_bf = np.zeros(N)
        for k, s in enumerate(all_S):
            for i in s:
                pi_bf[i] += probs[k]

        assert np.max(np.abs(pi - pi_bf)) < 1e-10


# ── Log normalizer ───────────────────────────────────────────────────────────

class TestLogNormalizer:
    def test_finite(self, cls):
        w = np.random.default_rng(0).exponential(1.0, 20)
        cp = cls.from_weights(7, w)
        assert np.isfinite(cp.log_normalizer)

    def test_matches_brute_force(self, cls):
        N, n = 8, 3
        w = np.random.default_rng(42).exponential(1.0, N)
        cp = cls.from_weights(n, w)

        log_w = np.log(w)
        all_S = list(combinations(range(N), n))
        log_probs = np.array([log_w[list(s)].sum() for s in all_S])
        log_Z_bf = np.log(np.exp(log_probs - log_probs.max()).sum()) + log_probs.max()

        assert abs(cp.log_normalizer - log_Z_bf) < 1e-10

    def test_agrees_across_implementations(self):
        """All implementations should agree on log Z."""
        w = np.random.default_rng(0).exponential(1.0, 20)
        results = {}
        for c in ALL_CLASSES:
            cp = c.from_weights(7, w)
            results[c.__name__] = cp.log_normalizer
        values = list(results.values())
        for v in values[1:]:
            assert abs(v - values[0]) < 1e-8, f"log_Z mismatch: {results}"


# ── Sampling ─────────────────────────────────────────────────────────────────

class TestSampling:
    def test_correct_shape(self, cls):
        w = np.random.default_rng(0).exponential(1.0, 20)
        cp = cls.from_weights(7, w)
        rng = np.random.default_rng(42)
        S = np.stack([to_numpy(cp.sample(rng=rng)) for _ in range(10)])
        assert S.shape == (10, 7)

    def test_sorted_indices(self, cls):
        w = np.random.default_rng(0).exponential(1.0, 20)
        cp = cls.from_weights(7, w)
        rng = np.random.default_rng(42)
        S = np.stack([to_numpy(cp.sample(rng=rng)) for _ in range(100)])
        assert np.all(np.diff(S, axis=1) > 0), "samples not sorted"

    def test_valid_indices(self, cls):
        N = 20
        w = np.random.default_rng(0).exponential(1.0, N)
        cp = cls.from_weights(7, w)
        rng = np.random.default_rng(42)
        S = np.stack([to_numpy(cp.sample(rng=rng)) for _ in range(100)])
        assert np.all(S >= 0) and np.all(S < N)

    def test_empirical_pi_matches(self, cls):
        N, n = 15, 5
        w = np.random.default_rng(0).exponential(1.0, N)
        cp = cls.from_weights(n, w)
        pi = to_numpy(cp.incl_prob)
        M = 50_000
        rng = np.random.default_rng(0)
        S = np.stack([to_numpy(cp.sample(rng=rng)) for _ in range(M)])
        pi_emp = np.bincount(S.ravel(), minlength=N) / M
        assert np.max(np.abs(pi_emp - pi)) < 0.02

    def test_distribution_matches_brute_force(self, cls):
        """For small N choose n, verify P(S) matches true distribution."""
        N, n = 8, 3
        w = np.random.default_rng(42).exponential(1.0, N)
        cp = cls.from_weights(n, w)

        # True P(S)
        log_w = np.log(w)
        all_S = list(combinations(range(N), n))
        log_probs = np.array([log_w[list(s)].sum() for s in all_S])
        log_Z = np.log(np.exp(log_probs - log_probs.max()).sum()) + log_probs.max()
        true_probs = dict(zip(all_S, np.exp(log_probs - log_Z)))

        # Empirical P(S) — use persistent RNG for all backends
        M = 200_000
        rng = np.random.default_rng(0)
        counts = {}
        for _ in range(M):
            s = tuple(sorted(to_numpy(cp.sample(rng=rng))))
            counts[s] = counts.get(s, 0) + 1

        max_err = max(abs(counts.get(s, 0) / M - true_probs[s]) for s in all_S)
        assert max_err < 3 / np.sqrt(M), f"max|P_hat(S) - P(S)| = {max_err:.5f}"


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_n_equals_1(self, cls):
        N = 10
        w = np.random.default_rng(1).exponential(1.0, N)
        cp = cls.from_weights(1, w)
        pi = to_numpy(cp.incl_prob)
        assert abs(pi.sum() - 1.0) < 1e-10
        rng = np.random.default_rng(42)
        S = np.stack([to_numpy(cp.sample(rng=rng)) for _ in range(100)])
        assert S.shape == (100, 1)

    def test_n_equals_N_minus_1(self, cls):
        N = 6
        w = np.random.default_rng(2).exponential(1.0, N)
        cp = cls.from_weights(N - 1, w)
        pi = to_numpy(cp.incl_prob)
        assert abs(pi.sum() - (N - 1)) < 1e-10


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
