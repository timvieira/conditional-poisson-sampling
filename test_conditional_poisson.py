"""Tests for conditional_poisson.py"""

import numpy as np
import pytest
from itertools import combinations
from conditional_poisson import ConditionalPoisson


def test_forward_pass():
    rng = np.random.default_rng(0)
    N, n = 20, 7
    q_true = rng.exponential(1.0, N)
    cp = ConditionalPoisson.from_weights(n, q_true)
    pi = cp.pi
    assert abs(pi.sum() - n) < 1e-6
    assert np.all((pi > 0) & (pi < 1))
    assert np.isfinite(cp.log_normalizer)


def test_fitting():
    rng = np.random.default_rng(0)
    N, n = 20, 7
    q_true = rng.exponential(1.0, N)
    cp = ConditionalPoisson.from_weights(n, q_true)
    cp_fit = ConditionalPoisson.fit(cp.pi, n)
    assert np.max(np.abs(cp.pi - cp_fit.pi)) < 1e-8


def test_log_prob_normalises():
    rng = np.random.default_rng(0)
    cp = ConditionalPoisson.from_weights(3, rng.exponential(1.0, 6))
    all_S = list(combinations(range(6), 3))
    lps = np.array([cp.log_prob(list(S)) for S in all_S])
    lps_b = cp.log_prob(np.array([list(S) for S in all_S]))
    assert abs(np.exp(lps).sum() - 1.0) < 1e-8
    assert np.max(np.abs(lps - lps_b)) < 1e-12


def test_sampling():
    rng = np.random.default_rng(0)
    N, n = 20, 7
    q_true = rng.exponential(1.0, N)
    cp = ConditionalPoisson.from_weights(n, q_true)
    M = 100_000
    S = cp.sample(M, rng=rng)
    assert S.shape == (M, n)
    assert np.all(np.diff(S, axis=1) > 0), "samples not sorted"
    pi_emp = np.bincount(S.ravel(), minlength=N) / M
    assert np.max(np.abs(pi_emp - cp.pi)) < 0.02


def test_hvp():
    rng = np.random.default_rng(0)
    N, n = 20, 7
    q_true = rng.exponential(1.0, N)
    cp = ConditionalPoisson.from_weights(n, q_true)
    cp_fit = ConditionalPoisson.fit(cp.pi, n)
    v = rng.standard_normal(N)
    Hv = cp_fit.hvp(v)
    eps = 1e-5
    J = np.zeros((N, N))
    for j in range(N):
        ej = np.zeros(N); ej[j] = 1.0
        J[:, j] = (ConditionalPoisson(n, cp_fit.theta + eps * ej).pi -
                    ConditionalPoisson(n, cp_fit.theta - eps * ej).pi) / (2 * eps)
    assert np.max(np.abs(Hv - J @ v)) < 1e-5
    assert np.linalg.norm(cp_fit.hvp(np.ones(N))) < 1e-8


def test_numerical_stability():
    rng = np.random.default_rng(0)
    cases = [
        (4, rng.uniform(30, 50, 10)),
        (4, rng.uniform(-50, -30, 10)),
        (4, np.linspace(-30, 30, 10)),
        (50, np.full(100, 2.0)),
        (100, np.full(200, 10.0)),
        (250, np.full(500, 5.0)),
    ]
    for n, theta in cases:
        cp = ConditionalPoisson(n, theta)
        pi = cp.pi
        assert np.isfinite(pi).all(), f"non-finite pi for n={n}"
        assert np.isfinite(cp.log_normalizer), f"non-finite log_normalizer for n={n}"
        assert abs(pi.sum() - n) < 1e-4, f"pi.sum()={pi.sum()} != {n}"


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_n_equals_1():
    rng = np.random.default_rng(1)
    N = 10
    cp = ConditionalPoisson.from_weights(1, rng.exponential(1.0, N))
    pi = cp.pi
    assert abs(pi.sum() - 1.0) < 1e-10
    assert np.all((pi > 0) & (pi < 1))
    S = cp.sample(1000, rng=rng)
    assert S.shape == (1000, 1)
    assert np.all((S >= 0) & (S < N))


def test_n_equals_N_minus_1():
    rng = np.random.default_rng(2)
    N = 6
    cp = ConditionalPoisson.from_weights(N - 1, rng.exponential(1.0, N))
    pi = cp.pi
    assert abs(pi.sum() - (N - 1)) < 1e-10
    assert np.all((pi > 0) & (pi < 1))
    # brute force: C(6,5) = 6 subsets
    all_S = list(combinations(range(N), N - 1))
    lps = np.array([cp.log_prob(list(S)) for S in all_S])
    assert abs(np.exp(lps).sum() - 1.0) < 1e-8


def test_small_N():
    cp = ConditionalPoisson.from_weights(1, np.array([2.0, 3.0]))
    pi = cp.pi
    assert abs(pi.sum() - 1.0) < 1e-10
    assert abs(pi[0] - 2.0 / 5.0) < 1e-10
    assert abs(pi[1] - 3.0 / 5.0) < 1e-10


# ── Constructors ──────────────────────────────────────────────────────────────

def test_uniform():
    cp = ConditionalPoisson.uniform(10, 3)
    pi = cp.pi
    assert np.allclose(pi, 0.3)
    assert abs(pi.sum() - 3.0) < 1e-10


def test_from_weights():
    q = np.array([1.0, 1.0, 1.0, 1.0])
    cp = ConditionalPoisson.from_weights(2, q)
    assert np.allclose(cp.pi, 0.5)
    assert np.allclose(cp.theta, 0.0)


# ── Theta setter / cache invalidation ────────────────────────────────────────

def test_theta_setter_invalidates_cache():
    rng = np.random.default_rng(3)
    cp = ConditionalPoisson.from_weights(3, rng.exponential(1.0, 8))
    pi_old = cp.pi.copy()
    log_en_old = cp.log_normalizer

    cp.theta = rng.standard_normal(8)
    pi_new = cp.pi
    assert not np.allclose(pi_old, pi_new), "pi should change after theta update"
    assert cp.log_normalizer != log_en_old


# ── Input validation ─────────────────────────────────────────────────────────

def test_n_less_than_1_raises():
    with pytest.raises(ValueError, match="n must be >= 1"):
        ConditionalPoisson(0, np.zeros(5))


def test_N_less_than_n_raises():
    with pytest.raises(ValueError, match="N=.*must be >= n"):
        ConditionalPoisson(5, np.zeros(3))


def test_theta_not_1d_raises():
    with pytest.raises(ValueError, match="theta must be 1-D"):
        ConditionalPoisson(2, np.zeros((3, 3)))


def test_from_weights_nonpositive_raises():
    with pytest.raises(ValueError, match="all weights must be positive"):
        ConditionalPoisson.from_weights(2, np.array([1.0, -1.0, 2.0]))


def test_fit_bad_pi_sum_raises():
    with pytest.raises(ValueError, match="sum\\(pi_star\\)/n"):
        ConditionalPoisson.fit(np.array([0.5, 0.5, 0.5]), n=2)


def test_fit_pi_out_of_range_raises():
    with pytest.raises(ValueError, match="strictly in \\(0, 1\\)"):
        ConditionalPoisson.fit(np.array([0.0, 1.0, 1.0]), n=2)


# ── log_prob input formats ───────────────────────────────────────────────────

def test_log_prob_bool_formats():
    rng = np.random.default_rng(4)
    cp = ConditionalPoisson.from_weights(3, rng.exponential(1.0, 6))

    S_idx = np.array([0, 2, 4])
    S_bool = np.zeros(6, dtype=bool)
    S_bool[[0, 2, 4]] = True

    lp_idx = cp.log_prob(S_idx)
    lp_bool = cp.log_prob(S_bool)
    assert abs(lp_idx - lp_bool) < 1e-12

    # batch bool
    all_S_idx = np.array(list(combinations(range(6), 3)))
    all_S_bool = np.zeros((len(all_S_idx), 6), dtype=bool)
    for i, s in enumerate(all_S_idx):
        all_S_bool[i, s] = True

    lps_idx = cp.log_prob(all_S_idx)
    lps_bool = cp.log_prob(all_S_bool)
    assert np.max(np.abs(lps_idx - lps_bool)) < 1e-12


# ── Sampling reproducibility ─────────────────────────────────────────────────

def test_sampling_deterministic_with_seed():
    cp = ConditionalPoisson.from_weights(3, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    s1 = cp.sample(50, rng=42)
    s2 = cp.sample(50, rng=42)
    assert np.array_equal(s1, s2)


# ── Brute-force equivalence ───────────────────────────────────────────────────

def _brute_force(theta, n):
    """Compute everything by explicit enumeration over all C(N,n) subsets."""
    theta = np.asarray(theta, float)
    N = len(theta)
    all_S = list(combinations(range(N), n))

    # unnormalised log-probs
    log_w = np.array([theta[list(s)].sum() for s in all_S])
    log_w_max = log_w.max()
    w = np.exp(log_w - log_w_max)
    Z = w.sum()
    log_en = np.log(Z) + log_w_max     # log e_n(exp(theta))
    probs = w / Z                       # P(S) for each subset

    # inclusion probabilities: pi_i = sum_{S : i in S} P(S)
    pi = np.zeros(N)
    for k, s in enumerate(all_S):
        for i in s:
            pi[i] += probs[k]

    # pairwise inclusion: pi_ij = sum_{S : i,j in S} P(S)
    pi2 = np.zeros((N, N))
    for k, s in enumerate(all_S):
        for i in s:
            for j in s:
                pi2[i, j] += probs[k]

    # covariance: Cov[Z_i, Z_j] = pi_ij - pi_i pi_j  (i != j), pi_i(1-pi_i) on diag
    Cov = pi2 - np.outer(pi, pi)

    return log_en, pi, Cov, all_S, probs


_BRUTE_CASES = [
    (5, 2),
    (6, 3),
    (7, 1),
    (7, 6),
    (8, 4),
]


def test_brute_force_pi():
    rng = np.random.default_rng(10)
    for N, n in _BRUTE_CASES:
        theta = rng.standard_normal(N)
        cp = ConditionalPoisson(n, theta)
        _, pi_bf, _, _, _ = _brute_force(theta, n)
        assert np.allclose(cp.pi, pi_bf, atol=1e-10), \
            f"pi mismatch for N={N}, n={n}: max err={np.max(np.abs(cp.pi - pi_bf)):.2e}"


def test_brute_force_log_normalizer():
    rng = np.random.default_rng(11)
    for N, n in _BRUTE_CASES:
        theta = rng.standard_normal(N)
        cp = ConditionalPoisson(n, theta)
        log_en_bf, _, _, _, _ = _brute_force(theta, n)
        assert abs(cp.log_normalizer - log_en_bf) < 1e-10, \
            f"log_normalizer mismatch for N={N}, n={n}: {cp.log_normalizer} vs {log_en_bf}"


def test_brute_force_log_prob():
    rng = np.random.default_rng(12)
    for N, n in _BRUTE_CASES:
        theta = rng.standard_normal(N)
        cp = ConditionalPoisson(n, theta)
        _, _, _, all_S, probs_bf = _brute_force(theta, n)
        for k, s in enumerate(all_S):
            lp = cp.log_prob(list(s))
            assert abs(lp - np.log(probs_bf[k])) < 1e-10, \
                f"log_prob mismatch for N={N}, n={n}, S={s}"


def test_brute_force_hvp():
    rng = np.random.default_rng(13)
    for N, n in _BRUTE_CASES:
        theta = rng.standard_normal(N)
        cp = ConditionalPoisson(n, theta)
        _, _, Cov_bf, _, _ = _brute_force(theta, n)
        # test multiple random directions
        for _ in range(3):
            v = rng.standard_normal(N)
            Hv = cp.hvp(v)
            Hv_bf = Cov_bf @ v
            assert np.allclose(Hv, Hv_bf, atol=1e-8), \
                f"hvp mismatch for N={N}, n={n}: max err={np.max(np.abs(Hv - Hv_bf)):.2e}"


def test_brute_force_sampling_distribution():
    """Verify that sample frequencies converge to brute-force probabilities."""
    rng = np.random.default_rng(14)
    N, n = 5, 2
    theta = rng.standard_normal(N)
    cp = ConditionalPoisson(n, theta)
    _, _, _, all_S, probs_bf = _brute_force(theta, n)

    M = 200_000
    samples = cp.sample(M, rng=rng)

    # map each sample to its subset index
    S_to_idx = {s: i for i, s in enumerate(all_S)}
    counts = np.zeros(len(all_S))
    for row in samples:
        counts[S_to_idx[tuple(row)]] += 1
    freq = counts / M

    assert np.max(np.abs(freq - probs_bf)) < 0.01, \
        f"sampling distribution mismatch: max err={np.max(np.abs(freq - probs_bf)):.4f}"


if __name__ == "__main__":
    import time

    tests = [
        test_forward_pass,
        test_fitting,
        test_log_prob_normalises,
        test_sampling,
        test_hvp,
        test_numerical_stability,
        test_n_equals_1,
        test_n_equals_N_minus_1,
        test_small_N,
        test_uniform,
        test_from_weights,
        test_theta_setter_invalidates_cache,
        test_log_prob_bool_formats,
        test_sampling_deterministic_with_seed,
        test_brute_force_pi,
        test_brute_force_log_normalizer,
        test_brute_force_log_prob,
        test_brute_force_hvp,
        test_brute_force_sampling_distribution,
    ]
    for t in tests:
        t0 = time.perf_counter()
        t()
        ms = (time.perf_counter() - t0) * 1000
        print(f"  {t.__name__:40s} passed  ({ms:.0f} ms)")
