"""Tests for conditional_poisson.py"""

import numpy as np
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


if __name__ == "__main__":
    import time

    tests = [
        test_forward_pass,
        test_fitting,
        test_log_prob_normalises,
        test_sampling,
        test_hvp,
        test_numerical_stability,
    ]
    for t in tests:
        t0 = time.perf_counter()
        t()
        ms = (time.perf_counter() - t0) * 1000
        print(f"  {t.__name__:40s} passed  ({ms:.0f} ms)")
