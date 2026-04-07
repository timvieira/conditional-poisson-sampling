#!/usr/bin/env python3
"""
Tests for every theoretical identity and equality stated in the blog post.

Each test function is named after the claim it verifies, with a docstring
quoting or paraphrasing the relevant passage.
"""

import numpy as np
from itertools import combinations
from scipy.signal import convolve as poly_mul
from conditional_poisson_numpy import ConditionalPoissonNumPy


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
    return float(sum(np.prod([w[i] for i in s])
                     for s in combinations(range(N), k)))


def all_probs_bf(w, n):
    """Return dict {frozenset(S): P(S)} by brute force."""
    N = len(w)
    all_S = list(combinations(range(N), n))
    log_u = np.array([np.sum(np.log(w[list(s)])) for s in all_S])
    log_Z = np.log(np.sum(np.exp(log_u)))
    probs = np.exp(log_u - log_Z)
    return {frozenset(s): p for s, p in zip(all_S, probs)}, log_Z


def pi_bf(w, n):
    """Brute-force inclusion probabilities."""
    N = len(w)
    probs, _ = all_probs_bf(w, n)
    pi = np.zeros(N)
    for S, p in probs.items():
        for i in S:
            pi[i] += p
    return pi


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
W_SMALL = RNG.exponential(1.0, 8)  # N=8
N_SMALL = 3
W_MED = RNG.exponential(1.0, 12)   # N=12
N_MED = 5


# ===========================================================================
# 1. Definition of the distribution (Cell 2)
# ===========================================================================

def test_distribution_definition():
    """P(S) = prod(w_i, i in S) / Z(w,n) for all size-n subsets."""
    w, n = W_SMALL, N_SMALL
    cp = ConditionalPoissonNumPy.from_weights(n, w)
    probs, _ = all_probs_bf(w, n)
    Z = Z_bf(w, n)
    for S, p_bf in probs.items():
        p_formula = np.prod(w[list(S)]) / Z
        assert np.isclose(p_bf, p_formula, rtol=1e-12), \
            f"P({S}): brute_force={p_bf}, formula={p_formula}"


def test_Z_equals_binomial_when_uniform():
    """When w = 1, Z(w,n) = C(N,n)."""
    from math import comb
    for N in [5, 10, 15]:
        w = np.ones(N)
        for n in [0, 1, N // 2, N]:
            assert np.isclose(Z_bf(w, n), comb(N, n), rtol=1e-10), \
                f"N={N}, n={n}"


def test_Z_is_elementary_symmetric_poly():
    """Z(w,n) = e_n(w), the n-th elementary symmetric polynomial."""
    w, n = W_SMALL, N_SMALL
    cp = ConditionalPoissonNumPy.from_weights(n, w)
    assert np.isclose(np.exp(cp.log_normalizer), Z_bf(w, n), rtol=1e-10)


# ===========================================================================
# 2. Poisson sampling connection (Cell 2)
# ===========================================================================

def test_weight_is_odds():
    """w_i = p_i / (1 - p_i), i.e., the odds of the coin flip."""
    w = W_SMALL
    p = w / (1 + w)
    w_from_p = p / (1 - p)
    assert np.allclose(w, w_from_p, rtol=1e-14)


def test_conditional_poisson_from_bernoulli():
    """Conditioning independent Bernoulli(p_i) on exactly n heads gives CPS."""
    w, n = W_SMALL, N_SMALL
    N = len(w)
    p = w / (1 + w)
    # P(S | |S|=n) = prod(p_i, i in S) * prod(1-p_j, j not in S) / Pr[|S|=n]
    # = prod(w_i, i in S) * prod(1/(1+w_i)) / Pr[|S|=n]
    # ∝ prod(w_i, i in S)
    probs_bf, _ = all_probs_bf(w, n)
    all_S = list(combinations(range(N), n))
    # Compute via Bernoulli conditioning
    bern_probs = {}
    for S in all_S:
        S_set = frozenset(S)
        p_S = np.prod(p[list(S)]) * np.prod(1 - np.delete(p, list(S)))
        bern_probs[S_set] = p_S
    total = sum(bern_probs.values())
    for S_set in bern_probs:
        bern_probs[S_set] /= total
        assert np.isclose(bern_probs[S_set], probs_bf[S_set], rtol=1e-12), \
            f"S={S_set}"


# ===========================================================================
# 3. Scaling invariance (Cell 32)
# ===========================================================================

def test_scaling_invariance():
    """Scaling all weights by alpha > 0 doesn't change the distribution."""
    w, n = W_SMALL, N_SMALL
    alpha = 3.7
    probs1, _ = all_probs_bf(w, n)
    probs2, _ = all_probs_bf(alpha * w, n)
    for S in probs1:
        assert np.isclose(probs1[S], probs2[S], rtol=1e-12), \
            f"S={S}, P1={probs1[S]}, P2={probs2[S]}"


# ===========================================================================
# 4. Inclusion probabilities (Cells 9, 32)
# ===========================================================================

def test_pi_sums_to_n():
    """sum(pi_i) = n."""
    w, n = W_MED, N_MED
    cp = ConditionalPoissonNumPy.from_weights(n, w)
    assert np.isclose(cp.incl_prob.sum(), n, rtol=1e-12)


def test_pi_in_unit_interval():
    """Each pi_i in [0, 1]."""
    w, n = W_MED, N_MED
    cp = ConditionalPoissonNumPy.from_weights(n, w)
    assert np.all(cp.incl_prob >= -1e-15) and np.all(cp.incl_prob <= 1 + 1e-15)


def test_pi_matches_brute_force():
    """pi_i = P(i in S) computed by enumeration."""
    w, n = W_SMALL, N_SMALL
    cp = ConditionalPoissonNumPy.from_weights(n, w)
    pi_exact = pi_bf(w, n)
    assert np.allclose(cp.incl_prob, pi_exact, rtol=1e-10)


def test_pi_is_gradient_of_log_Z():
    """pi_i = d log Z / d theta_i (exponential family gradient identity)."""
    w, n = W_SMALL, N_SMALL
    theta = np.log(w)
    eps = 1e-7
    log_Z_0 = np.log(Z_bf(w, n))
    grad_numerical = np.empty(len(w))
    for i in range(len(w)):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        w_plus = np.exp(theta_plus)
        log_Z_plus = np.log(Z_bf(w_plus, n))
        grad_numerical[i] = (log_Z_plus - log_Z_0) / eps
    cp = ConditionalPoissonNumPy.from_weights(n, w)
    assert np.allclose(cp.incl_prob, grad_numerical, rtol=1e-5), \
        f"max err = {np.max(np.abs(cp.incl_prob - grad_numerical))}"


# ===========================================================================
# 5. Leave-one-out formula for pi (Cell 32)
# ===========================================================================

def test_pi_leave_one_out():
    """pi_i = w_i * Z(w^{-i}, n-1) / Z(w, n)."""
    w, n = W_SMALL, N_SMALL
    Z_full = Z_bf(w, n)
    cp = ConditionalPoissonNumPy.from_weights(n, w)
    for i in range(len(w)):
        w_minus_i = np.delete(w, i)
        Z_leave = Z_bf(w_minus_i, n - 1)
        pi_formula = w[i] * Z_leave / Z_full
        assert np.isclose(cp.incl_prob[i], pi_formula, rtol=1e-10), \
            f"i={i}: cp.incl_prob={cp.incl_prob[i]}, formula={pi_formula}"


# ===========================================================================
# 6. Higher-order inclusion probabilities (Cell 32)
# ===========================================================================

def test_higher_order_inclusion():
    """pi(X) = P(X ⊆ S) = prod(w_i, i in X) * Z(w^{-X}, n-|X|) / Z(w, n)."""
    w, n = W_SMALL, N_SMALL
    N = len(w)
    probs, _ = all_probs_bf(w, n)
    Z_full = Z_bf(w, n)
    all_S = list(probs.keys())
    for X in [(0,), (1,), (0, 2), (3, 5), (0, 1, 2)]:
        X = list(X)
        if len(X) > n:
            continue
        pi_bf_X = sum(p for S, p in probs.items() if set(X).issubset(S))
        w_minus_X = np.delete(w, X)
        pi_formula = np.prod(w[X]) * Z_bf(w_minus_X, n - len(X)) / Z_full
        assert np.isclose(pi_bf_X, pi_formula, rtol=1e-10), \
            f"X={X}: bf={pi_bf_X}, formula={pi_formula}"


# ===========================================================================
# 7. Covariance is Hessian of log Z (Cell 32)
# ===========================================================================

def test_covariance_is_hessian():
    """Cov[1_{i in S}, 1_{j in S}] = d^2 log Z / d theta_i d theta_j."""
    w, n = W_SMALL, N_SMALL
    N = len(w)
    probs, _ = all_probs_bf(w, n)
    pi_exact = pi_bf(w, n)

    # Brute-force covariance
    cov_bf = np.zeros((N, N))
    for S, p in probs.items():
        indicator = np.array([1.0 if i in S else 0.0 for i in range(N)])
        cov_bf += p * np.outer(indicator - pi_exact, indicator - pi_exact)

    # Numerical Hessian of log Z
    theta = np.log(w)
    eps = 1e-5
    hessian = np.zeros((N, N))
    log_Z_0 = np.log(Z_bf(w, n))
    for i in range(N):
        for j in range(i, N):
            t_pp = theta.copy(); t_pp[i] += eps; t_pp[j] += eps
            t_pm = theta.copy(); t_pm[i] += eps; t_pm[j] -= eps
            t_mp = theta.copy(); t_mp[i] -= eps; t_mp[j] += eps
            t_mm = theta.copy(); t_mm[i] -= eps; t_mm[j] -= eps
            hessian[i, j] = (np.log(Z_bf(np.exp(t_pp), n))
                           - np.log(Z_bf(np.exp(t_pm), n))
                           - np.log(Z_bf(np.exp(t_mp), n))
                           + np.log(Z_bf(np.exp(t_mm), n))) / (4 * eps * eps)
            hessian[j, i] = hessian[i, j]

    assert np.allclose(cov_bf, hessian, atol=1e-4), \
        f"max err = {np.max(np.abs(cov_bf - hessian))}"


# ===========================================================================
# 8. Product polynomial gives Z (Cell 15)
# ===========================================================================

def test_product_polynomial_coefficients():
    """[z^k] prod(1 + w_i z) = Z(w, k) = e_k(w)."""
    w = W_SMALL
    # Build product polynomial
    poly = np.array([1.0])
    for wi in w:
        poly = poly_mul(poly, [1.0, wi])
    for k in range(len(w) + 1):
        assert np.isclose(poly[k], Z_bf(w, k), rtol=1e-10), \
            f"k={k}: poly={poly[k]}, Z_bf={Z_bf(w, k)}"


# ===========================================================================
# 9. Parameterization relation (Cell 32)
# ===========================================================================

def test_prob_odds_generating_function_relation():
    """prod((1-p_i) + p_i z) = prod(1-p_i) * prod(1 + w_i z).

    Equivalently: Pr[k heads] = Z(w,k) / prod(1 + w_i).
    """
    w = W_SMALL
    p = w / (1 + w)
    # Build both generating functions
    gf_p = np.array([1.0])
    for pi in p:
        gf_p = poly_mul(gf_p, [1 - pi, pi])
    gf_w = np.array([1.0])
    for wi in w:
        gf_w = poly_mul(gf_w, [1.0, wi])
    prod_1_minus_p = np.prod(1 - p)
    # Check relation
    assert np.allclose(gf_p, prod_1_minus_p * gf_w, rtol=1e-10)
    # Check Pr[k heads] = Z(w,k) / prod(1+w_i)
    prod_1_plus_w = np.prod(1 + w)
    for k in range(len(w) + 1):
        pr_k = gf_p[k]
        z_k = Z_bf(w, k)
        assert np.isclose(pr_k, z_k / prod_1_plus_w, rtol=1e-10)


# ===========================================================================
# 10. Weighted Pascal recurrence (Cell 32)
# ===========================================================================

def test_weighted_pascal_recurrence():
    """Z(w1..wm, k) = Z(w1..w_{m-1}, k) + w_m * Z(w1..w_{m-1}, k-1)."""
    w = W_SMALL
    N = len(w)
    n = N_SMALL
    for m in range(1, N + 1):
        for k in range(n + 1):
            z_include = w[m - 1] * Z_bf(w[:m - 1], k - 1) if k > 0 else 0.0
            z_exclude = Z_bf(w[:m - 1], k)
            z_total = z_include + z_exclude
            z_direct = Z_bf(w[:m], k)
            assert np.isclose(z_total, z_direct, rtol=1e-10), \
                f"m={m}, k={k}: recurrence={z_total}, direct={z_direct}"


# ===========================================================================
# 11. Weighted Vandermonde identity (Cell 32)
# ===========================================================================

def test_weighted_vandermonde():
    """Z((a,b), k) = sum_{j=0}^{k} Z(a, j) * Z(b, k-j)."""
    w = W_SMALL
    # Split into two groups
    mid = len(w) // 2
    a, b = w[:mid], w[mid:]
    for k in range(len(w) + 1):
        lhs = Z_bf(w, k)
        rhs = sum(Z_bf(a, j) * Z_bf(b, k - j) for j in range(k + 1))
        assert np.isclose(lhs, rhs, rtol=1e-10), \
            f"k={k}: lhs={lhs}, rhs={rhs}"


# ===========================================================================
# 12. Newton's identities (Cell 32)
# ===========================================================================

def test_newtons_identities():
    """Z(w, k) = sum_{i=1}^{k} (-1)^{i-1}/k * Z(w, k-i) * g_i,
    where g_i = sum(w_j^i).
    """
    w = W_SMALL
    N = len(w)
    # Compute power sums
    g = [np.sum(w ** i) for i in range(N + 1)]  # g[0] unused
    # Build Z via Newton's identities
    Z_newton = np.zeros(N + 1)
    Z_newton[0] = 1.0
    for k in range(1, N + 1):
        Z_newton[k] = sum((-1) ** (i - 1) / k * Z_newton[k - i] * g[i]
                          for i in range(1, k + 1))
    # Compare to brute force
    for k in range(N + 1):
        assert np.isclose(Z_newton[k], Z_bf(w, k), rtol=1e-8), \
            f"k={k}: newton={Z_newton[k]}, bf={Z_bf(w, k)}"


# ===========================================================================
# 13. K-DPP with diagonal kernel (Cell 32)
# ===========================================================================

def test_kdpp_diagonal():
    """CPS with weights w equals K-DPP with diagonal kernel L = diag(w)."""
    w, n = W_SMALL, N_SMALL
    N = len(w)
    probs_cps, _ = all_probs_bf(w, n)
    # K-DPP: P(S) = det(L_S) / sum_{|S'|=K} det(L_{S'})
    # For diagonal L, det(L_S) = prod(w_i, i in S)
    all_S = list(combinations(range(N), n))
    dets = {frozenset(s): np.prod(w[list(s)]) for s in all_S}
    total = sum(dets.values())
    for S in dets:
        p_dpp = dets[S] / total
        assert np.isclose(p_dpp, probs_cps[S], rtol=1e-12), \
            f"S={S}: DPP={p_dpp}, CPS={probs_cps[S]}"


# ===========================================================================
# 14. Acceptance rates (Cells 3, 5)
# ===========================================================================

def test_bernoulli_acceptance_rate():
    """Bernoulli acceptance = Pr[exactly n heads] = Z(w,n) / prod(1+w_i)."""
    w, n = W_SMALL, N_SMALL
    p = w / (1 + w)
    # Exact: probability of exactly n heads from N independent coins
    N = len(w)
    all_S = list(combinations(range(N), n))
    acc = sum(np.prod(p[list(s)]) * np.prod(1 - np.delete(p, list(s)))
              for s in all_S)
    acc_formula = Z_bf(w, n) / np.prod(1 + w)
    assert np.isclose(acc, acc_formula, rtol=1e-10)


def test_categorical_acceptance_rate():
    """Categorical acceptance = n! * sum_{|S|=n} prod(p_i, i in S),
    where p_i = w_i / sum(w).
    """
    w, n = W_SMALL, N_SMALL
    N = len(w)
    p_cat = w / w.sum()
    all_S = list(combinations(range(N), n))
    from math import factorial
    acc = factorial(n) * sum(np.prod(p_cat[list(s)]) for s in all_S)
    # Verify by simulation
    rng = np.random.default_rng(123)
    trials = 200_000
    accepted = 0
    for _ in range(trials):
        draws = rng.choice(N, size=n, replace=True, p=p_cat)
        if len(set(draws)) == n:
            accepted += 1
    acc_sim = accepted / trials
    assert abs(acc - acc_sim) < 0.01, \
        f"formula={acc:.4f}, simulated={acc_sim:.4f}"


# ===========================================================================
# 15. Rejection samplers produce CPS (Cell 3)
# ===========================================================================

def test_rejection_bernoulli_produces_cps():
    """Bernoulli rejection sampling produces P(S) ∝ prod(w_i, i in S)."""
    w, n = W_SMALL[:6], 2  # small for tractable simulation
    N = len(w)
    probs_exact, _ = all_probs_bf(w, n)
    p = w / (1 + w)
    rng = np.random.default_rng(0)
    counts = {}
    trials = 500_000
    for _ in range(trials):
        S = frozenset(np.where(rng.random(N) < p)[0])
        if len(S) == n:
            counts[S] = counts.get(S, 0) + 1
    total = sum(counts.values())
    for S, p_exact in probs_exact.items():
        p_sim = counts.get(S, 0) / total
        assert abs(p_sim - p_exact) < 0.01, \
            f"S={S}: exact={p_exact:.4f}, sim={p_sim:.4f}"


# ===========================================================================
# 16. Horvitz-Thompson estimator (Cell 31)
# ===========================================================================

def test_horvitz_thompson_unbiased():
    """E[sum_{i in S} p(i)/pi_i * f(i)] = sum_i p(i) * f(i)."""
    w, n = W_SMALL, N_SMALL
    N = len(w)
    cp = ConditionalPoissonNumPy.from_weights(n, w)
    rng = np.random.default_rng(0)

    # Arbitrary f and p
    f = rng.standard_normal(N)
    p_dist = rng.dirichlet(np.ones(N))
    mu_true = np.sum(p_dist * f)

    # Monte Carlo estimate of E[HT]
    M = 200_000
    samples = cp.sample(M, rng=rng)
    ht_estimates = np.array([
        np.sum(p_dist[s] / cp.incl_prob[s] * f[s]) for s in samples
    ])
    mu_ht = ht_estimates.mean()
    se = ht_estimates.std() / np.sqrt(M)
    assert abs(mu_ht - mu_true) < 4 * se, \
        f"mu_true={mu_true:.6f}, mu_ht={mu_ht:.6f}, 4*se={4*se:.6f}"


# ===========================================================================
# 17. Maximum entropy (Cell 2)
# ===========================================================================

def test_max_entropy():
    """CPS is the max-entropy distribution over size-n subsets with given pi."""
    from scipy.optimize import minimize

    w, n = W_SMALL[:6], 2
    N = len(w)
    probs_cps, _ = all_probs_bf(w, n)
    pi_target = pi_bf(w, n)

    # CPS entropy
    H_cps = -sum(p * np.log(p) for p in probs_cps.values() if p > 0)

    # Enumerate all size-n subsets and build the membership matrix A
    # where A[i, j] = 1 if item i is in subset j.
    all_S = sorted(probs_cps.keys(), key=lambda s: tuple(sorted(s)))
    M = len(all_S)
    A = np.zeros((N, M))
    for j, S in enumerate(all_S):
        for i in S:
            A[i, j] = 1.0

    # Maximize entropy subject to: A @ q = pi_target.
    # Softmax parameterization q = softmax(u) ensures q > 0 and sum(q) = 1,
    # leaving only the marginal constraint, enforced via penalty.
    def softmax(u):
        u = u - u.max()
        e = np.exp(u)
        return e / e.sum()

    p_cps = np.array([probs_cps[S] for S in all_S])

    # Penalty method: minimize -H(softmax(u)) + mu * ||A @ softmax(u) - pi||^2
    # Increasing mu drives the constraint violation to zero.
    u = np.zeros(M)
    for mu in [1e2, 1e4, 1e6, 1e8]:
        def objective(u):
            q = softmax(u)
            neg_H = np.sum(q * np.log(q))
            violation = A @ q - pi_target
            return neg_H + mu * np.dot(violation, violation)
        result = minimize(objective, u, method='L-BFGS-B',
                          options={'ftol': 1e-15, 'maxiter': 2000})
        u = result.x

    q_opt = softmax(u)
    H_opt = -np.sum(q_opt * np.log(q_opt))
    assert np.allclose(A @ q_opt, pi_target, atol=1e-6), \
        f"Marginal constraint violated: max err {np.max(np.abs(A @ q_opt - pi_target)):.2e}"
    assert np.isclose(H_opt, H_cps, rtol=1e-6), \
        f"CPS entropy {H_cps:.10f} != optimal {H_opt:.10f}"
    assert np.allclose(q_opt, p_cps, atol=1e-4), \
        f"Optimal distribution differs from CPS: max diff {np.max(np.abs(q_opt - p_cps)):.2e}"


# ===========================================================================
# 18. Fitting: gradient and optimality (Cell 19)
# ===========================================================================

def test_fitting_gradient():
    """Gradient of L(theta) = pi* - pi(theta)."""
    w, n = W_SMALL, N_SMALL
    cp = ConditionalPoissonNumPy.from_weights(n, w)
    pi_star = np.full(len(w), n / len(w))  # uniform target

    # L(theta) = pi*^T theta - log Z(w, n)
    theta = np.log(w)
    grad = pi_star - cp.incl_prob
    # Numerical gradient
    eps = 1e-7
    grad_num = np.empty(len(w))
    for i in range(len(w)):
        theta_p = theta.copy(); theta_p[i] += eps
        theta_m = theta.copy(); theta_m[i] -= eps
        L_p = pi_star @ theta_p - np.log(Z_bf(np.exp(theta_p), n))
        L_m = pi_star @ theta_m - np.log(Z_bf(np.exp(theta_m), n))
        grad_num[i] = (L_p - L_m) / (2 * eps)
    assert np.allclose(grad, grad_num, rtol=1e-5)


def test_fitting_recovers_target():
    """Fitting to target pi* recovers pi = pi*."""
    N, n = 10, 4
    rng = np.random.default_rng(7)
    # Generate valid target: each pi_i strictly in (0,1), sum = n
    pi_star = rng.uniform(0.1, 0.9, N)
    pi_star *= n / pi_star.sum()
    # Ensure strict (0,1) after rescaling
    pi_star = np.clip(pi_star, 0.05, 0.95)
    pi_star *= n / pi_star.sum()
    cp = ConditionalPoissonNumPy.fit(pi_star, n)
    assert np.allclose(cp.incl_prob, pi_star, atol=1e-10), \
        f"max err = {np.max(np.abs(cp.incl_prob - pi_star))}"


# ===========================================================================
# 19. Contour scaling identity (Cell 26)
# ===========================================================================

def test_contour_scaling():
    """[z^n] prod(1 + w_i r z) = Z(w, n) * r^n."""
    w, n = W_SMALL, N_SMALL
    r = 2.5
    # Build product polynomial with scaled weights
    poly_scaled = np.array([1.0])
    for wi in w:
        poly_scaled = poly_mul(poly_scaled, [1.0, wi * r])
    Z_direct = Z_bf(w, n)
    assert np.isclose(poly_scaled[n], Z_direct * r ** n, rtol=1e-10)


def test_contour_r_solves_expected_size():
    """The optimal r satisfies sum(w_i r / (1 + w_i r)) = n."""
    w, n = W_MED, N_MED
    cp = ConditionalPoissonNumPy.from_weights(n, w)
    # The internal scaling factor: when we set p_i = w_i r / (1 + w_i r),
    # the expected number of heads is n.
    # We can recover r from pi: pi_i = w_i * Z(w^{-i}, n-1) / Z(w, n)
    # But more directly, let's just verify the equation for a few r values
    from scipy.optimize import brentq

    def expected_size(log_r):
        r = np.exp(log_r)
        return np.sum(w * r / (1 + w * r)) - n

    log_r = brentq(expected_size, -20, 20)
    r = np.exp(log_r)
    assert np.isclose(np.sum(w * r / (1 + w * r)), n, rtol=1e-10)


def test_rescaling_dynamic_range():
    """Numerical validation table: r=1 has DR~10^16, r=r* has DR~1."""
    N, n = 200, 10
    rng = np.random.default_rng(123)

    for regime in ['mild', 'heavy']:
        if regime == 'mild':
            w = rng.exponential(1.0, N)
        else:
            # Heavy tails: mix of very small and very large weights
            w = np.exp(rng.normal(0, 4, N))

        # Compute exact log Z via ConditionalPoissonNumPy (uses optimal r internally)
        cp = ConditionalPoissonNumPy.from_weights(n, w)
        log_Z_exact = cp.log_normalizer

        # Find optimal r
        from scipy.optimize import brentq
        def expected_size(log_r):
            r = np.exp(log_r)
            return np.sum(w * r / (1 + w * r)) - n
        log_r_star = brentq(expected_size, -50, 50)
        r_star = np.exp(log_r_star)

        for r, expect_good in [(1.0, False), (n / np.sum(w), True), (r_star, True)]:
            # Build product polynomial with scaled weights
            poly = np.array([1.0])
            for wi in w:
                poly = poly_mul(poly, [1.0, wi * r])
                if len(poly) > n + 1:
                    poly = poly[:n + 1]

            c_n = poly[n]
            dr = np.max(np.abs(poly)) / np.abs(c_n)
            log_Z_fft = np.log(c_n) - n * np.log(r)
            err = np.abs(log_Z_fft - log_Z_exact)

            if expect_good:
                # r=n/W and r=r* should give small error
                assert err < 1e-6, \
                    f"{regime} r={r:.2g}: err={err:.1e} too large"
            # r=r* should have DR close to 1
            if r == r_star:
                assert dr < 100, \
                    f"{regime} r=r*: DR={dr:.1e} too large"


# ===========================================================================
# 20. Exponential family structure (Cell 2)
# ===========================================================================

def test_exponential_family_form():
    """log P(S) = sum_{i in S} theta_i - log Z, where theta_i = log w_i."""
    w, n = W_SMALL, N_SMALL
    theta = np.log(w)
    probs, log_Z = all_probs_bf(w, n)
    for S, p in probs.items():
        log_p_formula = sum(theta[i] for i in S) - log_Z
        assert np.isclose(np.log(p), log_p_formula, rtol=1e-10), \
            f"S={S}"


# ===========================================================================
# 21. D-tree product rule (Cell 19)
# ===========================================================================

def test_dtree_product_rule():
    """D[node] = D[L] * P[R] + P[L] * D[R] gives the directional derivative
    of the product polynomial.
    """
    w = W_SMALL
    N = len(w)
    v = RNG.standard_normal(N)

    # P(z) = prod(1 + w_i z), D(z) = sum_i v_i w_i prod_{j≠i}(1 + w_j z)
    # Split in half
    mid = N // 2
    wL, wR = w[:mid], w[mid:]
    vL, vR = v[:mid], v[mid:]

    # Build poly_L, poly_R
    PL = np.array([1.0])
    for wi in wL:
        PL = poly_mul(PL, [1.0, wi])
    PR = np.array([1.0])
    for wi in wR:
        PR = poly_mul(PR, [1.0, wi])

    # Build D_L = sum_{i in L} v_i w_i prod_{j in L, j≠i}(1 + w_j z)
    DL = np.zeros(mid)  # degree mid-1
    for i in range(mid):
        term = np.array([vL[i] * wL[i]])
        for j in range(mid):
            if j != i:
                term = poly_mul(term, [1.0, wL[j]])
        if len(term) > len(DL):
            DL = np.pad(DL, (0, len(term) - len(DL)))
        DL[:len(term)] += term

    DR = np.zeros(len(wR))
    for i in range(len(wR)):
        term = np.array([vR[i] * wR[i]])
        for j in range(len(wR)):
            if j != i:
                term = poly_mul(term, [1.0, wR[j]])
        if len(term) > len(DR):
            DR = np.pad(DR, (0, len(term) - len(DR)))
        DR[:len(term)] += term

    # Product rule: D_root = D_L * poly_R + poly_L * D_R
    D_root = poly_mul(DL, PR) + poly_mul(PL, DR)

    # Direct computation: D_root = sum_i v_i w_i prod_{j≠i}(1 + w_j z)
    D_direct = np.zeros(N)
    for i in range(N):
        term = np.array([v[i] * w[i]])
        for j in range(N):
            if j != i:
                term = poly_mul(term, [1.0, w[j]])
        if len(term) > len(D_direct):
            D_direct = np.pad(D_direct, (0, len(term) - len(D_direct)))
        D_direct[:len(term)] += term

    assert np.allclose(D_root[:N], D_direct[:N], rtol=1e-10)


# ===========================================================================
# 22. Sampling correctness (Cell 15)
# ===========================================================================

def test_sampling_distribution():
    """Sampling produces the correct distribution (chi-squared test)."""
    w, n = W_SMALL[:6], 2  # small enough for brute force
    N = len(w)
    cp = ConditionalPoissonNumPy.from_weights(n, w)
    probs_exact, _ = all_probs_bf(w, n)
    all_S = sorted(probs_exact.keys(), key=lambda s: tuple(sorted(s)))
    p_expected = np.array([probs_exact[S] for S in all_S])

    M = 300_000
    rng = np.random.default_rng(42)
    samples = cp.sample(M, rng=rng)
    counts = {}
    for s in samples:
        key = frozenset(s)
        counts[key] = counts.get(key, 0) + 1
    p_observed = np.array([counts.get(S, 0) / M for S in all_S])

    # Chi-squared statistic
    chi2 = M * np.sum((p_observed - p_expected) ** 2 / p_expected)
    df = len(all_S) - 1
    # Should be roughly chi2(df); reject at p < 0.001
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df)
    assert p_value > 0.001, f"chi2={chi2:.1f}, df={df}, p={p_value:.6f}"


# ---------------------------------------------------------------------------
# Boundary weights: w_i = 0 (excluded) and w_i = inf (forced inclusion)
# ---------------------------------------------------------------------------

def test_boundary_zero_weight_pi():
    """w_i = 0 implies pi_i = 0 (item never selected); pi still sums to n."""
    w = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    cp = ConditionalPoissonNumPy.from_weights(n=2, w=w)
    assert cp.incl_prob[0] == 0.0, f"pi[0] should be 0 when w[0]=0, got {cp.incl_prob[0]}"
    assert np.isclose(np.sum(cp.incl_prob), 2), f"pi should sum to n=2, got {np.sum(cp.incl_prob)}"


def test_boundary_inf_weight_pi():
    """w_i = inf implies pi_i = 1 (item always selected); pi still sums to n."""
    w = np.array([np.inf, 1.0, 2.0, 3.0, 4.0])
    cp = ConditionalPoissonNumPy.from_weights(n=3, w=w)
    assert cp.incl_prob[0] == 1.0, f"pi[0] should be 1 when w[0]=inf, got {cp.incl_prob[0]}"
    assert np.isclose(np.sum(cp.incl_prob), 3), f"pi should sum to n=3, got {np.sum(cp.incl_prob)}"


def test_boundary_mixed_zero_inf():
    """Mix of w=0, w=inf, and finite; pi=0 for zeros, pi=1 for infs, sum=n."""
    w = np.array([np.inf, 0.0, 1.0, 2.0, 3.0, np.inf, 0.0])
    n = 4
    cp = ConditionalPoissonNumPy.from_weights(n=n, w=w)
    assert cp.incl_prob[0] == 1.0
    assert cp.incl_prob[5] == 1.0
    assert cp.incl_prob[1] == 0.0
    assert cp.incl_prob[6] == 0.0
    assert np.isclose(np.sum(cp.incl_prob), n)
    # Remaining pi should sum to n - #inf = 2
    finite_mask = np.isfinite(w) & (w > 0)
    assert np.isclose(np.sum(cp.incl_prob[finite_mask]), n - 2)


def test_boundary_all_determined():
    """n items have w=inf, rest have w=0 — fully determined subset."""
    w = np.array([np.inf, np.inf, 0.0, 0.0, 0.0])
    cp = ConditionalPoissonNumPy.from_weights(n=2, w=w)
    assert np.allclose(cp.incl_prob, [1, 1, 0, 0, 0])
    # Only one possible subset, so log_prob = 0
    assert np.isclose(cp.log_prob(np.array([0, 1])), 0.0, atol=1e-12)


def test_boundary_sampling_respects_zero_inf():
    """Samples never include w=0 items, always include w=inf items."""
    w = np.array([np.inf, 0.0, 1.0, 2.0, 3.0])
    cp = ConditionalPoissonNumPy.from_weights(n=3, w=w)
    rng = np.random.default_rng(123)
    samples = cp.sample(500, rng=rng)
    # Item 0 (inf) must appear in every sample
    assert np.all(np.isin(0, samples.T)), "w=inf item missing from some samples"
    # Item 1 (zero) must never appear
    assert not np.any(np.isin(1, samples.T)), "w=0 item appeared in a sample"


# ===========================================================================
# Non-asymptotic Poisson approximation bound
# ===========================================================================

def test_poisson_approximation_bound():
    """Non-asymptotic: |pi_i - p_i| <= p_i(1-p_i) / d for all i."""
    from scipy.optimize import brentq
    rng = np.random.default_rng(0)
    for _ in range(500):
        N = rng.integers(3, 50)
        n = rng.integers(1, N)
        w = np.exp(rng.uniform(-5, 5, N))
        cp = ConditionalPoissonNumPy.from_weights(n, w)
        pi = cp.incl_prob
        def f(logr):
            r = np.exp(logr)
            return np.sum(w * r / (1 + w * r)) - n
        logr = brentq(f, -100, 100)
        r = np.exp(logr)
        p = w * r / (1 + w * r)
        d = np.sum(p * (1 - p))
        for i in range(N):
            if p[i] < 1e-12 or p[i] > 1 - 1e-12:
                continue
            bound = p[i] * (1 - p[i]) / d
            err = abs(pi[i] - p[i])
            assert err <= bound * (1 + 1e-9) + 1e-14, (
                f"Bound violated: |pi[{i}]-p[{i}]|={err:.2e} > p(1-p)/d={bound:.2e} "
                f"(N={N}, n={n}, d={d:.3f})"
            )


# ===========================================================================
# Runner
# ===========================================================================

ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith('test_')]

if __name__ == '__main__':
    import time
    failed = []
    for test in ALL_TESTS:
        name = test.__name__
        t0 = time.perf_counter()
        try:
            test()
            ms = (time.perf_counter() - t0) * 1000
            print(f'  {name:50s} passed  ({ms:.0f} ms)')
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            print(f'  {name:50s} FAILED  ({ms:.0f} ms)')
            print(f'    {e}')
            failed.append(name)
    print()
    if failed:
        print(f'{len(failed)} FAILED: {", ".join(failed)}')
    else:
        print(f'All {len(ALL_TESTS)} tests passed!')
