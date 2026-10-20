"""Contour radius for rescaling weights.

For any r > 0, Z(w, n) = r^{-n} [z^n] prod_i (1 + w_i r z), so rescaling every
weight by r leaves the distribution unchanged.  The choice of r only affects
floating-point accuracy: the coefficients of prod_i (1 + w_i r z) are
proportional to the pmf of a sum of Bernoulli(w_i r / (1 + w_i r)) variables,
and choosing r so that this sum has mean n puts the peak of the coefficients
at degree n.  Then the coefficients that matter are never tiny relative to the
largest one, which is what both the FFT product tree and the scaled dynamic
program need.
"""

import numpy as np
from scipy.special import expit

_EPS = np.finfo(float).eps


def log_contour_radius(theta, n: int) -> float:
    """Return log r solving sum_i sigmoid(theta_i + log r) = n.

    At n = 0 or n = N the root is at -inf or +inf, so the target is clipped
    to [1/2, N - 1/2]; the coefficient at degree n is then still near the peak.

    g(t) = sum_i sigmoid(theta_i + t) - target is increasing in t.  We run
    Newton's method safeguarded by bisection on a bracket [lo, hi] with
    g(lo) <= 0 <= g(hi), and stop when g is zero to rounding error or the
    bracket has shrunk to machine precision.
    """
    theta = np.asarray(theta, dtype=float)
    N = len(theta)
    if N == 0:
        return 0.0
    target = min(max(float(n), 0.5), N - 0.5)
    base = np.log(target / (N - target))      # logit(target / N)

    # sigmoid(theta_i + lo) <= target/N for all i, so g(lo) <= 0; likewise g(hi) >= 0.
    lo, hi = base - theta.max(), base - theta.min()
    t = min(max(base - np.median(theta), lo), hi)
    g_tol = 4 * N * _EPS                      # rounding error of a sum of N sigmoids

    for _ in range(200):                      # safety net; converges in far fewer
        p = expit(theta + t)
        g = p.sum() - target
        if abs(g) <= g_tol:
            break
        if g > 0:
            hi = t
        else:
            lo = t
        if hi - lo <= 4 * _EPS * max(1.0, abs(t)):
            break
        gp = (p * (1 - p)).sum()
        t_newton = t - g / gp if gp > 0 else np.nan
        t = t_newton if lo < t_newton < hi else 0.5 * (lo + hi)
    return float(t)
