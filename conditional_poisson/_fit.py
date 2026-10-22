"""Fit theta to target inclusion probabilities.  Shared by all four classes.

The loss is the negated dual objective from the article,

    f(theta) = log Z(theta) - target . theta,    grad f(theta) = pi(theta) - target,

which is convex.  It is minimized by L-BFGS until max |pi - target| <= tol.

Near the optimum, the decrease in f per step is about |grad|^2, which falls
below the rounding error in f (about machine epsilon times |f|, and |f| grows
with N).  A line search that compares values of f then rejects good steps on
rounding alone and stalls with |grad| around 1e-8.  The gradient stays accurate
to about machine epsilon, so the line search here also accepts a step on the
approximate Wolfe conditions of Hager & Zhang (2005), which use only the slope
along the search direction.
"""

import warnings
import numpy as np
from scipy.special import logit

# Line search parameters, the values used by Hager & Zhang (2005), Section 5.
DELTA, SIGMA, EPSILON = 0.1, 0.9, 1e-6
HISTORY = 10           # number of L-BFGS correction pairs kept
MAX_TRIALS = 20        # safety guard on evaluations per line search


def fit_theta(evaluate, target_incl, n, *, tol, max_iter, verbose):
    """Minimize f(theta) and return theta, centered to mean zero.

    evaluate(theta) returns (log Z, pi) for the distribution with parameters
    theta, with pi as a float64 array.  Returns the evaluated point with the
    smallest max |pi - target_incl|, which is the first point within tol when
    the fit converges.  Warns if the fit stops before reaching tol.

    Hager, W. W. and Zhang, H. (2005).  A new conjugate gradient method with
    guaranteed descent and an efficient line search.  SIAM Journal on
    Optimization 16(1), 170-192.
    """
    target = np.asarray(target_incl, float)
    N = len(target)
    if not np.all((target > 0) & (target < 1)):
        raise ValueError("target inclusion probabilities must lie strictly between 0 and 1")
    # pi always sums to n, so max |pi - target| >= |sum(target) - n| / N.
    if abs(target.sum() - n) > N * tol:
        raise ValueError(f"target inclusion probabilities sum to {float(target.sum())!r}, not n = {n}")

    theta = logit(target)
    best_err, best_theta = np.inf, theta - theta.mean()

    def value_and_grad(theta):
        nonlocal best_err, best_theta
        # f is unchanged by adding a constant to theta.  Center theta here, so
        # the point returned is exactly a point evaluated.
        theta = theta - theta.mean()
        log_Z, pi = evaluate(theta)
        g = pi - target
        err = float(np.max(np.abs(g)))
        if err < best_err:
            best_err, best_theta = err, theta.copy()
        if verbose:
            print(f"  max|pi-pi*| = {err:.3e}")
        return log_Z - target @ theta, g, err

    # Initial inverse Hessian.  If the coins were independent (Poisson
    # sampling) with probabilities target, the Hessian of f would be
    # diag(target (1 - target)), and theta = logit(target) would be optimal.
    h0 = 1 / (target * (1 - target))
    f, g, err = value_and_grad(theta)
    memory = []                                   # (s, y, 1 / s.y), oldest first
    reason = f"max_iter = {max_iter} reached"
    for _ in range(max_iter):
        if err <= tol:
            break
        d = _direction(g, h0, memory)
        if not g @ d < 0:                         # lost descent to rounding
            memory.clear()
            d = -h0 * g
        step = _line_search(value_and_grad, theta, f, g, d, tol)
        if step is None and memory:               # retry along -h0 g
            memory.clear()
            d = -h0 * g
            step = _line_search(value_and_grad, theta, f, g, d, tol)
        if step is None:
            reason = "the line search found no acceptable step"
            break
        alpha, f, g_new, err = step
        s, y = alpha * d, g_new - g
        # s.y > 0 by the curvature condition, unless the step met tol and
        # skipped the line search's tests.
        if s @ y > 0:
            memory.append((s, y, 1 / (s @ y)))
            del memory[:-HISTORY]
        theta, g = theta + s, g_new

    if best_err > tol:
        warnings.warn(f"fit stopped at max|pi - target_incl| = {best_err:.2e}, "
                      f"above tol = {tol:.0e}: {reason}", RuntimeWarning, stacklevel=3)
    return best_theta


def _direction(g, h0, memory):
    """L-BFGS two-loop recursion: -H g for the inverse-Hessian estimate H."""
    q = -g
    a = []
    for s, y, rho in reversed(memory):
        a.append(rho * (s @ q))
        q = q - a[-1] * y
    r = h0 * q
    for (s, y, rho), a_i in zip(memory, reversed(a)):
        r = r + (a_i - rho * (y @ r)) * s
    return r


def _line_search(value_and_grad, theta, f0, g0, d, tol):
    """Find a step length alpha along d that meets the Wolfe conditions or the
    approximate Wolfe conditions (Hager & Zhang 2005, Section 4).

    phi(alpha) = f(theta + alpha d) is convex, so its slope phi'(alpha) is
    nondecreasing, and a step is too short exactly when phi'(alpha) is below
    SIGMA phi'(0).  Returns (alpha, f, g, err) at the accepted point, or None
    if MAX_TRIALS evaluations find none.
    """
    dphi0 = g0 @ d
    eps_f = EPSILON * abs(f0)
    lo, dlo = 0.0, dphi0                          # longest step known to be too short
    hi, dhi = None, 0.0                           # shortest step known to be too long
    alpha = 1.0
    for _ in range(MAX_TRIALS):
        f, g, err = value_and_grad(theta + alpha * d)
        dphi = g @ d
        curvature = dphi >= SIGMA * dphi0
        wolfe = curvature and f <= f0 + DELTA * alpha * dphi0
        approx_wolfe = curvature and dphi <= (2 * DELTA - 1) * dphi0 and f <= f0 + eps_f
        if err <= tol or wolfe or approx_wolfe:
            return alpha, f, g, err
        # The next trial aims at the zero of phi', estimated by the secant
        # through the last two slopes.
        if not curvature:                         # too short
            prev, dprev, lo, dlo = lo, dlo, alpha, dphi
            if hi is None:                        # no upper end yet: grow by at most 10x
                alpha = 10 * lo
                if dprev < dlo:
                    alpha = min(alpha, _secant_zero(prev, dprev, lo, dlo))
                continue
        else:                                     # too long
            hi, dhi = alpha, dphi
        # Keep the trial in the middle 80% of the bracket, so it shrinks by at
        # least 10% each time; bisect if the slopes do not bracket a zero.
        assert hi is not None
        width = hi - lo
        if dlo < 0 < dhi:
            alpha = min(max(_secant_zero(lo, dlo, hi, dhi), lo + 0.1 * width), hi - 0.1 * width)
        else:
            alpha = lo + 0.5 * width
    return None


def _secant_zero(a, da, b, db):
    """Zero of the line through (a, da) and (b, db)."""
    return b - db * (b - a) / (db - da)
