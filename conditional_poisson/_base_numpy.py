"""Shared base class for NumPy conditional Poisson implementations."""

from functools import cached_property
import numpy as np


class ConditionalPoissonNumPyBase:
    """Base class for NumPy implementations.

    Subclasses must implement:
        _forward (cached_property) -> circuit-specific forward data
        incl_prob (cached_property) -> inclusion probability vector
        sample(self) -> array of indices
        _sample_data (cached_property) -> data for sampling loop

    Subclass must ensure that log_normalizer can be computed from _forward.
    """

    def __init__(self, n: int, theta: np.ndarray):
        theta = np.asarray(theta, float)
        assert theta.ndim == 1
        assert np.all(np.isfinite(theta))
        assert 0 <= n <= len(theta)
        self.n = int(n)
        self.N = len(theta)
        self.theta = theta.copy()

    @classmethod
    def from_weights(cls, n: int, w):
        w = np.asarray(w, float)
        if np.any(w <= 0) or not np.all(np.isfinite(w)):
            raise ValueError("all weights must be finite and positive")
        return cls(n, np.log(w))

    @classmethod
    def fit(cls, target_incl, n, *, tol=1e-10, max_iter=200, verbose=False):
        """Fit to target inclusion probabilities via L-BFGS.

        Returns the evaluated point with the smallest max |pi - target_incl|,
        which is not always the point where L-BFGS stops.  Near the optimum,
        the decrease in the loss falls below its rounding error, so the line
        search can reject a better point because of rounding alone.  The
        gradient pi - target_incl stays accurate, so it decides.
        """
        from scipy.optimize import minimize
        from scipy.special import logit

        target_incl = np.asarray(target_incl, float)
        cp = cls(n, logit(target_incl))
        best_err, best_theta = np.inf, cp.theta

        def neg_ll_and_grad(theta):
            nonlocal best_err, best_theta
            cp.theta = theta
            cp.clear()
            pi = cp.incl_prob
            loss = cp.log_normalizer - float(target_incl @ theta)
            grad = pi - target_incl
            err = float(np.max(np.abs(grad)))
            if err < best_err:
                best_err, best_theta = err, theta.copy()
            if verbose:
                print(f"  max|pi-pi*| = {err:.3e}")
            return loss, grad

        minimize(
            neg_ll_and_grad, logit(target_incl),
            method='L-BFGS-B', jac=True,
            options={'maxiter': max_iter, 'gtol': tol, 'ftol': 0},
        )
        theta = best_theta - best_theta.mean()
        return cls(n, theta)

    def clear(self):
        """Flush all cached computations."""
        for attr in ('_forward', 'log_normalizer', 'incl_prob', '_sample_data'):
            self.__dict__.pop(attr, None)

    def log_prob(self, S) -> float:
        """log P(S) = sum_{i in S} theta_i - log Z."""
        S = np.asarray(S)
        return float(self.theta[S].sum() - self.log_normalizer)

    def sample(self):
        raise NotImplementedError
