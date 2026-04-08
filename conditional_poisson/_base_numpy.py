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
        """Fit to target inclusion probabilities via L-BFGS."""
        from scipy.optimize import minimize
        from scipy.special import logit

        target_incl = np.asarray(target_incl, float)
        obj = cls(n, logit(target_incl))

        def neg_ll_and_grad(theta):
            obj.theta = theta
            obj.clear()
            pi = obj.incl_prob
            loss = obj.log_normalizer - float(target_incl @ theta)
            grad = pi - target_incl
            if verbose:
                print(f"  max|pi-pi*| = {np.max(np.abs(grad)):.3e}")
            return loss, grad

        result = minimize(
            neg_ll_and_grad, logit(target_incl),
            method='L-BFGS-B', jac=True,
            options={'maxiter': max_iter, 'gtol': tol, 'ftol': 0},
        )
        theta = result.x
        theta -= theta.mean()
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
