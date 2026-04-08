"""
Sequential O(Nn) implementation of the conditional Poisson distribution.

    P(S) ∝ prod_{i in S} w_i,   |S| = n

Uses the elementary symmetric polynomial (ESP) recurrence for all
computations: normalizing constant, inclusion probabilities (via
reverse-mode AD on the DP), and sampling (right-to-left scan on the
forward table).
"""

from __future__ import annotations
from functools import cached_property
import numpy as np


class ConditionalPoissonSequentialNumPy:
    """Conditional Poisson distribution via O(Nn) sequential DP.

    Constructors: __init__(n, theta), from_weights(n, w), fit(target_incl, n)
    Properties:   incl_prob, log_normalizer, n, N, theta
    Methods:      log_prob(S), sample(), clear()
    """

    def __init__(self, n: int, theta: np.ndarray):
        theta = np.asarray(theta, float)
        assert theta.ndim == 1
        assert 0 <= n <= len(theta)
        self.n = n
        self.N = len(theta)
        self.theta = theta.copy()

    @classmethod
    def from_weights(cls, n: int, w) -> ConditionalPoissonSequentialNumPy:
        w = np.asarray(w, float)
        if np.any(w <= 0) or not np.all(np.isfinite(w)):
            raise ValueError("all weights must be finite and positive")
        return cls(n, np.log(w))

    def clear(self):
        """Flush all cached computations."""
        for attr in ('_E', 'log_normalizer', 'incl_prob', '_sample_data'):
            self.__dict__.pop(attr, None)

    @cached_property
    def _E(self):
        """Forward ESP table E[k, n] = e_k(w[0:n]).  O(Nn)."""
        w = np.exp(self.theta)
        N, K = self.N, self.n
        E = np.zeros((K + 1, N + 1))
        E[0, :] = 1.0
        for n in range(N):
            E[1:, n+1] = E[1:, n] + w[n] * E[:K, n]
        return E, w

    @cached_property
    def log_normalizer(self) -> float:
        """log Z(w, n).  O(Nn).  Does not trigger backward pass."""
        E, _ = self._E
        return float(np.log(E[self.n, self.N]))

    @cached_property
    def incl_prob(self) -> np.ndarray:
        """Inclusion probabilities via reverse-mode AD on the ESP DP.  O(Nn)."""
        E, w = self._E
        N, K = self.N, self.n
        Z = E[K, N]
        dE = np.zeros((K + 1, N + 1))
        dw = np.zeros(N)
        dE[K, N] = 1.0
        for n in reversed(range(N)):
            for k in range(K, 0, -1):
                dE[k, n] += dE[k, n+1]
                dw[n] += dE[k, n+1] * E[k-1, n]
                dE[k-1, n] += dE[k, n+1] * w[n]
        return w * dw / Z

    @cached_property
    def _sample_data(self):
        """Convert ESP table to plain lists for fast sampling loop."""
        E, w = self._E
        return E.tolist(), w.tolist()

    def sample(self) -> np.ndarray:
        """Draw one sample by scanning items right-to-left.

        P(include i | k remaining) = w[i] * E[k-1, i] / E[k, i+1]

        Complexity: O(Nn) to build table [cached] + O(N).
        """
        import random
        E, w = self._sample_data
        N, K = self.N, self.n
        selected = []
        k = K
        for i in reversed(range(N)):
            if k == 0:
                break
            if random.random() * E[k][i+1] <= w[i] * E[k-1][i]:
                selected.append(i)
                k -= 1
        selected.reverse()
        return np.array(selected, dtype=np.int32)

    def log_prob(self, S) -> float:
        """log P(S) = sum_{i in S} theta_i - log Z."""
        S = np.asarray(S)
        return float(self.theta[S].sum() - self.log_normalizer)

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

    def __repr__(self):
        return f"ConditionalPoissonSequentialNumPy(N={self.N}, n={self.n})"
