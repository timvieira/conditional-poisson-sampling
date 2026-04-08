"""
Sequential O(Nn) implementation of the conditional Poisson distribution.

    P(S) ∝ prod_{i in S} w_i,   |S| = n

Uses the elementary symmetric polynomial (ESP) recurrence for all
computations: normalizing constant, inclusion probabilities (via
reverse-mode AD on the DP), and sampling (right-to-left scan on the
forward table).
"""

from __future__ import annotations
import numpy as np


class ConditionalPoissonSequentialNumPy:
    """Conditional Poisson distribution via O(Nn) sequential DP.

    Constructors: __init__(n, theta), from_weights(n, w), fit(target_incl, n)
    Properties:   incl_prob, log_normalizer, n, N, theta
    Methods:      log_prob(S), sample()
    """

    def __init__(self, n: int, theta: np.ndarray):
        theta = np.asarray(theta, float)
        assert theta.ndim == 1
        assert 0 <= n <= len(theta)
        self.n = n
        self.N = len(theta)
        self.theta = theta.copy()
        self._cache: dict = {}

    @classmethod
    def from_weights(cls, n: int, w) -> ConditionalPoissonSequentialNumPy:
        w = np.asarray(w, float)
        if np.any(w <= 0) or not np.all(np.isfinite(w)):
            raise ValueError("all weights must be finite and positive")
        return cls(n, np.log(w))

    def _get_E(self):
        """Build and cache the ESP table E[k, n+1].

        E[k, n] = e_k(w[0:n]), the k-th elementary symmetric polynomial
        of the first n weights.  With row-wise rescaling for stability:

            E[k, n+1] = E[k, n] + w[n] * E[k-1, n]

        Scale factors sf[n] are stored per column (per item added).
        True value: e_k(w[0:n]) = E[k, n] * prod(sf[1:n+1]).
        """
        if "E" not in self._cache:
            w = np.exp(self.theta)
            N, K = self.N, self.n
            E = np.zeros((K + 1, N + 1))
            E[0, :] = 1.0
            for n in range(N):
                E[1:, n+1] = E[1:, n] + w[n] * E[:K, n]
            self._cache["E"] = (E, w)
        return self._cache["E"]

    @property
    def log_normalizer(self) -> float:
        """log Z(w, n).  O(Nn), cached."""
        if "log_Z" not in self._cache:
            E, _ = self._get_E()
            self._cache["log_Z"] = float(np.log(E[self.n, self.N]))
        return self._cache["log_Z"]

    @property
    def incl_prob(self) -> np.ndarray:
        """Inclusion probabilities via reverse-mode AD on the ESP DP.

        Forward: E[k, n+1] = E[k, n] + w[n] * E[k-1, n]
        Backward: dZ/dw[n] = sum_k dE[k, n+1] * E[k-1, n]
        Inclusion: pi[n] = w[n] * dZ/dw[n] / Z
        """
        if "pi" not in self._cache:
            E, w = self._get_E()
            N, K = self.N, self.n
            Z = E[K, N]

            # Reverse-mode AD on the ESP recurrence
            dE = np.zeros((K + 1, N + 1))
            dw = np.zeros(N)
            dE[K, N] = 1.0
            for n in reversed(range(N)):
                for k in range(K, 0, -1):
                    dE[k, n] += dE[k, n+1]
                    dw[n] += dE[k, n+1] * E[k-1, n]
                    dE[k-1, n] += dE[k, n+1] * w[n]

            self._cache["pi"] = w * dw / Z
        return self._cache["pi"].copy()

    def sample(self) -> np.ndarray:
        """Draw one sample by scanning items right-to-left.

        P(include i | k remaining) = w[i] * E[k-1, i] / E[k, i+1]

        Complexity: O(Nn) to build table [cached] + O(N).
        """
        if "sample_data" not in self._cache:
            E, w = self._get_E()
            self._cache["sample_data"] = (E.tolist(), w.tolist())
        E, w = self._cache["sample_data"]
        N, K = self.N, self.n
        selected = []
        k = K
        for i in reversed(range(N)):
            if k == 0:
                break
            if np.random.random() * E[k][i+1] <= w[i] * E[k-1][i]:
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
            obj._cache.clear()
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
