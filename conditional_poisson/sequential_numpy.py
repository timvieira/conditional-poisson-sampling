"""
conditional_poisson_sequential_numpy.py
=======================================

Conditional Poisson distribution using O(Nn) sequential DP.

    P(S) ∝ prod_{i in S} w_i,   |S| = n

This implementation uses the weighted Pascal recurrence for all computations:
normalizing constant, inclusion probabilities, and sampling.  It is simpler
and faster than the product-tree implementation for small n, but scales as
O(Nn) rather than O(N log^2 n).

The sampling algorithm builds a table of sequential conditional probabilities
q[i, k] = P(include item i | k items still needed from items i..N-1), then
draws one sample by scanning items and flipping biased coins.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Union


class ConditionalPoissonSequentialNumPy:
    """Conditional Poisson distribution via O(Nn) sequential DP.

    Construction
    ------------
    ConditionalPoissonSequentialNumPy(n, theta)           from log-weights
    ConditionalPoissonSequentialNumPy.from_weights(n, w)  from weights

    Properties
    ----------
    log_normalizer   log Z(w, n)
    incl_prob        inclusion probability vector pi

    Methods
    -------
    sample(size)     draw samples via sequential scan, O(N) per sample
    """

    def __init__(self, n: int, theta: np.ndarray):
        theta = np.asarray(theta, float)
        if theta.ndim != 1:
            raise ValueError(f"theta must be 1-D, got shape {theta.shape}")
        N = len(theta)
        if n < 0 or n > N:
            raise ValueError(f"n={n} out of range [0, {N}]")
        self.n = n
        self.N = N
        self._theta = theta.copy()
        self._cache: dict = {}

    @classmethod
    def from_weights(cls, n: int, w) -> ConditionalPoissonSequentialNumPy:
        w = np.asarray(w, float)
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        if np.any(w == 0) or np.any(np.isinf(w)):
            raise ValueError("sequential DP requires finite positive weights")
        return cls(n, np.log(w))

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def theta(self) -> np.ndarray: return self._theta

    # ── DP tables ────────────────────────────────────────────────────────────

    def _get_dp(self):
        """Build and cache forward/backward DP tables."""
        if "dp" not in self._cache:
            self._cache["dp"] = self._compute_table()
        return self._cache["dp"]

    def _compute_table(self):
        w = np.exp(self._theta)
        N, n = self.N, self.n
        log_gm = np.mean(self._theta)
        q = w / np.exp(log_gm)

        # Forward: F[i, k] = e_k(q[0:i]) (scaled)
        F, Fls = self._forward_dp(q)

        # Backward: B[i, k] = e_k(q[i:N]) (scaled)
        B = np.zeros((N + 1, n + 1))
        Bls = np.zeros(N + 1)
        B[N, 0] = 1.0
        for i in range(N - 1, -1, -1):
            row = np.empty(n + 1)
            row[0] = 1.0
            if n >= 1:
                row[1] = B[i+1, 1] + q[i] * np.exp(-Bls[i+1])
            if n >= 2:
                row[2:] = B[i+1, 2:] + q[i] * B[i+1, 1:n]
            mx = np.max(row[1:]) if n >= 1 else 1.0
            if mx > 0:
                row[1:] /= mx
                Bls[i] = Bls[i+1] + np.log(mx)
            else:
                Bls[i] = Bls[i+1]
            B[i] = row
        return (q, F, Fls, B, Bls, log_gm)


    # ── Log normalizer ───────────────────────────────────────────────────────

    @property
    def log_normalizer(self) -> float:
        """log Z(w, n).  O(Nn)."""
        if "log_Z" not in self._cache:
            _, F, Fls, _, _, log_gm = self._get_dp()
            self._cache["log_Z"] = (
                np.log(abs(F[self.N, self.n])) + Fls[self.N]
                + self.n * log_gm
            )
        return self._cache["log_Z"]

    # ── Inclusion probabilities ──────────────────────────────────────────────

    @property
    def incl_prob(self) -> np.ndarray:
        """Inclusion probability vector pi.  O(Nn)."""
        if "pi" not in self._cache:
            q, F, Fls, B, Bls, _ = self._get_dp()
            N, n = self.N, self.n
            log_Z = np.log(abs(F[N, n])) + Fls[N]

            pi = np.zeros(N)
            for i in range(N):
                max_j = min(n, i + 1)
                total_log = -np.inf
                for j in range(max_j):
                    k = n - 1 - j
                    if k < 0 or k > N - i - 1:
                        continue
                    f_val, b_val = F[i, j], B[i+1, k]
                    if f_val == 0 or b_val == 0:
                        continue
                    log_f = np.log(abs(f_val)) + (Fls[i] if j >= 1 else 0.0)
                    log_b = np.log(abs(b_val)) + (Bls[i+1] if k >= 1 else 0.0)
                    log_term = log_f + log_b
                    if total_log == -np.inf:
                        total_log = log_term
                    else:
                        mx = max(total_log, log_term)
                        total_log = mx + np.log(
                            np.exp(total_log - mx) + np.exp(log_term - mx)
                        )
                pi[i] = q[i] * np.exp(total_log - log_Z)
            self._cache["pi"] = pi
        return self._cache["pi"].copy()

    # ── Sampling ─────────────────────────────────────────────────────────────

    def _get_seq_q(self):
        """Build and cache the sequential conditional probability table q.

        q[i, k] = P(include item i | k items still needed from i..N-1).
        Uses the backward ESP recurrence.  O(Nn).
        """
        if "seq_q" not in self._cache:
            w = np.exp(self._theta)
            N, n = self.N, self.n
            # Backward ESP recurrence: expa[i, k] = e_k(w[i:N])
            expa = np.zeros((N, n))
            for i in range(N):
                expa[i, 0] = np.sum(w[i:N])
            for i in range(N - n, N):
                expa[i, N - i - 1] = np.prod(w[i:N])
            for i in range(N - 3, -1, -1):
                for k in range(1, min(N - i - 1, n)):
                    expa[i, k] = w[i] * expa[i + 1, k - 1] + expa[i + 1, k]
            # Sequential conditional probabilities
            q = np.zeros((N, n))
            for i in range(N - 1, -1, -1):
                q[i, 0] = w[i] / expa[i, 0]
            for i in range(N - n, N):
                q[i, N - i - 1] = 1.0
            for i in range(N - 3, -1, -1):
                for k in range(1, min(N - i - 1, n)):
                    q[i, k] = w[i] * expa[i + 1, k - 1] / expa[i, k]
            self._cache["seq_q"] = q
        return self._cache["seq_q"]

    def sample(
        self,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> np.ndarray:
        """
        Draw one sample via sequential scan.

        Parameters
        ----------
        rng  : seed or np.random.Generator

        Returns
        -------
        (n,) int array of sorted item indices.

        Complexity: O(Nn) to build q table [cached] + O(N).
        """
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        q = self._get_seq_q()
        N, n = self.N, self.n
        selected = []
        k = n
        for i in range(N):
            if k == 0:
                break
            if rng.random() < q[i, k - 1]:
                selected.append(i)
                k -= 1
        return np.array(selected, dtype=np.int32)

    # ── Log probability ───────────────────────────────────────────────────────

    def log_prob(self, S) -> float:
        """log P(S) = sum_{i in S} theta_i - log Z."""
        S = np.asarray(S)
        return float(self._theta[S].sum() - self.log_normalizer)

    # ── Fitting ──────────────────────────────────────────────────────────────

    @classmethod
    def fit(cls, target_incl, n, *, tol=1e-10, max_iter=200, verbose=False):
        """Fit to target inclusion probabilities via L-BFGS."""
        from scipy.optimize import minimize
        from scipy.special import logit

        target_incl = np.asarray(target_incl, float)
        obj = cls(n, logit(target_incl))

        def neg_ll_and_grad(theta):
            obj._theta = theta
            obj._cache.clear()
            pi = obj.incl_prob
            loss = obj.log_normalizer - float(np.dot(target_incl, theta))
            grad = pi - target_incl
            if verbose:
                err = float(np.max(np.abs(grad)))
                print(f"  max|pi-pi*| = {err:.3e}")
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

    def _forward_dp(self, q):
        """Weighted Pascal DP table with row-wise scaling.

        True value: Z(q[0:i] choose k) = W[i, k] * exp(ls[i])  for k >= 1.
        W[i, 0] = 1 always (unscaled).

        Recurrence:  Z[i, k] = Z[i-1, k] + q[i-1] * Z[i-1, k-1]
        """
        N, n = self.N, self.n
        W = np.zeros((N + 1, n + 1))
        ls = np.zeros(N + 1)
        W[0, 0] = 1.0
        for i in range(1, N + 1):
            row = np.empty(n + 1)
            row[0] = 1.0
            if n >= 1:
                row[1] = W[i-1, 1] + q[i-1] * np.exp(-ls[i-1])
            if n >= 2:
                row[2:] = W[i-1, 2:] + q[i-1] * W[i-1, 1:n]
            mx = np.max(row[1:]) if n >= 1 else 1.0
            if mx > 0:
                row[1:] /= mx
                ls[i] = ls[i-1] + np.log(mx)
            else:
                ls[i] = ls[i-1]
            W[i] = row
        return W, ls
