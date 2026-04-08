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


class ConditionalPoissonSequentialNumPy:
    """Conditional Poisson distribution via O(Nn) sequential DP.

    Constructors: __init__(n, theta), from_weights(n, w), fit(target_incl, n)
    Properties:   incl_prob, log_normalizer, theta, n, N
    Methods:      log_prob(S), sample(rng)
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
        self.theta = theta.copy()
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

    # ── DP tables ────────────────────────────────────────────────────────────

    def _get_dp(self):
        """Build and cache forward/backward DP tables."""
        if "dp" not in self._cache:
            self._cache["dp"] = self._compute_table()
        return self._cache["dp"]

    def _compute_table(self):
        w = np.exp(self.theta)
        N, n = self.N, self.n
        log_gm = np.mean(self.theta)
        q = w / np.exp(log_gm)

        F, Fls = self._forward_dp(q)

        # Backward: B[i, k] = e_k(q[i:N]) with uniform row-wise scaling.
        # scale_factor[i] is the rescaling applied at row i; the true value
        # is B[i,k] * prod(scale_factor[i:N]).
        B = np.zeros((N + 1, n + 1))
        scale_factor = np.ones(N + 1)
        B[N, 0] = 1.0
        for i in range(N - 1, -1, -1):
            B[i] = B[i+1].copy()
            B[i, 1:] += q[i] * B[i+1, :n]
            mx = np.max(B[i])
            if mx > 0:
                B[i] /= mx
                scale_factor[i] = mx

        # Cumulative log scale (only needed for incl_prob, not sampling)
        Bls = np.cumsum(np.log(scale_factor[::-1]))[::-1].copy()

        return (q, F, Fls, B, Bls, scale_factor, log_gm)

    def _forward_dp(self, q):
        """Weighted Pascal DP table with uniform row-wise scaling.

        true value: e_k(q[0:i]) = W[i, k] * exp(ls[i])

        Recurrence: W[i, k] = W[i-1, k] + q[i-1] * W[i-1, k-1]
        """
        N, n = self.N, self.n
        W = np.zeros((N + 1, n + 1))
        ls = np.zeros(N + 1)
        W[0, 0] = 1.0
        for i in range(1, N + 1):
            W[i] = W[i-1]
            W[i, 1:] += q[i-1] * W[i-1, :n]
            mx = np.max(W[i])
            if mx > 0:
                W[i] /= mx
                ls[i] = ls[i-1] + np.log(mx)
            else:
                ls[i] = ls[i-1]
        return W, ls

    # ── Log normalizer ───────────────────────────────────────────────────────

    @property
    def log_normalizer(self) -> float:
        """log Z(w, n).  O(Nn)."""
        if "log_Z" not in self._cache:
            _, F, Fls, _, _, _, log_gm = self._get_dp()
            self._cache["log_Z"] = (
                np.log(F[self.N, self.n]) + Fls[self.N]
                + self.n * log_gm
            )
        return self._cache["log_Z"]

    # ── Inclusion probabilities ──────────────────────────────────────────────

    @property
    def incl_prob(self) -> np.ndarray:
        """Inclusion probability vector pi.  O(Nn)."""
        if "pi" not in self._cache:
            q, F, Fls, B, Bls, _, _ = self._get_dp()
            N, n = self.N, self.n
            log_Z = np.log(F[N, n]) + Fls[N]

            # TODO: this looks does not look efficient -- remember that the
            # algorithm to compute the incl_prob should match backpropagation for \nabla_{\theta} log Z

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
                    log_f = np.log(f_val) + Fls[i]
                    log_b = np.log(b_val) + Bls[i+1]
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

    def sample(self) -> np.ndarray:
        """
        Draw one sample via sequential scan.

        P(include i | k remaining) = q[i] * B[i+1,k-1] / (B[i,k] * scale_factor[i])

        Complexity: O(Nn) to build tables [cached] + O(N).
        """
        if "sample_data" not in self._cache:
            q, _, _, B, _, scale_factor, _ = self._get_dp()
            self._cache["sample_data"] = (q.tolist(), B.tolist(), scale_factor.tolist())
        q, B, sf = self._cache["sample_data"]
        N, n = self.N, self.n
        selected = []
        k = n
        for i in range(N):
            if k == 0:
                break
            prob = q[i] * B[i+1][k-1] / (B[i][k] * sf[i])
            if np.random.random() < prob:
                selected.append(i)
                k -= 1
        return np.array(selected, dtype=np.int32)

    # ── Log probability ───────────────────────────────────────────────────────

    def log_prob(self, S) -> float:
        """log P(S) = sum_{i in S} theta_i - log Z."""
        S = np.asarray(S)
        return float(self.theta[S].sum() - self.log_normalizer)

    # ── Fitting ──────────────────────────────────────────────────────────────

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
