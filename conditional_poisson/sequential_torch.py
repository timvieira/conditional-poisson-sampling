"""
conditional_poisson_sequential_torch.py
=======================================

Conditional Poisson distribution using O(Nn) sequential DP (PyTorch).

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
import torch
import math
from typing import Optional


class ConditionalPoissonSequentialTorch:
    """Conditional Poisson distribution via O(Nn) sequential DP (PyTorch).

    Construction
    ------------
    ConditionalPoissonSequentialTorch(n, theta)           from log-weights
    ConditionalPoissonSequentialTorch.from_weights(n, w)  from weights

    Properties
    ----------
    log_normalizer   log Z(w, n)
    incl_prob        inclusion probability vector pi

    Methods
    -------
    sample(size)     draw samples via sequential scan, O(N) per sample
    """

    def __init__(self, n: int, theta):
        theta = torch.as_tensor(theta, dtype=torch.float64)
        if theta.ndim != 1:
            raise ValueError(f"theta must be 1-D, got shape {theta.shape}")
        N = len(theta)
        if n < 0 or n > N:
            raise ValueError(f"n={n} out of range [0, {N}]")
        self._n = n
        self._N = N
        self._theta = theta.detach().clone()
        self._cache: dict = {}

    @classmethod
    def from_weights(cls, n: int, w) -> ConditionalPoissonSequentialTorch:
        w = torch.as_tensor(w, dtype=torch.float64)
        if torch.any(w < 0):
            raise ValueError("weights must be non-negative")
        if torch.any(w == 0) or torch.any(torch.isinf(w)):
            raise ValueError("sequential DP requires finite positive weights")
        return cls(n, torch.log(w))

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def n(self) -> int:   return self._n
    @property
    def N(self) -> int:   return self._N
    @property
    def theta(self) -> torch.Tensor: return self._theta.clone()
    @property
    def w(self) -> torch.Tensor: return torch.exp(self._theta)

    # ── DP tables ────────────────────────────────────────────────────────────

    def _get_dp(self):
        """Build and cache forward/backward DP tables."""
        if "dp" not in self._cache:
            w = self.w
            N, n = self._N, self._n
            log_gm = self._theta.mean().item()
            q = (w / math.exp(log_gm)).tolist()

            # Forward: F[i, k] = e_k(q[0:i]) (scaled)
            F, Fls = _build_dp_table(q, n)

            # Backward: B[i, k] = e_k(q[i:N]) (scaled)
            B = [[0.0] * (n + 1) for _ in range(N + 1)]
            Bls = [0.0] * (N + 1)
            B[N][0] = 1.0
            for i in range(N - 1, -1, -1):
                row = [0.0] * (n + 1)
                row[0] = 1.0
                if n >= 1:
                    row[1] = B[i+1][1] + q[i] * math.exp(-Bls[i+1])
                for k in range(2, n + 1):
                    row[k] = B[i+1][k] + q[i] * B[i+1][k-1]
                mx = max(abs(row[k]) for k in range(1, n + 1)) if n >= 1 else 1.0
                if mx > 0:
                    for k in range(1, n + 1):
                        row[k] /= mx
                    Bls[i] = Bls[i+1] + math.log(mx)
                else:
                    Bls[i] = Bls[i+1]
                B[i] = row

            self._cache["dp"] = (q, F, Fls, B, Bls, log_gm)
        return self._cache["dp"]

    # ── Log normalizer ───────────────────────────────────────────────────────

    @property
    def log_normalizer(self) -> float:
        """log Z(w, n).  O(Nn)."""
        if "log_Z" not in self._cache:
            q, F, Fls, B, Bls, log_gm = self._get_dp()
            self._cache["log_Z"] = (
                math.log(abs(F[self._N][self._n])) + Fls[self._N]
                + self._n * log_gm
            )
        return self._cache["log_Z"]

    # ── Inclusion probabilities ──────────────────────────────────────────────

    @property
    def incl_prob(self) -> torch.Tensor:
        """Inclusion probability vector pi.  O(Nn)."""
        if "pi" not in self._cache:
            q, F, Fls, B, Bls, log_gm = self._get_dp()
            N, n = self._N, self._n
            log_Z = math.log(abs(F[N][n])) + Fls[N]

            pi = [0.0] * N
            for i in range(N):
                max_j = min(n, i + 1)
                total_log = -math.inf
                for j in range(max_j):
                    k = n - 1 - j
                    if k < 0 or k > N - i - 1:
                        continue
                    f_val, b_val = F[i][j], B[i+1][k]
                    if f_val == 0 or b_val == 0:
                        continue
                    log_f = math.log(abs(f_val)) + (Fls[i] if j >= 1 else 0.0)
                    log_b = math.log(abs(b_val)) + (Bls[i+1] if k >= 1 else 0.0)
                    log_term = log_f + log_b
                    if total_log == -math.inf:
                        total_log = log_term
                    else:
                        mx = max(total_log, log_term)
                        total_log = mx + math.log(
                            math.exp(total_log - mx) + math.exp(log_term - mx)
                        )
                pi[i] = q[i] * math.exp(total_log - log_Z)
            self._cache["pi"] = torch.tensor(pi, dtype=torch.float64)
        return self._cache["pi"].clone()

    # ── Sampling ─────────────────────────────────────────────────────────────

    def _get_seq_q(self):
        """Build and cache the sequential conditional probability table q.

        q[i, k] = P(include item i | k items still needed from i..N-1).
        Uses the backward ESP recurrence.  O(Nn).
        """
        if "seq_q" not in self._cache:
            w = self.w.tolist()
            N, n = self._N, self._n
            # Backward ESP recurrence: expa[i, k] = e_k(w[i:N])
            expa = [[0.0] * n for _ in range(N)]
            for i in range(N):
                expa[i][0] = sum(w[i:N])
            for i in range(N - n, N):
                p = 1.0
                for j in range(i, N):
                    p *= w[j]
                expa[i][N - i - 1] = p
            for i in range(N - 3, -1, -1):
                for k in range(1, min(N - i - 1, n)):
                    expa[i][k] = w[i] * expa[i + 1][k - 1] + expa[i + 1][k]
            # Sequential conditional probabilities
            q = [[0.0] * n for _ in range(N)]
            for i in range(N - 1, -1, -1):
                q[i][0] = w[i] / expa[i][0]
            for i in range(N - n, N):
                q[i][N - i - 1] = 1.0
            for i in range(N - 3, -1, -1):
                for k in range(1, min(N - i - 1, n)):
                    q[i][k] = w[i] * expa[i + 1][k - 1] / expa[i][k]
            self._cache["seq_q"] = q
        return self._cache["seq_q"]

    def sample(self, size: int = 1, rng=None) -> torch.Tensor:
        """
        Draw independent samples via sequential scan.

        Parameters
        ----------
        size : number of subsets to draw
        rng  : int seed, random.Random, or np.random.Generator

        Returns
        -------
        (size, n) long tensor of sorted indices.

        Complexity: O(Nn) to build q table [cached] + O(size * N).
        """
        import random as _random
        import numpy as np

        q = self._get_seq_q()
        N, n = self._N, self._n

        # Accept any object with .random() method
        if isinstance(rng, np.random.Generator) or isinstance(rng, _random.Random):
            pass
        else:
            rng = _random.Random(rng)

        samples = torch.empty(size, n, dtype=torch.long)
        for m in range(size):
            k = n
            cursor = 0
            for i in range(N):
                if k == 0:
                    break
                if rng.random() < q[i][k - 1]:
                    samples[m, cursor] = i
                    cursor += 1
                    k -= 1
        return samples

    def __repr__(self):
        return f"ConditionalPoissonSequentialTorch(N={self._N}, n={self._n})"


# ── Helper: weighted Pascal DP table ─────────────────────────────────────────

def _build_dp_table(q, n):
    """
    Weighted Pascal DP table with row-wise scaling (pure Python lists).

    True value: Z(q[0:i] choose k) = W[i][k] * exp(ls[i])  for k >= 1
    W[i][0] = 1 always (unscaled).

    Recurrence:  Z[i, k] = Z[i-1, k] + q[i-1] * Z[i-1, k-1]
    """
    N = len(q)
    W = [[0.0] * (n + 1) for _ in range(N + 1)]
    ls = [0.0] * (N + 1)
    W[0][0] = 1.0
    for i in range(1, N + 1):
        row = [0.0] * (n + 1)
        row[0] = 1.0
        if n >= 1:
            row[1] = W[i-1][1] + q[i-1] * math.exp(-ls[i-1])
        for k in range(2, n + 1):
            row[k] = W[i-1][k] + q[i-1] * W[i-1][k-1]
        mx = max(abs(row[k]) for k in range(1, n + 1)) if n >= 1 else 1.0
        if mx > 0:
            for k in range(1, n + 1):
                row[k] /= mx
            ls[i] = ls[i-1] + math.log(mx)
        else:
            ls[i] = ls[i-1]
        W[i] = row
    return W, ls
