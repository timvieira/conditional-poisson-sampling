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
        self.n = n
        self.N = N
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
    def theta(self) -> torch.Tensor: return self._theta

    # ── Log normalizer ───────────────────────────────────────────────────────

    def _log_Z(self):
        """Compute log Z as a differentiable scalar tensor.

        Uses the weighted Pascal recurrence:
            F[i, k] = F[i-1, k] + w[i-1] * F[i-1, k-1]
        with row-wise renormalization for numerical stability.
        log Z = log|F[N, n]| + cumulative_log_scale + n * log(geometric_mean).
        """
        theta = self._theta
        N, n = self.N, self.n
        w = torch.exp(theta)

        # Geometric-mean normalization for stability
        log_gm = theta.mean()
        q = w / torch.exp(log_gm)

        # Forward DP: F[k] = e_k(q[0:i]) after processing i items.
        # Row-wise scaling: true value at k>=1 is F[k] * exp(log_scale).
        # F[0] = 1 always (unscaled), so the k=1 update must compensate:
        #   F_new[1] = F[1] + q[i] * exp(-log_scale)
        #   F_new[k] = F[k] + q[i] * F[k-1]   for k >= 2
        F = torch.zeros(n + 1, dtype=theta.dtype, device=theta.device)
        F[0] = 1.0
        log_scale = torch.zeros(1, dtype=theta.dtype, device=theta.device)

        for i in range(N):
            parts = [F[:1]]  # F[0] = 1 unchanged
            # k=1: compensate for unscaled F[0]
            parts.append(F[1:2] + q[i] * torch.exp(-log_scale))
            if n >= 2:
                # k=2..n: both F[k] and F[k-1] share the same scale
                parts.append(F[2:] + q[i] * F[1:n])
            new_tail = torch.cat(parts[1:])
            mx = new_tail.abs().max()
            if mx > 0:
                new_tail = new_tail / mx
                log_scale = log_scale + torch.log(mx)
            F = torch.cat([F[:1], new_tail])

        return torch.log(F[n]) + log_scale.squeeze() + n * log_gm

    @property
    def log_normalizer(self) -> float:
        """log Z(w, n).  O(Nn)."""
        return self._log_Z().item()

    # ── Inclusion probabilities ──────────────────────────────────────────────

    @property
    def incl_prob(self) -> torch.Tensor:
        """Inclusion probabilities pi_i = d(log Z)/d(theta_i) via autograd."""
        saved = self._theta
        self._theta = saved.detach().requires_grad_(True)
        log_Z = self._log_Z()
        pi = torch.autograd.grad(log_Z, self._theta)[0].detach()
        self._theta = saved
        return pi.clone()

    # ── Sampling ─────────────────────────────────────────────────────────────

    def _get_seq_q(self):
        """Build and cache the sequential conditional probability table q.

        q[i, k] = P(include item i | k items still needed from i..N-1).
        Uses the backward ESP recurrence.  O(Nn).
        """
        if "seq_q" not in self._cache:
            w = torch.exp(self._theta).tolist()
            N, n = self.N, self.n
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

    def sample(self, rng=None) -> torch.Tensor:
        """
        Draw one sample via sequential scan.

        Parameters
        ----------
        rng  : int seed, random.Random, or np.random.Generator

        Returns
        -------
        (n,) long tensor of sorted indices.

        Complexity: O(Nn) to build q table [cached] + O(N).
        """
        import random as _random
        import numpy as np

        q = self._get_seq_q()
        N, n = self.N, self.n

        if isinstance(rng, np.random.Generator) or isinstance(rng, _random.Random):
            pass
        else:
            rng = _random.Random(rng)

        selected = []
        k = n
        for i in range(N):
            if k == 0:
                break
            if rng.random() < q[i][k - 1]:
                selected.append(i)
                k -= 1
        return torch.tensor(selected, dtype=torch.long)

    # ── Log probability ───────────────────────────────────────────────────────

    def log_prob(self, S) -> float:
        """log P(S) = sum_{i in S} theta_i - log Z."""
        S = torch.as_tensor(S, dtype=torch.long)
        return float(self._theta[S].sum() - self.log_normalizer)

    # ── Fitting ──────────────────────────────────────────────────────────────

    @classmethod
    def fit(cls, target_incl, n, *, tol=1e-7,
            dtype=torch.float64, device=None):
        """Fit to target inclusion probabilities via L-BFGS."""
        from conditional_poisson.tree_torch import _to_tensor
        target_incl = _to_tensor(target_incl, dtype).to(device=device)
        cp = cls(n, torch.logit(target_incl))
        cp._theta.requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [cp._theta],
            max_iter=200,
            history_size=5,
            line_search_fn='strong_wolfe',
            tolerance_grad=tol,
            tolerance_change=0,
        )

        def closure():
            optimizer.zero_grad()
            loss = cp._log_Z() - target_incl @ cp._theta
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            cp._theta -= cp._theta.mean()
        cp._theta = cp._theta.detach()

        return cp

    def __repr__(self):
        return f"ConditionalPoissonSequentialTorch(N={self.N}, n={self.n})"
