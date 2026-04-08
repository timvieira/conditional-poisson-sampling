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

    Constructors: __init__(n, theta), from_weights(n, w), fit(target_incl, n)
    Properties:   incl_prob, log_normalizer, theta, n, N
    Methods:      log_prob(S), sample(rng)
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
        self.theta = theta.detach().clone()
        self._cache: dict = {}

    @classmethod
    def from_weights(cls, n: int, w) -> ConditionalPoissonSequentialTorch:
        w = torch.as_tensor(w, dtype=torch.float64)
        if torch.any(w < 0):
            raise ValueError("weights must be non-negative")
        if torch.any(w == 0) or torch.any(torch.isinf(w)):
            raise ValueError("sequential DP requires finite positive weights")
        return cls(n, torch.log(w))

    # ── Log normalizer ───────────────────────────────────────────────────────

    def _log_Z(self):
        """Compute log Z as a differentiable scalar tensor.

        Uses the weighted Pascal recurrence F[k] += q[i] * F[k-1]
        with uniform row-wise rescaling for numerical stability.
        """
        theta = self.theta
        N, n = self.N, self.n
        log_gm = theta.mean()
        q = torch.exp(theta - log_gm)

        # F[k] tracks e_k(q[0:i]) with uniform row-wise scaling.
        # We avoid in-place ops for autograd by creating new tensors.
        F = torch.zeros(n + 1, dtype=theta.dtype, device=theta.device)
        F[0] = 1.0
        log_scale = torch.zeros(1, dtype=theta.dtype, device=theta.device)

        for i in range(N):
            new_F = F.clone()
            new_F[1:] = F[1:] + q[i] * F[:n]
            mx = new_F.max()
            if mx > 0:
                new_F = new_F / mx
                log_scale = log_scale + torch.log(mx)
            F = new_F

        return torch.log(F[n]) + log_scale.squeeze() + n * log_gm

    @property
    def log_normalizer(self) -> float:
        """log Z(w, n).  O(Nn)."""
        return self._log_Z().item()

    # ── Inclusion probabilities ──────────────────────────────────────────────

    @property
    def incl_prob(self) -> torch.Tensor:
        """Inclusion probabilities pi_i = d(log Z)/d(theta_i) via autograd."""
        saved = self.theta
        self.theta = saved.detach().requires_grad_(True)
        log_Z = self._log_Z()
        pi = torch.autograd.grad(log_Z, self.theta)[0].detach()
        self.theta = saved
        return pi.clone()

    # ── Sampling ─────────────────────────────────────────────────────────────

    def _get_suffix_table(self):
        """Build and cache the suffix ESP table for sampling.

        B[i] is the uniformly-scaled ESP vector for items i..N-1.
        true value: e_k(q[i:N]) = B[i,k] * exp(Bls[i]).
        """
        if "suffix" not in self._cache:
            N, n = self.N, self.n
            theta = self.theta.detach()
            log_gm = theta.mean()
            q = torch.exp(theta - log_gm)

            B = torch.zeros(N + 1, n + 1, dtype=theta.dtype)
            scale_factor = torch.ones(N + 1, dtype=theta.dtype)
            B[N, 0] = 1.0
            for i in range(N - 1, -1, -1):
                B[i] = B[i+1].clone()
                B[i, 1:] += q[i] * B[i+1, :n]
                mx = B[i].max().item()
                if mx > 0:
                    B[i] /= mx
                    scale_factor[i] = mx
            self._cache["suffix"] = (q, B, scale_factor)
        return self._cache["suffix"]

    def sample(self) -> torch.Tensor:
        """
        Draw one sample via sequential scan.

        Complexity: O(Nn) to build suffix table [cached] + O(N).
        """
        if "sample_data" not in self._cache:
            q, B, sf = self._get_suffix_table()
            self._cache["sample_data"] = (
                q.tolist(), B.tolist(), sf.tolist()
            )
        q, B, sf = self._cache["sample_data"]
        N, n = self.N, self.n
        import random
        selected = []
        k = n
        for i in range(N):
            if k == 0:
                break
            prob = q[i] * B[i+1][k-1] / (B[i][k] * sf[i])
            if random.random() < prob:
                selected.append(i)
                k -= 1
        return torch.tensor(selected, dtype=torch.long)

    # ── Log probability ───────────────────────────────────────────────────────

    def log_prob(self, S) -> float:
        """log P(S) = sum_{i in S} theta_i - log Z."""
        S = torch.as_tensor(S, dtype=torch.long)
        return float(self.theta[S].sum() - self.log_normalizer)

    # ── Fitting ──────────────────────────────────────────────────────────────

    @classmethod
    def fit(cls, target_incl, n, *, tol=1e-7,
            dtype=torch.float64, device=None):
        """Fit to target inclusion probabilities via L-BFGS."""
        from conditional_poisson.tree_torch import _to_tensor
        target_incl = _to_tensor(target_incl, dtype).to(device=device)
        cp = cls(n, torch.logit(target_incl))
        cp.theta.requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [cp.theta],
            max_iter=200,
            history_size=5,
            line_search_fn='strong_wolfe',
            tolerance_grad=tol,
            tolerance_change=0,
        )

        def closure():
            optimizer.zero_grad()
            loss = cp._log_Z() - target_incl @ cp.theta
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            cp.theta -= cp.theta.mean()
        cp.theta = cp.theta.detach()

        return cp

    def __repr__(self):
        return f"ConditionalPoissonSequentialTorch(N={self.N}, n={self.n})"
