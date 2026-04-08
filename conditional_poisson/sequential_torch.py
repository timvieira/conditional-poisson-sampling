"""
Sequential O(Nn) implementation of the conditional Poisson distribution (PyTorch).

    P(S) ∝ prod_{i in S} w_i,   |S| = n

Uses the elementary symmetric polynomial (ESP) recurrence for all
computations: normalizing constant, inclusion probabilities (via
torch.autograd on the DP), and sampling (right-to-left scan on the
forward table).
"""

from __future__ import annotations
from functools import cached_property
import torch


class ConditionalPoissonSequentialTorch:
    """Conditional Poisson distribution via O(Nn) sequential DP (PyTorch).

    Constructors: __init__(n, theta), from_weights(n, w), fit(target_incl, n)
    Properties:   incl_prob, log_normalizer, n, N, theta
    Methods:      log_prob(S), sample(), clear()
    """

    def __init__(self, n: int, theta):
        theta = torch.as_tensor(theta, dtype=torch.float64)
        assert theta.ndim == 1
        assert 0 <= n <= len(theta)
        self.n = n
        self.N = len(theta)
        self.theta = theta.detach().clone()

    @classmethod
    def from_weights(cls, n: int, w) -> ConditionalPoissonSequentialTorch:
        w = torch.as_tensor(w, dtype=torch.float64)
        if torch.any(w <= 0) or not torch.isfinite(w).all():
            raise ValueError("all weights must be finite and positive")
        return cls(n, torch.log(w))

    def clear(self):
        """Flush all cached computations."""
        for attr in ('_E_differentiable', 'log_normalizer', 'incl_prob', '_sample_data'):
            self.__dict__.pop(attr, None)

    def _build_E(self, theta):
        """Build the full ESP table E[k, n] = e_k(w[0:n]).

        Returns (E, w) where E is a (K+1, N+1) tensor on the autograd graph.
        """
        N, K = self.N, self.n
        w = torch.exp(theta)
        cols = [torch.zeros(K + 1, dtype=theta.dtype, device=theta.device)]
        cols[0][0] = 1.0
        for i in range(N):
            new_col = cols[i].clone()
            new_col[1:] = cols[i][1:] + w[i] * cols[i][:K]
            cols.append(new_col)
        E = torch.stack(cols, dim=1)
        return E, w

    @cached_property
    def log_normalizer(self) -> float:
        """log Z(w, n).  O(Nn).  Does not trigger backward pass."""
        E, _ = self._build_E(self.theta.detach())
        return float(torch.log(E[self.n, self.N]).item())

    @cached_property
    def incl_prob(self) -> torch.Tensor:
        """Inclusion probabilities pi_i = d(log Z)/d(theta_i) via autograd.  O(Nn)."""
        theta = self.theta.detach().requires_grad_(True)
        E, _ = self._build_E(theta)
        log_Z = torch.log(E[self.n, self.N])
        return torch.autograd.grad(log_Z, theta)[0].detach()

    @cached_property
    def _sample_data(self):
        """Build ESP table as plain lists for fast sampling loop."""
        E, w = self._build_E(self.theta.detach())
        return E.tolist(), w.tolist()

    def sample(self) -> torch.Tensor:
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
        return torch.tensor(selected, dtype=torch.long)

    def log_prob(self, S) -> float:
        """log P(S) = sum_{i in S} theta_i - log Z."""
        S = torch.as_tensor(S, dtype=torch.long)
        return float(self.theta[S].sum() - self.log_normalizer)

    @classmethod
    def fit(cls, target_incl, n, *, tol=1e-7, dtype=torch.float64, device=None):
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
            cp.clear()
            E, _ = cp._build_E(cp.theta)
            loss = torch.log(E[cp.n, cp.N]) - target_incl @ cp.theta
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            cp.theta -= cp.theta.mean()
        cp.theta = cp.theta.detach()
        cp.clear()

        return cp

    def __repr__(self):
        return f"ConditionalPoissonSequentialTorch(N={self.N}, n={self.n})"
