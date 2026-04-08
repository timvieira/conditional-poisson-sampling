"""
Sequential O(Nn) implementation of the conditional Poisson distribution (PyTorch).

    P(S) ∝ prod_{i in S} w_i,   |S| = n

Uses the elementary symmetric polynomial (ESP) recurrence for all
computations: normalizing constant, inclusion probabilities (via
torch.autograd on the DP), and sampling (right-to-left scan on the
forward table).
"""

from __future__ import annotations
import torch


class ConditionalPoissonSequentialTorch:
    """Conditional Poisson distribution via O(Nn) sequential DP (PyTorch).

    Constructors: __init__(n, theta), from_weights(n, w), fit(target_incl, n)
    Properties:   incl_prob, log_normalizer, n, N, theta
    Methods:      log_prob(S), sample()
    """

    def __init__(self, n: int, theta):
        theta = torch.as_tensor(theta, dtype=torch.float64)
        assert theta.ndim == 1
        assert 0 <= n <= len(theta)
        self.n = n
        self.N = len(theta)
        self.theta = theta.detach().clone()
        self._cache: dict = {}

    @classmethod
    def from_weights(cls, n: int, w) -> ConditionalPoissonSequentialTorch:
        w = torch.as_tensor(w, dtype=torch.float64)
        if torch.any(w <= 0) or not torch.isfinite(w).all():
            raise ValueError("all weights must be finite and positive")
        return cls(n, torch.log(w))

    def _log_Z(self):
        """Compute log Z as a differentiable scalar tensor.

        ESP recurrence: E[k, n+1] = E[k, n] + w[n] * E[k-1, n]
        with per-column rescaling. Scale factors stored for log Z.
        """
        theta = self.theta
        N, K = self.N, self.n
        w = torch.exp(theta)

        E = torch.zeros(K + 1, dtype=theta.dtype, device=theta.device)
        E[0] = 1.0
        sf = torch.ones(N, dtype=theta.dtype, device=theta.device)

        for n in range(N):
            new_E = E.clone()
            new_E[1:] = E[1:] + w[n] * E[:K]
            mx = new_E.max()
            if mx > 0:
                new_E = new_E / mx
                sf[n] = mx
            E = new_E

        return torch.log(E[K]) + sf.log().sum()

    @property
    def log_normalizer(self) -> float:
        """log Z(w, n).  O(Nn)."""
        return self._log_Z().item()

    @property
    def incl_prob(self) -> torch.Tensor:
        """Inclusion probabilities pi_i = d(log Z)/d(theta_i) via autograd."""
        saved = self.theta
        self.theta = saved.detach().requires_grad_(True)
        log_Z = self._log_Z()
        pi = torch.autograd.grad(log_Z, self.theta)[0].detach()
        self.theta = saved
        return pi

    def _get_E_for_sampling(self):
        """Build and cache the ESP table for sampling (detached, as lists)."""
        if "sample_E" not in self._cache:
            w = torch.exp(self.theta).detach()
            N, K = self.N, self.n
            E = torch.zeros(K + 1, N + 1, dtype=self.theta.dtype)
            sf = torch.ones(N + 1, dtype=self.theta.dtype)
            E[0, :] = 1.0
            for n in range(N):
                E[1:, n+1] = E[1:, n] + w[n] * E[:K, n]
                mx = E[:, n+1].max().item()
                if mx > 0:
                    E[:, n+1] /= mx
                    sf[n+1] = mx
            self._cache["sample_E"] = (
                E.tolist(), sf.tolist(), w.tolist()
            )
        return self._cache["sample_E"]

    def sample(self) -> torch.Tensor:
        """Draw one sample by scanning items right-to-left.

        P(include i) = w[i] * E[k-1, i] / (E[k, i+1] * sf[i+1])

        Complexity: O(Nn) to build table [cached] + O(N).
        """
        import random
        E, sf, w = self._get_E_for_sampling()
        N, K = self.N, self.n
        selected = []
        k = K
        for i in reversed(range(N)):
            if k == 0:
                break
            prob = w[i] * E[k-1][i] / (E[k][i+1] * sf[i+1])
            if random.random() < prob:
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
