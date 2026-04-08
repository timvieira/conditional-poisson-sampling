"""Sequential O(Nn) conditional Poisson distribution (PyTorch)."""

from functools import cached_property
import torch
from conditional_poisson._base_torch import ConditionalPoissonTorchBase


class ConditionalPoissonSequentialTorch(ConditionalPoissonTorchBase):
    """Sequential O(Nn) DP via ESP recurrence (PyTorch).

    Constructors: __init__(n, theta), from_weights(n, w), fit(target_incl, n)
    Properties:   incl_prob, log_normalizer, n, N, theta
    Methods:      log_prob(S), sample(), clear()
    """

    def _log_Z(self, theta):
        """ESP recurrence: E[k, n+1] = E[k, n] + w[n] * E[k-1, n]."""
        N, K = self.N, self.n
        w = torch.exp(theta)
        E = torch.zeros(K + 1, dtype=theta.dtype, device=theta.device)
        E[0] = 1.0
        cols = [E]
        for i in range(N):
            new_E = cols[i].clone()
            new_E[1:] = cols[i][1:] + w[i] * cols[i][:K]
            cols.append(new_E)
        return torch.log(cols[N][K])

    @cached_property
    def _sample_data(self):
        E_full, w = [], torch.exp(self.theta).detach().tolist()
        theta = self.theta.detach()
        N, K = self.N, self.n
        E = [0.0] * (K + 1)
        E[0] = 1.0
        E_full.append(list(E))
        wt = torch.exp(theta).tolist()
        for i in range(N):
            new_E = list(E)
            for k in range(1, K + 1):
                new_E[k] = E[k] + wt[i] * E[k - 1]
            E = new_E
            E_full.append(list(E))
        return E_full, w

    def sample(self) -> torch.Tensor:
        """Right-to-left scan on the forward ESP table.  O(N)."""
        import random
        E, w = self._sample_data
        N, K = self.N, self.n
        selected = []
        k = K
        for i in reversed(range(N)):
            if k == 0:
                break
            if random.random() * E[i + 1][k] <= w[i] * E[i][k - 1]:
                selected.append(i)
                k -= 1
        selected.reverse()
        return torch.tensor(selected, dtype=torch.long)

    def __repr__(self):
        return f"ConditionalPoissonSequentialTorch(N={self.N}, n={self.n})"
