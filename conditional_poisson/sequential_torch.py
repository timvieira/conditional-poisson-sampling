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

    def _circuit(self, theta):
        """ESP recurrence returning (log_Z, E_columns)."""
        N, K = self.N, self.n
        w = torch.exp(theta)
        E = torch.zeros(K + 1, dtype=theta.dtype, device=theta.device)
        E[0] = 1.0
        cols = [E]
        for i in range(N):
            new_E = cols[i].clone()
            new_E[1:] = cols[i][1:] + w[i] * cols[i][:K]
            cols.append(new_E)
        log_Z = torch.log(cols[N][K])
        return log_Z, (cols, w)

    @cached_property
    def _sample_data(self):
        """ESP table and weights from _forward, detached."""
        _, _, (cols, w) = self._forward
        E = torch.stack(cols).detach()
        return E, w.detach()

    def sample(self) -> torch.Tensor:
        """Right-to-left scan on the forward ESP table.  O(N)."""
        E, w = self._sample_data
        N, K = self.N, self.n
        selected = []
        k = K
        for i in reversed(range(N)):
            if k == 0:
                break
            if torch.rand(1).item() * E[i + 1, k].item() <= w[i].item() * E[i, k - 1].item():
                selected.append(i)
                k -= 1
        selected.reverse()
        return torch.tensor(selected, dtype=torch.long)

    def __repr__(self):
        return f"ConditionalPoissonSequentialTorch(N={self.N}, n={self.n})"
