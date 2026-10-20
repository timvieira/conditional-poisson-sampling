"""Sequential O(Nn) conditional Poisson distribution (PyTorch)."""

from functools import cached_property
import torch
import torch.nn.functional as F
from conditional_poisson._base_torch import ConditionalPoissonTorchBase


class ConditionalPoissonSequentialTorch(ConditionalPoissonTorchBase):
    """Sequential O(Nn) DP via ESP recurrence (PyTorch).

    Constructors: __init__(n, theta), from_weights(n, w), fit(target_incl, n)
    Properties:   incl_prob, log_normalizer, n, N, theta
    Methods:      log_prob(S), sample(), clear()
    """

    def _circuit(self, theta):
        """Normalized ESP recurrence.  Returns (log_Z, (rows, p, q)).

        Same table as ConditionalPoissonSequentialNumPy._forward: with
        p_j = sigmoid(theta_j + log r), row j is the pmf of a sum of
        Bernoulli(p_i) for i < j, truncated to k <= n, so no entry can overflow.
        """
        N, K = self.N, self.n
        x = theta + self._log_r
        p, q = torch.sigmoid(x), torch.sigmoid(-x)      # q = 1 - p, accurate near 1
        row = F.pad(torch.ones(1, dtype=theta.dtype, device=theta.device), (0, K))
        rows = [row]
        for j in range(N):
            row = q[j] * row + p[j] * F.pad(row[:K], (1, 0))
            rows.append(row)
        # log prod_i (1 + v_i) with v_i = exp(theta_i + log r)
        # (torch.logaddexp, not F.softplus, which is linear above x = 20)
        log_Z = torch.log(row[K]) + torch.logaddexp(torch.zeros_like(x), x).sum() - K * self._log_r
        return log_Z, (rows, p, q)

    @cached_property
    def _sample_data(self):
        _, _, (rows, p, q) = self._forward
        E = torch.stack(rows).detach().cpu().numpy()
        return E.data, p.tolist(), q.tolist()           # fast scalar reads, no copy

    def sample(self) -> torch.Tensor:
        """Right-to-left scan on the forward table.  O(N) per sample.

        With k items still to choose from items 0..j, include item j with
        probability p_j E[j, k-1] / ((1 - p_j) E[j, k] + p_j E[j, k-1]).
        """
        E, p, q = self._sample_data
        u = torch.rand(self.N, dtype=torch.float64).tolist()
        selected = []
        k = self.n
        for j in reversed(range(self.N)):
            if k == 0:
                break
            incl = p[j] * E[j, k - 1]
            if u[j] * (q[j] * E[j, k] + incl) < incl:
                selected.append(j)
                k -= 1
        selected.reverse()
        return torch.tensor(selected, dtype=torch.long)

    def __repr__(self):
        return f"ConditionalPoissonSequentialTorch(N={self.N}, n={self.n})"
