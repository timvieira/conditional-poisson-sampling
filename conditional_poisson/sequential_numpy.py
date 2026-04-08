"""Sequential O(Nn) conditional Poisson distribution (NumPy)."""

from functools import cached_property
import numpy as np
from conditional_poisson._base_numpy import ConditionalPoissonNumPyBase


class ConditionalPoissonSequentialNumPy(ConditionalPoissonNumPyBase):
    """Sequential O(Nn) DP via ESP recurrence (NumPy).

    Constructors: __init__(n, theta), from_weights(n, w), fit(target_incl, n)
    Properties:   incl_prob, log_normalizer, n, N, theta
    Methods:      log_prob(S), sample(), clear()
    """

    @cached_property
    def _forward(self):
        """Forward ESP table E[k, n] = e_k(w[0:n]).  O(Nn)."""
        w = np.exp(self.theta)
        N, K = self.N, self.n
        E = np.zeros((K + 1, N + 1))
        E[0, :] = 1.0
        for n in range(N):
            E[1:, n + 1] = E[1:, n] + w[n] * E[:K, n]
        return E, w

    @cached_property
    def log_normalizer(self) -> float:
        E, _ = self._forward
        return float(np.log(E[self.n, self.N]))

    @cached_property
    def incl_prob(self) -> np.ndarray:
        """Reverse-mode AD on the ESP recurrence.  O(Nn)."""
        E, w = self._forward
        N, K = self.N, self.n
        Z = E[K, N]
        dE = np.zeros((K + 1, N + 1))
        dw = np.zeros(N)
        dE[K, N] = 1.0
        for n in reversed(range(N)):
            for k in range(K, 0, -1):
                dE[k, n] += dE[k, n + 1]
                dw[n] += dE[k, n + 1] * E[k - 1, n]
                dE[k - 1, n] += dE[k, n + 1] * w[n]
        return w * dw / Z

    def sample(self) -> np.ndarray:
        """Right-to-left scan on the forward ESP table.  O(N)."""
        E, w = self._forward
        N, K = self.N, self.n
        selected = []
        k = K
        for i in reversed(range(N)):
            if np.random.random() * E[k, i + 1] <= w[i] * E[k - 1, i]:
                selected.append(i)
                k -= 1
            if k == 0:
                break
        selected.reverse()
        return np.array(selected, dtype=np.int32)

    def __repr__(self):
        return f"ConditionalPoissonSequentialNumPy(N={self.N}, n={self.n})"
