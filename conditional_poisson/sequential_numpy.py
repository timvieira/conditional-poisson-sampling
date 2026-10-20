"""Sequential O(Nn) conditional Poisson distribution (NumPy)."""

from functools import cached_property
import numpy as np
from scipy.special import expit
from conditional_poisson._base_numpy import ConditionalPoissonNumPyBase
from conditional_poisson._contour import log_contour_radius


class ConditionalPoissonSequentialNumPy(ConditionalPoissonNumPyBase):
    """Sequential O(Nn) DP via ESP recurrence (NumPy).

    Constructors: __init__(n, theta), from_weights(n, w), fit(target_incl, n)
    Properties:   incl_prob, log_normalizer, n, N, theta
    Methods:      log_prob(S), sample(), clear()
    """

    @cached_property
    def _forward(self):
        """Forward table of prefix ESPs, normalized.  O(Nn).

        Rescale the weights by the contour radius r, v = r w (this does not
        change the distribution; see conditional_poisson._contour), and divide
        the prefix ESPs by prod_{i<j} (1 + v_i):

            E[j, k] = e_k(v[0:j]) / prod_{i<j} (1 + v_i).

        Then row j is the pmf of a sum of Bernoulli(p_i) for i < j, where
        p_i = v_i / (1 + v_i), truncated to k <= n, and the ESP recurrence
        becomes a convex combination

            E[j+1, k] = (1 - p_j) E[j, k] + p_j E[j, k-1],

        so no entry can overflow.
        """
        N, K = self.N, self.n
        log_r = log_contour_radius(self.theta, K)
        x = self.theta + log_r
        p, q = expit(x), expit(-x)              # q = 1 - p, accurate when p is near 1
        E = np.zeros((N + 1, K + 1))
        E[0, 0] = 1.0
        for j, (pj, qj) in enumerate(zip(p.tolist(), q.tolist())):
            np.multiply(E[j], qj, out=E[j + 1])
            E[j + 1, 1:] += pj * E[j, :K]
        # log prod_i (1 + v_i) = sum_i log(1 + exp(theta_i + log r))
        log_scale = float(np.logaddexp(0.0, x).sum())
        return E, (p, q), log_scale, log_r

    @cached_property
    def log_normalizer(self) -> float:
        E, _, log_scale, log_r = self._forward
        N, K = self.N, self.n
        return float(np.log(E[N, K]) + log_scale - K * log_r)

    @cached_property
    def incl_prob(self) -> np.ndarray:
        """Reverse-mode AD on the ESP recurrence.  O(Nn).

        The adjoint of row j+1 is (proportional to) the vector of suffix
        quantities delta[k] = Pr[sum_{i>j} Bernoulli(p_i) = n - k], which obeys
        the mirror-image recurrence.  Splitting the subsets counted at row j+1
        by whether they contain item j gives

            incl = p_j  * (E[j, :n] . delta[1:])
            excl = (1 - p_j) * (E[j] . delta)
            pi_j = incl / (incl + excl).
        """
        E, (p, q), _, _ = self._forward
        N, K = self.N, self.n
        pi = np.zeros(N)
        if K == 0:
            return pi
        p, q = p.tolist(), q.tolist()
        delta = np.zeros(K + 1)
        delta[K] = 1.0
        for j in reversed(range(N)):
            row = E[j]
            incl = p[j] * (row[:K] @ delta[1:])
            excl = q[j] * (row @ delta)
            pi[j] = incl / (incl + excl)
            shifted = p[j] * delta[1:]
            delta *= q[j]
            delta[:K] += shifted
        return pi

    @cached_property
    def _sample_data(self):
        E, (p, q), _, _ = self._forward
        return E.data, p.tolist(), q.tolist()   # fast scalar reads, no copy

    def sample(self) -> np.ndarray:
        """Right-to-left scan on the forward table.  O(N) per sample.

        With k items still to choose from items 0..j, include item j with
        probability p_j E[j, k-1] / ((1 - p_j) E[j, k] + p_j E[j, k-1]).
        """
        E, p, q = self._sample_data
        u = np.random.random(self.N).tolist()
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
        return np.array(selected, dtype=np.int32)

    def __repr__(self):
        return f"ConditionalPoissonSequentialNumPy(N={self.N}, n={self.n})"
