"""Product-tree conditional Poisson distribution (NumPy).

Multiplies the leaf polynomials up a balanced binary tree, truncating every
product to degree n, for O(N log^2 n) time.  The weights are rescaled by the
contour radius r (see conditional_poisson._contour) and each leaf (1 + w_i r z)
is divided by (1 + w_i r), as in the scaled DP of
conditional_poisson.sequential_numpy.  The leaf is then the generating function
(1 - p_i) + p_i z of a coin with p_i = w_i r / (1 + w_i r), so every node holds
the distribution of the number of heads among its coins (truncated to degree
n): every coefficient is between 0 and 1, nothing can overflow, and the
coefficient that log Z needs is the largest one at the root.  Inclusion
probabilities come from a reverse pass over the same tree, run only when
incl_prob is accessed.
"""

from bisect import bisect_right
from functools import cached_property
from itertools import accumulate
import numpy as np
from typing import Union
from scipy.signal import convolve
from scipy.special import expit
from conditional_poisson._base_numpy import ConditionalPoissonNumPyBase
from conditional_poisson._contour import log_contour_radius

__all__ = ["ConditionalPoissonNumPy"]


class ConditionalPoissonNumPy(ConditionalPoissonNumPyBase):
    """Product-tree O(N log^2 n) via scipy convolve (NumPy).

    Constructors: __init__(n, theta), from_weights(n, w), fit(target_incl, n)
    Properties:   incl_prob, log_normalizer, n, N, theta
    Methods:      log_prob(S), sample(), clear()
    """

    @cached_property
    def _forward(self):
        """Build the product tree.  O(N log^2 n).

        Nodes are in heap order: the root is 1, the children of node i are 2i
        and 2i + 1, and the leaves are tree_n, ..., 2 tree_n - 1.  P[i] holds
        Pr[K_i = k] for k = 0, ..., min(size_i, n), where K_i is the number of
        heads among the coins under node i and size_i is how many there are.
        Padding leaves hold no coins, so they are the constant polynomial 1.

        Returns (P, tree_n, (p, q), log prod_i (1 + w_i r), log r).
        """
        N, n = self.N, self.n
        log_r = log_contour_radius(self.theta, n)
        x = self.theta + log_r
        p, q = expit(x), expit(-x)              # q = 1 - p, accurate when p is near 1

        tree_n = 1
        while tree_n < N:
            tree_n <<= 1
        P: list = [None] * (2 * tree_n)
        for i in range(tree_n):
            P[tree_n + i] = np.array([q[i], p[i]])[:n + 1] if i < N else np.ones(1)
        for i in range(tree_n - 1, 0, -1):
            P[i] = convolve(P[2 * i], P[2 * i + 1])[:n + 1]

        log_scale = float(np.logaddexp(0.0, x).sum())
        return P, tree_n, (p, q), log_scale, log_r

    @cached_property
    def log_normalizer(self) -> float:
        """log Z = log Pr[K = n] + sum_i log(1 + w_i r) - n log r.
        Does not trigger the reverse pass."""
        P, _, _, log_scale, log_r = self._forward
        return float(np.log(P[1][self.n]) + log_scale - self.n * log_r)

    @cached_property
    def incl_prob(self) -> np.ndarray:
        """Inclusion probabilities by a reverse pass over the tree.  O(N log^2 n).

        G[i][k] is the derivative of Pr[K = n] with respect to P[i][k], which is
        Pr[number of heads among the coins not under node i = n - k]; it is
        also a probability.  A parent c = a * b (truncated) passes its children

            G[a][j] = sum_m G[c][j + m] P[b][m],

        a correlation.  At leaf i, splitting the subsets by whether they
        contain item i gives

            incl = p_i G[i][1],   excl = (1 - p_i) G[i][0],   pi_i = incl / (incl + excl).
        """
        P, tree_n, (p, q), _, _ = self._forward
        N, n = self.N, self.n
        if n == 0:
            return np.zeros(N)

        G: list = [None] * (2 * tree_n)
        G[1] = np.zeros(n + 1)
        G[1][n] = 1.0
        for c in range(1, tree_n):
            Lc = len(P[c])
            for a, b in ((2 * c, 2 * c + 1), (2 * c + 1, 2 * c)):
                full = convolve(G[c][::-1], P[b])
                G[a] = full[Lc - len(P[a]):Lc][::-1]

        pi = np.empty(N)
        for i in range(N):
            g = G[tree_n + i]
            incl = p[i] * g[1]
            pi[i] = incl / (incl + q[i] * g[0])
        return pi

    @cached_property
    def _sample_data(self):
        P, tree_n, _, _, _ = self._forward
        # At depth d, at most min(2^d, n) nodes receive a nonzero quota.
        max_splits = sum(min(1 << d, self.n) for d in range(tree_n.bit_length() - 1))
        return [None] + [poly.tolist() for poly in P[1:]], tree_n, max_splits

    def sample(self) -> np.ndarray:
        """Top-down quota splitting on the product tree.  O(n log N) per sample.

        A node with quota k gives j items to its left child and k - j to its
        right with probability proportional to Pr[K_left = j] Pr[K_right = k - j].
        """
        P, tree_n, max_splits = self._sample_data
        u = iter(np.random.random(max_splits).tolist())
        selected = []
        stack = [(1, self.n)] if self.n > 0 else []
        while stack:
            node, k = stack.pop()
            if node >= tree_n:                  # a real leaf with quota 1
                selected.append(node - tree_n)
                continue
            a, b = 2 * node, 2 * node + 1
            L, R = P[a], P[b]
            # A child of size s holds coefficients up to degree min(s, n) >= its
            # quota, so these bounds are exactly the children's capacities.
            lo, hi = max(0, k - (len(R) - 1)), min(k, len(L) - 1)
            if lo == hi:
                j = lo
            elif k == 1:                        # the common case near the leaves
                left, right = L[1] * R[0], L[0] * R[1]
                j = 1 if next(u) * (left + right) < left else 0
            else:
                cdf = list(accumulate(L[j] * R[k - j] for j in range(lo, hi + 1)))
                j = lo + min(bisect_right(cdf, next(u) * cdf[-1]), hi - lo)
            if j:
                stack.append((a, j))
            if k - j:
                stack.append((b, k - j))
        selected.sort()
        return np.array(selected, dtype=np.int32)

    def log_prob(self, S: Union[np.ndarray, list]) -> Union[float, np.ndarray]:
        """log P(S) = sum_{i in S} theta_i - log Z.  Supports batch."""
        S = np.asarray(S)
        lz = self.log_normalizer
        th = self.theta
        if S.dtype == bool:
            return (th @ S.T - lz) if S.ndim == 2 else float(th[S].sum() - lz)
        else:
            return (th[S].sum(axis=1) - lz) if S.ndim == 2 else float(th[S].sum() - lz)

    def __repr__(self):
        return f"ConditionalPoissonNumPy(N={self.N}, n={self.n})"
