"""
Product-tree implementation of the conditional Poisson distribution (NumPy).

    P(S) ∝ prod_{i in S} w_i,   |S| = n

Uses a polynomial product tree for O(N (log N)^2) computation of Z, pi,
and sampling.  The downward pass for inclusion probabilities is only
triggered when incl_prob is accessed.

TODO: truncate polynomials to degree n throughout the tree to achieve
O(N log^2 n) instead of O(N log^2 N).
"""

from __future__ import annotations
from functools import cached_property
import numpy as np
from typing import Union
from scipy.signal import convolve

__all__ = ["ConditionalPoissonNumPy"]


class ConditionalPoissonNumPy:
    """Conditional Poisson distribution over fixed-size subsets.

        P(S) ∝ prod_{i in S} w_i,   |S| = n

    Constructors: __init__(n, theta), from_weights(n, w), fit(target_incl, n)
    Properties:   incl_prob, log_normalizer, n, N, theta
    Methods:      log_prob(S), sample(), fit_inplace(target_incl), clear()
    """

    def __init__(self, n: int, theta: np.ndarray):
        theta = np.asarray(theta, float)
        assert theta.ndim == 1
        assert np.all(np.isfinite(theta))
        assert 0 <= n <= len(theta)
        self.n = int(n)
        self.N = len(theta)
        self.theta = theta.copy()

    @classmethod
    def from_weights(cls, n: int, w: np.ndarray) -> "ConditionalPoissonNumPy":
        w = np.asarray(w, float)
        if np.any(w <= 0) or not np.all(np.isfinite(w)):
            raise ValueError("all weights must be finite and positive")
        return cls(n, np.log(w))

    @classmethod
    def fit(cls, target_incl, n, *, tol=1e-10, max_iter=200, verbose=False):
        obj = cls(n, np.zeros(len(target_incl)))
        obj.fit_inplace(target_incl, tol=tol, max_iter=max_iter, verbose=verbose)
        return obj

    def clear(self):
        """Flush all cached computations."""
        for attr in ('_tree', 'log_normalizer', 'incl_prob', '_sample_tree'):
            self.__dict__.pop(attr, None)

    def _scale(self, c):
        m = np.max(c)
        if m == 0:
            return c, -np.inf
        return c / m, np.log(m)

    def _pmul(self, a, als, b, bls):
        c = convolve(a, b)
        cn, inc = self._scale(c)
        return cn, als + bls + inc

    @cached_property
    def _tree(self):
        """Build the product tree.  O(N (log N)^2)."""
        N, n = self.N, self.n
        log_gm = float(np.mean(self.theta))
        q_s = np.exp(self.theta - log_gm)

        tree_n = 1
        while tree_n < N:
            tree_n <<= 1
        Pc = [None] * (2 * tree_n)
        Pls = np.full(2 * tree_n, -np.inf)

        for i in range(tree_n):
            if i < N:
                Pc[tree_n + i], Pls[tree_n + i] = self._scale(np.array([1.0, q_s[i]]))
            else:
                Pc[tree_n + i] = np.array([1.0])
                Pls[tree_n + i] = 0.0

        for i in range(tree_n - 1, 0, -1):
            l, r = 2 * i, 2 * i + 1
            Pc[i], Pls[i] = self._pmul(Pc[l], Pls[l], Pc[r], Pls[r])

        return Pc, Pls, tree_n, q_s, log_gm

    @cached_property
    def log_normalizer(self) -> float:
        """Log normalizing constant.  O(N (log N)^2).  Does not trigger downward pass."""
        Pc, Pls, _, _, log_gm = self._tree
        n = self.n
        en_n = float(Pc[1][n]) if len(Pc[1]) > n else 0.0
        root_ls = float(Pls[1])
        return (np.log(en_n) + root_ls + n * log_gm) if en_n != 0.0 else -np.inf

    @cached_property
    def incl_prob(self) -> np.ndarray:
        """Inclusion probabilities via downward pass.  O(N (log N)^2)."""
        Pc, Pls, tree_n, q_s, _ = self._tree
        N, n = self.N, self.n

        N2 = 2 * tree_n
        oPc = [None] * N2
        oPls = np.full(N2, -np.inf)
        oPc[1] = np.array([1.0])
        oPls[1] = 0.0
        for i in range(1, tree_n):
            if oPc[i] is None:
                continue
            l, r = 2 * i, 2 * i + 1
            for c, s in ((l, r), (r, l)):
                oPc[c], oPls[c] = self._pmul(oPc[i], oPls[i], Pc[s], Pls[s])

        en_n = float(Pc[1][n]) if len(Pc[1]) > n else 0.0
        root_ls = float(Pls[1])

        pi = np.zeros(N)
        for i in range(N):
            op = oPc[tree_n + i]
            op_ls = float(oPls[tree_n + i])
            if en_n != 0.0 and op is not None and len(op) > n - 1:
                pi[i] = q_s[i] * float(op[n - 1]) / en_n * np.exp(op_ls - root_ls)
        return pi

    @cached_property
    def _sample_tree(self):
        """Prepare tree data for fast sampling loop."""
        Pc, Pls, tree_n, _, _ = self._tree
        ratio = [1.0] * (2 * tree_n)
        for i in range(1, tree_n):
            ratio[i] = np.exp(Pls[i] - Pls[2*i] - Pls[2*i+1])
        return (
            [p.tolist() if p is not None else [] for p in Pc],
            ratio, tree_n,
        )

    def sample(self) -> np.ndarray:
        """Draw one sample via top-down quota splitting.

        Complexity: O(N (log N)^2) to build tree [cached] + O(n log N).
        """
        import random
        Pc, ratio, tree_n = self._sample_tree
        N, n = self.N, self.n
        selected = []
        stack = [(1, n)]
        while stack:
            node, k = stack.pop()
            if k == 0:
                continue
            if node >= tree_n:
                if node - tree_n < N:
                    selected.append(node - tree_n)
                continue
            L = Pc[2 * node]
            R = Pc[2 * node + 1]
            u = random.random() * Pc[node][k] * ratio[node]
            acc = 0.0
            for j in range(k + 1):
                lv = L[j] if j < len(L) else 0.0
                rv = R[k - j] if k - j < len(R) else 0.0
                acc += lv * rv
                if acc >= u:
                    break
            stack.append((2 * node + 1, k - j))
            stack.append((2 * node, j))
        selected.sort()
        return np.array(selected, dtype=np.int32)

    def log_prob(self, S: Union[np.ndarray, list]) -> Union[float, np.ndarray]:
        """log P(S) = sum_{i in S} theta_i - log Z."""
        S = np.asarray(S)
        lz = self.log_normalizer
        th = self.theta
        if S.dtype == bool:
            return (th @ S.T - lz) if S.ndim == 2 else float(th[S].sum() - lz)
        else:
            return (th[S].sum(axis=1) - lz) if S.ndim == 2 else float(th[S].sum() - lz)

    def fit_inplace(self, target_incl, *, tol=1e-10, max_iter=200, verbose=False):
        """Update weights to match target inclusion probabilities.  Returns self."""
        from scipy.optimize import minimize
        from scipy.special import logit

        target_incl = np.asarray(target_incl, float)
        assert len(target_incl) == self.N
        assert np.all((target_incl > 0) & (target_incl < 1))

        def neg_ll_and_grad(theta):
            self.theta = theta
            self.clear()
            pi = self.incl_prob
            loss = self.log_normalizer - float(target_incl @ theta)
            grad = pi - target_incl
            if verbose:
                print(f"  max|pi-pi*| = {np.max(np.abs(grad)):.3e}")
            return loss, grad

        result = minimize(
            neg_ll_and_grad, logit(target_incl),
            method='L-BFGS-B', jac=True,
            options={'maxiter': max_iter, 'gtol': tol, 'ftol': 0},
        )
        theta = result.x
        theta -= theta.mean()
        self.theta = theta
        self.clear()
        return self

    def __repr__(self):
        return f"ConditionalPoissonNumPy(N={self.N}, n={self.n})"
