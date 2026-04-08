"""
conditional_poisson_numpy.py
======================

Conditional Poisson distribution over fixed-size subsets.

    P(S) ∝ prod_{i in S} w_i,   |S| = n

where w_i are non-negative weights (zero and infinity allowed).  Internally parameterised by
log-weights theta_i = log(w_i).

Everything — forward pass, gradient, and sampling —
uses a polynomial product tree built once and cached.

Tree structure
--------------
Upward pass (divide-and-conquer):

  P_T(z)  = prod_{i in T}(1 + w_i z)

  P[node] = convolve(P[L], P[R])

The root polynomial's k-th coefficient is Z(w choose k), the elementary
symmetric polynomial e_k(w).  No degree truncation is applied; each node
holds the full-degree polynomial for its subtree.

Downward pass propagates outside polynomials to every leaf:

  oP[leaf i] = P^{(-i)}(z) = prod_{j != i}(1 + w_j z)

Extraction:
  pi_i   = w_i * [z^{n-1}] P^{(-i)} / [z^n] P_root

Sampling
--------
The P-tree is cached and reused for sampling.
At each internal node with quota k, sample the left/right quota split:

  P(j items from L | k, T) ∝ poly_L[j] * poly_R[k-j]

Recurse down the tree; each sample costs O(n log N) in expectation.

Numerics
--------
Geometric-mean normalisation w -> w/exp(mean(log w)) is applied before
every tree build.  The scale factor alpha = exp(mean(log w)) is tracked
separately and compensated when extracting log Z:

  log Z = log(P_root[n]) + n * log(alpha)

Complexity
----------
The tree build follows the divide-and-conquer recurrence:

  T(N) = 2 T(N/2) + O(N log N)   =>   T(N) = O(N (log N)^2)

where the O(N log N) term is FFT-based polynomial multiplication.

  _build_tree            O(N (log N)^2)
  _forward (+ backward)  O(N (log N)^2)
  incl_prob / log_normalizer    O(N (log N)^2)  [cached]
  sample                 O(N (log N)^2 + n log N)
  fit(target_incl)       O(N (log N)^2 * L-BFGS_iters)

TODO: truncate polynomials to degree n throughout the tree to achieve
O(N log^2 n) instead of O(N log^2 N).  This requires per-coefficient
scaling to avoid underflow when the degree-n coefficient is much smaller
than the peak coefficient at degree ~subtree_size/2.
"""

from __future__ import annotations
import numpy as np
from typing import Union
from scipy.signal import convolve

__all__ = ["ConditionalPoissonNumPy"]


# ── Scaled polynomial arithmetic ─────────────────────────────────────────────
#
# Each polynomial is a pair (c, ls) where the true polynomial is c * exp(ls)
# and max|c| = 1.  Unlike the previous implementation, polynomials are NOT
# truncated — they keep their full natural degree.




# ══ ConditionalPoissonNumPy ════════════════════════════════════════════════════════

class ConditionalPoissonNumPy:
    """
    Conditional Poisson distribution over fixed-size subsets.

        P(S) ∝ prod_{i in S} w_i,   |S| = n

    Constructors: __init__(n, theta), from_weights(n, w), fit(target_incl, n)
    Properties:   incl_prob, log_normalizer, theta, n, N
    Methods:      log_prob(S), sample(rng), fit_inplace(target_incl)
    """

    def __init__(self, n: int, theta: np.ndarray):
        theta = np.asarray(theta, float)
        if theta.ndim != 1:   raise ValueError("theta must be 1-D")
        if not np.all(np.isfinite(theta)):
            raise ValueError("all theta must be finite (no w=0 or w=inf)")
        assert 0 <= n <= len(theta), f"n={n} must be in [0, {len(theta)}]"

        self.n      = int(n)
        self.N      = len(theta)
        self.theta = theta.copy()
        self._cache: dict = {}

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_weights(cls, n: int, w: np.ndarray) -> "ConditionalPoissonNumPy":
        """Construct from positive finite weights."""
        w = np.asarray(w, float)
        if np.any(w <= 0) or not np.all(np.isfinite(w)):
            raise ValueError("all weights must be finite and positive")
        return cls(n, np.log(w))

    @classmethod
    def fit(
        cls,
        target_incl: np.ndarray,
        n: int,
        *,
        tol: float = 1e-10,
        max_iter: int = 200,
        verbose: bool = False,
    ) -> "ConditionalPoissonNumPy":
        """
        Fit to target inclusion probabilities via L-BFGS.

        Parameters
        ----------
        target_incl : (N,) array with entries in (0, 1) summing to n
        n       : subset size
        tol     : convergence threshold on max|pi* - pi|
        max_iter: maximum L-BFGS iterations
        verbose : print iteration progress

        Returns
        -------
        ConditionalPoissonNumPy with weights fit to match target_incl.
        """
        obj = cls(n, np.zeros(len(target_incl)))
        obj.fit_inplace(target_incl, tol=tol, max_iter=max_iter, verbose=verbose)
        return obj

    # ── Internal: scaled polynomial arithmetic ─────────────────────────────────

    def _scale(self, c):
        """Normalize so max(c) = 1, return (normalized_coeffs, log_scale)."""
        m = np.max(c)
        if m == 0:
            return c, -np.inf
        return c / m, np.log(m)

    def _pmul(self, a, als, b, bls):
        """Multiply two scaled polynomials. Returns (c, cls)."""
        c = convolve(a, b)
        cn, inc = self._scale(c)
        return cn, als + bls + inc

    # ── Internal: P-tree (shared by pi, log_Z, sampling) ──────────────────────

    def _build_tree(self):
        """Build and cache the product tree and geometric-mean-scaled weights.

        The tree is a segment tree: Pc[i] is the polynomial at node i,
        Pls[i] is its log scale. tree_n is the leaf offset (power of 2 >= N).
        """
        if "tree" not in self._cache:
            N, n = self.N, self.n
            log_gm = float(np.mean(self.theta))
            q_s = np.exp(self.theta - log_gm)

            tree_n = 1
            while tree_n < N:
                tree_n <<= 1
            Pc  = [None] * (2 * tree_n)
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

            self._cache["tree"] = (Pc, Pls, tree_n, q_s, log_gm)
        return self._cache["tree"]

    # ── Internal: backward (downward) pass for inclusion probabilities ────────

    def _compute_pi(self):
        """Downward pass: propagate outside polynomials to every leaf.

        oP[leaf i] = prod_{j != i} (1 + q_j z)
        pi_i = q_i * oP[i][n-1] / root[n] * exp(scale correction)
        """
        if "pi" not in self._cache:
            Pc, Pls, tree_n, q_s, _ = self._build_tree()
            N, n = self.N, self.n

            N2 = 2 * tree_n
            oPc  = [None] * N2
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

            self._cache["pi"] = pi

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def incl_prob(self) -> np.ndarray:
        """Inclusion probabilities pi_i = P(i in S).  O(N (log N)^2), cached."""
        self._compute_pi()
        return self._cache["pi"].copy()

    @property
    def log_normalizer(self) -> float:
        """Log normalizing constant.  O(N (log N)^2), cached."""
        Pc, Pls, _, _, log_gm = self._build_tree()
        n = self.n
        en_n = float(Pc[1][n]) if len(Pc[1]) > n else 0.0
        root_ls = float(Pls[1])
        return (np.log(en_n) + root_ls + n * log_gm) if en_n != 0.0 else -np.inf

    # ── Log probability ───────────────────────────────────────────────────────

    def log_prob(self, S: Union[np.ndarray, list]) -> Union[float, np.ndarray]:
        """
        Log-probability of one subset or a batch.

            log P(S) = sum_{i in S} theta_i  -  log Z

        S may be:
          int array (n,)     single subset (item indices)
          bool array (N,)    single subset (indicator)
          int array (M, n)   batch of M subsets
          bool array (M, N)  batch of M subsets
        """
        S   = np.asarray(S)
        lz  = self.log_normalizer
        th  = self.theta
        if S.dtype == bool:
            return (th @ S.T - lz) if S.ndim == 2 else float(th[S].sum() - lz)
        else:
            return (th[S].sum(axis=1) - lz) if S.ndim == 2 else float(th[S].sum() - lz)

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample(self) -> np.ndarray:
        """Draw one sample via top-down quota splitting on the product tree.

        At each internal node with quota k, sample the left quota j
        from pmf[j] = L[j] * R[k-j].

        Complexity: O(N (log N)^2) to build tree [cached] + O(n log N).
        """
        import random
        if "sample_tree" not in self._cache:
            Pc, Pls, tree_n, _, _ = self._build_tree()
            # Per-node scale ratio: sum L[j]*R[k-j] = P[k] * ratio[node]
            ratio = [1.0] * (2 * tree_n)
            for i in range(1, tree_n):
                ratio[i] = np.exp(Pls[i] - Pls[2*i] - Pls[2*i+1])
            self._cache["sample_tree"] = (
                [p.tolist() if p is not None else [] for p in Pc],
                ratio, tree_n,
            )
        Pc, ratio, tree_n = self._cache["sample_tree"]
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

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit_inplace(
        self,
        target_incl: np.ndarray,
        *,
        tol: float = 1e-10,
        max_iter: int = 200,
        verbose: bool = False,
    ) -> "ConditionalPoissonNumPy":
        """
        Update weights to match target inclusion probabilities pi*.

        Minimizes -E_π*[log P_θ(S)] = -(π*ᵀθ - log Z(θ, n)) via L-BFGS.
        The gradient is π(θ) - π*, so convergence means max|π - π*| ≤ tol.
        Log-weights are zero-centered on completion (shift-invariant distribution).

        Returns self (for chaining).
        """
        from scipy.optimize import minimize

        target_incl = np.asarray(target_incl, float)
        if len(target_incl) != self.N:
            raise ValueError(f"len(target_incl)={len(target_incl)} != N={self.N}")
        if abs(target_incl.sum() / self.n - 1.0) > 1e-6:
            raise ValueError(f"sum(target_incl)/n = {target_incl.sum()/self.n:.8f}, expected 1.0")
        if not np.all((target_incl > 0) & (target_incl < 1)):
            raise ValueError("all target_incl must lie strictly in (0, 1)")

        # Warm start: logit(pi*)
        from scipy.special import logit
        theta0 = logit(target_incl)

        iter_count = [0]

        def neg_ll_and_grad(theta):
            self.theta = theta
            self._cache.clear()
            pi = self.incl_prob
            loss = self.log_normalizer - float(np.dot(target_incl, theta))
            grad = pi - target_incl
            if verbose:
                err = float(np.max(np.abs(target_incl - pi)))
                print(f"  iter {iter_count[0]:3d}:  max|pi*-pi| = {err:.3e}")
            iter_count[0] += 1
            return loss, grad

        result = minimize(
            neg_ll_and_grad, theta0,
            method='L-BFGS-B', jac=True,
            options={'maxiter': max_iter, 'gtol': tol, 'ftol': 0},
        )

        theta = result.x
        theta -= theta.mean()   # zero-center (shift-invariant)
        self.theta = theta
        self._cache.clear()
        return self

    def __repr__(self) -> str:
        if "log_Z" in self._cache:
            return (f"ConditionalPoissonNumPy(N={self.N}, n={self.n}, "
                    f"log_normalizer={self._cache['log_Z']:.3f})")
        return f"ConditionalPoissonNumPy(N={self.N}, n={self.n})"
