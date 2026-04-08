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

  _build_p_tree          O(N (log N)^2)
  _downward_pass         O(N (log N)^2)
  incl_prob / log_normalizer    O(N (log N)^2)  [cached]
  sample(M)              O(N (log N)^2 + M n log N)
  fit(target_incl)           O(N (log N)^2 * L-BFGS_iters)

TODO: truncate polynomials to degree n throughout the tree to achieve
O(N log^2 n) instead of O(N log^2 N).  This requires per-coefficient
scaling to avoid underflow when the degree-n coefficient is much smaller
than the peak coefficient at degree ~subtree_size/2.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Union
from scipy.signal import convolve

__all__ = ["ConditionalPoissonNumPy"]


# ── Scaled polynomial arithmetic ─────────────────────────────────────────────
#
# Each polynomial is a pair (c, ls) where the true polynomial is c * exp(ls)
# and max|c| = 1.  Unlike the previous implementation, polynomials are NOT
# truncated — they keep their full natural degree.

def _scale(c):
    """Normalize so max|c| = 1, return (normalized_coeffs, log_scale)."""
    m = np.max(np.abs(c))
    if m == 0:
        return c, -np.inf
    return c / m, np.log(m)

def _pmul(a, als, b, bls):
    """Multiply two scaled polynomials. Returns (c, cls)."""
    c = convolve(a, b)
    cn, inc = _scale(c)
    return cn, als + bls + inc


# ── Upward pass: P-tree ──────────────────────────────────────────────────────

def _build_p_tree(q_s):
    """
    Build product tree for P_T(z) = prod_{i in T}(1 + q_i z).

    Each node stores a scaled polynomial (c, ls) for its subtree.
    No degree truncation.  Uses scipy.signal.convolve (auto FFT/direct).

    Returns (Pc, Pls, S) where Pc[i], Pls[i] is the scaled poly at node i,
    S is the leaf offset (power of 2 >= N).

    Complexity: O(N (log N)^2).
    """
    N = len(q_s)
    S = 1
    while S < N:
        S <<= 1
    Pc  = [None] * (2 * S)
    Pls = np.full(2 * S, -np.inf)

    for i in range(S):
        if i < N:
            c = np.array([1.0, q_s[i]])
            Pc[S + i], Pls[S + i] = _scale(c)
        else:
            Pc[S + i] = np.array([1.0])
            Pls[S + i] = 0.0   # identity

    for i in range(S - 1, 0, -1):
        l, r = 2 * i, 2 * i + 1
        Pc[i], Pls[i] = _pmul(Pc[l], Pls[l], Pc[r], Pls[r])

    return Pc, Pls, S


# ── Downward pass ─────────────────────────────────────────────────────────────

def _downward_pass(Pc, Pls, S):
    """
    Top-down pass propagating outside polynomials to every leaf.

    At leaf i on return:
      oPc[S+i], oPls[S+i]  encodes  P^{(-i)}(z)

    Update rule (child c with sibling s):
      oP[c] = oP[parent] * P[s]

    Complexity: O(N (log N)^2).
    """
    N2 = 2 * S
    oPc  = [None] * N2
    oPls = np.full(N2, -np.inf)
    oPc[1] = np.array([1.0])
    oPls[1] = 0.0

    for i in range(1, S):
        if oPc[i] is None:
            continue
        l, r = 2 * i, 2 * i + 1
        for c, s in ((l, r), (r, l)):
            oPc[c], oPls[c] = _pmul(oPc[i], oPls[i], Pc[s], Pls[s])

    return oPc, oPls


# ── Extraction: pi, log_Z ────────────────────────────────────────────────────

def _extract(q_s, log_gm, Pc, Pls, oPc, oPls, S, N, n):
    """
    Extract pi and log_Z from tree + downward-pass results.

    pi_i   = q_s[i] * oPc[S+i][n-1] / Pc[1][n]  * exp(oPls[S+i] - Pls[1])
    log Z  = log|Pc[1][n]| + Pls[1] + n * log_gm
    """
    en_n    = float(Pc[1][n]) if len(Pc[1]) > n else 0.0
    root_ls = float(Pls[1])
    log_Z   = (np.log(abs(en_n)) + root_ls + n * log_gm) if en_n != 0.0 else -np.inf

    pi = np.zeros(N)
    for i in range(N):
        op = oPc[S + i]
        op_ls = float(oPls[S + i])
        if en_n != 0.0 and op is not None and len(op) > n - 1:
            pi[i] = q_s[i] * float(op[n - 1]) / en_n * np.exp(op_ls - root_ls)

    return pi, log_Z


# ── Sampler: divide-and-conquer on the P-tree ─────────────────────────────────

def _build_sample_cdfs(Pc, S, n):
    """Precompute normalized CDFs for every (node, quota) pair.

    For each internal node with children L, R and quota k, the split
    distribution has PMF  pmf[j] = L[j] * R[k-j].  The normalizer is
    the parent's k-th coefficient (the convolution sum), which is already
    stored in the tree (up to a scale factor that cancels).

    Returns cdfs: list where cdfs[node][k] is a list of cumulative
    probabilities for splitting quota k at that node, or None if
    that quota is impossible.
    """
    cdfs = [None] * (2 * S)
    for node in range(1, S):
        La = np.maximum(np.asarray(Pc[2 * node]), 0.0)
        Ra = np.maximum(np.asarray(Pc[2 * node + 1]), 0.0)
        max_k = min(n, len(La) - 1 + len(Ra) - 1)
        # Pad to length max_k+1 so indexing is unconditional
        Lp = np.pad(La, (0, max(0, max_k + 1 - len(La))))
        Rp = np.pad(Ra, (0, max(0, max_k + 1 - len(Ra))))
        node_cdfs = [None] * (max_k + 1)
        for k in range(1, max_k + 1):
            # PMF of split distribution: pmf[j] = L[j] * R[k-j]
            pmf = Lp[:k+1] * Rp[k::-1]
            total = pmf.sum()
            if total > 0:
                np.cumsum(pmf, out=pmf)
                pmf /= total
                node_cdfs[k] = pmf
        cdfs[node] = node_cdfs
    return cdfs


def _tree_sample(cdfs, S, N, n, rng):
    """Draw one sample via top-down quota splitting with precomputed CDFs.

    At each internal node with quota k, split k items between left and
    right children proportional to Pc_L[j] * Pc_R[k-j].

    Complexity: O(n log N).
    """
    selected = []
    stack = [(1, n)]
    while stack:
        node, k = stack.pop()
        if k == 0:
            continue
        if node >= S:
            if node - S < N:
                selected.append(node - S)
            continue
        cdf = cdfs[node][k]
        j = int(np.searchsorted(cdf, rng.random()))
        stack.append((2 * node + 1, k - j))
        stack.append((2 * node, j))
    selected.sort()
    return np.array(selected, dtype=np.int32)



# ══ ConditionalPoissonNumPy ════════════════════════════════════════════════════════

class ConditionalPoissonNumPy:
    """
    Conditional Poisson distribution over fixed-size subsets.

        P(S) ∝ prod_{i in S} w_i,   |S| = n

    Construction
    ------------
    ConditionalPoissonNumPy(n, theta)           direct from log-weights theta = log(w)
    ConditionalPoissonNumPy.from_weights(n, w)  from non-negative weights w_i
    ConditionalPoissonNumPy.fit(target_incl, n)     moment-match to target probs

    Properties  (all cached; cache invalidated when theta changes)
    ----------
    incl_prob       (N,) inclusion probabilities
    w               (N,) weights (= exp(theta))
    log_normalizer  log normalizing constant  — never overflows

    Methods
    -------
    log_prob(S)         scalar or (M,) log-probabilities
    sample(M, rng)      (M, n) int array of sorted subsets
    fit_inplace(target_incl)  update theta in-place via L-BFGS; returns self

    Notes
    -----
    - The P-tree is built once on first access and shared by pi, log_normalizer,
      and sample.
    - theta is zero-centered after fitting (shift-invariant distribution).
    """

    def __init__(self, n: int, theta: np.ndarray):
        theta = np.asarray(theta, float)
        if theta.ndim != 1:   raise ValueError("theta must be 1-D")
        if not np.all(np.isfinite(theta)):
            raise ValueError("all theta must be finite (no w=0 or w=inf)")
        assert 0 <= n <= len(theta), f"n={n} must be in [0, {len(theta)}]"

        self.n      = int(n)
        self.N      = len(theta)
        self._theta = theta.copy()
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

    # ── Parameters ───────────────────────────────────────────────────────────

    @property
    def theta(self) -> np.ndarray: return self._theta

    @theta.setter
    def theta(self, value):
        value = np.asarray(value, float)
        if len(value) != self.N:
            raise ValueError(f"len(theta)={len(value)} != N={self.N}")
        self.__init__(self.n, value)

    # ── Internal: P-tree (shared by pi, log_Z, sampling) ──────────────────────

    def _get_p_tree(self):
        """Build and cache the P-tree and normalized weights."""
        if "p_tree" not in self._cache:
            log_gm = float(np.mean(self._theta))
            q_s    = np.exp(self._theta - log_gm)
            Pc, Pls, S = _build_p_tree(q_s)
            self._cache["p_tree"] = (Pc, Pls, S, q_s, log_gm)
        return self._cache["p_tree"]

    # ── Internal: forward pass (pi, log_Z) ────────────────────────────────────

    def _forward(self):
        if "pi" not in self._cache:
            Pc, Pls, S, q_s, log_gm = self._get_p_tree()
            oPc, oPls = _downward_pass(Pc, Pls, S)
            pi, log_Z = _extract(q_s, log_gm, Pc, Pls, oPc, oPls,
                                 S, self.N, self.n)
            self._cache["pi"]    = pi
            self._cache["log_Z"] = log_Z

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def incl_prob(self) -> np.ndarray:
        """Inclusion probabilities pi_i = P(i in S).  O(N (log N)^2), cached."""
        self._forward(); return self._cache["pi"].copy()

    @property
    def log_normalizer(self) -> float:
        """Log normalizing constant.  Never overflows.  O(N (log N)^2), cached."""
        self._forward(); return self._cache["log_Z"]

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
        th  = self._theta
        if S.dtype == bool:
            return (th @ S.T - lz) if S.ndim == 2 else float(th[S].sum() - lz)
        else:
            return (th[S].sum(axis=1) - lz) if S.ndim == 2 else float(th[S].sum() - lz)

    # ── Sampling ──────────────────────────────────────────────────────────────

    def _get_sample_cdfs(self):
        """Build and cache the CDFs for sampling."""
        if "sample_cdfs" not in self._cache:
            Pc, _, S, _, _ = self._get_p_tree()
            self._cache["sample_cdfs"] = _build_sample_cdfs(Pc, S, self.n)
        return self._cache["sample_cdfs"]

    def sample(
        self,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> np.ndarray:
        """
        Draw one sample using the cached P-tree.

        Parameters
        ----------
        rng  : seed or np.random.Generator

        Returns
        -------
        (n,) int array of sorted item indices.

        Complexity: O(N (log N)^2) to build tree [cached] + O(n log N).
        """
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        _, _, S, _, _ = self._get_p_tree()
        cdfs = self._get_sample_cdfs()
        return _tree_sample(cdfs, S, self.N, self.n, rng)

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
            self.theta = theta  # clears cache
            pi = self.incl_prob
            log_Z = self._cache["log_Z"]
            loss = log_Z - float(np.dot(target_incl, theta))
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
        return self

    def __repr__(self) -> str:
        if "log_Z" in self._cache:
            return (f"ConditionalPoissonNumPy(N={self.N}, n={self.n}, "
                    f"log_normalizer={self._cache['log_Z']:.3f})")
        return f"ConditionalPoissonNumPy(N={self.N}, n={self.n})"
