"""
conditional_poisson_numpy.py
======================

Conditional Poisson distribution over fixed-size subsets.

    P(S) ∝ prod_{i in S} w_i,   |S| = n

where w_i are non-negative weights (zero and infinity allowed).  Internally parameterised by
log-weights theta_i = log(w_i).

Everything — forward pass, gradient, Hessian-vector product, and sampling —
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
  pi / log_normalizer    O(N (log N)^2)  [cached]
  hvp(v)                 O(N (log N)^2)  [P-tree cached; D-tree + downward fresh]
  sample(M)              O(N (log N)^2 + M n log N)
  fit(pi_star)           O(N (log N)^2 * Newton_iters * CG_iters)

TODO: truncate polynomials to degree n throughout the tree to achieve
O(N log^2 n) instead of O(N log^2 N).  This requires per-coefficient
scaling to avoid underflow when the degree-n coefficient is much smaller
than the peak coefficient at degree ~subtree_size/2.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Union
from scipy.signal import convolve

__all__ = ["ConditionalPoisson"]


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

def _padd(a, als, b, bls):
    """Add two scaled polynomials (possibly different lengths). Returns (c, cls)."""
    if als == -np.inf:
        return b.copy(), bls
    if bls == -np.inf:
        return a.copy(), als
    s = max(als, bls)
    la, lb = len(a), len(b)
    if la >= lb:
        out = a * np.exp(als - s)
        out[:lb] += b * np.exp(bls - s)
    else:
        out = b * np.exp(bls - s)
        out[:la] += a * np.exp(als - s)
    cn, inc = _scale(out)
    return cn, s + inc


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


# ── Upward pass: D-tree (for HVP; requires P-tree already built) ─────────────

def _build_d_tree(Pc, Pls, q_s, v, S):
    """
    Build D-tree: D_T(z) = sum_{i in T} q_i v_i prod_{j in T, j!=i}(1+q_j z).

    D[node] = convolve(D[L], P[R]) + convolve(P[L], D[R])   (product rule)

    Complexity: O(N (log N)^2).
    """
    N = len(q_s)
    Dc  = [None] * (2 * S)
    Dls = np.full(2 * S, -np.inf)

    for i in range(S):
        d_val = q_s[i] * v[i] if i < N else 0.0
        if d_val == 0.0:
            Dc[S + i] = np.zeros(1)
            Dls[S + i] = -np.inf
        else:
            sign = 1.0 if d_val > 0 else -1.0
            Dc[S + i] = np.array([sign])
            Dls[S + i] = np.log(abs(d_val))

    for i in range(S - 1, 0, -1):
        l, r = 2 * i, 2 * i + 1
        t1c, t1s = _pmul(Dc[l], Dls[l], Pc[r], Pls[r])
        t2c, t2s = _pmul(Pc[l], Pls[l], Dc[r], Dls[r])
        Dc[i], Dls[i] = _padd(t1c, t1s, t2c, t2s)

    return Dc, Dls


# ── Downward pass ─────────────────────────────────────────────────────────────

def _downward_pass(Pc, Pls, Dc, Dls, S):
    """
    Top-down pass propagating outside polynomials to every leaf.

    Pass Dc=None, Dls=None to skip D (pi-only mode).

    At leaf i on return:
      oPc[S+i], oPls[S+i]  encodes  P^{(-i)}(z)
      oDc[S+i], oDls[S+i]  encodes  D^{(-i)}(z)  (if D requested)

    Update rule (child c with sibling s):
      oP[c] = oP[parent] * P[s]
      oD[c] = oD[parent] * P[s]  +  oP[parent] * D[s]

    Complexity: O(N (log N)^2).
    """
    do_d = (Dc is not None)
    N2 = 2 * S

    oPc  = [None] * N2;  oPls = np.full(N2, -np.inf)
    oDc  = [None] * N2;  oDls = np.full(N2, -np.inf)

    oPc[1] = np.array([1.0]); oPls[1] = 0.0
    if do_d:
        oDc[1] = np.zeros(1); oDls[1] = -np.inf

    for i in range(1, S):
        if oPc[i] is None:
            continue
        l, r = 2 * i, 2 * i + 1
        for c, s in ((l, r), (r, l)):
            oPc[c], oPls[c] = _pmul(oPc[i], oPls[i], Pc[s], Pls[s])
            if do_d:
                t1c, t1s = _pmul(oDc[i], oDls[i], Pc[s], Pls[s])
                t2c, t2s = _pmul(oPc[i], oPls[i], Dc[s], Dls[s])
                oDc[c], oDls[c] = _padd(t1c, t1s, t2c, t2s)

    return oPc, oPls, oDc, oDls


# ── Extraction: pi, log_Z, and (optionally) Hv ───────────────────────────────

def _extract(q_s, log_gm, Pc, Pls, oPc, oPls, oDc, oDls, S, N, n, v):
    """
    Extract pi, Hv (optional), and log_Z from tree + downward-pass results.

    pi_i   = q_s[i] * oPc[S+i][n-1] / Pc[1][n]  * exp(oPls[S+i] - Pls[1])
    log Z  = log|Pc[1][n]| + Pls[1] + n * log_gm
    """
    en_n    = float(Pc[1][n]) if len(Pc[1]) > n else 0.0
    root_ls = float(Pls[1])
    log_Z   = (np.log(abs(en_n)) + root_ls + n * log_gm) if en_n != 0.0 else -np.inf

    do_hv = (oDc is not None) and (v is not None)
    pi   = np.zeros(N)
    SumZ = np.zeros(N) if do_hv else None

    for i in range(N):
        op = oPc[S + i]
        op_ls = float(oPls[S + i])
        if en_n != 0.0 and op is not None and len(op) > n - 1:
            pi[i] = q_s[i] * float(op[n - 1]) / en_n * np.exp(op_ls - root_ls)
        if do_hv and n >= 2:
            od = oDc[S + i]
            od_ls = float(oDls[S + i])
            if en_n != 0.0 and od is not None and len(od) > n - 2:
                SumZ[i] = q_s[i] * float(od[n - 2]) / en_n * np.exp(od_ls - root_ls)

    if do_hv:
        alpha = float(np.dot(pi, v))
        Hv = SumZ + pi * v - pi * alpha
    else:
        Hv = None

    return pi, Hv, log_Z


# ── Sampler: divide-and-conquer on the P-tree ─────────────────────────────────

def _tree_sample(Pc, S, N, n, M, rng):
    """
    Sample M subsets of size n using the cached P-tree.

    At each internal node with quota k, the split distribution is:

      P(j items from L | k) ∝ Pc_L[j] * Pc_R[k-j],   j = 0..k

    The log-scale factors cancel in the ratio, so only the normalized
    coefficient arrays Pc are needed.

    Processes each tree level as a batch to minimise Python loop overhead.

    Complexity: O(n log N) per sample, O(M n log N) total.
    """
    quotas = np.zeros((2 * S, M), dtype=np.int32)
    quotas[1] = n

    depth = int(np.log2(S))  # number of levels (S is a power of 2)
    for d in range(depth):
        level_start = 1 << d        # first node at this depth
        level_end   = 1 << (d + 1)  # one past last node at this depth

        # ── Fast path: levels where max quota ≤ 1 (Bernoulli decisions) ──
        # Check if all nodes at this level have max_k ≤ 1
        level_quotas = quotas[level_start:level_end]   # (num_nodes, M)
        global_max_k = int(level_quotas.max())

        if global_max_k == 0:
            # All quotas are 0 — nothing to split
            quotas[level_start * 2 : level_end * 2] = 0
            continue

        if global_max_k == 1:
            # All decisions are Bernoulli: include in left child or right?
            # prob(j=1|k=1) = Pl[1]*Pr[0] / (Pl[1]*Pr[0] + Pl[0]*Pr[1])
            num_nodes = level_end - level_start
            probs = np.zeros(num_nodes)
            for idx in range(num_nodes):
                node = level_start + idx
                Pl, Pr = Pc[2 * node], Pc[2 * node + 1]
                w1 = max(Pl[1], 0.0) * max(Pr[0], 0.0) if len(Pl) > 1 else 0.0
                w0 = max(Pl[0], 0.0) * (max(Pr[1], 0.0) if len(Pr) > 1 else 0.0)
                total = w0 + w1
                probs[idx] = w1 / total if total > 0 else 0.5

            # Vectorised Bernoulli across all nodes × all samples
            u = rng.random((num_nodes, M))
            is_k1 = level_quotas == 1  # (num_nodes, M)
            j_left = (is_k1 & (u < probs[:, None])).astype(np.int32)

            # Batch assignment to children
            quotas[level_start * 2 : level_end * 2 : 2] = j_left
            quotas[level_start * 2 + 1 : level_end * 2 + 1 : 2] = (
                level_quotas - j_left
            )
            continue

        # ── General case: per-node CDF sampling ──
        for node in range(level_start, level_end):
            k_arr = quotas[node]
            Pl = Pc[2 * node]
            Pr = Pc[2 * node + 1]
            max_l = len(Pl)
            max_r = len(Pr)
            j_arr = np.zeros(M, dtype=np.int32)

            max_k = int(k_arr.max())
            if max_k == 0:
                quotas[2 * node] = 0
                quotas[2 * node + 1] = 0
                continue

            # Precompute CDF for each possible quota k = 1..max_k
            cdf = np.zeros((max_k + 1, max_k + 1))
            for k in range(1, max_k + 1):
                km = min(k + 1, max_l)
                jv = np.arange(km)
                kv = k - jv
                valid = kv < max_r
                if not valid.any():
                    continue
                jv_v = jv[valid]
                kv_v = kv[valid]
                w = np.maximum(Pl[jv_v], 0.0) * np.maximum(Pr[kv_v], 0.0)
                cdf[k, jv_v] = w
            np.cumsum(cdf, axis=1, out=cdf)
            row_totals = cdf[:, -1].copy()
            row_totals[row_totals == 0] = 1.0
            cdf /= row_totals[:, None]

            u = rng.random(M)
            active = k_arr > 0
            if active.any():
                active_idx = np.where(active)[0]
                ki = k_arr[active_idx]
                sample_cdfs = cdf[ki]
                j_arr[active_idx] = np.argmax(
                    sample_cdfs >= u[active_idx, None], axis=1,
                )

            quotas[2 * node] = j_arr
            quotas[2 * node + 1] = k_arr - j_arr

    # Collect leaf decisions into (M, n) output
    # leaf_quotas[i, m] > 0 means item i is included in sample m
    leaf_quotas = quotas[S : S + N]          # (N, M)
    included = leaf_quotas.T > 0             # (M, N) bool
    # Each row of `included` has exactly n True entries.
    # np.where gives (sample_idx, item_idx) pairs sorted by sample then item.
    sample_idx, item_idx = np.where(included)
    out = item_idx.astype(np.int32).reshape(M, n)

    return out


# ── CG solver ─────────────────────────────────────────────────────────────────

def _cg(matvec, b, tol=1e-12, max_iter=None):
    n = len(b); max_iter = max_iter or min(n, 500)
    x, r, p = np.zeros(n), b.copy(), b.copy()
    rr = float(np.dot(r, r))
    for _ in range(max_iter):
        if rr < tol ** 2: break
        Ap  = matvec(p);  pAp = float(np.dot(p, Ap))
        if pAp <= 0: break
        a   = rr / pAp;  x += a * p;  r -= a * Ap
        rr2 = float(np.dot(r, r))
        p   = r + (rr2 / rr) * p;  rr = rr2
    return x


# ══ ConditionalPoisson ════════════════════════════════════════════════════════

class ConditionalPoisson:
    """
    Conditional Poisson distribution over fixed-size subsets.

        P(S) ∝ prod_{i in S} w_i,   |S| = n

    Construction
    ------------
    ConditionalPoisson(n, theta)           direct from log-weights theta = log(w)
    ConditionalPoisson.from_weights(n, w)  from non-negative weights w_i
    ConditionalPoisson.fit(pi_star, n)     moment-match to target probs

    Properties  (all cached; cache invalidated when theta changes)
    ----------
    pi              (N,) inclusion probabilities
    w               (N,) weights (= exp(theta))
    log_normalizer  log normalizing constant  — never overflows

    Methods
    -------
    log_prob(S)         scalar or (M,) log-probabilities
    sample(M, rng)      (M, n) int array of sorted subsets
    hvp(v)              Cov[Z] v  (positive semi-definite, null-space = span{1})
    fit_inplace(pi_star)  update theta in-place; returns self

    Notes
    -----
    - The P-tree is built once on first access and shared by pi, log_normalizer,
      and sample. The D-tree is rebuilt per hvp(v) call (depends on v).
    - theta is zero-centered after fitting (shift-invariant distribution).
    """

    def __init__(self, n: int, theta: np.ndarray):
        theta = np.asarray(theta, float)
        if theta.ndim != 1:   raise ValueError("theta must be 1-D")
        if len(theta) < n:    raise ValueError(f"N={len(theta)} must be >= n={n}")
        if n < 1:             raise ValueError("n must be >= 1")

        forced_in  = np.where(theta == np.inf)[0]
        forced_out = np.where(theta == -np.inf)[0]
        interior   = np.where(np.isfinite(theta))[0]

        n_in = len(forced_in)
        n_red = n - n_in
        if n_in > n:
            raise ValueError(f"{n_in} items have w=inf but n={n}")
        if n_red > len(interior):
            raise ValueError(
                f"Need {n_red} items from {len(interior)} interior items "
                f"(after {n_in} forced-in), impossible"
            )

        self._n     = int(n)
        self._theta = theta.copy()
        self._cache: dict = {}

        # Boundary handling by reduction: strip forced-in/out items,
        # delegate to a reduced instance on interior items with adjusted n.
        self._forced_in  = forced_in
        self._forced_out = forced_out
        self._interior   = interior
        self._reduced: Optional["ConditionalPoisson"] = None

        if len(forced_in) > 0 or len(forced_out) > 0:
            if n_red > 0 and len(interior) > 0:
                self._reduced = ConditionalPoisson(n_red, theta[interior])
            # else: fully degenerate (all items determined), no reduced instance

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_weights(cls, n: int, w: np.ndarray) -> "ConditionalPoisson":
        """Construct from non-negative weights w_i (0 and inf allowed)."""
        w = np.asarray(w, float)
        if np.any(w < 0): raise ValueError("all weights must be non-negative")
        with np.errstate(divide='ignore'):
            return cls(n, np.log(w))

    @classmethod
    def fit(
        cls,
        pi_star: np.ndarray,
        n: int,
        *,
        tol: float = 1e-10,
        max_iter: int = 50,
        verbose: bool = False,
    ) -> "ConditionalPoisson":
        """
        Fit to target inclusion probabilities.

        Parameters
        ----------
        pi_star : (N,) array with entries in (0, 1) summing to n
        n       : subset size
        tol     : convergence threshold on max|pi* - pi|
        max_iter: maximum Newton iterations
        verbose : print iteration progress

        Returns
        -------
        ConditionalPoisson with weights fit to match pi_star.
        """
        obj = cls(n, np.zeros(len(pi_star)))
        obj.fit_inplace(pi_star, tol=tol, max_iter=max_iter, verbose=verbose)
        return obj

    # ── Parameters ───────────────────────────────────────────────────────────

    @property
    def n(self) -> int:   return self._n
    @property
    def N(self) -> int:   return len(self._theta)
    @property
    def theta(self) -> np.ndarray: return self._theta.copy()
    @property
    def w(self) -> np.ndarray:
        """Weights w_i = exp(theta_i)."""
        return np.exp(self._theta)

    @theta.setter
    def theta(self, value):
        value = np.asarray(value, float)
        if len(value) != self.N:
            raise ValueError(f"len(theta)={len(value)} != N={self.N}")
        # Re-initialize to re-partition boundary items
        self.__init__(self._n, value)

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
            oPc, oPls, _, _ = _downward_pass(Pc, Pls, None, None, S)
            pi, _, log_Z = _extract(q_s, log_gm, Pc, Pls, oPc, oPls,
                                    None, None, S, self.N, self._n, None)
            self._cache["pi"]    = pi
            self._cache["log_Z"] = log_Z

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def pi(self) -> np.ndarray:
        """Inclusion probabilities pi_i = P(i in S).  O(N (log N)^2), cached."""
        if self._reduced is not None:
            pi = np.zeros(self.N)
            pi[self._forced_in] = 1.0
            pi[self._interior] = self._reduced.pi
            return pi
        if len(self._forced_in) > 0:
            # Fully degenerate: all forced-in, rest forced-out
            pi = np.zeros(self.N)
            pi[self._forced_in] = 1.0
            return pi
        self._forward(); return self._cache["pi"].copy()

    @property
    def log_normalizer(self) -> float:
        """Log normalizing constant.  Never overflows.  O(N (log N)^2), cached."""
        if self._reduced is not None:
            return self._reduced.log_normalizer
        if len(self._forced_in) > 0:
            return 0.0  # only one possible subset
        self._forward(); return self._cache["log_Z"]

    # ── Log probability ───────────────────────────────────────────────────────

    def log_prob(self, S: Union[np.ndarray, list]) -> Union[float, np.ndarray]:
        """
        Log-probability of one subset or a batch.

            log P(S) = sum_{i in S} log(w_i)  -  log Z

        Returns -inf for impossible subsets (containing a w=0 item or
        missing a w=inf item).

        S may be:
          int array (n,)     single subset (item indices)
          bool array (N,)    single subset (indicator)
          int array (M, n)   batch of M subsets
          bool array (M, N)  batch of M subsets
        """
        if self._reduced is None and len(self._forced_in) == 0:
            # No boundary items — fast path using theta directly
            S   = np.asarray(S)
            lz  = self.log_normalizer
            th  = self._theta
            if S.dtype == bool:
                return (th @ S.T - lz) if S.ndim == 2 else float(th[S].sum() - lz)
            else:
                return (th[S].sum(axis=1) - lz) if S.ndim == 2 else float(th[S].sum() - lz)

        # Has boundary items — convert to bool indicator, validate, reduce.
        S = np.asarray(S)
        single = S.ndim == 1
        if S.dtype == bool:
            indicators = S if S.ndim == 2 else S[None, :]
        else:
            idx = S if S.ndim == 2 else S[None, :]
            indicators = np.zeros((idx.shape[0], self.N), dtype=bool)
            for m in range(idx.shape[0]):
                indicators[m, idx[m]] = True

        results = np.full(indicators.shape[0], -np.inf)
        for m in range(indicators.shape[0]):
            ind = indicators[m]
            # Check boundary: forced-in must be in S, forced-out must not
            if not np.all(ind[self._forced_in]):
                continue  # -inf
            if np.any(ind[self._forced_out]):
                continue  # -inf
            if self._reduced is None:
                results[m] = 0.0  # degenerate: only one valid subset
            else:
                int_ind = ind[self._interior]
                results[m] = self._reduced.log_prob(int_ind)

        return float(results[0]) if single else results

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample(
        self,
        size: int = 1,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> np.ndarray:
        """
        Draw independent samples using the cached P-tree.

        Parameters
        ----------
        size : number of subsets to draw
        rng  : seed or np.random.Generator

        Returns
        -------
        (size, n) int array; each row is a sorted list of n item indices.

        Complexity: O(N (log N)^2) to build tree [cached] + O(size * n * log N).
        """
        rng = np.random.default_rng(rng)
        if self._reduced is not None:
            # Sample from reduced problem, map indices back, merge with forced-in
            int_samples = self._reduced.sample(size, rng)
            int_samples = self._interior[int_samples]  # remap to full indices
            forced_tile = np.tile(self._forced_in, (size, 1))
            merged = np.concatenate([forced_tile, int_samples], axis=1)
            merged.sort(axis=1)
            return merged
        if len(self._forced_in) > 0:
            return np.tile(np.sort(self._forced_in), (size, 1))
        Pc, Pls, S, q_s, _ = self._get_p_tree()
        return _tree_sample(Pc, S, self.N, self._n, size, rng)

    # ── Hessian-vector product ────────────────────────────────────────────────

    def hvp(self, v: np.ndarray) -> np.ndarray:
        """
        Compute Cov[1_S] v.  O(N (log N)^2).

        Cov[1_S] is the covariance matrix of the inclusion indicators.
        It is positive semi-definite with rank N-1; null space = span{1}
        since sum(1_{i in S}) = n is constant.

        Uses the cached P-tree; rebuilds D-tree (which depends on v).
        """
        v = np.asarray(v, float)
        if self._reduced is not None:
            Hv = np.zeros(self.N)
            Hv[self._interior] = self._reduced.hvp(v[self._interior])
            return Hv
        if len(self._forced_in) > 0:
            return np.zeros(self.N)
        Pc, Pls, S, q_s, log_gm = self._get_p_tree()
        Dc, Dls = _build_d_tree(Pc, Pls, q_s, v, S)
        oPc, oPls, oDc, oDls = _downward_pass(Pc, Pls, Dc, Dls, S)
        _, Hv, _ = _extract(q_s, log_gm, Pc, Pls, oPc, oPls, oDc, oDls,
                            S, self.N, self._n, v)
        return Hv

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit_inplace(
        self,
        pi_star: np.ndarray,
        *,
        tol: float = 1e-10,
        max_iter: int = 50,
        verbose: bool = False,
    ) -> "ConditionalPoisson":
        """
        Update weights to match target inclusion probabilities pi*.

        Maximizes the log-likelihood via Newton-CG with Armijo backtracking.
        Log-weights are zero-centered on completion (shift-invariant distribution).

        Returns self (for chaining).
        """
        pi_star = np.asarray(pi_star, float)
        if len(pi_star) != self.N:
            raise ValueError(f"len(pi_star)={len(pi_star)} != N={self.N}")
        if abs(pi_star.sum() / self._n - 1.0) > 1e-6:
            raise ValueError(f"sum(pi_star)/n = {pi_star.sum()/self._n:.8f}, expected 1.0")
        if not np.all((pi_star > 0) & (pi_star < 1)):
            raise ValueError("all pi_star must lie strictly in (0, 1)")

        # Warm start: logit(pi*) = log(pi*/(1-pi*)).  This is the Poisson
        # sampling odds — asymptotically exact by Hájek (1964), Thm 5.2:
        # the CPS inclusion probs satisfy pi_i = p_i + O(1/N) where
        # p_i = w_i/(1+w_i) are the unconditional Poisson probs.
        theta = np.log(pi_star / (1.0 - pi_star))

        converged = False
        for it in range(max_iter):
            self.theta  = theta          # sets theta, clears cache
            pi          = self._cache.get("pi") or self.pi
            log_Z       = self._cache["log_Z"]
            grad        = pi_star - pi
            err         = float(np.max(np.abs(grad)))
            if verbose:
                print(f"  iter {it:3d}:  max|pi*-pi| = {err:.3e}")
            if err < tol:
                converged = True
                break

            delta = _cg(self.hvp, grad)

            slope = float(np.dot(grad, delta))
            L0    = float(np.dot(pi_star, theta)) - log_Z
            step  = 1.0
            for _ in range(20):
                th_new        = theta + step * delta
                tmp           = ConditionalPoisson(self._n, th_new)
                L_new         = float(np.dot(pi_star, th_new)) - tmp.log_normalizer
                if L_new >= L0 + 1e-4 * step * slope: break
                step         *= 0.5

            theta += step * delta

        if not converged:
            import warnings
            warnings.warn(
                f"fit did not converge after {max_iter} iterations "
                f"(max|pi*-pi| = {err:.3e}, tol = {tol:.3e}). "
                f"Increase max_iter or loosen tol.",
                stacklevel=3,
            )

        theta      -= theta.mean()   # zero-center (shift-invariant)
        self.theta  = theta
        return self

    def __repr__(self) -> str:
        if "log_Z" in self._cache:
            return (f"ConditionalPoisson(N={self.N}, n={self._n}, "
                    f"log_normalizer={self._cache['log_Z']:.3f})")
        return f"ConditionalPoisson(N={self.N}, n={self._n})"
