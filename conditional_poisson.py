"""
conditional_poisson.py
======================

Conditional Poisson distribution over fixed-size subsets.

    P(S; theta) = exp(sum_{i in S} theta_i) / e_n(exp(theta)),   |S| = n

Everything — forward pass, gradient, Hessian-vector product, and sampling —
uses a single augmented polynomial product tree built once and cached.

Tree structure
--------------
Upward pass builds two ScaledPoly trees in one sweep:

  P_T(z)  = prod_{i in T}(1 + q_i z)                  [degree <= n]
  D_T(z)  = sum_{i in T} q_i v_i prod_{j in T, j!=i}(1+q_j z)   [degree <= n-1]

  P[node] = P[L] * P[R]
  D[node] = D[L]*P[R] + P[L]*D[R]    (product rule; D depends on v)

Downward pass propagates outside-(P,D) pairs to every leaf:

  oP[leaf i] = P^{(-i)}(z)           leave-one-out product poly
  oD[leaf i] = D^{(-i)}(z)           leave-one-out weighted-deriv poly

Extraction (at leaf i):
  pi_i     = q_s[i] * [z^{n-1}] oP_i / e_n(q_s) * exp(oP_ls - root_ls)
  SumZ_i   = q_s[i] * [z^{n-2}] oD_i / e_n(q_s) * exp(oD_ls - root_ls)
  (Cov v)_i = SumZ_i  +  pi_i v_i  -  pi_i (pi.v)

Sampling
--------
The P-tree (upward pass only, no D) is cached and reused for sampling.
At each internal node with quota k, sample the left/right quota split:

  P(j items from L | k, T) ∝ P_L[j] * P_R[k-j]       (Pls factors cancel)

Recurse down the tree; each sample costs O(n log N) in expectation.
No O(Nn) backward table is built or stored.

Numerics
--------
Every polynomial is a ScaledPoly (coeffs_norm, log_scale) with
max|coeffs_norm| = 1. All FFTs operate on O(1) numbers, preventing
float64 overflow and FFT rounding blowup. Geometric-mean normalisation
q -> q/exp(mean(log q)) is applied before every tree build; pi and Cov[Z]v
are invariant under this rescaling.

Complexity
----------
  _build_p_tree          O(N log^2 n)
  _build_d_tree          O(N log^2 n)
  _downward_pass         O(N log^2 n)
  pi / log_normalizer    O(N log^2 n)  [cached]
  hvp(v)                 O(N log^2 n)  [P-tree cached; D-tree + downward fresh]
  sample(M)              O(N log^2 n + M n log N)
  fit(pi_star)           O(N log^2 n * Newton_iters * CG_iters)
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Union

# ── Scaled polynomial arithmetic ─────────────────────────────────────────────
#
# ScaledPoly = (cn, ls) representing the polynomial cn * exp(ls),
# with invariant  max|cn[k]| == 1  (or cn all-zero and ls == -inf).
# Keeps all FFT inputs in [-1, 1]: no overflow, rounding stays O(eps).

_DIRECT = 48
_NEGINF = -np.inf

def _pmul_raw(a, b, d):
    if d < 0: return np.zeros(0)
    la, lb = len(a), len(b)
    if la == 0 or lb == 0: return np.zeros(d + 1)
    if la + lb - 2 <= _DIRECT:
        c = np.convolve(a, b)
    else:
        m  = la + lb - 1
        nf = 1 << (m - 1).bit_length()
        c  = np.fft.irfft(np.fft.rfft(a, nf) * np.fft.rfft(b, nf), nf).real
    out = np.zeros(d + 1)
    out[:min(len(c), d + 1)] = c[:d + 1]
    return out

def _pad(a, d):
    out = np.zeros(d + 1); out[:min(len(a), d + 1)] = a[:d + 1]; return out

def _scale(c, d):
    c = _pad(c, d); m = np.max(np.abs(c))
    if m == 0: return np.zeros(d + 1), _NEGINF
    return c / m, np.log(m)

def _pmul_s(an, als, bn, bls, d):
    cn, inc = _scale(_pmul_raw(an, bn, d), d)
    return cn, als + bls + inc

def _padd_s(an, als, bn, bls, d):
    if als == _NEGINF: return _pad(bn, d), bls
    if bls == _NEGINF: return _pad(an, d), als
    s = max(als, bls)
    c = _pad(an, d) * np.exp(als - s) + _pad(bn, d) * np.exp(bls - s)
    cn, inc = _scale(c, d); return cn, s + inc


# ── Upward pass: P-tree (for pi, log_en, sampling) ───────────────────────────

def _build_p_tree(q_s, n):
    """
    Build scaled product tree for P_T(z) = prod_{i in T}(1 + q_i z).

    Returns (Pn, Pls, S) where Pn[i], Pls[i] is the ScaledPoly at node i,
    S is the leaf offset (power of 2 >= N).

    Complexity: O(N log^2 n).
    """
    N = len(q_s); S = 1
    while S < N: S <<= 1
    Pn  = [None] * (2 * S)
    Pls = np.full(2 * S, _NEGINF)

    for i in range(S):
        if i < N:
            p = np.array([1.0, q_s[i]]) if n >= 1 else np.array([1.0])
            Pn[S+i], Pls[S+i] = _scale(p, min(1, n))
        else:
            Pn[S+i] = np.array([1.0]); Pls[S+i] = 0.0   # identity

    for i in range(S - 1, 0, -1):
        l, r = 2*i, 2*i+1
        Pn[i], Pls[i] = _pmul_s(Pn[l], Pls[l], Pn[r], Pls[r], n)

    return Pn, Pls, S


# ── Upward pass: D-tree (for HVP; requires P-tree already built) ─────────────

def _build_d_tree(Pn, Pls, q_s, v, n, S):
    """
    Build scaled D-tree: D_T(z) = sum_{i in T} q_i v_i prod_{j!=i}(1+q_j z).

    D[node] = D[L]*P[R] + P[L]*D[R]    (product rule)

    Requires the P-tree (Pn, Pls) already built for this q_s.

    Complexity: O(N log^2 n).
    """
    N = len(q_s)
    Dn  = [None] * (2 * S)
    Dls = np.full(2 * S, _NEGINF)

    for i in range(S):
        d_val = q_s[i] * v[i] if i < N else 0.0
        if d_val == 0.0:
            Dn[S+i] = np.zeros(1); Dls[S+i] = _NEGINF
        else:
            sign = 1.0 if d_val > 0 else -1.0
            Dn[S+i] = np.array([sign]); Dls[S+i] = np.log(abs(d_val))

    for i in range(S - 1, 0, -1):
        l, r = 2*i, 2*i+1
        t1n, t1s      = _pmul_s(Dn[l], Dls[l], Pn[r], Pls[r], n - 1)
        t2n, t2s      = _pmul_s(Pn[l], Pls[l], Dn[r], Dls[r], n - 1)
        Dn[i], Dls[i] = _padd_s(t1n, t1s, t2n, t2s, n - 1)

    return Dn, Dls


# ── Downward pass ─────────────────────────────────────────────────────────────

def _downward_pass(Pn, Pls, Dn, Dls, S, n):
    """
    Top-down pass propagating outside-(P, D) pairs to every leaf.

    Pass Dn=None, Dls=None to skip D (pi-only mode, for forward pass).

    At leaf i on return:
      oPn[S+i], oPls[S+i]  encodes  P^{(-i)}(z)  mod z^n
      oDn[S+i], oDls[S+i]  encodes  D^{(-i)}(z)  mod z^{n-1}  (if D requested)

    Update rule (child c with sibling s):
      oP[c] = oP[parent] * P[s]
      oD[c] = oD[parent] * P[s]  +  oP[parent] * D[s]

    Complexity: O(N log^2 n).
    """
    do_d = (Dn is not None)
    N2   = 2 * S

    oPn  = [None] * N2;  oPls = np.full(N2, _NEGINF)
    oDn  = [None] * N2;  oDls = np.full(N2, _NEGINF)

    oPn[1] = np.array([1.0]); oPls[1] = 0.0
    if do_d:
        oDn[1] = np.zeros(1); oDls[1] = _NEGINF

    for i in range(1, S):
        if oPn[i] is None: continue
        l, r = 2*i, 2*i+1
        for c, s in ((l, r), (r, l)):
            oPn[c], oPls[c] = _pmul_s(oPn[i], oPls[i], Pn[s], Pls[s], n - 1)
            if do_d:
                t1n, t1s        = _pmul_s(oDn[i], oDls[i], Pn[s], Pls[s], n - 1)
                t2n, t2s        = _pmul_s(oPn[i], oPls[i], Dn[s], Dls[s], n - 1)
                oDn[c], oDls[c] = _padd_s(t1n, t1s, t2n, t2s, n - 1)

    return oPn, oPls, oDn, oDls


# ── Extraction: pi, log_en, and (optionally) Hv ──────────────────────────────

def _extract(q_s, log_gm, Pn, Pls, oPn, oPls, oDn, oDls, S, N, n, v):
    """
    Extract pi, Hv (optional), and log_en from tree + downward-pass results.

    Formulas (all numerically stable):
      log e_n(q) = log|en_n| + root_ls + n * log_gm

    At leaf i, with dP = oPls[i] - root_ls, dD = oDls[i] - root_ls:
      pi_i   = q_s[i] * oPn[i][n-1] / en_n * exp(dP)
      SumZ_i = q_s[i] * oDn[i][n-2] / en_n * exp(dD)
             = sum_{j != i} pi_{ij} v_j

    (Cov[Z] v)_i = SumZ_i  +  pi_i v_i  -  pi_i (pi . v)

    exp(dP) and exp(dD) are bounded because pi_i in (0,1).
    """
    en_n    = float(Pn[1][n]) if len(Pn[1]) > n else 0.0
    root_ls = float(Pls[1])
    log_en  = (np.log(abs(en_n)) + root_ls + n * log_gm) if en_n != 0.0 else _NEGINF

    do_hv = (oDn is not None) and (v is not None)
    pi    = np.zeros(N)
    SumZ  = np.zeros(N) if do_hv else None

    for i in range(N):
        oP    = oPn[S + i];  oP_ls = float(oPls[S + i])
        if en_n != 0.0 and oP_ls != _NEGINF:
            oP_c   = float(oP[n-1]) if len(oP) > n - 1 else 0.0
            pi[i]  = q_s[i] * oP_c / en_n * np.exp(oP_ls - root_ls)
        if do_hv and n >= 2:
            oD    = oDn[S + i];  oD_ls = float(oDls[S + i])
            if en_n != 0.0 and oD_ls != _NEGINF:
                oD_c      = float(oD[n-2]) if len(oD) > n - 2 else 0.0
                SumZ[i]   = q_s[i] * oD_c / en_n * np.exp(oD_ls - root_ls)

    if do_hv:
        alpha = float(np.dot(pi, v))
        Hv    = SumZ + pi * v - pi * alpha
    else:
        Hv = None

    return pi, Hv, log_en


# ── Sampler: divide-and-conquer on the P-tree ─────────────────────────────────

def _tree_sample(Pn, Pls, S, N, n, M, rng):
    """
    Sample M subsets of size n using the cached P-tree.

    At each internal node with quota k, the split distribution is:

      P(j items from L | k) ∝ Pn_L[j] * Pn_R[k-j],   j = 0..k

    The Pls log-scale factors are common across j and cancel in the ratio,
    so only the normalised coefficient arrays Pn are needed — no new
    polynomial evaluations.

    Complexity: O(n log N) per sample, O(M n log N) total.
    """
    # quotas[node] = (M,) array of how many items to pick from this subtree
    quotas = np.zeros((2 * S, M), dtype=np.int32)
    quotas[1] = n

    for node in range(1, S):
        l, r  = 2 * node, 2 * node + 1
        k_arr = quotas[node]
        Pl    = Pn[l]
        Pr    = Pn[r]
        max_l = len(Pl)
        max_r = len(Pr)
        j_arr = np.zeros(M, dtype=np.int32)

        for k in np.unique(k_arr):
            if k == 0:
                continue
            mask  = k_arr == k
            count = int(mask.sum())

            # w[j] = Pl[j] * Pr[k-j]  for j = 0..k
            # (Pls factors cancel; coefficients are non-negative for ESP of q>0)
            jv    = np.arange(k + 1, dtype=int)
            kv    = k - jv
            safe_j  = np.minimum(jv, max_l - 1)
            safe_kv = np.minimum(kv, max_r - 1)
            wl    = np.where(jv  < max_l, Pl[safe_j],  0.0)
            wr    = np.where(kv  < max_r, Pr[safe_kv], 0.0)
            w     = np.maximum(wl, 0.0) * np.maximum(wr, 0.0)  # clip FFT rounding
            ws    = w.sum()
            if ws > 0:
                j_arr[mask] = rng.choice(k + 1, size=count, p=w / ws)
            # ws == 0 only for padding subtrees with no real items; j stays 0

        quotas[l] = j_arr
        quotas[r] = k_arr - j_arr

    # Collect leaf decisions into (M, n) output
    out     = np.full((M, n), -1, dtype=np.int32)
    cursors = np.zeros(M, dtype=np.int32)
    for i in range(N):
        inc = quotas[S + i] > 0       # quota 1 = include, 0 = skip
        idx = np.where(inc)[0]
        out[idx, cursors[idx]] = i
        cursors[idx] += 1

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

        P(S; theta) = exp(sum_{i in S} theta_i) / e_n(exp theta),  |S| = n

    Construction
    ------------
    ConditionalPoisson(n, theta)           direct
    ConditionalPoisson.uniform(N, n)       uniform inclusion probs n/N
    ConditionalPoisson.from_weights(n, q)  from positive weights q_i
    ConditionalPoisson.fit(pi_star, n)     moment-match to target probs

    Properties  (all cached; cache invalidated when theta changes)
    ----------
    pi              (N,) inclusion probabilities
    log_normalizer  log e_n(q)  — never overflows

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
    - theta is zero-centred after fitting (shift-invariant distribution).
    """

    def __init__(self, n: int, theta: np.ndarray):
        theta = np.asarray(theta, float)
        if theta.ndim != 1:   raise ValueError("theta must be 1-D")
        if len(theta) < n:    raise ValueError(f"N={len(theta)} must be >= n={n}")
        if n < 1:             raise ValueError("n must be >= 1")
        self._n     = int(n)
        self._theta = theta.copy()
        self._cache: dict = {}

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def uniform(cls, N: int, n: int) -> "ConditionalPoisson":
        """Uniform: all theta_i = 0, pi_i = n/N."""
        return cls(n, np.zeros(N))

    @classmethod
    def from_weights(cls, n: int, q: np.ndarray) -> "ConditionalPoisson":
        """Construct from positive weights q_i = exp(theta_i)."""
        q = np.asarray(q, float)
        if np.any(q <= 0): raise ValueError("all weights must be positive")
        return cls(n, np.log(q))

    @classmethod
    def fit(cls, pi_star: np.ndarray, n: int, **kw) -> "ConditionalPoisson":
        """Fit to target inclusion probabilities. See fit_inplace for options."""
        obj = cls(n, np.zeros(len(pi_star)))
        obj.fit_inplace(pi_star, **kw)
        return obj

    # ── Parameters ───────────────────────────────────────────────────────────

    @property
    def n(self) -> int:   return self._n
    @property
    def N(self) -> int:   return len(self._theta)
    @property
    def theta(self) -> np.ndarray: return self._theta.copy()
    @property
    def q(self) -> np.ndarray: return np.exp(self._theta)

    @theta.setter
    def theta(self, value):
        self._theta = np.asarray(value, float).copy()
        self._cache.clear()

    # ── Internal: P-tree (shared by pi, log_en, sampling) ────────────────────

    def _get_p_tree(self):
        """Build and cache the P-tree and normalised weights."""
        if "p_tree" not in self._cache:
            log_gm = float(np.mean(self._theta))
            q_s    = np.exp(self._theta - log_gm)
            Pn, Pls, S = _build_p_tree(q_s, self._n)
            self._cache["p_tree"] = (Pn, Pls, S, q_s, log_gm)
        return self._cache["p_tree"]

    # ── Internal: forward pass (pi, log_en) ──────────────────────────────────

    def _forward(self):
        if "pi" not in self._cache:
            Pn, Pls, S, q_s, log_gm = self._get_p_tree()
            oPn, oPls, _, _ = _downward_pass(Pn, Pls, None, None, S, self._n)
            pi, _, log_en = _extract(q_s, log_gm, Pn, Pls, oPn, oPls,
                                     None, None, S, self.N, self._n, None)
            self._cache["pi"]     = pi
            self._cache["log_en"] = log_en

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def pi(self) -> np.ndarray:
        """Inclusion probabilities pi_i = P(i in S).  O(N log^2 n), cached."""
        self._forward(); return self._cache["pi"].copy()

    @property
    def log_normalizer(self) -> float:
        """log e_n(q).  Never overflows.  O(N log^2 n), cached."""
        self._forward(); return self._cache["log_en"]

    # ── Log probability ───────────────────────────────────────────────────────

    def log_prob(self, S: Union[np.ndarray, list]) -> Union[float, np.ndarray]:
        """
        Log-probability of one subset or a batch.

            log P(S) = sum_{i in S} theta_i  -  log e_n(q)

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

    def sample(
        self,
        size: int = 1,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> np.ndarray:
        """
        Draw independent samples using the cached P-tree.

        At each internal node with quota k, sample the left/right split:

            P(j from L | k) ∝ Pn_L[j] * Pn_R[k-j]

        The Pls log-scale factors cancel; only the normalised Pn arrays
        are needed, which are already in the cached tree. No separate
        backward-ESP table is built.

        Parameters
        ----------
        size : number of subsets to draw
        rng  : seed or np.random.Generator

        Returns
        -------
        (size, n) int array; each row is a sorted list of n item indices.

        Complexity: O(N log^2 n) to build tree [cached] + O(size * n * log N).
        """
        rng             = np.random.default_rng(rng)
        Pn, Pls, S, q_s, _ = self._get_p_tree()
        return _tree_sample(Pn, Pls, S, self.N, self._n, size, rng)

    # ── Hessian-vector product ────────────────────────────────────────────────

    def hvp(self, v: np.ndarray) -> np.ndarray:
        """
        Compute Cov[Z] v.  O(N log^2 n).

        Cov[Z] = d^2/dtheta^2 log e_n(q) is the Fisher information matrix.
        It is positive semi-definite with rank N-1; null space = span{1}
        since sum(Z_i) = n is constant.

        Uses the cached P-tree; rebuilds D-tree (which depends on v).
        """
        v = np.asarray(v, float)
        Pn, Pls, S, q_s, log_gm = self._get_p_tree()
        Dn, Dls = _build_d_tree(Pn, Pls, q_s, v, self._n, S)
        oPn, oPls, oDn, oDls = _downward_pass(Pn, Pls, Dn, Dls, S, self._n)
        _, Hv, _ = _extract(q_s, log_gm, Pn, Pls, oPn, oPls, oDn, oDls,
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
        Update theta to match target inclusion probabilities pi*.

        Maximises L(theta) = pi* . theta - log e_n(exp theta)
        via Newton-CG with Armijo backtracking.
        theta is zero-centred on completion (shift-invariant distribution).

        Returns self (for chaining).
        """
        pi_star = np.asarray(pi_star, float)
        if len(pi_star) != self.N:
            raise ValueError(f"len(pi_star)={len(pi_star)} != N={self.N}")
        if abs(pi_star.sum() / self._n - 1.0) > 1e-6:
            raise ValueError(f"sum(pi_star)/n = {pi_star.sum()/self._n:.8f}, expected 1.0")
        if not np.all((pi_star > 0) & (pi_star < 1)):
            raise ValueError("all pi_star must lie strictly in (0, 1)")

        theta = np.log(pi_star / (1.0 - pi_star))   # warm start

        for it in range(max_iter):
            self.theta  = theta          # sets theta, clears cache
            pi          = self._cache.get("pi") or self.pi
            log_en      = self._cache["log_en"]
            grad        = pi_star - pi
            err         = float(np.max(np.abs(grad)))
            if verbose:
                print(f"  iter {it:3d}:  max|pi*-pi| = {err:.3e}")
            if err < tol:
                break

            delta = _cg(self.hvp, grad)

            slope = float(np.dot(grad, delta))
            L0    = float(np.dot(pi_star, theta)) - log_en
            step  = 1.0
            for _ in range(20):
                th_new        = theta + step * delta
                tmp           = ConditionalPoisson(self._n, th_new)
                L_new         = float(np.dot(pi_star, th_new)) - tmp.log_normalizer
                if L_new >= L0 + 1e-4 * step * slope: break
                step         *= 0.5

            theta += step * delta

        theta      -= theta.mean()   # zero-centre (shift-invariant)
        self.theta  = theta
        return self

    def __repr__(self) -> str:
        return (f"ConditionalPoisson(N={self.N}, n={self._n}, "
                f"log_normalizer={self.log_normalizer:.3f})")


# ── Tests ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math, time
    rng = np.random.default_rng(0)

    # 1. Basic properties
    print("=" * 60)
    print("Test 1: construction and forward pass")
    N, n   = 20, 7
    q_true = rng.exponential(1.0, N)
    cp     = ConditionalPoisson.from_weights(n, q_true)
    print(f"  {cp}")
    print(f"  sum(pi) = {cp.pi.sum():.6f}  (should be {n})")
    print(f"  pi in (0,1): {bool(np.all((cp.pi > 0) & (cp.pi < 1)))}\n")

    # 2. Fitting
    print("=" * 60)
    print("Test 2: fitting  (N=20, n=7)")
    cp_fit = ConditionalPoisson.fit(cp.pi, n, verbose=True)
    print(f"  max|pi_true - pi_fit| = {np.max(np.abs(cp.pi - cp_fit.pi)):.2e}\n")

    # 3. log_prob sums to 1
    print("=" * 60)
    print("Test 3: log_prob normalises correctly  (N=6, n=3 brute force)")
    from itertools import combinations
    cp3 = ConditionalPoisson.from_weights(3, rng.exponential(1.0, 6))
    all_S    = list(combinations(range(6), 3))
    lps      = np.array([cp3.log_prob(list(S)) for S in all_S])
    lps_b    = cp3.log_prob(np.array([list(S) for S in all_S]))
    print(f"  sum P(S) = {np.exp(lps).sum():.8f}  (should be 1)")
    print(f"  max |single - batch| = {np.max(np.abs(lps - lps_b)):.2e}\n")

    # 4. Sampling: check empirical pi
    print("=" * 60)
    print("Test 4: sampling via product tree  (M=100k)")
    M  = 100_000
    t0 = time.perf_counter()
    S  = cp.sample(M, rng=rng)
    t1 = time.perf_counter()
    print(f"  Drew {M:,} samples in {(t1-t0)*1000:.0f} ms")
    assert S.shape == (M, n)
    assert np.all(np.diff(S, axis=1) > 0), "samples not sorted"
    pi_emp = np.bincount(S.ravel(), minlength=N) / M
    err    = np.max(np.abs(pi_emp - cp.pi))
    print(f"  max|pi_emp - pi| = {err:.4f}  (expect ~{1/np.sqrt(M):.4f} MC noise)\n")

    # 5. HVP correctness
    print("=" * 60)
    print("Test 5: Hessian-vector product")
    v   = rng.standard_normal(N)
    Hv  = cp_fit.hvp(v)
    eps = 1e-5
    J   = np.zeros((N, N))
    for j in range(N):
        ej  = np.zeros(N); ej[j] = 1.0
        J[:, j] = (ConditionalPoisson(n, cp_fit.theta + eps*ej).pi -
                   ConditionalPoisson(n, cp_fit.theta - eps*ej).pi) / (2*eps)
    print(f"  max|Hv - Hv_fd|   = {np.max(np.abs(Hv - J @ v)):.2e}")
    print(f"  ||Cov[Z] * 1||    = {np.linalg.norm(cp_fit.hvp(np.ones(N))):.2e}  (null-space)\n")

    # 6. Numerical stability
    print("=" * 60)
    print("Test 6: numerical stability")
    cases = [
        ("theta ~ U[30,50],   N=10 n=4",   ConditionalPoisson(4,  rng.uniform(30,  50, 10))),
        ("theta ~ U[-50,-30], N=10 n=4",   ConditionalPoisson(4,  rng.uniform(-50,-30, 10))),
        ("linspace(-30,30),   N=10 n=4",   ConditionalPoisson(4,  np.linspace(-30, 30, 10))),
        ("N=100 n=50  theta=2  [old: sign error]", ConditionalPoisson(50,  np.full(100, 2.0))),
        ("N=200 n=100 theta=10 [old: nan]",        ConditionalPoisson(100, np.full(200, 10.0))),
        ("N=500 n=250 theta=5",                    ConditionalPoisson(250, np.full(500, 5.0))),
    ]
    for name, cpt in cases:
        pi_t = cpt.pi
        ok   = (np.isfinite(pi_t).all() and np.isfinite(cpt.log_normalizer)
                and abs(pi_t.sum() - cpt.n) < 1e-4)
        print(f"  {name}")
        print(f"    log_normalizer={cpt.log_normalizer:.2f}  pi.sum={pi_t.sum():.4f}  ok={ok}")

    # 7. Timing
    print()
    print("=" * 60)
    print("Test 7: timing")
    print(f"  {'N':>5}  {'n':>5}  {'pi ms':>8}  {'hvp ms':>8}  {'10k samp ms':>12}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*12}")
    for N7, n7 in [(50,20),(100,40),(200,80),(500,200)]:
        q7  = rng.exponential(1.0, N7)
        v7  = rng.standard_normal(N7)
        cp7 = ConditionalPoisson.from_weights(n7, q7)

        reps  = max(1, int(800 / (N7 * n7**0.5)))

        t0 = time.perf_counter()
        for _ in range(reps): cp7._cache.clear(); cp7.pi
        pi_ms = (time.perf_counter() - t0) / reps * 1000

        t0 = time.perf_counter()
        for _ in range(reps): cp7.hvp(v7)
        hvp_ms = (time.perf_counter() - t0) / reps * 1000

        t0 = time.perf_counter()
        cp7.sample(10_000, rng=rng)
        samp_ms = (time.perf_counter() - t0) * 1000

        print(f"  {N7:>5}  {n7:>5}  {pi_ms:>8.1f}  {hvp_ms:>8.1f}  {samp_ms:>12.0f}")
