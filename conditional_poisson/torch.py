"""
conditional_poisson_torch.py
============================

PyTorch implementation of the conditional Poisson distribution.

Uses FFT-based polynomial product tree with contour radius scaling for
O(N log² n) complexity and full autograd support.  Gradients (π) and
fitting come from torch.autograd — no hand-coded downward pass needed.

Weights must be finite and positive.  For forced inclusion/exclusion,
pre-filter the universe before constructing.

The key numerical trick: **contour radius scaling**.  Before building the
polynomial product tree, rescale each weight w_i -> w_i * r, where r is
chosen so the peak of the product polynomial falls at degree n (the
coefficient we need).  This collapses the dynamic range from ~10^{-300}
to ~O(1), making standard float64 FFT accurate for the target coefficient.

The optimal r = exp(t) where t solves:

    sum_i  w_i * r / (1 + w_i * r)  =  n

This is the Lagrange multiplier from the Poisson sampling connection:
under Poisson sampling with probabilities p_i = w_i*r / (1 + w_i*r),
the expected sample size is n.

After multiplying the rescaled polynomials prod_i (1 + w_i*r*z) via FFT,
the n-th coefficient equals Z(w, n) * r^n.  We recover log Z as:

    log Z = log(root_rescaled[n]) - n * log(r)

Complexity: O(N log² n) with truncation to degree n.
Precision: full float64 — the coefficient at degree n is near the peak.
Autograd: fully compatible — rescaling + torch.fft are differentiable.

Pure PyTorch internally — no numpy or scipy dependency.
"""

import torch
import torch.nn.functional as F
import math
from bisect import bisect_left as _bisect_left
from typing import Union


def _to_tensor(x, dtype=torch.float64):
    """Convert input to a torch tensor if it isn't one already."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.tensor(x, dtype=dtype)


# Threshold: use FFT when polynomials are large enough to amortize overhead.
# With contour scaling, FFT precision is no longer a concern.
_FFT_THRESHOLD = 64


class ConditionalPoissonTorch:
    """
    Conditional Poisson distribution over fixed-size subsets (PyTorch).

        P(S) ∝ prod_{i in S} w_i,   |S| = n

    Construction
    ------------
    ConditionalPoissonTorch(n, theta)                direct from log-weights
    ConditionalPoissonTorch.from_weights(n, w)       from positive weights
    ConditionalPoissonTorch.fit(target_incl, n)           fit to target probs

    Properties
    ----------
    incl_prob       (N,) inclusion probabilities
    w               (N,) weights (= exp(theta))
    log_normalizer  log normalizing constant

    Methods
    -------
    log_prob(S)         log-probability of subset(s)
    sample(M)           draw M independent subsets
    """

    def __init__(self, n: int, theta, *, dtype=torch.float64, device=None):
        """Construct from log-weights theta_i = log(w_i)."""
        self._theta = _to_tensor(theta, dtype).detach().clone().to(device=device)
        self.n = int(n)
        self.N = len(self._theta)
        self._sample_tree = None

        if self.n < 0 or self.n > self.N:
            raise ValueError(f"n={self.n} must be in [0, {self.N}]")

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_weights(cls, n: int, w, **kw) -> "ConditionalPoissonTorch":
        """Construct from positive weights w_i."""
        w = _to_tensor(w, kw.get('dtype', torch.float64))
        if (w <= 0).any() or not torch.isfinite(w).all():
            raise ValueError("all weights must be finite and positive")
        with torch.no_grad():
            theta = torch.log(w)
        return cls(n, theta, **kw)

    @classmethod
    def fit(cls, target_incl, n: int, *, tol: float = 1e-7,
            dtype=torch.float64, device=None, **kw) -> "ConditionalPoissonTorch":
        """
        Fit to target inclusion probabilities via L-BFGS.

        Minimizes log Z(θ, n) - π*ᵀθ (negative log-likelihood under π*).
        The gradient is π(θ) - π*, so each L-BFGS step is driven
        directly by the gap between current and target inclusion
        probabilities.  Convergence (max|π(θ) - π*| ≤ tol) is
        therefore an infinity-norm gradient test — the optimizer's
        own stopping criterion.

        All values in target_incl must be in (0, 1) and sum to n.

        Parameters
        ----------
        target_incl : (N,) target inclusion probabilities
        tol : convergence tolerance on max|π(θ) - π*|.
        """
        target_incl = _to_tensor(target_incl, dtype).to(device=device)
        cp = cls(n, torch.logit(target_incl), dtype=dtype, device=device, **kw)
        cp._theta.requires_grad_(True)

        # L-BFGS (Nocedal & Wright, Ch. 7): memory m, Wolfe line search.
        # The gradient is π(θ) - π*, so tolerance_grad = tol stops
        # when max|π - π*| ≤ tol.
        optimizer = torch.optim.LBFGS(
            [cp._theta],
            max_iter=200,
            history_size=5,
            line_search_fn='strong_wolfe',
            tolerance_grad=tol,
            tolerance_change=0,   # only stop on gradient criterion, not loss plateau
        )

        def closure():
            optimizer.zero_grad()
            loss = cp._log_Z() - target_incl @ cp._theta
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            cp._theta -= cp._theta.mean()   # zero-center (shift-invariant)
        cp._theta = cp._theta.detach()

        return cp

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def theta(self) -> torch.Tensor:
        return self._theta.clone()

    @property
    def w(self) -> torch.Tensor:
        return torch.exp(self._theta)

    @property
    def incl_prob(self) -> torch.Tensor:
        """Inclusion probabilities pi_i = P(i in S).

        pi_i = d(log Z) / d(theta_i).  This is backpropagation (reverse-mode AD)
        applied to the forward pass — the Baur-Strassen theorem guarantees the
        cost is O(1)x the forward pass.  Griewank-Walther guarantees the numerical
        stability is inherited from the forward pass.
        """
        saved = self._theta
        self._theta = saved.detach().requires_grad_(True)
        log_Z = self._log_Z()
        pi = torch.autograd.grad(log_Z, self._theta)[0].detach()
        self._theta = saved
        return pi

    @property
    def log_normalizer(self) -> float:
        """Log normalizing constant."""
        return self._log_Z().item()

    # ── Log probability ───────────────────────────────────────────────────────

    def log_prob(self, S) -> Union[float, torch.Tensor]:
        """
        Log-probability: log P(S) = sum_{i in S} theta_i - log Z.

        S: (n,) or (M, n) int tensor, or (N,) or (M, N) bool tensor.
        """
        S = _to_tensor(S, torch.long) if not isinstance(S, torch.Tensor) else S
        th = self._theta.detach()
        lz = self.log_normalizer

        if S.dtype == torch.bool:
            if S.dim() == 2:
                return th @ S.float().T - lz
            return float(th[S].sum() - lz)
        if S.dim() == 2:
            return torch.stack([th[row].sum() - lz for row in S])
        return float(th[S].sum() - lz)

    # ── Sampling ──────────────────────────────────────────────────────────────

    def _get_sample_cdfs(self):
        """Precompute CDFs for sampling from the product tree (cached)."""
        if self._sample_tree is not None:
            return self._sample_tree

        tree, tree_n, _, _ = self._build_tree(store=True)
        n = self.n

        # Convert tree tensors to lists for fast Python CDF computation.
        Pc = [None] * (2 * tree_n)
        for i in range(1, 2 * tree_n):
            if tree[i] is not None:
                Pc[i] = tree[i].detach().tolist()

        # Precompute normalized CDFs for each (node, quota) pair.
        # PMF of split distribution: pmf[j] = L[j] * R[k-j]
        cdfs = [None] * (2 * tree_n)
        for node in range(1, tree_n):
            L, R = Pc[2 * node], Pc[2 * node + 1]
            max_k = min(n, len(L) - 1 + len(R) - 1)
            node_cdfs = [None] * (max_k + 1)
            for k in range(1, max_k + 1):
                pmf = []
                for j in range(k + 1):
                    rem = k - j
                    lv = L[j] if j < len(L) else 0.0
                    rv = R[rem] if rem < len(R) else 0.0
                    pmf.append(max(lv, 0.0) * max(rv, 0.0))
                total = sum(pmf)
                if total > 0:
                    acc = 0.0
                    cdf = []
                    for p in pmf:
                        acc += p / total
                        cdf.append(acc)
                    node_cdfs[k] = cdf
            cdfs[node] = node_cdfs

        self._sample_tree = (cdfs, tree_n)
        return self._sample_tree

    @staticmethod
    def _draw_one(cdfs, tree_n, N, n, rng):
        """Draw one sample via top-down quota splitting with precomputed CDFs.

        Complexity: O(n log N).
        """
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
            cdf = cdfs[node][k]
            j = _bisect_left(cdf, rng.random())
            stack.append((2 * node + 1, k - j))
            stack.append((2 * node, j))
        selected.sort()
        return selected

    def sample(self, size: int = 1, rng=None) -> torch.Tensor:
        """
        Draw independent samples using the cached product tree.

        Parameters
        ----------
        size : number of subsets to draw
        rng  : int seed, random.Random, or np.random.Generator

        Returns
        -------
        (size, n) long tensor of sorted indices.

        Complexity: O(N log^2 n) to build tree [cached] + O(size * n * log N).
        """
        import random as _random
        import numpy as np

        cdfs, tree_n = self._get_sample_cdfs()
        N, n = self.N, self.n

        # XXX: are these special cases necessary?
        if n == 0:
            return torch.empty(size, 0, dtype=torch.long)
        if n == N:
            return torch.arange(N).unsqueeze(0).expand(size, -1)

        # Accept any object with .random() method (random.Random, np.random.Generator)
        if isinstance(rng, np.random.Generator) or isinstance(rng, _random.Random):
            pass  # use as-is
        else:
            rng = _random.Random(rng)

        # XXX: remove the option to sampled more than one element; eliminate the size argument
        if size == 1:
            s = self._draw_one(cdfs, tree_n, N, n, rng)
            return torch.tensor(s, dtype=torch.long).unsqueeze(0)
        return torch.stack([
            torch.tensor(self._draw_one(cdfs, tree_n, N, n, rng), dtype=torch.long)
            for _ in range(size)
        ])

    # ── Internal: product tree with contour scaling ───────────────────────────

    @staticmethod
    def _find_r(w, n, tol=1e-12, max_iter=100):
        """Find r such that sum_i w_i*r / (1 + w_i*r) = n.

        This is a monotone equation in log(r): the LHS is strictly increasing
        from 0 (as r->0) to N (as r->inf).  Newton's method converges quickly.

        Parameters
        ----------
        w : (N,) tensor of positive weights (detached, no grad needed)
        n : target sum (int)

        Returns
        -------
        r : scalar (float, not a tensor — this is a numerical parameter,
            not part of the differentiable computation)
        """

        # XXX: make find_r a proper method that uses theta and n

        # Work in log-space: let t = log(r), solve g(t) = sum p_i(t) - n = 0
        # where p_i(t) = w_i*exp(t) / (1 + w_i*exp(t)) = sigmoid(log(w_i) + t)
        log_w = torch.log(w).detach().double()

        # Initial guess: t such that n/N of the items have p_i ≈ 0.5
        # i.e., median(log_w) + t ≈ 0 => t ≈ -median(log_w)
        t = -torch.median(log_w).item()

        for _ in range(max_iter):
            s = log_w + t
            # Numerically stable sigmoid and its derivative
            # Clamp to avoid exp overflow in the tails
            s_clamped = s.clamp(-500, 500)
            p = torch.sigmoid(s_clamped)
            g = p.sum().item() - n
            gp = (p * (1 - p)).sum().item()

            if abs(g) < tol:
                break
            if gp < 1e-100:
                # All probabilities saturated; do a bisection-style step
                if g > 0:
                    t -= 1.0
                else:
                    t += 1.0
                continue

            t -= g / gp

        return math.exp(t)

    @staticmethod
    def _batch_poly_mul_fft(a_batch, b_batch):
        """Multiply pairs of polynomials via batched FFT.

        O(B * d log d) where d = La + Lb.  Uses torch.fft which is
        differentiable, so autograd can backpropagate through this.
        """
        La = a_batch.shape[1]
        Lb = b_batch.shape[1]
        n_out = La + Lb - 1
        fa = torch.fft.rfft(a_batch, n=n_out)
        fb = torch.fft.rfft(b_batch, n=n_out)
        return torch.fft.irfft(fa * fb, n=n_out)

    @staticmethod
    def _batch_poly_mul_direct(a_batch, b_batch):
        """Multiply pairs of polynomials via grouped conv1d (direct O(d²))."""
        B = a_batch.shape[0]
        Lb = b_batch.shape[1]
        out = F.conv1d(
            a_batch.unsqueeze(0),
            b_batch.flip(-1).unsqueeze(1),
            padding=Lb - 1,
            groups=B,
        )
        return out.squeeze(0)

    @classmethod
    def _batch_poly_mul(cls, a_batch, b_batch):
        """Hybrid: direct for small polys, FFT for large."""
        if a_batch.shape[1] + b_batch.shape[1] <= _FFT_THRESHOLD:
            return cls._batch_poly_mul_direct(a_batch, b_batch)
        return cls._batch_poly_mul_fft(a_batch, b_batch)

    def _log_Z(self):
        """Log normalizing constant as a differentiable scalar tensor."""
        root, root_log_scale, log_r = self._build_tree()
        return torch.log(root[self.n]) + root_log_scale - self.n * log_r

    def _build_tree(self, store=False):
        """Build the product tree via batched FFT with contour scaling.

        If store=False (default), returns only the root polynomial — fast
        path for log_Z and incl_prob.

        If store=True, also populates and returns the full segment tree
        (needed for sampling CDFs).

        Returns (root_or_tree, root_log_scale_or_tree_n, log_r[, ...]):
        - store=False: (root_poly, root_log_scale, log_r)
        - store=True:  (tree, tree_n, log_r, root_log_scale)
        """
        n = self.n
        N = self.N
        theta = self._theta
        dtype = theta.dtype
        device = theta.device

        w = torch.exp(theta)

        # Find optimal contour radius (not on autograd graph).
        # Rescaling z -> r*z shifts the product polynomial's peak to degree n,
        # making FFT numerically stable for the coefficient we need.
        # r is NOT differentiable — it's a conditioning choice, not part of
        # the mathematical function.  Gradients flow through w * r.
        r = self._find_r(w.detach(), n)
        log_r = math.log(r)
        w_scaled = w * r   # on autograd graph

        # Pad to power of 2 for clean binary tree layout.
        tree_n = 1 << max(1, (N - 1).bit_length())

        # Leaf polynomials: (1 + w_i'*z) for real items, (1) for padding.
        polys = torch.ones(tree_n, 2, dtype=dtype, device=device)
        polys[:N, 1] = w_scaled
        polys[N:, 1] = 0.0

        if store:
            tree = [None] * (2 * tree_n)
            for i in range(tree_n):
                tree[tree_n + i] = polys[i]

        # Per-polynomial log-scale for renormalization.
        # True polynomial = polys[i] * exp(scales[i]).
        scales = torch.zeros(tree_n, dtype=dtype, device=device)

        # Bottom-up: batched multiplication, one level at a time.
        level_size = tree_n
        while level_size > 1:
            left = polys[0::2]
            right = polys[1::2]

            if left.shape[1] > n + 1:
                left = left[:, :n + 1]
            if right.shape[1] > n + 1:
                right = right[:, :n + 1]

            products = self._batch_poly_mul(left, right)
            if products.shape[1] > n + 1:
                products = products[:, :n + 1]

            new_scales = scales[0::2] + scales[1::2]
            max_abs = products.abs().max(dim=1).values.clamp(min=1e-300)
            products = products / max_abs.unsqueeze(1)
            new_scales = new_scales + torch.log(max_abs)

            level_size //= 2
            if store:
                for i in range(level_size):
                    tree[level_size + i] = products[i]

            polys = products
            scales = new_scales

        if store:
            return tree, tree_n, log_r, scales[0]
        return polys[0], scales[0], log_r

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self):
        return (f"ConditionalPoissonTorch(n={self.n}, N={self.N}, "
                f"log_Z={self.log_normalizer:.4f})")
