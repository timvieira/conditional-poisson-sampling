"""
conditional_poisson_torch.py
============================

PyTorch implementation of the conditional Poisson distribution.

Uses FFT-based polynomial product tree with contour radius scaling for
O(N log² n) complexity and full autograd support.  Gradients (π), HVP
(Cov·v), and fitting come from torch.autograd — no hand-coded downward
pass or D-tree needed.

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
from typing import Optional, Union


# ── Utilities ────────────────────────────────────────────────────────────────

def _to_tensor(x, dtype=torch.float64):
    """Convert input to a torch tensor if it isn't one already."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.tensor(x, dtype=dtype)


# ── Find the optimal rescaling parameter r ───────────────────────────────────

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


# ── Batched FFT polynomial multiplication ────────────────────────────────────

def _batch_poly_mul_fft(a_batch, b_batch):
    """Multiply pairs of polynomials via batched FFT.

    O(B * d log d) where d = La + Lb.  Uses torch.fft which is
    differentiable, so autograd can backpropagate through this.

    Parameters
    ----------
    a_batch : (B, La) tensor
    b_batch : (B, Lb) tensor

    Returns
    -------
    (B, La + Lb - 1) tensor of product polynomials
    """
    La = a_batch.shape[1]
    Lb = b_batch.shape[1]
    n_out = La + Lb - 1
    # Convolution theorem: poly_mul(a, b) = ifft(fft(a) * fft(b))
    fa = torch.fft.rfft(a_batch, n=n_out)
    fb = torch.fft.rfft(b_batch, n=n_out)
    return torch.fft.irfft(fa * fb, n=n_out)


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


# Threshold: use FFT when polynomials are large enough to amortize overhead.
# With contour scaling, FFT precision is no longer a concern.
_FFT_THRESHOLD = 64

def _batch_poly_mul(a_batch, b_batch):
    """Hybrid: direct for small polys, FFT for large."""
    if a_batch.shape[1] + b_batch.shape[1] <= _FFT_THRESHOLD:
        return _batch_poly_mul_direct(a_batch, b_batch)
    return _batch_poly_mul_fft(a_batch, b_batch)


# ── Forward pass: product tree with contour scaling ──────────────────────────

def forward_log_Z(theta, n):
    """Compute log Z via FFT-based product tree with contour scaling.

    Steps:
    1. Compute r = optimal contour radius (non-differentiable root-find)
    2. Rescale weights: w_i' = w_i * r  (differentiable)
    3. Build product tree: prod_i (1 + w_i' z) via batched FFT
    4. Extract: log Z = log(root[n]) - n * log(r)

    Parameters
    ----------
    theta : (N,) tensor of log-weights (requires_grad=True)
    n     : subset size (int)

    Returns
    -------
    log_Z : scalar tensor (differentiable w.r.t. theta)
    """
    N = theta.shape[0]
    dtype = theta.dtype
    device = theta.device

    w = torch.exp(theta)

    # Step 1: find optimal contour radius (not on autograd graph).
    #
    # Why this works: the product polynomial P(z) = prod_i (1 + w_i z) has
    # its peak coefficient near degree N/2 (for uniform weights).  The
    # coefficient at degree n can be ~10^{-300} smaller.  FFT introduces
    # errors ~eps * max|c|, which drowns the small coefficient.
    #
    # Rescaling z -> r*z gives P(r*z) = prod_i (1 + w_i*r*z).  The k-th
    # coefficient of P(r*z) is c_k * r^k, so choosing r shifts the peak.
    # The optimal r makes the expected sample size under Poisson sampling
    # equal n, placing the peak at degree n — exactly where we need it.
    #
    # r is NOT on the autograd graph: it's a numerical conditioning choice,
    # not part of the mathematical function.  The gradient of log_Z w.r.t.
    # theta flows through w_scaled = w * r (where r is a constant).
    r = _find_r(w.detach(), n)
    log_r = math.log(r)

    # Step 2: rescale weights (this IS on the autograd graph).
    # After rescaling, the product polynomial's peak is near degree n,
    # so FFT rounding errors are relative to the coefficient we need.
    w_scaled = w * r

    # Step 3: build leaf polynomials (1 + w_i' * z)
    ones = torch.ones(N, dtype=dtype, device=device)
    polys = torch.stack([ones, w_scaled], dim=1)  # (N, 2)

    # Per-polynomial log-scale for renormalization.
    # True polynomial = polys[i] * exp(log_scales[i]).
    # Keeps convolution inputs O(1) for numerical stability.
    log_scales = torch.zeros(N, dtype=dtype, device=device)

    # Bottom-up tree with batched multiplication
    while polys.shape[0] > 1:
        B = polys.shape[0]
        if B % 2 == 1:
            leftover_poly = polys[-1:]
            leftover_scale = log_scales[-1:]
            polys = polys[:-1]
            log_scales = log_scales[:-1]
            B -= 1
        else:
            leftover_poly = None
            leftover_scale = None

        left = polys[0::2]
        right = polys[1::2]

        # Truncate inputs to degree n (we only need coeff[0..n] of root)
        if left.shape[1] > n + 1:
            left = left[:, :n + 1]
        if right.shape[1] > n + 1:
            right = right[:, :n + 1]

        products = _batch_poly_mul(left, right)

        # Truncate output to degree n
        if products.shape[1] > n + 1:
            products = products[:, :n + 1]

        # Renormalize: scale to max|c| = 1
        new_scales = log_scales[0::2] + log_scales[1::2]
        max_abs = products.abs().max(dim=1).values.clamp(min=1e-300)
        products = products / max_abs.unsqueeze(1)
        new_scales = new_scales + torch.log(max_abs)

        if leftover_poly is not None:
            pad_size = products.shape[1] - leftover_poly.shape[1]
            if pad_size > 0:
                leftover_poly = F.pad(leftover_poly, (0, pad_size))
            products = torch.cat([products, leftover_poly], dim=0)
            new_scales = torch.cat([new_scales, leftover_scale], dim=0)

        polys = products
        log_scales = new_scales

    root = polys[0]
    # root[n] * exp(log_scales[0]) = [z^n] prod(1 + w_i*r*z) = Z(w,n) * r^n
    # => log Z = log(root[n]) + log_scales[0] - n * log(r)
    log_Z = torch.log(root[n]) + log_scales[0] - n * log_r
    return log_Z


# ── Convenience wrappers ─────────────────────────────────────────────────────

def compute_pi(theta, n):
    """Inclusion probabilities via autograd on log_Z.

    pi_i = d(log Z) / d(theta_i).  This is backpropagation (reverse-mode AD)
    applied to the forward pass — the Baur-Strassen theorem guarantees the
    cost is O(1)x the forward pass.  Griewank-Walther guarantees the numerical
    stability is inherited from the forward pass.
    """
    theta = theta.detach().requires_grad_(True)
    log_Z = forward_log_Z(theta, n)
    pi = torch.autograd.grad(log_Z, theta, create_graph=True)[0]
    return pi


def compute_hvp(theta, n, v):
    """Hessian-vector product Cov[1_S] v via double backward.

    Pearlmutter's R-operator: forward-mode AD applied to the backward pass
    (or equivalently, backward-mode applied to the gradient computation).
    Cost is O(1)x the gradient, i.e., O(1)x the forward pass.
    """
    theta = theta.detach().requires_grad_(True)
    log_Z = forward_log_Z(theta, n)
    grad = torch.autograd.grad(log_Z, theta, create_graph=True)[0]
    hv = torch.autograd.grad(grad, theta, grad_outputs=v)[0]
    return hv


# ── Class interface ──────────────────────────────────────────────────────────

class ConditionalPoissonTorch:
    """
    Conditional Poisson distribution over fixed-size subsets (PyTorch).

        P(S) ∝ prod_{i in S} w_i,   |S| = n

    Construction
    ------------
    ConditionalPoissonTorch(n, theta)                direct from log-weights
    ConditionalPoissonTorch.from_weights(n, w)       from positive weights
    ConditionalPoissonTorch.fit(pi_star, n)           fit to target probs

    Properties
    ----------
    incl_prob       (N,) inclusion probabilities
    w               (N,) weights (= exp(theta))
    log_normalizer  log normalizing constant

    Methods
    -------
    log_prob(S)         log-probability of subset(s)
    sample(M)           draw M independent subsets
    hvp(v)              Cov[1_S] v
    """

    def __init__(self, n: int, theta, *, dtype=torch.float64, device=None):
        """Construct from log-weights theta_i = log(w_i)."""
        self._theta = _to_tensor(theta, dtype).detach().clone().to(device=device)
        self._n = int(n)
        self._N = len(self._theta)
        self._sample_tree = None

        if self._n < 0 or self._n > self._N:
            raise ValueError(f"n={self._n} must be in [0, {self._N}]")

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
    def fit(cls, pi_star, n: int, *, tol: float = 1e-7,
            dtype=torch.float64, device=None, **kw) -> "ConditionalPoissonTorch":
        """
        Fit to target inclusion probabilities via L-BFGS.

        Minimizes -E_π★[log P_θ(S)] = -(π★ᵀθ - log Z(θ, n)).
        The gradient is π(θ) - π★, so each L-BFGS step is driven
        directly by the gap between current and target inclusion
        probabilities.  Convergence (max|π(θ) - π★| ≤ tol) is
        therefore an infinity-norm gradient test — the optimizer's
        own stopping criterion.

        All values in π★ must be in (0, 1) and sum to n.

        Parameters
        ----------
        tol : convergence tolerance on max|π(θ) - π★|.
        """
        pi_star = _to_tensor(pi_star, dtype).to(device=device)
        theta = torch.log(pi_star / (1.0 - pi_star)).clone().requires_grad_(True)

        # L-BFGS (Nocedal & Wright, Ch. 7): memory m, Wolfe line search.
        # The gradient is π(θ) - π★, so tolerance_grad = tol stops
        # when max|π - π★| ≤ tol.
        optimizer = torch.optim.LBFGS(
            [theta],
            max_iter=200,
            history_size=5,
            line_search_fn='strong_wolfe',
            tolerance_grad=tol,
            tolerance_change=0,
        )

        def closure():
            optimizer.zero_grad()
            loss = -(pi_star @ theta - forward_log_Z(theta, n))
            loss.backward()
            return loss

        optimizer.step(closure)

        # Check fit quality: how close are the fitted π to the targets?
        pi_fit = compute_pi(theta.detach(), n)
        fit_err = (pi_fit - pi_star).abs().max().item()
        if fit_err > tol:
            import warnings
            warnings.warn(
                f"fit did not reach tol={tol:.0e}: max|pi - pi*| = {fit_err:.2e}. "
                f"L-BFGS exhausted max_iter=200 without converging.")

        with torch.no_grad():
            theta -= theta.mean()

        return cls(n, theta.detach(), dtype=dtype, device=device, **kw)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        return self._n

    @property
    def N(self) -> int:
        return self._N

    @property
    def theta(self) -> torch.Tensor:
        return self._theta.clone()

    @property
    def w(self) -> torch.Tensor:
        return torch.exp(self._theta)

    @property
    def incl_prob(self) -> torch.Tensor:
        """Inclusion probabilities pi_i = P(i in S)."""
        return compute_pi(self._theta, self._n).detach()

    @property
    def log_normalizer(self) -> float:
        """Log normalizing constant."""
        return forward_log_Z(self._theta, self._n).item()

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

    def _build_sample_tree(self):
        """Build the product tree and precompute CDFs for sampling.

        Cached on first call to sample().
        """
        N = self._N
        n = self._n
        w = torch.exp(self._theta).detach()

        r = _find_r(w, n)
        w_scaled = w * r

        tree_n = 1 << (N - 1).bit_length()
        Pc = [None] * (2 * tree_n)

        for i in range(N):
            Pc[tree_n + i] = [1.0, w_scaled[i].item()]
        for i in range(N, tree_n):
            Pc[tree_n + i] = [1.0]

        for i in range(tree_n - 1, 0, -1):
            L = torch.tensor(Pc[2 * i], dtype=self._theta.dtype)
            R = torch.tensor(Pc[2 * i + 1], dtype=self._theta.dtype)
            p = _batch_poly_mul(L.unsqueeze(0), R.unsqueeze(0)).squeeze(0)
            if len(p) > n + 1:
                p = p[:n + 1]
            mx = p.abs().max()
            if mx > 0:
                p = p / mx
            Pc[i] = p.tolist()

        # Precompute normalized CDFs for each (node, quota) pair
        cdfs = [None] * (2 * tree_n)
        for node in range(1, tree_n):
            L, R = Pc[2 * node], Pc[2 * node + 1]
            max_k = min(n, len(L) - 1 + len(R) - 1)
            node_cdfs = [None] * (max_k + 1)
            for k in range(1, max_k + 1):
                cdf = []
                total = 0.0
                for j in range(k + 1):
                    rem = k - j
                    lv = L[j] if j < len(L) else 0.0
                    rv = R[rem] if rem < len(R) else 0.0
                    total += max(lv, 0.0) * max(rv, 0.0)
                    cdf.append(total)
                if total > 0:
                    node_cdfs[k] = [c / total for c in cdf]
            cdfs[node] = node_cdfs

        self._sample_tree = (cdfs, tree_n)

    def sample(self, size: int = 1, rng: Optional[int] = None) -> torch.Tensor:
        """
        Draw independent samples.

        Returns (size, n) long tensor of sorted indices.
        """
        import random as _random

        if self._sample_tree is None:
            self._build_sample_tree()

        if self._n == 0:
            return torch.empty(size, 0, dtype=torch.long)
        if self._n == self._N:
            return torch.arange(self._N).unsqueeze(0).expand(size, -1)

        cdfs, tree_n = self._sample_tree
        N, n = self._N, self._n

        if rng is not None:
            r = _random.Random(rng)
        else:
            r = _random.Random()

        samples = []
        for _ in range(size):
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
                u = r.random()
                j = 0
                while j < k and cdf[j] < u:
                    j += 1
                stack.append((2 * node + 1, k - j))
                stack.append((2 * node, j))
            selected.sort()
            samples.append(torch.tensor(selected, dtype=torch.long))

        return torch.stack(samples)

    # ── Hessian-vector product ────────────────────────────────────────────────

    def hvp(self, v) -> torch.Tensor:
        """Compute Cov[1_S] v via autograd (double backward)."""
        return compute_hvp(self._theta, self._n, _to_tensor(v, self._theta.dtype)).detach()

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self):
        return (f"ConditionalPoissonTorch(n={self._n}, N={self._N}, "
                f"log_Z={self.log_normalizer:.4f})")
