"""
conditional_poisson_torch.py
============================

Drop-in PyTorch replacement for ConditionalPoisson (conditional_poisson.py).

Uses FFT-based polynomial product tree with contour radius scaling for
O(N log² n) complexity and full autograd support.  Gradients (π), HVP
(Cov·v), and fitting come from torch.autograd — no hand-coded downward
pass or D-tree needed.

Weights must be finite and positive.  For forced inclusion/exclusion,
pre-filter the universe before constructing.

Pure PyTorch internally — no numpy or scipy dependency.
"""

import torch
from typing import Optional, Union

from torch_fft_prototype import forward_log_Z, compute_pi, compute_hvp, _find_r
from torch_fft_prototype import _batch_poly_mul


def _to_tensor(x, dtype=torch.float64):
    """Convert input to a torch tensor if it isn't one already."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.tensor(x, dtype=dtype)


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
    pi              (N,) inclusion probabilities
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

        All values in pi_star must be in (0, 1) and sum to n.

        Parameters
        ----------
        tol : convergence tolerance on max|pi - pi_star|.
              Since the gradient of the objective is exactly pi_star - pi(theta),
              this is equivalent to an infinity-norm gradient tolerance.
        """
        pi_star = _to_tensor(pi_star, dtype).to(device=device)
        theta = torch.log(pi_star / (1.0 - pi_star)).clone().requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [theta], line_search_fn='strong_wolfe',
            tolerance_grad=0, tolerance_change=0,  # we check convergence ourselves
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
                f"The LBFGS line search terminated early.")

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
    def pi(self) -> torch.Tensor:
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
        """Build the product tree for sampling (cached on first call).

        tree[node][k] = Z(w_T, k) for items in subtree T (up to a
        per-node scale factor that cancels in the sampling ratios).
        """
        N = self._N
        n = self._n
        w = torch.exp(self._theta).detach()

        r = _find_r(w, n)
        w_scaled = w * r

        tree_n = 1 << (N - 1).bit_length()
        tree = [None] * (2 * tree_n)

        for i in range(N):
            tree[tree_n + i] = torch.tensor([1.0, w_scaled[i].item()],
                                            dtype=self._theta.dtype)
        for i in range(N, tree_n):
            tree[tree_n + i] = torch.ones(1, dtype=self._theta.dtype)

        for i in range(tree_n - 1, 0, -1):
            p = _batch_poly_mul(tree[2 * i].unsqueeze(0),
                                tree[2 * i + 1].unsqueeze(0)).squeeze(0)
            if len(p) > n + 1:
                p = p[:n + 1]
            mx = p.abs().max()
            if mx > 0:
                p = p / mx
            tree[i] = p.detach()

        self._sample_tree = (tree, tree_n)

    def _draw_one_sample(self, generator):
        """Draw one sample via top-down quota splitting."""
        tree, tree_n = self._sample_tree
        n = self._n
        N = self._N

        selected = []
        stack = [(1, n)]
        while stack:
            node, k = stack.pop()
            if k == 0:
                continue
            if node >= tree_n:
                leaf = node - tree_n
                if leaf < N and k == 1:
                    selected.append(leaf)
                continue
            left = tree[2 * node]
            right = tree[2 * node + 1]
            probs = torch.zeros(k + 1, dtype=self._theta.dtype)
            for j in range(k + 1):
                rem = k - j
                if j < len(left) and rem < len(right):
                    probs[j] = max(left[j].item(), 0.0) * max(right[rem].item(), 0.0)
            total = probs.sum()
            if total <= 0:
                continue
            probs /= total
            j = torch.multinomial(probs, 1, generator=generator).item()
            stack.append((2 * node + 1, k - j))
            stack.append((2 * node, j))

        return torch.sort(torch.tensor(selected, dtype=torch.long))[0]

    def sample(self, size: int = 1, rng: Optional[int] = None) -> torch.Tensor:
        """
        Draw independent samples.

        Returns (size, n) long tensor of sorted indices.
        """
        generator = torch.Generator()
        if rng is not None:
            generator.manual_seed(rng)
        else:
            generator.seed()

        if self._sample_tree is None:
            self._build_sample_tree()

        if self._n == 0:
            return torch.empty(size, 0, dtype=torch.long)
        if self._n == self._N:
            return torch.arange(self._N).unsqueeze(0).expand(size, -1)

        return torch.stack([self._draw_one_sample(generator) for _ in range(size)])

    # ── Hessian-vector product ────────────────────────────────────────────────

    def hvp(self, v) -> torch.Tensor:
        """Compute Cov[1_S] v via autograd (double backward)."""
        return compute_hvp(self._theta, self._n, _to_tensor(v, self._theta.dtype)).detach()

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self):
        return (f"ConditionalPoissonTorch(n={self._n}, N={self._N}, "
                f"log_Z={self.log_normalizer:.4f})")
