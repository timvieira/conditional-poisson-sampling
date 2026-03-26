"""
conditional_poisson_torch.py
============================

Drop-in PyTorch replacement for ConditionalPoisson (conditional_poisson.py).

Uses FFT-based polynomial product tree with contour radius scaling for
O(N log² n) complexity and full autograd support.  Gradients (π), HVP
(Cov·v), and fitting come from torch.autograd — no hand-coded downward
pass or D-tree needed.

Supports an optional `items` parameter: a list of labels for the universe
elements.  When provided, pi returns a dict, sample returns lists of items,
and log_prob accepts item-valued subsets.

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
    ConditionalPoissonTorch(n, theta)                    direct from log-weights
    ConditionalPoissonTorch.from_weights(n, w, items=)   from positive weights
    ConditionalPoissonTorch.fit(pi_star, n, items=)      fit to target probs

    Properties
    ----------
    pi              inclusion probabilities (dict if items, else tensor)
    w               weights (= exp(theta))
    log_normalizer  log normalizing constant

    Methods
    -------
    log_prob(S)         log-probability of subset(s)
    sample(M)           draw M independent subsets
    hvp(v)              Cov[1_S] v
    """

    def __init__(self, n: int, theta, *, items=None,
                 dtype=torch.float64, device=None):
        """Construct from log-weights theta_i = log(w_i)."""
        self._theta = _to_tensor(theta, dtype).detach().clone().to(device=device)
        self._n = int(n)
        self._N = len(self._theta)
        self._items = items
        self._sample_tree = None  # lazily built on first sample() call

        if self._n < 0 or self._n > self._N:
            raise ValueError(f"n={self._n} must be in [0, {self._N}]")
        if items is not None and len(items) != self._N:
            raise ValueError(f"items has {len(items)} elements but theta has {self._N}")

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_weights(cls, n: int, w, *, items=None, **kw) -> "ConditionalPoissonTorch":
        """Construct from positive weights w_i."""
        w = _to_tensor(w, kw.get('dtype', torch.float64))
        if (w <= 0).any() or not torch.isfinite(w).all():
            raise ValueError("all weights must be finite and positive")
        with torch.no_grad():
            theta = torch.log(w)
        return cls(n, theta, items=items, **kw)

    @classmethod
    def fit(
        cls,
        pi_star,
        n: int,
        *,
        items=None,
        tol: float = 1e-7,
        max_iter: int = 200,
        verbose: bool = False,
        dtype=torch.float64,
        device=None,
    ) -> "ConditionalPoissonTorch":
        """
        Fit to target inclusion probabilities via L-BFGS.

        Parameters
        ----------
        pi_star : target inclusion probabilities (tensor, list, or dict)
                  All values must be in (0, 1) and sum to n.
        n       : subset size
        items   : optional item labels
        tol     : convergence tolerance on max|pi - pi_star|
        max_iter: maximum L-BFGS iterations
        verbose : print progress
        """
        if isinstance(pi_star, dict):
            if items is None:
                items = list(pi_star.keys())
            pi_star = torch.tensor([pi_star[k] for k in items], dtype=dtype, device=device)

        pi_star = _to_tensor(pi_star, dtype).to(device=device)

        # Warm start: logit(pi*) is asymptotically exact (Hájek 1964)
        theta = torch.log(pi_star / (1.0 - pi_star)).clone().requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [theta], lr=1.0, max_iter=20, history_size=20,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-14, tolerance_change=1e-16,
        )

        nit = [0]
        last_grad = [float('inf')]

        def closure():
            nit[0] += 1
            optimizer.zero_grad()
            log_Z = forward_log_Z(theta, n)
            loss = -(pi_star @ theta - log_Z)
            loss.backward()
            last_grad[0] = theta.grad.abs().max().item()
            return loss

        for _ in range(max_iter // 20 + 1):
            optimizer.step(closure)
            if verbose:
                print(f"  iter {nit[0]:3d}  max|pi - pi*| = {last_grad[0]:.2e}")
            if last_grad[0] < tol:
                break

        if last_grad[0] >= tol:
            import warnings
            warnings.warn(f"fit did not converge: max|pi - pi*| = {last_grad[0]:.2e} "
                          f"after {nit[0]} iterations (tol={tol})")

        with torch.no_grad():
            theta -= theta.mean()

        return cls(n, theta.detach(), items=items, dtype=dtype, device=device)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        """Subset size."""
        return self._n

    @property
    def N(self) -> int:
        """Universe size."""
        return self._N

    @property
    def theta(self) -> torch.Tensor:
        """Log-weights."""
        return self._theta.clone()

    @property
    def w(self) -> torch.Tensor:
        """Weights."""
        return torch.exp(self._theta)

    @property
    def pi(self):
        """Inclusion probabilities.

        Returns dict {item: prob} if items were provided, else (N,) tensor.
        """
        pi_val = compute_pi(self._theta, self._n).detach()
        if self._items is not None:
            return {item: pi_val[i].item() for i, item in enumerate(self._items)}
        return pi_val

    @property
    def log_normalizer(self) -> float:
        """Log normalizing constant."""
        return forward_log_Z(self._theta, self._n).item()

    # ── Log probability ───────────────────────────────────────────────────────

    def log_prob(self, S) -> Union[float, torch.Tensor]:
        """
        Log-probability of one subset or a batch.

        S may be:
          - list of items (if items were provided)
          - (n,) or (M, n) int tensor/list of indices
          - (N,) or (M, N) bool tensor

        Returns log P(S) = sum_{i in S} theta_i - log Z.
        """
        if self._items is not None and not isinstance(S, torch.Tensor):
            if isinstance(S, (list, tuple)) and len(S) > 0:
                item_to_idx = {item: i for i, item in enumerate(self._items)}
                if isinstance(S[0], (list, tuple)):
                    S = torch.tensor([[item_to_idx[x] for x in sub] for sub in S])
                elif not isinstance(S[0], int):
                    S = torch.tensor([item_to_idx[x] for x in S])

        S = _to_tensor(S, torch.long) if not isinstance(S, torch.Tensor) else S
        single = S.dim() == 1
        th = self._theta.detach()
        lz = self.log_normalizer

        if S.dtype == torch.bool:
            if S.dim() == 2:
                return (th @ S.float().T - lz)
            return float(th[S].sum() - lz)
        if S.dim() == 2:
            return torch.stack([th[row].sum() - lz for row in S])
        return float(th[S].sum() - lz)

    # ── Sampling ──────────────────────────────────────────────────────────────

    def _build_sample_tree(self):
        """Build the product tree for sampling (cached on first call).

        Each tree[node][k] = Z(w_T, k) for items in subtree T (up to
        a per-node scale factor that cancels in the sampling ratios).
        Uses torch FFT for polynomial multiplication with contour scaling.
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

    def sample(self, size: int = 1, rng: Optional[int] = None):
        """
        Draw independent samples.

        Parameters
        ----------
        size : number of subsets to draw
        rng  : random seed (int) or None

        Returns
        -------
        If items: list of lists of items
        If no items: (size, n) long tensor of sorted indices
        """
        generator = torch.Generator()
        if rng is not None:
            generator.manual_seed(rng)
        else:
            generator.seed()

        if self._sample_tree is None:
            self._build_sample_tree()

        if self._n == 0:
            raw = torch.empty(size, 0, dtype=torch.long)
        elif self._n == self._N:
            raw = torch.arange(self._N).unsqueeze(0).expand(size, -1)
        else:
            raw = torch.stack([self._draw_one_sample(generator) for _ in range(size)])

        if self._items is not None:
            return [[self._items[i] for i in row.tolist()] for row in raw]
        return raw

    # ── Hessian-vector product ────────────────────────────────────────────────

    def hvp(self, v) -> torch.Tensor:
        """Compute Cov[1_S] v via autograd (double backward)."""
        return compute_hvp(self._theta, self._n, _to_tensor(v, self._theta.dtype)).detach()

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self):
        return (f"ConditionalPoissonTorch(n={self._n}, N={self._N}, "
                f"log_Z={self.log_normalizer:.4f})")
