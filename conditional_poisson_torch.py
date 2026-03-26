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

Boundary weights (w=0, w=∞) are handled by a preprocessing reduction:
strip forced-out/forced-in items, solve the reduced problem, remap results.

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
    ConditionalPoissonTorch.from_weights(n, w, items=)   from non-negative weights
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
        """Construct from log-weights (interior items only, no boundary)."""
        self._theta = _to_tensor(theta, dtype).detach().clone().to(device=device)
        self._n = int(n)
        self._N_interior = len(self._theta)

        if self._n < 0 or self._n > self._N_interior:
            raise ValueError(f"n={self._n} must be in [0, {self._N_interior}]")

        self._items = items
        self._interior_idx = torch.arange(self._N_interior, dtype=torch.long)
        self._forced_in_idx = torch.empty(0, dtype=torch.long)
        self._forced_out_idx = torch.empty(0, dtype=torch.long)
        self._N_full = self._N_interior
        self._sample_tree = None  # lazily built on first sample() call

    def _with_boundary(self, interior_idx, forced_in_idx, forced_out_idx, N_full):
        """Attach boundary mapping (called by factory methods)."""
        self._interior_idx = interior_idx
        self._forced_in_idx = forced_in_idx
        self._forced_out_idx = forced_out_idx
        self._N_full = N_full
        return self

    @property
    def _has_boundary(self):
        return len(self._forced_in_idx) > 0 or len(self._forced_out_idx) > 0

    # ── Preprocessing ─────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess_weights(w, n):
        """Separate boundary items from interior, validate, return mapping."""
        w = _to_tensor(w)

        forced_out_idx = torch.where(w == 0)[0]
        forced_in_idx = torch.where(torch.isinf(w))[0]
        interior_idx = torch.where(torch.isfinite(w) & (w > 0))[0]

        n_forced = len(forced_in_idx)
        n_interior = n - n_forced

        if n_interior < 0:
            raise ValueError(f"More forced-in items ({n_forced}) than n={n}")
        if n_interior > len(interior_idx):
            raise ValueError(f"Not enough interior items ({len(interior_idx)}) "
                             f"for n - forced_in = {n_interior}")

        with torch.no_grad():
            interior_theta = torch.log(w[interior_idx])

        return interior_idx, forced_in_idx, forced_out_idx, interior_theta, n_interior

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_weights(cls, n: int, w, *, items=None, **kw) -> "ConditionalPoissonTorch":
        """Construct from non-negative weights w_i (0 and inf allowed)."""
        w = _to_tensor(w, kw.get('dtype', torch.float64))
        if (w < 0).any():
            raise ValueError("all weights must be non-negative")

        N_full = len(w)
        if items is not None and len(items) != N_full:
            raise ValueError(f"items has {len(items)} elements but w has {N_full}")

        interior_idx, forced_in_idx, forced_out_idx, interior_theta, n_int = \
            cls._preprocess_weights(w, n)

        if n_int == 0:
            interior_theta = torch.zeros(0, dtype=w.dtype)

        cp = cls(n_int, interior_theta, items=items, **kw)
        cp._with_boundary(interior_idx, forced_in_idx, forced_out_idx, N_full)
        return cp

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

        forced_in = torch.where(pi_star == 1.0)[0]
        forced_out = torch.where(pi_star == 0.0)[0]
        interior = torch.where((pi_star > 0) & (pi_star < 1))[0]

        pi_int = pi_star[interior]
        n_int = n - len(forced_in)

        if n_int == 0:
            cp = cls(0, torch.zeros(0, dtype=dtype), items=items, dtype=dtype, device=device)
            cp._with_boundary(interior, forced_in, forced_out, len(pi_star))
            return cp

        theta = torch.log(pi_int / (1.0 - pi_int)).clone().requires_grad_(True)

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
            log_Z = forward_log_Z(theta, n_int)
            loss = -(pi_int @ theta - log_Z)
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

        cp = cls(n_int, theta.detach(), items=items, dtype=dtype, device=device)
        cp._with_boundary(interior, forced_in, forced_out, len(pi_star))
        return cp

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        """Subset size."""
        return self._n + len(self._forced_in_idx)

    @property
    def N(self) -> int:
        """Universe size (including boundary items)."""
        return self._N_full

    @property
    def theta(self) -> torch.Tensor:
        """Log-weights for interior items."""
        return self._theta.clone()

    @property
    def w(self) -> torch.Tensor:
        """Weights for all items (including boundary)."""
        if not self._has_boundary:
            return torch.exp(self._theta)
        w_full = torch.zeros(self._N_full, dtype=self._theta.dtype)
        w_full[self._interior_idx] = torch.exp(self._theta)
        w_full[self._forced_in_idx] = float('inf')
        return w_full

    @property
    def pi(self):
        """Inclusion probabilities.

        Returns dict {item: prob} if items were provided, else (N,) tensor.
        """
        pi_full = torch.zeros(self._N_full, dtype=self._theta.dtype)
        pi_full[self._forced_in_idx] = 1.0
        if self._n > 0 and self._N_interior > 0:
            pi_full[self._interior_idx if self._has_boundary
                    else slice(None)] = compute_pi(self._theta, self._n).detach()

        if self._items is not None:
            return {item: pi_full[i].item() for i, item in enumerate(self._items)}
        return pi_full

    @property
    def log_normalizer(self) -> float:
        """Log normalizing constant."""
        if self._n == 0 or self._N_interior == 0:
            return 0.0
        return forward_log_Z(self._theta, self._n).item()

    # ── Log probability ───────────────────────────────────────────────────────

    def log_prob(self, S) -> Union[float, torch.Tensor]:
        """
        Log-probability of one subset or a batch.

        S may be:
          - list of items (if items were provided)
          - (n,) or (M, n) int tensor/list of indices
          - (N,) or (M, N) bool tensor

        Returns -inf for impossible subsets.
        """
        # Convert item-valued input to indices
        if self._items is not None and not isinstance(S, torch.Tensor):
            if isinstance(S, (list, tuple)) and len(S) > 0:
                item_to_idx = {item: i for i, item in enumerate(self._items)}
                if isinstance(S[0], (list, tuple)):
                    S = torch.tensor([[item_to_idx[x] for x in sub] for sub in S])
                elif not isinstance(S[0], (int,)):
                    S = torch.tensor([item_to_idx[x] for x in S])

        S = _to_tensor(S, torch.long) if not isinstance(S, torch.Tensor) else S
        single = S.dim() == 1
        th = self._theta.detach()
        lz = self.log_normalizer

        if not self._has_boundary:
            if S.dtype == torch.bool:
                if S.dim() == 2:
                    return (th @ S.float().T - lz)
                return float(th[S].sum() - lz)
            if S.dim() == 2:
                return torch.stack([th[row].sum() - lz for row in S])
            return float(th[S].sum() - lz)

        # Boundary: convert to indicators, validate
        if S.dtype != torch.bool:
            idx = S if S.dim() == 2 else S.unsqueeze(0)
            indicators = torch.zeros(idx.shape[0], self._N_full, dtype=torch.bool)
            for m in range(idx.shape[0]):
                indicators[m, idx[m]] = True
        else:
            indicators = S if S.dim() == 2 else S.unsqueeze(0)

        results = torch.full((indicators.shape[0],), float('-inf'))
        for m in range(indicators.shape[0]):
            ind = indicators[m]
            if not ind[self._forced_in_idx].all():
                continue
            if ind[self._forced_out_idx].any():
                continue
            if self._n == 0:
                results[m] = 0.0
            else:
                results[m] = th[ind[self._interior_idx]].sum() - lz

        return float(results[0]) if single else results

    # ── Sampling ──────────────────────────────────────────────────────────────

    def _build_sample_tree(self):
        """Build the product tree for sampling (cached on first call).

        Uses torch FFT for polynomial multiplication with contour scaling.
        """
        N = self._N_interior
        n = self._n
        w = torch.exp(self._theta).detach()

        r = _find_r(w, n)
        w_scaled = w * r

        # Pad to next power of 2
        tree_n = 1 << (N - 1).bit_length()

        # tree[i] = polynomial tensor at node i (truncated to degree n)
        tree = [None] * (2 * tree_n)

        # Leaves
        for i in range(N):
            tree[tree_n + i] = torch.tensor([1.0, w_scaled[i].item()],
                                            dtype=self._theta.dtype)
        for i in range(N, tree_n):
            tree[tree_n + i] = torch.ones(1, dtype=self._theta.dtype)

        # Build bottom-up using batched FFT poly mul
        for i in range(tree_n - 1, 0, -1):
            left = tree[2 * i]
            right = tree[2 * i + 1]
            # Use the same FFT mul as forward_log_Z
            p = _batch_poly_mul(left.unsqueeze(0), right.unsqueeze(0)).squeeze(0)
            if len(p) > n + 1:
                p = p[:n + 1]
            # Renormalize to prevent overflow/underflow
            mx = p.abs().max()
            if mx > 0:
                p = p / mx
            tree[i] = p.detach()

        self._sample_tree = (tree, tree_n, N)

    def _draw_samples(self, size, generator):
        """Draw samples from the cached product tree via top-down quota splitting."""
        tree, tree_n, N = self._sample_tree
        n = self._n

        if n == 0:
            return torch.empty(size, 0, dtype=torch.long)
        if n == N:
            return torch.arange(N).unsqueeze(0).expand(size, -1)

        samples = torch.empty(size, n, dtype=torch.long)
        for m in range(size):
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
                left_poly = tree[2 * node]
                right_poly = tree[2 * node + 1]
                probs = torch.zeros(k + 1, dtype=self._theta.dtype)
                for j in range(k + 1):
                    rem = k - j
                    if j < len(left_poly) and rem < len(right_poly):
                        probs[j] = max(left_poly[j].item(), 0.0) * \
                                   max(right_poly[rem].item(), 0.0)
                total = probs.sum()
                if total <= 0:
                    continue
                probs /= total
                j = torch.multinomial(probs, 1, generator=generator).item()
                stack.append((2 * node + 1, k - j))
                stack.append((2 * node, j))
            samples[m] = torch.sort(torch.tensor(selected, dtype=torch.long))[0]

        return samples

    def sample(
        self,
        size: int = 1,
        rng: Optional[int] = None,
    ):
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

        if self._n == 0 or self._N_interior == 0:
            forced = torch.sort(self._forced_in_idx)[0]
            raw = forced.unsqueeze(0).expand(size, -1)
        else:
            if self._sample_tree is None:
                self._build_sample_tree()
            int_samples = self._draw_samples(size, generator)

            if self._has_boundary:
                int_samples = self._interior_idx[int_samples]
                forced = self._forced_in_idx.unsqueeze(0).expand(size, -1)
                raw = torch.cat([forced, int_samples], dim=1)
                raw = torch.sort(raw, dim=1)[0]
            else:
                raw = int_samples

        if self._items is not None:
            return [[self._items[i] for i in row.tolist()] for row in raw]
        return raw

    # ── Hessian-vector product ────────────────────────────────────────────────

    def hvp(self, v) -> torch.Tensor:
        """Compute Cov[1_S] v via autograd (double backward)."""
        v = _to_tensor(v, self._theta.dtype)

        result = torch.zeros(self._N_full, dtype=self._theta.dtype)
        if self._n > 0 and self._N_interior > 0:
            v_int = v[self._interior_idx] if self._has_boundary else v
            hvp_int = compute_hvp(self._theta, self._n, v_int).detach()
            if self._has_boundary:
                result[self._interior_idx] = hvp_int
            else:
                result = hvp_int
        return result

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self):
        return (f"ConditionalPoissonTorch(n={self.n}, N={self._N_full}, "
                f"log_Z={self.log_normalizer:.4f})")
