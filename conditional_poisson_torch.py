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
"""

import numpy as np
import torch
from typing import Optional, Union

from torch_fft_prototype import forward_log_Z, compute_pi, compute_hvp


def _preprocess_weights(w, n):
    """Separate boundary items from interior, validate, return mapping.

    Returns
    -------
    interior_idx : int array — indices of finite positive-weight items
    forced_in_idx : int array — indices of w=∞ items
    forced_out_idx : int array — indices of w=0 items
    interior_theta : tensor — log-weights for interior items only
    n_interior : int — reduced subset size (n minus forced-in count)
    items : list or None — original items (passed through)
    """
    if isinstance(w, np.ndarray):
        w = torch.tensor(w, dtype=torch.float64)
    w = w.double()

    forced_out_idx = torch.where(w == 0)[0].numpy()
    forced_in_idx = torch.where(torch.isinf(w))[0].numpy()
    interior_idx = torch.where(torch.isfinite(w) & (w > 0))[0].numpy()

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
    sample(M, rng)      draw M independent subsets
    hvp(v)              Cov[1_S] v
    """

    def __init__(self, n: int, theta, *, items=None,
                 dtype=torch.float64, device=None):
        """Construct from log-weights (interior items only, no boundary)."""
        if isinstance(theta, np.ndarray):
            theta = torch.tensor(theta, dtype=dtype, device=device)
        self._theta = theta.detach().clone().to(dtype=dtype, device=device)
        self._n = int(n)
        self._N_interior = len(self._theta)

        if self._n < 0 or self._n > self._N_interior:
            raise ValueError(f"n={self._n} must be in [0, {self._N_interior}]")

        # Item mapping (set by from_weights / fit, not by __init__ directly)
        self._items = items
        self._interior_idx = None
        self._forced_in_idx = np.array([], dtype=int)
        self._forced_out_idx = np.array([], dtype=int)
        self._N_full = self._N_interior  # overridden if boundary items exist

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

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_weights(cls, n: int, w, *, items=None, **kw) -> "ConditionalPoissonTorch":
        """Construct from non-negative weights w_i (0 and inf allowed)."""
        if isinstance(w, (list, tuple)):
            w = np.array(w, dtype=float)
        if isinstance(w, np.ndarray):
            w = torch.tensor(w, dtype=kw.get('dtype', torch.float64))
        if (w < 0).any():
            raise ValueError("all weights must be non-negative")

        N_full = len(w)
        if items is not None and len(items) != N_full:
            raise ValueError(f"items has {len(items)} elements but w has {N_full}")

        interior_idx, forced_in_idx, forced_out_idx, interior_theta, n_int = \
            _preprocess_weights(w, n)

        if n_int == 0:
            # Fully determined: all selected items are forced-in
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
        pi_star : target inclusion probabilities (tensor, array, or dict)
        n       : subset size
        items   : optional item labels
        tol     : convergence tolerance on max|pi - pi_star|
        max_iter: maximum L-BFGS iterations
        verbose : print progress
        """
        # Handle dict input
        if isinstance(pi_star, dict):
            if items is None:
                items = list(pi_star.keys())
            pi_star = np.array([pi_star[k] for k in items])

        if isinstance(pi_star, np.ndarray):
            pi_star = torch.tensor(pi_star, dtype=dtype, device=device)
        pi_star = pi_star.to(dtype=dtype, device=device)

        # Separate boundary items: pi=0 → forced out, pi=1 → forced in
        forced_in = torch.where(pi_star == 1.0)[0].numpy()
        forced_out = torch.where(pi_star == 0.0)[0].numpy()
        interior = torch.where((pi_star > 0) & (pi_star < 1))[0].numpy()

        pi_int = pi_star[interior]
        n_int = n - len(forced_in)

        if n_int == 0:
            # Fully determined
            cp = cls(0, torch.zeros(0, dtype=dtype), items=items, dtype=dtype, device=device)
            cp._with_boundary(interior, forced_in, forced_out, len(pi_star))
            return cp

        # Warm start: logit(pi*) is asymptotically exact (Hájek 1964)
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

    @theta.setter
    def theta(self, value):
        if isinstance(value, np.ndarray):
            value = torch.tensor(value, dtype=self._theta.dtype, device=self._theta.device)
        if value.shape != self._theta.shape:
            raise ValueError(f"theta must have shape {self._theta.shape}, got {value.shape}")
        self._theta = value.detach().clone()

    @property
    def w(self) -> torch.Tensor:
        """Weights for all items (including boundary)."""
        if not self._has_boundary:
            return torch.exp(self._theta)
        w_full = torch.zeros(self._N_full, dtype=self._theta.dtype)
        w_full[self._interior_idx] = torch.exp(self._theta)
        w_full[self._forced_in_idx] = float('inf')
        # forced_out stays 0
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

    def log_prob(self, S) -> Union[float, np.ndarray]:
        """
        Log-probability of one subset or a batch.

        S may be:
          - list of items (if items were provided)
          - int array (n,) or (M, n) of indices
          - bool array (N,) or (M, N)

        Returns -inf for impossible subsets.
        """
        # Convert item-valued input to indices
        if self._items is not None and not isinstance(S, np.ndarray):
            if isinstance(S, (list, tuple)) and len(S) > 0:
                if isinstance(S[0], (list, tuple)):
                    # Batch of item lists
                    item_to_idx = {item: i for i, item in enumerate(self._items)}
                    S = np.array([[item_to_idx[x] for x in sub] for sub in S])
                elif not isinstance(S[0], (int, np.integer)):
                    # Single item list
                    item_to_idx = {item: i for i, item in enumerate(self._items)}
                    S = np.array([item_to_idx[x] for x in S])

        S = np.asarray(S)
        single = S.ndim == 1

        if not self._has_boundary:
            th = self._theta.detach().cpu().numpy()
            lz = self.log_normalizer
            if S.dtype == bool:
                return (th @ S.T - lz) if S.ndim == 2 else float(th[S].sum() - lz)
            return (th[S].sum(axis=1) - lz) if S.ndim == 2 else float(th[S].sum() - lz)

        # Convert to indicators for boundary validation
        if S.dtype == bool:
            indicators = S if S.ndim == 2 else S[None, :]
        else:
            idx = S if S.ndim == 2 else S[None, :]
            indicators = np.zeros((idx.shape[0], self._N_full), dtype=bool)
            for m in range(idx.shape[0]):
                indicators[m, idx[m]] = True

        th_int = self._theta.detach().cpu().numpy()
        lz = self.log_normalizer

        results = np.full(indicators.shape[0], -np.inf)
        for m in range(indicators.shape[0]):
            ind = indicators[m]
            if not np.all(ind[self._forced_in_idx]):
                continue
            if np.any(ind[self._forced_out_idx]):
                continue
            if self._n == 0:
                results[m] = 0.0
            else:
                int_ind = ind[self._interior_idx]
                results[m] = float(th_int[int_ind].sum() - lz)

        return float(results[0]) if single else results

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample(
        self,
        size: int = 1,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ):
        """
        Draw independent samples.

        Returns
        -------
        If items: list of lists of items
        If no items: (size, n) int array of sorted indices
        """
        rng = np.random.default_rng(rng)

        if self._n == 0 or self._N_interior == 0:
            # Fully determined
            forced = np.sort(self._forced_in_idx)
            raw = np.tile(forced, (size, 1))
        else:
            # Sample interior items via NumPy implementation
            from conditional_poisson import ConditionalPoisson
            cp_np = ConditionalPoisson(self._n, self._theta.detach().cpu().numpy())
            int_samples = cp_np.sample(size, rng)

            if self._has_boundary:
                # Remap interior indices to full indices, merge with forced-in
                int_samples = self._interior_idx[int_samples]
                forced = np.tile(self._forced_in_idx, (size, 1))
                raw = np.concatenate([forced, int_samples], axis=1)
                raw.sort(axis=1)
            else:
                raw = int_samples

        if self._items is not None:
            return [[self._items[i] for i in row] for row in raw]
        return raw

    # ── Hessian-vector product ────────────────────────────────────────────────

    def hvp(self, v) -> torch.Tensor:
        """Compute Cov[1_S] v via autograd (double backward)."""
        if isinstance(v, np.ndarray):
            v = torch.tensor(v, dtype=self._theta.dtype, device=self._theta.device)

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
        n_total = self.n
        return (f"ConditionalPoissonTorch(n={n_total}, N={self._N_full}, "
                f"log_Z={self.log_normalizer:.4f})")
