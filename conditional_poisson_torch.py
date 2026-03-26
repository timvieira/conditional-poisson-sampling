"""
conditional_poisson_torch.py
============================

Drop-in PyTorch replacement for ConditionalPoisson (conditional_poisson.py).

Uses FFT-based polynomial product tree with contour radius scaling for
O(N log² n) complexity and full autograd support.  Gradients (π), HVP
(Cov·v), and fitting come from torch.autograd — no hand-coded downward
pass or D-tree needed.

API is intentionally parallel to ConditionalPoisson so the two can be
swapped with minimal code changes.  Key differences:
  - Parameters and outputs are torch tensors, not numpy arrays
  - Autograd-compatible: log_normalizer and pi are differentiable w.r.t. theta
  - Fitting uses L-BFGS (gradient only) instead of Newton-CG (gradient + HVP)
"""

import numpy as np
import torch
from typing import Optional, Union

from torch_fft_prototype import forward_log_Z, compute_pi, compute_hvp


class ConditionalPoissonTorch:
    """
    Conditional Poisson distribution over fixed-size subsets (PyTorch).

        P(S) ∝ prod_{i in S} w_i,   |S| = n

    Construction
    ------------
    ConditionalPoissonTorch(n, theta)              direct from log-weights
    ConditionalPoissonTorch.uniform(N, n)          uniform inclusion probs n/N
    ConditionalPoissonTorch.from_weights(n, w)     from non-negative weights
    ConditionalPoissonTorch.fit(pi_star, n)        fit to target inclusion probs

    Properties (recomputed each access via autograd)
    ----------
    pi              (N,) inclusion probabilities
    w               (N,) weights (= exp(theta))
    log_normalizer  log normalizing constant

    Methods
    -------
    log_prob(S)         scalar or (M,) log-probabilities
    sample(M, rng)      (M, n) int array of sorted subsets
    hvp(v)              Cov[1_S] v  (positive semi-definite)
    """

    def __init__(self, n: int, theta, *, dtype=torch.float64, device=None):
        if isinstance(theta, np.ndarray):
            theta = torch.tensor(theta, dtype=dtype, device=device)
        self._theta = theta.detach().clone().to(dtype=dtype, device=device)
        self._n = int(n)
        self._N = len(self._theta)

        if self._n < 0 or self._n > self._N:
            raise ValueError(f"n={self._n} must be in [0, {self._N}]")

        # Handle boundary items: w=0 (forced out) and w=inf (forced in)
        w = torch.exp(self._theta)
        self._forced_out = torch.where(w == 0)[0]
        self._forced_in = torch.where(torch.isinf(w))[0]
        self._interior = torch.where(torch.isfinite(w) & (w > 0))[0]

        n_forced = len(self._forced_in)
        n_remaining = self._n - n_forced

        if n_remaining < 0:
            raise ValueError(f"More forced-in items ({n_forced}) than n={self._n}")
        if n_remaining > len(self._interior):
            raise ValueError(f"Not enough interior items ({len(self._interior)}) "
                             f"for n - forced_in = {n_remaining}")

        # Build reduced problem on interior items
        if len(self._forced_in) > 0 or len(self._forced_out) > 0:
            if n_remaining > 0 and len(self._interior) > 0:
                self._reduced = ConditionalPoissonTorch(
                    n_remaining, self._theta[self._interior],
                    dtype=dtype, device=device)
            else:
                self._reduced = None  # fully determined
        else:
            self._reduced = None  # no boundary items

        self._has_boundary = len(self._forced_in) > 0 or len(self._forced_out) > 0

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def uniform(cls, N: int, n: int, **kw) -> "ConditionalPoissonTorch":
        """Uniform: all weights equal, pi_i = n/N."""
        return cls(n, torch.zeros(N, dtype=kw.get('dtype', torch.float64)), **kw)

    @classmethod
    def from_weights(cls, n: int, w, **kw) -> "ConditionalPoissonTorch":
        """Construct from non-negative weights w_i (0 and inf allowed)."""
        if isinstance(w, np.ndarray):
            w = torch.tensor(w, dtype=kw.get('dtype', torch.float64))
        if (w < 0).any():
            raise ValueError("all weights must be non-negative")
        with torch.no_grad():
            theta = torch.log(w.to(dtype=kw.get('dtype', torch.float64)))
        return cls(n, theta, **kw)

    @classmethod
    def fit(
        cls,
        pi_star,
        n: int,
        *,
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
        pi_star : (N,) target inclusion probabilities (must sum to n)
        n       : subset size
        tol     : convergence tolerance on max|pi - pi_star|
        max_iter: maximum L-BFGS iterations
        verbose : print progress

        Returns
        -------
        ConditionalPoissonTorch with fitted weights
        """
        if isinstance(pi_star, np.ndarray):
            pi_star = torch.tensor(pi_star, dtype=dtype, device=device)
        pi_star = pi_star.to(dtype=dtype, device=device)

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
            # grad = -(pi_star - pi), so |grad|_inf = max|pi - pi_star|
            last_grad[0] = theta.grad.abs().max().item()
            return loss

        for outer in range(max_iter // 20 + 1):
            optimizer.step(closure)
            if verbose:
                print(f"  iter {nit[0]:3d}  max|pi - pi*| = {last_grad[0]:.2e}")
            if last_grad[0] < tol:
                break

        if last_grad[0] >= tol:
            import warnings
            warnings.warn(f"fit did not converge: max|pi - pi*| = {last_grad[0]:.2e} "
                          f"after {nit[0]} iterations (tol={tol})")

        # Zero-center theta (shift invariance)
        with torch.no_grad():
            theta -= theta.mean()

        return cls(n, theta.detach(), dtype=dtype, device=device)

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

    @theta.setter
    def theta(self, value):
        if isinstance(value, np.ndarray):
            value = torch.tensor(value, dtype=self._theta.dtype, device=self._theta.device)
        if value.shape != self._theta.shape:
            raise ValueError(f"theta must have shape {self._theta.shape}, got {value.shape}")
        self._theta = value.detach().clone()

    @property
    def w(self) -> torch.Tensor:
        return torch.exp(self._theta)

    @property
    def pi(self) -> torch.Tensor:
        """Inclusion probabilities pi_i = P(i in S)."""
        if self._has_boundary:
            pi = torch.zeros(self._N, dtype=self._theta.dtype, device=self._theta.device)
            pi[self._forced_in] = 1.0
            if self._reduced is not None:
                pi[self._interior] = self._reduced.pi
            return pi
        return compute_pi(self._theta, self._n).detach()

    @property
    def log_normalizer(self) -> float:
        """Log normalizing constant."""
        if self._has_boundary:
            if self._reduced is not None:
                return self._reduced.log_normalizer
            return 0.0  # fully determined
        return forward_log_Z(self._theta, self._n).item()

    # ── Log probability ───────────────────────────────────────────────────────

    def log_prob(self, S) -> Union[float, np.ndarray]:
        """
        Log-probability of one subset or a batch.

            log P(S) = sum_{i in S} theta_i  -  log Z

        S may be:
          int array (n,)     single subset (item indices)
          bool array (N,)    single subset (indicator)
          int array (M, n)   batch of M subsets
          bool array (M, N)  batch of M subsets

        Returns -inf for impossible subsets.
        """
        S = np.asarray(S)
        th = self._theta.detach().cpu().numpy()
        lz = self.log_normalizer

        if not self._has_boundary:
            if S.dtype == bool:
                return (th @ S.T - lz) if S.ndim == 2 else float(th[S].sum() - lz)
            else:
                return (th[S].sum(axis=1) - lz) if S.ndim == 2 else float(th[S].sum() - lz)

        # Boundary items: validate
        single = S.ndim == 1
        if S.dtype == bool:
            indicators = S if S.ndim == 2 else S[None, :]
        else:
            idx = S if S.ndim == 2 else S[None, :]
            indicators = np.zeros((idx.shape[0], self._N), dtype=bool)
            for m in range(idx.shape[0]):
                indicators[m, idx[m]] = True

        forced_in_np = self._forced_in.cpu().numpy()
        forced_out_np = self._forced_out.cpu().numpy()
        interior_np = self._interior.cpu().numpy()

        results = np.full(indicators.shape[0], -np.inf)
        for m in range(indicators.shape[0]):
            ind = indicators[m]
            if not np.all(ind[forced_in_np]):
                continue
            if np.any(ind[forced_out_np]):
                continue
            if self._reduced is None:
                results[m] = 0.0
            else:
                results[m] = self._reduced.log_prob(ind[interior_np])

        return float(results[0]) if single else results

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample(
        self,
        size: int = 1,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> np.ndarray:
        """
        Draw independent samples using top-down tree walk.

        Parameters
        ----------
        size : number of subsets to draw
        rng  : seed or np.random.Generator

        Returns
        -------
        (size, n) int array; each row is a sorted list of n item indices.

        Uses the NumPy sampler internally (sampling is inherently sequential
        and doesn't benefit from torch/GPU).
        """
        rng = np.random.default_rng(rng)

        if self._has_boundary:
            if self._reduced is not None:
                int_samples = self._reduced.sample(size, rng)
                int_indices = self._interior.cpu().numpy()
                int_samples = int_indices[int_samples]
                forced_tile = np.tile(self._forced_in.cpu().numpy(), (size, 1))
                merged = np.concatenate([forced_tile, int_samples], axis=1)
                merged.sort(axis=1)
                return merged
            return np.tile(np.sort(self._forced_in.cpu().numpy()), (size, 1))

        # Delegate to the NumPy implementation's tree sampler — sampling is
        # inherently sequential and doesn't benefit from torch/GPU.
        from conditional_poisson import ConditionalPoisson
        cp_np = ConditionalPoisson(self._n, self._theta.detach().cpu().numpy())
        return cp_np.sample(size, rng)

    # ── Hessian-vector product ────────────────────────────────────────────────

    def hvp(self, v) -> torch.Tensor:
        """Compute Cov[1_S] v via autograd (double backward)."""
        if isinstance(v, np.ndarray):
            v = torch.tensor(v, dtype=self._theta.dtype, device=self._theta.device)

        if self._has_boundary:
            result = torch.zeros(self._N, dtype=self._theta.dtype, device=self._theta.device)
            if self._reduced is not None:
                result[self._interior] = self._reduced.hvp(v[self._interior])
            return result

        return compute_hvp(self._theta, self._n, v).detach()

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self):
        return (f"ConditionalPoissonTorch(n={self._n}, N={self._N}, "
                f"log_Z={self.log_normalizer:.4f})")
