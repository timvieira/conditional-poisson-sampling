"""Shared base class for PyTorch conditional Poisson implementations."""

from functools import cached_property
import torch
from conditional_poisson._contour import log_contour_radius
from conditional_poisson._fit import fit_theta


def _to_tensor(x, dtype=torch.float64):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.tensor(x, dtype=dtype)


class ConditionalPoissonTorchBase:
    """Base class for PyTorch implementations.

    Subclasses must implement:
        _circuit(self, theta) -> (log_Z, aux)
            log_Z: differentiable scalar tensor
            aux: arbitrary data needed by _sample_data and sample
        _sample_data (cached_property) -> data for sampling loop
        sample(self) -> tensor of indices
    """

    def __init__(self, n: int, theta, *, dtype=torch.float64, device=None):
        self.theta = _to_tensor(theta, dtype).to(device=device)
        self.n = int(n)
        self.N = len(self.theta)
        assert 0 <= self.n <= self.N

    @classmethod
    def from_weights(cls, n: int, w, **kw):
        w = _to_tensor(w, kw.get('dtype', torch.float64))
        if (w <= 0).any() or not torch.isfinite(w).all():
            raise ValueError("all weights must be finite and positive")
        return cls(n, torch.log(w), **kw)

    @classmethod
    def fit(cls, target_incl, n: int, *, tol: float = 1e-10, max_iter: int = 200,
            verbose: bool = False, dtype=torch.float64, device=None, **kw):
        """Fit theta so that every |pi_i - target_incl_i| <= tol.

        L-BFGS on the convex dual objective; see conditional_poisson._fit.  If
        it stops before reaching tol, it warns and returns the best point found.
        """
        def evaluate(theta):
            cp = cls(n, theta, dtype=dtype, device=device, **kw)
            return cp.log_normalizer, cp.incl_prob.cpu().double().numpy()

        target = _to_tensor(target_incl).detach().cpu().numpy()
        theta = fit_theta(evaluate, target, n, tol=tol, max_iter=max_iter, verbose=verbose)
        return cls(n, theta, dtype=dtype, device=device, **kw)

    def clear(self):
        """Flush all cached computations."""
        for attr in ('_forward', 'log_normalizer', 'incl_prob', '_sample_data', '_log_r'):
            self.__dict__.pop(attr, None)

    @cached_property
    def _log_r(self) -> float:
        """Log of the contour radius (see conditional_poisson._contour)."""
        return log_contour_radius(self.theta.detach().cpu().numpy(), self.n)

    def _circuit(self, theta):
        """Override: return (log_Z, aux) where log_Z is differentiable."""
        raise NotImplementedError

    @cached_property
    def _forward(self):
        """Cached forward pass with autograd-enabled theta."""
        theta = self.theta.detach().requires_grad_(True)
        log_Z, aux = self._circuit(theta)
        return log_Z, theta, aux

    @cached_property
    def log_normalizer(self) -> float:
        log_Z, _, _ = self._forward
        return log_Z.item()

    @cached_property
    def incl_prob(self) -> torch.Tensor:
        """Inclusion probabilities pi_i = d(log Z)/d(theta_i) via autograd."""
        log_Z, theta, _ = self._forward
        return torch.autograd.grad(log_Z, theta)[0].detach()

    def log_prob(self, S) -> float:
        S = torch.as_tensor(S, dtype=torch.long)
        return float(self.theta[S].sum() - self.log_normalizer)

    def sample(self):
        raise NotImplementedError
