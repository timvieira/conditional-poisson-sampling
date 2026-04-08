"""Shared base class for PyTorch conditional Poisson implementations."""

from functools import cached_property
import torch


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
    def fit(cls, target_incl, n: int, *, tol: float = 1e-7,
            dtype=torch.float64, device=None, **kw):
        """Fit to target inclusion probabilities via L-BFGS."""
        target_incl = _to_tensor(target_incl, dtype).to(device=device)
        cp = cls(n, torch.logit(target_incl).clone(), dtype=dtype, device=device, **kw)
        cp.theta.requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [cp.theta],
            max_iter=200, history_size=5,
            line_search_fn='strong_wolfe',
            tolerance_grad=tol, tolerance_change=0,
        )

        def closure():
            optimizer.zero_grad()
            cp.clear()
            log_Z, _ = cp._circuit(cp.theta)
            loss = log_Z - target_incl @ cp.theta
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            cp.theta -= cp.theta.mean()
        cp.theta = cp.theta.detach()
        return cp

    def clear(self):
        """Flush all cached computations."""
        for attr in ('_forward', 'log_normalizer', 'incl_prob', '_sample_data'):
            self.__dict__.pop(attr, None)

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
