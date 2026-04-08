"""
Product-tree implementation of the conditional Poisson distribution (PyTorch).

Uses FFT-based polynomial product tree with contour radius scaling for
O(N log^2 n) complexity and full autograd support.

The key numerical trick: **contour radius scaling**.  Rescale each
weight w_i -> w_i * r, where r is chosen so the peak of the product
polynomial falls at degree n.  This collapses the dynamic range from
~10^{-300} to ~O(1), making standard float64 FFT accurate for the
target coefficient.

Pure PyTorch internally — no numpy or scipy dependency.
"""

import torch
import torch.nn.functional as F
import math
from typing import Union
from functools import cached_property


def _to_tensor(x, dtype=torch.float64):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.tensor(x, dtype=dtype)


_FFT_THRESHOLD = 64


class ConditionalPoissonTorch:
    """Conditional Poisson distribution over fixed-size subsets (PyTorch).

        P(S) ∝ prod_{i in S} w_i,   |S| = n

    Constructors: __init__(n, theta), from_weights(n, w), fit(target_incl, n)
    Properties:   incl_prob, log_normalizer, n, N, theta
    Methods:      log_prob(S), sample(), clear()
    """

    def __init__(self, n: int, theta, *, dtype=torch.float64, device=None):
        self.theta = _to_tensor(theta, dtype).detach().clone().to(device=device)
        self.n = int(n)
        self.N = len(self.theta)
        assert 0 <= self.n <= self.N

    @classmethod
    def from_weights(cls, n: int, w, **kw) -> "ConditionalPoissonTorch":
        w = _to_tensor(w, kw.get('dtype', torch.float64))
        if (w <= 0).any() or not torch.isfinite(w).all():
            raise ValueError("all weights must be finite and positive")
        return cls(n, torch.log(w), **kw)

    @classmethod
    def fit(cls, target_incl, n: int, *, tol: float = 1e-7,
            dtype=torch.float64, device=None, **kw) -> "ConditionalPoissonTorch":
        """Fit to target inclusion probabilities via L-BFGS."""
        target_incl = _to_tensor(target_incl, dtype).to(device=device)
        cp = cls(n, torch.logit(target_incl), dtype=dtype, device=device, **kw)
        cp.theta.requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [cp.theta],
            max_iter=200, history_size=5,
            line_search_fn='strong_wolfe',
            tolerance_grad=tol,
            tolerance_change=0,
        )

        def closure():
            optimizer.zero_grad()
            cp.__dict__.pop('_r', None)  # r depends on theta
            tree, tree_n, log_r, root_log_scale, _ = cp._build_forward()
            loss = torch.log(tree[1][n]) + root_log_scale - n * log_r - target_incl @ cp.theta
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            cp.theta -= cp.theta.mean()
        cp.theta = cp.theta.detach()
        return cp

    def clear(self):
        """Flush all cached computations."""
        for attr in ('_r', '_forward', 'log_normalizer', 'incl_prob', '_sample_data'):
            self.__dict__.pop(attr, None)

    @cached_property
    def _forward(self):
        """Differentiable forward pass: builds tree, computes log Z (cached).

        Returns (log_Z_tensor, tree, tree_n, node_scale, theta_grad).
        theta_grad is the requires_grad copy used for the build.
        """
        theta_grad = self.theta.detach().requires_grad_(True)
        saved = self.theta
        self.theta = theta_grad
        tree, tree_n, log_r, root_log_scale, node_scale = self._build_forward()
        self.theta = saved
        log_Z = torch.log(tree[1][self.n]) + root_log_scale - self.n * log_r
        return log_Z, tree, tree_n, node_scale, theta_grad

    @cached_property
    def log_normalizer(self) -> float:
        """Log normalizing constant."""
        log_Z, _, _, _, _ = self._forward
        return float(log_Z.item())

    @cached_property
    def incl_prob(self) -> torch.Tensor:
        """Inclusion probabilities pi_i = d(log Z)/d(theta_i) via autograd."""
        log_Z, _, _, _, theta_grad = self._forward
        return torch.autograd.grad(log_Z, theta_grad)[0].detach()

    def log_prob(self, S) -> Union[float, torch.Tensor]:
        """log P(S) = sum_{i in S} theta_i - log Z."""
        S = _to_tensor(S, torch.long) if not isinstance(S, torch.Tensor) else S
        th = self.theta.detach()
        lz = self.log_normalizer
        if S.dtype == torch.bool:
            if S.dim() == 2:
                return th @ S.float().T - lz
            return float(th[S].sum() - lz)
        if S.dim() == 2:
            return torch.stack([th[row].sum() - lz for row in S])
        return float(th[S].sum() - lz)

    @cached_property
    def _sample_data(self):
        """Convert cached tree to plain lists for fast sampling loop."""
        _, tree, tree_n, node_scale, _ = self._forward
        return (
            [t.detach().tolist() if t is not None else [] for t in tree],
            node_scale, tree_n,
        )

    def sample(self) -> torch.Tensor:
        """Draw one sample via top-down quota splitting.

        Complexity: O(N log^2 n) to build tree [cached] + O(n log N).
        """
        import random
        Pc, ratio, tree_n = self._sample_data
        N, n = self.N, self.n
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
            L = Pc[2 * node]
            R = Pc[2 * node + 1]
            u = random.random() * Pc[node][k] * ratio[node]
            acc = 0.0
            for j in range(k + 1):
                lv = L[j] if j < len(L) else 0.0
                rv = R[k - j] if k - j < len(R) else 0.0
                acc += lv * rv
                if acc >= u:
                    break
            stack.append((2 * node + 1, k - j))
            stack.append((2 * node, j))
        selected.sort()
        return torch.tensor(selected, dtype=torch.long)

    @cached_property
    def _r(self):
        """Optimal contour radius (cached).  Not on the autograd graph."""
        log_w = self.theta.detach().double()
        n = self.n
        t = -torch.median(log_w).item()
        for _ in range(100):
            p = torch.sigmoid(log_w + t)
            g = p.sum().item() - n
            gp = (p * (1 - p)).sum().item()
            if abs(g) < 1e-12:
                break
            if gp < 1e-100:
                t += -1.0 if g > 0 else 1.0
                continue
            t -= g / gp
        return math.exp(t)

    @staticmethod
    def _batch_poly_mul_fft(a_batch, b_batch):
        La, Lb = a_batch.shape[1], b_batch.shape[1]
        n_out = La + Lb - 1
        fa = torch.fft.rfft(a_batch, n=n_out)
        fb = torch.fft.rfft(b_batch, n=n_out)
        return torch.fft.irfft(fa * fb, n=n_out)

    @staticmethod
    def _batch_poly_mul_direct(a_batch, b_batch):
        B, Lb = a_batch.shape[0], b_batch.shape[1]
        return F.conv1d(
            a_batch.unsqueeze(0),
            b_batch.flip(-1).unsqueeze(1),
            padding=Lb - 1, groups=B,
        ).squeeze(0)

    @classmethod
    def _batch_poly_mul(cls, a_batch, b_batch):
        if a_batch.shape[1] + b_batch.shape[1] <= _FFT_THRESHOLD:
            return cls._batch_poly_mul_direct(a_batch, b_batch)
        return cls._batch_poly_mul_fft(a_batch, b_batch)

    def _build_forward(self):
        """Build the product tree via batched FFT with contour scaling.

        Returns (tree, tree_n, log_r, root_log_scale, node_scale).
        """
        n, N = self.n, self.N
        theta = self.theta
        dtype, device = theta.dtype, theta.device

        w = torch.exp(theta)
        r = self._r
        log_r = math.log(r)
        w_scaled = w * r

        tree_n = 1 << max(1, (N - 1).bit_length())
        tree = [None] * (2 * tree_n)
        node_scale = [1.0] * (2 * tree_n)

        polys = torch.ones(tree_n, 2, dtype=dtype, device=device)
        polys[:N, 1] = w_scaled
        polys[N:, 1] = 0.0
        for i in range(tree_n):
            tree[tree_n + i] = polys[i]

        scales = torch.zeros(tree_n, dtype=dtype, device=device)
        level_size = tree_n
        while level_size > 1:
            left, right = polys[0::2], polys[1::2]
            if left.shape[1] > n + 1: left = left[:, :n + 1]
            if right.shape[1] > n + 1: right = right[:, :n + 1]

            products = self._batch_poly_mul(left, right)
            if products.shape[1] > n + 1: products = products[:, :n + 1]

            new_scales = scales[0::2] + scales[1::2]
            max_abs = products.abs().max(dim=1).values.clamp(min=1e-300)
            products = products / max_abs.unsqueeze(1)
            new_scales = new_scales + torch.log(max_abs)

            level_size //= 2
            for i in range(level_size):
                tree[level_size + i] = products[i]
                node_scale[level_size + i] = max_abs[i].item()

            polys, scales = products, new_scales

        return tree, tree_n, log_r, scales[0], node_scale

    def __repr__(self):
        return f"ConditionalPoissonTorch(n={self.n}, N={self.N})"
