"""Product-tree conditional Poisson distribution (PyTorch).

Uses FFT-based polynomial product tree with contour radius scaling for
O(N log^2 n) complexity.  The contour radius r rescales weights so the
peak of the product polynomial falls at degree n, making float64 FFT
accurate for the target coefficient.
"""

from functools import cached_property
import torch
import torch.nn.functional as F
import math
from conditional_poisson._base_torch import ConditionalPoissonTorchBase


_FFT_THRESHOLD = 64


class ConditionalPoissonTorch(ConditionalPoissonTorchBase):
    """Product-tree O(N log^2 n) via batched FFT (PyTorch).

    Constructors: __init__(n, theta), from_weights(n, w), fit(target_incl, n)
    Properties:   incl_prob, log_normalizer, n, N, theta
    Methods:      log_prob(S), sample(), clear()
    """

    def clear(self):
        super().clear()
        self.__dict__.pop('_r', None)

    @cached_property
    def _r(self):
        """Optimal contour radius (cached)."""
        log_w = self.theta.detach().double()
        t = -torch.median(log_w).item()
        for _ in range(100):
            p = torch.sigmoid(log_w + t)
            g = p.sum().item() - self.n
            gp = (p * (1 - p)).sum().item()
            if abs(g) < 1e-12:
                break
            if gp < 1e-100:
                t += -1.0 if g > 0 else 1.0
                continue
            t -= g / gp
        return math.exp(t)

    def _circuit(self, theta):
        """Product tree with contour scaling.  Returns (log_Z, (tree, tree_n, node_scale))."""
        n, N = self.n, self.N
        dtype, device = theta.dtype, theta.device

        w = torch.exp(theta)
        r = self._r
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
            m = products.max(dim=1).values
            products = products / m.unsqueeze(1)
            new_scales = new_scales + torch.log(m)
            level_size //= 2
            for i in range(level_size):
                tree[level_size + i] = products[i]
                node_scale[level_size + i] = m[i].item()
            polys, scales = products, new_scales

        log_Z = torch.log(polys[0][n]) + scales[0] - n * math.log(r)
        return log_Z, (tree, tree_n, node_scale)

    @cached_property
    def _sample_data(self):
        _, _, (tree, tree_n, node_scale) = self._forward
        return tree, node_scale, tree_n

    def sample(self) -> torch.Tensor:
        """Top-down quota splitting on the product tree.  O(n log N)."""
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
            u = torch.rand(1).item() * Pc[node][k].item() * ratio[node]
            acc = 0.0
            for j in range(k + 1):
                lv = L[j].item() if j < len(L) else 0.0
                rv = R[k - j].item() if k - j < len(R) else 0.0
                acc += lv * rv
                if acc >= u:
                    break
            stack.append((2 * node + 1, k - j))
            stack.append((2 * node, j))
        selected.sort()
        return torch.tensor(selected, dtype=torch.long)

    @staticmethod
    def _batch_poly_mul_fft(a_batch, b_batch):
        La, Lb = a_batch.shape[1], b_batch.shape[1]
        n_out = La + Lb - 1
        return torch.fft.irfft(
            torch.fft.rfft(a_batch, n=n_out) * torch.fft.rfft(b_batch, n=n_out),
            n=n_out,
        )

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

    def __repr__(self):
        return f"ConditionalPoissonTorch(n={self.n}, N={self.N})"
