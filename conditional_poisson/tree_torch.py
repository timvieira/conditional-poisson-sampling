"""Product-tree conditional Poisson distribution (PyTorch).

Uses FFT-based polynomial product tree with contour radius scaling for
O(N log^2 n) complexity.  The contour radius r rescales weights so the
peak of the product polynomial falls at degree n, making float64 FFT
accurate for the target coefficient.
"""

from bisect import bisect_right
from functools import cached_property
from itertools import accumulate
import torch
import torch.nn.functional as F
from scipy.fft import next_fast_len
from conditional_poisson._base_torch import ConditionalPoissonTorchBase


# Below this output length, schoolbook multiplication is exact and about as
# fast as FFT; above it, FFT wins.
_FFT_THRESHOLD = 16


class ConditionalPoissonTorch(ConditionalPoissonTorchBase):
    """Product-tree O(N log^2 n) via batched FFT (PyTorch).

    Constructors: __init__(n, theta), from_weights(n, w), fit(target_incl, n)
    Properties:   incl_prob, log_normalizer, n, N, theta
    Methods:      log_prob(S), sample(), clear()
    """

    def _circuit(self, theta):
        """Product tree with contour scaling.  Returns (log_Z, levels).

        levels[d] holds the polynomials of the nodes at depth d (root first),
        one row per node, truncated to degree n and each divided by its largest
        coefficient.  Each level multiplies adjacent pairs of nodes in the level
        below; if that level has an odd number of nodes, its last node is
        multiplied by the constant polynomial 1, which is not stored.  So the
        children of node i at depth d are nodes 2i and 2i + 1 at depth d + 1
        (the second may be missing), and with the leaves at depth D, node i
        covers items i 2^(D-d), ..., (i + 1) 2^(D-d) - 1, up to item N - 1.
        Every level is computed with one batched multiplication; there is no
        per-node Python work.
        """
        n, N = self.n, self.N
        log_r = self._log_r

        # Leaves are (1 + w_i r z).  With N = 0 there is one leaf, the constant 1.
        polys = torch.zeros(max(N, 1), 2, dtype=theta.dtype, device=theta.device)
        polys[:, 0] = 1.0
        polys[:N, 1] = torch.exp(theta + log_r)
        log_scale = torch.zeros(len(polys), dtype=theta.dtype, device=theta.device)
        levels = [polys]
        while len(polys) > 1:
            if len(polys) % 2:                  # pair the last node with the constant 1
                one = polys.new_zeros(1, polys.shape[1])
                one[0, 0] = 1.0
                polys = torch.cat([polys, one])
                log_scale = torch.cat([log_scale, log_scale.new_zeros(1)])
            products = self._batch_poly_mul(polys[0::2, :n + 1], polys[1::2, :n + 1])
            products = products[:, :n + 1]
            # Divide each node by its largest coefficient so nothing overflows.
            # log Z does not depend on the divisor, so it is detached from autograd.
            m = products.detach().amax(dim=1)
            polys = products / m.unsqueeze(1)
            log_scale = log_scale[0::2] + log_scale[1::2] + torch.log(m)
            levels.append(polys)
        levels.reverse()

        log_Z = torch.log(polys[0, n]) + log_scale[0] - n * log_r
        return log_Z, levels

    @cached_property
    def _sample_data(self):
        """Tree levels as zero-copy 2-D memoryviews (fast scalar reads), and
        an upper bound on the number of splits in one sample."""
        _, _, levels = self._forward
        P = [lv.detach().cpu().numpy().data for lv in levels]
        D = len(P) - 1
        # At depth d, at most min(2^d, n) nodes receive a nonzero quota.
        max_splits = sum(min(1 << d, self.n) for d in range(D))
        return P, max_splits

    def sample(self) -> torch.Tensor:
        """Top-down quota splitting on the product tree.  O(n log N) per sample."""
        P, max_splits = self._sample_data
        N, n = self.N, self.n
        D = len(P) - 1                          # leaves are at depth D
        u = iter(torch.rand(max_splits, dtype=torch.float64).tolist())
        selected = []
        stack = [(0, 0, n)] if n > 0 else []    # (depth, index within level, quota)
        while stack:
            d, i, k = stack.pop()
            if d == D:                          # a real leaf with quota 1
                selected.append(i)
                continue
            # Give j items to the left child and k - j to the right.  Each child
            # can take at most its number of items (see _circuit).  A missing
            # right child has none, so then j = k and Q[b] is never read.
            size = 1 << (D - d - 1)
            n_left = min(max(N - 2 * i * size, 0), size)
            n_right = min(max(N - (2 * i + 1) * size, 0), size)
            lo, hi = max(0, k - n_right), min(k, n_left)
            a, b = 2 * i, 2 * i + 1
            Q = P[d + 1]
            if lo == hi:
                j = lo
            elif k == 1:                        # the common case near the leaves
                left, right = Q[a, 1] * Q[b, 0], Q[a, 0] * Q[b, 1]
                j = 1 if next(u) * (left + right) < left else 0
            else:
                cdf = list(accumulate(Q[a, j] * Q[b, k - j] for j in range(lo, hi + 1)))
                j = lo + min(bisect_right(cdf, next(u) * cdf[-1]), hi - lo)
            if j:
                stack.append((d + 1, a, j))
            if k - j:
                stack.append((d + 1, b, k - j))
        selected.sort()
        return torch.tensor(selected, dtype=torch.long)

    @staticmethod
    def _batch_poly_mul_fft(a_batch, b_batch):
        n_out = a_batch.shape[1] + b_batch.shape[1] - 1
        # Most products here have 2^k + 1 coefficients, and FFTs of lengths with
        # large prime factors (8193 = 3 * 2731) are slow, so pad to a length
        # with only small prime factors.
        L = next_fast_len(n_out, real=True)
        return torch.fft.irfft(
            torch.fft.rfft(a_batch, n=L) * torch.fft.rfft(b_batch, n=L),
            n=L,
        )[:, :n_out]

    @staticmethod
    def _batch_poly_mul_direct(a_batch, b_batch):
        """Schoolbook multiplication: one shifted multiply-add per coefficient of a."""
        La = a_batch.shape[1]
        out = 0
        for i in range(La):
            out = out + F.pad(a_batch[:, i:i + 1] * b_batch, (i, La - 1 - i))
        return out

    @classmethod
    def _batch_poly_mul(cls, a_batch, b_batch):
        if a_batch.shape[1] + b_batch.shape[1] <= _FFT_THRESHOLD:
            return cls._batch_poly_mul_direct(a_batch, b_batch)
        return cls._batch_poly_mul_fft(a_batch, b_batch)

    def __repr__(self):
        return f"ConditionalPoissonTorch(n={self.n}, N={self.N})"
