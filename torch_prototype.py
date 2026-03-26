"""
torch_prototype.py
==================

PyTorch implementation of the conditional Poisson forward pass.

Only the forward pass (tree-structured polynomial multiplication) is
implemented explicitly.  Everything else comes from autograd:
  - Inclusion probabilities: backprop on log Z  (Baur & Strassen)
  - Hessian-vector products: backprop on pi      (Pearlmutter)

The forward pass is numerically stable (per-level renormalization with
truncation to degree n), so Griewank-Walther guarantees that the
autograd-computed derivatives are also stable.

Performance: batched grouped conv1d reduces O(N) torch calls to O(log N)
per tree level.  Polynomials are truncated to degree n (we only need
the n-th coefficient of the root), bounding memory and preventing
high-degree coefficients from drowning out the signal.

Complexity: O(Nn) — conv1d uses direct convolution (oneDNN/MKLDNN on
CPU, never FFT).  Same asymptotic cost as the naive DP, but faster in
practice due to oneDNN's vectorized kernels and batched execution.
Recovering O(N log² n) requires numerically stable FFT-based polynomial
multiplication (open problem: FFT rounding errors corrupt small
coefficients after renormalization).
"""

import torch
import torch.nn.functional as F


def _batch_poly_mul(a_batch, b_batch):
    """Multiply pairs of polynomials in a batch via grouped conv1d.

    Parameters
    ----------
    a_batch : (B, La) tensor — left polynomials
    b_batch : (B, Lb) tensor — right polynomials

    Returns
    -------
    (B, La + Lb - 1) tensor of product polynomials
    """
    B = a_batch.shape[0]
    Lb = b_batch.shape[1]
    # grouped conv1d: input (1, B, La), weight (B, 1, Lb)
    out = F.conv1d(
        a_batch.unsqueeze(0),
        b_batch.flip(-1).unsqueeze(1),
        padding=Lb - 1,
        groups=B,
    )
    return out.squeeze(0)


def forward_log_Z(theta, n):
    """Compute log Z via batched binary tree of polynomial multiplications.

    Each level of the tree renormalizes polynomial coefficients to have
    max |c| = 1, tracking the scale factor through the log domain.
    This keeps all conv1d inputs O(1) while remaining differentiable.

    Parameters
    ----------
    theta : (N,) tensor of log-weights (requires_grad=True for differentiation)
    n     : subset size (int)

    Returns
    -------
    log_Z : scalar tensor (differentiable)
    """
    N = theta.shape[0]
    dtype = theta.dtype
    device = theta.device

    # Geometric-mean normalisation (keeps polynomial coefficients O(1))
    log_gm = theta.mean()
    q = torch.exp(theta - log_gm)

    # Leaf polynomials: (1 + q_i * z), shape (N, 2)
    ones = torch.ones(N, dtype=dtype, device=device)
    polys = torch.stack([ones, q], dim=1)  # (N, 2)

    # Track log-scale per polynomial: true poly = polys[i] * exp(log_scales[i])
    log_scales = torch.zeros(N, dtype=dtype, device=device)

    # Bottom-up: batch all multiplications at each level
    while polys.shape[0] > 1:
        B = polys.shape[0]
        if B % 2 == 1:
            leftover_poly = polys[-1:]
            leftover_scale = log_scales[-1:]
            polys = polys[:-1]
            log_scales = log_scales[:-1]
            B -= 1
        else:
            leftover_poly = None
            leftover_scale = None

        left = polys[0::2]                 # (B//2, L)
        right = polys[1::2]                # (B//2, L)

        # Truncate inputs to degree n before multiplying: coeff[k] of the
        # product for k <= n depends only on coeff[0..n] of each factor.
        # This bounds polynomial size, prevents FFT from wasting precision
        # on high-degree coefficients we'll never use, and gives O(N log²n).
        if left.shape[1] > n + 1:
            left = left[:, :n + 1]
        if right.shape[1] > n + 1:
            right = right[:, :n + 1]

        products = _batch_poly_mul(left, right)  # (B//2, ≤ 2n+1)

        # Truncate output to degree n
        if products.shape[1] > n + 1:
            products = products[:, :n + 1]

        # Combined scales: left_scale + right_scale
        new_scales = log_scales[0::2] + log_scales[1::2]

        # Renormalize: divide out max|c| per polynomial, track in log_scales
        max_abs = products.abs().max(dim=1).values.clamp(min=1e-300)
        products = products / max_abs.unsqueeze(1)
        new_scales = new_scales + torch.log(max_abs)

        if leftover_poly is not None:
            pad_size = products.shape[1] - leftover_poly.shape[1]
            if pad_size > 0:
                leftover_poly = F.pad(leftover_poly, (0, pad_size))
            products = torch.cat([products, leftover_poly], dim=0)
            new_scales = torch.cat([new_scales, leftover_scale], dim=0)

        polys = products
        log_scales = new_scales

    root = polys[0]                        # (N+1,) polynomial
    log_Z = torch.log(root[n]) + log_scales[0] + n * log_gm
    return log_Z


def compute_pi(theta, n):
    """Compute inclusion probabilities via autograd.

    Parameters
    ----------
    theta : (N,) tensor of log-weights
    n     : subset size

    Returns
    -------
    pi : (N,) tensor of inclusion probabilities
    """
    theta = theta.detach().requires_grad_(True)
    log_Z = forward_log_Z(theta, n)
    pi = torch.autograd.grad(log_Z, theta, create_graph=True)[0]
    return pi


def compute_hvp(theta, n, v):
    """Compute Hessian-vector product Cov[1_S] v via double backward.

    Parameters
    ----------
    theta : (N,) tensor of log-weights
    n     : subset size
    v     : (N,) tensor direction

    Returns
    -------
    hv : (N,) tensor
    """
    theta = theta.detach().requires_grad_(True)
    log_Z = forward_log_Z(theta, n)
    grad = torch.autograd.grad(log_Z, theta, create_graph=True)[0]
    hv = torch.autograd.grad(grad, theta, grad_outputs=v)[0]
    return hv


# ══ Comparison script ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import numpy as np
    import time
    from conditional_poisson import ConditionalPoisson

    np.random.seed(42)

    for N, n in [(50, 20), (200, 80), (500, 200), (1000, 400)]:
        theta_np = np.random.randn(N) * 0.5
        v_np = np.random.randn(N)
        theta_t = torch.tensor(theta_np, dtype=torch.float64)
        v_t = torch.tensor(v_np, dtype=torch.float64)

        cp = ConditionalPoisson(n, theta_np)

        # ── Correctness ──
        pi_np = cp.pi
        pi_t = compute_pi(theta_t, n).detach().numpy()
        err_pi = np.max(np.abs(pi_np - pi_t))

        hv_np = cp.hvp(v_np)
        hv_t = compute_hvp(theta_t, n, v_t).detach().numpy()
        err_hv = np.max(np.abs(hv_np - hv_t))

        # ── Timing (warm start) ──
        reps = max(3, 100 // max(1, N // 50))

        # NumPy pi
        for _ in range(3):
            cp._cache.clear(); cp.pi
        t0 = time.perf_counter()
        for _ in range(reps):
            cp._cache.clear(); cp.pi
        t_np_pi = (time.perf_counter() - t0) / reps

        # Torch pi
        for _ in range(3):
            compute_pi(theta_t, n)
        t0 = time.perf_counter()
        for _ in range(reps):
            compute_pi(theta_t, n)
        t_t_pi = (time.perf_counter() - t0) / reps

        # NumPy hvp
        _ = cp.pi  # ensure tree cached
        for _ in range(3):
            cp.hvp(v_np)
        t0 = time.perf_counter()
        for _ in range(reps):
            cp.hvp(v_np)
        t_np_hv = (time.perf_counter() - t0) / reps

        # Torch hvp
        for _ in range(3):
            compute_hvp(theta_t, n, v_t)
        t0 = time.perf_counter()
        for _ in range(reps):
            compute_hvp(theta_t, n, v_t)
        t_t_hv = (time.perf_counter() - t0) / reps

        print(f"N={N:>5d}, n={n:>4d}  |  "
              f"pi: np={t_np_pi*1000:.1f}ms  torch={t_t_pi*1000:.1f}ms  "
              f"err={err_pi:.1e}  |  "
              f"hvp: np={t_np_hv*1000:.1f}ms  torch={t_t_hv*1000:.1f}ms  "
              f"err={err_hv:.1e}")
