"""
torch_prototype.py
==================

PyTorch autograd prototype for the conditional Poisson forward pass.

Validates that:
1. torch.autograd.grad on log_Z gives inclusion probabilities (matching the
   hand-coded downward pass)
2. torch.autograd.functional.vhp on log_Z gives the Hessian-vector product
   (matching the hand-coded D-tree)
"""

import torch
import numpy as np
import time


def poly_mul(a, b):
    """Multiply two 1-D polynomials via conv1d.

    a, b: 1-D tensors of polynomial coefficients.
    Returns 1-D tensor of length len(a) + len(b) - 1.
    """
    # conv1d expects (batch, channels, length)
    # We flip b for convolution (conv1d does cross-correlation)
    out = torch.nn.functional.conv1d(
        a.view(1, 1, -1),
        b.flip(0).view(1, 1, -1),
        padding=len(b) - 1,
    )
    return out.view(-1)


def forward_log_Z(theta, n):
    """Compute log Z via binary tree of polynomial multiplications.

    Parameters
    ----------
    theta : 1-D tensor of log-weights (requires_grad=True)
    n     : subset size

    Returns
    -------
    log_Z : scalar tensor (differentiable)
    """
    N = theta.shape[0]

    # Geometric-mean normalisation
    log_gm = theta.mean()
    q = torch.exp(theta - log_gm)

    # Build leaf polynomials: (1 + q_i * z)
    polys = []
    for i in range(N):
        polys.append(torch.stack([torch.ones(1, dtype=theta.dtype, device=theta.device).squeeze(), q[i]]))

    # Bottom-up tree multiplication
    while len(polys) > 1:
        new_polys = []
        for i in range(0, len(polys), 2):
            if i + 1 < len(polys):
                new_polys.append(poly_mul(polys[i], polys[i + 1]))
            else:
                new_polys.append(polys[i])
        polys = new_polys

    root = polys[0]
    log_Z = torch.log(root[n]) + n * log_gm
    return log_Z


def compute_pi_autograd(theta_np, n):
    """Compute inclusion probabilities via autograd on log_Z."""
    theta = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)
    log_Z = forward_log_Z(theta, n)
    pi = torch.autograd.grad(log_Z, theta, create_graph=True)[0]
    return pi.detach().numpy()


def compute_hvp_autograd(theta_np, n, v_np):
    """Compute Hessian-vector product via torch.autograd.functional.vhp."""
    theta = torch.tensor(theta_np, dtype=torch.float64)
    v = torch.tensor(v_np, dtype=torch.float64)

    def f(th):
        return forward_log_Z(th, n)

    # vhp returns (f(theta), v^T H) — but H is symmetric so v^T H = H v
    _, hvp = torch.autograd.functional.vhp(f, theta, v)
    return hvp.numpy()


# ══ Comparison script ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    from conditional_poisson import ConditionalPoisson

    np.random.seed(42)
    N, n = 50, 20
    theta = np.random.randn(N) * 0.5

    cp = ConditionalPoisson(n, theta)

    # ── Inclusion probabilities ──────────────────────────────────────────────

    t0 = time.perf_counter()
    pi_numpy = cp.pi
    t_numpy_pi = time.perf_counter() - t0

    t0 = time.perf_counter()
    pi_torch = compute_pi_autograd(theta, n)
    t_torch_pi = time.perf_counter() - t0

    err_pi = np.max(np.abs(pi_numpy - pi_torch))
    print(f"Inclusion probabilities (pi)")
    print(f"  NumPy  time: {t_numpy_pi:.4f}s")
    print(f"  Torch  time: {t_torch_pi:.4f}s")
    print(f"  Max |error|: {err_pi:.2e}")
    print()

    # ── Hessian-vector product ───────────────────────────────────────────────

    v = np.random.randn(N)

    t0 = time.perf_counter()
    hvp_numpy = cp.hvp(v)
    t_numpy_hvp = time.perf_counter() - t0

    t0 = time.perf_counter()
    hvp_torch = compute_hvp_autograd(theta, n, v)
    t_torch_hvp = time.perf_counter() - t0

    err_hvp = np.max(np.abs(hvp_numpy - hvp_torch))
    print(f"Hessian-vector product (Hv)")
    print(f"  NumPy  time: {t_numpy_hvp:.4f}s")
    print(f"  Torch  time: {t_torch_hvp:.4f}s")
    print(f"  Max |error|: {err_hvp:.2e}")
    print()

    # ── Sanity checks ────────────────────────────────────────────────────────

    print(f"pi sum (should be {n}): numpy={pi_numpy.sum():.6f}, torch={pi_torch.sum():.6f}")
    print(f"Hv . 1 (should be ~0):  numpy={hvp_numpy.sum():.2e}, torch={hvp_torch.sum():.2e}")
