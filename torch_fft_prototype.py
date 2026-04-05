"""
torch_fft_prototype.py
======================

FFT-based polynomial product tree for conditional Poisson sampling.

The key numerical trick: **contour radius scaling**.  Before building the
polynomial product tree, rescale each weight w_i -> w_i * r, where r is
chosen so the peak of the product polynomial falls at degree n (the
coefficient we need).  This collapses the dynamic range from ~10^{-300}
to ~O(1), making standard float64 FFT accurate for the target coefficient.

The optimal r = exp(t) where t solves:

    sum_i  w_i * r / (1 + w_i * r)  =  n

This is the Lagrange multiplier from the Poisson sampling connection:
under Poisson sampling with probabilities p_i = w_i*r / (1 + w_i*r),
the expected sample size is n.  The conditional Poisson solver already
computes this parameter.

After multiplying the rescaled polynomials prod_i (1 + w_i*r*z) via FFT,
the n-th coefficient equals Z(w, n) * r^n.  We recover log Z as:

    log Z = log(root_rescaled[n]) - n * log(r)

Complexity: O(N log² n) with truncation to degree n.
Precision: full float64 — the coefficient at degree n is near the peak.
Autograd: fully compatible — rescaling + torch.fft are differentiable.
"""

import torch
import torch.nn.functional as F
import math


# ── Find the optimal rescaling parameter r ───────────────────────────────────

def _find_r(w, n, tol=1e-12, max_iter=100):
    """Find r such that sum_i w_i*r / (1 + w_i*r) = n.

    This is a monotone equation in log(r): the LHS is strictly increasing
    from 0 (as r->0) to N (as r->inf).  Newton's method converges quickly.

    Parameters
    ----------
    w : (N,) tensor of positive weights (detached, no grad needed)
    n : target sum (int)

    Returns
    -------
    r : scalar (float, not a tensor — this is a numerical parameter,
        not part of the differentiable computation)
    """
    # Work in log-space: let t = log(r), solve g(t) = sum p_i(t) - n = 0
    # where p_i(t) = w_i*exp(t) / (1 + w_i*exp(t)) = sigmoid(log(w_i) + t)
    log_w = torch.log(w).detach().double()
    N_items = len(w)

    # Initial guess: t such that n/N of the items have p_i ≈ 0.5
    # i.e., median(log_w) + t ≈ 0 => t ≈ -median(log_w)
    t = -torch.median(log_w).item()

    for _ in range(max_iter):
        s = log_w + t
        # Numerically stable sigmoid and its derivative
        # Clamp to avoid exp overflow in the tails
        s_clamped = s.clamp(-500, 500)
        p = torch.sigmoid(s_clamped)
        g = p.sum().item() - n
        gp = (p * (1 - p)).sum().item()

        if abs(g) < tol:
            break
        if gp < 1e-100:
            # All probabilities saturated; do a bisection-style step
            if g > 0:
                t -= 1.0
            else:
                t += 1.0
            continue

        t -= g / gp

    return math.exp(t)


# ── Batched FFT polynomial multiplication ────────────────────────────────────

def _batch_poly_mul_fft(a_batch, b_batch):
    """Multiply pairs of polynomials via batched FFT.

    O(B * d log d) where d = La + Lb.  Uses torch.fft which is
    differentiable, so autograd can backpropagate through this.

    Parameters
    ----------
    a_batch : (B, La) tensor
    b_batch : (B, Lb) tensor

    Returns
    -------
    (B, La + Lb - 1) tensor of product polynomials
    """
    La = a_batch.shape[1]
    Lb = b_batch.shape[1]
    n_out = La + Lb - 1
    # Convolution theorem: poly_mul(a, b) = ifft(fft(a) * fft(b))
    fa = torch.fft.rfft(a_batch, n=n_out)
    fb = torch.fft.rfft(b_batch, n=n_out)
    return torch.fft.irfft(fa * fb, n=n_out)


def _batch_poly_mul_direct(a_batch, b_batch):
    """Multiply pairs of polynomials via grouped conv1d (direct O(d²))."""
    B = a_batch.shape[0]
    Lb = b_batch.shape[1]
    out = F.conv1d(
        a_batch.unsqueeze(0),
        b_batch.flip(-1).unsqueeze(1),
        padding=Lb - 1,
        groups=B,
    )
    return out.squeeze(0)


# Threshold: use FFT when polynomials are large enough to amortize overhead.
# With contour scaling, FFT precision is no longer a concern.
_FFT_THRESHOLD = 64

def _batch_poly_mul(a_batch, b_batch):
    """Hybrid: direct for small polys, FFT for large."""
    if a_batch.shape[1] + b_batch.shape[1] <= _FFT_THRESHOLD:
        return _batch_poly_mul_direct(a_batch, b_batch)
    return _batch_poly_mul_fft(a_batch, b_batch)


# ── Forward pass: product tree with contour scaling ──────────────────────────

def forward_log_Z(theta, n):
    """Compute log Z via FFT-based product tree with contour scaling.

    Steps:
    1. Compute r = optimal contour radius (non-differentiable root-find)
    2. Rescale weights: w_i' = w_i * r  (differentiable)
    3. Build product tree: prod_i (1 + w_i' z) via batched FFT
    4. Extract: log Z = log(root[n]) - n * log(r)

    Parameters
    ----------
    theta : (N,) tensor of log-weights (requires_grad=True)
    n     : subset size (int)

    Returns
    -------
    log_Z : scalar tensor (differentiable w.r.t. theta)
    """
    N = theta.shape[0]
    dtype = theta.dtype
    device = theta.device

    w = torch.exp(theta)

    # Step 1: find optimal contour radius (not on autograd graph).
    #
    # Why this works: the product polynomial P(z) = prod_i (1 + w_i z) has
    # its peak coefficient near degree N/2 (for uniform weights).  The
    # coefficient at degree n can be ~10^{-300} smaller.  FFT introduces
    # errors ~eps * max|c|, which drowns the small coefficient.
    #
    # Rescaling z -> r*z gives P(r*z) = prod_i (1 + w_i*r*z).  The k-th
    # coefficient of P(r*z) is c_k * r^k, so choosing r shifts the peak.
    # The optimal r makes the expected sample size under Poisson sampling
    # equal n, placing the peak at degree n — exactly where we need it.
    #
    # r is NOT on the autograd graph: it's a numerical conditioning choice,
    # not part of the mathematical function.  The gradient of log_Z w.r.t.
    # theta flows through w_scaled = w * r (where r is a constant).
    r = _find_r(w.detach(), n)
    log_r = math.log(r)

    # Step 2: rescale weights (this IS on the autograd graph).
    # After rescaling, the product polynomial's peak is near degree n,
    # so FFT rounding errors are relative to the coefficient we need.
    w_scaled = w * r

    # Step 3: build leaf polynomials (1 + w_i' * z)
    ones = torch.ones(N, dtype=dtype, device=device)
    polys = torch.stack([ones, w_scaled], dim=1)  # (N, 2)

    # Per-polynomial log-scale for renormalization.
    # True polynomial = polys[i] * exp(log_scales[i]).
    # Keeps convolution inputs O(1) for numerical stability.
    log_scales = torch.zeros(N, dtype=dtype, device=device)

    # Bottom-up tree with batched multiplication
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

        left = polys[0::2]
        right = polys[1::2]

        # Truncate inputs to degree n (we only need coeff[0..n] of root)
        if left.shape[1] > n + 1:
            left = left[:, :n + 1]
        if right.shape[1] > n + 1:
            right = right[:, :n + 1]

        products = _batch_poly_mul(left, right)

        # Truncate output to degree n
        if products.shape[1] > n + 1:
            products = products[:, :n + 1]

        # Renormalize: scale to max|c| = 1
        new_scales = log_scales[0::2] + log_scales[1::2]
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

    root = polys[0]
    # root[n] * exp(log_scales[0]) = [z^n] prod(1 + w_i*r*z) = Z(w,n) * r^n
    # => log Z = log(root[n]) + log_scales[0] - n * log(r)
    log_Z = torch.log(root[n]) + log_scales[0] - n * log_r
    return log_Z


# ── Convenience wrappers (same interface as torch_prototype) ─────────────────

def compute_pi(theta, n):
    """Inclusion probabilities via autograd on log_Z.

    pi_i = d(log Z) / d(theta_i).  This is backpropagation (reverse-mode AD)
    applied to the forward pass — the Baur-Strassen theorem guarantees the
    cost is O(1)x the forward pass.  Griewank-Walther guarantees the numerical
    stability is inherited from the forward pass.
    """
    theta = theta.detach().requires_grad_(True)
    log_Z = forward_log_Z(theta, n)
    pi = torch.autograd.grad(log_Z, theta, create_graph=True)[0]
    return pi


def compute_hvp(theta, n, v):
    """Hessian-vector product Cov[1_S] v via double backward.

    Pearlmutter's R-operator: forward-mode AD applied to the backward pass
    (or equivalently, backward-mode applied to the gradient computation).
    Cost is O(1)x the gradient, i.e., O(1)x the forward pass.
    """
    theta = theta.detach().requires_grad_(True)
    log_Z = forward_log_Z(theta, n)
    grad = torch.autograd.grad(log_Z, theta, create_graph=True)[0]
    hv = torch.autograd.grad(grad, theta, grad_outputs=v)[0]
    return hv


# ══ Test and benchmark ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import numpy as np
    import time
    from conditional_poisson_numpy import ConditionalPoisson

    rng = np.random.default_rng(42)

    # ── Correctness ──
    print("Correctness (FFT with contour scaling):")
    for N in [50, 100, 500, 1000, 2000, 3000, 5000]:
        n = N * 2 // 5
        theta_np = rng.standard_normal(N) * 0.5
        v_np = rng.standard_normal(N)
        theta_t = torch.tensor(theta_np, dtype=torch.float64)
        v_t = torch.tensor(v_np, dtype=torch.float64)
        cp = ConditionalPoisson(n, theta_np)

        with torch.no_grad():
            lz_t = forward_log_Z(theta_t, n).item()
        lz_err = abs(cp.log_normalizer - lz_t)
        if np.isnan(lz_err):
            lz_err = float('inf')

        pi_t = compute_pi(theta_t, n).detach().numpy()
        pi_err = np.max(np.abs(cp.pi - pi_t))

        hv_t = compute_hvp(theta_t, n, v_t).detach().numpy()
        hv_err = np.max(np.abs(cp.hvp(v_np) - hv_t))

        print(f"  N={N:>5d}  logZ={lz_err:.1e}  pi={pi_err:.1e}  hvp={hv_err:.1e}")

    # ── Autograd check ──
    print("\nAutograd:")
    theta = torch.tensor(rng.standard_normal(200) * 0.5,
                         dtype=torch.float64, requires_grad=True)
    log_Z = forward_log_Z(theta, 80)
    pi = torch.autograd.grad(log_Z, theta, create_graph=True)[0]
    v = torch.randn(200, dtype=torch.float64)
    hv = torch.autograd.grad(pi, theta, grad_outputs=v)[0]
    print(f"  log_Z grad_fn: {log_Z.grad_fn is not None}")
    print(f"  pi grad_fn:    {pi.grad_fn is not None}")
    print(f"  HVP finite:    {torch.isfinite(hv).all().item()}")

    # ── Timing comparison ──
    print(f"\n{'N':>6} {'n':>5} | {'np pi':>8} {'fft pi':>8} {'direct pi':>10} | "
          f"{'np hvp':>8} {'fft hvp':>8}")
    print("-" * 75)

    from torch_prototype import compute_pi as compute_pi_direct
    from torch_prototype import compute_hvp as compute_hvp_direct

    for N in [50, 200, 500, 1000, 2000, 3000, 5000]:
        n = N * 2 // 5
        theta_np = rng.standard_normal(N) * 0.5
        v_np = rng.standard_normal(N)
        theta_t = torch.tensor(theta_np, dtype=torch.float64)
        v_t = torch.tensor(v_np, dtype=torch.float64)
        cp = ConditionalPoisson(n, theta_np)
        reps = max(3, 50 // max(1, N // 100))

        # NumPy
        for _ in range(3): cp._cache.clear(); cp.pi
        t0 = time.perf_counter()
        for _ in range(reps): cp._cache.clear(); cp.pi
        t_np_pi = (time.perf_counter() - t0) / reps

        _ = cp.pi
        for _ in range(3): cp.hvp(v_np)
        t0 = time.perf_counter()
        for _ in range(reps): cp.hvp(v_np)
        t_np_hv = (time.perf_counter() - t0) / reps

        # FFT (this file)
        for _ in range(3): compute_pi(theta_t, n)
        t0 = time.perf_counter()
        for _ in range(reps): compute_pi(theta_t, n)
        t_fft_pi = (time.perf_counter() - t0) / reps

        for _ in range(3): compute_hvp(theta_t, n, v_t)
        t0 = time.perf_counter()
        for _ in range(reps): compute_hvp(theta_t, n, v_t)
        t_fft_hv = (time.perf_counter() - t0) / reps

        # Direct conv1d (torch_prototype)
        for _ in range(3): compute_pi_direct(theta_t, n)
        t0 = time.perf_counter()
        for _ in range(reps): compute_pi_direct(theta_t, n)
        t_dir_pi = (time.perf_counter() - t0) / reps

        print(f"{N:>6d} {n:>5d} | {t_np_pi*1000:>6.1f}ms {t_fft_pi*1000:>6.1f}ms "
              f"{t_dir_pi*1000:>8.1f}ms | {t_np_hv*1000:>6.1f}ms {t_fft_hv*1000:>6.1f}ms")
