"""Empirical scaling of the tree-based P-tree build time."""

import numpy as np
import time
import pylab as pl
from conditional_poisson_numpy import ConditionalPoisson


def time_op(fn, min_reps=5, min_seconds=0.5):
    """Time a function, returning all measurements."""
    times = []
    for _ in range(min_reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    while sum(times) < min_seconds:
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.array(times)


def measure_build(Ns, ns_fn, label):
    """Measure tree build times for a parameter sweep."""
    rng = np.random.default_rng(0)
    results = []
    for N in Ns:
        n = ns_fn(N)
        w = rng.exponential(1.0, N)
        cp = ConditionalPoisson.from_weights(n, w)
        times = time_op(lambda: (cp._cache.clear(), cp._get_p_tree()))
        med = np.median(times) * 1000
        lo = np.percentile(times, 25) * 1000
        hi = np.percentile(times, 75) * 1000
        results.append((N, n, med, lo, hi))
        print(f'  {label}: N={N:5d} n={n:4d}  {med:8.2f} ms  [{lo:.2f}, {hi:.2f}]')
    return results


def fit_power(xs, ys):
    """Fit y = c * x^alpha, return (alpha, c)."""
    log_x = np.log(np.array(xs, float))
    log_y = np.log(np.array(ys, float))
    alpha, log_c = np.polyfit(log_x, log_y, 1)
    return alpha, np.exp(log_c)


# ── Collect data ──────────────────────────────────────────────────────────────

print('=== Vary N, fixed n=50 ===')
Ns_fixn = [50, 100, 200, 500, 1000, 2000]
res_fixn = measure_build(Ns_fixn, lambda N: 50, 'fix n')

print('\n=== Vary n, fixed N=1000 ===')
ns_vals = [10, 20, 50, 100, 200, 500]
ns_iter = iter(ns_vals)
res_fixN = measure_build([1000]*len(ns_vals), lambda N: next(ns_iter), 'fix N')

print('\n=== Vary N, n=N/5 ===')
Ns_prop = [50, 100, 200, 500, 1000, 2000]
res_prop = measure_build(Ns_prop, lambda N: max(N // 5, 2), 'prop')


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = pl.subplots(1, 3, figsize=(15, 4.5))

# Panel 1: vary N, fixed n
ax = axes[0]
Ns = [r[0] for r in res_fixn]
meds = [r[2] for r in res_fixn]
los = [r[3] for r in res_fixn]
his = [r[4] for r in res_fixn]
ax.errorbar(Ns, meds, yerr=[np.array(meds)-np.array(los), np.array(his)-np.array(meds)],
            fmt='o-', color='#2196F3', capsize=3, label='measured')
alpha, c = fit_power(Ns, meds)
ax.loglog(Ns, [c * N**alpha for N in Ns], '--', color='gray',
          label=f'$O(N^{{{alpha:.2f}}})$ fit')
ax.set_xlabel('N')
ax.set_ylabel('time (ms)')
ax.set_title(f'Vary N (n=50 fixed)')
ax.legend()
ax.set_xscale('log'); ax.set_yscale('log')

# Panel 2: vary n, fixed N
ax = axes[1]
ns = [r[1] for r in res_fixN]
meds = [r[2] for r in res_fixN]
los = [r[3] for r in res_fixN]
his = [r[4] for r in res_fixN]
ax.errorbar(ns, meds, yerr=[np.array(meds)-np.array(los), np.array(his)-np.array(meds)],
            fmt='o-', color='#E91E63', capsize=3, label='measured')
# Reference lines
alpha_n, c_n = fit_power(ns, meds)
ax.loglog(ns, [c_n * n**alpha_n for n in ns], '--', color='gray',
          label=f'$O(n^{{{alpha_n:.2f}}})$ fit')
# Show what O(log^2 n) and O(n) would look like
log2ns = np.array([np.log2(n) for n in ns])
c_log = np.median(np.array(meds) / log2ns**2)
ax.loglog(ns, [c_log * l**2 for l in log2ns], ':', color='orange',
          alpha=0.6, label=r'$O(\log^2 n)$ ref')
ax.set_xlabel('n')
ax.set_ylabel('time (ms)')
ax.set_title(f'Vary n (N=1000 fixed)')
ax.legend(fontsize=8)
ax.set_xscale('log'); ax.set_yscale('log')

# Panel 3: vary N, n=N/5
ax = axes[2]
Ns = [r[0] for r in res_prop]
ns_p = [r[1] for r in res_prop]
meds = [r[2] for r in res_prop]
los = [r[3] for r in res_prop]
his = [r[4] for r in res_prop]
ax.errorbar(Ns, meds, yerr=[np.array(meds)-np.array(los), np.array(his)-np.array(meds)],
            fmt='o-', color='#4CAF50', capsize=3, label='measured')
alpha_p, c_p = fit_power(Ns, meds)
ax.loglog(Ns, [c_p * N**alpha_p for N in Ns], '--', color='gray',
          label=f'$O(N^{{{alpha_p:.2f}}})$ fit')
# O(N (log N)^2) reference
log2Ns = np.array([np.log2(N) for N in Ns])
c_nlog = np.median(np.array(meds) / (np.array(Ns, float) * log2Ns**2))
ax.loglog(Ns, [c_nlog * N * l**2 for N, l in zip(Ns, log2Ns)],
          ':', color='orange', alpha=0.6, label=r'$O(N \log^2 N)$ ref')
# O(N^2) reference
c_n2 = meds[0] / Ns[0]**2
ax.loglog(Ns, [c_n2 * N**2 for N in Ns], ':', color='red',
          alpha=0.4, label=r'$O(N^2)$ ref')
ax.set_xlabel('N')
ax.set_ylabel('time (ms)')
ax.set_title(r'Vary N ($n = N/5$)')
ax.legend(fontsize=8)
ax.set_xscale('log'); ax.set_yscale('log')

pl.suptitle('P-tree build time scaling', fontsize=14)
pl.tight_layout()
pl.savefig('bench_scaling.png', dpi=150, bbox_inches='tight')
print('\nSaved bench_scaling.png')
pl.close()
