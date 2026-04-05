#!/usr/bin/env python3
"""
3D surface plots of timing data as f(N, n).

Reads timing_grid.json, produces one figure per experiment with
a 3D surface for each method.

Usage:
    python3 plot_timing_3d.py
    python3 plot_timing_3d.py timing_grid.json   # custom input
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata

COLORS = {
    # Z
    "DP forward":           "#E91E63",
    "NumPy tree":           "#5b9bd5",
    "PyTorch FFT":          "#c0504d",
    # Pi
    "Fwd-bwd DP":           "#E91E63",
    "N×DP loo":             "#aaa",
    "PyTorch FFT + autograd": "#c0504d",
}

COMPLEXITY = {
    "DP forward":             r"$O(Nn)$",
    "NumPy tree":             r"$O(N \log^2 N)$",
    "PyTorch FFT":            r"$O(N \log^2 n)$",
    "Fwd-bwd DP":             r"$O(Nn)$",
    "N×DP loo":               r"$O(N^2 n)$",
    "PyTorch FFT + autograd": r"$O(N \log^2 n)$",
}


def load_data(path):
    with open(path) as f:
        return json.load(f)


def plot_3d_experiment(data, experiment, title, out_path):
    """One figure with a 3D surface per method, overlaid."""
    methods = {}
    for d in data:
        if d["experiment"] != experiment:
            continue
        m = d["method"]
        methods.setdefault(m, {"N": [], "n": [], "t": []})
        methods[m]["N"].append(d["N"])
        methods[m]["n"].append(d["n"])
        methods[m]["t"].append(d["time_ms"])

    fig = plt.figure(figsize=(14, 5))

    # One subplot per method (side by side)
    n_methods = len(methods)
    sorted_methods = sorted(methods.keys())

    for idx, method in enumerate(sorted_methods):
        ax = fig.add_subplot(1, n_methods, idx + 1, projection='3d')

        pts = methods[method]
        N = np.array(pts["N"], dtype=float)
        n = np.array(pts["n"], dtype=float)
        t = np.array(pts["t"], dtype=float)

        logN = np.log10(N)
        logn = np.log10(n)
        logt = np.log10(t)

        # Create grid for surface
        Ni = np.linspace(logN.min(), logN.max(), 30)
        ni = np.linspace(logn.min(), logn.max(), 30)
        Ng, ng = np.meshgrid(Ni, ni)

        # Mask grid points where n >= N (invalid)
        mask = ng < Ng

        # Interpolate
        tg = griddata((logN, logn), logt, (Ng, ng), method='cubic')

        # Apply mask
        tg = np.where(mask, tg, np.nan)

        color = COLORS.get(method, "#999")
        surf = ax.plot_surface(Ng, ng, tg, alpha=0.7,
                               color=color, edgecolor='none',
                               antialiased=True)

        # Also plot the actual data points
        ax.scatter(logN, logn, logt, c='k', s=15, zorder=5, depthshade=False)

        complexity = COMPLEXITY.get(method, "")
        ax.set_title(f"{method}\n{complexity}", fontsize=9)
        ax.set_xlabel("log₁₀ N", fontsize=8)
        ax.set_ylabel("log₁₀ n", fontsize=8)
        ax.set_zlabel("log₁₀ ms", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=25, azim=-60)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_crosssections(data, experiment, title, out_path):
    """Fixed-n cross-sections: time vs N at several n values.
    This is the clearest way to verify slopes."""
    methods = {}
    for d in data:
        if d["experiment"] != experiment:
            continue
        m = d["method"]
        methods.setdefault(m, {})
        n = d["n"]
        methods[m].setdefault(n, {"N": [], "t": []})
        methods[m][n]["N"].append(d["N"])
        methods[m][n]["t"].append(d["time_ms"])

    sorted_methods = sorted(methods.keys())
    n_methods = len(sorted_methods)

    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4),
                             squeeze=False)

    # Pick a few n values for cross-sections
    all_ns = sorted(set(d["n"] for d in data if d["experiment"] == experiment))
    # Pick ~4 spread across range
    if len(all_ns) > 4:
        indices = np.linspace(0, len(all_ns) - 1, 4, dtype=int)
        cross_ns = [all_ns[i] for i in indices]
    else:
        cross_ns = all_ns

    cmap = plt.cm.viridis
    n_colors = {n: cmap(i / max(1, len(cross_ns) - 1))
                for i, n in enumerate(cross_ns)}

    for idx, method in enumerate(sorted_methods):
        ax = axes[0, idx]
        for n in cross_ns:
            if n not in methods[method]:
                continue
            pts = methods[method][n]
            Ns = np.array(pts["N"])
            ts = np.array(pts["t"])
            order = np.argsort(Ns)
            Ns, ts = Ns[order], ts[order]

            if len(Ns) >= 2:
                slope = np.polyfit(np.log(Ns), np.log(ts), 1)[0]
                label = f"n={n} (slope={slope:.2f})"
            else:
                label = f"n={n}"

            ax.loglog(Ns, ts, 'o-', color=n_colors[n], label=label,
                      lw=1.5, ms=5)

        complexity = COMPLEXITY.get(method, "")
        ax.set_title(f"{method}  {complexity}", fontsize=9)
        ax.set_xlabel("N")
        ax.set_ylabel("time (ms)")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.grid(True, alpha=0.15, which="both")

    fig.suptitle(f"{title} — fixed-n cross-sections", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "timing_grid.json"
    data = load_data(data_path)
    base = os.path.dirname(__file__)

    for experiment, title in [("Z", "Computing Z"), ("pi", "Computing π")]:
        plot_3d_experiment(
            data, experiment, title,
            os.path.join(base, f"timing_3d_{experiment}.png"))
        plot_crosssections(
            data, experiment, f"{title}",
            os.path.join(base, f"timing_cross_{experiment}.png"))


if __name__ == "__main__":
    main()
