#!/usr/bin/env python3
"""
Generate timing comparison plots from timing_data.json.

Produces four SVG figures:
  - content/figures/timing_Z.svg       — computing normalizing constant Z
  - content/figures/timing_pi.svg      — computing inclusion probabilities pi
  - content/figures/timing_fit.svg     — fitting target pi -> weights
  - content/figures/timing_samples.svg — drawing samples

Usage:
    python3 plot_timing.py
    python3 plot_timing.py timing_data.json   # custom input path
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Style ────────────────────────────────────────────────────────────────────

STYLE = {
    # Z experiment
    "DP forward":                {"color": "#E91E63", "marker": "s", "ls": "--", "lw": 1.5, "ms": 5},
    "NumPy tree":                {"color": "#5b9bd5", "marker": "o", "ls": "-",  "lw": 2,   "ms": 6},
    "PyTorch FFT":               {"color": "#c0504d", "marker": "^", "ls": "-",  "lw": 2,   "ms": 6},
    # Pi experiment
    "N×DP (leave-one-out)":      {"color": "#aaa",    "marker": "s", "ls": "--", "lw": 1.5, "ms": 5},
    "N×Tree (leave-one-out)":    {"color": "#bbb",    "marker": "D", "ls": "--", "lw": 1.5, "ms": 5},
    "Forward-backward DP":       {"color": "#E91E63", "marker": "s", "ls": "-",  "lw": 2,   "ms": 6},
    "R:sampling (pi)":           {"color": "#FF9800", "marker": "p", "ls": "-.", "lw": 1.5, "ms": 7},
    "NumPy tree + backprop":     {"color": "#5b9bd5", "marker": "o", "ls": "-",  "lw": 2,   "ms": 6},  # alias
    "PyTorch FFT + autograd":    {"color": "#c0504d", "marker": "^", "ls": "-",  "lw": 2,   "ms": 6},
    # Fit experiment
    "NumPy tree (Newton-CG)":    {"color": "#5b9bd5", "marker": "o", "ls": "-",  "lw": 2,   "ms": 6},
    "R:sampling (fit)":          {"color": "#FF9800", "marker": "p", "ls": "-.", "lw": 1.5, "ms": 7},
    # Samples experiment
    "R:sampling (1 sample, incl. DP)":      {"color": "#FF9800", "marker": "p", "ls": "-.", "lw": 1.5, "ms": 7},
    "R:sampling (1 sample, excl. DP)":      {"color": "#FF9800", "marker": "h", "ls": ":",  "lw": 1.5, "ms": 7},
    "NumPy tree (1 sample, incl. build)":   {"color": "#5b9bd5", "marker": "D", "ls": "--", "lw": 1.5, "ms": 5},
    "NumPy tree (1 sample)":     {"color": "#5b9bd5", "marker": "o", "ls": "--", "lw": 1.5, "ms": 5},
}

# Map method names in pi experiment that reuse Z names
PI_LABEL_MAP = {
    "NumPy tree": "NumPy tree + backprop",
}

# Complexity labels for the legend
COMPLEXITY = {
    # Z
    "DP forward":                r"$\mathcal{O}(Nn)$",
    "NumPy tree":                r"$\mathcal{O}(N \log^2 N)$",
    "PyTorch FFT":               r"$\mathcal{O}(N \log^2 n)$",
    # Pi
    "N×DP (leave-one-out)":      r"$\mathcal{O}(N^2 n)$",
    "N×Tree (leave-one-out)":    r"$\mathcal{O}(N^2 \log^2 n)$",
    "Forward-backward DP":       r"$\mathcal{O}(Nn)$",
    "R:sampling (pi)":           r"R sampling",
    "NumPy tree + backprop":     r"$\mathcal{O}(N \log^2 N)$",
    "PyTorch FFT + autograd":    r"$\mathcal{O}(N \log^2 n)$",
    # Fit
    "NumPy tree (Newton-CG)":    r"product tree + Newton-CG",
    "R:sampling (fit)":          r"R sampling (fixed-point)",
    # Samples
    "R:sampling (1 sample, incl. DP)":      r"R sampling (1 sample, incl. DP)",
    "R:sampling (1 sample, excl. DP)":      r"R sampling (1 sample, excl. DP)",
    "NumPy tree (1 sample, incl. build)":   r"product tree (1 sample, incl. build)",
    "NumPy tree (1 sample)":     r"product tree (1 sample, excl. build)",
}


def load_data(path):
    with open(path) as f:
        return json.load(f)


def plot_experiment(ax, data, experiment, label_map=None):
    """Plot all methods for one experiment on ax."""
    # Group by method
    methods = {}
    for d in data:
        if d["experiment"] != experiment:
            continue
        m = d["method"]
        if label_map and m in label_map:
            m = label_map[m]
        methods.setdefault(m, ([], []))
        methods[m][0].append(d["N"])
        methods[m][1].append(d["time_ms"])

    # Plot order: slow methods first (background), fast last (foreground)
    order = list(STYLE.keys())
    sorted_methods = sorted(methods.keys(),
                            key=lambda m: order.index(m) if m in order else 999)

    for m in sorted_methods:
        Ns, ts = methods[m]
        # Sort by N
        idx = np.argsort(Ns)
        Ns = [Ns[i] for i in idx]
        ts = [ts[i] for i in idx]

        s = STYLE.get(m, {"color": "gray", "marker": ".", "ls": "-", "lw": 1, "ms": 4})
        complexity = COMPLEXITY.get(m, "")
        label = f"{m}  {complexity}" if complexity else m
        ax.loglog(Ns, ts, marker=s["marker"], color=s["color"],
                  ls=s["ls"], lw=s["lw"], ms=s["ms"], label=label)

    ax.set_xlabel("$N$")
    ax.set_ylabel("time (ms)")
    ax.grid(True, alpha=0.15, which="both")
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def make_plot(data, experiment, title, out_path, label_map=None, figsize=(7, 4.5)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_experiment(ax, data, experiment, label_map=label_map)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "results/timing_data.json"
    data = load_data(data_path)

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dirs = [
        os.path.join(repo, "content", "figures"),
    ]

    plots = [
        ("Z", "Computing the Normalizing Constant $Z$", "timing_Z.svg", None),
        ("pi", "Computing Inclusion Probabilities $\\pi$", "timing_pi.svg", PI_LABEL_MAP),
        ("fit", "Fitting: Target $\\pi$ → Weights", "timing_fit.svg", None),
        ("samples", "Drawing Samples", "timing_samples.svg", None),
    ]

    for experiment, title, filename, label_map in plots:
        for out_dir in out_dirs:
            if os.path.isdir(out_dir):
                make_plot(data, experiment, title,
                          os.path.join(out_dir, filename),
                          label_map=label_map)


if __name__ == "__main__":
    main()
