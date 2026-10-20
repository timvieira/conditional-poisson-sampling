#!/usr/bin/env python3
"""Draw the article's static timing plots from content/figures/timing.json.

Writes content/figures/timing_{Z,pi,samples}.svg, each a log-log plot of time
against N with n = 0.4 N (the "ratio" sweep of bench/bench_timing.py).
Empirical slopes are reported by the slope table in the article, which is
computed in the browser from the (N, n) grid in the same file.

Usage:
    python bench/plot_timing.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIGURES = REPO / "content" / "figures"

STYLE = {
    "tree":     dict(color="#c0504d", marker="^", ls="-",  lw=2.0, ms=6),
    "dp":       dict(color="#5b9bd5", marker="s", ls="-",  lw=2.0, ms=5),
    "tree_loo": dict(color="#c0504d", marker="^", ls="--", lw=1.3, ms=5, alpha=0.55),
    "dp_loo":   dict(color="#5b9bd5", marker="s", ls="--", lw=1.3, ms=4, alpha=0.55),
    "r":        dict(color="#FF9800", marker="o", ls="-.", lw=1.5, ms=5),
}

LABELS = {
    "Z": {
        "tree": r"product tree (PyTorch), $\mathcal{O}(N \log^2 n)$",
        "dp":   r"DP (NumPy), $\mathcal{O}(Nn)$",
    },
    "pi": {
        "tree":     r"product tree + autograd, $\mathcal{O}(N \log^2 n)$",
        "dp":       r"DP + reverse pass, $\mathcal{O}(Nn)$",
        "tree_loo": r"leave-one-out, $N \times$ tree, $\mathcal{O}(N^2 \log^2 n)$",
        "dp_loo":   r"leave-one-out, $N \times$ DP, $\mathcal{O}(N^2 n)$",
        "r":        r"R sampling package, $\mathcal{O}(Nn)$",
    },
    "sample": {
        "tree": r"product tree, $\mathcal{O}(n \log N)$ per sample",
        "dp":   r"DP table, $\mathcal{O}(N)$ per sample",
        "r":    r"R sampling package, $\mathcal{O}(N)$ per sample",
    },
}

TITLES = {
    "Z": r"Computing $Z$",
    "pi": r"Computing $\boldsymbol{\pi}$",
    "sample": r"Drawing One Sample (after setup)",
}

FILES = {"Z": "timing_Z.svg", "pi": "timing_pi.svg", "sample": "timing_samples.svg"}


def fmt_seconds(t, _pos):
    if t >= 1:
        return f"{t:g} s"
    if t >= 1e-3:
        return f"{t * 1e3:g} ms"
    return f"{t * 1e6:g} µs"


def plot(results, experiment):
    rows = [r for r in results if r["sweep"] == "ratio" and r["experiment"] == experiment]
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for method, label in LABELS[experiment].items():
        pts = sorted((r["N"], r["seconds"]) for r in rows
                     if r["method"] == method and r["seconds"] is not None)
        if not pts:
            continue
        Ns, ts = map(np.array, zip(*pts))
        ax.loglog(Ns, ts, label=label, **STYLE[method])
        failed = [r for r in rows if r["method"] == method and r.get("failure")]
        for r in failed:
            ax.annotate(f"overflow at $N = {r['N']}$", xy=(Ns[-1], ts[-1]), xytext=(8, -3),
                        textcoords="offset points", fontsize=8, color=STYLE[method]["color"])
    ax.set_xlabel(r"$N$ (with $n = 0.4N$)")
    ax.set_ylabel("time")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_seconds))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.grid(True, alpha=0.15, which="both")
    ax.set_title(TITLES[experiment], fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = FIGURES / FILES[experiment]
    fig.savefig(out, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    print(f"Wrote {out.relative_to(REPO)}")


def main():
    plt.rcParams["svg.hashsalt"] = "timing"      # deterministic SVG ids
    data = json.loads((FIGURES / "timing.json").read_text())
    for experiment in FILES:
        plot(data["results"], experiment)


if __name__ == "__main__":
    main()
