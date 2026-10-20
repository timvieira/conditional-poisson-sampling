#!/usr/bin/env python3
"""Build the article's timing figures from content/figures/timing.json.

Writes content/figures/timing_plots.js, which sets window.TIMING_PLOTS to

    figures   Plotly figures, drawn in the article by plotly.js:
              "Z", "pi", "sample"   time against N with n = 0.4 N (the "ratio"
                                    sweep of bench/bench_timing.py)
              "Z_grid", "pi_grid", "sample_grid"
                                    time over the (N, n) grid, in 3D
    slopes    the cells of the article's slope table

The article includes this file with a <script> tag rather than fetching it, so
that the site's build rewrites its path for every page that shows the article.

Plotly cannot typeset LaTeX with the page's MathJax (CHTML output), so the
figures contain no math: their titles, the x-axis label of the line plots, and
the complexity bounds are HTML in the article.  The figures' heights are set
there too, by CSS, so that they can change with the screen width.

Usage:
    python bench/plot_timing.py
"""

import json
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

REPO = Path(__file__).resolve().parent.parent
FIGURES = REPO / "content" / "figures"
DATA = FIGURES / "timing.json"
OUT = FIGURES / "timing_plots.js"

RED, BLUE, ORANGE = "#c0504d", "#5b9bd5", "#e69500"
STYLE = {
    "tree":     dict(name="product tree", color=RED, symbol="triangle-up", dash="solid", width=2, size=8),
    "dp":       dict(name="DP", color=BLUE, symbol="square", dash="solid", width=2, size=7),
    "tree_loo": dict(name="leave-one-out (tree)", color=RED, symbol="triangle-up", dash="dash",
                     width=1.5, size=7, opacity=0.55),
    "dp_loo":   dict(name="leave-one-out (DP)", color=BLUE, symbol="square", dash="dash",
                     width=1.5, size=6, opacity=0.55),
    "r":        dict(name="R sampling package", color=ORANGE, symbol="circle", dash="dashdot",
                     width=1.5, size=7),
}
METHODS = {"Z": ["tree", "dp"],
           "pi": ["tree", "dp", "tree_loo", "dp_loo", "r"],
           "sample": ["tree", "dp", "r"]}

# Complexity bound of each (operation, method), for the slope table.
BOUND = {
    "Z":      {"tree": lambda N, n: N * math.log(n) ** 2, "dp": lambda N, n: N * n},
    "pi":     {"tree": lambda N, n: N * math.log(n) ** 2, "dp": lambda N, n: N * n},
    "sample": {"tree": lambda N, n: n * math.log(N),      "dp": lambda N, n: N},
}
SLOPE_POINTS = 4      # fit each slope over this many of the largest sizes
SLOPE_FLAG = 0.2      # flag cells whose measured and bound slopes differ by more

FONT = dict(family="system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif", size=12, color="#333")
GRID, MINOR_GRID, AXIS_LINE = "#e3e3e3", "#f2f2f2", "#999"


# ── Formatting ───────────────────────────────────────────────────────────────

def fmt_time(t):
    """Seconds as e.g. '4.82 ms', to three significant figures."""
    t = float(f"{t:.3g}")
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("µs", 1e-6)):
        if t >= scale:
            return f"{t / scale:.3g} {unit}"
    return f"{t / 1e-6:.3g} µs"


def fmt_count(x):
    return f"{x:,.0f}"


def decades(values, fmt):
    """Ticks at the powers of ten spanning values, for a log axis.  The minor
    ticks (2-9 times each power) are explicit too: with explicit major ticks,
    Plotly spaces automatic minor ticks linearly, thousands of them."""
    lo = math.floor(math.log10(min(values)))
    hi = math.ceil(math.log10(max(values)))
    vals = [10.0 ** e for e in range(lo, hi + 1)]
    minor = [m * 10.0 ** e for e in range(lo, hi) for m in range(2, 10)]
    return dict(tickvals=vals, ticktext=[fmt(v) for v in vals], minor_tickvals=minor)


# ── Line plots: time against N with n = 0.4 N ────────────────────────────────

def line_figure(results, experiment):
    rows = [r for r in results if r["sweep"] == "ratio" and r["experiment"] == experiment]
    fig = go.Figure()
    Ns_all, ts_all = [], []
    for method in METHODS[experiment]:
        pts = sorted((r["N"], r["seconds"]) for r in rows
                     if r["method"] == method and r["seconds"] is not None)
        if not pts:
            continue
        Ns, ts = map(list, zip(*pts))
        Ns_all += Ns
        ts_all += ts
        s = STYLE[method]
        fig.add_trace(go.Scatter(
            x=Ns, y=ts, name=s["name"], mode="lines+markers", opacity=s.get("opacity", 1),
            line=dict(color=s["color"], dash=s["dash"], width=s["width"]),
            marker=dict(symbol=s["symbol"], size=s["size"], color=s["color"]),
            customdata=[fmt_time(t) for t in ts], hovertemplate="%{customdata}",
        ))
        if any(r["method"] == method and r.get("failure") for r in rows):
            fig.add_annotation(x=math.log10(Ns[-1]), y=math.log10(ts[-1]), text="NaN at the next size",
                               showarrow=False, xanchor="left", xshift=8, yshift=-2,
                               font=dict(color=s["color"], size=11))
    axis = dict(type="log", showgrid=True, gridcolor=GRID, zeroline=False, showline=True,
                linecolor=AXIS_LINE, ticks="outside", tickcolor=AXIS_LINE,
                minor=dict(showgrid=True, gridcolor=MINOR_GRID, tickmode="array"))

    def log_axis(values, fmt, **kw):
        ticks = decades(values, fmt)
        return dict(axis, tickvals=ticks["tickvals"], ticktext=ticks["ticktext"],
                    minor=dict(axis["minor"], tickvals=ticks["minor_tickvals"]), **kw)

    fig.update_layout(
        margin=dict(l=64, r=16, t=8, b=36),
        paper_bgcolor="white", plot_bgcolor="white", font=FONT,
        xaxis=log_axis(Ns_all, fmt_count, hoverformat=","),
        yaxis=log_axis(ts_all, fmt_time, title=dict(text="time", standoff=6)),
        legend=dict(orientation="h", x=0, xanchor="left", y=1, yanchor="bottom", font=dict(size=12)),
        hovermode="x unified",
    )
    return fig


# ── 3D plots: time over the (N, n) grid ──────────────────────────────────────

def grid_figure(results, experiment):
    rows = [r for r in results if r["sweep"] == "grid" and r["experiment"] == experiment
            and r["seconds"] is not None]
    fig = go.Figure()
    for method in ("dp", "tree"):
        pts = [r for r in rows if r["method"] == method]
        Ns = sorted({r["N"] for r in pts})
        ns = sorted({r["n"] for r in pts})
        seconds = {(r["N"], r["n"]): r["seconds"] for r in pts}
        s = STYLE[method]
        fig.add_trace(go.Surface(               # sizes with 2n > N are not timed: gaps
            x=Ns, y=ns, z=[[seconds.get((N, n)) for N in Ns] for n in ns],
            colorscale=[[0, s["color"]], [1, s["color"]]], showscale=False, opacity=0.4,
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[r["N"] for r in pts], y=[r["n"] for r in pts], z=[r["seconds"] for r in pts],
            mode="markers", name=s["name"], marker=dict(size=3, color=s["color"]),
            customdata=[fmt_time(r["seconds"]) for r in pts],
            hovertemplate=(f"{s['name']}<br><i>N</i> = %{{x:,}}, <i>n</i> = %{{y:,}}"
                           "<br>%{customdata}<extra></extra>"),
        ))
    # Explicit tick labels: Plotly's own drop the tick at 1 on 3D log axes.
    axis = dict(type="log", gridcolor="#d9d9d9", backgroundcolor="rgba(0,0,0,0)")

    def log_axis(values, fmt, title):
        ticks = decades(values, fmt)
        return dict(axis, title=title, tickvals=ticks["tickvals"], ticktext=ticks["ticktext"])

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", font=FONT,
        modebar=dict(bgcolor="rgba(255,255,255,0.8)", color="#aaa", activecolor="#555"),
        scene=dict(
            xaxis=log_axis([r["N"] for r in rows], fmt_count, "<i>N</i>"),
            yaxis=log_axis([r["n"] for r in rows], fmt_count, "<i>n</i>"),
            zaxis=log_axis([r["seconds"] for r in rows], fmt_time, "time"),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
    )
    return fig


# ── Slope table ──────────────────────────────────────────────────────────────

def loglog_slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def slopes(results):
    """Log-log slope of time against N at n = 1000 and against n at N = 20,000,
    measured and for the bound, keyed like the article's table cells:
    "Z tree N 1000" is the slope of the tree's time for Z in N at n = 1000."""
    grid = [r for r in results if r["sweep"] == "grid" and r["seconds"] is not None]
    cells = {}
    for experiment, bounds in BOUND.items():
        for method, bound in bounds.items():
            for x, other, fixed in (("N", "n", 1000), ("n", "N", 20000)):
                pts = sorted((r for r in grid if r["experiment"] == experiment
                              and r["method"] == method and r[other] == fixed),
                             key=lambda r: r[x])[-SLOPE_POINTS:]
                if len(pts) < 2:
                    continue
                xs = [r[x] for r in pts]
                measured = loglog_slope(xs, [r["seconds"] for r in pts])
                expected = loglog_slope(xs, [bound(r["N"], r["n"]) for r in pts])
                cells[f"{experiment} {method} {x} {fixed}"] = dict(
                    text=f"{measured:.2f} / {expected:.2f}",
                    flag=abs(measured - expected) > SLOPE_FLAG)
    return cells


def main():
    results = json.loads(DATA.read_text())["results"]
    figures = {}
    for experiment in METHODS:
        figures[experiment] = line_figure(results, experiment)
        figures[f"{experiment}_grid"] = grid_figure(results, experiment)
    figures = {name: fig.to_plotly_json() for name, fig in figures.items()}
    for fig in figures.values():
        fig["layout"].pop("template", None)     # style is set explicitly; keeps the file small
    payload = json.dumps(dict(figures=figures, slopes=slopes(results)),
                         cls=PlotlyJSONEncoder, separators=(",", ":"))
    OUT.write_text("// Generated by bench/plot_timing.py from timing.json; do not edit.\n"
                   f"window.TIMING_PLOTS = {payload};\n")
    print(f"Wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
