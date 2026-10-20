#!/usr/bin/env python3
"""Timing experiments for the article's Timing section.

Writes content/figures/timing.json, the single data file behind every timing
figure: bench/plot_timing.py draws the static plots from it, and the article's
3D plots load it in the browser.

Methods (all compute exact conditional Poisson quantities):
  tree      ConditionalPoissonTorch: FFT product tree; pi by autograd
  dp        ConditionalPoissonSequentialNumPy: O(Nn) DP; pi by its reverse pass
  tree_loo  pi by leave-one-out with the tree: N forward passes, no backprop
  dp_loo    pi by leave-one-out with the DP
  r         R package `sampling` (UPMEqfromw, UPMEpikfromq, UPMEsfromq)

Protocol: everything runs serially in this one process (R in a subprocess),
single-threaded, with the same weights w ~ Exp(1) for every method.  Each
measurement makes one warm-up call, then times repeats of enough calls that a
repeat lasts >= 0.2 s, and reports the fastest repeat per call (as Python's
timeit recommends: slower repeats measure interference from other processes,
not the code).  Each output is checked against an independent method.  A
series over N stops after a call takes more than STOP_AFTER seconds.

Usage:
    python bench/bench_timing.py                     # full run
    python bench/bench_timing.py --quick             # small sizes, to test the pipeline
    python bench/bench_timing.py --only Z:dp,pi:r    # re-measure some (experiment:method)
                                                     # pairs, keeping the other results
"""

import os
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import gc
import json
import math
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np
import scipy
import torch

from conditional_poisson.tree_torch import ConditionalPoissonTorch as Tree
from conditional_poisson.sequential_numpy import ConditionalPoissonSequentialNumPy as DP

torch.set_num_threads(1)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "content" / "figures" / "timing.json"
R_SCRIPT = REPO / "bench" / "bench_timing_r.R"

MIN_REPEAT = 0.2      # seconds per timed repeat
STOP_AFTER = 20.0     # stop a series over N once one call takes longer than this


# ── Leave-one-out baseline ────────────────────────────────────────────────────

def loo_incl_prob(cls, n, w):
    """pi_i = w_i Z(w without item i, n - 1) / Z(w, n): one forward pass per item.

    This is how one computes pi without backprop.  It costs N forward passes.
    """
    w = np.asarray(w, float)
    log_Z = cls.from_weights(n, w).log_normalizer
    pi = np.empty(len(w))
    for i in range(len(w)):
        pi[i] = w[i] * math.exp(cls.from_weights(n - 1, np.delete(w, i)).log_normalizer - log_Z)
    return pi


# ── Timing ───────────────────────────────────────────────────────────────────

def _time_loops(fn, loops):
    t0 = time.perf_counter()
    for _ in range(loops):
        fn()
    return (time.perf_counter() - t0) / loops


def measure(fn):
    """Return (seconds per call, output of the warm-up call).  See module docstring."""
    gc.collect()
    t0 = time.perf_counter()
    out = fn()                                  # warm-up; also sizes the repeats
    first = time.perf_counter() - t0
    loops = max(1, math.ceil(MIN_REPEAT / max(first, 1e-9)))
    repeats = 5 if first < 1 else 3 if first < 10 else 1
    gc.disable()
    try:
        times = [_time_loops(fn, loops) for _ in range(repeats)]
    finally:
        gc.enable()
    return min(times), out


def to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


# ── R ────────────────────────────────────────────────────────────────────────

def find_rscript():
    """Rscript from $RSCRIPT, the cps-r conda env (`make install-r`), or PATH."""
    if os.environ.get("RSCRIPT"):
        return os.environ["RSCRIPT"]
    conda = shutil.which("conda")
    if conda:
        out = subprocess.run([conda, "run", "-n", "cps-r", "which", "Rscript"],
                             capture_output=True, text=True)
        path = out.stdout.strip()
        if out.returncode == 0 and path:
            return path
    return shutil.which("Rscript")


def run_r(rscript, n, w, timeout=3600):
    """Time R's pi and per-sample cost on the same weights.  Returns (times, pi)."""
    with tempfile.TemporaryDirectory() as tmp:
        wfile, pifile = Path(tmp) / "w.txt", Path(tmp) / "pi.txt"
        np.savetxt(wfile, w, fmt="%.17g")
        proc = subprocess.run([rscript, str(R_SCRIPT), str(wfile), str(n), str(pifile)],
                              capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip()[-300:])
        times = {}
        for line in proc.stdout.splitlines():
            if line.startswith("{"):
                rec = json.loads(line)
                times[rec["experiment"]] = rec
        pi = np.loadtxt(pifile, ndmin=1) if pifile.exists() else None
    return times, pi


# ── Experiments ──────────────────────────────────────────────────────────────

def weights(N):
    return np.random.default_rng(N).exponential(1.0, N)


def calls(experiment, method, n, w):
    """The timed call for (experiment, method), plus any untimed setup."""
    if experiment == "Z":
        cls = {"tree": Tree, "dp": DP}[method]
        return lambda: cls.from_weights(n, w).log_normalizer
    if experiment == "pi":
        if method in ("tree", "dp"):
            cls = {"tree": Tree, "dp": DP}[method]
            return lambda: cls.from_weights(n, w).incl_prob
        cls = {"tree_loo": Tree, "dp_loo": DP}[method]
        return lambda: loo_incl_prob(cls, n, w)
    if experiment == "sample":
        cls = {"tree": Tree, "dp": DP}[method]
        cp = cls.from_weights(n, w)
        cp.sample()                             # build the tree / table once
        return cp.sample
    raise ValueError(experiment)


class Reference:
    """Exact log Z and pi from the tree, cached per (N, n), for checking outputs."""

    def __init__(self):
        self.cache = {}

    def __call__(self, N, n):
        if (N, n) not in self.cache:
            cp = Tree.from_weights(n, weights(N))
            self.cache = {(N, n): (cp.log_normalizer, to_numpy(cp.incl_prob))}
        return self.cache[(N, n)]


def check(experiment, method, out, N, n, ref):
    """Error of out against the reference (the DP for the tree itself)."""
    if experiment == "sample":
        s = to_numpy(out)
        assert len(s) == n and len(np.unique(s)) == n and s.min() >= 0 and s.max() < N
        return None
    if method == "tree":
        other = DP.from_weights(n, weights(N))
        exact = other.log_normalizer if experiment == "Z" else other.incl_prob
    else:
        log_Z, pi = ref(N, n)
        exact = log_Z if experiment == "Z" else pi
    return float(np.max(np.abs(to_numpy(out) - exact)))


def run_series(results, experiment, method, sizes, sweep, ref):
    """Time one method over increasing N; stop after a call exceeds STOP_AFTER."""
    for N, n in sizes:
        w = weights(N)
        try:
            seconds, out = measure(calls(experiment, method, n, w))
        except MemoryError:
            print(f"  {experiment:6s} {method:8s} N={N:6d} n={n:6d}  out of memory", file=sys.stderr)
            break
        err = check(experiment, method, out, N, n, ref)
        results.append(dict(experiment=experiment, method=method, sweep=sweep,
                            N=N, n=n, seconds=seconds, err=err))
        print(f"  {experiment:6s} {method:8s} N={N:6d} n={n:6d}  {seconds * 1e3:12.3f} ms"
              f"  err={err if err is None else f'{err:.1e}'}", file=sys.stderr)
        if seconds > STOP_AFTER:
            break


def run_r_series(results, rscript, sizes, sweep, ref):
    """R's pi and per-sample cost; stops at the first size where R's output is not finite."""
    for N, n in sizes:
        times, pi = run_r(rscript, n, weights(N))
        if pi is None or not np.all(np.isfinite(pi)):
            results.append(dict(experiment="pi", method="r", sweep=sweep, N=N, n=n,
                                seconds=None, err=None, failure="non-finite output (overflow)"))
            print(f"  R      N={N:6d} n={n:6d}  non-finite output (overflow)", file=sys.stderr)
            break
        err = float(np.max(np.abs(pi - ref(N, n)[1])))
        for experiment in ("pi", "sample"):
            seconds = times[experiment]["seconds"]
            results.append(dict(experiment=experiment, method="r", sweep=sweep, N=N, n=n,
                                seconds=seconds, err=err if experiment == "pi" else None))
            print(f"  {experiment:6s} r        N={N:6d} n={n:6d}  {seconds * 1e3:12.3f} ms",
                  file=sys.stderr)
        if times["pi"]["seconds"] > STOP_AFTER:
            break


def metadata(rscript):
    cpu = platform.processor()
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    meta = dict(
        date=date.today().isoformat(), cpu=cpu, threads=1,
        python=platform.python_version(), numpy=np.__version__,
        scipy=scipy.__version__, torch=torch.__version__,
        protocol=("serial, single-threaded; one warm-up call, then the fastest of up to 5 "
                  f"repeats of >= {MIN_REPEAT} s each, per call; weights w ~ Exp(1)"),
    )
    if rscript:
        out = subprocess.run([rscript, "-e", 'cat(R.version.string, "|", '
                              'as.character(packageVersion("sampling")))'],
                             capture_output=True, text=True)
        meta["r"] = out.stdout.strip()
    return meta


def main():
    quick = "--quick" in sys.argv
    only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None

    def wanted(experiment, method):
        return only is None or f"{experiment}:{method}" in only
    if quick:
        ratio_Ns = [50, 100, 200, 500]
        grid_Ns, grid_ns = [100, 200, 500], [5, 10, 20, 50, 100]
    else:
        ratio_Ns = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        grid_Ns = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        grid_ns = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    ratio = [(N, round(0.4 * N)) for N in ratio_Ns]
    rscript = find_rscript()
    if not rscript:
        print("warning: Rscript not found; skipping R (see `make install-r`)", file=sys.stderr)
    ref = Reference()
    results = []
    r_wanted = wanted("pi", "r") or wanted("sample", "r")       # R runs both together
    if only:
        results = [r for r in json.loads(OUT.read_text())["results"]
                   if not (r_wanted if r["method"] == "r" else wanted(r["experiment"], r["method"]))]

    def save():
        OUT.write_text(json.dumps(dict(meta=metadata(rscript), results=results), indent=1) + "\n")

    print("n = 0.4 N", file=sys.stderr)
    for experiment, methods in [("Z", ["tree", "dp"]),
                                ("pi", ["tree", "dp", "tree_loo", "dp_loo"]),
                                ("sample", ["tree", "dp"])]:
        for method in methods:
            if wanted(experiment, method):
                run_series(results, experiment, method, ratio, "ratio", ref)
                save()
    if rscript and r_wanted:
        run_r_series(results, rscript, ratio, "ratio", ref)
        save()

    print("grid over (N, n)", file=sys.stderr)
    for n in grid_ns:
        sizes = [(N, n) for N in grid_Ns if 2 * n <= N]
        for experiment in ("Z", "pi", "sample"):
            for method in ("tree", "dp"):
                if wanted(experiment, method):
                    run_series(results, experiment, method, sizes, "grid", ref)
        save()
    print(f"Wrote {len(results)} results to {OUT.relative_to(REPO)}", file=sys.stderr)


if __name__ == "__main__":
    main()
