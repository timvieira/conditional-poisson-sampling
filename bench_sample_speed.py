#!/usr/bin/env python3
"""Benchmark single-sample speed across sampler implementations.

Compares: Python sequential, simple tree (precomputed CDFs), NumPy tree,
PyTorch tree, and R sequential — all drawing one sample from a
precomputed structure.

Usage:
    python bench_sample_speed.py [--quick]
"""

import json
import subprocess
import shutil
import sys
import time

import numpy as np

from conditional_poisson_numpy import ConditionalPoissonNumPy
from conditional_poisson_torch import ConditionalPoissonTorch
from bench_timing import (
    build_q_table,
    sequential_sample_from_q,
    build_tree_cdfs,
    simple_tree_sample,
)


def find_rscript():
    """Find Rscript binary — check cps-r conda env first, then PATH."""
    conda = shutil.which("conda")
    if conda:
        import subprocess as sp
        result = sp.run(
            [conda, "run", "-n", "cps-r", "which", "Rscript"],
            capture_output=True, text=True,
        )
        path = result.stdout.strip()
        if path and not path.startswith("ERROR"):
            return path
    return shutil.which("Rscript")


def time_fn(fn, reps):
    """Time fn over reps, return median ms."""
    fn()  # warmup
    times = [0.0] * reps
    for i in range(reps):
        t0 = time.perf_counter()
        fn()
        times[i] = (time.perf_counter() - t0) * 1000
    return float(np.median(times))


def run_r_sample(rscript, N, n, seed, reps):
    """Get R sampling times via subprocess."""
    result = subprocess.run(
        [rscript, "bench_timing_r.R", str(N), str(n), str(seed), str(reps)],
        capture_output=True, text=True, timeout=120,
    )
    times = {}
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        if "excl. DP" in r["method"]:
            times["R sequential"] = r["time_ms"]
        elif "incl. DP" in r["method"]:
            times["R sequential (incl. DP)"] = r["time_ms"]
    return times


def main():
    quick = "--quick" in sys.argv

    if quick:
        sizes = [(50, 20), (100, 40), (200, 80)]
    else:
        sizes = [(50, 20), (100, 40), (200, 80), (500, 200),
                 (1000, 400), (2000, 800)]

    seed = 42
    reps = 500
    rscript = find_rscript()

    methods = [
        "Python sequential",
        "Simple tree",
        "NumPy tree",
        "PyTorch tree",
        "R sequential",
    ]

    # Header
    print(f"{'N':>6s} {'n':>5s}", end="")
    for m in methods:
        print(f"  {m:>20s}", end="")
    print()
    print("-" * (13 + 22 * len(methods)))

    for N, n in sizes:
        w = np.random.RandomState(seed).exponential(1.0, N)
        rng = np.random.default_rng(seed)
        timings = {}

        # Python sequential
        q_table = build_q_table(w, n)
        timings["Python sequential"] = time_fn(
            lambda: sequential_sample_from_q(q_table, rng), reps
        )

        # Simple tree (precomputed CDFs)
        cp = ConditionalPoissonNumPy.from_weights(n, w)
        Pc, _, S, _, _ = cp._get_p_tree()
        tree_cdfs = build_tree_cdfs(Pc, S, N, n)
        timings["Simple tree"] = time_fn(
            lambda: simple_tree_sample(tree_cdfs, S, N, n, rng), reps
        )

        # NumPy tree
        cp.sample(1, rng=seed)  # warmup
        timings["NumPy tree"] = time_fn(
            lambda: cp.sample(1, rng=seed), reps
        )

        # PyTorch tree
        cpt = ConditionalPoissonTorch.from_weights(n, w)
        cpt.sample(1, rng=seed)  # warmup (builds sample tree)
        timings["PyTorch tree"] = time_fn(
            lambda: cpt.sample(1, rng=seed), reps
        )

        # R sequential
        if rscript:
            r_times = run_r_sample(rscript, N, n, seed, reps)
            timings["R sequential"] = r_times.get("R sequential", float("nan"))
        else:
            timings["R sequential"] = float("nan")

        # Print row
        print(f"{N:>6d} {n:>5d}", end="")
        for m in methods:
            ms = timings.get(m, float("nan"))
            print(f"  {ms:>19.3f}ms", end="")
        print()


if __name__ == "__main__":
    main()
