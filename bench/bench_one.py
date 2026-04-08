#!/usr/bin/env python3
"""Timing benchmark for a single (method, experiment, N, n) combination.

Usage:
    python bench_one.py <method> <experiment> <N> <n> [--reps R] [--timeout T]

Prints a JSON line to stdout with timing result or error.
"""

import argparse
import json
import signal
import sys
import time
import numpy as np


class Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise Timeout()


def time_fn(fn, reps=5, timeout=None):
    """Time fn, return median ms.  Raises Timeout if any rep exceeds timeout."""
    if timeout:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
    try:
        fn()  # warmup
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
        return float(np.median(times))
    finally:
        if timeout:
            signal.alarm(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["Sequential", "NumPy", "PyTorch"])
    parser.add_argument("experiment", choices=["Z", "pi", "fit", "sample"])
    parser.add_argument("N", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("--reps", type=int, default=0, help="0 = auto")
    parser.add_argument("--timeout", type=int, default=120, help="seconds per job")
    args = parser.parse_args()

    N, n = args.N, args.n
    reps = args.reps or max(3, 200 // max(1, N // 50))
    result = {"method": args.method, "experiment": args.experiment, "N": N, "n": n}

    rng = np.random.RandomState(42)
    w = rng.exponential(1.0, N)

    if args.method == "Sequential":
        from conditional_poisson.sequential_numpy import ConditionalPoissonSequentialNumPy as Cls
    elif args.method == "NumPy":
        from conditional_poisson.tree_numpy import ConditionalPoissonNumPy as Cls
    elif args.method == "PyTorch":
        from conditional_poisson.tree_torch import ConditionalPoissonTorch as Cls

    t0 = time.perf_counter()
    try:
        if args.experiment == "Z":
            def bench():
                cp = Cls.from_weights(n, w)
                return cp.log_normalizer
            ms = time_fn(bench, reps=reps, timeout=args.timeout)

        elif args.experiment == "pi":
            def bench():
                cp = Cls.from_weights(n, w)
                return cp.incl_prob
            ms = time_fn(bench, reps=reps, timeout=args.timeout)

        elif args.experiment == "fit":
            from conditional_poisson.tree_numpy import ConditionalPoissonNumPy
            pi_target = ConditionalPoissonNumPy.from_weights(n, w).incl_prob
            ms = time_fn(lambda: Cls.fit(pi_target, n), reps=reps, timeout=args.timeout)

        elif args.experiment == "sample":
            cp = Cls.from_weights(n, w)
            _ = cp.log_normalizer
            ms = time_fn(cp.sample, reps=reps, timeout=args.timeout)

        elapsed = time.perf_counter() - t0
        result["time_ms"] = round(ms, 4)
        result["wall_s"] = round(elapsed, 1)
        print(json.dumps(result))
        print(f"  {args.method} {args.experiment} N={N} n={n}  {ms:.1f}ms  ({elapsed:.0f}s wall)",
              file=sys.stderr)

    except Timeout:
        elapsed = time.perf_counter() - t0
        result["error"] = f"timeout ({args.timeout}s)"
        result["wall_s"] = round(elapsed, 1)
        print(json.dumps(result))
        print(f"  TIMEOUT {args.method} {args.experiment} N={N} n={n}  ({elapsed:.0f}s wall)",
              file=sys.stderr)

    except Exception as e:
        elapsed = time.perf_counter() - t0
        result["error"] = str(e)[:200]
        result["wall_s"] = round(elapsed, 1)
        print(json.dumps(result))
        print(f"  ERROR {args.method} {args.experiment} N={N} n={n}: {e}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
