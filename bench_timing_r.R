#!/usr/bin/env Rscript
# Benchmark R packages for conditional Poisson sampling.
# Called by bench_timing.py via subprocess.
#
# Usage:
#   Rscript bench_timing_r.R <N> <n> <seed> <reps>
#
# Output: JSON lines to stdout, one per method/operation.

library(sampling)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
    cat("Usage: Rscript bench_timing_r.R <N> <n> <seed> <reps>\n", file = stderr())
    quit(status = 1)
}

N    <- as.integer(args[1])
n    <- as.integer(args[2])
seed <- as.integer(args[3])
reps <- as.integer(args[4])

set.seed(seed)
w <- rexp(N)

# Helper: time a function over reps, return median ms.
# Uses batch timing when individual calls are too fast.
time_it <- function(fn, reps) {
    # Warmup
    tryCatch(fn(), error = function(e) NULL)

    # First, try individual timing
    times <- numeric(reps)
    ok <- TRUE
    for (i in seq_len(reps)) {
        t0 <- proc.time()["elapsed"]
        tryCatch(fn(), error = function(e) { ok <<- FALSE })
        if (!ok) return(NA)
        times[i] <- (proc.time()["elapsed"] - t0) * 1000
    }
    med <- median(times)

    # If too fast for ms resolution, batch
    if (med < 1.0) {
        batch <- max(10, as.integer(ceiling(5.0 / max(med, 0.001))))
        t0 <- proc.time()["elapsed"]
        for (i in seq_len(batch)) fn()
        med <- (proc.time()["elapsed"] - t0) * 1000 / batch
    }
    med
}

emit <- function(method, ms) {
    if (!is.na(ms)) {
        cat(sprintf('{"method":"%s","N":%d,"n":%d,"time_ms":%.4f}\n', method, N, n, ms))
    }
}

# --- Computing pi via sampling::UPMEqfromw + UPMEpikfromq ---
ms <- time_it(function() {
    q <- UPMEqfromw(w, n)
    UPMEpikfromq(q)
}, reps)
emit("R:sampling (pi)", ms)

# Save pik for sampling benchmarks
q <- UPMEqfromw(w, n)
pik <- UPMEpikfromq(q)

# --- Drawing 1 sample via sampling internals (from weights) ---
# UPmaxentropy(pik) includes fitting (pik -> w), which is unfair to compare
# against our sampler that starts from weights.  Instead, time the DP + sample
# steps directly: UPMEqfromw (O(Nn) DP) + UPMEsfromq (O(N) sequential sample).
ms <- time_it(function() {
    q2 <- UPMEqfromw(w, n)
    UPMEsfromq(q2)
}, reps)
emit("R:sampling (1 sample)", ms)
