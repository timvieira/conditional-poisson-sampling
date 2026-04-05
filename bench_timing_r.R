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

# Save pik for fitting and sampling benchmarks
q <- UPMEqfromw(w, n)
pik <- UPMEpikfromq(q)

# --- Fitting: target pi -> weights via UPMEpiktildefrompik ---
ms <- time_it(function() {
    UPMEpiktildefrompik(pik)
}, reps)
emit("R:sampling (fit)", ms)

# --- Drawing 1 sample (DP + sample, from weights) ---
# Includes the O(Nn) DP rebuild every call.
ms <- time_it(function() {
    q2 <- UPMEqfromw(w, n)
    UPMEsfromq(q2)
}, reps)
emit("R:sampling (1 sample, incl. DP)", ms)

# --- Drawing 1 sample (sample step only, from precomputed DP) ---
# Fair comparison against our tree sampler: pre-compute DP outside timing loop.
q_pre <- UPMEqfromw(w, n)
ms <- time_it(function() {
    UPMEsfromq(q_pre)
}, reps)
emit("R:sampling (1 sample, excl. DP)", ms)
