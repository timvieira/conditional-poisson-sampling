#!/usr/bin/env Rscript
# Times the R `sampling` package on one weight vector.  Called by bench_timing.py.
#
# Usage:
#   Rscript bench_timing_r.R <weights.txt> <n> <pi_out.txt>
#
# Prints one JSON line per operation and writes the inclusion probabilities
# to <pi_out.txt> so the caller can check them.  Uses the same protocol as
# bench_timing.py: one warm-up call, then the fastest of several repeats,
# each repeat lasting at least 0.2 s.

suppressMessages(library(sampling))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
    cat("Usage: Rscript bench_timing_r.R <weights.txt> <n> <pi_out.txt>\n", file = stderr())
    quit(status = 1)
}
w <- scan(args[1], quiet = TRUE)
n <- as.integer(args[2])

now <- function() proc.time()[["elapsed"]]

time_it <- function(fn) {
    t0 <- now(); fn(); first <- now() - t0
    loops <- max(1, ceiling(0.2 / max(first, 1e-6)))
    repeats <- if (first < 1) 5 else if (first < 10) 3 else 1
    best <- Inf
    for (r in seq_len(repeats)) {
        t0 <- now()
        for (i in seq_len(loops)) fn()
        best <- min(best, (now() - t0) / loops)
    }
    best
}

emit <- function(experiment, seconds) {
    cat(sprintf('{"experiment": "%s", "seconds": %.9g}\n', experiment, seconds))
}

# --- Inclusion probabilities: UPMEqfromw builds the O(Nn) table, UPMEpikfromq reads off pi ---
q <- UPMEqfromw(w, n)
pik <- UPMEpikfromq(q)
write(format(pik, digits = 17), args[3], ncolumns = 1)
if (!all(is.finite(pik))) quit(status = 0)      # overflow: the caller records the failure

emit("pi", time_it(function() UPMEpikfromq(UPMEqfromw(w, n))))

# --- One sample: UPMEsfromq scans the precomputed table, O(N) ---
emit("sample", time_it(function() UPMEsfromq(q)))
