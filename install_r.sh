#!/usr/bin/env bash
# Install R and required packages for the timing benchmarks.
# Creates a dedicated conda environment to avoid dependency conflicts.
#
# Usage:
#   bash install_r.sh
#
# After installation, activate with:
#   conda activate cps-r
#
# Or just use the Rscript path directly:
#   ~/anaconda3/envs/cps-r/bin/Rscript bench_timing_r.R ...

set -euo pipefail

ENV_NAME="cps-r"

# Check for conda
if ! command -v conda &>/dev/null; then
    echo "Error: conda not found. Install miniconda first." >&2
    exit 1
fi

# Create env with R (skip if exists)
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists, skipping creation."
else
    echo "Creating conda environment '${ENV_NAME}' with R..."
    conda create -y -n "$ENV_NAME" -c conda-forge --override-channels r-base
fi

# Get the env's Rscript path
RSCRIPT="$(conda run -n "$ENV_NAME" which Rscript)"
echo "Using Rscript: $RSCRIPT"

# Install R packages from CRAN
echo "Installing R package: sampling..."
conda run -n "$ENV_NAME" Rscript -e '
pkgs <- c("sampling")
missing <- pkgs[!pkgs %in% installed.packages()[,"Package"]]
if (length(missing) > 0) {
    install.packages(missing, repos="https://cloud.r-project.org", quiet=TRUE)
}
cat("Installed:", paste(pkgs, collapse=", "), "\n")
'

echo ""
echo "Done. To use R in benchmarks:"
echo "  conda run -n $ENV_NAME Rscript bench_timing_r.R ..."
echo "  # or"
echo "  conda activate $ENV_NAME && Rscript bench_timing_r.R ..."
