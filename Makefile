.PHONY: help test bench figures data snippets deploy dev clean install install-r

help:
	@echo "Usage: make <target>"
	@echo "  test          Run all tests"
	@echo "  bench         Run timing benchmarks (serial; ~30 min)"
	@echo "  figures       Regenerate SVG plots from timing data"
	@echo "  data          bench + figures"
	@echo "  snippets      Regenerate popover.js code snippets"
	@echo "  deploy        Build and deploy to GitHub Pages"

test:
	python -m pytest tests/ -v

# Timing benchmarks: serial and single-threaded, so runs do not compete for CPU.
# Writes content/figures/timing.json, the data behind every timing figure.
# `make figures` builds the article's Plotly figures from it (timing_plots.js).
bench:
	python bench/bench_timing.py

figures: content/figures/timing.json
	python bench/plot_timing.py

data: bench figures

snippets:
	python extract_snippets.py

deploy: test snippets
	blog deploy

dev:
	blog dev

install:
	pip install -e ".[dev]"

install-r:
	bash install_r.sh

clean:
	rm -rf __pycache__ conditional_poisson/__pycache__ tests/__pycache__ bench/__pycache__
	rm -rf *.egg-info build dist .pytest_cache
