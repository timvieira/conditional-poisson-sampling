.PHONY: help test test-identity test-unit test-all test-r \
       bench bench-timing bench-grid bench-sample-speed bench-accuracy \
       figures data deploy dev clean install lint

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Testing:"
	@echo "  test              Run all tests (identity + unit + parameterized)"
	@echo "  test-identity     Run identity tests (blog post claims)"
	@echo "  test-unit         Run unit tests for ConditionalPoissonNumPy"
	@echo "  test-all          Run parameterized tests across all implementations"
	@echo "  test-r            Run R agreement tests (requires R + sampling package)"
	@echo ""
	@echo "Benchmarks:"
	@echo "  bench             Run all benchmarks"
	@echo "  bench-timing      Run timing benchmarks → bench/results/timing_data.json"
	@echo "  bench-grid        Run timing grid sweep → bench/results/timing_grid.json"
	@echo "  bench-sample-speed  Compare single-sample speed across implementations"
	@echo "  bench-accuracy    Compare numerical accuracy across implementations"
	@echo ""
	@echo "Figures & data:"
	@echo "  data              Regenerate all benchmark data (timing + grid)"
	@echo "  figures           Regenerate all SVG plots from benchmark data"
	@echo "  snippets          Regenerate popover.js code snippets"
	@echo ""
	@echo "Site:"
	@echo "  dev               Start dev server with auto-rebuild"
	@echo "  deploy            Build site and deploy to GitHub Pages"
	@echo ""
	@echo "Setup:"
	@echo "  install           Install package in editable mode with dev deps"
	@echo "  clean             Remove build artifacts and caches"

# ── Testing ──────────────────────────────────────────────────────────────────

test: test-identity test-unit test-all

test-identity:
	python tests/test_identities.py

test-unit:
	python -m pytest tests/test_conditional_poisson.py tests/test_torch.py -v

test-all:
	python -m pytest tests/test_all_implementations.py -v

test-r:
	python -m pytest tests/test_r_agreement.py -v

# ── Benchmarks ───────────────────────────────────────────────────────────────

bench: bench-timing bench-grid bench-sample-speed bench-accuracy

bench-timing:
	python bench/bench_timing.py
	@echo "Wrote bench/results/timing_data.json"

bench-grid:
	python bench/bench_timing_grid.py
	@echo "Wrote bench/results/timing_grid.json"

bench-sample-speed:
	python bench/bench_sample_speed.py

bench-accuracy:
	python bench/bench_accuracy.py

# ── Figures & data ───────────────────────────────────────────────────────────

data: bench-timing bench-grid

figures: bench/results/timing_data.json
	python bench/plot_timing.py bench/results/timing_data.json
	@echo "Wrote content/figures/timing_*.svg"

bench/results/timing_data.json:
	$(MAKE) bench-timing

snippets:
	python extract_snippets.py

# ── Site ─────────────────────────────────────────────────────────────────────

dev:
	blog dev

deploy: test snippets
	blog deploy

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

clean:
	rm -rf __pycache__ conditional_poisson/__pycache__ tests/__pycache__ bench/__pycache__
	rm -rf *.egg-info build dist .pytest_cache
	rm -rf bench/results/*.json
