TIMEOUT ?= 120

.PHONY: help test bench-timing bench-grid figures data snippets deploy dev clean install

help:
	@echo "Usage: make [-j4] <target>"
	@echo "  test          Run all tests"
	@echo "  bench-timing  Run all timing benchmarks (use -j for parallel)"
	@echo "  bench-grid    Run timing grid sweep"
	@echo "  figures       Regenerate SVG plots from timing data"
	@echo "  snippets      Regenerate popover.js code snippets"
	@echo "  deploy        Build and deploy to GitHub Pages"

test:
	python -m pytest tests/ -v

# Timing benchmarks: one file per (method, experiment, N, n).
# Use `make -j8 bench-timing` for parallel execution.
METHODS := Sequential NumPy PyTorch
EXPERIMENTS := Z pi fit sample
define SIZE_template
BENCH_TARGETS += $(foreach m,$(METHODS),$(foreach e,$(EXPERIMENTS),bench/results/$(m)_$(e)_$(1)_$(2).json))
endef
$(eval $(call SIZE_template,50,20))
$(eval $(call SIZE_template,100,40))
$(eval $(call SIZE_template,200,80))
$(eval $(call SIZE_template,500,200))
$(eval $(call SIZE_template,1000,400))
$(eval $(call SIZE_template,2000,800))
$(eval $(call SIZE_template,5000,2000))

bench/results/%.json:
	@mkdir -p bench/results
	@method=$$(echo $* | cut -d_ -f1); \
	 exp=$$(echo $* | cut -d_ -f2); \
	 N=$$(echo $* | cut -d_ -f3); \
	 n=$$(echo $* | cut -d_ -f4); \
	 python bench/bench_one.py $$method $$exp $$N $$n --timeout $(TIMEOUT) > $@

bench-timing: $(BENCH_TARGETS)
	@python -c "import json, glob; \
	results = [json.load(open(f)) for f in sorted(glob.glob('bench/results/*.json')) if 'error' not in json.load(open(f))]; \
	json.dump(results, open('bench/results/timing_data.json','w'), indent=2); \
	print(f'Wrote {len(results)} results to bench/results/timing_data.json')"

bench-grid:
	python bench/bench_timing_grid.py

data: bench-timing bench-grid

figures: bench/results/timing_data.json
	python bench/plot_timing.py bench/results/timing_data.json

snippets:
	python extract_snippets.py

deploy: test snippets
	blog deploy

dev:
	blog dev

install:
	pip install -e ".[dev]"

clean:
	rm -rf __pycache__ conditional_poisson/__pycache__ tests/__pycache__ bench/__pycache__
	rm -rf *.egg-info build dist .pytest_cache
	rm -rf bench/results/
