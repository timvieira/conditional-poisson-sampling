# TODO

## Content

- [x] Add a disclaimer that this post was written with extensive help from Claude Code, especially in the interactive widgets, which would not have been possible without Claude Code.

- [x] Widget contents spill out of containers on narrow screens (e.g., phone). Looks very bad.
- [ ] Sampling animation widget: visual language is inconsistent with the product tree widget, and some of the pop-up histograms are confusing. Consider highlighting the current step in the pseudocode to anchor the viewer.
- [x] Review the sampling pseudocode
- [ ] Would evaluating/plotting the polynomial as a function of z tell us anything interesting?
- [x] Add link to Wikipedia page on [elementary symmetric polynomials](https://en.wikipedia.org/wiki/Elementary_symmetric_polynomial)

## Timing Section

- [x] Timing section is still a mess — results are mostly missing, and many results don't make sense yet.
- [x] Is the 3D plot using old data? It doesn't appear to be updated. Where is it getting its data from?
- [ ] Contour diagram needs work.
- [ ] "Numerical validation" section — not sure this is the best way to present this information.
- [x] Hide all Python/plotting code—just show the plot. Link to relevant implementations using the checkmark/footnote pill system (e.g., each curve links to its implementation in the repo).
- [x] Add more baselines for computing $\pi$ — (1) backprop-on-DP at $O(Nn)$, (2) naive loop versions: $N \times O(Nn)$ DP and $N \times O(N \log^2 n)$ product tree. These show the value of backprop vs. leave-one-out recomputation.
- [x] Add a programmatically generated empirical slope table. For each method, fit log-log slopes (time vs $N$ at fixed $n$, and time vs $n$ at fixed $N$) from the grid sweep data and display alongside theoretical complexity bounds. Auto-flag rows where empirical slope deviates significantly from prediction.
- [x] Scrutinize timing experiments. R's `sampling` package functions (`UPMEqfromw`, `UPMEpikfromq`, `UPMEsfromq`, `UPMEpiktildefrompik`) are confirmed pure R (no compiled C). Subprocess overhead and measurement methodology still need investigation. Verify that all methods compute the same quantity.
- [x] ~~Explore the fitting algorithm in the R `sampling` package. Compare the number of iterations required by the R fixed-point method vs. our L-BFGS solver.~~ — done (`bench/bench_fit.py`, article's "Comparison with R"): about the same counts on typical targets (3–9 vs 4–15 evaluations to 1e-10); with very uneven weights the fixed-point iteration slows down (median 32.5 vs 9 at sigma = 10, 3 of 100 targets unfinished after 2000).
- [x] 3D timing plots: replace the single dropdown plot with three separate 3D plots (computing $Z$, computing $\pi$, drawing samples). Sampling timing is currently only a static 2D SVG—it has two variables ($N$, $n$) just like the others, so it should be 3D too.

## Implementation

### Code cleanup
- [x] ~~L-BFGS fitting convergence is much slower than the old Newton-CG (24 iterations vs 5 for N=10, non-monotone).~~ — fixed: L-BFGS now starts from the Poisson approximation's inverse Hessian and accepts steps on the approximate Wolfe conditions (`conditional_poisson/_fit.py`); N=10 takes 9 evaluations, and fits reach 1e-10 at N=1000.

### Benchmarks
- [x] Rerun `bench_timing.py` and regenerate `timing_data.json` + SVG plots (sampling data is stale after sampler rewrite)
- [x] Rerun `bench_timing_grid.py` and update inline 3D widget data in the article (sampling rows are stale)
- [x] Update `plot_timing.py` to include new sampling methods (sequential, PyTorch tree) in the SVG
- [x] The static `timing_samples.svg` still shows old curves from the vectorized sampler — must be regenerated

### Other
- [ ] NumPy tree timing slopes (~1.15–1.45 in $N$) are higher than expected for $O(N \log^2 N)$. Investigate whether forcing FFT throughout gives cleaner scaling.
- [ ] Use numpy/pytorch's built-in bisect_left methods instead of manual binary search — likely faster. Verify empirically with a benchmark before switching.
- [ ] Test GPU performance — `conditional_poisson/tree_torch.py` (FFT) should be benchmarked on GPU. Float32 precision risk needs testing (contour scaling helps but may not fully compensate).

## Blog Polish
- [x] ~~Add a note encouraging readers to report issues on the [GitHub issue tracker](https://github.com/timvieira/conditional-poisson-sampling/issues)~~ — done
- [x] ~~Add a citation section at the end of the post (e.g., BibTeX snippet, suggested citation format)~~ — done
- [ ] Consistently color code math symbols and widgets — use the same colors for $P(S)$, $\pi$, and $w$ in both LaTeX and D3 widgets. Previously removed all color coding; bring it back in a principled way with a shared palette.

## Widget UX

- [ ] Scrutinize the little histograms in the sampling animation. What are they supposed to be? Why do they appear to change after sampling?
- [ ] Make sure the sampling animation widget is faithful to the implementation.
- [x] Several widgets overflow their containers on mobile. Fix the layout to fill viewport width on small screens, then add horizontal scroll for wide SVGs.
- [ ] Centralize widget/layout styles into the blog's CSS (`~/projects/blog/main/content/css/blog.css`) instead of inline styles in the .md file.

## Fidelity Audit

- [ ] Scrutinize the fidelity of the implementation to the article. Gaps must be noted for science!

## Checklist (final pass)

- All vector quantities use `\boldsymbol` in LaTeX
- All subsection titles use Title Case
- Consolidated notation box present and complete
- w/p/θ parameterization transitions are explicit
- Verification pills appear after punctuation
- Every mathematical claim has a ✓ pill linking to its test
- Code and math use consistent notation (per CLAUDE.md)
- No undefined abbreviations
- Citations verified against actual publications
- No hardcoded stale numbers

## I WILL DIE ON THIS HILL
- there will be no such thing as a suffix table in torch sequential.

## Repo Housekeeping

### Cleanup

- [ ] Tidy up the repo (clean up unused files, organize structure)
- [ ] Fix `test_animation.mjs` — update paths, verify it runs, integrate into CI
- [ ] Add mathematical correctness tests for JS widget algorithms (product tree, quota splitting, polynomial multiplication, CDF computation). Either extend `test_animation.mjs` to check computed values against known answers, or extract JS algorithms into testable modules and test with Node.js.
- [x] ~~Decide: remove `plot_timing_3d.py`?~~ — removed (superseded by inline Plotly widget)
- [x] ~~Decide: remove `bench_scaling.py`?~~ — removed (redundant with `bench_timing.py`)
- [x] ~~Delete stale branches~~ — done
- [ ] tests are too slow


### Packaging and Distribution

- [ ] Publish to PyPI

### Dev Quality

- [ ] Add linting config (ruff/flake8/mypy)
- [ ] Lint checkers
- [ ] Coverage
- [ ] Update README
