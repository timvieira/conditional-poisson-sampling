# TODO

## Content

- [ ] $P(i \in S)$ is unclear notation
- [x] Add a disclaimer that this post was written with extensive help from Claude Code, especially in the interactive widgets, which would not have been possible without Claude Code.

- [ ] Widget contents spill out of containers on narrow screens (e.g., phone). Looks very bad.
- [ ] Do we like the linear-gradient background on the widgets?
- [ ] Sampling animation widget: visual language is inconsistent with the product tree widget, and some of the pop-up histograms are confusing. Consider highlighting the current step in the pseudocode to anchor the viewer.
- [x] Review the sampling pseudocode
- [ ] Should we refer to polynomials more consistently as generating functions here?
- [ ] Would evaluating/plotting the polynomial as a function of z tell us anything interesting?
- [x] Add link to Wikipedia page on [elementary symmetric polynomials](https://en.wikipedia.org/wiki/Elementary_symmetric_polynomial)

## Timing Section

- [ ] Timing section is still a mess — results are mostly missing, and many results don't make sense yet.
- [ ] Is the 3D plot using old data? It doesn't appear to be updated. Where is it getting its data from?
- [ ] Contour diagram needs work.
- [ ] "Numerical validation" section — not sure this is the best way to present this information.
- [ ] Hide all Python/plotting code—just show the plot. Link to relevant implementations using the checkmark/footnote pill system (e.g., each curve links to its implementation in the repo).
- [ ] Add more baselines for computing $\pi$ — (1) backprop-on-DP at $O(Nn)$, (2) naive loop versions: $N \times O(Nn)$ DP and $N \times O(N \log^2 n)$ product tree. These show the value of backprop vs. leave-one-out recomputation.
- [ ] Add a programmatically generated empirical slope table. For each method, fit log-log slopes (time vs $N$ at fixed $n$, and time vs $n$ at fixed $N$) from the grid sweep data and display alongside theoretical complexity bounds. Auto-flag rows where empirical slope deviates significantly from prediction.
- [ ] Scrutinize timing experiments. R's `sampling` package functions (`UPMEqfromw`, `UPMEpikfromq`, `UPMEsfromq`, `UPMEpiktildefrompik`) are confirmed pure R (no compiled C). Subprocess overhead and measurement methodology still need investigation. Verify that all methods compute the same quantity.
- [ ] Explore the fitting algorithm in the R `sampling` package. It may be faster simply because its fixed-point iteration has less overhead than L-BFGS (e.g., no secant equations). Compare the number of iterations required by the R fixed-point method vs. our L-BFGS solver.
- [ ] 3D timing plots: replace the single dropdown plot with three separate 3D plots (computing $Z$, computing $\pi$, drawing samples). Sampling timing is currently only a static 2D SVG—it has two variables ($N$, $n$) just like the others, so it should be 3D too.

## Implementation

### Code cleanup
- [ ] Add fixed-point iteration `fit` method (matching R's `UPMEpiktildefrompik`: `theta += pi_star - pi(theta)`) as an alternative to L-BFGS. Compare running time of L-BFGS vs fixed-point iteration.
- [ ] L-BFGS fitting convergence is much slower than the old Newton-CG (24 iterations vs 5 for N=10, non-monotone). Consider restoring Newton-CG as the default optimizer (requires HVP internally, not as public API) or tuning L-BFGS parameters.

### Benchmarks
- [ ] Rerun `bench_timing.py` and regenerate `timing_data.json` + SVG plots (sampling data is stale after sampler rewrite)
- [ ] Rerun `bench_timing_grid.py` and update inline 3D widget data in the article (sampling rows are stale)
- [ ] Update `plot_timing.py` to include new sampling methods (sequential, PyTorch tree) in the SVG
- [ ] The static `timing_samples.svg` still shows old curves from the vectorized sampler — must be regenerated

### Other
- [ ] NumPy tree timing slopes (~1.15–1.45 in $N$) are higher than expected for $O(N \log^2 N)$. Investigate whether forcing FFT throughout gives cleaner scaling.
- [ ] Use numpy/pytorch's built-in bisect_left methods instead of manual binary search — likely faster. Verify empirically with a benchmark before switching.
- [ ] Test GPU performance — `conditional_poisson/tree_torch.py` (FFT) should be benchmarked on GPU. Float32 precision risk needs testing (contour scaling helps but may not fully compensate).

## Blog Polish
- [x] ~~Add a note encouraging readers to report issues on the [GitHub issue tracker](https://github.com/timvieira/conditional-poisson-sampling/issues)~~ — done
- [x] ~~Add a citation section at the end of the post (e.g., BibTeX snippet, suggested citation format)~~ — done
- [ ] Why are there still references to the NumPy implementation in the article? The timing section legitimately benchmarks the NumPy tree, but it should be framed as the reference/pedagogical implementation, not the recommended one.
- [ ] Consistently color code math symbols and widgets — use the same colors for $P(S)$, $\pi$, and $w$ in both LaTeX and D3 widgets. Previously removed all color coding; bring it back in a principled way with a shared palette.

## Widget UX

- [ ] Inclusion probability widget redesign:
  - [x] Problem: forward and adjoint values are interleaved in the same node boxes, making it hard to parse
  - [ ] **Side-by-side trees**: forward pass (blue) on the left, backward pass (purple) on the right. Same layout, nodes correspond visually. Simplest win.
  - [ ] **Animate the two passes**: show forward first, then backward flowing back, rather than everything at once
  - [ ] **Highlight on hover**: when hovering over a pi_i, highlight the path from root adjoint down to that leaf
  - [ ] **Simplify node labels**: show the key coefficient (e.g., c_n for the root) prominently, with the full histogram as supporting detail
  - [ ] Move inclusion probabilities to the top (near the leaves they come from) rather than the bottom
- [ ] Scrutinize the little histograms in the sampling animation. What are they supposed to be? Why do they appear to change after sampling?
- [ ] Make sure the sampling animation widget is faithful to the implementation.
- [ ] Several widgets overflow their containers on mobile. Fix the layout to fill viewport width on small screens, then add horizontal scroll for wide SVGs.
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
- [ ] remove all the useless ascii art: # ── Fitting ─────────────────


### Packaging and Distribution

- [ ] Publish to PyPI

### Dev Quality

- [ ] Add linting config (ruff/flake8/mypy)
- [ ] Lint checkers
- [ ] Coverage
- [ ] Update README
