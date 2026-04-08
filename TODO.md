# TODO

## Content

- [x] Add a disclaimer that this post was written with extensive help from Claude Code, especially in the interactive widgets, which would not have been possible without Claude Code.
- [ ] Need to explain that controlling the inclusion probabilities is the key to the optimal, unbiased, k-sparse estimator of the distribution, which is what HT provides. The optimal inclusion probabilities are $\pi = \min(1, p_i \tau)$ where $\tau$ is the solution to $n = \sum_i \min(1, p_i \tau)$. Give a citation (or just prove it).
- [ ] Compare CPS to priority sampling in the HT estimation section — replicate the style of the "estimating means in a finite universe" blog post (https://timvieira.github.io/blog/post/2017/07/03/estimating-means-in-a-finite-universe/). Show variance reduction of CPS-HT vs priority sampling vs i.i.d. Monte Carlo on the same estimation problem. CPS gives optimal variance (max-entropy) but is more expensive to set up; priority sampling is $O(N \log N)$ but suboptimal. The comparison would make the HT section much more concrete.
- [ ] Review the sampling pseudocode
- [ ] Should we refer to polynomials more consistently as generating functions here?
- [x] Add link to Wikipedia page on [elementary symmetric polynomials](https://en.wikipedia.org/wiki/Elementary_symmetric_polynomial)

## Timing Section

- [ ] Hide all Python/plotting code—just show the plot. Link to relevant implementations using the checkmark/footnote pill system (e.g., each curve links to its implementation in the repo).
- [ ] Add more baselines for computing $\pi$ — (1) backprop-on-DP at $O(Nn)$, (2) naive loop versions: $N \times O(Nn)$ DP and $N \times O(N \log^2 n)$ product tree. These show the value of backprop vs. leave-one-out recomputation.
- [ ] Add a programmatically generated empirical slope table. For each method, fit log-log slopes (time vs $N$ at fixed $n$, and time vs $n$ at fixed $N$) from the grid sweep data and display alongside theoretical complexity bounds. Auto-flag rows where empirical slope deviates significantly from prediction.
- [ ] Scrutinize timing experiments. R's `sampling` package functions (`UPMEqfromw`, `UPMEpikfromq`, `UPMEsfromq`, `UPMEpiktildefrompik`) are confirmed pure R (no compiled C). Subprocess overhead and measurement methodology still need investigation. Verify that all methods compute the same quantity.
- [ ] Explore the fitting algorithm in the R `sampling` package. It may be faster simply because its fixed-point iteration has less overhead than L-BFGS (e.g., no secant equations). Compare the number of iterations required by the R fixed-point method vs. our L-BFGS solver.
- [ ] 3D timing plots: replace the single dropdown plot with three separate 3D plots (computing $Z$, computing $\pi$, drawing samples). Sampling timing is currently only a static 2D SVG—it has two variables ($N$, $n$) just like the others, so it should be 3D too.

## Implementation

### Code cleanup
- [x] ~~Remove stale aliases and floating functions~~ — done
- [x] ~~Replace Newton-CG with L-BFGS; remove hvp/D-tree/CG~~ — done
- [x] ~~Remove `sample_sequential`~~ — done
- [x] ~~Move tests to `tests/`~~ — done
- [ ] Add fixed-point iteration `fit` method (matching R's `UPMEpiktildefrompik`: `theta += pi_star - pi(theta)`) as an alternative to L-BFGS. Compare running time of L-BFGS vs fixed-point iteration.
- [ ] L-BFGS fitting convergence is much slower than the old Newton-CG (24 iterations vs 5 for N=10, non-monotone). Consider restoring Newton-CG as the default optimizer (requires HVP internally, not as public API) or tuning L-BFGS parameters.
- [x] ~~Remove boundary handling (w=0/inf) from `conditional_poisson/numpy.py`~~ — done (all implementations now require finite positive weights)
- [x] ~~Rename `conditional_poisson/numpy.py` → `tree_numpy.py` and `conditional_poisson/torch.py` → `tree_torch.py`~~ — done
- [x] ~~Remove stupid wrapper methods~~ — done (moved `compute_pi`, `forward_log_Z`, `_find_r`, poly mul into class as instance/static methods)
- [ ]  _get_sample_cdfs look way too complicated (and potentially slow) - I suspect it can be simplified considerably.
- [x] ~~apply a similar refactor to numpy implementation (and sequential implementations too)~~ — done

### Sequential implementations
- [x] ~~Add `fit` and `log_prob` to `ConditionalPoissonSequentialNumPy`~~ — done
- [x] ~~Add `fit` and `log_prob` to `ConditionalPoissonSequentialTorch`~~ — done
- [ ] Fix numerical overflow in sequential `_get_seq_q` — the ESP recurrence operates in linear space without log-scaling, producing NaN at N ≥ 500. Affects both NumPy and Torch sequential classes.
- [x] ~~`ConditionalPoissonSequentialTorch` should use `torch.autograd` for `incl_prob` (backprop on `log_normalizer`) instead of manual forward-backward DP~~ — done
- [x] ~~Sequential sampling should work directly from the forward DP table instead of recomputing a separate backward ESP table (`_get_seq_q`)~~ — done (samples from the backward table computed in `_compute_table`)
- [ ] Tree-based sampling should work similarly — compute split probabilities directly from the tree polynomials during the top-down walk, without a separate CDF precomputation pass
- [x] ~~All four implementations should have the same public interface~~ — done (`from_weights`, `fit`, `sample`, `log_prob`, `incl_prob`, `log_normalizer`, `n`, `N`, `theta`; `w` removed)
- [x] ~~Extend `tests/test_all_implementations.py` to cover `fit` and `log_prob`~~ — done

### Benchmarks
- [ ] Rerun `bench_timing.py` and regenerate `timing_data.json` + SVG plots (sampling data is stale after sampler rewrite)
- [ ] Rerun `bench_timing_grid.py` and update inline 3D widget data in the article (sampling rows are stale)
- [ ] Update `plot_timing.py` to include new sampling methods (sequential, PyTorch tree) in the SVG
- [x] ~~Update `bench_timing_grid.py` to use class methods~~ — done
- [ ] The static `timing_samples.svg` still shows old curves from the vectorized sampler — must be regenerated

### Other
- [ ] NumPy tree timing slopes (~1.15–1.45 in $N$) are higher than expected for $O(N \log^2 N)$. Investigate whether forcing FFT throughout gives cleaner scaling.
- [ ] Use numpy/pytorch's built-in bisect_left methods instead of manual binary search — likely faster. Verify empirically with a benchmark before switching.
- [ ] Test GPU performance — `conditional_poisson/torch.py` (FFT) should be benchmarked on GPU. Float32 precision risk needs testing (contour scaling helps but may not fully compensate).
- [x] Promote `conditional_poisson_torch.py` to the primary library implementation — merged `torch_fft_prototype.py` into it with full `ConditionalPoissonTorch` class (from_weights, fit, sample, log_prob, pi, hvp). Blog post Basic Usage updated.
- [x] The maximum-entropy test could be strengthened by actually optimizing over the space of all distributions over size-$n$ sets, rather than just checking against a few specific alternatives.

## Blog Polish

- [x] ~~Add a note encouraging readers to report issues on the [GitHub issue tracker](https://github.com/timvieira/conditional-poisson-sampling/issues)~~ — done
- [x] ~~Add a citation section at the end of the post (e.g., BibTeX snippet, suggested citation format)~~ — done
- [ ] Why are there still references to the NumPy implementation in the article? The timing section legitimately benchmarks the NumPy tree, but it should be framed as the reference/pedagogical implementation, not the recommended one.
- [ ] Consistently color code math symbols and widgets — use the same colors for $P(S)$, $\pi$, and $w$ in both LaTeX and D3 widgets. Previously removed all color coding; bring it back in a principled way with a shared palette.

## Interactive Documentation

- [ ] Integrate Pyodide-based interactive Python cells into the blog post so readers can edit and run code in the browser. Prototype in `mockup_pyodide.html`. Uses a pure-NumPy `ConditionalPoissonNumPy` class (O(N²n) DP—fine for demo sizes). Basic Usage section is the obvious first candidate; fitting and sampling sections could follow.

## Widget UX

- [ ] Inclusion probability widget: hard to distinguish forward values from adjoint (backward) values. Explore different layouts. Also, inclusion probabilities appear at the bottom rather than the top—keep the forward-value + backward-value visual metaphor consistent.
- [ ] Scrutinize the little histograms in the sampling animation. What are they supposed to be? Why do they appear to change after sampling?
- [ ] Make sure the sampling animation widget is faithful to the implementation.
- [ ] Several widgets overflow their containers on mobile. Fix the layout to fill viewport width on small screens, then add horizontal scroll for wide SVGs.
- [ ] Centralize widget/layout styles into the blog's CSS (`~/projects/blog/main/content/css/blog.css`) instead of inline styles in the .md file.
- [x] Clicking "Gradient Descent" and "Archive" in the header should navigate to the blog's archive page.

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

## Repo Housekeeping

### Cleanup

- [ ] Tidy up the repo (clean up unused files, organize structure)
- [x] ~~Remove `memory/` directory (Claude session state, shouldn't be checked in)~~ — done
- [x] ~~Remove `bench_scaling.png`~~ — done
- [x] ~~Audit dead code: `torch_prototype.py`~~ — removed
- [x] ~~Remove `display_utils.py`~~ — removed (only consumer was the deleted notebook)
- [ ] Fix `test_animation.mjs` — update paths, verify it runs, integrate into CI
- [ ] Add mathematical correctness tests for JS widget algorithms (product tree, quota splitting, polynomial multiplication, CDF computation). Either extend `test_animation.mjs` to check computed values against known answers, or extract JS algorithms into testable modules and test with Node.js.
- [x] ~~Decide: remove `plot_timing_3d.py`?~~ — removed (superseded by inline Plotly widget)
- [x] ~~Decide: remove `bench_scaling.py`?~~ — removed (redundant with `bench_timing.py`)
- [x] ~~Delete stale branches~~ — done

### Packaging and Distribution

- [x] ~~Add a `LICENSE` file (pyproject.toml says MIT but no license file exists)~~ — done
- [x] ~~Package as `conditional_poisson` with all four implementations~~ — done
- [x] ~~Add optional-dependencies for dev/torch~~ — done
- [ ] Package up the JavaScript library as an easy-to-install npm module
- [ ] Publish to PyPI

### Dev Quality

- [x] ~~Set up CI via GitHub Actions~~ — done (Python tests + JS tests)
- [ ] Add linting config (ruff/flake8/mypy)
- [ ] Update README

## Bugs

(none)

