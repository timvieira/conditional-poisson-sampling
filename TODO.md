# TODO

## Content

- [ ] Would evaluating/plotting the polynomial as a function of z tell us anything interesting?

## Timing Section

- [ ] Contour diagram needs work.

## Implementation

- [ ] NumPy tree timing slopes (~1.15–1.45 in $N$) are higher than expected for $O(N \log^2 N)$. Investigate whether forcing FFT throughout gives cleaner scaling.
- [ ] Use numpy/pytorch's built-in bisect_left methods instead of manual binary search — likely faster. Verify empirically with a benchmark before switching.
- [ ] Test GPU performance — `conditional_poisson/tree_torch.py` (FFT) should be benchmarked on GPU. Float32 precision risk needs testing (contour scaling helps but may not fully compensate).

## Blog Polish

- [ ] Consistently color code math symbols and widgets — use the same colors for $P(S)$, $\pi$, and $w$ in both LaTeX and D3 widgets. Previously removed all color coding; bring it back in a principled way with a shared palette.

## Widget UX

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

### Packaging and Distribution

- [ ] Publish to PyPI
