# Blog Post TODO

## Content

- [x] Add a disclaimer that this post was written with extensive help from Claude Code, especially in the interactive widgets, which would not have been possible without Claude Code.
- [ ] Need to explain that controlling the inclusion probabilities is the key to the optimal, unbiased, k-sparse estimator of the distribution, which is what HT provides. The optimal inclusion probabilities are $\pi = \min(1, p_i \tau)$ where $\tau$ is the solution to $n = \sum_i \min(1, p_i \tau)$. Give a citation (or just prove it).
- [ ] Compare CPS to priority sampling in the HT estimation section — replicate the style of the "estimating means in a finite universe" blog post (https://timvieira.github.io/blog/post/2017/07/03/estimating-means-in-a-finite-universe/). Show variance reduction of CPS-HT vs priority sampling vs i.i.d. Monte Carlo on the same estimation problem. CPS gives optimal variance (max-entropy) but is more expensive to set up; priority sampling is $O(N \log N)$ but suboptimal. The comparison would make the HT section much more concrete.
- [ ] Should we refer to polynomials more consistently as generating functions here?
- [x] Add link to Wikipedia page on [elementary symmetric polynomials](https://en.wikipedia.org/wiki/Elementary_symmetric_polynomial)

## Timing Section

- [ ] Hide all Python/plotting code—just show the plot. Link to relevant implementations using the checkmark/footnote pill system (e.g., each curve links to its implementation in the repo).
- [ ] Add more baselines for computing $\pi$ — (1) backprop-on-DP at $O(Nn)$, (2) naive loop versions: $N \times O(Nn)$ DP and $N \times O(N \log^2 n)$ product tree. These show the value of backprop vs. leave-one-out recomputation.
- [ ] Add a programmatically generated empirical slope table. For each method, fit log-log slopes (time vs $N$ at fixed $n$, and time vs $n$ at fixed $N$) from the grid sweep data and display alongside theoretical complexity bounds. Auto-flag rows where empirical slope deviates significantly from prediction.
- [ ] 3D timing plots: replace the single dropdown plot with three separate 3D plots (computing $Z$, computing $\pi$, drawing samples). Sampling timing is currently only a static 2D SVG—it has two variables ($N$, $n$) just like the others, so it should be 3D too.

## Implementation

- [ ] NumPy tree timing slopes (~1.15–1.45 in $N$) are higher than expected for $O(N \log^2 N)$. Investigate whether forcing FFT throughout gives cleaner scaling.
- [ ] Test GPU performance — both `torch_fft_prototype.py` (FFT) and `torch_prototype.py` (direct conv1d) should be benchmarked on GPU. Float32 precision risk needs testing (contour scaling helps but may not fully compensate).
- [ ] Promote `torch_fft_prototype.py` to the primary library implementation — it's faster ($O(N \log^2 n)$ vs $O(N \log^2 N)$), simpler, and supports autograd. Need to: add sampling, match the full `ConditionalPoisson` API (fit, log_prob, etc.), add tests, decide on package structure. The NumPy implementation becomes the pedagogical/reference version.
- [ ] Package up the NumPy, PyTorch, and JavaScript libraries as easy-to-install single-file libraries via pip/npm. (`pyproject.toml` and `package.json` exist but are not polished for distribution.)
- [ ] The maximum-entropy test could be strengthened by actually optimizing over the space of all distributions over size-$n$ sets, rather than just checking against a few specific alternatives.

## Blog Polish

- [ ] Why are there still references to the NumPy implementation in the article? The timing section legitimately benchmarks the NumPy tree, but it should be framed as the reference/pedagogical implementation, not the recommended one.
- [ ] Consistently color code math symbols and widgets — use the same colors for $P(S)$, $\pi$, and $w$ in both LaTeX and D3 widgets. Previously removed all color coding; bring it back in a principled way with a shared palette.
- [ ] Make the $P(S)$ horizontal bars in the interactive explorer taller — ideally same width as the other bars.

## Widget UX

- [ ] Inclusion probability widget: hard to distinguish forward values from adjoint (backward) values. Explore different layouts. Also, inclusion probabilities appear at the bottom rather than the top—keep the forward-value + backward-value visual metaphor consistent.
- [ ] Several widgets overflow their containers on mobile. Fix the layout to fill viewport width on small screens, then add horizontal scroll for wide SVGs.
- [ ] Centralize widget/layout styles into the blog's CSS (`~/projects/blog/main/content/css/blog.css`) instead of inline styles in the .md file.
- [ ] Clicking "Gradient Descent" and "Archive" in the header should navigate to the blog's archive page.

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

## Bugs

- [ ] Implementation pill links are broken — the pills render fine but their href targets don't resolve to anything. Fix the links to point at actual implementations. Make the linking robust/future-proof (e.g., link to a function name search or a stable anchor rather than fragile line numbers).

