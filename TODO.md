# Blog post TODO

## Next Up

- [x] Weighted Pascal recurrence needs base cases.
- [x] Weighted Vandermonde identity: the statement jumps straight to the generating function proof without connecting the combinatorial claim to the generating function setup. Bridge the gap — explain why the identity follows from factoring the generating function before diving into the proof.
- [ ] Speed comparison against the R packages (UPmaxentropy in `sampling`, BalancedSampling) — show wall-clock times for computing π and drawing samples at various N.
- [ ] Make the P(S) horizontal bars in the interactive explorer taller - ideally, as similar/same width to the other bars
- [ ] Add a disclaimer that this post was written with extensive help from Claude Code, especially in the interactive widgets, which would not have been possible without Claude Code.
- [ ] Timing section: hide all Python/plotting code—just show the plot. Link to relevant implementations using the checkmark/footnote pill system (e.g., each curve links to its implementation in the repo).
- [ ] Timing section: add more baselines for computing π — (1) backprop-on-DP at O(Nn), (2) naive loop versions: N × O(Nn) DP and N × O(N log² n) product tree. These show the value of backprop vs. leave-one-out recomputation.
- [ ] Need to explain that controlling the inclusion probabilities is the key to the optimal, unbiased, k-sparse estimator of the distribuion, which is what HT provides.  The optimal inclusion probabilities are \pi = min(1, p_i \tau) where \tau is the solution to n = \sum_i min(1, p_i \tau).  Give a citation (or just prove it).

- [ ] Should we refer to polynomials more consistently as generating functions here?
- [ ] Add link to Wikipedia page on [elementary symmetric polynomials](https://en.wikipedia.org/wiki/Elementary_symmetric_polynomial)
- [ ] Why are there still references to the NumPy implementation in the article? Remove all remaining mentions.
- [ ] Package up the NumPy, PyTorch, and JavaScript libraries that are byproducts of this post. They should be easy to install or use as single-file libraries, and installable via package managers (pip, npm).

## Confusing / Poorly Explained

- [x] Motivate the name "conditional Poisson" — bridge from Poisson sampling (independent inclusion) to conditioning on size
- [x] Introduce Scott bracket notation $\llbracket f \rrbracket(z^k)$ before first use, or switch to the more standard $[z^k] f(z)$
- [x] Explain or remove the D-tree parenthetical — currently a black box ("piggybacks on the P-tree" says nothing)
- [x] Fix $\text{Cov}[Z]$ notation — $Z$ is the normalizing constant everywhere else; use $\text{Cov}[\mathbf{1}_S]$ or $\Sigma$ for the inclusion covariance
- [x] Expand the downward pass explanation — a small worked example (N=4) with actual polynomial coefficients at each node would help a lot
- [x] Justify the $O(n \log N)$ per-sample cost — why not $O(N \log N)$? (only $O(n)$ nodes have nonzero quota per level)
- [x] Add a one-line derivation for the fitting log-likelihood ($\log P(S) = \sum_{i \in S} \theta_i - \log Z$, take expectation under target)

## Potentially Unnecessary / Uninteresting (fix or cut)

- [x] Numerical stability section: either show a case that *breaks* without scaling (before/after) or cut to one sentence
- [x] Timing section: add comparison against naive $O(Nn)$ DP or R implementations to give context; raw ms alone is low-signal
- [x] Construction 2 (Categorical): develop the birthday-problem connection and crossover analysis, or demote to a footnote

## Missing Visual Aids / Pseudocode / Math

- [x] Add the Mermaid tree diagrams from the README into the blog post (upward pass, downward pass, sampling)
- [x] Add pseudocode for the sampling procedure (recursive quota-splitting, ~5-10 lines)
- [x] Add a scatter plot of $w_i$ vs $\pi_i$ to build intuition for the nonlinear relationship
- [x] Small example (Cell 4): added bar chart comparing rescaled weights ($w_i / W \cdot n$) vs $\pi_i$ — both sum to $n$ for visual comparability
- [x] ~~Investigate whether there's a more principled scale correction factor~~ — addressed: contour radius r derived from Poisson approximation / saddlepoint is already principled.
- [x] Justify convexity of the fitting problem in one sentence (log-partition function of an exponential family)
- [x] Move the "maximum entropy" characterization earlier — it's a major selling point, currently buried in the summary
- [x] Add a "Motivation" paragraph with 2-3 concrete use cases (survey sampling, beam search, subset selection in ML)

## Mathematical / Logical Errors

- [x] Check whether the expected size equation needs clipping: no, each term $w_i r/(1+w_i r) \in (0,1)$ for $w_i, r > 0$.
- [x] Rejection bound in Cell 4 — added expandable derivation following De Vita (2023) and defined $W$
- [x] Brute-force log Z verification (Cell 15) is circular — it adds `cp_small.log_normalizer` to already-normalized log-probs and recovers it; compute log Z from brute force independently instead
- [x] Fitting objective (Cell 16) is called "log-likelihood" but is actually the expected log-probability E_{S~P*}[log P_θ(S)]; call it "expected log-probability" or "cross-entropy objective"
- [x] I don't think P_L and P_R are defined anywhere - can we avoid using them or do we need to define them?
- [ ] write the sampling pseudocode in the same style as the rejection sampler.

## Undefined / Missing Definitions

- [x] $W = \sum_i w_i$ — now defined in Cell 4 below the acceptance rate table
- [x] "Exponential fixed-size design" (Cell 1) — explain where the name comes from (exponential family structure)
- [x] Notation switch from $\mathcal{S}$ to $\mathcal{Y}$ in the HT estimator section (Cell 23) is unexplained

## Missing Citations / Links

- [x] No citation for the product tree algorithm — cite subproduct tree literature (Ben-Or 1983, Macdonald) or state explicitly what's novel about this combination
- [x] Newton's identities (Cell 24) stated without citation — add pointer (e.g., Stanley's Enumerative Combinatorics)
- [x] Horvitz-Thompson estimator (Cell 23) not cited — add Horvitz & Thompson (1952)
- [x] Pull key intuition from the "heaps for incremental computation" link forward instead of just linking

## Structure and Flow

- [x] Add a roadmap after the intro so readers know the arc of the post (rejection samplers → tree → fitting)
- [x] Lead with a small concrete example before the abstract generating function machinery — anchor notation in numbers before abstraction
- [x] Integrate the N=4 worked example *into* the tree explanation (upward/downward) rather than placing it after all three passes
- [x] The intro paragraph does too much at once — split definition, etymology, max-entropy property, and exponential family connection into digestible chunks
- [x] "Basic usage" (Cells 7-10) arrives before the reader understands what it's computing — either commit to a tutorial-first framing (and say so) or defer until after the tree explanation
- [x] The rejection sampling section's punchline ("neither gives you Z, π, or gradients") should be emphasized as the gap the tree fills — frame the contrast as "easy to understand, impossible to scale" vs. "the tree makes it all efficient"
- [x] The intro introduces ~10 symbols ($\mathcal{S}$, $N$, $n$, $w_i$, $S$, $\binom{\mathcal{S}}{n}$, $Z$, $p_i$, $\theta_i$, $\pi_i$) before any concrete example — anchor notation in numbers first
- [x] The product tree section covers too much in one pass (upward, downward, sampling, worked example, brute-force) — consider breaking it into smaller sections or interleaving explanation with examples

## Exposition Gaps

- [x] Give the D-tree more space — but frame it as Pearlmutter's R-operator applied to the backward pass, not as a separate algorithm. Currently it's a single opaque sentence; it should be presented as the next mechanical transformation in the stack (forward → backward → HVP)
- [x] The fitting section is compressed overall — objective, convexity, gradient, Hessian, Newton-CG, and D-tree are all packed into one paragraph; the D-tree is the worst case but the whole section needs room to breathe
- [x] Explain *why* the sampling split is exact — the specific insight is that the split probabilities $P_L[j] \cdot P_R[k-j]$ are the *conditional* probabilities of the target distribution (via Vandermonde); state this explicitly near the pseudocode
- [x] The downward pass *is* backpropagation (not an analogy) — the post should make this unmistakable: define the forward pass (polynomial multiplication in a tree), then the backward pass follows mechanically. The current explanation jumps through too many intermediate framings (gradient, exponential family, leave-one-out, backprop, tree walk) instead of letting backprop do the explanatory work
- [x] The entire algorithmic stack is mechanical program transformations with known cost guarantees — the post should make this explicit:
    - Forward pass ($Z$): $O(N \log^2 N)$ — tree-structured polynomial multiplication
    - Gradient ($\pi$): $O(N \log^2 N)$ — reverse-mode AD / backprop (Baur & Strassen, 1983)
    - HVP ($\text{Cov}[\mathbf{1}_S]\,v$): $O(N \log^2 N)$ — Pearlmutter's R-operator (1994) applied to the gradient computation
    Each level costs O(1)× the previous by a general theorem, not a problem-specific derivation. Keep Tim's blog post links as the primary exposition; cite Baur & Strassen (1983), Griewank & Walther (2008), and Pearlmutter (1994) in the references
- [x] The Horvitz-Thompson section needs both setup/framing (it arrives with no transition) and a small worked example — currently reads as "see also" rather than a demonstration
- [x] The max-entropy property is stated with a citation but never demonstrated or given intuition

## Restructuring

- [x] Restructure the "Identities" section — added Vandermonde cross-reference to tree/sampling sections, added parameterization cross-reference from intro, annotated Vandermonde with its role
- [x] The identities section has redundancy — deduplicated differential identities into brief recap with cross-references to exp family and fitting sections
- [x] K-DPP connection: collapsed into a `<details>` block (interesting but tangential)
- [x] The summary repeats but doesn't synthesize — replaced bullet list with a capabilities table showing the mechanical transformation stack, added "no problem-specific derivations" takeaway

## Implementation

- [ ] NumPy tree timing slopes (~1.15–1.45 in N) are higher than expected for O(N log² N). The implementation has complicated dispatching that may switch between convolution methods (direct vs FFT) at different sizes. Investigate whether forcing FFT throughout gives cleaner scaling. May also be a truncation issue.
- [x] ~~Batch polynomial multiplications in NumPy~~ — dropped. The NumPy implementation is the pedagogical/reference version; optimize for readability, not speed. The torch implementation is the fast path.
- [x] Recover sub-O(Nn) complexity with numerical stability — **solved via contour radius scaling** (torch_fft_prototype.py). Rescale weights w_i -> w_i*r where r shifts the product polynomial's peak to degree n. FFT rounding errors are now relative to the coefficient we need, not a distant peak. r = exp(t) where t is the Poisson sampling Lagrange multiplier (sum sigmoid(log w_i + t) = n). Result: O(N log² n), machine-epsilon precision, 10-16x faster than NumPy, fully differentiable.
    - The earlier obstacle (FFT/Karatsuba subtractions cancelling small coefficients) was a dynamic range problem, not a fundamental algebraic one — contour scaling eliminates the dynamic range.
- [ ] Test GPU performance — both torch_fft_prototype.py (FFT) and torch_prototype.py (direct conv1d) should be benchmarked on GPU. The FFT version is faster on CPU, but direct convolution may have better GPU characteristics (no FFT synchronization, better memory access patterns). Keep both implementations as the benchmark in torch_fft_prototype.py's __main__ compares them. float32 precision risk needs testing (contour scaling helps but may not fully compensate).
- [x] Compare Newton-CG fitting to simpler optimizers — L-BFGS with gradient only is 2-4x faster than Newton-CG with HVP. D-tree/HVP demoted to bonus appendix in blog post.
- [x] **Fitting optimizer is a hot mess** — resolved: torch's LBFGS does implement H₀ scaling (sᵀy/yᵀy), and setting `tolerance_grad=tol` gives a principled stopping rule: the gradient is π(θ) - π★, so the optimizer stops when max|π - π★| ≤ tol. Documented in docstring.
- [ ] Promote torch_fft_prototype.py to the primary library implementation — it's faster (O(N log² n) vs O(N log² N)), simpler (no hand-coded downward pass or D-tree), and supports autograd for integration as a neural network layer. The NumPy implementation (conditional_poisson_numpy.py) becomes a pedagogical/reference implementation that makes the algorithm transparent without requiring PyTorch — keep it, but frame it as the teaching version. Need to: add sampling to the torch version, match the full ConditionalPoisson API (fit, log_prob, etc.), add tests, decide on the package structure.

## Bugs

- [ ] Implementation pill links are broken — the pills render fine but their href targets don't resolve to anything. Fix the links to point at the actual implementations. Make the linking robust/future-proof so links don't rot when code moves (e.g., link to a function name search or a stable anchor rather than fragile line numbers).

- [x] HT section (Cell 40): orphaned sentence — rewrote section, moved unbiasedness right after formula
- [x] HT section (Cell 40): redundant "advantage over MC" paragraph — removed, covered by Setup
- [x] Speedup discussion (Cell 39): fixed `torch_fft_prototype.py` link to `/blob/main/`
- [x] Identities parameterization table (Cell 41): fixed `[0,∞)` → `[0,∞]` to match intro
- [x] Fitting section (Cell 27): fixed "convex" → "concave (since log Z is convex as a log-partition function)"
- [x] Jaynes (1957) added to References section (Cell 42)

## Confusing

- [x] Micro-example (Cell 4) uses 1-indexed, code uses 0-indexed — added indexing note in code, kept math 1-indexed
- [x] Downward pass diagram N=8 vs code N=4 — switched worked example to N=8, n=3 to match diagrams
- [x] "## Sampling" heading confused with "### Sampling" — renamed to "## Drawing samples"

## Editorial

- [x] Cell 5: "CPS" abbreviation never introduced — replaced with "the conditional Poisson distribution"
- [x] Cell 10: "**The gap.**" split into its own paragraph
- [x] Cell 42: extra blank line before References removed

## Notation / Formatting

- [x] Bold vectors consistently: added `\btheta` and `\bpip` macros, applied throughout notebook. Scalars (θ_i, π_i) stay unbolded.
- [x] Title Case all subsection headings.
- [x] Verification checkmark pills: placed after punctuation like footnote markers. (Sizing/padding still TODO.)
- [x] Contour scaling section rewritten with full derivation: Cauchy integral formula motivation, why we get to choose r, saddlepoint/tilting-parameter derivation, connection to Poisson expected size. Title-cased heading.
- [x] Add a diagram of the contour integral in the complex plane — done: interactive D3 widget in div#contour-diagram.
- [x] Numerical validation of contour radius: added code cell showing r=1 gives 10^16 dynamic range and wrong log Z, while r=r* gives dynamic range 1 and machine-precision log Z.
- [x] Fix complexity column in summary table: replaced +O(1)× with O(N log² n), fixed D-tree table too. Also fixed O(N log² N) → O(N log² n) throughout (truncation). Baur-Strassen 5× constant cited in Cell 35.
- [x] Fitting cost breakdown: added per-iteration cost paragraph to Cell 27 (line search × O(N log² n) eval + O(mN) two-loop).
- [x] Summary table: replaced "~15-50 iterations" with "O(N log² n) per iteration".
- [ ] Compare CPS to priority sampling in the HT estimation section — replicate the style of the "estimating means in a finite universe" blog post (https://timvieira.github.io/blog/post/2017/07/03/estimating-means-in-a-finite-universe/). Show variance reduction of CPS-HT vs priority sampling vs i.i.d. Monte Carlo on the same estimation problem. CPS gives optimal variance (max-entropy) but is more expensive to set up; priority sampling is O(N log N) but suboptimal. The comparison would make the HT section much more concrete.
- [x] Explain truncation to degree n: added paragraph in Cell 11 after Master Theorem — convolution of truncated polynomials gives correct coefficients up to degree n.
- [x] PyTorch timing: merged into a single log-log plot with DP baseline, NumPy tree, PyTorch FFT, and HVP. Removed text table.
- [x] Stray `[z^k]` notation in worked example code (Cell 16 table header) — fixed to Scott brackets.
- [x] Sampling timing plot: added O(n log N) reference line as baseline.
- [x] Mention alternative O(Nn) algorithm: added note after Pascal recurrence in Cell 41 — DP forward gives Z, backprop gives π, all in O(Nn).
- [x] Rename tree subsections: "Upward Pass" → "Computing the Normalizing Constant Z", "Downward Pass (Backpropagation)" → "Computing Inclusion Probabilities π", "Sampling" → "Drawing Exact Samples". Updated #Sampling anchor in Cell 41.
- [x] Remove NumPy downward pass code: deleted the hand-coded downward pass and π computation code cells. Kept the conceptual explanation and mermaid diagram.
- [x] Audit and remove stale "pass" terminology: replaced "upward/downward pass" with "product tree"/"backpropagation" in prose. Kept "forward/backward pass" in summary table (standard AD terms). The headings were renamed but the body text still references "upward pass", "downward pass", "forward pass", "backward pass" etc. Since the PyTorch implementation uses autograd (no hand-coded downward pass), much of this language is outdated. Reframe in terms of *what is computed* (Z, π, samples) rather than implementation passes. Check: Cell 6 outline, Cell 13 body, Cell 27 D-tree section, Cell 35 intro, summary table.
- [x] Explain polynomial = coefficient array + FFT trick: added "Polynomials as arrays" paragraph to Cell 11, covering representation, convolution, and O(d log d) FFT speedup. References (3B1B, CLRS) can be added later.
- [x] Rejection sampling Cell 9: converted from executable code + histogram plots to pseudocode in a markdown cell.
- [x] Smooth out the opening: dropped bold question, removed unsupported applications, used ∝, merged imports into first code cell, tightened Cell 3→5 flow.
- [x] Small example uses non-integer weights: changed to w = (1.5, 3.2, 0.8, 4.5). Updated Cell 4 table, Z, π values, bar chart, and Cell 12 w_ex.
- [x] Add indicator columns to the Cell 4 example table: show each subset as both a set label $\{1,2\}$ and a row of 0/1 indicators for each item. This makes the indicator structure visible and sets up π as the (weighted) column sum. Keep the set label as the row label. Add a bottom row showing $\pi_i = \mathbb{E}[\mathbf{1}_i]$ for each column—the inclusion probability is literally the expected value of the indicator, computed as the P(S)-weighted column sum.
- [x] ~~Worked examples should not assume N is a clean power of 2~~ — dropped; power-of-2 examples are clearer for teaching and the code handles arbitrary N transparently.
- [x] Reconcile w_i domain: added WLOG blockquote in Cell 41 parameterizations section — w_i=0 excluded, w_i=∞ deterministically included, general case reduces to finite positive.
- [x] Refactor display-heavy code cells into `display_utils.py` — done: html_table, check_mark, poly_html moved to display_utils.py.
- [x] Removed color coding from all macros (kept macro structure). Removed color legend.
- [x] Vector concatenation notation: defined on first use before Vandermonde identity in Cell 41.
- [x] Weighted Vandermonde identity: promoted to Proposition with collapsible proof (factor the product, equate z^k coefficients). Cited unweighted Vandermonde's identity. The proof follows from expanding the product $\prod_{i \in A \cup B}(1 + w_i z) = \prod_{i \in A}(1 + w_i z) \cdot \prod_{i \in B}(1 + w_i z)$ and collecting the coefficient of $z^k$ via convolution. Currently stated without proof in the Recurrences and Algorithms subsection of Cell 42. Cite the unweighted version: [Vandermonde's identity](https://en.wikipedia.org/wiki/Vandermonde%27s_identity). Include a short proof (e.g., in a footnote or `<details>` block) — it's only one line: factor the product, equate z^k coefficients.

- [ ] Consistently color code math symbols and widgets — use the same colors for $P(S)$ (set probabilities), $\pi$ (inclusion probabilities), and $w$ (weights) in both LaTeX math and the interactive D3 widgets. Previously removed all color coding; bring it back in a principled way with a shared palette.

## Widget UX

- [ ] Inclusion probability widget: hard to distinguish forward values from adjoint (backward) values. Explore different layouts to make this clearer. Also, inclusion probabilities appear at the bottom rather than the top—should keep the forward-value + backward-value visual metaphor consistent (forward on top, adjoint/backward below).
- [ ] Sampling pseudocode: move the `W=` line outside the loop—currently makes the line too long.
- [x] Remove stale references to the defunct NumPy implementation.

## Visualization

- [x] Interactive widget: D3 widget with draggable w and π bars, live subset table, N/n controls. Embedded in blog post, replaces static example. where readers can drag sliders to adjust weights and see the subset probabilities, normalizing constant, inclusion probabilities, and bar chart update in real time. Embed directly in the notebook (Jupyter supports `%%html`/`%%javascript` cells). This would make the relationship between weights and inclusion probabilities tangible—currently it's just a table of numbers. **Bonus:** eliminates the hardcoded-number problem entirely—all values are computed programmatically, so they can never be stale. Should support switching parameterizations: specify w and compute π (forward), or specify target π and fit w (inverse/fitting). This ties the intro example directly to the fitting section and lets readers see both directions interactively.
- [x] D3 animation of the sampling algorithm: show the binary tree with quotas at each node, animate the top-down walk step by step. At each internal node, display the split equation P_L[j] · P_R[k−j] as a small distribution, sample from it, then propagate the resulting quotas to the children. Highlight the path from root to leaves, show items being selected (quota=1) or excluded (quota=0) at the leaves. Could be an interactive widget embedded in the notebook or a standalone HTML page linked from the blog post.

## Checklist (standing editorial rules)

Apply before every commit touching the notebook:
- [x] All vector quantities use `\boldsymbol` in LaTeX
- [x] All subsection titles use Title Case
- [x] Consolidated notation box: added collapsible table after the intro defining all symbols (w, θ, p, π, Z, n, N, S, z, Scott brackets). Uses `<details class="derivation">` for blog styling.
- [x] Clean up w/p/θ parameterization confusion. Audited: every transition between w, p, θ is explicit with a \defeq or conversion formula. The post defines w (odds) first, introduces p as the conversion, uses θ for exponential family/fitting. No implicit switches found.
- [x] "is asymptotically exact" → replaced with "has initialization error O(1/N) per item (Hájek 1964, Theorem 5.2)" with the regularity condition stated.
- [x] Verification pills appear after punctuation, not before
- [x] Color coding removed; legend removed
- [x] Every mathematical claim has a ✓ pill linking to its test — audited; all testable claims have pills; unpilled equations are definitions, complexity results, or cited theorems
- [x] Code and math use consistent notation (per CLAUDE.md) — audited w/p/θ parameterization; all transitions explicit
- [x] No undefined abbreviations (spell out on first use)
- [x] Citations verified against the actual publication — Kulesza & Taskar year fixed from 2011→2012; all others verified correct
- [x] Every hardcoded number verified: static example tables (the source of past bugs) replaced by the widget. Remaining code cells contain only input values (w_ex, plot params) and computed outputs. No stale values found.

## Minor

- [x] Fold brute-force verification into a collapsible block or just reference the test suite
- [x] Sampling cost $O(n \log N)$ assumes $n \ll N$; when $n \approx N$ essentially all nodes are visited — note this or clarify
- [x] Consider switching from Scott bracket notation to the more standard $[z^k] f(z)$ — keeping Scott brackets (less familiar but avoids ambiguity)
- [x] Link to `conditional_poisson_numpy.py` points to repo root, not the file itself
- [x] HTML acceptance rate table (Cell 5) — kept as HTML because Jupyter doesn't render LaTeX in markdown tables
- [x] DP baseline in timing section (Cell 25) is $O(N^2 n)$ (recomputes from scratch per leave-one-out), not $O(Nn)$ as stated
- [x] Color coding (weights blue, π crimson, Z orange) is never explained in the text; invisible to colorblind readers or raw markdown viewers
- [x] Several forward references (exponential family, D-tree) introduce concepts long before they're used — consider deferring or adding a brief "we'll return to this" signpost


- [x] Mermaid diagrams aren't rendering now that they moved out of the notebook into the .md file. Fixed: added _protect_mermaid/_restore_mermaid to build.py to convert ```mermaid blocks into <div class="mermaid"> elements.

## Misc

- [x] If we are going to bother explaining the name conditional Poisson, we should say

  Poisson sampling is named after mathematician Siméon Denis Poisson.  It has
  nothing to do with fish or fishing.

  In Poisson sampling, each unit i is included in the sample independently with some probability p_i

  Here the sample size is n = \sum 1[i∈S] is random.  The *conditional* Poisson
  distribution is simple a Poisson distribution conditioned to have n be a fixed
  value (with probability one).  The inclusion probabilities under Poisson
  sampling are \pi_=p_i, but they become more complicated when we condition.
  [Explain that part better.]


## MIsc

[x] ~~"Why the tree matters."~~ — addressed: blog post now explains that forward + backprop is the key insight, the tree is just the fastest forward pass.
