# Blog post TODO

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
- [ ] Investigate whether there's a more principled scale correction factor (tilting parameter $\lambda$, saddlepoint, etc.) for comparing weights and inclusion probabilities
- [x] Justify convexity of the fitting problem in one sentence (log-partition function of an exponential family)
- [x] Move the "maximum entropy" characterization earlier — it's a major selling point, currently buried in the summary
- [x] Add a "Motivation" paragraph with 2-3 concrete use cases (survey sampling, beam search, subset selection in ML)

## Mathematical / Logical Errors

- [x] Rejection bound in Cell 4 — added expandable derivation following De Vita (2023) and defined $W$
- [x] Brute-force log Z verification (Cell 15) is circular — it adds `cp_small.log_normalizer` to already-normalized log-probs and recovers it; compute log Z from brute force independently instead
- [x] Fitting objective (Cell 16) is called "log-likelihood" but is actually the expected log-probability E_{S~P*}[log P_θ(S)]; call it "expected log-probability" or "cross-entropy objective"

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

- [x] ~~Batch polynomial multiplications in NumPy~~ — dropped. The NumPy implementation is the pedagogical/reference version; optimize for readability, not speed. The torch implementation is the fast path.
- [x] Recover sub-O(Nn) complexity with numerical stability — **solved via contour radius scaling** (torch_fft_prototype.py). Rescale weights w_i -> w_i*r where r shifts the product polynomial's peak to degree n. FFT rounding errors are now relative to the coefficient we need, not a distant peak. r = exp(t) where t is the Poisson sampling Lagrange multiplier (sum sigmoid(log w_i + t) = n). Result: O(N log² n), machine-epsilon precision, 10-16x faster than NumPy, fully differentiable.
    - The earlier obstacle (FFT/Karatsuba subtractions cancelling small coefficients) was a dynamic range problem, not a fundamental algebraic one — contour scaling eliminates the dynamic range.
- [ ] Test GPU performance — both torch_fft_prototype.py (FFT) and torch_prototype.py (direct conv1d) should be benchmarked on GPU. The FFT version is faster on CPU, but direct convolution may have better GPU characteristics (no FFT synchronization, better memory access patterns). Keep both implementations as the benchmark in torch_fft_prototype.py's __main__ compares them. float32 precision risk needs testing (contour scaling helps but may not fully compensate).
- [x] Compare Newton-CG fitting to simpler optimizers — L-BFGS with gradient only is 2-4x faster than Newton-CG with HVP. D-tree/HVP demoted to bonus appendix in blog post.
- [x] **Fitting optimizer is a hot mess** — resolved: torch's LBFGS does implement H₀ scaling (sᵀy/yᵀy), and setting `tolerance_grad=tol` gives a principled stopping rule: the gradient is π(θ) - π★, so the optimizer stops when max|π - π★| ≤ tol. Documented in docstring.
- [ ] Promote torch_fft_prototype.py to the primary library implementation — it's faster (O(N log² n) vs O(N log² N)), simpler (no hand-coded downward pass or D-tree), and supports autograd for integration as a neural network layer. The NumPy implementation (conditional_poisson.py) becomes a pedagogical/reference implementation that makes the algorithm transparent without requiring PyTorch — keep it, but frame it as the teaching version. Need to: add sampling to the torch version, match the full ConditionalPoisson API (fit, log_prob, etc.), add tests, decide on the package structure.

## Bugs

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
- [x] Title case all subsection headings
- [x] Verification checkmark pills: placed after punctuation like footnote markers. (Sizing/padding still TODO.)
- [x] Contour scaling section rewritten with full derivation: Cauchy integral formula motivation, why we get to choose r, saddlepoint/tilting-parameter derivation, connection to Poisson expected size. Title-cased heading.
- [ ] Add a diagram of the contour integral in the complex plane — show the circle |z|=r with the generating function's poles at z = -1/w_i, and how changing r shifts which poles are inside/outside the contour. (The text now explains the math; a visual would reinforce it.)
- [x] Numerical validation of contour radius: added code cell showing r=1 gives 10^16 dynamic range and wrong log Z, while r=r* gives dynamic range 1 and machine-precision log Z.
- [x] Fix complexity column in summary table: replaced +O(1)× with O(N log² n), fixed D-tree table too. Also fixed O(N log² N) → O(N log² n) throughout (truncation). Baur-Strassen 5× constant cited in Cell 35.
- [x] Fitting cost breakdown: added per-iteration cost paragraph to Cell 27 (line search × O(N log² n) eval + O(mN) two-loop).
- [x] Summary table: replaced "~15-50 iterations" with "O(N log² n) per iteration".
- [ ] Compare CPS to priority sampling in the HT estimation section — replicate the style of the "estimating means in a finite universe" blog post (https://timvieira.github.io/blog/post/2017/07/03/estimating-means-in-a-finite-universe/). Show variance reduction of CPS-HT vs priority sampling vs i.i.d. Monte Carlo on the same estimation problem. CPS gives optimal variance (max-entropy) but is more expensive to set up; priority sampling is O(N log N) but suboptimal. The comparison would make the HT section much more concrete.
- [x] Explain truncation to degree n: added paragraph in Cell 11 after Master Theorem — convolution of truncated polynomials gives correct coefficients up to degree n.
- [ ] PyTorch timing section (Cell 38) outputs a plain text table of numbers — should be a plot (log-log, NumPy vs FFT vs FFT+autograd) consistent with the earlier timing plot. Show speedup visually, not as a wall of text.
- [x] Stray `[z^k]` notation in worked example code (Cell 16 table header) — fixed to Scott brackets.
- [ ] Sampling timing plot (right panel) just shows one line with no baseline — it's not clear what the reader should take away. Add baselines: rejection sampling (Bernoulli construction), sequential DP-based sampling, or at minimum show cost per sample vs amortized cost (tree build once, then O(n log N) per sample). Or cut the plot and just state the complexity.
- [x] Mention alternative O(Nn) algorithm: added note after Pascal recurrence in Cell 41 — DP forward gives Z, backprop gives π, all in O(Nn).
- [x] Rename tree subsections: "Upward Pass" → "Computing the Normalizing Constant Z", "Downward Pass (Backpropagation)" → "Computing Inclusion Probabilities π", "Sampling" → "Drawing Exact Samples". Updated #Sampling anchor in Cell 41.
- [x] Explain polynomial = coefficient array + FFT trick: added "Polynomials as arrays" paragraph to Cell 11, covering representation, convolution, and O(d log d) FFT speedup. References (3B1B, CLRS) can be added later.
- [x] Rejection sampling Cell 9: converted from executable code + histogram plots to pseudocode in a markdown cell.
- [x] Small example uses non-integer weights: changed to w = (1.5, 3.2, 0.8, 4.5). Updated Cell 4 table, Z, π values, bar chart, and Cell 12 w_ex.
- [ ] Worked examples should not assume N is a clean power of 2 — currently N=4 (micro-example) and N=8 (tree diagrams/worked example). Using e.g. N=5 or N=7 would show how padding works and avoid the impression that the algorithm requires power-of-2 input. The tree diagrams and w_ex array in Cell 12 would need updating.
- [x] Reconcile w_i domain: added WLOG blockquote in Cell 41 parameterizations section — w_i=0 excluded, w_i=∞ deterministically included, general case reduces to finite positive.
- [ ] Refactor display-heavy code cells into `display_utils.py` — many notebook cells are mostly formatting cruft (html_table, display(HTML(...)), plot boilerplate). Move these to helper functions so cells become one-liners like `show_upward_pass(w_ex, n_ex)`. Alternatively, use cell metadata `"jupyter": {"source_hidden": true}` to collapse pure-display cells.
- [x] Removed color coding from all macros (kept macro structure). Removed color legend.
- [x] Vector concatenation notation: defined on first use before Vandermonde identity in Cell 41.

## Visualization

- [ ] D3 animation of the sampling algorithm: show the binary tree with quotas at each node, animate the top-down walk step by step. At each internal node, display the split equation P_L[j] · P_R[k−j] as a small distribution, sample from it, then propagate the resulting quotas to the children. Highlight the path from root to leaves, show items being selected (quota=1) or excluded (quota=0) at the leaves. Could be an interactive widget embedded in the notebook or a standalone HTML page linked from the blog post.

## Checklist (standing editorial rules)

Apply before every commit touching the notebook:
- [x] All vector quantities use `\boldsymbol` in LaTeX
- [x] All subsection titles are Title Case
- [x] Verification pills appear after punctuation, not before
- [x] Color coding removed; legend removed
- [ ] Every mathematical claim has a ✓ pill linking to its test
- [ ] Code and math use consistent notation (per CLAUDE.md)
- [ ] No undefined abbreviations (spell out on first use)
- [ ] Citations verified against the actual publication

## Minor

- [x] Fold brute-force verification into a collapsible block or just reference the test suite
- [x] Sampling cost $O(n \log N)$ assumes $n \ll N$; when $n \approx N$ essentially all nodes are visited — note this or clarify
- [x] Consider switching from Scott bracket notation to the more standard $[z^k] f(z)$ — keeping Scott brackets (less familiar but avoids ambiguity)
- [x] Link to `conditional_poisson.py` points to repo root, not the file itself
- [x] HTML acceptance rate table (Cell 5) — kept as HTML because Jupyter doesn't render LaTeX in markdown tables
- [x] DP baseline in timing section (Cell 25) is $O(N^2 n)$ (recomputes from scratch per leave-one-out), not $O(Nn)$ as stated
- [x] Color coding (weights blue, π crimson, Z orange) is never explained in the text; invisible to colorblind readers or raw markdown viewers
- [x] Several forward references (exponential family, D-tree) introduce concepts long before they're used — consider deferring or adding a brief "we'll return to this" signpost
