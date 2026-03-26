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

- [ ] Batch polynomial multiplications per tree level in the NumPy implementation (same trick as torch_prototype: one vectorized operation per level instead of O(N) individual convolve calls). The torch version gets 3-5x from this; NumPy should see similar gains.
- [x] Recover sub-O(Nn) complexity with numerical stability — **solved via contour radius scaling** (torch_fft_prototype.py). Rescale weights w_i -> w_i*r where r shifts the product polynomial's peak to degree n. FFT rounding errors are now relative to the coefficient we need, not a distant peak. r = exp(t) where t is the Poisson sampling Lagrange multiplier (sum sigmoid(log w_i + t) = n). Result: O(N log² n), machine-epsilon precision, 10-16x faster than NumPy, fully differentiable.
    - The earlier obstacle (FFT/Karatsuba subtractions cancelling small coefficients) was a dynamic range problem, not a fundamental algebraic one — contour scaling eliminates the dynamic range.
- [ ] Test GPU performance — the FFT-based implementation (torch_fft_prototype.py) should benefit from GPU since torch.fft is well-optimized for CUDA. float32 precision risk needs testing (contour scaling helps but may not fully compensate).
- [ ] Compare Newton-CG fitting to simpler optimizers (e.g., gradient descent, L-BFGS) — the PyTorch version makes this easy since autograd gives gradients for free. Newton-CG requires the hand-coded HVP (D-tree in NumPy, double backward in PyTorch); if a first-order method converges fast enough, it simplifies the implementation and may be competitive for moderate precision.
- [ ] Promote torch_fft_prototype.py to the primary library implementation — it's faster (O(N log² n) vs O(N log² N)), simpler (no hand-coded downward pass or D-tree), and supports autograd for integration as a neural network layer. The NumPy implementation (conditional_poisson.py) becomes a pedagogical/reference implementation that makes the algorithm transparent without requiring PyTorch — keep it, but frame it as the teaching version. Need to: add sampling to the torch version, match the full ConditionalPoisson API (fit, log_prob, etc.), add tests, decide on the package structure.

## Bugs

- [x] HT section (Cell 40): orphaned sentence — rewrote section, moved unbiasedness right after formula
- [x] HT section (Cell 40): redundant "advantage over MC" paragraph — removed, covered by Setup
- [x] Speedup discussion (Cell 39): fixed `torch_fft_prototype.py` link to `/blob/main/`
- [x] Identities parameterization table (Cell 41): fixed `[0,∞)` → `[0,∞]` to match intro
- [x] Fitting section (Cell 27): fixed "convex" → "concave (since log Z is convex as a log-partition function)"
- [x] Jaynes (1957) added to References section (Cell 42)

## Confusing

- [ ] Micro-example (Cell 4) uses 1-indexed items ({1,2}, {1,3}...) but worked example code (Cell 12) uses 0-indexed arrays — could trip readers up
- [ ] Downward pass (Cell 13) Mermaid diagram is N=8 but interleaved code is N=4 — no explicit bridge between them
- [ ] "## Sampling" heading (Cell 23) in the library API section could be confused with "### Sampling" subsection of the product tree — rename to "## Drawing samples" or similar

## Editorial

- [ ] Cell 5: abbreviation "CPS" used ("CPS spreads probability...") but never introduced — spell out or define
- [ ] Cell 10: "**The gap.**" sentence is tacked onto a paragraph about categorical vs. Bernoulli tradeoff — would land harder as its own paragraph
- [ ] Cell 42 (summary): two blank lines between the `(coeffs, log_scale)` paragraph and **References:** — minor formatting

## Minor

- [x] Fold brute-force verification into a collapsible block or just reference the test suite
- [x] Sampling cost $O(n \log N)$ assumes $n \ll N$; when $n \approx N$ essentially all nodes are visited — note this or clarify
- [x] Consider switching from Scott bracket notation to the more standard $[z^k] f(z)$ — keeping Scott brackets (less familiar but avoids ambiguity)
- [x] Link to `conditional_poisson.py` points to repo root, not the file itself
- [x] HTML acceptance rate table (Cell 5) — kept as HTML because Jupyter doesn't render LaTeX in markdown tables
- [x] DP baseline in timing section (Cell 25) is $O(N^2 n)$ (recomputes from scratch per leave-one-out), not $O(Nn)$ as stated
- [x] Color coding (weights blue, π crimson, Z orange) is never explained in the text; invisible to colorblind readers or raw markdown viewers
- [x] Several forward references (exponential family, D-tree) introduce concepts long before they're used — consider deferring or adding a brief "we'll return to this" signpost
