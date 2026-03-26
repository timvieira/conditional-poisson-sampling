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

- [ ] Add a roadmap after the intro so readers know the arc of the post (rejection samplers → tree → fitting)
- [ ] Lead with a small concrete example before the abstract generating function machinery — anchor notation in numbers before abstraction
- [ ] Integrate the N=4 worked example *into* the tree explanation (upward/downward) rather than placing it after all three passes
- [ ] The intro paragraph does too much at once — split definition, etymology, max-entropy property, and exponential family connection into digestible chunks
- [ ] "Basic usage" (Cells 7-10) arrives before the reader understands what it's computing — either commit to a tutorial-first framing (and say so) or defer until after the tree explanation
- [ ] The rejection sampling section's punchline ("neither gives you Z, π, or gradients") should be emphasized as the gap the tree fills — frame the contrast as "easy to understand, impossible to scale" vs. "the tree makes it all efficient"
- [ ] The intro introduces ~10 symbols ($\mathcal{S}$, $N$, $n$, $w_i$, $S$, $\binom{\mathcal{S}}{n}$, $Z$, $p_i$, $\theta_i$, $\pi_i$) before any concrete example — anchor notation in numbers first
- [ ] The product tree section covers too much in one pass (upward, downward, sampling, worked example, brute-force) — consider breaking it into smaller sections or interleaving explanation with examples

## Exposition Gaps

- [ ] Give the D-tree more space — but frame it as Pearlmutter's R-operator applied to the backward pass, not as a separate algorithm. Currently it's a single opaque sentence; it should be presented as the next mechanical transformation in the stack (forward → backward → HVP)
- [ ] The fitting section is compressed overall — objective, convexity, gradient, Hessian, Newton-CG, and D-tree are all packed into one paragraph; the D-tree is the worst case but the whole section needs room to breathe
- [ ] Explain *why* the sampling split is exact — the specific insight is that the split probabilities $P_L[j] \cdot P_R[k-j]$ are the *conditional* probabilities of the target distribution (via Vandermonde); state this explicitly near the pseudocode
- [ ] The downward pass *is* backpropagation (not an analogy) — the post should make this unmistakable: define the forward pass (polynomial multiplication in a tree), then the backward pass follows mechanically. The current explanation jumps through too many intermediate framings (gradient, exponential family, leave-one-out, backprop, tree walk) instead of letting backprop do the explanatory work
- [ ] The entire algorithmic stack is mechanical program transformations with known cost guarantees — the post should make this explicit:
    - Forward pass ($Z$): $O(N \log^2 N)$ — tree-structured polynomial multiplication
    - Gradient ($\pi$): $O(N \log^2 N)$ — reverse-mode AD / backprop (Baur & Strassen, 1983)
    - HVP ($\text{Cov}[\mathbf{1}_S]\,v$): $O(N \log^2 N)$ — Pearlmutter's R-operator (1994) applied to the gradient computation
    Each level costs O(1)× the previous by a general theorem, not a problem-specific derivation. Keep Tim's blog post links as the primary exposition; cite Baur & Strassen (1983), Griewank & Walther (2008), and Pearlmutter (1994) in the references
- [ ] The Horvitz-Thompson section needs both setup/framing (it arrives with no transition) and a small worked example — currently reads as "see also" rather than a demonstration
- [ ] The max-entropy property is stated with a citation but never demonstrated or given intuition

## Restructuring

- [ ] Restructure the "Identities" section (Cell 27) — it's a grab-bag; promote Vandermonde identity to the tree section (it's *why* the tree works), move parameterization table earlier (clarifies the intro), demote the rest to an appendix
- [ ] The identities section has redundancy — repeats the $\pi_i$ formula and convexity argument from earlier sections; deduplicate or make the second occurrence explicitly a "recap for reference"
- [ ] K-DPP connection: decide whether it belongs in the main text or an appendix — it's interesting but tangential to the main narrative
- [ ] The summary repeats but doesn't synthesize — add a takeaway ("the tree unifies normalizing constant computation, marginal inference, sampling, and parameter fitting into a single data structure")

## Implementation

- [ ] Batch polynomial multiplications per tree level in the NumPy implementation (same trick as torch_prototype: one vectorized operation per level instead of O(N) individual convolve calls). The torch version gets 3-5x from this; NumPy should see similar gains.
- [ ] Investigate numerically stable FFT-based polynomial multiplication to recover O(N log² n) — current torch version is O(Nn) because conv1d always uses direct convolution (oneDNN on CPU). FFT rounding errors corrupt small polynomial coefficients; the fundamental issue is that convolution of two degree-n polynomials with O(1) coefficients produces a peak at degree ~n but the coefficient we need (also at degree n) can be much smaller in magnitude.
- [ ] Test GPU performance — cuDNN has FFT and Winograd codepaths for conv1d (unlike CPU's oneDNN which is always direct), so GPU could recover O(n log n) per multiply but at the cost of the FFT precision issue. Also: float32 on GPU adds further precision risk for the renormalization scheme.

## Minor

- [x] Fold brute-force verification into a collapsible block or just reference the test suite
- [x] Sampling cost $O(n \log N)$ assumes $n \ll N$; when $n \approx N$ essentially all nodes are visited — note this or clarify
- [ ] Consider switching from Scott bracket notation to the more standard $[z^k] f(z)$ — less notation to introduce
- [ ] Link to `conditional_poisson.py` points to repo root, not the file itself
- [ ] HTML acceptance rate table (Cell 5) is visually inconsistent with the rest of the markdown
- [ ] DP baseline in timing section (Cell 25) is $O(N^2 n)$ (recomputes from scratch per leave-one-out), not $O(Nn)$ as stated
- [ ] Color coding (weights blue, π crimson, Z orange) is never explained in the text; invisible to colorblind readers or raw markdown viewers
- [ ] Several forward references (exponential family, D-tree) introduce concepts long before they're used — consider deferring or adding a brief "we'll return to this" signpost
