# Blog post TODO

## Confusing / Poorly Explained

- [x] Motivate the name "conditional Poisson" — bridge from Poisson sampling (independent inclusion) to conditioning on size
- [ ] Introduce Scott bracket notation $\llbracket f \rrbracket(z^k)$ before first use, or switch to the more standard $[z^k] f(z)$
- [ ] Explain or remove the D-tree parenthetical — currently a black box ("piggybacks on the P-tree" says nothing)
- [x] Fix $\text{Cov}[Z]$ notation — $Z$ is the normalizing constant everywhere else; use $\text{Cov}[\mathbf{1}_S]$ or $\Sigma$ for the inclusion covariance
- [ ] Expand the downward pass explanation — a small worked example (N=4) with actual polynomial coefficients at each node would help a lot
- [x] Justify the $O(n \log N)$ per-sample cost — why not $O(N \log N)$? (only $O(n)$ nodes have nonzero quota per level)
- [x] Add a one-line derivation for the fitting log-likelihood ($\log P(S) = \sum_{i \in S} \theta_i - \log Z$, take expectation under target)

## Potentially Unnecessary / Uninteresting (fix or cut)

- [ ] Numerical stability section: either show a case that *breaks* without scaling (before/after) or cut to one sentence
- [ ] Timing section: add comparison against naive $O(Nn)$ DP or R implementations to give context; raw ms alone is low-signal
- [ ] Construction 2 (Categorical): develop the birthday-problem connection and crossover analysis, or demote to a footnote

## Missing Visual Aids / Pseudocode / Math

- [x] Add the Mermaid tree diagrams from the README into the blog post (upward pass, downward pass, sampling)
- [x] Add pseudocode for the sampling procedure (recursive quota-splitting, ~5-10 lines)
- [ ] Add a scatter plot of $w_i$ vs $\pi_i$ to build intuition for the nonlinear relationship
- [x] Justify convexity of the fitting problem in one sentence (log-partition function of an exponential family)
- [x] Move the "maximum entropy" characterization earlier — it's a major selling point, currently buried in the summary
- [x] Add a "Motivation" paragraph with 2-3 concrete use cases (survey sampling, beam search, subset selection in ML)

## Mathematical / Logical Errors

- [x] Rejection bound in Cell 4 — added expandable derivation following De Vita (2023) and defined $W$
- [x] Brute-force log Z verification (Cell 15) is circular — it adds `cp_small.log_normalizer` to already-normalized log-probs and recovers it; compute log Z from brute force independently instead
- [x] Fitting objective (Cell 16) is called "log-likelihood" but is actually the expected log-probability E_{S~P*}[log P_θ(S)]; call it "expected log-probability" or "cross-entropy objective"

## Undefined / Missing Definitions

- [x] $W = \sum_i w_i$ — now defined in Cell 4 below the acceptance rate table
- [ ] "Exponential fixed-size design" (Cell 1) — explain where the name comes from (exponential family structure)
- [ ] Notation switch from $\mathcal{S}$ to $\mathcal{Y}$ in the HT estimator section (Cell 23) is unexplained

## Missing Citations / Links

- [ ] No citation for the product tree algorithm — cite subproduct tree literature (Ben-Or 1983, Macdonald) or state explicitly what's novel about this combination
- [ ] Newton's identities (Cell 24) stated without citation — add pointer (e.g., Stanley's Enumerative Combinatorics)
- [ ] Horvitz-Thompson estimator (Cell 23) not cited — add Horvitz & Thompson (1952)
- [ ] Pull key intuition from the "heaps for incremental computation" link forward instead of just linking

## Minor

- [ ] Fold brute-force verification into a collapsible block or just reference the test suite
- [ ] Sampling cost $O(n \log N)$ assumes $n \ll N$; when $n \approx N$ essentially all nodes are visited — note this or clarify
