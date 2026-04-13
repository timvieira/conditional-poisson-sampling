## Application: Horvitz-Thompson Estimation

So far we've built machinery for sampling fixed-size subsets and computing inclusion probabilities.  A natural question: what can you *do* with this?  One important application is unbiased estimation.

**Setup.** Suppose you have a distribution $p$ over a universe $\mathcal{S}$ of $N$ items, and a function $f$ that is expensive to evaluate.  You want to estimate $\mu = \sum_{i=1}^{N} p(i)\, f(i)$ using only $n$ evaluations of $f$.  With i.i.d. Monte Carlo, you'd draw $n$ samples—but some items may repeat, wasting evaluations.

**The estimator.** The **Horvitz-Thompson estimator** ([Horvitz & Thompson, 1952](https://doi.org/10.1080/01621459.1952.10483446)) uses sampling *without* replacement to guarantee $n$ *distinct* evaluations.  Draw a fixed-size subset $S \sim \Ps_n$ using conditional Poisson sampling, then form:

$$
\hat{\mu}_{\text{HT}}(S) = \sum_{i \in S} \frac{p(i)}{\pip_i}\, f(i), \quad S \sim \Ps_n
$$

This gives an unbiased estimate: $\mathbb{E}[\hat{\mu}_{\text{HT}}] = \mu$,<a href="tests/test_identities.py#test_horvitz_thompson_unbiased" title="test_horvitz_thompson_unbiased" class="verified" target="_blank">✓</a> provided $\pip_i > 0$ whenever $p(i) > 0$.  The inverse-probability weighting $p(i)/\pip_i$ corrects for the sampling bias—items with higher inclusion probability are down-weighted, and vice versa.

**Example.** With $N = 100$ items and $n = 5$, set weights proportional to $p(i)$ so that high-probability items are more likely to be selected.  Each sample gives 5 distinct evaluations of $f$; the HT formula reweights them to produce an unbiased estimate of the full sum.

For more on SWOR-based estimation (including the near-optimal priority sampling scheme), see my earlier post on [estimating means in a finite universe](https://timvieira.github.io/blog/post/2017/07/03/estimating-means-in-a-finite-universe/).

## TODOs

- [ ] Need to explain that controlling the inclusion probabilities is the key to the optimal, unbiased, k-sparse estimator of the distribution, which is what HT provides. The optimal inclusion probabilities are $\pi = \min(1, p_i \tau)$ where $\tau$ is the solution to $n = \sum_i \min(1, p_i \tau)$. Give a citation (or just prove it).
- [ ] Compare CPS to priority sampling in the HT estimation section — replicate the style of the "estimating means in a finite universe" blog post (https://timvieira.github.io/blog/post/2017/07/03/estimating-means-in-a-finite-universe/). Show variance reduction of CPS-HT vs priority sampling vs i.i.d. Monte Carlo on the same estimation problem. CPS gives optimal variance (max-entropy) but is more expensive to set up; priority sampling is $O(N \log N)$ but suboptimal. The comparison would make the HT section much more concrete.
