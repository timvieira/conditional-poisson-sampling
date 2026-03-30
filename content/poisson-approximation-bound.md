title: A Non-Asymptotic Bound for the Poisson Approximation to Conditional Poisson Sampling
date: 2026-03-29
status: draft
comments: true
tags: notebook, sampling, open-problems, inequalities

<macros>
\newcommand{\w}{w}
\newcommand{\bw}{\boldsymbol{w}}
\newcommand{\W}{W}
\newcommand{\Z}{Z}
\newcommand{\Zw}[2]{\binom{#1}{#2}}
\newcommand{\pip}{\pi}
\def\btheta{{\boldsymbol{\theta}}}
\def\bpip{\boldsymbol{\pi}}
\newcommand{\ba}{\boldsymbol{a}}
\newcommand{\bb}{\boldsymbol{b}}
\newcommand{\z}{z}
\newcommand{\llbracket}{[\![}
\newcommand{\rrbracket}{]\!]}
\newcommand{\defeq}{\overset{\small\text{def}}{=}}
</macros>

## Setup

The **conditional Poisson distribution** draws a random subset $S$ of exactly $n$ items from a universe of $N$ items, with probability proportional to the product of weights:

$$P(S) \propto \prod_{i \in S} \w_i, \qquad |S| = n.$$

The **inclusion probability** $\pip_i \defeq P(i \in S)$ is the marginal probability that item $i$ appears in the random subset.  Computing $\bpip$ from $\bw$ requires summing over $\binom{N}{n}$ subsets (or using the polynomial product tree described in the [main article](../conditional-poisson-sampling/)).

The **Poisson approximation** replaces the exact $\pip_i$ with independent Bernoulli probabilities.  Choose a **tilting parameter** $r > 0$ such that $\sum_i p_i = n$, where

$$p_i \defeq \frac{\w_i \, r}{1 + \w_i \, r}.$$

Then $\pip_i \approx p_i$.  The quality of this approximation is the subject of this post.

## The Conjecture

**Conjecture.**<a href="test_identities.py#test_poisson_approximation_bound" title="test_poisson_approximation_bound" class="verified" target="_blank">✓</a>  For all $N \geq 2$, $1 \leq n < N$, positive weights $\w_1, \ldots, \w_N$, and all items $i$:

$$|\pip_i - p_i| \;\leq\; \frac{p_i(1 - p_i)}{d}$$

where $d \defeq \sum_j p_j(1-p_j) = \text{Var}(K)$ is the variance of the Poisson sample size $K = \sum_j I_j$.

**Numerical evidence.** The ratio $|\pip_i - p_i| \cdot d / (p_i(1-p_i))$ has been computed across $>10^6$ random instances with $N$ up to $200$ and weight spreads up to $e^{12}$.  It grows slowly with $N$—from $\approx 0.38$ at $N = 3$ to $\approx 0.58$ at $N = 29$—and appears to converge to a constant strictly less than $1$.  No violation has been found.

## Asymptotic Results (Known)

[Hájek (1964, Theorem 5.2)](https://doi.org/10.1214/aoms/1177700375) showed $\pip_i = p_i + \mathcal{O}(1/N)$.  [Boistard, Lopuhaä & Ruiz-Gazen (2012)](https://arxiv.org/abs/1207.5654) gave the explicit first-order correction:

$$\pip_i = p_i\Big(1 - d^{-1}(p_i - \bar{p})(1 - p_i) + \mathcal{O}(d^{-2})\Big)$$

where $\bar{p} \defeq d^{-1}\sum_i p_i^2(1-p_i)$.  The $\mathcal{O}(d^{-2})$ remainder has **unspecified constants**—which is what motivates the search for a non-asymptotic bound.

**Intuition.** When $p_i > \bar{p}$, conditioning on $|S| = n$ forces the other items to "make room," so $\pip_i > p_i$.  The factor $(1 - p_i)$ modulates this—items near certain inclusion have less room to adjust.  The denominator $d = \text{Var}(K)$ measures how much slack the system has.

<details class="derivation draft">
<summary><b>Derivation sketch</b> (Boistard, Lopuhaä & Ruiz-Gazen, 2012)</summary>

**Step 1: Bayes reduction.**  Let $I_1, \ldots, I_N$ be independent Bernoulli variables with $P(I_i = 1) = p_i$, and let $K \defeq \sum_i I_i$ be the (random) Poisson sample size.  The conditional inclusion probability is

$$\pip_i = P(I_i = 1 \mid K = n) = p_i \cdot \frac{P(K = n \mid I_i = 1)}{P(K = n)}.$$

So the entire problem reduces to computing a ratio of Poisson-binomial probabilities.

**Step 2: Edgeworth expansion.**  Since $K$ is a sum of $N$ independent (but non-identical) Bernoullis with variance $d = \sum_i p_i(1-p_i)$, the local probability $P(K = n)$ has an Edgeworth expansion around the Gaussian approximation.  Writing $x = (K - n)/\sqrt{d}$, the expansion is

$$P(K = n) = \frac{1}{\sqrt{2\pi d}}\Big(1 + \sum_{j=1}^{s} P_j(0)\, d^{-j/2} + \mathcal{O}(d^{-(s+1)/2})\Big)$$

where the $P_j$ are polynomials in Hermite polynomials and normalized cumulants $S_m \defeq \kappa_m / d^{m-1}$.  Crucially, $S_m = \mathcal{O}(1)$ (since each cumulant $\kappa_m$ of a sum of Bernoullis satisfies $\kappa_m = \mathcal{O}(d)$).  Odd-order Hermite polynomials vanish at $x = 0$ (we are evaluating at the mean), so only even powers of $d^{-1/2}$ survive:

$$P(K = n) = \frac{1}{\sqrt{2\pi d}}\Big(1 + c_1\, d^{-1} + \mathcal{O}(d^{-2})\Big)$$

where $c_1$ is an explicit constant depending on $\kappa_3$ and $\kappa_4$.

**Step 3: Conditional expansion.**  Conditioning on $I_i = 1$ removes item $i$ from the sum.  Define $\tilde{K} = K - I_i$, so $\tilde{K} = n - 1$ given $K = n$ and $I_i = 1$.  The conditional sum $\tilde{K}$ has mean $n - p_i$, variance $d - p_i(1-p_i)$, and its expansion is centered at $\tilde{x} = (p_i - 1)/\sqrt{d - p_i(1-p_i)}$, which is $\mathcal{O}(d^{-1/2})$.

Expanding around $\tilde{x} = 0$ introduces additional $d^{-1/2}$ corrections from the Gaussian density $\phi(\tilde{x}) = \phi(0)(1 - \tilde{x}^2/2 + \cdots)$ and from the shifted Edgeworth polynomials $P_j(\tilde{x})$.  The result is

$$P(K = n \mid I_i = 1) = \frac{1}{\sqrt{2\pi d}}\Big(1 + c_2\, d^{-1} + \mathcal{O}(d^{-2})\Big)$$

where $c_2$ depends on $i$ through $p_i$.

**Step 4: Ratio.**  Taking the ratio:

$$\frac{P(K = n \mid I_i = 1)}{P(K = n)} = \frac{1 + c_2\, d^{-1} + \mathcal{O}(d^{-2})}{1 + c_1\, d^{-1} + \mathcal{O}(d^{-2})} = 1 + (c_2 - c_1)\, d^{-1} + \mathcal{O}(d^{-2}).$$

The explicit computation of $c_2 - c_1$ (carried out in detail in Boistard et al., Section 4) yields $c_2 - c_1 = -(p_i - \bar{p})(1 - p_i)$.

**Regularity condition.**  The expansion requires $d \to \infty$, which holds under $\limsup N/d < \infty$—equivalently, the $p_i$ don't all concentrate near 0 or 1.

**Higher-order inclusion probabilities.**  The same technique extends to joint inclusion probabilities $\pip_{i_1, \ldots, i_k} = P(I_{i_1} = \cdots = I_{i_k} = 1 \mid K = n)$.  The result (Theorem 1 in Boistard et al.) is

$$\pip_{i_1, \ldots, i_k} = \prod_{j=1}^{k} p_{i_j} \cdot \Big(1 + a\, d^{-1} + \mathcal{O}(d^{-2})\Big)$$

where $a$ is an explicit polynomial in the $p_{i_j}$ and the population summaries $\bar{p}$, $d$.

</details>


## Exact Formula

The Bayes formula gives an exact expression with no remainder:

$$\frac{\pip_i}{p_i} = \frac{1}{p_i + (1 - p_i)\, R_i}, \qquad R_i \defeq \frac{P(\tilde{K} = n)}{P(\tilde{K} = n-1)}$$

where $\tilde{K} \defeq \sum_{j \neq i} I_j$.  Rearranging:

$$\pip_i - p_i = \frac{p_i(1-p_i)(1 - R_i)}{p_i + (1 - p_i)\, R_i}.$$

The entire quality of the Poisson approximation is controlled by $R_i$, the ratio of consecutive Poisson-binomial probabilities.  The conjecture is equivalent to:

$$(1 + \w_i)\,|1 - R_i|\, d \;\leq\; R_i + \w_i.$$


## Equivalent $R_i$ Bounds

The bound is equivalent to two clean inequalities on the ESP ratio $R_i = e_n(\bw_{-i})/e_{n-1}(\bw_{-i})$:

$$R_i \geq 1 \implies R_i \leq \frac{\w_i + (1+\w_i)d}{(1+\w_i)d - 1}, \qquad R_i \leq 1 \implies R_i \geq \frac{[d + \w_i(d-1)]^+}{1 + (1+\w_i)d}$$

Both verified over 1.6M $R_i$ values with zero violations.  These are purely algebraic inequalities about elementary symmetric polynomials of positive reals, given the constraint $\sum \w_j/(1+\w_j) = n$.


## Proof Attempt 1: Markov Matrix Structure

Define $h(\btheta) \defeq \log P(K = n \mid \btheta)$ where $\theta_i = \log \w_i$.  Then $\pip_i - p_i = \partial h / \partial \theta_i$ and $h$ is concave in $\btheta$ (since $\log e_n$ is concave in $\log \bw$).

The Hessian $H(\btheta) = \nabla^2 h$ has a key structural property: the matrix $-H$ is **entry-wise non-negative** (since $-H_{ij} = -\text{Cov}(I_i, I_j \mid K=n)$ for $i \neq j$, and CPS is negatively associated), with **row sums exactly $p_i(1-p_i)$** (since $\sum_j \text{Cov}(I_i, I_j \mid K=n) = \text{Cov}(I_i, K \mid K=n) = 0$).

Therefore $M(\btheta) \defeq D^{-1}(-H)$ is a **row-stochastic (Markov) matrix**, where $D = \text{diag}(p_i(1-p_i))$.  This holds at every $\btheta$.

Let $\btheta^*$ be the global maximum of $h$ (the uniform-weight point where $\bpip = \mathbf{p}$).  By the mean value theorem:

$$\bpip - \mathbf{p} = \nabla h(\btheta) = \underbrace{\left(\int_0^1 H(\btheta^* + t(\btheta - \btheta^*))\, dt\right)}_{\displaystyle A} (\btheta - \btheta^*)$$

The integrated matrix $-A$ is entry-wise non-negative with row sums $q_i \defeq \int_0^1 p_i(1-p_i)(\btheta^* + t\boldsymbol{\delta})\, dt$, so $D_q^{-1}(-A)$ is row-stochastic.  Writing $\boldsymbol{\delta} = \btheta - \btheta^*$:

$$\frac{\pip_i - p_i}{q_i} = \big[D_q^{-1}(-A)\big]_i \cdot (-\boldsymbol{\delta}) = \text{convex combination of } (-\delta_j).$$

Therefore $|\pip_i - p_i| \leq q_i \cdot \|\boldsymbol{\delta}\|_\infty$.

**Gap.** Numerically, $\|\boldsymbol{\delta}\|_\infty \cdot d$ can be large (up to $\sim 10^4$), so this bound is too loose.  The Markov structure is real but the $\ell^\infty$ norm is the wrong quantity—the bound holds because items with large $|\delta_j|$ have small $q_i$, a cancellation not captured by the product $q_i \cdot \|\boldsymbol{\delta}\|_\infty$.


## Proof Attempt 2: Algebraic Verification for Small $N$

For fixed $N$, the bound reduces to a polynomial nonnegativity condition.  Writing the ratio as $\text{numer}/\text{denom}$, the bound $|\text{ratio}| \leq 1$ is equivalent to

$$(\text{denom} + \text{numer})(\text{denom} - \text{numer}) \geq 0$$

on the feasible polytope $\{p_j \in (0,1) : \sum p_j = n\}$.

**$N = 3$, $n = 2$ (proved).**  In coordinates $s = p + q - 1$, $t = p - q$, the ratio factors as $-2 f_1 f_2 / g$ where $f_1$ and $f_2$ both vanish at the uniform point.  Both factors $g \pm 2 f_1 f_2$ are non-negative on the feasible region (verified symbolically: boundary values are non-negative, interior critical points satisfy the bound).

**$N = 4, 5, 6$ (verified).** Both factors non-negative for all $n < N$, verified by random sampling with $>500\text{k}$ points per $(N, n)$ pair, zero violations.

**Gap.** The polynomial has $N - 2$ variables and degree growing with $N$, so direct verification doesn't scale.  An inductive argument using deletion-contraction would need to overcome the mismatch between sub-problem parameters and the original $d$.


## Proof Attempt 3: Exponential Tilting

By exponential tilting to the midpoint mean $n - 1/2$:

$$R_i = \exp(-t_{\text{mid}}) \cdot \frac{P_t(\tilde{K} = n)}{P_t(\tilde{K} = n - 1)}$$

where $P_t$ is the tilted Poisson-binomial with mean exactly $n - 1/2$.

- The **Gaussian contribution**: $|t_{\text{mid}}| \approx |p_i - 1/2| / \tilde{d}$ where $\tilde{d} = d - p_i(1-p_i)$.
- The **tilted ratio**: $P_t(n)/P_t(n{-}1) = 1 + \delta$ where $|\delta|$ is controlled by the third cumulant.  Since $n$ and $n-1$ are symmetric about the tilted mean, the Gaussian ratio is exactly 1.  The first Edgeworth correction (from $\kappa_3$) contributes $|\delta| \leq 1/(2\tilde{d})$.

Combining: $|\log R_i| \leq |p_i - 1/2|/\tilde{d} + 1/(2\tilde{d}) = \max(p_i, 1{-}p_i)/\tilde{d}$.

This bound is numerically verified (max ratio 0.71) and the "slack" of $1/2$ in $\max(p_i, 1{-}p_i)$ vs $|p_i - 1/2|$ exactly absorbs the Edgeworth $P_1$ correction.

**Gap.**  The log-ratio bound $|\log R_i| \leq \max(p_i, 1{-}p_i)/\tilde{d}$ does **not** imply the original bound $(1+\w_i)|1-R_i|d \leq R_i + \w_i$ (the $|1 - e^\alpha|$ step is lossy—verified: 2.5M violations in the implication).

Moreover, the **leading saddlepoint term exceeds the bound** in some regimes (small $p_i$, small $d$).  The bound holds only through exact cancellation between leading and correction terms.  This rules out any proof that bounds these separately.


## Proof Attempt 4: Induction via Deletion-Contraction

By deletion-contraction on item $j \neq i$:

$$R_i = \frac{R' + \w_j}{1 + \w_j / S'} = \frac{(R' + \w_j) S'}{S' + \w_j}$$

where $R' = e_n(w_{-i,-j})/e_{n-1}(w_{-i,-j})$ and $S' = e_{n-1}(w_{-i,-j})/e_{n-2}(w_{-i,-j})$ are ESP ratios for the $(N-2)$-item set.  By Newton's inequality (log-concavity), $R' \leq R_i \leq S'$.

This gives $|1 - R_i| \leq \max(|1-R'|, |S'-1|)$ — the ratio for the $N$-item set is bounded by the worst ratio at two adjacent index positions of the $(N-2)$-item set.  However, this monotonicity does **not** give the $1/d$ factor because:

1. The sub-problem $(N-2, n)$ has a different variance $d'$ than the original $(N, n)$ problem.
2. The induction hypothesis applies to the $(N-1)$-item CPS, but the deletion-contraction removes two items ($i$ and $j$), giving an $(N-2)$-item set.

**Gap.** The mismatch between the sub-problem's parameters and the original $d$ prevents the induction from closing.


## Summary

| Approach | Gives | Needed | Gap |
|---|---|---|---|
| Markov matrix | $\|\pi - p\|_i \leq q_i \|\delta\|_\infty$ | $\|\delta\|_\infty \leq 1/d$ | $\|\delta\|_\infty$ too large |
| Algebraic (fixed $N$) | Proof for $N \leq 6$ | General $N$ | Polynomials grow exponentially |
| Tilting/Edgeworth | $\|\log R_i\| \leq \max(p,1{-}p)/\tilde{d}$ | Implication to original | $\|1-e^\alpha\|$ step too lossy |
| Induction | $\|1-R_i\| \leq \max(\|1-R'\|, \|S'-1\|)$ | $1/d$ factor | Sub-problem $d$ mismatch |

The conjecture is an algebraic inequality about elementary symmetric polynomials of real-rooted polynomials.  A proof likely requires a structural argument from this theory—perhaps using interlacing, ultra-log-concavity, or the strongly Rayleigh property—that captures the joint cancellation between leading and correction terms.
