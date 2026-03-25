# Conditional Poisson Sampling

Sample random subsets of exactly $n$ items from a universe $\mathcal{S}$ of $N$ items, where each item $i$ has a specified inclusion probability $\pi_i$.

Given a weight vector $\boldsymbol{w} = (w_1, \dots, w_N)$ with $w_i > 0$, the probability of drawing a particular subset $S \in \binom{\mathcal{S}}{n}$ is proportional to the product of its weights:

$$P(S) = \frac{\prod_{i \in S} w_i}{Z\tbinom{\boldsymbol{w}}{n}}, \quad S \in \tbinom{\mathcal{S}}{n}$$

where $Z\binom{\boldsymbol{w}}{n} = \sum_{S \in \binom{\mathcal{S}}{n}} \prod_{i \in S} w_i$ is the normalizing constant — a weighted generalization of the binomial coefficient (when $\boldsymbol{w} = \mathbf{1}$, we recover $Z\binom{\mathbf{1}}{n} = \binom{N}{n}$).

This is the **conditional Poisson distribution** (also called the *exponential* or *maximum-entropy* fixed-size design). It arises by running independent Bernoulli trials — include item $i$ with probability $p_i = w_i/(1+w_i)$ — and conditioning on exactly $n$ items being selected. The weight $w_i$ is the *odds* of the $i$-th coin flip: $w_i = p_i / (1 - p_i)$.

## Installation

Single-file library — copy `conditional_poisson.py` into your project, or install from a local clone:

```bash
pip install .
```

Requires Python 3.8+ and NumPy.

## Usage

```python
import numpy as np
from conditional_poisson import ConditionalPoisson

# From weights (Bernoulli odds)
w = np.array([1.0, 2.0, 3.0, 0.5, 1.5])
cp = ConditionalPoisson.from_weights(n=2, w=w)

# Inclusion probabilities: P(item i is in the sample)
print(cp.pi)          # shape (5,), sums to n=2

# Log-normalizer
print(cp.log_normalizer)

# Sample 1000 subsets of size 2
samples = cp.sample(1000, rng=42)   # shape (1000, 2)

# Log-probability of a specific subset
print(cp.log_prob([0, 3]))

# Hessian-vector product: Cov[Z] v
v = np.random.randn(5)
print(cp.hvp(v))
```

### Constructors

| Constructor | Description |
|---|---|
| `ConditionalPoisson(n, theta)` | Direct from log-weights `theta`, where `theta[i]` $= \log w_i$ |
| `ConditionalPoisson.uniform(N, n)` | Uniform: every item has inclusion probability $n/N$ |
| `ConditionalPoisson.from_weights(n, w)` | From weight vector $\boldsymbol{w}$ |
| `ConditionalPoisson.fit(pi_star, n)` | Find $\boldsymbol{w}$ that produces target inclusion probabilities $\pi^{\ast}$ |

### Fitting to target probabilities

A common use case: you have desired inclusion probabilities and need to find weights that achieve them.

```python
pi_star = np.array([0.6, 0.4, 0.8, 0.3, 0.9])  # must sum to n
cp = ConditionalPoisson.fit(pi_star, n=3, tol=1e-10, verbose=True)
print(np.max(np.abs(cp.pi - pi_star)))  # should be < tol
```

This solves a convex optimization problem (Newton-CG with Armijo backtracking) to find $\boldsymbol{w}$ such that the resulting inclusion probabilities match $\pi^{\ast}$.

## How it works

All operations use a **polynomial product tree** that computes $Z\binom{\boldsymbol{w}}{n}$, inclusion probabilities, and samples in $O(N \log^2 N)$ time. The tree uses a divide-and-conquer strategy: the product $\prod_i (1 + w_i z)$ is computed by recursively splitting the items in half and multiplying the sub-products via FFT-based polynomial multiplication (`scipy.signal.convolve`). The recurrence is $T(N) = 2\,T(N/2) + O(N \log N)$, which solves to $T(N) = O(N \log^2 N)$ by the Master Theorem.

See the [blog post](content/conditional-poisson-sampling.ipynb) for a detailed walkthrough. The diagrams below give the high-level picture.

### Upward pass: build the product tree

Each leaf holds one factor $(1 + w_i z)$. Internal nodes multiply their children's polynomials (truncated to degree $n$). The root's $n$-th coefficient is $Z\binom{\boldsymbol{w}}{n}$.

```mermaid
graph BT
    L1["(1 + w₁z)"] --> N12["P₁₂ = P₁ · P₂"]
    L2["(1 + w₂z)"] --> N12
    L3["(1 + w₃z)"] --> N34["P₃₄ = P₃ · P₄"]
    L4["(1 + w₄z)"] --> N34
    L5["(1 + w₅z)"] --> N56["P₅₆ = P₅ · P₆"]
    L6["(1 + w₆z)"] --> N56
    L7["(1 + w₇z)"] --> N78["P₇₈ = P₇ · P₈"]
    L8["(1 + w₈z)"] --> N78
    N12 --> N1234["P₁₋₄ = P₁₂ · P₃₄"]
    N34 --> N1234
    N56 --> N5678["P₅₋₈ = P₅₆ · P₇₈"]
    N78 --> N5678
    N1234 --> ROOT["root: P₁₋₈"]
    N5678 --> ROOT

    style ROOT fill:#4a90d9,color:#fff
    style N1234 fill:#5fa0d9,color:#fff
    style N5678 fill:#5fa0d9,color:#fff
    style N12 fill:#7ab8e0,color:#fff
    style N34 fill:#7ab8e0,color:#fff
    style N56 fill:#7ab8e0,color:#fff
    style N78 fill:#7ab8e0,color:#fff
    style L1 fill:#b8d4e8,color:#000
    style L2 fill:#b8d4e8,color:#000
    style L3 fill:#b8d4e8,color:#000
    style L4 fill:#b8d4e8,color:#000
    style L5 fill:#b8d4e8,color:#000
    style L6 fill:#b8d4e8,color:#000
    style L7 fill:#b8d4e8,color:#000
    style L8 fill:#b8d4e8,color:#000
```

### Downward pass: leave-one-out polynomials

Each child receives the product of its parent's outside context with its sibling's subtree. At the leaves, this yields $P^{(-i)}(z) = \prod_{j \neq i}(1 + w_j z)$, from which $\pi_i = w_i \cdot \llbracket P^{(-i)} \rrbracket(z^{n-1}) / Z\binom{\boldsymbol{w}}{n}$.

```mermaid
graph TB
    ROOT["outside = 1"] --> N1234["outside = P₅₋₈"]
    ROOT --> N5678["outside = P₁₋₄"]
    N1234 --> N12["outside = P₅₋₈ · P₃₄"]
    N1234 --> N34["outside = P₅₋₈ · P₁₂"]
    N5678 --> N56["outside = P₁₋₄ · P₇₈"]
    N5678 --> N78["outside = P₁₋₄ · P₅₆"]
    N12 --> L1["P⁽⁻¹⁾ = out₁₂ · P₂"]
    N12 --> L2["P⁽⁻²⁾ = out₁₂ · P₁"]
    N34 --> L3["P⁽⁻³⁾ = out₃₄ · P₄"]
    N34 --> L4["P⁽⁻⁴⁾ = out₃₄ · P₃"]
    N56 --> L5["P⁽⁻⁵⁾ = out₅₆ · P₆"]
    N56 --> L6["P⁽⁻⁶⁾ = out₅₆ · P₅"]
    N78 --> L7["P⁽⁻⁷⁾ = out₇₈ · P₈"]
    N78 --> L8["P⁽⁻⁸⁾ = out₇₈ · P₇"]

    style ROOT fill:#4a90d9,color:#fff
    style N1234 fill:#5fa0d9,color:#fff
    style N5678 fill:#5fa0d9,color:#fff
    style N12 fill:#7ab8e0,color:#fff
    style N34 fill:#7ab8e0,color:#fff
    style N56 fill:#7ab8e0,color:#fff
    style N78 fill:#7ab8e0,color:#fff
    style L1 fill:#d4e8b8,color:#000
    style L2 fill:#d4e8b8,color:#000
    style L3 fill:#d4e8b8,color:#000
    style L4 fill:#d4e8b8,color:#000
    style L5 fill:#d4e8b8,color:#000
    style L6 fill:#d4e8b8,color:#000
    style L7 fill:#d4e8b8,color:#000
    style L8 fill:#d4e8b8,color:#000
```

### Sampling: top-down quota splitting

Starting with quota $k = n$ at the root, each internal node splits its quota between children: draw $j$ from the left with probability $\propto P_L[j] \cdot P_R[k-j]$. Leaves with quota 1 are included; quota 0 excluded.

```mermaid
graph TB
    ROOT["quota = n"] -->|"j₁ items"| LEFT["quota = j₁"]
    ROOT -->|"n − j₁ items"| RIGHT["quota = n − j₁"]
    LEFT -->|"j₂"| LL["quota = j₂"]
    LEFT -->|"j₁ − j₂"| LR["quota = j₁ − j₂"]
    RIGHT -->|"j₃"| RL["quota = j₃"]
    RIGHT -->|"n−j₁−j₃"| RR["quota = n−j₁−j₃"]
    LL --> L1["0 or 1"]
    LL --> L2["0 or 1"]
    LR --> L3["0 or 1"]
    LR --> L4["0 or 1"]
    RL --> L5["0 or 1"]
    RL --> L6["0 or 1"]
    RR --> L7["0 or 1"]
    RR --> L8["0 or 1"]

    style ROOT fill:#4a90d9,color:#fff
    style LEFT fill:#5fa0d9,color:#fff
    style RIGHT fill:#5fa0d9,color:#fff
    style LL fill:#7ab8e0,color:#fff
    style LR fill:#7ab8e0,color:#fff
    style RL fill:#7ab8e0,color:#fff
    style RR fill:#7ab8e0,color:#fff
    style L1 fill:#b8d4e8,color:#000
    style L2 fill:#b8d4e8,color:#000
    style L3 fill:#b8d4e8,color:#000
    style L4 fill:#b8d4e8,color:#000
    style L5 fill:#b8d4e8,color:#000
    style L6 fill:#b8d4e8,color:#000
    style L7 fill:#b8d4e8,color:#000
    style L8 fill:#b8d4e8,color:#000
```

### Complexity

The tree build dominates and costs $O(N \log^2 N)$, independent of $n$. This follows from the divide-and-conquer recurrence $T(N) = 2\,T(N/2) + O(N \log N)$ where the $O(N \log N)$ term is the FFT-based polynomial multiplication at each level.

| Operation | Time |
|---|---|
| `pi` / `log_normalizer` | $O(N \log^2 N)$ (cached) |
| `hvp(v)` | $O(N \log^2 N)$ (P-tree cached; D-tree rebuilt) |
| `sample(M)` | $O(N \log^2 N + M n \log N)$ |
| `fit(pi_star)` | $O(N \log^2 N \cdot T_{\text{Newton}} \cdot T_{\text{CG}})$ |

## Tests

```bash
pytest                              # with pytest
python test_conditional_poisson.py  # standalone
```

The test suite includes brute-force equivalence tests that verify `pi`, `log_normalizer`, `log_prob`, `hvp`, and sampling against explicit enumeration over all $\binom{N}{n}$ subsets.

## References

- Hájek (1964). ["Asymptotic Theory of Rejective Sampling with Varying Probabilities from a Finite Population."](https://doi.org/10.1214/aoms/1177700375) *The Annals of Mathematical Statistics*, 35(4), 1491–1523. — Introduces the conditional Poisson sampling design.

- Chen, Dempster & Liu (1994). ["Weighted Finite Population Sampling to Maximize Entropy."](https://academic.oup.com/biomet/article-abstract/81/3/457/256956) *Biometrika*, 81(3), 457–469. — Characterizes CPS as maximum-entropy and establishes uniqueness of weights given inclusion probabilities.

- Tillé (2006). [*Sampling Algorithms*](https://link.springer.com/book/10.1007/978-0-387-34240-0). Springer. — Comprehensive reference on sampling-without-replacement designs including CPS.

- Meister, Amini, Vieira & Cotterell (2021). ["Conditional Poisson Stochastic Beams."](https://aclanthology.org/2021.emnlp-main.52/) *Proceedings of EMNLP 2021*. — Applies CPS to stochastic beam search in NLP; uses the $O(NK)$ dynamic programming recurrence for the normalizing constant.

## License

MIT
