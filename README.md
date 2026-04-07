# Conditional Poisson Sampling

Sample random subsets of exactly $n$ items from a universe of $N$ items, where each item has a weight $w_i \ge 0$ (including $w_i = \infty$ for forced inclusion). The probability of drawing subset $S$ is proportional to $\prod_{i \in S} w_i$.

This is the **conditional Poisson distribution** — the unique maximum-entropy distribution over fixed-size subsets with given inclusion probabilities. It arises by running independent Bernoulli trials and conditioning on exactly $n$ items being selected.

All operations use a polynomial product tree running in $O(N \log^2 N)$ time.

**[Read the blog post](https://timvieira.github.io/conditional-poisson-sampling/conditional-poisson-sampling/)**

## Installation

Single-file library — copy `conditional_poisson_numpy.py` into your project, or install from a local clone:

```bash
pip install .
```

Requires Python 3.8+, NumPy, and SciPy.

## Usage

```python
import numpy as np
from conditional_poisson_numpy import ConditionalPoissonNumPy

# From weights (Bernoulli odds)
w = np.array([1.0, 2.0, 3.0, 0.5, 1.5])
cp = ConditionalPoissonNumPy.from_weights(n=2, w=w)

# Inclusion probabilities: P(item i is in the sample)
print(cp.incl_prob)          # shape (5,), sums to n=2

# Log-normalizer
print(cp.log_normalizer)

# Sample 1000 subsets of size 2
samples = cp.sample(1000, rng=42)   # shape (1000, 2)

# Log-probability of a specific subset
print(cp.log_prob([0, 3]))

# Hessian-vector product: Cov[1_S] v
v = np.random.randn(5)
print(cp.hvp(v))
```

### Constructors

| Constructor | Description |
|---|---|
| `ConditionalPoissonNumPy(n, theta)` | Direct from log-weights `theta`, where `theta[i]` $= \log w_i$ |
| `ConditionalPoissonNumPy.from_weights(n, w)` | From weight vector $\boldsymbol{w}$ (uniform: `from_weights(n, np.ones(N))`) |
| `ConditionalPoissonNumPy.fit(pi_star, n)` | Find $\boldsymbol{w}$ that produces target inclusion probabilities $\pi^{\ast}$ |

### Fitting to target probabilities

```python
pi_star = np.array([0.6, 0.4, 0.8, 0.3, 0.9])  # must sum to n
cp = ConditionalPoissonNumPy.fit(pi_star, n=3, tol=1e-10, verbose=True)
print(np.max(np.abs(cp.incl_prob - pi_star)))  # should be < tol
```

### Complexity

| Operation | Time |
|---|---|
| `incl_prob` / `log_normalizer` | $O(N \log^2 N)$ (cached) |
| `hvp(v)` | $O(N \log^2 N)$ (P-tree cached; D-tree rebuilt) |
| `sample(M)` | $O(N \log^2 N + M n \log N)$ |
| `fit(pi_star)` | $O(N \log^2 N \cdot T_{\text{Newton}} \cdot T_{\text{CG}})$ |

## Tests

```bash
pytest                              # with pytest
python test_conditional_poisson_numpy.py  # standalone
```

## References

- Hájek (1964). ["Asymptotic Theory of Rejective Sampling with Varying Probabilities from a Finite Population."](https://doi.org/10.1214/aoms/1177700375) *The Annals of Mathematical Statistics*, 35(4), 1491–1523.

- Chen, Dempster & Liu (1994). ["Weighted Finite Population Sampling to Maximize Entropy."](https://academic.oup.com/biomet/article-abstract/81/3/457/256956) *Biometrika*, 81(3), 457–469.

- Tillé (2006). [*Sampling Algorithms*](https://link.springer.com/book/10.1007/978-0-387-34240-0). Springer.

- Meister, Amini, Vieira & Cotterell (2021). ["Conditional Poisson Stochastic Beams."](https://aclanthology.org/2021.emnlp-main.52/) *Proceedings of EMNLP 2021*.

## License

MIT
