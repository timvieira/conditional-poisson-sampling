# Conditional Poisson Sampling

Sample random subsets of exactly $n$ items from a universe of $N$ items, where each item has a positive, finite weight $w_i$. The probability of drawing subset $S$ is proportional to $\prod_{i \in S} w_i$. (To exclude an item, leave it out; to force one in, leave it out and draw $n - 1$ items from the rest.)

This is the **conditional Poisson distribution**—the unique maximum-entropy distribution over fixed-size subsets with given inclusion probabilities. It arises by running independent Bernoulli trials and conditioning on exactly $n$ items being selected.

All operations use a polynomial product tree running in $\mathcal{O}(N \log^2 n)$ time.

**[Read the blog post](https://timvieira.github.io/conditional-poisson-sampling/conditional-poisson-sampling/)**

## Installation

Install from a local clone:

```bash
pip install .              # NumPy classes
pip install '.[torch]'     # also ConditionalPoissonTorch and ConditionalPoissonSequentialTorch
```

Requires Python 3.8+, NumPy, and SciPy; PyTorch is optional.

## Usage

```python
import numpy as np
from conditional_poisson import ConditionalPoissonNumPy

# From weights (Bernoulli odds)
w = np.array([1.0, 2.0, 3.0, 0.5, 1.5])
cp = ConditionalPoissonNumPy.from_weights(n=2, w=w)

# Inclusion probabilities: P(item i is in the sample)
print(cp.incl_prob)          # shape (5,), sums to n=2

# Log-normalizer
print(cp.log_normalizer)

# Sample 1000 subsets of size 2
samples = np.stack([cp.sample() for _ in range(1000)])  # shape (1000, 2)

# Log-probability of a specific subset
print(cp.log_prob([0, 3]))

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
print(np.max(np.abs(cp.incl_prob - pi_star)))  # 6.9e-11
```

### Complexity

| Operation | Time |
|---|---|
| `incl_prob` / `log_normalizer` | $\mathcal{O}(N \log^2 n)$ (cached) |
| `sample()`, $M$ times | $\mathcal{O}(N \log^2 n + M n \log N)$ |
| `fit(pi_star)` | $\mathcal{O}(N \log^2 n \cdot T_{\text{L-BFGS}})$ |

## Tests

```bash
make test                          # all tests, with pytest
python3 tests/test_identities.py   # the blog post's identities, standalone
```

## References

- Hájek (1964). ["Asymptotic Theory of Rejective Sampling with Varying Probabilities from a Finite Population."](https://doi.org/10.1214/aoms/1177700375) *The Annals of Mathematical Statistics*, 35(4), 1491–1523.

- Chen, Dempster & Liu (1994). ["Weighted Finite Population Sampling to Maximize Entropy."](https://academic.oup.com/biomet/article-abstract/81/3/457/256956) *Biometrika*, 81(3), 457–469.

- Tillé (2006). [*Sampling Algorithms*](https://link.springer.com/book/10.1007/978-0-387-34240-0). Springer.

- Meister, Amini, Vieira & Cotterell (2021). ["Conditional Poisson Stochastic Beams."](https://aclanthology.org/2021.emnlp-main.52/) *Proceedings of EMNLP 2021*.

## License

MIT
