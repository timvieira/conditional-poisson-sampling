# Conditional Poisson Sampling

A NumPy implementation of the **conditional Poisson distribution** over fixed-size subsets:

$$P(S;\, \theta) = \frac{\exp\!\bigl(\sum_{i \in S} \theta_i\bigr)}{e_n(\exp \theta)}, \quad |S| = n$$

where $e_n$ is the $n$-th elementary symmetric polynomial.

## Features

- **Inclusion probabilities** $\pi_i = P(i \in S)$ via product-tree downward pass
- **Log-normalizer** $\log e_n(q)$ — numerically stable, never overflows
- **Exact sampling** using divide-and-conquer on the product tree (no backward ESP table)
- **Hessian-vector products** $\text{Cov}[Z]\, v$ for Newton-CG fitting
- **Moment matching** — fit $\theta$ to target inclusion probabilities $\pi^*$ via Newton-CG with Armijo backtracking

## Usage

```python
import numpy as np
from conditional_poisson import ConditionalPoisson

# From weights
q = np.array([1.0, 2.0, 3.0, 0.5, 1.5])
cp = ConditionalPoisson.from_weights(n=2, q=q)

# Inclusion probabilities
print(cp.pi)          # shape (5,), sums to n=2

# Log-normalizer
print(cp.log_normalizer)

# Sample 1000 subsets of size 2
samples = cp.sample(1000, rng=42)   # shape (1000, 2)

# Log-probability of a subset
print(cp.log_prob([0, 3]))

# Hessian-vector product: Cov[Z] v
v = np.random.randn(5)
print(cp.hvp(v))
```

### Constructors

| Constructor | Description |
|---|---|
| `ConditionalPoisson(n, theta)` | Direct from log-weights $\theta$ |
| `ConditionalPoisson.uniform(N, n)` | Uniform inclusion probabilities $n/N$ |
| `ConditionalPoisson.from_weights(n, q)` | From positive weights $q_i = \exp(\theta_i)$ |
| `ConditionalPoisson.fit(pi_star, n)` | Fit $\theta$ to target inclusion probabilities |

### Fitting to target probabilities

```python
pi_star = np.array([0.6, 0.4, 0.8, 0.3, 0.9])  # must sum to n
cp = ConditionalPoisson.fit(pi_star, n=3, tol=1e-10, verbose=True)
print(np.max(np.abs(cp.pi - pi_star)))  # should be < tol
```

## Algorithm

All operations are built on a single **augmented polynomial product tree** over the factors $(1 + q_i z)$.

- **Upward pass** builds the P-tree: $P_T(z) = \prod_{i \in T}(1 + q_i z)$ truncated to degree $n$.
- **D-tree** (for HVP): $D_T(z) = \sum_{i \in T} q_i v_i \prod_{j \in T,\, j \neq i}(1 + q_j z)$, using the product rule.
- **Downward pass** propagates leave-one-out polynomials $P^{(-i)}(z)$ and $D^{(-i)}(z)$ to each leaf.
- **Sampling** walks the P-tree top-down, splitting a quota $k$ at each node proportional to $P_L[j] \cdot P_R[k-j]$.

### Numerical stability

Every polynomial is stored as a **ScaledPoly** `(coeffs_norm, log_scale)` with $\max|c_k| = 1$. FFT convolutions operate on $O(1)$-magnitude numbers, preventing float64 overflow and FFT rounding blowup. Weights are geometrically normalised ($q \to q / \exp(\overline{\log q})$) before each tree build.

### Complexity

| Operation | Time |
|---|---|
| `pi` / `log_normalizer` | $O(N \log^2 n)$ (cached) |
| `hvp(v)` | $O(N \log^2 n)$ (P-tree cached; D-tree rebuilt) |
| `sample(M)` | $O(N \log^2 n + M\, n \log N)$ |
| `fit(pi_star)` | $O(N \log^2 n \times \text{Newton\_iters} \times \text{CG\_iters})$ |

## Tests

Run the built-in test suite:

```bash
python conditional_poisson.py
```

This verifies: forward-pass correctness, normalisation, sampling accuracy, HVP against finite differences, numerical stability at extreme parameter scales, and timing benchmarks.

## Requirements

- Python 3.8+
- NumPy
