# Conditional Poisson Sampling

A NumPy implementation of the **conditional Poisson distribution** over fixed-size subsets:

$$P(S; \theta) = \frac{\exp \bigl(\sum_{i \in S} \theta_i\bigr)}{e_n(\exp \theta)}, \quad |S| = n$$

where $e_n$ is the $n$-th elementary symmetric polynomial.

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
- **D-tree** (for HVP): $D_T(z) = \sum_{i \in T} q_i v_i \prod_{j \in T, j \neq i}(1 + q_j z)$, using the product rule.
- **Downward pass** propagates leave-one-out polynomials $P^{(-i)}(z)$ and $D^{(-i)}(z)$ to each leaf.
- **Sampling** walks the P-tree top-down, splitting a quota $k$ at each node proportional to $P_L[j] \cdot P_R[k-j]$.

### Upward pass (P-tree)

The P-tree computes the product polynomial $P_T(z) = \prod_{i \in T}(1 + q_i z)$ bottom-up. Each leaf holds a degree-1 factor, and internal nodes multiply their children's polynomials:

```mermaid
graph BT
    L1["leaf 1<br/>(1 + q₁z)"] --> N12["node 1,2<br/>P₁ · P₂"]
    L2["leaf 2<br/>(1 + q₂z)"] --> N12
    L3["leaf 3<br/>(1 + q₃z)"] --> N34["node 3,4<br/>P₃ · P₄"]
    L4["leaf 4<br/>(1 + q₄z)"] --> N34
    N12 --> ROOT["root<br/>P₁₂ · P₃₄ = e(z)"]
    N34 --> ROOT

    style ROOT fill:#4a90d9,color:#fff
    style N12 fill:#7ab8e0,color:#fff
    style N34 fill:#7ab8e0,color:#fff
    style L1 fill:#b8d4e8,color:#000
    style L2 fill:#b8d4e8,color:#000
    style L3 fill:#b8d4e8,color:#000
    style L4 fill:#b8d4e8,color:#000
```

The root polynomial's $n$-th coefficient is the elementary symmetric polynomial $e_n(q)$, which is the normalizing constant.

### Downward pass (leave-one-out)

The downward pass propagates "outside" polynomials top-down. At each node, the child receives the product of its parent's outside polynomial with its sibling's inside polynomial. At the leaves, this yields the leave-one-out products $P^{(-i)}(z) = \prod_{j \neq i}(1 + q_j z)$:

```mermaid
graph TB
    ROOT["root<br/>outside = 1"] --> N12["node 1,2<br/>outside = P₃₄"]
    ROOT --> N34["node 3,4<br/>outside = P₁₂"]
    N12 --> L1["leaf 1<br/>P⁽⁻¹⁾ = P₃₄ · P₂"]
    N12 --> L2["leaf 2<br/>P⁽⁻²⁾ = P₃₄ · P₁"]
    N34 --> L3["leaf 3<br/>P⁽⁻³⁾ = P₁₂ · P₄"]
    N34 --> L4["leaf 4<br/>P⁽⁻⁴⁾ = P₁₂ · P₃"]

    style ROOT fill:#4a90d9,color:#fff
    style N12 fill:#7ab8e0,color:#fff
    style N34 fill:#7ab8e0,color:#fff
    style L1 fill:#d4e8b8,color:#000
    style L2 fill:#d4e8b8,color:#000
    style L3 fill:#d4e8b8,color:#000
    style L4 fill:#d4e8b8,color:#000
```

The inclusion probability is then $\pi_i = q_i \cdot [z^{n-1}] P^{(-i)}(z) / e_n(q)$.

### Sampling (top-down split)

Sampling walks the P-tree top-down with a quota $k$ (initially $n$). At each internal node, the quota is split between the left and right children proportional to their polynomial coefficients:

```mermaid
graph TB
    ROOT["root<br/>quota k = n"] -->|"j items"| LEFT["left subtree<br/>quota = j"]
    ROOT -->|"k − j items"| RIGHT["right subtree<br/>quota = k − j"]
    LEFT -->|"..."| LL["..."]
    LEFT -->|"..."| LR["..."]
    RIGHT -->|"..."| RL["..."]
    RIGHT -->|"..."| RR["..."]

    style ROOT fill:#4a90d9,color:#fff
    style LEFT fill:#7ab8e0,color:#fff
    style RIGHT fill:#7ab8e0,color:#fff
    style LL fill:#b8d4e8,color:#000
    style LR fill:#b8d4e8,color:#000
    style RL fill:#b8d4e8,color:#000
    style RR fill:#b8d4e8,color:#000
```

At each split: $P(j \text{ from left} \mid k) \propto P_L[j] \cdot P_R[k-j]$. Leaves with quota 1 are included in the sample; quota 0 are excluded.

### Numerical stability

Every polynomial is stored as a **ScaledPoly** `(coeffs_norm, log_scale)` with $\max \lvert c_k \rvert = 1$. FFT convolutions operate on $O(1)$-magnitude numbers, preventing float64 overflow and FFT rounding blowup. Weights are geometrically normalised before each tree build: $q \to q / \exp(\bar{\mu})$ where $\bar{\mu} = \text{mean}(\log q)$.

### Complexity

| Operation | Time |
|---|---|
| `pi` / `log_normalizer` | $O(N \log^2 n)$ (cached) |
| `hvp(v)` | $O(N \log^2 n)$ (P-tree cached; D-tree rebuilt) |
| `sample(M)` | $O(N \log^2 n + M n \log N)$ |
| `fit(pi_star)` | $O(N \log^2 n \cdot T_{\text{Newton}} \cdot T_{\text{CG}})$ |

## Tests

```bash
pytest                              # with pytest
python test_conditional_poisson.py  # standalone
```

The test suite includes brute-force equivalence tests that verify `pi`, `log_normalizer`, `log_prob`, `hvp`, and sampling against explicit enumeration over all $\binom{N}{n}$ subsets.

## References

- Chen, Dempster & Liu (1994). "Weighted Finite Population Sampling to Maximize Entropy." *Biometrika*, 81(3), 457–469. — Introduces conditional Poisson sampling and the connection to elementary symmetric polynomials.

- Vieira (2014). ["Subsets and Superset Sampling."](https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/) — Blog post describing divide-and-conquer sampling on product trees.

## License

MIT
