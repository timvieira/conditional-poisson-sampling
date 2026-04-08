"""
conditional_poisson — Conditional Poisson distribution over fixed-size subsets.

    P(S) ∝ prod_{i in S} w_i,   |S| = n

Four implementations with the same interface:

  ConditionalPoissonNumPy          — O(N log² N) product tree (NumPy)
  ConditionalPoissonTorch          — O(N log² n) FFT product tree (PyTorch)
  ConditionalPoissonSequentialNumPy — O(Nn) sequential DP (NumPy)
  ConditionalPoissonSequentialTorch — O(Nn) sequential DP (PyTorch)

Basic usage:

    from conditional_poisson import ConditionalPoissonNumPy
    cp = ConditionalPoissonNumPy.from_weights(n=3, w=[1.0, 2.0, 3.0, 4.0, 5.0])
    print(cp.incl_prob)       # inclusion probabilities
    print(cp.log_normalizer)  # log Z
    print(cp.sample(10))      # draw 10 samples
"""

from conditional_poisson.tree_numpy import ConditionalPoissonNumPy
from conditional_poisson.sequential_numpy import ConditionalPoissonSequentialNumPy

__all__ = [
    "ConditionalPoissonNumPy",
    "ConditionalPoissonSequentialNumPy",
]

# PyTorch implementations are optional (torch may not be installed)
try:
    from conditional_poisson.tree_torch import ConditionalPoissonTorch
    __all__.append("ConditionalPoissonTorch")
except ImportError:
    pass

try:
    from conditional_poisson.sequential_torch import ConditionalPoissonSequentialTorch
    __all__.append("ConditionalPoissonSequentialTorch")
except ImportError:
    pass
