# This is a reference implementation of the sequential algorithm.
#
# It does not have a strategy to prevent overflow.
#
# It does not use theta as it's parameterization
#
# It has not been optimized for efficiency (other than having the right big-O)
#
# Notice that it doesn't have any notion of a "suffix table" in it sampling
# algorithm.
#
import numpy as np

class ConditionalPoissonSampling:
    """Conditional Poisson Sampling (aka k-Bernoulli Point Process)

    This class implements efficient algorithms for sampling, normalization, and
    marginalization based on the elementary symmetric polynomials.

    p(Y=y) ∝ { \prod_{i ∈ y} v[i]  if |y| = k,
              {  0                   otherwise .

    This distribution is called the "conditional Poisson sampling".  Poisson
    sampling is where you interpret the probabilities (which sum to one over n)
    as Bernoulli random variates.  Conditioning here refers to /rejecting/ draws
    that are of size != k.  We can see artifacts of the Bernoulli draws in the
    sampling method.

    """

    def __init__(self, v, K, zero=0, one=1):

        self.v = v
        [self.N] = v.shape
        self.K = K
        N = self.N

        # Below, is Algorithm 7 of Kulesza & Taskar (2012).  It is a dynamic
        # program that (efficiently) evaluates the elementary symmetric
        # polynomials on v (and it supports general semirings).  For more a more
        # numerically stable computation pass in v in the log-semiring.  To
        # compute the entropy pass in v in the Entropy-semiring.
        E = np.full((K+1,N+1), zero, dtype=v.dtype)
        E[0,:] = one                     # initialization
        for k in range(1, K+1):
            for n in range(N):
                E[k,n+1] = E[k,n] + v[n] * E[k-1,n]
        self.E = E
        self.Z = E[K,N]   # normalizing constant

        # gradient computations (dZ/dv, dz/dE)
        d_v = np.full(self.N, zero, dtype=v.dtype)
        d_E = np.full((K+1,N+1), zero, dtype=v.dtype)
        d_E[self.K, self.N] = one
        for r in reversed(range(1,self.K+1)):
            for n in reversed(range(self.N)):
                d_E[r,n]   += d_E[r,n+1]
                d_v[n]     += d_E[r,n+1] * self.E[r-1,n]
                d_E[r-1,n] += d_E[r,n+1] * self.v[n]
        self.d_v = d_v

    def score(self, Y):
        return np.prod(self.v[list(Y)]) if len(Y) == self.K else 0

    def P(self, Y):
        return self.score(Y) / self.Z

    def dlogZ(self):
        return self.d_v/self.Z

    def inclusion(self):
        "p(i ∈ Y)"
        return self.d_v * self.v / self.Z

    def sample(self):
        # shift index into E, to make code more readable.
        E = self.E[:,1:]; v = self.v
        k = self.K
        Y = set()
        for i in reversed(range(self.N)):
            # Sample from conditional probability of element i
            if np.random.uniform(0,1) * E[k, i] <= v[i] * E[k-1, i-1]:
                Y.add(i)
                k -= 1
                if k == 0: break
        return frozenset(Y)
