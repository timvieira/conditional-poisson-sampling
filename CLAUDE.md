# Project conventions

## Notation consistency (critical)

Code and documentation must use identical notation. Stale or inconsistent notation is unacceptable.

- **Weight vector**: `\boldsymbol{w}` in math, `w` in code/prose — always bold in LaTeX to distinguish from elements
- **Weight element**: `w_i` — never bold, always subscripted
- **Log-weights**: `theta` (or `theta_i`) — only where referring to the actual code parameter
- **Normalizing constant**: `Z\binom{\boldsymbol{w}}{n}` — the weighted binomial coefficient notation; equals `\binom{N}{n}` when `w = 1`
- **Weights are odds**: `w_i = p_i / (1 - p_i)` — odds of the Bernoulli coin flip, NOT "importance weights"
- **Inclusion probabilities**: `pi` (or `pi_i`)
- **Subsets**: `S \in \binom{\mathcal{S}}{n}` for size-n subsets of the universe `\mathcal{S}`
- **Scott brackets**: `\llbracket f \rrbracket(z^k)` for coefficient extraction — not `[z^k]`
- **Color coding** (notebook only): weights blue (#2196F3), inclusion probs crimson (#E91E63), Z orange (#FF9800)

When changing notation or API names, update **all** of these in the same commit:
1. Code (parameter names, property names)
2. Docstrings (module, class, method)
3. README (prose, math, code examples, mermaid diagrams)
4. Tests (if they use keyword arguments or reference the old names)

## Workflow

- Commit and push after every logical change
- Run tests before committing
