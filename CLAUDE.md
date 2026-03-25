# Project conventions

## Notation consistency (critical)

Code and documentation must use identical notation. Stale or inconsistent notation is unacceptable.

- **Weights**: `w` (or `w_i`) everywhere — code, docstrings, README, diagrams
- **Log-weights**: `theta` (or `theta_i`) — only where referring to the actual code parameter
- **Normalizing constant**: `Z` — not `e_n(q)` or `e_n(w)` in user-facing docs
- **Inclusion probabilities**: `pi` (or `pi_i`)

When changing notation or API names, update **all** of these in the same commit:
1. Code (parameter names, property names)
2. Docstrings (module, class, method)
3. README (prose, math, code examples, mermaid diagrams)
4. Tests (if they use keyword arguments or reference the old names)

## Workflow

- Commit and push after every logical change
- Run tests before committing
