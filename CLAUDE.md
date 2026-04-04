# Project conventions

## Notation consistency (critical)

Code and documentation must use identical notation. Stale or inconsistent notation is unacceptable.

- **Weight vector**: `\boldsymbol{w}` in math, `w` in code/prose — always bold in LaTeX to distinguish from elements
- **Weight element**: `w_i` — never bold, always subscripted
- **Log-weights**: `theta` (or `theta_i`) — only where referring to the actual code parameter
- **Normalizing constant**: `\Zw{\boldsymbol{w}}{n}` macro (currently renders as `\binom{w}{n}`); equals `\binom{N}{n}` when `w = 1`
- **Weights are odds**: `w_i = p_i / (1 - p_i)` — odds of the Bernoulli coin flip, NOT "importance weights"
- **Inclusion probabilities**: `pi` (or `pi_i`)
- **Subsets**: `S \in \binom{\mathcal{S}}{n}` for size-n subsets of the universe `\mathcal{S}`
- **Scott brackets**: `\llbracket f \rrbracket(z^k)` for coefficient extraction — not `[z^k]`
- **Big-O notation**: `\mathcal{O}(\cdot)` in LaTeX — not plain `O(\cdot)`
- **Em-dashes**: tight, no spaces — `word—word` not `word — word`
- **Definitional equals**: `\defeq` macro (renders as = with "def" above) for defining equations
- **Approximately**: `\approx` not `\sim` — reserve `\sim` for "distributed as" (e.g., $S \sim P$)
- **Headings**: Title Case (capitalize all words except articles/conjunctions/prepositions: a, an, the, and, but, or, in, on, at, to, for, of)
- **Function arguments**: never drop arguments from functions — always write `\ell(\btheta)` not `\ell`, `\mathcal{L}(\Ps, \btheta)` not `\mathcal{L}`, `H(\Ps)` not `H`, etc.
- **Notation discipline**: define every symbol before first use, ideally in a consolidated notation box. Never introduce a symbol implicitly or switch parameterizations (w↔θ↔p) without flagging it. Readers should be able to trust that the same symbol means the same thing throughout.
- **Typesetting**: use MathJax for all mathematical labels and symbols, including in interactive widgets. For SVG, use `<foreignObject>` to embed MathJax-rendered HTML. Never use plain text for math symbols.

When changing notation or API names, update **all** of these in the same commit:
1. Code (parameter names, property names)
2. Docstrings (module, class, method)
3. README (prose, math, code examples, mermaid diagrams)
4. Tests (if they use keyword arguments or reference the old names)

## Blog post ↔ identity tests (critical)

The blog post (`content/conditional-poisson-sampling.ipynb`) and the identity test suite (`test_identities.py`) must be kept in exact correspondence. Every mathematical identity, equality, and theorem stated in the blog post has a matching test that verifies it numerically. Inline `<small>` references in the notebook link each claim to its test.

**Rules:**
- When adding or changing a mathematical claim in the blog post, add or update the corresponding test in `test_identities.py` in the same commit.
- When fixing a test, verify the blog post statement still matches what the test checks.
- Never delete a test without also removing or correcting the claim it verified.
- If a test fails, the blog post is wrong until proven otherwise — do not weaken the test to make it pass.
- Run `python3 test_identities.py` before committing any change to the notebook or the test file.

The code informs the tests and the tests inform the code. They are two views of the same truth. **Testing is not optional — it is the ground truth.** If the test says the math is wrong, the math is wrong. No exceptions, no "we'll fix it later."

## Blog post structure

The blog post is assembled from two files:

- **`content/conditional-poisson-sampling.md`** — the master template. Contains all prose, widget HTML/JS, and `{% notebook ... cells[X:Y] %}` directives that pull in code cells.
- **`content/conditional-poisson-sampling.ipynb`** — the Jupyter notebook. Only code cells are rendered from here (via `{% notebook %}` directives). Markdown cells in the notebook are stale copies; the .md file is authoritative for all prose.

The build system (`~/projects/blog/main/build.py`) renders the .md file using Python-Markdown with nbconvert for notebook cell inclusions, and a Jinja2 template for the page chrome.

## Build System

- **`blog dev`** — builds the site and serves it at `http://localhost:8000/conditional-poisson-sampling/` with file-watching auto-rebuild.
- **`blog deploy`** — builds, commits `docs/`, and pushes to GitHub. The site is served via GitHub Pages from the `main` branch, `/docs` directory.

**Site URL:** https://timvieira.github.io/conditional-poisson-sampling/

## Workflow

- Commit and push after every logical change
- Run `blog deploy` to publish changes to the live site
- Run tests before committing
- **Never write citations without verifying them** — check title, authors, year, venue, and URL against the actual publication. Hallucinated references are unacceptable.
