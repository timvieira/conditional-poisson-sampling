# Widget Design Principles

Collected from conversations, CLAUDE.md, and TODO.md.

## Visual Consistency
- **Color palette**: blue (#5b9bd5) = forward/primal, purple (#7b2d8e) = adjoint/backward, gold (#d4a24e) = weights, red (#c0504d) = highlighted coefficients and pi output
- Use the same colors for matching quantities in LaTeX and D3 widgets
- All mathematical labels rendered via MathJax, never plain text
- Use macros from the article (`$\pip_i$`, `$\w_i$`, `$\Zw{\bw}{n}$`, etc.)

## Pedagogical Value
- Each widget should teach ONE concept clearly; don't overload
- The forward pass and backward pass should be visually separable — either side-by-side or animated sequentially
- The connection between leaf adjoints and inclusion probabilities must be explicit: show $\pip_i = \w_i \cdot \bar{c}_1^{(i)}$
- The "seed" ($\bar{c}_n = 1/Z$) at the root of the backward pass is the key entry point — make it visually prominent
- Degree labels at the root help the reader track which coefficient is Z

## Histograms
- Each node shows polynomial coefficients as a bar chart
- Heights use log scale with a fixed reference (prevents wild rescaling on slider drag)
- Numeric labels above bars for precision
- The n-th coefficient at the root is highlighted in red (it's Z)

## Interaction
- Weight sliders are gold vertical drag bars above the leaves
- Dragging updates the tree in real time
- N and n are adjustable via number inputs
- On hover: highlight the path from root to leaf (future)

## Mathematical Fidelity
- The widget must compute the same values as the Python implementation
- Forward: polynomial product tree with convolution
- Backward: cross-correlation of parent adjoint with sibling polynomial
- Inclusion probability: pi_i = w_i * leaf_adjoint[1]
- All values should match the test suite's brute-force checks

## MathJax Label Placement
- Labels are absolutely positioned `<div>`s over the SVG container
- Use `transform: translateX(-50%)` for center-anchored labels
- MathJax renders asynchronously — call `MathJax.typesetPromise()` after building the DOM
- Use the `typeset: false` startup config to prevent premature typesetting
- Labels that reference bar positions (e.g., degree indices at root) must be computed after bar layout
- Keep labels short — long expressions break positioning
- For SVG-native labels (non-math), use `<text>` elements directly; only use MathJax for actual math

## Layout
- Leaves at top, root at bottom (matches the article's tree diagrams)
- Weights (input) above the tree, pi (output) below
- Separators between zones: input → tree → output
- Mobile: stack vertically if viewport < 600px
