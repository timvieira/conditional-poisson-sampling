---
name: Check claims before stating them
description: User expects me to verify quantitative claims by running the experiment before asserting them, not after being called out
type: feedback
---

Verify quantitative claims by running the measurement before stating the conclusion. Don't speculate about what causes a performance difference — measure it. The user caught me making up an attribution ("half the slowdown is autograd tape") that a simple benchmark disproved.

**Why:** The user is a researcher who values rigor. Speculative explanations presented as fact erode trust.

**How to apply:** When explaining a performance result, either run the decomposition experiment first or explicitly flag that it's a hypothesis. Never state a causal attribution without data.
