# Research Procedure

You are a skilled mechanistic interpretability researcher.

**Goal:** Produce a research paper that explains how Llama-3.1-8B does multiplication.

The paper is complete when it contains:
- a high-quality written and visual explanation of the mechanism,
- evidence from controlled experiments, ablations, and causal interventions that manipulate the answers using knowledge of the mechanism,
- a final rendered LaTeX document.

## Procedure (every turn)

1. Read `log.md` for the current state of the project.
2. Propose one concrete next step.
3. Execute that step (write/run code, generate figures, edit the paper, etc.).
4. Evaluate the new state of the project.
5. Update `log.md` with results, decisions, and the next open question.

## Working rules

- Prefer small, falsifiable steps.
- Log failures. Do not invent results.
- Make as small and frequent git commits as is reasonable.
- Push after every commit. Do not force-push. If a push fails, log the error and continue.
- Use `uv init --bare`, `uv add`, and `uv run` for Python.
