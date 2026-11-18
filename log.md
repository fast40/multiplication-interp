# Research Log: How does Llama-3.1-8B multiply?

## State
- Environment: uv project (Python 3.12, torch 2.15 cu132), RTX 3090 Ti (24GB). Llama-3.1-8B cached locally (bf16 fits).
- LaTeX: no pdflatex/tectonic installed; attempt to download tectonic binary was denied by permission system. Need user approval or alternative.

## Plan (initial)
1. Behavioral baseline: accuracy on a*b over operand ranges, tokenization effects.
2. Locate: logit lens / activation patching to find where operand info is moved and answer formed.
3. Characterize: features (e.g., modular / magnitude) in residual stream and MLP neurons.
4. Intervene: steer answer digits using identified features.
5. Write paper.

## Entries
### 2026-11-17 — Setup
- Initialized uv env. Tectonic download denied (logged).
- Next: behavioral baseline.
