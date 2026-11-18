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

### 2026-11-17 — Exp 01: Behavioral baseline (`scripts/01_baseline.py`, `results/baseline.csv`)
- Prompt: 4-shot `a*b=c\n` lines, then `a*b=`. Greedy decode.
- Tokenization: numbers ≤999 are single tokens; 4-digit answers split 3+1 (`1081` → `108`,`1`).
- Accuracy over a,b∈1..99: 1d×1d 100%, 2d×1d 100%, 2d×2d 79.4%. **Single-token products (<1000): 99.7%**; 4-digit: 74.9%.
- 4-digit errors: last digit right 90%, length right 99.9%, first token right 75%; median |error| = 10 → magnitude approximately right, units digit usually exact, tens digit is where it fails. Consistent with "approximate magnitude + exact modular" heuristics.
- Decision: study single-token products (a∈10..99, b∈2..99, a·b≤999). Problem is exactly 4 tokens `a * b =`, answer is one token.

### 2026-11-17 — Exp 02: Residual patching (`scripts/02_patch_resid.py`, `results/patch_resid.json`)
- Corrupt one operand (same digit count, different single-token product), patch clean residual at (layer, pos), metric = normalized logit-diff recovery. n=120 per operand.
- Result: operand info lives at its own token through L14 (recovery ≈1.0), transfers to `=` between L15–L17 (at L16 operand-pos 0.17/0.10, `=` 0.79/0.89), and is fully at `=` by L18. Same for a and b. `*` and `\n` positions carry nothing.
- Interpretation: attention in ~L14–17 moves operands to `=`; answer computed at `=` in L15–31.
- Next: attention vs MLP contribution per layer at `=` (path patching / ablation), then identify heads.

### 2026-11-17 — Exp 03: Component patching at `=` (`scripts/03_patch_components.py`, `results/patch_components.json`)
- Corrupt both operands (n=200), patch single component output at `=` only.
- Attention: only L15 (0.38) and L16 (0.29) matter; all other layers ≤0.04.
- Heads: **L15H14 (0.22)** and **L16H21 (0.20)** dominate; next best L15H15, L20H2 (0.04).
- MLPs: no effect before L15; L15–L31 each recover 0.07–0.33 (peak L18 0.33, L31 0.26) → answer computation distributed over many late MLP layers.
- Next: (a) attention patterns of L15H14/L16H21; (b) linear probes + logit lens at `=` by layer.
