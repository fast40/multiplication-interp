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

### 2026-11-17 — Exp 04: Probes + logit lens at `=` (`scripts/04_probe.py` GPU cache, `scripts/04b_probe_fit.py` CPU fits)
- Restructured: GPU script caches all 33 hidden states at `=` for all 2210 single-token problems; probes fit on CPU (PCA-256 + logistic/ridge).
- Partial result (running): from L1 on, `=` linearly decodes operand units digits (1.0) and log(ab) (R²=1.0) — these are additive in (a,b), so trivially decodable from operand identity; NOT evidence of computation (and patching showed `=` isn't *used* before L15). Non-additive targets are the informative ones. **Final curves:** product units digit flat ≈0.55 through L16, rises L17 (0.77) → L19 (0.93); hundreds digit L18–20 (0.73→0.97); **tens digit last**, L21–24 (0.25→0.91). Logit-lens top-1 over 0..999: 0.01 at L18, 0.16 L20, 0.40 L23, 0.83 L26, 0.96 L29, 1.0 L32. Order units→hundreds→tens matches the behavioral finding that the tens digit is the fragile part.
- Mistake logged: `pkill -f 04_probe.py` killed my own shell; first probe version was too slow (sklearn while holding GPU).

### 2026-11-17 — Exp 05: Mover heads (`scripts/05_head_attn.py`, `results/head_attn.json`)
- L15H14 attends from `=` to **b** (0.96); L16H21 attends to **a** (0.95). L15H15 also → b (0.85). L20H2 diffuse.
- Mean-ablating at `=` (n=300): none 99.7%, L15H14 99.0%, L16H21 97.0%, both 97.3%, two random heads 99.7%. → Movers identified but redundant; need larger ablation sets.

### 2026-11-17 — Exp 06/07: Answer-writing neurons (`scripts/06_neurons.py`, `scripts/07_classify_neurons.py`)
- DLA (direct logit attribution to the correct answer token, relative to mean over number tokens 0..999, final-norm linearized) for all MLP neurons L14–31 at `=`; top 60/layer cached (1080 neurons).
- DLA grows with depth: L14–17 negligible; L18–31 large; biggest single neurons L30N13826 (0.27), L31N10258 (0.25), L29N1647 (0.20). Distributed: no neuron >0.3 logits.
- Classified each neuron by output pattern over 0..999: smooth (low-freq, periods ≥100) = **magnitude**; else best modulus m (variance explained by n mod m) = **mod m**.
- Top 200 neurons: magnitude 74 (ΣDLA 2.75), modular 72 (ΣDLA ≈2.3; m ∈ {10,100,2,5,3,8,7,11,6,20,...}), other 54 (1.7).
- **Key finding — matched read/write:** modular neurons are activated by the *product's residue mod the same m* (R² 0.54) and not by magnitude (0.07); magnitude neurons are activated by log-magnitude (0.63) and not product residue (0.04).
- Examples: L30N13826 mod3 (output R² 0.93, act by ab mod 3 0.66); L29N1647 mod8 (0.78 / 0.82); L26N8382 mod7; L26N10704 mod11; L29N11462 mod5; L28N2505 mod2; L31N10258 magnitude (smooth 0.97, act by mag 0.86).
- Hypothesis: **the answer token is selected as the intersection of an approximate magnitude bump and residue constraints mod several small moduli** (residue-number-system–like code).
- Next: (1) figure of example neurons; (2) group-level logit decomposition; (3) double-dissociation ablation; (4) residue/magnitude steering interventions.

### 2026-11-17 — Figures v1 (`scripts/style.py`, `scripts/08_figs_basic.py`)
- fig_behavior (correctness map + 4-digit error histogram: spikes at ±10, ±20), fig_patching, fig_probes. Checked visually; fixed overlapping titles/colorbar.

### 2026-11-17 — Exp 09: Group logit decomposition (`scripts/09_group_decomp.py`, `results/group_decomp.json`, `figures/fig_group_decomp`)
- Summed centered direct contributions (act − mean act) × output pattern for all cached neurons by class: magnitude (n=209) vs modular (n=285), aligned on offset = candidate − ab.
- Magnitude group: broad tent peaked at 0 (half-width ~60). Alone: top-1 15%, within ±10 79%, units digit 28%.
- Modular group: comb with spikes at multiples of 10, taller at ±20/±60/±100/±120 (common multiples of moduli), max at 0. Alone: top-1 67%, units digit 95%, but aliases one period away.
- Sum: sharp peak at 0, **top-1 97%** from direct effects of 494 neurons only.
- Supports "magnitude bump × residue comb" mechanism. Next: causal tests — mean-ablation double dissociation and activation transplant steering.
