"""Do the neuron groups (intervened at the position that emits the number token) (found with the default prompt) work under other prompt formats?
Measures clean accuracy, group ablations, and both-group transplant under alternative prompts."""
import json, numpy as np, torch
from common import load, answer_id
from groups import load_groups, cache_acts, run_with

FORMATS = {
    "default": lambda a, b: f"3*4=12\n15*6=90\n21*13=273\n8*7=56\n{a}*{b}=",
    "times_symbol": lambda a, b: f"7×6=42\n11×9=99\n14×12=168\n{a}×{b}=",
    "spaced": lambda a, b: f"5 * 9 = 45\n13 * 4 = 52\n22 * 11 = 242\n{a} * {b} =",
    "zero_shot_text": lambda a, b: f"The product of {a} and {b} is",
}
tok, model = load(); torch.set_grad_enabled(False)
G = load_groups(); num_ids_plain = torch.tensor([answer_id(tok, n) for n in range(1000)]).cuda()
print("space tokenization of ' 161':", [tok.decode(t) for t in tok(" 161", add_special_tokens=False).input_ids])
rng = np.random.default_rng(5)
probs = [(a, b) for a in range(10, 100) for b in range(2, 100) if a * b <= 999]
sub = [probs[i] for i in rng.choice(len(probs), 400, replace=False)]; P = np.array([a * b for a, b in sub])
res = {}
for name, f in FORMATS.items():
    texts = [f(a, b) for a, b in sub]
    enc = [tok(t).input_ids for t in texts]; Lmax = max(map(len, enc)); assert all(len(e) == Lmax for e in enc), name
    ids = torch.tensor(enc).cuda()
    # if the model's next token is a bare space, append it: the number token is produced at that position
    lg = model(ids[:8]).logits[:, -1]
    top = [tok.decode(t) for t in lg.argmax(-1).tolist()]
    if top[0] == " ":
        sp = tok(" ", add_special_tokens=False).input_ids[0]
        ids = torch.cat([ids, torch.full((len(ids), 1), sp).cuda()], 1)
        top = [tok.decode(t) for t in model(ids[:8]).logits[:, -1].argmax(-1).tolist()]
    num_ids = num_ids_plain
    A = {k: cache_acts(model, ids, g) for k, g in G.items()}
    mean = {k: {l: v.mean(0, keepdim=True).expand(len(sub), -1) for l, v in a.items()} for k, a in A.items()}
    clean = run_with(model, ids, num_ids, [])
    r = dict(sample_top=top[:4], clean_acc=float((clean == P).mean()))
    for k in G:
        o = run_with(model, ids, num_ids, [(G[k], mean[k])]); w = o != P
        r[f"ablate_{k}"] = dict(acc=float((o == P).mean()), wrong_within10=float((np.abs(o - P) <= 10)[w].mean()) if w.any() else None,
                                wrong_units_ok=float((o % 10 == P % 10)[w].mean()) if w.any() else None)
    donor = rng.permutation(len(sub)); Py = P[donor]
    src = {k: {l: v[torch.tensor(donor).cuda()] for l, v in a.items()} for k, a in A.items()}
    o = run_with(model, ids, num_ids, [(G["modular"], src["modular"]), (G["magnitude"], src["magnitude"])])
    r["transplant_both_equals_donor"] = float((o == Py)[Py != P].mean())
    res[name] = r; print(name, r, flush=True)
json.dump(res, open("results/prompt_robustness.json", "w"), indent=1)
