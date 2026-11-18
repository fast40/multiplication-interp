"""Prediction test: writing an arbitrary target t via class means gets more exact as the modular
class mean conditions on a finer residue (t mod m for m = 10, 20, 40, 120)."""
import json, numpy as np, torch
from common import load, prompt, answer_id
from groups import load_groups, cache_acts, run_with

tok, model = load(); torch.set_grad_enabled(False)
G = load_groups()
probs = [(a, b) for a in range(10, 100) for b in range(2, 100) if a * b <= 999]; P = np.array([a * b for a, b in probs])
ids = torch.tensor([tok(prompt(a, b)).input_ids for a, b in probs]).cuda()
num_ids = torch.tensor([answer_id(tok, n) for n in range(1000)]).cuda()
A = {k: cache_acts(model, ids, g) for k, g in G.items()}
rng = np.random.default_rng(0)
sub = rng.choice(len(probs), 300, replace=False); ids_s = ids[torch.tensor(sub).cuda()]
# targets restricted to values whose residue class is populated for all m (so every condition is defined)
Ms = [10, 20, 40, 120]
ok_t = [t for t in range(100, 1000) if all(((P % m) == (t % m)).sum() >= 3 for m in Ms)]
targets = rng.choice(ok_t, len(sub))
mag_vals = {l: torch.cat([A["magnitude"][l][torch.tensor(np.abs(P - t) <= 25).cuda()].mean(0, keepdim=True) for t in targets]) for l in A["magnitude"]}
res = {"n_targets": len(sub), "n_valid_target_values": len(ok_t)}
for m in Ms:
    mod_vals = {l: torch.cat([A["modular"][l][torch.tensor(P % m == t % m).cuda()].mean(0, keepdim=True) for t in targets]) for l in A["modular"]}
    o = run_with(model, ids_s, num_ids, [(G["modular"], mod_vals), (G["magnitude"], mag_vals)])
    res[m] = dict(exact=float((o == targets).mean()), units=float((o % 10 == targets % 10).mean()), mod_m=float((o % m == targets % m).mean()),
                  within10=float((np.abs(o - targets) <= 10).mean()), med_abs=float(np.median(np.abs(o - targets))))
    print(m, res[m], flush=True)
# modular only (no magnitude write): should keep true magnitude
mod_vals = {l: torch.cat([A["modular"][l][torch.tensor(P % 120 == t % 120).cuda()].mean(0, keepdim=True) for t in targets]) for l in A["modular"]}
o = run_with(model, ids_s, num_ids, [(G["modular"], mod_vals)])
res["mod120_only"] = dict(exact=float((o == targets).mean()), mod120=float((o % 120 == targets % 120).mean()), med_abs_to_true=float(np.median(np.abs(o - P[sub]))))
print(res["mod120_only"])
json.dump(res, open("results/write_precision.json", "w"), indent=1)
