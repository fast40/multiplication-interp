"""Donor-free steering with class-mean activations:
(A) units: modular neurons := mean over problems whose product ends in d;
(B) magnitude: magnitude neurons := mean over problems with |ab - T| <= 25;
(C) write an arbitrary target t: (A) with d = t%10 and (B) with T = t."""
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
sub = rng.choice(len(probs), 300, replace=False); ids_s = ids[torch.tensor(sub).cuda()]; Ps = P[sub]
Pt = torch.tensor(P).cuda()
def const(group_acts, mask):
    return {l: v[torch.tensor(mask).cuda()].mean(0, keepdim=True) for l, v in group_acts.items()}
def tile(c, n): return {l: v.expand(n, -1) for l, v in c.items()}
def nearest_units(t, d):
    c = np.arange(1000); c = c[c % 10 == d]; return c[np.argmin(np.abs(c - t))]
res = {"A_units": {}, "B_mag": {}, "C_write": {}}
# (A)
for d in range(10):
    vals = tile(const(A["modular"], P % 10 == d), len(sub))
    o = run_with(model, ids_s, num_ids, [(G["modular"], vals)])
    keep = Ps % 10 != d
    tgt = np.array([nearest_units(p, d) for p in Ps])
    res["A_units"][d] = dict(units_is_d=float((o % 10 == d)[keep].mean()), hits_nearest=float((o == tgt)[keep].mean()),
                             within10=float((np.abs(o - Ps) <= 10)[keep].mean()), n=int(keep.sum()))
    print("A", d, res["A_units"][d], flush=True)
# (B)
for T in range(100, 1000, 50):
    vals = tile(const(A["magnitude"], np.abs(P - T) <= 25), len(sub))
    o = run_with(model, ids_s, num_ids, [(G["magnitude"], vals)])
    res["B_mag"][T] = dict(med_abs_to_T=float(np.median(np.abs(o - T))), within25=float((np.abs(o - T) <= 25).mean()),
                           units_kept=float((o % 10 == Ps % 10).mean()), outputs=o.tolist())
    print("B", T, {k: v for k, v in res["B_mag"][T].items() if k != "outputs"}, flush=True)
# (C)
targets = rng.integers(100, 1000, len(sub))
mod_vals = {l: torch.cat([A["modular"][l][torch.tensor(P % 10 == t % 10).cuda()].mean(0, keepdim=True) for t in targets]) for l in A["modular"]}
mag_vals = {l: torch.cat([A["magnitude"][l][torch.tensor(np.abs(P - t) <= 25).cuda()].mean(0, keepdim=True) for t in targets]) for l in A["magnitude"]}
o = run_with(model, ids_s, num_ids, [(G["modular"], mod_vals), (G["magnitude"], mag_vals)])
res["C_write"] = dict(exact=float((o == targets).mean()), units=float((o % 10 == targets % 10).mean()), within10=float((np.abs(o - targets) <= 10).mean()),
                      med_abs=float(np.median(np.abs(o - targets))), equals_true=float((o == Ps).mean()),
                      examples=[(int(probs[s][0]), int(probs[s][1]), int(t), int(x)) for s, t, x in zip(sub[:15], targets[:15], o[:15])])
print("C", res["C_write"])
res["sub"] = sub.tolist()
json.dump(res, open("results/steer.json", "w"))
