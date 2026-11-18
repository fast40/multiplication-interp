"""CPU: direct logit contributions of the magnitude and modular neuron groups, aligned to the true
answer (offset = n - ab). Uses cached activations (centered over problems) and output patterns."""
import json, numpy as np
from style import plt, C, INK2, MUTED, save
z = np.load("results/cache/neurons.npz"); cls = json.load(open("results/neuron_classes.json"))
probs = z["probs"]; P = probs[:, 0] * probs[:, 1]; N = len(P)
groups = {"magnitude": [d for d in cls if d["type"] == "magnitude"], "modular": [d for d in cls if d["type"].startswith("mod")]}
contrib = {}
for g, ds in groups.items():
    tot = np.zeros((N, 1000))
    for d in ds:
        l, nn = d["layer"], d["neuron"]
        act = z[f"acts_{l}"][:, list(z[f"idx_{l}"]).index(nn)].astype(np.float64)
        tot += np.outer(act - act.mean(), z[f"out_{l}_{nn}"])
    contrib[g] = tot - tot.mean(1, keepdims=True)
contrib["sum"] = contrib["magnitude"] + contrib["modular"]
W = 120; offs = np.arange(-W, W + 1)
aligned = {}
for g, M in contrib.items():
    rows = []
    for i in range(N):
        idx = P[i] + offs; ok = (idx >= 0) & (idx < 1000)
        r = np.full(len(offs), np.nan); r[ok] = M[i, idx[ok]]; rows.append(r)
    aligned[g] = np.nanmean(np.array(rows), 0)
res = {g: {"top1": float((M.argmax(1) == P).mean()),
           "units_ok": float((M.reshape(N, 100, 10).max(1).argmax(1) == P % 10).mean()),
           "within10": float((np.abs(M.argmax(1) - P) <= 10).mean()),
           "med_abs_err": float(np.median(np.abs(M.argmax(1) - P)))} for g, M in contrib.items()}
print(json.dumps(res, indent=1))
json.dump({"stats": res, "aligned": {g: v.tolist() for g, v in aligned.items()}, "offs": offs.tolist(),
           "n_neurons": {g: len(v) for g, v in groups.items()}}, open("results/group_decomp.json", "w"))
fig, ax = plt.subplots(3, 1, figsize=(7.2, 4.6), sharex=True)
for k, (g, lab, c) in enumerate([("magnitude", f"magnitude neurons (n={len(groups['magnitude'])})", C[0]),
                                  ("modular", f"modular neurons (n={len(groups['modular'])})", C[1]),
                                  ("sum", "sum", C[6])]):
    ax[k].plot(offs, aligned[g], color=c, lw=1.4); ax[k].axvline(0, color=MUTED, lw=0.8, ls=":")
    ax[k].set_ylabel("logit"); ax[k].set_title(f"({'abc'[k]}) {lab}: mean direct contribution", loc="left")
ax[2].set_xlabel("candidate answer − true product"); ax[2].set_xticks(np.arange(-120, 121, 20))
fig.tight_layout(); save(fig, "fig_group_decomp")
