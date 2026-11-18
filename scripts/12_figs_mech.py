"""Figures: example neurons; ablation + transplant results."""
import json, numpy as np
from style import plt, C, INK2, MUTED, GRID, save
z = np.load("results/cache/neurons.npz"); cls = {(d["layer"], d["neuron"]): d for d in json.load(open("results/neuron_classes.json"))}
probs = z["probs"]; P = probs[:, 0] * probs[:, 1]; n = np.arange(1000)
def act_of(l, nn): return z[f"acts_{l}"][:, list(z[f"idx_{l}"]).index(nn)]
top_mod10 = max((d for d in cls.values() if d["type"] == "mod10"), key=lambda d: d["mean_dla"])
EX = [(31, 10258, None), (top_mod10["layer"], top_mod10["neuron"], 10), (29, 1647, 8), (30, 13826, 3), (26, 8382, 7)]
fig, ax = plt.subplots(2, len(EX), figsize=(7.4, 3.6))
for j, (l, nn, m) in enumerate(EX):
    out, act = z[f"out_{l}_{nn}"], act_of(l, nn); d = cls[(l, nn)]
    if m is None:
        ax[0, j].plot(n, out, color=C[0], lw=1); ax[0, j].set_xlabel("number token n")
        ax[1, j].scatter(P, act, s=1.5, color=C[0], alpha=0.5, lw=0); ax[1, j].set_xscale("log"); ax[1, j].set_xlabel("product ab")
        ax[0, j].set_title(f"L{l}N{nn}\nmagnitude (DLA {d['mean_dla']:.2f})", fontsize=8)
    else:
        g = [out[n % m == r].mean() for r in range(m)]; e = [out[n % m == r].std() for r in range(m)]
        ax[0, j].bar(range(m), g, yerr=e, color=C[1], width=0.7, error_kw=dict(lw=0.6, ecolor=MUTED))
        ax[0, j].set_xlabel(f"n mod {m}"); ax[0, j].set_xticks(range(m))
        ga = [act[P % m == r].mean() for r in range(m)]; ea = [act[P % m == r].std() for r in range(m)]
        ax[1, j].bar(range(m), ga, yerr=ea, color=C[1], width=0.7, error_kw=dict(lw=0.6, ecolor=MUTED))
        ax[1, j].set_xlabel(f"ab mod {m}"); ax[1, j].set_xticks(range(m))
        ax[0, j].set_title(f"L{l}N{nn}\nmod {m} (DLA {d['mean_dla']:.2f})", fontsize=8)
    for i in range(2): ax[i, j].tick_params(labelsize=6.5)
ax[0, 0].set_ylabel("output → logit of n"); ax[1, 0].set_ylabel("activation at '='")
fig.tight_layout(); save(fig, "fig_neurons")

# ---- interventions
R = json.load(open("results/ablate_transplant.json")); T = np.load("results/ablate_transplant_outputs.npz"); RT = json.load(open("results/residue_transfer.json"))
Pt, Py = T["P"], T["P"][T["donor"]]
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw=dict(width_ratios=[1.1, 1.2, 1.0]))
conds = [("clean", None), ("random", "ablate_random(mod-size)"), ("modular", "ablate_modular"), ("magnitude", "ablate_magnitude")]
for i, (lab, k) in enumerate(conds):
    o = T[k] if k else T["transplant_random(mod-size)"] * 0 + Pt
    ok = (o == Pt).mean(); near = ((o != Pt) & (np.abs(o - Pt) <= 10)).mean(); far = ((np.abs(o - Pt) > 10)).mean()
    ax[0].bar(i, ok, color=C[0], width=0.65); ax[0].bar(i, near, bottom=ok, color=C[3], width=0.65); ax[0].bar(i, far, bottom=ok + near, color=C[7], width=0.65)
ax[0].set_xticks(range(4), [c[0] for c in conds], fontsize=7, rotation=25); ax[0].set_ylabel("fraction of problems")
ax[0].set_title("(a) mean-ablate group at '='")
from matplotlib.patches import Patch
ax[0].legend([Patch(color=C[0]), Patch(color=C[3]), Patch(color=C[7])], ["correct", "wrong, |err|≤10", "wrong, |err|>10"], loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7, ncol=1)
ms = ["2", "4", "5", "8", "10", "3", "7", "9"]; x = np.arange(len(ms))
ax[1].bar(x - 0.2, [RT["transplant_modular"][m]["follows_donor"] for m in ms], 0.38, color=C[1], label="modular transplant")
ax[1].bar(x + 0.2, [RT["transplant_magnitude"][m]["follows_donor"] for m in ms], 0.38, color=C[0], label="magnitude transplant")
ax[1].plot(x, [1 / int(m) for m in ms], ls="none", marker="_", ms=10, color=INK2, label="chance")
ax[1].set_xticks(x, ms); ax[1].set_xlabel("modulus m"); ax[1].set_ylabel("output ≡ donor (mod m)")
ax[1].set_title("(b) residues follow the modular group"); ax[1].legend(loc="upper right", fontsize=7)
o = T["transplant_magnitude"]
ax[2].scatter(Py, o, s=2, color=C[0], alpha=0.4, lw=0, label="magnitude transplant")
ax[2].plot([0, 999], [0, 999], color=MUTED, lw=0.8, ls=":")
ax[2].set_xlabel("donor product"); ax[2].set_ylabel("model output"); ax[2].set_title("(c) magnitude follows donor")
fig.tight_layout(); save(fig, "fig_interventions")
