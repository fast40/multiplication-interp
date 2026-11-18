"""Figures: behavior, patching, probes, example neurons."""
import json, numpy as np, pandas as pd
from style import plt, C, INK2, MUTED, save
from matplotlib.colors import LinearSegmentedColormap
seq = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#86b6ef", "#2a78d6", "#104281"])

# --- behavior
d = pd.read_csv("results/baseline.csv", dtype={"pred": str})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1, 1.15]))
M = np.full((99, 99), np.nan)
for r in d.itertuples(): M[r.a - 1, r.b - 1] = r.correct
ax[0].imshow(M, origin="lower", cmap=LinearSegmentedColormap.from_list("e", ["#e34948", "#cde2fb"]), extent=[0.5, 99.5, 0.5, 99.5])
bb = np.linspace(1, 99, 300); ax[0].plot(bb, 999 / bb, color=INK2, lw=1, ls="--"); ax[0].set_ylim(0.5, 99.5)
ax[0].text(60, 22, "ab = 999", color=INK2, fontsize=8); ax[0].grid(False)
ax[0].set_xlabel("b"); ax[0].set_ylabel("a"); ax[0].set_title("(a) correct (blue) / wrong (red)")
w = d[(d.true >= 1000) & ~d.correct].copy(); w["err"] = pd.to_numeric(w.pred, errors="coerce") - w.true; w = w.dropna(subset=["err"])
e = w.err.clip(-60, 60)
ax[1].hist(e, bins=np.arange(-60.5, 61.5, 1), color=C[0], edgecolor="white", linewidth=0.3)
ax[1].set_xlabel("prediction − true product"); ax[1].set_ylabel("count")
ax[1].set_title("(b) error of wrong 4-digit answers")
save(fig, "fig_behavior")

# --- patching
P = json.load(open("results/patch_resid.json")); pos = ["\\n", "a", "*", "b", "="]
fig = plt.figure(figsize=(7.4, 2.9), layout="constrained")
sf = fig.subfigures(1, 2, width_ratios=[1.15, 1])
ax = list(sf[0].subplots(1, 2, sharey=True)) + [sf[1].subplots()]
for i, k in enumerate(["a", "b"]):
    g = np.array(P[k]); im = ax[i].imshow(g, aspect="auto", origin="lower", cmap=seq, vmin=0, vmax=1)
    ax[i].set_xticks(range(5), pos); ax[i].set_title(f"({'ab'[i]}) corrupt {k}"); ax[i].grid(False)
    ax[i].set_xlabel("position")
ax[0].set_ylabel("layer")
sf[0].colorbar(im, ax=ax[:2], label="recovery", shrink=0.9)
Q = json.load(open("results/patch_components.json")); L = np.arange(32)
ax[2].plot(L, Q["attn"], color=C[0], label="attention out")
ax[2].plot(L, Q["mlp"], color=C[1], label="MLP out")
ax[2].set_xlabel("layer"); ax[2].set_ylabel("recovery of logit diff"); ax[2].legend(loc="upper left")
ax[2].annotate("L15H14 (→b)\nL16H21 (→a)", xy=(15.3, 0.37), xytext=(1, 0.22), color=INK2, fontsize=8, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax[2].set_title("(c) one component at '='")
save(fig, "fig_patching")

# --- probes
R = json.load(open("results/probe.json")); L = np.arange(len(R["prod_units"]))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for j, (k, lab) in enumerate([("a_units", "operand a units digit"), ("prod_units", "product units digit"), ("prod_hund", "product hundreds digit"), ("prod_tens", "product tens digit")]):
    ax[0].plot(L, R[k], color=C[j], label=lab)
ax[0].set_xlabel("layer"); ax[0].set_ylabel("held-out probe accuracy"); ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.5)
ax[0].set_title("(a) linear probes at '='")
ax[0].axvspan(15, 17, color="#e1e0d9", alpha=0.6, lw=0); ax[0].text(15.2, 0.02, "operands\narrive", fontsize=7, color=INK2)
ax[1].plot(L, R["lens_top1"], color=C[0], label="exact answer top-1 (of 0..999)")
ax[1].plot(L, R["lens_digit"], color=C[1], label="units-digit class top-1")
ax[1].set_xlabel("layer"); ax[1].set_ylabel("accuracy"); ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=7.5); ax[1].set_title("(b) logit lens at '='")
save(fig, "fig_probes")
