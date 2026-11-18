"""Mechanism schematic with inset curves from the group decomposition."""
import json, numpy as np
from style import plt, C, INK, INK2, MUTED, AXIS, save
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
D = json.load(open("results/group_decomp.json")); offs = np.array(D["offs"])
fig = plt.figure(figsize=(7.4, 3.3)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 45); ax.axis("off")
def box(x, y, w, h, text, fc="#f4f3ef", ec=AXIS, fs=8, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color)
def arrow(x0, y0, x1, y1, c=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=c, lw=1))
# tokens
for i, t in enumerate(["a", "*", "b", "="]):
    box(2 + i * 6, 3, 4, 4, t, fs=10)
ax.text(13, 0.5, "input tokens (layers 0–14: operands stay home)", ha="center", fontsize=7, color=INK2)
# movers
box(2, 13, 22, 6, "L16H21 copies a → '='\nL15H14 copies b → '='", fs=7.5)
arrow(4, 7.5, 8, 12.5); arrow(16, 7.5, 16, 12.5); arrow(13, 19.5, 13, 23)
# early MLP
box(2, 23.5, 22, 7, "MLPs ~L16–20 at '=' (inferred)\nbuild product features:\nresidues mod 2,4,5,8,10,(3)\n+ log-magnitude", fs=7)
arrow(24.8, 29, 33, 35); arrow(24.8, 25, 33, 13)
# groups
box(33.5, 30, 22, 10, "", fc="#fdf0ea", ec=C[1])
ax.text(44.5, 37.8, "modular neurons (285)", ha="center", fontsize=7.5, color=INK)
ax.text(44.5, 32.3, "read  ab mod m\nwrite comb over n ≡ ab (mod m)", ha="center", fontsize=6.5, color=INK2)
box(33.5, 7, 22, 10, "", fc="#eaf2fc", ec=C[0])
ax.text(44.5, 14.8, "magnitude neurons (209)", ha="center", fontsize=7.5, color=INK)
ax.text(44.5, 9.3, "read  log ab (range detectors)\nwrite smooth bump over n", ha="center", fontsize=6.5, color=INK2)
ax.text(44.5, 22.8, "layers 18–31", ha="center", fontsize=7, color=INK2)
# insets
def inset(rect, y, c, title):
    a = fig.add_axes(rect); a.plot(offs, y, color=c, lw=1.1); a.axvline(0, color=MUTED, lw=0.6, ls=":")
    a.set_xticks([-100, 0, 100]); a.set_yticks([]); a.tick_params(labelsize=6); a.grid(False); a.set_title(title, fontsize=7, pad=2)
    for s in ["left"]: a.spines[s].set_visible(False)
    return a
inset([0.61, 0.66, 0.16, 0.22], D["aligned"]["modular"], C[1], "comb")
inset([0.61, 0.15, 0.16, 0.22], D["aligned"]["magnitude"], C[0], "bump")
a = inset([0.83, 0.38, 0.16, 0.26], D["aligned"]["sum"], C[6], "sum → answer")
a.set_xlabel("n − ab", fontsize=6.5, labelpad=1)
arrow(56.5, 35, 60.5, 35); arrow(56.5, 12, 60.5, 12)
arrow(78.5, 35, 83, 26, c=C[1]); arrow(78.5, 12, 83, 18, c=C[0])
save(fig, "fig_schematic")
