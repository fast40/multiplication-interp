"""Shared matplotlib style (palette: reference dataviz categorical slots, light mode)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "figure.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
})
def save(fig, name):
    fig.savefig(f"figures/{name}.pdf"); fig.savefig(f"figures/{name}.png"); plt.close(fig)
