"""CPU: classify top-DLA neurons by their output pattern over number tokens 0..999:
'magnitude' (smooth, periods >= 100) or 'mod m' (best modulus m by variance explained by n mod m).
Then test what drives activation: product residue mod m vs operand-residue pair vs magnitude."""
import numpy as np, json, collections
z = np.load("results/cache/neurons.npz")
rows = json.load(open("results/neurons_top.json"))
probs = z["probs"]; A, B = probs[:, 0], probs[:, 1]; P = A * B
n = np.arange(1000)
MODS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 19, 20, 25, 50, 100]

def fg(y, g):
    y = y - y.mean(); m = np.zeros_like(y, dtype=np.float64)
    for v in np.unique(g): m[g == v] = y[g == v].mean()
    return float((m ** 2).sum() / (y ** 2).sum())

def lowpass(y, k=10):
    f = np.fft.rfft(y - y.mean()); p = np.abs(f) ** 2; p[0] = 0
    return float(p[1:k + 1].sum() / p.sum())

out = []
for r in rows:
    l, nn = r["layer"], r["neuron"]
    pat = z[f"out_{l}_{nn}"].astype(np.float64)
    act = z[f"acts_{l}"][:, list(z[f"idx_{l}"]).index(nn)].astype(np.float64)
    d = dict(r); d["out_smooth"] = lowpass(pat)
    # best modulus: maximize explained variance minus chance level (m-1)/999, prefer smaller m on ties
    sc = {m: fg(pat, n % m) - (m - 1) / 999 for m in MODS}
    # minimal modulus reaching 90% of best score (avoid 100 absorbing 10, etc.)
    best = max(sc.values()); m_star = min((m for m in MODS if sc[m] >= 0.9 * best), default=2)
    d["mod"] = m_star; d["out_mod"] = sc[m_star]
    d["type"] = "magnitude" if d["out_smooth"] > max(0.5, d["out_mod"]) else (f"mod{m_star}" if d["out_mod"] > 0.3 else "other")
    m = m_star
    d["act_prodmod"] = fg(act, P % m)                  # product residue
    d["act_pairmod"] = fg(act, (A % m) * 1000 + B % m)  # operand residue pair (superset of info)
    d["act_mag"] = fg(act, np.digitize(np.log(P), np.linspace(np.log(20), np.log(999), 40)))
    out.append(d)
json.dump(out, open("results/neuron_classes.json", "w"), indent=0)
top = sorted(out, key=lambda d: -d["mean_dla"])
for K in [50, 200, len(top)]:
    c = collections.Counter(d["type"] for d in top[:K]); s = collections.defaultdict(float)
    for d in top[:K]: s[d["type"]] += d["mean_dla"]
    print(f"top{K}:", sorted(((k, c[k], round(s[k], 2)) for k in c), key=lambda x: -x[2]))
print("activation drivers among top 200 (mean variance explained):")
g = collections.defaultdict(list)
for d in top[:200]: g["magnitude" if d["type"] == "magnitude" else ("modular" if d["type"].startswith("mod") else "other")].append(d)
for k, v in g.items():
    print(k, len(v), "prodmod %.2f pairmod %.2f mag %.2f" % tuple(np.mean([d[x] for d in v]) for x in ["act_prodmod", "act_pairmod", "act_mag"]))
for d in top[:25]:
    print(d["layer"], d["neuron"], d["type"], round(d["mean_dla"], 3), "smooth %.2f mod %.2f | act prodmod %.2f pairmod %.2f mag %.2f" % (d["out_smooth"], d["out_mod"], d["act_prodmod"], d["act_pairmod"], d["act_mag"]))
