"""CPU: are modular neurons divisibility detectors? Compare variance in activation explained by
the indicator [ab ≡ 0 mod m] vs by the full residue ab mod m; same for output pattern [n ≡ 0 mod m]."""
import json, numpy as np
z = np.load("results/cache/neurons.npz"); cls = json.load(open("results/neuron_classes.json"))
probs = z["probs"]; P = probs[:, 0] * probs[:, 1]; n = np.arange(1000)
def fg(y, g):
    y = y - y.mean(); m = np.zeros_like(y, dtype=np.float64)
    for v in np.unique(g): m[g == v] = y[g == v].mean()
    return float((m ** 2).sum() / (y ** 2).sum())
rows = []
for d in cls:
    if not d["type"].startswith("mod") or d["mod"] >= 50: continue
    l, nn, m = d["layer"], d["neuron"], d["mod"]
    act = z[f"acts_{l}"][:, list(z[f"idx_{l}"]).index(nn)].astype(float); out = z[f"out_{l}_{nn}"].astype(float)
    rows.append(dict(layer=l, neuron=nn, m=m, dla=d["mean_dla"],
                     act_full=fg(act, P % m), act_div=fg(act, P % m == 0), out_full=fg(out, n % m), out_div=fg(out, n % m == 0),
                     sign_agree=bool(np.sign(act[P % m == 0].mean() - act.mean()) == np.sign(out[n % m == 0].mean() - out.mean()))))
r = rows
ratio_a = np.array([x["act_div"] / x["act_full"] for x in r]); ratio_o = np.array([x["out_div"] / x["out_full"] for x in r])
w = np.array([x["dla"] for x in r])
summary = dict(n=len(r), act_div_over_full_median=float(np.median(ratio_a)), out_div_over_full_median=float(np.median(ratio_o)),
               frac_act_div_ge_0p8=float((ratio_a >= 0.8).mean()), frac_out_div_ge_0p8=float((ratio_o >= 0.8).mean()),
               frac_sign_agree=float(np.mean([x["sign_agree"] for x in r])),
               dla_weighted_frac_divisibility=float(w[(ratio_a >= 0.8) & (ratio_o >= 0.8)].sum() / w.sum()))
print(json.dumps(summary, indent=1))
from collections import Counter; print(Counter(x["m"] for x in r))
json.dump(dict(summary=summary, rows=r), open("results/divisibility.json", "w"), indent=0)
