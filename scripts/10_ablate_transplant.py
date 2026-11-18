"""Causal tests on the magnitude vs modular neuron groups at '=':
(1) mean-ablation (double dissociation) vs size-matched random neurons;
(2) activation transplant from a donor problem y into x for one group at a time."""
import json, numpy as np, torch
from collections import defaultdict
from common import load, prompt, answer_id

tok, model = load(); torch.set_grad_enabled(False)
z = np.load("results/cache/neurons.npz"); cls = json.load(open("results/neuron_classes.json"))
probs = [tuple(p) for p in z["probs"]]; N = len(probs)
P = np.array([a * b for a, b in probs])
ids = torch.tensor([tok(prompt(a, b)).input_ids for a, b in probs]).cuda()
num_ids = torch.tensor([answer_id(tok, n) for n in range(1000)]).cuda()
rng = np.random.default_rng(0)

def group(pred):
    g = defaultdict(list)
    for d in cls:
        if pred(d): g[d["layer"]].append(d["neuron"])
    return {l: torch.tensor(v).cuda() for l, v in g.items()}
G = {"magnitude": group(lambda d: d["type"] == "magnitude"), "modular": group(lambda d: d["type"].startswith("mod"))}
def rand_like(g):
    out = {}
    for l, v in g.items():
        cached = set(z[f"idx_{l}"].tolist()); pool = np.array([i for i in range(14336) if i not in cached])
        out[l] = torch.tensor(rng.choice(pool, len(v), replace=False)).cuda()
    return out
G["random(mag-size)"] = rand_like(G["magnitude"]); G["random(mod-size)"] = rand_like(G["modular"])

# cache activations of all group neurons at '=' for all problems
allidx = {l: torch.unique(torch.cat([G[k][l] for k in G if l in G[k]])) for l in range(14, 32) if any(l in G[k] for k in G)}
allidx = {l: v for l, v in allidx.items() if len(v)}
acts = {l: [] for l in allidx}
hs = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, i, l=l: acts[l].append(i[0][:, -1, allidx[l]].clone())) for l in allidx]
for i in range(0, N, 128): model(ids[i:i + 128])
[h.remove() for h in hs]
acts = {l: torch.cat(v) for l, v in acts.items()}  # N x k
pos = {l: {int(n): j for j, n in enumerate(allidx[l].tolist())} for l in allidx}

def run(gname, src):
    """src: N x ... per layer; replace group neurons at '=' with src values (rows aligned with batch)."""
    g = G[gname]; preds = []
    for i in range(0, N, 128):
        hs = []
        for l, idx in g.items():
            cols = torch.tensor([pos[l][int(n)] for n in idx.tolist()]).cuda()
            def pre(m, inp, l=l, idx=idx, cols=cols, i=i):
                x = inp[0].clone(); x[:, -1, idx] = src[l][i:i + x.shape[0]][:, cols].to(x.dtype); return (x,)
            hs.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(pre))
        lg = model(ids[i:i + 128]).logits[:, -1]
        [h.remove() for h in hs]
        preds.append(lg[:, num_ids].argmax(-1).cpu())  # restricted to numbers 0..999
    return torch.cat(preds).numpy()

def stats(o, ref=P):
    return dict(acc=float((o == ref).mean()), units_ok=float((o % 10 == ref % 10).mean()), mod3_ok=float((o % 3 == ref % 3).mean()),
                within10=float((np.abs(o - ref) <= 10).mean()), med_abs_err=float(np.median(np.abs(o - ref))),
                units_ok_given_wrong=float((o % 10 == ref % 10)[o != ref].mean()) if (o != ref).any() else None,
                within10_given_wrong=float((np.abs(o - ref) <= 10)[o != ref].mean()) if (o != ref).any() else None)
OUT = {}
res = {"n_neurons": {k: sum(len(v) for v in g.values()) for k, g in G.items()}}
mean = {l: acts[l].mean(0, keepdim=True).expand(N, -1) for l in acts}
res["clean"] = stats(run("magnitude", acts))
for k in G:
    o = run(k, mean); OUT[f"ablate_{k}"] = o; res[f"ablate_{k}"] = stats(o); print(k, res[f"ablate_{k}"], flush=True)

# transplant: donor y with |py-px| >= 100 and different units digit
donor = np.zeros(N, dtype=int)
for i in range(N):
    while True:
        j = rng.integers(N)
        if abs(P[j] - P[i]) >= 100 and P[j] % 10 != P[i] % 10: donor[i] = j; break
Py = P[donor]
src = {l: acts[l][torch.tensor(donor).cuda()] for l in acts}
def near_with_units(target, units):
    """Nearest number to `target` whose units digit is `units`."""
    c = np.arange(1000); out = []
    for t, u in zip(target, units):
        cc = c[c % 10 == u]; out.append(cc[np.argmin(np.abs(cc - t))])
    return np.array(out)
for k in ["modular", "magnitude", "random(mod-size)", "random(mag-size)"]:
    o = run(k, src); OUT[f"transplant_{k}"] = o
    r = dict(units_from_donor=float((o % 10 == Py % 10).mean()), units_from_orig=float((o % 10 == P % 10).mean()),
             equals_orig=float((o == P).mean()), equals_donor=float((o == Py).mean()),
             med_dist_orig=float(np.median(np.abs(o - P))), med_dist_donor=float(np.median(np.abs(o - Py))),
             hits_pred_orig_mag_donor_units=float((o == near_with_units(P, Py % 10)).mean()),
             hits_pred_donor_mag_orig_units=float((o == near_with_units(Py, P % 10)).mean()))
    res[f"transplant_{k}"] = r; print("transplant", k, r, flush=True)
both = {"both": {l: torch.unique(torch.cat([G[k][l] for k in ["magnitude", "modular"] if l in G[k]])) for l in allidx}}
G.update(both); o = run("both", src); OUT["transplant_both"] = o
res["transplant_both"] = dict(equals_donor=float((o == Py).mean()), equals_orig=float((o == P).mean()), med_dist_donor=float(np.median(np.abs(o - Py))))
print("both", res["transplant_both"])
json.dump(res, open("results/ablate_transplant.json", "w"), indent=1)
np.savez("results/ablate_transplant_outputs.npz", donor=donor, P=P, probs=np.array(probs), **OUT)
