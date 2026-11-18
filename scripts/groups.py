"""Neuron groups (from Exp 07) and helpers to cache / overwrite their activations at '='."""
import json, numpy as np, torch
from collections import defaultdict

def load_groups():
    cls = json.load(open("results/neuron_classes.json")); G = {"magnitude": defaultdict(list), "modular": defaultdict(list)}
    for d in cls:
        if d["type"] == "magnitude": G["magnitude"][d["layer"]].append(d["neuron"])
        elif d["type"].startswith("mod"): G["modular"][d["layer"]].append(d["neuron"])
    return {k: {l: torch.tensor(sorted(v)).cuda() for l, v in g.items()} for k, g in G.items()}

def cache_acts(model, ids, group, bs=128):
    acts = {l: [] for l in group}
    hs = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, i, l=l: acts[l].append(i[0][:, -1, group[l]].float().clone())) for l in group]
    for i in range(0, len(ids), bs): model(ids[i:i + bs])
    [h.remove() for h in hs]
    return {l: torch.cat(v) for l, v in acts.items()}

def run_with(model, ids, num_ids, groups_vals, bs=128):
    """groups_vals: list of (group, vals) where vals[l] is (N x k) to write at '='. Returns argmax over numbers."""
    preds = []
    for i in range(0, len(ids), bs):
        hs = []
        for group, vals in groups_vals:
            for l, idx in group.items():
                def pre(m, inp, l=l, idx=idx, vals=vals, i=i):
                    x = inp[0].clone(); x[:, -1, idx] = vals[l][i:i + x.shape[0]].to(x.dtype); return (x,)
                hs.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(pre))
        lg = model(ids[i:i + bs]).logits[:, -1]
        [h.remove() for h in hs]
        preds.append(lg[:, num_ids].argmax(-1).cpu())
    return torch.cat(preds).numpy()
