"""Residual-stream activation patching (clean -> corrupt) by layer and position.
Corrupt one operand (a or b) with another of the same digit count; patch clean
residual at (layer, pos) into corrupted run; measure recovery of clean-vs-corrupt logit diff."""
import numpy as np, torch, json
from common import load, prompt, problems, answer_id, layer_out

tok, model = load()
L = model.config.num_hidden_layers
rng = np.random.default_rng(0)

def make_pairs(which, n=120):
    pairs = []
    while len(pairs) < n:
        a, b = problems(rng, 1)[0]
        for _ in range(50):
            if which == "a":
                a2 = int(rng.integers(10, 100)); b2 = b
            else:
                b2 = int(rng.integers(2, 10) if b < 10 else rng.integers(10, 100)); a2 = a
            if (a2, b2) != (a, b) and a2 * b2 <= 999 and a2 * b2 != a * b:
                pairs.append(((a, b), (a2, b2))); break
    return pairs

def run(ids, hooks=None):
    with torch.no_grad():
        return model(ids).logits[:, -1].float()

res = {}
for which in ["a", "b"]:
    pairs = make_pairs(which)
    clean = torch.tensor([tok(prompt(*c)).input_ids for c, _ in pairs]).cuda()
    corr = torch.tensor([tok(prompt(*k)).input_ids for _, k in pairs]).cuda()
    ac = torch.tensor([answer_id(tok, c[0] * c[1]) for c, _ in pairs]).cuda()
    ak = torch.tensor([answer_id(tok, k[0] * k[1]) for _, k in pairs]).cuda()
    # cache clean residuals
    cache = {}
    hs = [model.model.layers[l].register_forward_hook(lambda m, i, o, l=l: cache.__setitem__(l, layer_out(o).detach().clone())) for l in range(L)]
    lc = run(clean); [h.remove() for h in hs]
    lk = run(corr)
    ar = torch.arange(len(pairs))
    d_clean = (lc[ar, ac] - lc[ar, ak]).mean().item(); d_corr = (lk[ar, ac] - lk[ar, ak]).mean().item()
    print(which, "clean ld", d_clean, "corrupt ld", d_corr)
    T = clean.shape[1]
    positions = list(range(T - 5, T))  # '\n', a, '*', b, '='
    grid = np.zeros((L, len(positions)))
    for l in range(L):
        for j, p in enumerate(positions):
            def hook(m, i, o, p=p, l=l):
                h = layer_out(o); h[:, p] = cache[l][:, p]; return o
            hh = model.model.layers[l].register_forward_hook(hook)
            lp = run(corr); hh.remove()
            grid[l, j] = ((lp[ar, ac] - lp[ar, ak]).mean().item() - d_corr) / (d_clean - d_corr)
    res[which] = grid.tolist()
    print(np.round(grid[::2], 2))
json.dump(res, open("results/patch_resid.json", "w"))
