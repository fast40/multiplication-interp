"""Find MLP neurons (layers 14-31) at '=' with the largest direct logit attribution (DLA) to the
correct answer; record their activations over all single-token problems and their output
patterns over number tokens 0..999."""
import numpy as np, torch, json
from common import load, prompt, answer_id

tok, model = load()
torch.set_grad_enabled(False)
LAYERS = list(range(14, 32))
probs = [(a, b) for a in range(10, 100) for b in range(2, 100) if a * b <= 999]
ids = torch.tensor([tok(prompt(a, b)).input_ids for a, b in probs]).cuda()
true = torch.tensor([a * b for a, b in probs]).cuda()
num_ids = torch.tensor([answer_id(tok, n) for n in range(1000)]).cuda()
WU = (model.lm_head.weight[num_ids].float() * model.model.norm.weight.float())  # 1000 x d (norm gain folded)
WU = WU - WU.mean(0, keepdim=True)  # logit differences among numbers only

acts = {}
hs = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, i, l=l: acts.setdefault(l, []).append(i[0][:, -1].float().cpu())) for l in LAYERS]
rms = []
fh = model.model.norm.register_forward_pre_hook(lambda m, i: rms.append(i[0][:, -1].float().pow(2).mean(-1).sqrt().cpu()))
for i in range(0, len(probs), 128):
    with torch.no_grad(): model(ids[i:i + 128])
[h.remove() for h in hs]; fh.remove()
rms = torch.cat(rms).cuda()
rows = []
outpat = {}
for l in LAYERS:
    A = torch.cat(acts[l]).cuda()  # N x 14336
    Wd = model.model.layers[l].mlp.down_proj.weight.float()  # d x 14336
    P = (WU @ Wd)  # 1000 x 14336: logit effect over numbers per unit activation
    dla = A * P[true] / rms[:, None]  # N x 14336 direct logit attribution
    m = dla.mean(0)
    top = torch.argsort(-m)[:60]
    for n in top.tolist():
        rows.append(dict(layer=l, neuron=n, mean_dla=m[n].item()))
        outpat[f"{l}_{n}"] = (P[:, n]).cpu().numpy()
    acts[l] = A[:, top].cpu().numpy().astype(np.float32); acts[l + 100] = top.cpu().numpy()
    print(l, "top dla", m[top[:5]].cpu().numpy().round(3), "sum top60", m[top].sum().item(), "sum all pos", m.clamp(min=0).sum().item(), flush=True)
rows.sort(key=lambda r: -r["mean_dla"])
np.savez("results/cache/neurons.npz", probs=np.array(probs), **{f"acts_{l}": acts[l] for l in LAYERS}, **{f"idx_{l}": acts[l + 100] for l in LAYERS},
         **{f"out_{k}": v for k, v in outpat.items()})
json.dump(rows, open("results/neurons_top.json", "w"))
print(rows[:20])
