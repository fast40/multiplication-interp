"""Patch attention-out, MLP-out (per layer) and individual heads at the '=' position, clean -> corrupt."""
import numpy as np, torch, json
from common import load, prompt, problems, answer_id, layer_out

tok, model = load()
L = model.config.num_hidden_layers; H = model.config.num_attention_heads; D = model.config.head_dim
rng = np.random.default_rng(1)

def make_pairs(n=200):
    pairs = []
    while len(pairs) < n:
        a, b = problems(rng, 1)[0]
        a2, b2 = problems(rng, 1)[0]
        if len(str(b2)) == len(str(b)) and a2 * b2 != a * b:
            pairs.append(((a, b), (a2, b2)))  # corrupt BOTH operands
    return pairs

pairs = make_pairs()
clean = torch.tensor([tok(prompt(*c)).input_ids for c, _ in pairs]).cuda()
corr = torch.tensor([tok(prompt(*k)).input_ids for _, k in pairs]).cuda()
ac = torch.tensor([answer_id(tok, c[0] * c[1]) for c, _ in pairs]).cuda()
ak = torch.tensor([answer_id(tok, k[0] * k[1]) for _, k in pairs]).cuda()
ar = torch.arange(len(pairs))
def ld(x): return (x[ar, ac] - x[ar, ak]).mean().item()
def run(ids):
    with torch.no_grad(): return model(ids).logits[:, -1].float()

mods = {"attn": [l.self_attn for l in model.model.layers], "mlp": [l.mlp for l in model.model.layers],
        "heads": [l.self_attn.o_proj for l in model.model.layers]}
cache = {k: {} for k in mods}
hs = []
for l in range(L):
    hs.append(mods["attn"][l].register_forward_hook(lambda m, i, o, l=l: cache["attn"].__setitem__(l, layer_out(o)[:, -1].clone())))
    hs.append(mods["mlp"][l].register_forward_hook(lambda m, i, o, l=l: cache["mlp"].__setitem__(l, o[:, -1].clone())))
    hs.append(mods["heads"][l].register_forward_pre_hook(lambda m, i, l=l: cache["heads"].__setitem__(l, i[0][:, -1].clone())))
lc = run(clean); [h.remove() for h in hs]
lk = run(corr); dc, dk = ld(lc), ld(lk)
print("clean", dc, "corr", dk)
norm = lambda x: (x - dk) / (dc - dk)
res = {"attn": [], "mlp": []}
for kind in ["attn", "mlp"]:
    for l in range(L):
        if kind == "attn":
            def hook(m, i, o, l=l):
                h = layer_out(o); h[:, -1] = cache["attn"][l]; return o
        else:
            def hook(m, i, o, l=l):
                o[:, -1] = cache["mlp"][l]; return o
        h = mods[kind][l].register_forward_hook(hook); res[kind].append(norm(ld(run(corr)))); h.remove()
    print(kind, np.round(res[kind], 2))
heads = np.zeros((L, H))
for l in range(L):
    for hd in range(H):
        def pre(m, i, l=l, hd=hd):
            x = i[0].clone(); x[:, -1, hd * D:(hd + 1) * D] = cache["heads"][l][:, hd * D:(hd + 1) * D]; return (x,)
        h = mods["heads"][l].register_forward_pre_hook(pre); heads[l, hd] = norm(ld(run(corr))); h.remove()
res["heads"] = heads.tolist()
top = np.argsort(-heads.ravel())[:20]
print("top heads", [(int(t // H), int(t % H), round(float(heads.ravel()[t]), 3)) for t in top])
json.dump(res, open("results/patch_components.json", "w"))
