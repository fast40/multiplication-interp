"""Attention patterns of the operand-mover heads at '=' and their effect when ablated."""
import numpy as np, torch, json
from common import load, prompt, problems, answer_id

tok, model = load()
model.config._attn_implementation = "eager"
for l in model.model.layers: l.self_attn.config._attn_implementation = "eager"
rng = np.random.default_rng(2)
probs = problems(rng, 300)
ids = torch.tensor([tok(prompt(a, b)).input_ids for a, b in probs]).cuda()
with torch.no_grad():
    out = model(ids, output_attentions=True)
names = ["\\n", "a", "*", "b", "="]
res = {}
for (l, h) in [(15, 14), (16, 21), (15, 15), (20, 2)]:
    att = out.attentions[l][:, h, -1].float().cpu().numpy()  # B x T
    m = att[:, -5:].mean(0); res[f"L{l}H{h}"] = dict(zip(names, m.tolist())); res[f"L{l}H{h}"]["bos"] = float(att[:, 0].mean())
    print(f"L{l}H{h}", {k: round(v, 3) for k, v in res[f"L{l}H{h}"].items()})
# ablation: zero (mean-ablate) the two heads' output at '=' -> accuracy
D = model.config.head_dim
ans = torch.tensor([answer_id(tok, a * b) for a, b in probs]).cuda()
def acc(heads):
    hs = []
    for (l, h) in heads:
        def pre(m, i, h=h):
            x = i[0].clone(); x[:, -1, h * D:(h + 1) * D] = x[:, -1, h * D:(h + 1) * D].mean(0, keepdim=True); return (x,)
        hs.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(pre))
    with torch.no_grad(): lg = model(ids).logits[:, -1]
    [x.remove() for x in hs]
    return (lg.argmax(-1) == ans).float().mean().item()
model.config._attn_implementation = "sdpa"
for l in model.model.layers: l.self_attn.config._attn_implementation = "sdpa"
res["acc"] = {"none": acc([]), "L15H14": acc([(15, 14)]), "L16H21": acc([(16, 21)]), "both": acc([(15, 14), (16, 21)]),
              "rand2": acc([(15, 3), (16, 5)])}
print(res["acc"])
json.dump(res, open("results/head_attn.json", "w"))
