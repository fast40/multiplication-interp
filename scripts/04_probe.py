"""Collect '=' residuals for all single-token problems; linear probes per layer for
product units digit, tens digit, hundreds digit, log-magnitude, and operand units digits.
Also logit-lens rank of the correct answer and of the correct units digit class."""
import numpy as np, torch, json
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from common import load, prompt, answer_id, layer_out

tok, model = load()
L = model.config.num_hidden_layers
probs = [(a, b) for a in range(10, 100) for b in range(2, 100) if a * b <= 999]
print("n problems", len(probs))
ids = torch.tensor([tok(prompt(a, b)).input_ids for a, b in probs]).cuda()
acts = np.zeros((L + 1, len(probs), model.config.hidden_size), dtype=np.float16)
num_ids = torch.tensor([answer_id(tok, n) for n in range(1000)]).cuda()
W = model.lm_head.weight[num_ids].float()  # 1000 x d
lens_rank = np.zeros((L + 1, len(probs))); lens_digit = np.zeros((L + 1, len(probs)))
true = np.array([a * b for a, b in probs])
for i in range(0, len(probs), 128):
    with torch.no_grad():
        out = model(ids[i:i + 128], output_hidden_states=True)
    for l, h in enumerate(out.hidden_states):
        x = h[:, -1].float()
        acts[l, i:i + 128] = x.cpu().numpy()
        lg = model.model.norm(x.to(torch.bfloat16)).float() @ W.T  # logit lens over 0..999
        t = torch.tensor(true[i:i + 128]).cuda()
        lens_rank[l, i:i + 128] = (lg > lg.gather(1, t[:, None])).sum(1).cpu().numpy()
        # units digit class via logsumexp over tokens with each units digit
        cls = torch.stack([torch.logsumexp(lg[:, d::10], 1) for d in range(10)], 1)
        lens_digit[l, i:i + 128] = (cls.argmax(1) == t % 10).float().cpu().numpy()
np.save("results/cache_acts.npy", acts) if False else None
A = np.array([a for a, b in probs]); B = np.array([b for a, b in probs])
targets = {"prod_units": true % 10, "prod_tens": (true // 10) % 10, "prod_hund": true // 100,
           "a_units": A % 10, "b_units": B % 10}
res = {k: [] for k in list(targets) + ["log_prod_r2"]}
idx_tr, idx_te = train_test_split(np.arange(len(probs)), test_size=0.3, random_state=0)
for l in range(L + 1):
    X = acts[l].astype(np.float32); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    for k, y in targets.items():
        clf = LogisticRegression(max_iter=300, C=0.05).fit(X[idx_tr], y[idx_tr])
        res[k].append(clf.score(X[idx_te], y[idx_te]))
    r = Ridge(alpha=100).fit(X[idx_tr], np.log(true[idx_tr]))
    res["log_prod_r2"].append(r.score(X[idx_te], np.log(true[idx_te])))
    print(l, {k: round(v[-1], 2) for k, v in res.items()}, "lens top1", (lens_rank[l] == 0).mean().round(2), "lens digit", lens_digit[l].mean().round(2), flush=True)
res["lens_top1"] = (lens_rank == 0).mean(1).tolist(); res["lens_digit"] = lens_digit.mean(1).tolist()
res["lens_median_rank"] = np.median(lens_rank, 1).tolist()
json.dump(res, open("results/probe.json", "w"))
np.save("results/cache/acts_eq.npy", acts[[16, 20, 24, 28, 32]])
