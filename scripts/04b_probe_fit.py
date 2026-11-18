"""CPU-only: fit linear probes on cached '=' residuals (from 04_probe.py).
Features additive in (a, b) (operand digits, log a + log b) are trivially decodable once operands are
present; the product's units/tens digit are not additive and index actual computation."""
import numpy as np, json, warnings
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
acts = np.load("results/cache/acts_eq_all.npy", mmap_mode="r")
lens = np.load("results/cache/lens.npz"); true = lens["true"]; lens_rank, lens_digit = lens["lens_rank"], lens["lens_digit"]
probs = [(a, b) for a in range(10, 100) for b in range(2, 100) if a * b <= 999]
L = acts.shape[0] - 1
A = np.array([a for a, b in probs]); B = np.array([b for a, b in probs])
targets = {"prod_units": true % 10, "prod_tens": (true // 10) % 10, "prod_hund": true // 100,
           "a_units": A % 10, "b_units": B % 10}
res = {k: [] for k in list(targets) + ["log_prod_r2"]}
idx_tr, idx_te = train_test_split(np.arange(len(probs)), test_size=0.3, random_state=0)
for l in range(L + 1):
    X = acts[l].astype(np.float32); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X = PCA(256, random_state=0).fit(X[idx_tr]).transform(X)
    for k, y in targets.items():
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(X[idx_tr], y[idx_tr])
        res[k].append(clf.score(X[idx_te], y[idx_te]))
    r = Ridge(alpha=10).fit(X[idx_tr], np.log(true[idx_tr]))
    res["log_prod_r2"].append(r.score(X[idx_te], np.log(true[idx_te])))
    print(l, {k: round(v[-1], 2) for k, v in res.items()}, "lens top1", (lens_rank[l] == 0).mean().round(2), "lens digit", lens_digit[l].mean().round(2), flush=True)
res["lens_top1"] = (lens_rank == 0).mean(1).tolist(); res["lens_digit"] = lens_digit.mean(1).tolist()
res["lens_median_rank"] = np.median(lens_rank, 1).tolist()
json.dump(res, open("results/probe.json", "w"))
